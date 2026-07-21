"""Shared in-process list cache for AWS resource searches.

Model: **stale-while-revalidate with a background daemon thread.**

The first call that misses the cache fetches from AWS synchronously and
registers a ``refresh_fn``. A daemon ``threading.Thread`` then wakes every
``refresh_interval`` seconds (default: 80 % of TTL), calls ``refresh_fn()``,
and silently replaces the cached list — so subsequent searches always read
from memory and never block on AWS.

Key properties
--------------
- **No external state.** Everything lives in the Python process heap.
- **Daemon threads.** They auto-die when the process exits; restart = fresh.
- **TTL as safety net only.** If a background refresh fails repeatedly and
  the entry goes stale, the next *synchronous* fetch will heal it.
- **force_refresh.** Evicts the entry immediately; the caller re-fetches
  synchronously. The existing background thread keeps running.

Env vars
--------
``MONTAI_RESOURCE_CACHE_TTL``
    Seconds before an entry with no background refresher is considered stale
    (default 300 s / 5 min). Entries with a running refresher are served
    indefinitely — the refresher keeps them current.
``MONTAI_RESOURCE_CACHE_REFRESH_FACTOR``
    Background refresh fires at ``TTL × factor`` (default 0.8 → every 4 min
    for a 5-min TTL).

Usage
-----
From a search module::

    from montai_joblib_utils_ai_mcp.tools import _resource_cache as cache

    def search_impl(query, region_name=None, force_refresh=False):
        raw = cache.get("sfn", region_name, force_refresh=force_refresh)
        if raw is None:
            client = boto3.client("stepfunctions", region_name=region_name)
            raw = _list_all_from_aws(client)
            cache.set(
                "sfn", region_name, raw,
                refresh_fn=lambda: _list_all_from_aws(
                    boto3.client("stepfunctions", region_name=region_name)
                ),
            )
        return _filter(raw, query)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)

_DEFAULT_TTL: float = float(os.environ.get("MONTAI_RESOURCE_CACHE_TTL", "300"))
_REFRESH_FACTOR: float = float(os.environ.get("MONTAI_RESOURCE_CACHE_REFRESH_FACTOR", "0.8"))


class ResourceCache:
    """Thread-safe stale-while-revalidate cache for full AWS resource lists.

    Each entry holds the complete paginated list for one (resource, region)
    pair. The first population starts a daemon background thread that silently
    refreshes the list on a timer so concurrent search workers always read
    from memory.
    """

    def __init__(
        self,
        ttl: float = _DEFAULT_TTL,
        refresh_factor: float = _REFRESH_FACTOR,
    ) -> None:
        self._ttl = ttl
        self._refresh_interval = ttl * refresh_factor
        self._lock = threading.Lock()
        # key -> (items, fetched_at)
        self._store: dict[str, tuple[list[Any], float]] = {}
        # keys for which a background refresh thread is already running
        self._refreshers: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        resource: str,
        region: str | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[Any] | None:
        """Return cached list, or ``None`` on miss / force-refresh.

        - If a background refresher is registered for this key, the TTL check
          is skipped — the thread keeps the data current, so we always serve it.
        - If no refresher, entries older than TTL are treated as a miss.
        - ``force_refresh=True`` evicts the entry immediately regardless.
          The background thread (if any) keeps running and will repopulate on
          its next tick; the caller re-fetches synchronously in the meantime.
        """
        key = self._key(resource, region)
        with self._lock:
            if force_refresh and key in self._store:
                logger.debug("Cache force-refresh: evicting %s", key)
                del self._store[key]
                return None
            entry = self._store.get(key)
            if entry is None:
                return None
            items, fetched_at = entry
            # Entries with an active refresher are never expired by TTL.
            if key not in self._refreshers:
                age = time.monotonic() - fetched_at
                if age > self._ttl:
                    logger.debug(
                        "Cache TTL expired (%.0fs > %.0fs): evicting %s",
                        age,
                        self._ttl,
                        key,
                    )
                    del self._store[key]
                    return None
            return items

    def set(
        self,
        resource: str,
        region: str | None,
        items: list[Any],
        *,
        refresh_fn: Callable[[], list[Any]] | None = None,
    ) -> None:
        """Store *items* and optionally start a background refresh thread.

        ``refresh_fn`` is a zero-argument callable that returns a fresh list
        (it should handle its own boto3 client creation). It is called only
        once per unique (resource, region) key — subsequent ``set`` calls with
        a ``refresh_fn`` are silently ignored if a thread is already running.
        """
        key = self._key(resource, region)
        with self._lock:
            self._store[key] = (items, time.monotonic())
            if refresh_fn is not None and key not in self._refreshers:
                self._refreshers.add(key)
                self._start_refresh_thread(key, resource, region, refresh_fn)
        logger.debug("Cache set: %s (%d items)", key, len(items))

    def invalidate(
        self,
        resource: str | None = None,
        region: str | None = None,
    ) -> list[str]:
        """Evict matching store entries (background threads keep running).

        The next ``get`` will return ``None``; the caller re-fetches
        synchronously. If a background thread is running it will re-populate
        the cache on its next tick anyway.

        Pass ``resource=None`` to clear all entries.
        Returns the list of evicted cache keys.
        """
        with self._lock:
            if resource is None:
                evicted = list(self._store.keys())
                self._store.clear()
                logger.info("Cache cleared: %d entries evicted", len(evicted))
                return evicted
            key = self._key(resource, region)
            if key in self._store:
                del self._store[key]
                logger.info("Cache invalidated: %s", key)
                return [key]
            return []

    def info(self) -> dict[str, Any]:
        """Snapshot of current cache state for diagnostics."""
        now = time.monotonic()
        with self._lock:
            return {
                k: {
                    "item_count": len(items),
                    "age_s": round(now - fetched_at, 1),
                    "background_refresh": k in self._refreshers,
                    "refresh_interval_s": round(self._refresh_interval, 0),
                }
                for k, (items, fetched_at) in self._store.items()
            }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _start_refresh_thread(
        self,
        key: str,
        resource: str,
        region: str | None,
        refresh_fn: Callable[[], list[Any]],
    ) -> None:
        """Start a daemon thread that refreshes *key* every ``refresh_interval`` s.

        Must be called with ``self._lock`` held.
        """
        interval = self._refresh_interval

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    new_items = refresh_fn()
                    with self._lock:
                        self._store[key] = (new_items, time.monotonic())
                    logger.info(
                        "Background refresh OK: %s (%d items)", key, len(new_items)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Background refresh failed: %s — %s", key, exc)

        t = threading.Thread(
            target=_loop,
            name=f"resource-cache-{key}",
            daemon=True,  # auto-dies on process exit; restart = fresh start
        )
        t.start()
        logger.info(
            "Background refresh thread started: %s (interval %.0fs)", key, interval
        )

    @staticmethod
    def _key(resource: str, region: str | None) -> str:
        return f"{resource}:{region or 'default'}"


# Module-level singleton — shared across all search modules in the same process.
_cache = ResourceCache()


# ── Public helpers (thin wrappers over the singleton) ─────────────────────────


def get(resource: str, region: str | None = None, *, force_refresh: bool = False) -> list[Any] | None:
    return _cache.get(resource, region, force_refresh=force_refresh)


def set(  # noqa: A001
    resource: str,
    region: str | None,
    items: list[Any],
    *,
    refresh_fn: Callable[[], list[Any]] | None = None,
) -> None:
    _cache.set(resource, region, items, refresh_fn=refresh_fn)


def invalidate(resource: str | None = None, region: str | None = None) -> list[str]:
    return _cache.invalidate(resource, region)


def info() -> dict[str, Any]:
    return _cache.info()


# ── Agent-callable tool ───────────────────────────────────────────────────────


@tool
def invalidate_resource_cache(resource: str = "all") -> dict[str, Any]:
    """Invalidate the in-process AWS resource list cache.

    Forces the next search to re-fetch live data from AWS immediately instead
    of waiting for the background refresh cycle. Use when AWS state has just
    changed (new deploy, new state machine, Lambda added) and you need results
    that reflect it right now.

    Background refresh threads (if running) are NOT stopped — they will
    repopulate the cache on their next scheduled tick regardless.

    Args:
        resource: Which resource family to invalidate.
            ``"sfn"``    — Step Functions state machine list only.
            ``"lambda"`` — Lambda function list only.
            ``"all"``    — Clear everything (default).

    Returns:
        dict with ``evicted_keys`` and ``cache_after`` snapshot.
    """
    target = None if resource == "all" else resource
    evicted = invalidate(target)
    return {
        "invalidated": resource,
        "evicted_keys": evicted,
        "cache_after": info(),
    }
