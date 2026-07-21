"""Deterministic name / text matching helpers for compute search."""

from __future__ import annotations


def matches_query(query: str | None, *haystacks: str | None) -> bool:
    """Return True if query is empty or appears (case-insensitive) in any haystack."""
    if not query or not query.strip():
        return True
    needle = query.strip().lower()
    for hay in haystacks:
        if hay and needle in hay.lower():
            return True
    return False


def tag_values(tags: list[dict] | dict | None) -> list[str]:
    """Normalize AWS tag shapes (list of {Key,Value} or dict) into searchable strings."""
    if not tags:
        return []
    if isinstance(tags, dict):
        return [f"{k}={v}" for k, v in tags.items()]
    out: list[str] = []
    for tag in tags:
        key = tag.get("Key") or tag.get("key") or ""
        value = tag.get("Value") or tag.get("value") or ""
        out.append(f"{key}={value}")
    return out
