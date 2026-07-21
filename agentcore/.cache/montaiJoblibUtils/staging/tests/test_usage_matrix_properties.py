"""Usage tests: discover_compute against matrix properties.

Run with AWS credentials:
  AWS_PROFILE=legacy-admin uv run pytest -m usage -q -k batch1

Batches of ~5 keep fan-out load and logs readable. After each batch, inspect
traces (strategy / primary / leaves) and tighten matching.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from montai_joblib_utils_ai_mcp.tools.discover_compute import discover_compute_impl
from montai_joblib_utils_ai_mcp.tools.state_machine_tree import (
    describe_state_machine_tree_impl,
)
from tests.fixtures.matrix_properties import MATRIX_PROPERTIES_50

pytestmark = pytest.mark.usage

TRACE_DIR = Path(__file__).resolve().parent / "_usage_traces"
TRACE_DIR.mkdir(exist_ok=True)

# Batches of 5 for incremental runs
BATCHES: list[list[str]] = [
    MATRIX_PROPERTIES_50[i : i + 5] for i in range(0, len(MATRIX_PROPERTIES_50), 5)
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _related(prop: str, hit_name: str | None) -> bool:
    if not hit_name:
        return False
    a, b = _norm(prop), _norm(hit_name)
    return a in b or b in a


def _write_trace(prop: str, payload: dict) -> None:
    path = TRACE_DIR / f"{prop}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))


@pytest.fixture(scope="module", autouse=True)
def _require_aws():
    if os.environ.get("SKIP_USAGE_TESTS") == "1":
        pytest.skip("SKIP_USAGE_TESTS=1")
    # Soft check — discover will error clearly if creds missing
    yield


def _run_discover(prop: str) -> dict:
    out = discover_compute_impl(prop)
    # If primary is a state machine, ensure tree is present (or fetch it).
    primary = out.get("primary") or {}
    if primary.get("job_type") in ("stepfunctions", "state_machine") and not primary.get(
        "tree"
    ):
        tree = describe_state_machine_tree_impl(
            state_machine_arn=primary.get("arn"),
            state_machine_name=primary.get("name"),
        )
        primary = {**primary, "tree": tree.get("tree"), "leaves": tree.get("leaves")}
        out["primary"] = primary
    _write_trace(
        prop,
        {
            "query": prop,
            "strategy": out.get("strategy"),
            "elapsed_ms": out.get("elapsed_ms"),
            "classification": out.get("classification"),
            "search_query": out.get("search_query"),
            "count": out.get("count"),
            "primary": {
                "name": primary.get("name"),
                "job_type": primary.get("job_type"),
                "arn": primary.get("arn"),
                "leaf_count": primary.get("leaf_count"),
                "leaves": primary.get("leaves"),
            },
            "hit_names": [h.get("name") for h in out.get("hits") or []],
            "batch_skipped": (out.get("by_job_type") or {}).get("batch", {}).get("skipped"),
        },
    )
    return out


def _assert_usable(prop: str, out: dict) -> None:
    assert out.get("error") is None, out
    assert out["elapsed_ms"] < 30_000, f"{prop} too slow: {out['elapsed_ms']}ms"
    primary = out.get("primary")
    hits = out.get("hits") or []
    assert primary or hits, f"{prop}: no hits"
    names = [primary.get("name") if primary else None] + [h.get("name") for h in hits]
    assert any(_related(prop, n) for n in names), (
        f"{prop}: no related hit among {names}. strategy={out.get('strategy')}"
    )
    if primary and primary.get("job_type") in ("stepfunctions", "state_machine"):
        assert primary.get("tree"), f"{prop}: state machine primary missing tree"
        assert primary.get("leaf_count", 0) >= 1 or (primary.get("leaves") is not None)


# Use zero-padded names so `-k batch_01` does not also match batch_10.
@pytest.mark.parametrize("prop", BATCHES[0])
def test_usage_batch_01(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[1])
def test_usage_batch_02(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[2])
def test_usage_batch_03(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[3])
def test_usage_batch_04(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[4])
def test_usage_batch_05(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[5])
def test_usage_batch_06(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[6])
def test_usage_batch_07(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[7])
def test_usage_batch_08(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[8])
def test_usage_batch_09(prop: str):
    _assert_usable(prop, _run_discover(prop))


@pytest.mark.parametrize("prop", BATCHES[9])
def test_usage_batch_10(prop: str):
    _assert_usable(prop, _run_discover(prop))
