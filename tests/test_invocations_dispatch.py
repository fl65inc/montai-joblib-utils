"""Unit tests for /invocations action dispatcher (no AWS calls)."""

from __future__ import annotations

from unittest.mock import patch

from montai_joblib_utils_ai_mcp.api import _dispatch_invocation


def test_dispatch_default_discover():
    with patch(
        "montai_joblib_utils_ai_mcp.api.discover_compute_impl",
        return_value={"count": 1, "results": [{"name": "acn"}]},
    ) as mock_discover:
        out = _dispatch_invocation({"prompt": "acn_predictions", "job_type": "batch"})
    assert out["count"] == 1
    mock_discover.assert_called_once()
    kwargs = mock_discover.call_args.kwargs
    assert kwargs["query"] == "acn_predictions"
    assert kwargs["job_type"] == "batch"


def test_dispatch_search_batch_runs():
    with patch(
        "montai_joblib_utils_ai_mcp.api.search_batch_runs_impl",
        return_value={"count": 2, "results": []},
    ) as mock_runs:
        out = _dispatch_invocation(
            {
                "action": "search_batch_runs",
                "query": "acn",
                "statuses": ["FAILED"],
                "max_results": 5,
            }
        )
    assert out["count"] == 2
    mock_runs.assert_called_once()
    kwargs = mock_runs.call_args.kwargs
    assert kwargs["query"] == "acn"
    assert kwargs["statuses"] == ["FAILED"]
    assert kwargs["max_results"] == 5


def test_dispatch_state_machine_tree():
    with patch(
        "montai_joblib_utils_ai_mcp.api.describe_state_machine_tree_impl",
        return_value={"leaves": []},
    ) as mock_tree:
        out = _dispatch_invocation(
            {
                "action": "describe_state_machine_tree",
                "state_machine_name": "matrix-foo",
            }
        )
    assert out == {"leaves": []}
    mock_tree.assert_called_once_with(
        state_machine_arn=None,
        state_machine_name="matrix-foo",
    )


def test_dispatch_unknown_action():
    out = _dispatch_invocation({"action": "submit_job", "prompt": "nope"})
    assert "error" in out
    assert "discover" in out["supported_actions"]
