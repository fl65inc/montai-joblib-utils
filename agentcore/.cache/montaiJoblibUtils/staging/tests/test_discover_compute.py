from unittest.mock import patch

from montai_joblib_utils_ai_mcp.tools.discover_compute import discover_compute_impl


@patch("montai_joblib_utils_ai_mcp.tools.discover_compute.search_compute_impl")
def test_known_type_uses_targeted(mock_search):
    mock_search.return_value = {
        "job_type": "sagemaker_pipeline",
        "count": 1,
        "results": [{"name": "MedChemTractabilityPipeline", "job_type": "sagemaker_pipeline"}],
    }
    out = discover_compute_impl("tractability inference pipeline")
    assert out["strategy"] == "targeted"
    assert out["classification"]["job_type"] == "sagemaker_pipeline"
    mock_search.assert_called_once()
    assert out["primary"]["name"] == "MedChemTractabilityPipeline"


@patch("montai_joblib_utils_ai_mcp.tools.discover_compute._fanout_workers")
def test_unknown_uses_parallel_fanout(mock_fanout):
    mock_fanout.return_value = {
        "lambda": {
            "job_type": "lambda",
            "count": 1,
            "results": [{"name": "helper", "job_type": "lambda"}],
        },
        "batch": {
            "job_type": "batch",
            "count": 0,
            "results": [],
            "skipped": True,
        },
        "sagemaker_pipeline": {
            "job_type": "sagemaker_pipeline",
            "count": 1,
            "results": [{"name": "Pipe", "job_type": "sagemaker_pipeline"}],
        },
        "stepfunctions": {"job_type": "stepfunctions", "count": 0, "results": []},
    }
    out = discover_compute_impl("tractability")
    assert out["strategy"] == "parallel_fanout"
    mock_fanout.assert_called_once()
    assert out["primary"]["job_type"] == "sagemaker_pipeline"


@patch("montai_joblib_utils_ai_mcp.tools.discover_compute.search_batch_compute_impl")
@patch("montai_joblib_utils_ai_mcp.tools.discover_compute.search_stepfunctions_compute_impl")
@patch("montai_joblib_utils_ai_mcp.tools.discover_compute.search_sagemaker_compute_impl")
@patch("montai_joblib_utils_ai_mcp.tools.discover_compute.search_lambda_compute_impl")
def test_fanout_skips_batch_when_fast_hits(
    mock_lambda, mock_sm, mock_sfn, mock_batch
):
    from montai_joblib_utils_ai_mcp.tools.discover_compute import _fanout_workers

    mock_lambda.return_value = {"job_type": "lambda", "count": 0, "results": []}
    mock_sm.return_value = {
        "job_type": "sagemaker_pipeline",
        "count": 1,
        "results": [{"name": "Pipe", "job_type": "sagemaker_pipeline"}],
    }
    mock_sfn.return_value = {"job_type": "stepfunctions", "count": 0, "results": []}

    out = _fanout_workers("tractability", max_results=5)
    assert out["sagemaker_pipeline"]["count"] == 1
    assert out["batch"].get("skipped") is True
    mock_batch.assert_not_called()
