from unittest.mock import patch

from montai_joblib_utils_ai_mcp.tools.search_compute import search_compute_impl


def test_unknown_job_type():
    result = search_compute_impl(job_type="glue")  # type: ignore[arg-type]
    assert result["count"] == 0
    assert "Unknown job_type" in result["error"]


@patch("montai_joblib_utils_ai_mcp.tools.search_compute.search_lambda_compute_impl")
def test_dispatch_lambda(mock_lambda):
    mock_lambda.return_value = {"job_type": "lambda", "count": 1, "results": []}
    out = search_compute_impl(job_type="lambda", query="cdd", name_prefix="montai")
    mock_lambda.assert_called_once_with(query="cdd", name_prefix="montai", max_results=25)
    assert out["job_type"] == "lambda"


@patch("montai_joblib_utils_ai_mcp.tools.search_compute.search_batch_compute_impl")
def test_dispatch_batch(mock_batch):
    mock_batch.return_value = {"job_type": "batch", "count": 0, "results": []}
    search_compute_impl(job_type="batch", query="acn")
    mock_batch.assert_called_once_with(query="acn", status="ACTIVE", max_results=25)


@patch("montai_joblib_utils_ai_mcp.tools.search_compute.search_sagemaker_compute_impl")
def test_dispatch_sagemaker_variants(mock_sm):
    mock_sm.return_value = {"job_type": "sagemaker_pipeline", "count": 0, "results": []}
    search_compute_impl(job_type="sagemaker_pipeline", query="main")
    mock_sm.assert_called_with(
        query="main", kind="pipeline", max_results=25, status_equals=None
    )

    search_compute_impl(job_type="sagemaker_training", status="Completed")
    mock_sm.assert_called_with(
        query=None, kind="training", max_results=25, status_equals="Completed"
    )

    search_compute_impl(job_type="sagemaker_hyperparameter")
    mock_sm.assert_called_with(
        query=None, kind="hyperparameter", max_results=25, status_equals=None
    )


@patch("montai_joblib_utils_ai_mcp.tools.search_compute.search_stepfunctions_compute_impl")
def test_dispatch_stepfunctions(mock_sfn):
    mock_sfn.return_value = {"job_type": "stepfunctions", "count": 1, "results": []}
    search_compute_impl(job_type="stepfunctions", query="tractability")
    mock_sfn.assert_called_with(
        query="tractability", max_results=25, state_machine_type=None
    )

    search_compute_impl(job_type="state_machine", query="fanout", status="STANDARD")
    mock_sfn.assert_called_with(
        query="fanout", max_results=25, state_machine_type="STANDARD"
    )
