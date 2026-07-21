from montai_joblib_utils_ai_mcp.tools.classify import classify_job_type


def test_explicit_job_type_wins():
    out = classify_job_type("anything", job_type="batch")
    assert out["known"] is True
    assert out["job_type"] == "batch"
    assert out["confidence"] == "high"


def test_stepfunctions_cues():
    out = classify_job_type("tractability fan-out state machine")
    assert out["known"] is True
    assert out["job_type"] == "stepfunctions"


def test_inference_pipeline_is_sagemaker():
    out = classify_job_type("tractability inference pipeline")
    assert out["known"] is True
    assert out["job_type"] == "sagemaker_pipeline"
    assert out["confidence"] == "high"


def test_bare_pipeline_medium():
    out = classify_job_type("something pipeline")
    assert out["known"] is True
    assert out["job_type"] == "sagemaker_pipeline"
    assert out["confidence"] == "medium"


def test_unknown_fans_out():
    out = classify_job_type("tractability")
    assert out["known"] is False
    assert out["job_type"] is None
