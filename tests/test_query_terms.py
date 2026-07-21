from montai_joblib_utils_ai_mcp.tools.query_terms import extract_search_query


def test_strips_pipeline_cues():
    assert extract_search_query("tractability inference pipeline") == "tractability"


def test_strips_state_machine_cues():
    assert extract_search_query("tractability fan-out state machine") == "tractability"


def test_keeps_bare_workload():
    assert extract_search_query("acn_predictions") == "acn_predictions"
