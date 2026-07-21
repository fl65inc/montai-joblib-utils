from montai_joblib_utils_ai_mcp.tools._match import matches_query, tag_values


def test_matches_query_empty_matches_all():
    assert matches_query(None, "acn_predictions")
    assert matches_query("", "anything")
    assert matches_query("  ", "x")


def test_matches_query_substring_case_insensitive():
    assert matches_query("ACN", "acn_predictions")
    assert matches_query("admet", "internal_admet_predictions")
    assert not matches_query("xyz", "acn_predictions")


def test_tag_values_list_and_dict():
    assert tag_values([{"Key": "job_name", "Value": "acn_predictions"}]) == [
        "job_name=acn_predictions"
    ]
    assert tag_values({"framework": "matrix"}) == ["framework=matrix"]
    assert tag_values(None) == []
