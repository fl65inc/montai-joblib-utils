"""Unit tests for ASL tree walking (no AWS required)."""

from montai_joblib_utils_ai_mcp.tools.state_machine_tree import (
    describe_state_machine_tree_impl,
)

ACN_ASL = {
    "Comment": "Generate ACN predictions",
    "StartAt": "GenerateBatchConfigs",
    "States": {
        "GenerateBatchConfigs": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:015551053535:function:generate_matrix_configs_lambda",
            "Next": "Generate_ACN_Predictions",
        },
        "Generate_ACN_Predictions": {
            "Type": "Map",
            "Iterator": {
                "StartAt": "ACN_Predictions",
                "States": {
                    "ACN_Predictions": {
                        "Type": "Task",
                        "Resource": "arn:aws:states:::batch:submitJob.sync",
                        "Parameters": {
                            "JobDefinition": (
                                "arn:aws:batch:us-east-1:015551053535:"
                                "job-definition/acn_predictions"
                            ),
                            "JobQueue": (
                                "arn:aws:batch:us-east-1:015551053535:"
                                "job-queue/PE_GPU_QUEUE_g4dn_8xlarge_400GiB"
                            ),
                        },
                        "End": True,
                    }
                },
            },
            "End": True,
        },
    },
}


def test_walk_acn_definition_finds_lambda_and_batch():
    out = describe_state_machine_tree_impl(
        state_machine_name="ACN_Predictions",
        definition=ACN_ASL,
    )
    assert out["tree"]["type"] == "StateMachine"
    assert out["tree"]["start_at"] == "GenerateBatchConfigs"
    kinds = {leaf["job_type"] for leaf in out["leaves"]}
    names = {leaf["name"] for leaf in out["leaves"]}
    assert "lambda" in kinds
    assert "batch" in kinds
    assert "generate_matrix_configs_lambda" in names
    assert "acn_predictions" in names
    # Job queues must not appear as batch leaves.
    assert "PE_GPU_QUEUE_g4dn_8xlarge_400GiB" not in names
    assert out["leaf_count"] == 2
