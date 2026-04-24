from claimsops_env.agent_interface import RolloutRunner
from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.tracing import trace_rollout


def test_trace_rollout_records_reward_deltas_generically() -> None:
    result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=1, scenario_family="covered_collision")
    trace = trace_rollout(result)

    reward_components = {
        delta.component
        for step in trace.steps
        for delta in step.reward_deltas
    }

    assert trace.claim_id.startswith("CLM-")
    assert trace.initial_summary["claim_phase"] == "intake"
    assert "workflow_progress" in reward_components
    assert "total" in reward_components
    assert trace.steps[0].rubric_evaluation["rubric_id"] == "rubric.covered_collision"


def test_trace_rollout_records_workflow_affordance_changes() -> None:
    result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=1, scenario_family="covered_collision")
    trace = trace_rollout(result)
    summaries = [change.summary for step in trace.steps for change in step.state_changes]

    assert any("workflow_affordances.claim_phase" in summary for summary in summaries)


def test_trace_rollout_captures_document_and_event_changes() -> None:
    result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=4, scenario_family="duplicate_line_item")
    trace = trace_rollout(result)
    summaries = [change.summary for step in trace.steps for change in step.state_changes]

    assert any("claim_documents" in summary and "repair_estimate_breakdown" in summary.lower() for summary in summaries)
    assert any("pending_events" in summary for summary in summaries)
    assert any("event_history" in summary and "document" in summary.lower() for summary in summaries)


def test_trace_rollout_captures_estimate_line_item_review_changes() -> None:
    result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=4, scenario_family="duplicate_line_item")
    trace = trace_rollout(result)
    summaries = [change.summary for step in trace.steps for change in step.state_changes]

    assert any("estimate_line_items" in summary and "review_status" in summary for summary in summaries)
    assert any("approved" in summary for summary in summaries)


def test_trace_markdown_does_not_expose_hidden_truth() -> None:
    result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=2, scenario_family="suspicious_inception")
    rendered = trace_rollout(result).to_markdown().lower()

    assert "expected_payable" not in rendered
    assert "required_documents" not in rendered
    assert "fraud_suspicious" not in rendered
    assert "hidden" not in rendered
