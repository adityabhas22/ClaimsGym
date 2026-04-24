from claimsops_env.agent_interface import RolloutRunner, parse_action_text, render_observation
from claimsops_env.environment import ClaimsOpsEnv
from claimsops_env.policies import ScriptedBaselinePolicy


def test_parse_action_text_accepts_fenced_json() -> None:
    action = parse_action_text('```json\n{"tool":"get_policy","args":{"policy_id":"POL-1"}}\n```')

    assert isinstance(action, dict)
    assert action["tool"] == "get_policy"


def test_render_observation_includes_catalog_not_hidden_truth() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=1)
    rendered = render_observation(observation)

    assert "action_catalog" in rendered
    assert "expected_payable" not in rendered
    assert "fraud_suspicious" not in rendered


def test_rollout_runner_is_shared_baseline_harness() -> None:
    result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=1, scenario_family="covered_collision")

    assert result.steps > 0
    assert result.reward_breakdown
    assert result.trajectory[0].next_observation
