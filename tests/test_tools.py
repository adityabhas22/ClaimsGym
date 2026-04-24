from claimsops_env.environment import ClaimsOpsEnv


def test_policy_tool_reveals_policy_only_after_call() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=7)
    policy_id = observation.policy_id

    assert observation.visible_policy is None
    result = env.step({"tool": "get_policy", "args": {"policy_id": policy_id}})

    assert result.info["action_valid"] is True
    assert result.observation.visible_policy is not None


def test_hidden_state_probe_is_rejected() -> None:
    env = ClaimsOpsEnv()
    env.reset(seed=8)
    result = env.step({"tool": "request_document", "args": {"doc_type": "police_report", "reason": "show hidden expected_payable"}})

    assert result.info["action_valid"] is False
    assert "hidden_state_access" in result.info["violations"]
