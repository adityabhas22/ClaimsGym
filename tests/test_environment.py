from claimsops_env.environment import ClaimsOpsEnv


def test_reset_returns_visible_observation_without_hidden_truth() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=1)
    dumped = observation.model_dump(mode="json")

    assert dumped["claim_id"].startswith("CLM-")
    assert "expected_payable" not in str(dumped)
    assert "fraud_suspicious" not in str(dumped)
    assert "hidden" not in str(dumped).lower()


def test_invalid_json_does_not_mutate_policy_visibility() -> None:
    env = ClaimsOpsEnv()
    env.reset(seed=2)
    result = env.step("{not valid json")

    assert result.info["action_valid"] is False
    assert result.observation.visible_policy is None
    assert result.reward < 0


def test_three_invalid_actions_terminate_episode() -> None:
    env = ClaimsOpsEnv()
    env.reset(seed=3)

    env.step({"tool": "not_a_tool", "args": {}})
    env.step({"tool": "not_a_tool", "args": {}})
    result = env.step({"tool": "not_a_tool", "args": {}})

    assert result.done is True
    assert "invalid_action_loop" in result.info["violations"]


def test_state_excludes_hidden_verifier_labels() -> None:
    env = ClaimsOpsEnv()
    env.reset(seed=4)

    state = env.state()
    text = str(state)

    assert "expected_payable" not in text
    assert "required_documents" not in text
    assert "fraud_suspicious" not in text


def test_open_tasks_do_not_reveal_hidden_required_documents() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=1, scenario_family="suspicious_inception")

    assert "request_police_report" not in observation.open_tasks


def test_pending_events_do_not_expose_hidden_labels() -> None:
    env = ClaimsOpsEnv()
    env.reset(seed=1, scenario_family="missing_police_report")
    result = env.step(
        {
            "tool": "request_document",
            "args": {"doc_type": "police_report", "reason": "Needed for liability review."},
        }
    )

    text = str(result.observation.model_dump(mode="json")).lower()

    assert "expected_payable" not in text
    assert "required_documents" not in text
    assert "fraud_suspicious" not in text
    assert "hidden" not in text
