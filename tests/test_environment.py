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


def test_workflow_affordances_surface_visible_blockers_without_hidden_truth() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=1, scenario_family="missing_police_report")
    affordances = observation.workflow_affordances

    assert affordances.claim_phase == "intake"
    assert "coverage_not_verified" in affordances.close_blockers
    assert "coverage" in affordances.recommended_action_categories
    assert affordances.action_availability["get_policy"] is True
    assert affordances.action_availability["submit_final_decision"] is False
    assert affordances.next_due_steps is not None

    text = str(affordances.model_dump()).lower()
    assert "expected_payable" not in text
    assert "required_documents" not in text
    assert "fraud_suspicious" not in text
    assert "hidden" not in text


def test_workflow_affordances_show_waiting_state_after_external_request() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=1, scenario_family="missing_police_report")
    result = env.step(
        {
            "tool": "request_document",
            "args": {"doc_type": "police_report", "reason": "Needed for liability review."},
        }
    )
    affordances = result.observation.workflow_affordances

    assert affordances.claim_phase == "waiting_on_external"
    assert any(item.startswith("document_arrival:") for item in affordances.waiting_on)
    assert "waiting_on_external" in affordances.close_blockers
    assert affordances.action_availability["submit_final_decision"] is False
