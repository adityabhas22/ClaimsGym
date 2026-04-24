from claimsops_env.environment import ClaimsOpsEnv


def test_good_covered_collision_scores_high() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=10, scenario_family="covered_collision")
    policy_id = observation.policy_id
    estimate_id = next(e.evidence_id for e in observation.available_evidence if e.kind == "repair_estimate")

    env.step({"tool": "get_policy", "args": {"policy_id": policy_id}})
    env.step({"tool": "check_policy_status", "args": {"policy_id": policy_id, "loss_date": observation.loss_date.isoformat()}})
    estimate_result = env.step({"tool": "inspect_repair_estimate", "args": {"estimate_id": estimate_id}})
    estimate = estimate_result.observation.latest_tool_result["data"]
    policy = estimate_result.observation.visible_policy
    payment = max(0.0, min(estimate["covered_amount"], policy["collision_limit"]) - policy["deductible"])
    env.step({"tool": "set_reserve", "args": {"amount": estimate["covered_amount"], "rationale": "Visible covered repair exposure."}})
    result = env.step(
        {
            "tool": "submit_final_decision",
            "args": {
                "decision": "approve",
                "payment_amount": payment,
                "reserve_amount": estimate["covered_amount"],
                "siu_referral": False,
                "subrogation": False,
                "claimant_message": "Your claim decision is approved. Payment is issued after the deductible. We will contact you with the next step.",
                "evidence_cited": ["EV-STATEMENT", estimate_id],
                "rationale": "Active policy; estimate reviewed; payment applies deductible and coverage terms.",
            },
        }
    )

    assert result.done is True
    assert result.info["reward_breakdown"]["coverage"] == 1.0
    assert result.info["reward_breakdown"]["payout"] == 1.0
    assert result.reward > 0.75


def test_fabricated_evidence_id_caps_reward() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=10, scenario_family="covered_collision")

    result = env.step(
        {
            "tool": "submit_final_decision",
            "args": {
                "decision": "approve",
                "payment_amount": 1000,
                "reserve_amount": observation.requested_amount,
                "siu_referral": False,
                "subrogation": False,
                "claimant_message": "Your claim decision is approved. We will contact you with the next step.",
                "evidence_cited": ["EV-FAKE"],
                "rationale": "Unsupported fabricated citation.",
            },
        }
    )

    assert result.info["reward_breakdown"]["safety_cap"] <= 0.2
    assert "fabricated_document_id" in result.info["violations"]
