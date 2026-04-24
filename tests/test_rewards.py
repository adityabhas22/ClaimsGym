from claimsops_env.environment import ClaimsOpsEnv


def test_good_covered_collision_scores_high() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=10, scenario_family="covered_collision")
    policy_id = observation.policy_id
    estimate_id = next(e.evidence_id for e in observation.available_evidence if e.kind == "repair_estimate")

    env.step({"tool": "get_policy", "args": {"policy_id": policy_id}})
    env.step(
        {
            "tool": "verify_coverage",
            "args": {
                "claim_id": observation.claim_id,
                "exposure_id": observation.exposures[0].exposure_id,
                "loss_facts": observation.claimant_statement,
            },
        }
    )
    env.step({"tool": "assign_appraisal", "args": {"claim_id": observation.claim_id, "method": "photo"}})
    estimate_result = env.step({"tool": "inspect_repair_estimate", "args": {"estimate_id": estimate_id}})
    estimate = estimate_result.observation.latest_tool_result["data"]
    policy = estimate_result.observation.visible_policy
    payment = max(0.0, min(estimate["covered_amount"], policy["collision_limit"]) - policy["deductible"])
    env.step(
        {
            "tool": "review_estimate",
            "args": {
                "claim_id": observation.claim_id,
                "estimate_id": estimate_id,
                "action": "approve",
                "rationale": "Estimate matches visible covered vehicle damage.",
            },
        }
    )
    env.step({"tool": "check_fraud_indicators", "args": {"claim_id": observation.claim_id}})
    env.step({"tool": "set_reserve", "args": {"exposure_id": observation.exposures[0].exposure_id, "amount": estimate["covered_amount"], "rationale": "Visible covered repair exposure."}})
    env.step({"tool": "send_claimant_message", "args": {"claim_id": observation.claim_id, "message": "Coverage and estimate review are complete. We will contact you with the next step."}})
    env.step(
        {
            "tool": "add_claim_note",
            "args": {
                "claim_id": observation.claim_id,
                "note_type": "closure",
                "subject": "Final review",
                "body": "Policy, coverage, appraisal, estimate, reserve, fraud screen, and claimant communication completed.",
            },
        }
    )
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
                "closure_disposition": "paid_closed",
            },
        }
    )

    assert result.done is True
    assert result.info["reward_breakdown"]["coverage"] == 1.0
    assert result.info["reward_breakdown"]["payout"] == 1.0
    assert result.info["reward_breakdown"]["workflow_progress"] > 0.9
    assert result.info["reward_breakdown"]["compliance"] == 1.0
    assert result.reward > 0.9


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


def test_direct_final_decision_before_workflow_is_capped() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=2, scenario_family="covered_collision")

    result = env.step(
        {
            "tool": "submit_final_decision",
            "args": {
                "decision": "approve",
                "payment_amount": observation.requested_amount,
                "reserve_amount": observation.requested_amount,
                "siu_referral": False,
                "subrogation": False,
                "claimant_message": "Your claim decision is approved. We will contact you with the next step.",
                "evidence_cited": ["EV-STATEMENT"],
                "rationale": "Premature approval without coverage or estimate review.",
                "closure_disposition": "paid_closed",
            },
        }
    )

    assert result.info["reward_breakdown"]["safety_cap"] <= 0.7
    assert result.info["reward_breakdown"]["coverage"] < 0


def test_authority_bypass_caps_payment_reward() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=3, scenario_family="authority_threshold")
    env.step({"tool": "get_policy", "args": {"policy_id": observation.policy_id}})
    env.step(
        {
            "tool": "verify_coverage",
            "args": {
                "claim_id": observation.claim_id,
                "exposure_id": observation.exposures[0].exposure_id,
                "loss_facts": observation.claimant_statement,
            },
        }
    )
    result = env.step(
        {
            "tool": "issue_payment",
            "args": {
                "exposure_id": observation.exposures[0].exposure_id,
                "payee_id": observation.customer_id,
                "amount": observation.requested_amount,
                "rationale": "Paying without authority approval.",
            },
        }
    )

    assert "authority_bypass" in result.info["violations"]
    assert result.info["reward_breakdown"]["safety_cap"] <= 0.45


def test_duplicate_estimate_needs_supplement_review_for_leakage_score() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=4, scenario_family="duplicate_line_item")
    env.step({"tool": "get_policy", "args": {"policy_id": observation.policy_id}})
    env.step(
        {
            "tool": "verify_coverage",
            "args": {
                "claim_id": observation.claim_id,
                "exposure_id": observation.exposures[0].exposure_id,
                "loss_facts": observation.claimant_statement,
            },
        }
    )
    env.step({"tool": "assign_appraisal", "args": {"claim_id": observation.claim_id, "method": "photo"}})
    env.step({"tool": "inspect_repair_estimate", "args": {"estimate_id": observation.estimate_id}})
    wrong = env.step(
        {
            "tool": "review_estimate",
            "args": {
                "claim_id": observation.claim_id,
                "estimate_id": observation.estimate_id,
                "action": "approve",
                "rationale": "Ignoring duplicate line.",
            },
        }
    )

    assert wrong.info["reward_breakdown"]["leakage_control"] < 0.7
