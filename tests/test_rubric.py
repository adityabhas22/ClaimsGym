from claimsops_env.environment import ClaimsOpsEnv
from claimsops_env.scenario_templates import SCENARIO_FAMILIES, get_template


def test_every_scenario_template_has_workflow_rubric() -> None:
    for family in SCENARIO_FAMILIES:
        rubric = get_template(family).rubric

        assert rubric is not None
        assert rubric.rubric_id == f"rubric.{family}"
        assert {condition.key for condition in rubric.conditions}
        assert any(condition.severity == "forbidden" for condition in rubric.conditions)


def test_rubric_evaluation_is_step_info_not_observation() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=1, scenario_family="covered_collision")
    result = env.step({"tool": "get_policy", "args": {"policy_id": observation.policy_id}})

    assert "rubric_evaluation" in result.info
    assert result.info["rubric_evaluation"]["rubric_id"] == "rubric.covered_collision"
    assert "rubric" not in result.observation.model_dump(mode="json")


def test_direct_final_decision_reports_rubric_misses() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=3, scenario_family="covered_collision")
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
                "rationale": "Premature final decision without investigation.",
                "closure_disposition": "paid_closed",
            },
        }
    )
    evaluation = result.info["rubric_evaluation"]

    assert "coverage_verified" in evaluation["missed_must"]
    assert "estimate_reviewed" in evaluation["missed_must"]
    assert evaluation["score_by_category"]["workflow"] < 1.0


def test_authority_bypass_violates_forbidden_rubric_condition() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=4, scenario_family="authority_threshold")
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
                "rationale": "Trying payment before authority.",
            },
        }
    )

    assert "no_authority_bypass" in result.info["rubric_evaluation"]["violated_forbidden"]


def test_required_document_rubric_uses_material_docs_without_name_leak() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=5, scenario_family="missing_police_report")
    result = env.step({"tool": "get_policy", "args": {"policy_id": observation.policy_id}})
    rendered = str(result.info["rubric_evaluation"]).lower()

    assert "docs_requested" in rendered
    assert "required_documents" not in rendered
    assert "hidden" not in rendered
