from claimsops_env.environment import ClaimsOpsEnv
from claimsops_env.generator import ScenarioGenerator


def test_estimate_line_items_reconcile_to_gross_and_payable_amount() -> None:
    generator = ScenarioGenerator()

    for family in ["covered_collision", "prior_damage_leakage", "duplicate_line_item", "rental_storage_leakage"]:
        spec = generator.generate_family(family, seed=11)
        line_total = round(sum(line.amount for line in spec.repair_estimate.line_items), 2)
        payable_total = round(sum(line.amount for line in spec.repair_estimate.line_items if line.payable), 2)

        assert line_total == round(spec.repair_estimate.gross_amount, 2)
        assert payable_total == round(spec.repair_estimate.covered_amount, 2)


def test_observation_exposes_claim_documents_without_hidden_truth() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=5, scenario_family="ownership_gap")
    text = str(observation.model_dump(mode="json")).lower()

    assert observation.claim_documents
    assert any(document.doc_type.value == "repair_estimate_breakdown" for document in observation.claim_documents)
    assert "expected_payable" not in text
    assert "required_documents" not in text
    assert "hidden" not in text


def test_document_request_lifecycle_moves_to_reviewed_after_inspection() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=7, scenario_family="missing_police_report")

    requested = env.step(
        {
            "tool": "request_document",
            "args": {"doc_type": "police_report", "reason": "Needed to confirm liability."},
        }
    ).observation
    assert any(document.doc_type.value == "police_report" and document.status == "requested" for document in requested.claim_documents)

    received = env.step({"tool": "check_fraud_indicators", "args": {"claim_id": observation.claim_id}}).observation
    police_doc = next(document for document in received.claim_documents if document.doc_type.value == "police_report")

    assert police_doc.status == "received"
    assert police_doc.evidence_id == "EV-DOC-POLICE_REPORT"

    reviewed = env.step({"tool": "inspect_evidence", "args": {"evidence_id": "EV-DOC-POLICE_REPORT"}}).observation
    police_doc = next(document for document in reviewed.claim_documents if document.doc_type.value == "police_report")

    assert police_doc.status == "reviewed"


def test_wrong_estimate_review_marks_nonpayable_line_as_mishandled() -> None:
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
    env.step({"tool": "inspect_repair_estimate", "args": {"estimate_id": observation.estimate_id}})
    result = env.step(
        {
            "tool": "review_estimate",
            "args": {
                "claim_id": observation.claim_id,
                "estimate_id": observation.estimate_id,
                "action": "approve",
                "rationale": "Incorrectly approving the duplicate line.",
            },
        }
    )
    inspected = env.step({"tool": "inspect_repair_estimate", "args": {"estimate_id": observation.estimate_id}})

    duplicate_lines = [
        line
        for line in inspected.observation.latest_tool_result["data"]["line_items"]
        if "duplicate_line" in line["flags"]
    ]
    assert duplicate_lines
    assert duplicate_lines[0]["review_status"] == "approved"
    assert result.info["reward_breakdown"]["leakage_control"] < 0.55
