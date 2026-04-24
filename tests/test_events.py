from claimsops_env.environment import ClaimsOpsEnv


def test_requested_document_arrives_as_delayed_event() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=2, scenario_family="missing_police_report")

    requested = env.step(
        {
            "tool": "request_document",
            "args": {"doc_type": "police_report", "reason": "Needed for liability and recovery review."},
        }
    ).observation

    assert requested.pending_events
    assert "police_report" not in {doc.value if hasattr(doc, "value") else doc for doc in env.state()["workflow"]["documents_received"]}

    advanced = env.step({"tool": "check_fraud_indicators", "args": {"claim_id": observation.claim_id}}).observation

    assert not advanced.pending_events
    assert "police_report" in {doc.value if hasattr(doc, "value") else doc for doc in env.state()["workflow"]["documents_received"]}
    assert any(evidence.evidence_id == "EV-DOC-POLICE_REPORT" for evidence in advanced.available_evidence)
    assert any(event.event_type == "document_arrival" for event in advanced.event_history)


def test_authority_approval_is_not_immediate() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=3, scenario_family="authority_threshold")

    requested = env.step(
        {
            "tool": "request_authority_approval",
            "args": {
                "exposure_id": observation.exposures[0].exposure_id,
                "amount": observation.requested_amount,
                "rationale": "Above adjuster authority.",
            },
        }
    ).observation

    assert requested.pending_events
    assert env.state()["workflow"]["authority_approved"] is False

    resolved = env.step({"tool": "add_claim_note", "args": {"claim_id": observation.claim_id, "note_type": "financial", "subject": "Awaiting authority", "body": "Manager approval is pending."}}).observation

    assert not resolved.pending_events
    assert env.state()["workflow"]["authority_approved"] is True
    assert any(event.event_type == "authority_decision" for event in resolved.event_history)


def test_valuation_report_arrives_before_total_loss_finalization() -> None:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=4, scenario_family="total_loss")

    requested = env.step({"tool": "request_valuation", "args": {"claim_id": observation.claim_id, "reason": "Near total-loss threshold."}}).observation

    assert requested.pending_events
    assert not any(evidence.evidence_id == "EV-VALUATION" for evidence in requested.available_evidence)

    resolved = env.step({"tool": "add_claim_note", "args": {"claim_id": observation.claim_id, "note_type": "estimate", "subject": "Awaiting valuation", "body": "Total-loss valuation vendor report is pending."}}).observation

    assert not resolved.pending_events
    assert any(evidence.evidence_id == "EV-VALUATION" for evidence in resolved.available_evidence)
