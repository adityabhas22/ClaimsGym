from __future__ import annotations

from collections.abc import Callable
from typing import Any

from claimsops_env.models import (
    Decision,
    DocumentType,
    EstimateReviewDecision,
    RubricCheck,
    RubricCondition,
    RubricEvaluation,
    RubricSeverity,
    WorkflowRubric,
)


RubricPredicate = Callable[[Any], tuple[bool, str]]


def evaluate_rubric(rubric: WorkflowRubric, context: Any) -> RubricEvaluation:
    checks = [_evaluate_condition(condition, context) for condition in rubric.conditions]
    score_by_category = _score_by_category(checks)
    total_weight = sum(check.weight for check in checks)
    passed_weight = sum(check.weight for check in checks if check.passed)
    return RubricEvaluation(
        rubric_id=rubric.rubric_id,
        title=rubric.title,
        checks=checks,
        score_by_category=score_by_category,
        overall_score=passed_weight / total_weight if total_weight else 1.0,
        missed_must=[check.key for check in checks if check.severity == RubricSeverity.MUST and not check.passed],
        missed_final=[check.key for check in checks if check.severity == RubricSeverity.FINAL and not check.passed],
        violated_forbidden=[check.key for check in checks if check.severity == RubricSeverity.FORBIDDEN and not check.passed],
    )


def category_score(evaluation: RubricEvaluation, category: str, default: float = 1.0) -> float:
    return evaluation.score_by_category.get(category, default)


def _evaluate_condition(condition: RubricCondition, context: Any) -> RubricCheck:
    predicate = CONDITION_REGISTRY.get(condition.key)
    if predicate is None:
        passed, detail = False, "No predicate registered for rubric condition."
    else:
        passed, detail = predicate(context)
    return RubricCheck(
        key=condition.key,
        category=condition.category,
        severity=condition.severity,
        description=condition.description,
        weight=condition.weight,
        passed=passed,
        detail=detail,
    )


def _score_by_category(checks: list[RubricCheck]) -> dict[str, float]:
    grouped: dict[str, list[RubricCheck]] = {}
    for check in checks:
        grouped.setdefault(check.category.value, []).append(check)
    scores: dict[str, float] = {}
    for category, category_checks in grouped.items():
        total = sum(check.weight for check in category_checks)
        passed = sum(check.weight for check in category_checks if check.passed)
        scores[category] = passed / total if total else 1.0
    return scores


def _required_doc_values(context: Any) -> set[str]:
    return {doc.value for doc in context.hidden.required_documents}


def _requested_doc_values(context: Any) -> set[str]:
    return {_doc_value(doc) for doc in context.requested_documents}


def _received_doc_values(context: Any) -> set[str]:
    return {_doc_value(doc) for doc in context.received_documents}


def _reviewed_doc_values(context: Any) -> set[str]:
    return {
        document.doc_type.value
        for document in context.platform_state.documents
        if document.status == "reviewed"
    }


def _final_message(context: Any) -> str:
    return (context.final_decision.claimant_message if context.final_decision else "").lower()


def _payment_amount(context: Any) -> float:
    return float(context.final_decision.payment_amount if context.final_decision else 0.0)


def _has_final(context: Any) -> bool:
    return context.final_decision is not None


def _condition(value: bool, detail: str) -> tuple[bool, str]:
    return value, detail


def _payout_within_tolerance(context: Any) -> tuple[bool, str]:
    if context.final_decision is None:
        return False, "No final decision."
    expected = context.hidden.expected_payable
    actual = context.final_decision.payment_amount
    tolerance = max(250.0, expected * 0.08)
    return abs(actual - expected) <= tolerance, f"actual={actual:.2f}, expected={expected:.2f}, tolerance={tolerance:.2f}"


def _correct_coverage_decision(context: Any) -> tuple[bool, str]:
    decision = context.final_decision
    if decision is None:
        return False, "No final decision."
    if context.hidden.authority_escalation_required and decision.decision == Decision.ESCALATE:
        return True, "Escalated authority-sensitive claim."
    if context.hidden.is_covered:
        passed = decision.decision in {Decision.APPROVE, Decision.APPROVE_PARTIAL}
        return passed, f"covered={context.hidden.is_covered}, decision={decision.decision.value}"
    passed = decision.decision == Decision.DENY
    return passed, f"covered={context.hidden.is_covered}, decision={decision.decision.value}"


def _nonpayable_lines_controlled(context: Any) -> tuple[bool, str]:
    nonpayable = [
        line
        for line in context.repair_estimate.line_items
        if not line.payable or {"duplicate_line", "prior_damage", "not_loss_related"} & set(line.flags)
    ]
    mishandled = [
        line.line_id
        for line in nonpayable
        if line.review_status not in {"questioned", "excluded", "supplement_pending"}
    ]
    return not mishandled, f"mishandled_nonpayable_lines={mishandled}"


def _required_requested(context: Any) -> tuple[bool, str]:
    required = _required_doc_values(context)
    missing = sorted(required - _requested_doc_values(context))
    return not missing, f"missing_requested={missing}"


def _required_received(context: Any) -> tuple[bool, str]:
    required = _required_doc_values(context)
    missing = sorted(required - _received_doc_values(context))
    return not missing, f"missing_received={missing}"


def _required_reviewed(context: Any) -> tuple[bool, str]:
    required = _required_doc_values(context)
    missing = sorted(required - _reviewed_doc_values(context))
    return not missing, f"missing_reviewed={missing}"


def _denial_clause_cited(context: Any) -> tuple[bool, str]:
    decision = context.final_decision
    if decision is None:
        return False, "No final decision."
    if decision.decision != Decision.DENY:
        return True, "Not a denial."
    clause = context.hidden.denial_clause
    if not clause:
        return True, "No specific denial clause expected."
    text = f"{decision.rationale} {decision.claimant_message}".lower()
    return clause.lower() in text, f"expected_clause={clause}"


def _valid_evidence_citations(context: Any) -> tuple[bool, str]:
    decision = context.final_decision
    if decision is None:
        return False, "No final decision."
    missing = [evidence_id for evidence_id in decision.evidence_cited if evidence_id not in context.evidence_ids]
    return bool(decision.evidence_cited) and not missing, f"missing_citations={missing}"


def _no_final_payment_before_docs(context: Any) -> tuple[bool, str]:
    if context.final_decision is None or context.final_decision.payment_amount <= 0:
        return True, "No final payment."
    required = _required_doc_values(context)
    missing = sorted(required - _received_doc_values(context))
    return not missing, f"missing_received={missing}"


def _doc_value(doc: DocumentType | str) -> str:
    return doc.value if isinstance(doc, DocumentType) else str(doc)


def _value(item: Any) -> str:
    return item.value if hasattr(item, "value") else str(item)


CONDITION_REGISTRY: dict[str, RubricPredicate] = {
    "policy_retrieved": lambda c: _condition(c.workflow.policy_seen, "workflow.policy_seen"),
    "coverage_verified": lambda c: _condition(c.platform_state.coverage_verified, "platform_state.coverage_verified"),
    "appraisal_assigned": lambda c: _condition(c.workflow.appraisal_assigned, "workflow.appraisal_assigned"),
    "estimate_seen": lambda c: _condition(c.workflow.estimate_seen, "workflow.estimate_seen"),
    "estimate_reviewed": lambda c: _condition(c.workflow.estimate_reviewed, "workflow.estimate_reviewed"),
    "reserve_set": lambda c: _condition(bool(c.platform_state.reserve_lines), "reserve_lines present"),
    "fraud_screened": lambda c: _condition(c.workflow.fraud_checked, "workflow.fraud_checked"),
    "claimant_updated": lambda c: _condition(c.workflow.claimant_updated, "workflow.claimant_updated"),
    "closure_note_added": lambda c: _condition(c.workflow.closure_note_added, "workflow.closure_note_added"),
    "final_decision_submitted": lambda c: _condition(_has_final(c), "final_decision present"),
    "valid_evidence_citations": _valid_evidence_citations,
    "claimant_message_present": lambda c: _condition(bool(_final_message(c).strip()), "claimant_message present"),
    "correct_coverage_decision": _correct_coverage_decision,
    "payout_within_tolerance": _payout_within_tolerance,
    "docs_requested": _required_requested,
    "docs_received": _required_received,
    "docs_reviewed": _required_reviewed,
    "prior_claims_queried": lambda c: _condition(c.workflow.prior_claims_seen, "workflow.prior_claims_seen"),
    "expected_estimate_review": lambda c: _condition(
        c.platform_state.estimate_review_decision == c.hidden.expected_estimate_review,
        f"actual={_value(c.platform_state.estimate_review_decision)}, expected={_value(c.hidden.expected_estimate_review)}",
    ),
    "nonpayable_lines_controlled": _nonpayable_lines_controlled,
    "siu_referral_if_suspicious": lambda c: _condition(
        bool(c.final_decision.siu_referral if c.final_decision else c.workflow.siu_referral),
        "siu referral expected",
    ),
    "no_unnecessary_siu_referral": lambda c: _condition(
        not bool(c.final_decision.siu_referral if c.final_decision else c.workflow.siu_referral),
        "no SIU referral expected",
    ),
    "subrogation_opened_if_expected": lambda c: _condition(
        bool(c.final_decision.subrogation if c.final_decision else c.workflow.subrogation_opened),
        "subrogation expected",
    ),
    "no_unnecessary_subrogation": lambda c: _condition(
        not bool(c.final_decision.subrogation if c.final_decision else c.workflow.subrogation_opened),
        "no subrogation expected",
    ),
    "authority_requested_if_needed": lambda c: _condition(c.workflow.authority_requested, "workflow.authority_requested"),
    "authority_approved_if_needed": lambda c: _condition(c.workflow.authority_approved, "workflow.authority_approved"),
    "valuation_received_if_total_loss": lambda c: _condition(c.platform_state.valuation_received, "platform_state.valuation_received"),
    "denial_clause_cited_if_denied": _denial_clause_cited,
    "no_payment_before_coverage": lambda c: _condition("payment_before_coverage" not in c.violations, "payment_before_coverage violation absent"),
    "no_authority_bypass": lambda c: _condition("authority_bypass" not in c.violations, "authority_bypass violation absent"),
    "no_close_with_open_activities": lambda c: _condition("closed_with_open_activities" not in c.violations, "closed_with_open_activities violation absent"),
    "no_final_with_pending_events": lambda c: _condition(not (_has_final(c) and c.platform_state.pending_events), "no pending events at final"),
    "no_internal_fraud_language": lambda c: _condition("fraud score" not in _final_message(c) and "siu score" not in _final_message(c), "no internal score language"),
    "no_final_payment_before_required_docs": _no_final_payment_before_docs,
}
