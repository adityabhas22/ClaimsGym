from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol

from claimsops_env.models import (
    ActionRecord,
    Decision,
    DocumentType,
    EstimateReviewDecision,
    FinalDecision,
    HiddenTruth,
    PlatformState,
    RepairEstimate,
    RubricEvaluation,
    RewardBreakdown,
    WorkflowState,
    WorkflowRubric,
)
from claimsops_env.rubric import category_score, evaluate_rubric
from claimsops_env.tools import reserve_band_for


@dataclass(frozen=True)
class RewardContext:
    hidden: HiddenTruth
    workflow: WorkflowState
    final_decision: FinalDecision | None
    approved_payment: float | None
    reserve_amount: float | None
    platform_state: PlatformState
    repair_estimate: RepairEstimate
    rubric: WorkflowRubric
    rubric_evaluation: RubricEvaluation | None
    action_log: list[ActionRecord]
    evidence_ids: set[str]
    requested_documents: list[DocumentType | str]
    received_documents: list[DocumentType | str]
    remaining_steps: int
    step_budget: int
    violations: list[str]
    valid_format: bool


class RewardComponent(Protocol):
    name: str

    def score(self, context: RewardContext) -> float:
        ...


class WorkflowProgressReward:
    name = "workflow_progress"

    def score(self, context: RewardContext) -> float:
        milestones = [
            context.workflow.policy_seen,
            context.platform_state.coverage_verified,
            context.workflow.appraisal_assigned,
            context.workflow.estimate_seen,
            context.workflow.estimate_reviewed,
            bool(context.platform_state.reserve_lines),
            context.workflow.fraud_checked,
            context.workflow.claimant_updated,
            context.workflow.closure_note_added or context.final_decision is not None,
            context.final_decision is not None,
        ]
        if context.hidden.required_documents:
            requested = {_doc_value(doc) for doc in context.requested_documents}
            received = {_doc_value(doc) for doc in context.received_documents}
            required = {doc.value for doc in context.hidden.required_documents}
            milestones.append(required.issubset(requested))
            milestones.append(required.issubset(received))
        if context.hidden.subrogation_expected:
            milestones.append(context.workflow.subrogation_opened)
        if context.hidden.authority_escalation_required:
            milestones.append(context.workflow.authority_requested)
            milestones.append(context.workflow.authority_approved)
        legacy_score = sum(1 for item in milestones if item) / len(milestones)
        return 0.75 * legacy_score + 0.25 * _rubric_category(context, "workflow")


class CoverageReward:
    name = "coverage"

    def score(self, context: RewardContext) -> float:
        decision = context.final_decision
        if decision is None:
            return 0.0
        if not context.platform_state.coverage_verified:
            return -0.5
        if context.hidden.authority_escalation_required:
            if decision.decision == Decision.ESCALATE:
                return 1.0
            if context.workflow.authority_requested and decision.decision in {Decision.APPROVE, Decision.APPROVE_PARTIAL}:
                return 1.0
            return -0.5
        if context.hidden.is_covered:
            if decision.decision in {Decision.APPROVE, Decision.APPROVE_PARTIAL}:
                return 1.0
            return -1.0
        if decision.decision == Decision.DENY:
            clause_ok = not context.hidden.denial_clause or context.hidden.denial_clause in decision.rationale
            return 1.0 if clause_ok else 0.5
        return -1.0


class PayoutReward:
    name = "payout"

    def score(self, context: RewardContext) -> float:
        decision = context.final_decision
        if decision is None:
            return 0.0
        if not context.workflow.estimate_reviewed:
            return 0.0
        expected = context.hidden.expected_payable
        actual = decision.payment_amount
        tolerance = max(250.0, expected * 0.08)
        if expected == actual:
            return 1.0
        return max(0.0, 1.0 - abs(actual - expected) / tolerance)


class EvidenceReward:
    name = "evidence"

    def score(self, context: RewardContext) -> float:
        required = {doc.value for doc in context.hidden.required_documents}
        requested = {_doc_value(doc) for doc in context.requested_documents}
        received = {_doc_value(doc) for doc in context.received_documents}
        if required:
            requested_score = len(required & requested) / len(required)
            received_score = len(required & received) / len(required)
            reviewed = {
                document.doc_type.value
                for document in context.platform_state.documents
                if document.status == "reviewed"
            }
            reviewed_score = len(required & reviewed) / len(required)
            completeness = 0.45 * requested_score + 0.35 * received_score + 0.20 * reviewed_score
        else:
            completeness = 1.0
        unnecessary = len(requested - required)
        premature_payment = bool(
            context.final_decision
            and context.final_decision.payment_amount > 0
            and not required.issubset(received)
        )
        legacy_score = max(0.0, completeness - 0.05 * unnecessary - (0.25 if premature_payment else 0.0))
        return 0.8 * legacy_score + 0.2 * _rubric_category(context, "evidence")


class LeakageControlReward:
    name = "leakage_control"

    def score(self, context: RewardContext) -> float:
        score = 1.0
        review = context.platform_state.estimate_review_decision
        if context.hidden.expected_estimate_review != EstimateReviewDecision.APPROVE:
            score -= 0.45 if review != context.hidden.expected_estimate_review else 0.0
        elif review not in {EstimateReviewDecision.APPROVE, None}:
            score -= 0.15
        nonpayable_lines = [
            line
            for line in context.repair_estimate.line_items
            if not line.payable or {"duplicate_line", "prior_damage", "not_loss_related"} & set(line.flags)
        ]
        if nonpayable_lines:
            mishandled = [
                line
                for line in nonpayable_lines
                if line.review_status not in {"questioned", "excluded", "supplement_pending"}
            ]
            score -= min(0.35, 0.18 * len(mishandled))

        issued = [payment for payment in context.platform_state.payments if payment.status == "issued"]
        paid = sum(payment.amount for payment in issued)
        if paid and paid > context.hidden.expected_payable + max(250.0, context.hidden.expected_payable * 0.08):
            score -= 0.5
        if context.hidden.expected_total_loss and not context.platform_state.valuation_received:
            score -= 0.25
        if context.platform_state.storage_charges > 150:
            score -= 0.1
        if context.platform_state.rental_days > 2:
            score -= 0.1
        if context.hidden.expected_payable == 0 and paid > 0:
            score -= 0.75
        legacy_score = max(-1.0, score)
        return max(-1.0, 0.85 * legacy_score + 0.15 * _rubric_category(context, "leakage"))


class FraudTriageReward:
    name = "fraud_triage"

    def score(self, context: RewardContext) -> float:
        referred = bool(context.final_decision.siu_referral if context.final_decision else context.workflow.siu_referral)
        suspicious = context.hidden.fraud_suspicious
        if referred and not context.workflow.fraud_checked:
            return -0.5
        if suspicious and referred:
            return 1.0
        if suspicious and not referred:
            return -1.0
        if not suspicious and not referred:
            return 1.0
        return -0.5


class SubrogationReward:
    name = "subrogation"

    def score(self, context: RewardContext) -> float:
        opened = bool(context.final_decision.subrogation if context.final_decision else context.workflow.subrogation_opened)
        expected = context.hidden.subrogation_expected
        if expected and opened:
            return 1.0
        if expected and not opened:
            return -0.75
        if not expected and not opened:
            return 1.0
        return -0.25


class CommunicationReward:
    name = "communication"

    def score(self, context: RewardContext) -> float:
        decision = context.final_decision
        if decision is None:
            return 0.0
        message = decision.claimant_message.lower()
        checks = [
            any(word in message for word in ["approved", "denied", "review", "decision", "coverage"]),
            "deductible" in message if decision.payment_amount > 0 else True,
            any(phrase in message for phrase in ["next step", "we will", "please", "contact", "follow up"]),
            "fraud score" not in message and "siu score" not in message,
            "you committed fraud" not in message and "fraudulent" not in message,
        ]
        if decision.decision == Decision.DENY:
            checks.append(bool(re.search(r"policy|clause|coverage|evidence|period", message)))
        return sum(1 for check in checks if check) / len(checks)


class ReserveReward:
    name = "reserve"

    def score(self, context: RewardContext) -> float:
        amount = context.reserve_amount
        if context.final_decision is not None:
            amount = context.final_decision.reserve_amount
        if amount is None:
            return 0.0
        actual = reserve_band_for(amount)
        expected = context.hidden.expected_reserve_band
        if actual == expected:
            return 1.0
        adjacent = {
            ("low", "medium"),
            ("medium", "low"),
            ("medium", "high"),
            ("high", "medium"),
        }
        return 0.5 if (actual.value, expected.value) in adjacent else 0.0


class ComplianceReward:
    name = "compliance"

    def score(self, context: RewardContext) -> float:
        open_or_overdue = [
            activity
            for activity in context.platform_state.activities
            if _value(activity.status) in {"open", "overdue"}
        ]
        overdue = [activity for activity in open_or_overdue if _value(activity.status) == "overdue"]
        score = 1.0
        score -= min(0.5, 0.1 * len(open_or_overdue))
        score -= min(0.4, 0.2 * len(overdue))
        if context.final_decision and not context.workflow.claimant_updated:
            score -= 0.25
        if context.final_decision and context.platform_state.pending_events:
            score -= 0.35
        if "closed_with_open_activities" in context.violations:
            score -= 0.4
        legacy_score = max(0.0, score)
        return 0.85 * legacy_score + 0.15 * _rubric_category(context, "compliance")


class FinancialControlsReward:
    name = "financial_controls"

    def score(self, context: RewardContext) -> float:
        score = 1.0
        issued = [payment for payment in context.platform_state.payments if payment.status == "issued"]
        blocked = [payment for payment in context.platform_state.payments if payment.status == "blocked_authority"]
        if context.hidden.authority_escalation_required and not context.workflow.authority_requested:
            score -= 0.5
        if context.hidden.authority_escalation_required and context.workflow.authority_requested and not context.workflow.authority_approved:
            score -= 0.25
        if "authority_bypass" in context.violations:
            score -= 0.6
        if "payment_before_coverage" in context.violations:
            score -= 0.4
        if blocked:
            score -= 0.2
        if issued and not context.platform_state.reserve_lines:
            score -= 0.4
        pending = [reserve for reserve in context.platform_state.reserve_lines if reserve.approval_status == "pending_authority"]
        if pending and not context.workflow.authority_requested:
            score -= 0.25
        legacy_score = max(0.0, score)
        return 0.85 * legacy_score + 0.15 * _rubric_category(context, "financial")


class EfficiencyReward:
    name = "efficiency"

    def score(self, context: RewardContext) -> float:
        if context.final_decision is None:
            return 0.0
        used = context.step_budget - context.remaining_steps
        useful_window = max(1, context.step_budget - 3)
        return max(0.0, 1.0 - max(0, used - 4) / useful_window)


class AuditTrailReward:
    name = "audit_trail"

    def score(self, context: RewardContext) -> float:
        decision = context.final_decision
        if decision is None:
            return 0.0
        if not decision.evidence_cited:
            return 0.0
        valid = [evidence_id for evidence_id in decision.evidence_cited if evidence_id in context.evidence_ids]
        citation_score = len(valid) / len(decision.evidence_cited)
        rationale_score = 1.0 if len(decision.rationale.strip()) >= 20 else 0.25
        note_score = 1.0 if context.platform_state.notes else 0.4
        closure_score = 1.0 if context.workflow.closure_note_added or decision.closure_disposition else 0.5
        legacy_score = 0.55 * citation_score + 0.25 * rationale_score + 0.10 * note_score + 0.10 * closure_score
        return 0.85 * legacy_score + 0.15 * _rubric_category(context, "audit")


DEFAULT_COMPONENTS: tuple[RewardComponent, ...] = (
    WorkflowProgressReward(),
    CoverageReward(),
    PayoutReward(),
    EvidenceReward(),
    LeakageControlReward(),
    FraudTriageReward(),
    SubrogationReward(),
    CommunicationReward(),
    ReserveReward(),
    ComplianceReward(),
    FinancialControlsReward(),
    EfficiencyReward(),
    AuditTrailReward(),
)


WEIGHTS = {
    "workflow_progress": 0.08,
    "coverage": 0.14,
    "payout": 0.14,
    "evidence": 0.10,
    "leakage_control": 0.10,
    "fraud_triage": 0.12,
    "subrogation": 0.06,
    "communication": 0.07,
    "reserve": 0.05,
    "compliance": 0.06,
    "financial_controls": 0.05,
    "efficiency": 0.01,
    "audit_trail": 0.02,
}


def score_episode(context: RewardContext) -> RewardBreakdown:
    context = _with_rubric_evaluation(context)
    scores = {component.name: component.score(context) for component in DEFAULT_COMPONENTS}
    breakdown = RewardBreakdown(
        format_validity=1.0 if context.valid_format else -1.0,
        workflow_progress=scores["workflow_progress"],
        coverage=scores["coverage"],
        payout=scores["payout"],
        evidence=scores["evidence"],
        leakage_control=scores["leakage_control"],
        fraud_triage=scores["fraud_triage"],
        subrogation=scores["subrogation"],
        communication=scores["communication"],
        reserve=scores["reserve"],
        compliance=scores["compliance"],
        financial_controls=scores["financial_controls"],
        efficiency=scores["efficiency"],
        audit_trail=scores["audit_trail"],
    )
    weighted = sum(WEIGHTS[name] * scores[name] for name in WEIGHTS)
    penalty = _penalties(context)
    cap = _safety_cap(context)
    breakdown.penalties = penalty
    breakdown.safety_cap = cap
    breakdown.total = min(cap, max(-1.0, weighted - penalty))
    return breakdown


def evaluate_context_rubric(context: RewardContext) -> RubricEvaluation:
    return evaluate_rubric(context.rubric, context)


def _with_rubric_evaluation(context: RewardContext) -> RewardContext:
    if context.rubric_evaluation is not None:
        return context
    return replace(context, rubric_evaluation=evaluate_context_rubric(context))


def _rubric_category(context: RewardContext, category: str) -> float:
    evaluation = context.rubric_evaluation or evaluate_context_rubric(context)
    return category_score(evaluation, category, default=1.0)


def _penalties(context: RewardContext) -> float:
    penalty = 0.0
    if not context.valid_format:
        penalty += 0.2
    if "tool_error" in context.violations:
        penalty += 0.1
    if "invalid_action_loop" in context.violations:
        penalty += 0.5
    if "hidden_state_access" in context.violations:
        penalty += 1.0
    if "authority_bypass" in context.violations:
        penalty += 0.3
    if "payment_before_coverage" in context.violations:
        penalty += 0.2
    if "closed_with_open_activities" in context.violations:
        penalty += 0.2
    return penalty


def _safety_cap(context: RewardContext) -> float:
    cap = 1.0
    if "fabricated_document_id" in context.violations:
        cap = min(cap, 0.2)
    if "hidden_state_access" in context.violations:
        cap = min(cap, 0.2)
    if "authority_bypass" in context.violations:
        cap = min(cap, 0.45)
    if "payment_before_coverage" in context.violations:
        cap = min(cap, 0.5)
    decision = context.final_decision
    if decision is None:
        return cap
    if not context.platform_state.coverage_verified or not context.workflow.estimate_reviewed:
        cap = min(cap, 0.7)
    if context.hidden.expected_payable and decision.payment_amount > context.hidden.expected_payable * 1.25 + 500:
        cap = min(cap, 0.2)
    if context.hidden.is_covered and decision.decision == Decision.DENY and not context.workflow.policy_status_checked:
        cap = min(cap, 0.3)
    required = {doc.value for doc in context.hidden.required_documents}
    requested = {_doc_value(doc) for doc in context.requested_documents}
    received = {_doc_value(doc) for doc in context.received_documents}
    if decision.payment_amount > 0 and not required.issubset(received):
        cap = min(cap, 0.5)
    if "fraud score" in decision.claimant_message.lower() or "siu score" in decision.claimant_message.lower():
        cap = min(cap, 0.4)
    if context.hidden.authority_escalation_required and not context.workflow.authority_requested:
        cap = min(cap, 0.6)
    if context.hidden.authority_escalation_required and context.workflow.authority_requested and not context.workflow.authority_approved:
        cap = min(cap, 0.75)
    if context.platform_state.pending_events:
        cap = min(cap, 0.85)
    if "closed_with_open_activities" in context.violations:
        cap = min(cap, 0.75)
    return cap


def _doc_value(doc: DocumentType | str) -> str:
    return doc.value if isinstance(doc, DocumentType) else str(doc)


def _value(item: object) -> str:
    return item.value if hasattr(item, "value") else str(item)
