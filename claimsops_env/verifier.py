from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from claimsops_env.models import (
    Decision,
    DocumentType,
    FinalDecision,
    HiddenTruth,
    RewardBreakdown,
    WorkflowState,
)
from claimsops_env.tools import reserve_band_for


@dataclass(frozen=True)
class RewardContext:
    hidden: HiddenTruth
    workflow: WorkflowState
    final_decision: FinalDecision | None
    approved_payment: float | None
    reserve_amount: float | None
    evidence_ids: set[str]
    requested_documents: list[DocumentType | str]
    remaining_steps: int
    step_budget: int
    violations: list[str]
    valid_format: bool


class RewardComponent(Protocol):
    name: str

    def score(self, context: RewardContext) -> float:
        ...


class CoverageReward:
    name = "coverage"

    def score(self, context: RewardContext) -> float:
        decision = context.final_decision
        if decision is None:
            return 0.0
        if context.hidden.authority_escalation_required:
            return 1.0 if decision.decision == Decision.ESCALATE else -0.5
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
        if required:
            completeness = len(required & requested) / len(required)
        else:
            completeness = 1.0
        unnecessary = len(requested - required)
        premature_payment = bool(
            context.final_decision
            and context.final_decision.payment_amount > 0
            and not required.issubset(requested)
        )
        return max(0.0, completeness - 0.05 * unnecessary - (0.25 if premature_payment else 0.0))


class FraudTriageReward:
    name = "fraud_triage"

    def score(self, context: RewardContext) -> float:
        referred = bool(context.final_decision.siu_referral if context.final_decision else context.workflow.siu_referral)
        suspicious = context.hidden.fraud_suspicious
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
        return 0.7 * citation_score + 0.3 * rationale_score


DEFAULT_COMPONENTS: tuple[RewardComponent, ...] = (
    CoverageReward(),
    PayoutReward(),
    EvidenceReward(),
    FraudTriageReward(),
    SubrogationReward(),
    CommunicationReward(),
    ReserveReward(),
    EfficiencyReward(),
    AuditTrailReward(),
)


WEIGHTS = {
    "coverage": 0.20,
    "payout": 0.20,
    "evidence": 0.15,
    "fraud_triage": 0.10,
    "subrogation": 0.10,
    "communication": 0.10,
    "reserve": 0.05,
    "efficiency": 0.05,
    "audit_trail": 0.05,
}


def score_episode(context: RewardContext) -> RewardBreakdown:
    scores = {component.name: component.score(context) for component in DEFAULT_COMPONENTS}
    breakdown = RewardBreakdown(
        format_validity=1.0 if context.valid_format else -1.0,
        coverage=scores["coverage"],
        payout=scores["payout"],
        evidence=scores["evidence"],
        fraud_triage=scores["fraud_triage"],
        subrogation=scores["subrogation"],
        communication=scores["communication"],
        reserve=scores["reserve"],
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
    return penalty


def _safety_cap(context: RewardContext) -> float:
    cap = 1.0
    if "fabricated_document_id" in context.violations:
        cap = min(cap, 0.2)
    decision = context.final_decision
    if decision is None:
        return cap
    if context.hidden.expected_payable and decision.payment_amount > context.hidden.expected_payable * 1.25 + 500:
        cap = min(cap, 0.2)
    if context.hidden.is_covered and decision.decision == Decision.DENY and not context.workflow.policy_status_checked:
        cap = min(cap, 0.3)
    required = {doc.value for doc in context.hidden.required_documents}
    requested = {_doc_value(doc) for doc in context.requested_documents}
    if decision.payment_amount > 0 and not required.issubset(requested):
        cap = min(cap, 0.5)
    if "fraud score" in decision.claimant_message.lower() or "siu score" in decision.claimant_message.lower():
        cap = min(cap, 0.4)
    return cap


def _doc_value(doc: DocumentType | str) -> str:
    return doc.value if isinstance(doc, DocumentType) else str(doc)
