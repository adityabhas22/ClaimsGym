from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from claimsops_env.models import (
    Action,
    Decision,
    DocumentType,
    Evidence,
    EvidenceKind,
    FinalDecision,
    ReserveBand,
    ToolName,
)


class ToolError(ValueError):
    pass


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    terminal: bool = False


class ToolHandler:
    name: ToolName

    def run(self, runtime: "RuntimeView", args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


@dataclass
class RuntimeView:
    spec: Any
    visible_policy: dict[str, Any] | None
    evidence: list[Evidence]
    diary: list[str]
    communications: list[str]
    financial_snapshot: Any
    workflow: Any
    final_decision: FinalDecision | None
    violations: list[str]


class GetPolicyArgs(BaseModel):
    policy_id: str


class CheckPolicyStatusArgs(BaseModel):
    policy_id: str
    loss_date: str


class InspectRepairEstimateArgs(BaseModel):
    estimate_id: str


class RequestDocumentArgs(BaseModel):
    doc_type: DocumentType
    reason: str


class QueryPriorClaimsArgs(BaseModel):
    customer_id: str
    vehicle_id: str


class CheckFraudIndicatorsArgs(BaseModel):
    claim_id: str


class SetReserveArgs(BaseModel):
    amount: float = Field(ge=0)
    rationale: str


class ApprovePaymentArgs(BaseModel):
    amount: float = Field(ge=0)
    coverages: list[str]
    rationale: str


class ReferToSiuArgs(BaseModel):
    reason: str
    evidence_ids: list[str]


class OpenSubrogationArgs(BaseModel):
    target_party: str
    rationale: str


class GetPolicyTool(ToolHandler):
    name = ToolName.GET_POLICY

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = GetPolicyArgs.model_validate(args)
        if parsed.policy_id != runtime.spec.policy.policy_id:
            raise ToolError("policy_id does not match this claim file")
        policy = runtime.spec.policy
        runtime.visible_policy = policy.model_dump(mode="json")
        runtime.workflow.policy_seen = True
        runtime.evidence.append(
            Evidence(
                evidence_id=f"EV-POLICY-{policy.policy_id}",
                kind=EvidenceKind.POLICY,
                summary="Policy declarations and coverage limits reviewed.",
            )
        )
        return ToolResult(ok=True, summary="Policy retrieved.", data=runtime.visible_policy)


class CheckPolicyStatusTool(ToolHandler):
    name = ToolName.CHECK_POLICY_STATUS

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = CheckPolicyStatusArgs.model_validate(args)
        policy = runtime.spec.policy
        if parsed.policy_id != policy.policy_id:
            raise ToolError("policy_id does not match this claim file")
        in_period = policy.effective_date.isoformat() <= parsed.loss_date <= policy.expiration_date.isoformat()
        active_on_loss = policy.status == "active" and in_period
        runtime.workflow.policy_status_checked = True
        result = {
            "status": policy.status,
            "active_on_loss": active_on_loss,
            "effective_date": policy.effective_date.isoformat(),
            "expiration_date": policy.expiration_date.isoformat(),
        }
        return ToolResult(ok=True, summary=f"Policy active on loss: {active_on_loss}.", data=result)


class InspectRepairEstimateTool(ToolHandler):
    name = ToolName.INSPECT_REPAIR_ESTIMATE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = InspectRepairEstimateArgs.model_validate(args)
        estimate = runtime.spec.repair_estimate
        if parsed.estimate_id != estimate.estimate_id:
            raise ToolError("estimate_id does not match this claim file")
        runtime.workflow.estimate_seen = True
        return ToolResult(ok=True, summary="Repair estimate inspected.", data=estimate.model_dump(mode="json"))


class RequestDocumentTool(ToolHandler):
    name = ToolName.REQUEST_DOCUMENT

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = RequestDocumentArgs.model_validate(args)
        runtime.workflow.documents_requested.append(parsed.doc_type)
        if parsed.doc_type in runtime.spec.hidden.required_documents:
            runtime.workflow.documents_received.append(parsed.doc_type)
            runtime.evidence.append(
                Evidence(
                    evidence_id=f"EV-DOC-{parsed.doc_type.value.upper()}",
                    kind=EvidenceKind.REQUESTED_DOCUMENT,
                    summary=f"Requested {parsed.doc_type.value} received and added to file.",
                )
            )
            summary = f"{parsed.doc_type.value} requested and received."
        else:
            summary = f"{parsed.doc_type.value} requested; not required for this scenario."
        runtime.diary.append(f"Document request: {parsed.doc_type.value}. Reason: {parsed.reason}")
        return ToolResult(ok=True, summary=summary, data={"doc_type": parsed.doc_type.value})


class QueryPriorClaimsTool(ToolHandler):
    name = ToolName.QUERY_PRIOR_CLAIMS

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = QueryPriorClaimsArgs.model_validate(args)
        if parsed.customer_id != runtime.spec.policy.customer_id or parsed.vehicle_id != runtime.spec.policy.vehicle_id:
            raise ToolError("customer_id or vehicle_id does not match this claim file")
        runtime.workflow.prior_claims_seen = True
        if runtime.spec.prior_claim_summaries:
            runtime.evidence.append(
                Evidence(
                    evidence_id="EV-PRIOR-CLAIMS",
                    kind=EvidenceKind.PRIOR_CLAIMS,
                    summary="; ".join(runtime.spec.prior_claim_summaries),
                    flags=["prior_damage"],
                )
            )
        return ToolResult(
            ok=True,
            summary="Prior claim search complete.",
            data={"prior_claims": runtime.spec.prior_claim_summaries},
        )


class CheckFraudIndicatorsTool(ToolHandler):
    name = ToolName.CHECK_FRAUD_INDICATORS

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = CheckFraudIndicatorsArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        runtime.workflow.fraud_checked = True
        indicators = []
        if runtime.spec.hidden.fraud_suspicious:
            indicators = ["inception_timing_or_statement_conflict", "requires_siu_review"]
        runtime.evidence.append(
            Evidence(
                evidence_id="EV-FRAUD-INDICATORS",
                kind=EvidenceKind.FRAUD_REPORT,
                summary="Fraud indicator screen completed.",
                flags=indicators,
            )
        )
        return ToolResult(ok=True, summary="Fraud indicator screen complete.", data={"indicators": indicators})


class SetReserveTool(ToolHandler):
    name = ToolName.SET_RESERVE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = SetReserveArgs.model_validate(args)
        runtime.financial_snapshot.reserve_amount = parsed.amount
        band = reserve_band_for(parsed.amount).value
        runtime.diary.append(f"Reserve set to ${parsed.amount:,.2f}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary=f"Reserve set in {band} band.", data={"amount": parsed.amount, "band": band})


class ApprovePaymentTool(ToolHandler):
    name = ToolName.APPROVE_PAYMENT

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = ApprovePaymentArgs.model_validate(args)
        runtime.financial_snapshot.approved_payment = parsed.amount
        runtime.financial_snapshot.payment_coverages = parsed.coverages
        runtime.diary.append(f"Payment approved for ${parsed.amount:,.2f}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary="Payment approval recorded.", data=parsed.model_dump())


class ReferToSiuTool(ToolHandler):
    name = ToolName.REFER_TO_SIU

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = ReferToSiuArgs.model_validate(args)
        existing_ids = {e.evidence_id for e in runtime.evidence}
        missing = [evidence_id for evidence_id in parsed.evidence_ids if evidence_id not in existing_ids]
        if missing:
            runtime.violations.append("fabricated_document_id")
            raise ToolError(f"unknown evidence_ids: {missing}")
        runtime.workflow.siu_referral = True
        runtime.diary.append(f"SIU referral opened. Reason: {parsed.reason}")
        return ToolResult(ok=True, summary="SIU referral opened.", data=parsed.model_dump())


class OpenSubrogationTool(ToolHandler):
    name = ToolName.OPEN_SUBROGATION

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = OpenSubrogationArgs.model_validate(args)
        runtime.workflow.subrogation_opened = True
        runtime.diary.append(f"Subrogation opened against {parsed.target_party}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary="Subrogation file opened.", data=parsed.model_dump())


class SubmitFinalDecisionTool(ToolHandler):
    name = ToolName.SUBMIT_FINAL_DECISION

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = FinalDecision.model_validate(args)
        existing_ids = {e.evidence_id for e in runtime.evidence}
        missing = [evidence_id for evidence_id in parsed.evidence_cited if evidence_id not in existing_ids]
        if missing:
            runtime.violations.append("fabricated_document_id")
            raise ToolError(f"unknown evidence_ids: {missing}")
        runtime.workflow.final_decision_submitted = True
        runtime.final_decision = parsed
        runtime.communications.append(parsed.claimant_message)
        runtime.diary.append(f"Final decision submitted: {parsed.decision.value}.")
        return ToolResult(ok=True, summary="Final decision submitted.", data=parsed.model_dump(), terminal=True)


def reserve_band_for(amount: float) -> ReserveBand:
    if amount < 3500:
        return ReserveBand.LOW
    if amount < 8000:
        return ReserveBand.MEDIUM
    return ReserveBand.HIGH


def build_tool_registry() -> dict[ToolName, ToolHandler]:
    tools: list[ToolHandler] = [
        GetPolicyTool(),
        CheckPolicyStatusTool(),
        InspectRepairEstimateTool(),
        RequestDocumentTool(),
        QueryPriorClaimsTool(),
        CheckFraudIndicatorsTool(),
        SetReserveTool(),
        ApprovePaymentTool(),
        ReferToSiuTool(),
        OpenSubrogationTool(),
        SubmitFinalDecisionTool(),
    ]
    return {tool.name: tool for tool in tools}


def validate_action(raw_action: dict[str, Any] | Action) -> Action:
    if isinstance(raw_action, Action):
        return raw_action
    try:
        return Action.model_validate(raw_action)
    except ValidationError as exc:
        raise ToolError(str(exc)) from exc
