from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from claimsops_env.models import (
    Action,
    ActivityStatus,
    AppraisalStatus,
    ClaimNote,
    CoverageType,
    Decision,
    DocumentType,
    EstimateReviewDecision,
    Evidence,
    EvidenceKind,
    Exposure,
    ExposureStatus,
    Evidence,
    FinalDecision,
    NoteType,
    Payment,
    ReserveLine,
    ReserveBand,
    ToolName,
    VendorAssignment,
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
    platform_state: Any
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


class InspectEvidenceArgs(BaseModel):
    evidence_id: str


class RequestDocumentArgs(BaseModel):
    doc_type: DocumentType
    reason: str


class QueryPriorClaimsArgs(BaseModel):
    customer_id: str
    vehicle_id: str


class CheckFraudIndicatorsArgs(BaseModel):
    claim_id: str


class SetReserveArgs(BaseModel):
    exposure_id: str = "EXP-VEHICLE-1"
    cost_type: str = "claim_cost"
    cost_category: str = "auto_body"
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


class CreateOrUpdateExposureArgs(BaseModel):
    exposure_id: str = "EXP-VEHICLE-1"
    coverage: CoverageType
    claimant_id: str
    incident_id: str


class VerifyCoverageArgs(BaseModel):
    claim_id: str
    exposure_id: str = "EXP-VEHICLE-1"
    loss_facts: str


class AssignAppraisalArgs(BaseModel):
    claim_id: str
    method: str = Field(pattern="^(photo|field|shop)$")


class ReviewEstimateArgs(BaseModel):
    claim_id: str
    estimate_id: str
    action: EstimateReviewDecision
    rationale: str


class RequestValuationArgs(BaseModel):
    claim_id: str
    reason: str


class IssuePaymentArgs(BaseModel):
    exposure_id: str = "EXP-VEHICLE-1"
    payee_id: str
    amount: float = Field(ge=0)
    payment_type: str = Field(default="final", pattern="^(partial|final|expense|recovery_refund)$")
    method: str = Field(default="eft", pattern="^(eft|check)$")
    rationale: str


class RequestAuthorityApprovalArgs(BaseModel):
    exposure_id: str = "EXP-VEHICLE-1"
    amount: float = Field(ge=0)
    rationale: str


class SendClaimantMessageArgs(BaseModel):
    claim_id: str
    message: str


class AddClaimNoteArgs(BaseModel):
    claim_id: str
    note_type: NoteType
    subject: str
    body: str
    related_object_id: str | None = None


class CloseClaimArgs(BaseModel):
    claim_id: str
    disposition: str = Field(pattern="^(paid_closed|denied_closed|escalated|withdrawn)$")
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


class GetPolicySnapshotTool(GetPolicyTool):
    name = ToolName.GET_POLICY_SNAPSHOT


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
        _complete_activity(runtime, "estimate", "repair estimate inspected")
        return ToolResult(ok=True, summary="Repair estimate inspected.", data=estimate.model_dump(mode="json"))


class InspectEvidenceTool(ToolHandler):
    name = ToolName.INSPECT_EVIDENCE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = InspectEvidenceArgs.model_validate(args)
        for evidence in runtime.evidence:
            if evidence.evidence_id == parsed.evidence_id:
                if evidence.kind == EvidenceKind.REPAIR_ESTIMATE:
                    runtime.workflow.estimate_seen = True
                    _complete_activity(runtime, "estimate", "repair estimate inspected")
                return ToolResult(ok=True, summary=f"Evidence {parsed.evidence_id} inspected.", data=evidence.model_dump(mode="json"))
        raise ToolError("evidence_id does not match visible claim file")


class RequestDocumentTool(ToolHandler):
    name = ToolName.REQUEST_DOCUMENT

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = RequestDocumentArgs.model_validate(args)
        runtime.workflow.documents_requested.append(parsed.doc_type)
        if parsed.doc_type not in runtime.workflow.documents_received:
            runtime.workflow.documents_received.append(parsed.doc_type)
        if parsed.doc_type in runtime.spec.hidden.required_documents:
            runtime.evidence.append(
                Evidence(
                    evidence_id=f"EV-DOC-{parsed.doc_type.value.upper()}",
                    kind=EvidenceKind.REQUESTED_DOCUMENT,
                    summary=f"Requested {parsed.doc_type.value} received and added to file.",
                )
            )
            summary = f"{parsed.doc_type.value} requested and received."
        else:
            runtime.evidence.append(
                Evidence(
                    evidence_id=f"EV-DOC-{parsed.doc_type.value.upper()}",
                    kind=EvidenceKind.REQUESTED_DOCUMENT,
                    summary=f"Requested {parsed.doc_type.value} received; no material discrepancy identified.",
                )
            )
            summary = f"{parsed.doc_type.value} requested and received."
        runtime.diary.append(f"Document request: {parsed.doc_type.value}. Reason: {parsed.reason}")
        if parsed.doc_type == DocumentType.POLICE_REPORT:
            _complete_activity(runtime, "coverage", "police report requested")
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
        _complete_activity(runtime, "siu", "SIU screen completed")
        return ToolResult(ok=True, summary="Fraud indicator screen complete.", data={"indicators": indicators})


class CreateOrUpdateExposureTool(ToolHandler):
    name = ToolName.CREATE_OR_UPDATE_EXPOSURE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = CreateOrUpdateExposureArgs.model_validate(args)
        existing = _find_exposure(runtime, parsed.exposure_id)
        if existing:
            existing.coverage = parsed.coverage
            existing.claimant_id = parsed.claimant_id
            existing.incident_id = parsed.incident_id
            existing.status = ExposureStatus.OPEN
            existing.validation_level = "coverage_pending"
            summary = "Exposure updated."
        else:
            runtime.platform_state.exposures.append(
                Exposure(
                    exposure_id=parsed.exposure_id,
                    coverage=parsed.coverage,
                    claimant_id=parsed.claimant_id,
                    incident_id=parsed.incident_id,
                    status=ExposureStatus.OPEN,
                    validation_level="coverage_pending",
                )
            )
            summary = "Exposure created."
        runtime.diary.append(f"{summary} {parsed.exposure_id} on {parsed.coverage.value}.")
        return ToolResult(ok=True, summary=summary, data={"exposure_id": parsed.exposure_id})


class VerifyCoverageTool(ToolHandler):
    name = ToolName.VERIFY_COVERAGE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = VerifyCoverageArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        exposure = _require_exposure(runtime, parsed.exposure_id)
        policy = runtime.spec.policy
        active_on_loss = policy.status == "active" and policy.effective_date <= runtime.spec.claim.loss_date <= policy.expiration_date
        covered = active_on_loss and runtime.spec.hidden.is_covered
        exposure.status = ExposureStatus.COVERAGE_VERIFIED if covered else ExposureStatus.OPEN
        exposure.validation_level = "ability_to_pay" if covered else "coverage_pending"
        runtime.platform_state.coverage_verified = True
        runtime.platform_state.coverage_result = {
            "covered": covered,
            "active_on_loss": active_on_loss,
            "coverage": exposure.coverage.value,
            "reason": "covered physical damage loss" if covered else runtime.spec.hidden.coverage_reason,
        }
        runtime.workflow.policy_status_checked = True
        _complete_activity(runtime, "coverage", "coverage verified")
        runtime.diary.append(f"Coverage verified for {parsed.exposure_id}: covered={covered}.")
        return ToolResult(ok=True, summary=f"Coverage verified: {covered}.", data=runtime.platform_state.coverage_result)


class AssignAppraisalTool(ToolHandler):
    name = ToolName.ASSIGN_APPRAISAL

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = AssignAppraisalArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        status_by_method = {
            "photo": AppraisalStatus.PHOTO_ASSIGNED,
            "field": AppraisalStatus.FIELD_ASSIGNED,
            "shop": AppraisalStatus.SHOP_ASSIGNED,
        }
        runtime.platform_state.appraisal_status = status_by_method[parsed.method]
        runtime.workflow.appraisal_assigned = True
        runtime.platform_state.vendor_assignments.append(
            VendorAssignment(
                vendor_id=f"VND-{parsed.method.upper()}-APPRAISER",
                vendor_type=f"{parsed.method}_appraiser" if parsed.method in {"photo", "field"} else "repair_shop",  # type: ignore[arg-type]
                eta_steps=1 if parsed.method == "photo" else 2,
                notes=f"{parsed.method} appraisal assigned",
            )
        )
        runtime.diary.append(f"{parsed.method.title()} appraisal assigned.")
        return ToolResult(ok=True, summary="Appraisal assigned.", data={"method": parsed.method})


class ReviewEstimateTool(ToolHandler):
    name = ToolName.REVIEW_ESTIMATE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = ReviewEstimateArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        if parsed.estimate_id != runtime.spec.repair_estimate.estimate_id:
            raise ToolError("estimate_id does not match this claim file")
        runtime.workflow.estimate_reviewed = True
        runtime.platform_state.estimate_review_decision = parsed.action
        runtime.platform_state.appraisal_status = (
            AppraisalStatus.SUPPLEMENT_REQUESTED
            if parsed.action == EstimateReviewDecision.REQUEST_SUPPLEMENT
            else AppraisalStatus.TOTAL_LOSS_REVIEW
            if parsed.action == EstimateReviewDecision.CONFIRM_TOTAL_LOSS
            else AppraisalStatus.ESTIMATE_REVIEWED
        )
        if parsed.action == EstimateReviewDecision.REQUEST_SUPPLEMENT:
            runtime.evidence.append(
                Evidence(
                    evidence_id="EV-SUPPLEMENT-REQUEST",
                    kind=EvidenceKind.REQUESTED_DOCUMENT,
                    summary="Supplement request sent to repair facility for estimate clarification.",
                    flags=["supplement_requested"],
                )
            )
        if parsed.action == EstimateReviewDecision.CONFIRM_TOTAL_LOSS:
            runtime.platform_state.valuation_requested = True
        _complete_activity(runtime, "estimate", f"estimate review action={parsed.action.value}")
        runtime.diary.append(f"Estimate review: {parsed.action.value}. Rationale: {parsed.rationale}")
        return ToolResult(
            ok=True,
            summary=f"Estimate review recorded: {parsed.action.value}.",
            data={"action": parsed.action.value},
        )


class RequestValuationTool(ToolHandler):
    name = ToolName.REQUEST_VALUATION

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = RequestValuationArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        runtime.platform_state.valuation_requested = True
        runtime.platform_state.valuation_received = True
        runtime.workflow.valuation_seen = True
        runtime.platform_state.appraisal_status = AppraisalStatus.VALUATION_RECEIVED
        actual_cash_value = max(runtime.spec.hidden.expected_payable + runtime.spec.policy.deductible, runtime.spec.repair_estimate.covered_amount * 0.82)
        runtime.evidence.append(
            Evidence(
                evidence_id="EV-VALUATION",
                kind=EvidenceKind.REQUESTED_DOCUMENT,
                summary=f"Total-loss valuation report estimates ACV at ${actual_cash_value:,.2f}.",
                amount=round(actual_cash_value, 2),
                flags=["valuation"],
            )
        )
        runtime.diary.append(f"Valuation requested. Reason: {parsed.reason}")
        return ToolResult(ok=True, summary="Valuation report received.", data={"actual_cash_value": round(actual_cash_value, 2)})


class SetReserveTool(ToolHandler):
    name = ToolName.SET_RESERVE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = SetReserveArgs.model_validate(args)
        exposure = _require_exposure(runtime, parsed.exposure_id)
        runtime.financial_snapshot.reserve_amount = parsed.amount
        runtime.financial_snapshot.total_reserved = parsed.amount
        exposure.reserve_amount = parsed.amount
        approval_status = "pending_authority" if parsed.amount > runtime.spec.policy.authority_limit and not runtime.platform_state.authority_approved else "approved"
        reserve_id = f"RSV-{len(runtime.platform_state.reserve_lines) + 1:03d}"
        runtime.platform_state.reserve_lines.append(
            ReserveLine(
                reserve_id=reserve_id,
                exposure_id=parsed.exposure_id,
                cost_type="claim_cost",
                cost_category=parsed.cost_category,
                amount=parsed.amount,
                rationale=parsed.rationale,
                approval_status=approval_status,  # type: ignore[arg-type]
            )
        )
        _complete_activity(runtime, "reserve", "reserve set")
        band = reserve_band_for(parsed.amount).value
        runtime.diary.append(f"Reserve set to ${parsed.amount:,.2f}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary=f"Reserve set in {band} band.", data={"amount": parsed.amount, "band": band, "reserve_id": reserve_id})


class ApprovePaymentTool(ToolHandler):
    name = ToolName.APPROVE_PAYMENT

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = ApprovePaymentArgs.model_validate(args)
        runtime.financial_snapshot.approved_payment = parsed.amount
        runtime.financial_snapshot.payment_coverages = parsed.coverages
        runtime.diary.append(f"Payment approved for ${parsed.amount:,.2f}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary="Payment approval recorded.", data=parsed.model_dump())


class IssuePaymentTool(ToolHandler):
    name = ToolName.ISSUE_PAYMENT

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = IssuePaymentArgs.model_validate(args)
        exposure = _require_exposure(runtime, parsed.exposure_id)
        if parsed.amount > runtime.spec.policy.authority_limit and not runtime.platform_state.authority_approved:
            runtime.violations.append("authority_bypass")
            status = "blocked_authority"
            summary = "Payment blocked: authority approval required."
        elif not runtime.platform_state.coverage_verified:
            runtime.violations.append("payment_before_coverage")
            status = "draft"
            summary = "Payment drafted before coverage verification."
        else:
            status = "issued"
            summary = "Payment issued."
            exposure.status = ExposureStatus.READY_TO_PAY
            exposure.paid_amount += parsed.amount
            runtime.financial_snapshot.approved_payment = parsed.amount
            runtime.financial_snapshot.total_paid += parsed.amount
            runtime.financial_snapshot.payment_coverages = [exposure.coverage.value]
        payment_id = f"PAY-{len(runtime.platform_state.payments) + 1:03d}"
        runtime.platform_state.payments.append(
            Payment(
                payment_id=payment_id,
                exposure_id=parsed.exposure_id,
                payee_id=parsed.payee_id,
                amount=parsed.amount,
                payment_type=parsed.payment_type,  # type: ignore[arg-type]
                method=parsed.method,  # type: ignore[arg-type]
                status=status,  # type: ignore[arg-type]
                rationale=parsed.rationale,
            )
        )
        runtime.diary.append(f"{summary} {payment_id} amount=${parsed.amount:,.2f}.")
        return ToolResult(ok=True, summary=summary, data={"payment_id": payment_id, "status": status, "amount": parsed.amount})


class RequestAuthorityApprovalTool(ToolHandler):
    name = ToolName.REQUEST_AUTHORITY_APPROVAL

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = RequestAuthorityApprovalArgs.model_validate(args)
        _require_exposure(runtime, parsed.exposure_id)
        runtime.platform_state.authority_requested = True
        runtime.platform_state.authority_approved = True
        runtime.workflow.authority_requested = True
        runtime.workflow.authority_approved = True
        _complete_activity(runtime, "authority", "authority approval requested and approved")
        runtime.diary.append(f"Authority approval granted for ${parsed.amount:,.2f}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary="Authority approval granted.", data={"amount": parsed.amount, "approved": True})


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
        _complete_activity(runtime, "siu", "SIU referral opened")
        runtime.diary.append(f"SIU referral opened. Reason: {parsed.reason}")
        return ToolResult(ok=True, summary="SIU referral opened.", data=parsed.model_dump())


class OpenSiuReferralTool(ReferToSiuTool):
    name = ToolName.OPEN_SIU_REFERRAL


class OpenSubrogationTool(ToolHandler):
    name = ToolName.OPEN_SUBROGATION

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = OpenSubrogationArgs.model_validate(args)
        runtime.workflow.subrogation_opened = True
        _complete_activity(runtime, "subrogation", "subrogation opened")
        runtime.diary.append(f"Subrogation opened against {parsed.target_party}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary="Subrogation file opened.", data=parsed.model_dump())


class SendClaimantMessageTool(ToolHandler):
    name = ToolName.SEND_CLAIMANT_MESSAGE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = SendClaimantMessageArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        runtime.communications.append(parsed.message)
        runtime.workflow.claimant_updated = True
        _complete_activity(runtime, "communication", "claimant message sent")
        runtime.platform_state.notes.append(
            ClaimNote(
                note_id=f"NOTE-{len(runtime.platform_state.notes) + 1:03d}",
                note_type=NoteType.COMMUNICATION,
                subject="Claimant update",
                body=parsed.message,
                related_object_id=parsed.claim_id,
            )
        )
        return ToolResult(ok=True, summary="Claimant message sent.", data={"message": parsed.message})


class AddClaimNoteTool(ToolHandler):
    name = ToolName.ADD_CLAIM_NOTE

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = AddClaimNoteArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        note = ClaimNote(
            note_id=f"NOTE-{len(runtime.platform_state.notes) + 1:03d}",
            note_type=parsed.note_type,
            subject=parsed.subject,
            body=parsed.body,
            related_object_id=parsed.related_object_id,
        )
        runtime.platform_state.notes.append(note)
        if parsed.note_type == NoteType.CLOSURE:
            runtime.workflow.closure_note_added = True
        runtime.diary.append(f"Claim note added: {parsed.subject}")
        return ToolResult(ok=True, summary="Claim note added.", data=note.model_dump(mode="json"))


class CloseClaimTool(ToolHandler):
    name = ToolName.CLOSE_CLAIM

    def run(self, runtime: RuntimeView, args: dict[str, Any]) -> ToolResult:
        parsed = CloseClaimArgs.model_validate(args)
        if parsed.claim_id != runtime.spec.claim.claim_id:
            raise ToolError("claim_id does not match this claim file")
        open_activities = [activity.activity_id for activity in runtime.platform_state.activities if activity.status == ActivityStatus.OPEN]
        if open_activities:
            runtime.violations.append("closed_with_open_activities")
        runtime.platform_state.claim_closed = True
        for exposure in runtime.platform_state.exposures:
            exposure.status = ExposureStatus.CLOSED
            exposure.validation_level = "closed"
        runtime.diary.append(f"Claim closed as {parsed.disposition}. Rationale: {parsed.rationale}")
        return ToolResult(ok=True, summary="Claim closed.", data={"disposition": parsed.disposition, "open_activities": open_activities}, terminal=False)


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
        runtime.platform_state.claim_closed = True
        runtime.workflow.claimant_updated = True
        if parsed.closure_disposition:
            runtime.workflow.closure_note_added = True
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
        GetPolicySnapshotTool(),
        CheckPolicyStatusTool(),
        InspectRepairEstimateTool(),
        InspectEvidenceTool(),
        RequestDocumentTool(),
        QueryPriorClaimsTool(),
        CheckFraudIndicatorsTool(),
        CreateOrUpdateExposureTool(),
        VerifyCoverageTool(),
        AssignAppraisalTool(),
        ReviewEstimateTool(),
        RequestValuationTool(),
        SetReserveTool(),
        ApprovePaymentTool(),
        IssuePaymentTool(),
        RequestAuthorityApprovalTool(),
        ReferToSiuTool(),
        OpenSiuReferralTool(),
        OpenSubrogationTool(),
        SendClaimantMessageTool(),
        AddClaimNoteTool(),
        CloseClaimTool(),
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


def _find_exposure(runtime: RuntimeView, exposure_id: str) -> Exposure | None:
    for exposure in runtime.platform_state.exposures:
        if exposure.exposure_id == exposure_id:
            return exposure
    return None


def _require_exposure(runtime: RuntimeView, exposure_id: str) -> Exposure:
    exposure = _find_exposure(runtime, exposure_id)
    if exposure is None:
        raise ToolError("exposure_id does not match this claim file")
    return exposure


def _complete_activity(runtime: RuntimeView, category: str, reason: str) -> None:
    for activity in runtime.platform_state.activities:
        status = activity.status.value if hasattr(activity.status, "value") else str(activity.status)
        if activity.category == category and status in {"open", "overdue"}:
            activity.status = ActivityStatus.COMPLETED
            activity.close_reason = reason
            return
