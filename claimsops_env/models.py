from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CoverageType(str, Enum):
    COLLISION = "collision"
    COMPREHENSIVE = "comprehensive"
    PROPERTY_DAMAGE = "property_damage"


class ClaimType(str, Enum):
    COLLISION = "collision"
    COMPREHENSIVE = "comprehensive"


class DocumentType(str, Enum):
    POLICE_REPORT = "police_report"
    REPAIR_ESTIMATE_BREAKDOWN = "repair_estimate_breakdown"
    CLAIMANT_STATEMENT = "claimant_statement"
    PROOF_OF_OWNERSHIP = "proof_of_ownership"


class EvidenceKind(str, Enum):
    CLAIMANT_STATEMENT = "claimant_statement"
    REPAIR_ESTIMATE = "repair_estimate"
    POLICE_REPORT = "police_report"
    TELEMATICS = "telematics"
    PRIOR_CLAIMS = "prior_claims"
    REQUESTED_DOCUMENT = "requested_document"
    FRAUD_REPORT = "fraud_report"
    POLICY = "policy"
    VENDOR_REPORT = "vendor_report"


class Decision(str, Enum):
    APPROVE = "approve"
    APPROVE_PARTIAL = "approve_partial"
    DENY = "deny"
    ESCALATE = "escalate"


class ReserveBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolName(str, Enum):
    GET_POLICY = "get_policy"
    GET_POLICY_SNAPSHOT = "get_policy_snapshot"
    CHECK_POLICY_STATUS = "check_policy_status"
    INSPECT_REPAIR_ESTIMATE = "inspect_repair_estimate"
    INSPECT_EVIDENCE = "inspect_evidence"
    REQUEST_DOCUMENT = "request_document"
    QUERY_PRIOR_CLAIMS = "query_prior_claims"
    CHECK_FRAUD_INDICATORS = "check_fraud_indicators"
    CREATE_OR_UPDATE_EXPOSURE = "create_or_update_exposure"
    VERIFY_COVERAGE = "verify_coverage"
    ASSIGN_APPRAISAL = "assign_appraisal"
    REVIEW_ESTIMATE = "review_estimate"
    REQUEST_VALUATION = "request_valuation"
    SET_RESERVE = "set_reserve"
    APPROVE_PAYMENT = "approve_payment"
    ISSUE_PAYMENT = "issue_payment"
    REQUEST_AUTHORITY_APPROVAL = "request_authority_approval"
    REFER_TO_SIU = "refer_to_siu"
    OPEN_SIU_REFERRAL = "open_siu_referral"
    OPEN_SUBROGATION = "open_subrogation"
    SEND_CLAIMANT_MESSAGE = "send_claimant_message"
    ADD_CLAIM_NOTE = "add_claim_note"
    CLOSE_CLAIM = "close_claim"
    SUBMIT_FINAL_DECISION = "submit_final_decision"


class ExposureStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    COVERAGE_VERIFIED = "coverage_verified"
    READY_TO_PAY = "ready_to_pay"
    CLOSED = "closed"


class ActivityStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    OVERDUE = "overdue"


class AppraisalStatus(str, Enum):
    NOT_ASSIGNED = "not_assigned"
    PHOTO_ASSIGNED = "photo_assigned"
    FIELD_ASSIGNED = "field_assigned"
    SHOP_ASSIGNED = "shop_assigned"
    ESTIMATE_REVIEWED = "estimate_reviewed"
    SUPPLEMENT_REQUESTED = "supplement_requested"
    TOTAL_LOSS_REVIEW = "total_loss_review"
    VALUATION_RECEIVED = "valuation_received"


class EstimateReviewDecision(str, Enum):
    APPROVE = "approve"
    REQUEST_SUPPLEMENT = "request_supplement"
    CONFIRM_TOTAL_LOSS = "confirm_total_loss"
    ESCALATE_FIELD = "escalate_field"
    REQUEST_PHOTOS = "request_photos"


class NoteType(str, Enum):
    COVERAGE = "coverage"
    ESTIMATE = "estimate"
    FINANCIAL = "financial"
    COMMUNICATION = "communication"
    SIU = "siu"
    SUBROGATION = "subrogation"
    CLOSURE = "closure"


class Policy(BaseModel):
    policy_id: str
    customer_id: str
    vehicle_id: str
    effective_date: date
    expiration_date: date
    status: Literal["active", "cancelled", "lapsed"]
    deductible: float = Field(ge=0)
    collision_limit: float = Field(gt=0)
    comprehensive_limit: float = Field(gt=0)
    authority_limit: float = Field(gt=0)
    exclusions: list[str] = Field(default_factory=list)

    def limit_for(self, claim_type: ClaimType) -> float:
        if claim_type == ClaimType.COMPREHENSIVE:
            return self.comprehensive_limit
        return self.collision_limit


class Evidence(BaseModel):
    evidence_id: str
    kind: EvidenceKind
    summary: str
    amount: float | None = None
    flags: list[str] = Field(default_factory=list)


class RepairEstimate(BaseModel):
    estimate_id: str
    gross_amount: float = Field(ge=0)
    covered_amount: float = Field(ge=0)
    unrelated_damage_amount: float = Field(default=0, ge=0)
    duplicate_line_amount: float = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)
    labor_hours: float = Field(default=0, ge=0)
    parts_amount: float = Field(default=0, ge=0)
    paint_materials_amount: float = Field(default=0, ge=0)
    total_loss_threshold: float | None = None
    photo_quality: Literal["good", "partial", "poor"] = "good"


class Party(BaseModel):
    party_id: str
    role: str
    display_name: str
    contact_status: Literal["reachable", "unresponsive", "represented"] = "reachable"


class Incident(BaseModel):
    incident_id: str
    incident_type: Literal["vehicle_damage", "rental", "towing_storage", "third_party_property"]
    vehicle_id: str | None = None
    description: str
    severity: Literal["minor", "moderate", "severe", "possible_total_loss"]
    liability_signal: Literal["insured_fault", "third_party_fault", "disputed", "unknown"] = "unknown"


class Exposure(BaseModel):
    exposure_id: str
    coverage: CoverageType
    claimant_id: str
    incident_id: str
    status: ExposureStatus = ExposureStatus.DRAFT
    validation_level: Literal["new", "coverage_pending", "ability_to_pay", "closed"] = "new"
    reserve_amount: float = Field(default=0, ge=0)
    paid_amount: float = Field(default=0, ge=0)


class Activity(BaseModel):
    activity_id: str
    subject: str
    category: str
    due_in_steps: int = Field(ge=0)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    status: ActivityStatus = ActivityStatus.OPEN
    related_object_id: str | None = None
    close_reason: str | None = None


class ReserveLine(BaseModel):
    reserve_id: str
    exposure_id: str
    cost_type: Literal["claim_cost", "expense", "recovery"] = "claim_cost"
    cost_category: str
    amount: float = Field(ge=0)
    rationale: str
    approval_status: Literal["approved", "pending_authority"] = "approved"


class Payment(BaseModel):
    payment_id: str
    exposure_id: str
    payee_id: str
    amount: float = Field(ge=0)
    payment_type: Literal["partial", "final", "expense", "recovery_refund"] = "final"
    method: Literal["eft", "check"] = "eft"
    status: Literal["draft", "issued", "blocked_authority"] = "issued"
    rationale: str


class VendorAssignment(BaseModel):
    vendor_id: str
    vendor_type: Literal["photo_appraiser", "field_appraiser", "repair_shop", "rental", "salvage"]
    status: Literal["assigned", "completed", "cancelled"] = "assigned"
    eta_steps: int = Field(default=1, ge=0)
    notes: str = ""


class ClaimNote(BaseModel):
    note_id: str
    note_type: NoteType
    subject: str
    body: str
    related_object_id: str | None = None


class PendingEvent(BaseModel):
    event_id: str
    event_type: Literal[
        "document_arrival",
        "appraisal_complete",
        "valuation_complete",
        "authority_decision",
        "claimant_response",
        "supplement_received",
        "rental_day_accrual",
        "storage_fee_accrual",
    ]
    due_in_steps: int = Field(ge=0)
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EventRecord(BaseModel):
    event_id: str
    event_type: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class PlatformState(BaseModel):
    parties: list[Party] = Field(default_factory=list)
    incidents: list[Incident] = Field(default_factory=list)
    exposures: list[Exposure] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    reserve_lines: list[ReserveLine] = Field(default_factory=list)
    payments: list[Payment] = Field(default_factory=list)
    vendor_assignments: list[VendorAssignment] = Field(default_factory=list)
    notes: list[ClaimNote] = Field(default_factory=list)
    pending_events: list[PendingEvent] = Field(default_factory=list)
    event_history: list[EventRecord] = Field(default_factory=list)
    rental_days: int = Field(default=0, ge=0)
    storage_charges: float = Field(default=0, ge=0)
    appraisal_status: AppraisalStatus = AppraisalStatus.NOT_ASSIGNED
    estimate_review_decision: EstimateReviewDecision | None = None
    valuation_requested: bool = False
    valuation_received: bool = False
    authority_requested: bool = False
    authority_approved: bool = False
    coverage_verified: bool = False
    coverage_result: dict[str, Any] | None = None
    claim_closed: bool = False


class ClaimScenario(BaseModel):
    claim_id: str
    policy_id: str
    customer_id: str
    vehicle_id: str
    line_of_business: Literal["personal_auto"] = "personal_auto"
    claim_type: ClaimType
    loss_date: date
    reported_date: date
    claimant_statement: str
    requested_amount: float = Field(ge=0)
    estimate_id: str
    family: str
    difficulty: int = Field(ge=1, le=5)
    initial_evidence: list[Evidence]
    step_budget: int = Field(gt=0)


class HiddenTruth(BaseModel):
    is_covered: bool
    coverage_reason: str
    expected_payable: float = Field(ge=0)
    required_documents: set[DocumentType] = Field(default_factory=set)
    fraud_suspicious: bool
    subrogation_expected: bool
    expected_reserve_band: ReserveBand
    expected_estimate_review: EstimateReviewDecision = EstimateReviewDecision.APPROVE
    expected_exposure_coverages: set[CoverageType] = Field(default_factory=set)
    expected_activity_categories: set[str] = Field(default_factory=set)
    expected_min_reserve: float = Field(default=0, ge=0)
    expected_max_reserve: float = Field(default=0, ge=0)
    expected_total_loss: bool = False
    expected_salvage_value: float = Field(default=0, ge=0)
    liability_split_insured_pct: int = Field(default=0, ge=0, le=100)
    authority_escalation_required: bool = False
    denial_clause: str | None = None
    protected_fields: dict[str, Any] = Field(default_factory=dict)


class FinancialSnapshot(BaseModel):
    reserve_amount: float | None = None
    approved_payment: float | None = None
    payment_coverages: list[str] = Field(default_factory=list)
    total_reserved: float = 0.0
    total_paid: float = 0.0
    open_recovery_reserve: float = 0.0


class WorkflowState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    policy_seen: bool = False
    policy_status_checked: bool = False
    estimate_seen: bool = False
    estimate_reviewed: bool = False
    appraisal_assigned: bool = False
    valuation_seen: bool = False
    prior_claims_seen: bool = False
    fraud_checked: bool = False
    documents_requested: list[DocumentType] = Field(default_factory=list)
    documents_received: list[DocumentType] = Field(default_factory=list)
    siu_referral: bool = False
    subrogation_opened: bool = False
    authority_requested: bool = False
    authority_approved: bool = False
    claimant_updated: bool = False
    closure_note_added: bool = False
    final_decision_submitted: bool = False


class Action(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)


class ActionRecord(BaseModel):
    step: int
    action: dict[str, Any]
    valid: bool
    result_summary: str


class Observation(BaseModel):
    claim_id: str
    policy_id: str
    customer_id: str
    vehicle_id: str
    estimate_id: str
    line_of_business: str
    claim_type: ClaimType
    loss_date: date
    reported_date: date
    claimant_statement: str
    requested_amount: float
    latest_tool_result: dict[str, Any] | None = None
    visible_policy: dict[str, Any] | None = None
    available_evidence: list[Evidence]
    parties: list[Party] = Field(default_factory=list)
    incidents: list[Incident] = Field(default_factory=list)
    exposures: list[Exposure] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    reserve_lines: list[ReserveLine] = Field(default_factory=list)
    payments: list[Payment] = Field(default_factory=list)
    vendor_assignments: list[VendorAssignment] = Field(default_factory=list)
    claim_notes: list[ClaimNote] = Field(default_factory=list)
    pending_events: list[PendingEvent] = Field(default_factory=list)
    event_history: list[EventRecord] = Field(default_factory=list)
    rental_days: int = 0
    storage_charges: float = 0.0
    appraisal_status: AppraisalStatus = AppraisalStatus.NOT_ASSIGNED
    coverage_result: dict[str, Any] | None = None
    alerts: list[str] = Field(default_factory=list)
    audit_gaps: list[str] = Field(default_factory=list)
    claim_diary: list[str]
    financial_snapshot: FinancialSnapshot
    communications_sent: list[str]
    open_tasks: list[str]
    available_tools: list[ToolName]
    remaining_steps: int


class RewardBreakdown(BaseModel):
    format_validity: float = 0.0
    workflow_progress: float = 0.0
    coverage: float = 0.0
    payout: float = 0.0
    evidence: float = 0.0
    leakage_control: float = 0.0
    fraud_triage: float = 0.0
    subrogation: float = 0.0
    communication: float = 0.0
    reserve: float = 0.0
    compliance: float = 0.0
    financial_controls: float = 0.0
    efficiency: float = 0.0
    audit_trail: float = 0.0
    penalties: float = 0.0
    safety_cap: float = 1.0
    total: float = 0.0

    def as_log_row(self) -> dict[str, float]:
        return self.model_dump()


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class FinalDecision(BaseModel):
    decision: Decision
    payment_amount: float = Field(ge=0)
    reserve_amount: float = Field(ge=0)
    siu_referral: bool
    subrogation: bool
    claimant_message: str
    evidence_cited: list[str]
    rationale: str
    closure_disposition: Literal["paid_closed", "denied_closed", "escalated", "withdrawn"] | None = None

    @field_validator("claimant_message")
    @classmethod
    def message_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claimant_message is required")
        return value
