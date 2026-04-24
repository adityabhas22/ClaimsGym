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
    CHECK_POLICY_STATUS = "check_policy_status"
    INSPECT_REPAIR_ESTIMATE = "inspect_repair_estimate"
    REQUEST_DOCUMENT = "request_document"
    QUERY_PRIOR_CLAIMS = "query_prior_claims"
    CHECK_FRAUD_INDICATORS = "check_fraud_indicators"
    SET_RESERVE = "set_reserve"
    APPROVE_PAYMENT = "approve_payment"
    REFER_TO_SIU = "refer_to_siu"
    OPEN_SUBROGATION = "open_subrogation"
    SUBMIT_FINAL_DECISION = "submit_final_decision"


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
    authority_escalation_required: bool = False
    denial_clause: str | None = None
    protected_fields: dict[str, Any] = Field(default_factory=dict)


class FinancialSnapshot(BaseModel):
    reserve_amount: float | None = None
    approved_payment: float | None = None
    payment_coverages: list[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    policy_seen: bool = False
    policy_status_checked: bool = False
    estimate_seen: bool = False
    prior_claims_seen: bool = False
    fraud_checked: bool = False
    documents_requested: list[DocumentType] = Field(default_factory=list)
    documents_received: list[DocumentType] = Field(default_factory=list)
    siu_referral: bool = False
    subrogation_opened: bool = False
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
    claim_diary: list[str]
    financial_snapshot: FinancialSnapshot
    communications_sent: list[str]
    open_tasks: list[str]
    available_tools: list[ToolName]
    remaining_steps: int


class RewardBreakdown(BaseModel):
    format_validity: float = 0.0
    coverage: float = 0.0
    payout: float = 0.0
    evidence: float = 0.0
    fraud_triage: float = 0.0
    subrogation: float = 0.0
    communication: float = 0.0
    reserve: float = 0.0
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

    @field_validator("claimant_message")
    @classmethod
    def message_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claimant_message is required")
        return value
