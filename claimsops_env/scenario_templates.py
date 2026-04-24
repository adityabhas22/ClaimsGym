from __future__ import annotations

from dataclasses import dataclass, field

from claimsops_env.models import DocumentType, EstimateReviewDecision


@dataclass(frozen=True)
class ScenarioTemplate:
    family: str
    title: str
    level: int
    claim_type: str = "collision"
    statement: str = ""
    gross_range: tuple[int, int] = (3200, 8800)
    required_documents: frozenset[DocumentType] = frozenset()
    visible_estimate_flags: tuple[str, ...] = ()
    prior_claims: tuple[str, ...] = ()
    suspicious: bool = False
    subrogation: bool = False
    policy_status: str = "active"
    denial_clause: str | None = None
    expected_review: EstimateReviewDecision = EstimateReviewDecision.APPROVE
    expected_total_loss: bool = False
    authority_escalation: bool = False
    unrelated_damage: float = 0.0
    duplicate_line: float = 0.0
    photo_quality: str = "good"
    liability_split_insured_pct: int = 0
    needs_police_evidence: bool = False
    telematics_conflict: bool = False
    contact_status: str = "reachable"
    initial_event_profiles: tuple[str, ...] = ()
    extra_activity_categories: tuple[str, ...] = ()
    operational_rubric: tuple[str, ...] = field(default_factory=tuple)


SCENARIO_TEMPLATES: dict[str, ScenarioTemplate] = {
    "covered_collision": ScenarioTemplate(
        family="covered_collision",
        title="Straight-through covered collision",
        level=1,
        statement="I was hit while stopped at a light. The rear bumper and trunk are damaged.",
        operational_rubric=(
            "Verify active collision coverage.",
            "Review estimate for covered rear impact damage.",
            "Apply deductible before payment.",
        ),
    ),
    "comprehensive_deductible": ScenarioTemplate(
        family="comprehensive_deductible",
        title="Comprehensive weather loss with deductible",
        level=1,
        claim_type="comprehensive",
        statement="A tree branch fell during a storm and damaged the hood and windshield.",
        gross_range=(2200, 6500),
        operational_rubric=(
            "Classify as comprehensive rather than collision.",
            "Apply comprehensive deductible.",
        ),
    ),
    "policy_lapse": ScenarioTemplate(
        family="policy_lapse",
        title="Loss after policy lapse",
        level=1,
        policy_status="lapsed",
        denial_clause="policy_period",
        statement="I hit a pole last week and need my bumper and headlamp repaired.",
        operational_rubric=(
            "Confirm loss date falls outside the active term.",
            "Deny with policy-period rationale and no payment.",
        ),
    ),
    "limit_exceeded": ScenarioTemplate(
        family="limit_exceeded",
        title="Covered loss above available limit",
        level=1,
        visible_estimate_flags=("Estimate exceeds collision coverage limit.",),
        operational_rubric=(
            "Cap payment at the applicable coverage limit.",
            "Avoid paying the requested amount when it exceeds limits.",
        ),
    ),
    "missing_police_report": ScenarioTemplate(
        family="missing_police_report",
        title="Parking-lot third-party loss with pending police report",
        level=2,
        statement="A third-party vehicle backed into my car in a parking lot; report number is pending.",
        required_documents=frozenset({DocumentType.POLICE_REPORT}),
        subrogation=True,
        operational_rubric=(
            "Request police report before final liability and recovery decision.",
            "Open subrogation when the report supports third-party liability.",
        ),
    ),
    "incomplete_statement": ScenarioTemplate(
        family="incomplete_statement",
        title="Incomplete first notice facts",
        level=2,
        statement="My car was damaged but I do not have all of the details yet. The statement is incomplete.",
        required_documents=frozenset({DocumentType.CLAIMANT_STATEMENT}),
        operational_rubric=(
            "Request a complete claimant statement before final payment.",
            "Do not deny solely because the initial notice is incomplete.",
        ),
    ),
    "ownership_gap": ScenarioTemplate(
        family="ownership_gap",
        title="Potential ownership or insurable-interest gap",
        level=2,
        statement="The car is titled to my sibling but I am the regular driver and owner on the policy.",
        required_documents=frozenset({DocumentType.PROOF_OF_OWNERSHIP}),
        visible_estimate_flags=("Ownership documentation is not yet in file.",),
        operational_rubric=(
            "Request proof of ownership or insurable interest.",
            "Avoid payment before the ownership gap is cured.",
        ),
    ),
    "prior_damage_leakage": ScenarioTemplate(
        family="prior_damage_leakage",
        title="Repair estimate includes unrelated prior damage",
        level=3,
        unrelated_damage=1100.0,
        required_documents=frozenset({DocumentType.REPAIR_ESTIMATE_BREAKDOWN}),
        prior_claims=("Prior left quarter-panel damage paid 2025-11-04.",),
        visible_estimate_flags=("Left quarter-panel line appears unrelated to described rear impact.",),
        expected_review=EstimateReviewDecision.ESCALATE_FIELD,
        operational_rubric=(
            "Query prior claims and separate unrelated damage.",
            "Escalate field appraisal rather than paying the full estimate.",
        ),
    ),
    "duplicate_line_item": ScenarioTemplate(
        family="duplicate_line_item",
        title="Duplicate paint-material line in estimate",
        level=3,
        duplicate_line=425.0,
        required_documents=frozenset({DocumentType.REPAIR_ESTIMATE_BREAKDOWN}),
        visible_estimate_flags=("Paint materials line appears duplicated.",),
        expected_review=EstimateReviewDecision.REQUEST_SUPPLEMENT,
        operational_rubric=(
            "Request a supplement or corrected estimate breakdown.",
            "Deduct duplicate line amount from payable exposure.",
        ),
    ),
    "rental_storage_leakage": ScenarioTemplate(
        family="rental_storage_leakage",
        title="Towing, storage, and rental leakage pressure",
        level=3,
        statement="The car is sitting at the tow yard and I need a rental while repairs are reviewed.",
        visible_estimate_flags=("Tow yard storage charges accrue daily until release is arranged.",),
        extra_activity_categories=("rental", "towing_storage"),
        initial_event_profiles=("storage_fee_accrual", "rental_day_accrual"),
        operational_rubric=(
            "Move quickly on appraisal and communication to control storage leakage.",
            "Do not ignore visible daily expense accumulation.",
        ),
    ),
    "suspicious_inception": ScenarioTemplate(
        family="suspicious_inception",
        title="Loss reported shortly after inception",
        level=4,
        statement="My parked vehicle was damaged overnight two days after I started the policy.",
        suspicious=True,
        required_documents=frozenset({DocumentType.POLICE_REPORT}),
        visible_estimate_flags=("Loss reported shortly after policy inception.",),
        expected_review=EstimateReviewDecision.REQUEST_PHOTOS,
        photo_quality="partial",
        operational_rubric=(
            "Run fraud indicator screen and request objective support.",
            "Refer to SIU only when visible indicators support it.",
        ),
    ),
    "conflicting_statement": ScenarioTemplate(
        family="conflicting_statement",
        title="Statement conflicts with telematics",
        level=4,
        statement="I was parked when another car struck the front bumper.",
        suspicious=True,
        telematics_conflict=True,
        visible_estimate_flags=("Telematics indicates vehicle was moving at impact.",),
        expected_review=EstimateReviewDecision.ESCALATE_FIELD,
        operational_rubric=(
            "Inspect conflicting evidence before closure.",
            "Escalate the damage review and triage SIU.",
        ),
    ),
    "excluded_driver": ScenarioTemplate(
        family="excluded_driver",
        title="Named-driver exclusion surfaced in review",
        level=4,
        denial_clause="excluded_driver",
        statement="My roommate was borrowing the car when the front end was damaged.",
        visible_estimate_flags=("Driver identity requires coverage review under listed exclusions.",),
        operational_rubric=(
            "Verify policy exclusions before approving payment.",
            "Deny only with the correct exclusion rationale.",
        ),
    ),
    "subrogation_opportunity": ScenarioTemplate(
        family="subrogation_opportunity",
        title="Clear rear-end third-party recovery",
        level=5,
        statement="I was rear-ended by another driver who admitted fault at the scene.",
        subrogation=True,
        needs_police_evidence=True,
        operational_rubric=(
            "Pay insured under collision coverage after deductible.",
            "Open recovery against the liable third party.",
        ),
    ),
    "authority_threshold": ScenarioTemplate(
        family="authority_threshold",
        title="Payment above adjuster authority",
        level=5,
        authority_escalation=True,
        operational_rubric=(
            "Request authority approval before payment above limit.",
            "Keep reserves and payments in pending-authority state until approval arrives.",
        ),
    ),
    "total_loss": ScenarioTemplate(
        family="total_loss",
        title="Possible total loss with valuation and salvage",
        level=5,
        statement="The vehicle has severe front-end damage and the tow yard says it may be a total loss.",
        gross_range=(12000, 14500),
        expected_review=EstimateReviewDecision.CONFIRM_TOTAL_LOSS,
        expected_total_loss=True,
        operational_rubric=(
            "Confirm total-loss review when repair cost nears threshold.",
            "Request valuation before final settlement.",
        ),
    ),
}


SCENARIO_FAMILIES: tuple[str, ...] = tuple(SCENARIO_TEMPLATES)


def get_template(family: str) -> ScenarioTemplate:
    try:
        return SCENARIO_TEMPLATES[family]
    except KeyError as exc:
        raise ValueError(f"unknown scenario family: {family}") from exc
