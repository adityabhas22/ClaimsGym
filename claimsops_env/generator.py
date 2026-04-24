from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from claimsops_env.models import (
    Activity,
    AppraisalStatus,
    ClaimScenario,
    ClaimType,
    CoverageType,
    DocumentType,
    Evidence,
    EvidenceKind,
    EstimateReviewDecision,
    Exposure,
    ExposureStatus,
    HiddenTruth,
    Incident,
    Party,
    PlatformState,
    Policy,
    RepairEstimate,
    ReserveBand,
)


@dataclass(frozen=True)
class EpisodeSpec:
    policy: Policy
    claim: ClaimScenario
    repair_estimate: RepairEstimate
    platform_state: PlatformState
    hidden: HiddenTruth
    prior_claim_summaries: list[str]


class ScenarioGenerator:
    """Deterministic synthetic auto-claim generator.

    The generator intentionally produces hidden labels alongside visible claim
    records so verifiers can score behavior without exposing answers to agents.
    """

    def generate(self, seed: int | None = None) -> EpisodeSpec:
        rng = random.Random(seed)
        family = rng.choice(
            [
                "covered_collision",
                "comprehensive_deductible",
                "policy_lapse",
                "limit_exceeded",
                "missing_police_report",
                "prior_damage_leakage",
                "duplicate_line_item",
                "suspicious_inception",
                "conflicting_statement",
                "subrogation_opportunity",
                "authority_threshold",
                "total_loss",
            ]
        )
        return self._build_family(family, rng)

    def generate_family(self, family: str, seed: int | None = None) -> EpisodeSpec:
        return self._build_family(family, random.Random(seed))

    def _build_family(self, family: str, rng: random.Random) -> EpisodeSpec:
        suffix = rng.randint(1000, 9999)
        customer_id = f"CUST-{suffix}"
        vehicle_id = f"VIN-{rng.randint(10000, 99999)}"
        policy_id = f"POL-{suffix}"
        claim_id = f"CLM-{rng.randint(10000, 99999)}"
        estimate_id = f"EST-{rng.randint(10000, 99999)}"

        effective = date(2026, 1, 1)
        expiration = date(2026, 12, 31)
        loss_date = date(2026, 2, 12) + timedelta(days=rng.randint(0, 45))
        reported_date = loss_date + timedelta(days=rng.randint(0, 4))
        deductible = rng.choice([250.0, 500.0, 1000.0])
        collision_limit = rng.choice([7500.0, 10000.0, 15000.0])
        comprehensive_limit = rng.choice([5000.0, 7500.0, 10000.0])
        authority_limit = rng.choice([6500.0, 8000.0])
        claim_type = ClaimType.COLLISION
        gross = float(rng.randint(3200, 8800))
        unrelated = 0.0
        duplicate = 0.0
        required_docs: set[DocumentType] = set()
        suspicious = False
        subrogation = False
        status = "active"
        denial_clause = None
        coverage_reason = "Active policy with covered auto physical damage loss."
        statement = "I was hit while stopped at a light. The rear bumper and trunk are damaged."
        prior_claims: list[str] = []
        difficulty = 1
        authority_escalation = False
        notes: list[str] = []
        expected_review = EstimateReviewDecision.APPROVE
        expected_total_loss = False
        liability_split = 0
        photo_quality = "good"

        if family == "comprehensive_deductible":
            claim_type = ClaimType.COMPREHENSIVE
            statement = "A tree branch fell during a storm and damaged the hood and windshield."
            difficulty = 1
        elif family == "policy_lapse":
            status = "lapsed"
            loss_date = expiration + timedelta(days=7)
            denial_clause = "policy_period"
            coverage_reason = "Loss occurred after the policy expiration date."
            difficulty = 1
        elif family == "limit_exceeded":
            gross = collision_limit + rng.randint(1500, 3500)
            notes.append("Estimate exceeds collision coverage limit.")
            difficulty = 1
        elif family == "missing_police_report":
            required_docs.add(DocumentType.POLICE_REPORT)
            statement = "A third-party vehicle backed into my car in a parking lot; report number is pending."
            subrogation = True
            liability_split = 0
            difficulty = 2
        elif family == "prior_damage_leakage":
            unrelated = 1100.0
            required_docs.add(DocumentType.REPAIR_ESTIMATE_BREAKDOWN)
            prior_claims.append("Prior left quarter-panel damage paid 2025-11-04.")
            notes.append("Left quarter-panel line appears unrelated to described rear impact.")
            expected_review = EstimateReviewDecision.ESCALATE_FIELD
            difficulty = 3
        elif family == "duplicate_line_item":
            duplicate = 425.0
            required_docs.add(DocumentType.REPAIR_ESTIMATE_BREAKDOWN)
            notes.append("Paint materials line appears duplicated.")
            expected_review = EstimateReviewDecision.REQUEST_SUPPLEMENT
            difficulty = 3
        elif family == "suspicious_inception":
            effective = loss_date - timedelta(days=2)
            suspicious = True
            required_docs.add(DocumentType.POLICE_REPORT)
            notes.append("Loss reported shortly after policy inception.")
            expected_review = EstimateReviewDecision.REQUEST_PHOTOS
            photo_quality = "partial"
            difficulty = 4
        elif family == "conflicting_statement":
            suspicious = True
            statement = "I was parked when another car struck the front bumper."
            notes.append("Telematics indicates vehicle was moving at impact.")
            expected_review = EstimateReviewDecision.ESCALATE_FIELD
            difficulty = 4
        elif family == "subrogation_opportunity":
            subrogation = True
            statement = "I was rear-ended by another driver who admitted fault at the scene."
            liability_split = 0
            difficulty = 5
        elif family == "authority_threshold":
            gross = authority_limit + rng.randint(800, 1800)
            authority_escalation = True
            difficulty = 5
        elif family == "total_loss":
            gross = float(rng.randint(12000, 14500))
            collision_limit = max(collision_limit, 15000.0)
            expected_review = EstimateReviewDecision.CONFIRM_TOTAL_LOSS
            expected_total_loss = True
            statement = "The vehicle has severe front-end damage and the tow yard says it may be a total loss."
            difficulty = 5

        policy = Policy(
            policy_id=policy_id,
            customer_id=customer_id,
            vehicle_id=vehicle_id,
            effective_date=effective,
            expiration_date=expiration,
            status=status,
            deductible=deductible,
            collision_limit=collision_limit,
            comprehensive_limit=comprehensive_limit,
            authority_limit=authority_limit,
            exclusions=["wear_and_tear", "intentional_loss", "commercial_use"],
        )
        covered_amount = max(0.0, gross - unrelated - duplicate)
        if status != "active" or not (policy.effective_date <= loss_date <= policy.expiration_date):
            is_covered = False
            expected_payable = 0.0
        else:
            is_covered = True
            expected_payable = max(0.0, min(covered_amount, policy.limit_for(claim_type)) - policy.deductible)

        reserve_band = self._reserve_band(expected_payable if is_covered else gross)
        evidence = [
            Evidence(
                evidence_id="EV-STATEMENT",
                kind=EvidenceKind.CLAIMANT_STATEMENT,
                summary=statement,
            ),
            Evidence(
                evidence_id=estimate_id,
                kind=EvidenceKind.REPAIR_ESTIMATE,
                summary=f"Initial repair estimate totals ${gross:,.2f}.",
                amount=gross,
                flags=notes,
            ),
        ]
        if family == "conflicting_statement":
            evidence.append(
                Evidence(
                    evidence_id="EV-TELEMATICS",
                    kind=EvidenceKind.TELEMATICS,
                    summary="Telematics shows motion and front-end deceleration at time of loss.",
                    flags=["statement_conflict"],
                )
            )
        if family in {"subrogation_opportunity", "missing_police_report"}:
            evidence.append(
                Evidence(
                    evidence_id="EV-POLICE-AVAILABLE",
                    kind=EvidenceKind.POLICE_REPORT,
                    summary="Police narrative identifies another driver as likely at fault.",
                    flags=["third_party_fault"],
                )
            )

        claim = ClaimScenario(
            claim_id=claim_id,
            policy_id=policy_id,
            customer_id=customer_id,
            vehicle_id=vehicle_id,
            claim_type=claim_type,
            loss_date=loss_date,
            reported_date=reported_date,
            claimant_statement=statement,
            requested_amount=gross,
            estimate_id=estimate_id,
            family=family,
            difficulty=difficulty,
            initial_evidence=evidence,
            step_budget=18 if difficulty >= 4 else 16 if difficulty >= 2 else 14,
        )
        repair_estimate = RepairEstimate(
            estimate_id=estimate_id,
            gross_amount=gross,
            covered_amount=covered_amount,
            unrelated_damage_amount=unrelated,
            duplicate_line_amount=duplicate,
            notes=notes,
            labor_hours=round(gross / 120.0, 1),
            parts_amount=round(gross * 0.42, 2),
            paint_materials_amount=round(gross * 0.12, 2),
            total_loss_threshold=round(policy.limit_for(claim_type) * 0.75, 2),
            photo_quality=photo_quality,  # type: ignore[arg-type]
        )
        platform_state = self._platform_state(
            claim=claim,
            claim_type=claim_type,
            covered_amount=covered_amount,
            difficulty=difficulty,
            subrogation=subrogation,
            suspicious=suspicious,
            expected_total_loss=expected_total_loss,
        )
        min_reserve = max(500.0, expected_payable * 0.85 if expected_payable else gross * 0.5)
        max_reserve = max(min_reserve + 250.0, expected_payable * 1.25 if expected_payable else gross * 1.1)
        hidden = HiddenTruth(
            is_covered=is_covered,
            coverage_reason=coverage_reason,
            expected_payable=expected_payable,
            required_documents=required_docs,
            fraud_suspicious=suspicious,
            subrogation_expected=subrogation,
            expected_reserve_band=reserve_band,
            expected_estimate_review=expected_review,
            expected_exposure_coverages={CoverageType.COMPREHENSIVE if claim_type == ClaimType.COMPREHENSIVE else CoverageType.COLLISION},
            expected_activity_categories={"coverage", "estimate", "reserve", "communication", "closure"},
            expected_min_reserve=round(min_reserve, 2),
            expected_max_reserve=round(max_reserve, 2),
            expected_total_loss=expected_total_loss,
            expected_salvage_value=round(gross * 0.18, 2) if expected_total_loss else 0.0,
            liability_split_insured_pct=liability_split,
            authority_escalation_required=authority_escalation,
            denial_clause=denial_clause,
            protected_fields={
                "policy_id": policy_id,
                "claim_id": claim_id,
                "customer_id": customer_id,
                "vehicle_id": vehicle_id,
            },
        )
        return EpisodeSpec(
            policy=policy,
            claim=claim,
            repair_estimate=repair_estimate,
            platform_state=platform_state,
            hidden=hidden,
            prior_claim_summaries=prior_claims,
        )

    def _platform_state(
        self,
        claim: ClaimScenario,
        claim_type: ClaimType,
        covered_amount: float,
        difficulty: int,
        subrogation: bool,
        suspicious: bool,
        expected_total_loss: bool,
    ) -> PlatformState:
        insured = Party(
            party_id=claim.customer_id,
            role="insured",
            display_name="Synthetic Insured",
        )
        claimant = Party(
            party_id="PTY-CLAIMANT",
            role="claimant",
            display_name="Synthetic Claimant",
        )
        parties = [insured, claimant]
        if subrogation:
            parties.append(
                Party(
                    party_id="PTY-ADVERSE-DRIVER",
                    role="adverse_driver",
                    display_name="Other Driver",
                )
            )
        incident = Incident(
            incident_id="INC-VEHICLE-1",
            incident_type="vehicle_damage",
            vehicle_id=claim.vehicle_id,
            description=claim.claimant_statement,
            severity="possible_total_loss" if expected_total_loss else ("severe" if covered_amount > 8000 else "moderate"),
            liability_signal="third_party_fault" if subrogation else "unknown",
        )
        exposure = Exposure(
            exposure_id="EXP-VEHICLE-1",
            coverage=CoverageType.COMPREHENSIVE if claim_type == ClaimType.COMPREHENSIVE else CoverageType.COLLISION,
            claimant_id=claim.customer_id,
            incident_id=incident.incident_id,
            status=ExposureStatus.OPEN,
            validation_level="coverage_pending",
        )
        activities = [
            Activity(
                activity_id="ACT-COVERAGE",
                subject="Verify policy coverage and term status",
                category="coverage",
                due_in_steps=2,
                priority="high",
                related_object_id=exposure.exposure_id,
            ),
            Activity(
                activity_id="ACT-ESTIMATE",
                subject="Review repair estimate and damage evidence",
                category="estimate",
                due_in_steps=3,
                priority="normal",
                related_object_id=claim.estimate_id,
            ),
            Activity(
                activity_id="ACT-RESERVE",
                subject="Set initial reserve for open exposure",
                category="reserve",
                due_in_steps=2,
                priority="high",
                related_object_id=exposure.exposure_id,
            ),
            Activity(
                activity_id="ACT-COMMUNICATION",
                subject="Send claimant status update before closure",
                category="communication",
                due_in_steps=5,
                priority="normal",
                related_object_id=claim.claim_id,
            ),
        ]
        if suspicious:
            activities.append(
                Activity(
                    activity_id="ACT-SIU-SCREEN",
                    subject="Complete SIU indicator screen",
                    category="siu",
                    due_in_steps=3,
                    priority="high",
                    related_object_id=claim.claim_id,
                )
            )
        if subrogation:
            activities.append(
                Activity(
                    activity_id="ACT-SUBRO",
                    subject="Evaluate recovery potential",
                    category="subrogation",
                    due_in_steps=4,
                    priority="normal",
                    related_object_id=claim.claim_id,
                )
            )
        if difficulty >= 5:
            activities.append(
                Activity(
                    activity_id="ACT-AUTHORITY",
                    subject="Check authority threshold before payment",
                    category="authority",
                    due_in_steps=4,
                    priority="high",
                    related_object_id=exposure.exposure_id,
                )
            )
        return PlatformState(
            parties=parties,
            incidents=[incident],
            exposures=[exposure],
            activities=activities,
            appraisal_status=AppraisalStatus.NOT_ASSIGNED,
        )

    @staticmethod
    def _reserve_band(amount: float) -> ReserveBand:
        if amount < 3500:
            return ReserveBand.LOW
        if amount < 8000:
            return ReserveBand.MEDIUM
        return ReserveBand.HIGH
