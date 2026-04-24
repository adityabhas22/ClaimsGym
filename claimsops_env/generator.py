from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from claimsops_env.models import (
    Activity,
    AppraisalStatus,
    ClaimDocument,
    ClaimScenario,
    ClaimType,
    CoverageType,
    DocumentType,
    Evidence,
    EvidenceKind,
    EstimateLineItem,
    EstimateReviewDecision,
    Exposure,
    ExposureStatus,
    HiddenTruth,
    Incident,
    Party,
    PendingEvent,
    PlatformState,
    Policy,
    RepairEstimate,
    ReserveBand,
    WorkflowRubric,
)
from claimsops_env.scenario_templates import SCENARIO_FAMILIES, ScenarioTemplate, build_rubric, get_template


@dataclass(frozen=True)
class EpisodeSpec:
    policy: Policy
    claim: ClaimScenario
    repair_estimate: RepairEstimate
    platform_state: PlatformState
    hidden: HiddenTruth
    prior_claim_summaries: list[str]
    rubric: WorkflowRubric


class ScenarioGenerator:
    """Deterministic synthetic auto-claim generator.

    The generator intentionally produces hidden labels alongside visible claim
    records so verifiers can score behavior without exposing answers to agents.
    """

    def generate(self, seed: int | None = None) -> EpisodeSpec:
        rng = random.Random(seed)
        family = rng.choice(SCENARIO_FAMILIES)
        return self._build_family(family, rng)

    def generate_family(self, family: str, seed: int | None = None) -> EpisodeSpec:
        return self._build_family(family, random.Random(seed))

    def _build_family(self, family: str, rng: random.Random) -> EpisodeSpec:
        template = get_template(family)
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
        claim_type = ClaimType.COMPREHENSIVE if template.claim_type == "comprehensive" else ClaimType.COLLISION
        gross = float(rng.randint(*template.gross_range))
        unrelated = template.unrelated_damage
        duplicate = template.duplicate_line
        required_docs: set[DocumentType] = set(template.required_documents)
        suspicious = template.suspicious
        subrogation = template.subrogation
        status = template.policy_status
        denial_clause = template.denial_clause
        coverage_reason = "Active policy with covered auto physical damage loss."
        if denial_clause == "policy_period":
            coverage_reason = "Loss occurred after the policy expiration date."
        elif denial_clause == "excluded_driver":
            coverage_reason = "The reported driver is excluded under the policy driver exclusion."
        statement = template.statement or "I was hit while stopped at a light. The rear bumper and trunk are damaged."
        prior_claims: list[str] = list(template.prior_claims)
        difficulty = template.level
        authority_escalation = template.authority_escalation
        notes: list[str] = list(template.visible_estimate_flags)
        expected_review = template.expected_review
        expected_total_loss = template.expected_total_loss
        liability_split = template.liability_split_insured_pct
        photo_quality = template.photo_quality

        if family == "policy_lapse":
            loss_date = expiration + timedelta(days=7)
        elif family == "limit_exceeded":
            gross = collision_limit + rng.randint(1500, 3500)
        elif family == "suspicious_inception":
            effective = loss_date - timedelta(days=2)
        elif family == "authority_threshold":
            gross = authority_limit + rng.randint(800, 1800)
        elif family == "total_loss":
            gross = float(rng.randint(12000, 14500))
            collision_limit = max(collision_limit, 15000.0)
        elif family == "excluded_driver":
            status = "active"

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
        if family == "excluded_driver":
            policy.exclusions.append("excluded_driver")
        covered_amount = max(0.0, gross - unrelated - duplicate)
        if denial_clause or status != "active" or not (policy.effective_date <= loss_date <= policy.expiration_date):
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
        if template.telematics_conflict:
            evidence.append(
                Evidence(
                    evidence_id="EV-TELEMATICS",
                    kind=EvidenceKind.TELEMATICS,
                    summary="Telematics shows motion and front-end deceleration at time of loss.",
                    flags=["statement_conflict"],
                )
            )
        if template.needs_police_evidence:
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
            labor_hours=round(covered_amount / 120.0, 1),
            parts_amount=round(covered_amount * 0.42, 2),
            paint_materials_amount=round(covered_amount * 0.12, 2),
            total_loss_threshold=round(policy.limit_for(claim_type) * 0.75, 2),
            photo_quality=photo_quality,  # type: ignore[arg-type]
            line_items=self._estimate_line_items(
                estimate_id=estimate_id,
                claim_type=claim_type,
                covered_amount=covered_amount,
                unrelated=unrelated,
                duplicate=duplicate,
                family=family,
            ),
        )
        platform_state = self._platform_state(
            claim=claim,
            claim_type=claim_type,
            covered_amount=covered_amount,
            difficulty=difficulty,
            subrogation=subrogation,
            suspicious=suspicious,
            expected_total_loss=expected_total_loss,
            template=template,
            required_docs=required_docs,
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
            rubric=template.rubric or build_rubric(template),
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
        template: ScenarioTemplate,
        required_docs: set[DocumentType],
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
            contact_status=template.contact_status,  # type: ignore[arg-type]
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
        for category in template.extra_activity_categories:
            activities.append(
                Activity(
                    activity_id=f"ACT-{category.upper().replace('_', '-')}",
                    subject=f"Manage {category.replace('_', ' ')} exposure and leakage",
                    category=category,
                    due_in_steps=2,
                    priority="high" if category in {"towing_storage", "rental"} else "normal",
                    related_object_id=claim.claim_id,
                )
            )
        pending_events = [
            PendingEvent(
                event_id=f"EVT-{profile.upper().replace('_', '-')}-001",
                event_type=profile,  # type: ignore[arg-type]
                due_in_steps=1,
                summary=f"System will update {profile.replace('_', ' ')} if no action is taken.",
                payload={"source": "scenario_profile"},
            )
            for profile in template.initial_event_profiles
        ]
        documents = [
            ClaimDocument(
                document_id="DOC-FNOL-STATEMENT",
                doc_type=DocumentType.CLAIMANT_STATEMENT,
                title="First notice claimant statement",
                source="claimant",
                status="received",
                evidence_id="EV-STATEMENT",
                confidence="medium",
                summary=claim.claimant_statement,
                issues=["incomplete"] if DocumentType.CLAIMANT_STATEMENT in required_docs else [],
                related_object_id=claim.claim_id,
            ),
            ClaimDocument(
                document_id=f"DOC-{claim.estimate_id}",
                doc_type=DocumentType.REPAIR_ESTIMATE_BREAKDOWN,
                title="Initial repair estimate",
                source="vendor",
                status="received",
                evidence_id=claim.estimate_id,
                confidence="high" if not template.visible_estimate_flags else "medium",
                summary="Initial estimate imported with line-item detail.",
                issues=list(template.visible_estimate_flags),
                related_object_id=claim.estimate_id,
            ),
        ]
        return PlatformState(
            parties=parties,
            incidents=[incident],
            exposures=[exposure],
            activities=activities,
            documents=documents,
            pending_events=pending_events,
            appraisal_status=AppraisalStatus.NOT_ASSIGNED,
        )

    @staticmethod
    def _reserve_band(amount: float) -> ReserveBand:
        if amount < 3500:
            return ReserveBand.LOW
        if amount < 8000:
            return ReserveBand.MEDIUM
        return ReserveBand.HIGH

    @staticmethod
    def _estimate_line_items(
        estimate_id: str,
        claim_type: ClaimType,
        covered_amount: float,
        unrelated: float,
        duplicate: float,
        family: str,
    ) -> list[EstimateLineItem]:
        coverage = CoverageType.COMPREHENSIVE if claim_type == ClaimType.COMPREHENSIVE else CoverageType.COLLISION
        expense_amount = 360.0 if family == "rental_storage_leakage" else 0.0
        repair_amount = max(0.0, covered_amount - expense_amount)
        base_lines = [
            (
                "labor",
                "Body labor for covered loss damage",
                round(repair_amount * 0.34, 2),
                ["covered_damage"],
            ),
            (
                "parts",
                "Replacement parts for covered loss damage",
                round(repair_amount * 0.42, 2),
                ["covered_damage"],
            ),
            (
                "paint_materials",
                "Paint and materials for covered panels",
                round(repair_amount * 0.14, 2),
                ["covered_damage"],
            ),
        ]
        used = sum(line[2] for line in base_lines)
        base_lines.append(
            (
                "tax_fee",
                "Shop supplies and taxable fees",
                round(max(0.0, repair_amount - used), 2),
                ["covered_damage"],
            )
        )
        items = [
            EstimateLineItem(
                line_id=f"{estimate_id}-L{index:02d}",
                category=category,  # type: ignore[arg-type]
                description=description,
                amount=amount,
                coverage=coverage,
                payable=True,
                flags=flags,
            )
            for index, (category, description, amount, flags) in enumerate(base_lines, start=1)
            if amount > 0
        ]
        if unrelated:
            items.append(
                EstimateLineItem(
                    line_id=f"{estimate_id}-L{len(items) + 1:02d}",
                    category="prior_damage",
                    description="Left quarter-panel repair unrelated to reported loss",
                    amount=round(unrelated, 2),
                    coverage=coverage,
                    payable=False,
                    review_status="questioned",
                    flags=["prior_damage", "not_loss_related"],
                )
            )
        if duplicate:
            items.append(
                EstimateLineItem(
                    line_id=f"{estimate_id}-L{len(items) + 1:02d}",
                    category="duplicate",
                    description="Duplicate paint materials operation",
                    amount=round(duplicate, 2),
                    coverage=coverage,
                    payable=False,
                    review_status="questioned",
                    flags=["duplicate_line"],
                )
            )
        if family == "rental_storage_leakage":
            items.extend(
                [
                    EstimateLineItem(
                        line_id=f"{estimate_id}-L{len(items) + 1:02d}",
                        category="storage",
                        description="Initial tow-yard storage charge",
                        amount=150.0,
                        coverage=coverage,
                        payable=True,
                        flags=["expense_leakage"],
                    ),
                    EstimateLineItem(
                        line_id=f"{estimate_id}-L{len(items) + 2:02d}",
                        category="rental",
                        description="Rental authorization pending repair review",
                        amount=210.0,
                        coverage=coverage,
                        payable=True,
                        flags=["expense_leakage"],
                    ),
                ]
            )
        return items
