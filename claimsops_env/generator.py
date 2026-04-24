from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from claimsops_env.models import (
    ClaimScenario,
    ClaimType,
    DocumentType,
    Evidence,
    EvidenceKind,
    HiddenTruth,
    Policy,
    RepairEstimate,
    ReserveBand,
)


@dataclass(frozen=True)
class EpisodeSpec:
    policy: Policy
    claim: ClaimScenario
    repair_estimate: RepairEstimate
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
            difficulty = 2
        elif family == "prior_damage_leakage":
            unrelated = 1100.0
            prior_claims.append("Prior left quarter-panel damage paid 2025-11-04.")
            notes.append("Left quarter-panel line appears unrelated to described rear impact.")
            difficulty = 3
        elif family == "duplicate_line_item":
            duplicate = 425.0
            notes.append("Paint materials line appears duplicated.")
            difficulty = 3
        elif family == "suspicious_inception":
            effective = loss_date - timedelta(days=2)
            suspicious = True
            required_docs.add(DocumentType.POLICE_REPORT)
            notes.append("Loss reported shortly after policy inception.")
            difficulty = 4
        elif family == "conflicting_statement":
            suspicious = True
            statement = "I was parked when another car struck the front bumper."
            notes.append("Telematics indicates vehicle was moving at impact.")
            difficulty = 4
        elif family == "subrogation_opportunity":
            subrogation = True
            statement = "I was rear-ended by another driver who admitted fault at the scene."
            difficulty = 5
        elif family == "authority_threshold":
            gross = authority_limit + rng.randint(800, 1800)
            authority_escalation = True
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
            step_budget=12 if difficulty >= 3 else 8,
        )
        repair_estimate = RepairEstimate(
            estimate_id=estimate_id,
            gross_amount=gross,
            covered_amount=covered_amount,
            unrelated_damage_amount=unrelated,
            duplicate_line_amount=duplicate,
            notes=notes,
        )
        hidden = HiddenTruth(
            is_covered=is_covered,
            coverage_reason=coverage_reason,
            expected_payable=expected_payable,
            required_documents=required_docs,
            fraud_suspicious=suspicious,
            subrogation_expected=subrogation,
            expected_reserve_band=reserve_band,
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
            hidden=hidden,
            prior_claim_summaries=prior_claims,
        )

    @staticmethod
    def _reserve_band(amount: float) -> ReserveBand:
        if amount < 3500:
            return ReserveBand.LOW
        if amount < 8000:
            return ReserveBand.MEDIUM
        return ReserveBand.HIGH
