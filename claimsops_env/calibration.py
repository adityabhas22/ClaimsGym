from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

from pydantic import BaseModel, Field

from claimsops_env.agent_interface import AgentContext, ActionPolicy, RolloutResult, RolloutRunner
from claimsops_env.models import Observation
from claimsops_env.scenario_templates import SCENARIO_FAMILIES, SCENARIO_TEMPLATES


class CalibrationRow(BaseModel):
    scenario_family: str
    seed: int
    behavior: str
    expected_quality: str
    description: str
    total_reward: float
    reward_breakdown: dict[str, Any]
    rubric_overall: float = 0.0
    missed_must: list[str] = Field(default_factory=list)
    missed_final: list[str] = Field(default_factory=list)
    violated_forbidden: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    safety_cap: float = 1.0
    penalties: float = 0.0
    verdict: str
    expected_passed: bool
    notes: list[str] = Field(default_factory=list)
    steps: int
    done: bool
    rollout: dict[str, Any] | None = None


class OrderingFailure(BaseModel):
    scenario_family: str
    seed: int
    better_behavior: str
    worse_behavior: str
    better_reward: float
    worse_reward: float
    margin: float


class CalibrationReport(BaseModel):
    rows: list[CalibrationRow]
    ordering_failures: list[OrderingFailure] = Field(default_factory=list)

    @property
    def expectations_passed(self) -> bool:
        return all(row.expected_passed for row in self.rows)

    @property
    def ordering_passed(self) -> bool:
        return not self.ordering_failures

    @property
    def passed(self) -> bool:
        return self.expectations_passed and self.ordering_passed

    def to_markdown(self) -> str:
        expected_failures = [row for row in self.rows if not row.expected_passed]
        lines = [
            "# ClaimsOps Reward Calibration",
            "",
            f"- episodes: `{len(self.rows)}`",
            f"- calibration_passed: `{self.passed}`",
            f"- expected_passed: `{len(self.rows) - len(expected_failures)}/{len(self.rows)}`",
            f"- ordering_passed: `{self.ordering_passed}`",
            "",
            "| family | seed | behavior | expected | verdict | ok | reward | cap | penalties | rubric | misses | notes |",
            "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in self.rows:
            misses = len(row.missed_must) + len(row.missed_final) + len(row.violated_forbidden)
            notes = "; ".join(row.notes[:3])
            lines.append(
                f"| {row.scenario_family} | {row.seed} | {row.behavior} | {row.expected_quality} | "
                f"{row.verdict} | {row.expected_passed} | {row.total_reward:.3f} | "
                f"{row.safety_cap:.3f} | {row.penalties:.3f} | {row.rubric_overall:.3f} | "
                f"{misses} | {notes} |"
            )
        if expected_failures:
            lines.extend(["", "## Expectation Failures"])
            for row in expected_failures:
                lines.append(
                    "- "
                    f"{row.scenario_family} seed={row.seed} behavior=`{row.behavior}` "
                    f"expected `{row.expected_quality}` but measured `{row.verdict}` "
                    f"(reward={row.total_reward:.3f}, rubric={row.rubric_overall:.3f})"
                )
        if self.ordering_failures:
            lines.extend(["", "## Ordering Failures"])
            for failure in self.ordering_failures:
                lines.append(
                    "- "
                    f"{failure.scenario_family} seed={failure.seed}: expected `{failure.better_behavior}` "
                    f"> `{failure.worse_behavior}`, got {failure.better_reward:.3f} <= "
                    f"{failure.worse_reward:.3f} + margin {failure.margin:.3f}"
                )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    expected_quality: str
    description: str
    policy: ActionPolicy
    rank: int
    family_expectations: dict[str, str] = dataclass_field(default_factory=dict)

    def quality_for(self, family: str) -> str:
        return self.family_expectations.get(family, self.expected_quality)


def default_behaviors() -> list[BehaviorSpec]:
    document_gap_families = {
        family
        for family, template in SCENARIO_TEMPLATES.items()
        if template.required_documents
    }
    return [
        BehaviorSpec(
            name="careful_adjuster",
            expected_quality="good",
            description="Completes visible claim workflow, waits on events, reviews returned documents, and finalizes with supported payment.",
            policy=CarefulAdjusterPolicy(),
            rank=0,
        ),
        BehaviorSpec(
            name="missing_evidence",
            expected_quality="neutral",
            description="Handles basic workflow but ignores material document requests and closes without document review where documents matter.",
            policy=MissingEvidencePolicy(),
            rank=1,
            family_expectations={family: "bad" for family in document_gap_families},
        ),
        BehaviorSpec(
            name="siu_everything",
            expected_quality="bad",
            description="Over-refers claims to SIU regardless of visible support.",
            policy=SiuEverythingPolicy(),
            rank=1,
            family_expectations={
                "suspicious_inception": "neutral",
                "conflicting_statement": "neutral",
            },
        ),
        BehaviorSpec(
            name="overpay",
            expected_quality="bad",
            description="Performs light investigation then overpays relative to deductible, limits, and payable line items.",
            policy=OverpayPolicy(),
            rank=2,
        ),
        BehaviorSpec(
            name="premature_final",
            expected_quality="bad",
            description="Submits a final decision immediately with minimal evidence.",
            policy=PrematureFinalPolicy(),
            rank=3,
        ),
        BehaviorSpec(
            name="authority_bypass",
            expected_quality="bad",
            description="Attempts payment before required authority approval.",
            policy=AuthorityBypassPolicy(),
            rank=3,
        ),
    ]


def run_calibration(
    *,
    families: list[str] | None = None,
    seeds: list[int] | None = None,
    behaviors: list[BehaviorSpec] | None = None,
    include_rollouts: bool = False,
    ordering_margin: float = 0.02,
) -> CalibrationReport:
    selected_families = families or list(SCENARIO_FAMILIES)
    selected_seeds = seeds or [0]
    selected_behaviors = behaviors or default_behaviors()
    rows: list[CalibrationRow] = []
    for family in selected_families:
        for seed in selected_seeds:
            for behavior in selected_behaviors:
                rollout = RolloutRunner().run(behavior.policy, seed=seed, scenario_family=family)
                rows.append(_row_from_rollout(family, seed, behavior, rollout, include_rollouts))
    return CalibrationReport(
        rows=rows,
        ordering_failures=_ordering_failures(rows, selected_behaviors, ordering_margin),
    )


class CarefulAdjusterPolicy(ActionPolicy):
    request_documents = True
    inspect_documents = True
    force_siu_referral = False

    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        if "verify_policy" in observation.open_tasks:
            return {"tool": "get_policy", "args": {"policy_id": observation.policy_id}}
        if "verify_coverage" in observation.open_tasks:
            return {
                "tool": "verify_coverage",
                "args": {
                    "claim_id": observation.claim_id,
                    "exposure_id": observation.exposures[0].exposure_id,
                    "loss_facts": observation.claimant_statement,
                },
            }
        if "assign_appraisal" in observation.open_tasks:
            method = "field" if "possible_prior_damage" in observation.alerts else "photo"
            return {"tool": "assign_appraisal", "args": {"claim_id": observation.claim_id, "method": method}}
        if "inspect_estimate" in observation.open_tasks:
            return {"tool": "inspect_repair_estimate", "args": {"estimate_id": observation.estimate_id}}
        if "review_estimate" in observation.open_tasks:
            return {
                "tool": "review_estimate",
                "args": {
                    "claim_id": observation.claim_id,
                    "estimate_id": observation.estimate_id,
                    "action": _estimate_review_action(observation),
                    "rationale": "Estimate reviewed against visible line-item and workflow indicators.",
                },
            }
        if "request_valuation" in observation.open_tasks:
            return {"tool": "request_valuation", "args": {"claim_id": observation.claim_id, "reason": "Damage is near total-loss threshold."}}
        if "request_authority_approval" in observation.open_tasks:
            return {
                "tool": "request_authority_approval",
                "args": {
                    "exposure_id": observation.exposures[0].exposure_id,
                    "amount": observation.requested_amount,
                    "rationale": "Visible exposure exceeds adjuster authority threshold.",
                },
            }
        if self.request_documents:
            for task in observation.open_tasks:
                if task.startswith("request_"):
                    return {
                        "tool": "request_document",
                        "args": {
                            "doc_type": task.removeprefix("request_"),
                            "reason": "Material to coverage, liability, or estimate review.",
                        },
                    }
        if "await_pending_events" in observation.open_tasks:
            return _waiting_note_action(observation)
        if self.inspect_documents:
            document = _received_unreviewed_document(observation)
            if document and document.get("evidence_id"):
                return {"tool": "inspect_evidence", "args": {"evidence_id": document["evidence_id"]}}
        if "screen_fraud_indicators" in observation.open_tasks:
            return {"tool": "check_fraud_indicators", "args": {"claim_id": observation.claim_id}}
        if _should_query_prior_claims(observation, context):
            return {
                "tool": "query_prior_claims",
                "args": {"customer_id": observation.customer_id, "vehicle_id": observation.vehicle_id},
            }
        if _should_open_siu(observation, context, force=self.force_siu_referral):
            return {"tool": "open_siu_referral", "args": {"reason": "Visible fraud indicators require SIU review.", "evidence_ids": _evidence_ids(observation)}}
        if "evaluate_subrogation" in observation.open_tasks:
            return {
                "tool": "open_subrogation",
                "args": {"target_party": "third_party_driver", "rationale": "Visible evidence indicates another party may be liable."},
            }
        if "set_reserve" in observation.open_tasks:
            estimate = _latest_success_data(context, "inspect_repair_estimate")
            amount = float(estimate.get("covered_amount") or observation.requested_amount)
            return {
                "tool": "set_reserve",
                "args": {
                    "exposure_id": observation.exposures[0].exposure_id,
                    "amount": amount,
                    "cost_category": "auto_body",
                    "rationale": "Reserve set from visible covered exposure.",
                },
            }
        if "send_claimant_update" in observation.open_tasks:
            return {
                "tool": "send_claimant_message",
                "args": {
                    "claim_id": observation.claim_id,
                    "message": "We are reviewing coverage, damage, documents, and the deductible. We will contact you with the next step.",
                },
            }
        if "add_closure_note" in observation.open_tasks:
            return {
                "tool": "add_claim_note",
                "args": {
                    "claim_id": observation.claim_id,
                    "note_type": "closure",
                    "subject": "Final claim review",
                    "body": "Policy, coverage, documents, estimate, reserve, communication, and audit evidence reviewed before disposition.",
                    "related_object_id": observation.claim_id,
                },
            }
        return self._final_decision(observation, context)

    def _final_decision(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        policy = observation.visible_policy or {}
        estimate = _latest_success_data(context, "inspect_repair_estimate")
        status = observation.coverage_result or _latest_success_data(context, "verify_coverage")
        fraud = _latest_success_data(context, "check_fraud_indicators")
        active_on_loss = bool(status.get("active_on_loss")) if status else policy.get("status") == "active"
        covered = bool(status.get("covered", active_on_loss))
        payment = 0.0
        decision = "deny"
        rationale = "Coverage could not be confirmed under the policy period or applicable exclusions."
        if covered and policy and estimate:
            limit_key = "comprehensive_limit" if observation.claim_type.value == "comprehensive" else "collision_limit"
            payment = max(0.0, min(float(estimate["covered_amount"]), float(policy[limit_key])) - float(policy["deductible"]))
            decision = "approve" if payment > 0 else "deny"
            rationale = "Active coverage and reviewed estimate support payment after deductible, limits, and nonpayable line review."
        indicators = fraud.get("indicators", []) if fraud else []
        siu_referral = bool(indicators) or self.force_siu_referral
        message = (
            f"Your claim decision is {decision}. "
            f"The payable amount is ${payment:.2f} after applying the policy deductible where applicable. "
            "We will contact you with any next step."
        )
        return {
            "tool": "submit_final_decision",
            "args": {
                "decision": decision,
                "payment_amount": payment,
                "reserve_amount": observation.financial_snapshot.reserve_amount or max(payment, observation.requested_amount),
                "siu_referral": siu_referral,
                "subrogation": _subrogation_visible(observation),
                "claimant_message": message,
                "evidence_cited": _evidence_ids(observation),
                "rationale": rationale,
                "closure_disposition": "paid_closed" if decision == "approve" else "denied_closed",
            },
        }


class MissingEvidencePolicy(CarefulAdjusterPolicy):
    request_documents = False
    inspect_documents = False


class SiuEverythingPolicy(CarefulAdjusterPolicy):
    force_siu_referral = True


class PrematureFinalPolicy(ActionPolicy):
    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        return {
            "tool": "submit_final_decision",
            "args": {
                "decision": "approve",
                "payment_amount": observation.requested_amount,
                "reserve_amount": observation.requested_amount,
                "siu_referral": False,
                "subrogation": False,
                "claimant_message": "Your claim decision is approved. We will contact you with the next step.",
                "evidence_cited": ["EV-STATEMENT"],
                "rationale": "Premature approval without completed investigation.",
                "closure_disposition": "paid_closed",
            },
        }


class OverpayPolicy(ActionPolicy):
    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        if len(context.history) == 0:
            return {"tool": "get_policy", "args": {"policy_id": observation.policy_id}}
        if len(context.history) == 1:
            return {
                "tool": "verify_coverage",
                "args": {
                    "claim_id": observation.claim_id,
                    "exposure_id": observation.exposures[0].exposure_id,
                    "loss_facts": observation.claimant_statement,
                },
            }
        if len(context.history) == 2:
            return {"tool": "inspect_repair_estimate", "args": {"estimate_id": observation.estimate_id}}
        return {
            "tool": "submit_final_decision",
            "args": {
                "decision": "approve",
                "payment_amount": observation.requested_amount + 1500.0,
                "reserve_amount": observation.requested_amount + 1500.0,
                "siu_referral": False,
                "subrogation": False,
                "claimant_message": "Your claim decision is approved. The deductible was reviewed. We will contact you with the next step.",
                "evidence_cited": _evidence_ids(observation),
                "rationale": "Overpaying relative to reviewed payable exposure.",
                "closure_disposition": "paid_closed",
            },
        }


class AuthorityBypassPolicy(ActionPolicy):
    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        if len(context.history) == 0:
            return {"tool": "get_policy", "args": {"policy_id": observation.policy_id}}
        if len(context.history) == 1:
            return {
                "tool": "verify_coverage",
                "args": {
                    "claim_id": observation.claim_id,
                    "exposure_id": observation.exposures[0].exposure_id,
                    "loss_facts": observation.claimant_statement,
                },
            }
        return {
            "tool": "issue_payment",
            "args": {
                "exposure_id": observation.exposures[0].exposure_id,
                "payee_id": observation.customer_id,
                "amount": observation.requested_amount,
                "rationale": "Attempting payment without completing authority and workflow controls.",
            },
        }


def _row_from_rollout(
    family: str,
    seed: int,
    behavior: BehaviorSpec,
    rollout: RolloutResult,
    include_rollout: bool,
) -> CalibrationRow:
    final_info = rollout.trajectory[-1].info if rollout.trajectory else {}
    rubric = final_info.get("rubric_evaluation") or {}
    reward_breakdown = rollout.reward_breakdown
    violations = list(final_info.get("violations") or [])
    missed_must = list(rubric.get("missed_must") or [])
    missed_final = list(rubric.get("missed_final") or [])
    violated_forbidden = list(rubric.get("violated_forbidden") or [])
    verdict, expected_passed, notes = _assess_rollout_quality(
        expected_quality=behavior.quality_for(family),
        total_reward=rollout.total_reward,
        rubric_overall=float(rubric.get("overall_score", 0.0)),
        safety_cap=float(reward_breakdown.get("safety_cap", 1.0)),
        penalties=float(reward_breakdown.get("penalties", 0.0)),
        missed_must=missed_must,
        missed_final=missed_final,
        violated_forbidden=violated_forbidden,
        violations=violations,
        done=bool(rollout.trajectory[-1].done if rollout.trajectory else False),
    )
    return CalibrationRow(
        scenario_family=family,
        seed=seed,
        behavior=behavior.name,
        expected_quality=behavior.quality_for(family),
        description=behavior.description,
        total_reward=rollout.total_reward,
        reward_breakdown=reward_breakdown,
        rubric_overall=float(rubric.get("overall_score", 0.0)),
        missed_must=missed_must,
        missed_final=missed_final,
        violated_forbidden=violated_forbidden,
        violations=violations,
        safety_cap=float(reward_breakdown.get("safety_cap", 1.0)),
        penalties=float(reward_breakdown.get("penalties", 0.0)),
        verdict=verdict,
        expected_passed=expected_passed,
        notes=notes,
        steps=rollout.steps,
        done=bool(rollout.trajectory[-1].done if rollout.trajectory else False),
        rollout=rollout.model_dump(mode="json") if include_rollout else None,
    )


def _ordering_failures(rows: list[CalibrationRow], behaviors: list[BehaviorSpec], margin: float) -> list[OrderingFailure]:
    rank_by_behavior = {behavior.name: behavior.rank for behavior in behaviors}
    failures: list[OrderingFailure] = []
    grouped: dict[tuple[str, int], list[CalibrationRow]] = {}
    for row in rows:
        grouped.setdefault((row.scenario_family, row.seed), []).append(row)
    for (family, seed), family_rows in grouped.items():
        for better in family_rows:
            for worse in family_rows:
                if better.expected_quality != "good" or worse.expected_quality != "bad":
                    continue
                if rank_by_behavior[better.behavior] >= rank_by_behavior[worse.behavior]:
                    continue
                if better.total_reward <= worse.total_reward + margin:
                    failures.append(
                        OrderingFailure(
                            scenario_family=family,
                            seed=seed,
                            better_behavior=better.behavior,
                            worse_behavior=worse.behavior,
                            better_reward=better.total_reward,
                            worse_reward=worse.total_reward,
                            margin=margin,
                        )
                    )
    return failures


def _assess_rollout_quality(
    *,
    expected_quality: str,
    total_reward: float,
    rubric_overall: float,
    safety_cap: float,
    penalties: float,
    missed_must: list[str],
    missed_final: list[str],
    violated_forbidden: list[str],
    violations: list[str],
    done: bool,
) -> tuple[str, bool, list[str]]:
    notes: list[str] = []
    miss_count = len(missed_must) + len(missed_final) + len(violated_forbidden)
    if not done:
        notes.append("did not reach terminal decision")
    if safety_cap < 1.0:
        notes.append(f"safety cap {safety_cap:.2f}")
    if penalties:
        notes.append(f"penalties {penalties:.2f}")
    if miss_count:
        notes.append(f"{miss_count} rubric misses")
    if violations:
        notes.append(f"violations: {', '.join(violations[:3])}")

    if total_reward >= 0.82 and rubric_overall >= 0.85 and safety_cap >= 0.95 and not missed_must and not violated_forbidden:
        verdict = "good"
    elif total_reward >= 0.45 and rubric_overall >= 0.45 and safety_cap >= 0.45:
        verdict = "mixed"
    else:
        verdict = "bad"

    if expected_quality == "good":
        expected_passed = verdict == "good"
    elif expected_quality == "bad":
        expected_passed = verdict != "good"
    elif expected_quality == "neutral":
        expected_passed = True
    else:
        expected_passed = True
    return verdict, expected_passed, notes


def _estimate_review_action(observation: Observation) -> str:
    if "estimate_duplicate_line" in observation.alerts:
        return "request_supplement"
    if "possible_prior_damage" in observation.alerts:
        return "escalate_field"
    if "near_total_loss_threshold" in observation.alerts:
        return "confirm_total_loss"
    if any("inception" in flag.lower() for evidence in observation.available_evidence for flag in evidence.flags):
        return "request_photos"
    return "approve"


def _received_unreviewed_document(observation: Observation) -> dict[str, Any] | None:
    for document in observation.model_dump(mode="json").get("claim_documents", []):
        if document.get("status") == "received" and document.get("evidence_id"):
            if document.get("document_id") in {"DOC-FNOL-STATEMENT", f"DOC-{observation.estimate_id}"}:
                continue
            return document
    return None


def _waiting_note_action(observation: Observation) -> dict[str, Any]:
    pending = ", ".join(event.event_type for event in observation.pending_events) or "external events"
    return {
        "tool": "add_claim_note",
        "args": {
            "claim_id": observation.claim_id,
            "note_type": "estimate",
            "subject": "Pending external event",
            "body": f"Awaiting {pending} before final disposition.",
            "related_object_id": observation.claim_id,
        },
    }


def _should_query_prior_claims(observation: Observation, context: AgentContext) -> bool:
    if any(isinstance(step.action, dict) and step.action.get("tool") == "query_prior_claims" for step in context.history):
        return False
    flags = [flag for evidence in observation.available_evidence for flag in evidence.flags]
    return any("prior" in flag or "unrelated" in flag for flag in flags)


def _should_open_siu(observation: Observation, context: AgentContext, *, force: bool) -> bool:
    if any(isinstance(step.action, dict) and step.action.get("tool") in {"open_siu_referral", "refer_to_siu"} for step in context.history):
        return False
    if not force and not any("requires_siu_review" in flag for evidence in observation.available_evidence for flag in evidence.flags):
        return False
    return bool(_evidence_ids(observation))


def _subrogation_visible(observation: Observation) -> bool:
    return any("third_party" in flag for evidence in observation.available_evidence for flag in evidence.flags)


def _evidence_ids(observation: Observation) -> list[str]:
    return [evidence.evidence_id for evidence in observation.available_evidence[:4]]


def _latest_success_data(context: AgentContext, tool_name: str) -> dict[str, Any]:
    for step in reversed(context.history):
        action = step.action
        if not isinstance(action, dict) or action.get("tool") != tool_name:
            continue
        if not step.info.get("action_valid"):
            continue
        latest = step.next_observation.get("latest_tool_result") or {}
        return latest.get("data") or {}
    return {}
