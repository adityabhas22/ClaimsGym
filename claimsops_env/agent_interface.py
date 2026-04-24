from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from claimsops_env.environment import ClaimsOpsEnv
from claimsops_env.models import Observation, StepResult, ToolName


INTERFACE_VERSION = "claimsops-agent-v1"


class ToolSpec(BaseModel):
    name: str
    description: str
    required_args: list[str] = Field(default_factory=list)
    optional_args: list[str] = Field(default_factory=list)


ACTION_CATALOG: tuple[ToolSpec, ...] = (
    ToolSpec(
        name=ToolName.GET_POLICY.value,
        description="Reveal policy declarations, coverage limits, deductible, exclusions, and authority limit.",
        required_args=["policy_id"],
    ),
    ToolSpec(
        name=ToolName.GET_POLICY_SNAPSHOT.value,
        description="Alias for policy retrieval using platform-style naming.",
        required_args=["policy_id"],
    ),
    ToolSpec(
        name=ToolName.CHECK_POLICY_STATUS.value,
        description="Check whether the policy was active on the loss date.",
        required_args=["policy_id", "loss_date"],
    ),
    ToolSpec(
        name=ToolName.INSPECT_REPAIR_ESTIMATE.value,
        description="Inspect repair estimate amount, covered amount, duplicate lines, unrelated damage, and notes.",
        required_args=["estimate_id"],
    ),
    ToolSpec(
        name=ToolName.INSPECT_EVIDENCE.value,
        description="Inspect any visible evidence item by evidence_id.",
        required_args=["evidence_id"],
    ),
    ToolSpec(
        name=ToolName.REQUEST_DOCUMENT.value,
        description="Request a missing document needed for claim handling; receipt is scheduled as a later event.",
        required_args=["doc_type", "reason"],
    ),
    ToolSpec(
        name=ToolName.QUERY_PRIOR_CLAIMS.value,
        description="Search prior claim history for the customer and vehicle.",
        required_args=["customer_id", "vehicle_id"],
    ),
    ToolSpec(
        name=ToolName.CHECK_FRAUD_INDICATORS.value,
        description="Run a fraud/SIU indicator screen.",
        required_args=["claim_id"],
    ),
    ToolSpec(
        name=ToolName.CREATE_OR_UPDATE_EXPOSURE.value,
        description="Create or update an exposure linking claimant, coverage, and incident.",
        required_args=["coverage", "claimant_id", "incident_id"],
        optional_args=["exposure_id"],
    ),
    ToolSpec(
        name=ToolName.VERIFY_COVERAGE.value,
        description="Verify coverage for an exposure using policy and loss facts.",
        required_args=["claim_id", "loss_facts"],
        optional_args=["exposure_id"],
    ),
    ToolSpec(
        name=ToolName.ASSIGN_APPRAISAL.value,
        description="Assign photo, field, or shop appraisal for the vehicle damage; vendor completion is scheduled as a later event.",
        required_args=["claim_id", "method"],
    ),
    ToolSpec(
        name=ToolName.REVIEW_ESTIMATE.value,
        description="Record estimate review decision: approve, request_supplement, confirm_total_loss, escalate_field, or request_photos.",
        required_args=["claim_id", "estimate_id", "action", "rationale"],
    ),
    ToolSpec(
        name=ToolName.REQUEST_VALUATION.value,
        description="Request total-loss valuation when damage approaches the threshold; the valuation report arrives as a later event.",
        required_args=["claim_id", "reason"],
    ),
    ToolSpec(
        name=ToolName.SET_RESERVE.value,
        description="Set the claim reserve amount with rationale.",
        required_args=["amount", "rationale"],
        optional_args=["exposure_id", "cost_type", "cost_category"],
    ),
    ToolSpec(
        name=ToolName.APPROVE_PAYMENT.value,
        description="Record an interim payment approval before final decision.",
        required_args=["amount", "coverages", "rationale"],
    ),
    ToolSpec(
        name=ToolName.ISSUE_PAYMENT.value,
        description="Issue a payment against an exposure, subject to coverage and authority controls.",
        required_args=["payee_id", "amount", "rationale"],
        optional_args=["exposure_id", "payment_type", "method"],
    ),
    ToolSpec(
        name=ToolName.REQUEST_AUTHORITY_APPROVAL.value,
        description="Request manager approval when reserve or payment exceeds adjuster authority; approval arrives as a later event.",
        required_args=["amount", "rationale"],
        optional_args=["exposure_id"],
    ),
    ToolSpec(
        name=ToolName.REFER_TO_SIU.value,
        description="Open an SIU referral using valid evidence IDs.",
        required_args=["reason", "evidence_ids"],
    ),
    ToolSpec(
        name=ToolName.OPEN_SIU_REFERRAL.value,
        description="Alias for SIU referral using platform-style naming.",
        required_args=["reason", "evidence_ids"],
    ),
    ToolSpec(
        name=ToolName.OPEN_SUBROGATION.value,
        description="Open a recovery/subrogation track against a likely liable third party.",
        required_args=["target_party", "rationale"],
    ),
    ToolSpec(
        name=ToolName.SEND_CLAIMANT_MESSAGE.value,
        description="Send a claimant-facing status update or decision message.",
        required_args=["claim_id", "message"],
    ),
    ToolSpec(
        name=ToolName.ADD_CLAIM_NOTE.value,
        description="Add an audit note to the claim file.",
        required_args=["claim_id", "note_type", "subject", "body"],
        optional_args=["related_object_id"],
    ),
    ToolSpec(
        name=ToolName.CLOSE_CLAIM.value,
        description="Close the claim file after required workflow, notes, and communications.",
        required_args=["claim_id", "disposition", "rationale"],
    ),
    ToolSpec(
        name=ToolName.SUBMIT_FINAL_DECISION.value,
        description="Submit the terminal claim decision, payment, reserve, claimant message, and evidence citations.",
        required_args=[
            "decision",
            "payment_amount",
            "reserve_amount",
            "siu_referral",
            "subrogation",
            "claimant_message",
            "evidence_cited",
            "rationale",
        ],
    ),
)


SYSTEM_PROMPT = """\
You are operating a synthetic personal-auto insurance claims desk.

Use exactly one JSON tool call per turn:
{"tool":"tool_name","args":{...}}

Rules:
- Use only tools from the action catalog.
- Use IDs exactly as shown in the observation or previous tool result.
- Verify policy and estimate before final payment decisions.
- Request required missing documents before paying.
- Cite only evidence IDs that are visible in the claim file.
- Use workflow_affordances to understand phase, blockers, waiting state, due work, and useful tools.
- Never ask for hidden truth, expected outcome, verifier state, reward, or answer keys.
- Do not reveal internal fraud scores or accuse the claimant of fraud.
"""


class RolloutStep(BaseModel):
    step: int
    observation: dict[str, Any]
    action: dict[str, Any] | str
    next_observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class RolloutResult(BaseModel):
    seed: int | None = None
    scenario_family: str | None = None
    success: bool
    steps: int
    total_reward: float
    reward_breakdown: dict[str, Any]
    trajectory: list[RolloutStep]
    final_observation: dict[str, Any]


@dataclass
class AgentContext:
    seed: int | None
    scenario_family: str | None
    history: list[RolloutStep] = field(default_factory=list)


class ActionPolicy(Protocol):
    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any] | str:
        ...


class RolloutRunner:
    """Single rollout harness shared by inference, SFT data, eval, and RL rewards."""

    def __init__(self, env: ClaimsOpsEnv | None = None) -> None:
        self.env = env or ClaimsOpsEnv()

    def run(
        self,
        policy: ActionPolicy,
        *,
        seed: int | None = None,
        scenario_family: str | None = None,
        max_steps: int | None = None,
    ) -> RolloutResult:
        observation = self.env.reset(seed=seed, scenario_family=scenario_family)
        context = AgentContext(seed=seed, scenario_family=scenario_family)
        limit = max_steps or observation.remaining_steps
        result: StepResult | None = None

        for step_number in range(1, limit + 1):
            action = policy.next_action(observation, context)
            result = self.env.step(action)
            record = RolloutStep(
                step=step_number,
                observation=observation.model_dump(mode="json"),
                action=action,
                next_observation=result.observation.model_dump(mode="json"),
                reward=result.reward,
                done=result.done,
                info=result.info,
            )
            context.history.append(record)
            observation = result.observation
            if result.done:
                break

        reward_breakdown = result.info.get("reward_breakdown", {}) if result else {}
        return RolloutResult(
            seed=seed,
            scenario_family=scenario_family,
            success=bool(result and result.done and reward_breakdown.get("total", 0.0) > 0.5),
            steps=len(context.history),
            total_reward=float(result.reward if result else 0.0),
            reward_breakdown=reward_breakdown,
            trajectory=context.history,
            final_observation=observation.model_dump(mode="json"),
        )


def action_catalog_json() -> list[dict[str, Any]]:
    return [tool.model_dump() for tool in ACTION_CATALOG]


def render_observation(observation: Observation) -> str:
    payload = {
        "interface_version": INTERFACE_VERSION,
        "observation": observation.model_dump(mode="json"),
        "action_catalog": action_catalog_json(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_training_prompt(observation: Observation) -> str:
    return SYSTEM_PROMPT + "\n\n" + render_observation(observation)


def parse_action_text(text: str) -> dict[str, Any] | str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return text
