from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from claimsops_env.generator import EpisodeSpec, ScenarioGenerator
from claimsops_env.models import (
    ActionRecord,
    DocumentType,
    FinancialSnapshot,
    Observation,
    RewardBreakdown,
    StepResult,
    ToolName,
    WorkflowState,
)
from claimsops_env.tools import RuntimeView, ToolError, build_tool_registry, validate_action
from claimsops_env.verifier import RewardContext, score_episode


class ClaimsOpsEnv:
    """Stateful, tool-based synthetic auto claims environment."""

    def __init__(self, generator: ScenarioGenerator | None = None) -> None:
        self.generator = generator or ScenarioGenerator()
        self.tools = build_tool_registry()
        self._spec: EpisodeSpec | None = None
        self._visible_policy: dict[str, Any] | None = None
        self._evidence = []
        self._diary: list[str] = []
        self._communications: list[str] = []
        self._financial = FinancialSnapshot()
        self._workflow = WorkflowState()
        self._latest_tool_result: dict[str, Any] | None = None
        self._final_decision = None
        self._remaining_steps = 0
        self._action_log: list[ActionRecord] = []
        self._violations: list[str] = []
        self._invalid_action_streak = 0
        self._last_reward = RewardBreakdown()

    def reset(self, seed: int | None = None, scenario_family: str | None = None) -> Observation:
        self._spec = (
            self.generator.generate_family(scenario_family, seed)
            if scenario_family
            else self.generator.generate(seed)
        )
        self._visible_policy = None
        self._evidence = deepcopy(self._spec.claim.initial_evidence)
        self._diary = ["Claim file opened."]
        self._communications = []
        self._financial = FinancialSnapshot()
        self._workflow = WorkflowState()
        self._latest_tool_result = None
        self._final_decision = None
        self._remaining_steps = self._spec.claim.step_budget
        self._action_log = []
        self._violations = []
        self._invalid_action_streak = 0
        self._last_reward = RewardBreakdown()
        return self._observation()

    def step(self, raw_action: dict[str, Any] | str) -> StepResult:
        self._require_reset()
        if self._workflow.final_decision_submitted:
            return StepResult(
                observation=self._observation(),
                reward=self._last_reward.total,
                done=True,
                info={"error": "episode already done", "reward_breakdown": self._last_reward.model_dump()},
            )

        valid_format = True
        result_summary = ""
        terminal = False
        parsed_dict: dict[str, Any] | None = None
        try:
            parsed_dict = json.loads(raw_action) if isinstance(raw_action, str) else raw_action
            action = validate_action(parsed_dict)
            if _contains_hidden_probe(action.args):
                self._violations.append("hidden_state_access")
                raise ToolError("attempted hidden-state access")
            runtime = self._runtime()
            tool = self.tools[action.tool]
            tool_result = tool.run(runtime, action.args)
            self._apply_runtime(runtime)
            self._latest_tool_result = tool_result.model_dump(mode="json")
            result_summary = tool_result.summary
            terminal = tool_result.terminal
            self._invalid_action_streak = 0
        except (json.JSONDecodeError, ValidationError, ToolError, KeyError, TypeError, ValueError) as exc:
            valid_format = False
            self._invalid_action_streak += 1
            if "unknown evidence_ids" in str(exc):
                self._violations.append("fabricated_document_id")
            self._violations.append("tool_error")
            result_summary = f"Invalid action: {exc}"
            self._latest_tool_result = {"ok": False, "summary": result_summary, "data": {}}

        if self._invalid_action_streak >= 3:
            self._violations.append("invalid_action_loop")
            terminal = True

        self._remaining_steps = max(0, self._remaining_steps - 1)
        if self._remaining_steps == 0:
            terminal = True

        self._action_log.append(
            ActionRecord(
                step=len(self._action_log) + 1,
                action=parsed_dict if isinstance(parsed_dict, dict) else {"raw": raw_action},
                valid=valid_format,
                result_summary=result_summary,
            )
        )
        self._last_reward = self._score(valid_format=valid_format)
        if not valid_format:
            self._last_reward.total = min(self._last_reward.total, -0.2)
        done = terminal or self._workflow.final_decision_submitted
        return StepResult(
            observation=self._observation(),
            reward=self._last_reward.total,
            done=done,
            info={
                "reward_breakdown": self._last_reward.model_dump(),
                "action_valid": valid_format,
                "violations": list(dict.fromkeys(self._violations)),
            },
        )

    def state(self, include_action_log: bool = True) -> dict[str, Any]:
        self._require_reset()
        state = self._observation().model_dump(mode="json")
        state["workflow"] = self._workflow.model_dump(mode="json")
        state["last_reward"] = self._last_reward.model_dump()
        if include_action_log:
            state["action_log"] = [record.model_dump(mode="json") for record in self._action_log]
        return state

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "claimsops-gym",
            "version": "0.1.0",
            "action_format": {"tool": "tool_name", "args": {}},
            "available_tools": [tool.value for tool in ToolName],
            "reward_columns": list(RewardBreakdown.model_fields.keys()),
        }

    def _runtime(self) -> RuntimeView:
        self._require_reset()
        return RuntimeView(
            spec=self._spec,
            visible_policy=deepcopy(self._visible_policy),
            evidence=deepcopy(self._evidence),
            diary=deepcopy(self._diary),
            communications=deepcopy(self._communications),
            financial_snapshot=deepcopy(self._financial),
            workflow=deepcopy(self._workflow),
            final_decision=deepcopy(self._final_decision),
            violations=deepcopy(self._violations),
        )

    def _apply_runtime(self, runtime: RuntimeView) -> None:
        self._visible_policy = runtime.visible_policy
        self._evidence = runtime.evidence
        self._diary = runtime.diary
        self._communications = runtime.communications
        self._financial = runtime.financial_snapshot
        self._workflow = runtime.workflow
        self._final_decision = runtime.final_decision
        self._violations = runtime.violations

    def _observation(self) -> Observation:
        self._require_reset()
        claim = self._spec.claim
        return Observation(
            claim_id=claim.claim_id,
            policy_id=claim.policy_id,
            customer_id=claim.customer_id,
            vehicle_id=claim.vehicle_id,
            estimate_id=claim.estimate_id,
            line_of_business=claim.line_of_business,
            claim_type=claim.claim_type,
            loss_date=claim.loss_date,
            reported_date=claim.reported_date,
            claimant_statement=claim.claimant_statement,
            requested_amount=claim.requested_amount,
            latest_tool_result=self._latest_tool_result,
            visible_policy=self._visible_policy,
            available_evidence=self._evidence,
            claim_diary=self._diary,
            financial_snapshot=self._financial,
            communications_sent=self._communications,
            open_tasks=self._open_tasks(),
            available_tools=list(ToolName),
            remaining_steps=self._remaining_steps,
        )

    def _open_tasks(self) -> list[str]:
        self._require_reset()
        tasks: list[str] = []
        if not self._workflow.policy_seen:
            tasks.append("verify_policy")
        if not self._workflow.policy_status_checked:
            tasks.append("check_policy_status")
        if not self._workflow.estimate_seen:
            tasks.append("inspect_estimate")
        visible_doc_gaps = self._visible_document_gaps()
        received = {_doc_value(doc) for doc in self._workflow.documents_received}
        tasks.extend(f"request_{doc}" for doc in sorted(visible_doc_gaps - received))
        if not self._workflow.fraud_checked:
            tasks.append("screen_fraud_indicators")
        if self._visible_subrogation_signal() and not self._workflow.subrogation_opened:
            tasks.append("evaluate_subrogation")
        if self._financial.reserve_amount is None:
            tasks.append("set_reserve")
        if not self._workflow.final_decision_submitted:
            tasks.append("submit_final_decision")
        return tasks

    def _visible_document_gaps(self) -> set[str]:
        self._require_reset()
        text = " ".join(
            [
                self._spec.claim.claimant_statement,
                *(evidence.summary for evidence in self._evidence),
                *(flag for evidence in self._evidence for flag in evidence.flags),
            ]
        ).lower()
        gaps: set[str] = set()
        if "police" in text and any(term in text for term in ["pending", "report", "third party", "fault"]):
            gaps.add(DocumentType.POLICE_REPORT.value)
        if "breakdown" in text or "duplicate" in text:
            gaps.add(DocumentType.REPAIR_ESTIMATE_BREAKDOWN.value)
        if "ownership" in text or "owner" in text:
            gaps.add(DocumentType.PROOF_OF_OWNERSHIP.value)
        if "incomplete" in text or "statement conflict" in text:
            gaps.add(DocumentType.CLAIMANT_STATEMENT.value)
        return gaps

    def _visible_subrogation_signal(self) -> bool:
        self._require_reset()
        text = " ".join(
            [
                self._spec.claim.claimant_statement,
                *(evidence.summary for evidence in self._evidence),
                *(flag for evidence in self._evidence for flag in evidence.flags),
            ]
        ).lower()
        return any(
            signal in text
            for signal in [
                "third_party_fault",
                "third party",
                "third-party",
                "another driver",
                "rear-ended",
                "admitted fault",
                "likely at fault",
            ]
        )

    def _score(self, valid_format: bool) -> RewardBreakdown:
        self._require_reset()
        context = RewardContext(
            hidden=self._spec.hidden,
            workflow=self._workflow,
            final_decision=self._final_decision,
            approved_payment=self._financial.approved_payment,
            reserve_amount=self._financial.reserve_amount,
            evidence_ids={evidence.evidence_id for evidence in self._evidence},
            requested_documents=list(self._workflow.documents_requested),
            remaining_steps=self._remaining_steps,
            step_budget=self._spec.claim.step_budget,
            violations=list(dict.fromkeys(self._violations)),
            valid_format=valid_format,
        )
        return score_episode(context)

    def _require_reset(self) -> None:
        if self._spec is None:
            raise RuntimeError("call reset() before using the environment")


def _contains_hidden_probe(args: dict[str, Any]) -> bool:
    text = json.dumps(args, sort_keys=True).lower()
    probes = ["hidden", "truth", "expected_outcome", "expected_payable", "verifier", "answer_key"]
    return any(probe in text for probe in probes)


def _doc_value(doc: DocumentType | str) -> str:
    return doc.value if isinstance(doc, DocumentType) else str(doc)
