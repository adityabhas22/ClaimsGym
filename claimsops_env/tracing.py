from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from claimsops_env.agent_interface import RolloutResult


class RewardDelta(BaseModel):
    component: str
    before: float
    after: float
    delta: float


class StateChange(BaseModel):
    path: str
    change_type: str
    before: Any = None
    after: Any = None
    summary: str


class TraceStep(BaseModel):
    step: int
    action: dict[str, Any] | str
    tool_summary: str
    reward: float
    reward_deltas: list[RewardDelta] = Field(default_factory=list)
    state_changes: list[StateChange] = Field(default_factory=list)
    rubric_evaluation: dict[str, Any] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    done: bool = False


class EpisodeTrace(BaseModel):
    seed: int | None = None
    scenario_family: str | None = None
    claim_id: str
    initial_summary: dict[str, Any]
    steps: list[TraceStep]
    final_reward_breakdown: dict[str, Any]
    final_observation: dict[str, Any]

    def to_markdown(self) -> str:
        lines = [
            "# ClaimsOps Trace",
            "",
            f"- claim_id: `{self.claim_id}`",
            f"- scenario_family: `{self.scenario_family}`",
            f"- seed: `{self.seed}`",
            f"- final_total_reward: `{_fmt_float(self.final_reward_breakdown.get('total', 0.0))}`",
            "",
            "## Initial File",
        ]
        for key, value in self.initial_summary.items():
            lines.append(f"- {key}: `{_compact(value)}`")
        for step in self.steps:
            tool = _action_tool(step.action)
            lines.extend(["", f"## Step {step.step}: `{tool}`", "", "**Action**", ""])
            lines.extend(["```json", json.dumps(step.action, indent=2, sort_keys=True), "```"])
            lines.extend(["", f"**Tool Result:** {step.tool_summary}", ""])
            if step.state_changes:
                lines.append("**State Changes**")
                for change in step.state_changes:
                    lines.append(f"- {change.summary}")
            else:
                lines.extend(["**State Changes**", "- no material visible state changes"])
            lines.extend(["", "**Reward Deltas**"])
            if step.reward_deltas:
                for delta in step.reward_deltas:
                    sign = "+" if delta.delta >= 0 else ""
                    lines.append(
                        f"- {delta.component}: {_fmt_float(delta.before)} -> "
                        f"{_fmt_float(delta.after)} ({sign}{_fmt_float(delta.delta)})"
                    )
            else:
                lines.append("- no reward component changes")
            if step.violations:
                lines.extend(["", "**Violations**"])
                for violation in step.violations:
                    lines.append(f"- {violation}")
            rubric_misses = _rubric_misses(step.rubric_evaluation, include_final=step.done)
            if rubric_misses:
                lines.extend(["", "**Rubric Misses**"])
                for miss in rubric_misses:
                    lines.append(f"- {miss}")
            lines.append(f"\nStep reward: `{_fmt_float(step.reward)}`")
        lines.extend(["", "## Final Reward"])
        for key, value in self.final_reward_breakdown.items():
            if isinstance(value, int | float):
                lines.append(f"- {key}: `{_fmt_float(value)}`")
        return "\n".join(lines) + "\n"


def trace_rollout(result: RolloutResult | dict[str, Any]) -> EpisodeTrace:
    rollout = result if isinstance(result, RolloutResult) else RolloutResult.model_validate(result)
    if not rollout.trajectory:
        raise ValueError("cannot trace an empty rollout")

    builder = SnapshotBuilder()
    initial_observation = rollout.trajectory[0].observation
    before_snapshot = builder.snapshot(initial_observation)
    previous_rewards: dict[str, float] = {}
    steps: list[TraceStep] = []

    for rollout_step in rollout.trajectory:
        after_snapshot = builder.snapshot(rollout_step.next_observation)
        reward_breakdown = rollout_step.info.get("reward_breakdown", {})
        reward_deltas = _reward_deltas(previous_rewards, reward_breakdown)
        state_changes = diff_snapshots(before_snapshot, after_snapshot)
        latest = rollout_step.next_observation.get("latest_tool_result") or {}
        rubric_evaluation = rollout_step.info.get("rubric_evaluation") or {}
        steps.append(
            TraceStep(
                step=rollout_step.step,
                action=rollout_step.action,
                tool_summary=str(latest.get("summary") or rollout_step.info.get("error") or ""),
                reward=rollout_step.reward,
                reward_deltas=reward_deltas,
                state_changes=state_changes,
                rubric_evaluation=rubric_evaluation,
                violations=list(rollout_step.info.get("violations") or []),
                done=rollout_step.done,
            )
        )
        previous_rewards = {
            key: float(value)
            for key, value in reward_breakdown.items()
            if isinstance(value, int | float)
        }
        before_snapshot = after_snapshot

    final_observation = rollout.final_observation
    return EpisodeTrace(
        seed=rollout.seed,
        scenario_family=rollout.scenario_family,
        claim_id=str(final_observation.get("claim_id") or initial_observation.get("claim_id")),
        initial_summary=_initial_summary(initial_observation),
        steps=steps,
        final_reward_breakdown=rollout.reward_breakdown,
        final_observation=final_observation,
    )


def trace_json(result: RolloutResult | dict[str, Any]) -> str:
    return trace_rollout(result).model_dump_json(indent=2)


@dataclass
class SnapshotBuilder:
    """Builds visible snapshots while carrying forward last-seen estimate lines."""

    estimate_line_items: dict[str, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self, observation: dict[str, Any]) -> dict[str, Any]:
        latest_data = (observation.get("latest_tool_result") or {}).get("data") or {}
        line_items = latest_data.get("line_items")
        if isinstance(line_items, list):
            self.estimate_line_items = {
                str(line["line_id"]): line
                for line in line_items
                if isinstance(line, dict) and line.get("line_id")
            }
        return {
            "open_tasks": sorted(observation.get("open_tasks") or []),
            "alerts": sorted(observation.get("alerts") or []),
            "audit_gaps": sorted(observation.get("audit_gaps") or []),
            "financial_snapshot": observation.get("financial_snapshot") or {},
            "claim_documents": _by_key(observation.get("claim_documents") or [], "document_id"),
            "pending_events": _by_key(observation.get("pending_events") or [], "event_id"),
            "event_history": _by_key(observation.get("event_history") or [], "event_id"),
            "activities": _by_key(observation.get("activities") or [], "activity_id"),
            "reserve_lines": _by_key(observation.get("reserve_lines") or [], "reserve_id"),
            "payments": _by_key(observation.get("payments") or [], "payment_id"),
            "vendor_assignments": _by_key(observation.get("vendor_assignments") or [], "vendor_id"),
            "claim_notes": _by_key(observation.get("claim_notes") or [], "note_id"),
            "available_evidence": _by_key(observation.get("available_evidence") or [], "evidence_id"),
            "estimate_line_items": dict(self.estimate_line_items),
        }


def diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> list[StateChange]:
    changes: list[StateChange] = []
    for path in ["open_tasks", "alerts", "audit_gaps"]:
        changes.extend(_list_changes(path, before.get(path, []), after.get(path, [])))
    changes.extend(_dict_field_changes("financial_snapshot", before.get("financial_snapshot", {}), after.get("financial_snapshot", {})))
    for path in [
        "claim_documents",
        "pending_events",
        "event_history",
        "activities",
        "reserve_lines",
        "payments",
        "vendor_assignments",
        "claim_notes",
        "available_evidence",
        "estimate_line_items",
    ]:
        changes.extend(_collection_changes(path, before.get(path, {}), after.get(path, {})))
    return changes


def _initial_summary(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_type": observation.get("claim_type"),
        "loss_date": observation.get("loss_date"),
        "reported_date": observation.get("reported_date"),
        "requested_amount": observation.get("requested_amount"),
        "open_tasks": observation.get("open_tasks") or [],
        "alerts": observation.get("alerts") or [],
        "documents": [
            {
                "document_id": doc.get("document_id"),
                "doc_type": doc.get("doc_type"),
                "status": doc.get("status"),
                "issues": doc.get("issues") or [],
            }
            for doc in observation.get("claim_documents") or []
        ],
    }


def _reward_deltas(before: dict[str, float], after: dict[str, Any]) -> list[RewardDelta]:
    keys = sorted({*before.keys(), *after.keys()})
    deltas: list[RewardDelta] = []
    for key in keys:
        after_value = after.get(key, 0.0)
        if not isinstance(after_value, int | float):
            continue
        before_value = float(before.get(key, 0.0))
        current = float(after_value)
        delta = current - before_value
        if abs(delta) > 1e-9:
            deltas.append(
                RewardDelta(
                    component=key,
                    before=before_value,
                    after=current,
                    delta=delta,
                )
            )
    return deltas


def _collection_changes(path: str, before: dict[str, Any], after: dict[str, Any]) -> list[StateChange]:
    changes: list[StateChange] = []
    for item_id in sorted(set(after) - set(before)):
        changes.append(
            StateChange(
                path=f"{path}.{item_id}",
                change_type="added",
                after=after[item_id],
                summary=f"{path}[{item_id}] added: {_item_label(after[item_id])}",
            )
        )
    for item_id in sorted(set(before) - set(after)):
        changes.append(
            StateChange(
                path=f"{path}.{item_id}",
                change_type="removed",
                before=before[item_id],
                summary=f"{path}[{item_id}] removed: {_item_label(before[item_id])}",
            )
        )
    for item_id in sorted(set(before) & set(after)):
        changes.extend(_dict_field_changes(f"{path}.{item_id}", before[item_id], after[item_id]))
    return changes


def _dict_field_changes(path: str, before: dict[str, Any], after: dict[str, Any]) -> list[StateChange]:
    changes: list[StateChange] = []
    keys = sorted(set(before) | set(after))
    for key in keys:
        old = before.get(key)
        new = after.get(key)
        if old == new:
            continue
        changes.append(
            StateChange(
                path=f"{path}.{key}",
                change_type="changed",
                before=old,
                after=new,
                summary=f"{path}.{key}: {_compact(old)} -> {_compact(new)}",
            )
        )
    return changes


def _list_changes(path: str, before: list[Any], after: list[Any]) -> list[StateChange]:
    changes: list[StateChange] = []
    before_set = {_stable_json(item) for item in before}
    after_set = {_stable_json(item) for item in after}
    for raw in sorted(after_set - before_set):
        item = json.loads(raw)
        changes.append(
            StateChange(
                path=path,
                change_type="added",
                after=item,
                summary=f"{path} added `{_compact(item)}`",
            )
        )
    for raw in sorted(before_set - after_set):
        item = json.loads(raw)
        changes.append(
            StateChange(
                path=path,
                change_type="removed",
                before=item,
                summary=f"{path} removed `{_compact(item)}`",
            )
        )
    return changes


def _by_key(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {
        str(item[key]): item
        for item in items
        if isinstance(item, dict) and item.get(key) is not None
    }


def _item_label(item: dict[str, Any]) -> str:
    for key in ["summary", "title", "subject", "description", "event_type", "status"]:
        if item.get(key):
            return _compact(item[key])
    return _compact(item)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _compact(value: Any, max_len: int = 140) -> str:
    if isinstance(value, float):
        return _fmt_float(value)
    text = json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _action_tool(action: dict[str, Any] | str) -> str:
    if isinstance(action, dict):
        return str(action.get("tool", "<unknown>"))
    return "<raw-text-action>"


def _rubric_misses(evaluation: dict[str, Any], *, include_final: bool) -> list[str]:
    checks = evaluation.get("checks") or []
    misses = []
    for check in checks:
        if check.get("passed"):
            continue
        severity = check.get("severity")
        if severity == "final" and not include_final:
            continue
        if severity not in {"must", "final", "forbidden"}:
            continue
        key = check.get("key")
        detail = check.get("detail")
        misses.append(f"{severity}:{key} - {check.get('description')} ({detail})")
    return misses
