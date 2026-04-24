from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from claimsops_env.agent_interface import ActionPolicy, RolloutResult, RolloutRunner
from claimsops_env.suites import ScenarioSuite, SuiteEpisode, get_suite


class SuiteRunRow(BaseModel):
    suite: str
    episode_id: str
    policy_name: str
    scenario_family: str
    seed: int
    split: str
    tags: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    success: bool
    verdict: str
    total_reward: float
    reward_breakdown: dict[str, Any]
    rubric_overall: float = 0.0
    missed_must: list[str] = Field(default_factory=list)
    missed_final: list[str] = Field(default_factory=list)
    violated_forbidden: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    safety_cap: float = 1.0
    penalties: float = 0.0
    steps: int
    done: bool
    rollout: dict[str, Any] | None = None


class SuiteFamilySummary(BaseModel):
    scenario_family: str
    episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    reward_breakdown_mean: dict[str, float]


class SuiteRunReport(BaseModel):
    suite: str
    purpose: str
    policy_name: str
    heldout: bool = False
    rows: list[SuiteRunRow]
    reward_breakdown_mean: dict[str, float]
    family_summaries: list[SuiteFamilySummary]

    @property
    def episodes(self) -> int:
        return len(self.rows)

    @property
    def mean_reward(self) -> float:
        return mean(row.total_reward for row in self.rows) if self.rows else 0.0

    @property
    def success_rate(self) -> float:
        return mean(1.0 if row.success else 0.0 for row in self.rows) if self.rows else 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# ClaimsOps Suite: {self.suite}",
            "",
            f"- policy: `{self.policy_name}`",
            f"- purpose: {self.purpose}",
            f"- heldout: `{self.heldout}`",
            f"- episodes: `{self.episodes}`",
            f"- mean_reward: `{self.mean_reward:.3f}`",
            f"- success_rate: `{self.success_rate:.3f}`",
            "",
            "## Reward Columns",
            "",
            "| component | mean |",
            "|---|---:|",
        ]
        for component, value in sorted(self.reward_breakdown_mean.items()):
            lines.append(f"| {component} | {value:.3f} |")

        lines.extend(
            [
                "",
                "## Family Summary",
                "",
                "| family | episodes | success | reward | steps |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for summary in self.family_summaries:
            lines.append(
                f"| {summary.scenario_family} | {summary.episodes} | {summary.success_rate:.3f} | "
                f"{summary.mean_reward:.3f} | {summary.mean_steps:.1f} |"
            )

        lines.extend(
            [
                "",
                "## Episodes",
                "",
                "| episode | family | seed | verdict | reward | cap | penalties | rubric | misses |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in self.rows:
            misses = len(row.missed_must) + len(row.missed_final) + len(row.violated_forbidden)
            lines.append(
                f"| {row.episode_id} | {row.scenario_family} | {row.seed} | {row.verdict} | "
                f"{row.total_reward:.3f} | {row.safety_cap:.3f} | {row.penalties:.3f} | "
                f"{row.rubric_overall:.3f} | {misses} |"
            )
        return "\n".join(lines) + "\n"


def run_suite(
    policy: ActionPolicy,
    *,
    suite: str | ScenarioSuite,
    policy_name: str,
    include_rollouts: bool = False,
) -> SuiteRunReport:
    resolved_suite = get_suite(suite) if isinstance(suite, str) else suite
    runner = RolloutRunner()
    rows = [
        _row_from_rollout(
            suite=resolved_suite,
            episode=episode,
            policy_name=policy_name,
            rollout=runner.run(
                policy,
                seed=episode.seed,
                scenario_family=episode.scenario_family,
                max_steps=episode.max_steps,
            ),
            include_rollout=include_rollouts,
        )
        for episode in resolved_suite.episodes
    ]
    return SuiteRunReport(
        suite=resolved_suite.name,
        purpose=resolved_suite.purpose,
        policy_name=policy_name,
        heldout=resolved_suite.heldout,
        rows=rows,
        reward_breakdown_mean=_mean_reward_breakdown(rows),
        family_summaries=_family_summaries(rows),
    )


def _row_from_rollout(
    *,
    suite: ScenarioSuite,
    episode: SuiteEpisode,
    policy_name: str,
    rollout: RolloutResult,
    include_rollout: bool,
) -> SuiteRunRow:
    final_info = rollout.trajectory[-1].info if rollout.trajectory else {}
    rubric = final_info.get("rubric_evaluation") or {}
    reward_breakdown = rollout.reward_breakdown
    missed_must = list(rubric.get("missed_must") or [])
    missed_final = list(rubric.get("missed_final") or [])
    violated_forbidden = list(rubric.get("violated_forbidden") or [])
    violations = list(final_info.get("violations") or [])
    safety_cap = float(reward_breakdown.get("safety_cap", 1.0))
    penalties = float(reward_breakdown.get("penalties", 0.0))
    rubric_overall = float(rubric.get("overall_score", 0.0))
    return SuiteRunRow(
        suite=suite.name,
        episode_id=episode.episode_id,
        policy_name=policy_name,
        scenario_family=episode.scenario_family,
        seed=episode.seed,
        split=episode.split,
        tags=list(episode.tags),
        max_steps=episode.max_steps,
        success=rollout.success,
        verdict=_rollout_verdict(
            total_reward=rollout.total_reward,
            rubric_overall=rubric_overall,
            safety_cap=safety_cap,
            missed_must=missed_must,
            violated_forbidden=violated_forbidden,
        ),
        total_reward=rollout.total_reward,
        reward_breakdown=reward_breakdown,
        rubric_overall=rubric_overall,
        missed_must=missed_must,
        missed_final=missed_final,
        violated_forbidden=violated_forbidden,
        violations=violations,
        safety_cap=safety_cap,
        penalties=penalties,
        steps=rollout.steps,
        done=bool(rollout.trajectory[-1].done if rollout.trajectory else False),
        rollout=rollout.model_dump(mode="json") if include_rollout else None,
    )


def _rollout_verdict(
    *,
    total_reward: float,
    rubric_overall: float,
    safety_cap: float,
    missed_must: list[str],
    violated_forbidden: list[str],
) -> str:
    if total_reward >= 0.82 and rubric_overall >= 0.85 and safety_cap >= 0.95 and not missed_must and not violated_forbidden:
        return "good"
    if total_reward >= 0.45 and rubric_overall >= 0.45 and safety_cap >= 0.45:
        return "mixed"
    return "bad"


def _mean_reward_breakdown(rows: list[SuiteRunRow]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.reward_breakdown.items() if isinstance(value, int | float)})
    return {
        key: mean(float(row.reward_breakdown[key]) for row in rows if isinstance(row.reward_breakdown.get(key), int | float))
        for key in keys
    }


def _family_summaries(rows: list[SuiteRunRow]) -> list[SuiteFamilySummary]:
    families = sorted({row.scenario_family for row in rows})
    summaries: list[SuiteFamilySummary] = []
    for family in families:
        family_rows = [row for row in rows if row.scenario_family == family]
        summaries.append(
            SuiteFamilySummary(
                scenario_family=family,
                episodes=len(family_rows),
                success_rate=mean(1.0 if row.success else 0.0 for row in family_rows),
                mean_reward=mean(row.total_reward for row in family_rows),
                mean_steps=mean(row.steps for row in family_rows),
                reward_breakdown_mean=_mean_reward_breakdown(family_rows),
            )
        )
    return summaries
