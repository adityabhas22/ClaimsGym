from __future__ import annotations

from dataclasses import dataclass

from claimsops_env.scenario_templates import SCENARIO_FAMILIES


@dataclass(frozen=True)
class SuiteEpisode:
    scenario_family: str
    seed: int
    split: str = "train"
    tags: tuple[str, ...] = ()
    max_steps: int | None = None

    @property
    def episode_id(self) -> str:
        return f"{self.scenario_family}:{self.seed}"


@dataclass(frozen=True)
class ScenarioSuite:
    name: str
    purpose: str
    episodes: tuple[SuiteEpisode, ...]
    heldout: bool = False

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(episode.scenario_family for episode in self.episodes))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(episode.seed for episode in self.episodes))


def get_suite(name: str) -> ScenarioSuite:
    try:
        return SUITES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SUITES))
        raise ValueError(f"unknown suite: {name}. available suites: {available}") from exc


def list_suites() -> tuple[ScenarioSuite, ...]:
    return tuple(SUITES[name] for name in sorted(SUITES))


def make_episodes(
    *,
    families: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    split: str = "train",
    tags: tuple[str, ...] = (),
    max_steps: int | None = None,
) -> tuple[SuiteEpisode, ...]:
    _validate_families(families)
    return tuple(
        SuiteEpisode(
            scenario_family=family,
            seed=seed,
            split=split,
            tags=tags,
            max_steps=max_steps,
        )
        for family in families
        for seed in seeds
    )


def resolve_suite_episodes(
    *,
    suite_name: str | None = None,
    families: list[str] | tuple[str, ...] | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
    max_steps: int | None = None,
) -> tuple[SuiteEpisode, ...]:
    if suite_name:
        episodes = get_suite(suite_name).episodes
        if max_steps is None:
            return episodes
        return tuple(_with_max_steps(episode, max_steps) for episode in episodes)
    selected_families = tuple(families or SCENARIO_FAMILIES)
    selected_seeds = tuple(seeds or (0,))
    return make_episodes(
        families=selected_families,
        seeds=selected_seeds,
        split="ad_hoc",
        tags=("ad_hoc",),
        max_steps=max_steps,
    )


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _with_max_steps(episode: SuiteEpisode, max_steps: int) -> SuiteEpisode:
    return SuiteEpisode(
        scenario_family=episode.scenario_family,
        seed=episode.seed,
        split=episode.split,
        tags=episode.tags,
        max_steps=max_steps,
    )


def _validate_families(families: list[str] | tuple[str, ...]) -> None:
    unknown = sorted(set(families) - set(SCENARIO_FAMILIES))
    if unknown:
        available = ", ".join(SCENARIO_FAMILIES)
        raise ValueError(f"unknown scenario families: {unknown}. available families: {available}")


LEVEL_1_FAMILIES = (
    "covered_collision",
    "comprehensive_deductible",
    "policy_lapse",
    "limit_exceeded",
)

LEVEL_2_FAMILIES = (
    "missing_police_report",
    "incomplete_statement",
    "ownership_gap",
)

LEVEL_3_FAMILIES = (
    "prior_damage_leakage",
    "duplicate_line_item",
    "rental_storage_leakage",
)

LEVEL_4_FAMILIES = (
    "suspicious_inception",
    "conflicting_statement",
    "excluded_driver",
)

LEVEL_5_FAMILIES = (
    "subrogation_opportunity",
    "authority_threshold",
    "total_loss",
)


SUITES: dict[str, ScenarioSuite] = {
    "smoke": ScenarioSuite(
        name="smoke",
        purpose="Fast health check across straight-through, evidence, leakage, and authority workflows.",
        episodes=(
            SuiteEpisode("covered_collision", 0, tags=("smoke", "coverage")),
            SuiteEpisode("missing_police_report", 0, tags=("smoke", "evidence")),
            SuiteEpisode("duplicate_line_item", 0, tags=("smoke", "leakage")),
            SuiteEpisode("authority_threshold", 0, tags=("smoke", "authority")),
        ),
    ),
    "calibration": ScenarioSuite(
        name="calibration",
        purpose="Reward sanity gate for careful and shortcut workflow probes.",
        episodes=make_episodes(
            families=("covered_collision", "missing_police_report", "duplicate_line_item", "authority_threshold"),
            seeds=(0, 1),
            split="calibration",
            tags=("calibration",),
        ),
    ),
    "curriculum_phase_1": ScenarioSuite(
        name="curriculum_phase_1",
        purpose="Basic coverage, deductible, lapse, and policy-limit handling.",
        episodes=make_episodes(families=LEVEL_1_FAMILIES, seeds=(0, 1, 2), tags=("curriculum", "phase_1")),
    ),
    "curriculum_phase_2": ScenarioSuite(
        name="curriculum_phase_2",
        purpose="Material evidence requests, receipt, review, and no-premature-payment behavior.",
        episodes=make_episodes(families=LEVEL_2_FAMILIES, seeds=(0, 1, 2), tags=("curriculum", "phase_2")),
    ),
    "curriculum_phase_3": ScenarioSuite(
        name="curriculum_phase_3",
        purpose="Leakage control for prior damage, duplicate lines, rental, towing, and storage pressure.",
        episodes=make_episodes(families=LEVEL_3_FAMILIES, seeds=(0, 1, 2), tags=("curriculum", "phase_3")),
    ),
    "curriculum_phase_4": ScenarioSuite(
        name="curriculum_phase_4",
        purpose="SIU, subrogation, authority escalation, exclusions, conflicts, and total-loss workflows.",
        episodes=make_episodes(
            families=LEVEL_4_FAMILIES + LEVEL_5_FAMILIES,
            seeds=(0, 1, 2),
            tags=("curriculum", "phase_4"),
        ),
    ),
    "hard_eval": ScenarioSuite(
        name="hard_eval",
        purpose="Evaluation mix for edge cases that require multi-step workflow control.",
        episodes=make_episodes(
            families=LEVEL_3_FAMILIES + LEVEL_4_FAMILIES + LEVEL_5_FAMILIES,
            seeds=(20, 21),
            split="eval",
            tags=("eval", "hard"),
        ),
    ),
    "heldout": ScenarioSuite(
        name="heldout",
        purpose="Heldout deterministic split reserved for final model evaluation, not training.",
        episodes=make_episodes(
            families=SCENARIO_FAMILIES,
            seeds=(100, 101),
            split="heldout",
            tags=("heldout",),
        ),
        heldout=True,
    ),
    "demo": ScenarioSuite(
        name="demo",
        purpose="Small narrative set for before/after examples.",
        episodes=(
            SuiteEpisode("covered_collision", 0, split="demo", tags=("demo", "simple")),
            SuiteEpisode("policy_lapse", 0, split="demo", tags=("demo", "denial")),
            SuiteEpisode("duplicate_line_item", 0, split="demo", tags=("demo", "hard")),
        ),
    ),
}
