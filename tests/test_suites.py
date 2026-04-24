from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.suite_runner import run_suite
from claimsops_env.suites import get_suite, list_suites, resolve_suite_episodes


def test_named_suites_are_deterministic_and_have_known_splits() -> None:
    suites = {suite.name: suite for suite in list_suites()}

    assert "smoke" in suites
    assert "calibration" in suites
    assert "heldout" in suites
    assert suites["heldout"].heldout is True
    assert all(episode.split == "heldout" for episode in suites["heldout"].episodes)
    assert [episode.episode_id for episode in get_suite("smoke").episodes] == [
        "covered_collision:0",
        "missing_police_report:0",
        "duplicate_line_item:0",
        "authority_threshold:0",
    ]


def test_resolve_suite_episodes_can_override_max_steps_without_mutating_suite() -> None:
    original = get_suite("smoke").episodes[0]
    resolved = resolve_suite_episodes(suite_name="smoke", max_steps=3)

    assert resolved[0].max_steps == 3
    assert original.max_steps is None
    assert resolved[0].episode_id == original.episode_id


def test_suite_runner_reports_dynamic_reward_columns() -> None:
    suite = get_suite("smoke")
    small_suite = type(suite)(
        name="unit_smoke",
        purpose=suite.purpose,
        episodes=suite.episodes[:2],
    )

    report = run_suite(ScriptedBaselinePolicy(), suite=small_suite, policy_name="baseline")
    rendered = report.to_markdown()

    assert report.episodes == 2
    assert report.reward_breakdown_mean["coverage"] >= 0.0
    assert {summary.scenario_family for summary in report.family_summaries} == {
        "covered_collision",
        "missing_police_report",
    }
    assert "# ClaimsOps Suite: unit_smoke" in rendered
    assert "expected_payable" not in report.model_dump_json()


def test_suite_runner_can_attach_rollout_payloads() -> None:
    suite = get_suite("demo")
    small_suite = type(suite)(name="unit_demo", purpose=suite.purpose, episodes=suite.episodes[:1])

    report = run_suite(
        ScriptedBaselinePolicy(),
        suite=small_suite,
        policy_name="baseline",
        include_rollouts=True,
    )

    assert report.rows[0].rollout is not None
    assert report.rows[0].rollout["trajectory"]
