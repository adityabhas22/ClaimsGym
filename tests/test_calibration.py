from claimsops_env.calibration import default_behaviors, run_calibration


def _behaviors(*names: str):
    wanted = set(names)
    return [behavior for behavior in default_behaviors() if behavior.name in wanted]


def test_calibration_marks_careful_workflow_good() -> None:
    report = run_calibration(
        families=["covered_collision"],
        seeds=[0],
        behaviors=_behaviors("careful_adjuster", "premature_final"),
    )

    rows = {row.behavior: row for row in report.rows}
    assert report.passed is True
    assert rows["careful_adjuster"].verdict == "good"
    assert rows["careful_adjuster"].expected_passed is True
    assert rows["careful_adjuster"].total_reward > rows["premature_final"].total_reward
    assert rows["premature_final"].verdict != "good"
    assert rows["premature_final"].expected_passed is True


def test_calibration_careful_workflow_handles_material_document_gap() -> None:
    report = run_calibration(
        families=["missing_police_report"],
        seeds=[0],
        behaviors=_behaviors("careful_adjuster", "missing_evidence"),
    )

    rows = {row.behavior: row for row in report.rows}
    assert rows["careful_adjuster"].verdict == "good"
    assert rows["careful_adjuster"].missed_must == []
    assert rows["careful_adjuster"].violated_forbidden == []
    assert rows["missing_evidence"].expected_quality == "bad"
    assert rows["careful_adjuster"].total_reward > rows["missing_evidence"].total_reward


def test_calibration_report_surfaces_component_diagnostics() -> None:
    report = run_calibration(
        families=["duplicate_line_item"],
        seeds=[0],
        behaviors=_behaviors("careful_adjuster", "overpay"),
    )
    overpay = next(row for row in report.rows if row.behavior == "overpay")
    rendered = report.to_markdown()

    assert overpay.safety_cap < 1.0
    assert overpay.reward_breakdown["leakage_control"] < 1.0
    assert overpay.notes
    assert "expected_passed" in rendered
    assert "overpay" in rendered


def test_calibration_catches_unsupported_siu_over_referral() -> None:
    report = run_calibration(
        families=["covered_collision"],
        seeds=[0],
        behaviors=_behaviors("careful_adjuster", "siu_everything"),
    )

    rows = {row.behavior: row for row in report.rows}
    assert rows["careful_adjuster"].verdict == "good"
    assert rows["siu_everything"].expected_quality == "bad"
    assert rows["siu_everything"].verdict != "good"
    assert rows["careful_adjuster"].total_reward > rows["siu_everything"].total_reward


def test_calibration_can_attach_rollout_payload_for_debugging() -> None:
    report = run_calibration(
        families=["covered_collision"],
        seeds=[0],
        behaviors=_behaviors("careful_adjuster"),
        include_rollouts=True,
    )

    row = report.rows[0]
    assert row.rollout is not None
    assert row.rollout["trajectory"]
    assert row.rollout["reward_breakdown"] == row.reward_breakdown
