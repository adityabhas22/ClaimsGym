from __future__ import annotations

import argparse
from pathlib import Path

from claimsops_env.calibration import default_behaviors, run_calibration
from claimsops_env.scenario_templates import SCENARIO_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reward calibration trajectories across ClaimsOps scenarios.")
    parser.add_argument("--families", default="covered_collision,missing_police_report,duplicate_line_item,authority_threshold")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--behaviors", default="all")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--include-rollouts", action="store_true")
    parser.add_argument("--ordering-margin", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    families = list(SCENARIO_FAMILIES) if args.families == "all" else _split_csv(args.families)
    seeds = [int(seed) for seed in _split_csv(args.seeds)]
    behaviors = default_behaviors()
    if args.behaviors != "all":
        wanted = set(_split_csv(args.behaviors))
        behaviors = [behavior for behavior in behaviors if behavior.name in wanted]
        missing = wanted - {behavior.name for behavior in behaviors}
        if missing:
            raise ValueError(f"unknown behaviors: {sorted(missing)}")

    report = run_calibration(
        families=families,
        seeds=seeds,
        behaviors=behaviors,
        include_rollouts=args.include_rollouts,
        ordering_margin=args.ordering_margin,
    )
    rendered = report.model_dump_json(indent=2) if args.format == "json" else report.to_markdown()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered)
        print(f"wrote calibration report to {args.output}")
        return
    print(rendered)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
