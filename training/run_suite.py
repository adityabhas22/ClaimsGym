from __future__ import annotations

import argparse
from pathlib import Path

from claimsops_env.calibration import default_behaviors
from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.suite_runner import run_suite
from claimsops_env.suites import list_suites


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named ClaimsOps scenario suite through the shared rollout harness.")
    parser.add_argument("--suite", default="smoke", choices=[suite.name for suite in list_suites()])
    parser.add_argument("--policy", default="baseline")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--include-rollouts", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    policy = _policy_for(args.policy)
    report = run_suite(
        policy,
        suite=args.suite,
        policy_name=args.policy,
        include_rollouts=args.include_rollouts,
    )
    rendered = report.model_dump_json(indent=2) if args.format == "json" else report.to_markdown()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered)
        print(f"wrote suite report to {args.output}")
        return
    print(rendered)


def _policy_for(name: str):
    if name == "baseline":
        return ScriptedBaselinePolicy()
    for behavior in default_behaviors():
        if behavior.name == name:
            return behavior.policy
    available = ["baseline", *(behavior.name for behavior in default_behaviors())]
    raise ValueError(f"unknown policy: {name}. available policies: {available}")


if __name__ == "__main__":
    main()
