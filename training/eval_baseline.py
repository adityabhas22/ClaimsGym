from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any

from claimsops_env.agent_interface import RolloutRunner
from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.suite_runner import run_suite


def run_episode(seed: int, scenario_family: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
    result = RolloutRunner().run(
        ScriptedBaselinePolicy(),
        seed=seed,
        scenario_family=scenario_family,
        max_steps=max_steps,
    )
    return result.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--scenario-family", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.suite:
        report = run_suite(ScriptedBaselinePolicy(), suite=args.suite, policy_name="baseline")
        print(report.model_dump_json(indent=2) if args.json else report.to_markdown())
        return

    results = [run_episode(seed, args.scenario_family, args.max_steps) for seed in range(args.seeds)]
    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"episodes={len(results)} mean_reward={mean(result['total_reward'] for result in results):.3f}")
    for key in [
        "coverage",
        "payout",
        "evidence",
        "fraud_triage",
        "subrogation",
        "communication",
        "reserve",
        "audit_trail",
    ]:
        print(f"{key}={mean(result['reward_breakdown'][key] for result in results):.3f}")


if __name__ == "__main__":
    main()
