from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from claimsops_env.agent_interface import RolloutRunner, RolloutResult
from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.tracing import trace_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a ClaimsOps rollout with state diffs and reward deltas.")
    parser.add_argument("--input", default=None, help="Optional JSON file containing a RolloutResult or a list with one RolloutResult.")
    parser.add_argument("--scenario-family", default="duplicate_line_item")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--policy", default="baseline", choices=["baseline"])
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rollout = _load_rollout(Path(args.input)) if args.input else _run_rollout(args)
    trace = trace_rollout(rollout)
    rendered = trace.model_dump_json(indent=2) if args.format == "json" else trace.to_markdown()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered)
        print(f"wrote trace to {args.output}")
        return
    print(rendered)


def _run_rollout(args: argparse.Namespace) -> RolloutResult:
    policy = ScriptedBaselinePolicy()
    return RolloutRunner().run(
        policy,
        seed=args.seed,
        scenario_family=args.scenario_family,
        max_steps=args.max_steps,
    )


def _load_rollout(path: Path) -> RolloutResult:
    payload: Any = json.loads(path.read_text())
    if isinstance(payload, list):
        if not payload:
            raise ValueError("input rollout list is empty")
        payload = payload[0]
    if isinstance(payload, dict) and "trajectory" not in payload and "rollout" in payload:
        payload = payload["rollout"]
    return RolloutResult.model_validate(payload)


if __name__ == "__main__":
    main()
