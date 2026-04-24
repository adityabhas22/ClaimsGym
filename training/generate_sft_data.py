from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimsops_env.agent_interface import RolloutRunner, SYSTEM_PROMPT
from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.scenario_templates import SCENARIO_FAMILIES
from claimsops_env.suites import SuiteEpisode, get_suite, list_suites


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT warm-start data from the shared rollout harness.")
    parser.add_argument("--output", default="outputs/sft-warmstart.jsonl")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--families", default=",".join(SCENARIO_FAMILIES))
    parser.add_argument("--suite", default=None, choices=[suite.name for suite in list_suites()])
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes = _episodes_from_args(args)
    count = 0
    with output.open("w") as handle:
        runner = RolloutRunner()
        for episode in episodes:
            result = runner.run(
                ScriptedBaselinePolicy(),
                seed=episode.seed,
                scenario_family=episode.scenario_family,
                max_steps=episode.max_steps,
            )
            if not result.trajectory:
                continue
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for step in result.trajectory:
                messages.append({"role": "user", "content": json.dumps(step.observation, sort_keys=True)})
                messages.append({"role": "assistant", "content": json.dumps(step.action, sort_keys=True)})
            record = {
                "scenario_family": episode.scenario_family,
                "seed": episode.seed,
                "suite": args.suite,
                "split": episode.split,
                "tags": list(episode.tags),
                "reward": result.total_reward,
                "reward_breakdown": result.reward_breakdown,
                "messages": messages,
                "prompt": render_observation_from_step(result),
                "completion": "\n".join(json.dumps(step.action, sort_keys=True) for step in result.trajectory),
            }
            handle.write(json.dumps(record) + "\n")
            count += 1
    print(f"wrote {count} examples to {output}")


def _episodes_from_args(args: argparse.Namespace) -> tuple[SuiteEpisode, ...]:
    if args.suite:
        return get_suite(args.suite).episodes
    families = [family.strip() for family in args.families.split(",") if family.strip()]
    return tuple(
        SuiteEpisode(scenario_family=family, seed=seed, split="sft", tags=("sft",))
        for family in families
        for seed in range(args.seeds)
    )


def render_observation_from_step(result) -> str:
    first = result.trajectory[0].observation
    return json.dumps({"system": SYSTEM_PROMPT, "initial_observation": first}, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
