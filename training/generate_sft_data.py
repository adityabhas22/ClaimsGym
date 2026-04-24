from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimsops_env.agent_interface import RolloutRunner, SYSTEM_PROMPT
from claimsops_env.policies import ScriptedBaselinePolicy
from claimsops_env.scenario_templates import SCENARIO_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT warm-start data from the shared rollout harness.")
    parser.add_argument("--output", default="outputs/sft-warmstart.jsonl")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--families", default=",".join(SCENARIO_FAMILIES))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    families = [family.strip() for family in args.families.split(",") if family.strip()]
    count = 0
    with output.open("w") as handle:
        for family in families:
            for seed in range(args.seeds):
                result = RolloutRunner().run(ScriptedBaselinePolicy(), seed=seed, scenario_family=family)
                if not result.trajectory:
                    continue
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for step in result.trajectory:
                    messages.append({"role": "user", "content": json.dumps(step.observation, sort_keys=True)})
                    messages.append({"role": "assistant", "content": json.dumps(step.action, sort_keys=True)})
                record = {
                    "scenario_family": family,
                    "seed": seed,
                    "reward": result.total_reward,
                    "reward_breakdown": result.reward_breakdown,
                    "messages": messages,
                    "prompt": render_observation_from_step(result),
                    "completion": "\n".join(json.dumps(step.action, sort_keys=True) for step in result.trajectory),
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
    print(f"wrote {count} examples to {output}")


def render_observation_from_step(result) -> str:
    first = result.trajectory[0].observation
    return json.dumps({"system": SYSTEM_PROMPT, "initial_observation": first}, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
