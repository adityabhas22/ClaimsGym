from __future__ import annotations

"""Minimal GRPO/RLVR scaffold.

This file is intentionally light until the environment contract is stable. The
important extension point is `claimsops_reward`, which runs an episode and
returns the scalar reward while the environment logs component columns.
"""

import json
from dataclasses import dataclass
from typing import Any

from claimsops_env.agent_interface import parse_action_text, render_training_prompt
from claimsops_env.environment import ClaimsOpsEnv


DEFAULT_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


@dataclass(frozen=True)
class TrainingRunConfig:
    model_name: str = DEFAULT_MODEL_CANDIDATES[0]
    output_dir: str = "outputs/claimsops-grpo"
    max_episode_steps: int = 12
    num_generations: int = 4
    use_vllm: bool = False
    use_unsloth: bool = False


def build_prompt(seed: int, scenario_family: str | None = None) -> str:
    env = ClaimsOpsEnv()
    observation = env.reset(seed=seed, scenario_family=scenario_family)
    return render_training_prompt(observation)


def claimsops_reward(completions: list[str], prompts: list[str] | None = None, **_: Any) -> list[float]:
    rewards = []
    for index, completion in enumerate(completions):
        seed = index
        env = ClaimsOpsEnv()
        env.reset(seed=seed)
        done = False
        last_reward = 0.0
        # First scaffold treats each completion as a single action. The next
        # version should use TRL/OpenEnv multi-turn rollouts directly.
        action = parse_action_text(completion)
        result = env.step(action)
        done = result.done
        last_reward = result.reward
        while not done:
            break
        rewards.append(last_reward)
    return rewards


def main() -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install training dependencies first: pip install -e '.[train]' "
            "or add '.[train-unsloth]' on supported GPU hosts."
        ) from exc

    config = TrainingRunConfig()
    dataset = Dataset.from_list([{"prompt": build_prompt(seed)} for seed in range(32)])
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        max_completion_length=512,
        use_vllm=config.use_vllm,
    )
    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=claimsops_reward,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
