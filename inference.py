from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from claimsops_env.agent_interface import (
    AgentContext,
    RolloutRunner,
    SYSTEM_PROMPT,
    parse_action_text,
    render_observation,
)
from claimsops_env.models import Observation


BENCHMARK = os.getenv("CLAIMSOPS_BENCHMARK", "claimsops-gym")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "missing"
SEED = int(os.getenv("CLAIMSOPS_SEED", "7"))
SCENARIO_FAMILY = os.getenv("CLAIMSOPS_SCENARIO_FAMILY") or None
MAX_STEPS = int(os.getenv("CLAIMSOPS_MAX_STEPS", "12"))
TEMPERATURE = float(os.getenv("CLAIMSOPS_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("CLAIMSOPS_MAX_TOKENS", "512"))
MAX_HISTORY_STEPS = int(os.getenv("CLAIMSOPS_MAX_HISTORY_STEPS", "8"))


class OpenAIChatPolicy:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any] | str:
        messages = self._messages(observation, context)
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = response.choices[0].message.content or ""
        return parse_action_text(content)

    def _messages(self, observation: Observation, context: AgentContext) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for step in context.history[-MAX_HISTORY_STEPS:]:
            messages.append({"role": "user", "content": json.dumps(step.observation, sort_keys=True)})
            messages.append({"role": "assistant", "content": json.dumps(step.action, sort_keys=True)})
            tool_result = (step.next_observation.get("latest_tool_result") or {}).get("summary", "")
            messages.append(
                {
                    "role": "user",
                    "content": "Previous action result: "
                    + json.dumps(
                        {"reward": step.reward, "done": step.done, "tool_result": tool_result},
                        sort_keys=True,
                    ),
                }
            )
        messages.append({"role": "user", "content": render_observation(observation)})
        return messages


def format_start_line(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def format_step_line(step: int, action: dict[str, Any] | str, reward: float, done: bool, error: str | None) -> str:
    action_text = json.dumps(action, separators=(",", ":"), sort_keys=True) if isinstance(action, dict) else str(action)
    flat_error = " ".join(str(error).split()) if error else "null"
    return f"[STEP] step={step} action={action_text} reward={reward:.2f} done={str(done).lower()} error={flat_error}"


def format_end_line(success: bool, steps: int, score: float, rewards: list[float]) -> str:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    return f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={reward_text}"


def main() -> None:
    task = SCENARIO_FAMILY or "seeded_claim"
    print(format_start_line(task, BENCHMARK, MODEL_NAME), flush=True)
    result = RolloutRunner().run(
        OpenAIChatPolicy(),
        seed=SEED,
        scenario_family=SCENARIO_FAMILY,
        max_steps=MAX_STEPS,
    )
    rewards = []
    for step in result.trajectory:
        rewards.append(step.reward)
        error = None
        if not step.info.get("action_valid", True):
            error = (step.next_observation.get("latest_tool_result") or {}).get("summary", "invalid action")
        print(format_step_line(step.step, step.action, step.reward, step.done, error), flush=True)
    print(format_end_line(result.success, result.steps, result.total_reward, rewards), flush=True)


if __name__ == "__main__":
    main()
