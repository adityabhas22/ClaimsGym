"""Local Hugging Face inference policy for ClaimsOps.

Mirrors the OpenAI-backed policy in ``inference.py`` but runs a transformers
model on the host GPU. Designed for DGX Spark (Blackwell, bf16, SDPA) but also
works on any CUDA host with PyTorch + transformers installed.

The policy reuses the shared ``RolloutRunner``, ``parse_action_text``, and the
new ``render_compact_prompt`` / ``render_system_prompt_with_catalog`` helpers so
that SFT data, GRPO reward, baseline eval, and inference stay on one contract.

Usage:
    from training.hf_inference import HfChatPolicy, HfChatConfig
    from claimsops_env.agent_interface import RolloutRunner

    policy = HfChatPolicy(HfChatConfig(model_name="Qwen/Qwen3-4B-Instruct-2507"))
    result = RolloutRunner().run(policy, seed=0, scenario_family="covered_collision")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from claimsops_env.agent_interface import (
    ActionPolicy,
    AgentContext,
    parse_action_text,
    render_compact_prompt,
    render_system_prompt_with_catalog,
)
from claimsops_env.models import Observation


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class HfChatConfig:
    model_name: str = DEFAULT_MODEL
    tokenizer_name: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    enable_thinking: bool = False
    max_history_turns: int = 6
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"
    quantization: str | None = None
    stop_strings: tuple[str, ...] = field(default_factory=lambda: ("</tool_call>",))


def config_from_env() -> HfChatConfig:
    cfg = HfChatConfig()
    cfg.model_name = os.getenv("CLAIMSOPS_MODEL", cfg.model_name)
    cfg.tokenizer_name = os.getenv("CLAIMSOPS_TOKENIZER") or cfg.tokenizer_name
    cfg.device_map = os.getenv("CLAIMSOPS_DEVICE_MAP", cfg.device_map)
    cfg.torch_dtype = os.getenv("CLAIMSOPS_DTYPE", cfg.torch_dtype)
    cfg.max_new_tokens = int(os.getenv("CLAIMSOPS_MAX_NEW_TOKENS", str(cfg.max_new_tokens)))
    cfg.temperature = float(os.getenv("CLAIMSOPS_TEMPERATURE", str(cfg.temperature)))
    cfg.top_p = float(os.getenv("CLAIMSOPS_TOP_P", str(cfg.top_p)))
    cfg.do_sample = os.getenv("CLAIMSOPS_DO_SAMPLE", "1").lower() not in {"0", "false", "no"}
    cfg.enable_thinking = os.getenv("CLAIMSOPS_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"}
    cfg.max_history_turns = int(os.getenv("CLAIMSOPS_MAX_HISTORY_TURNS", str(cfg.max_history_turns)))
    cfg.attn_implementation = os.getenv("CLAIMSOPS_ATTN", cfg.attn_implementation)
    cfg.quantization = os.getenv("CLAIMSOPS_QUANT") or None
    return cfg


class HfChatPolicy(ActionPolicy):
    """Transformers-backed chat policy.

    Each turn builds a chat message list of:
        system : instructions + action catalog (rendered once; stable across turns)
        user   : compact observation for the current step
        asst   : previous assistant tool call (from context.history)
        user   : next compact observation
        ...

    The full observation is not re-sent every turn because the environment
    places the relevant tool result into ``observation.latest_tool_result``.
    """

    def __init__(self, config: HfChatConfig | None = None) -> None:
        self.config = config or HfChatConfig()
        self._system_text = render_system_prompt_with_catalog()
        self._tokenizer, self._model = _load_backend(self.config)

    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any] | str:
        messages = self._build_messages(observation, context)
        prompt = self._apply_template(messages)
        completion = self._generate(prompt)
        return parse_action_text(_strip_tool_call_wrapper(completion))

    def _build_messages(self, observation: Observation, context: AgentContext) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_text}]
        history = context.history[-self.config.max_history_turns :] if self.config.max_history_turns else context.history
        for step in history:
            obs_dict = step.observation
            user_payload = {
                "claim_id": obs_dict.get("claim_id"),
                "open_tasks": obs_dict.get("open_tasks", []),
                "latest_tool_result": obs_dict.get("latest_tool_result"),
                "workflow_affordances": obs_dict.get("workflow_affordances"),
                "alerts": obs_dict.get("alerts", []),
                "remaining_steps": obs_dict.get("remaining_steps"),
            }
            messages.append({"role": "user", "content": json.dumps(user_payload, separators=(",", ":"), sort_keys=True)})
            assistant_content = (
                json.dumps(step.action, separators=(",", ":"), sort_keys=True)
                if isinstance(step.action, dict)
                else str(step.action)
            )
            messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": render_compact_prompt(observation)})
        return messages

    def _apply_template(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self._tokenizer.apply_chat_template(
                messages, enable_thinking=self.config.enable_thinking, **kwargs
            )
        except TypeError:
            # Tokenizers without the enable_thinking kwarg (non-Qwen3 families).
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _generate(self, prompt: str) -> str:
        import torch  # type: ignore[import-not-found]

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": max(1e-4, self.config.temperature),
            "top_p": self.config.top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.config.stop_strings:
            try:
                gen_kwargs["stop_strings"] = list(self.config.stop_strings)
                gen_kwargs["tokenizer"] = self._tokenizer
            except Exception:
                pass
        with torch.inference_mode():
            output = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, inputs.input_ids.shape[-1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text


def _load_backend(config: HfChatConfig) -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Install training extras before using HfChatPolicy: pip install -e '.[train]'"
        ) from exc

    dtype = getattr(torch, config.torch_dtype, torch.bfloat16)
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "attn_implementation": config.attn_implementation,
    }
    if config.quantization in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "bitsandbytes quantization requested but unavailable: pip install bitsandbytes"
            ) from exc
        if config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL)


def _strip_tool_call_wrapper(text: str) -> str:
    """Qwen3 may wrap the JSON action in <tool_call>...</tool_call> blocks."""

    match = _TOOL_CALL_RE.search(text)
    if match:
        return match.group(1).strip()
    return text
