"""Run a local HF model across ClaimsOps suites and report reward columns.

This is the first real base-model eval entry point. It is a thin CLI wrapper
around ``RolloutRunner`` + ``HfChatPolicy`` so SFT data, GRPO reward, scripted
baseline, and model eval all stay on the same rollout contract.

Typical usage on DGX Spark:

    export CLAIMSOPS_MODEL=Qwen/Qwen3-4B-Instruct-2507
    claimsops-local-baseline --suite heldout --output outputs/qwen3-4b-heldout.json
    claimsops-local-baseline --suite calibration --seeds 0,1 --families covered_collision

Output: aggregate reward columns, per-family means, format-validity rate,
violation counts, and (optionally) the full trajectory payload for later
trace debugging with ``claimsops-trace --input``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from claimsops_env.agent_interface import RolloutRunner
from claimsops_env.suites import SuiteEpisode, list_suites, resolve_suite_episodes

from training.hf_inference import HfChatConfig, HfChatPolicy, config_from_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local HF model across ClaimsOps suites.")
    parser.add_argument("--suite", default=None, choices=[suite.name for suite in list_suites()])
    parser.add_argument("--families", default=None, help="Comma-separated scenario families (used when --suite is omitted).")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (used when --suite is omitted).")
    parser.add_argument("--max-steps", type=int, default=None, help="Override episode step budget.")
    parser.add_argument("--model", default=None, help="HF model id; defaults to CLAIMSOPS_MODEL or Qwen3-4B.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id (defaults to --model).")
    parser.add_argument("--dtype", default=None, choices=[None, "bfloat16", "float16", "float32"])
    parser.add_argument("--quantization", default=None, choices=[None, "4bit", "8bit"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-history-turns", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 thinking mode (OFF by default; see Qwen3 issue #1817).")
    parser.add_argument("--output", default=None, help="Write JSON report to this path.")
    parser.add_argument("--include-rollouts", action="store_true", help="Include full trajectory JSON in the output.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> HfChatConfig:
    cfg = config_from_env()
    if args.model:
        cfg.model_name = args.model
    if args.tokenizer:
        cfg.tokenizer_name = args.tokenizer
    if args.dtype:
        cfg.torch_dtype = args.dtype
    if args.quantization:
        cfg.quantization = args.quantization
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.max_history_turns is not None:
        cfg.max_history_turns = args.max_history_turns
    if args.enable_thinking:
        cfg.enable_thinking = True
    return cfg


def _resolve_episodes(args: argparse.Namespace) -> tuple[SuiteEpisode, ...]:
    families = [item.strip() for item in args.families.split(",") if item.strip()] if args.families else None
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()] if args.seeds else None
    return resolve_suite_episodes(
        suite_name=args.suite,
        families=families,
        seeds=seeds,
        max_steps=args.max_steps,
    )


def _episode_row(result: Any, episode: SuiteEpisode) -> dict[str, Any]:
    breakdown = result.reward_breakdown or {}
    valid_steps = sum(1 for step in result.trajectory if step.info.get("action_valid", True))
    violations: list[str] = []
    for step in result.trajectory:
        violations.extend(step.info.get("violations", []) or [])
    return {
        "episode_id": episode.episode_id,
        "scenario_family": episode.scenario_family,
        "seed": episode.seed,
        "split": episode.split,
        "steps": result.steps,
        "success": result.success,
        "total_reward": result.total_reward,
        "format_validity_rate": (valid_steps / result.steps) if result.steps else 0.0,
        "violations": list(dict.fromkeys(violations)),
        "reward_breakdown": breakdown,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    reward_keys = sorted({key for row in rows for key in row["reward_breakdown"].keys() if isinstance(row["reward_breakdown"].get(key), (int, float))})
    family_buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family_buckets.setdefault(row["scenario_family"], []).append(row)
    per_family = {
        family: {
            "episodes": len(items),
            "mean_total": round(mean(item["total_reward"] for item in items), 4),
            "mean_format_validity": round(mean(item["format_validity_rate"] for item in items), 4),
            "mean_steps": round(mean(item["steps"] for item in items), 2),
        }
        for family, items in family_buckets.items()
    }
    violation_counter: Counter[str] = Counter()
    for row in rows:
        violation_counter.update(row["violations"])
    return {
        "episodes": len(rows),
        "mean_total_reward": round(mean(row["total_reward"] for row in rows), 4),
        "mean_format_validity": round(mean(row["format_validity_rate"] for row in rows), 4),
        "mean_steps": round(mean(row["steps"] for row in rows), 2),
        "reward_columns": {
            key: round(mean(row["reward_breakdown"].get(key, 0.0) for row in rows), 4) for key in reward_keys
        },
        "per_family": per_family,
        "violations": dict(violation_counter.most_common()),
    }


def main() -> None:
    args = _parse_args()
    episodes = _resolve_episodes(args)
    config = _config_from_args(args)
    print(f"[local-baseline] model={config.model_name} dtype={config.torch_dtype} episodes={len(episodes)}")

    policy = HfChatPolicy(config)
    runner = RolloutRunner()
    rows: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        result = runner.run(
            policy,
            seed=episode.seed,
            scenario_family=episode.scenario_family,
            max_steps=episode.max_steps,
        )
        row = _episode_row(result, episode)
        rows.append(row)
        print(
            f"[{index}/{len(episodes)}] {episode.episode_id}: "
            f"reward={row['total_reward']:.3f} steps={row['steps']} "
            f"fmt={row['format_validity_rate']:.2f} violations={row['violations']}"
        )
        if args.include_rollouts:
            rollouts.append({"episode": row, "rollout": result.model_dump(mode="json")})

    report = {
        "config": {
            "model_name": config.model_name,
            "tokenizer_name": config.tokenizer_name or config.model_name,
            "dtype": config.torch_dtype,
            "quantization": config.quantization,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "max_history_turns": config.max_history_turns,
            "enable_thinking": config.enable_thinking,
        },
        "suite": args.suite,
        "episodes": rows,
        "aggregate": _aggregate(rows),
    }
    if args.include_rollouts:
        report["rollouts"] = rollouts

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[local-baseline] wrote {path}")
    else:
        print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
