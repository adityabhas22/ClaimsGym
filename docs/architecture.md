# ClaimsOps Gym Architecture

ClaimsOps Gym is a synthetic, stateful claims operations simulator. One episode is one personal-auto physical-damage claim file. The agent acts through typed tools and receives compact observations, while hidden scenario truth stays available only to deterministic verifiers.

## Design Principles

- Keep simulator state, tool transitions, reward functions, and transport wrappers separate.
- Make every reward component independently inspectable.
- Expose workflow affordances that a real claims desk would show, such as open tasks and remaining steps.
- Never expose hidden expected outcomes in agent-visible observations.
- Make scenario generation deterministic from seeds so rollouts are reproducible.
- Prefer typed models and schema validation over ad hoc action parsing.

## Core Modules

- `claimsops_env.models`: Pydantic contracts for policies, claims, evidence, observations, actions, final decisions, and reward breakdowns.
- `claimsops_env.generator`: seeded scenario generator for claim families and hidden verifier labels.
- `claimsops_env.tools`: tool registry and state-mutating tool handlers.
- `claimsops_env.verifier`: composable reward component classes and safety caps.
- `claimsops_env.environment`: local `reset`, `step`, `state`, and metadata API.
- `claimsops_env.agent_interface`: shared action catalog, observation rendering, action parsing, rollout runner, and trajectory schema.
- `claimsops_env.policies`: baseline policies that run through the same interface as model inference.
- `claimsops_env.server`: thin FastAPI wrapper for OpenEnv/Space deployment.
- `training.eval_baseline`: transparent baseline runner for smoke tests and reward-column reporting.
- `training.generate_sft_data`: warm-start trajectory generation through the shared rollout runner.
- `training.train_grpo`: minimal TRL GRPO scaffold.

## Extension Points

Add a new claim scenario by extending `ScenarioGenerator._build_family`, then add a focused reward test that proves the hidden label is not exposed and the expected behavior scores higher than the wrong behavior.

Add a new tool by creating a `ToolHandler`, adding its args model, registering it in `build_tool_registry`, and adding at least one transition test. Tools are the only place where environment state should mutate.

Add a reward component by implementing `RewardComponent`, adding it to `DEFAULT_COMPONENTS`, adding a weight, and asserting its independent score in tests. Avoid reward logic that depends on model identity or trajectory text outside the validated action log.

All training and inference scripts should call `RolloutRunner` or the shared render/parse helpers instead of reimplementing the loop. This prevents drift when SFT data, GRPO reward code, local eval, and demo inference each invent a slightly different action contract.

## Training Direction

Start with JSON/tool-format competence before full RL. The first serious training loop should use TRL `GRPOTrainer` with a capable instruct model in the 4B to 8B class, then optionally use Unsloth when local GPU memory is the limiting factor. Keep model-specific choices in training configs, not in environment code.

Candidate model classes:

- Small/medium instruct models with strong JSON behavior for local iteration.
- Qwen or Llama instruct checkpoints in the 4B to 8B range for GRPO smoke runs.
- Larger hosted models only for generating baseline trajectories or teacher traces, not as hidden verifiers.

The exact checkpoint should be rechecked when training starts because model availability and TRL integration change quickly.
