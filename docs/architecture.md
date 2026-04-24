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
- `claimsops_env.scenario_templates`: typed scenario-template catalog with visible clues, workflow rubrics, document requirements, authority/SIU/subrogation labels, and initial event profiles.
- `claimsops_env.generator`: seeded scenario generator for claim families, visible claim-platform state, and hidden verifier labels.
- `claimsops_env.tools`: tool registry and state-mutating tool handlers.
- `claimsops_env.verifier`: composable reward component classes and safety caps.
- `claimsops_env.rubric`: verifier-side workflow rubric evaluator and condition registry.
- `claimsops_env.environment`: local `reset`, `step`, `state`, and metadata API.
- `claimsops_env.agent_interface`: shared action catalog, observation rendering, action parsing, rollout runner, and trajectory schema.
- `claimsops_env.suites`: named scenario suites for smoke checks, curriculum phases, calibration, hard evaluation, demo, and heldout splits.
- `claimsops_env.suite_runner`: aggregate suite runner and report schema over the shared rollout harness.
- `claimsops_env.tracing`: rollout trace builder for state diffs, reward deltas, and markdown/JSON debugging output.
- `claimsops_env.policies`: baseline policies that run through the same interface as model inference.
- `claimsops_env.calibration`: reward calibration suite for known good and bad workflow behaviors.
- `claimsops_env.server`: thin FastAPI wrapper for OpenEnv/Space deployment.
- `training.run_suite`: CLI wrapper for named suite runs and optional saved rollout payloads.
- `training.eval_baseline`: transparent baseline runner for smoke tests and reward-column reporting.
- `training.calibrate_rewards`: CLI wrapper for calibration reports and optional rollout payloads.
- `training.generate_sft_data`: warm-start trajectory generation through the shared rollout runner.
- `training.train_grpo`: minimal TRL GRPO scaffold.

## Extension Points

Add a new claim scenario by adding a `ScenarioTemplate`, then wire only the family-specific mechanics that cannot be expressed declaratively in `ScenarioGenerator._build_family`. Add a focused reward test that proves the hidden label is not exposed and the expected behavior scores higher than the wrong behavior.

Add a new tool by creating a `ToolHandler`, adding its args model, registering it in `build_tool_registry`, and adding at least one transition test. Tools are the only place where environment state should mutate.

Add a reward component by implementing `RewardComponent`, adding it to `DEFAULT_COMPONENTS`, adding a weight, and asserting its independent score in tests. Avoid reward logic that depends on model identity or trajectory text outside the validated action log.

Add a rubric condition by adding the condition key to a `WorkflowRubric` and registering a predicate in `claimsops_env.rubric.CONDITION_REGISTRY`. Rubrics stay verifier-side; do not add them to observations or tool results.

All training and inference scripts should call `RolloutRunner` or the shared render/parse helpers instead of reimplementing the loop. This prevents drift when SFT data, GRPO reward code, local eval, and demo inference each invent a slightly different action contract.

`claimsops_env.tracing.trace_rollout` consumes the shared `RolloutResult`
schema. It does not run a separate environment loop, so traces for scripted
baselines, model inference, saved eval rollouts, and future training examples
all use the same action/observation/reward contract.

`claimsops_env.calibration.run_calibration` also consumes the shared rollout
contract. Calibration policies are not trainer logic or hidden answer keys; they
are controlled probes for the verifier. They answer whether the reward system
ranks careful workflows above obvious shortcuts and what specific components,
rubric misses, safety caps, or violations explain a score.

Named suites are the episode-selection contract. Training, SFT generation,
baseline evaluation, calibration, trace selection, and future model inference
should refer to `claimsops_env.suites` instead of hardcoding separate family and
seed lists. This keeps curriculum, heldout evaluation, and debugging runs from
quietly drifting apart.

## Claim Platform State

The simulator now exposes a claim-system style file rather than a flat adjudication record:

- parties and roles
- incidents
- exposures with coverage, claimant, reserve, paid amount, and validation level
- activities/diary with due dates and close reasons
- reserve lines
- payments
- vendor/appraisal assignments
- claim notes
- claim documents with requested/received/reviewed status, source, confidence, issues, and evidence links
- repair estimate line items with category, amount, payable flag, review status, and leakage flags
- appraisal and estimate-review status
- pending external events and resolved event history
- rental days and storage charges for visible leakage pressure
- alerts and audit gaps derived from visible facts
- workflow affordances with phase, wait reasons, close blockers, due timing, recommended action categories, and useful tools

Hidden truth remains separate: expected payout, fraud truth, true leakage, required evidence, total-loss expectation, and authority requirements are used only by verifiers.

## Workflow Affordances

`Observation.workflow_affordances` is a model-facing navigation layer derived
only from visible claim-system state. It contains:

- `claim_phase`: coarse phase such as intake, investigation, waiting on external work, authority review, pre-closure, ready for decision, or closed.
- `waiting_on`: pending external events, requested documents, vendor assignments, or authority approval.
- `close_blockers`: visible tasks, audit gaps, open activities, and external waits that make final closure premature.
- `next_due_steps`: nearest visible activity or event deadline.
- `recommended_action_categories`: high-level categories such as coverage, evidence, leakage, reserve, communication, audit, authority, or decision.
- `action_availability`: tool-level booleans for currently useful actions.

These fields are not rewards and do not expose expected payout, required hidden
documents, fraud truth, or verifier rubrics. They are the equivalent of a claim
system surfacing work queues, blockers, and due work so a model can explore the
environment without needing brittle hidden task heuristics.

## Casework Depth

Documents are first-class claim-file records. A requested document starts with
status `requested`, moves to `received` when the event engine resolves the
arrival, and can move to `reviewed` when the agent inspects the linked evidence
ID. This creates a concrete distinction between asking for evidence and
actually incorporating it before closure.

Repair estimates contain structured line items. Each line has a category,
coverage, amount, payable flag, review status, and flags for leakage modes such
as duplicate operations, unrelated prior damage, rental/storage pressure, and
covered damage. The expected payable amount is still hidden, but the visible
line items let agents reason through the same kind of estimate review an
adjuster would perform.

## Event Simulation

The simulator has a lightweight operational event engine. Tools can schedule
pending events in `PlatformState.pending_events`; the environment decrements
them each step and resolves them into visible platform state.

Current event types:

- `document_arrival`
- `appraisal_complete`
- `supplement_received`
- `valuation_complete`
- `authority_decision`
- `claimant_response`
- `rental_day_accrual`
- `storage_fee_accrual`

This avoids the common benchmark failure mode where every tool is an immediate
oracle. The agent must request a missing item, keep working, wait when needed,
and only finalize after the returned evidence is visible.

## Scenario Families

The catalog currently covers straight-through coverage, comprehensive losses,
policy lapse, limit caps, missing evidence, ownership gaps, prior damage,
duplicate estimate lines, rental/storage leakage, suspicious inception,
statement conflicts, excluded-driver denials, subrogation, authority
thresholds, and total-loss valuation.

## Scenario Suites

`claimsops_env.suites` defines deterministic episode sets:

- `smoke`: fast health check across coverage, evidence, leakage, and authority.
- `calibration`: reward sanity gate for careful and shortcut workflow probes.
- `curriculum_phase_1`: basic coverage, deductible, lapse, and limits.
- `curriculum_phase_2`: missing documents and evidence handling.
- `curriculum_phase_3`: leakage control and estimate review.
- `curriculum_phase_4`: SIU, subrogation, authority, exclusions, conflicts, and total loss.
- `hard_eval`: edge-case evaluation mix.
- `heldout`: reserved split using separate seeds for final model evaluation.
- `demo`: three example episodes for storytelling.

`claimsops-run-suite` reports aggregate reward columns, per-family means, and
episode-level verdicts. `claimsops-baseline`, `claimsops-calibrate`,
`claimsops-trace`, and `claimsops-generate-sft` can all consume suite names so
they stay on the same scenario contract.

## Trace Debugging

`claimsops-trace` renders a single rollout as a developer-readable record:

- action JSON and tool summary
- visible state diffs between observations
- workflow-affordance phase, blocker, wait, and action-availability changes
- document/event/estimate-line changes
- reward deltas for every numeric reward column present in the rollout
- verifier-side rubric misses from step `info`
- violations and final reward breakdown

Reward columns are discovered dynamically from the `reward_breakdown` dict.
State diffs are lens-based over observation fields and keyed collections, so
new reward columns and most new visible state fields can be added without
rewriting the tracer.

## Reward Calibration

`claimsops-calibrate` runs a suite of deliberately different behaviors through
the environment:

- `careful_adjuster`: completes visible workflow tasks, waits on events,
  reviews returned documents, and finalizes with supported payment or denial.
- `missing_evidence`: skips material document request/review work.
- `siu_everything`: over-refers claims to SIU.
- `overpay`: investigates lightly but pays above supported exposure.
- `premature_final`: closes immediately.
- `authority_bypass`: tries to pay without required approval.

Each row reports total reward, component breakdown, rubric score, safety cap,
penalties, terminal state, and a good/mixed/bad verdict. Expectations are
scenario-aware: for example, the missing-evidence probe is neutral on a
straight-through claim with no required documents, but bad on a claim where a
police report, statement, ownership proof, or estimate breakdown is material.
Pass/fail ordering is focused on the important calibration claim: known good
workflows should beat known shortcut workflows. Scores among bad behaviors
remain diagnostic because some scenarios legitimately punish one bad shortcut
more severely than another.

## Workflow Rubrics

Scenario templates generate a hidden `WorkflowRubric` that describes good claim
handling in operational terms: required workflow milestones, material document
handling, expected estimate review, authority controls, SIU/subrogation
expectations, forbidden shortcuts, and final decision requirements. The agent
does not see the rubric in observations. The environment evaluates it in step
`info`, blends category scores into existing reward components, and records
misses for trace debugging.

## Training Direction

Start with JSON/tool-format competence before full RL. The first serious training loop should use TRL `GRPOTrainer` with a capable instruct model in the 4B to 8B class, then optionally use Unsloth when local GPU memory is the limiting factor. Keep model-specific choices in training configs, not in environment code.

Candidate model classes:

- Small/medium instruct models with strong JSON behavior for local iteration.
- Qwen or Llama instruct checkpoints in the 4B to 8B range for GRPO smoke runs.
- Larger hosted models only for generating baseline trajectories or teacher traces, not as hidden verifiers.

The exact checkpoint should be rechecked when training starts because model availability and TRL integration change quickly.
