# ClaimsOps Gym Project Brief

## Thesis

ClaimsOps Gym is not a static insurance adjudication benchmark. It is a lifecycle claims-operations environment where an LLM agent manages a synthetic personal-auto claim through tool calls: verify coverage, inspect evidence, request missing documents, control leakage, set reserve, triage SIU risk, open subrogation, communicate with the claimant, and submit a final decision with an audit trail.

The professional capability we want to train is sequential operations under uncertainty, not one-shot approve/deny classification.

## Hackathon Fit

The environment targets OpenEnv-style agentic RL:

- `reset(seed)` creates a synthetic claim file.
- `step(action)` applies one validated tool call.
- `state()` returns visible debug state without hidden verifier truth.
- Rewards are decomposed into independently logged columns.
- Hidden truth and safety caps prevent trivial reward hacking.
- Training and demo scripts compare baseline versus trained trajectories.

This fits the World Modeling / Professional Tasks theme: long-horizon workflow, partially observable state, API orchestration, auditability, and recovery from missing or conflicting evidence.

## MVP Scope

Personal auto physical damage claims only. Avoid health insurance and real consumer data. All data is synthetic.

Current scenario families live in `claimsops_env.scenario_templates` and include:

- Active covered collision.
- Comprehensive claim with deductible.
- Policy lapse.
- Coverage limit exceeded.
- Missing police report.
- Incomplete first notice facts.
- Ownership / insurable-interest gap.
- Prior damage leakage.
- Duplicate estimate line.
- Rental, towing, and storage leakage pressure.
- Suspicious policy inception timing.
- Statement conflict with telematics.
- Named-driver exclusion.
- Third-party subrogation opportunity.
- Authority threshold escalation.
- Possible total loss with valuation and salvage.

The generator consumes typed scenario templates instead of burying all domain
rubrics inside random branches. Each template names the visible facts,
operational tasks, required evidence, expected estimate review, fraud/SIU
signal, subrogation expectation, authority posture, and any initial platform
events.

## Environment Contract

Actions use strict JSON:

```json
{
  "tool": "inspect_repair_estimate",
  "args": {
    "estimate_id": "EST-12345"
  }
}
```

The initial observation is compact and includes the visible claim file, available evidence, open tasks, latest tool result, financial snapshot, available tools, and remaining steps. It does not include hidden labels such as expected payout, fraud truth, or expected coverage outcome.

Observations also expose realistic platform state:

- pending external events, such as document arrival, appraisal completion, valuation reports, and authority decisions
- event history
- visible storage/rental leakage counters
- alerts and audit gaps derived from visible facts

## Tool Design

The tool registry is intentionally claim-platform shaped for the MVP:

- `get_policy`
- `get_policy_snapshot`
- `check_policy_status`
- `verify_coverage`
- `create_or_update_exposure`
- `inspect_repair_estimate`
- `inspect_evidence`
- `assign_appraisal`
- `review_estimate`
- `request_valuation`
- `request_document`
- `query_prior_claims`
- `check_fraud_indicators`
- `set_reserve`
- `approve_payment`
- `issue_payment`
- `request_authority_approval`
- `refer_to_siu`
- `open_siu_referral`
- `open_subrogation`
- `send_claimant_message`
- `add_claim_note`
- `close_claim`
- `submit_final_decision`

Each tool owns its argument schema and mutation logic. Environment state should not be mutated outside tools except for step accounting, action logging, and terminal state.

Several tools intentionally schedule events rather than completing work
immediately. `request_document`, `assign_appraisal`, `request_valuation`,
`review_estimate` with supplement, and `request_authority_approval` place
pending events into the claim file. The environment advances those events on
subsequent steps. This gives the model a real workflow problem: continue useful
work while waiting, then incorporate returned documents and reports before
closure.

## Reward Design

The scalar reward is only the reporting aggregate. The real design surface is the independent reward columns:

- format validity
- workflow progress
- coverage
- payout
- evidence
- leakage control
- fraud triage
- subrogation
- communication
- reserve
- compliance
- financial controls
- efficiency
- audit trail
- penalties
- safety cap

Safety caps currently cover overpayment, premature payment before required evidence, denial of a covered claim without investigation, claimant-facing fraud-score disclosure, fabricated evidence IDs, hidden-state probing, and repeated invalid actions.

Evidence reward now separates requested documents from received documents.
Requesting the right document helps, but final payment before the document
arrives remains capped. Compliance and leakage rewards also account for
pending external events, authority approval state, rental days, and storage
charges.

## Extensibility Rules

Add scenario depth through `claimsops_env.scenario_templates`, the generator,
and the hidden rubric, not by hardcoding policy behavior into the trainer.

Add workflow realism through visible operational affordances such as tasks, blockers, required document statuses, and financial snapshots. This is fair because real adjusters use claim systems that surface this structure.

Add reward functions as composable classes. Every new reward should have a unit test that isolates the behavior it scores.

Keep deployment glue thin. The FastAPI app should expose the environment, not contain domain logic.

## Shared Harness Rule

OpsArena taught us not to scatter rollout loops across SFT, RL, eval, and inference scripts. ClaimsOps Gym has one shared agent interface in `claimsops_env.agent_interface`:

- one action catalog
- one observation renderer
- one action parser
- one `RolloutRunner`
- one trajectory schema

SFT generation, GRPO reward functions, scripted baselines, and `inference.py` should all use this interface. If the environment API changes, update the shared interface first and keep scripts thin.

## Model And Training Direction

Use a capable instruct model with reliable JSON/tool-call behavior for the first training runs. Start in the 4B to 8B range so rollouts are cheap enough to inspect manually. Good candidate classes are Qwen instruct and Llama instruct models; the exact checkpoint should be refreshed before training starts.

Use TRL GRPO/RLVR once the environment is stable. Unsloth is useful when GPU memory is the bottleneck. Use light SFT only if the model cannot reliably emit valid JSON actions.

Training should proceed as curriculum:

1. Simple coverage and deductible cases.
2. Missing docs and policy lapse.
3. Leakage and payout calculations.
4. SIU, subrogation, authority thresholds, and mixed hidden tests.

## Demo Story

Show three episodes:

- Simple covered collision.
- Policy lapse or excluded claim.
- Hard claim with leakage, SIU signal, or subrogation.

For each, show baseline trajectory, reward breakdown, trained trajectory, reward breakdown, and one short explanation of what improved.

## Sources To Revisit

- OpenEnv framework: https://github.com/meta-pytorch/OpenEnv
- TRL documentation: https://huggingface.co/docs/trl/main/en/index
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Reward engineering paper: https://arxiv.org/abs/2408.10215
- Reward/benchmark design paper: https://arxiv.org/abs/2601.19100
