# ClaimsOps Gym

ClaimsOps Gym is a synthetic personal-auto claims operations environment for OpenEnv-compatible RL training.

The agent does not only approve or deny a claim. It runs a claims desk through validated tool calls: policy verification, evidence inspection, missing-document requests, payout calculation, leakage/fraud triage, reserve setting, subrogation, claimant communication, and final audit-trail submission.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
claimsops-baseline --seeds 5
claimsops-run-suite --suite smoke
claimsops-trace --scenario-family duplicate_line_item --seed 4
claimsops-calibrate --suite calibration
claimsops-server
```

The server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /metadata`
- `GET /docs`

## Repo Layout

```text
claimsops_env/
  agent_interface.py
  models.py
  scenario_templates.py
  generator.py
  tools.py
  verifier.py
  environment.py
  policies.py
  suites.py
  suite_runner.py
  calibration.py
  server.py
training/
  run_suite.py
  generate_sft_data.py
  eval_baseline.py
  calibrate_rewards.py
  train_grpo.py
inference.py
tests/
docs/
configs/
```

## Example Action

```json
{
  "tool": "get_policy",
  "args": {
    "policy_id": "POL-1234"
  }
}
```

## Reward Columns

Rewards are logged as separate components: workflow progress, coverage, payout, evidence, leakage control, fraud triage, subrogation, communication, reserve, compliance, financial controls, efficiency, audit trail, penalties, and safety cap.

## Environment Depth

Scenario templates now cover 15+ realistic auto physical-damage workflows, including missing evidence, ownership gaps, prior damage, duplicate estimate lines, rental/storage leakage, SIU triage, subrogation, authority approval, excluded-driver denial, and total loss valuation.

Several tools schedule pending operational events instead of instantly returning answers. Requested documents, appraisals, supplements, valuations, and authority decisions arrive on later steps and are exposed through the observation as pending events and event history.

The claim file also tracks document lifecycle records and repair estimate line
items. This lets rewards distinguish requested versus received versus reviewed
evidence, and lets leakage control score duplicate/prior-damage line handling
instead of relying only on one aggregate estimate amount.

## Trace Debugger

Use `claimsops-trace` to inspect one rollout as a black-box recorder:

```bash
claimsops-trace --scenario-family duplicate_line_item --seed 4
claimsops-trace --suite hard_eval --episode-index 3
claimsops-trace --input outputs/model-rollout.json --format json
```

The trace shows each action, tool result, visible state diffs, reward-component
deltas, verifier-side rubric misses, violations, and final reward breakdown. It
reads the shared rollout schema, so saved baseline/model trajectories can be
inspected without a separate harness.

## Reward Calibration

Use `claimsops-calibrate` before changing training code. It runs known workflow
behaviors through the shared rollout harness and reports reward columns,
rubric misses, safety caps, and a good/mixed/bad verdict:

```bash
claimsops-calibrate --suite calibration
claimsops-calibrate --families all --seeds 0,1 --format json --include-rollouts --output outputs/calibration.json
```

The goal is to catch reward-shaping mistakes early: a careful adjuster workflow
should outrank shortcut policies, while overpaying, premature closure, SIU
over-referral, and missing-evidence workflows should be visibly degraded.
Expectations are scenario-aware, so a probe can be neutral when the shortcut it
tests is not relevant to that claim family.

## Scenario Suites

Named suites keep training, eval, tracing, and calibration on the same episode
contract:

```bash
claimsops-run-suite --suite smoke
claimsops-run-suite --suite heldout --format json --include-rollouts --output outputs/heldout-baseline.json
claimsops-baseline --suite curriculum_phase_1
claimsops-generate-sft --suite curriculum_phase_1 --output outputs/phase1-sft.jsonl
```

Suites live in `claimsops_env.suites` and include `smoke`, `calibration`,
`curriculum_phase_1` through `curriculum_phase_4`, `hard_eval`, `heldout`, and
`demo`. The `heldout` suite uses separate seeds and should stay reserved for
final model evaluation.

See `CODEX.md` for the full project brief and extension plan.

All rollout paths share `claimsops_env.agent_interface.RolloutRunner`; keep SFT generation, GRPO rewards, baseline evaluation, suite runs, and model inference on that one interface.
