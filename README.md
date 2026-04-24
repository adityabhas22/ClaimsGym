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
  server.py
training/
  generate_sft_data.py
  eval_baseline.py
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

See `CODEX.md` for the full project brief and extension plan.

All rollout paths share `claimsops_env.agent_interface.RolloutRunner`; keep SFT generation, GRPO rewards, baseline evaluation, and model inference on that one interface.
