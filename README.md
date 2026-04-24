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

Rewards are logged as separate components: coverage, payout, evidence, fraud triage, subrogation, communication, reserve, efficiency, audit trail, penalties, and safety cap.

See `CODEX.md` for the full project brief and extension plan.

All rollout paths share `claimsops_env.agent_interface.RolloutRunner`; keep SFT generation, GRPO rewards, baseline evaluation, and model inference on that one interface.
