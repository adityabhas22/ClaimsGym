from __future__ import annotations

from typing import Any

from claimsops_env.agent_interface import AgentContext, ActionPolicy
from claimsops_env.models import Observation


class ScriptedBaselinePolicy(ActionPolicy):
    """Transparent baseline for smoke tests and first SFT traces.

    It uses only visible observations and previous tool results captured by the
    shared rollout runner. It is intentionally conservative and generic rather
    than a scenario oracle.
    """

    def next_action(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        if "verify_policy" in observation.open_tasks:
            return {"tool": "get_policy", "args": {"policy_id": observation.policy_id}}
        if "check_policy_status" in observation.open_tasks:
            return {
                "tool": "check_policy_status",
                "args": {"policy_id": observation.policy_id, "loss_date": observation.loss_date.isoformat()},
            }
        if "inspect_estimate" in observation.open_tasks:
            return {"tool": "inspect_repair_estimate", "args": {"estimate_id": observation.estimate_id}}
        for task in observation.open_tasks:
            if task.startswith("request_"):
                return {
                    "tool": "request_document",
                    "args": {
                        "doc_type": task.removeprefix("request_"),
                        "reason": "Required to complete coverage and liability review.",
                    },
                }
        if "screen_fraud_indicators" in observation.open_tasks:
            return {"tool": "check_fraud_indicators", "args": {"claim_id": observation.claim_id}}
        if self._should_query_prior_claims(observation, context):
            return {
                "tool": "query_prior_claims",
                "args": {"customer_id": observation.customer_id, "vehicle_id": observation.vehicle_id},
            }
        if "evaluate_subrogation" in observation.open_tasks:
            return {
                "tool": "open_subrogation",
                "args": {
                    "target_party": "third_party_driver",
                    "rationale": "Visible evidence indicates another party may be liable.",
                },
            }
        if "set_reserve" in observation.open_tasks:
            estimate = self._latest_success_data(context, "inspect_repair_estimate")
            amount = float(estimate.get("covered_amount") or observation.requested_amount)
            return {
                "tool": "set_reserve",
                "args": {"amount": amount, "rationale": "Reserve set from visible repair exposure."},
            }
        return self._final_decision(observation, context)

    def _final_decision(self, observation: Observation, context: AgentContext) -> dict[str, Any]:
        policy = observation.visible_policy or {}
        estimate = self._latest_success_data(context, "inspect_repair_estimate")
        status = self._latest_success_data(context, "check_policy_status")
        fraud = self._latest_success_data(context, "check_fraud_indicators")

        active_on_loss = bool(status.get("active_on_loss")) if status else policy.get("status") == "active"
        payment = 0.0
        decision = "deny"
        rationale = "policy_period"
        if active_on_loss and policy and estimate:
            limit_key = "comprehensive_limit" if observation.claim_type.value == "comprehensive" else "collision_limit"
            payment = max(0.0, min(float(estimate["covered_amount"]), float(policy[limit_key])) - float(policy["deductible"]))
            decision = "approve" if payment > 0 else "deny"
            rationale = "Active policy and estimate reviewed; payment applies deductible, limit, and covered damage."

        indicators = fraud.get("indicators", []) if fraud else []
        evidence_ids = [evidence.evidence_id for evidence in observation.available_evidence[:4]]
        reserve_amount = observation.financial_snapshot.reserve_amount or max(payment, observation.requested_amount)
        message = (
            f"Your claim decision is {decision}. "
            f"The payable amount is ${payment:.2f} after applying the policy deductible where applicable. "
            "We will contact you with any next step."
        )
        return {
            "tool": "submit_final_decision",
            "args": {
                "decision": decision,
                "payment_amount": payment,
                "reserve_amount": reserve_amount,
                "siu_referral": bool(indicators),
                "subrogation": any(
                    "third_party" in flag
                    for evidence in observation.available_evidence
                    for flag in evidence.flags
                ),
                "claimant_message": message,
                "evidence_cited": evidence_ids,
                "rationale": rationale,
            },
        }

    @staticmethod
    def _should_query_prior_claims(observation: Observation, context: AgentContext) -> bool:
        if any(step.action and isinstance(step.action, dict) and step.action.get("tool") == "query_prior_claims" for step in context.history):
            return False
        estimate_flags = [flag for evidence in observation.available_evidence for flag in evidence.flags]
        return any("unrelated" in flag or "prior" in flag for flag in estimate_flags)

    @staticmethod
    def _latest_success_data(context: AgentContext, tool_name: str) -> dict[str, Any]:
        for step in reversed(context.history):
            action = step.action
            if not isinstance(action, dict) or action.get("tool") != tool_name:
                continue
            if not step.info.get("action_valid"):
                continue
            latest = step.next_observation.get("latest_tool_result") or {}
            return latest.get("data") or {}
        return {}
