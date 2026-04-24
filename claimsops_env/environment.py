from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from claimsops_env.generator import EpisodeSpec, ScenarioGenerator
from claimsops_env.models import (
    ActionRecord,
    ActivityStatus,
    AppraisalStatus,
    ClaimDocument,
    DocumentType,
    EventRecord,
    Evidence,
    EvidenceKind,
    FinancialSnapshot,
    Observation,
    PlatformState,
    RewardBreakdown,
    RubricEvaluation,
    StepResult,
    ToolName,
    WorkflowState,
)
from claimsops_env.tools import RuntimeView, ToolError, build_tool_registry, validate_action
from claimsops_env.verifier import RewardContext, evaluate_context_rubric, score_episode


class ClaimsOpsEnv:
    """Stateful, tool-based synthetic auto claims environment."""

    def __init__(self, generator: ScenarioGenerator | None = None) -> None:
        self.generator = generator or ScenarioGenerator()
        self.tools = build_tool_registry()
        self._spec: EpisodeSpec | None = None
        self._visible_policy: dict[str, Any] | None = None
        self._evidence = []
        self._platform_state = PlatformState()
        self._diary: list[str] = []
        self._communications: list[str] = []
        self._financial = FinancialSnapshot()
        self._workflow = WorkflowState()
        self._latest_tool_result: dict[str, Any] | None = None
        self._final_decision = None
        self._remaining_steps = 0
        self._action_log: list[ActionRecord] = []
        self._violations: list[str] = []
        self._invalid_action_streak = 0
        self._last_reward = RewardBreakdown()
        self._last_rubric_evaluation: RubricEvaluation | None = None

    def reset(self, seed: int | None = None, scenario_family: str | None = None) -> Observation:
        self._spec = (
            self.generator.generate_family(scenario_family, seed)
            if scenario_family
            else self.generator.generate(seed)
        )
        self._visible_policy = None
        self._evidence = deepcopy(self._spec.claim.initial_evidence)
        self._platform_state = deepcopy(self._spec.platform_state)
        self._diary = ["Claim file opened."]
        self._communications = []
        self._financial = FinancialSnapshot()
        self._workflow = WorkflowState()
        self._latest_tool_result = None
        self._final_decision = None
        self._remaining_steps = self._spec.claim.step_budget
        self._action_log = []
        self._violations = []
        self._invalid_action_streak = 0
        self._last_reward = RewardBreakdown()
        self._last_rubric_evaluation = None
        return self._observation()

    def step(self, raw_action: dict[str, Any] | str) -> StepResult:
        self._require_reset()
        if self._workflow.final_decision_submitted:
            return StepResult(
                observation=self._observation(),
                reward=self._last_reward.total,
                done=True,
                info={"error": "episode already done", "reward_breakdown": self._last_reward.model_dump()},
            )

        valid_format = True
        result_summary = ""
        terminal = False
        parsed_dict: dict[str, Any] | None = None
        try:
            parsed_dict = json.loads(raw_action) if isinstance(raw_action, str) else raw_action
            action = validate_action(parsed_dict)
            if _contains_hidden_probe(action.args):
                self._violations.append("hidden_state_access")
                raise ToolError("attempted hidden-state access")
            runtime = self._runtime()
            tool = self.tools[action.tool]
            tool_result = tool.run(runtime, action.args)
            self._apply_runtime(runtime)
            self._latest_tool_result = tool_result.model_dump(mode="json")
            result_summary = tool_result.summary
            terminal = tool_result.terminal
            self._invalid_action_streak = 0
        except (json.JSONDecodeError, ValidationError, ToolError, KeyError, TypeError, ValueError) as exc:
            valid_format = False
            self._invalid_action_streak += 1
            if "unknown evidence_ids" in str(exc):
                self._violations.append("fabricated_document_id")
            self._violations.append("tool_error")
            result_summary = f"Invalid action: {exc}"
            self._latest_tool_result = {"ok": False, "summary": result_summary, "data": {}}

        if self._invalid_action_streak >= 3:
            self._violations.append("invalid_action_loop")
            terminal = True

        self._remaining_steps = max(0, self._remaining_steps - 1)
        if self._remaining_steps == 0:
            terminal = True
        self._advance_activities()
        self._advance_events()

        self._action_log.append(
            ActionRecord(
                step=len(self._action_log) + 1,
                action=parsed_dict if isinstance(parsed_dict, dict) else {"raw": raw_action},
                valid=valid_format,
                result_summary=result_summary,
            )
        )
        context = self._reward_context(valid_format=valid_format)
        self._last_rubric_evaluation = evaluate_context_rubric(context)
        context = replace(context, rubric_evaluation=self._last_rubric_evaluation)
        self._last_reward = score_episode(context)
        if not valid_format:
            self._last_reward.total = min(self._last_reward.total, -0.2)
        done = terminal or self._workflow.final_decision_submitted
        return StepResult(
            observation=self._observation(),
            reward=self._last_reward.total,
            done=done,
            info={
                "reward_breakdown": self._last_reward.model_dump(),
                "rubric_evaluation": self._last_rubric_evaluation.model_dump(mode="json") if self._last_rubric_evaluation else {},
                "action_valid": valid_format,
                "violations": list(dict.fromkeys(self._violations)),
            },
        )

    def state(self, include_action_log: bool = True) -> dict[str, Any]:
        self._require_reset()
        state = self._observation().model_dump(mode="json")
        state["workflow"] = self._workflow.model_dump(mode="json")
        state["last_reward"] = self._last_reward.model_dump()
        if include_action_log:
            state["action_log"] = [record.model_dump(mode="json") for record in self._action_log]
        return state

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "claimsops-gym",
            "version": "0.1.0",
            "action_format": {"tool": "tool_name", "args": {}},
            "available_tools": [tool.value for tool in ToolName],
            "reward_columns": list(RewardBreakdown.model_fields.keys()),
        }

    def _runtime(self) -> RuntimeView:
        self._require_reset()
        return RuntimeView(
            spec=self._spec,
            visible_policy=deepcopy(self._visible_policy),
            evidence=deepcopy(self._evidence),
            platform_state=deepcopy(self._platform_state),
            diary=deepcopy(self._diary),
            communications=deepcopy(self._communications),
            financial_snapshot=deepcopy(self._financial),
            workflow=deepcopy(self._workflow),
            final_decision=deepcopy(self._final_decision),
            violations=deepcopy(self._violations),
        )

    def _advance_activities(self) -> None:
        for activity in self._platform_state.activities:
            if _status_value(activity.status) == "open":
                activity.due_in_steps = max(0, activity.due_in_steps - 1)
                if activity.due_in_steps == 0:
                    activity.status = ActivityStatus.OVERDUE

    def _advance_events(self) -> None:
        resolved_ids: set[str] = set()
        for event in self._platform_state.pending_events:
            event.due_in_steps = max(0, event.due_in_steps - 1)
            if event.due_in_steps > 0:
                continue
            self._resolve_event(event.event_type, event.event_id, event.summary, event.payload)
            resolved_ids.add(event.event_id)
        if resolved_ids:
            self._platform_state.pending_events = [
                event for event in self._platform_state.pending_events if event.event_id not in resolved_ids
            ]

    def _resolve_event(self, event_type: str, event_id: str, summary: str, payload: dict[str, Any]) -> None:
        record_summary = summary
        if event_type == "document_arrival":
            doc_type = str(payload.get("doc_type", "requested_document"))
            evidence_id = str(payload.get("evidence_id") or f"EV-DOC-{doc_type.upper()}")
            try:
                document = DocumentType(doc_type)
            except ValueError:
                document = None
            if document and document not in self._workflow.documents_received:
                self._workflow.documents_received.append(document)
            for claim_document in self._platform_state.documents:
                if claim_document.doc_type == document and claim_document.status == "requested":
                    claim_document.status = "received"
                    claim_document.evidence_id = evidence_id
                    claim_document.confidence = "high" if document == DocumentType.POLICE_REPORT else "medium"
                    claim_document.summary = f"Requested {doc_type} received and indexed."
                    if document == DocumentType.POLICE_REPORT and self._spec.hidden.subrogation_expected:
                        claim_document.issues.append("third_party_fault")
                    break
            if not any(evidence.evidence_id == evidence_id for evidence in self._evidence):
                flags = [doc_type]
                if document in self._spec.hidden.required_documents:
                    flags.append("document_received")
                if document == DocumentType.POLICE_REPORT and self._spec.hidden.subrogation_expected:
                    flags.append("third_party_fault")
                kind = EvidenceKind.POLICE_REPORT if document == DocumentType.POLICE_REPORT else EvidenceKind.REQUESTED_DOCUMENT
                self._evidence.append(
                    Evidence(
                        evidence_id=evidence_id,
                        kind=kind,
                        summary=f"Requested {doc_type} received and added to the claim file.",
                        flags=flags,
                    )
                )
            record_summary = f"Document received: {doc_type}."
        elif event_type == "appraisal_complete":
            vendor_id = str(payload.get("vendor_id", ""))
            method = str(payload.get("method", "vendor"))
            for assignment in self._platform_state.vendor_assignments:
                if assignment.vendor_id == vendor_id or not vendor_id:
                    assignment.status = "completed"
                    assignment.eta_steps = 0
                    break
            evidence_id = f"EV-APPRAISAL-{len(self._platform_state.event_history) + 1:03d}"
            if not any(evidence.evidence_id == evidence_id for evidence in self._evidence):
                self._evidence.append(
                    Evidence(
                        evidence_id=evidence_id,
                        kind=EvidenceKind.VENDOR_REPORT,
                        summary=f"{method.title()} appraisal vendor completed damage review.",
                        flags=["appraisal_complete", method],
                    )
                )
            record_summary = f"Appraisal completed: {method}."
        elif event_type == "supplement_received":
            evidence_id = str(payload.get("evidence_id", "EV-SUPPLEMENT-RECEIVED"))
            if not any(evidence.evidence_id == evidence_id for evidence in self._evidence):
                self._evidence.append(
                    Evidence(
                        evidence_id=evidence_id,
                        kind=EvidenceKind.REQUESTED_DOCUMENT,
                        summary="Repair facility returned a supplement/corrected estimate for review.",
                        flags=["supplement_received"],
                    )
                )
            self._platform_state.documents.append(
                _claim_document(
                    document_id="DOC-SUPPLEMENT",
                    doc_type=DocumentType.REPAIR_ESTIMATE_BREAKDOWN,
                    title="Repair facility supplement",
                    source="vendor",
                    status="received",
                    evidence_id=evidence_id,
                    confidence="medium",
                    summary="Supplement/corrected estimate received from repair facility.",
                    issues=["supplement_received"],
                    related_object_id=self._spec.claim.estimate_id,
                )
            )
            record_summary = "Supplement received from repair facility."
        elif event_type == "valuation_complete":
            actual_cash_value = float(payload.get("actual_cash_value", 0.0))
            self._platform_state.valuation_received = True
            self._workflow.valuation_seen = True
            self._platform_state.appraisal_status = AppraisalStatus.VALUATION_RECEIVED
            if not any(evidence.evidence_id == "EV-VALUATION" for evidence in self._evidence):
                self._evidence.append(
                    Evidence(
                        evidence_id="EV-VALUATION",
                        kind=EvidenceKind.REQUESTED_DOCUMENT,
                        summary=f"Total-loss valuation report estimates ACV at ${actual_cash_value:,.2f}.",
                        amount=round(actual_cash_value, 2),
                        flags=["valuation"],
                    )
                )
            self._platform_state.documents.append(
                _claim_document(
                    document_id="DOC-VALUATION",
                    doc_type=DocumentType.REPAIR_ESTIMATE_BREAKDOWN,
                    title="Total-loss valuation report",
                    source="vendor",
                    status="received",
                    evidence_id="EV-VALUATION",
                    confidence="high",
                    summary="Valuation report received from total-loss vendor.",
                    issues=["valuation"],
                    related_object_id=self._spec.claim.estimate_id,
                )
            )
            record_summary = "Total-loss valuation report received."
        elif event_type == "authority_decision":
            approved = bool(payload.get("approved", True))
            self._platform_state.authority_approved = approved
            self._workflow.authority_approved = approved
            if approved:
                _complete_activity_by_category(self._platform_state, "authority", "authority approval received")
            record_summary = "Authority approval received." if approved else "Authority approval declined."
        elif event_type == "claimant_response":
            for party in self._platform_state.parties:
                if party.role == "claimant":
                    party.contact_status = "reachable"
                    break
            record_summary = "Claimant response received."
        elif event_type == "rental_day_accrual":
            self._platform_state.rental_days += 1
            record_summary = f"Rental day accrued; open rental days={self._platform_state.rental_days}."
        elif event_type == "storage_fee_accrual":
            self._platform_state.storage_charges = round(self._platform_state.storage_charges + 75.0, 2)
            record_summary = f"Storage fee accrued; total storage=${self._platform_state.storage_charges:,.2f}."
        self._platform_state.event_history.append(
            EventRecord(event_id=event_id, event_type=event_type, summary=record_summary, payload=payload)
        )

    def _apply_runtime(self, runtime: RuntimeView) -> None:
        self._visible_policy = runtime.visible_policy
        self._evidence = runtime.evidence
        self._platform_state = runtime.platform_state
        self._diary = runtime.diary
        self._communications = runtime.communications
        self._financial = runtime.financial_snapshot
        self._workflow = runtime.workflow
        self._final_decision = runtime.final_decision
        self._violations = runtime.violations

    def _observation(self) -> Observation:
        self._require_reset()
        claim = self._spec.claim
        return Observation(
            claim_id=claim.claim_id,
            policy_id=claim.policy_id,
            customer_id=claim.customer_id,
            vehicle_id=claim.vehicle_id,
            estimate_id=claim.estimate_id,
            line_of_business=claim.line_of_business,
            claim_type=claim.claim_type,
            loss_date=claim.loss_date,
            reported_date=claim.reported_date,
            claimant_statement=claim.claimant_statement,
            requested_amount=claim.requested_amount,
            latest_tool_result=self._latest_tool_result,
            visible_policy=self._visible_policy,
            available_evidence=self._evidence,
            parties=self._platform_state.parties,
            incidents=self._platform_state.incidents,
            exposures=self._platform_state.exposures,
            activities=self._platform_state.activities,
            reserve_lines=self._platform_state.reserve_lines,
            payments=self._platform_state.payments,
            vendor_assignments=self._platform_state.vendor_assignments,
            claim_notes=self._platform_state.notes,
            claim_documents=self._platform_state.documents,
            pending_events=self._platform_state.pending_events,
            event_history=self._platform_state.event_history,
            rental_days=self._platform_state.rental_days,
            storage_charges=self._platform_state.storage_charges,
            appraisal_status=self._platform_state.appraisal_status,
            coverage_result=self._platform_state.coverage_result,
            alerts=self._alerts(),
            audit_gaps=self._audit_gaps(),
            claim_diary=self._diary,
            financial_snapshot=self._financial,
            communications_sent=self._communications,
            open_tasks=self._open_tasks(),
            available_tools=list(ToolName),
            remaining_steps=self._remaining_steps,
        )

    def _open_tasks(self) -> list[str]:
        self._require_reset()
        tasks: list[str] = []
        if not self._workflow.policy_seen:
            tasks.append("verify_policy")
        if not self._platform_state.coverage_verified:
            tasks.append("verify_coverage")
        if not self._workflow.appraisal_assigned:
            tasks.append("assign_appraisal")
        if not self._workflow.estimate_seen:
            tasks.append("inspect_estimate")
        if self._workflow.estimate_seen and not self._workflow.estimate_reviewed:
            tasks.append("review_estimate")
        if "near_total_loss_threshold" in self._alerts() and not self._platform_state.valuation_requested:
            tasks.append("request_valuation")
        visible_doc_gaps = self._visible_document_gaps()
        received = {_doc_value(doc) for doc in self._workflow.documents_received}
        requested = {_doc_value(doc) for doc in self._workflow.documents_requested}
        tasks.extend(f"request_{doc}" for doc in sorted(visible_doc_gaps - received - requested))
        if self._platform_state.pending_events:
            tasks.append("await_pending_events")
        if not self._workflow.fraud_checked:
            tasks.append("screen_fraud_indicators")
        if self._visible_subrogation_signal() and not self._workflow.subrogation_opened:
            tasks.append("evaluate_subrogation")
        if self._financial.reserve_amount is None:
            tasks.append("set_reserve")
        authority_limit = self._visible_policy.get("authority_limit") if self._visible_policy else None
        if authority_limit and self._spec.claim.requested_amount > authority_limit and not self._workflow.authority_requested:
            tasks.append("request_authority_approval")
        if not self._workflow.claimant_updated:
            tasks.append("send_claimant_update")
        if not self._workflow.closure_note_added:
            tasks.append("add_closure_note")
        if not self._workflow.final_decision_submitted:
            tasks.append("submit_final_decision")
        return tasks

    def _alerts(self) -> list[str]:
        self._require_reset()
        alerts: list[str] = []
        if self._visible_document_gaps():
            alerts.append("visible_document_gap")
        if self._visible_subrogation_signal():
            alerts.append("possible_subrogation")
        if self._spec.repair_estimate.gross_amount >= (self._spec.repair_estimate.total_loss_threshold or float("inf")):
            alerts.append("near_total_loss_threshold")
        if self._spec.repair_estimate.duplicate_line_amount > 0:
            alerts.append("estimate_duplicate_line")
        if self._spec.repair_estimate.unrelated_damage_amount > 0:
            alerts.append("possible_prior_damage")
        authority_limit = self._visible_policy.get("authority_limit") if self._visible_policy else None
        if authority_limit and self._spec.claim.requested_amount > authority_limit:
            alerts.append("authority_threshold_risk")
        overdue = [activity.activity_id for activity in self._platform_state.activities if _status_value(activity.status) in {"open", "overdue"} and activity.due_in_steps <= 0]
        if overdue:
            alerts.append("activity_overdue")
        if self._platform_state.pending_events:
            alerts.append("pending_external_event")
        if self._platform_state.rental_days >= 2:
            alerts.append("rental_leakage_risk")
        if self._platform_state.storage_charges >= 120:
            alerts.append("storage_fee_leakage_risk")
        return alerts

    def _audit_gaps(self) -> list[str]:
        self._require_reset()
        gaps: list[str] = []
        if not self._platform_state.coverage_verified:
            gaps.append("coverage_not_verified")
        if not self._workflow.estimate_reviewed:
            gaps.append("estimate_not_reviewed")
        if not self._platform_state.reserve_lines:
            gaps.append("reserve_not_set")
        if not self._workflow.claimant_updated:
            gaps.append("claimant_not_updated")
        if not any(note.note_type.value == "closure" for note in self._platform_state.notes):
            gaps.append("closure_note_missing")
        if any(document.status == "received" and document.issues for document in self._platform_state.documents):
            gaps.append("document_review_needed")
        if self._platform_state.pending_events:
            gaps.append("pending_external_events")
        open_activities = [activity.activity_id for activity in self._platform_state.activities if _status_value(activity.status) in {"open", "overdue"}]
        if open_activities:
            gaps.append("open_activities")
        return gaps

    def _visible_document_gaps(self) -> set[str]:
        self._require_reset()
        text = " ".join(
            [
                self._spec.claim.claimant_statement,
                *(evidence.summary for evidence in self._evidence),
                *(flag for evidence in self._evidence for flag in evidence.flags),
                *(issue for document in self._platform_state.documents for issue in document.issues),
            ]
        ).lower()
        gaps: set[str] = set()
        if "police" in text and any(term in text for term in ["pending", "report", "third party", "fault"]):
            gaps.add(DocumentType.POLICE_REPORT.value)
        if "breakdown" in text or "duplicate" in text:
            gaps.add(DocumentType.REPAIR_ESTIMATE_BREAKDOWN.value)
        if "ownership" in text or "owner" in text:
            gaps.add(DocumentType.PROOF_OF_OWNERSHIP.value)
        if "incomplete" in text or "statement conflict" in text:
            gaps.add(DocumentType.CLAIMANT_STATEMENT.value)
        return gaps

    def _visible_subrogation_signal(self) -> bool:
        self._require_reset()
        text = " ".join(
            [
                self._spec.claim.claimant_statement,
                *(evidence.summary for evidence in self._evidence),
                *(flag for evidence in self._evidence for flag in evidence.flags),
            ]
        ).lower()
        return any(
            signal in text
            for signal in [
                "third_party_fault",
                "third party",
                "third-party",
                "another driver",
                "rear-ended",
                "admitted fault",
                "likely at fault",
            ]
        )

    def _reward_context(self, valid_format: bool) -> RewardContext:
        self._require_reset()
        return RewardContext(
            hidden=self._spec.hidden,
            workflow=self._workflow,
            final_decision=self._final_decision,
            approved_payment=self._financial.approved_payment,
            reserve_amount=self._financial.reserve_amount,
            platform_state=self._platform_state,
            repair_estimate=self._spec.repair_estimate,
            rubric=self._spec.rubric,
            rubric_evaluation=None,
            action_log=self._action_log,
            evidence_ids={evidence.evidence_id for evidence in self._evidence},
            requested_documents=list(self._workflow.documents_requested),
            received_documents=list(self._workflow.documents_received),
            remaining_steps=self._remaining_steps,
            step_budget=self._spec.claim.step_budget,
            violations=list(dict.fromkeys(self._violations)),
            valid_format=valid_format,
        )

    def _score(self, valid_format: bool) -> RewardBreakdown:
        return score_episode(self._reward_context(valid_format))

    def _require_reset(self) -> None:
        if self._spec is None:
            raise RuntimeError("call reset() before using the environment")


def _contains_hidden_probe(args: dict[str, Any]) -> bool:
    text = json.dumps(args, sort_keys=True).lower()
    probes = ["hidden", "truth", "expected_outcome", "expected_payable", "verifier", "answer_key"]
    return any(probe in text for probe in probes)


def _doc_value(doc: DocumentType | str) -> str:
    return doc.value if isinstance(doc, DocumentType) else str(doc)


def _status_value(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _complete_activity_by_category(platform_state: PlatformState, category: str, reason: str) -> None:
    for activity in platform_state.activities:
        if activity.category == category and _status_value(activity.status) in {"open", "overdue"}:
            activity.status = ActivityStatus.COMPLETED
            activity.close_reason = reason
            return


def _claim_document(
    *,
    document_id: str,
    doc_type: DocumentType,
    title: str,
    source: str,
    status: str,
    evidence_id: str,
    confidence: str,
    summary: str,
    issues: list[str],
    related_object_id: str,
) -> ClaimDocument:
    return ClaimDocument(
        document_id=document_id,
        doc_type=doc_type,
        title=title,
        source=source,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        evidence_id=evidence_id,
        confidence=confidence,  # type: ignore[arg-type]
        summary=summary,
        issues=issues,
        related_object_id=related_object_id,
    )
