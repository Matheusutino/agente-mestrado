from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .planner_agent import AgentExecutionRecord
from src.types import (
    AttemptSummary,
    EvaluationResult,
    OptimizationHistory,
    PipelineResult,
    RevisionAttemptSummary,
    RevisionRequest,
)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths(attempt_dir: Path) -> dict[str, str]:
    artifacts = {
        "agent_all_messages": attempt_dir / "agent_all_messages.json",
        "agent_new_messages": attempt_dir / "agent_new_messages.json",
        "agent_events": attempt_dir / "agent_events.json",
        "agent_usage": attempt_dir / "agent_usage.json",
        "agent_run": attempt_dir / "agent_run.json",
        "splits": attempt_dir / "splits.npz",
        "dataset_info": attempt_dir / "dataset_info.json",
        "metrics": attempt_dir / "metrics.json",
        "classification_report": attempt_dir / "classification_report.txt",
        "predictions": attempt_dir / "predictions.csv",
        "confusion_matrix": attempt_dir / "confusion_matrix.csv",
        "representation_metadata": attempt_dir / "representation_metadata.json",
        "representation_config": attempt_dir / "representation_config.json",
        "model_config": attempt_dir / "model_config.json",
        "report": attempt_dir / "report.md",
        "attempt_result": attempt_dir / "result.json",
    }
    return {name: str(path) for name, path in artifacts.items() if path.exists()}


def summarize_attempt(
    task: str,
    attempt_index: int,
    attempt_dir: Path,
    result: PipelineResult | None = None,
    error: str | None = None,
) -> AttemptSummary:
    metrics = None
    metrics_path = attempt_dir / "metrics.json"
    if metrics_path.exists():
        metrics = EvaluationResult.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    agent_run = load_json_if_exists(attempt_dir / "agent_run.json")
    agent_usage = load_json_if_exists(attempt_dir / "agent_usage.json")

    return AttemptSummary(
        attempt_index=attempt_index,
        attempt_dir=str(attempt_dir.resolve()),
        status="success" if error is None and result is not None else "failed",
        task=task,
        result=result,
        metrics=metrics,
        agent_run_id=None if agent_run is None else agent_run.get("run_id"),
        agent_conversation_id=None if agent_run is None else agent_run.get("conversation_id"),
        agent_usage=agent_usage,
        representation_config=load_json_if_exists(attempt_dir / "representation_config.json"),
        model_parameters=load_json_if_exists(attempt_dir / "model_config.json"),
        artifact_paths=artifact_paths(attempt_dir),
        error=error,
    )


def save_attempt_result(attempt_dir: Path, result: PipelineResult) -> None:
    (attempt_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_agent_trace(attempt_dir: Path, record: AgentExecutionRecord) -> None:
    all_messages = json.loads(record.all_messages_json)
    new_messages = json.loads(record.new_messages_json)
    (attempt_dir / "agent_all_messages.json").write_text(
        json.dumps(all_messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (attempt_dir / "agent_new_messages.json").write_text(
        json.dumps(new_messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (attempt_dir / "agent_events.json").write_text(
        json.dumps(record.events, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (attempt_dir / "agent_usage.json").write_text(
        json.dumps(record.usage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (attempt_dir / "agent_run.json").write_text(
        json.dumps(
            {
                "run_id": record.run_id,
                "conversation_id": record.conversation_id,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_revision_request(attempts: list[AttemptSummary]) -> RevisionRequest:
    previous_attempt = RevisionAttemptSummary(
        attempt_index=attempts[-1].attempt_index,
        attempt_dir=attempts[-1].attempt_dir,
        status=attempts[-1].status,
        task=attempts[-1].task,
        result=attempts[-1].result,
        metrics=attempts[-1].metrics,
        agent_run_id=attempts[-1].agent_run_id,
        agent_conversation_id=attempts[-1].agent_conversation_id,
        agent_usage=attempts[-1].agent_usage,
        representation_config=attempts[-1].representation_config,
        model_parameters=attempts[-1].model_parameters,
        error=attempts[-1].error,
    )

    prior_attempts: list[RevisionAttemptSummary] = []
    for attempt in attempts[:-1]:
        prior_attempts.append(
            RevisionAttemptSummary(
                attempt_index=attempt.attempt_index,
                attempt_dir=attempt.attempt_dir,
                status=attempt.status,
                task=attempt.task,
                result=attempt.result,
                metrics=attempt.metrics,
                agent_run_id=attempt.agent_run_id,
                agent_conversation_id=attempt.agent_conversation_id,
                agent_usage=attempt.agent_usage,
                representation_config=attempt.representation_config,
                model_parameters=attempt.model_parameters,
                error=attempt.error,
            )
        )

    return RevisionRequest(
        previous_attempt=previous_attempt,
        prior_attempts=prior_attempts,
    )


def select_best_attempt(attempts: list[AttemptSummary]) -> AttemptSummary | None:
    successful = [
        attempt
        for attempt in attempts
        if attempt.status == "success" and attempt.result is not None and attempt.metrics is not None
    ]
    if not successful:
        return None
    return max(
        successful,
        key=lambda attempt: (
            float("-inf") if attempt.metrics.f1_macro is None else attempt.metrics.f1_macro,
            float("-inf") if attempt.metrics.accuracy is None else attempt.metrics.accuracy,
        ),
    )


def save_history(output_dir: Path, history: OptimizationHistory) -> None:
    (output_dir / "optimization_history.json").write_text(
        json.dumps(history.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
