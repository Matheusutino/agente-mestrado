from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from src.types import (
    EvaluationResult,
    OptimizationHistory,
    PipelineResult,
    RevisionRequest,
    RevisionRoundSummary,
    RoundSummary,
)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths(round_dir: Path) -> dict[str, str]:
    artifacts = {
        "agent_trace": round_dir / "agent_trace.json",
        "splits": round_dir / "splits.npz",
        "dataset_info": round_dir / "dataset_info.json",
        "metrics": round_dir / "metrics.json",
        "classification_report": round_dir / "classification_report.txt",
        "predictions": round_dir / "predictions.csv",
        "confusion_matrix": round_dir / "confusion_matrix.csv",
        "representation_metadata": round_dir / "representation_metadata.json",
        "representation_config": round_dir / "representation_config.json",
        "model_config": round_dir / "model_config.json",
        "report": round_dir / "report.md",
        "round_result": round_dir / "result.json",
    }
    return {name: str(path) for name, path in artifacts.items() if path.exists()}


def summarize_round(
    task: str,
    round_index: int,
    round_dir: Path,
    result: PipelineResult | None = None,
    error: str | None = None,
) -> RoundSummary:
    metrics = None
    metrics_path = round_dir / "metrics.json"
    if metrics_path.exists():
        metrics = EvaluationResult.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    agent_run = load_json_if_exists(round_dir / "agent_run.json")
    agent_usage = load_json_if_exists(round_dir / "agent_usage.json")

    return RoundSummary(
        round_index=round_index,
        round_dir=str(round_dir.resolve()),
        status="success" if error is None and result is not None else "failed",
        task=task,
        result=result,
        metrics=metrics,
        agent_run_id=None if agent_run is None else agent_run.get("run_id"),
        agent_conversation_id=None if agent_run is None else agent_run.get("conversation_id"),
        agent_usage=agent_usage,
        representation_config=load_json_if_exists(round_dir / "representation_config.json"),
        model_parameters=load_json_if_exists(round_dir / "model_config.json"),
        artifact_paths=artifact_paths(round_dir),
        error=error,
    )


def save_round_result(round_dir: Path, result: PipelineResult) -> None:
    (round_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_agent_trace(round_dir: Path, record: Any) -> None:
    (round_dir / "agent_trace.json").write_text(
        json.dumps(record.events, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_revision_request(rounds: list[RoundSummary]) -> RevisionRequest:
    previous_round = RevisionRoundSummary(
        round_index=rounds[-1].round_index,
        round_dir=rounds[-1].round_dir,
        status=rounds[-1].status,
        task=rounds[-1].task,
        result=rounds[-1].result,
        metrics=rounds[-1].metrics,
        agent_run_id=rounds[-1].agent_run_id,
        agent_conversation_id=rounds[-1].agent_conversation_id,
        agent_usage=rounds[-1].agent_usage,
        representation_config=rounds[-1].representation_config,
        model_parameters=rounds[-1].model_parameters,
        error=rounds[-1].error,
    )

    prior_rounds: list[RevisionRoundSummary] = []
    for round_summary in rounds[:-1]:
        prior_rounds.append(
            RevisionRoundSummary(
                round_index=round_summary.round_index,
                round_dir=round_summary.round_dir,
                status=round_summary.status,
                task=round_summary.task,
                result=round_summary.result,
                metrics=round_summary.metrics,
                agent_run_id=round_summary.agent_run_id,
                agent_conversation_id=round_summary.agent_conversation_id,
                agent_usage=round_summary.agent_usage,
                representation_config=round_summary.representation_config,
                model_parameters=round_summary.model_parameters,
                error=round_summary.error,
            )
        )

    return RevisionRequest(
        previous_round=previous_round,
        prior_rounds=prior_rounds,
    )


def select_best_round(rounds: list[RoundSummary]) -> RoundSummary | None:
    successful = [
        round_summary
        for round_summary in rounds
        if round_summary.status == "success"
        and round_summary.result is not None
        and round_summary.metrics is not None
    ]
    if not successful:
        return None
    return max(
        successful,
        key=lambda round_summary: (
            float("-inf")
            if round_summary.metrics.f1_macro is None
            else round_summary.metrics.f1_macro,
            float("-inf")
            if round_summary.metrics.accuracy is None
            else round_summary.metrics.accuracy,
        ),
    )


def save_history(output_dir: Path, history: OptimizationHistory) -> None:
    (output_dir / "optimization_history.json").write_text(
        json.dumps(history.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
