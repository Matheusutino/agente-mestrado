from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    accuracy: float | None = None
    f1_macro: float | None = None
    precision_macro: float | None = None
    recall_macro: float | None = None


class ReportResult(BaseModel):
    report_path: str


class PipelineResult(BaseModel):
    dataset_path: str
    text_column: str
    label_column: str
    representation: Literal["tfidf", "bow", "sentence_transformer"]
    model: Literal[
        "logistic_regression",
        "linear_svm",
        "multinomial_nb",
        "decision_tree",
        "random_forest",
        "knn",
    ]
    metrics_requested: list[Literal["accuracy", "f1_macro", "precision_macro", "recall_macro"]]
    assumptions: list[str]
    justification: str


class RoundSummary(BaseModel):
    round_index: int
    round_dir: str
    status: Literal["success", "failed"]
    task: str
    result: PipelineResult | None = None
    metrics: EvaluationResult | None = None
    agent_run_id: str | None = None
    agent_conversation_id: str | None = None
    agent_usage: dict[str, Any] | None = None
    representation_config: dict[str, Any] | None = None
    model_parameters: dict[str, Any] | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class RevisionRoundSummary(BaseModel):
    round_index: int
    round_dir: str
    status: Literal["success", "failed"]
    task: str
    result: PipelineResult | None = None
    metrics: EvaluationResult | None = None
    agent_run_id: str | None = None
    agent_conversation_id: str | None = None
    agent_usage: dict[str, Any] | None = None
    representation_config: dict[str, Any] | None = None
    model_parameters: dict[str, Any] | None = None
    error: str | None = None


class RevisionRequest(BaseModel):
    previous_round: RevisionRoundSummary
    prior_rounds: list[RevisionRoundSummary] = Field(default_factory=list)


class OptimizationHistory(BaseModel):
    task: str
    max_rounds: int
    max_minutes: int | None = None
    rounds: list[RoundSummary] = Field(default_factory=list)
    selected_round_index: int | None = None
    selected_round_dir: str | None = None
    final_result: PipelineResult | None = None
    finished_reason: str
