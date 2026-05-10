from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.types import EvaluationResult, RepresentationResult


def evaluate_classifier(run_dir: str, metrics: list[str]) -> EvaluationResult:
    """Evaluate the persisted model on the persisted test split and save artifacts.

    Example:
        `evaluate_classifier("/abs/path/to/run_dir", ["accuracy", "f1_macro", "precision_macro", "recall_macro"])`
    """
    run_path = Path(run_dir).expanduser().resolve()
    model = joblib.load(run_path / "model.joblib")
    representation_metadata = RepresentationResult.model_validate_json(
        (run_path / "representation_metadata.json").read_text(encoding="utf-8")
    )
    if representation_metadata.feature_storage_format == "sparse_npz":
        X_test = sp.load_npz(str(run_path / "X_test_features.npz"))
    elif representation_metadata.feature_storage_format == "dense_npy":
        X_test = np.load(run_path / "X_test_features.npy")
    else:
        raise ValueError(
            f"Unsupported feature storage format: {representation_metadata.feature_storage_format}"
        )
    split_data = np.load(run_path / "splits.npz", allow_pickle=True)
    y_test = split_data["y_test"].tolist()
    X_test_raw = split_data["X_test"].tolist()
    y_pred = model.predict(X_test).tolist()

    metric_values: dict[str, float] = {}
    if "accuracy" in metrics:
        metric_values["accuracy"] = float(accuracy_score(y_test, y_pred))
    if "f1_macro" in metrics:
        metric_values["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
    if "precision_macro" in metrics:
        metric_values["precision_macro"] = float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
    if "recall_macro" in metrics:
        metric_values["recall_macro"] = float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )

    labels = sorted(set(y_test) | set(y_pred))
    report_text = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    pd.DataFrame({"text": X_test_raw, "y_true": y_test, "y_pred": y_pred}).to_csv(
        run_path / "predictions.csv",
        index=False,
    )
    (run_path / "metrics.json").write_text(
        json.dumps(metric_values, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_path / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        run_path / "confusion_matrix.csv"
    )
    return EvaluationResult(**metric_values)
