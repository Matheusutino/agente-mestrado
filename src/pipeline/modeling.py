from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from pydantic import TypeAdapter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from src.types import ModelConfig, RepresentationResult, TrainingResult


def _coerce_model_config(config: ModelConfig | dict | str) -> ModelConfig:
    """Normalize model configuration input into a typed config object.

    Args:
        config: Config object, plain dict, or JSON string sent by the LLM.

    Returns:
        A validated discriminated `ModelConfig`.
    """
    adapter = TypeAdapter(ModelConfig)
    if isinstance(config, str):
        return adapter.validate_json(config)
    return adapter.validate_python(config)


def build_model(config: ModelConfig):
    """Build a supported text classification model from a structured config.

    Args:
        config: Typed configuration describing which estimator to instantiate.

    Returns:
        An unfitted scikit-learn compatible estimator.

    Raises:
        ValueError: If the requested model is not supported.
    """
    if config.model == "logistic_regression":
        return LogisticRegression(
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            C=config.c,
        )
    if config.model == "linear_svm":
        return LinearSVC(
            class_weight=config.class_weight,
            C=config.c,
        )
    if config.model == "multinomial_nb":
        return MultinomialNB(alpha=config.alpha)
    if config.model == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            class_weight=config.class_weight,
            random_state=config.random_state,
        )
    if config.model == "knn":
        return KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            metric=config.metric,
        )
    raise ValueError(f"Unsupported model: {config.model}")


def train_classifier(run_dir: str, config: ModelConfig | dict | str) -> TrainingResult:
    """Train a classifier from persisted representation features.

    Args:
        run_dir: Directory containing persisted training features and labels.
        config: Typed config, plain dict, or JSON string describing the model.

    Returns:
        A summary describing the trained model and training set size.
    """
    run_path = Path(run_dir).expanduser().resolve()
    model_config = _coerce_model_config(config)
    representation_metadata = RepresentationResult.model_validate_json(
        (run_path / "representation_metadata.json").read_text(encoding="utf-8")
    )
    if (
        representation_metadata.config.representation == "sentence_transformer"
        and model_config.model == "multinomial_nb"
    ):
        raise ValueError(
            "multinomial_nb is not supported with sentence_transformer features; "
            "use linear_svm, logistic_regression, decision_tree, or knn instead."
        )

    if representation_metadata.feature_storage_format == "sparse_npz":
        X_train = sp.load_npz(str(run_path / "X_train_features.npz"))
    elif representation_metadata.feature_storage_format == "dense_npy":
        X_train = np.load(run_path / "X_train_features.npy")
    else:
        raise ValueError(
            f"Unsupported feature storage format: {representation_metadata.feature_storage_format}"
        )
    split_data = np.load(run_path / "splits.npz", allow_pickle=True)
    y_train = split_data["y_train"].tolist()

    model = build_model(model_config)
    model.fit(X_train, y_train)
    joblib.dump(model, run_path / "model.joblib")
    (run_path / "model_config.json").write_text(
        json.dumps(model_config.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return TrainingResult(
        model_type=model_config.model,
        trained_on_rows=len(y_train),
        config=model_config,
    )
