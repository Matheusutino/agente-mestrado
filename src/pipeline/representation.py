from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.sparse as sp
from pydantic import TypeAdapter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.types import (
    DenseRepresentationConfig,
    RepresentationConfig,
    RepresentationResult,
    SparseRepresentationConfig,
)


def _build_sparse_representation(config: SparseRepresentationConfig):
    common_kwargs = {
        "max_features": config.max_features,
        "lowercase": config.lowercase,
        "ngram_range": (config.ngram_min, config.ngram_max),
        "min_df": config.min_df,
        "max_df": config.max_df,
    }

    if config.representation == "bow":
        return CountVectorizer(
            **common_kwargs,
            binary=config.binary,
        )
    return TfidfVectorizer(
        **common_kwargs,
        binary=config.binary,
        use_idf=config.use_idf,
        sublinear_tf=config.sublinear_tf,
    )


def _build_dense_representation(config: DenseRepresentationConfig):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for dense representations."
        ) from exc
    return SentenceTransformer(config.model_name, device=config.device)


def build_representation_model(config: RepresentationConfig):
    """Build a text representation object from a structured configuration."""
    if isinstance(config, SparseRepresentationConfig):
        return _build_sparse_representation(config)
    if isinstance(config, DenseRepresentationConfig):
        return _build_dense_representation(config)
    raise ValueError(f"Unsupported representation config: {type(config)!r}")


def _coerce_representation_config(
    config: RepresentationConfig | dict[str, Any] | str,
) -> RepresentationConfig:
    adapter = TypeAdapter(RepresentationConfig)
    if isinstance(config, str):
        config = json.loads(config)
    return adapter.validate_python(config)


def build_representation(
    run_dir: str,
    config: RepresentationConfig | dict[str, Any] | str,
) -> RepresentationResult:
    """Build and persist a text representation for train and test splits.

    The representation settings are passed as one structured config so the
    caller chooses within a coherent family of parameters instead of a large
    flat argument list. JSON-string configs are also accepted defensively
    because some models serialize nested tool arguments that way.
    """
    run_path = Path(run_dir).expanduser().resolve()
    split_data = np.load(run_path / "splits.npz", allow_pickle=True)
    X_train = split_data["X_train"].tolist()
    X_test = split_data["X_test"].tolist()

    config = _coerce_representation_config(config)
    representation_model = build_representation_model(config)
    if isinstance(config, DenseRepresentationConfig):
        X_train_features = representation_model.encode(
            X_train,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        X_test_features = representation_model.encode(
            X_test,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if config.truncate_dim is not None:
            X_train_features = X_train_features[:, : config.truncate_dim]
            X_test_features = X_test_features[:, : config.truncate_dim]

        try:
            joblib.dump(representation_model, run_path / "representation.joblib")
        except Exception:
            pass

        np.save(run_path / "X_train_features.npy", X_train_features)
        np.save(run_path / "X_test_features.npy", X_test_features)
        result = RepresentationResult(
            representation=config.representation,
            vocab_size=None,
            feature_dim=int(X_train_features.shape[1]),
            feature_storage_format="dense_npy",
            config=config,
        )
    else:
        X_train_features = representation_model.fit_transform(X_train)
        X_test_features = representation_model.transform(X_test)

        joblib.dump(representation_model, run_path / "representation.joblib")
        sp.save_npz(str(run_path / "X_train_features.npz"), X_train_features)
        sp.save_npz(str(run_path / "X_test_features.npz"), X_test_features)
        result = RepresentationResult(
            representation=config.representation,
            vocab_size=len(representation_model.vocabulary_),
            feature_dim=int(X_train_features.shape[1]),
            feature_storage_format="sparse_npz",
            config=config,
        )

    (run_path / "representation_config.json").write_text(
        json.dumps(result.config.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_path / "representation_metadata.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result
