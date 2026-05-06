from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.types import FIXED_RANDOM_SEED, FIXED_TEST_SIZE, PreprocessingResult


def preprocess_dataset(
    dataset_path: str,
    text_column: str,
    label_column: str,
    run_dir: str,
    test_size: float = FIXED_TEST_SIZE,
    random_state: int = FIXED_RANDOM_SEED,
) -> PreprocessingResult:
    """Load, clean, and split a dataset, then persist split artifacts into run_dir."""
    run_path = Path(run_dir).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    df = pd.read_csv(Path(dataset_path))

    working_df = df[[text_column, label_column]].copy()
    working_df = working_df.dropna(subset=[text_column, label_column])
    working_df[text_column] = working_df[text_column].astype(str).str.strip()
    working_df[label_column] = working_df[label_column].astype(str).str.strip()
    working_df = working_df[
        working_df[text_column].astype(bool) & working_df[label_column].astype(bool)
    ]

    if working_df.empty:
        raise ValueError("No rows remain after removing empty text/label values.")

    label_counts = working_df[label_column].value_counts()
    classes = sorted(label_counts.index.astype(str).tolist())
    class_distribution = {
        str(label): int(count) for label, count in label_counts.items()
    }
    stratify = working_df[label_column] if int(label_counts.min()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        working_df[text_column].tolist(),
        working_df[label_column].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    np.savez_compressed(
        run_path / "splits.npz",
        X_train=np.asarray(X_train, dtype=object),
        X_test=np.asarray(X_test, dtype=object),
        y_train=np.asarray(y_train, dtype=object),
        y_test=np.asarray(y_test, dtype=object),
    )

    train_label_counts = pd.Series(y_train).value_counts()
    test_label_counts = pd.Series(y_test).value_counts()

    result = PreprocessingResult(
        dataset_path=dataset_path,
        text_column=text_column,
        label_column=label_column,
        num_classes=len(classes),
        classes=classes,
        class_distribution=class_distribution,
        train_class_distribution={
            str(label): int(count) for label, count in train_label_counts.items()
        },
        test_class_distribution={
            str(label): int(count) for label, count in test_label_counts.items()
        },
        train_rows=len(X_train),
        test_rows=len(X_test),
        test_size=test_size,
        random_state=random_state,
    )
    (run_path / "dataset_info.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result
