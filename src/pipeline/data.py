from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.types import DatasetProfile, DiscoveredDatasets, ensure_path_string

DEFAULT_DATASETS_DIR = "datasets"


def _class_distribution(series: pd.Series) -> dict[str, int]:
    counts = series.dropna().astype(str).value_counts()
    return {str(label): int(count) for label, count in counts.items()}


def _safe_sample_rows(df: pd.DataFrame, sample_size: int = 5) -> list[dict[str, Any]]:
    sample = df.head(sample_size).copy()
    sample = sample.where(pd.notnull(sample), None)
    return sample.to_dict(orient="records")


def _safe_describe_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty:
        return {}
    summary = df.describe(include="all").transpose().copy()
    summary = summary.where(pd.notnull(summary), None)
    return {
        str(column): {
            str(metric): value.item() if hasattr(value, "item") else value
            for metric, value in metrics.items()
        }
        for column, metrics in summary.to_dict(orient="index").items()
    }


def discover_datasets(path: str = DEFAULT_DATASETS_DIR) -> DiscoveredDatasets:
    """Discover CSV datasets under a directory.

    Args:
        path: Directory path searched recursively for `.csv` files.

    Returns:
        A structured list of absolute dataset paths.

    Raises:
        FileNotFoundError: If the discovery path does not exist or no CSV files are found.
        ValueError: If the discovery path is not a directory.
    """
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Discovery path not found: {target}")
    if not target.is_dir():
        raise ValueError(f"Discovery path must be a directory: {target}")

    datasets = sorted(str(item.resolve()) for item in target.rglob("*.csv"))
    if not datasets:
        raise FileNotFoundError(f"No CSV files were found under: {target}")

    return DiscoveredDatasets(dataset_paths=datasets)


def dataset_profile(dataset_path: str | Path, sample_size: int = 5) -> DatasetProfile:
    """Profile a dataset for downstream LLM planning.

    Args:
        dataset_path: Path to the CSV dataset.
        sample_size: Number of head rows to include in the profile sample.

    Returns:
        A structured dataset profile.
    """
    path = Path(dataset_path).expanduser().resolve()
    df = pd.read_csv(path)

    column_types = {column: str(dtype) for column, dtype in df.dtypes.items()}
    missing_values = {column: int(df[column].isna().sum()) for column in df.columns}

    class_distributions: dict[str, dict[str, int]] = {}
    notes: list[str] = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        if 2 <= unique_count <= 50:
            class_distributions[str(column)] = _class_distribution(series)

    if df.empty:
        notes.append("Dataset is empty.")
    notes.append(
        "Text and label candidates were not inferred in profiling; the LLM must decide from columns, types, samples, and class distributions."
    )
    if not class_distributions:
        notes.append("No low-cardinality columns were found for class distribution summaries.")

    return DatasetProfile(
        dataset_path=ensure_path_string(path),
        num_rows=int(len(df)),
        num_columns=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
        column_types=column_types,
        missing_values=missing_values,
        describe_summary=_safe_describe_summary(df),
        sample_rows=_safe_sample_rows(df, sample_size=sample_size),
        class_distributions=class_distributions,
        notes=notes,
    )
