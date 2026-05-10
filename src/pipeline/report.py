from __future__ import annotations

import json
from pathlib import Path

from src.types import FIXED_RANDOM_SEED, FIXED_TEST_SIZE, ReportResult


def generate_report(
    run_dir: str,
    task: str,
    representation: str,
    model_type: str,
    assumptions: list[str],
    justification: str,
) -> ReportResult:
    """Generate and save a markdown report from persisted pipeline artifacts.

    Example:
        `generate_report("/abs/path/to/run_dir", "Classify sentiment texts", "tfidf", "logistic_regression", ["The text column is 'text'."], "TF-IDF with logistic regression is a strong baseline for balanced sentiment data.")`
    """
    run_path = Path(run_dir).expanduser().resolve()
    info = json.loads((run_path / "dataset_info.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    metrics_lines = "\n".join(f"- `{name}`: {value:.4f}" for name, value in metrics.items())
    assumptions_lines = "\n".join(f"- {assumption}" for assumption in assumptions) or "- None recorded."

    report = f"""# Text Classification Pipeline Report

## Task
{task}

## Dataset
- Path: `{info['dataset_path']}`
- Text column: `{info['text_column']}`
- Label column: `{info['label_column']}`
- Classes: {info['classes']}
- Train rows: {info['train_rows']} | Test rows: {info['test_rows']}
- test_size: {FIXED_TEST_SIZE} (fixed) | random_seed: {FIXED_RANDOM_SEED} (fixed)

## Choices
- Representation: `{representation}`
- Model: `{model_type}`

## Assumptions
{assumptions_lines}

## Justification
{justification}

## Metrics
{metrics_lines}
"""
    report_path = run_path / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return ReportResult(report_path=str(report_path))
