from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.planner import optimize_pipeline
from src.types import AttemptSummary, OptimizationHistory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fully agentic text classification pipeline."
    )
    parser.add_argument("--task", required=True, help="Task description for the agent.")
    parser.add_argument("--output-root", required=True, help="Directory for run artifacts.")
    parser.add_argument(
        "--llm-provider",
        default="nvidia_nim",
        choices=["nvidia_nim"],
        help="LLM provider backend (kept for compatibility, currently ignored).",
    )
    parser.add_argument("--llm-model", required=True, help="Model name.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of planner attempts.",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=10,
        help="Wall-clock budget for all attempts.",
    )
    parser.add_argument(
        "--quiet-llm",
        action="store_true",
        help="Disable agent event logs in the terminal.",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Enable reasoning/thinking for the agent (low, medium, high). Omit to disable.",
    )
    return parser.parse_args()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _selected_attempt(history: OptimizationHistory) -> AttemptSummary | None:
    if history.selected_attempt_index is None:
        return None
    for attempt in history.attempts:
        if attempt.attempt_index == history.selected_attempt_index:
            return attempt
    return None


def _build_final_report(history: OptimizationHistory) -> str:
    lines = [
        "# Optimization Summary",
        "",
        f"- Task: `{history.task}`",
        f"- Finished reason: `{history.finished_reason}`",
        f"- Max attempts: `{history.max_attempts}`",
        f"- Max minutes: `{history.max_minutes}`",
        "",
        "## Attempts",
    ]
    for attempt in history.attempts:
        lines.append(
            f"- Attempt {attempt.attempt_index:02d}: status=`{attempt.status}` dir=`{attempt.attempt_dir}`"
        )
        if attempt.result is not None:
            lines.append(
                f"  choices: dataset=`{attempt.result.dataset_path}` text=`{attempt.result.text_column}` label=`{attempt.result.label_column}` representation=`{attempt.result.representation}` model=`{attempt.result.model}`"
            )
        if attempt.metrics is not None:
            lines.append(
                "  metrics: "
                f"accuracy={attempt.metrics.accuracy} "
                f"f1_macro={attempt.metrics.f1_macro} "
                f"precision_macro={attempt.metrics.precision_macro} "
                f"recall_macro={attempt.metrics.recall_macro}"
            )
        if attempt.error:
            lines.append(f"  error: {attempt.error}")

    selected = _selected_attempt(history)
    lines.extend(["", "## Final Selection"])
    if selected is None or selected.result is None:
        lines.append("- No successful attempt was selected.")
    else:
        lines.append(f"- Selected attempt: `{selected.attempt_index:02d}`")
        lines.append(
            f"- Dataset: `{selected.result.dataset_path}` | Text: `{selected.result.text_column}` | Label: `{selected.result.label_column}`"
        )
        lines.append(
            f"- Representation: `{selected.result.representation}` | Model: `{selected.result.model}`"
        )
        if selected.metrics is not None:
            lines.append(
                f"- Best metrics: accuracy={selected.metrics.accuracy} f1_macro={selected.metrics.f1_macro}"
            )
        if selected.result.optimization_comment:
            lines.append(f"- Planner note: {selected.result.optimization_comment}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_dotenv()
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    history = optimize_pipeline(
        task=args.task,
        model_name=args.llm_model,
        output_dir=output_root,
        max_attempts=args.max_attempts,
        max_minutes=args.max_minutes,
        verbose=not args.quiet_llm,
        thinking_effort=args.thinking_effort,
    )

    if history is None:
        print("Pipeline failed — no result produced.", file=sys.stderr)
        return 1

    selected = _selected_attempt(history)
    if selected is not None:
        save_json(output_root / "result.json", selected.model_dump(mode="json"))
    final_report = _build_final_report(history)
    (output_root / "final_report.md").write_text(final_report, encoding="utf-8")

    if selected is None:
        print(
            f"Run finished without a successful attempt. See {output_root / 'optimization_history.json'}",
            file=sys.stderr,
        )
        return 1

    print(f"Done. Results saved to {output_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
