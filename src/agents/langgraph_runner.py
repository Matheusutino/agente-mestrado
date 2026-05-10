from __future__ import annotations

import sys
import time
from pathlib import Path

from src.types import OptimizationHistory, RoundSummary

from .langgraph_agent import run_langgraph_round
from .langgraph_history import (
    build_revision_request,
    save_agent_trace,
    save_round_result,
    save_history,
    select_best_round,
    summarize_round,
)
from .langgraph_prompts import initial_prompt, revision_prompt


def optimize_pipeline_langgraph(
    task: str,
    model_name: str,
    output_dir: Path,
    llm_provider: str = "openrouter",
    max_rounds: int = 3,
    max_minutes: int | None = 10,
    verbose: bool = False,
    thinking_effort: str | None = None,
    max_tool_errors_per_round: int = 3,
) -> OptimizationHistory | None:
    if llm_provider != "openrouter":
        print(
            "LangGraph runtime currently supports only the `openrouter` provider.",
            file=sys.stderr,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[RoundSummary] = []
    start_time = time.monotonic()
    finished_reason = "max_rounds_reached"

    for round_index in range(1, max_rounds + 1):
        if max_minutes is not None and (time.monotonic() - start_time) >= max_minutes * 60:
            finished_reason = "time_budget_exhausted"
            break

        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        prompt = (
            revision_prompt(
                task,
                round_dir,
                build_revision_request(rounds),
                round_index=round_index,
                max_rounds=max_rounds,
            )
            if rounds
            else initial_prompt(
                task,
                round_dir,
                round_index=round_index,
                max_rounds=max_rounds,
            )
        )

        try:
            execution = run_langgraph_round(
                task=task,
                prompt=prompt,
                round_dir=str(round_dir),
                model_name=model_name,
                verbose=verbose,
                thinking_effort=thinking_effort,
                max_tool_errors=max_tool_errors_per_round,
            )
            save_agent_trace(round_dir, execution)
            save_round_result(round_dir, execution.result)
            rounds.append(
                summarize_round(
                    task=task,
                    round_index=round_index,
                    round_dir=round_dir,
                    result=execution.result,
                )
            )
        except Exception as exc:
            print(f"Round {round_index} failed: {exc}", file=sys.stderr)
            rounds.append(
                summarize_round(
                    task=task,
                    round_index=round_index,
                    round_dir=round_dir,
                    error=str(exc),
                )
            )

        best_round = select_best_round(rounds)
        history = OptimizationHistory(
            task=task,
            max_rounds=max_rounds,
            max_minutes=max_minutes,
            rounds=rounds,
            selected_round_index=None if best_round is None else best_round.round_index,
            selected_round_dir=None if best_round is None else best_round.round_dir,
            final_result=None if best_round is None else best_round.result,
            finished_reason=finished_reason,
        )
        save_history(output_dir, history)

    best_round = select_best_round(rounds)
    if best_round is None and finished_reason == "max_rounds_reached":
        finished_reason = "all_rounds_failed"

    history = OptimizationHistory(
        task=task,
        max_rounds=max_rounds,
        max_minutes=max_minutes,
        rounds=rounds,
        selected_round_index=None if best_round is None else best_round.round_index,
        selected_round_dir=None if best_round is None else best_round.round_dir,
        final_result=None if best_round is None else best_round.result,
        finished_reason=finished_reason,
    )
    save_history(output_dir, history)
    return history
