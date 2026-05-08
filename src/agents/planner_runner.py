from __future__ import annotations

import sys
import time
from pathlib import Path

from src.types import AttemptSummary, OptimizationHistory

from .planner_agent import build_planner_agent, run_agent
from .planner_history import (
    build_revision_request,
    save_agent_trace,
    save_attempt_result,
    save_history,
    select_best_attempt,
    summarize_attempt,
)
from .planner_prompts import initial_prompt, revision_prompt


def optimize_pipeline(
    task: str,
    model_name: str,
    output_dir: Path,
    llm_provider: str = "nvidia_nim",
    max_attempts: int = 3,
    max_minutes: int | None = 10,
    verbose: bool = False,
    thinking_effort: str | None = None,
) -> OptimizationHistory | None:
    try:
        agent = build_planner_agent(
            model_name=model_name,
            llm_provider=llm_provider,
            thinking_effort=thinking_effort,
        )
    except Exception as exc:
        print(f"Agent init failed: {exc}", file=sys.stderr)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[AttemptSummary] = []
    start_time = time.monotonic()
    finished_reason = "max_attempts_reached"

    for attempt_index in range(1, max_attempts + 1):
        if max_minutes is not None and (time.monotonic() - start_time) >= max_minutes * 60:
            finished_reason = "time_budget_exhausted"
            break

        attempt_dir = output_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        prompt = (
            revision_prompt(task, attempt_dir, build_revision_request(attempts))
            if attempts
            else initial_prompt(task, attempt_dir)
        )

        try:
            execution = run_agent(agent, prompt, verbose=verbose)
            save_agent_trace(attempt_dir, execution)
            save_attempt_result(attempt_dir, execution.result)
            attempts.append(
                summarize_attempt(
                    task=task,
                    attempt_index=attempt_index,
                    attempt_dir=attempt_dir,
                    result=execution.result,
                )
            )
        except Exception as exc:
            print(f"Attempt {attempt_index} failed: {exc}", file=sys.stderr)
            attempts.append(
                summarize_attempt(
                    task=task,
                    attempt_index=attempt_index,
                    attempt_dir=attempt_dir,
                    error=str(exc),
                )
            )

        best_attempt = select_best_attempt(attempts)
        history = OptimizationHistory(
            task=task,
            max_attempts=max_attempts,
            max_minutes=max_minutes,
            attempts=attempts,
            selected_attempt_index=None if best_attempt is None else best_attempt.attempt_index,
            selected_attempt_dir=None if best_attempt is None else best_attempt.attempt_dir,
            final_result=None if best_attempt is None else best_attempt.result,
            finished_reason=finished_reason,
        )
        save_history(output_dir, history)

    best_attempt = select_best_attempt(attempts)
    if best_attempt is None and finished_reason == "max_attempts_reached":
        finished_reason = "all_attempts_failed"

    history = OptimizationHistory(
        task=task,
        max_attempts=max_attempts,
        max_minutes=max_minutes,
        attempts=attempts,
        selected_attempt_index=None if best_attempt is None else best_attempt.attempt_index,
        selected_attempt_dir=None if best_attempt is None else best_attempt.attempt_dir,
        final_result=None if best_attempt is None else best_attempt.result,
        finished_reason=finished_reason,
    )
    save_history(output_dir, history)
    return history
