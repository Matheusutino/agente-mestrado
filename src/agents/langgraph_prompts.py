from __future__ import annotations

import json
from pathlib import Path

from src.types import RevisionRequest


def initial_prompt(task: str, output_dir: Path, round_index: int, max_rounds: int) -> str:
    return json.dumps(
        {
            "mode": "initial_round",
            "task": task,
            "run_dir": str(output_dir.resolve()),
            "round_index": round_index,
            "max_rounds": max_rounds,
            "rounds_remaining_including_this_one": max_rounds - round_index + 1,
        },
        ensure_ascii=False,
    )


def revision_prompt(
    task: str,
    output_dir: Path,
    revision: RevisionRequest,
    round_index: int,
    max_rounds: int,
) -> str:
    return json.dumps(
        {
            "mode": "revision_round",
            "task": task,
            "run_dir": str(output_dir.resolve()),
            "round_index": round_index,
            "max_rounds": max_rounds,
            "rounds_remaining_including_this_one": max_rounds - round_index + 1,
            "revision_goal": (
                "Improve over the previous round. Use the prior result, metrics, "
                "configuration, and errors to choose a stronger next pipeline. "
                "Do not repeat the same dataset/representation/model/configuration "
                "unless you have a concrete reason based on the previous outcome."
            ),
            "revision_context": revision.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
