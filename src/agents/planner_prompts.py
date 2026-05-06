from __future__ import annotations

import json
from pathlib import Path

from src.types import RevisionRequest


def initial_prompt(task: str, output_dir: Path) -> str:
    return json.dumps(
        {
            "mode": "initial_attempt",
            "task": task,
            "run_dir": str(output_dir.resolve()),
        },
        ensure_ascii=False,
    )


def revision_prompt(task: str, output_dir: Path, revision: RevisionRequest) -> str:
    return json.dumps(
        {
            "mode": "revision_attempt",
            "task": task,
            "run_dir": str(output_dir.resolve()),
            "revision_context": revision.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
