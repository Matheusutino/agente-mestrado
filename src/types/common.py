from __future__ import annotations

from pathlib import Path


FIXED_TEST_SIZE: float = 0.2
FIXED_RANDOM_SEED: int = 42


def ensure_path_string(path: str | Path) -> str:
    return str(Path(path).resolve())
