from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_file(path: str) -> dict[str, Any] | list[Any]:
    """Read a JSON artifact from disk and return its parsed contents.

    Args:
        path: Absolute or relative path to a JSON file.

    Returns:
        Parsed JSON content as a dictionary or list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path does not point to a `.json` file.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(path).expanduser().resolve()
    if file_path.suffix.lower() != ".json":
        raise ValueError("read_json_file only supports .json files.")
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))
