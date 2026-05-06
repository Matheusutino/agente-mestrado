from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DiscoveredDatasets(BaseModel):
    dataset_paths: list[str]


class DatasetProfile(BaseModel):
    dataset_path: str
    num_rows: int
    num_columns: int
    columns: list[str]
    column_types: dict[str, str]
    missing_values: dict[str, int]
    describe_summary: dict[str, dict[str, Any]]
    sample_rows: list[dict[str, Any]]
    class_distributions: dict[str, dict[str, int]]
    notes: list[str] = Field(default_factory=list)
