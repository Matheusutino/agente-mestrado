from __future__ import annotations

from pydantic import BaseModel


class PreprocessingResult(BaseModel):
    dataset_path: str
    text_column: str
    label_column: str
    num_classes: int
    classes: list[str]
    class_distribution: dict[str, int]
    train_class_distribution: dict[str, int]
    test_class_distribution: dict[str, int]
    train_rows: int
    test_rows: int
    test_size: float
    random_state: int
