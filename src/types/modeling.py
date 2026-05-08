from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

class LogisticRegressionConfig(BaseModel):
    model: Literal["logistic_regression"]
    max_iter: int = 1000
    class_weight: Literal["balanced"] | None = "balanced"
    c: float = 1.0

    @field_validator("max_iter")
    @classmethod
    def validate_max_iter(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_iter must be at least 1.")
        return value

    @field_validator("c")
    @classmethod
    def validate_c(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("c must be greater than 0.")
        return value


class LinearSVMConfig(BaseModel):
    model: Literal["linear_svm"]
    class_weight: Literal["balanced"] | None = "balanced"
    c: float = 1.0

    @field_validator("c")
    @classmethod
    def validate_c(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("c must be greater than 0.")
        return value


class MultinomialNBConfig(BaseModel):
    model: Literal["multinomial_nb"]
    alpha: float = 1.0

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("alpha must be greater than 0.")
        return value


class DecisionTreeConfig(BaseModel):
    model: Literal["decision_tree"]
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: Literal["balanced"] | None = None

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_depth must be at least 1 when provided.")
        return value

    @field_validator("min_samples_split")
    @classmethod
    def validate_min_samples_split(cls, value: int) -> int:
        if value < 2:
            raise ValueError("min_samples_split must be at least 2.")
        return value

    @field_validator("min_samples_leaf")
    @classmethod
    def validate_min_samples_leaf(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        return value


class KNNConfig(BaseModel):
    model: Literal["knn"]
    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"
    metric: str = "minkowski"

    @field_validator("n_neighbors")
    @classmethod
    def validate_n_neighbors(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_neighbors must be at least 1.")
        return value


ModelConfig = Annotated[
    LogisticRegressionConfig
    | LinearSVMConfig
    | MultinomialNBConfig
    | DecisionTreeConfig
    | KNNConfig,
    Field(discriminator="model"),
]


class TrainingResult(BaseModel):
    model_type: Literal[
        "logistic_regression",
        "linear_svm",
        "multinomial_nb",
        "decision_tree",
        "knn",
    ]
    trained_on_rows: int
    config: ModelConfig
