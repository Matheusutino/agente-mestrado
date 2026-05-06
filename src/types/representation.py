from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SparseRepresentationConfig(BaseModel):
    representation: Literal["tfidf", "bow"]
    max_features: int = 10000
    lowercase: bool = True
    ngram_min: int = 1
    ngram_max: int = 1
    min_df: int | float = 1
    max_df: int | float = 1.0
    binary: bool = False
    use_idf: bool = True
    sublinear_tf: bool = False

    @model_validator(mode="after")
    def validate_sparse_fields(self) -> "SparseRepresentationConfig":
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError(
                "ngram_min must be at least 1 and ngram_max must be >= ngram_min."
            )
        return self


class DenseRepresentationConfig(BaseModel):
    representation: Literal["sentence_transformer"]
    model_name: str
    normalize_embeddings: bool = True
    batch_size: int = 32
    truncate_dim: int | None = None
    device: str | None = None

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("batch_size must be at least 1.")
        return value

    @field_validator("truncate_dim")
    @classmethod
    def validate_truncate_dim(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("truncate_dim must be at least 1 when provided.")
        return value


RepresentationConfig = Annotated[
    SparseRepresentationConfig | DenseRepresentationConfig,
    Field(discriminator="representation"),
]


class RepresentationResult(BaseModel):
    representation: Literal["tfidf", "bow", "sentence_transformer"]
    vocab_size: int | None
    feature_dim: int
    feature_storage_format: Literal["sparse_npz", "dense_npy"]
    config: RepresentationConfig
