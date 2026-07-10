"""Immutable dataset preparation and verification."""

from autodidact.data.archive import fetch_prepared_dataset
from autodidact.data.config import (
    DEFAULT_CONFIG,
    PREPARED_DATASET_ARCHIVE,
    PipelineConfig,
)
from autodidact.data.integrity import DatasetIntegrityError, verify_dataset
from autodidact.data.pipeline import build_dataset

__all__ = [
    "DEFAULT_CONFIG",
    "DatasetIntegrityError",
    "PREPARED_DATASET_ARCHIVE",
    "PipelineConfig",
    "build_dataset",
    "fetch_prepared_dataset",
    "verify_dataset",
]
