"""Immutable dataset preparation and verification."""

from autodidact.data.config import DEFAULT_CONFIG, PipelineConfig
from autodidact.data.integrity import DatasetIntegrityError, verify_dataset
from autodidact.data.pipeline import build_dataset

__all__ = [
    "DEFAULT_CONFIG",
    "DatasetIntegrityError",
    "PipelineConfig",
    "build_dataset",
    "verify_dataset",
]
