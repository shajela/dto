"""
DTO - Distributed Training Orchestrator

A framework for distributed training with support for various data formats
and cloud storage backends.
"""

from .distributed_trainer import DistributedTrainer
from .dataset_utils import (
    create_s3_dataset,
    create_custom_dataset,
    S3Dataset,
    CustomDataset
)

__version__ = "0.1.0"
__author__ = "DTO Team"

__all__ = [
    "DistributedTrainer",
    "create_s3_dataset",
    "create_custom_dataset",
    "S3Dataset",
    "CustomDataset"
]
