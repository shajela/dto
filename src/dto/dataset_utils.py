"""
S3 Dataset Utilities for DTO Framework

Provides utilities for creating datasets that are loaded from S3 with various data formats.
All training data in the DTO framework must be stored in S3.
"""

import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Optional, List, Union, Any
import os


class S3Dataset(Dataset):
    """
    A dataset wrapper that can load data from S3 in various formats.
    
    This class serves as a placeholder/metadata container when the actual
    data loading is deferred until distributed training starts.
    
    Args:
        s3_path: Path to data file in S3 bucket (without bucket name)
        target_column: Name of target column (for CSV/Parquet)
        feature_columns: List of feature column names (optional)
        data_format: Expected data format ('auto', 'csv', 'pickle', 'torch', etc.)
        **kwargs: Additional metadata for data loading
    """
    
    def __init__(self, 
                 s3_path: str,
                 target_column: str = 'target',
                 feature_columns: Optional[List[str]] = None,
                 data_format: str = 'auto',
                 **kwargs):
        self.s3_path = s3_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.data_format = data_format
        self.metadata = kwargs
        
        # These will be set when data is actually loaded
        self._data = None
        self._length = None
    
    def __len__(self):
        if self._length is None:
            raise RuntimeError("Dataset not loaded yet. This is expected before distributed training starts.")
        return self._length
    
    def __getitem__(self, idx):
        if self._data is None:
            raise RuntimeError("Dataset not loaded yet. This is expected before distributed training starts.")
        return self._data[idx]


class CustomDataset(Dataset):
    """
    Base class for custom datasets with S3 support.
    
    Users can extend this class for custom data loading logic.
    """
    
    def __init__(self, s3_path: str, **kwargs):
        self.s3_path = s3_path
        self.metadata = kwargs
        self._data = None
    
    def load_from_s3(self, local_path: str):
        """
        Override this method to implement custom data loading logic.
        
        Args:
            local_path: Path to downloaded file from S3
        """
        raise NotImplementedError("Subclasses must implement load_from_s3")
    
    def __len__(self):
        if self._data is None:
            raise RuntimeError("Dataset not loaded yet")
        return len(self._data)
    
    def __getitem__(self, idx):
        if self._data is None:
            raise RuntimeError("Dataset not loaded yet")
        return self._data[idx]


def create_s3_dataset(s3_path: str, 
                     target_column: str = 'target',
                     feature_columns: Optional[List[str]] = None,
                     **kwargs) -> S3Dataset:
    """
    Factory function to create an S3Dataset.
    
    Args:
        s3_path: Path to data in S3 bucket
        target_column: Target column name
        feature_columns: Feature column names
        **kwargs: Additional metadata
        
    Returns:
        S3Dataset: Dataset ready for distributed training
    """
    return S3Dataset(
        s3_path=s3_path,
        target_column=target_column,
        feature_columns=feature_columns,
        **kwargs
    )


def create_custom_dataset(custom_dataset_class, s3_path: str, **kwargs):
    """
    Factory function to create a CustomDataset subclass instance.
    
    Args:
        custom_dataset_class: The CustomDataset subclass to instantiate
        s3_path: Path to data in S3 bucket
        **kwargs: Additional metadata passed to the dataset constructor
        
    Returns:
        CustomDataset: Instance of the custom dataset class
        
    Example:
        dataset = create_custom_dataset(MyCustomDataset, "data/file.bin", param1="value")
    """
    if not issubclass(custom_dataset_class, CustomDataset):
        raise ValueError(f"custom_dataset_class must be a subclass of CustomDataset, got {custom_dataset_class}")
    
    return custom_dataset_class(s3_path=s3_path, **kwargs)
