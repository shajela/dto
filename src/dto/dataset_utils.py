"""
S3 Dataset Utilities for DTO Framework

Provides utilities for creating datasets that are loaded from S3 with various data formats.
All training data in the DTO framework must be stored in S3.
"""

import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Optional, List, Union, Any
import boto3
import tempfile
import os
import pandas as pd
import pickle
import json

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
                 s3_bucket_name: Optional[str] = None,
                 **kwargs):
        self.s3_path = s3_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.data_format = self._determine_format(s3_path, data_format)
        self.s3_bucket_name = s3_bucket_name
        self.metadata = kwargs
        
        # These will be set when data is actually loaded
        self._data = None
        self._targets = None
        self._length = None
        self._loaded = False
        
        # Load metadata immediately to get length
        self._load_metadata()
    
    def _determine_format(self, s3_path: str, data_format: str) -> str:
        """Determine data format from file extension if 'auto'"""
        if data_format != 'auto':
            return data_format
            
        # Auto-detect from file extension
        ext = s3_path.lower().split('.')[-1]
        format_map = {
            'csv': 'csv',
            'tsv': 'csv',
            'txt': 'csv',
            'parquet': 'parquet',
            'pkl': 'pickle',
            'pickle': 'pickle',
            'pt': 'torch',
            'pth': 'torch',
            'json': 'json',
            'jsonl': 'jsonl'
        }
        return format_map.get(ext, 'csv')  # Default to CSV
    
    def _load_metadata(self):
        """Load just the metadata (like length) without loading full data"""
        if self.s3_bucket_name is None:
            # Can't load metadata without bucket name, use placeholder
            self._length = 1000  # Placeholder length
            return
            
        try:
            s3_client = boto3.client('s3')
            
            # Different approaches based on data format
            if self.data_format == 'csv':
                self._load_csv_metadata(s3_client)
            elif self.data_format == 'parquet':
                self._load_parquet_metadata(s3_client)
            elif self.data_format in ['torch', 'pickle']:
                self._load_binary_metadata(s3_client)
            elif self.data_format in ['json', 'jsonl']:
                self._load_json_metadata(s3_client)
            else:
                # Unknown format, use placeholder
                self._length = 1000
                
        except Exception as e:
            print(f"Warning: Could not load metadata for {self.s3_path}: {e}")
            # If metadata loading fails, use placeholder
            self._length = 1000
    
    def _load_csv_metadata(self, s3_client):
        """Load metadata for CSV files - count rows efficiently"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                s3_client.download_file(self.s3_bucket_name, self.s3_path, tmp_file.name)
                # Count rows efficiently
                with open(tmp_file.name, 'r') as f:
                    self._length = sum(1 for line in f) - 1  # Subtract header
            finally:
                os.unlink(tmp_file.name)
    
    def _load_parquet_metadata(self, s3_client):
        """Load metadata for Parquet files"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                try:
                    s3_client.download_file(self.s3_bucket_name, self.s3_path, tmp_file.name)
                    # Parquet can read metadata without loading full data
                    df_info = pd.read_parquet(tmp_file.name, columns=[])  # Just metadata
                    self._length = len(pd.read_parquet(tmp_file.name))
                finally:
                    os.unlink(tmp_file.name)
        except ImportError:
            print("Warning: pandas not available for parquet metadata, using placeholder")
            self._length = 1000
    
    def _load_binary_metadata(self, s3_client):
        """Load metadata for torch/pickle files"""
        # For binary formats, we might need to load the full file to get length
        # Use a reasonable placeholder to avoid downloading
        self._length = 1000
    
    def _load_json_metadata(self, s3_client):
        """Load metadata for JSON/JSONL files"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                s3_client.download_file(self.s3_bucket_name, self.s3_path, tmp_file.name)
                if self.data_format == 'jsonl':
                    # Count lines for JSONL
                    with open(tmp_file.name, 'r') as f:
                        self._length = sum(1 for line in f)
                else:
                    # For JSON, assume it's a list
                    import json
                    with open(tmp_file.name, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self._length = len(data)
                        else:
                            self._length = 1  # Single object
            finally:
                os.unlink(tmp_file.name)
    
    def set_s3_bucket(self, bucket_name: str):
        """Set the S3 bucket name and reload metadata"""
        self.s3_bucket_name = bucket_name
        self._load_metadata()
    
    def _load_data_if_needed(self):
        """Load data from S3 if not already loaded"""
        if self._loaded:
            return
            
        if self.s3_bucket_name is None:
            raise ValueError("S3 bucket name not set. Call set_s3_bucket() first.")
            
        # Download from S3
        s3_client = boto3.client('s3')
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                s3_client.download_file(self.s3_bucket_name, self.s3_path, tmp_file.name)
                
                # Load based on format
                if self.data_format == 'csv':
                    self._load_csv_data(tmp_file.name)
                elif self.data_format == 'parquet':
                    self._load_parquet_data(tmp_file.name)
                elif self.data_format == 'torch':
                    self._load_torch_data(tmp_file.name)
                elif self.data_format == 'pickle':
                    self._load_pickle_data(tmp_file.name)
                elif self.data_format in ['json', 'jsonl']:
                    self._load_json_data(tmp_file.name)
                else:
                    raise ValueError(f"Unsupported data format: {self.data_format}")
                    
            finally:
                os.unlink(tmp_file.name)
        
        self._loaded = True
    
    def _load_csv_data(self, file_path: str):
        """Load CSV data"""
        df = pd.read_csv(file_path)
        
        # Extract features and targets
        if self.feature_columns:
            features = df[self.feature_columns].values
        else:
            # Use all columns except target
            feature_cols = [col for col in df.columns if col != self.target_column]
            features = df[feature_cols].values
        
        targets = df[self.target_column].values
        
        # Convert to tensors
        self._data = torch.FloatTensor(features)
        self._targets = torch.LongTensor(targets)
        self._length = len(df)  # Update with actual length
    
    def _load_parquet_data(self, file_path: str):
        """Load Parquet data"""
        df = pd.read_parquet(file_path)
        
        # Extract features and targets (same logic as CSV)
        if self.feature_columns:
            features = df[self.feature_columns].values
        else:
            feature_cols = [col for col in df.columns if col != self.target_column]
            features = df[feature_cols].values
        
        targets = df[self.target_column].values
        
        # Convert to tensors
        self._data = torch.FloatTensor(features)
        self._targets = torch.LongTensor(targets)
        self._length = len(df)
    
    def _load_torch_data(self, file_path: str):
        """Load PyTorch tensor data"""
        data = torch.load(file_path, map_location='cpu')
        if isinstance(data, dict):
            self._data = data['features']
            self._targets = data['targets']
        elif isinstance(data, tuple):
            self._data, self._targets = data
        else:
            # Assume it's just features, no targets
            self._data = data
            self._targets = torch.zeros(len(data))  # Placeholder targets
        
        self._length = len(self._data)
    
    def _load_pickle_data(self, file_path: str):
        """Load pickled data"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self._data = torch.FloatTensor(data['features'])
            self._targets = torch.LongTensor(data['targets'])
        elif isinstance(data, tuple):
            features, targets = data
            self._data = torch.FloatTensor(features)
            self._targets = torch.LongTensor(targets)
        else:
            # Assume it's a pandas DataFrame or similar
            if hasattr(data, 'values'):  # pandas DataFrame
                if self.feature_columns:
                    features = data[self.feature_columns].values
                else:
                    feature_cols = [col for col in data.columns if col != self.target_column]
                    features = data[feature_cols].values
                
                targets = data[self.target_column].values
                self._data = torch.FloatTensor(features)
                self._targets = torch.LongTensor(targets)
            else:
                raise ValueError(f"Unsupported pickle data format: {type(data)}")
        
        self._length = len(self._data)
    
    def _load_json_data(self, file_path: str):
        """Load JSON/JSONL data"""
        if self.data_format == 'jsonl':
            # Load JSONL (one JSON object per line)
            data_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    data_list.append(json.loads(line.strip()))
        else:
            # Load regular JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    data_list = data
                else:
                    data_list = [data]
        
        # Convert JSON to features/targets
        if self.feature_columns:
            features = [[item[col] for col in self.feature_columns] for item in data_list]
        else:
            # Use all keys except target
            first_item = data_list[0]
            feature_keys = [key for key in first_item.keys() if key != self.target_column]
            features = [[item[key] for key in feature_keys] for item in data_list]
        
        targets = [item[self.target_column] for item in data_list]
        
        self._data = torch.FloatTensor(features)
        self._targets = torch.LongTensor(targets)
        self._length = len(data_list)
    
    def __len__(self):
        return self._length if self._length is not None else 0
    
    def __getitem__(self, idx):
        self._load_data_if_needed()
        return self._data[idx], self._targets[idx]


# Keep your existing CustomDataset and factory functions unchanged
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
                     data_format: str = 'auto',
                     **kwargs) -> S3Dataset:
    """
    Factory function to create an S3Dataset.
    
    Args:
        s3_path: Path to data in S3 bucket
        target_column: Target column name
        feature_columns: Feature column names
        data_format: Data format ('auto', 'csv', 'parquet', 'torch', 'pickle', 'json', 'jsonl')
        **kwargs: Additional metadata
        
    Returns:
        S3Dataset: Dataset ready for distributed training
    """
    return S3Dataset(
        s3_path=s3_path,
        target_column=target_column,
        feature_columns=feature_columns,
        data_format=data_format,
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
