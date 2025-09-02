"""
Distributed Training Framework

A simple framework that handles all Horovod complexity while allowing users
to provide their own training logic. The framework automatically handles:
- Horovod initialization and cleanup
- Learning rate scaling
- Parameter broadcasting
- Optimizer wrapping
- Distributed data sampling
- Progress logging (only on rank 0)
- Leader checkpointing (automatic save/resume)
- S3 checkpoint backup (optional)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import horovod.torch as hvd
from typing import Callable, Optional, Any, Dict, Union, Tuple
import logging
import os
import json
import glob
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import tempfile


class DistributedTrainer:
    """
    A framework for distributed training that handles all Horovod complexity.
    Users only need to provide their training logic.
    """
    
    def __init__(self, 
                 s3_bucket_arn: str,
                 auto_scale_lr: bool = True,
                 auto_scale_batch_size: bool = False,
                 verbose: bool = True,
                 checkpoint_interval: int = 10,
                 keep_last_n_checkpoints: int = 3,
                 save_best_only: bool = False):
        """
        Initialize the distributed trainer.
        
        Args:
            s3_bucket_arn: S3 bucket ARN for checkpoints and training data (required)
            auto_scale_lr: Whether to automatically scale learning rate by world size
            auto_scale_batch_size: Whether to scale batch size by world size
            verbose: Whether to print progress information
            checkpoint_interval: Save checkpoint every N epochs (0 to disable)
            keep_last_n_checkpoints: Number of recent checkpoints to keep (0 to keep all)
            save_best_only: Only save checkpoint if validation metric improves
        """
        # Initialize Horovod
        hvd.init()
        
        # Setup device
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
            self.device = torch.device('cuda', hvd.local_rank())
        else:
            self.device = torch.device('cpu')
        
        self.rank = hvd.rank()
        self.size = hvd.size()
        self.local_rank = hvd.local_rank()
        self.auto_scale_lr = auto_scale_lr
        self.auto_scale_batch_size = auto_scale_batch_size

        # Setup logging (only on rank 0)
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Checkpointing configuration
        self.checkpoint_interval = checkpoint_interval
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float('inf')  # Assumes lower is better (loss)
        self.checkpoint_dir = self._setup_checkpoint_dir()

        # S3 configuration
        self.s3_bucket_arn = s3_bucket_arn
        self.s3_client = None
        self.s3_bucket_name = None
        self._setup_s3_client()
        
        if self.is_master():
            self.log(f"Initialized distributed training on {self.size} processes")
            self.log(f"Using device: {self.device}")
            self.log(f"S3 bucket: {self.s3_bucket_name}")
            if self.checkpoint_interval > 0:
                self.log(f"Checkpointing enabled: every {self.checkpoint_interval} epochs")
                self.log(f"Checkpoint directory: {self.checkpoint_dir}")
                self.log(f"S3 backup enabled for checkpoints")
    
    def _setup_s3_client(self):
        """Setup S3 client and extract bucket name from ARN"""
        try:
            # Extract bucket name from ARN (arn:aws:s3:::bucket-name)
            if self.s3_bucket_arn.startswith('arn:aws:s3:::') or self.s3_bucket_arn.startswith('arn:aws-us-gov:s3:::'):
                self.s3_bucket_name = self.s3_bucket_arn.split(':::')[1]
            else:
                # Assume it's just the bucket name
                self.s3_bucket_name = self.s3_bucket_arn
                
            self.s3_client = boto3.client('s3')
            
            # Test access by listing bucket (only on master)
            if self.is_master():
                self.s3_client.head_bucket(Bucket=self.s3_bucket_name)
                self.log(f"S3 access verified for bucket: {self.s3_bucket_name}")
                
        except (ClientError, NoCredentialsError) as e:
            error_msg = f"Failed to setup S3 client: {e}. S3 bucket ARN is required for all training data and checkpoints."
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected S3 setup error: {e}. S3 bucket ARN is required for all training data and checkpoints."
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _setup_checkpoint_dir(self) -> str:
        """Setup checkpoint directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"./checkpoints/run_{timestamp}"
        
        if self.is_master():
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        return checkpoint_dir
    
    def _load_dataset_from_s3(self, dataloader: DataLoader) -> Dataset:
        """
        Load dataset from S3 supporting multiple formats.
        
        Args:
            dataloader: Original dataloader (used for batch_size and other configs)
            
        Returns:
            Dataset: Loaded dataset ready for distributed training
        """
        if not self.s3_client or not self.s3_bucket_name:
            raise ValueError("S3 client not configured for remote data loading")
        
        s3_path = dataloader.dataset.s3_path
        self.log(f"Loading dataset from S3: {s3_path}")
        
        # Create temporary directory for downloading data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "data")
            os.makedirs(temp_path, exist_ok=True)
            temp_path = os.path.join(temp_path, os.path.basename(s3_path))
            
            # Download data from S3
            try:
                self.s3_client.download_file(self.s3_bucket_name, s3_path, temp_path)
                self.log(f"Downloaded data from s3://{self.s3_bucket_name}/{s3_path}")
            except ClientError as e:
                raise RuntimeError(f"Failed to download data from S3: {e}")
            
            # Check if dataset has custom loading method (for CustomDataset subclasses)
            if hasattr(dataloader.dataset, 'load_from_s3'):
                self.log("Using custom dataset loading method")
                dataloader.dataset.load_from_s3(temp_path)
                return dataloader.dataset._data
            else:
                # Load data based on file extension for S3Dataset
                return self._create_dataset_from_file(temp_path, dataloader.dataset)
    
    def _create_dataset_from_file(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """
        Create a PyTorch Dataset from various file formats.
        
        Args:
            file_path: Path to the downloaded data file
            original_dataset: Original dataset object (may contain metadata)
            
        Returns:
            Dataset: PyTorch dataset ready for training
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext in ['.csv', '.tsv']:
                return self._load_csv_dataset(file_path, original_dataset)
            elif file_ext in ['.pkl', '.pickle']:
                return self._load_pickle_dataset(file_path, original_dataset)
            elif file_ext in ['.pt', '.pth']:
                return self._load_torch_dataset(file_path, original_dataset)
            elif file_ext in ['.npy', '.npz']:
                return self._load_numpy_dataset(file_path, original_dataset)
            elif file_ext in ['.parquet']:
                return self._load_parquet_dataset(file_path, original_dataset)
            elif file_ext in ['.h5', '.hdf5']:
                return self._load_hdf5_dataset(file_path, original_dataset)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.log(f"Error loading dataset from {file_path}: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def _load_csv_dataset(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """Load dataset from CSV/TSV file."""
        # Check if original dataset has column specifications
        target_col = getattr(original_dataset, 'target_column', 'target')
        feature_cols = getattr(original_dataset, 'feature_columns', None)
        
        # Load CSV
        df = pd.read_csv(file_path)
        self.log(f"Loaded CSV with shape: {df.shape}")
        
        # Split features and targets
        if target_col in df.columns:
            y = torch.tensor(df[target_col].values, dtype=torch.long)
            if feature_cols:
                X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            else:
                # Use all columns except target
                feature_cols = [col for col in df.columns if col != target_col]
                X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        else:
            # Assume last column is target
            X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
            y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
        
        return TensorDataset(X, y)
    
    def _load_pickle_dataset(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """Load dataset from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.log(f"Loaded pickle data with type: {type(data)}")
        
        if isinstance(data, dict):
            # Expect {'X': features, 'y': targets} format
            X = torch.tensor(data['X'], dtype=torch.float32)
            y = torch.tensor(data['y'], dtype=torch.long)
            return TensorDataset(X, y)
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            # Expect (features, targets) format
            X = torch.tensor(data[0], dtype=torch.float32)
            y = torch.tensor(data[1], dtype=torch.long)
            return TensorDataset(X, y)
        elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            # Already a dataset-like object
            return data
        else:
            raise ValueError(f"Unsupported pickle data format: {type(data)}")
    
    def _load_torch_dataset(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """Load dataset from PyTorch file."""
        data = torch.load(file_path, map_location='cpu')
        
        if isinstance(data, dict):
            if 'dataset' in data:
                return data['dataset']
            elif 'X' in data and 'y' in data:
                return TensorDataset(data['X'], data['y'])
            else:
                # Try to find tensor pairs
                tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
                if len(tensors) == 2:
                    return TensorDataset(tensors[0], tensors[1])
        elif isinstance(data, Dataset):
            return data
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            X = data[0] if isinstance(data[0], torch.Tensor) else torch.tensor(data[0])
            y = data[1] if isinstance(data[1], torch.Tensor) else torch.tensor(data[1])
            return TensorDataset(X, y)
        
        raise ValueError(f"Unsupported PyTorch data format: {type(data)}")
    
    def _load_numpy_dataset(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """Load dataset from NumPy file."""
        if file_path.endswith('.npz'):
            data = np.load(file_path)
            # Expect 'X' and 'y' keys, or 'features' and 'labels'
            if 'X' in data and 'y' in data:
                X = torch.tensor(data['X'], dtype=torch.float32)
                y = torch.tensor(data['y'], dtype=torch.long)
            elif 'features' in data and 'labels' in data:
                X = torch.tensor(data['features'], dtype=torch.float32)
                y = torch.tensor(data['labels'], dtype=torch.long)
            else:
                # Take first two arrays
                arrays = list(data.values())
                if len(arrays) >= 2:
                    X = torch.tensor(arrays[0], dtype=torch.float32)
                    y = torch.tensor(arrays[1], dtype=torch.long)
                else:
                    raise ValueError("NPZ file must contain at least 2 arrays")
        else:
            # Single .npy file - assume it contains both features and targets
            data = np.load(file_path)
            if data.ndim == 2 and data.shape[1] > 1:
                # Split features (all but last) and targets (last column)
                X = torch.tensor(data[:, :-1], dtype=torch.float32)
                y = torch.tensor(data[:, -1], dtype=torch.long)
            else:
                raise ValueError("Single NPY file must be 2D with multiple columns")
        
        return TensorDataset(X, y)
    
    def _load_parquet_dataset(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """Load dataset from Parquet file."""
        df = pd.read_parquet(file_path)
        self.log(f"Loaded Parquet with shape: {df.shape}")
        
        # Use same logic as CSV
        return self._load_csv_dataset(file_path, original_dataset)
    
    def _load_hdf5_dataset(self, file_path: str, original_dataset: Dataset) -> Dataset:
        """Load dataset from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 support. Install with: pip install h5py")
        
        with h5py.File(file_path, 'r') as f:
            # Common HDF5 dataset names
            if 'X' in f and 'y' in f:
                X = torch.tensor(f['X'][:], dtype=torch.float32)
                y = torch.tensor(f['y'][:], dtype=torch.long)
            elif 'features' in f and 'labels' in f:
                X = torch.tensor(f['features'][:], dtype=torch.float32)
                y = torch.tensor(f['labels'][:], dtype=torch.long)
            elif 'data' in f and 'target' in f:
                X = torch.tensor(f['data'][:], dtype=torch.float32)
                y = torch.tensor(f['target'][:], dtype=torch.long)
            else:
                # Take first two datasets
                keys = list(f.keys())
                if len(keys) >= 2:
                    X = torch.tensor(f[keys[0]][:], dtype=torch.float32)
                    y = torch.tensor(f[keys[1]][:], dtype=torch.long)
                else:
                    raise ValueError("HDF5 file must contain at least 2 datasets")
        
        return TensorDataset(X, y)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging that only outputs on rank 0"""
        logger = logging.getLogger(f'DistributedTrainer_rank_{self.rank}')
        logger.setLevel(logging.INFO)
        
        if self.is_master() and self.verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[Rank 0] %(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def is_master(self) -> bool:
        """Check if current process is the master (rank 0)"""
        return self.rank == 0
    
    def log(self, message: str):
        """Log message only on master process"""
        if self.is_master() and self.verbose:
            self.logger.info(message)
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for distributed training.
        Moves to device and broadcasts parameters from rank 0.
        """
        model = model.to(self.device)
        
        # Broadcast parameters from rank 0 to all other processes
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        
        self.log(f"Model prepared for distributed training")
        self.log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def prepare_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        """
        Prepare optimizer for distributed training.
        Scales learning rate and wraps with Horovod DistributedOptimizer.
        """
        # Scale learning rate by world size if enabled
        if self.auto_scale_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.size
                self.log(f"Scaled learning rate to {param_group['lr']:.6f} (original * {self.size})")
        
        # Wrap optimizer with Horovod DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(
            optimizer, 
            named_parameters=model.named_parameters()
        )
        
        # Broadcast optimizer state from rank 0 to all other processes
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        
        self.log("Optimizer prepared for distributed training")
        
        return optimizer
    
    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Prepare dataloader for distributed training.
        Creates distributed sampler to ensure each worker sees different data.
        All training data is loaded from S3.
        """
        # Load training data from S3
        self.log("Loading training data from S3")
        dataset = self._load_dataset_from_s3(dataloader)

        # Create distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=self.size, 
            rank=self.rank
        )
        
        # Get batch size (scale if enabled)
        batch_size = dataloader.batch_size
        if self.auto_scale_batch_size:
            batch_size *= self.size
            self.log(f"Scaled batch size to {batch_size} (original * {self.size})")
        
        # Create new dataloader with distributed sampler
        distributed_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last
        )
        
        self.log(f"DataLoader prepared with distributed sampler")
        self.log(f"Total samples: {len(dataset)}, Samples per worker: {len(distributed_dataloader.dataset) // self.size}")
        
        return distributed_dataloader
    
    def allreduce(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """
        Perform allreduce operation across all workers.
        Useful for averaging metrics across workers.
        """
        return hvd.allreduce(tensor, average=average)
    
    def save_checkpoint(self, 
                       epoch: int, 
                       model: nn.Module, 
                       optimizer: optim.Optimizer, 
                       metrics: Dict[str, Any],
                       is_best: bool = False) -> Optional[str]:
        """
        Save checkpoint (only on rank 0 to avoid race conditions).
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            metrics: Training metrics to save
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint (None if not master)
        """
        if not self.is_master():
            return None
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'horovod_size': self.size,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            self.log(f"New best checkpoint saved (epoch {epoch})")
        
        # Save latest checkpoint link
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_path)
        
        # Save training metadata
        metadata = {
            'latest_epoch': epoch,
            'latest_checkpoint': checkpoint_path,
            'best_checkpoint': os.path.join(self.checkpoint_dir, "best_checkpoint.pt") if is_best else None,
            'total_epochs': epoch,
            'metrics_history': metrics
        }
        metadata_path = os.path.join(self.checkpoint_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.log(f"Checkpoint saved: {checkpoint_path}")
        
        # Upload to S3 if configured
        if self.s3_client and self.s3_bucket_name:
            self._upload_checkpoint_to_s3(checkpoint_path, epoch, is_best)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: str, 
                       model: nn.Module, 
                       optimizer: Optional[optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            Dictionary containing loaded metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.log(f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint['epoch']})")
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'horovod_size': checkpoint.get('horovod_size', 1),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint in the checkpoint directory.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        # Fallback: find highest numbered checkpoint
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return checkpoints[-1]
    
    def _upload_checkpoint_to_s3(self, checkpoint_path: str, epoch: int, is_best: bool = False):
        """
        Upload checkpoint to S3 bucket.
        
        Args:
            checkpoint_path: Local path to checkpoint file
            epoch: Epoch number
            is_best: Whether this is the best checkpoint
        """
        if not self.s3_client or not self.s3_bucket_name:
            return
        
        try:
            # Create S3 key structure: checkpoints/run_timestamp/filename
            run_id = os.path.basename(self.checkpoint_dir)
            filename = os.path.basename(checkpoint_path)
            s3_key = f"checkpoints/{run_id}/{filename}"
            
            # Upload regular checkpoint
            self.s3_client.upload_file(checkpoint_path, self.s3_bucket_name, s3_key)
            self.log(f"Checkpoint uploaded to S3: s3://{self.s3_bucket_name}/{s3_key}")
            
            # Upload as latest checkpoint
            latest_s3_key = f"checkpoints/{run_id}/latest_checkpoint.pt"
            self.s3_client.upload_file(checkpoint_path, self.s3_bucket_name, latest_s3_key)
            
            # Upload as best checkpoint if applicable
            if is_best:
                best_s3_key = f"checkpoints/{run_id}/best_checkpoint.pt"
                self.s3_client.upload_file(checkpoint_path, self.s3_bucket_name, best_s3_key)
                self.log(f"Best checkpoint uploaded to S3: s3://{self.s3_bucket_name}/{best_s3_key}")
            
            # Upload metadata
            metadata_path = os.path.join(self.checkpoint_dir, "training_metadata.json")
            if os.path.exists(metadata_path):
                metadata_s3_key = f"checkpoints/{run_id}/training_metadata.json"
                self.s3_client.upload_file(metadata_path, self.s3_bucket_name, metadata_s3_key)
                
        except ClientError as e:
            self.log(f"Failed to upload checkpoint to S3: {e}")
        except Exception as e:
            self.log(f"Unexpected error during S3 upload: {e}")
    
    def download_checkpoint_from_s3(self, s3_key: str, local_path: Optional[str] = None) -> Optional[str]:
        """
        Download checkpoint from S3.
        
        Args:
            s3_key: S3 key for the checkpoint
            local_path: Local path to save checkpoint (optional)
            
        Returns:
            Path to downloaded checkpoint or None if failed
        """
        if not self.s3_client or not self.s3_bucket_name:
            self.log("S3 not configured, cannot download checkpoint")
            return None
        
        if local_path is None:
            # Create local path in checkpoint directory
            filename = os.path.basename(s3_key)
            local_path = os.path.join(self.checkpoint_dir, f"downloaded_{filename}")
        
        try:
            self.s3_client.download_file(self.s3_bucket_name, s3_key, local_path)
            self.log(f"Checkpoint downloaded from S3: s3://{self.s3_bucket_name}/{s3_key} -> {local_path}")
            return local_path
        except ClientError as e:
            self.log(f"Failed to download checkpoint from S3: {e}")
            return None
        except Exception as e:
            self.log(f"Unexpected error during S3 download: {e}")
            return None
    
    def list_s3_checkpoints(self, run_id: Optional[str] = None) -> list:
        """
        List available checkpoints in S3.
        
        Args:
            run_id: Specific run ID to list checkpoints for (optional)
            
        Returns:
            List of S3 keys for available checkpoints
        """
        if not self.s3_client or not self.s3_bucket_name:
            self.log("S3 not configured, cannot list checkpoints")
            return []
        
        try:
            if run_id:
                prefix = f"checkpoints/{run_id}/"
            else:
                prefix = "checkpoints/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket_name,
                Prefix=prefix
            )
            
            checkpoints = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.pt'):
                        checkpoints.append(obj['Key'])
            
            return sorted(checkpoints)
            
        except ClientError as e:
            self.log(f"Failed to list S3 checkpoints: {e}")
            return []
        except Exception as e:
            self.log(f"Unexpected error listing S3 checkpoints: {e}")
            return []
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N"""
        if self.keep_last_n_checkpoints <= 0:
            return
        
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoints = glob.glob(pattern)
        
        if len(checkpoints) <= self.keep_last_n_checkpoints:
            return
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.keep_last_n_checkpoints]
        for checkpoint in to_remove:
            try:
                os.remove(checkpoint)
                self.log(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
            except OSError as e:
                self.log(f"Failed to remove checkpoint {checkpoint}: {e}")
    
    def _should_save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """
        Determine if checkpoint should be saved based on configuration.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            
        Returns:
            True if checkpoint should be saved
        """
        if self.checkpoint_interval <= 0:
            return False
        
        # Save on interval
        if (epoch + 1) % self.checkpoint_interval == 0:
            return True
        
        # Save if best (if enabled)
        if self.save_best_only and 'loss' in metrics:
            current_metric = metrics['loss']
            if current_metric < self.best_metric:
                return True
        
        return False
    
    def _is_best_checkpoint(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if current metrics represent the best checkpoint so far.
        
        Args:
            metrics: Current metrics
            
        Returns:
            True if this is the best checkpoint
        """
        if 'loss' not in metrics:
            return False
        
        current_metric = metrics['loss']
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            return True
        
        return False
    
    def save_final_model_to_file(self, 
                        model: nn.Module, 
                        optimizer: optim.Optimizer, 
                        final_metrics: Dict[str, Any], 
                        model_name: str = "final_model") -> str:
        """
        Save the final trained model (only on master node).
        
        Args:
            model: Trained model to save
            optimizer: Final optimizer state
            final_metrics: Final training metrics
            model_name: Name for the final model file
            
        Returns:
            Path to saved model file (empty string on non-master nodes)
        """
        if not self.is_master():
            return ""
        
        # Create final model filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pt"
        model_path = os.path.join(self.checkpoint_dir, model_filename)
        
        # Save final model
        final_state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_metrics': final_metrics,
            'training_complete': True,
            'save_timestamp': timestamp,
            'model_architecture': str(model.__class__.__name__)
        }
        
        try:
            torch.save(final_state, model_path)
            self.log(f"Final model saved: {model_path}")
            
            # Backup to S3 if configured
            if self.s3_client and self.s3_bucket_name:
                try:
                    s3_key = f"final_models/{model_filename}"
                    self.s3_client.upload_file(model_path, self.s3_bucket_name, s3_key)
                    self.log(f"Final model backed up to S3: s3://{self.s3_bucket_name}/{s3_key}")
                except ClientError as e:
                    self.log(f"Warning: Failed to backup final model to S3: {e}")
            
            return model_path
            
        except Exception as e:
            self.log(f"Error saving final model: {e}")
            return ""
    
    def load_final_model(self, model_path: str, model: nn.Module) -> Dict[str, Any]:
        """
        Load a final trained model.
        
        Args:
            model_path: Path to the saved model file
            model: Model instance to load weights into
            
        Returns:
            Dictionary containing model info and metrics
        """
        if not os.path.exists(model_path):
            # Try to download from S3 if configured
            if self.s3_client and self.s3_bucket_name:
                model_filename = os.path.basename(model_path)
                s3_key = f"final_models/{model_filename}"
                try:
                    self.log(f"Downloading final model from S3: {s3_key}")
                    self.s3_client.download_file(self.s3_bucket_name, s3_key, model_path)
                    self.log(f"Final model downloaded from S3")
                except ClientError as e:
                    raise FileNotFoundError(f"Final model not found locally or in S3: {model_path}")
            else:
                raise FileNotFoundError(f"Final model not found: {model_path}")
        
        # Load final model
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model_info = {
            'final_metrics': checkpoint.get('final_metrics', {}),
            'save_timestamp': checkpoint.get('save_timestamp', 'unknown'),
            'model_architecture': checkpoint.get('model_architecture', 'unknown'),
            'training_complete': checkpoint.get('training_complete', False)
        }
        
        self.log(f"Final model loaded: {model_path}")
        return model_info
    
    def fit(self,
            model: nn.Module,
            train_loader: DataLoader,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            num_epochs: int,
            train_step_fn: Optional[Callable] = None,
            resume_from_checkpoint: Optional[str] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Main training method that orchestrates the distributed training.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            optimizer: Optimizer for training
            criterion: Loss function
            num_epochs: Number of training epochs
            train_step_fn: Custom training step function (optional)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            **kwargs: Additional arguments passed to training functions
        
        Returns:
            Dictionary containing training history and metrics
        """
        self.log("Starting distributed training...")
        
        # Prepare components for distributed training
        model = self.prepare_model(model)
        optimizer = self.prepare_optimizer(optimizer, model)
        train_loader = self.prepare_dataloader(train_loader)
        
        # Handle checkpoint resume
        start_epoch = 0
        if resume_from_checkpoint:
            if resume_from_checkpoint == "auto":
                # Auto-find latest checkpoint
                resume_from_checkpoint = self.get_latest_checkpoint()
                if resume_from_checkpoint:
                    self.log(f"Auto-resuming from: {resume_from_checkpoint}")
                else:
                    self.log("No checkpoint found for auto-resume, starting from scratch")
            
            if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
                checkpoint_info = self.load_checkpoint(resume_from_checkpoint, model, optimizer)
                start_epoch = checkpoint_info['epoch']
                self.best_metric = checkpoint_info.get('metrics', {}).get('loss', float('inf'))
                self.log(f"Resumed training from epoch {start_epoch}")
        
        # Training history
        history = {
            'train_loss': [],
            'epochs': []
        }
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler (important for shuffling)
            train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            if train_step_fn is not None:
                # Use custom training step
                train_metrics = self._train_epoch_custom(
                    model, train_loader, optimizer, criterion, 
                    train_step_fn, epoch, **kwargs
                )
            else:
                # Use default training step
                train_metrics = self._train_epoch_default(
                    model, train_loader, optimizer, criterion, epoch
                )
            
            # Update history
            history['train_loss'].append(train_metrics.get('loss', 0.0))
            history['epochs'].append(epoch + 1)
            
            # Save checkpoint if needed (only on master)
            if self._should_save_checkpoint(epoch, train_metrics):
                is_best = self._is_best_checkpoint(train_metrics)
                self.save_checkpoint(epoch + 1, model, optimizer, train_metrics, is_best)
            
            # Log progress
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1:
                log_msg = f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_metrics.get('loss', 0.0):.4f}"
                self.log(log_msg)
        
        # Save final model after training completion
        final_metrics = {
            'final_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
            'total_epochs': num_epochs,
            'best_loss': self.best_metric if self.best_metric != float('inf') else None
        }
        
        final_model_path = self.save_final_model_to_file(model, optimizer, final_metrics)
        if final_model_path:
            history['final_model_path'] = final_model_path
        
        self.log("Training completed!")
        return history
    
    def _train_epoch_default(self, 
                           model: nn.Module, 
                           train_loader: DataLoader, 
                           optimizer: optim.Optimizer, 
                           criterion: nn.Module, 
                           epoch: int) -> Dict[str, Any]:
        """Default training step implementation"""
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        # Average loss across all workers
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = self.allreduce(avg_loss_tensor, average=True).item()
        
        return {'loss': avg_loss}
    
    def _train_epoch_custom(self, 
                          model: nn.Module, 
                          train_loader: DataLoader, 
                          optimizer: optim.Optimizer, 
                          criterion: nn.Module, 
                          train_step_fn: Callable, 
                          epoch: int, 
                          **kwargs) -> Dict[str, Any]:
        """Custom training step implementation"""
        model.train()
        
        # Call user's custom training function
        metrics = train_step_fn(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            trainer=self,
            epoch=epoch,
            **kwargs
        )
        
        return metrics if metrics is not None else {}


# Convenience function for quick setup
def create_distributed_trainer(s3_bucket_arn: str, **kwargs) -> DistributedTrainer:
    """
    Create and return a DistributedTrainer instance.
    
    Args:
        s3_bucket_arn: S3 bucket ARN for training data and checkpoints (required)
        **kwargs: Additional arguments passed to DistributedTrainer
        
    Returns:
        DistributedTrainer instance configured for S3-based training
    """
    return DistributedTrainer(s3_bucket_arn=s3_bucket_arn, **kwargs)
