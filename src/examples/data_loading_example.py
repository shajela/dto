"""
This example shows how to use the DTO framework with all training data
stored in S3. The framework has been simplified to only support S3-based
data loading for a fully cloud-native distributed training experience.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import DTO framework
from dto import DistributedTrainer, create_s3_dataset
from examples.simple_net import SimpleNet


def main():
    """Example showing S3-only training workflow."""
    
    # S3 bucket ARN is now required for all operations
    s3_bucket_arn = "arn:aws:s3:::my-ml-training-data"
    
    # Create dataset from S3 - this is the only supported method now
    dataset = create_s3_dataset(
        s3_path="training-data/housing_prices.csv",
        target_column="price_category",
        feature_columns=["bedrooms", "bathrooms", "sqft", "age", "location_score"]
    )
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleNet(input_size=5, hidden_size=64, num_classes=3)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create distributed trainer - S3 bucket ARN is required
    trainer = DistributedTrainer(
        s3_bucket_arn=s3_bucket_arn,  # Required parameter
        auto_scale_lr=True,
        checkpoint_interval=5,
        verbose=True
    )
    
    # Train the model - all data will be loaded from S3
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50
    )
    
    print("Training completed!")
    return history


def example_with_different_formats():
    """Examples with different S3 data formats."""
    
    s3_bucket_arn = "arn:aws:s3:::my-ml-training-data"
    
    # Example 1: Pickle data from S3
    pickle_dataset = create_s3_dataset(
        s3_path="preprocessed/features.pkl",
        data_format="pickle"
    )
    
    # Example 2: PyTorch dataset from S3
    torch_dataset = create_s3_dataset(
        s3_path="datasets/processed_dataset.pt",
        data_format="torch"
    )
    
    # Example 3: Parquet data from S3
    parquet_dataset = create_s3_dataset(
        s3_path="big-data/training_data.parquet",
        target_column="label",
        feature_columns=["feature_1", "feature_2", "feature_3"]
    )
    
    # All datasets work the same way with the trainer
    trainer = DistributedTrainer(s3_bucket_arn=s3_bucket_arn)
    
    # Just change the dataset in the DataLoader
    for name, dataset in [("pickle", pickle_dataset), ("torch", torch_dataset), ("parquet", parquet_dataset)]:
        print(f"Training with {name} dataset from S3...")
        # Would train here...


def example_custom_dataset():
    """Example with custom S3 dataset."""
    
    from dto import CustomDataset
    import torch
    from torch.utils.data import TensorDataset
    import json
    
    class JSONLDataset(CustomDataset):
        """Custom dataset for JSON Lines format stored in S3."""
        
        def load_from_s3(self, local_path: str):
            """Load JSON Lines data from downloaded S3 file."""
            features = []
            labels = []
            
            with open(local_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    features.append(data['features'])
                    labels.append(data['label'])
            
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            self._data = TensorDataset(X, y)
    
    # Create custom dataset
    custom_dataset = JSONLDataset(s3_path="custom-data/training.jsonl")
    
    train_loader = DataLoader(custom_dataset, batch_size=64)
    
    # Use with trainer
    trainer = DistributedTrainer(s3_bucket_arn="arn:aws:s3:::my-bucket")
    
    # The trainer will automatically call the custom load_from_s3 method
    # when it needs to load the data
