"""
Local PyTorch Training Script

Standard PyTorch training loop using the same model and parameters as
the distributed trainer example. No distributed training or S3 - just
traditional single-GPU/CPU PyTorch training for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import time
from simple_net import SimpleNet


def load_local_data(csv_path: str) -> TensorDataset:
    """
    Load housing data from local CSV file and create PyTorch dataset.
    
    Args:
        csv_path: Path to housing_prices.csv file
        
    Returns:
        TensorDataset with features and targets
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Same columns as distributed trainer
    feature_columns = ["bedrooms", "bathrooms", "sqft", "age", "location_score"]
    target_column = "price_category"
    
    # Extract features and targets
    features = df[feature_columns].values  # Shape: (n_samples, 5)
    targets = df[target_column].values     # Shape: (n_samples,)
    
    # Convert to tensors (same as S3Dataset does)
    features_tensor = torch.FloatTensor(features)
    targets_tensor = torch.LongTensor(targets)
    
    # Create dataset
    dataset = TensorDataset(features_tensor, targets_tensor)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Features shape: {features_tensor.shape}")
    print(f"Target distribution: {np.bincount(targets)}")
    
    return dataset


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch using standard PyTorch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        num_batches += 1
    
    # Calculate average loss
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(csv_path: str, model_path: Optional[str] = None):
    """Main training function with exact same setup as distributed trainer."""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data from local csv
    dataset = load_local_data(csv_path)
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleNet(input_size=5, hidden_size=64, num_classes=2)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    num_epochs = 200
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 50)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train one epoch
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Log progress (every 10 epochs, same as distributed trainer default)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] - Loss: {avg_loss:.4f}")
    
    print("=" * 50)
    print("Training completed!")
    print(f"Final loss: {avg_loss:.4f}")
    
    # Optional: Save the trained model
    if model_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': avg_loss,
            'epoch': num_epochs
        }, model_path)
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    start_time = time.time()
    main(csv_path="./data/housing_prices.csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
