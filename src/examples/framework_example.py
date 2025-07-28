"""
Example usage of the DistributedTrainer framework.

This shows how to convert your existing training script to use the framework
without modifying the core training logic. Demonstrates:
- Basic distributed training
- Custom training steps
- Checkpointing and resume
- S3 backup integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
import numpy as np
from sklearn.datasets import make_classification
from simple_net import SimpleNet
from distributed_trainer import DistributedTrainer


def create_synthetic_data() -> TensorDataset:
    """Create the same synthetic data as in your original script"""
    input_size = 20
    num_classes = 2
    
    X, y = make_classification(
        n_samples=1000,
        n_features=input_size,
        n_classes=num_classes,
        n_redundant=0,
        n_informative=input_size,
        random_state=42
    )
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    return TensorDataset(X_tensor, y_tensor)


def main_simple() -> None:
    """
    Simple usage - let the framework handle everything with default training loop
    """
    # Create trainer with basic checkpointing
    trainer = DistributedTrainer(
        auto_scale_lr=True, 
        verbose=True,
        checkpoint_interval=10,  # Save every 10 epochs
        keep_last_n_checkpoints=3  # Keep only last 3 checkpoints
    )
    
    # Create data
    dataset = create_synthetic_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train using framework (automatically handles all Horovod complexity)
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50
    )
    
    if trainer.is_master():
        print("Training completed!")
        print(f"Final loss: {history['train_loss'][-1]:.4f}")
        print(f"Checkpoints saved in: {trainer.checkpoint_dir}")


def main_with_s3_backup() -> None:
    """
    Training with S3 backup for checkpoints
    """
    # Create trainer with S3 backup (replace with your actual S3 bucket)
    trainer = DistributedTrainer(
        auto_scale_lr=True,
        verbose=True,
        checkpoint_interval=5,  # More frequent checkpoints for demo
        keep_last_n_checkpoints=2,
        s3_bucket_arn="arn:aws:s3:::my-training-checkpoints"  # Your S3 bucket
    )
    
    # Create data and model
    dataset = create_synthetic_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train with automatic S3 backup
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=30
    )
    
    if trainer.is_master():
        print("Training with S3 backup completed!")
        print(f"Final loss: {history['train_loss'][-1]:.4f}")
        
        # List S3 checkpoints
        s3_checkpoints = trainer.list_s3_checkpoints()
        print(f"S3 checkpoints: {s3_checkpoints}")


def main_with_resume() -> None:
    """
    Training with checkpoint resume functionality
    """
    # Create trainer
    trainer = DistributedTrainer(
        auto_scale_lr=True,
        verbose=True,
        checkpoint_interval=5,
        save_best_only=False  # Save all interval checkpoints
    )
    
    # Create data and model
    dataset = create_synthetic_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    if trainer.is_master():
        print("=== Phase 1: Initial Training ===")
    
    # Train for initial epochs
    trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=15
    )
    
    if trainer.is_master():
        print("=== Phase 2: Resume Training ===")
    
    # Continue training from latest checkpoint
    trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=30,  # Will continue from epoch 15 to 30
        resume_from_checkpoint="auto"  # Auto-find latest checkpoint
    )
    
    if trainer.is_master():
        print("Resume training completed!")


def main_best_checkpoint_only() -> None:
    """
    Training that only saves the best performing checkpoints
    """
    trainer = DistributedTrainer(
        auto_scale_lr=True,
        verbose=True,
        checkpoint_interval=5,
        save_best_only=True,  # Only save when loss improves
        keep_last_n_checkpoints=0  # Keep all best checkpoints
    )
    
    # Create data and model
    dataset = create_synthetic_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train with best-only checkpointing
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=40
    )
    
    if trainer.is_master():
        print("Best-checkpoint-only training completed!")
        print(f"Best loss achieved: {min(history['train_loss']):.4f}")
        print(f"Final loss: {history['train_loss'][-1]:.4f}")


def custom_training_step(model: nn.Module, 
                        train_loader: DataLoader, 
                        optimizer: optim.Optimizer, 
                        criterion: nn.Module, 
                        device: torch.device, 
                        epoch: int, 
                        trainer: 'DistributedTrainer', 
                        **kwargs) -> Dict[str, Any]:
    """
    Custom training step function - defined training logic.
    The framework handles all the Horovod setup, you just write the training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Your custom training logic here
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # You could add custom regularization, gradient clipping, etc.
        # Example: L2 regularization
        l2_lambda = kwargs.get('l2_lambda', 0.01)
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        loss.backward()
        
        # Example: gradient clipping
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        # Custom logging every N batches
        if batch_idx % 50 == 0 and trainer.is_master():
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Calculate average loss across all workers
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss = trainer.allreduce(avg_loss_tensor, average=True).item()
    
    return {'loss': avg_loss, 'num_batches': num_batches}


def main_custom() -> None:
    """
    Advanced usage - provide your own training step function with checkpointing.
    """
    # Create trainer with enhanced checkpointing
    trainer = DistributedTrainer(
        auto_scale_lr=True,
        verbose=True,
        checkpoint_interval=8,
        keep_last_n_checkpoints=2,
        save_best_only=False
    )
    
    # Create data
    dataset = create_synthetic_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train using framework with custom training step
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50,
        train_step_fn=custom_training_step,  # Your custom training logic
        l2_lambda=0.01,  # Custom parameters passed to your function
        max_grad_norm=1.0
    )
    
    if trainer.is_master():
        print("Custom training completed!")
        print(f"Final loss: {history['train_loss'][-1]:.4f}")


def manual_checkpoint_example() -> None:
    """
    Example of manual checkpoint management
    """
    trainer = DistributedTrainer(
        checkpoint_interval=0,  # Disable automatic checkpointing
        verbose=True
    )
    
    # Create data and model
    dataset = create_synthetic_data()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare components
    model = trainer.prepare_model(model)
    optimizer = trainer.prepare_optimizer(optimizer, model)
    train_loader = trainer.prepare_dataloader(train_loader)
    
    # Manual training loop with custom checkpoint logic
    for epoch in range(20):
        train_loader.sampler.set_epoch(epoch)
        
        # Simple training epoch
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(trainer.device), target.to(trainer.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Manual checkpoint at specific epochs
        if (epoch + 1) % 7 == 0:  # Save every 7 epochs
            checkpoint_path = trainer.save_checkpoint(
                epoch + 1, 
                model, 
                optimizer, 
                {'loss': avg_loss},
                is_best=(epoch + 1 == 14)  # Mark epoch 14 as best
            )
            if trainer.is_master() and checkpoint_path:
                print(f"Manual checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")
        
        if trainer.is_master() and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/20, Loss: {avg_loss:.4f}")
    
    if trainer.is_master():
        print("Manual checkpoint training completed!")


def s3_download_example() -> None:
    """
    Example of downloading checkpoints from S3
    """
    trainer = DistributedTrainer(
        s3_bucket_arn="arn:aws:s3:::my-training-checkpoints",
        verbose=True
    )
    
    if trainer.is_master():
        print("=== S3 Checkpoint Download Example ===")
        
        # List available checkpoints
        checkpoints = trainer.list_s3_checkpoints()
        print(f"Available S3 checkpoints: {checkpoints}")
        
        if checkpoints:
            # Download a specific checkpoint
            s3_key = checkpoints[0]  # First checkpoint
            local_path = trainer.download_checkpoint_from_s3(s3_key)
            
            if local_path:
                print(f"Downloaded checkpoint: {local_path}")
                
                # You could now use this for training
                # model = SimpleNet(...)
                # optimizer = optim.Adam(...)
                # trainer.load_checkpoint(local_path, model, optimizer)
            else:
                print("Failed to download checkpoint")


if __name__ == "__main__":
    print("=== Basic Usage (Framework handles everything) ===")
    main_simple()
    
    print("\n=== Training with S3 Backup ===")
    # main_with_s3_backup()  # Uncomment if you have S3 bucket configured
    
    print("\n=== Training with Resume Functionality ===")
    main_with_resume()
    
    print("\n=== Best Checkpoint Only ===")
    main_best_checkpoint_only()
    
    print("\n=== Custom Training Step with Checkpointing ===")
    main_custom()
    
    print("\n=== Manual Checkpoint Management ===")
    manual_checkpoint_example()
    
    print("\n=== S3 Download Example ===")
    # s3_download_example()  # Uncomment if you have S3 bucket with checkpoints
