"""
Simple Neural Network for DTO Framework Examples

A basic fully connected neural network for demonstration purposes.
Used in the distributed training examples to show how the DTO framework works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    A simple fully connected neural network with customizable architecture.
    
    This is a basic feedforward network with:
    - Input layer
    - Hidden layer with ReLU activation
    - Output layer with optional dropout
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of neurons in the hidden layer
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout probability (default: 0.2)
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_rate: float = 0.2):
        super(SimpleNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Input layer to first hidden layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # First hidden layer to second hidden layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer to output
        x = self.fc3(x)
        
        return x
