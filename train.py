import torch
from torch.optim import Optimizer
from model import RollingStaticGCN, WeightedBCELoss
from torch_geometric.data import Data
from typing import Dict, List
import numpy as np

def train_model(
    model: RollingStaticGCN, 
    snapshots: Dict[int, Data], 
    train_steps: List[int],
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    pos_weight: float = 10.0
) -> RollingStaticGCN:
    """
    Train a GCN on a sequence of temporal snapshots.
    We iterate through the specified training time steps.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for t in train_steps:
            data = snapshots[t]
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y, data.train_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}, Loss: {total_loss / len(train_steps):.4f}")
            
    return model
