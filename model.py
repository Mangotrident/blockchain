import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple

class RollingStaticGCN(nn.Module):
    """
    Baseline Model: Rolling Static GCN.
    Applied independently per time step.
    
    Mathematical Formulation:
    H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})
    where \tilde{A} = A + I, and \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super(RollingStaticGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
        # Proper initialization (PyG usually does this, but being explicit)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns:
            torch.Tensor: Logits for the binary classification task.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class WeightedBCELoss(nn.Module):
    """
    Custom BCE loss with positive weight to handle class imbalance.
    In Elliptic dataset, illicit (1) is much rarer than licit (0).
    """
    def __init__(self, pos_weight: float):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # Only compute loss on labeled nodes
        masked_logits = logits[mask].squeeze()
        masked_labels = labels[mask].float()
        
        # Using binary_cross_entropy_with_logits for numerical stability
        loss = F.binary_cross_entropy_with_logits(
            masked_logits, 
            masked_labels, 
            pos_weight=self.pos_weight.to(logits.device)
        )
        return loss
