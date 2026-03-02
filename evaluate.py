import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import Data
from typing import Dict, Any

def evaluate_model(model: torch.nn.Module, data: Data) -> Dict[str, Any]:
    """
    Evaluate the model on a specific snapshot.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits).squeeze()
        
        mask = data.train_mask
        y_true = data.y[mask].cpu().numpy()
        y_probs = probs[mask].cpu().numpy()
        y_pred = (y_probs > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_probs)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def compute_rpd(auc_orig: float, auc_shifted: float) -> float:
    """
    Compute Relative Performance Degradation (RPD).
    RPD = (AUC_original - AUC_shifted) / AUC_original
    """
    if auc_orig == 0:
        return 0.0
    return (auc_orig - auc_shifted) / auc_orig
