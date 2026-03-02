import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import os

class EllipticDataLoader:
    """
    Data loader for the Elliptic Bitcoin Dataset.
    Constructs temporal graph snapshots per time step.
    
    Mathematical Formulation:
    - Nodes: V_t = {v | timestep(v) = t}
    - Features: X_t \in \mathbb{R}^{|V_t| \times 166}
    - Edges: E_t = {(u, v) | u, v \in V_t}
    - Labels: y_t \in \{0, 1, \text{unknown}\}
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.features_path = os.path.join(data_dir, 'elliptic_txs_features.csv')
        self.edges_path = os.path.join(data_dir, 'elliptic_txs_edgelist.csv')
        self.classes_path = os.path.join(data_dir, 'elliptic_txs_classes.csv')

    def load_snapshots(self) -> Dict[int, Data]:
        """
        Loads the dataset and organizes it into temporal snapshots.
        Returns:
            Dict[int, Data]: Dictionary mapping timestep to PyG Data object.
        """
        print("Loading features...")
        df_features = pd.read_csv(self.features_path, header=None)
        # First column is txId, second is timestep, rest are features (165 features)
        df_features.columns = ['txId', 'timestep'] + [f'f{i}' for i in range(165)]
        
        print("Loading classes...")
        df_classes = pd.read_csv(self.classes_path)
        # Mapper for classes: '1' -> 1 (illicit), '2' -> 0 (licit), 'unknown' -> -1
        class_map = {'1': 1, '2': 0, 'unknown': -1}
        df_classes['label'] = df_classes['class'].map(class_map)
        
        print("Loading edges...")
        df_edges = pd.read_csv(self.edges_path)
        
        # Merge features and classes
        df_merged = pd.merge(df_features, df_classes[['txId', 'label']], on='txId')
        
        snapshots = {}
        timesteps = sorted(df_merged['timestep'].unique())
        
        for t in timesteps:
            print(f"Processing timestep {t}...")
            # Nodes at this timestep
            nodes_t = df_merged[df_merged['timestep'] == t]
            node_ids = nodes_t['txId'].values
            
            # Map original txId to 0-indexed IDs for this snapshot
            id_map = {old_id: i for i, old_id in enumerate(node_ids)}
            
            # Features (excluding txId and timestep)
            x = torch.tensor(nodes_t.iloc[:, 2:-1].values, dtype=torch.float)
            y = torch.tensor(nodes_t['label'].values, dtype=torch.long)
            
            # Edges at this timestep (only those connecting nodes in this snapshot)
            mask_src = df_edges['txId1'].isin(node_ids)
            mask_dst = df_edges['txId2'].isin(node_ids)
            edges_t = df_edges[mask_src & mask_dst]
            
            edge_index = torch.tensor([
                [id_map[src] for src in edges_t['txId1']],
                [id_map[dst] for dst in edges_t['txId2']]
            ], dtype=torch.long)
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            # Add a mask for nodes with known labels (0 or 1)
            data.train_mask = (y != -1)
            
            snapshots[t] = data
            
        print(f"Loaded {len(snapshots)} snapshots.")
        return snapshots

if __name__ == "__main__":
    # Quick test
    loader = EllipticDataLoader('data')
    snapshots = loader.load_snapshots()
    t1 = snapshots[1]
    print(f"Snapshot 1: Nodes={t1.num_nodes}, Edges={t1.num_edges}, Labeled={t1.train_mask.sum().item()}")
