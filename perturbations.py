import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, List
import copy

class StructuralPerturbator:
    """
    Structural Regime Shift Engine.
    Implements adversarial structural shifts in graphs.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def transaction_fragmentation(self, data: Data, fragmentation_rate: float, k: int = 3) -> Data:
        """
        Simulate laundering by splitting large transaction flows.
        Procedure:
        1. Identify high-degree nodes (potential laundering hubs).
        2. Remove a fraction of their outgoing edges.
        3. Replace each removed edge with k smaller flows via synthetic relay nodes.
        """
        if fragmentation_rate <= 0:
            return data

        perturbed_data = copy.deepcopy(data)
        edge_index = perturbed_data.edge_index
        num_nodes = perturbed_data.num_nodes
        num_edges = edge_index.shape[1]

        # 1. Identify high-degree nodes
        out_degrees = torch.zeros(num_nodes)
        out_degrees.scatter_add_(0, edge_index[0], torch.ones(num_edges))
        
        # Select nodes to fragment (top 10% by degree or similar)
        num_to_fragment = int(fragmentation_rate * num_nodes)
        if num_to_fragment == 0:
            return data
            
        target_nodes = torch.argsort(out_degrees, descending=True)[:num_to_fragment]
        
        new_edges = []
        new_features = []
        new_labels = []
        
        current_node_count = num_nodes
        
        # Mask for edges to keep
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        
        for u in target_nodes:
            # Edges from u
            u_edges_idx = (edge_index[0] == u).nonzero(as_tuple=True)[0]
            if len(u_edges_idx) == 0:
                continue
            
            # Remove original edges
            keep_mask[u_edges_idx] = False
            
            for idx in u_edges_idx:
                v = edge_index[1, idx]
                
                # Flow splitting: u -> relays -> v
                for _ in range(k):
                    relay_node = current_node_count
                    # Feature for relay: mean of u and v (simple approximation)
                    relay_feat = (perturbed_data.x[u] + perturbed_data.x[v]) / 2.0
                    new_features.append(relay_feat)
                    new_labels.append(-1) # Synthetic nodes are unknown
                    
                    # New edges: u -> relay, relay -> v
                    new_edges.append([u.item(), relay_node])
                    new_edges.append([relay_node, v.item()])
                    
                    current_node_count += 1

        # Reconstruct perturbed graph
        final_edge_index = edge_index[:, keep_mask]
        if new_edges:
            final_edge_index = torch.cat([final_edge_index, torch.tensor(new_edges).T], dim=1)
            final_x = torch.cat([perturbed_data.x, torch.stack(new_features)], dim=0)
            final_y = torch.cat([perturbed_data.y, torch.tensor(new_labels)], dim=0)
        else:
            final_x = perturbed_data.x
            final_y = perturbed_data.y

        perturbed_data.x = final_x
        perturbed_data.edge_index = final_edge_index
        perturbed_data.y = final_y
        perturbed_data.train_mask = (final_y != -1)
        
        return perturbed_data

    def motif_camouflage(self, data: Data, camouflage_intensity: float) -> Data:
        """
        Hide illicit nodes within benign structural patterns by injecting edges
        between illicit nodes and benign clusters.
        """
        if camouflage_intensity <= 0:
            return data

        perturbed_data = copy.deepcopy(data)
        y = perturbed_data.y
        illicit_indices = (y == 1).nonzero(as_tuple=True)[0]
        benign_indices = (y == 0).nonzero(as_tuple=True)[0]

        if len(illicit_indices) == 0 or len(benign_indices) == 0:
            return data

        num_edges_to_add = int(camouflage_intensity * len(illicit_indices))
        if num_edges_to_add == 0:
            return data

        # Randomly connect illicit nodes to benign ones
        new_src = illicit_indices[torch.randint(0, len(illicit_indices), (num_edges_to_add,))]
        new_dst = benign_indices[torch.randint(0, len(benign_indices), (num_edges_to_add,))]
        
        new_edges = torch.stack([new_src, new_dst], dim = 0)
        perturbed_data.edge_index = torch.cat([perturbed_data.edge_index, new_edges], dim=1)
        
        return perturbed_data
