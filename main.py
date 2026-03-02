import torch
from data_loader import EllipticDataLoader
from model import RollingStaticGCN
from perturbations import StructuralPerturbator
from train import train_model
from evaluate import evaluate_model, compute_rpd
import pandas as pd

def run_experiment():
    # 1. Load Data
    loader = EllipticDataLoader('data')
    snapshots = loader.load_snapshots()
    
    # 2. Define Experiment Protocol
    # Train on first 30 time steps
    train_steps = list(range(1, 31))
    # Test on future time steps (e.g., 35 to 49)
    test_steps = list(range(35, 50))
    
    # 3. Initialize Model
    in_channels = snapshots[1].num_features
    model = RollingStaticGCN(in_channels=in_channels, hidden_channels=64, out_channels=1)
    
    # 4. Train Baseline
    print("\n--- Training Baseline Model ---")
    model = train_model(model, snapshots, train_steps, epochs=50)
    
    # 5. Evaluate Clean Performance
    print("\n--- Evaluating on Clean Test Slices ---")
    results_orig = []
    for t in test_steps:
        res = evaluate_model(model, snapshots[t])
        results_orig.append(res['auc'])
    avg_auc_orig = sum(results_orig) / len(results_orig)
    
    # 6. Apply Perturbations and Re-evaluate
    perturbator = StructuralPerturbator(seed=42)
    
    # A. Transaction Fragmentation
    print("\n--- Evaluating Fragmented Test Slices ---")
    results_frag = []
    for t in test_steps:
        perturbed_data = perturbator.transaction_fragmentation(snapshots[t], fragmentation_rate=0.2)
        res = evaluate_model(model, perturbed_data)
        results_frag.append(res['auc'])
    avg_auc_frag = sum(results_frag) / len(results_frag)
    
    # B. Motif Camouflage
    print("\n--- Evaluating Camouflaged Test Slices ---")
    results_camo = []
    for t in test_steps:
        perturbed_data = perturbator.motif_camouflage(snapshots[t], camouflage_intensity=0.5)
        res = evaluate_model(model, perturbed_data)
        results_camo.append(res['auc'])
    avg_auc_camo = sum(results_camo) / len(results_camo)
    
    # 7. Final Results Table
    data = {
        'Scenario': ['Clean', 'Transaction Fragmentation', 'Motif Camouflage'],
        'Avg AUC': [avg_auc_orig, avg_auc_frag, avg_auc_camo],
        'RPD (%)': [
            0.0, 
            compute_rpd(avg_auc_orig, avg_auc_frag) * 100,
            compute_rpd(avg_auc_orig, avg_auc_camo) * 100
        ]
    }
    
    df_results = pd.DataFrame(data)
    print("\n" + "="*40)
    print("STRUCTURAL REGIME SHIFT BENCHMARK V1")
    print("="*40)
    print(df_results.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    run_experiment()
