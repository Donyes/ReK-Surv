"""
Hyperparameter Sensitivity Analysis for ReK-Surv

This script performs systematic hyperparameter sensitivity experiments:
- Stage A: Capacity (spline_order × grid_size)
- Stage B: Regularization (L1 τ × L2 wd) - Log scale
- Stage C: Width sensitivity
- Stage D: Depth sensitivity

Default dataset: METABRIC
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import time
import warnings
warnings.filterwarnings('ignore')

from models import KAN
from utils import negative_log_likelihood, proximal_l1, load_metabric

# Configuration
SEEDS = [42, 123, 456]  # Multiple seeds for averaging

BEST_PARAMS = {
    'grid_size': 1,
    'spline_order': 3,
    'tau': 0.25,
    'wd': 0.5,
    'depth': 2,
    'width': 1,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_seed(X_train_full, X_test, T_train_full, T_test, E_train_full, E_test,
                      grid_size, spline_order, tau, wd, depth, width,
                      seed, epochs=300, patience=50):
    """Train with a single seed, return test C-index."""
    set_seed(seed)
    
    # Split train/val
    E = E_train_full
    stratify_by = []
    for e, t in zip(E, T_train_full):
        if e == 1:
            time_bin = np.digitize(t, bins=np.percentile(T_train_full[E == 1], [33, 66]))
            stratify_by.append(f"event_{time_bin}")
        else:
            stratify_by.append("non_event")
    
    X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
        X_train_full, T_train_full, E_train_full,
        test_size=0.2, stratify=stratify_by, random_state=seed
    )
    
    # Sort by time
    sort_train = np.argsort(-T_train)
    X_train, T_train, E_train = X_train[sort_train], T_train[sort_train], E_train[sort_train]
    sort_val = np.argsort(-T_val)
    X_val, T_val, E_val = X_val[sort_val], T_val[sort_val], E_val[sort_val]
    
    # Build model
    n_features = X_train.shape[1]
    hidden = [width] * depth
    layers = [n_features] + hidden + [1]
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    E_train_t = torch.tensor(E_train, dtype=torch.float32)
    
    model = KAN(layers, grid_size=grid_size, spline_order=spline_order)
    n_params = model.count_parameters()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.5, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6)
    
    best_val_c = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = negative_log_likelihood(model(X_train_t), E_train_t)
        if torch.isnan(loss):
            return np.nan, n_params, np.nan
        loss.backward()
        optimizer.step()
        proximal_l1(model, tau, optimizer.param_groups[0]['lr'])
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).cpu().numpy().flatten()
            try:
                val_c = concordance_index(T_val, -val_out, E_val)
            except:
                val_c = 0.5
        
        if val_c > best_val_c:
            best_val_c = val_c
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        scheduler.step(val_c)
        if no_improve >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            _ = model(X_test_t)
        inference_time = (time.perf_counter() - start) / 100 * 1000
        
        test_out = model(X_test_t).cpu().numpy().flatten()
        try:
            test_c = concordance_index(T_test, -test_out, E_test)
        except:
            test_c = 0.5
    
    return test_c, n_params, inference_time


def train_config(X_train_full, X_test, T_train_full, T_test, E_train_full, E_test,
                 grid_size, spline_order, tau, wd, depth, width, verbose=False):
    """Train with multiple seeds and return averaged results."""
    c_indices = []
    n_params = None
    inf_time = None
    
    for seed in SEEDS:
        c, params, t = train_single_seed(
            X_train_full, X_test, T_train_full, T_test, E_train_full, E_test,
            grid_size, spline_order, tau, wd, depth, width, seed
        )
        if not np.isnan(c):
            c_indices.append(c)
        if n_params is None:
            n_params = params
            inf_time = t
    
    if len(c_indices) == 0:
        return np.nan, np.nan, n_params, inf_time
    
    mean_c = np.mean(c_indices)
    std_c = np.std(c_indices) if len(c_indices) > 1 else 0.0
    
    if verbose:
        print(f"   C-index: {mean_c:.4f} ± {std_c:.4f}, Params: {n_params}")
    
    return mean_c, std_c, n_params, inf_time


def stage_a_capacity(X_train, X_test, T_train, T_test, E_train, E_test):
    """Stage A: Capacity (order × grid) = 3×3 = 9 configs"""
    print("\n" + "=" * 60)
    print("Stage A: Capacity (Spline Order × Grid Size)")
    print("=" * 60)
    
    order_star = BEST_PARAMS['spline_order']
    grid_star = BEST_PARAMS['grid_size']
    
    orders = [max(1, order_star - 1), order_star, order_star + 1]
    grids = [max(1, grid_star - 1), grid_star, grid_star + 1]
    
    results = []
    for order in orders:
        for grid in grids:
            print(f"  order={order}, grid={grid}", end=" -> ")
            mean_c, std_c, n_params, inf_time = train_config(
                X_train, X_test, T_train, T_test, E_train, E_test,
                grid, order, BEST_PARAMS['tau'], BEST_PARAMS['wd'],
                BEST_PARAMS['depth'], BEST_PARAMS['width'], verbose=True
            )
            results.append({
                'spline_order': order, 'grid_size': grid,
                'test_c_index': mean_c, 'test_c_std': std_c,
                'n_params': n_params, 'inference_ms': inf_time
            })
    
    return pd.DataFrame(results)


def stage_b_regularization(X_train, X_test, T_train, T_test, E_train, E_test):
    """Stage B: Regularization (τ × wd) = 5×5 = 25 configs (log scale)"""
    print("\n" + "=" * 60)
    print("Stage B: Regularization (L1 τ × L2 wd) - Log Scale")
    print("=" * 60)
    
    taus = [0, 0.01, 0.1, 0.5, 1.0]
    wds = [0, 0.01, 0.1, 0.5, 1.0]
    
    results = []
    for tau in taus:
        for wd in wds:
            print(f"  τ={tau:.2f}, wd={wd:.2f}", end=" -> ")
            mean_c, std_c, _, _ = train_config(
                X_train, X_test, T_train, T_test, E_train, E_test,
                BEST_PARAMS['grid_size'], BEST_PARAMS['spline_order'],
                tau, wd, BEST_PARAMS['depth'], BEST_PARAMS['width'], verbose=True
            )
            results.append({
                'tau': tau, 'wd': wd,
                'test_c_index': mean_c, 'test_c_std': std_c
            })
    
    return pd.DataFrame(results)


def stage_c_width(X_train, X_test, T_train, T_test, E_train, E_test):
    """Stage C: Width = 3 configs"""
    print("\n" + "=" * 60)
    print("Stage C: Width Sensitivity")
    print("=" * 60)
    
    widths = [1, 4, 8]
    results = []
    
    for width in widths:
        print(f"  width={width}", end=" -> ")
        mean_c, std_c, n_params, inf_time = train_config(
            X_train, X_test, T_train, T_test, E_train, E_test,
            BEST_PARAMS['grid_size'], BEST_PARAMS['spline_order'],
            BEST_PARAMS['tau'], BEST_PARAMS['wd'],
            BEST_PARAMS['depth'], width, verbose=True
        )
        results.append({
            'width': width, 'test_c_index': mean_c, 'test_c_std': std_c,
            'n_params': n_params, 'inference_ms': inf_time
        })
    
    return pd.DataFrame(results)


def stage_d_depth(X_train, X_test, T_train, T_test, E_train, E_test):
    """Stage D: Depth = 3 configs"""
    print("\n" + "=" * 60)
    print("Stage D: Depth Sensitivity")
    print("=" * 60)
    
    depth_star = BEST_PARAMS['depth']
    depths = [max(1, depth_star - 1), depth_star, depth_star + 1]
    results = []
    
    for depth in depths:
        print(f"  depth={depth}", end=" -> ")
        mean_c, std_c, n_params, inf_time = train_config(
            X_train, X_test, T_train, T_test, E_train, E_test,
            BEST_PARAMS['grid_size'], BEST_PARAMS['spline_order'],
            BEST_PARAMS['tau'], BEST_PARAMS['wd'],
            depth, BEST_PARAMS['width'], verbose=True
        )
        results.append({
            'depth': depth, 'test_c_index': mean_c, 'test_c_std': std_c,
            'n_params': n_params, 'inference_ms': inf_time
        })
    
    return pd.DataFrame(results)


def plot_results(df_a, df_b, df_c, df_d, save_path='sensitivity_results.png'):
    """Plot 2×2 sensitivity analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ReK-Surv Hyperparameter Sensitivity (METABRIC)', fontsize=16, fontweight='bold')
    
    # (A) Capacity heatmap
    ax1 = axes[0, 0]
    pivot = df_a.pivot(index='spline_order', columns='grid_size', values='test_c_index')
    pivot_std = df_a.pivot(index='spline_order', columns='grid_size', values='test_c_std')
    annot = pivot.copy().astype(str)
    for i in pivot.index:
        for j in pivot.columns:
            annot.loc[i, j] = f"{pivot.loc[i,j]:.3f}\n±{pivot_std.loc[i,j]:.3f}"
    sns.heatmap(pivot, annot=annot, fmt='', cmap='RdYlGn', ax=ax1,
                vmin=max(0.5, pivot.values.min() - 0.02), vmax=min(1.0, pivot.values.max() + 0.01),
                cbar_kws={'label': 'C-index'}, annot_kws={'size': 10})
    ax1.set_title('(A) Capacity: Spline Order × Grid Size', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Spline Order')
    
    # (B) Regularization heatmap
    ax2 = axes[0, 1]
    pivot_b = df_b.pivot(index='tau', columns='wd', values='test_c_index')
    pivot_b_std = df_b.pivot(index='tau', columns='wd', values='test_c_std')
    annot_b = pivot_b.copy().astype(str)
    for i in pivot_b.index:
        for j in pivot_b.columns:
            annot_b.loc[i, j] = f"{pivot_b.loc[i,j]:.3f}\n±{pivot_b_std.loc[i,j]:.3f}"
    sns.heatmap(pivot_b, annot=annot_b, fmt='', cmap='RdYlGn', ax=ax2,
                vmin=max(0.5, pivot_b.values.min() - 0.02), vmax=min(1.0, pivot_b.values.max() + 0.01),
                cbar_kws={'label': 'C-index'}, annot_kws={'size': 9})
    ax2.set_title('(B) Regularization: L1 (τ) × L2 (wd)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Weight Decay (L2)')
    ax2.set_ylabel('Proximal L1 (τ)')
    
    # (C) Width bar chart
    ax3 = axes[1, 0]
    x = np.arange(len(df_c))
    w = 0.35
    ax3.bar(x - w/2, df_c['test_c_index'], w, yerr=df_c['test_c_std'], capsize=5,
            label='C-index', color='steelblue', alpha=0.85)
    ax3.set_ylabel('Test C-index', color='steelblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3.set_ylim([max(0.5, df_c['test_c_index'].min() - 0.1), 1.05])
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + w/2, df_c['n_params'], w, label='Params', color='coral', alpha=0.85)
    ax3_twin.set_ylabel('# Parameters', color='coral')
    ax3_twin.tick_params(axis='y', labelcolor='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'width={w}' for w in df_c['width']])
    ax3.set_title('(C) Width Sensitivity', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # (D) Depth bar chart
    ax4 = axes[1, 1]
    x = np.arange(len(df_d))
    ax4.bar(x - w/2, df_d['test_c_index'], w, yerr=df_d['test_c_std'], capsize=5,
            label='C-index', color='steelblue', alpha=0.85)
    ax4.set_ylabel('Test C-index', color='steelblue')
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4.set_ylim([max(0.5, df_d['test_c_index'].min() - 0.1), 1.05])
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + w/2, df_d['n_params'], w, label='Params', color='coral', alpha=0.85)
    ax4_twin.set_ylabel('# Parameters', color='coral')
    ax4_twin.tick_params(axis='y', labelcolor='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'depth={d}' for d in df_d['depth']])
    ax4.set_title('(D) Depth Sensitivity', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {save_path}")


def main():
    print("=" * 60)
    print("ReK-Surv Hyperparameter Sensitivity Analysis")
    print("Dataset: METABRIC")
    print(f"Seeds: {SEEDS}")
    print(f"Best params: {BEST_PARAMS}")
    print("=" * 60)
    
    # Load data
    print("\nLoading METABRIC dataset...")
    X_train, X_test, T_train, T_test, E_train, E_test = load_metabric('data/metabric.h5')
    print(f"Train: {len(X_train)} ({int(E_train.sum())} events)")
    print(f"Test:  {len(X_test)} ({int(E_test.sum())} events)")
    
    # Run experiments
    df_a = stage_a_capacity(X_train, X_test, T_train, T_test, E_train, E_test)
    df_b = stage_b_regularization(X_train, X_test, T_train, T_test, E_train, E_test)
    df_c = stage_c_width(X_train, X_test, T_train, T_test, E_train, E_test)
    df_d = stage_d_depth(X_train, X_test, T_train, T_test, E_train, E_test)
    
    # Save results
    df_a.to_csv('stage_a_capacity.csv', index=False)
    df_b.to_csv('stage_b_regularization.csv', index=False)
    df_c.to_csv('stage_c_width.csv', index=False)
    df_d.to_csv('stage_d_depth.csv', index=False)
    
    # Plot
    plot_results(df_a, df_b, df_c, df_d)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
