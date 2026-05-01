"""
Training script for ReK-Surv model.

Usage:
    python train.py --dataset metabric --epochs 500
"""

import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from lifelines.utils import concordance_index

from models import KAN
from utils import (
    negative_log_likelihood,
    proximal_l1,
    load_metabric,
    load_support,
    load_whas,
    load_rgbsg,
    load_hlb,
    prepare_tensors,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    X_train: torch.Tensor,
    E_train: torch.Tensor,
    X_val: torch.Tensor,
    T_val: np.ndarray,
    E_val: np.ndarray,
    model: KAN,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    tau: float = 0.1,
    epochs: int = 500,
    patience: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Train the ReK-Surv model.
    
    Args:
        X_train, E_train: Training data tensors
        X_val, T_val, E_val: Validation data
        model: KAN model instance
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        tau: L1 regularization strength
        epochs: Maximum training epochs
        patience: Early stopping patience
        verbose: Print training progress
    
    Returns:
        Dictionary with training history and best metrics
    """
    best_c_index = 0
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_c_index': []}
    
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = negative_log_likelihood(outputs, E_train)
        
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch}. Stopping.")
            break
        
        loss.backward()
        optimizer.step()
        
        # Apply proximal L1 regularization
        current_lr = optimizer.param_groups[0]['lr']
        proximal_l1(model, tau, current_lr)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).cpu().numpy().flatten()
            val_c_index = concordance_index(T_val, -val_outputs, E_val)
        
        history['train_loss'].append(loss.item())
        history['val_c_index'].append(val_c_index)
        
        # Early stopping check
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        scheduler.step(val_c_index)
        
        if verbose and epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val C-index: {val_c_index:.4f} | LR: {lr:.6f}")
        
        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return {
        'history': history,
        'best_c_index': best_c_index,
        'final_epoch': epoch,
    }


def main():
    parser = argparse.ArgumentParser(description='Train ReK-Surv model')
    parser.add_argument('--dataset', type=str, default='metabric', help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data file')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.5, help='Weight decay (L2)')
    parser.add_argument('--tau', type=float, default=0.25, help='L1 regularization')
    parser.add_argument('--grid_size', type=int, default=1, help='B-spline grid size')
    parser.add_argument('--spline_order', type=int, default=3, help='B-spline order')
    parser.add_argument('--depth', type=int, default=2, help='Network depth')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Model save path')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    dataset = args.dataset.strip().lower()
    loaders = {
        'metabric': load_metabric,
        'support': load_support,
        'whas': load_whas,
        'rgbsg': load_rgbsg,
        'hlb': load_hlb,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from: {', '.join(loaders.keys())}")

    if args.data_path is None:
        if dataset == 'hlb':
            args.data_path = 'data/hlb.xlsx'
        else:
            args.data_path = f"data/{dataset}.h5"

    X_train, X_test, T_train, T_test, E_train, E_test = loaders[dataset](args.data_path)
    
    print(f"Train: {len(X_train)} samples ({int(E_train.sum())} events)")
    print(f"Test:  {len(X_test)} samples ({int(E_test.sum())} events)")
    print(f"Features: {X_train.shape[1]}")
    
    # Prepare tensors
    tensors = prepare_tensors(X_train, X_test, T_train, T_test, E_train, E_test)
    
    # Build model
    n_features = X_train.shape[1]
    hidden_layers = [1] * args.depth
    layers = [n_features] + hidden_layers + [1]
    
    model = KAN(
        layers,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    )
    print(f"Model parameters: {model.count_parameters()}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6
    )
    
    # Train
    print("\nTraining...")
    result = train(
        tensors['X_train'], tensors['E_train'],
        tensors['X_test'], T_test, E_test,
        model, optimizer, scheduler,
        tau=args.tau,
        epochs=args.epochs,
        patience=args.patience,
    )
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(tensors['X_train']).numpy().flatten()
        test_pred = model(tensors['X_test']).numpy().flatten()
        
        train_c = concordance_index(T_train, -train_pred, E_train)
        test_c = concordance_index(T_test, -test_pred, E_test)
    
    print(f"\nFinal Results:")
    print(f"  Train C-index: {train_c:.4f}")
    print(f"  Test C-index:  {test_c:.4f}")
    
    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == '__main__':
    main()
