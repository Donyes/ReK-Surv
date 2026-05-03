"""
Training entry point for the dynamic HLB survival model.

Usage:
    python train_dynamic.py --epochs 200 --batch_size 32
"""
from __future__ import annotations
import argparse
import random
import json
import pathlib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ReKDynSurv
from utils.dynamic_data import (
    load_processed, fit_scalers, apply_scalers,
    TreeSequenceDataset, collate_trees, stratified_kfold_indices,
)
from utils.dynamic_loss import total_loss, proximal_l1
from utils.dynamic_metrics import evaluate_all_landmarks


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: dict, device):
    batch['env_periods'] = [ep.to(device) for ep in batch['env_periods']]
    batch['static'] = batch['static'].to(device)
    batch['T'] = batch['T'].to(device)
    batch['E'] = batch['E'].to(device)
    batch['ct_seq'] = batch['ct_seq'].to(device)
    batch['ct_mask'] = batch['ct_mask'].to(device)
    return batch


def mean_std_over_folds(fold_results):
    """fold_results: list of dict {(s,d): {'cindex':.., 'brier':..}}"""
    if not fold_results:
        return {}
    keys = fold_results[0].keys()
    agg = {}
    for k in keys:
        cidxs = [f[k]['cindex'] for f in fold_results if not np.isnan(f[k]['cindex'])]
        briers = [f[k]['brier'] for f in fold_results if not np.isnan(f[k]['brier'])]
        agg[k] = {
            'cindex_mean': float(np.mean(cidxs)) if cidxs else float('nan'),
            'cindex_std': float(np.std(cidxs)) if cidxs else float('nan'),
            'brier_mean': float(np.mean(briers)) if briers else float('nan'),
            'brier_std': float(np.std(briers)) if briers else float('nan'),
            'n_folds': len(cidxs),
        }
    return agg


def train_one_fold(fold_idx, train_idx, val_idx, processed, args, device):
    env_scaler, static_scaler, medians = fit_scalers(
        processed['env_daily_raw'], processed['static_raw'], train_idx,
    )
    env_scaled, static_scaled = apply_scalers(
        processed['env_daily_raw'], processed['static_raw'],
        env_scaler, static_scaler, medians,
    )

    train_ds = TreeSequenceDataset(processed, train_idx, env_scaled, static_scaled)
    val_ds = TreeSequenceDataset(processed, val_idx, env_scaled, static_scaled)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_trees, drop_last=False)

    d_env = processed['env_daily_raw'].shape[1]
    d_static = processed['static_raw'].shape[1]
    K = processed['K']

    model = ReKDynSurv(
        d_env=d_env, d_static=d_static, K=K,
        d_h=args.d_h, n_heads=args.n_heads, dropout=args.dropout,
        kan_grid_size=args.kan_grid_size, kan_spline_order=args.kan_spline_order,
        tail_mask_p=args.tail_mask_p,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fold {fold_idx}] model params = {n_params}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6,
    )

    # target C-index cell to track for early stopping
    track_cell = (args.track_landmark, args.track_horizon)

    best_metric = -float('inf')
    best_state = None
    best_results = None
    no_improve = 0

    # pre-compute train T/E arrays for sksurv IPCW baseline
    T_train_np = processed['T'][train_idx]
    E_train_np = processed['E'][train_idx]

    for epoch in range(args.epochs):
        model.train()
        train_stats = {'total': 0.0, 'haz_nll': 0.0, 'rank': 0.0, 'aux': 0.0, 'ent_reg': 0.0}
        n_batches = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch['env_periods'], batch['static'])
            loss_dict = total_loss(outputs, batch,
                                   alpha=(args.alpha_haz, args.alpha_rank,
                                          args.alpha_aux, args.alpha_ent),
                                   sigma_rank=args.sigma_rank)
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if args.tau_l1 > 0:
                lr_now = optimizer.param_groups[0]['lr']
                # target: only the per-period encoder's first conv layer + kan layers
                tgt_params = list(model.period_encoder.block1.parameters()) \
                    + list(model.fusion.parameters())
                proximal_l1(tgt_params, args.tau_l1, lr_now)

            for k, v in loss_dict.items():
                train_stats[k] += float(v.item() if torch.is_tensor(v) else v)
            n_batches += 1
        for k in train_stats:
            train_stats[k] /= max(1, n_batches)

        # validation
        model.eval()
        val_results = evaluate_all_landmarks(
            model, val_ds, T_train_np, E_train_np,
            landmarks=args.landmarks, horizons=args.horizons,
            batch_size=args.batch_size, device=device,
        )
        track_val = val_results.get(track_cell, {'cindex': float('nan')})['cindex']
        if np.isnan(track_val):
            # fallback: average over all cells
            cs = [v['cindex'] for v in val_results.values() if not np.isnan(v['cindex'])]
            track_val = float(np.mean(cs)) if cs else 0.0

        scheduler.step(track_val)

        if track_val > best_metric:
            best_metric = track_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_results = val_results
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"[fold {fold_idx}] ep {epoch:3d} "
                f"| total {train_stats['total']:.3f} "
                f"haz {train_stats['haz_nll']:.3f} "
                f"rank {train_stats['rank']:.3f} "
                f"aux {train_stats['aux']:.3f} "
                f"| val C@{track_cell} {track_val:.4f} "
                f"| best {best_metric:.4f} | lr {lr:.2e}"
            )

        if no_improve >= args.patience:
            print(f"[fold {fold_idx}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/hlb_processed.pt')
    parser.add_argument('--out_dir', default='runs/hlb_dynamic')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    # model
    parser.add_argument('--d_h', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--kan_grid_size', type=int, default=5)
    parser.add_argument('--kan_spline_order', type=int, default=3)
    parser.add_argument('--tail_mask_p', type=float, default=0.3)

    # optim
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--tau_l1', type=float, default=1e-3)

    # loss weights
    parser.add_argument('--alpha_haz', type=float, default=1.0)
    parser.add_argument('--alpha_rank', type=float, default=0.5)
    parser.add_argument('--alpha_aux', type=float, default=0.3)
    parser.add_argument('--alpha_ent', type=float, default=0.01)
    parser.add_argument('--sigma_rank', type=float, default=0.1)

    # evaluation
    parser.add_argument('--landmarks', type=int, nargs='+', default=[3, 6, 9])
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--track_landmark', type=int, default=6)
    parser.add_argument('--track_horizon', type=int, default=3)

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"loading {args.data_path}")
    processed = load_processed(args.data_path)
    N = len(processed['tree_ids'])
    K = processed['K']
    print(f"N = {N}, K = {K}, events = {int(processed['E'].sum())}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    fold_models = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        stratified_kfold_indices(processed['E'], n_splits=args.n_folds,
                                 random_state=args.seed)
    ):
        print(f"\n===== FOLD {fold_idx + 1}/{args.n_folds} | train={len(train_idx)} val={len(val_idx)} =====")
        model, results = train_one_fold(fold_idx, train_idx, val_idx, processed, args, device)
        fold_results.append(results)
        # save model
        torch.save(model.state_dict(), out_dir / f'fold{fold_idx}.pt')

    # aggregate
    agg = mean_std_over_folds(fold_results)
    print("\n===== SUMMARY (mean ± std over {} folds) =====".format(args.n_folds))
    print(f"{'(s, Δ)':<10} {'C^td':>18} {'BS^td':>18}")
    for (s, d), v in sorted(agg.items()):
        print(f"{str((s,d)):<10} "
              f"{v['cindex_mean']:.4f} ± {v['cindex_std']:.4f}    "
              f"{v['brier_mean']:.4f} ± {v['brier_std']:.4f}")

    # dump json
    results_json = {
        'args': vars(args),
        'per_fold': [{str(k): v for k, v in r.items()} for r in fold_results],
        'summary': {str(k): v for k, v in agg.items()},
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
