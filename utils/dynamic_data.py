"""
PyTorch Dataset and DataLoader utilities for the dynamic HLB survival task.

Organises each tree as a per-period stack of daily env windows so the model can
process the original daily series without summarising it down to scalar features.
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_processed(path: str = 'data/hlb_processed.pt') -> dict:
    return torch.load(str(pathlib.Path(path)), weights_only=False)


def fit_scalers(env_daily_raw: np.ndarray, static_raw: np.ndarray, train_idx: np.ndarray):
    """Fit scalers on training fold only.

    The env scaler is fit on all env days (the daily series is shared across trees,
    so train/val/test only differ in tree subset; we still split based on which
    rows of env_daily contribute to TRAINING trees' periods to avoid leakage of
    val/test trees' realised periods into the scaler).

    For simplicity and because the daily env is shared, we fit env scaler on the
    full env_daily once per fold using all daily rows: this is acceptable since
    no per-tree label info enters the scaler.
    """
    env_scaler = StandardScaler().fit(env_daily_raw)

    static_train = static_raw[train_idx]
    # impute with column median from training fold
    medians = np.nanmedian(static_train, axis=0)
    static_train_imp = np.where(np.isnan(static_train), medians, static_train)
    static_scaler = StandardScaler().fit(static_train_imp)
    return env_scaler, static_scaler, medians


def apply_scalers(env_daily_raw, static_raw, env_scaler, static_scaler, medians):
    env = env_scaler.transform(env_daily_raw).astype(np.float32)
    static = np.where(np.isnan(static_raw), medians, static_raw)
    static = static_scaler.transform(static).astype(np.float32)
    return env, static


class TreeSequenceDataset(Dataset):
    """One sample per tree.

    __getitem__ returns a dict with:
        env_periods : list[Tensor (L_k, d_env)] of length K
        static      : Tensor (d_static,)
        T           : int
        E           : int
        ct_seq      : Tensor (K,) float (NaN where invalid)
        ct_mask     : Tensor (K,) bool
        tree_id     : str
    """

    def __init__(self, processed: dict, indices: np.ndarray,
                 env_daily: np.ndarray, static: np.ndarray):
        self.indices = indices
        self.env_daily = env_daily  # (D_total, d_env), already scaled
        self.static = static  # (N, d_static), already scaled
        self.period_bounds = processed['period_bounds']  # (K, 2)
        self.T = processed['T']
        self.E = processed['E']
        self.ct_seq = processed['ct_seq']
        self.ct_valid_mask = processed['ct_valid_mask']
        self.tree_ids = processed['tree_ids']
        self.K = processed['K']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        env_periods = []
        for k in range(self.K):
            lo, hi = self.period_bounds[k]
            env_periods.append(torch.from_numpy(self.env_daily[lo:hi].copy()))  # (L_k, d_env)
        return {
            'env_periods': env_periods,
            'static': torch.from_numpy(self.static[i]),
            'T': int(self.T[i]),
            'E': int(self.E[i]),
            'ct_seq': torch.from_numpy(self.ct_seq[i]),
            'ct_mask': torch.from_numpy(self.ct_valid_mask[i]),
            'tree_id': self.tree_ids[i],
        }


def collate_trees(batch: list) -> dict:
    """Collate a list of tree samples into batched tensors.

    Per-period env windows have variable length L_k, but L_k is the SAME across
    trees (env is shared). So we can stack: env_periods[k] -> (B, L_k, d_env).
    """
    B = len(batch)
    K = len(batch[0]['env_periods'])
    env_periods = []
    for k in range(K):
        per_k = torch.stack([batch[b]['env_periods'][k] for b in range(B)], dim=0)
        env_periods.append(per_k)  # (B, L_k, d_env)

    static = torch.stack([b['static'] for b in batch], dim=0)
    T = torch.tensor([b['T'] for b in batch], dtype=torch.long)
    E = torch.tensor([b['E'] for b in batch], dtype=torch.long)
    ct_seq = torch.stack([b['ct_seq'] for b in batch], dim=0)  # (B, K), NaN allowed
    ct_mask = torch.stack([b['ct_mask'] for b in batch], dim=0)
    tree_ids = [b['tree_id'] for b in batch]
    return {
        'env_periods': env_periods,
        'static': static,
        'T': T,
        'E': E,
        'ct_seq': ct_seq,
        'ct_mask': ct_mask,
        'tree_ids': tree_ids,
    }


def stratified_kfold_indices(E: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """Yield (train_idx, val_idx) per fold, stratified on event indicator."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(E))
    for train_idx, val_idx in skf.split(indices, E):
        yield train_idx, val_idx
