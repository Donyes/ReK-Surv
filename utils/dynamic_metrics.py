"""
Time-dependent evaluation metrics for the discrete-time HLB survival model.

We implement:
    C_td(s, s+Δ): Antolini's time-dependent C-index, restricted to the risk set
                  at landmark s. Predictions are CIF values F(s+Δ | history ≤ s).
    BS_td(s+Δ):   IPCW Brier score (Graf 1999, Gerds & Schumacher 2006), again
                  restricted to T_i > s.

For each landmark s ∈ {3, 6, 9} and horizon Δ ∈ {1, 3} we report both metrics.

Predictions per tree are obtained by re-running the model with env_periods[s:]
zeroed out, so the model can only use environment data up to landmark s. The
FiLM-modulated per-period encoder still uses zeros for periods >= s, but the
causal attention guarantees context[k<s] is unchanged versus the full-env pass.
"""
from __future__ import annotations
import numpy as np
import torch
from sksurv.metrics import concordance_index_ipcw, brier_score
from sksurv.nonparametric import kaplan_meier_estimator


def predict_landmark(model, dataset, landmark: int, batch_size: int = 32,
                     device: torch.device = torch.device('cpu')):
    """Return cif (N, K) predicted with env masked from period `landmark`.

    landmark is 0-based: env_periods[landmark:] are zeroed.
    """
    from torch.utils.data import DataLoader
    from .dynamic_data import collate_trees

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_trees)
    cif_all, T_all, E_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            env_periods = [ep.to(device) for ep in batch['env_periods']]
            static = batch['static'].to(device)
            out = model(env_periods, static, env_mask_from=landmark)
            cif_all.append(out['cif'].cpu().numpy())
            T_all.append(batch['T'].numpy())
            E_all.append(batch['E'].numpy())
    return np.concatenate(cif_all), np.concatenate(T_all), np.concatenate(E_all)


def _structured_array(T: np.ndarray, E: np.ndarray):
    """Build the structured array sksurv expects."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(E, T)],
        dtype=[('event', '?'), ('time', '<f8')],
    )


def time_dependent_cindex(cif: np.ndarray, T: np.ndarray, E: np.ndarray,
                          landmark: int, horizon: int,
                          T_train: np.ndarray | None = None,
                          E_train: np.ndarray | None = None) -> float:
    """C^td at (landmark s, horizon Δ).

    Restrict to the risk set {i : T_i > s}. Use F(s+Δ) as the risk score.
    Higher risk => earlier event => sksurv expects positive risk score
    correlated with hazard.
    """
    K = cif.shape[1]
    eval_idx = landmark + horizon - 1  # 0-based column index for period s+Δ
    eval_idx = min(eval_idx, K - 1)

    risk_mask = T > landmark
    if risk_mask.sum() < 2:
        return float('nan')
    cif_eval = cif[risk_mask, eval_idx]
    T_eval = T[risk_mask].astype(float)
    E_eval = E[risk_mask].astype(bool)
    if E_eval.sum() < 1:
        return float('nan')

    survival_train = _structured_array(
        T_train if T_train is not None else T,
        E_train if E_train is not None else E,
    )
    survival_test = _structured_array(T_eval, E_eval.astype(int))
    try:
        c, *_ = concordance_index_ipcw(
            survival_train, survival_test, cif_eval,
            tau=float(landmark + horizon),
        )
        return float(c)
    except Exception as e:
        return float('nan')


def time_dependent_brier(cif: np.ndarray, T: np.ndarray, E: np.ndarray,
                         landmark: int, horizon: int,
                         T_train: np.ndarray, E_train: np.ndarray) -> float:
    """BS^td at landmark s evaluated at time s+Δ, IPCW-corrected."""
    K = cif.shape[1]
    eval_idx = min(landmark + horizon - 1, K - 1)
    eval_time = float(landmark + horizon)

    risk_mask = T > landmark
    if risk_mask.sum() < 2:
        return float('nan')

    cif_eval = cif[risk_mask, eval_idx]
    T_eval = T[risk_mask].astype(float)
    E_eval = E[risk_mask].astype(bool)

    survival_train = _structured_array(T_train, E_train)
    survival_test = _structured_array(T_eval, E_eval.astype(int))

    # sksurv brier_score wants (n,) survival predictions for each eval time
    surv_pred = 1.0 - cif_eval
    try:
        # ensure eval_time is within both train and test follow-up
        max_time = min(T_train.max(), T_eval.max())
        t = min(eval_time, max_time - 1e-6)
        times, bs = brier_score(survival_train, survival_test,
                                surv_pred.reshape(-1, 1), [t])
        return float(bs[0])
    except Exception:
        return float('nan')


def evaluate_all_landmarks(model, val_dataset, T_train: np.ndarray, E_train: np.ndarray,
                           landmarks=(3, 6, 9), horizons=(1, 3),
                           batch_size: int = 32,
                           device: torch.device = torch.device('cpu')):
    """Compute C^td and BS^td for every (s, Δ) cell.

    Returns dict {(s, Δ): {'cindex': ..., 'brier': ...}}.
    """
    results = {}
    K = val_dataset.K
    for s in landmarks:
        if s >= K:
            continue
        cif, T_val, E_val = predict_landmark(model, val_dataset, landmark=s,
                                             batch_size=batch_size, device=device)
        for d in horizons:
            if s + d > K:
                continue
            c = time_dependent_cindex(cif, T_val, E_val, s, d, T_train, E_train)
            b = time_dependent_brier(cif, T_val, E_val, s, d, T_train, E_train)
            results[(s, d)] = {'cindex': c, 'brier': b}
    return results
