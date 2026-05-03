"""
Loss functions for the dynamic HLB survival model.

Closely follows Dynamic-DeepHit (Lee et al., IEEE TBME 2020) but adapted to the
single-cause discrete-time setting with the following differences:

1. Hazard NLL uses the per-period hazard head's survival product, while the PMF
   NLL uses the dedicated PMF head; both are summed for redundancy.
2. Ranking loss is the standard pairwise exponential loss restricted to event
   trees i with T_i < T_j.
3. Auxiliary task predicts delta_CT_{k -> k+1} at each period k, masked by the
   per-tree validity mask (post-event, missing, T=1 cases).
4. Period attention entropy is regularised toward higher entropy to discourage
   over-concentration on the very last day(s) of any single period.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def hazard_nll(hazards: torch.Tensor, T: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Discrete-time hazard NLL via product-survival.

    For an event at period T_i (1-indexed):
        L_i = -log(h_{T_i}) - sum_{k < T_i} log(1 - h_k)
    For censoring at period T_i:
        L_i = -sum_{k <= T_i} log(1 - h_k)

    hazards : (B, K) in (0, 1)
    T       : (B,)   long, in [1, K]
    E       : (B,)   long, 0/1
    """
    B, K = hazards.shape
    eps = 1e-7
    log_1mh = torch.log(torch.clamp(1.0 - hazards, min=eps))  # (B, K)
    log_h = torch.log(torch.clamp(hazards, min=eps))          # (B, K)

    # build a (B, K) mask where positions k < T_i are 1 (i.e. survived periods)
    k_idx = torch.arange(K, device=hazards.device).unsqueeze(0) + 1  # 1..K, (1, K)
    T_exp = T.unsqueeze(1)
    survived_mask = (k_idx < T_exp).float()                  # 1 for k = 1..T_i-1
    at_T_mask_event = ((k_idx == T_exp) & (E.unsqueeze(1) == 1)).float()
    at_T_mask_cens = ((k_idx == T_exp) & (E.unsqueeze(1) == 0)).float()

    nll = -(log_1mh * survived_mask).sum(dim=1)
    nll = nll - (log_h * at_T_mask_event).sum(dim=1)
    nll = nll - (log_1mh * at_T_mask_cens).sum(dim=1)  # treat last period censor like surviving past
    return nll.mean()


def pmf_nll(*args, **kwargs):  # kept as a stub for backwards compatibility; not used
    raise RuntimeError("pmf_nll is deprecated; the model now uses hazard-derived CIF only.")


def ranking_loss(cif: torch.Tensor, T: torch.Tensor, E: torch.Tensor,
                 sigma: float = 0.1) -> torch.Tensor:
    """Cause-specific ranking loss (single cause version of DeepHit).

    For every pair (i, j) with E_i = 1 and T_i < T_j (regardless of E_j):
        L_pair = exp(-(F_i(T_i) - F_j(T_i)) / sigma)
    F is the cumulative incidence at the event time of i, evaluated under
    each tree's predicted CIF.
    """
    B, K = cif.shape
    # gather F_*(T_i): each row uses column T_i - 1
    Ti_idx = (T - 1).clamp(min=0, max=K - 1)
    F_at_T = cif.gather(1, Ti_idx.unsqueeze(1)).squeeze(1)  # (B,)

    # for tree j, we need its CIF evaluated at T_i (the OTHER tree's event time)
    # build matrix M[i, j] = F_j(T_i - 1)
    M = cif[:, Ti_idx]  # (B, B): M[j, i] = cif[j, Ti_idx[i]]; we want (i, j) -> cif[j, Ti_idx[i]]
    # reshape: M.T gives (i, j) -> cif[j, Ti_idx[i]]
    F_ij = M.t()  # (B, B), entry [i, j] = F_j(T_i)

    diff = F_at_T.unsqueeze(1) - F_ij  # (B, B); positive when tree i ranks above j

    # acceptable pairs: E_i = 1 and T_i < T_j
    Ti = T.unsqueeze(1)
    Tj = T.unsqueeze(0)
    Ei = E.unsqueeze(1)
    pair_mask = (Ei == 1) & (Ti < Tj)
    pair_mask = pair_mask.float()

    if pair_mask.sum() < 1:
        return torch.zeros((), device=cif.device)

    eta = torch.exp(-diff / sigma)
    loss = (eta * pair_mask).sum() / (pair_mask.sum() + 1e-7)
    return loss


def aux_delta_ct_loss(delta_ct_hat: torch.Tensor, ct_seq: torch.Tensor,
                      ct_mask: torch.Tensor) -> torch.Tensor:
    """Predict delta_CT at each period: delta_CT_k = CT_{k+1} - CT_k.

    delta_ct_hat : (B, K)  predictions for periods 0..K-1 (where index k predicts
                           the change between period k and k+1; we use index k for
                           CT_{k+2} - CT_{k+1} mapping below).
    ct_seq       : (B, K)  with NaN where invalid
    ct_mask      : (B, K)  bool

    For k in 0..K-2: target = ct_seq[:, k+1] - ct_seq[:, k] when both valid.
    The model's prediction at period k (delta_ct_hat[:, k]) targets that delta.
    """
    B, K = ct_seq.shape
    if K < 2:
        return torch.zeros((), device=delta_ct_hat.device)

    pred = delta_ct_hat[:, :K - 1]                      # (B, K-1)
    target = ct_seq[:, 1:] - ct_seq[:, :-1]             # (B, K-1)
    pair_mask = ct_mask[:, 1:] & ct_mask[:, :-1]        # (B, K-1)
    pair_mask = pair_mask.to(pred.dtype)

    target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))

    sq = (pred - target) ** 2 * pair_mask
    denom = pair_mask.sum() + 1e-7
    return sq.sum() / denom


def attn_entropy_reg(tree_attn: torch.Tensor) -> torch.Tensor:
    """Penalise low entropy of cross-period attention so the model spreads
    attention across history rather than collapsing onto a single period.

    tree_attn : (B, n_heads, K, K)
    """
    eps = 1e-7
    # only consider the diagonal causal context: each query position k uses
    # at most k+1 keys; we average entropy over (B, H, K).
    a = tree_attn.clamp(min=eps)
    H = -(a * torch.log(a)).sum(dim=-1)  # (B, n_heads, K)
    # we want HIGH entropy => minimize -H => loss = -mean(H)
    return -H.mean()


def proximal_l1(params, tau: float, lr: float) -> None:
    """Soft-thresholding proximal step. Same interface as utils.loss.proximal_l1
    but accepting an iterable of parameters instead of a model.
    """
    threshold = tau * lr
    with torch.no_grad():
        for p in params:
            if p.requires_grad:
                sign = torch.sign(p)
                shrinkage = torch.clamp(torch.abs(p) - threshold, min=0.0)
                p.data = sign * shrinkage


def total_loss(outputs: dict, batch: dict, alpha=(1.0, 0.5, 0.3, 0.01),
               sigma_rank: float = 0.1) -> dict:
    """Combine all terms into one scalar loss.

    alpha = (haz_nll, rank, aux, ent_reg)
    """
    haz_loss = hazard_nll(outputs['hazards'], batch['T'], batch['E'])
    rank_loss = ranking_loss(outputs['cif'], batch['T'], batch['E'], sigma=sigma_rank)
    aux_loss = aux_delta_ct_loss(outputs['delta_ct_hat'], batch['ct_seq'], batch['ct_mask'])
    ent_loss = attn_entropy_reg(outputs['tree_attn'])

    total = (alpha[0] * haz_loss + alpha[1] * rank_loss
             + alpha[2] * aux_loss + alpha[3] * ent_loss)
    return {
        'total': total,
        'haz_nll': haz_loss.detach(),
        'rank': rank_loss.detach(),
        'aux': aux_loss.detach(),
        'ent_reg': ent_loss.detach(),
    }
