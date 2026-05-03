"""
ReKDynSurv: dynamic survival model for HLB prediction.

Architecture:
    Per-period env encoder : multi-scale dilated 1D-CNN + FiLM(static) over each
                             tree's daily env window, producing env_period_emb[k].
    Tree-specific period attention : static-modulated query over the K period
                                     embeddings, with causal mask so context[k]
                                     only sees periods 1..k.
    KAN fusion head : the existing KANLinear from models/kan.py, applied to
                      (context[k] || static_emb || time_embed[k]).
    Heads :
        hazard head -> discrete-time hazard h_k in (0, 1)
        PMF head    -> P(T = k) for k = 1..K and censoring bin K+1
        aux head    -> scalar prediction of delta_CT_{k->k+1}
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import KANLinear


class FiLM(nn.Module):
    """Generate per-channel gamma, beta from static features to modulate a feature map."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)
        self.feat_dim = feat_dim

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, L), cond: (B, cond_dim)
        gb = self.proj(cond)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return feat * (1 + gamma) + beta


class MultiScaleConvBlock(nn.Module):
    """Parallel 1D convs with kernels {3,7,14} + dilated conv (d=2) to capture
    sustained multi-day weather patterns rather than last-day effects.
    """

    def __init__(self, in_ch: int, out_ch: int, static_dim: int, dropout: float = 0.2):
        super().__init__()
        assert out_ch % 4 == 0, "out_ch must be divisible by 4"
        br = out_ch // 4
        self.conv_k3 = nn.Conv1d(in_ch, br, kernel_size=3, padding=1)
        self.conv_k7 = nn.Conv1d(in_ch, br, kernel_size=7, padding=3)
        self.conv_k14 = nn.Conv1d(in_ch, br, kernel_size=14, padding=6)
        self.conv_d2 = nn.Conv1d(in_ch, br, kernel_size=7, padding=6, dilation=2)
        self.film = FiLM(static_dim, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        h3 = self.conv_k3(x)
        h7 = self.conv_k7(x)
        h14 = self.conv_k14(x)  # (B, br, L-1)
        if h14.size(-1) != x.size(-1):
            # right-pad to match length
            pad = x.size(-1) - h14.size(-1)
            if pad > 0:
                h14 = F.pad(h14, (0, pad))
            elif pad < 0:
                h14 = h14[..., :x.size(-1)]
        hd = self.conv_d2(x)
        if hd.size(-1) != x.size(-1):
            pad = x.size(-1) - hd.size(-1)
            if pad > 0:
                hd = F.pad(hd, (0, pad))
            elif pad < 0:
                hd = hd[..., :x.size(-1)]
        h = torch.cat([h3, h7, h14, hd], dim=1)  # (B, out_ch, L)
        h = self.bn(h)
        h = self.film(h, static)
        h = self.act(h)
        h = self.drop(h)
        return h


class PerPeriodEnvEncoder(nn.Module):
    """Encode each period's daily window into a single d_h embedding, per tree.

    All trees share the underlying daily env, but FiLM + static-modulated pooling
    ensure each tree gets its own per-period embedding.
    """

    def __init__(self, d_env: int, d_h: int, static_dim: int,
                 dropout: float = 0.2, tail_mask_p: float = 0.3,
                 tail_mask_max: int = 5):
        super().__init__()
        self.block1 = MultiScaleConvBlock(d_env, d_h, static_dim, dropout)
        self.block2 = MultiScaleConvBlock(d_h, d_h, static_dim, dropout)
        self.pool_proj = nn.Linear(static_dim, d_h)
        self.d_h = d_h
        self.tail_mask_p = tail_mask_p
        self.tail_mask_max = tail_mask_max

    def forward(self, env_period: torch.Tensor, static: torch.Tensor, training: bool) -> torch.Tensor:
        """
        env_period: (B, L_k, d_env)
        static:     (B, static_dim)
        returns:    (B, d_h)
        """
        B, L, _ = env_period.shape
        x = env_period.transpose(1, 2)  # (B, d_env, L)

        if training and self.tail_mask_p > 0 and L > self.tail_mask_max + 1:
            if torch.rand(()).item() < self.tail_mask_p:
                n_mask = int(torch.randint(2, self.tail_mask_max + 1, ()).item())
                # zero out the last n_mask days across the whole batch
                mask = torch.ones(L, device=x.device)
                mask[-n_mask:] = 0
                x = x * mask.view(1, 1, L)

        h = self.block1(x, static)
        h = self.block2(h, static)  # (B, d_h, L)

        # attention pooling across days, query derived from static so pooling
        # weights differ per tree
        q = self.pool_proj(static)  # (B, d_h)
        attn_logits = torch.einsum('bdl,bd->bl', h, q) / math.sqrt(self.d_h)
        attn = F.softmax(attn_logits, dim=-1)  # (B, L)
        pooled = torch.einsum('bdl,bl->bd', h, attn)  # (B, d_h)
        return pooled, attn  # return attn for potential regularisation/viz


class TreePeriodAttention(nn.Module):
    """Attention over K period embeddings with causal mask.

    query[k]   depends on static + time embedding of period k (tree-specific).
    key/value  are the K period embeddings env_period_emb[1..K].
    causal mask: query at step k only attends to keys 1..k.
    """

    def __init__(self, d_h: int, static_dim: int, K: int, n_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.d_h = d_h
        self.n_heads = n_heads
        assert d_h % n_heads == 0
        self.time_embed = nn.Parameter(torch.randn(K, d_h) * 0.02)
        self.q_proj = nn.Linear(static_dim + d_h, d_h)
        self.k_proj = nn.Linear(d_h, d_h)
        self.v_proj = nn.Linear(d_h, d_h)
        self.o_proj = nn.Linear(d_h, d_h)
        self.drop = nn.Dropout(dropout)

    def forward(self, period_emb: torch.Tensor, static: torch.Tensor):
        """
        period_emb : (B, K, d_h)
        static     : (B, static_dim)
        returns    : context (B, K, d_h), attn (B, n_heads, K, K)
        """
        B = period_emb.size(0)
        time_emb = self.time_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_h)
        static_rep = static.unsqueeze(1).expand(-1, self.K, -1)  # (B, K, static_dim)
        q_in = torch.cat([static_rep, time_emb], dim=-1)  # (B, K, static_dim + d_h)

        Q = self.q_proj(q_in)
        K_ = self.k_proj(period_emb)
        V = self.v_proj(period_emb)

        # reshape for multi-head
        def split_h(t):
            return t.view(B, self.K, self.n_heads, self.d_h // self.n_heads).transpose(1, 2)
        Qh, Kh, Vh = split_h(Q), split_h(K_), split_h(V)  # (B, H, K, d_head)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(Qh.size(-1))  # (B, H, K, K)

        # causal mask: query position i only attends to keys j where j <= i
        mask = torch.tril(torch.ones(self.K, self.K, device=scores.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, Vh)  # (B, H, K, d_head)
        out = out.transpose(1, 2).contiguous().view(B, self.K, self.d_h)
        return self.o_proj(out), attn


class KANFusionHead(nn.Module):
    """Two-layer KAN fusion on (context || static || time_embed) -> d_h."""

    def __init__(self, in_dim: int, d_h: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.kan1 = KANLinear(in_dim, d_h, grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-3, 3])
        self.kan2 = KANLinear(d_h, d_h, grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-3, 3])
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, in_dim)
        B, K, _ = x.shape
        x = x.reshape(B * K, -1)
        h = self.kan1(x)
        h = self.kan2(h)
        h = self.norm(h)
        return h.view(B, K, -1)


class ReKDynSurv(nn.Module):
    def __init__(self, d_env: int, d_static: int, K: int,
                 d_h: int = 32, n_heads: int = 2, dropout: float = 0.3,
                 kan_grid_size: int = 5, kan_spline_order: int = 3,
                 tail_mask_p: float = 0.3,
                 train_landmark_mask_p: float = 1.0):
        super().__init__()
        self.K = K
        self.d_h = d_h
        self.train_landmark_mask_p = train_landmark_mask_p

        self.static_encoder = nn.Sequential(
            nn.Linear(d_static, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_h),
        )

        # per-period encoder is shared across periods (same module applied K times)
        self.period_encoder = PerPeriodEnvEncoder(
            d_env=d_env, d_h=d_h, static_dim=d_h,
            dropout=dropout, tail_mask_p=tail_mask_p,
        )

        self.period_attn = TreePeriodAttention(
            d_h=d_h, static_dim=d_h, K=K, n_heads=n_heads, dropout=dropout,
        )

        self.fusion = KANFusionHead(
            in_dim=d_h + d_h + d_h,  # context + static + time
            d_h=d_h,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

        self.hazard_head = nn.Linear(d_h, 1)           # per-period hazard logit
        self.aux_head = nn.Linear(d_h, 1)              # delta-CT prediction

        # initialise hazard bias so initial hazard ~ 0.1 (avoids early NLL explosion)
        nn.init.constant_(self.hazard_head.bias, -2.2)

    def forward(self, env_periods: list, static: torch.Tensor,
                env_mask_from: int | None = None):
        """
        env_periods : list of length K, env_periods[k] is (B, L_k, d_env)
        static      : (B, d_static)
        env_mask_from : if given, env_periods[k] for k >= env_mask_from is zeroed
                        out before encoding. Used for landmark evaluation so the
                        model only sees env data up to the landmark.

        returns dict:
            hazards   : (B, K)   per-period discrete hazard
            survival  : (B, K)   cumulative survival S_k = prod_{j<=k}(1-h_j)
            cif       : (B, K)   cumulative incidence F(k) = 1 - S(k)
            delta_ct_hat : (B, K) predicted delta_CT at each period
            tree_attn    : (B, n_heads, K, K) cross-period attention
            context      : (B, K, d_h) fused context vector per period
        """
        B = static.size(0)
        static_emb = self.static_encoder(static)  # (B, d_h)

        # per-sample random landmark masking during training: each sample gets
        # its own s_b ~ Uniform{0..K}, periods k >= s_b are zeroed out so the
        # model learns to predict from partial env. s_b == K means "no masking".
        # This matches the eval-time distribution where env_periods[landmark:]
        # are zeroed via env_mask_from.
        if (self.training and env_mask_from is None
                and self.train_landmark_mask_p > 0.0):
            if torch.rand((), device=static.device).item() < self.train_landmark_mask_p:
                s_b = torch.randint(0, self.K + 1, (B,), device=static.device)
            else:
                s_b = None
        else:
            s_b = None

        period_embs = []
        within_attn = []
        for k in range(self.K):
            ep = env_periods[k]
            if env_mask_from is not None and k >= env_mask_from:
                ep = torch.zeros_like(ep)
            elif s_b is not None:
                keep = (k < s_b).to(ep.dtype).view(B, 1, 1)  # (B, 1, 1)
                ep = ep * keep
            pooled, attn_k = self.period_encoder(ep, static_emb, training=self.training)
            period_embs.append(pooled)
            within_attn.append(attn_k)
        period_emb = torch.stack(period_embs, dim=1)  # (B, K, d_h)

        context, tree_attn = self.period_attn(period_emb, static_emb)  # (B, K, d_h)

        # fusion input
        time_emb = self.period_attn.time_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_h)
        static_rep = static_emb.unsqueeze(1).expand(-1, self.K, -1)
        fusion_in = torch.cat([context, static_rep, time_emb], dim=-1)
        h = self.fusion(fusion_in)  # (B, K, d_h)

        hazard_logits = self.hazard_head(h).squeeze(-1)  # (B, K)
        hazards = torch.sigmoid(hazard_logits)
        # numerically stable survival = prod(1 - h)
        survival = torch.cumprod(torch.clamp(1.0 - hazards, min=1e-7, max=1.0 - 1e-7), dim=1)
        cif = 1.0 - survival

        delta_ct_hat = self.aux_head(h).squeeze(-1)  # (B, K)

        return {
            'hazards': hazards,
            'hazard_logits': hazard_logits,
            'survival': survival,
            'cif': cif,
            'delta_ct_hat': delta_ct_hat,
            'tree_attn': tree_attn,
            'period_emb': period_emb,
            'context': h,
        }
