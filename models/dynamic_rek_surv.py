"""
Dynamic ReK-Surv models for time-varying survival analysis.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .kan import KAN


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(weights.dtype)
    denominator = weights.sum(dim=dim, keepdim=True)
    return torch.where(denominator > 0, weights / denominator.clamp_min(1e-8), torch.zeros_like(weights))


class DynamicReKSurv(nn.Module):
    """
    Unified dynamic survival model supporting:
    - legacy: KAN + RNN + attention
    - legacy_period_mean: KAN + RNN + attention over 13 period-mean tokens
    - period_ms: period-level multiscale attention with agricultural features
    - period_ms_tree_query: tree-specific period attention conditioned on static features
    - trigger_orchard: daily multiscale trigger-window search
    - trigger_orchard_v2: tree-conditioned continuous-window trigger search
    """

    def __init__(
        self,
        env_dim: int,
        static_dim: int,
        day_to_period,
        num_periods: int = 13,
        use_time_features: bool = False,
        rnn_type: str = "gru",
        hidden_size: int = 64,
        projection_dim: int = 32,
        landmark_embedding_dim: int = 8,
        grid_size: int = 1,
        spline_order: int = 3,
        model_type: str = "legacy",
        period_env_dim: int | None = None,
        env_feature_names: Sequence[str] | None = None,
        period_start_day_index: Sequence[int] | None = None,
        period_end_day_index: Sequence[int] | None = None,
        env_aux_mode: str = "next_day",
        trigger_topk: int = 5,
        tree_attention_dropout: float = 0.10,
        static_attention_dim: int = 32,
        ct_delta_output_dim: int = 1,
    ):
        super().__init__()
        self.env_dim = env_dim
        self.static_dim = static_dim
        self.num_periods = num_periods
        self.use_time_features = use_time_features
        self.hidden_size = hidden_size
        self.model_type = model_type.strip().lower()
        self.env_aux_mode = env_aux_mode.strip().lower()
        self.trigger_topk = int(trigger_topk)
        self.ct_delta_output_dim = max(int(ct_delta_output_dim), 1)

        day_to_period_tensor = torch.tensor(day_to_period, dtype=torch.long)
        max_days = int(day_to_period_tensor.numel())
        day_positions = torch.arange(max_days, dtype=torch.float32) / float(max(max_days, 1))

        self.register_buffer("day_to_period_long", day_to_period_tensor.view(1, max_days), persistent=False)
        self.register_buffer("day_to_period", day_to_period_tensor.view(1, max_days, 1).float())
        self.register_buffer("day_positions", day_positions.view(1, max_days, 1))
        self.register_buffer(
            "period_ids",
            torch.arange(1, num_periods + 1, dtype=torch.long).view(1, num_periods),
        )

        if period_start_day_index is None or period_end_day_index is None:
            start_day_index, end_day_index = self._infer_period_day_bounds(day_to_period_tensor, num_periods)
        else:
            start_day_index = torch.tensor(period_start_day_index, dtype=torch.long)
            end_day_index = torch.tensor(period_end_day_index, dtype=torch.long)
        self.register_buffer("period_start_day_index", start_day_index.view(1, num_periods), persistent=False)
        self.register_buffer("period_end_day_index", end_day_index.view(1, num_periods), persistent=False)

        self.period_env_dim = int(period_env_dim or 0)
        self.env_feature_names = list(env_feature_names or [])
        self.expert_feature_indices = self._resolve_expert_feature_indices(self.env_feature_names)

        if self.model_type == "legacy":
            self._init_legacy(
                rnn_type=rnn_type,
                projection_dim=projection_dim,
                hidden_size=hidden_size,
                landmark_embedding_dim=landmark_embedding_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                seq_input_dim=self.env_dim,
                aux_target_dim=self.env_dim,
            )
        elif self.model_type == "legacy_period_mean":
            if self.period_env_dim <= 0:
                raise ValueError("legacy_period_mean requires period_env_dim > 0.")
            self._init_legacy(
                rnn_type=rnn_type,
                projection_dim=projection_dim,
                hidden_size=hidden_size,
                landmark_embedding_dim=landmark_embedding_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                seq_input_dim=self.period_env_dim,
                aux_target_dim=self.period_env_dim,
            )
        elif self.model_type == "period_ms":
            self._init_period_ms(
                projection_dim=projection_dim,
                landmark_embedding_dim=landmark_embedding_dim,
                grid_size=grid_size,
                spline_order=spline_order,
            )
        elif self.model_type == "period_ms_tree_query":
            self._init_period_ms_tree_query(
                projection_dim=projection_dim,
                landmark_embedding_dim=landmark_embedding_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                tree_attention_dropout=tree_attention_dropout,
                static_attention_dim=static_attention_dim,
            )
        elif self.model_type == "trigger_orchard":
            self._init_trigger_orchard(
                projection_dim=projection_dim,
                hidden_size=hidden_size,
                landmark_embedding_dim=landmark_embedding_dim,
                grid_size=grid_size,
                spline_order=spline_order,
            )
        elif self.model_type == "trigger_orchard_v2":
            self._init_trigger_orchard_v2(
                projection_dim=projection_dim,
                hidden_size=hidden_size,
                landmark_embedding_dim=landmark_embedding_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                ct_delta_output_dim=self.ct_delta_output_dim,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _init_legacy(
        self,
        rnn_type: str,
        projection_dim: int,
        hidden_size: int,
        landmark_embedding_dim: int,
        grid_size: int,
        spline_order: int,
        seq_input_dim: int,
        aux_target_dim: int,
    ) -> None:
        time_feature_dim = 2 if self.use_time_features else 0
        input_dim = seq_input_dim + self.static_dim + time_feature_dim
        self.legacy_seq_input_dim = int(seq_input_dim)
        self.legacy_aux_target_dim = int(aux_target_dim)

        self.day_projector = KAN(
            [input_dim, projection_dim, projection_dim],
            grid_size=grid_size,
            spline_order=spline_order,
        )

        rnn_type = rnn_type.strip().lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=projection_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=projection_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        self.static_projector = nn.Sequential(
            nn.Linear(self.static_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.attention_score = nn.Linear(hidden_size, 1)
        self.landmark_embedding = nn.Embedding(self.num_periods + 1, landmark_embedding_dim)
        self.fused_dim = hidden_size * 2 + projection_dim
        self.survival_head = KAN(
            [self.fused_dim, projection_dim, self.num_periods + 1],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.auxiliary_head = nn.Linear(hidden_size, self.legacy_aux_target_dim)
        self.ct_aux_head = nn.Sequential(
            nn.Linear(self.fused_dim + landmark_embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 1),
        )

    def _init_period_ms(
        self,
        projection_dim: int,
        landmark_embedding_dim: int,
        grid_size: int,
        spline_order: int,
    ) -> None:
        if self.period_env_dim <= 0:
            raise ValueError("period_ms requires period_env_dim > 0.")

        self.period_projector = KAN(
            [self.period_env_dim, projection_dim, projection_dim],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.period_static_projector = nn.Sequential(
            nn.Linear(self.static_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.period_lag_embedding = nn.Embedding(self.num_periods + 1, projection_dim)
        self.period_attention_score = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, 1),
        )
        self.no_history_period = nn.Parameter(torch.zeros(projection_dim))
        self.landmark_embedding = nn.Embedding(self.num_periods + 1, landmark_embedding_dim)
        self.fused_dim = projection_dim * 3
        self.period_survival_head = KAN(
            [self.fused_dim, projection_dim, self.num_periods + 1],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.period_ct_aux_head = nn.Sequential(
            nn.Linear(self.fused_dim + landmark_embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 1),
        )

    def _init_period_ms_tree_query(
        self,
        projection_dim: int,
        landmark_embedding_dim: int,
        grid_size: int,
        spline_order: int,
        tree_attention_dropout: float,
        static_attention_dim: int,
    ) -> None:
        if self.period_env_dim <= 0:
            raise ValueError("period_ms_tree_query requires period_env_dim > 0.")

        self.period_projector = KAN(
            [self.period_env_dim, projection_dim, projection_dim],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        hidden_static_dim = max(int(static_attention_dim), projection_dim)
        self.period_static_projector = nn.Sequential(
            nn.Linear(self.static_dim, hidden_static_dim),
            nn.ReLU(),
            nn.Linear(hidden_static_dim, projection_dim),
        )
        self.period_lag_embedding = nn.Embedding(self.num_periods + 1, projection_dim)
        self.tree_period_film = nn.Linear(projection_dim, projection_dim * 2)
        self.tree_static_gate = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Sigmoid(),
        )
        self.tree_attn_period_proj = nn.Linear(projection_dim, projection_dim, bias=False)
        self.tree_attn_lag_proj = nn.Linear(projection_dim, projection_dim, bias=False)
        self.tree_attn_static_proj = nn.Linear(projection_dim, projection_dim, bias=True)
        self.tree_attn_interaction_proj = nn.Linear(projection_dim, projection_dim, bias=False)
        self.tree_attention_dropout = nn.Dropout(float(tree_attention_dropout))
        self.tree_attention_score = nn.Linear(projection_dim, 1)

        self.no_history_period = nn.Parameter(torch.zeros(projection_dim))
        self.landmark_embedding = nn.Embedding(self.num_periods + 1, landmark_embedding_dim)
        self.fused_dim = projection_dim * 3 + landmark_embedding_dim
        self.period_survival_head = KAN(
            [self.fused_dim, projection_dim, self.num_periods + 1],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.period_ct_aux_head = nn.Sequential(
            nn.Linear(self.fused_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 1),
        )

    def _init_trigger_orchard(
        self,
        projection_dim: int,
        hidden_size: int,
        landmark_embedding_dim: int,
        grid_size: int,
        spline_order: int,
    ) -> None:
        kernel_sizes = [7, 14, 30, 60]
        branch_hidden = max(hidden_size // len(kernel_sizes), 8)

        self.trigger_input_proj = nn.Linear(self.env_dim, projection_dim)
        self.trigger_convs = nn.ModuleList(
            [
                nn.Conv1d(projection_dim, branch_hidden, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        self.trigger_daily_encoder = nn.Sequential(
            nn.Linear(projection_dim + branch_hidden * len(kernel_sizes), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        subset_dim = max(projection_dim // 2, 8)
        self.expert_subset_projectors = nn.ModuleList()
        self.expert_score_heads = nn.ModuleList()
        for indices in self.expert_feature_indices:
            input_dim = max(len(indices), 1)
            self.expert_subset_projectors.append(nn.Linear(input_dim, subset_dim))
            self.expert_score_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_size + subset_dim, projection_dim),
                    nn.ReLU(),
                    nn.Linear(projection_dim, 1),
                )
            )

        self.static_projector = nn.Sequential(
            nn.Linear(self.static_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.static_expert_gate = nn.Sequential(
            nn.Linear(self.static_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 3),
        )
        self.no_history_day = nn.Parameter(torch.zeros(hidden_size))
        self.landmark_embedding = nn.Embedding(self.num_periods + 1, landmark_embedding_dim)
        self.fused_dim = hidden_size * 4 + projection_dim
        self.trigger_survival_head = KAN(
            [self.fused_dim, hidden_size, self.num_periods + 1],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.trigger_ct_aux_head = nn.Sequential(
            nn.Linear(self.fused_dim + landmark_embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 1),
        )

    def _init_trigger_orchard_v2(
        self,
        projection_dim: int,
        hidden_size: int,
        landmark_embedding_dim: int,
        grid_size: int,
        spline_order: int,
        ct_delta_output_dim: int,
    ) -> None:
        kernel_sizes = [7, 14, 30, 60]
        branch_hidden = max(hidden_size // len(kernel_sizes), 8)

        self.trigger_input_proj = nn.Linear(self.env_dim, projection_dim)
        self.trigger_convs = nn.ModuleList(
            [
                nn.Conv1d(projection_dim, branch_hidden, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        self.trigger_daily_encoder = nn.Sequential(
            nn.Linear(projection_dim + branch_hidden * len(kernel_sizes), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        (
            trigger_window_starts,
            trigger_window_ends,
            trigger_window_lengths,
            trigger_window_types,
            trigger_window_end_periods,
        ) = self._build_trigger_window_catalog(kernel_sizes)
        self.register_buffer("trigger_window_start_index", trigger_window_starts, persistent=False)
        self.register_buffer("trigger_window_end_index", trigger_window_ends, persistent=False)
        self.register_buffer("trigger_window_length", trigger_window_lengths, persistent=False)
        self.register_buffer("trigger_window_type_id", trigger_window_types, persistent=False)
        self.register_buffer("trigger_window_end_period", trigger_window_end_periods, persistent=False)

        self.trigger_num_window_types = int(trigger_window_types.max().item()) + 1
        self.trigger_window_type_embedding = nn.Embedding(self.trigger_num_window_types, projection_dim)
        self.trigger_window_lag_embedding = nn.Embedding(self.num_periods + 1, projection_dim)

        subset_dim = max(projection_dim // 2, 8)
        token_input_dim = hidden_size + subset_dim + projection_dim * 2 + 2
        self.expert_subset_projectors = nn.ModuleList()
        self.expert_window_projectors = nn.ModuleList()
        self.expert_query_projectors = nn.ModuleList()
        for indices in self.expert_feature_indices:
            input_dim = max(len(indices), 1)
            self.expert_subset_projectors.append(nn.Linear(input_dim, subset_dim))
            self.expert_window_projectors.append(
                nn.Sequential(
                    nn.Linear(token_input_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )
            self.expert_query_projectors.append(
                nn.Sequential(
                    nn.Linear(self.static_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )

        self.static_projector = nn.Sequential(
            nn.Linear(self.static_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.static_expert_gate = nn.Sequential(
            nn.Linear(self.static_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 3),
        )
        self.no_history_day = nn.Parameter(torch.zeros(hidden_size))
        self.landmark_embedding = nn.Embedding(self.num_periods + 1, landmark_embedding_dim)
        self.fused_dim = hidden_size * 4 + projection_dim
        self.trigger_survival_head = KAN(
            [self.fused_dim, hidden_size, self.num_periods + 1],
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.trigger_ct_aux_head = nn.Sequential(
            nn.Linear(self.fused_dim + landmark_embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, ct_delta_output_dim),
        )

    def forward(
        self,
        daily_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_days: torch.Tensor,
        landmark_period: torch.Tensor,
        period_env_prefix: torch.Tensor | None = None,
        seq_len_periods: torch.Tensor | None = None,
        period_observed_mask: torch.Tensor | None = None,
    ) -> dict:
        if daily_env_prefix.dim() != 3:
            raise ValueError("daily_env_prefix must have shape [batch, time, env_dim].")

        if seq_len_periods is None:
            seq_len_periods = landmark_period.long()

        if self.model_type == "legacy":
            return self._forward_legacy(
                daily_env_prefix=daily_env_prefix,
                static_x=static_x,
                seq_len_days=seq_len_days,
                landmark_period=landmark_period,
            )
        if self.model_type == "legacy_period_mean":
            if period_env_prefix is None:
                raise ValueError("legacy_period_mean requires period_env_prefix.")
            return self._forward_legacy_period_mean(
                period_env_prefix=period_env_prefix,
                static_x=static_x,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_period,
            )
        if self.model_type in {"period_ms", "period_ms_tree_query"}:
            if period_env_prefix is None:
                raise ValueError(f"{self.model_type} requires period_env_prefix.")
            if self.model_type == "period_ms_tree_query":
                return self._forward_period_ms_tree_query(
                    daily_env_prefix=daily_env_prefix,
                    period_env_prefix=period_env_prefix,
                    static_x=static_x,
                    seq_len_days=seq_len_days,
                    seq_len_periods=seq_len_periods,
                    landmark_period=landmark_period,
                    period_observed_mask=period_observed_mask,
                )
            return self._forward_period_ms(
                daily_env_prefix=daily_env_prefix,
                period_env_prefix=period_env_prefix,
                static_x=static_x,
                seq_len_days=seq_len_days,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_period,
                period_observed_mask=period_observed_mask,
            )
        if self.model_type == "trigger_orchard_v2":
            return self._forward_trigger_orchard_v2(
                daily_env_prefix=daily_env_prefix,
                static_x=static_x,
                seq_len_days=seq_len_days,
                landmark_period=landmark_period,
            )
        return self._forward_trigger_orchard(
            daily_env_prefix=daily_env_prefix,
            static_x=static_x,
            seq_len_days=seq_len_days,
            landmark_period=landmark_period,
        )

    def _forward_legacy(
        self,
        daily_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_days: torch.Tensor,
        landmark_period: torch.Tensor,
    ) -> dict:
        batch_size, max_days, _ = daily_env_prefix.shape
        static_repeated = static_x.unsqueeze(1).expand(batch_size, max_days, static_x.size(1))

        feature_parts = [daily_env_prefix, static_repeated]
        if self.use_time_features:
            period_feature = self.day_to_period[:, :max_days, :] / float(self.num_periods)
            day_feature = self.day_positions[:, :max_days, :]
            period_feature = period_feature.expand(batch_size, -1, -1)
            day_feature = day_feature.expand(batch_size, -1, -1)
            feature_parts.extend([day_feature, period_feature])

        day_inputs = torch.cat(feature_parts, dim=-1)
        projected = self.day_projector(day_inputs)

        packed = pack_padded_sequence(
            projected,
            lengths=seq_len_days.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden_state = self.rnn(packed)
        rnn_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=max_days,
        )

        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]
        last_hidden = hidden_state[-1]

        time_index = torch.arange(max_days, device=daily_env_prefix.device).unsqueeze(0)
        valid_mask = time_index < seq_len_days.unsqueeze(1)

        attention_logits = self.attention_score(rnn_outputs).squeeze(-1)
        attention_weights = _masked_softmax(attention_logits, valid_mask, dim=1)
        attention_context = torch.bmm(
            attention_weights.unsqueeze(1),
            rnn_outputs,
        ).squeeze(1)

        static_summary = self.static_projector(static_x)
        head_input = torch.cat([last_hidden, attention_context, static_summary], dim=1)
        all_logits = self.survival_head(head_input)
        landmark_summary = self.landmark_embedding(
            landmark_period.long().clamp(min=0, max=self.num_periods)
        )
        pred_ct_delta = self.ct_aux_head(torch.cat([head_input, landmark_summary], dim=1))

        aux_next_env = (
            self.auxiliary_head(rnn_outputs)
            if self.env_aux_mode == "next_day"
            else daily_env_prefix.new_zeros(batch_size, max_days, self.env_dim)
        )
        return self._finalize_outputs(
            all_logits=all_logits,
            landmark_period=landmark_period,
            pred_ct_delta=pred_ct_delta,
            aux_next_env=aux_next_env,
            attention_weights=attention_weights,
            period_attention_weights=None,
            spread_entropy=None,
            spread_valid_mask=None,
        )

    def _forward_period_ms(
        self,
        daily_env_prefix: torch.Tensor,
        period_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_days: torch.Tensor,
        seq_len_periods: torch.Tensor,
        landmark_period: torch.Tensor,
        period_observed_mask: torch.Tensor | None,
    ) -> dict:
        batch_size, _, _ = period_env_prefix.shape
        period_tokens = self.period_projector(period_env_prefix)
        period_ids = self.period_ids.to(seq_len_periods.device)
        valid_period_mask = period_ids <= seq_len_periods.unsqueeze(1)
        if period_observed_mask is not None:
            valid_period_mask = valid_period_mask & period_observed_mask.bool()

        lag_indices = (landmark_period.unsqueeze(1) - period_ids + 1).clamp(min=0, max=self.num_periods)
        lag_embeddings = self.period_lag_embedding(lag_indices)
        attention_logits = self.period_attention_score(torch.tanh(period_tokens + lag_embeddings)).squeeze(-1)
        period_attention_weights = _masked_softmax(attention_logits, valid_period_mask, dim=1)

        weighted_tokens = period_attention_weights.unsqueeze(-1) * period_tokens
        attention_context = weighted_tokens.sum(dim=1)

        masked_tokens = period_tokens.masked_fill(~valid_period_mask.unsqueeze(-1), -1e9)
        period_maxpool = masked_tokens.max(dim=1).values
        no_history_mask = valid_period_mask.sum(dim=1) <= 0
        if no_history_mask.any():
            attention_context = attention_context.clone()
            period_maxpool = period_maxpool.clone()
            attention_context[no_history_mask] = self.no_history_period
            period_maxpool[no_history_mask] = self.no_history_period

        static_summary = self.period_static_projector(static_x)
        head_input = torch.cat([attention_context, period_maxpool, static_summary], dim=1)
        all_logits = self.period_survival_head(head_input)
        landmark_summary = self.landmark_embedding(
            landmark_period.long().clamp(min=0, max=self.num_periods)
        )
        pred_ct_delta = self.period_ct_aux_head(torch.cat([head_input, landmark_summary], dim=1))
        attention_weights = self._expand_period_attention_to_days(
            period_attention_weights=period_attention_weights,
            seq_len_days=seq_len_days,
            seq_len_periods=seq_len_periods,
            max_days=daily_env_prefix.size(1),
            period_observed_mask=period_observed_mask,
        )
        spread_entropy = -(period_attention_weights.clamp_min(1e-8) * period_attention_weights.clamp_min(1e-8).log()).sum(dim=1)
        spread_valid_mask = valid_period_mask.sum(dim=1) >= 3
        return self._finalize_outputs(
            all_logits=all_logits,
            landmark_period=landmark_period,
            pred_ct_delta=pred_ct_delta,
            aux_next_env=daily_env_prefix.new_zeros(batch_size, daily_env_prefix.size(1), self.env_dim),
            attention_weights=attention_weights,
            period_attention_weights=period_attention_weights,
            spread_entropy=spread_entropy,
            spread_valid_mask=spread_valid_mask,
        )

    def _forward_legacy_period_mean(
        self,
        period_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_periods: torch.Tensor,
        landmark_period: torch.Tensor,
    ) -> dict:
        batch_size, max_periods, _ = period_env_prefix.shape
        static_repeated = static_x.unsqueeze(1).expand(batch_size, max_periods, static_x.size(1))

        feature_parts = [period_env_prefix, static_repeated]
        if self.use_time_features:
            period_positions = (
                torch.arange(max_periods, device=period_env_prefix.device, dtype=torch.float32)
                / float(max(max_periods, 1))
            ).view(1, max_periods, 1).expand(batch_size, -1, -1)
            period_feature = (
                self.period_ids[:, :max_periods].to(period_env_prefix.device).float() / float(self.num_periods)
            ).unsqueeze(-1).expand(batch_size, -1, -1)
            feature_parts.extend([period_positions, period_feature])

        period_inputs = torch.cat(feature_parts, dim=-1)
        projected = self.day_projector(period_inputs)

        packed = pack_padded_sequence(
            projected,
            lengths=seq_len_periods.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden_state = self.rnn(packed)
        rnn_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=max_periods,
        )

        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]
        last_hidden = hidden_state[-1]

        time_index = torch.arange(max_periods, device=period_env_prefix.device).unsqueeze(0)
        valid_mask = time_index < seq_len_periods.unsqueeze(1)

        attention_logits = self.attention_score(rnn_outputs).squeeze(-1)
        attention_weights = _masked_softmax(attention_logits, valid_mask, dim=1)
        attention_context = torch.bmm(
            attention_weights.unsqueeze(1),
            rnn_outputs,
        ).squeeze(1)

        static_summary = self.static_projector(static_x)
        head_input = torch.cat([last_hidden, attention_context, static_summary], dim=1)
        all_logits = self.survival_head(head_input)
        landmark_summary = self.landmark_embedding(
            landmark_period.long().clamp(min=0, max=self.num_periods)
        )
        pred_ct_delta = self.ct_aux_head(torch.cat([head_input, landmark_summary], dim=1))
        aux_next_env = (
            self.auxiliary_head(rnn_outputs)
            if self.env_aux_mode == "next_day"
            else period_env_prefix.new_zeros(batch_size, max_periods, self.legacy_aux_target_dim)
        )
        spread_entropy = -(attention_weights.clamp_min(1e-8) * attention_weights.clamp_min(1e-8).log()).sum(dim=1)
        spread_valid_mask = valid_mask.sum(dim=1) >= 3
        return self._finalize_outputs(
            all_logits=all_logits,
            landmark_period=landmark_period,
            pred_ct_delta=pred_ct_delta,
            aux_next_env=aux_next_env,
            attention_weights=attention_weights,
            period_attention_weights=attention_weights,
            spread_entropy=spread_entropy,
            spread_valid_mask=spread_valid_mask,
        )

    def _forward_period_ms_tree_query(
        self,
        daily_env_prefix: torch.Tensor,
        period_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_days: torch.Tensor,
        seq_len_periods: torch.Tensor,
        landmark_period: torch.Tensor,
        period_observed_mask: torch.Tensor | None,
    ) -> dict:
        batch_size, _, _ = period_env_prefix.shape
        period_tokens = self.period_projector(period_env_prefix)
        static_summary = self.period_static_projector(static_x)

        film_gamma, film_beta = self.tree_period_film(static_summary).chunk(2, dim=1)
        period_tokens = period_tokens * (1.0 + 0.1 * torch.tanh(film_gamma).unsqueeze(1))
        period_tokens = period_tokens + 0.1 * film_beta.unsqueeze(1)

        period_ids = self.period_ids.to(seq_len_periods.device)
        valid_period_mask = period_ids <= seq_len_periods.unsqueeze(1)
        if period_observed_mask is not None:
            valid_period_mask = valid_period_mask & period_observed_mask.bool()

        lag_indices = (landmark_period.unsqueeze(1) - period_ids + 1).clamp(min=0, max=self.num_periods)
        lag_embeddings = self.period_lag_embedding(lag_indices)
        static_gate = self.tree_static_gate(static_summary).unsqueeze(1)
        interaction_tokens = period_tokens * static_gate

        attention_hidden = (
            self.tree_attn_period_proj(period_tokens)
            + self.tree_attn_lag_proj(lag_embeddings)
            + self.tree_attn_static_proj(static_summary).unsqueeze(1)
            + self.tree_attn_interaction_proj(interaction_tokens)
        )
        attention_logits = self.tree_attention_score(
            self.tree_attention_dropout(torch.tanh(attention_hidden))
        ).squeeze(-1)
        period_attention_weights = _masked_softmax(attention_logits, valid_period_mask, dim=1)

        attention_context = (period_attention_weights.unsqueeze(-1) * period_tokens).sum(dim=1)
        masked_tokens = period_tokens.masked_fill(~valid_period_mask.unsqueeze(-1), -1e9)
        period_maxpool = masked_tokens.max(dim=1).values
        no_history_mask = valid_period_mask.sum(dim=1) <= 0
        if no_history_mask.any():
            attention_context = attention_context.clone()
            period_maxpool = period_maxpool.clone()
            attention_context[no_history_mask] = self.no_history_period
            period_maxpool[no_history_mask] = self.no_history_period

        landmark_summary = self.landmark_embedding(
            landmark_period.long().clamp(min=0, max=self.num_periods)
        )
        head_input = torch.cat([attention_context, period_maxpool, static_summary, landmark_summary], dim=1)
        all_logits = self.period_survival_head(head_input)
        pred_ct_delta = self.period_ct_aux_head(head_input)
        attention_weights = self._expand_period_attention_to_days(
            period_attention_weights=period_attention_weights,
            seq_len_days=seq_len_days,
            seq_len_periods=seq_len_periods,
            max_days=daily_env_prefix.size(1),
            period_observed_mask=period_observed_mask,
        )
        spread_entropy = -(period_attention_weights.clamp_min(1e-8) * period_attention_weights.clamp_min(1e-8).log()).sum(dim=1)
        spread_valid_mask = valid_period_mask.sum(dim=1) >= 3
        return self._finalize_outputs(
            all_logits=all_logits,
            landmark_period=landmark_period,
            pred_ct_delta=pred_ct_delta,
            aux_next_env=daily_env_prefix.new_zeros(batch_size, daily_env_prefix.size(1), self.env_dim),
            attention_weights=attention_weights,
            period_attention_weights=period_attention_weights,
            spread_entropy=spread_entropy,
            spread_valid_mask=spread_valid_mask,
        )

    def _forward_trigger_orchard(
        self,
        daily_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_days: torch.Tensor,
        landmark_period: torch.Tensor,
    ) -> dict:
        batch_size, max_days, _ = daily_env_prefix.shape
        time_index = torch.arange(max_days, device=daily_env_prefix.device).unsqueeze(0)
        valid_mask = time_index < seq_len_days.unsqueeze(1)

        base = self.trigger_input_proj(daily_env_prefix)
        conv_input = base.transpose(1, 2)
        conv_outputs = []
        for conv in self.trigger_convs:
            kernel_size = int(conv.kernel_size[0])
            padded = F.pad(conv_input, (kernel_size - 1, 0))
            conv_outputs.append(F.relu(conv(padded)).transpose(1, 2))
        day_hidden = self.trigger_daily_encoder(torch.cat([base] + conv_outputs, dim=-1))

        expert_contexts: List[torch.Tensor] = []
        expert_weights: List[torch.Tensor] = []
        for expert_index, feature_indices in enumerate(self.expert_feature_indices):
            if len(feature_indices) == 0:
                subset = daily_env_prefix[..., :1]
            else:
                subset = daily_env_prefix[..., feature_indices]
            subset_proj = self.expert_subset_projectors[expert_index](subset)
            score_input = torch.cat([day_hidden, subset_proj], dim=-1)
            scores = self.expert_score_heads[expert_index](score_input).squeeze(-1)
            expert_context, day_weights = self._topk_pool(
                values=day_hidden,
                scores=scores,
                valid_mask=valid_mask,
                topk=self.trigger_topk,
            )
            expert_contexts.append(expert_context)
            expert_weights.append(day_weights)

        expert_context_tensor = torch.stack(expert_contexts, dim=1)
        expert_weight_tensor = torch.stack(expert_weights, dim=1)
        expert_gate = torch.softmax(self.static_expert_gate(static_x), dim=1)
        weighted_context = (expert_gate.unsqueeze(-1) * expert_context_tensor).sum(dim=1)
        attention_weights = (expert_gate.unsqueeze(-1) * expert_weight_tensor).sum(dim=1)

        static_summary = self.static_projector(static_x)
        head_input = torch.cat(
            [
                expert_context_tensor[:, 0, :],
                expert_context_tensor[:, 1, :],
                expert_context_tensor[:, 2, :],
                weighted_context,
                static_summary,
            ],
            dim=1,
        )
        all_logits = self.trigger_survival_head(head_input)
        landmark_summary = self.landmark_embedding(
            landmark_period.long().clamp(min=0, max=self.num_periods)
        )
        pred_ct_delta = self.trigger_ct_aux_head(torch.cat([head_input, landmark_summary], dim=1))
        return self._finalize_outputs(
            all_logits=all_logits,
            landmark_period=landmark_period,
            pred_ct_delta=pred_ct_delta,
            aux_next_env=daily_env_prefix.new_zeros(batch_size, max_days, self.env_dim),
            attention_weights=attention_weights,
            period_attention_weights=None,
            spread_entropy=None,
            spread_valid_mask=None,
        )

    def _forward_trigger_orchard_v2(
        self,
        daily_env_prefix: torch.Tensor,
        static_x: torch.Tensor,
        seq_len_days: torch.Tensor,
        landmark_period: torch.Tensor,
    ) -> dict:
        batch_size, max_days, _ = daily_env_prefix.shape

        base = self.trigger_input_proj(daily_env_prefix)
        conv_input = base.transpose(1, 2)
        conv_outputs = []
        for conv in self.trigger_convs:
            kernel_size = int(conv.kernel_size[0])
            padded = F.pad(conv_input, (kernel_size - 1, 0))
            conv_outputs.append(F.relu(conv(padded)).transpose(1, 2))
        day_hidden = self.trigger_daily_encoder(torch.cat([base] + conv_outputs, dim=-1))

        window_starts = self.trigger_window_start_index.to(daily_env_prefix.device)
        window_ends = self.trigger_window_end_index.to(daily_env_prefix.device)
        window_lengths = self.trigger_window_length.to(daily_env_prefix.device)
        window_types = self.trigger_window_type_id.to(daily_env_prefix.device)
        window_end_periods = self.trigger_window_end_period.to(daily_env_prefix.device)

        valid_window_mask = window_ends.unsqueeze(0) < seq_len_days.unsqueeze(1)
        window_hidden_mean = self._pool_window_means(
            values=day_hidden,
            window_starts=window_starts,
            window_ends=window_ends,
            window_lengths=window_lengths,
        )
        window_type_embedding = self.trigger_window_type_embedding(window_types).unsqueeze(0).expand(batch_size, -1, -1)
        lag_indices = (
            landmark_period.unsqueeze(1) - window_end_periods.unsqueeze(0) + 1
        ).clamp(min=0, max=self.num_periods)
        lag_embedding = self.trigger_window_lag_embedding(lag_indices)
        normalized_length = (
            window_lengths.float() / float(max(max_days, 1))
        ).view(1, -1, 1).expand(batch_size, -1, -1)
        day_gap = (
            seq_len_days.unsqueeze(1).float() - 1.0 - window_ends.unsqueeze(0).float()
        ).clamp(min=0.0)
        normalized_day_gap = (day_gap / float(max(max_days, 1))).unsqueeze(-1)

        expert_contexts: List[torch.Tensor] = []
        expert_day_weights: List[torch.Tensor] = []
        expert_window_weights: List[torch.Tensor] = []
        for expert_index, feature_indices in enumerate(self.expert_feature_indices):
            if len(feature_indices) == 0:
                subset = daily_env_prefix[..., :1]
            else:
                subset = daily_env_prefix[..., feature_indices]
            subset_mean = self._pool_window_means(
                values=subset,
                window_starts=window_starts,
                window_ends=window_ends,
                window_lengths=window_lengths,
            )
            subset_proj = self.expert_subset_projectors[expert_index](subset_mean)
            window_token_input = torch.cat(
                [
                    window_hidden_mean,
                    subset_proj,
                    window_type_embedding,
                    lag_embedding,
                    normalized_length,
                    normalized_day_gap,
                ],
                dim=-1,
            )
            window_tokens = self.expert_window_projectors[expert_index](window_token_input)
            tree_query = F.normalize(self.expert_query_projectors[expert_index](static_x), dim=1)
            window_keys = F.normalize(window_tokens, dim=-1)
            scores = (window_keys * tree_query.unsqueeze(1)).sum(dim=-1)
            expert_context, day_weights, window_weights = self._topk_window_pool(
                values=window_tokens,
                scores=scores,
                valid_mask=valid_window_mask,
                window_starts=window_starts,
                window_ends=window_ends,
                topk=self.trigger_topk,
                max_days=max_days,
            )
            expert_contexts.append(expert_context)
            expert_day_weights.append(day_weights)
            expert_window_weights.append(window_weights)

        expert_context_tensor = torch.stack(expert_contexts, dim=1)
        expert_day_weight_tensor = torch.stack(expert_day_weights, dim=1)
        expert_window_weight_tensor = torch.stack(expert_window_weights, dim=1)

        expert_gate = torch.softmax(self.static_expert_gate(static_x), dim=1)
        weighted_context = (expert_gate.unsqueeze(-1) * expert_context_tensor).sum(dim=1)
        attention_weights = (expert_gate.unsqueeze(-1) * expert_day_weight_tensor).sum(dim=1)
        mixed_window_weights = (expert_gate.unsqueeze(-1) * expert_window_weight_tensor).sum(dim=1)

        static_summary = self.static_projector(static_x)
        head_input = torch.cat(
            [
                expert_context_tensor[:, 0, :],
                expert_context_tensor[:, 1, :],
                expert_context_tensor[:, 2, :],
                weighted_context,
                static_summary,
            ],
            dim=1,
        )
        all_logits = self.trigger_survival_head(head_input)
        landmark_summary = self.landmark_embedding(
            landmark_period.long().clamp(min=0, max=self.num_periods)
        )
        pred_ct_delta = self.trigger_ct_aux_head(torch.cat([head_input, landmark_summary], dim=1))
        output = self._finalize_outputs(
            all_logits=all_logits,
            landmark_period=landmark_period,
            pred_ct_delta=pred_ct_delta,
            aux_next_env=daily_env_prefix.new_zeros(batch_size, max_days, self.env_dim),
            attention_weights=attention_weights,
            period_attention_weights=None,
            spread_entropy=None,
            spread_valid_mask=None,
        )
        if not self.training:
            output["trigger_window_weights"] = mixed_window_weights
            output["trigger_window_start_index"] = window_starts
            output["trigger_window_end_index"] = window_ends
            output["trigger_window_type_id"] = window_types
            output["trigger_window_end_period"] = window_end_periods
        return output

    def _finalize_outputs(
        self,
        all_logits: torch.Tensor,
        landmark_period: torch.Tensor,
        pred_ct_delta: torch.Tensor,
        aux_next_env: torch.Tensor,
        attention_weights: torch.Tensor,
        period_attention_weights: torch.Tensor | None,
        spread_entropy: torch.Tensor | None,
        spread_valid_mask: torch.Tensor | None,
    ) -> dict:
        event_logits = all_logits[:, : self.num_periods]
        tail_logit = all_logits[:, self.num_periods :]
        masked_event_logits = event_logits.masked_fill(
            self.period_ids.to(landmark_period.device) <= landmark_period.unsqueeze(1),
            float("-inf"),
        )
        survival_logits = torch.cat([masked_event_logits, tail_logit], dim=1)
        survival_probs = torch.softmax(survival_logits, dim=1)
        event_probs = survival_probs[:, : self.num_periods]
        tail_prob = survival_probs[:, self.num_periods]

        output = {
            "event_probs": event_probs,
            "tail_prob": tail_prob,
            "aux_next_env": aux_next_env,
            "pred_ct_delta": pred_ct_delta,
            "attention_weights": attention_weights,
        }
        if period_attention_weights is not None:
            output["period_attention_weights"] = period_attention_weights
        if spread_entropy is not None and spread_valid_mask is not None:
            output["spread_entropy"] = spread_entropy
            output["spread_valid_mask"] = spread_valid_mask
        return output

    def _expand_period_attention_to_days(
        self,
        period_attention_weights: torch.Tensor,
        seq_len_days: torch.Tensor,
        seq_len_periods: torch.Tensor,
        max_days: int,
        period_observed_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = period_attention_weights.size(0)
        device = period_attention_weights.device
        day_period_ids = self.day_to_period_long[:, :max_days].to(device).expand(batch_size, -1)
        time_index = torch.arange(max_days, device=device).unsqueeze(0)
        valid_day_mask = time_index < seq_len_days.unsqueeze(1)
        observed_period_mask = day_period_ids <= seq_len_periods.unsqueeze(1)
        if period_observed_mask is not None:
            gathered_mask = period_observed_mask.to(device).bool().gather(
                1,
                (day_period_ids.clamp(min=1, max=self.num_periods) - 1).long(),
            )
            observed_period_mask = observed_period_mask & gathered_mask
        valid_mask = valid_day_mask & observed_period_mask

        daily_attention = period_attention_weights.new_zeros(batch_size, max_days)
        for period_index in range(1, self.num_periods + 1):
            period_day_mask = valid_mask & (day_period_ids == period_index)
            if not period_day_mask.any():
                continue
            counts = period_day_mask.sum(dim=1).clamp_min(1)
            period_weight = period_attention_weights[:, period_index - 1]
            daily_attention = daily_attention + period_day_mask.float() * (
                period_weight.unsqueeze(1) / counts.unsqueeze(1)
            )
        return daily_attention

    def _topk_pool(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_days, hidden_dim = values.shape
        device = values.device
        full_weights = values.new_zeros(batch_size, max_days)
        contexts = values.new_zeros(batch_size, hidden_dim)

        masked_scores = scores.masked_fill(~valid_mask, -1e9)
        for sample_index in range(batch_size):
            sample_valid = valid_mask[sample_index]
            valid_count = int(sample_valid.sum().item())
            if valid_count <= 0:
                contexts[sample_index] = self.no_history_day
                continue

            sample_scores = masked_scores[sample_index]
            k = min(int(topk), valid_count)
            top_scores, top_indices = torch.topk(sample_scores, k=k, dim=0)
            top_weights = torch.softmax(top_scores, dim=0)
            full_weights[sample_index, top_indices] = top_weights
            contexts[sample_index] = (top_weights.unsqueeze(1) * values[sample_index, top_indices]).sum(dim=0)

        return contexts, full_weights

    def _topk_window_pool(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        window_starts: torch.Tensor,
        window_ends: torch.Tensor,
        topk: int,
        max_days: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_windows, hidden_dim = values.shape
        day_weights = values.new_zeros(batch_size, max_days)
        window_weights = values.new_zeros(batch_size, num_windows)
        contexts = values.new_zeros(batch_size, hidden_dim)

        for sample_index in range(batch_size):
            sample_valid = torch.nonzero(valid_mask[sample_index], as_tuple=False).view(-1)
            if sample_valid.numel() <= 0:
                contexts[sample_index] = self.no_history_day
                continue

            sample_scores = scores[sample_index, sample_valid]
            ordered_local = torch.argsort(sample_scores, descending=True)
            selected_indices: List[int] = []
            selected_ranges: List[tuple[int, int]] = []

            for local_index in ordered_local.tolist():
                window_index = int(sample_valid[local_index].item())
                start_day = int(window_starts[window_index].item())
                end_day = int(window_ends[window_index].item())
                overlaps_existing = any(
                    not (end_day < chosen_start or start_day > chosen_end)
                    for chosen_start, chosen_end in selected_ranges
                )
                if overlaps_existing:
                    continue
                selected_indices.append(window_index)
                selected_ranges.append((start_day, end_day))
                if len(selected_indices) >= int(topk):
                    break

            if not selected_indices:
                fallback_index = int(sample_valid[torch.argmax(sample_scores)].item())
                selected_indices = [fallback_index]
                selected_ranges = [
                    (
                        int(window_starts[fallback_index].item()),
                        int(window_ends[fallback_index].item()),
                    )
                ]

            selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=values.device)
            selected_scores = scores[sample_index, selected_tensor]
            selected_weights = torch.softmax(selected_scores, dim=0)
            window_weights[sample_index, selected_tensor] = selected_weights
            contexts[sample_index] = (
                selected_weights.unsqueeze(1) * values[sample_index, selected_tensor]
            ).sum(dim=0)

            for local_weight, (start_day, end_day) in zip(selected_weights, selected_ranges):
                length = max(end_day - start_day + 1, 1)
                day_weights[sample_index, start_day : end_day + 1] += local_weight / float(length)

        return contexts, day_weights, window_weights

    @staticmethod
    def _pool_window_means(
        values: torch.Tensor,
        window_starts: torch.Tensor,
        window_ends: torch.Tensor,
        window_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, feature_dim = values.shape
        prefix = torch.cat(
            [values.new_zeros(batch_size, 1, feature_dim), values.cumsum(dim=1)],
            dim=1,
        )
        gather_start = window_starts.view(1, -1, 1).expand(batch_size, -1, feature_dim)
        gather_end = (window_ends + 1).view(1, -1, 1).expand(batch_size, -1, feature_dim)
        window_sum = prefix.gather(1, gather_end) - prefix.gather(1, gather_start)
        return window_sum / window_lengths.view(1, -1, 1).clamp_min(1.0)

    @staticmethod
    def _infer_period_day_bounds(day_to_period: torch.Tensor, num_periods: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_indices = []
        end_indices = []
        for period_index in range(1, num_periods + 1):
            positions = torch.nonzero(day_to_period == period_index, as_tuple=False).view(-1)
            if positions.numel() == 0:
                start_indices.append(0)
                end_indices.append(0)
            else:
                start_indices.append(int(positions.min().item()))
                end_indices.append(int(positions.max().item()))
        return torch.tensor(start_indices, dtype=torch.long), torch.tensor(end_indices, dtype=torch.long)

    def _build_trigger_window_catalog(
        self,
        short_window_lengths: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_days = int(self.day_to_period_long.size(1))
        day_to_period = self.day_to_period_long.view(-1).tolist()
        period_starts = self.period_start_day_index.view(-1).tolist()
        period_ends = self.period_end_day_index.view(-1).tolist()

        starts: List[int] = []
        ends: List[int] = []
        lengths: List[float] = []
        type_ids: List[int] = []
        end_periods: List[int] = []

        for type_id, window_length in enumerate(short_window_lengths):
            for end_day in range(window_length - 1, max_days):
                start_day = end_day - window_length + 1
                starts.append(start_day)
                ends.append(end_day)
                lengths.append(float(window_length))
                type_ids.append(type_id)
                end_periods.append(int(day_to_period[end_day]))

        offset = len(short_window_lengths)
        for period_index in range(self.num_periods):
            start_day = int(period_starts[period_index])
            end_day = int(period_ends[period_index])
            starts.append(start_day)
            ends.append(end_day)
            lengths.append(float(end_day - start_day + 1))
            type_ids.append(offset)
            end_periods.append(period_index + 1)

        for span, type_id in ((2, offset + 1), (3, offset + 2)):
            for end_period_index in range(span - 1, self.num_periods):
                start_period_index = end_period_index - span + 1
                start_day = int(period_starts[start_period_index])
                end_day = int(period_ends[end_period_index])
                starts.append(start_day)
                ends.append(end_day)
                lengths.append(float(end_day - start_day + 1))
                type_ids.append(type_id)
                end_periods.append(end_period_index + 1)

        prefix_type_id = offset + 3
        for end_period_index in range(self.num_periods):
            start_day = int(period_starts[0])
            end_day = int(period_ends[end_period_index])
            starts.append(start_day)
            ends.append(end_day)
            lengths.append(float(end_day - start_day + 1))
            type_ids.append(prefix_type_id)
            end_periods.append(end_period_index + 1)

        return (
            torch.tensor(starts, dtype=torch.long),
            torch.tensor(ends, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.float32),
            torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(end_periods, dtype=torch.long),
        )

    @staticmethod
    def _resolve_expert_feature_indices(env_feature_names: Sequence[str]) -> List[List[int]]:
        names = [name.lower() for name in env_feature_names]

        def collect_indices(keywords: Sequence[str]) -> List[int]:
            indices = [
                index
                for index, name in enumerate(names)
                if any(keyword in name for keyword in keywords)
            ]
            if not indices:
                return list(range(len(names)))
            return sorted(set(indices))

        flush_indices = collect_indices(
            [
                "air_temp",
                "air_humidity",
                "light",
                "wind_speed",
                "wind_sin",
                "wind_cos",
                "rain",
                "temp_25_28",
                "temp_gt_33",
                "rainy_day",
                "dry_day",
                "low_light",
            ]
        )
        root_indices = collect_indices(
            [
                "soil_temp",
                "soil_moisture",
                "soil_ec",
                "rain",
                "low_soil_moist",
                "high_soil_moist",
                "high_ec",
            ]
        )
        mixed_indices = list(range(len(names)))
        return [flush_indices, root_indices, mixed_indices]
