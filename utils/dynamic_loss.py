"""
Loss functions for dynamic HLB survival modeling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dynamic_nll_loss(
    event_probs: torch.Tensor,
    tail_prob: torch.Tensor,
    event_flag: torch.Tensor,
    time_period: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Conditional negative log-likelihood for single-event discrete survival.
    """
    event_flag = event_flag.view(-1)
    time_period = time_period.view(-1).long()
    batch_indices = torch.arange(event_probs.size(0), device=event_probs.device)

    losses = []

    event_mask = event_flag > 0.5
    if event_mask.any():
        event_probability = event_probs[
            batch_indices[event_mask],
            time_period[event_mask] - 1,
        ]
        losses.append(-torch.log(event_probability.clamp_min(eps)))

    censor_mask = ~event_mask
    if censor_mask.any():
        censor_period = time_period[censor_mask]
        survival_terms = []
        for sample_index, period in zip(batch_indices[censor_mask], censor_period):
            period_index = int(period.item())
            future_event_mass = event_probs[sample_index, period_index:].sum()
            survival_terms.append(future_event_mass + tail_prob[sample_index])
        survival_probability = torch.stack(survival_terms)
        losses.append(-torch.log(survival_probability.clamp_min(eps)))

    if not losses:
        return event_probs.sum() * 0.0
    return torch.cat(losses).mean()


def dynamic_ranking_loss(
    event_probs: torch.Tensor,
    event_flag: torch.Tensor,
    time_period: torch.Tensor,
    landmark_period: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Dynamic-DeepHit style ranking loss using comparable pairs within each landmark.
    """
    event_flag = event_flag.view(-1)
    time_period = time_period.view(-1).long()
    landmark_period = landmark_period.view(-1).long()

    ranking_terms = []
    unique_landmarks = torch.unique(landmark_period)

    for landmark_value in unique_landmarks.tolist():
        group_mask = landmark_period == landmark_value
        group_indices = torch.nonzero(group_mask, as_tuple=False).view(-1)
        if group_indices.numel() <= 1:
            continue

        group_times = time_period[group_indices]
        group_events = event_flag[group_indices]
        group_probs = event_probs[group_indices]

        event_indices = torch.nonzero(group_events > 0.5, as_tuple=False).view(-1)
        for local_event_index in event_indices.tolist():
            event_period = int(group_times[local_event_index].item())
            comparable_mask = group_times > event_period
            comparable_mask[local_event_index] = False
            comparable_indices = torch.nonzero(comparable_mask, as_tuple=False).view(-1)
            if comparable_indices.numel() == 0:
                continue

            risk_slice = slice(landmark_value, event_period)
            cumulative_risk = group_probs[:, risk_slice].sum(dim=1)
            event_risk = cumulative_risk[local_event_index]
            comparison_risk = cumulative_risk[comparable_indices]
            ranking_terms.append(torch.exp(-(event_risk - comparison_risk) / sigma).mean())

    if not ranking_terms:
        return event_probs.sum() * 0.0
    return torch.stack(ranking_terms).mean()


def auxiliary_next_day_loss(
    aux_next_env: torch.Tensor,
    env_targets: torch.Tensor,
    seq_len_days: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error for next-day environment prediction.
    """
    if aux_next_env.size(1) <= 1:
        return aux_next_env.sum() * 0.0

    pred = aux_next_env[:, :-1, :]
    target = env_targets[:, 1:, :]
    valid_steps = torch.arange(pred.size(1), device=pred.device).unsqueeze(0)
    mask = valid_steps < (seq_len_days.unsqueeze(1) - 1)

    if not mask.any():
        return aux_next_env.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def ct_delta_auxiliary_loss(
    pred_ct_delta: torch.Tensor,
    ct_delta_target: torch.Tensor,
    ct_aux_mask: torch.Tensor,
    target_mean: float,
    target_std: float,
    loss_type: str = "huber",
    ct_aux_window_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CT auxiliary loss on valid prefix samples only.
    """
    mask = _combined_ct_aux_mask(ct_aux_mask, ct_aux_window_mask)
    if not mask.any():
        return pred_ct_delta.sum() * 0.0

    safe_std = max(float(target_std), 1e-8)
    pred, target = _align_ct_delta_tensors(pred_ct_delta, ct_delta_target, mask)
    pred = pred[mask]
    target = (target[mask] - float(target_mean)) / safe_std

    if loss_type == "huber":
        return F.huber_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported CT auxiliary loss: {loss_type}")


def ct_delta_mean_absolute_error(
    pred_ct_delta: torch.Tensor,
    ct_delta_target: torch.Tensor,
    ct_aux_mask: torch.Tensor,
    target_mean: float,
    target_std: float,
    ct_aux_window_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Mean absolute error in the original CT-delta scale.
    """
    mask = _combined_ct_aux_mask(ct_aux_mask, ct_aux_window_mask)
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return pred_ct_delta.sum() * 0.0, 0

    safe_std = max(float(target_std), 1e-8)
    pred, target = _align_ct_delta_tensors(pred_ct_delta, ct_delta_target, mask)
    pred = pred[mask] * safe_std + float(target_mean)
    target = target[mask]
    return torch.abs(pred - target).mean(), valid_count


def ct_endpoint_auxiliary_loss(
    pred_ct_state_logits: torch.Tensor,
    ct_state_target: torch.Tensor,
    ct_state_mask: torch.Tensor,
    pred_ct_value: torch.Tensor,
    ct_value_target: torch.Tensor,
    ct_value_mask: torch.Tensor,
    target_mean: float,
    target_std: float,
    loss_type: str = "huber",
) -> dict[str, torch.Tensor | int]:
    """
    CT auxiliary loss for window-aligned endpoint state/value heads.
    """
    zero = pred_ct_state_logits.sum() * 0.0 + pred_ct_value.sum() * 0.0
    state_mask = ct_state_mask > 0.5
    value_mask = ct_value_mask > 0.5

    state_valid_n = int(state_mask.sum().item())
    value_valid_n = int(value_mask.sum().item())

    state_loss = zero
    if state_valid_n > 0:
        state_loss = F.binary_cross_entropy_with_logits(
            pred_ct_state_logits[state_mask],
            ct_state_target[state_mask],
        )

    value_loss = zero
    if value_valid_n > 0:
        safe_std = max(float(target_std), 1e-8)
        normalized_target = (ct_value_target[value_mask] - float(target_mean)) / safe_std
        pred_value = pred_ct_value[value_mask]
        if loss_type == "huber":
            value_loss = F.huber_loss(pred_value, normalized_target)
        elif loss_type == "mse":
            value_loss = F.mse_loss(pred_value, normalized_target)
        else:
            raise ValueError(f"Unsupported CT auxiliary loss: {loss_type}")

    active_losses = []
    if state_valid_n > 0:
        active_losses.append(state_loss)
    if value_valid_n > 0:
        active_losses.append(value_loss)
    total_loss = zero if not active_losses else torch.stack(active_losses).mean()

    return {
        "total": total_loss,
        "state": state_loss,
        "value": value_loss,
        "state_valid_n": state_valid_n,
        "value_valid_n": value_valid_n,
    }


def ct_delta_endpoint_lod_auxiliary_loss(
    pred_ct_delta: torch.Tensor,
    ct_delta_target: torch.Tensor,
    ct_aux_mask: torch.Tensor,
    pred_ct_state_logits: torch.Tensor,
    ct_state_target: torch.Tensor,
    ct_state_mask: torch.Tensor,
    target_mean: float,
    target_std: float,
    loss_type: str = "huber",
    ct_aux_window_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """
    CT auxiliary loss for global delta regression plus endpoint LOD classification.
    """
    zero = pred_ct_delta.sum() * 0.0 + pred_ct_state_logits.sum() * 0.0
    delta_mask = _combined_ct_aux_mask(ct_aux_mask, ct_aux_window_mask)
    delta_valid_n = int(delta_mask.sum().item())
    state_mask = ct_state_mask > 0.5
    state_valid_n = int(state_mask.sum().item())

    delta_loss = zero
    if delta_valid_n > 0:
        delta_loss = ct_delta_auxiliary_loss(
            pred_ct_delta=pred_ct_delta,
            ct_delta_target=ct_delta_target,
            ct_aux_mask=ct_aux_mask,
            target_mean=target_mean,
            target_std=target_std,
            loss_type=loss_type,
            ct_aux_window_mask=ct_aux_window_mask,
        )

    state_loss = zero
    if state_valid_n > 0:
        state_loss = F.binary_cross_entropy_with_logits(
            pred_ct_state_logits[state_mask],
            ct_state_target[state_mask],
        )

    active_losses = []
    if delta_valid_n > 0:
        active_losses.append(delta_loss)
    if state_valid_n > 0:
        active_losses.append(state_loss)
    total_loss = zero if not active_losses else torch.stack(active_losses).mean()

    return {
        "total": total_loss,
        "delta": delta_loss,
        "state": state_loss,
        "delta_valid_n": delta_valid_n,
        "state_valid_n": state_valid_n,
    }


def ct_value_mean_absolute_error(
    pred_ct_value: torch.Tensor,
    ct_value_target: torch.Tensor,
    ct_value_mask: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> tuple[torch.Tensor, int]:
    """
    Mean absolute error in the original CT scale on exact-value targets only.
    """
    valid_mask = ct_value_mask > 0.5
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return pred_ct_value.sum() * 0.0, 0

    safe_std = max(float(target_std), 1e-8)
    pred = pred_ct_value[valid_mask] * safe_std + float(target_mean)
    target = ct_value_target[valid_mask]
    return torch.abs(pred - target).mean(), valid_count


def ct_state_confusion_counts(
    pred_ct_state_logits: torch.Tensor,
    ct_state_target: torch.Tensor,
    ct_state_mask: torch.Tensor,
) -> dict[str, int]:
    """
    Confusion counts for CT>=40 endpoint classification on valid windows only.
    """
    valid_mask = ct_state_mask > 0.5
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    pred_positive = pred_ct_state_logits[valid_mask] >= 0.0
    target_positive = ct_state_target[valid_mask] > 0.5
    return {
        "tp": int((pred_positive & target_positive).sum().item()),
        "tn": int((~pred_positive & ~target_positive).sum().item()),
        "fp": int((pred_positive & ~target_positive).sum().item()),
        "fn": int((~pred_positive & target_positive).sum().item()),
    }


def _combined_ct_aux_mask(
    ct_aux_mask: torch.Tensor,
    ct_aux_window_mask: torch.Tensor | None,
) -> torch.Tensor:
    sample_mask = ct_aux_mask.float()
    if ct_aux_window_mask is None:
        return sample_mask > 0.5

    window_mask = ct_aux_window_mask.float()
    if window_mask.dim() == sample_mask.dim():
        combined = sample_mask * window_mask
    elif window_mask.dim() == sample_mask.dim() + 1:
        combined = sample_mask.unsqueeze(1) * window_mask
    else:
        raise ValueError(
            "Unsupported CT auxiliary mask shapes: "
            f"sample={tuple(sample_mask.shape)}, window={tuple(window_mask.shape)}"
        )
    return combined > 0.5


def _align_ct_delta_tensors(
    pred_ct_delta: torch.Tensor,
    ct_delta_target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_ct_delta
    target = ct_delta_target
    while pred.dim() < mask.dim():
        pred = pred.unsqueeze(1)
    while target.dim() < mask.dim():
        target = target.unsqueeze(1)
    return pred.expand_as(mask), target.expand_as(mask)


def attention_spread_regularizer(
    spread_entropy: torch.Tensor,
    spread_valid_mask: torch.Tensor,
    seq_len_periods: torch.Tensor,
    entropy_ratio: float = 0.8,
) -> torch.Tensor:
    """
    Penalize collapsed period attention without forcing a uniform history profile.
    """
    valid_mask = spread_valid_mask.view(-1).bool() & (seq_len_periods.view(-1) >= 3)
    if not valid_mask.any():
        return spread_entropy.sum() * 0.0

    effective_lengths = seq_len_periods.view(-1)[valid_mask].float().clamp(min=3.0, max=4.0)
    target_entropy = torch.log(effective_lengths) * float(entropy_ratio)
    entropy_values = spread_entropy.view(-1)[valid_mask]
    return F.relu(target_entropy - entropy_values).mean()


def compute_dynamic_loss(
    model_output: dict,
    batch: dict,
    alpha: float,
    beta: float,
    gamma_env: float,
    gamma_ct: float,
    use_ct_aux_task: bool = False,
    ct_target_mean: float = 0.0,
    ct_target_std: float = 1.0,
    ct_aux_loss: str = "huber",
    env_aux_mode: str = "next_day",
    lag_spread_weight: float = 0.0,
) -> dict:
    """
    Compute the total dynamic training loss and its components.
    """
    ll_loss = dynamic_nll_loss(
        event_probs=model_output["event_probs"],
        tail_prob=model_output["tail_prob"],
        event_flag=batch["event_flag"],
        time_period=batch["time_period"],
    )
    ranking_loss = dynamic_ranking_loss(
        event_probs=model_output["event_probs"],
        event_flag=batch["event_flag"],
        time_period=batch["time_period"],
        landmark_period=batch["landmark_period"],
    )

    if env_aux_mode == "next_day":
        env_aux_loss = auxiliary_next_day_loss(
            aux_next_env=model_output["aux_next_env"],
            env_targets=batch["env"],
            seq_len_days=batch["seq_len_days"],
        )
    else:
        env_aux_loss = model_output["event_probs"].sum() * 0.0

    ct_aux_loss_value = env_aux_loss.new_zeros(())
    ct_state_loss_value = env_aux_loss.new_zeros(())
    ct_delta_loss_value = env_aux_loss.new_zeros(())
    ct_value_loss_value = env_aux_loss.new_zeros(())
    if use_ct_aux_task:
        if "pred_ct_state_logits" in model_output and "pred_ct_delta" in model_output:
            ct_hybrid_losses = ct_delta_endpoint_lod_auxiliary_loss(
                pred_ct_delta=model_output["pred_ct_delta"],
                ct_delta_target=batch["ct_delta_target"],
                ct_aux_mask=batch["ct_aux_mask"],
                pred_ct_state_logits=model_output["pred_ct_state_logits"],
                ct_state_target=batch["ct_state_target"],
                ct_state_mask=batch["ct_state_mask"],
                target_mean=ct_target_mean,
                target_std=ct_target_std,
                loss_type=ct_aux_loss,
                ct_aux_window_mask=batch.get("ct_aux_window_mask"),
            )
            ct_aux_loss_value = ct_hybrid_losses["total"]
            ct_delta_loss_value = ct_hybrid_losses["delta"]
            ct_state_loss_value = ct_hybrid_losses["state"]
        elif "pred_ct_state_logits" in model_output and "pred_ct_value" in model_output:
            ct_endpoint_losses = ct_endpoint_auxiliary_loss(
                pred_ct_state_logits=model_output["pred_ct_state_logits"],
                ct_state_target=batch["ct_state_target"],
                ct_state_mask=batch["ct_state_mask"],
                pred_ct_value=model_output["pred_ct_value"],
                ct_value_target=batch["ct_value_target"],
                ct_value_mask=batch["ct_value_mask"],
                target_mean=ct_target_mean,
                target_std=ct_target_std,
                loss_type=ct_aux_loss,
            )
            ct_aux_loss_value = ct_endpoint_losses["total"]
            ct_state_loss_value = ct_endpoint_losses["state"]
            ct_value_loss_value = ct_endpoint_losses["value"]
        else:
            ct_aux_loss_value = ct_delta_auxiliary_loss(
                pred_ct_delta=model_output["pred_ct_delta"],
                ct_delta_target=batch["ct_delta_target"],
                ct_aux_mask=batch["ct_aux_mask"],
                target_mean=ct_target_mean,
                target_std=ct_target_std,
                loss_type=ct_aux_loss,
                ct_aux_window_mask=batch.get("ct_aux_window_mask"),
            )
            ct_delta_loss_value = ct_aux_loss_value

    spread_loss = env_aux_loss.new_zeros(())
    if lag_spread_weight > 0.0 and "spread_entropy" in model_output and "spread_valid_mask" in model_output:
        spread_loss = attention_spread_regularizer(
            spread_entropy=model_output["spread_entropy"],
            spread_valid_mask=model_output["spread_valid_mask"],
            seq_len_periods=batch["seq_len_periods"],
        )

    total_loss = (
        alpha * ll_loss
        + beta * ranking_loss
        + gamma_env * env_aux_loss
        + gamma_ct * ct_aux_loss_value
        + lag_spread_weight * spread_loss
    )
    return {
        "total": total_loss,
        "ll": ll_loss,
        "rank": ranking_loss,
        "env_aux": env_aux_loss,
        "ct_aux": ct_aux_loss_value,
        "ct_state": ct_state_loss_value,
        "ct_delta": ct_delta_loss_value,
        "ct_value": ct_value_loss_value,
        "spread": spread_loss,
        "aux": env_aux_loss,
    }
