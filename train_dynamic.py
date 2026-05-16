"""
Training script for dynamic HLB survival analysis.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from models import DynamicReKSurv
from utils import (
    PrefixSample,
    PrefixSampleDataset,
    build_prefix_samples,
    compute_dynamic_loss,
    ct_delta_mean_absolute_error,
    ct_state_confusion_counts,
    ct_value_mean_absolute_error,
    fit_ct_delta_preprocessor,
    fit_ct_value_preprocessor,
    fit_static_preprocessor,
    load_dynamic_hlb_dataset,
    load_fixed_split_indices,
    split_tree_indices,
    transform_static_features,
    weighted_brier_score,
    weighted_c_index,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dynamic ReK-Surv on HLB dataset")
    parser.add_argument("--data_path", type=str, default="data/hlb_dataset.xlsx", help="Path to the HLB dynamic Excel file")
    parser.add_argument("--landmarks", type=int, nargs="+", default=[6, 9], help="Landmark periods to evaluate")
    parser.add_argument("--pred_horizons", type=int, nargs="+", default=[1, 3], help="Prediction horizons in periods")
    parser.add_argument(
        "--window_pairs",
        type=str,
        nargs="+",
        default=None,
        help="Explicit landmark:horizon pairs, e.g. 3:2 3:8 6:6",
    )
    parser.add_argument(
        "--early_stop_window_pairs",
        type=str,
        nargs="+",
        default=None,
        help="Optional landmark:horizon pairs used only for validation early stopping.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated train/val/test splits")
    parser.add_argument("--test_size", type=float, default=0.2, help="Tree-level test split fraction")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split fraction within train")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--fixed_split_json",
        type=str,
        default=None,
        help="Optional JSON file with fixed train/val/test tree assignments to keep repeat splits stable across label edits.",
    )
    parser.add_argument("--use_time_features", action="store_true", help="Append explicit time-position features for legacy mode")
    parser.add_argument("--use_ct_aux_task", action="store_true", help="Use Sheet4 CT values as an auxiliary supervision task only")
    parser.add_argument("--use_agro_features", action="store_true", help="Use agricultural multiscale environment features instead of the legacy raw daily features")
    parser.add_argument(
        "--env_feature_set",
        type=str,
        default="auto",
        choices=["auto", "raw9", "agro75", "basic10"],
        help="Daily environment feature construction scheme.",
    )
    parser.add_argument(
        "--build_period_env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to construct period-level environment tensors.",
    )
    parser.add_argument("--use_tree_id_spatial", action="store_true", help="Append parsed tree-row/tree-column coordinates as static features")
    parser.add_argument(
        "--period_feature_mode",
        type=str,
        default="full",
        choices=["full", "compact", "mean_only"],
        help="How to build period-level environment features when agro features are enabled",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="legacy",
        choices=["legacy", "legacy_period_mean", "period_ms", "period_ms_tree_query", "trigger_orchard", "trigger_orchard_v2", "trigger_orchard_v3"],
        help="Dynamic model architecture",
    )
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"], help="Sequence encoder type for the legacy model")
    parser.add_argument(
        "--env_aux_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "next_day"],
        help="Environment auxiliary task mode; defaults to next_day for legacy and none otherwise",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the log-likelihood loss")
    parser.add_argument("--beta", type=float, default=0.2, help="Weight for the ranking loss")
    parser.add_argument("--gamma", type=float, default=None, help="Legacy alias for --gamma_env")
    parser.add_argument("--gamma_env", type=float, default=0.05, help="Weight for the next-day environment auxiliary loss")
    parser.add_argument("--gamma_ct", type=float, default=0.10, help="Weight for the CT auxiliary loss")
    parser.add_argument("--ct_aux_loss", type=str, default="huber", choices=["huber", "mse"], help="Loss type for the CT auxiliary head")
    parser.add_argument("--lag_spread_weight", type=float, default=0.01, help="Weight for the period-attention spread regularizer")
    parser.add_argument("--tree_attention_dropout", type=float, default=0.10, help="Dropout inside the tree-specific period attention scorer")
    parser.add_argument("--v3_dropout", type=float, default=0.0, help="Dropout probability for trigger_orchard_v3 MLP blocks")
    parser.add_argument("--static_attention_dim", type=int, default=32, help="Hidden size for static-conditioned attention encoders")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="artifacts/dynamic_hlb", help="Directory for dynamic experiment outputs")
    parser.add_argument(
        "--train_landmark_subset",
        type=int,
        nargs="+",
        default=None,
        help="Optional restriction on which landmark prefixes are used for training/validation, e.g. 6 or 6 9.",
    )
    args = parser.parse_args()
    args.model_type = str(args.model_type).strip().lower()

    if args.gamma is not None:
        args.gamma_env = args.gamma
    if args.env_aux_mode == "auto":
        args.env_aux_mode = "next_day" if args.model_type == "legacy" else "none"
    if args.env_aux_mode == "none":
        args.gamma_env = 0.0
    if args.model_type != "legacy":
        args.use_time_features = False
    if not 0.0 <= args.v3_dropout < 1.0:
        raise ValueError("--v3_dropout must be in the range [0, 1).")
    configure_model_data_settings(args)
    return args


def configure_model_data_settings(args: argparse.Namespace) -> None:
    args.env_feature_set = str(getattr(args, "env_feature_set", "auto")).strip().lower()
    args.build_period_env = bool(getattr(args, "build_period_env", True))
    if args.model_type == "trigger_orchard_v3":
        args.env_feature_set = "basic10"
        args.build_period_env = False
        args.period_feature_mode = "none"


def build_window_specs(landmarks: Sequence[int], pred_horizons: Sequence[int], num_periods: int) -> List[Tuple[int, int]]:
    window_specs = []
    for landmark in landmarks:
        if landmark < 0 or landmark >= num_periods:
            raise ValueError(f"Invalid landmark period: {landmark}")
        for horizon in pred_horizons:
            if horizon <= 0:
                raise ValueError(f"Prediction horizon must be positive: {horizon}")
            if landmark + horizon > num_periods:
                raise ValueError(
                    f"Window {landmark}->{horizon} exceeds the {num_periods}-period study horizon."
                )
            window_specs.append((landmark, horizon))
    return window_specs


def build_explicit_window_specs(window_pairs: Sequence[str], num_periods: int) -> List[Tuple[int, int]]:
    """
    Parse explicit landmark:horizon window pairs.
    """
    window_specs: List[Tuple[int, int]] = []
    for raw_pair in window_pairs:
        if ":" not in raw_pair:
            raise ValueError(f"Invalid window pair '{raw_pair}'. Expected format landmark:horizon.")
        landmark_text, horizon_text = raw_pair.split(":", maxsplit=1)
        landmark = int(landmark_text)
        horizon = int(horizon_text)
        if landmark < 0 or landmark >= num_periods:
            raise ValueError(f"Invalid landmark period: {landmark}")
        if horizon <= 0:
            raise ValueError(f"Prediction horizon must be positive: {horizon}")
        if landmark + horizon > num_periods:
            raise ValueError(
                f"Window {landmark}->{horizon} exceeds the {num_periods}-period study horizon."
            )
        window_specs.append((landmark, horizon))
    return window_specs


def configure_ct_aux_target(args: argparse.Namespace, window_specs: Sequence[Tuple[int, int]]) -> None:
    model_type = str(args.model_type).strip().lower()
    if not args.use_ct_aux_task:
        args.ct_aux_target_mode = "disabled"
        args.ct_aux_output_dim = 1
        args.ct_delta_output_dim = 1
        args.ct_state_output_dim = 1
        args.ct_aux_window_specs = []
        return

    if model_type == "trigger_orchard_v3":
        max_endpoint = max((int(landmark) + int(horizon) for landmark, horizon in window_specs), default=1)
        args.ct_aux_target_mode = "window_prefix_vector_with_endpoint_lod"
        args.ct_delta_output_dim = max(max_endpoint - 1, 1)
        args.ct_state_output_dim = max(len(window_specs), 1)
        args.ct_aux_output_dim = int(args.ct_delta_output_dim)
        args.ct_aux_window_specs = [(int(landmark), int(horizon)) for landmark, horizon in window_specs]
        return

    if model_type == "trigger_orchard_v2":
        max_endpoint = max((int(landmark) + int(horizon) for landmark, horizon in window_specs), default=1)
        args.ct_aux_target_mode = "window_prefix_vector"
        args.ct_aux_output_dim = max(max_endpoint - 1, 1)
        args.ct_delta_output_dim = int(args.ct_aux_output_dim)
        args.ct_state_output_dim = 1
        args.ct_aux_window_specs = [(int(landmark), int(horizon)) for landmark, horizon in window_specs]
        return

    args.ct_aux_target_mode = "next_delta"
    args.ct_aux_output_dim = 1
    args.ct_delta_output_dim = 1
    args.ct_state_output_dim = 1
    args.ct_aux_window_specs = []


def ct_prefix_sample_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    target_mode = normalize_ct_target_mode(getattr(args, "ct_aux_target_mode", ""))
    if target_mode == "window_prefix_vector_with_endpoint_lod":
        return {
            "ct_delta_output_dim": int(args.ct_delta_output_dim),
            "ct_state_output_dim": int(args.ct_state_output_dim),
            "ct_aux_window_specs": args.ct_aux_window_specs,
            "ct_aux_target_mode": target_mode,
        }
    if target_mode not in {"window_endpoint_heads", "window_prefix_vector"}:
        return {}
    return {
        "ct_aux_output_dim": int(args.ct_aux_output_dim),
        "ct_aux_window_specs": args.ct_aux_window_specs,
        "ct_aux_target_mode": target_mode,
    }


def count_ct_aux_valid_targets(samples: Sequence[PrefixSample]) -> int:
    return int(sum(np.asarray(sample.ct_aux_mask, dtype=np.float32).sum() for sample in samples))


def count_ct_delta_supervision_targets(samples: Sequence[PrefixSample]) -> int:
    total = 0
    for sample in samples:
        mask = np.asarray(sample.ct_aux_mask, dtype=np.float32).reshape(-1) > 0.5
        if sample.ct_aux_window_mask is None:
            total += int(mask.sum())
            continue

        window_mask = np.asarray(sample.ct_aux_window_mask, dtype=np.float32) > 0.5
        if window_mask.ndim == mask.ndim:
            combined = window_mask & mask
        elif window_mask.ndim == mask.ndim + 1:
            combined = window_mask & mask.reshape((1,) + mask.shape)
        else:
            raise ValueError(
                "Unsupported CT auxiliary mask shapes in sample counting: "
                f"sample={mask.shape}, window={window_mask.shape}"
            )
        total += int(combined.sum())
    return total


def count_ct_endpoint_valid_targets(samples: Sequence[PrefixSample]) -> Dict[str, int]:
    return {
        "state": int(sum(np.asarray(sample.ct_state_mask, dtype=np.float32).sum() for sample in samples)),
        "value": int(sum(np.asarray(sample.ct_value_mask, dtype=np.float32).sum() for sample in samples)),
    }


def compute_balanced_accuracy(tp: int, tn: int, fp: int, fn: int) -> float | None:
    recalls = []
    positive_n = tp + fn
    negative_n = tn + fp
    if positive_n > 0:
        recalls.append(tp / positive_n)
    if negative_n > 0:
        recalls.append(tn / negative_n)
    if not recalls:
        return None
    return float(np.mean(recalls))


def normalize_ct_target_mode(target_mode: str) -> str:
    return str(target_mode).strip().lower()


def is_window_endpoint_ct_mode(target_mode: str) -> bool:
    return normalize_ct_target_mode(target_mode) == "window_endpoint_heads"


def is_window_delta_lod_ct_mode(target_mode: str) -> bool:
    return normalize_ct_target_mode(target_mode) == "window_prefix_vector_with_endpoint_lod"


def format_prefix_metrics_for_output(
    prefix_metrics: Dict[str, float | None],
    target_mode: str,
) -> Dict[str, float | None]:
    payload = {
        "total": prefix_metrics["total"],
        "ll": prefix_metrics["ll"],
        "rank": prefix_metrics["rank"],
        "env_aux": prefix_metrics["env_aux"],
        "ct_aux": prefix_metrics["ct_aux"],
        "spread": prefix_metrics["spread"],
    }
    if is_window_delta_lod_ct_mode(target_mode):
        payload.update(
            {
                "ct_delta_loss": prefix_metrics["ct_delta"],
                "ct_state_loss": prefix_metrics["ct_state"],
                "ct_delta_mae_exact": prefix_metrics["ct_delta_mae_exact"],
                "ct_delta_valid_n": prefix_metrics["ct_delta_valid_n"],
                "ct_state_bal_acc": prefix_metrics["ct_state_bal_acc"],
                "ct_state_valid_n": prefix_metrics["ct_state_valid_n"],
            }
        )
    elif is_window_endpoint_ct_mode(target_mode):
        payload.update(
            {
                "ct_state_loss": prefix_metrics["ct_state"],
                "ct_value_loss": prefix_metrics["ct_value"],
                "ct_state_bal_acc": prefix_metrics["ct_state_bal_acc"],
                "ct_state_valid_n": prefix_metrics["ct_state_valid_n"],
                "ct_value_mae_exact": prefix_metrics["ct_value_mae_exact"],
                "ct_value_valid_n": prefix_metrics["ct_value_valid_n"],
            }
        )
    else:
        payload.update(
            {
                "ct_aux_mae": prefix_metrics["ct_delta_mae_exact"],
                "ct_aux_valid_n": prefix_metrics["ct_delta_valid_n"],
            }
        )
    return payload


def build_weighted_sampler(samples: Sequence[PrefixSample]) -> WeightedRandomSampler:
    landmark_counts = Counter(sample.landmark_period for sample in samples)
    future_event_count = sum(sample.future_event for sample in samples)
    non_event_count = max(len(samples) - future_event_count, 1)

    positive_weight = len(samples) / (2.0 * max(future_event_count, 1))
    negative_weight = len(samples) / (2.0 * non_event_count)
    median_landmark_count = float(np.median(list(landmark_counts.values())))

    weights = []
    for sample in samples:
        class_weight = positive_weight if sample.future_event == 1 else negative_weight
        landmark_weight = median_landmark_count / landmark_counts[sample.landmark_period]
        weights.append(class_weight * landmark_weight)

    weight_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(samples), replacement=True)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def instantiate_model(args: argparse.Namespace, tree_data, static_x: np.ndarray) -> DynamicReKSurv:
    return DynamicReKSurv(
        env_dim=len(tree_data.env_feature_names),
        static_dim=static_x.shape[1],
        day_to_period=tree_data.day_to_period,
        num_periods=tree_data.num_periods,
        use_time_features=args.use_time_features,
        rnn_type=args.rnn_type,
        model_type=args.model_type,
        period_env_dim=len(tree_data.period_feature_names),
        env_feature_names=tree_data.env_feature_names,
        period_start_day_index=tree_data.period_start_day_index,
        period_end_day_index=tree_data.period_end_day_index,
        env_aux_mode=args.env_aux_mode,
        tree_attention_dropout=args.tree_attention_dropout,
        v3_dropout=args.v3_dropout,
        static_attention_dim=args.static_attention_dim,
        ct_delta_output_dim=getattr(args, "ct_delta_output_dim", 1),
        ct_aux_output_dim=getattr(args, "ct_aux_output_dim", 1),
        ct_state_output_dim=getattr(args, "ct_state_output_dim", 1),
        ct_aux_target_mode=getattr(args, "ct_aux_target_mode", ""),
    )


def forward_model(model: DynamicReKSurv, batch: Dict[str, torch.Tensor]) -> dict:
    sequence_env = batch["env"]
    sequence_len_days = batch["seq_len_days"]
    if getattr(model, "model_type", "") == "legacy_period_mean":
        sequence_env = batch["period_env"]
        sequence_len_days = batch["seq_len_periods"]
    return model(
        daily_env_prefix=sequence_env,
        period_env_prefix=batch["period_env"],
        static_x=batch["static_x"],
        seq_len_days=sequence_len_days,
        seq_len_periods=batch["seq_len_periods"],
        landmark_period=batch["landmark_period"],
    )


def train_one_epoch(
    model: DynamicReKSurv,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float,
    beta: float,
    gamma_env: float,
    gamma_ct: float,
    use_ct_aux_task: bool,
    ct_target_mean: float,
    ct_target_std: float,
    ct_aux_loss: str,
    env_aux_mode: str,
    lag_spread_weight: float,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0
    running_ll = 0.0
    running_rank = 0.0
    running_env_aux = 0.0
    running_ct_aux = 0.0
    running_ct_state = 0.0
    running_ct_delta = 0.0
    running_ct_value = 0.0
    running_spread = 0.0
    num_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        model_output = forward_model(model, batch)
        losses = compute_dynamic_loss(
            model_output=model_output,
            batch=batch,
            alpha=alpha,
            beta=beta,
            gamma_env=gamma_env,
            gamma_ct=gamma_ct,
            use_ct_aux_task=use_ct_aux_task,
            ct_target_mean=ct_target_mean,
            ct_target_std=ct_target_std,
            ct_aux_loss=ct_aux_loss,
            env_aux_mode=env_aux_mode,
            lag_spread_weight=lag_spread_weight,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = int(batch["env"].size(0))
        num_samples += batch_size
        running_total += float(losses["total"].item()) * batch_size
        running_ll += float(losses["ll"].item()) * batch_size
        running_rank += float(losses["rank"].item()) * batch_size
        running_env_aux += float(losses["env_aux"].item()) * batch_size
        running_ct_aux += float(losses["ct_aux"].item()) * batch_size
        running_ct_state += float(losses["ct_state"].item()) * batch_size
        running_ct_delta += float(losses["ct_delta"].item()) * batch_size
        running_ct_value += float(losses["ct_value"].item()) * batch_size
        running_spread += float(losses["spread"].item()) * batch_size

    return {
        "total": running_total / max(num_samples, 1),
        "ll": running_ll / max(num_samples, 1),
        "rank": running_rank / max(num_samples, 1),
        "env_aux": running_env_aux / max(num_samples, 1),
        "ct_aux": running_ct_aux / max(num_samples, 1),
        "ct_state": running_ct_state / max(num_samples, 1),
        "ct_delta": running_ct_delta / max(num_samples, 1),
        "ct_value": running_ct_value / max(num_samples, 1),
        "spread": running_spread / max(num_samples, 1),
    }


@torch.no_grad()
def evaluate_prefix_losses(
    model: DynamicReKSurv,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    beta: float,
    gamma_env: float,
    gamma_ct: float,
    use_ct_aux_task: bool,
    ct_target_mean: float,
    ct_target_std: float,
    ct_aux_loss: str,
    env_aux_mode: str,
    lag_spread_weight: float,
) -> Dict[str, float | None]:
    model.eval()
    running_total = 0.0
    running_ll = 0.0
    running_rank = 0.0
    running_env_aux = 0.0
    running_ct_aux = 0.0
    running_ct_state = 0.0
    running_ct_delta = 0.0
    running_ct_value = 0.0
    running_spread = 0.0
    running_ct_delta_abs = 0.0
    running_ct_delta_count = 0
    running_ct_value_abs = 0.0
    running_ct_value_count = 0
    state_tp = 0
    state_tn = 0
    state_fp = 0
    state_fn = 0
    num_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        model_output = forward_model(model, batch)
        losses = compute_dynamic_loss(
            model_output=model_output,
            batch=batch,
            alpha=alpha,
            beta=beta,
            gamma_env=gamma_env,
            gamma_ct=gamma_ct,
            use_ct_aux_task=use_ct_aux_task,
            ct_target_mean=ct_target_mean,
            ct_target_std=ct_target_std,
            ct_aux_loss=ct_aux_loss,
            env_aux_mode=env_aux_mode,
            lag_spread_weight=lag_spread_weight,
        )

        batch_size = int(batch["env"].size(0))
        num_samples += batch_size
        running_total += float(losses["total"].item()) * batch_size
        running_ll += float(losses["ll"].item()) * batch_size
        running_rank += float(losses["rank"].item()) * batch_size
        running_env_aux += float(losses["env_aux"].item()) * batch_size
        running_ct_aux += float(losses["ct_aux"].item()) * batch_size
        running_ct_state += float(losses["ct_state"].item()) * batch_size
        running_ct_delta += float(losses["ct_delta"].item()) * batch_size
        running_ct_value += float(losses["ct_value"].item()) * batch_size
        running_spread += float(losses["spread"].item()) * batch_size

        if use_ct_aux_task:
            if "pred_ct_state_logits" in model_output and "pred_ct_delta" in model_output:
                confusion = ct_state_confusion_counts(
                    pred_ct_state_logits=model_output["pred_ct_state_logits"],
                    ct_state_target=batch["ct_state_target"],
                    ct_state_mask=batch["ct_state_mask"],
                )
                state_tp += confusion["tp"]
                state_tn += confusion["tn"]
                state_fp += confusion["fp"]
                state_fn += confusion["fn"]
                batch_ct_delta_mae, batch_ct_delta_count = ct_delta_mean_absolute_error(
                    pred_ct_delta=model_output["pred_ct_delta"],
                    ct_delta_target=batch["ct_delta_target"],
                    ct_aux_mask=batch["ct_aux_mask"],
                    target_mean=ct_target_mean,
                    target_std=ct_target_std,
                    ct_aux_window_mask=batch.get("ct_aux_window_mask"),
                )
                running_ct_delta_abs += float(batch_ct_delta_mae.item()) * batch_ct_delta_count
                running_ct_delta_count += batch_ct_delta_count
            elif "pred_ct_state_logits" in model_output and "pred_ct_value" in model_output:
                confusion = ct_state_confusion_counts(
                    pred_ct_state_logits=model_output["pred_ct_state_logits"],
                    ct_state_target=batch["ct_state_target"],
                    ct_state_mask=batch["ct_state_mask"],
                )
                state_tp += confusion["tp"]
                state_tn += confusion["tn"]
                state_fp += confusion["fp"]
                state_fn += confusion["fn"]
                batch_ct_value_mae, batch_ct_value_count = ct_value_mean_absolute_error(
                    pred_ct_value=model_output["pred_ct_value"],
                    ct_value_target=batch["ct_value_target"],
                    ct_value_mask=batch["ct_value_mask"],
                    target_mean=ct_target_mean,
                    target_std=ct_target_std,
                )
                running_ct_value_abs += float(batch_ct_value_mae.item()) * batch_ct_value_count
                running_ct_value_count += batch_ct_value_count
            else:
                batch_ct_mae, batch_ct_count = ct_delta_mean_absolute_error(
                    pred_ct_delta=model_output["pred_ct_delta"],
                    ct_delta_target=batch["ct_delta_target"],
                    ct_aux_mask=batch["ct_aux_mask"],
                    target_mean=ct_target_mean,
                    target_std=ct_target_std,
                    ct_aux_window_mask=batch.get("ct_aux_window_mask"),
                )
                running_ct_delta_abs += float(batch_ct_mae.item()) * batch_ct_count
                running_ct_delta_count += batch_ct_count

    return {
        "total": running_total / max(num_samples, 1),
        "ll": running_ll / max(num_samples, 1),
        "rank": running_rank / max(num_samples, 1),
        "env_aux": running_env_aux / max(num_samples, 1),
        "ct_aux": running_ct_aux / max(num_samples, 1),
        "ct_state": running_ct_state / max(num_samples, 1),
        "ct_delta": running_ct_delta / max(num_samples, 1),
        "ct_value": running_ct_value / max(num_samples, 1),
        "spread": running_spread / max(num_samples, 1),
        "ct_state_bal_acc": compute_balanced_accuracy(state_tp, state_tn, state_fp, state_fn),
        "ct_state_valid_n": int(state_tp + state_tn + state_fp + state_fn),
        "ct_delta_mae_exact": running_ct_delta_abs / running_ct_delta_count if running_ct_delta_count > 0 else None,
        "ct_delta_valid_n": int(running_ct_delta_count),
        "ct_value_mae_exact": running_ct_value_abs / running_ct_value_count if running_ct_value_count > 0 else None,
        "ct_value_valid_n": int(running_ct_value_count),
    }


@torch.no_grad()
def evaluate_windows(
    model: DynamicReKSurv,
    tree_data,
    static_x: np.ndarray,
    evaluation_indices: np.ndarray,
    train_indices: np.ndarray,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
) -> tuple[List[Dict[str, float]], float]:
    model.eval()

    train_times = tree_data.time_period[train_indices]
    train_events = tree_data.event_flag[train_indices]
    records = []

    for landmark, horizon in window_specs:
        eligible_mask = tree_data.time_period[evaluation_indices] > landmark
        eligible_indices = evaluation_indices[eligible_mask]
        future_event_mask = (
            (tree_data.event_flag[eligible_indices] == 1)
            & (tree_data.time_period[eligible_indices] <= landmark + horizon)
        )

        record = {
            "landmark": int(landmark),
            "pred_horizon": int(horizon),
            "eligible_n": int(len(eligible_indices)),
            "future_event_n": int(future_event_mask.sum()),
            "ctd": float("nan"),
            "bstd": float("nan"),
        }

        if len(eligible_indices) == 0:
            records.append(record)
            continue

        period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
        if getattr(model, "model_type", "") == "legacy_period_mean":
            env = period_env
        else:
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
        static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
        seq_len_periods = torch.full(
            (len(eligible_indices),),
            int(tree_data.landmark_seq_len_periods[landmark]),
            dtype=torch.long,
            device=device,
        )
        if getattr(model, "model_type", "") == "legacy_period_mean":
            seq_len_days = seq_len_periods
        else:
            seq_len_days = torch.full(
                (len(eligible_indices),),
                int(tree_data.landmark_seq_len_days[landmark]),
                dtype=torch.long,
                device=device,
            )
        landmark_tensor = torch.full(
            (len(eligible_indices),),
            int(landmark),
            dtype=torch.long,
            device=device,
        )
        model_output = model(
            daily_env_prefix=env,
            period_env_prefix=period_env,
            static_x=static_tensor,
            seq_len_days=seq_len_days,
            seq_len_periods=seq_len_periods,
            landmark_period=landmark_tensor,
        )
        risk = model_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1).cpu().numpy()
        eval_times = tree_data.time_period[eligible_indices]
        eval_events = tree_data.event_flag[eligible_indices]
        absolute_horizon = landmark + horizon

        record["ctd"] = weighted_c_index(
            train_times=train_times,
            train_events=train_events,
            prediction=risk,
            test_times=eval_times,
            test_events=eval_events,
            horizon=absolute_horizon,
        )
        record["bstd"] = weighted_brier_score(
            train_times=train_times,
            train_events=train_events,
            prediction=risk,
            test_times=eval_times,
            test_events=eval_events,
            horizon=absolute_horizon,
        )
        records.append(record)

    valid_ctd = [record["ctd"] for record in records if np.isfinite(record["ctd"])]
    mean_ctd = float(np.mean(valid_ctd)) if len(valid_ctd) > 0 else float("-inf")
    return records, mean_ctd


def mean_window_metric(records: Sequence[Dict[str, float]], metric_key: str) -> float | None:
    valid_values = [float(record[metric_key]) for record in records if np.isfinite(record[metric_key])]
    if len(valid_values) == 0:
        return None
    return float(np.mean(valid_values))


def summarize_window_group(group: pd.DataFrame) -> Dict[str, float]:
    return {
        "eligible_n_mean": float(group["eligible_n"].mean()),
        "eligible_n_std": float(group["eligible_n"].std(ddof=0)),
        "future_event_n_mean": float(group["future_event_n"].mean()),
        "future_event_n_std": float(group["future_event_n"].std(ddof=0)),
        "ctd_mean": float(group["ctd"].mean(skipna=True)),
        "ctd_std": float(group["ctd"].std(skipna=True, ddof=0)),
        "bstd_mean": float(group["bstd"].mean(skipna=True)),
        "bstd_std": float(group["bstd"].std(skipna=True, ddof=0)),
    }


def run_repeat(
    repeat_index: int,
    args: argparse.Namespace,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    early_stop_window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_root: Path,
) -> List[Dict[str, float]]:
    repeat_seed = args.seed + repeat_index * 1000
    set_seed(repeat_seed)

    if args.fixed_split_json:
        train_indices, val_indices, test_indices = load_fixed_split_indices(
            tree_data=tree_data,
            split_json_path=args.fixed_split_json,
            repeat_index=repeat_index,
        )
    else:
        train_indices, val_indices, test_indices = split_tree_indices(
            tree_data=tree_data,
            test_size=args.test_size,
            val_ratio=args.val_ratio,
            random_state=repeat_seed,
        )
    preprocessor = fit_static_preprocessor(
        tree_data,
        train_indices,
        use_tree_id_spatial=args.use_tree_id_spatial,
    )
    static_x, static_feature_names = transform_static_features(tree_data, preprocessor)

    include_landmark_zero = args.model_type != "legacy_period_mean"
    ct_sample_kwargs = ct_prefix_sample_kwargs(args)
    train_samples = build_prefix_samples(
        tree_data,
        train_indices,
        include_landmark_zero=include_landmark_zero,
        **ct_sample_kwargs,
    )
    val_samples = build_prefix_samples(
        tree_data,
        val_indices,
        include_landmark_zero=include_landmark_zero,
        **ct_sample_kwargs,
    )
    if args.train_landmark_subset is not None:
        train_samples = build_prefix_samples(
            tree_data,
            train_indices,
            include_landmark_zero=include_landmark_zero,
            allowed_landmarks=args.train_landmark_subset,
            **ct_sample_kwargs,
        )
        val_samples = build_prefix_samples(
            tree_data,
            val_indices,
            include_landmark_zero=include_landmark_zero,
            allowed_landmarks=args.train_landmark_subset,
            **ct_sample_kwargs,
        )

    ct_target_stats = {"mean": 0.0, "std": 1.0, "count": 0}
    if args.use_ct_aux_task:
        if is_window_endpoint_ct_mode(args.ct_aux_target_mode):
            ct_target_stats = fit_ct_value_preprocessor(train_samples)
        else:
            ct_target_stats = fit_ct_delta_preprocessor(train_samples)

    train_dataset = PrefixSampleDataset(tree_data, static_x, train_samples)
    val_dataset = PrefixSampleDataset(tree_data, static_x, val_samples)
    train_sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = instantiate_model(args, tree_data, static_x).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(args.patience // 4, 2),
        min_lr=1e-5,
    )

    best_state = None
    best_epoch = 0
    best_val_ctd = float("-inf")
    best_val_prefix_metrics: Dict[str, float | None] = {}
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            gamma_env=args.gamma_env,
            gamma_ct=args.gamma_ct,
            use_ct_aux_task=args.use_ct_aux_task,
            ct_target_mean=float(ct_target_stats["mean"]),
            ct_target_std=float(ct_target_stats["std"]),
            ct_aux_loss=args.ct_aux_loss,
            env_aux_mode=args.env_aux_mode,
            lag_spread_weight=args.lag_spread_weight,
        )
        val_prefix_metrics = evaluate_prefix_losses(
            model=model,
            loader=val_loader,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            gamma_env=args.gamma_env,
            gamma_ct=args.gamma_ct,
            use_ct_aux_task=args.use_ct_aux_task,
            ct_target_mean=float(ct_target_stats["mean"]),
            ct_target_std=float(ct_target_stats["std"]),
            ct_aux_loss=args.ct_aux_loss,
            env_aux_mode=args.env_aux_mode,
            lag_spread_weight=args.lag_spread_weight,
        )
        val_records, val_ctd = evaluate_windows(
            model=model,
            tree_data=tree_data,
            static_x=static_x,
            evaluation_indices=val_indices,
            train_indices=train_indices,
            window_specs=window_specs,
            device=device,
        )
        if list(early_stop_window_specs) != list(window_specs):
            _, val_ctd = evaluate_windows(
                model=model,
                tree_data=tree_data,
                static_x=static_x,
                evaluation_indices=val_indices,
                train_indices=train_indices,
                window_specs=early_stop_window_specs,
                device=device,
            )
        scheduler.step(val_ctd if np.isfinite(val_ctd) else -1.0)
        current_lr = optimizer.param_groups[0]["lr"]

        history_record = {
            "epoch": epoch,
            "train_total_loss": train_loss["total"],
            "train_ll_loss": train_loss["ll"],
            "train_rank_loss": train_loss["rank"],
            "train_env_aux_loss": train_loss["env_aux"],
            "train_ct_aux_loss": train_loss["ct_aux"],
            "train_spread_loss": train_loss["spread"],
            "val_total_loss": val_prefix_metrics["total"],
            "val_ll_loss": val_prefix_metrics["ll"],
            "val_rank_loss": val_prefix_metrics["rank"],
            "val_env_aux_loss": val_prefix_metrics["env_aux"],
            "val_ct_aux_loss": val_prefix_metrics["ct_aux"],
            "val_spread_loss": val_prefix_metrics["spread"],
            "val_mean_ctd": val_ctd,
            "lr": current_lr,
        }
        if is_window_delta_lod_ct_mode(args.ct_aux_target_mode):
            history_record.update(
                {
                    "train_ct_delta_loss": train_loss["ct_delta"],
                    "train_ct_state_loss": train_loss["ct_state"],
                    "val_ct_delta_loss": val_prefix_metrics["ct_delta"],
                    "val_ct_state_loss": val_prefix_metrics["ct_state"],
                    "val_ct_delta_mae_exact": val_prefix_metrics["ct_delta_mae_exact"],
                    "val_ct_delta_valid_n": val_prefix_metrics["ct_delta_valid_n"],
                    "val_ct_state_bal_acc": val_prefix_metrics["ct_state_bal_acc"],
                    "val_ct_state_valid_n": val_prefix_metrics["ct_state_valid_n"],
                }
            )
        elif is_window_endpoint_ct_mode(args.ct_aux_target_mode):
            history_record.update(
                {
                    "train_ct_state_loss": train_loss["ct_state"],
                    "train_ct_value_loss": train_loss["ct_value"],
                    "val_ct_state_loss": val_prefix_metrics["ct_state"],
                    "val_ct_value_loss": val_prefix_metrics["ct_value"],
                    "val_ct_state_bal_acc": val_prefix_metrics["ct_state_bal_acc"],
                    "val_ct_state_valid_n": val_prefix_metrics["ct_state_valid_n"],
                    "val_ct_value_mae_exact": val_prefix_metrics["ct_value_mae_exact"],
                    "val_ct_value_valid_n": val_prefix_metrics["ct_value_valid_n"],
                }
            )
        else:
            history_record.update(
                {
                    "val_ct_aux_mae": val_prefix_metrics["ct_delta_mae_exact"],
                    "val_ct_aux_valid_n": val_prefix_metrics["ct_delta_valid_n"],
                }
            )
        history.append(history_record)

        should_update_best = best_state is None
        if np.isfinite(val_ctd) and val_ctd > best_val_ctd:
            should_update_best = True

        if should_update_best:
            best_val_ctd = val_ctd
            best_epoch = epoch
            best_val_prefix_metrics = dict(val_prefix_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or no_improve == 0:
            if is_window_delta_lod_ct_mode(args.ct_aux_target_mode):
                print(
                    f"Repeat {repeat_index} | Epoch {epoch:03d} | "
                    f"Loss {train_loss['total']:.4f} | "
                    f"Val mean Ctd {val_ctd:.4f} | "
                    f"Val CT delta MAE {format_optional_metric(val_prefix_metrics['ct_delta_mae_exact'])} | "
                    f"Val CT bal acc {format_optional_metric(val_prefix_metrics['ct_state_bal_acc'])} | "
                    f"LR {current_lr:.6f}"
                )
            elif is_window_endpoint_ct_mode(args.ct_aux_target_mode):
                print(
                    f"Repeat {repeat_index} | Epoch {epoch:03d} | "
                    f"Loss {train_loss['total']:.4f} | "
                    f"Val mean Ctd {val_ctd:.4f} | "
                    f"Val CT bal acc {format_optional_metric(val_prefix_metrics['ct_state_bal_acc'])} | "
                    f"Val CT exact MAE {format_optional_metric(val_prefix_metrics['ct_value_mae_exact'])} | "
                    f"LR {current_lr:.6f}"
                )
            else:
                print(
                    f"Repeat {repeat_index} | Epoch {epoch:03d} | "
                    f"Loss {train_loss['total']:.4f} | "
                    f"Val mean Ctd {val_ctd:.4f} | "
                    f"Val CT MAE {format_optional_metric(val_prefix_metrics['ct_delta_mae_exact'])} | "
                    f"LR {current_lr:.6f}"
                )

        if no_improve >= args.patience:
            print(f"Repeat {repeat_index} early stopped at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    train_records, _ = evaluate_windows(
        model=model,
        tree_data=tree_data,
        static_x=static_x,
        evaluation_indices=train_indices,
        train_indices=train_indices,
        window_specs=window_specs,
        device=device,
    )
    val_records, _ = evaluate_windows(
        model=model,
        tree_data=tree_data,
        static_x=static_x,
        evaluation_indices=val_indices,
        train_indices=train_indices,
        window_specs=window_specs,
        device=device,
    )
    test_records, _ = evaluate_windows(
        model=model,
        tree_data=tree_data,
        static_x=static_x,
        evaluation_indices=test_indices,
        train_indices=train_indices,
        window_specs=window_specs,
        device=device,
    )
    split_window_metrics = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    split_mean_metrics = {
        split_name: {
            "ctd_mean": mean_window_metric(records, "ctd"),
            "bstd_mean": mean_window_metric(records, "bstd"),
        }
        for split_name, records in split_window_metrics.items()
    }

    repeat_dir = output_root / f"repeat_{repeat_index}"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = repeat_dir / "best_model.pth"
    torch.save(best_state, checkpoint_path)

    if is_window_delta_lod_ct_mode(args.ct_aux_target_mode):
        train_ct_target_counts = {
            "state": int(sum(np.asarray(sample.ct_state_mask, dtype=np.float32).sum() for sample in train_samples)),
            "delta": count_ct_delta_supervision_targets(train_samples),
        }
        val_ct_target_counts = {
            "state": int(sum(np.asarray(sample.ct_state_mask, dtype=np.float32).sum() for sample in val_samples)),
            "delta": count_ct_delta_supervision_targets(val_samples),
        }
    elif is_window_endpoint_ct_mode(args.ct_aux_target_mode):
        train_ct_target_counts = count_ct_endpoint_valid_targets(train_samples)
        val_ct_target_counts = count_ct_endpoint_valid_targets(val_samples)
    else:
        legacy_train_ct_valid_n = count_ct_aux_valid_targets(train_samples)
        legacy_val_ct_valid_n = count_ct_aux_valid_targets(val_samples)
        train_ct_target_counts = {"state": 0, "value": legacy_train_ct_valid_n}
        val_ct_target_counts = {"state": 0, "value": legacy_val_ct_valid_n}
    best_val_prefix_metrics_payload = format_prefix_metrics_for_output(
        best_val_prefix_metrics,
        args.ct_aux_target_mode,
    )

    metrics_payload = {
        "repeat": repeat_index,
        "seed": repeat_seed,
        "train_trees": int(len(train_indices)),
        "val_trees": int(len(val_indices)),
        "test_trees": int(len(test_indices)),
        "train_tree_ids": [str(tree_data.tree_ids[index]) for index in train_indices],
        "val_tree_ids": [str(tree_data.tree_ids[index]) for index in val_indices],
        "test_tree_ids": [str(tree_data.tree_ids[index]) for index in test_indices],
        "train_prefix_samples": int(len(train_samples)),
        "val_prefix_samples": int(len(val_samples)),
        "best_epoch": int(best_epoch),
        "best_val_mean_ctd": float(best_val_ctd),
        "model_type": args.model_type,
        "use_agro_features": bool(args.use_agro_features),
        "env_feature_set": args.env_feature_set,
        "build_period_env": bool(args.build_period_env),
        "use_tree_id_spatial": bool(args.use_tree_id_spatial),
        "ct_aux_enabled": bool(args.use_ct_aux_task),
        "ct_aux_target_mode": getattr(
            args,
            "ct_aux_target_mode",
            "disabled" if not args.use_ct_aux_task else "next_delta",
        ),
        "ct_aux_window_specs": [list(spec) for spec in getattr(args, "ct_aux_window_specs", [])],
        "ct_aux_loss": args.ct_aux_loss,
        "env_aux_mode": args.env_aux_mode,
        "lag_spread_weight": float(args.lag_spread_weight),
        "tree_attention_dropout": float(args.tree_attention_dropout),
        "v3_dropout": float(args.v3_dropout),
        "static_attention_dim": int(args.static_attention_dim),
        "train_landmark_subset": None if args.train_landmark_subset is None else [int(value) for value in args.train_landmark_subset],
        "early_stop_window_specs": [list(spec) for spec in early_stop_window_specs],
        "env_feature_dim": int(len(tree_data.env_feature_names)),
        "period_feature_dim": int(len(tree_data.period_feature_names)),
        "env_feature_names": tree_data.env_feature_names,
        "period_feature_names": tree_data.period_feature_names,
        "static_feature_dim": int(static_x.shape[1]),
        "static_feature_names": static_feature_names,
        "history": history,
        "train_windows": train_records,
        "validation_windows": val_records,
        "best_validation_prefix_metrics": best_val_prefix_metrics_payload,
        "test_windows": test_records,
        "split_window_metrics": split_window_metrics,
        "split_mean_metrics": split_mean_metrics,
    }
    if is_window_delta_lod_ct_mode(args.ct_aux_target_mode):
        metrics_payload.update(
            {
                "train_ct_state_valid_n": int(train_ct_target_counts["state"]),
                "val_ct_state_valid_n": int(val_ct_target_counts["state"]),
                "train_ct_delta_valid_n": int(train_ct_target_counts["delta"]),
                "val_ct_delta_valid_n": int(val_ct_target_counts["delta"]),
                "ct_delta_output_dim": int(getattr(args, "ct_delta_output_dim", 1)),
                "ct_state_output_dim": int(getattr(args, "ct_state_output_dim", 1)),
                "ct_target_mean": float(ct_target_stats["mean"]),
                "ct_target_std": float(ct_target_stats["std"]),
                "ct_target_count": int(ct_target_stats["count"]),
            }
        )
    elif is_window_endpoint_ct_mode(args.ct_aux_target_mode):
        metrics_payload.update(
            {
                "train_ct_state_valid_n": int(train_ct_target_counts["state"]),
                "val_ct_state_valid_n": int(val_ct_target_counts["state"]),
                "train_ct_value_valid_n": int(train_ct_target_counts["value"]),
                "val_ct_value_valid_n": int(val_ct_target_counts["value"]),
                "ct_aux_output_dim": int(getattr(args, "ct_aux_output_dim", 1)),
                "ct_value_target_mean": float(ct_target_stats["mean"]),
                "ct_value_target_std": float(ct_target_stats["std"]),
                "ct_value_target_count": int(ct_target_stats["count"]),
            }
        )
    else:
        metrics_payload.update(
            {
                "train_ct_aux_valid_n": int(train_ct_target_counts["value"]),
                "val_ct_aux_valid_n": int(val_ct_target_counts["value"]),
                "ct_delta_output_dim": int(getattr(args, "ct_delta_output_dim", 1)),
                "ct_target_mean": float(ct_target_stats["mean"]),
                "ct_target_std": float(ct_target_stats["std"]),
                "ct_target_count": int(ct_target_stats["count"]),
            }
        )
    with open(repeat_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(to_serializable(metrics_payload), file, ensure_ascii=False, indent=2)

    flat_records = []
    for split_name, split_records in split_window_metrics.items():
        for record in split_records:
            flat_records.append(
                {
                    "repeat": repeat_index,
                    "seed": repeat_seed,
                    "best_epoch": best_epoch,
                    "split": split_name,
                    **record,
                }
            )
    return flat_records


def summarize_results(
    all_records: List[Dict[str, float]],
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    summary_df = pd.DataFrame(all_records)
    summary_df = summary_df.sort_values(["split", "landmark", "pred_horizon", "repeat"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "summary.csv", index=False)

    split_summaries: Dict[str, Dict[str, object]] = {}
    for split_name, split_group in summary_df.groupby("split", sort=False):
        windows = []
        for (landmark, pred_horizon), group in split_group.groupby(["landmark", "pred_horizon"], sort=True):
            window_summary = summarize_window_group(group)
            window_summary.update(
                {
                    "landmark": int(landmark),
                    "pred_horizon": int(pred_horizon),
                }
            )
            windows.append(window_summary)
        split_summaries[str(split_name)] = {
            "overall": {
                "ctd_mean": float(split_group["ctd"].mean(skipna=True)),
                "ctd_std": float(split_group["ctd"].std(skipna=True, ddof=0)),
                "bstd_mean": float(split_group["bstd"].mean(skipna=True)),
                "bstd_std": float(split_group["bstd"].std(skipna=True, ddof=0)),
            },
            "windows": windows,
        }

    summary_payload = {
        "config": vars(args),
        "overall": {split_name: split_summary["overall"] for split_name, split_summary in split_summaries.items()},
        "splits": split_summaries,
        "windows": split_summaries.get("test", {}).get("windows", []),
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as file:
        json.dump(to_serializable(summary_payload), file, ensure_ascii=False, indent=2)


def format_optional_metric(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def to_serializable(value):
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().tolist())
    return value


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tree_data = load_dynamic_hlb_dataset(
        args.data_path,
        use_ct_aux_task=args.use_ct_aux_task,
        use_agro_features=args.use_agro_features,
        env_feature_set=args.env_feature_set,
        build_period_env=args.build_period_env,
        period_feature_mode=args.period_feature_mode,
    )
    if args.window_pairs:
        window_specs = build_explicit_window_specs(args.window_pairs, tree_data.num_periods)
    else:
        window_specs = build_window_specs(args.landmarks, args.pred_horizons, tree_data.num_periods)
    if args.early_stop_window_pairs:
        early_stop_window_specs = build_explicit_window_specs(args.early_stop_window_pairs, tree_data.num_periods)
    else:
        early_stop_window_specs = window_specs
    configure_ct_aux_target(args, window_specs)

    print(f"Loaded {len(tree_data.tree_ids)} trees")
    print(f"Event trees: {int(tree_data.event_flag.sum())} | Censored trees: {int((1 - tree_data.event_flag).sum())}")
    print(f"Periods: {tree_data.num_periods} | Max days: {tree_data.max_days}")
    print(f"Daily env dim: {len(tree_data.env_feature_names)} | Period env dim: {len(tree_data.period_feature_names)}")
    print(f"Windows: {window_specs}")
    print(f"Early-stop windows: {early_stop_window_specs}")
    print(
        f"Model type: {args.model_type} | Agro features: {args.use_agro_features} "
        f"| Env feature set: {args.env_feature_set} | Build period env: {args.build_period_env} "
        f"| Period features: {args.period_feature_mode} | Env aux: {args.env_aux_mode} "
        f"| Tree spatial: {args.use_tree_id_spatial}"
    )
    if args.model_type == "trigger_orchard_v3":
        print(f"trigger_orchard_v3 dropout: {args.v3_dropout:.3f}")
    if is_window_delta_lod_ct_mode(args.ct_aux_target_mode):
        print(
            f"CT auxiliary task: {'on' if args.use_ct_aux_task else 'off'} | "
            f"CT target mode: {args.ct_aux_target_mode} | "
            f"CT delta dim: {args.ct_delta_output_dim} | CT state dim: {args.ct_state_output_dim}"
        )
    else:
        print(
            f"CT auxiliary task: {'on' if args.use_ct_aux_task else 'off'} | "
            f"CT target mode: {args.ct_aux_target_mode} | CT output dim: {args.ct_aux_output_dim}"
        )

    start_time = time.perf_counter()
    all_records: List[Dict[str, float]] = []
    for repeat_index in range(args.repeats):
        print(f"\n===== Repeat {repeat_index} / {args.repeats - 1} =====")
        repeat_records = run_repeat(
            repeat_index=repeat_index,
            args=args,
            tree_data=tree_data,
            window_specs=window_specs,
            early_stop_window_specs=early_stop_window_specs,
            device=device,
            output_root=output_root,
        )
        all_records.extend(repeat_records)

    summarize_results(all_records, args, output_root)
    elapsed = time.perf_counter() - start_time
    print(f"\nFinished dynamic training in {elapsed:.2f} seconds")
    print(f"Results saved to {output_root}")


if __name__ == "__main__":
    main()
