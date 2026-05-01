"""
Environment time-focus analysis for the dynamic HLB survival model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    fit_ct_delta_preprocessor,
    fit_static_preprocessor,
    load_dynamic_hlb_dataset,
    split_tree_indices,
    transform_static_features,
    weighted_brier_score,
    weighted_c_index,
)


LAG_BINS = [
    ("last_2_days", 0, 1),
    ("days_3_7", 2, 6),
    ("days_8_14", 7, 13),
    ("days_15_30", 14, 29),
    ("days_31_60", 30, 59),
    ("days_60_plus", 60, None),
]
RECENT_LAG_BINS = {"last_2_days", "days_3_7"}
PERIOD_LAG_BINS = [
    ("last_1_period", 0, 0),
    ("periods_2_3", 1, 2),
    ("periods_4_6", 3, 5),
    ("periods_7_plus", 6, None),
]
PERIOD_RECENT_LAG_BINS = {"last_1_period", "periods_2_3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether the dynamic HLB model over-relies on CT test-day-adjacent environment."
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts/dynamic_hlb_ctaux_full5",
        help="Directory containing the trained CT+environment auxiliary dynamic model repeats.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/hlb_dataset.xlsx",
        help="Path to the HLB Excel dataset.",
    )
    parser.add_argument(
        "--window_pairs",
        type=str,
        nargs="+",
        default=["6:1", "6:3", "9:1", "9:3"],
        help="Landmark:horizon window pairs to analyze.",
    )
    parser.add_argument(
        "--analysis_unit",
        type=str,
        default="auto",
        choices=["auto", "day", "period"],
        help="Use day-level or period-level perturbation analysis. 'auto' picks period for period_ms and day otherwise.",
    )
    parser.add_argument(
        "--history_days",
        type=int,
        nargs="+",
        default=[2, 7, 14, 30],
        help="History block lengths to test for occlusion and retraining.",
    )
    parser.add_argument(
        "--history_periods",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="History block lengths in periods for period-level analysis.",
    )
    parser.add_argument(
        "--perm_repeats",
        type=int,
        default=10,
        help="Number of random within-block permutations per position.",
    )
    parser.add_argument(
        "--retrain_repeats",
        type=int,
        default=5,
        help="Number of artifact repeats to reuse for split reconstruction and history ablation retraining.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/env_time_focus_ctaux",
        help="Output directory for the environment time-focus experiment.",
    )
    parser.add_argument(
        "--retrain_epochs",
        type=int,
        default=None,
        help="Optional epoch override for history ablation retraining.",
    )
    parser.add_argument(
        "--retrain_patience",
        type=int,
        default=None,
        help="Optional early-stopping patience override for history ablation retraining.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_window_specs(window_pairs: Sequence[str], num_periods: int) -> List[Tuple[int, int]]:
    specs: List[Tuple[int, int]] = []
    for pair in window_pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid window pair '{pair}'. Expected landmark:horizon.")
        landmark_text, horizon_text = pair.split(":", maxsplit=1)
        landmark = int(landmark_text)
        horizon = int(horizon_text)
        if landmark < 0 or landmark >= num_periods:
            raise ValueError(f"Invalid landmark period: {landmark}")
        if horizon <= 0 or landmark + horizon > num_periods:
            raise ValueError(f"Invalid prediction horizon {horizon} for landmark {landmark}.")
        specs.append((landmark, horizon))
    return sorted(set(specs))


def group_windows_by_landmark(window_specs: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for landmark, horizon in window_specs:
        grouped[int(landmark)].append(int(horizon))
    return {landmark: sorted(horizons) for landmark, horizons in grouped.items()}


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
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().tolist())
    return value


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


def get_artifact_config(artifact_dir: Path) -> Dict:
    summary_path = artifact_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing artifact summary: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["config"]


def get_repeat_metrics(artifact_dir: Path, repeat_index: int) -> Dict:
    metrics_path = artifact_dir / f"repeat_{repeat_index}" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing repeat metrics: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_available_repeats(artifact_dir: Path) -> List[int]:
    repeats = []
    for path in sorted(artifact_dir.glob("repeat_*")):
        suffix = path.name.split("_")[-1]
        if suffix.isdigit():
            repeats.append(int(suffix))
    if not repeats:
        raise ValueError(f"No repeat_* directories found in {artifact_dir}")
    return repeats


def reconstruct_repeat_state(
    tree_data,
    artifact_dir: Path,
    repeat_index: int,
    config: Dict,
) -> Dict:
    repeat_seed = int(config["seed"]) + repeat_index * 1000
    train_indices, val_indices, test_indices = split_tree_indices(
        tree_data=tree_data,
        test_size=float(config["test_size"]),
        val_ratio=float(config["val_ratio"]),
        random_state=repeat_seed,
    )
    metrics = get_repeat_metrics(artifact_dir, repeat_index)
    if (
        len(train_indices) != int(metrics["train_trees"])
        or len(val_indices) != int(metrics["val_trees"])
        or len(test_indices) != int(metrics["test_trees"])
    ):
        raise ValueError(
            f"Repeat {repeat_index} split reconstruction mismatch: "
            f"reconstructed=({len(train_indices)}, {len(val_indices)}, {len(test_indices)}) "
            f"artifact=({metrics['train_trees']}, {metrics['val_trees']}, {metrics['test_trees']})"
        )

    preprocessor = fit_static_preprocessor(
        tree_data,
        train_indices,
        use_tree_id_spatial=bool(config.get("use_tree_id_spatial", False)),
    )
    static_x, static_feature_names = transform_static_features(tree_data, preprocessor)
    return {
        "seed": repeat_seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "preprocessor": preprocessor,
        "static_x": static_x,
        "static_feature_names": static_feature_names,
        "artifact_metrics": metrics,
    }


def instantiate_model(tree_data, static_x: np.ndarray, config: Dict) -> DynamicReKSurv:
    return DynamicReKSurv(
        env_dim=len(tree_data.env_feature_names),
        static_dim=static_x.shape[1],
        day_to_period=tree_data.day_to_period,
        num_periods=tree_data.num_periods,
        use_time_features=bool(config.get("use_time_features", False)),
        rnn_type=str(config.get("rnn_type", "gru")),
        model_type=str(config.get("model_type", "legacy")),
        period_env_dim=len(tree_data.period_feature_names),
        env_feature_names=tree_data.env_feature_names,
        period_start_day_index=tree_data.period_start_day_index,
        period_end_day_index=tree_data.period_end_day_index,
        env_aux_mode=str(config.get("env_aux_mode", "next_day")),
        tree_attention_dropout=float(config.get("tree_attention_dropout", 0.10)),
        static_attention_dim=int(config.get("static_attention_dim", 32)),
    )


def forward_model(
    model: DynamicReKSurv,
    env: torch.Tensor,
    period_env: torch.Tensor,
    static_x: torch.Tensor,
    seq_len_days: torch.Tensor,
    seq_len_periods: torch.Tensor,
    landmark_period: torch.Tensor,
    period_observed_mask: torch.Tensor | None = None,
) -> dict:
    return model(
        daily_env_prefix=env,
        period_env_prefix=period_env,
        static_x=static_x,
        seq_len_days=seq_len_days,
        seq_len_periods=seq_len_periods,
        landmark_period=landmark_period,
        period_observed_mask=period_observed_mask,
    )


def resolve_analysis_unit(config: Dict, args: argparse.Namespace) -> str:
    if args.analysis_unit != "auto":
        return args.analysis_unit
    return "period" if str(config.get("model_type", "legacy")) in {"period_ms", "period_ms_tree_query"} else "day"


def get_lag_bin_from_bins(lag: int, lag_bins: Sequence[Tuple[str, int, int | None]]) -> str:
    for label, min_lag, max_lag in lag_bins:
        if max_lag is None:
            if lag >= min_lag:
                return label
        elif min_lag <= lag <= max_lag:
            return label
    raise ValueError(f"Unexpected lag value: {lag}")


def build_mask_for_bin(seq_len: int, label: str, lag_bins: Sequence[Tuple[str, int, int | None]]) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=bool)
    for index in range(seq_len):
        lag = seq_len - 1 - index
        if get_lag_bin_from_bins(lag, lag_bins) == label:
            mask[index] = True
    return mask


def get_history_lengths(args: argparse.Namespace, analysis_unit: str) -> List[int]:
    return sorted(set(args.history_periods if analysis_unit == "period" else args.history_days))


def load_repeat_model(
    tree_data,
    static_x: np.ndarray,
    artifact_dir: Path,
    repeat_index: int,
    config: Dict,
    device: torch.device,
) -> DynamicReKSurv:
    model = instantiate_model(tree_data, static_x, config)
    checkpoint_path = artifact_dir / f"repeat_{repeat_index}" / "best_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_lag_bin(lag: int) -> str:
    for label, min_lag, max_lag in LAG_BINS:
        if max_lag is None:
            if lag >= min_lag:
                return label
        elif min_lag <= lag <= max_lag:
            return label
    raise ValueError(f"Unexpected lag value: {lag}")


def get_bin_length(label: str, seq_len: int) -> int:
    if label == "last_2_days":
        return max(min(seq_len, 2), 0)
    if label == "days_3_7":
        return max(min(seq_len, 7) - 2, 0)
    if label == "days_8_14":
        return max(min(seq_len, 14) - 7, 0)
    if label == "days_15_30":
        return max(min(seq_len, 30) - 14, 0)
    if label == "days_31_60":
        return max(min(seq_len, 60) - 30, 0)
    if label == "days_60_plus":
        return max(seq_len - 60, 0)
    raise ValueError(f"Unknown lag bin: {label}")


def build_attention_mask_for_bin(seq_len: int, label: str) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=bool)
    for day_index in range(seq_len):
        lag = seq_len - 1 - day_index
        if get_lag_bin(lag) == label:
            mask[day_index] = True
    return mask


def get_window_record(
    tree_data,
    eligible_indices: np.ndarray,
    landmark: int,
    horizon: int,
) -> Dict[str, int]:
    future_event_mask = (
        (tree_data.event_flag[eligible_indices] == 1)
        & (tree_data.time_period[eligible_indices] <= landmark + horizon)
    )
    return {
        "eligible_n": int(len(eligible_indices)),
        "future_event_n": int(future_event_mask.sum()),
    }


def save_dataframe_and_json(
    df: pd.DataFrame,
    csv_path: Path,
    json_path: Path,
    payload: Dict,
) -> None:
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, ensure_ascii=False, indent=2)


@torch.no_grad()
def run_attention_analysis(
    args: argparse.Namespace,
    artifact_dir: Path,
    config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    grouped_windows = group_windows_by_landmark(window_specs)
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, config)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_repeat_model(tree_data, static_x, artifact_dir, repeat_index, config, device)

        for landmark, horizons in grouped_windows.items():
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue

            seq_len = int(tree_data.landmark_seq_len_days[landmark])
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
            period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
            seq_len_tensor = torch.full((len(eligible_indices),), seq_len, dtype=torch.long, device=device)
            seq_len_periods = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            landmark_tensor = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            output = forward_model(
                model=model,
                env=env,
                period_env=period_env,
                static_x=static_tensor,
                seq_len_days=seq_len_tensor,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_tensor,
            )
            attention = output["attention_weights"].detach().cpu().numpy()[:, :seq_len]
            attention_sum = attention.sum(axis=1)
            if not np.allclose(attention_sum, 1.0, atol=1e-5):
                raise ValueError(
                    f"Attention weights do not sum to 1 for repeat={repeat_index}, landmark={landmark}."
                )

            base_by_label = {}
            for label, _, _ in LAG_BINS:
                mask = build_attention_mask_for_bin(seq_len, label)
                if mask.sum() == 0:
                    continue
                actual_mass = attention[:, mask].sum(axis=1)
                uniform_mass = float(mask.sum()) / float(seq_len)
                base_by_label[label] = {
                    "actual_mass_mean": float(actual_mass.mean()),
                    "actual_mass_std": float(actual_mass.std(ddof=0)),
                    "uniform_mass": uniform_mass,
                    "enrichment": float(actual_mass.mean() / uniform_mass) if uniform_mass > 0 else float("nan"),
                }

            for horizon in horizons:
                window_info = get_window_record(tree_data, eligible_indices, landmark, horizon)
                for label, stats in base_by_label.items():
                    records.append(
                        {
                            "repeat": repeat_index,
                            "landmark": landmark,
                            "pred_horizon": horizon,
                            "lag_bin": label,
                            "eligible_n": window_info["eligible_n"],
                            "future_event_n": window_info["future_event_n"],
                            **stats,
                        }
                    )

    df = pd.DataFrame(records)
    summary_rows = (
        df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            actual_mass_mean=("actual_mass_mean", "mean"),
            actual_mass_std=("actual_mass_mean", "std"),
            uniform_mass=("uniform_mass", "mean"),
            enrichment_mean=("enrichment", "mean"),
            enrichment_std=("enrichment", "std"),
        )
        .fillna(0.0)
    )
    overall_rows = (
        df.groupby(["lag_bin"], as_index=False)
        .agg(
            actual_mass_mean=("actual_mass_mean", "mean"),
            actual_mass_std=("actual_mass_mean", "std"),
            uniform_mass=("uniform_mass", "mean"),
            enrichment_mean=("enrichment", "mean"),
            enrichment_std=("enrichment", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "attention_lag_enrichment.csv",
        json_path=output_dir / "attention_lag_enrichment.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "window_specs": [list(spec) for spec in window_specs],
                "repeats_used": repeat_indices,
            },
            "summary": summary_rows.to_dict(orient="records"),
            "overall": overall_rows.to_dict(orient="records"),
        },
    )
    return df


@torch.no_grad()
def run_occlusion_analysis(
    args: argparse.Namespace,
    artifact_dir: Path,
    config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    grouped_windows = group_windows_by_landmark(window_specs)
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        print(f"[Occlusion] repeat {repeat_index}")
        repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, config)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_repeat_model(tree_data, static_x, artifact_dir, repeat_index, config, device)

        for landmark, horizons in grouped_windows.items():
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue

            seq_len = int(tree_data.landmark_seq_len_days[landmark])
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
            period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
            seq_len_tensor = torch.full((len(eligible_indices),), seq_len, dtype=torch.long, device=device)
            seq_len_periods = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            landmark_tensor = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)

            original_output = forward_model(
                model=model,
                env=env,
                period_env=period_env,
                static_x=static_tensor,
                seq_len_days=seq_len_tensor,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_tensor,
            )
            original_risk = {
                horizon: original_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                for horizon in horizons
            }

            for history_days in sorted(set(args.history_days)):
                if history_days > seq_len:
                    continue
                max_start = seq_len - history_days
                for start_day in range(max_start + 1):
                    masked_env = env.clone()
                    masked_env[:, start_day : start_day + history_days, :] = 0.0
                    masked_output = forward_model(
                        model=model,
                        env=masked_env,
                        period_env=period_env,
                        static_x=static_tensor,
                        seq_len_days=seq_len_tensor,
                        seq_len_periods=seq_len_periods,
                        landmark_period=landmark_tensor,
                    )
                    block_end_lag = seq_len - (start_day + history_days)
                    lag_bin = get_lag_bin(block_end_lag)

                    for horizon in horizons:
                        risk = masked_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                        delta = torch.abs(risk - original_risk[horizon]).detach().cpu().numpy()
                        window_info = get_window_record(tree_data, eligible_indices, landmark, horizon)
                        records.append(
                            {
                                "repeat": repeat_index,
                                "landmark": landmark,
                                "pred_horizon": horizon,
                                "history_days": history_days,
                                "start_day": start_day,
                                "block_end_lag": block_end_lag,
                                "lag_bin": lag_bin,
                                "eligible_n": window_info["eligible_n"],
                                "future_event_n": window_info["future_event_n"],
                                "delta_risk_mean": float(delta.mean()),
                                "delta_risk_std": float(delta.std(ddof=0)),
                            }
                        )

    df = pd.DataFrame(records)
    summary_by_k = (
        df.groupby(["landmark", "pred_horizon", "history_days", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )
    summary_aggregate = (
        df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "occlusion_lag_sensitivity.csv",
        json_path=output_dir / "occlusion_lag_sensitivity.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "window_specs": [list(spec) for spec in window_specs],
                "history_days": sorted(set(args.history_days)),
                "repeats_used": repeat_indices,
            },
            "summary_by_history_days": summary_by_k.to_dict(orient="records"),
            "summary_aggregate": summary_aggregate.to_dict(orient="records"),
        },
    )
    return df


@torch.no_grad()
def run_permutation_analysis(
    args: argparse.Namespace,
    artifact_dir: Path,
    config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    grouped_windows = group_windows_by_landmark(window_specs)
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    permutation_days = [day for day in sorted(set(args.history_days)) if day >= 7]
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        print(f"[Permutation] repeat {repeat_index}")
        repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, config)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_repeat_model(tree_data, static_x, artifact_dir, repeat_index, config, device)
        rng = np.random.default_rng(repeat_state["seed"] + 2026)

        for landmark, horizons in grouped_windows.items():
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue

            seq_len = int(tree_data.landmark_seq_len_days[landmark])
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
            period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
            seq_len_tensor = torch.full((len(eligible_indices),), seq_len, dtype=torch.long, device=device)
            seq_len_periods = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            landmark_tensor = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)

            original_output = forward_model(
                model=model,
                env=env,
                period_env=period_env,
                static_x=static_tensor,
                seq_len_days=seq_len_tensor,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_tensor,
            )
            original_risk = {
                horizon: original_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                for horizon in horizons
            }

            for history_days in permutation_days:
                if history_days > seq_len:
                    continue
                max_start = seq_len - history_days
                for start_day in range(max_start + 1):
                    block_end_lag = seq_len - (start_day + history_days)
                    lag_bin = get_lag_bin(block_end_lag)
                    delta_collect = {horizon: [] for horizon in horizons}

                    for _ in range(args.perm_repeats):
                        permuted_env = env.clone()
                        for sample_index in range(permuted_env.size(0)):
                            permutation = torch.tensor(
                                rng.permutation(history_days),
                                dtype=torch.long,
                                device=device,
                            )
                            permuted_env[sample_index, start_day : start_day + history_days, :] = (
                                permuted_env[sample_index, start_day : start_day + history_days, :][permutation]
                            )

                        permuted_output = forward_model(
                            model=model,
                            env=permuted_env,
                            period_env=period_env,
                            static_x=static_tensor,
                            seq_len_days=seq_len_tensor,
                            seq_len_periods=seq_len_periods,
                            landmark_period=landmark_tensor,
                        )
                        for horizon in horizons:
                            risk = permuted_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                            delta_collect[horizon].append(
                                torch.abs(risk - original_risk[horizon]).detach().cpu().numpy()
                            )

                    for horizon in horizons:
                        delta_matrix = np.stack(delta_collect[horizon], axis=0)
                        delta = delta_matrix.mean(axis=0)
                        window_info = get_window_record(tree_data, eligible_indices, landmark, horizon)
                        records.append(
                            {
                                "repeat": repeat_index,
                                "landmark": landmark,
                                "pred_horizon": horizon,
                                "history_days": history_days,
                                "start_day": start_day,
                                "block_end_lag": block_end_lag,
                                "lag_bin": lag_bin,
                                "eligible_n": window_info["eligible_n"],
                                "future_event_n": window_info["future_event_n"],
                                "delta_risk_mean": float(delta.mean()),
                                "delta_risk_std": float(delta.std(ddof=0)),
                            }
                        )

    df = pd.DataFrame(records)
    summary_by_k = (
        df.groupby(["landmark", "pred_horizon", "history_days", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )
    summary_aggregate = (
        df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "permutation_sensitivity.csv",
        json_path=output_dir / "permutation_sensitivity.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "window_specs": [list(spec) for spec in window_specs],
                "history_days": permutation_days,
                "perm_repeats": int(args.perm_repeats),
                "repeats_used": repeat_indices,
            },
            "summary_by_history_days": summary_by_k.to_dict(orient="records"),
            "summary_aggregate": summary_aggregate.to_dict(orient="records"),
        },
    )
    return df


def apply_history_view(
    env: torch.Tensor,
    seq_len_days: torch.Tensor,
    history_mode: str,
    history_days: int | None,
    use_time_features: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if history_mode == "full":
        return env, seq_len_days

    if history_days is None or history_days <= 0:
        raise ValueError(f"history_days must be positive for mode={history_mode}.")

    if history_mode == "exclude_last_k":
        new_seq_len = torch.clamp(seq_len_days - int(history_days), min=1)
        return env, new_seq_len

    if history_mode == "only_last_k":
        if use_time_features:
            raise ValueError("only_last_k retraining is not supported when use_time_features=true.")
        new_env = torch.zeros_like(env)
        new_seq_len = torch.clamp(seq_len_days, max=int(history_days))
        for sample_index in range(env.size(0)):
            seq_len = int(seq_len_days[sample_index].item())
            keep_days = int(new_seq_len[sample_index].item())
            start = seq_len - keep_days
            new_env[sample_index, :keep_days, :] = env[sample_index, start:seq_len, :]
        return new_env, new_seq_len

    raise ValueError(f"Unsupported history mode: {history_mode}")


def train_one_epoch_with_history(
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
    history_mode: str,
    history_days: int | None,
    use_time_features: bool,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0
    running_ll = 0.0
    running_rank = 0.0
    running_env_aux = 0.0
    running_ct_aux = 0.0
    num_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        env_view, seq_view = apply_history_view(
            env=batch["env"],
            seq_len_days=batch["seq_len_days"],
            history_mode=history_mode,
            history_days=history_days,
            use_time_features=use_time_features,
        )
        model_output = forward_model(
            model=model,
            env=env_view,
            period_env=batch["period_env"],
            static_x=batch["static_x"],
            seq_len_days=seq_view,
            seq_len_periods=batch["seq_len_periods"],
            landmark_period=batch["landmark_period"],
        )
        losses = compute_dynamic_loss(
            model_output=model_output,
            batch={**batch, "env": env_view, "seq_len_days": seq_view},
            alpha=alpha,
            beta=beta,
            gamma_env=gamma_env,
            gamma_ct=gamma_ct,
            use_ct_aux_task=use_ct_aux_task,
            ct_target_mean=ct_target_mean,
            ct_target_std=ct_target_std,
            ct_aux_loss=ct_aux_loss,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = int(env_view.size(0))
        num_samples += batch_size
        running_total += float(losses["total"].item()) * batch_size
        running_ll += float(losses["ll"].item()) * batch_size
        running_rank += float(losses["rank"].item()) * batch_size
        running_env_aux += float(losses["env_aux"].item()) * batch_size
        running_ct_aux += float(losses["ct_aux"].item()) * batch_size

    return {
        "total": running_total / max(num_samples, 1),
        "ll": running_ll / max(num_samples, 1),
        "rank": running_rank / max(num_samples, 1),
        "env_aux": running_env_aux / max(num_samples, 1),
        "ct_aux": running_ct_aux / max(num_samples, 1),
    }


@torch.no_grad()
def evaluate_windows_with_history(
    model: DynamicReKSurv,
    tree_data,
    static_x: np.ndarray,
    evaluation_indices: np.ndarray,
    train_indices: np.ndarray,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    history_mode: str,
    history_days: int | None,
    use_time_features: bool,
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

        env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
        period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
        static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
        seq_len_days = torch.full(
            (len(eligible_indices),),
            int(tree_data.landmark_seq_len_days[landmark]),
            dtype=torch.long,
            device=device,
        )
        seq_len_periods = torch.full(
            (len(eligible_indices),),
            int(landmark),
            dtype=torch.long,
            device=device,
        )
        landmark_tensor = torch.full(
            (len(eligible_indices),),
            int(landmark),
            dtype=torch.long,
            device=device,
        )
        env_view, seq_view = apply_history_view(
            env=env,
            seq_len_days=seq_len_days,
            history_mode=history_mode,
            history_days=history_days,
            use_time_features=use_time_features,
        )
        model_output = forward_model(
            model=model,
            env=env_view,
            period_env=period_env,
            static_x=static_tensor,
            seq_len_days=seq_view,
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
    mean_ctd = float(np.mean(valid_ctd)) if valid_ctd else float("-inf")
    return records, mean_ctd


def run_history_variant_repeat(
    tree_data,
    artifact_config: Dict,
    repeat_index: int,
    artifact_dir: Path,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    history_mode: str,
    history_days: int | None,
    retrain_epochs: int | None,
    retrain_patience: int | None,
) -> List[Dict]:
    repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, artifact_config)
    repeat_seed = repeat_state["seed"]
    set_seed(repeat_seed)

    train_indices = repeat_state["train_indices"]
    val_indices = repeat_state["val_indices"]
    static_x = repeat_state["static_x"]

    train_samples = build_prefix_samples(tree_data, train_indices, include_landmark_zero=True)
    train_dataset = PrefixSampleDataset(tree_data, static_x, train_samples)
    train_sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(artifact_config.get("batch_size", 32)),
        sampler=train_sampler,
        drop_last=False,
    )

    ct_target_stats = {"mean": 0.0, "std": 1.0, "count": 0}
    if bool(artifact_config.get("use_ct_aux_task", False)):
        ct_target_stats = fit_ct_delta_preprocessor(train_samples)

    model = instantiate_model(tree_data, static_x, artifact_config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(artifact_config.get("lr", 1e-3)),
        weight_decay=float(artifact_config.get("weight_decay", 1e-4)),
    )
    patience = int(retrain_patience if retrain_patience is not None else artifact_config.get("patience", 20))
    epochs = int(retrain_epochs if retrain_epochs is not None else artifact_config.get("epochs", 200))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(patience // 4, 2),
        min_lr=1e-5,
    )

    best_state = None
    best_val_ctd = float("-inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_with_history(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=float(artifact_config.get("alpha", 1.0)),
            beta=float(artifact_config.get("beta", 0.2)),
            gamma_env=float(artifact_config.get("gamma_env", artifact_config.get("gamma", 0.0) or 0.0)),
            gamma_ct=float(artifact_config.get("gamma_ct", 0.0)),
            use_ct_aux_task=bool(artifact_config.get("use_ct_aux_task", False)),
            ct_target_mean=float(ct_target_stats["mean"]),
            ct_target_std=float(ct_target_stats["std"]),
            ct_aux_loss=str(artifact_config.get("ct_aux_loss", "huber")),
            history_mode=history_mode,
            history_days=history_days,
            use_time_features=bool(artifact_config.get("use_time_features", False)),
        )
        val_records, val_ctd = evaluate_windows_with_history(
            model=model,
            tree_data=tree_data,
            static_x=static_x,
            evaluation_indices=val_indices,
            train_indices=train_indices,
            window_specs=window_specs,
            device=device,
            history_mode=history_mode,
            history_days=history_days,
            use_time_features=bool(artifact_config.get("use_time_features", False)),
        )
        scheduler.step(val_ctd if np.isfinite(val_ctd) else -1.0)
        if val_ctd > best_val_ctd:
            best_val_ctd = val_ctd
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or no_improve == 0:
            print(
                f"[History] repeat {repeat_index} | mode={history_mode} | k={history_days} | "
                f"epoch={epoch:03d} | loss={train_loss['total']:.4f} | val_ctd={val_ctd:.4f}"
            )
        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"History ablation training failed to produce a checkpoint for repeat={repeat_index}, "
            f"mode={history_mode}, k={history_days}"
        )

    model.load_state_dict(best_state)
    test_records, _ = evaluate_windows_with_history(
        model=model,
        tree_data=tree_data,
        static_x=static_x,
        evaluation_indices=repeat_state["test_indices"],
        train_indices=train_indices,
        window_specs=window_specs,
        device=device,
        history_mode=history_mode,
        history_days=history_days,
        use_time_features=bool(artifact_config.get("use_time_features", False)),
    )
    return test_records


def run_history_ablation(
    args: argparse.Namespace,
    artifact_dir: Path,
    artifact_config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        metrics = get_repeat_metrics(artifact_dir, repeat_index)
        for record in metrics["test_windows"]:
            key = (int(record["landmark"]), int(record["pred_horizon"]))
            if key not in set(window_specs):
                continue
            records.append(
                {
                    "variant": "full",
                    "history_mode": "full",
                    "history_days": None,
                    "repeat": repeat_index,
                    **record,
                }
            )

    for history_days in sorted(set(args.history_days)):
        for history_mode in ("only_last_k", "exclude_last_k"):
            if bool(artifact_config.get("use_time_features", False)) and history_mode == "only_last_k":
                raise ValueError("only_last_k ablation is not supported for time-feature-enabled checkpoints.")
            for repeat_index in repeat_indices:
                print(f"[History] {history_mode} | k={history_days} | repeat {repeat_index}")
                test_records = run_history_variant_repeat(
                    tree_data=tree_data,
                    artifact_config=artifact_config,
                    repeat_index=repeat_index,
                    artifact_dir=artifact_dir,
                    window_specs=window_specs,
                    device=device,
                    history_mode=history_mode,
                    history_days=history_days,
                    retrain_epochs=args.retrain_epochs,
                    retrain_patience=args.retrain_patience,
                )
                for record in test_records:
                    records.append(
                        {
                            "variant": f"{history_mode}_{history_days}",
                            "history_mode": history_mode,
                            "history_days": history_days,
                            "repeat": repeat_index,
                            **record,
                        }
                    )

    df = pd.DataFrame(records)
    summary_rows = (
        df.groupby(["variant", "history_mode", "history_days", "landmark", "pred_horizon"], dropna=False, as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            eligible_n_std=("eligible_n", "std"),
            future_event_n_mean=("future_event_n", "mean"),
            future_event_n_std=("future_event_n", "std"),
            ctd_mean=("ctd", "mean"),
            ctd_std=("ctd", "std"),
            bstd_mean=("bstd", "mean"),
            bstd_std=("bstd", "std"),
        )
        .fillna(0.0)
    )
    overall_rows = (
        df.groupby(["variant", "history_mode", "history_days"], dropna=False, as_index=False)
        .agg(
            ctd_mean=("ctd", "mean"),
            ctd_std=("ctd", "std"),
            bstd_mean=("bstd", "mean"),
            bstd_std=("bstd", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "history_ablation_summary.csv",
        json_path=output_dir / "history_ablation_summary.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "window_specs": [list(spec) for spec in window_specs],
                "history_days": sorted(set(args.history_days)),
                "repeats_used": repeat_indices,
                "retrain_epochs": args.retrain_epochs,
                "retrain_patience": args.retrain_patience,
            },
            "summary": summary_rows.to_dict(orient="records"),
            "overall": overall_rows.to_dict(orient="records"),
        },
    )
    return df


def build_period_observed_mask(
    landmark_period: torch.Tensor,
    num_periods: int,
    history_mode: str,
    history_periods: int | None,
) -> torch.Tensor:
    landmark_period = landmark_period.view(-1).long()
    period_ids = torch.arange(1, num_periods + 1, device=landmark_period.device).view(1, -1)
    observed_mask = period_ids <= landmark_period.unsqueeze(1)

    if history_mode == "full":
        return observed_mask
    if history_periods is None or history_periods <= 0:
        raise ValueError(f"history_periods must be positive for mode={history_mode}.")

    if history_mode == "only_last_k":
        start_period = torch.clamp(landmark_period - int(history_periods) + 1, min=1)
        return observed_mask & (period_ids >= start_period.unsqueeze(1))
    if history_mode == "exclude_last_k":
        cutoff_period = torch.clamp(landmark_period - int(history_periods), min=0)
        return period_ids <= cutoff_period.unsqueeze(1)
    raise ValueError(f"Unsupported period history mode: {history_mode}")


@torch.no_grad()
def run_period_attention_analysis(
    args: argparse.Namespace,
    artifact_dir: Path,
    config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    grouped_windows = group_windows_by_landmark(window_specs)
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, config)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_repeat_model(tree_data, static_x, artifact_dir, repeat_index, config, device)

        for landmark, horizons in grouped_windows.items():
            if landmark <= 0:
                continue
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue

            seq_len = int(tree_data.landmark_seq_len_days[landmark])
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
            period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
            seq_len_days = torch.full((len(eligible_indices),), seq_len, dtype=torch.long, device=device)
            seq_len_periods = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            landmark_tensor = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            output = forward_model(
                model=model,
                env=env,
                period_env=period_env,
                static_x=static_tensor,
                seq_len_days=seq_len_days,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_tensor,
            )
            attention = output["period_attention_weights"].detach().cpu().numpy()[:, :landmark]
            attention_sum = attention.sum(axis=1)
            if not np.allclose(attention_sum, 1.0, atol=1e-5):
                raise ValueError(
                    f"Period attention weights do not sum to 1 for repeat={repeat_index}, landmark={landmark}."
                )

            base_by_label = {}
            for label, _, _ in PERIOD_LAG_BINS:
                mask = build_mask_for_bin(landmark, label, PERIOD_LAG_BINS)
                if mask.sum() == 0:
                    continue
                actual_mass = attention[:, mask].sum(axis=1)
                uniform_mass = float(mask.sum()) / float(landmark)
                base_by_label[label] = {
                    "actual_mass_mean": float(actual_mass.mean()),
                    "actual_mass_std": float(actual_mass.std(ddof=0)),
                    "uniform_mass": uniform_mass,
                    "enrichment": float(actual_mass.mean() / uniform_mass) if uniform_mass > 0 else float("nan"),
                }

            for horizon in horizons:
                window_info = get_window_record(tree_data, eligible_indices, landmark, horizon)
                for label, stats in base_by_label.items():
                    records.append(
                        {
                            "repeat": repeat_index,
                            "landmark": landmark,
                            "pred_horizon": horizon,
                            "lag_bin": label,
                            "eligible_n": window_info["eligible_n"],
                            "future_event_n": window_info["future_event_n"],
                            **stats,
                        }
                    )

    df = pd.DataFrame(records)
    summary_rows = (
        df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            actual_mass_mean=("actual_mass_mean", "mean"),
            actual_mass_std=("actual_mass_mean", "std"),
            uniform_mass=("uniform_mass", "mean"),
            enrichment_mean=("enrichment", "mean"),
            enrichment_std=("enrichment", "std"),
        )
        .fillna(0.0)
    )
    overall_rows = (
        df.groupby(["lag_bin"], as_index=False)
        .agg(
            actual_mass_mean=("actual_mass_mean", "mean"),
            actual_mass_std=("actual_mass_mean", "std"),
            uniform_mass=("uniform_mass", "mean"),
            enrichment_mean=("enrichment", "mean"),
            enrichment_std=("enrichment", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "attention_lag_enrichment.csv",
        json_path=output_dir / "attention_lag_enrichment.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "analysis_unit": "period",
                "window_specs": [list(spec) for spec in window_specs],
                "repeats_used": repeat_indices,
            },
            "summary": summary_rows.to_dict(orient="records"),
            "overall": overall_rows.to_dict(orient="records"),
        },
    )
    return df


@torch.no_grad()
def run_period_occlusion_analysis(
    args: argparse.Namespace,
    artifact_dir: Path,
    config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    grouped_windows = group_windows_by_landmark(window_specs)
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    history_periods_list = get_history_lengths(args, "period")
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        print(f"[Period Occlusion] repeat {repeat_index}")
        repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, config)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_repeat_model(tree_data, static_x, artifact_dir, repeat_index, config, device)

        for landmark, horizons in grouped_windows.items():
            if landmark <= 0:
                continue
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue

            seq_len = int(tree_data.landmark_seq_len_days[landmark])
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
            period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
            seq_len_days = torch.full((len(eligible_indices),), seq_len, dtype=torch.long, device=device)
            seq_len_periods = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            landmark_tensor = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)

            original_output = forward_model(
                model=model,
                env=env,
                period_env=period_env,
                static_x=static_tensor,
                seq_len_days=seq_len_days,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_tensor,
            )
            original_risk = {
                horizon: original_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                for horizon in horizons
            }

            for history_periods in history_periods_list:
                if history_periods > landmark:
                    continue
                max_start = landmark - history_periods
                for start_period in range(max_start + 1):
                    masked_period_env = period_env.clone()
                    masked_period_env[:, start_period : start_period + history_periods, :] = 0.0
                    masked_output = forward_model(
                        model=model,
                        env=env,
                        period_env=masked_period_env,
                        static_x=static_tensor,
                        seq_len_days=seq_len_days,
                        seq_len_periods=seq_len_periods,
                        landmark_period=landmark_tensor,
                    )
                    block_end_lag = landmark - (start_period + history_periods)
                    lag_bin = get_lag_bin_from_bins(block_end_lag, PERIOD_LAG_BINS)

                    for horizon in horizons:
                        risk = masked_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                        delta = torch.abs(risk - original_risk[horizon]).detach().cpu().numpy()
                        window_info = get_window_record(tree_data, eligible_indices, landmark, horizon)
                        records.append(
                            {
                                "repeat": repeat_index,
                                "landmark": landmark,
                                "pred_horizon": horizon,
                                "history_periods": history_periods,
                                "start_period": start_period,
                                "block_end_lag": block_end_lag,
                                "lag_bin": lag_bin,
                                "eligible_n": window_info["eligible_n"],
                                "future_event_n": window_info["future_event_n"],
                                "delta_risk_mean": float(delta.mean()),
                                "delta_risk_std": float(delta.std(ddof=0)),
                            }
                        )

    df = pd.DataFrame(records)
    summary_by_k = (
        df.groupby(["landmark", "pred_horizon", "history_periods", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )
    summary_aggregate = (
        df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "occlusion_lag_sensitivity.csv",
        json_path=output_dir / "occlusion_lag_sensitivity.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "analysis_unit": "period",
                "window_specs": [list(spec) for spec in window_specs],
                "history_periods": history_periods_list,
                "repeats_used": repeat_indices,
            },
            "summary_by_history_periods": summary_by_k.to_dict(orient="records"),
            "summary_aggregate": summary_aggregate.to_dict(orient="records"),
        },
    )
    return df


@torch.no_grad()
def run_period_permutation_analysis(
    args: argparse.Namespace,
    artifact_dir: Path,
    config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    grouped_windows = group_windows_by_landmark(window_specs)
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    history_periods_list = [value for value in get_history_lengths(args, "period") if value >= 2]
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        print(f"[Period Permutation] repeat {repeat_index}")
        repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, config)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_repeat_model(tree_data, static_x, artifact_dir, repeat_index, config, device)
        rng = np.random.default_rng(repeat_state["seed"] + 2027)

        for landmark, horizons in grouped_windows.items():
            if landmark <= 1:
                continue
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue

            seq_len = int(tree_data.landmark_seq_len_days[landmark])
            env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
            period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
            seq_len_days = torch.full((len(eligible_indices),), seq_len, dtype=torch.long, device=device)
            seq_len_periods = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)
            landmark_tensor = torch.full((len(eligible_indices),), landmark, dtype=torch.long, device=device)

            original_output = forward_model(
                model=model,
                env=env,
                period_env=period_env,
                static_x=static_tensor,
                seq_len_days=seq_len_days,
                seq_len_periods=seq_len_periods,
                landmark_period=landmark_tensor,
            )
            original_risk = {
                horizon: original_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                for horizon in horizons
            }

            for history_periods in history_periods_list:
                if history_periods > landmark:
                    continue
                max_start = landmark - history_periods
                for start_period in range(max_start + 1):
                    block_end_lag = landmark - (start_period + history_periods)
                    lag_bin = get_lag_bin_from_bins(block_end_lag, PERIOD_LAG_BINS)
                    delta_collect = {horizon: [] for horizon in horizons}

                    for _ in range(args.perm_repeats):
                        permuted_period_env = period_env.clone()
                        for sample_index in range(permuted_period_env.size(0)):
                            permutation = torch.tensor(
                                rng.permutation(history_periods),
                                dtype=torch.long,
                                device=device,
                            )
                            permuted_period_env[sample_index, start_period : start_period + history_periods, :] = (
                                permuted_period_env[sample_index, start_period : start_period + history_periods, :][permutation]
                            )
                        permuted_output = forward_model(
                            model=model,
                            env=env,
                            period_env=permuted_period_env,
                            static_x=static_tensor,
                            seq_len_days=seq_len_days,
                            seq_len_periods=seq_len_periods,
                            landmark_period=landmark_tensor,
                        )
                        for horizon in horizons:
                            risk = permuted_output["event_probs"][:, landmark : landmark + horizon].sum(dim=1)
                            delta_collect[horizon].append(
                                torch.abs(risk - original_risk[horizon]).detach().cpu().numpy()
                            )

                    for horizon in horizons:
                        delta = np.concatenate(delta_collect[horizon], axis=0)
                        window_info = get_window_record(tree_data, eligible_indices, landmark, horizon)
                        records.append(
                            {
                                "repeat": repeat_index,
                                "landmark": landmark,
                                "pred_horizon": horizon,
                                "history_periods": history_periods,
                                "start_period": start_period,
                                "block_end_lag": block_end_lag,
                                "lag_bin": lag_bin,
                                "eligible_n": window_info["eligible_n"],
                                "future_event_n": window_info["future_event_n"],
                                "delta_risk_mean": float(delta.mean()),
                                "delta_risk_std": float(delta.std(ddof=0)),
                            }
                        )

    df = pd.DataFrame(records)
    summary_by_k = (
        df.groupby(["landmark", "pred_horizon", "history_periods", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )
    summary_aggregate = (
        df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            delta_risk_mean=("delta_risk_mean", "mean"),
            delta_risk_std=("delta_risk_mean", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "permutation_sensitivity.csv",
        json_path=output_dir / "permutation_sensitivity.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "analysis_unit": "period",
                "window_specs": [list(spec) for spec in window_specs],
                "history_periods": history_periods_list,
                "perm_repeats": args.perm_repeats,
                "repeats_used": repeat_indices,
            },
            "summary_by_history_periods": summary_by_k.to_dict(orient="records"),
            "summary_aggregate": summary_aggregate.to_dict(orient="records"),
        },
    )
    return df


def train_one_epoch_with_period_history(
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
    history_mode: str,
    history_periods: int | None,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0
    running_ll = 0.0
    running_rank = 0.0
    running_env_aux = 0.0
    running_ct_aux = 0.0
    running_spread = 0.0
    num_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        period_observed_mask = build_period_observed_mask(
            landmark_period=batch["landmark_period"],
            num_periods=model.num_periods,
            history_mode=history_mode,
            history_periods=history_periods,
        )
        visible_periods = period_observed_mask.sum(dim=1).long()
        model_output = forward_model(
            model=model,
            env=batch["env"],
            period_env=batch["period_env"],
            static_x=batch["static_x"],
            seq_len_days=batch["seq_len_days"],
            seq_len_periods=batch["seq_len_periods"],
            landmark_period=batch["landmark_period"],
            period_observed_mask=period_observed_mask,
        )
        losses = compute_dynamic_loss(
            model_output=model_output,
            batch={**batch, "seq_len_periods": visible_periods},
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
        running_spread += float(losses["spread"].item()) * batch_size

    return {
        "total": running_total / max(num_samples, 1),
        "ll": running_ll / max(num_samples, 1),
        "rank": running_rank / max(num_samples, 1),
        "env_aux": running_env_aux / max(num_samples, 1),
        "ct_aux": running_ct_aux / max(num_samples, 1),
        "spread": running_spread / max(num_samples, 1),
    }


@torch.no_grad()
def evaluate_windows_with_period_history(
    model: DynamicReKSurv,
    tree_data,
    static_x: np.ndarray,
    evaluation_indices: np.ndarray,
    train_indices: np.ndarray,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    history_mode: str,
    history_periods: int | None,
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

        env = torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32, device=device)
        period_env = torch.tensor(tree_data.period_env[eligible_indices], dtype=torch.float32, device=device)
        static_tensor = torch.tensor(static_x[eligible_indices], dtype=torch.float32, device=device)
        seq_len_days = torch.full(
            (len(eligible_indices),),
            int(tree_data.landmark_seq_len_days[landmark]),
            dtype=torch.long,
            device=device,
        )
        seq_len_periods = torch.full((len(eligible_indices),), int(landmark), dtype=torch.long, device=device)
        landmark_tensor = torch.full((len(eligible_indices),), int(landmark), dtype=torch.long, device=device)
        period_observed_mask = build_period_observed_mask(
            landmark_period=landmark_tensor,
            num_periods=model.num_periods,
            history_mode=history_mode,
            history_periods=history_periods,
        )
        model_output = forward_model(
            model=model,
            env=env,
            period_env=period_env,
            static_x=static_tensor,
            seq_len_days=seq_len_days,
            seq_len_periods=seq_len_periods,
            landmark_period=landmark_tensor,
            period_observed_mask=period_observed_mask,
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
    mean_ctd = float(np.mean(valid_ctd)) if valid_ctd else float("-inf")
    return records, mean_ctd


def run_period_history_variant_repeat(
    tree_data,
    artifact_config: Dict,
    repeat_index: int,
    artifact_dir: Path,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    history_mode: str,
    history_periods: int | None,
    retrain_epochs: int | None,
    retrain_patience: int | None,
) -> List[Dict]:
    repeat_state = reconstruct_repeat_state(tree_data, artifact_dir, repeat_index, artifact_config)
    repeat_seed = repeat_state["seed"]
    set_seed(repeat_seed)

    train_indices = repeat_state["train_indices"]
    val_indices = repeat_state["val_indices"]
    static_x = repeat_state["static_x"]

    train_samples = build_prefix_samples(tree_data, train_indices, include_landmark_zero=True)
    train_dataset = PrefixSampleDataset(tree_data, static_x, train_samples)
    train_sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(artifact_config.get("batch_size", 32)),
        sampler=train_sampler,
        drop_last=False,
    )

    ct_target_stats = {"mean": 0.0, "std": 1.0, "count": 0}
    if bool(artifact_config.get("use_ct_aux_task", False)):
        ct_target_stats = fit_ct_delta_preprocessor(train_samples)

    model = instantiate_model(tree_data, static_x, artifact_config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(artifact_config.get("lr", 1e-3)),
        weight_decay=float(artifact_config.get("weight_decay", 1e-4)),
    )
    patience = int(retrain_patience if retrain_patience is not None else artifact_config.get("patience", 20))
    epochs = int(retrain_epochs if retrain_epochs is not None else artifact_config.get("epochs", 200))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(patience // 4, 2),
        min_lr=1e-5,
    )

    best_state = None
    best_val_ctd = float("-inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_with_period_history(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=float(artifact_config.get("alpha", 1.0)),
            beta=float(artifact_config.get("beta", 0.2)),
            gamma_env=float(artifact_config.get("gamma_env", artifact_config.get("gamma", 0.0) or 0.0)),
            gamma_ct=float(artifact_config.get("gamma_ct", 0.0)),
            use_ct_aux_task=bool(artifact_config.get("use_ct_aux_task", False)),
            ct_target_mean=float(ct_target_stats["mean"]),
            ct_target_std=float(ct_target_stats["std"]),
            ct_aux_loss=str(artifact_config.get("ct_aux_loss", "huber")),
            env_aux_mode=str(artifact_config.get("env_aux_mode", "none")),
            lag_spread_weight=float(artifact_config.get("lag_spread_weight", 0.0)),
            history_mode=history_mode,
            history_periods=history_periods,
        )
        val_records, val_ctd = evaluate_windows_with_period_history(
            model=model,
            tree_data=tree_data,
            static_x=static_x,
            evaluation_indices=val_indices,
            train_indices=train_indices,
            window_specs=window_specs,
            device=device,
            history_mode=history_mode,
            history_periods=history_periods,
        )
        scheduler.step(val_ctd if np.isfinite(val_ctd) else -1.0)
        if val_ctd > best_val_ctd:
            best_val_ctd = val_ctd
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or no_improve == 0:
            print(
                f"[Period History] repeat {repeat_index} | mode={history_mode} | k={history_periods} | "
                f"epoch={epoch:03d} | loss={train_loss['total']:.4f} | val_ctd={val_ctd:.4f}"
            )
        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"Period history ablation training failed for repeat={repeat_index}, "
            f"mode={history_mode}, k={history_periods}"
        )

    model.load_state_dict(best_state)
    test_records, _ = evaluate_windows_with_period_history(
        model=model,
        tree_data=tree_data,
        static_x=static_x,
        evaluation_indices=repeat_state["test_indices"],
        train_indices=train_indices,
        window_specs=window_specs,
        device=device,
        history_mode=history_mode,
        history_periods=history_periods,
    )
    return test_records


def run_period_history_ablation(
    args: argparse.Namespace,
    artifact_dir: Path,
    artifact_config: Dict,
    tree_data,
    window_specs: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    repeat_indices = get_available_repeats(artifact_dir)[: args.retrain_repeats]
    history_periods_list = get_history_lengths(args, "period")
    records: List[Dict] = []

    for repeat_index in repeat_indices:
        metrics = get_repeat_metrics(artifact_dir, repeat_index)
        for record in metrics["test_windows"]:
            key = (int(record["landmark"]), int(record["pred_horizon"]))
            if key not in set(window_specs):
                continue
            records.append(
                {
                    "variant": "full",
                    "history_mode": "full",
                    "history_periods": None,
                    "repeat": repeat_index,
                    **record,
                }
            )

    for history_periods in history_periods_list:
        for history_mode in ("only_last_k", "exclude_last_k"):
            for repeat_index in repeat_indices:
                print(f"[Period History] {history_mode} | k={history_periods} | repeat {repeat_index}")
                test_records = run_period_history_variant_repeat(
                    tree_data=tree_data,
                    artifact_config=artifact_config,
                    repeat_index=repeat_index,
                    artifact_dir=artifact_dir,
                    window_specs=window_specs,
                    device=device,
                    history_mode=history_mode,
                    history_periods=history_periods,
                    retrain_epochs=args.retrain_epochs,
                    retrain_patience=args.retrain_patience,
                )
                for record in test_records:
                    records.append(
                        {
                            "variant": f"{history_mode}_{history_periods}",
                            "history_mode": history_mode,
                            "history_periods": history_periods,
                            "repeat": repeat_index,
                            **record,
                        }
                    )

    df = pd.DataFrame(records)
    summary_rows = (
        df.groupby(["variant", "history_mode", "history_periods", "landmark", "pred_horizon"], dropna=False, as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            eligible_n_std=("eligible_n", "std"),
            future_event_n_mean=("future_event_n", "mean"),
            future_event_n_std=("future_event_n", "std"),
            ctd_mean=("ctd", "mean"),
            ctd_std=("ctd", "std"),
            bstd_mean=("bstd", "mean"),
            bstd_std=("bstd", "std"),
        )
        .fillna(0.0)
    )
    overall_rows = (
        df.groupby(["variant", "history_mode", "history_periods"], dropna=False, as_index=False)
        .agg(
            ctd_mean=("ctd", "mean"),
            ctd_std=("ctd", "std"),
            bstd_mean=("bstd", "mean"),
            bstd_std=("bstd", "std"),
        )
        .fillna(0.0)
    )

    save_dataframe_and_json(
        df=df,
        csv_path=output_dir / "history_ablation_summary.csv",
        json_path=output_dir / "history_ablation_summary.json",
        payload={
            "config": {
                "artifact_dir": str(artifact_dir),
                "analysis_unit": "period",
                "window_specs": [list(spec) for spec in window_specs],
                "history_periods": history_periods_list,
                "repeats_used": repeat_indices,
                "retrain_epochs": args.retrain_epochs,
                "retrain_patience": args.retrain_patience,
            },
            "summary": summary_rows.to_dict(orient="records"),
            "overall": overall_rows.to_dict(orient="records"),
        },
    )
    return df


def determine_window_conclusion_period(
    attention_rankings: List[Tuple[str, float]],
    occlusion_rankings: List[Tuple[str, float]],
    permutation_rankings: List[Tuple[str, float]],
    full_metrics: Dict[str, float],
    only_last_metrics: Dict[int, Dict[str, float] | None],
    exclude_last_metrics: Dict[int, Dict[str, float] | None],
) -> str:
    attn_top_recent = bool(attention_rankings and attention_rankings[0][0] in PERIOD_RECENT_LAG_BINS)
    occ_top_recent = bool(occlusion_rankings and occlusion_rankings[0][0] in PERIOD_RECENT_LAG_BINS)
    perm_top_recent = bool(permutation_rankings and permutation_rankings[0][0] in PERIOD_RECENT_LAG_BINS)

    full_ctd = full_metrics["ctd_mean"]
    recent_only = [metrics for k, metrics in only_last_metrics.items() if k in (1, 2) and metrics is not None]
    recent_exclude = [metrics for k, metrics in exclude_last_metrics.items() if k in (1, 2) and metrics is not None]
    older_exclude = [metrics for k, metrics in exclude_last_metrics.items() if k >= 3 and metrics is not None]

    recent_only_preserved = len(recent_only) > 0 and max(item["ctd_mean"] for item in recent_only) >= full_ctd - 0.05
    recent_exclusion_hurts = len(recent_exclude) > 0 and min(item["ctd_mean"] for item in recent_exclude) <= full_ctd - 0.05
    recent_exclusion_mild = len(recent_exclude) > 0 and max(item["ctd_mean"] for item in recent_exclude) >= full_ctd - 0.03
    older_bins_present = bool(
        (attention_rankings and attention_rankings[0][0] not in PERIOD_RECENT_LAG_BINS)
        or (occlusion_rankings and occlusion_rankings[0][0] not in PERIOD_RECENT_LAG_BINS)
        or (permutation_rankings and permutation_rankings[0][0] not in PERIOD_RECENT_LAG_BINS)
    )
    older_exclusion_not_worse = len(older_exclude) > 0 and max(item["ctd_mean"] for item in older_exclude) >= full_ctd - 0.05

    if attn_top_recent and occ_top_recent and perm_top_recent and recent_only_preserved and recent_exclusion_hurts:
        return "支持近期相邻时期主导：模型主要依赖 landmark 前最近 1 到 2 个时期。"
    if older_bins_present and not recent_only_preserved and recent_exclusion_mild and older_exclusion_not_worse:
        return "支持更长历史的阶段组合与滞后效应：只保留最近时期无法维持性能，而去掉最近 1 到 2 个时期后仍保留较多判别力。"
    return "证据混合：模型并非只看最近时期，但也没有完全摆脱相邻时期的影响，需要结合各窗口分别解释。"


def generate_period_final_report(
    output_dir: Path,
    artifact_dir: Path,
    window_specs: Sequence[Tuple[int, int]],
    attention_df: pd.DataFrame,
    occlusion_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> None:
    attention_summary = (
        attention_df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            enrichment_mean=("enrichment", "mean"),
        )
        .fillna(0.0)
    )
    occlusion_summary = (
        occlusion_df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(delta_risk_mean=("delta_risk_mean", "mean"))
        .fillna(0.0)
    )
    permutation_summary = (
        permutation_df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(delta_risk_mean=("delta_risk_mean", "mean"))
        .fillna(0.0)
    )
    history_summary = (
        history_df.groupby(["variant", "landmark", "pred_horizon"], as_index=False)
        .agg(
            ctd_mean=("ctd", "mean"),
            bstd_mean=("bstd", "mean"),
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
        )
        .fillna(0.0)
    )

    attention_rankings = get_top_rankings(attention_summary, "enrichment_mean", ["landmark", "pred_horizon"])
    occlusion_rankings = get_top_rankings(occlusion_summary, "delta_risk_mean", ["landmark", "pred_horizon"])
    permutation_rankings = get_top_rankings(permutation_summary, "delta_risk_mean", ["landmark", "pred_horizon"])

    report_lines = [
        "# period_ms 时期级环境时间关注实验报告",
        "",
        f"- 分析对象：`{artifact_dir}`",
        f"- 窗口：{', '.join(f'{landmark}->{horizon}' for landmark, horizon in window_specs)}",
        "",
    ]

    overall_full = history_df[history_df["variant"] == "full"][["ctd", "bstd"]].mean(numeric_only=True)
    overall_best_only_last = (
        history_df[history_df["history_mode"] == "only_last_k"]
        .groupby("variant")[["ctd", "bstd"]]
        .mean()
        .sort_values("ctd", ascending=False)
    )
    overall_best_exclude = (
        history_df[history_df["history_mode"] == "exclude_last_k"]
        .groupby("variant")[["ctd", "bstd"]]
        .mean()
        .sort_values("ctd", ascending=False)
    )
    if not overall_best_only_last.empty and not overall_best_exclude.empty:
        report_lines.extend(
            [
                "## Overall",
                "",
                f"- full: Ctd={overall_full['ctd']:.4f}, BStd={overall_full['bstd']:.4f}",
                f"- best only_last_k: {overall_best_only_last.index[0]} -> Ctd={overall_best_only_last.iloc[0]['ctd']:.4f}, BStd={overall_best_only_last.iloc[0]['bstd']:.4f}",
                f"- best exclude_last_k: {overall_best_exclude.index[0]} -> Ctd={overall_best_exclude.iloc[0]['ctd']:.4f}, BStd={overall_best_exclude.iloc[0]['bstd']:.4f}",
                "",
            ]
        )

    for landmark, horizon in window_specs:
        full_metrics = summarize_ablation_variant(history_summary, "full", landmark, horizon)
        if full_metrics is None:
            continue
        available_history_periods = sorted(int(value) for value in history_df["history_periods"].dropna().unique())
        only_last_metrics = {
            history_periods: summarize_ablation_variant(history_summary, f"only_last_k_{history_periods}", landmark, horizon)
            for history_periods in available_history_periods
        }
        exclude_last_metrics = {
            history_periods: summarize_ablation_variant(history_summary, f"exclude_last_k_{history_periods}", landmark, horizon)
            for history_periods in available_history_periods
        }
        attn_ranking = attention_rankings.get((landmark, horizon), [])
        occ_ranking = occlusion_rankings.get((landmark, horizon), [])
        perm_ranking = permutation_rankings.get((landmark, horizon), [])
        conclusion = determine_window_conclusion_period(
            attention_rankings=attn_ranking,
            occlusion_rankings=occ_ranking,
            permutation_rankings=perm_ranking,
            full_metrics=full_metrics,
            only_last_metrics=only_last_metrics,
            exclude_last_metrics=exclude_last_metrics,
        )

        report_lines.extend(
            [
                f"## Window {landmark}->{horizon}",
                "",
                f"- 风险集规模：eligible_n={full_metrics['eligible_n_mean']:.1f}, future_event_n={full_metrics['future_event_n_mean']:.1f}",
                f"- attention enrichment 排名：{', '.join(f'{label}({value:.3f})' for label, value in attn_ranking)}",
                f"- occlusion Δrisk 排名：{', '.join(f'{label}({value:.4f})' for label, value in occ_ranking)}",
                f"- permutation Δrisk 排名：{', '.join(f'{label}({value:.4f})' for label, value in perm_ranking)}",
                f"- full：Ctd={full_metrics['ctd_mean']:.4f}, BStd={full_metrics['bstd_mean']:.4f}",
                "- only_last_k："
                + ", ".join(
                    f"k={int(k)} -> Ctd={metrics['ctd_mean']:.4f}, BStd={metrics['bstd_mean']:.4f}"
                    for k, metrics in only_last_metrics.items()
                    if metrics is not None
                ),
                "- exclude_last_k："
                + ", ".join(
                    f"k={int(k)} -> Ctd={metrics['ctd_mean']:.4f}, BStd={metrics['bstd_mean']:.4f}"
                    for k, metrics in exclude_last_metrics.items()
                    if metrics is not None
                ),
                f"- 结论：{conclusion}",
                "",
            ]
        )

    (output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def get_top_rankings(
    df: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str],
) -> Dict[Tuple[int, int], List[Tuple[str, float]]]:
    rankings: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
    for key, group in df.groupby(list(group_columns)):
        ordered = group.sort_values(value_column, ascending=False)
        rankings[key] = [
            (str(row["lag_bin"]), float(row[value_column]))
            for _, row in ordered.iterrows()
        ]
    return rankings


def summarize_ablation_variant(history_summary: pd.DataFrame, variant: str, landmark: int, horizon: int) -> Dict[str, float] | None:
    subset = history_summary[
        (history_summary["variant"] == variant)
        & (history_summary["landmark"] == landmark)
        & (history_summary["pred_horizon"] == horizon)
    ]
    if subset.empty:
        return None
    row = subset.iloc[0]
    return {
        "ctd_mean": float(row["ctd_mean"]),
        "bstd_mean": float(row["bstd_mean"]),
        "eligible_n_mean": float(row.get("eligible_n_mean", 0.0)),
        "future_event_n_mean": float(row.get("future_event_n_mean", 0.0)),
    }


def determine_window_conclusion(
    attention_rankings: List[Tuple[str, float]],
    occlusion_rankings: List[Tuple[str, float]],
    permutation_rankings: List[Tuple[str, float]],
    full_metrics: Dict[str, float],
    only_last_metrics: Dict[int, Dict[str, float]],
    exclude_last_metrics: Dict[int, Dict[str, float]],
) -> str:
    attn_top_recent = bool(attention_rankings and attention_rankings[0][0] in RECENT_LAG_BINS)
    occ_top_recent = bool(occlusion_rankings and occlusion_rankings[0][0] in RECENT_LAG_BINS)
    perm_top_recent = bool(permutation_rankings and permutation_rankings[0][0] in RECENT_LAG_BINS)

    full_ctd = full_metrics["ctd_mean"]
    full_bstd = full_metrics["bstd_mean"]
    recent_only = [
        metrics for k, metrics in only_last_metrics.items()
        if k in (2, 7) and metrics is not None
    ]
    recent_exclude = [
        metrics for k, metrics in exclude_last_metrics.items()
        if k in (2, 7) and metrics is not None
    ]
    longer_exclude = [
        metrics for k, metrics in exclude_last_metrics.items()
        if k in (14, 30) and metrics is not None
    ]

    recent_only_preserved = (
        len(recent_only) > 0
        and max(metric["ctd_mean"] for metric in recent_only) >= full_ctd - 0.05
        and min(metric["bstd_mean"] for metric in recent_only) <= full_bstd + 0.05
    )
    recent_exclusion_hurts = (
        len(recent_exclude) > 0
        and min(metric["ctd_mean"] for metric in recent_exclude) <= full_ctd - 0.05
    )
    recent_exclusion_mild = (
        len(recent_exclude) > 0
        and max(metric["ctd_mean"] for metric in recent_exclude) >= full_ctd - 0.03
    )
    longer_bins_present = bool(
        occlusion_rankings and occlusion_rankings[0][0] not in RECENT_LAG_BINS
        or permutation_rankings and permutation_rankings[0][0] not in RECENT_LAG_BINS
        or attention_rankings and attention_rankings[0][0] not in RECENT_LAG_BINS
    )
    longer_exclusion_not_worse = (
        len(longer_exclude) > 0
        and max(metric["ctd_mean"] for metric in longer_exclude) >= full_ctd - 0.05
    )

    if attn_top_recent and occ_top_recent and perm_top_recent and recent_only_preserved and recent_exclusion_hurts:
        return "支持近期环境主导：最后2天/7天在注意力、遮挡、置乱三类实验中都更敏感，且仅保留近期历史时性能未明显塌陷。"
    if longer_bins_present and not recent_only_preserved and recent_exclusion_mild and longer_exclusion_not_worse:
        return "支持连续多日组合与滞后效应：模型对更早历史也敏感，且只保留最近2/7天无法维持性能，删除最近2/7天后仍能保留较多判别力。"
    return "证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。"


def generate_final_report(
    output_dir: Path,
    artifact_dir: Path,
    window_specs: Sequence[Tuple[int, int]],
    attention_df: pd.DataFrame,
    occlusion_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> None:
    attention_summary = (
        attention_df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
            enrichment_mean=("enrichment", "mean"),
        )
        .fillna(0.0)
    )
    occlusion_summary = (
        occlusion_df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(delta_risk_mean=("delta_risk_mean", "mean"))
        .fillna(0.0)
    )
    permutation_summary = (
        permutation_df.groupby(["landmark", "pred_horizon", "lag_bin"], as_index=False)
        .agg(delta_risk_mean=("delta_risk_mean", "mean"))
        .fillna(0.0)
    )
    history_summary = (
        history_df.groupby(["variant", "landmark", "pred_horizon"], as_index=False)
        .agg(
            ctd_mean=("ctd", "mean"),
            bstd_mean=("bstd", "mean"),
            eligible_n_mean=("eligible_n", "mean"),
            future_event_n_mean=("future_event_n", "mean"),
        )
        .fillna(0.0)
    )

    attention_rankings = get_top_rankings(attention_summary, "enrichment_mean", ["landmark", "pred_horizon"])
    occlusion_rankings = get_top_rankings(occlusion_summary, "delta_risk_mean", ["landmark", "pred_horizon"])
    permutation_rankings = get_top_rankings(permutation_summary, "delta_risk_mean", ["landmark", "pred_horizon"])

    report_lines = [
        "# 环境时间关注实验报告",
        "",
        f"- 分析对象：`{artifact_dir}`",
        f"- 窗口：{', '.join(f'{landmark}->{horizon}' for landmark, horizon in window_specs)}",
        "",
    ]

    overall_full = history_df[history_df["variant"] == "full"][["ctd", "bstd"]].mean(numeric_only=True)
    overall_best_only_last = (
        history_df[history_df["history_mode"] == "only_last_k"]
        .groupby("variant")[["ctd", "bstd"]]
        .mean()
        .sort_values("ctd", ascending=False)
    )
    overall_best_exclude = (
        history_df[history_df["history_mode"] == "exclude_last_k"]
        .groupby("variant")[["ctd", "bstd"]]
        .mean()
        .sort_values("ctd", ascending=False)
    )
    if not overall_best_only_last.empty and not overall_best_exclude.empty:
        report_lines.extend(
            [
                "## Overall",
                "",
                f"- `full` 平均 `Ctd={overall_full['ctd']:.4f}`，`BStd={overall_full['bstd']:.4f}`",
                f"- 最佳 `only_last_k` 变体：`{overall_best_only_last.index[0]}`，平均 `Ctd={overall_best_only_last.iloc[0]['ctd']:.4f}`，`BStd={overall_best_only_last.iloc[0]['bstd']:.4f}`",
                f"- 最佳 `exclude_last_k` 变体：`{overall_best_exclude.index[0]}`，平均 `Ctd={overall_best_exclude.iloc[0]['ctd']:.4f}`，`BStd={overall_best_exclude.iloc[0]['bstd']:.4f}`",
                "",
            ]
        )

    for landmark, horizon in window_specs:
        key = (landmark, horizon)
        attn_ranking = attention_rankings.get(key, [])
        occ_ranking = occlusion_rankings.get(key, [])
        perm_ranking = permutation_rankings.get(key, [])

        full_metrics = summarize_ablation_variant(history_summary, "full", landmark, horizon)
        if full_metrics is None:
            continue
        only_last_metrics = {
            history_days: summarize_ablation_variant(history_summary, f"only_last_k_{history_days}", landmark, horizon)
            for history_days in (2, 7, 14, 30)
        }
        exclude_last_metrics = {
            history_days: summarize_ablation_variant(history_summary, f"exclude_last_k_{history_days}", landmark, horizon)
            for history_days in (2, 7, 14, 30)
        }
        conclusion = determine_window_conclusion(
            attention_rankings=attn_ranking,
            occlusion_rankings=occ_ranking,
            permutation_rankings=perm_ranking,
            full_metrics=full_metrics,
            only_last_metrics=only_last_metrics,
            exclude_last_metrics=exclude_last_metrics,
        )

        window_meta = history_summary[
            (history_summary["variant"] == "full")
            & (history_summary["landmark"] == landmark)
            & (history_summary["pred_horizon"] == horizon)
        ].iloc[0]
        report_lines.extend(
            [
                f"## Window {landmark}->{horizon}",
                "",
                f"- 风险集规模：`eligible_n≈{window_meta['eligible_n_mean']:.1f}`，未来事件数：`future_event_n≈{window_meta['future_event_n_mean']:.1f}`",
                f"- Attention enrichment 排名：{', '.join(f'{label}({value:.3f})' for label, value in attn_ranking)}",
                f"- 遮挡 Δrisk 排名：{', '.join(f'{label}({value:.4f})' for label, value in occ_ranking)}",
                f"- 置乱 Δrisk 排名：{', '.join(f'{label}({value:.4f})' for label, value in perm_ranking)}",
                f"- `full`：`Ctd={full_metrics['ctd_mean']:.4f}`，`BStd={full_metrics['bstd_mean']:.4f}`",
                "- `only_last_k`："
                + ", ".join(
                    f"`k={k}` -> Ctd={metrics['ctd_mean']:.4f}, BStd={metrics['bstd_mean']:.4f}"
                    for k, metrics in only_last_metrics.items()
                    if metrics is not None
                ),
                "- `exclude_last_k`："
                + ", ".join(
                    f"`k={k}` -> Ctd={metrics['ctd_mean']:.4f}, BStd={metrics['bstd_mean']:.4f}"
                    for k, metrics in exclude_last_metrics.items()
                    if metrics is not None
                ),
                f"- 结论：{conclusion}",
                "",
            ]
        )

    final_report_path = output_dir / "final_report.md"
    final_report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    artifact_config = get_artifact_config(artifact_dir)
    analysis_unit = resolve_analysis_unit(artifact_config, args)
    tree_data = load_dynamic_hlb_dataset(
        data_path=args.data_path,
        use_ct_aux_task=bool(artifact_config.get("use_ct_aux_task", False)),
        use_agro_features=bool(artifact_config.get("use_agro_features", False)),
        period_feature_mode=str(artifact_config.get("period_feature_mode", "full")),
    )
    window_specs = build_window_specs(args.window_pairs, tree_data.num_periods)
    print(f"Artifact config loaded from {artifact_dir}")
    print(f"Analysis unit: {analysis_unit}")
    print(f"Windows: {window_specs}")
    print(f"History days: {sorted(set(args.history_days))}")
    print(f"History periods: {sorted(set(args.history_periods))}")
    print(f"Permutation repeats: {args.perm_repeats}")
    print(f"Retrain repeats: {args.retrain_repeats}")

    start_time = time.perf_counter()
    if analysis_unit == "period":
        attention_df = run_period_attention_analysis(
            args=args,
            artifact_dir=artifact_dir,
            config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        occlusion_df = run_period_occlusion_analysis(
            args=args,
            artifact_dir=artifact_dir,
            config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        permutation_df = run_period_permutation_analysis(
            args=args,
            artifact_dir=artifact_dir,
            config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        history_df = run_period_history_ablation(
            args=args,
            artifact_dir=artifact_dir,
            artifact_config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        generate_period_final_report(
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            window_specs=window_specs,
            attention_df=attention_df,
            occlusion_df=occlusion_df,
            permutation_df=permutation_df,
            history_df=history_df,
        )
    else:
        attention_df = run_attention_analysis(
            args=args,
            artifact_dir=artifact_dir,
            config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        occlusion_df = run_occlusion_analysis(
            args=args,
            artifact_dir=artifact_dir,
            config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        permutation_df = run_permutation_analysis(
            args=args,
            artifact_dir=artifact_dir,
            config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        history_df = run_history_ablation(
            args=args,
            artifact_dir=artifact_dir,
            artifact_config=artifact_config,
            tree_data=tree_data,
            window_specs=window_specs,
            device=device,
            output_dir=output_dir,
        )
        generate_final_report(
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            window_specs=window_specs,
            attention_df=attention_df,
            occlusion_df=occlusion_df,
            permutation_df=permutation_df,
            history_df=history_df,
        )

    elapsed = time.perf_counter() - start_time
    print(f"Finished environment time-focus experiment in {elapsed:.2f}s")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
