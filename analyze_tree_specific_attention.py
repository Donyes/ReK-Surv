"""
Tree-specific period-attention diagnostics for dynamic HLB survival models.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from models import DynamicReKSurv
from utils import (
    fit_static_preprocessor,
    load_dynamic_hlb_dataset,
    split_tree_indices,
    transform_static_features,
    weighted_brier_score,
    weighted_c_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tree-specific period attention.")
    parser.add_argument("--artifact_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--data_path", type=str, default="data/hlb_dataset.xlsx")
    parser.add_argument(
        "--window_pairs",
        type=str,
        nargs="+",
        default=["3:3", "4:3", "6:3", "6:6", "9:3", "6:1", "9:1", "3:2", "3:8"],
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/tree_specific_attention_analysis")
    parser.add_argument("--random_init_repeats", type=int, default=50)
    parser.add_argument("--static_shuffle_repeats", type=int, default=10)
    parser.add_argument("--max_repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_window_specs(window_pairs: Sequence[str], num_periods: int) -> List[Tuple[int, int]]:
    specs = []
    for pair in window_pairs:
        landmark_text, horizon_text = pair.split(":", maxsplit=1)
        landmark = int(landmark_text)
        horizon = int(horizon_text)
        if landmark < 0 or landmark >= num_periods or horizon <= 0 or landmark + horizon > num_periods:
            raise ValueError(f"Invalid window pair: {pair}")
        specs.append((landmark, horizon))
    return sorted(set(specs))


def load_config(artifact_dir: Path) -> Dict:
    with open(artifact_dir / "summary.json", "r", encoding="utf-8") as file:
        return json.load(file)["config"]


def available_repeats(artifact_dir: Path, max_repeats: int | None) -> List[int]:
    repeats = []
    for path in sorted(artifact_dir.glob("repeat_*")):
        suffix = path.name.split("_")[-1]
        if suffix.isdigit():
            repeats.append(int(suffix))
    if max_repeats is not None:
        repeats = repeats[:max_repeats]
    if not repeats:
        raise ValueError(f"No repeat directories found in {artifact_dir}")
    return repeats


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
        env_aux_mode=str(config.get("env_aux_mode", "none")),
        tree_attention_dropout=float(config.get("tree_attention_dropout", 0.10)),
        static_attention_dim=int(config.get("static_attention_dim", 32)),
        ct_delta_output_dim=int(config.get("ct_delta_output_dim", 1)),
    )


def reconstruct_repeat(tree_data, config: Dict, repeat_index: int) -> Dict:
    repeat_seed = int(config["seed"]) + repeat_index * 1000
    train_indices, val_indices, test_indices = split_tree_indices(
        tree_data=tree_data,
        test_size=float(config["test_size"]),
        val_ratio=float(config["val_ratio"]),
        random_state=repeat_seed,
    )
    preprocessor = fit_static_preprocessor(
        tree_data,
        train_indices,
        use_tree_id_spatial=bool(config.get("use_tree_id_spatial", False)),
    )
    static_x, static_names = transform_static_features(tree_data, preprocessor)
    return {
        "seed": repeat_seed,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "static_x": static_x,
        "static_names": static_names,
    }


def load_model(artifact_dir: Path, repeat_index: int, tree_data, static_x: np.ndarray, config: Dict) -> DynamicReKSurv:
    model = instantiate_model(tree_data, static_x, config)
    state_dict = torch.load(artifact_dir / f"repeat_{repeat_index}" / "best_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def period_range(tree_data, period: int) -> str:
    if period == 1:
        start = pd.Timestamp(tree_data.period_bounds[0])
    else:
        start = pd.Timestamp(tree_data.period_end_dates[period - 2]) + pd.Timedelta(days=1)
    end = pd.Timestamp(tree_data.period_end_dates[period - 1])
    return f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}"


def model_forward(model, tree_data, static_x: np.ndarray, indices: np.ndarray, landmark: int) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        return model(
            daily_env_prefix=torch.tensor(tree_data.daily_env[indices], dtype=torch.float32),
            period_env_prefix=torch.tensor(tree_data.period_env[indices], dtype=torch.float32),
            static_x=torch.tensor(static_x[indices], dtype=torch.float32),
            seq_len_days=torch.full((len(indices),), int(tree_data.landmark_seq_len_days[landmark]), dtype=torch.long),
            seq_len_periods=torch.full((len(indices),), int(tree_data.landmark_seq_len_periods[landmark]), dtype=torch.long),
            landmark_period=torch.full((len(indices),), int(landmark), dtype=torch.long),
        )


def risk_from_output(output: Dict[str, torch.Tensor], landmark: int, horizon: int) -> np.ndarray:
    return output["event_probs"][:, landmark : landmark + horizon].sum(dim=1).detach().cpu().numpy()


def pairwise_js_divergence(attention: np.ndarray) -> Tuple[float, float, float]:
    if len(attention) <= 1:
        return 0.0, 0.0, 0.0
    values = []
    eps = 1e-8
    attention = np.asarray(attention, dtype=np.float64)
    attention = attention / np.clip(attention.sum(axis=1, keepdims=True), eps, None)
    for i in range(len(attention)):
        for j in range(i + 1, len(attention)):
            p = np.clip(attention[i], eps, 1.0)
            q = np.clip(attention[j], eps, 1.0)
            m = 0.5 * (p + q)
            js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            values.append(float(js))
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.max())


def raw_static_rows(tree_data, static_names: Sequence[str], static_x: np.ndarray, indices: np.ndarray) -> List[Dict]:
    rows = []
    for tree_index in indices:
        row = {
            "tree_age": float(tree_data.static_cont[tree_index, 0]),
            "tree_vigor": float(tree_data.static_cont[tree_index, 1]),
            "east_west_canopy": float(tree_data.static_cont[tree_index, 2]),
            "north_south_canopy": float(tree_data.static_cont[tree_index, 3]),
            "trunk_height": float(tree_data.static_cont[tree_index, 4]),
            "tree_height": float(tree_data.static_cont[tree_index, 5]),
            "canopy_area_proxy": float(tree_data.static_cont[tree_index, 2] * tree_data.static_cont[tree_index, 3]),
            "variety": int(tree_data.variety[tree_index]),
            "has_diseased_neighbor": int(tree_data.has_diseased_neighbor[tree_index]),
        }
        for name, value in zip(static_names, static_x[tree_index]):
            row[f"static_x_{name}"] = float(value)
        rows.append(row)
    return rows


def top_period_fields(attention: np.ndarray, landmark: int, tree_data) -> Dict:
    valid_periods = list(range(1, landmark + 1))
    ordered = sorted(valid_periods, key=lambda period: float(attention[period - 1]), reverse=True)
    fields = {}
    for rank, period in enumerate(ordered[:3], start=1):
        fields[f"top{rank}_period"] = int(period)
        fields[f"top{rank}_weight"] = float(attention[period - 1])
        fields[f"top{rank}_range"] = period_range(tree_data, period)
    return fields


def collect_attention_outputs(artifact_label: str, artifact_dir: Path, tree_data, config: Dict, window_specs, repeats):
    sample_rows = []
    diversity_rows = []
    top_distribution_rows = []
    for repeat_index in repeats:
        repeat_state = reconstruct_repeat(tree_data, config, repeat_index)
        static_x = repeat_state["static_x"]
        static_names = repeat_state["static_names"]
        train_indices = repeat_state["train_indices"]
        test_indices = repeat_state["test_indices"]
        model = load_model(artifact_dir, repeat_index, tree_data, static_x, config)

        for landmark, horizon in window_specs:
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue
            output = model_forward(model, tree_data, static_x, eligible_indices, landmark)
            attention = output["period_attention_weights"].detach().cpu().numpy()
            risk = risk_from_output(output, landmark, horizon)
            train_times = tree_data.time_period[train_indices]
            train_events = tree_data.event_flag[train_indices]
            ctd = weighted_c_index(
                train_times=train_times,
                train_events=train_events,
                prediction=risk,
                test_times=tree_data.time_period[eligible_indices],
                test_events=tree_data.event_flag[eligible_indices],
                horizon=landmark + horizon,
            )
            bstd = weighted_brier_score(
                train_times=train_times,
                train_events=train_events,
                prediction=risk,
                test_times=tree_data.time_period[eligible_indices],
                test_events=tree_data.event_flag[eligible_indices],
                horizon=landmark + horizon,
            )
            history_attention = attention[:, :landmark]
            js_mean, js_median, js_max = pairwise_js_divergence(history_attention)
            rounded_unique = int(np.unique(np.round(history_attention, 6), axis=0).shape[0])
            diversity_rows.append(
                {
                    "artifact": artifact_label,
                    "repeat": repeat_index,
                    "landmark": landmark,
                    "pred_horizon": horizon,
                    "eligible_n": int(len(eligible_indices)),
                    "future_event_n": int(
                        (
                            (tree_data.event_flag[eligible_indices] == 1)
                            & (tree_data.time_period[eligible_indices] <= landmark + horizon)
                        ).sum()
                    ),
                    "unique_attention_rows_rounded_6dp": rounded_unique,
                    "pairwise_js_mean": js_mean,
                    "pairwise_js_median": js_median,
                    "pairwise_js_max": js_max,
                    "ctd": ctd,
                    "bstd": bstd,
                }
            )

            top_counts = Counter(np.argmax(history_attention, axis=1) + 1)
            for period, count in sorted(top_counts.items()):
                top_distribution_rows.append(
                    {
                        "artifact": artifact_label,
                        "repeat": repeat_index,
                        "landmark": landmark,
                        "pred_horizon": horizon,
                        "top_period": int(period),
                        "top_period_range": period_range(tree_data, int(period)),
                        "count": int(count),
                        "fraction": float(count / len(eligible_indices)),
                    }
                )

            raw_rows = raw_static_rows(tree_data, static_names, static_x, eligible_indices)
            for local_index, tree_index in enumerate(eligible_indices):
                row = {
                    "artifact": artifact_label,
                    "repeat": repeat_index,
                    "landmark": landmark,
                    "pred_horizon": horizon,
                    "tree_id": str(tree_data.tree_ids[tree_index]),
                    "event_flag": int(tree_data.event_flag[tree_index]),
                    "event_period": int(tree_data.time_period[tree_index]),
                    "event_date": ""
                    if pd.isna(tree_data.event_dates[tree_index])
                    else pd.Timestamp(tree_data.event_dates[tree_index]).strftime("%Y-%m-%d"),
                    "risk": float(risk[local_index]),
                    **raw_rows[local_index],
                    **top_period_fields(attention[local_index], landmark, tree_data),
                }
                for period in range(1, tree_data.num_periods + 1):
                    row[f"attn_P{period}"] = float(attention[local_index, period - 1])
                sample_rows.append(row)
    return pd.DataFrame(sample_rows), pd.DataFrame(diversity_rows), pd.DataFrame(top_distribution_rows)


def add_quantile_group(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    try:
        return pd.qcut(frame[column], q=3, labels=[f"{label}_low", f"{label}_mid", f"{label}_high"], duplicates="drop")
    except ValueError:
        return pd.Series([f"{label}_all"] * len(frame), index=frame.index)


def summarize_static_groups(sample_df: pd.DataFrame, tree_data) -> pd.DataFrame:
    rows = []
    if sample_df.empty:
        return pd.DataFrame(rows)
    working = sample_df.copy()
    working["age_group"] = add_quantile_group(working, "tree_age", "age")
    working["canopy_group"] = add_quantile_group(working, "canopy_area_proxy", "canopy")
    working["height_group"] = add_quantile_group(working, "tree_height", "height")
    group_specs = [
        ("age_group", "age_group"),
        ("tree_vigor", "vigor"),
        ("variety", "variety"),
        ("has_diseased_neighbor", "neighbor"),
        ("canopy_group", "canopy_group"),
        ("height_group", "height_group"),
    ]
    attention_columns = [f"attn_P{period}" for period in range(1, tree_data.num_periods + 1)]
    for (artifact, landmark, horizon), base_group in working.groupby(["artifact", "landmark", "pred_horizon"]):
        for column, group_type in group_specs:
            for group_value, group in base_group.groupby(column, dropna=False, observed=False):
                if len(group) == 0:
                    continue
                means = group[attention_columns].mean()
                valid_periods = list(range(1, int(landmark) + 1))
                ordered = sorted(valid_periods, key=lambda period: float(means[f"attn_P{period}"]), reverse=True)
                rows.append(
                    {
                        "artifact": artifact,
                        "landmark": int(landmark),
                        "pred_horizon": int(horizon),
                        "group_type": group_type,
                        "group_value": str(group_value),
                        "n": int(len(group)),
                        "risk_mean": float(group["risk"].mean()),
                        "top1_period": int(ordered[0]),
                        "top1_range": period_range(tree_data, int(ordered[0])),
                        "top1_weight_mean": float(means[f"attn_P{ordered[0]}"]),
                        "top2_period": int(ordered[1]) if len(ordered) > 1 else None,
                        "top2_range": period_range(tree_data, int(ordered[1])) if len(ordered) > 1 else None,
                        "top2_weight_mean": float(means[f"attn_P{ordered[1]}"]) if len(ordered) > 1 else None,
                    }
                )
    return pd.DataFrame(rows)


def run_static_shuffle(artifact_label: str, artifact_dir: Path, tree_data, config: Dict, window_specs, repeats, shuffle_repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for repeat_index in repeats:
        repeat_state = reconstruct_repeat(tree_data, config, repeat_index)
        static_x = repeat_state["static_x"]
        train_indices = repeat_state["train_indices"]
        test_indices = repeat_state["test_indices"]
        model = load_model(artifact_dir, repeat_index, tree_data, static_x, config)
        for landmark, horizon in window_specs:
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue
            base_output = model_forward(model, tree_data, static_x, eligible_indices, landmark)
            base_risk = risk_from_output(base_output, landmark, horizon)
            train_times = tree_data.time_period[train_indices]
            train_events = tree_data.event_flag[train_indices]
            for shuffle_index in range(shuffle_repeats):
                shuffled_indices = np.asarray(eligible_indices).copy()
                rng.shuffle(shuffled_indices)
                shuffled_static_x = static_x.copy()
                shuffled_static_x[eligible_indices] = static_x[shuffled_indices]
                shuffled_output = model_forward(model, tree_data, shuffled_static_x, eligible_indices, landmark)
                shuffled_risk = risk_from_output(shuffled_output, landmark, horizon)
                ctd = weighted_c_index(
                    train_times=train_times,
                    train_events=train_events,
                    prediction=shuffled_risk,
                    test_times=tree_data.time_period[eligible_indices],
                    test_events=tree_data.event_flag[eligible_indices],
                    horizon=landmark + horizon,
                )
                bstd = weighted_brier_score(
                    train_times=train_times,
                    train_events=train_events,
                    prediction=shuffled_risk,
                    test_times=tree_data.time_period[eligible_indices],
                    test_events=tree_data.event_flag[eligible_indices],
                    horizon=landmark + horizon,
                )
                attn = shuffled_output["period_attention_weights"].detach().cpu().numpy()[:, :landmark]
                js_mean, _, _ = pairwise_js_divergence(attn)
                rows.append(
                    {
                        "artifact": artifact_label,
                        "repeat": repeat_index,
                        "landmark": landmark,
                        "pred_horizon": horizon,
                        "shuffle_index": shuffle_index,
                        "base_risk_mean": float(base_risk.mean()),
                        "shuffled_risk_mean": float(shuffled_risk.mean()),
                        "mean_abs_risk_delta": float(np.abs(shuffled_risk - base_risk).mean()),
                        "ctd": ctd,
                        "bstd": bstd,
                        "pairwise_js_mean": js_mean,
                    }
                )
    return pd.DataFrame(rows)


def run_random_init(artifact_label: str, tree_data, config: Dict, window_specs, repeats, random_repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    for repeat_index in repeats[:1]:
        repeat_state = reconstruct_repeat(tree_data, config, repeat_index)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        for init_index in range(random_repeats):
            set_seed(seed + init_index)
            model = instantiate_model(tree_data, static_x, config)
            model.eval()
            for landmark, horizon in window_specs:
                eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
                if len(eligible_indices) == 0:
                    continue
                output = model_forward(model, tree_data, static_x, eligible_indices, landmark)
                attn = output["period_attention_weights"].detach().cpu().numpy()[:, :landmark]
                js_mean, js_median, js_max = pairwise_js_divergence(attn)
                rows.append(
                    {
                        "artifact": artifact_label,
                        "repeat": repeat_index,
                        "landmark": landmark,
                        "pred_horizon": horizon,
                        "init_index": init_index,
                        "unique_attention_rows_rounded_6dp": int(np.unique(np.round(attn, 6), axis=0).shape[0]),
                        "pairwise_js_mean": js_mean,
                        "pairwise_js_median": js_median,
                        "pairwise_js_max": js_max,
                    }
                )
    return pd.DataFrame(rows)


def run_static_counterfactual(artifact_label: str, artifact_dir: Path, tree_data, config: Dict, window_specs, repeats) -> pd.DataFrame:
    rows = []
    for repeat_index in repeats:
        repeat_state = reconstruct_repeat(tree_data, config, repeat_index)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_model(artifact_dir, repeat_index, tree_data, static_x, config)
        for landmark, horizon in window_specs:
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) < 2:
                continue
            output = model_forward(model, tree_data, static_x, eligible_indices, landmark)
            risk = risk_from_output(output, landmark, horizon)
            order = np.argsort(risk)
            low_local = int(order[0])
            high_local = int(order[-1])
            pair_indices = np.asarray([eligible_indices[low_local], eligible_indices[high_local]], dtype=np.int64)
            swapped_static_x = static_x.copy()
            swapped_static_x[pair_indices[0]] = static_x[pair_indices[1]]
            swapped_static_x[pair_indices[1]] = static_x[pair_indices[0]]
            swapped_output = model_forward(model, tree_data, swapped_static_x, pair_indices, landmark)
            swapped_risk = risk_from_output(swapped_output, landmark, horizon)
            swapped_attention = swapped_output["period_attention_weights"].detach().cpu().numpy()[:, :landmark]
            base_attention = output["period_attention_weights"].detach().cpu().numpy()[[low_local, high_local], :landmark]
            for local_pair_pos, label in enumerate(["low_to_high_static", "high_to_low_static"]):
                tree_index = int(pair_indices[local_pair_pos])
                rows.append(
                    {
                        "artifact": artifact_label,
                        "repeat": repeat_index,
                        "landmark": landmark,
                        "pred_horizon": horizon,
                        "swap_case": label,
                        "tree_id": str(tree_data.tree_ids[tree_index]),
                        "base_risk": float(risk[[low_local, high_local][local_pair_pos]]),
                        "swapped_risk": float(swapped_risk[local_pair_pos]),
                        "risk_delta": float(swapped_risk[local_pair_pos] - risk[[low_local, high_local][local_pair_pos]]),
                        "attention_l1_delta": float(np.abs(swapped_attention[local_pair_pos] - base_attention[local_pair_pos]).sum()),
                        **top_period_fields(swapped_attention[local_pair_pos], landmark, tree_data),
                    }
                )
    return pd.DataFrame(rows)


def run_period_occlusion(artifact_label: str, artifact_dir: Path, tree_data, config: Dict, window_specs, repeats) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for repeat_index in repeats:
        repeat_state = reconstruct_repeat(tree_data, config, repeat_index)
        static_x = repeat_state["static_x"]
        test_indices = repeat_state["test_indices"]
        model = load_model(artifact_dir, repeat_index, tree_data, static_x, config)
        for landmark, horizon in window_specs:
            eligible_indices = test_indices[tree_data.time_period[test_indices] > landmark]
            if len(eligible_indices) == 0:
                continue
            base_output = model_forward(model, tree_data, static_x, eligible_indices, landmark)
            base_risk = risk_from_output(base_output, landmark, horizon)
            base_attention = base_output["period_attention_weights"].detach().cpu().numpy()
            for period in range(1, landmark + 1):
                masked_period_env = tree_data.period_env[eligible_indices].copy()
                masked_period_env[:, period - 1, :] = 0.0
                with torch.no_grad():
                    masked_output = model(
                        daily_env_prefix=torch.tensor(tree_data.daily_env[eligible_indices], dtype=torch.float32),
                        period_env_prefix=torch.tensor(masked_period_env, dtype=torch.float32),
                        static_x=torch.tensor(static_x[eligible_indices], dtype=torch.float32),
                        seq_len_days=torch.full((len(eligible_indices),), int(tree_data.landmark_seq_len_days[landmark]), dtype=torch.long),
                        seq_len_periods=torch.full((len(eligible_indices),), int(tree_data.landmark_seq_len_periods[landmark]), dtype=torch.long),
                        landmark_period=torch.full((len(eligible_indices),), int(landmark), dtype=torch.long),
                    )
                masked_risk = risk_from_output(masked_output, landmark, horizon)
                delta = np.abs(masked_risk - base_risk)
                for local_index, tree_index in enumerate(eligible_indices):
                    rows.append(
                        {
                            "artifact": artifact_label,
                            "repeat": repeat_index,
                            "landmark": landmark,
                            "pred_horizon": horizon,
                            "tree_id": str(tree_data.tree_ids[tree_index]),
                            "period": period,
                            "period_range": period_range(tree_data, period),
                            "attention_weight": float(base_attention[local_index, period - 1]),
                            "delta_risk": float(delta[local_index]),
                        }
                    )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()
    summary_df = (
        detail_df.groupby(["artifact", "landmark", "pred_horizon", "period", "period_range"], as_index=False)
        .agg(delta_risk_mean=("delta_risk", "mean"), attention_weight_mean=("attention_weight", "mean"))
        .sort_values(["artifact", "landmark", "pred_horizon", "delta_risk_mean"], ascending=[True, True, True, False])
    )
    return detail_df, summary_df


def write_report(output_dir: Path, sample_df: pd.DataFrame, diversity_df: pd.DataFrame, random_df: pd.DataFrame, shuffle_df: pd.DataFrame) -> None:
    lines = ["# Tree-specific period attention diagnostics", ""]
    if not diversity_df.empty:
        div_summary = (
            diversity_df.groupby(["artifact", "landmark", "pred_horizon"], as_index=False)
            .agg(
                eligible_n=("eligible_n", "mean"),
                future_event_n=("future_event_n", "mean"),
                ctd=("ctd", "mean"),
                bstd=("bstd", "mean"),
                pairwise_js_mean=("pairwise_js_mean", "mean"),
                unique_attention_rows=("unique_attention_rows_rounded_6dp", "mean"),
            )
        )
        lines.append("## Trained attention diversity")
        for row in div_summary.itertuples(index=False):
            lines.append(
                f"- {row.artifact} {row.landmark}->{row.pred_horizon}: "
                f"Ctd={row.ctd:.4f}, BStd={row.bstd:.4f}, "
                f"eligible={row.eligible_n:.1f}, events={row.future_event_n:.1f}, "
                f"JS={row.pairwise_js_mean:.6f}, unique_rows={row.unique_attention_rows:.1f}"
            )
        lines.append("")
    if not random_df.empty:
        rand_summary = random_df.groupby(["artifact", "landmark", "pred_horizon"], as_index=False).agg(
            random_js=("pairwise_js_mean", "mean")
        )
        lines.append("## Random-init control")
        for row in rand_summary.itertuples(index=False):
            lines.append(f"- {row.artifact} {row.landmark}->{row.pred_horizon}: random JS={row.random_js:.6f}")
        lines.append("")
    if not shuffle_df.empty:
        shuffle_summary = shuffle_df.groupby(["artifact", "landmark", "pred_horizon"], as_index=False).agg(
            shuffled_ctd=("ctd", "mean"),
            shuffled_bstd=("bstd", "mean"),
            mean_abs_risk_delta=("mean_abs_risk_delta", "mean"),
        )
        lines.append("## Static-shuffle control")
        for row in shuffle_summary.itertuples(index=False):
            lines.append(
                f"- {row.artifact} {row.landmark}->{row.pred_horizon}: "
                f"shuffled Ctd={row.shuffled_ctd:.4f}, BStd={row.shuffled_bstd:.4f}, "
                f"|delta risk|={row.mean_abs_risk_delta:.4f}"
            )
        lines.append("")
    if not sample_df.empty:
        lines.append("## Outputs")
        lines.append("- all_sample_attention.csv: every eligible test tree with risk and attn_P1...attn_P13.")
        lines.append("- static_group_attention.csv: attention summaries grouped by Sheet2 static attributes.")
        lines.append("- period_occlusion_summary.csv: periods whose masking changes risk the most.")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    all_diversity = []
    all_top_dist = []
    all_groups = []
    all_shuffle = []
    all_random = []
    all_counterfactual = []
    all_occlusion_detail = []
    all_occlusion_summary = []

    for artifact_text in args.artifact_dirs:
        artifact_dir = Path(artifact_text)
        artifact_label = artifact_dir.name
        config = load_config(artifact_dir)
        tree_data = load_dynamic_hlb_dataset(
            data_path=args.data_path,
            use_ct_aux_task=bool(config.get("use_ct_aux_task", False)),
            use_agro_features=bool(config.get("use_agro_features", False)),
            env_feature_set=str(config.get("env_feature_set", "auto")),
            build_period_env=bool(config.get("build_period_env", True)),
            period_feature_mode=str(config.get("period_feature_mode", "full")),
        )
        window_specs = build_window_specs(args.window_pairs, tree_data.num_periods)
        repeats = available_repeats(artifact_dir, args.max_repeats)
        sample_df, diversity_df, top_dist_df = collect_attention_outputs(
            artifact_label, artifact_dir, tree_data, config, window_specs, repeats
        )
        group_df = summarize_static_groups(sample_df, tree_data)
        shuffle_df = run_static_shuffle(
            artifact_label, artifact_dir, tree_data, config, window_specs, repeats, args.static_shuffle_repeats, args.seed
        )
        random_df = run_random_init(
            artifact_label, tree_data, config, window_specs, repeats, args.random_init_repeats, args.seed
        )
        counterfactual_df = run_static_counterfactual(artifact_label, artifact_dir, tree_data, config, window_specs, repeats)
        occlusion_detail_df, occlusion_summary_df = run_period_occlusion(
            artifact_label, artifact_dir, tree_data, config, window_specs, repeats
        )

        all_samples.append(sample_df)
        all_diversity.append(diversity_df)
        all_top_dist.append(top_dist_df)
        all_groups.append(group_df)
        all_shuffle.append(shuffle_df)
        all_random.append(random_df)
        all_counterfactual.append(counterfactual_df)
        all_occlusion_detail.append(occlusion_detail_df)
        all_occlusion_summary.append(occlusion_summary_df)

    sample_df = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    diversity_df = pd.concat(all_diversity, ignore_index=True) if all_diversity else pd.DataFrame()
    top_dist_df = pd.concat(all_top_dist, ignore_index=True) if all_top_dist else pd.DataFrame()
    group_df = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
    shuffle_df = pd.concat(all_shuffle, ignore_index=True) if all_shuffle else pd.DataFrame()
    random_df = pd.concat(all_random, ignore_index=True) if all_random else pd.DataFrame()
    counterfactual_df = pd.concat(all_counterfactual, ignore_index=True) if all_counterfactual else pd.DataFrame()
    occlusion_detail_df = pd.concat(all_occlusion_detail, ignore_index=True) if all_occlusion_detail else pd.DataFrame()
    occlusion_summary_df = pd.concat(all_occlusion_summary, ignore_index=True) if all_occlusion_summary else pd.DataFrame()

    save_csv(sample_df, output_dir / "all_sample_attention.csv")
    save_csv(diversity_df, output_dir / "attention_diversity.csv")
    save_csv(top_dist_df, output_dir / "top_period_distribution.csv")
    save_csv(group_df, output_dir / "static_group_attention.csv")
    save_csv(shuffle_df, output_dir / "static_shuffle_metrics.csv")
    save_csv(random_df, output_dir / "random_init_diversity.csv")
    save_csv(counterfactual_df, output_dir / "static_counterfactual.csv")
    save_csv(occlusion_detail_df, output_dir / "period_occlusion_detail.csv")
    save_csv(occlusion_summary_df, output_dir / "period_occlusion_summary.csv")
    write_report(output_dir, sample_df, diversity_df, random_df, shuffle_df)
    print(f"Saved tree-specific attention diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
