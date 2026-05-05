"""
Static KAN survival experiment on HLB Sheet2 + Sheet3 only.

This runner keeps the original KAN architecture intact and evaluates
repeated fixed splits with Cox-style calibration for horizon-specific
window metrics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from lifelines.utils import concordance_index

from models import KAN
from utils import negative_log_likelihood, proximal_l1, weighted_brier_score, weighted_c_index


TREE_ID_COLUMN = "编号"
CONT_COLUMNS = ["树龄", "树势", "东西冠幅", "南北冠幅", "树干高", "树高"]
VARIETY_COLUMN = "品种"
NEIGHBOR_COLUMN = "周围是否有病树"
EVENT_DATE_COLUMN = "首次发病日期"
EVENT_FLAG_COLUMN = "观测期内发病"


@dataclass(frozen=True)
class StaticHLBData:
    tree_ids: np.ndarray
    cont_x: np.ndarray
    variety: np.ndarray
    neighbor: np.ndarray
    event_flag: np.ndarray
    time_period: np.ndarray
    period_dates: list[pd.Timestamp]
    feature_names: list[str]


@dataclass(frozen=True)
class StaticSplit:
    repeat_index: int
    seed: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_tree_ids: list[str]
    val_tree_ids: list[str]
    test_tree_ids: list[str]


@dataclass(frozen=True)
class CandidateKey:
    lr: float
    tau: float
    weight_decay: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_tree_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:g}"

    text = str(value).strip()
    if text.endswith(".0"):
        numeric_part = text[:-2]
        if numeric_part.isdigit():
            return numeric_part
    return text


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().tolist())
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train static KAN on HLB Sheet2 + Sheet3 only.")
    parser.add_argument("--data_path", type=str, default="data/hlb_dataset.xlsx", help="Path to hlb_dataset.xlsx")
    parser.add_argument(
        "--fixed_split_json",
        type=str,
        default="artifacts/dynamic_hlb_trigger_v2_windowset_shared_splits_seed42.json",
        help="Fixed tree-level repeated splits JSON",
    )
    parser.add_argument(
        "--window_csv",
        type=str,
        default="artifacts/recommended_eval_windows_latefix3.csv",
        help="Window definition CSV",
    )
    parser.add_argument(
        "--window_tier",
        type=str,
        default="strict",
        help="Window tier to evaluate from the CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/static_hlb_sheet23_kan",
        help="Output directory",
    )
    return parser.parse_args()


def load_static_hlb_dataset(data_path: str) -> StaticHLBData:
    workbook = Path(data_path)
    sheet2 = pd.read_excel(workbook, sheet_name="Sheet2")
    sheet3 = pd.read_excel(workbook, sheet_name="Sheet3")

    period_dates = [pd.Timestamp(sheet3.columns[0]).normalize()]
    period_dates.extend(pd.to_datetime(sheet3.iloc[:, 0], errors="raise").dt.normalize().tolist())
    if len(period_dates) < 2:
        raise ValueError("Sheet3 must provide at least two period boundaries.")

    event_lookup = {date.normalize(): index + 1 for index, date in enumerate(period_dates)}
    num_periods = len(period_dates)

    tree_ids = sheet2[TREE_ID_COLUMN].map(normalize_tree_id).to_numpy()
    cont_x = sheet2[CONT_COLUMNS].to_numpy(dtype=np.float32)
    variety = sheet2[VARIETY_COLUMN].astype(int).to_numpy(dtype=np.int64)
    neighbor = sheet2[NEIGHBOR_COLUMN].to_numpy(dtype=np.float32)
    event_flag = sheet2[EVENT_FLAG_COLUMN].astype(int).to_numpy(dtype=np.int64)
    event_dates = pd.to_datetime(sheet2[EVENT_DATE_COLUMN], errors="coerce")

    time_period = np.full(len(sheet2), num_periods, dtype=np.int64)
    for index, (flag, event_date) in enumerate(zip(event_flag, event_dates)):
        if flag == 0:
            continue
        if pd.isna(event_date):
            raise ValueError(f"Tree {tree_ids[index]} is marked as event but has no event date.")
        normalized = pd.Timestamp(event_date).normalize()
        if normalized not in event_lookup:
            raise ValueError(
                f"Tree {tree_ids[index]} has event date {normalized.date()} not present in Sheet3."
            )
        time_period[index] = int(event_lookup[normalized])

    feature_names = CONT_COLUMNS + [VARIETY_COLUMN, NEIGHBOR_COLUMN]
    return StaticHLBData(
        tree_ids=tree_ids,
        cont_x=cont_x,
        variety=variety,
        neighbor=neighbor,
        event_flag=event_flag,
        time_period=time_period,
        period_dates=period_dates,
        feature_names=feature_names,
    )


def load_window_specs(window_csv: str, window_tier: str) -> list[dict[str, int]]:
    window_df = pd.read_csv(window_csv)
    if "recommended_tier" not in window_df.columns:
        raise ValueError(f"{window_csv} is missing the 'recommended_tier' column.")

    tier = str(window_tier).strip().lower()
    selected = window_df[window_df["recommended_tier"].astype(str).str.strip().str.lower() == tier].copy()
    if selected.empty:
        available = sorted(window_df["recommended_tier"].astype(str).str.strip().str.lower().unique().tolist())
        raise ValueError(f"No windows found for tier '{window_tier}'. Available tiers: {available}")

    specs: list[dict[str, int]] = []
    for _, row in selected.iterrows():
        specs.append(
            {
                "landmark": int(row["obs_periods"]),
                "pred_horizon": int(row["pred_periods"]),
                "absolute_horizon": int(row["absolute_horizon"]),
            }
        )
    return specs


def load_fixed_splits(data: StaticHLBData, fixed_split_json: str) -> list[StaticSplit]:
    payload = json.loads(Path(fixed_split_json).read_text(encoding="utf-8"))
    repeat_splits = payload.get("repeat_splits")
    if not isinstance(repeat_splits, dict):
        raise ValueError("fixed_split_json must contain a repeat_splits object.")

    tree_lookup = {tree_id: index for index, tree_id in enumerate(data.tree_ids)}
    splits: list[StaticSplit] = []

    for repeat_key in sorted(repeat_splits.keys(), key=lambda item: int(item)):
        repeat_payload = repeat_splits[repeat_key]
        train_tree_ids = [normalize_tree_id(tree_id) for tree_id in repeat_payload["train_tree_ids"]]
        val_tree_ids = [normalize_tree_id(tree_id) for tree_id in repeat_payload["val_tree_ids"]]
        test_tree_ids = [normalize_tree_id(tree_id) for tree_id in repeat_payload["test_tree_ids"]]

        train_indices = np.asarray([tree_lookup[tree_id] for tree_id in train_tree_ids], dtype=np.int64)
        val_indices = np.asarray([tree_lookup[tree_id] for tree_id in val_tree_ids], dtype=np.int64)
        test_indices = np.asarray([tree_lookup[tree_id] for tree_id in test_tree_ids], dtype=np.int64)

        all_indices = np.concatenate([train_indices, val_indices, test_indices])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError(f"Repeat {repeat_key} contains overlapping tree IDs across splits.")

        splits.append(
            StaticSplit(
                repeat_index=int(repeat_key),
                seed=int(repeat_payload.get("seed", 42 + int(repeat_key) * 1000)),
                train_indices=np.sort(train_indices),
                val_indices=np.sort(val_indices),
                test_indices=np.sort(test_indices),
                train_tree_ids=train_tree_ids,
                val_tree_ids=val_tree_ids,
                test_tree_ids=test_tree_ids,
            )
        )

    return splits


def fit_preprocessor(data: StaticHLBData, train_indices: np.ndarray) -> dict[str, np.ndarray | list[int]]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    cont_train = data.cont_x[train_indices]
    cont_mean = cont_train.mean(axis=0)
    cont_std = cont_train.std(axis=0)
    cont_std[cont_std == 0] = 1.0

    variety_levels = sorted(np.unique(data.variety).astype(int).tolist())
    return {
        "cont_mean": cont_mean.astype(np.float32),
        "cont_std": cont_std.astype(np.float32),
        "variety_levels": variety_levels,
    }


def transform_static_features(data: StaticHLBData, preprocessor: dict[str, np.ndarray | list[int]]) -> tuple[np.ndarray, list[str]]:
    cont_mean = np.asarray(preprocessor["cont_mean"], dtype=np.float32)
    cont_std = np.asarray(preprocessor["cont_std"], dtype=np.float32)
    variety_levels = list(preprocessor["variety_levels"])

    cont_features = (data.cont_x - cont_mean) / cont_std
    variety_onehot = np.stack([(data.variety == level).astype(np.float32) for level in variety_levels], axis=1)
    neighbor_feature = data.neighbor.reshape(-1, 1).astype(np.float32)

    x = np.concatenate([cont_features, variety_onehot, neighbor_feature], axis=1).astype(np.float32)
    feature_names = CONT_COLUMNS + [f"品种_{level}" for level in variety_levels] + [NEIGHBOR_COLUMN]
    return x, feature_names


def split_arrays(
    x: np.ndarray,
    time_period: np.ndarray,
    event_flag: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = np.asarray(indices, dtype=np.int64)
    return x[subset], time_period[subset], event_flag[subset]


def sort_by_time_desc(
    x: np.ndarray,
    time_period: np.ndarray,
    event_flag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(time_period, dtype=np.int64))
    return x[order], np.asarray(time_period)[order], np.asarray(event_flag)[order]


def set_torch_arrays(x: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(e, dtype=torch.float32),
    )


def fit_breslow_cumulative_hazard(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.int64).reshape(-1)
    events = np.asarray(events, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    max_time = int(times.max()) if len(times) > 0 else 0
    cumulative = np.zeros(max_time + 1, dtype=np.float64)
    running = 0.0

    for time_point in range(1, max_time + 1):
        event_mask = (times == time_point) & (events == 1)
        d_t = int(event_mask.sum())
        if d_t > 0:
            risk_mask = times >= time_point
            risk_scores = scores[risk_mask]
            if len(risk_scores) == 0:
                raise ValueError("Empty risk set encountered while fitting Breslow hazard.")
            max_score = float(np.max(risk_scores))
            denom = float(np.exp(risk_scores - max_score).sum())
            running += float(d_t) * math.exp(-max_score) / denom
        cumulative[time_point] = running

    return cumulative


def conditional_risk(scores: np.ndarray, cumulative_hazard: np.ndarray, landmark: int, absolute_horizon: int) -> np.ndarray:
    landmark = int(landmark)
    absolute_horizon = int(absolute_horizon)
    if absolute_horizon >= len(cumulative_hazard):
        absolute_horizon = len(cumulative_hazard) - 1
    if landmark >= len(cumulative_hazard):
        landmark = len(cumulative_hazard) - 1

    delta_hazard = max(float(cumulative_hazard[absolute_horizon] - cumulative_hazard[landmark]), 0.0)
    risk = 1.0 - np.exp(-delta_hazard * np.exp(np.asarray(scores, dtype=np.float64).reshape(-1)))
    return np.clip(risk, 0.0, 1.0)


def evaluate_windows(
    train_times: np.ndarray,
    train_events: np.ndarray,
    eval_times: np.ndarray,
    eval_events: np.ndarray,
    eval_scores: np.ndarray,
    cumulative_hazard: np.ndarray,
    window_specs: list[dict[str, int]],
) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for spec in window_specs:
        landmark = int(spec["landmark"])
        pred_horizon = int(spec["pred_horizon"])
        absolute_horizon = int(spec["absolute_horizon"])

        eligible_mask = eval_times > landmark
        eligible_scores = eval_scores[eligible_mask]
        eligible_times = eval_times[eligible_mask]
        eligible_events = eval_events[eligible_mask]

        record = {
            "landmark": landmark,
            "pred_horizon": pred_horizon,
            "absolute_horizon": absolute_horizon,
            "eligible_n": int(eligible_mask.sum()),
            "future_event_n": int(np.sum((eligible_times <= absolute_horizon) & (eligible_events == 1))),
            "ctd": float("nan"),
            "bstd": float("nan"),
        }

        if len(eligible_times) == 0:
            records.append(record)
            continue

        horizon_risk = conditional_risk(eligible_scores, cumulative_hazard, landmark, absolute_horizon)
        ctd = weighted_c_index(
            train_times=train_times,
            train_events=train_events,
            prediction=horizon_risk,
            test_times=eligible_times,
            test_events=eligible_events,
            horizon=absolute_horizon,
        )
        bstd = weighted_brier_score(
            train_times=train_times,
            train_events=train_events,
            prediction=horizon_risk,
            test_times=eligible_times,
            test_events=eligible_events,
            horizon=absolute_horizon,
        )
        record["ctd"] = float(ctd)
        record["bstd"] = float(bstd)
        records.append(record)

    return records


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan")
    return float(finite.mean())


def compare_metric_tuple(
    current: tuple[float, float, float],
    best: tuple[float, float, float] | None,
) -> bool:
    if best is None:
        return True
    current_ctd, current_bstd, current_cindex = current
    best_ctd, best_bstd, best_cindex = best
    if current_ctd != best_ctd:
        return current_ctd > best_ctd
    if current_bstd != best_bstd:
        return current_bstd < best_bstd
    return current_cindex > best_cindex


def safe_concordance_index(times: np.ndarray, scores: np.ndarray, events: np.ndarray) -> float:
    try:
        return float(concordance_index(times, -scores, events))
    except Exception:
        return float("nan")


def build_model(input_dim: int) -> KAN:
    return KAN([input_dim, 1, 1, 1], grid_size=1, spline_order=3)


def train_one_repeat(
    data: StaticHLBData,
    split: StaticSplit,
    window_specs: list[dict[str, int]],
    hyperparams: CandidateKey,
    epochs: int = 500,
    patience: int = 100,
    device: torch.device | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    set_seed(split.seed)

    preprocessor = fit_preprocessor(data, split.train_indices)
    x_all_transformed, feature_names = transform_static_features(data, preprocessor)

    x_train, t_train, e_train = split_arrays(x_all_transformed, data.time_period, data.event_flag, split.train_indices)
    x_val, t_val, e_val = split_arrays(x_all_transformed, data.time_period, data.event_flag, split.val_indices)
    x_test, t_test, e_test = split_arrays(x_all_transformed, data.time_period, data.event_flag, split.test_indices)
    x_train, t_train, e_train = sort_by_time_desc(x_train, t_train, e_train)
    x_val, t_val, e_val = sort_by_time_desc(x_val, t_val, e_val)
    x_test, t_test, e_test = sort_by_time_desc(x_test, t_test, e_test)

    x_train_t, t_train_t, e_train_t = set_torch_arrays(x_train, t_train, e_train)
    x_val_t, _, _ = set_torch_arrays(x_val, t_val, e_val)
    x_test_t = torch.tensor(x_test, dtype=torch.float32) if evaluate_test else None

    model = build_model(x_all_transformed.shape[1]).to(device)
    x_train_t = x_train_t.to(device)
    x_val_t = x_val_t.to(device)
    if x_test_t is not None:
        x_test_t = x_test_t.to(device)
    t_train_t = t_train_t.to(device)
    e_train_t = e_train_t.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(patience // 4, 2),
        min_lr=1e-6,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics: dict[str, Any] | None = None
    best_tuple: tuple[float, float, float] | None = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_scores_tensor = model(x_train_t)
        train_loss = negative_log_likelihood(train_scores_tensor, e_train_t)
        if torch.isnan(train_loss):
            raise RuntimeError(f"NaN loss encountered at epoch {epoch}.")
        train_loss.backward()
        optimizer.step()
        proximal_l1(model, hyperparams.tau, optimizer.param_groups[0]["lr"])

        model.eval()
        with torch.no_grad():
            train_scores = model(x_train_t).detach().cpu().numpy().reshape(-1)
            val_scores = model(x_val_t).detach().cpu().numpy().reshape(-1)

        cumulative_hazard = fit_breslow_cumulative_hazard(t_train, e_train, train_scores)
        val_window_records = evaluate_windows(
            train_times=t_train,
            train_events=e_train,
            eval_times=t_val,
            eval_events=e_val,
            eval_scores=val_scores,
            cumulative_hazard=cumulative_hazard,
            window_specs=window_specs,
        )
        val_mean_ctd = mean_or_nan(record["ctd"] for record in val_window_records)
        val_mean_bstd = mean_or_nan(record["bstd"] for record in val_window_records)
        val_global_cindex = safe_concordance_index(t_val, val_scores, e_val)

        scheduler.step(val_mean_ctd if np.isfinite(val_mean_ctd) else -1.0)

        current_tuple = (
            float(val_mean_ctd) if np.isfinite(val_mean_ctd) else float("-inf"),
            float(val_mean_bstd) if np.isfinite(val_mean_bstd) else float("inf"),
            float(val_global_cindex),
        )
        if compare_metric_tuple(current_tuple, best_tuple):
            best_tuple = current_tuple
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_metrics = {
                "val_mean_ctd": float(val_mean_ctd),
                "val_mean_bstd": float(val_mean_bstd),
                "val_global_c_index": float(val_global_cindex),
                "val_window_records": val_window_records,
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_scores = model(x_train_t).detach().cpu().numpy().reshape(-1)
        test_scores = (
            model(x_test_t).detach().cpu().numpy().reshape(-1)
            if x_test_t is not None
            else None
        )

    cumulative_hazard = fit_breslow_cumulative_hazard(t_train, e_train, train_scores)

    repeat_metrics = {
        "repeat": int(split.repeat_index),
        "seed": int(split.seed),
        "split_sizes": {
            "train": int(len(split.train_indices)),
            "val": int(len(split.val_indices)),
            "test": int(len(split.test_indices)),
        },
        "split_tree_ids": {
            "train": split.train_tree_ids,
            "val": split.val_tree_ids,
            "test": split.test_tree_ids,
        },
        "best_epoch": int(best_epoch),
        "hyperparams": asdict(hyperparams),
        "feature_names": feature_names,
        "best_validation": best_val_metrics,
        "test_global_c_index": None,
        "test_window_records": [],
        "test_mean_ctd": None,
        "test_mean_bstd": None,
        "test_c_index": None,
    }
    repeat_metrics["cumulative_hazard"] = cumulative_hazard.tolist()

    if evaluate_test and test_scores is not None:
        test_window_records = evaluate_windows(
            train_times=t_train,
            train_events=e_train,
            eval_times=t_test,
            eval_events=e_test,
            eval_scores=test_scores,
            cumulative_hazard=cumulative_hazard,
            window_specs=window_specs,
        )
        repeat_metrics["test_global_c_index"] = safe_concordance_index(t_test, test_scores, e_test)
        repeat_metrics["test_window_records"] = test_window_records
        repeat_metrics["test_mean_ctd"] = mean_or_nan(record["ctd"] for record in test_window_records)
        repeat_metrics["test_mean_bstd"] = mean_or_nan(record["bstd"] for record in test_window_records)
        repeat_metrics["test_c_index"] = repeat_metrics["test_global_c_index"]

    return repeat_metrics


def aggregate_candidate_results(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = (float(row["lr"]), float(row["tau"]), float(row["weight_decay"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (lr, tau, weight_decay), rows in grouped.items():
        val_ctd = np.asarray([float(row["best_validation"]["val_mean_ctd"]) for row in rows], dtype=np.float64)
        val_bstd = np.asarray([float(row["best_validation"]["val_mean_bstd"]) for row in rows], dtype=np.float64)
        val_cindex = np.asarray([float(row["best_validation"]["val_global_c_index"]) for row in rows], dtype=np.float64)
        aggregated.append(
            {
                "lr": lr,
                "tau": tau,
                "weight_decay": weight_decay,
                "val_mean_ctd_mean": float(np.nanmean(val_ctd)),
                "val_mean_ctd_std": float(np.nanstd(val_ctd, ddof=0)),
                "val_mean_bstd_mean": float(np.nanmean(val_bstd)),
                "val_mean_bstd_std": float(np.nanstd(val_bstd, ddof=0)),
                "val_global_c_index_mean": float(np.nanmean(val_cindex)),
                "val_global_c_index_std": float(np.nanstd(val_cindex, ddof=0)),
                "repeat_n": int(len(rows)),
            }
        )

    aggregated.sort(
        key=lambda row: (
            -float(row["val_mean_ctd_mean"]),
            float(row["val_mean_bstd_mean"]),
            -float(row["val_global_c_index_mean"]),
            float(row["lr"]),
            float(row["tau"]),
            float(row["weight_decay"]),
        )
    )
    for rank, row in enumerate(aggregated, start=1):
        row["rank"] = int(rank)
    return aggregated


def select_best_candidate(candidate_summary: list[dict[str, Any]]) -> CandidateKey:
    if not candidate_summary:
        raise ValueError("No candidate summary rows available.")
    best = candidate_summary[0]
    return CandidateKey(lr=float(best["lr"]), tau=float(best["tau"]), weight_decay=float(best["weight_decay"]))


def run_tuning(
    data: StaticHLBData,
    splits: list[StaticSplit],
    window_specs: list[dict[str, int]],
    output_dir: Path,
    device: torch.device,
) -> tuple[CandidateKey, list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_grid = list(
        itertools.product(
            [0.1, 0.3, 0.5],
            [0.0, 0.1, 0.25, 0.5, 1.0],
            [0.0, 0.1, 0.5, 1.0],
        )
    )
    tuning_rows: list[dict[str, Any]] = []

    for lr, tau, weight_decay in candidate_grid:
        hyperparams = CandidateKey(lr=float(lr), tau=float(tau), weight_decay=float(weight_decay))
        for split in splits:
            repeat_metrics = train_one_repeat(
                data=data,
                split=split,
                window_specs=window_specs,
                hyperparams=hyperparams,
                device=device,
                evaluate_test=False,
            )
            tuning_rows.append(
                {
                    "repeat": int(repeat_metrics["repeat"]),
                    "seed": int(repeat_metrics["seed"]),
                    "split_sizes": repeat_metrics["split_sizes"],
                    "hyperparams": repeat_metrics["hyperparams"],
                    "lr": float(hyperparams.lr),
                    "tau": float(hyperparams.tau),
                    "weight_decay": float(hyperparams.weight_decay),
                    "best_epoch": int(repeat_metrics["best_epoch"]),
                    "best_validation": repeat_metrics["best_validation"],
                }
            )

    tuning_summary = aggregate_candidate_results(tuning_rows)
    best_candidate = select_best_candidate(tuning_summary)

    tuning_csv_rows = [
        {
            "lr": row["lr"],
            "tau": row["tau"],
            "weight_decay": row["weight_decay"],
            "rank": row["rank"],
            "repeat_n": row["repeat_n"],
            "val_mean_ctd_mean": row["val_mean_ctd_mean"],
            "val_mean_ctd_std": row["val_mean_ctd_std"],
            "val_mean_bstd_mean": row["val_mean_bstd_mean"],
            "val_mean_bstd_std": row["val_mean_bstd_std"],
            "val_global_c_index_mean": row["val_global_c_index_mean"],
            "val_global_c_index_std": row["val_global_c_index_std"],
        }
        for row in tuning_summary
    ]

    pd.DataFrame(tuning_csv_rows).to_csv(output_dir / "tuning_results.csv", index=False)
    write_json(
        output_dir / "tuning_results.json",
        {
            "selected_hyperparams": asdict(best_candidate),
            "candidate_summary": tuning_summary,
            "repeat_details": tuning_rows,
        },
    )

    return best_candidate, tuning_summary, tuning_rows


def run_final_evaluation(
    data: StaticHLBData,
    splits: list[StaticSplit],
    window_specs: list[dict[str, int]],
    best_candidate: CandidateKey,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repeat_rows: list[dict[str, Any]] = []

    for split in splits:
        repeat_metrics = train_one_repeat(
            data=data,
            split=split,
            window_specs=window_specs,
            hyperparams=best_candidate,
            device=device,
            evaluate_test=True,
        )
        repeat_rows.append(repeat_metrics)

        repeat_dir = output_dir / f"repeat_{split.repeat_index}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = repeat_dir / "metrics.json"
        write_json(metrics_path, repeat_metrics)

    repeat_test_cindex = np.asarray([row["test_c_index"] for row in repeat_rows], dtype=np.float64)
    repeat_test_mean_ctd = np.asarray([row["test_mean_ctd"] for row in repeat_rows], dtype=np.float64)
    repeat_test_mean_bstd = np.asarray([row["test_mean_bstd"] for row in repeat_rows], dtype=np.float64)

    summary_overall = {
        "best_hyperparams": asdict(best_candidate),
        "test_c_index_mean": float(np.nanmean(repeat_test_cindex)),
        "test_c_index_std": float(np.nanstd(repeat_test_cindex, ddof=0)),
        "mean_ctd_mean": float(np.nanmean(repeat_test_mean_ctd)),
        "mean_ctd_std": float(np.nanstd(repeat_test_mean_ctd, ddof=0)),
        "mean_bstd_mean": float(np.nanmean(repeat_test_mean_bstd)),
        "mean_bstd_std": float(np.nanstd(repeat_test_mean_bstd, ddof=0)),
        "repeat_count": int(len(repeat_rows)),
    }

    window_summary_rows: list[dict[str, Any]] = []
    for spec in window_specs:
        landmark = int(spec["landmark"])
        pred_horizon = int(spec["pred_horizon"])
        absolute_horizon = int(spec["absolute_horizon"])
        ctd_values: list[float] = []
        bstd_values: list[float] = []
        eligible_values: list[float] = []
        future_event_values: list[float] = []

        for row in repeat_rows:
            record = next(
                item
                for item in row["test_window_records"]
                if int(item["landmark"]) == landmark
                and int(item["pred_horizon"]) == pred_horizon
                and int(item["absolute_horizon"]) == absolute_horizon
            )
            ctd_values.append(float(record["ctd"]))
            bstd_values.append(float(record["bstd"]))
            eligible_values.append(float(record["eligible_n"]))
            future_event_values.append(float(record["future_event_n"]))

        window_summary_rows.append(
            {
                "landmark": landmark,
                "pred_horizon": pred_horizon,
                "absolute_horizon": absolute_horizon,
                "eligible_n_mean": float(np.mean(eligible_values)),
                "eligible_n_std": float(np.std(eligible_values, ddof=0)),
                "future_event_n_mean": float(np.mean(future_event_values)),
                "future_event_n_std": float(np.std(future_event_values, ddof=0)),
                "ctd_mean": float(np.nanmean(ctd_values)),
                "ctd_std": float(np.nanstd(ctd_values, ddof=0)),
                "bstd_mean": float(np.nanmean(bstd_values)),
                "bstd_std": float(np.nanstd(bstd_values, ddof=0)),
            }
        )

    summary_csv_rows = [
        {
            "metric_type": "overall",
            "test_c_index_mean": summary_overall["test_c_index_mean"],
            "test_c_index_std": summary_overall["test_c_index_std"],
            "mean_ctd_mean": summary_overall["mean_ctd_mean"],
            "mean_ctd_std": summary_overall["mean_ctd_std"],
            "mean_bstd_mean": summary_overall["mean_bstd_mean"],
            "mean_bstd_std": summary_overall["mean_bstd_std"],
        }
    ]
    summary_csv_rows.extend(
        {
            "metric_type": "window",
            **row,
        }
        for row in window_summary_rows
    )

    pd.DataFrame(summary_csv_rows).to_csv(output_dir / "summary.csv", index=False)
    summary_json = {
        "best_hyperparams": asdict(best_candidate),
        "overall": summary_overall,
        "windows": window_summary_rows,
        "repeat_metrics": repeat_rows,
    }
    write_json(output_dir / "summary.json", summary_json)
    return repeat_rows, summary_json


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_static_hlb_dataset(args.data_path)
    window_specs = load_window_specs(args.window_csv, args.window_tier)
    splits = load_fixed_splits(data, args.fixed_split_json)

    if len(splits) != 5:
        print(f"Warning: expected 5 repeats, found {len(splits)}.")

    print(f"Loaded {len(data.tree_ids)} trees | events={int(data.event_flag.sum())} | censored={int((data.event_flag == 0).sum())}")
    print(f"Periods: {len(data.period_dates)}")
    print(f"Windows: {window_specs}")
    for split in splits:
        print(
            f"Repeat {split.repeat_index} split sizes: "
            f"{len(split.train_indices)}/{len(split.val_indices)}/{len(split.test_indices)}"
        )

    best_candidate, _, _ = run_tuning(
        data=data,
        splits=splits,
        window_specs=window_specs,
        output_dir=output_dir,
        device=device,
    )
    print(f"Selected hyperparams: {best_candidate}")

    repeat_rows, summary_json = run_final_evaluation(
        data=data,
        splits=splits,
        window_specs=window_specs,
        best_candidate=best_candidate,
        output_dir=output_dir,
        device=device,
    )

    print(
        "Final test c-index: "
        f"{summary_json['overall']['test_c_index_mean']:.4f} ± {summary_json['overall']['test_c_index_std']:.4f}"
    )
    print(
        "Final strict-window mean Ctd: "
        f"{summary_json['overall']['mean_ctd_mean']:.4f} ± {summary_json['overall']['mean_ctd_std']:.4f}"
    )
    print(
        "Final strict-window mean Bstd: "
        f"{summary_json['overall']['mean_bstd_mean']:.4f} ± {summary_json['overall']['mean_bstd_std']:.4f}"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
