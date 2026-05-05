"""
Dynamic data utilities for HLB time-varying survival analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


BASELINE_DATE = pd.Timestamp("2024-06-01")

RAW_ENV_COLUMN_MAP = {
    "空气温度": "air_temp",
    "空气湿度": "air_humidity",
    "光照强度": "light_intensity",
    "风向": "wind_direction",
    "风速": "wind_speed",
    "日累计雨量": "rainfall",
    "土壤温度": "soil_temp",
    "土壤水分": "soil_moisture",
    "土壤EC值": "soil_ec",
}
RAW9_ENV_FEATURES = list(RAW_ENV_COLUMN_MAP.values())
AGRO_PERIOD_STAT_FEATURES = [
    "air_temp",
    "air_humidity",
    "light_intensity",
    "wind_sin",
    "wind_cos",
    "wind_speed",
    "rainfall_log1p",
    "soil_temp",
    "soil_moisture",
    "soil_ec",
]
BASIC10_ENV_FEATURES = [
    "air_temp",
    "air_humidity",
    "light_intensity",
    "wind_sin",
    "wind_cos",
    "wind_speed",
    "rainfall",
    "soil_temp",
    "soil_moisture",
    "soil_ec",
]
COMPACT_PERIOD_ENDPOINT_FEATURES = [
    "air_temp",
    "air_humidity",
    "light_intensity",
    "wind_speed",
    "rainfall_log1p",
    "soil_temp",
    "soil_moisture",
    "soil_ec",
    "air_temp_mean_7d",
    "air_temp_mean_30d",
    "air_humidity_mean_7d",
    "light_intensity_mean_7d",
    "rainfall_sum_7d_log1p",
    "rainfall_sum_30d_log1p",
    "soil_moisture_mean_7d",
    "soil_moisture_mean_30d",
    "soil_ec_mean_7d",
    "soil_ec_mean_30d",
    "air_temp_slope_14d",
    "soil_moisture_slope_14d",
    "soil_ec_slope_14d",
    "temp_25_28_count_14d",
    "temp_gt_33_count_14d",
    "rainy_day_count_14d",
    "dry_day_count_14d",
    "low_soil_moist_count_14d",
    "high_soil_moist_count_14d",
    "high_ec_count_14d",
    "low_light_count_14d",
]
ROLLING_MEAN_FEATURES = [
    "air_temp",
    "air_humidity",
    "light_intensity",
    "wind_speed",
    "rainfall_log1p",
    "soil_temp",
    "soil_moisture",
    "soil_ec",
]
SLOPE_FEATURES = ["air_temp", "soil_moisture", "soil_ec"]
ROLLING_WINDOWS = [3, 7, 14, 30]
COUNT_WINDOWS = [7, 14, 30]


@dataclass
class DynamicTreeData:
    tree_ids: np.ndarray
    daily_env: np.ndarray
    period_env: np.ndarray
    shared_daily_env: np.ndarray
    shared_period_env: np.ndarray
    static_cont: np.ndarray
    tree_spatial: np.ndarray
    variety: np.ndarray
    has_diseased_neighbor: np.ndarray
    event_flag: np.ndarray
    time_period: np.ndarray
    seq_len_days: np.ndarray
    event_dates: np.ndarray
    env_feature_names: List[str]
    period_feature_names: List[str]
    static_cont_feature_names: List[str]
    variety_levels: List[int]
    period_end_dates: List[pd.Timestamp]
    period_bounds: List[pd.Timestamp]
    day_to_period: np.ndarray
    period_start_day_index: np.ndarray
    period_end_day_index: np.ndarray
    landmark_seq_len_days: np.ndarray
    landmark_seq_len_periods: np.ndarray
    max_days: int
    num_periods: int
    use_agro_features: bool
    env_feature_set: str
    build_period_env: bool
    period_feature_mode: str
    ct_values: np.ndarray | None = None
    ct_valid_mask: np.ndarray | None = None


@dataclass
class PrefixSample:
    tree_index: int
    landmark_period: int
    seq_len_days: int
    future_event: int
    ct_aux_mask: int | np.ndarray = 0
    ct_delta_target: float | np.ndarray = 0.0
    ct_state_target: float | np.ndarray = 0.0
    ct_state_mask: int | np.ndarray = 0
    ct_value_target: float | np.ndarray = 0.0
    ct_value_mask: int | np.ndarray = 0
    ct_aux_window_mask: np.ndarray | None = None
    current_ct: float = float("nan")
    next_ct: float = float("nan")


class PrefixSampleDataset(Dataset):
    """Dataset of landmark-prefix samples for dynamic training."""

    def __init__(
        self,
        tree_data: DynamicTreeData,
        static_x: np.ndarray,
        samples: Sequence[PrefixSample],
    ):
        self.tree_data = tree_data
        self.samples = list(samples)
        self.daily_env = torch.tensor(tree_data.daily_env, dtype=torch.float32)
        self.period_env = torch.tensor(tree_data.period_env, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.event_flag = torch.tensor(tree_data.event_flag, dtype=torch.float32)
        self.time_period = torch.tensor(tree_data.time_period, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        tree_index = sample.tree_index
        ct_aux_mask = np.asarray(sample.ct_aux_mask, dtype=np.float32).reshape(-1)
        ct_delta_target = np.asarray(sample.ct_delta_target, dtype=np.float32).reshape(-1)
        ct_state_target = np.asarray(sample.ct_state_target, dtype=np.float32).reshape(-1)
        ct_state_mask = np.asarray(sample.ct_state_mask, dtype=np.float32).reshape(-1)
        ct_value_target = np.asarray(sample.ct_value_target, dtype=np.float32).reshape(-1)
        ct_value_mask = np.asarray(sample.ct_value_mask, dtype=np.float32).reshape(-1)
        if sample.ct_aux_window_mask is None:
            ct_aux_window_mask = np.ones((1, ct_delta_target.size), dtype=np.float32)
        else:
            ct_aux_window_mask = np.asarray(sample.ct_aux_window_mask, dtype=np.float32)
        return {
            "env": self.daily_env[tree_index],
            "period_env": self.period_env[tree_index],
            "static_x": self.static_x[tree_index],
            "seq_len_days": torch.tensor(sample.seq_len_days, dtype=torch.long),
            "seq_len_periods": torch.tensor(sample.landmark_period, dtype=torch.long),
            "landmark_period": torch.tensor(sample.landmark_period, dtype=torch.long),
            "time_period": self.time_period[tree_index],
            "event_flag": self.event_flag[tree_index],
            "tree_index": torch.tensor(tree_index, dtype=torch.long),
            "ct_aux_mask": torch.tensor(ct_aux_mask, dtype=torch.float32),
            "ct_delta_target": torch.tensor(ct_delta_target, dtype=torch.float32),
            "ct_state_target": torch.tensor(ct_state_target, dtype=torch.float32),
            "ct_state_mask": torch.tensor(ct_state_mask, dtype=torch.float32),
            "ct_value_target": torch.tensor(ct_value_target, dtype=torch.float32),
            "ct_value_mask": torch.tensor(ct_value_mask, dtype=torch.float32),
            "ct_aux_window_mask": torch.tensor(ct_aux_window_mask, dtype=torch.float32),
            "current_ct": torch.tensor([sample.current_ct], dtype=torch.float32),
            "next_ct": torch.tensor([sample.next_ct], dtype=torch.float32),
        }


def load_dynamic_hlb_dataset(
    data_path: str,
    baseline_date: str | pd.Timestamp = BASELINE_DATE,
    use_ct_aux_task: bool = False,
    use_agro_features: bool = False,
    env_feature_set: str = "auto",
    build_period_env: bool = True,
    period_feature_mode: str = "full",
) -> DynamicTreeData:
    """
    Load the HLB dynamic survival dataset from Excel.
    """
    baseline = pd.Timestamp(baseline_date)
    path = Path(data_path)

    env_df = pd.read_excel(path, sheet_name="Sheet1")
    tree_df = pd.read_excel(path, sheet_name="Sheet2")
    measure_df = pd.read_excel(path, sheet_name="Sheet3")

    env_date_col = env_df.columns[0]
    env_df[env_date_col] = pd.to_datetime(env_df[env_date_col])
    env_df = env_df.sort_values(env_date_col).reset_index(drop=True)

    period_end_dates = _parse_measurement_dates(measure_df)
    period_bounds = [baseline] + period_end_dates
    final_date = period_end_dates[-1]

    env_window = env_df.loc[
        (env_df[env_date_col] >= baseline) & (env_df[env_date_col] <= final_date)
    ].reset_index(drop=True)
    expected_dates = pd.date_range(baseline, final_date, freq="D")
    observed_dates = pd.DatetimeIndex(env_window[env_date_col])
    missing_dates = expected_dates.difference(observed_dates)
    if len(missing_dates) > 0:
        raise ValueError(
            f"Environment data is missing {len(missing_dates)} calendar days after baseline."
        )

    resolved_env_feature_set = _resolve_env_feature_set(
        use_agro_features=use_agro_features,
        env_feature_set=env_feature_set,
    )
    day_to_period = _build_day_to_period(expected_dates, period_end_dates)
    (
        daily_env_values,
        env_feature_names,
        period_env_values,
        period_feature_names,
        period_start_day_index,
        period_end_day_index,
    ) = _build_environment_features(
        env_window=env_window,
        expected_dates=expected_dates,
        period_end_dates=period_end_dates,
        env_feature_set=resolved_env_feature_set,
        build_period_env=build_period_env,
        period_feature_mode=period_feature_mode,
    )

    num_periods = len(period_end_dates)
    max_days = len(expected_dates)

    tree_id_col = tree_df.columns[0]
    static_cont_feature_names = list(tree_df.columns[1:7])
    variety_col = tree_df.columns[7]
    neighbor_col = tree_df.columns[8]
    event_date_col = tree_df.columns[9]
    event_flag_col = tree_df.columns[10]

    tree_df[event_date_col] = pd.to_datetime(tree_df[event_date_col], errors="coerce")
    variety_levels = sorted(tree_df[variety_col].astype(int).unique().tolist())
    period_lookup = {date.normalize(): idx + 1 for idx, date in enumerate(period_end_dates)}

    tree_ids = tree_df[tree_id_col].map(_normalize_tree_id).to_numpy()
    static_cont = tree_df[static_cont_feature_names].to_numpy(dtype=np.float32)
    tree_spatial = np.stack([_parse_tree_id_spatial(tree_id) for tree_id in tree_ids], axis=0).astype(np.float32)
    variety = tree_df[variety_col].astype(int).to_numpy()
    has_diseased_neighbor = tree_df[neighbor_col].astype(np.float32).to_numpy()
    event_flag = tree_df[event_flag_col].astype(np.int64).to_numpy()

    ct_values = None
    ct_valid_mask = None
    if use_ct_aux_task:
        ct_values, ct_valid_mask = _load_ct_measurements(
            data_path=path,
            tree_ids=tree_ids,
            period_end_dates=period_end_dates,
        )

    time_period = np.zeros(len(tree_df), dtype=np.int64)
    seq_len_days = np.zeros(len(tree_df), dtype=np.int64)
    event_dates = tree_df[event_date_col].to_numpy()
    daily_env = np.repeat(daily_env_values[None, :, :], len(tree_df), axis=0).astype(np.float32)
    period_env = np.repeat(period_env_values[None, :, :], len(tree_df), axis=0).astype(np.float32)

    for row_index, (_, row) in enumerate(tree_df.iterrows()):
        is_event = int(row[event_flag_col]) == 1
        event_date = row[event_date_col]
        if is_event:
            if pd.isna(event_date):
                raise ValueError(f"Tree {row[tree_id_col]} is marked as event but has no event date.")
            normalized_event = pd.Timestamp(event_date).normalize()
            if normalized_event not in period_lookup:
                raise ValueError(
                    f"Tree {row[tree_id_col]} has event date {normalized_event.date()} not present in Sheet3."
                )
            time_period[row_index] = period_lookup[normalized_event]
            seq_len_days[row_index] = int((normalized_event - baseline).days) + 1
        else:
            time_period[row_index] = num_periods
            seq_len_days[row_index] = max_days

        if seq_len_days[row_index] < max_days:
            daily_env[row_index, seq_len_days[row_index] :] = 0.0

        if ct_valid_mask is not None and is_event:
            event_period_index = int(time_period[row_index]) - 1
            if event_period_index + 1 < num_periods:
                ct_valid_mask[row_index, event_period_index + 1 :] = False
                ct_values[row_index, event_period_index + 1 :] = 0.0

    landmark_seq_len_days = np.zeros(num_periods, dtype=np.int64)
    landmark_seq_len_periods = np.arange(num_periods, dtype=np.int64)
    landmark_seq_len_days[0] = 1
    for landmark in range(1, num_periods):
        landmark_seq_len_days[landmark] = int((period_end_dates[landmark - 1] - baseline).days) + 1

    return DynamicTreeData(
        tree_ids=tree_ids,
        daily_env=daily_env,
        period_env=period_env,
        shared_daily_env=daily_env_values.astype(np.float32),
        shared_period_env=period_env_values.astype(np.float32),
        static_cont=static_cont,
        tree_spatial=tree_spatial,
        variety=variety,
        has_diseased_neighbor=has_diseased_neighbor,
        event_flag=event_flag,
        time_period=time_period,
        seq_len_days=seq_len_days,
        event_dates=event_dates,
        env_feature_names=env_feature_names,
        period_feature_names=period_feature_names,
        static_cont_feature_names=static_cont_feature_names,
        variety_levels=variety_levels,
        period_end_dates=period_end_dates,
        period_bounds=period_bounds,
        day_to_period=day_to_period,
        period_start_day_index=period_start_day_index,
        period_end_day_index=period_end_day_index,
        landmark_seq_len_days=landmark_seq_len_days,
        landmark_seq_len_periods=landmark_seq_len_periods,
        max_days=max_days,
        num_periods=num_periods,
        use_agro_features=use_agro_features,
        env_feature_set=resolved_env_feature_set,
        build_period_env=bool(build_period_env),
        period_feature_mode=period_feature_mode,
        ct_values=ct_values,
        ct_valid_mask=ct_valid_mask,
    )


def fit_static_preprocessor(
    tree_data: DynamicTreeData,
    train_indices: Sequence[int],
    use_tree_id_spatial: bool = False,
) -> Dict[str, np.ndarray | List[int]]:
    """
    Fit static-feature preprocessing on training trees only.
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    cont_train = tree_data.static_cont[train_indices]
    cont_mean = cont_train.mean(axis=0)
    cont_std = cont_train.std(axis=0)
    cont_std[cont_std == 0] = 1.0
    preprocessor = {
        "cont_mean": cont_mean.astype(np.float32),
        "cont_std": cont_std.astype(np.float32),
        "variety_levels": tree_data.variety_levels,
        "use_tree_id_spatial": bool(use_tree_id_spatial),
    }
    if use_tree_id_spatial:
        spatial_train = tree_data.tree_spatial[train_indices]
        spatial_mean = spatial_train.mean(axis=0)
        spatial_std = spatial_train.std(axis=0)
        spatial_std[spatial_std == 0] = 1.0
        preprocessor["spatial_mean"] = spatial_mean.astype(np.float32)
        preprocessor["spatial_std"] = spatial_std.astype(np.float32)
    return preprocessor


def fit_ct_delta_preprocessor(samples: Sequence[PrefixSample]) -> Dict[str, float]:
    """
    Fit train-split-only standardization stats for valid CT delta targets.
    """
    valid_target_chunks = []
    for sample in samples:
        target = np.asarray(sample.ct_delta_target, dtype=np.float32).reshape(-1)
        mask = np.asarray(sample.ct_aux_mask, dtype=np.float32).reshape(-1) > 0.5
        if target.shape != mask.shape:
            raise ValueError(
                f"CT target and mask shapes do not match: target={target.shape}, mask={mask.shape}"
            )
        if mask.any():
            valid_target_chunks.append(target[mask])
    valid_targets = (
        np.concatenate(valid_target_chunks, axis=0)
        if valid_target_chunks
        else np.asarray([], dtype=np.float32)
    )
    if valid_targets.size == 0:
        return {
            "mean": 0.0,
            "std": 1.0,
            "count": 0,
        }

    target_std = float(valid_targets.std())
    if target_std == 0.0:
        target_std = 1.0

    return {
        "mean": float(valid_targets.mean()),
        "std": target_std,
        "count": int(valid_targets.size),
    }


def fit_ct_value_preprocessor(samples: Sequence[PrefixSample]) -> Dict[str, float]:
    """
    Fit train-split-only standardization stats for valid exact CT targets.
    """
    valid_target_chunks = []
    for sample in samples:
        target = np.asarray(sample.ct_value_target, dtype=np.float32).reshape(-1)
        mask = np.asarray(sample.ct_value_mask, dtype=np.float32).reshape(-1) > 0.5
        if target.shape != mask.shape:
            raise ValueError(
                f"CT value target and mask shapes do not match: target={target.shape}, mask={mask.shape}"
            )
        if mask.any():
            valid_target_chunks.append(target[mask])
    valid_targets = (
        np.concatenate(valid_target_chunks, axis=0)
        if valid_target_chunks
        else np.asarray([], dtype=np.float32)
    )
    if valid_targets.size == 0:
        return {
            "mean": 0.0,
            "std": 1.0,
            "count": 0,
        }

    target_std = float(valid_targets.std())
    if target_std == 0.0:
        target_std = 1.0

    return {
        "mean": float(valid_targets.mean()),
        "std": target_std,
        "count": int(valid_targets.size),
    }


def transform_static_features(
    tree_data: DynamicTreeData,
    preprocessor: Dict[str, np.ndarray | List[int]],
) -> tuple[np.ndarray, List[str]]:
    """
    Transform static features into a numeric matrix.
    """
    cont_mean = np.asarray(preprocessor["cont_mean"], dtype=np.float32)
    cont_std = np.asarray(preprocessor["cont_std"], dtype=np.float32)
    variety_levels = list(preprocessor["variety_levels"])

    cont_features = (tree_data.static_cont - cont_mean) / cont_std
    variety_onehot = np.stack(
        [(tree_data.variety == level).astype(np.float32) for level in variety_levels],
        axis=1,
    )
    neighbor_feature = tree_data.has_diseased_neighbor.reshape(-1, 1).astype(np.float32)

    feature_names = (
        tree_data.static_cont_feature_names
        + [f"variety_{level}" for level in variety_levels]
        + ["has_diseased_neighbor"]
    )
    feature_parts = [cont_features, variety_onehot, neighbor_feature]
    if bool(preprocessor.get("use_tree_id_spatial", False)):
        spatial_mean = np.asarray(preprocessor["spatial_mean"], dtype=np.float32)
        spatial_std = np.asarray(preprocessor["spatial_std"], dtype=np.float32)
        spatial_features = (tree_data.tree_spatial - spatial_mean) / spatial_std
        feature_parts.append(spatial_features.astype(np.float32))
        feature_names = feature_names + ["tree_row_coord", "tree_col_coord"]
    static_x = np.concatenate(feature_parts, axis=1)
    return static_x.astype(np.float32), feature_names


def make_stratification_labels(
    event_flag: np.ndarray,
    time_period: np.ndarray,
    num_event_bins: int = 3,
) -> np.ndarray:
    """
    Build stable stratification labels for tree-level splitting.
    """
    event_flag = np.asarray(event_flag, dtype=np.int64)
    time_period = np.asarray(time_period, dtype=np.int64)

    event_periods = time_period[event_flag == 1]
    if len(event_periods) == 0:
        return np.array(["censor"] * len(event_flag))

    quantiles = np.linspace(0.0, 1.0, num_event_bins + 1)[1:-1]
    if len(quantiles) == 0:
        cut_points = np.array([], dtype=np.float32)
    else:
        cut_points = np.quantile(event_periods, quantiles)
        cut_points = np.unique(cut_points)

    labels = []
    for is_event, period in zip(event_flag, time_period):
        if is_event == 1:
            period_bin = int(np.digitize(period, cut_points, right=True))
            labels.append(f"event_{period_bin}")
        else:
            labels.append("censor")
    return np.asarray(labels)


def split_tree_indices(
    tree_data: DynamicTreeData,
    test_size: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split tree indices into train/validation/test at the tree level.
    """
    all_indices = np.arange(len(tree_data.tree_ids))
    strat_labels = _choose_stratify_labels(
        event_flag=tree_data.event_flag,
        time_period=tree_data.time_period,
    )
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=_safe_stratify_labels(strat_labels),
    )

    train_strat_labels = _choose_stratify_labels(
        event_flag=tree_data.event_flag[train_indices],
        time_period=tree_data.time_period[train_indices],
    )
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=val_ratio,
        random_state=random_state,
        stratify=_safe_stratify_labels(train_strat_labels),
    )
    return (
        np.sort(train_indices.astype(np.int64)),
        np.sort(val_indices.astype(np.int64)),
        np.sort(test_indices.astype(np.int64)),
    )


def _resolve_split_section(payload: dict[str, Any], repeat_index: int | None) -> dict[str, Any]:
    if repeat_index is None or "repeat_splits" not in payload:
        return payload
    repeat_splits = payload["repeat_splits"]
    repeat_key = str(int(repeat_index))
    if repeat_key not in repeat_splits:
        raise KeyError(f"Repeat {repeat_index} was not found in fixed split payload.")
    merged = dict(payload)
    merged.update(repeat_splits[repeat_key])
    return merged


def _resolve_split_indices(
    tree_data: DynamicTreeData,
    payload: dict[str, Any],
    split_name: str,
) -> np.ndarray:
    indices_key = f"{split_name}_indices"
    tree_ids_key = f"{split_name}_tree_ids"
    if indices_key in payload:
        indices = np.asarray(payload[indices_key], dtype=np.int64)
        return np.sort(indices)
    if tree_ids_key not in payload:
        raise KeyError(f"Missing '{indices_key}' or '{tree_ids_key}' in fixed split payload.")

    tree_lookup = {str(tree_id): idx for idx, tree_id in enumerate(tree_data.tree_ids)}
    resolved_indices = []
    missing_ids = []
    for raw_tree_id in payload[tree_ids_key]:
        tree_id = str(raw_tree_id)
        if tree_id not in tree_lookup:
            missing_ids.append(tree_id)
            continue
        resolved_indices.append(tree_lookup[tree_id])
    if missing_ids:
        raise KeyError(f"Fixed split references unknown tree IDs for {split_name}: {missing_ids}")
    return np.sort(np.asarray(resolved_indices, dtype=np.int64))


def load_fixed_split_indices(
    tree_data: DynamicTreeData,
    split_json_path: str | Path,
    repeat_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    effective_payload = _resolve_split_section(payload, repeat_index)

    train_indices = _resolve_split_indices(tree_data, effective_payload, "train")
    val_indices = _resolve_split_indices(tree_data, effective_payload, "val")
    test_indices = _resolve_split_indices(tree_data, effective_payload, "test")

    combined = np.concatenate([train_indices, val_indices, test_indices])
    unique_indices = np.unique(combined)
    if len(unique_indices) != len(combined):
        raise ValueError("Fixed split contains duplicate tree assignments across train/val/test.")
    return train_indices, val_indices, test_indices


def build_prefix_samples(
    tree_data: DynamicTreeData,
    tree_indices: Sequence[int],
    include_landmark_zero: bool = True,
    allowed_landmarks: Sequence[int] | None = None,
    ct_delta_output_dim: int | None = None,
    ct_aux_output_dim: int | None = None,
    ct_state_output_dim: int | None = None,
    ct_aux_window_specs: Sequence[tuple[int, int]] | None = None,
    ct_aux_target_mode: str | None = None,
) -> List[PrefixSample]:
    """
    Build prefix samples at period boundaries for trees still at risk.
    """
    lower_landmark = 0 if include_landmark_zero else 1
    upper_landmark = tree_data.num_periods - 1
    prefix_samples: List[PrefixSample] = []
    landmark_filter = None if allowed_landmarks is None else {int(value) for value in allowed_landmarks}
    resolved_target_mode = str(ct_aux_target_mode or "").strip().lower()
    endpoint_ct_dim = None
    if resolved_target_mode == "window_endpoint_heads":
        resolved_output_dim = ct_aux_output_dim if ct_aux_output_dim is not None else ct_state_output_dim
        vector_ct_dim = None
        endpoint_ct_dim = None if resolved_output_dim is None else max(int(resolved_output_dim), 1)
    elif resolved_target_mode == "window_prefix_vector_with_endpoint_lod":
        resolved_output_dim = ct_delta_output_dim if ct_delta_output_dim is not None else ct_aux_output_dim
        vector_ct_dim = None if resolved_output_dim is None else max(int(resolved_output_dim), 1)
        endpoint_ct_dim = max(int(ct_state_output_dim or 1), 1)
    else:
        resolved_output_dim = ct_aux_output_dim if ct_aux_output_dim is not None else ct_delta_output_dim
        vector_ct_dim = None if resolved_output_dim is None else max(int(resolved_output_dim), 1)
    if not resolved_target_mode:
        resolved_target_mode = "next_delta" if vector_ct_dim is None else "window_prefix_vector"
    ct_aux_window_mask = (
        None
        if vector_ct_dim is None or resolved_target_mode not in {"window_prefix_vector", "window_prefix_vector_with_endpoint_lod"}
        else _build_ct_aux_window_mask(ct_aux_window_specs, vector_ct_dim)
    )

    for tree_index in tree_indices:
        tree_time_period = int(tree_data.time_period[tree_index])
        last_landmark = min(upper_landmark, tree_time_period - 1)
        for landmark_period in range(lower_landmark, last_landmark + 1):
            if landmark_filter is not None and int(landmark_period) not in landmark_filter:
                continue
            if resolved_target_mode == "window_endpoint_heads":
                (
                    ct_state_target,
                    ct_state_mask,
                    ct_value_target,
                    ct_value_mask,
                ) = _build_ct_endpoint_targets(
                    tree_data=tree_data,
                    tree_index=int(tree_index),
                    landmark_period=landmark_period,
                    window_specs=ct_aux_window_specs,
                    output_dim=endpoint_ct_dim or 0,
                )
                ct_aux_mask = 0
                ct_delta_target = 0.0
                sample_window_mask = None
                current_ct = float("nan")
                next_ct = float("nan")
            elif resolved_target_mode == "window_prefix_vector_with_endpoint_lod":
                ct_aux_mask, ct_delta_target = _build_ct_aux_vector_target(
                    tree_data=tree_data,
                    tree_index=int(tree_index),
                    output_dim=vector_ct_dim or 0,
                    require_exact_values=True,
                )
                ct_state_target, ct_state_mask = _build_ct_endpoint_state_targets(
                    tree_data=tree_data,
                    tree_index=int(tree_index),
                    landmark_period=landmark_period,
                    window_specs=ct_aux_window_specs,
                    output_dim=endpoint_ct_dim or 0,
                )
                ct_value_target = np.zeros(endpoint_ct_dim or 0, dtype=np.float32)
                ct_value_mask = np.zeros(endpoint_ct_dim or 0, dtype=np.float32)
                current_ct = float("nan")
                next_ct = float("nan")
                sample_window_mask = ct_aux_window_mask
            elif vector_ct_dim is None:
                ct_aux_mask, ct_delta_target, current_ct, next_ct = _build_ct_aux_target(
                    tree_data=tree_data,
                    tree_index=int(tree_index),
                    landmark_period=landmark_period,
                )
                ct_state_target = 0.0
                ct_state_mask = 0
                ct_value_target = 0.0
                ct_value_mask = 0
                sample_window_mask = None
            else:
                ct_aux_mask, ct_delta_target = _build_ct_aux_vector_target(
                    tree_data=tree_data,
                    tree_index=int(tree_index),
                    output_dim=vector_ct_dim,
                    require_exact_values=False,
                )
                ct_state_target = np.zeros(vector_ct_dim, dtype=np.float32)
                ct_state_mask = np.zeros(vector_ct_dim, dtype=np.float32)
                ct_value_target = np.zeros(vector_ct_dim, dtype=np.float32)
                ct_value_mask = np.zeros(vector_ct_dim, dtype=np.float32)
                current_ct = float("nan")
                next_ct = float("nan")
                sample_window_mask = ct_aux_window_mask
            prefix_samples.append(
                PrefixSample(
                    tree_index=int(tree_index),
                    landmark_period=landmark_period,
                    seq_len_days=int(tree_data.landmark_seq_len_days[landmark_period]),
                    future_event=int(tree_data.event_flag[tree_index]),
                    ct_aux_mask=ct_aux_mask,
                    ct_delta_target=ct_delta_target,
                    ct_state_target=ct_state_target,
                    ct_state_mask=ct_state_mask,
                    ct_value_target=ct_value_target,
                    ct_value_mask=ct_value_mask,
                    ct_aux_window_mask=sample_window_mask,
                    current_ct=current_ct,
                    next_ct=next_ct,
                )
            )
    return prefix_samples


def _build_environment_features(
    env_window: pd.DataFrame,
    expected_dates: pd.DatetimeIndex,
    period_end_dates: Sequence[pd.Timestamp],
    env_feature_set: str,
    build_period_env: bool,
    period_feature_mode: str,
) -> tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray, np.ndarray]:
    renamed = env_window.rename(columns=RAW_ENV_COLUMN_MAP).copy()
    raw_frame = renamed[list(RAW_ENV_COLUMN_MAP.values())].reset_index(drop=True)
    if len(raw_frame) != len(expected_dates):
        raise ValueError("Environment frame length does not match the expected daily calendar.")

    if env_feature_set == "agro75":
        daily_frame = _build_agro_daily_feature_frame(raw_frame)
        period_base_features = AGRO_PERIOD_STAT_FEATURES
    elif env_feature_set == "basic10":
        daily_frame = _build_basic10_daily_feature_frame(raw_frame)
        period_base_features = BASIC10_ENV_FEATURES
    else:
        daily_frame = raw_frame.copy()
        period_base_features = list(daily_frame.columns)

    daily_values = daily_frame.to_numpy(dtype=np.float32)
    daily_mean = daily_values.mean(axis=0)
    daily_std = daily_values.std(axis=0)
    daily_std[daily_std == 0] = 1.0
    standardized_daily_values = (daily_values - daily_mean) / daily_std
    standardized_daily_frame = pd.DataFrame(standardized_daily_values, columns=daily_frame.columns)

    start_indices, end_indices = _compute_period_day_indices(
        period_end_dates=period_end_dates,
        expected_dates=expected_dates,
    )
    if build_period_env:
        period_values, period_feature_names, _, _ = _build_period_feature_matrix(
            daily_frame=standardized_daily_frame,
            period_end_dates=period_end_dates,
            expected_dates=expected_dates,
            base_feature_names=period_base_features,
            period_feature_mode=period_feature_mode,
            env_feature_set=env_feature_set,
        )

        period_mean = period_values.mean(axis=0)
        period_std = period_values.std(axis=0)
        period_std[period_std == 0] = 1.0
        standardized_period_values = (period_values - period_mean) / period_std
    else:
        standardized_period_values = np.zeros((len(period_end_dates), 0), dtype=np.float32)
        period_feature_names = []

    return (
        standardized_daily_values.astype(np.float32),
        list(daily_frame.columns),
        standardized_period_values.astype(np.float32),
        period_feature_names,
        start_indices,
        end_indices,
    )


def _resolve_env_feature_set(use_agro_features: bool, env_feature_set: str | None) -> str:
    normalized = "auto" if env_feature_set is None else str(env_feature_set).strip().lower()
    if normalized == "auto":
        return "agro75" if use_agro_features else "raw9"
    if normalized not in {"raw9", "agro75", "basic10"}:
        raise ValueError(f"Unsupported env_feature_set: {env_feature_set}")
    return normalized


def _build_basic10_daily_feature_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw_frame.index)
    wind_rad = np.deg2rad(raw_frame["wind_direction"].to_numpy(dtype=np.float64) % 360.0)
    frame["air_temp"] = raw_frame["air_temp"].to_numpy(dtype=np.float64)
    frame["air_humidity"] = raw_frame["air_humidity"].to_numpy(dtype=np.float64)
    frame["light_intensity"] = raw_frame["light_intensity"].to_numpy(dtype=np.float64)
    frame["wind_sin"] = np.sin(wind_rad)
    frame["wind_cos"] = np.cos(wind_rad)
    frame["wind_speed"] = raw_frame["wind_speed"].to_numpy(dtype=np.float64)
    frame["rainfall"] = raw_frame["rainfall"].to_numpy(dtype=np.float64)
    frame["soil_temp"] = raw_frame["soil_temp"].to_numpy(dtype=np.float64)
    frame["soil_moisture"] = raw_frame["soil_moisture"].to_numpy(dtype=np.float64)
    frame["soil_ec"] = raw_frame["soil_ec"].to_numpy(dtype=np.float64)
    return frame.astype(np.float32)


def _build_agro_daily_feature_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw_frame.index)
    wind_rad = np.deg2rad(raw_frame["wind_direction"].to_numpy(dtype=np.float64) % 360.0)
    rainfall = raw_frame["rainfall"].to_numpy(dtype=np.float64)
    soil_moisture = raw_frame["soil_moisture"].to_numpy(dtype=np.float64)
    soil_ec = raw_frame["soil_ec"].to_numpy(dtype=np.float64)
    light = raw_frame["light_intensity"].to_numpy(dtype=np.float64)

    frame["air_temp"] = raw_frame["air_temp"].to_numpy(dtype=np.float64)
    frame["air_humidity"] = raw_frame["air_humidity"].to_numpy(dtype=np.float64)
    frame["light_intensity"] = light
    frame["wind_sin"] = np.sin(wind_rad)
    frame["wind_cos"] = np.cos(wind_rad)
    frame["wind_speed"] = raw_frame["wind_speed"].to_numpy(dtype=np.float64)
    frame["rainfall_log1p"] = np.log1p(np.clip(rainfall, a_min=0.0, a_max=None))
    frame["soil_temp"] = raw_frame["soil_temp"].to_numpy(dtype=np.float64)
    frame["soil_moisture"] = soil_moisture
    frame["soil_ec"] = soil_ec

    for window in ROLLING_WINDOWS:
        for feature_name in ROLLING_MEAN_FEATURES:
            frame[f"{feature_name}_mean_{window}d"] = _rolling_mean(frame[feature_name].to_numpy(), window)

    for window in (7, 14, 30):
        rainfall_sum = _rolling_sum(rainfall, window)
        frame[f"rainfall_sum_{window}d_log1p"] = np.log1p(np.clip(rainfall_sum, a_min=0.0, a_max=None))

    for window in (7, 14):
        for feature_name in SLOPE_FEATURES:
            frame[f"{feature_name}_slope_{window}d"] = _rolling_slope(
                frame[feature_name].to_numpy(),
                window,
            )

    soil_moist_q25 = float(np.quantile(soil_moisture, 0.25))
    soil_moist_q75 = float(np.quantile(soil_moisture, 0.75))
    soil_ec_q75 = float(np.quantile(soil_ec, 0.75))
    light_q25 = float(np.quantile(light, 0.25))

    indicator_map = {
        "temp_25_28": ((frame["air_temp"] >= 25.0) & (frame["air_temp"] <= 28.0)).to_numpy(dtype=np.float64),
        "temp_gt_33": (frame["air_temp"] > 33.0).to_numpy(dtype=np.float64),
        "rainy_day": (rainfall > 0.5).astype(np.float64),
        "dry_day": (rainfall < 0.1).astype(np.float64),
        "low_soil_moist": (soil_moisture < soil_moist_q25).astype(np.float64),
        "high_soil_moist": (soil_moisture > soil_moist_q75).astype(np.float64),
        "high_ec": (soil_ec > soil_ec_q75).astype(np.float64),
        "low_light": (light < light_q25).astype(np.float64),
    }
    for window in COUNT_WINDOWS:
        for label, indicator in indicator_map.items():
            frame[f"{label}_count_{window}d"] = _rolling_sum(indicator, window)

    return frame.astype(np.float32)


def _build_period_feature_matrix(
    daily_frame: pd.DataFrame,
    period_end_dates: Sequence[pd.Timestamp],
    expected_dates: pd.DatetimeIndex,
    base_feature_names: Sequence[str],
    period_feature_mode: str,
    env_feature_set: str,
) -> tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    period_feature_mode = str(period_feature_mode).strip().lower()
    if period_feature_mode not in {"full", "compact", "mean_only"}:
        raise ValueError(f"Unsupported period_feature_mode: {period_feature_mode}")

    if period_feature_mode == "compact" and env_feature_set != "agro75":
        raise ValueError("period_feature_mode=compact requires use_agro_features=True.")

    if period_feature_mode == "compact":
        stat_ops = ("mean", "std", "slope")
        endpoint_feature_names = [
            feature_name
            for feature_name in COMPACT_PERIOD_ENDPOINT_FEATURES
            if feature_name in daily_frame.columns
        ]
    elif period_feature_mode == "mean_only":
        stat_ops = ("mean",)
        endpoint_feature_names = []
    else:
        stat_ops = ("mean", "std", "min", "max", "slope")
        endpoint_feature_names = list(daily_frame.columns)

    period_features: List[np.ndarray] = []
    period_feature_names: List[str] = []
    start_indices: List[int] = []
    end_indices: List[int] = []

    start_indices_array, end_indices_array = _compute_period_day_indices(
        period_end_dates=period_end_dates,
        expected_dates=expected_dates,
    )
    for period_index, (start_index, end_index) in enumerate(
        zip(start_indices_array.tolist(), end_indices_array.tolist()),
        start=1,
    ):

        period_slice = daily_frame.iloc[start_index : end_index + 1]
        row_values: List[float] = []
        row_names: List[str] = []

        for feature_name in base_feature_names:
            feature_values = period_slice[feature_name].to_numpy(dtype=np.float64)
            stat_values = {
                "mean": float(feature_values.mean()),
                "std": float(feature_values.std(ddof=0)),
                "min": float(feature_values.min()),
                "max": float(feature_values.max()),
                "slope": float(_segment_slope(feature_values)),
            }
            for stat_name in stat_ops:
                row_values.append(stat_values[stat_name])
                row_names.append(f"period_{feature_name}_{stat_name}")

        endpoint_row = daily_frame.iloc[end_index]
        for feature_name in endpoint_feature_names:
            row_values.append(float(endpoint_row[feature_name]))
            row_names.append(f"end_{feature_name}")

        if not period_feature_names:
            period_feature_names = row_names
        period_features.append(np.asarray(row_values, dtype=np.float32))
        start_indices.append(int(start_index))
        end_indices.append(int(end_index))

    return (
        np.stack(period_features, axis=0).astype(np.float32),
        period_feature_names,
        np.asarray(start_indices, dtype=np.int64),
        np.asarray(end_indices, dtype=np.int64),
    )


def _compute_period_day_indices(
    period_end_dates: Sequence[pd.Timestamp],
    expected_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    start_indices: List[int] = []
    end_indices: List[int] = []
    previous_end_index = -1
    for period_index, period_end_date in enumerate(period_end_dates, start=1):
        end_index = int((pd.Timestamp(period_end_date) - expected_dates[0]).days)
        start_index = previous_end_index + 1
        if end_index < start_index:
            raise ValueError(f"Invalid period boundary at period {period_index}.")
        start_indices.append(start_index)
        end_indices.append(end_index)
        previous_end_index = end_index
    return (
        np.asarray(start_indices, dtype=np.int64),
        np.asarray(end_indices, dtype=np.int64),
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).sum().to_numpy(dtype=np.float64)


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    slopes = np.zeros(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        slopes[index] = _segment_slope(values[start : index + 1])
    return slopes


def _segment_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())
    if denominator == 0.0:
        return 0.0
    y_centered = values - values.mean()
    return float((x_centered * y_centered).sum() / denominator)


def _parse_measurement_dates(measure_df: pd.DataFrame) -> List[pd.Timestamp]:
    first_date = pd.Timestamp(measure_df.columns[0])
    remaining_dates = pd.to_datetime(measure_df.iloc[:, 0], errors="coerce").dropna().tolist()
    measurement_dates = [first_date] + [pd.Timestamp(date) for date in remaining_dates]
    return [date.normalize() for date in measurement_dates]


def _load_ct_measurements(
    data_path: Path,
    tree_ids: np.ndarray,
    period_end_dates: Sequence[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray]:
    ct_df = pd.read_excel(data_path, sheet_name="Sheet4")
    ct_tree_col = ct_df.columns[0]
    ct_value_cols = list(ct_df.columns[1:])
    ct_dates = [_coerce_date(column_name) for column_name in ct_value_cols]

    expected_dates = [pd.Timestamp(date).normalize() for date in period_end_dates]
    if ct_dates != expected_dates:
        raise ValueError(
            "Sheet4 CT columns do not align with Sheet3 measurement dates. "
            f"Expected {expected_dates}, got {ct_dates}."
        )

    ct_df["_tree_id"] = ct_df[ct_tree_col].map(_normalize_tree_id)
    if ct_df["_tree_id"].duplicated().any():
        duplicate_ids = ct_df.loc[ct_df["_tree_id"].duplicated(), "_tree_id"].tolist()
        raise ValueError(f"Duplicate tree IDs found in Sheet4: {duplicate_ids[:5]}")

    ct_df = ct_df.set_index("_tree_id")
    missing_tree_ids = [tree_id for tree_id in tree_ids if tree_id not in ct_df.index]
    if missing_tree_ids:
        raise ValueError(
            f"Sheet4 is missing CT rows for {len(missing_tree_ids)} trees, e.g. {missing_tree_ids[:5]}"
        )

    ct_values = ct_df.loc[list(tree_ids), ct_value_cols].to_numpy(dtype=np.float32)
    ct_valid_mask = ~np.isnan(ct_values)
    ct_values = np.nan_to_num(ct_values, nan=0.0).astype(np.float32)
    return ct_values, ct_valid_mask.astype(bool)


def _build_ct_aux_window_mask(
    window_specs: Sequence[tuple[int, int]] | None,
    output_dim: int,
) -> np.ndarray:
    if output_dim <= 0:
        raise ValueError(f"CT output dimension must be positive, got {output_dim}.")
    if not window_specs:
        return np.ones((1, output_dim), dtype=np.float32)

    window_masks = []
    for landmark, horizon in window_specs:
        endpoint = int(landmark) + int(horizon)
        target_length = max(min(endpoint - 1, output_dim), 0)
        mask = np.zeros(output_dim, dtype=np.float32)
        if target_length > 0:
            mask[:target_length] = 1.0
        window_masks.append(mask)
    return np.stack(window_masks, axis=0).astype(np.float32)


def _build_ct_aux_vector_target(
    tree_data: DynamicTreeData,
    tree_index: int,
    output_dim: int,
    require_exact_values: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if tree_data.ct_values is None or tree_data.ct_valid_mask is None:
        return (
            np.zeros(output_dim, dtype=np.float32),
            np.zeros(output_dim, dtype=np.float32),
        )

    max_delta_count = max(int(tree_data.num_periods) - 1, 0)
    target = np.zeros(output_dim, dtype=np.float32)
    mask = np.zeros(output_dim, dtype=np.float32)
    usable_count = min(output_dim, max_delta_count)
    if usable_count <= 0:
        return mask, target

    current_values = tree_data.ct_values[tree_index, :usable_count]
    next_values = tree_data.ct_values[tree_index, 1 : usable_count + 1]
    current_valid = tree_data.ct_valid_mask[tree_index, :usable_count]
    next_valid = tree_data.ct_valid_mask[tree_index, 1 : usable_count + 1]
    valid_delta_mask = current_valid & next_valid
    if require_exact_values:
        valid_delta_mask &= (current_values < 40.0) & (next_values < 40.0)

    target[:usable_count] = (next_values - current_values).astype(np.float32)
    mask[:usable_count] = valid_delta_mask.astype(np.float32)
    return mask, target


def _build_ct_endpoint_state_targets(
    tree_data: DynamicTreeData,
    tree_index: int,
    landmark_period: int,
    window_specs: Sequence[tuple[int, int]] | None,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_target = np.zeros(max(int(output_dim), 0), dtype=np.float32)
    state_mask = np.zeros(max(int(output_dim), 0), dtype=np.float32)

    if output_dim <= 0 or tree_data.ct_values is None or tree_data.ct_valid_mask is None or not window_specs:
        return state_target, state_mask

    for target_index, (window_landmark, horizon) in enumerate(window_specs[:output_dim]):
        if int(window_landmark) != int(landmark_period):
            continue
        endpoint_period = int(window_landmark) + int(horizon)
        endpoint_index = endpoint_period - 1
        if endpoint_index < 0 or endpoint_index >= int(tree_data.num_periods):
            continue
        if not tree_data.ct_valid_mask[tree_index, endpoint_index]:
            continue

        endpoint_value = float(tree_data.ct_values[tree_index, endpoint_index])
        state_mask[target_index] = 1.0
        state_target[target_index] = 1.0 if endpoint_value >= 40.0 else 0.0

    return state_target, state_mask


def _build_ct_endpoint_targets(
    tree_data: DynamicTreeData,
    tree_index: int,
    landmark_period: int,
    window_specs: Sequence[tuple[int, int]] | None,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_target, state_mask = _build_ct_endpoint_state_targets(
        tree_data=tree_data,
        tree_index=tree_index,
        landmark_period=landmark_period,
        window_specs=window_specs,
        output_dim=output_dim,
    )
    value_target = np.zeros(max(int(output_dim), 0), dtype=np.float32)
    value_mask = np.zeros(max(int(output_dim), 0), dtype=np.float32)

    if output_dim <= 0 or tree_data.ct_values is None or tree_data.ct_valid_mask is None or not window_specs:
        return state_target, state_mask, value_target, value_mask

    for target_index, (window_landmark, horizon) in enumerate(window_specs[:output_dim]):
        endpoint_period = int(window_landmark) + int(horizon)
        endpoint_index = endpoint_period - 1
        if endpoint_index < 0 or endpoint_index >= int(tree_data.num_periods):
            continue
        if int(window_landmark) != int(landmark_period) or not tree_data.ct_valid_mask[tree_index, endpoint_index]:
            continue

        endpoint_value = float(tree_data.ct_values[tree_index, endpoint_index])
        if endpoint_value >= 40.0:
            continue

        value_target[target_index] = endpoint_value
        value_mask[target_index] = 1.0

    return state_target, state_mask, value_target, value_mask


def _build_day_to_period(
    day_dates: Sequence[pd.Timestamp],
    period_end_dates: Sequence[pd.Timestamp],
) -> np.ndarray:
    day_to_period = np.zeros(len(day_dates), dtype=np.int64)
    period_index = 0
    for day_index, date in enumerate(day_dates):
        while period_index < len(period_end_dates) - 1 and date > period_end_dates[period_index]:
            period_index += 1
        day_to_period[day_index] = period_index + 1
    return day_to_period


def _build_ct_aux_target(
    tree_data: DynamicTreeData,
    tree_index: int,
    landmark_period: int,
) -> tuple[int, float, float, float]:
    if tree_data.ct_values is None or tree_data.ct_valid_mask is None:
        return 0, 0.0, float("nan"), float("nan")
    if landmark_period < 1 or landmark_period >= tree_data.num_periods:
        return 0, 0.0, float("nan"), float("nan")

    current_ct_index = landmark_period - 1
    next_ct_index = landmark_period
    if not (
        tree_data.ct_valid_mask[tree_index, current_ct_index]
        and tree_data.ct_valid_mask[tree_index, next_ct_index]
    ):
        return 0, 0.0, float("nan"), float("nan")

    is_event = int(tree_data.event_flag[tree_index]) == 1
    event_period = int(tree_data.time_period[tree_index])
    if is_event and landmark_period >= event_period:
        return 0, 0.0, float("nan"), float("nan")

    current_ct = float(tree_data.ct_values[tree_index, current_ct_index])
    next_ct = float(tree_data.ct_values[tree_index, next_ct_index])
    ct_delta = float(next_ct - current_ct)
    return 1, ct_delta, current_ct, next_ct


def _safe_stratify_labels(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    value_counts = pd.Series(labels).value_counts()
    if len(value_counts) <= 1 or int(value_counts.min()) < 2:
        return None
    return labels


def _choose_stratify_labels(
    event_flag: np.ndarray,
    time_period: np.ndarray,
) -> np.ndarray | None:
    """
    Prefer event-period-aware stratification, but always fall back to event/censoring
    stratification when period bins become too sparse.
    """
    detailed_labels = make_stratification_labels(event_flag, time_period)
    detailed_labels = _safe_stratify_labels(detailed_labels)
    if detailed_labels is not None:
        return detailed_labels

    event_censor_labels = np.where(np.asarray(event_flag, dtype=np.int64) == 1, "event", "censor")
    event_censor_labels = _safe_stratify_labels(event_censor_labels)
    if event_censor_labels is not None:
        return event_censor_labels

    return None


def _coerce_date(value) -> pd.Timestamp:
    coerced = pd.Timestamp(value)
    return coerced.normalize()


def _normalize_tree_id(value) -> str:
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


def _parse_tree_id_spatial(tree_id: str) -> np.ndarray:
    parts = str(tree_id).split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse tree ID as row-column coordinates: {tree_id}")
    try:
        return np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32)
    except ValueError as error:
        raise ValueError(f"Cannot parse tree ID as row-column coordinates: {tree_id}") from error
