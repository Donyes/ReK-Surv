"""
Evaluation utilities for dynamic time-dependent survival metrics.
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter


def _censoring_survival(train_times: np.ndarray, train_events: np.ndarray) -> np.ndarray:
    train_times = np.asarray(train_times).reshape(-1)
    train_events = np.asarray(train_events).reshape(-1)

    kmf = KaplanMeierFitter()
    kmf.fit(train_times, event_observed=(train_events == 0).astype(int))
    survival_curve = np.asarray(kmf.survival_function_.reset_index()).transpose()
    non_zero = survival_curve[1, survival_curve[1, :] != 0]
    if len(non_zero) > 0:
        survival_curve[1, survival_curve[1, :] == 0] = non_zero[-1]
    return survival_curve


def weighted_c_index(
    train_times: np.ndarray,
    train_events: np.ndarray,
    prediction: np.ndarray,
    test_times: np.ndarray,
    test_events: np.ndarray,
    horizon: int,
) -> float:
    """
    IPCW time-dependent concordance index for event risk by a fixed horizon.
    """
    censoring_curve = _censoring_survival(train_times, train_events)

    prediction = np.asarray(prediction).reshape(-1)
    test_times = np.asarray(test_times).reshape(-1)
    test_events = np.asarray(test_events).reshape(-1)
    num_samples = len(prediction)

    comparable = np.zeros((num_samples, num_samples), dtype=np.float64)
    ordered = np.zeros((num_samples, num_samples), dtype=np.float64)
    event_before_horizon = np.zeros((num_samples, num_samples), dtype=np.float64)

    for row_index in range(num_samples):
        candidate_index = np.where(censoring_curve[0, :] >= test_times[row_index])[0]
        if len(candidate_index) == 0:
            weight = (1.0 / censoring_curve[1, -1]) ** 2
        else:
            weight = (1.0 / censoring_curve[1, candidate_index[0]]) ** 2

        comparable[row_index, np.where(test_times[row_index] < test_times)[0]] = weight
        ordered[row_index, np.where(prediction[row_index] > prediction)[0]] = 1.0
        if test_times[row_index] <= horizon and test_events[row_index] == 1:
            event_before_horizon[row_index, :] = 1.0

    numerator = np.sum((comparable * event_before_horizon) * ordered)
    denominator = np.sum(comparable * event_before_horizon)
    if numerator == 0 and denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def weighted_brier_score(
    train_times: np.ndarray,
    train_events: np.ndarray,
    prediction: np.ndarray,
    test_times: np.ndarray,
    test_events: np.ndarray,
    horizon: int,
) -> float:
    """
    IPCW time-dependent Brier score for event risk by a fixed horizon.
    """
    censoring_curve = _censoring_survival(train_times, train_events)

    prediction = np.asarray(prediction).reshape(-1)
    test_times = np.asarray(test_times).reshape(-1)
    test_events = np.asarray(test_events).reshape(-1)

    num_samples = len(prediction)
    weights = np.zeros(num_samples, dtype=np.float64)
    survival_indicator = (test_times > horizon).astype(np.float64)

    for sample_index in range(num_samples):
        at_time_index = np.where(censoring_curve[0, :] >= test_times[sample_index])[0]
        at_horizon_index = np.where(censoring_curve[0, :] >= horizon)[0]

        if len(at_time_index) == 0:
            g_at_time = censoring_curve[1, -1]
        else:
            g_at_time = censoring_curve[1, at_time_index[0]]

        if len(at_horizon_index) == 0:
            g_at_horizon = censoring_curve[1, -1]
        else:
            g_at_horizon = censoring_curve[1, at_horizon_index[0]]

        weights[sample_index] = (
            (1.0 - survival_indicator[sample_index]) * float(test_events[sample_index]) / g_at_time
            + survival_indicator[sample_index] / g_at_horizon
        )

    y_true = ((test_times <= horizon) * test_events).astype(np.float64)
    return float(np.mean(weights * (prediction - y_true) ** 2))
