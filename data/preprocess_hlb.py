"""
Preprocess hlb_dataset.xlsx into a cached tensor dict for dynamic survival analysis.

Output: data/hlb_processed.pt
    {
        'tree_ids': list[str],                      # length N
        'env_daily_raw': np.ndarray (D_total, d_env),  # shared daily env from 2024-06-01 to last measurement date
        'env_dates': list[pd.Timestamp],            # length D_total
        'period_dates': list[pd.Timestamp],         # length K+1, [baseline, d_1, ..., d_K]
        'period_bounds': np.ndarray (K, 2),         # [start, end) indices into env_daily_raw for each period
        'static_raw': np.ndarray (N, d_static),     # unscaled static features
        'static_cols': list[str],
        'T': np.ndarray (N,) int,                   # event/censoring period index in [1, K]
        'E': np.ndarray (N,) int 0/1,
        'ct_seq': np.ndarray (N, K) float,          # CT values, NaN after event period
        'ct_valid_mask': np.ndarray (N, K) bool,    # True iff CT at this period is usable for aux task
    }
"""
import argparse
import pathlib
import pandas as pd
import numpy as np
import torch

BASELINE = pd.Timestamp('2024-06-01')


def load_sheets(xlsx_path: str):
    xl = pd.ExcelFile(xlsx_path)
    sheet1 = pd.read_excel(xl, sheet_name='Sheet1')
    sheet2 = pd.read_excel(xl, sheet_name='Sheet2')
    sheet3 = pd.read_excel(xl, sheet_name='Sheet3')
    sheet4 = pd.read_excel(xl, sheet_name='Sheet4')
    return sheet1, sheet2, sheet3, sheet4


def build_period_dates(sheet3: pd.DataFrame, sheet4: pd.DataFrame) -> list:
    """Derive the canonical list of CT measurement dates.

    Sheet3's column header is the first measurement date (2024-06-22), and its 12
    rows hold the remaining 12 dates. Sheet4's column names (excluding '编号') also
    list all 13 measurement dates; we use sheet4 as the authoritative source and
    cross-check with sheet3.
    """
    s4_dates = [pd.Timestamp(c) for c in sheet4.columns if c != '编号']
    header_date = pd.Timestamp(sheet3.columns[0])
    body_dates = [pd.Timestamp(v) for v in sheet3.iloc[:, 0].tolist()]
    s3_dates = [header_date] + body_dates
    assert s4_dates == s3_dates, f"sheet3 and sheet4 date lists disagree: {s3_dates} vs {s4_dates}"
    return [BASELINE] + s4_dates  # length K+1 = 14


def build_env_daily(sheet1: pd.DataFrame, period_dates: list):
    """Filter daily env data to [BASELINE, last_date] and return (env_array, dates)."""
    df = sheet1.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df[(df['日期'] >= BASELINE) & (df['日期'] <= period_dates[-1])]
    df = df.sort_values('日期').reset_index(drop=True)

    expected_days = (period_dates[-1] - BASELINE).days + 1
    assert len(df) == expected_days, (
        f"Daily env has {len(df)} rows but expected {expected_days} (one per day)"
    )

    env_cols = ['空气温度', '空气湿度', '光照强度', '风向', '风速',
                '日累计雨量', '土壤温度', '土壤水分', '土壤EC值']
    env_array = df[env_cols].values.astype(np.float32)
    dates = df['日期'].tolist()
    return env_array, dates, env_cols


def build_period_bounds(env_dates: list, period_dates: list) -> np.ndarray:
    """For each period k = 1..K, return [start, end) indices into env_dates.

    Period k covers the daily env rows with dates in (d_{k-1}, d_k] (left-open,
    right-closed). Baseline day (2024-06-01) is excluded from any period because
    its role is just to mark the starting CT-healthy state.
    """
    env_dates_np = pd.to_datetime(pd.Series(env_dates)).values  # np.datetime64
    K = len(period_dates) - 1
    bounds = np.zeros((K, 2), dtype=np.int64)
    for k in range(1, K + 1):
        lo = period_dates[k - 1]
        hi = period_dates[k]
        mask = (env_dates_np > np.datetime64(lo)) & (env_dates_np <= np.datetime64(hi))
        idx = np.where(mask)[0]
        assert len(idx) > 0, f"period {k} has no env days"
        bounds[k - 1] = [idx[0], idx[-1] + 1]
    return bounds


def build_static(sheet2: pd.DataFrame):
    """Extract static features from sheet2.

    Drops: 编号 (id), 首次发病日期 (label), 观测期内发病 (label).
    Returns unscaled raw array and column names.
    """
    drop_cols = ['编号', '首次发病日期', '观测期内发病']
    static_cols = [c for c in sheet2.columns if c not in drop_cols]
    raw = sheet2[static_cols].copy()
    # 树龄 may have missing values (per earlier inspection, but we guard anyway)
    for c in static_cols:
        if raw[c].dtype.kind in ('f', 'i'):
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    return raw.values.astype(np.float32), static_cols


def build_labels(sheet2: pd.DataFrame, period_dates: list):
    """Return T (period index, 1..K) and E (0/1) per tree.

    T_i = index k such that period_dates[k] == 首次发病日期 (for E_i=1)
    T_i = K (last period) for E_i=0
    """
    K = len(period_dates) - 1
    measurement_dates = period_dates[1:]  # d_1..d_K
    date_to_idx = {d: i + 1 for i, d in enumerate(measurement_dates)}  # 1-based

    T = np.zeros(len(sheet2), dtype=np.int64)
    E = sheet2['观测期内发病'].values.astype(np.int64)
    for i, row in sheet2.iterrows():
        if E[i] == 1:
            dt = pd.to_datetime(row['首次发病日期'])
            assert dt in date_to_idx, f"tree {row['编号']} 首次发病日期 {dt} not in measurement dates"
            T[i] = date_to_idx[dt]
        else:
            T[i] = K
    return T, E


def build_ct(sheet2: pd.DataFrame, sheet4: pd.DataFrame, T: np.ndarray, E: np.ndarray):
    """Construct per-tree CT sequence and validity mask for the auxiliary task.

    ct_seq[i, k-1] = sheet4 value at period k's measurement date, but cleared to
    NaN for k >= T_i when E_i=1 (never use post-event CT).

    ct_valid_mask[i, k-1] = True iff ct_seq[i, k-1] is finite AND k <= T_i (or E_i=0).
    Used by the auxiliary loss only when both k and k+1 are valid.
    """
    # align sheet2 and sheet4 by 编号
    ids_s2 = sheet2['编号'].tolist()
    ids_s4 = sheet4['编号'].tolist()
    assert ids_s2 == ids_s4, "sheet2 and sheet4 tree order differ"

    date_cols = [c for c in sheet4.columns if c != '编号']
    K = len(date_cols)
    ct = sheet4[date_cols].values.astype(np.float32)  # (N, K)

    # mask post-event values for event trees
    for i in range(len(ct)):
        if E[i] == 1:
            # T_i is the event period (1-based). CT at period T_i is the one that triggered
            # detection (<30); we keep it (it IS the event signal), but clear everything after.
            # For the auxiliary task we only use pairs (k, k+1) with BOTH <= T_i so the post-
            # event pair is never supervised.
            ct[i, T[i]:] = np.nan

    valid = np.isfinite(ct)
    return ct, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', default='data/hlb_dataset.xlsx')
    parser.add_argument('--out', default='data/hlb_processed.pt')
    args = parser.parse_args()

    xlsx_path = pathlib.Path(args.xlsx)
    assert xlsx_path.exists(), f"{xlsx_path} not found"

    print(f"[1/6] loading {xlsx_path}")
    sheet1, sheet2, sheet3, sheet4 = load_sheets(str(xlsx_path))

    print("[2/6] building period dates")
    period_dates = build_period_dates(sheet3, sheet4)
    K = len(period_dates) - 1
    print(f"      K = {K} periods, baseline = {period_dates[0].date()}, last = {period_dates[-1].date()}")
    period_days = [(period_dates[k] - period_dates[k - 1]).days for k in range(1, K + 1)]
    print(f"      period lengths (days): {period_days}")

    print("[3/6] building daily env series")
    env_daily_raw, env_dates, env_cols = build_env_daily(sheet1, period_dates)
    print(f"      env_daily shape = {env_daily_raw.shape}, cols = {env_cols}")

    print("[4/6] building period bounds into env array")
    period_bounds = build_period_bounds(env_dates, period_dates)

    print("[5/6] building static, labels, CT")
    static_raw, static_cols = build_static(sheet2)
    T, E = build_labels(sheet2, period_dates)
    ct_seq, ct_valid_mask = build_ct(sheet2, sheet4, T, E)
    tree_ids = sheet2['编号'].astype(str).tolist()

    N = len(tree_ids)
    n_events = int(E.sum())
    print(f"      N = {N} trees, events = {n_events}, censored = {N - n_events}")
    print(f"      T distribution: min={T.min()}, max={T.max()}, unique T for events = "
          f"{sorted(set(T[E == 1].tolist()))}")
    print(f"      static_cols = {static_cols}")

    # sanity: for events, CT at T_i should be <30 (but some may be filled to 40 placeholder);
    # just warn if unusual.
    event_ct_at_T = [ct_seq[i, T[i] - 1] for i in range(N) if E[i] == 1 and not np.isnan(ct_seq[i, T[i] - 1])]
    if event_ct_at_T:
        mx = max(event_ct_at_T)
        if mx >= 30:
            print(f"      [warn] max CT at event period for event trees = {mx:.2f} (>=30); "
                  f"check label alignment")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'tree_ids': tree_ids,
        'env_daily_raw': env_daily_raw,
        'env_dates': env_dates,
        'period_dates': period_dates,
        'period_bounds': period_bounds,
        'static_raw': static_raw,
        'static_cols': static_cols,
        'env_cols': env_cols,
        'T': T,
        'E': E,
        'ct_seq': ct_seq,
        'ct_valid_mask': ct_valid_mask,
        'baseline': BASELINE,
        'K': K,
    }, str(out_path))
    print(f"[6/6] saved to {out_path}")


if __name__ == '__main__':
    main()
