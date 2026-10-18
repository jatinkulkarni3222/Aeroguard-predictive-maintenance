"""
Feature Engineering Pipeline for NASA CMAPSS Predictive Maintenance.
Extracts rolling statistical features (mean, std, min, max) and linear degradation
trends across sensor telemetry channels for each engine unit.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def add_rolling_features(df: pd.DataFrame, sensors=None, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Computes rolling mean, std, min, and max for each sensor per engine (unit).
    Uses a dictionary accumulator to eliminate pandas fragmentation warnings.
    """
    if sensors is None:
        sensors = [col for col in df.columns if col.startswith("sensor_")]

    df_sorted = df.sort_values(["unit", "time_cycle"]).copy()
    rolling_cols = {}

    for sensor in sensors:
        grouped = df_sorted.groupby("unit")[sensor]
        for w in windows:
            roll = grouped.rolling(w, min_periods=1)
            rolling_cols[f"{sensor}_r{w}_mean"] = roll.mean().reset_index(level=0, drop=True)
            rolling_cols[f"{sensor}_r{w}_std"] = roll.std().fillna(0.0).reset_index(level=0, drop=True)
            rolling_cols[f"{sensor}_r{w}_min"] = roll.min().reset_index(level=0, drop=True)
            rolling_cols[f"{sensor}_r{w}_max"] = roll.max().reset_index(level=0, drop=True)

    rolling_df = pd.DataFrame(rolling_cols, index=df_sorted.index)
    return pd.concat([df_sorted, rolling_df], axis=1)


def add_trend_features(df: pd.DataFrame, sensors=None, window=20) -> pd.DataFrame:
    """
    Computes linear degradation slope (trend) over a sliding window per unit.
    Uses vectorized convolution with analytic regression kernel for high performance (~100x faster than polyfit loop).
    """
    if sensors is None:
        sensors = [col for col in df.columns if col.startswith("sensor_")]

    df_sorted = df.sort_values(["unit", "time_cycle"]).copy()

    # Precompute regression kernel weights for window length w
    w = int(window)
    x = np.arange(w, dtype=float)
    x_mean = (w - 1) / 2.0
    denom = np.sum((x - x_mean) ** 2)
    kernel = (x - x_mean) / denom

    trend_cols = {}
    for sensor in sensors:
        slopes_by_unit = []
        for _, group in df_sorted.groupby("unit", sort=False):
            vals = group[sensor].values.astype(float)
            if len(vals) < w:
                slopes_by_unit.append(np.zeros(len(vals), dtype=float))
            else:
                valid_slopes = np.convolve(vals, kernel[::-1], mode="valid")
                pad = np.zeros(w - 1, dtype=float)
                slopes_by_unit.append(np.concatenate([pad, valid_slopes]))
        trend_cols[f"{sensor}_trend"] = np.concatenate(slopes_by_unit)

    trend_df = pd.DataFrame(trend_cols, index=df_sorted.index)
    return pd.concat([df_sorted, trend_df], axis=1)


def generate_features(df: pd.DataFrame, windows=(5, 10, 20), trend_window=20) -> pd.DataFrame:
    """
    End-to-end feature generation: combines base telemetry with rolling statistics and degradation trends.
    """
    sensors = [col for col in df.columns if col.startswith("sensor_")]

    df_sorted = df.sort_values(["unit", "time_cycle"]).copy()
    feature_dict = {}

    # 1. Rolling statistics
    for sensor in sensors:
        grouped = df_sorted.groupby("unit")[sensor]
        for w in windows:
            roll = grouped.rolling(w, min_periods=1)
            feature_dict[f"{sensor}_r{w}_mean"] = roll.mean().reset_index(level=0, drop=True)
            feature_dict[f"{sensor}_r{w}_std"] = roll.std().fillna(0.0).reset_index(level=0, drop=True)
            feature_dict[f"{sensor}_r{w}_min"] = roll.min().reset_index(level=0, drop=True)
            feature_dict[f"{sensor}_r{w}_max"] = roll.max().reset_index(level=0, drop=True)

    # 2. Vectorized trend
    w = int(trend_window)
    x = np.arange(w, dtype=float)
    x_mean = (w - 1) / 2.0
    denom = np.sum((x - x_mean) ** 2)
    kernel = (x - x_mean) / denom

    for sensor in sensors:
        slopes = []
        for _, group in df_sorted.groupby("unit", sort=False):
            vals = group[sensor].values.astype(float)
            if len(vals) < w:
                slopes.append(np.zeros(len(vals), dtype=float))
            else:
                valid_slopes = np.convolve(vals, kernel[::-1], mode="valid")
                pad = np.zeros(w - 1, dtype=float)
                slopes.append(np.concatenate([pad, valid_slopes]))
        feature_dict[f"{sensor}_trend"] = np.concatenate(slopes)

    new_features_df = pd.DataFrame(feature_dict, index=df_sorted.index)
    full_df = pd.concat([df_sorted, new_features_df], axis=1)
    return full_df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    prep_path = root / "data" / "02-preprocessed" / "FD001_preprocessed.csv"
    feat_path = root / "data" / "03-features" / "FD001_features.csv"

    print(f"Loading preprocessed data from {prep_path}...")
    df_raw = pd.read_csv(prep_path)
    print(f"Generating features for {len(df_raw)} records...")
    df_feat = generate_features(df_raw)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(feat_path, index=False)
    print(f"Successfully saved {df_feat.shape[1]} columns to {feat_path}")
