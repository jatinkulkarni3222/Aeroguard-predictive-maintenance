"""
MLOps Unit Tests: Feature Engineering Pipeline Integrity.
Validates temporal rolling features, trend extraction, and shape invariants.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.pipelines.feature_pipeline import generate_features, add_rolling_features, add_trend_features


@pytest.fixture
def sample_telemetry_data():
    """Generates mock multivariate sensor telemetry for 2 engine units."""
    rows = []
    for unit in [1, 2]:
        for cycle in range(1, 35):
            row = {
                "unit": unit,
                "time_cycle": cycle,
                "setting1": 0.001 * cycle,
                "setting2": -0.0002,
                "setting3": 100.0,
            }
            for s in range(1, 22):
                row[f"sensor_{s}"] = 500.0 + s * 10.0 + 0.5 * cycle
            rows.append(row)
    return pd.DataFrame(rows)


def test_add_rolling_features(sample_telemetry_data):
    """Verifies that rolling features produce correct columns and no NaN/nulls."""
    df_roll = add_rolling_features(sample_telemetry_data, windows=[5, 10, 20])
    
    # 21 sensors * 3 windows * 4 stats = 252 rolling columns + 26 base = 278
    assert df_roll.shape[1] == 26 + 252
    assert "sensor_11_r5_mean" in df_roll.columns
    assert "sensor_4_r10_std" in df_roll.columns
    assert df_roll["sensor_11_r5_mean"].isnull().sum() == 0


def test_add_trend_features(sample_telemetry_data):
    """Verifies 1D convolution trend slopes for linear signals."""
    df_trend = add_trend_features(sample_telemetry_data, window=20)
    assert "sensor_11_trend" in df_trend.columns
    assert df_trend.shape[0] == len(sample_telemetry_data)
    # Slope for synthetic signal with rate +0.5 per cycle should be positive
    assert df_trend["sensor_11_trend"].iloc[-1] > 0.0


def test_generate_features_output_schema(sample_telemetry_data):
    """Validates end-to-end feature pipeline execution and total feature count."""
    df_feat = generate_features(sample_telemetry_data)
    # 26 base + 252 rolling + 21 trend = 299 columns
    assert df_feat.shape[1] == 299
    assert not df_feat.empty
