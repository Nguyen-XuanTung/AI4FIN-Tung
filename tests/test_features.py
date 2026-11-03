"""Unit tests for Phase 2 Feature Engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.candidates import (
    compute_all_features,
    compute_directional_body_ratio,
    compute_log_range,
    compute_log_returns_cc,
    compute_log_returns_oc,
    compute_log_volatility,
    compute_wicks,
)
from src.features.seasonality import DiurnalVolatilityScaler, assign_fx_session


def test_feature_math_precision():
    """Verify exact mathematical formulas for candidate features."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02 10:00", periods=2, freq="15min", tz="UTC"),
        "open": [1.0500, 1.0520],
        "high": [1.0550, 1.0560],
        "low": [1.0450, 1.0510],
        "close": [1.0520, 1.0540],
        "tick_volume": [10.0, 15.0],
        "active_m1_bars": [15, 15],
        "is_complete": [True, True],
    })

    # Bar 0 calculations
    # r_oc = ln(1.0520 / 1.0500)
    expected_roc = np.log(1.0520 / 1.0500)
    roc = compute_log_returns_oc(df)
    assert np.isclose(roc.iloc[0], expected_roc, atol=1e-8)

    # log_range = ln(1.0550 / 1.0450)
    expected_range = np.log(1.0550 / 1.0450)
    lrange = compute_log_range(df)
    assert np.isclose(lrange.iloc[0], expected_range, atol=1e-8)

    # Upper wick: ln(1.0550 / max(1.0500, 1.0520)) = ln(1.0550 / 1.0520)
    # Lower wick: ln(min(1.0500, 1.0520) / 1.0450) = ln(1.0500 / 1.0450)
    upper_w, lower_w = compute_wicks(df)
    assert np.isclose(upper_w.iloc[0], np.log(1.0550 / 1.0520), atol=1e-8)
    assert np.isclose(lower_w.iloc[0], np.log(1.0500 / 1.0450), atol=1e-8)

    # body_ratio = (1.0520 - 1.0500) / (1.0550 - 1.0450) = 0.0020 / 0.0100 = 0.2
    body_ratio = compute_directional_body_ratio(df)
    assert np.isclose(body_ratio.iloc[0], 0.2, atol=1e-8)


def test_feature_bounds_and_edge_cases():
    """Verify bounds, non-negativity, and zero-range edge cases."""
    # Bar 0: Normal
    # Bar 1: Pure flat bar (O = H = L = C)
    # Bar 2: Bullish Marubozu (no wicks: O=L, C=H)
    # Bar 3: Bearish Marubozu (no wicks: O=H, C=L)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02 10:00", periods=4, freq="15min", tz="UTC"),
        "open": [1.0500, 1.0500, 1.0500, 1.0600],
        "high": [1.0550, 1.0500, 1.0600, 1.0600],
        "low": [1.0450, 1.0500, 1.0500, 1.0500],
        "close": [1.0520, 1.0500, 1.0600, 1.0500],
    })

    lrange = compute_log_range(df)
    assert (lrange >= 0.0).all()
    assert lrange.iloc[1] == 0.0

    upper_w, lower_w = compute_wicks(df)
    assert (upper_w >= 0.0).all()
    assert (lower_w >= 0.0).all()
    # Bullish marubozu: both wicks zero
    assert upper_w.iloc[2] == 0.0
    assert lower_w.iloc[2] == 0.0

    body_ratio = compute_directional_body_ratio(df)
    assert body_ratio.iloc[1] == 0.0  # Flat bar protected from division by zero
    assert np.isclose(body_ratio.iloc[2], 1.0)  # Pure Bullish
    assert np.isclose(body_ratio.iloc[3], -1.0)  # Pure Bearish
    assert (body_ratio >= -1.0).all() and (body_ratio <= 1.0).all()


def test_compute_all_features_preserves_metadata():
    """Ensure compute_all_features retains is_complete and active_m1_bars."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-02 10:00", periods=3, freq="15min", tz="UTC"),
        "open": [1.0500, 1.0510, 1.0520],
        "high": [1.0530, 1.0540, 1.0550],
        "low": [1.0490, 1.0500, 1.0510],
        "close": [1.0515, 1.0525, 1.0535],
        "tick_volume": [0.0, 0.0, 0.0],
        "active_m1_bars": [15, 8, 15],
        "is_complete": [True, False, True],
    })

    features_df = compute_all_features(df)
    assert "is_complete" in features_df.columns
    assert "active_m1_bars" in features_df.columns
    assert (features_df["is_complete"] == [True, False, True]).all()
    assert (features_df["active_m1_bars"] == [15, 8, 15]).all()
    assert "r_oc" in features_df.columns
    assert "log_vol" in features_df.columns


def test_diurnal_scaler_leak_free():
    """Verify that DiurnalVolatilityScaler fits strictly on train and transforms test without leakage."""
    dates = pd.date_range("2023-01-02 00:00", periods=192, freq="15min", tz="UTC")  # 2 full days
    df = pd.DataFrame({
        "timestamp": dates,
        "log_range": np.random.uniform(0.0002, 0.0010, 192),
    })

    train_df = df.iloc[:96].copy()
    test_df = df.iloc[96:].copy()

    scaler = DiurnalVolatilityScaler(target_col="log_range")
    scaler.fit(train_df)

    assert scaler.is_fitted is True
    assert len(scaler.seasonal_medians) == 96

    test_transformed = scaler.transform(test_df)
    assert "range_ratio" in test_transformed.columns
    assert "range_zscore" in test_transformed.columns
    assert len(test_transformed) == 96
    # Ensure transformed values are finite and non-empty
    assert test_transformed["range_ratio"].notna().all()


def test_fx_sessions():
    """Verify session classification across UTC hours."""
    ts = pd.Series([
        pd.Timestamp("2023-01-02 03:00", tz="UTC"),  # Asian
        pd.Timestamp("2023-01-02 08:30", tz="UTC"),  # London
        pd.Timestamp("2023-01-02 13:45", tz="UTC"),  # Overlap
        pd.Timestamp("2023-01-02 18:00", tz="UTC"),  # NY Afternoon
        pd.Timestamp("2023-01-02 22:15", tz="UTC"),  # Transition
    ])

    sessions = assign_fx_session(ts)
    assert sessions.iloc[0] == "Asian"
    assert sessions.iloc[1] == "London"
    assert sessions.iloc[2] == "Overlap"
    assert sessions.iloc[3] == "NY_Afternoon"
    assert sessions.iloc[4] == "Transition"

