"""Unit tests for Phase 1 Data Quality and Resampling."""

import numpy as np
import pandas as pd
import pytest

from src.data.quality import check_data_quality, classify_time_gap
from src.data.resample import resample_m1_to_m15


def test_ohlc_consistency_valid():
    """Verify that a valid OHLC dataset passes with 0 invalid records."""
    dates = pd.date_range("2023-01-02 00:00", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.0500, 1.0510, 1.0520, 1.0515, 1.0530, 1.0525, 1.0540, 1.0535, 1.0550, 1.0545],
            "high": [1.0515, 1.0525, 1.0530, 1.0535, 1.0545, 1.0540, 1.0555, 1.0550, 1.0565, 1.0560],
            "low": [1.0495, 1.0505, 1.0510, 1.0505, 1.0520, 1.0515, 1.0530, 1.0525, 1.0540, 1.0535],
            "close": [1.0510, 1.0520, 1.0515, 1.0530, 1.0525, 1.0540, 1.0535, 1.0550, 1.0545, 1.0555],
            "tick_volume": [100] * 10,
        }
    )

    report = check_data_quality(df)
    assert report.invalid_ohlc_count == 0
    assert report.is_monotonic_increasing is True
    assert report.duplicate_timestamps == 0
    assert report.total_bars == 10
    assert report.total_gaps == 0


def test_ohlc_consistency_violations():
    """Verify that all types of invalid OHLC data are properly flagged."""
    dates = pd.date_range("2023-01-02 00:00", periods=5, freq="15min", tz="UTC")
    # Row 0: Valid
    # Row 1: High < Low
    # Row 2: High < Open
    # Row 3: Low > Close
    # Row 4: Negative price
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.0500, 1.0500, 1.0600, 1.0500, -1.0500],
            "high": [1.0550, 1.0400, 1.0500, 1.0600, 1.0600],
            "low": [1.0450, 1.0500, 1.0400, 1.0550, 1.0400],
            "close": [1.0520, 1.0450, 1.0450, 1.0400, 1.0500],
        }
    )

    report = check_data_quality(df)
    assert report.invalid_ohlc_count >= 4
    issues = [item["issue"] for item in report.invalid_ohlc_details]
    assert any("High" in s and "Low" in s for s in issues)
    assert any("max(Open, Close)" in s for s in issues)
    assert any("min(Open, Close)" in s for s in issues)
    assert any("Non-positive" in s for s in issues)


def test_timestamp_ordering_and_duplicates():
    """Verify detection of unordered timestamps and duplicates."""
    dates = [
        pd.Timestamp("2023-01-02 00:00", tz="UTC"),
        pd.Timestamp("2023-01-02 00:15", tz="UTC"),
        pd.Timestamp("2023-01-02 00:15", tz="UTC"),  # Duplicate
        pd.Timestamp("2023-01-02 00:10", tz="UTC"),  # Out of order
    ]
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.05, 1.05, 1.05, 1.05],
            "high": [1.06, 1.06, 1.06, 1.06],
            "low": [1.04, 1.04, 1.04, 1.04],
            "close": [1.05, 1.05, 1.05, 1.05],
        }
    )

    report = check_data_quality(df)
    assert report.is_monotonic_increasing is False
    assert report.duplicate_timestamps == 1


def test_gap_classification():
    """Verify exact categorization of gaps into Class 1, 2, 3 (Weekend/Holiday)."""
    # 1. Weekend closure (Friday 21:45 to Sunday 22:00)
    fri = pd.Timestamp("2023-01-06 21:45", tz="UTC")
    sun = pd.Timestamp("2023-01-08 22:00", tz="UTC")
    gap_cls, desc, missing = classify_time_gap(fri, sun)
    assert "Class 3 (Weekend Closure)" in gap_cls
    assert missing > 100

    # 2. Single missing bar in active session (Wednesday 10:00 to 10:30)
    wed_1 = pd.Timestamp("2023-01-11 10:00", tz="UTC")
    wed_2 = pd.Timestamp("2023-01-11 10:30", tz="UTC")
    gap_cls, desc, missing = classify_time_gap(wed_1, wed_2)
    assert "Class 1" in gap_cls
    assert missing == 1

    # 3. Multiple missing bars in active session (Wednesday 10:00 to 11:30)
    wed_3 = pd.Timestamp("2023-01-11 11:30", tz="UTC")
    gap_cls, desc, missing = classify_time_gap(wed_1, wed_3)
    assert "Class 2" in gap_cls
    assert missing == 5  # 90 mins / 15 = 6, 6 - 1 = 5

    # 4. Holiday closure (Christmas)
    xmas_1 = pd.Timestamp("2023-12-25 04:00", tz="UTC")
    xmas_2 = pd.Timestamp("2023-12-25 18:00", tz="UTC")
    gap_cls, desc, missing = classify_time_gap(xmas_1, xmas_2)
    assert "Class 3 (Holiday Closure)" in gap_cls


def test_resample_strict_no_fabrication():
    """Verify that resampling does NOT fabricate candles or interpolate gaps."""
    # Create two disjoint clusters of M1 bars:
    # Cluster 1: 10:00 to 10:14 (15 bars) -> 1 M15 bar at 10:00
    # Cluster 2: 11:00 to 11:14 (15 bars) -> 1 M15 bar at 11:00
    # Gap: 10:15, 10:30, 10:45 should be completely absent!
    c1 = pd.date_range("2023-01-11 10:00", periods=15, freq="1min", tz="UTC")
    c2 = pd.date_range("2023-01-11 11:00", periods=15, freq="1min", tz="UTC")
    all_dates = c1.append(c2)

    df_m1 = pd.DataFrame(
        {
            "timestamp": all_dates,
            "open": np.linspace(1.05, 1.06, 30),
            "high": np.linspace(1.051, 1.061, 30),
            "low": np.linspace(1.049, 1.059, 30),
            "close": np.linspace(1.0505, 1.0605, 30),
            "volume": np.zeros(30),
        }
    )

    df_m15 = resample_m1_to_m15(df_m1)
    # Must have exactly 2 bars, NOT 5 bars!
    assert len(df_m15) == 2
    assert df_m15["timestamp"].iloc[0] == pd.Timestamp("2023-01-11 10:00", tz="UTC")
    assert df_m15["timestamp"].iloc[1] == pd.Timestamp("2023-01-11 11:00", tz="UTC")
    assert (df_m15["active_m1_bars"] == 15).all()

    # Verify quality check detects this gap as Class 2 (Multiple missing bars)
    report = check_data_quality(df_m15)
    assert report.total_gaps == 1
    assert report.class_2_gaps_count == 1
    assert report.class_2_missing_bars == 3  # 10:15, 10:30, 10:45


def test_resample_completeness_flag():
    """Verify that complete and partial bars are properly distinguished with is_complete."""
    # Bar 1: Complete 15 M1 bars (10:00 to 10:14)
    # Bar 2: Partial with only 7 M1 bars (10:15 to 10:21)
    dates_full = pd.date_range("2023-01-11 10:00", periods=15, freq="1min", tz="UTC")
    dates_part = pd.date_range("2023-01-11 10:15", periods=7, freq="1min", tz="UTC")
    all_dates = dates_full.append(dates_part)

    df_m1 = pd.DataFrame(
        {
            "timestamp": all_dates,
            "open": [1.0500] * 15 + [1.0520] * 7,
            "high": [1.0510] * 15 + [1.0530] * 7,
            "low": [1.0490] * 15 + [1.0510] * 7,
            "close": [1.0505] * 15 + [1.0525] * 7,
            "volume": [0.0] * 22,
        }
    )

    df_m15 = resample_m1_to_m15(df_m1)
    assert len(df_m15) == 2
    assert "is_complete" in df_m15.columns

    # Bar 1 checks
    assert df_m15.loc[0, "active_m1_bars"] == 15
    assert df_m15.loc[0, "is_complete"] == True
    assert df_m15.loc[0, "open"] == 1.0500
    assert df_m15.loc[0, "close"] == 1.0505

    # Bar 2 checks (partial)
    assert df_m15.loc[1, "active_m1_bars"] == 7
    assert df_m15.loc[1, "is_complete"] == False
    assert df_m15.loc[1, "open"] == 1.0520
    assert df_m15.loc[1, "high"] == 1.0530
    assert df_m15.loc[1, "low"] == 1.0510
    assert df_m15.loc[1, "close"] == 1.0525

    # Quality report checks
    report = check_data_quality(df_m15)
    assert report.complete_bars_count == 1
    assert report.partial_bars_count == 1
    assert report.active_m1_bars_distribution == {7: 1, 15: 1}
