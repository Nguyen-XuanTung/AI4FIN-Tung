"""
tests/test_visualization.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Phase 8: Historical Regime Visualization module.

Verifies all 10 mandatory scientific and engineering requirements:
1. Probability vectors sum to 1.0 within numerical precision.
2. No probabilities outside [0, 1].
3. Timestamps are strictly monotonically increasing.
4. Join between OHLC and posterior does not drop or duplicate records.
5. Missing bars are not interpolated (no forward-fill).
6. No synthetic candles exist in visual data feeds.
7. Regime label mapping strictly matches model state IDs (0: Low, 1: High, 2: Med).
8. Entropy handles p=0 and uniform distributions without NaN/Inf, norm <= 1.0.
9. Transition counts do not count across session or market gap boundaries.
10. Retrospective full-year and as-of replay datasets are strictly segregated.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.visualization.data_loader import (
    STATE_LABELS,
    STATE_COLORS,
    load_canonical_model,
    load_as_of_replay_data,
    load_full_year_data,
    compute_entropy_series,
    extract_continuous_segments,
    compute_transition_and_run_lengths,
    validate_probability_frame,
)


@pytest.fixture(scope="module")
def replay_df():
    """Load Mode B (As-Of Replay) data once for testing."""
    return load_as_of_replay_data()


@pytest.fixture(scope="module")
def full_year_df():
    """Load Mode A (Full-Year Retrospective) data once for testing."""
    return load_full_year_data()


def test_01_probability_sum_unity(replay_df, full_year_df):
    """1. Probability vectors sum to 1.0 within numerical precision."""
    for name, df in [("Replay", replay_df), ("FullYear", full_year_df)]:
        probs = df[["prob_state_0", "prob_state_1", "prob_state_2"]].values
        sums = probs.sum(axis=1)
        max_err = np.max(np.abs(sums - 1.0))
        assert max_err < 1e-10, f"{name}: Probability sum deviates from 1.0 (max err: {max_err:.2e})"


def test_02_probabilities_in_unit_interval(replay_df, full_year_df):
    """2. No probability outside [0, 1]."""
    for name, df in [("Replay", replay_df), ("FullYear", full_year_df)]:
        probs = df[["prob_state_0", "prob_state_1", "prob_state_2"]].values
        assert np.all(probs >= 0.0), f"{name}: Negative probability detected"
        assert np.all(probs <= 1.0), f"{name}: Probability > 1.0 detected"


def test_03_timestamp_monotonicity(replay_df, full_year_df):
    """3. Timestamps are strictly monotonically increasing."""
    for name, df in [("Replay", replay_df), ("FullYear", full_year_df)]:
        ts = pd.to_datetime(df["timestamp"])
        assert ts.is_monotonic_increasing, f"{name}: Timestamps are not monotonically increasing"
        # Zero duplicate timestamps
        assert len(ts) == len(ts.drop_duplicates()), f"{name}: Duplicate timestamps detected"


def test_04_join_preserves_record_counts(replay_df):
    """4. Join between OHLC and posterior does not drop or duplicate records."""
    raw_replay = pd.read_parquet("data/processed/as_of_replay_inference_log.parquet")
    complete_raw = raw_replay[(raw_replay["is_complete"] == True) & (raw_replay["status"] == "valid_update")]
    assert len(replay_df) == len(complete_raw) == 8335, f"Expected 8335 records, got {len(replay_df)}"
    # Check that key OHLC columns exist and have non-null values
    for col in ["open", "high", "low", "close", "tick_volume"]:
        assert col in replay_df.columns
        assert replay_df[col].isna().sum() == 0, f"NaN found in OHLC column {col}"


def test_05_missing_bars_not_interpolated():
    """5. Missing bars are not interpolated (no forward-fill)."""
    # Create sample dataframe with a 2-hour missing gap
    ts = [
        pd.Timestamp("2023-05-10 12:00:00+00:00"),
        pd.Timestamp("2023-05-10 12:15:00+00:00"),
        # Gap of 2 hours
        pd.Timestamp("2023-05-10 14:15:00+00:00"),
        pd.Timestamp("2023-05-10 14:30:00+00:00"),
    ]
    df = pd.DataFrame({"timestamp": ts, "close": [1.1000, 1.1005, 1.1020, 1.1025]})
    segments = extract_continuous_segments(df, max_gap_seconds=900)
    # Must be split into 2 separate segments
    assert len(segments) == 2, f"Expected 2 segments due to 2-hour gap, got {len(segments)}"
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_06_no_synthetic_candles(full_year_df):
    """6. No synthetic candles exist (all records trace to raw complete parquet)."""
    raw_df = pd.read_parquet("data/processed/EURUSD_M15_2023.parquet")
    raw_complete = raw_df[raw_df["is_complete"] == True]
    assert len(full_year_df) == len(raw_complete) == 20836, (
        f"Full-year record count ({len(full_year_df)}) does not match raw complete count ({len(raw_complete)})"
    )


def test_07_regime_label_mapping_consistency():
    """7. Regime label mapping strictly matches model state ID."""
    assert STATE_LABELS[0] == "LowVol_Neutral"
    assert STATE_LABELS[1] == "HighVol_Neutral"
    assert STATE_LABELS[2] == "MedVol_Neutral"

    # Verify colors exist and are valid hex codes
    for s_id in [0, 1, 2]:
        col = STATE_COLORS[s_id]
        assert col.startswith("#") and len(col) == 7


def test_08_entropy_math_and_zero_handling():
    """8. Entropy handles zero probabilities correctly without NaN/Inf, norm <= 1.0."""
    # Case A: Degenerate certain belief [1, 0, 0] -> H = 0
    p_certain = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ent_cert, n_ent_cert = compute_entropy_series(p_certain, n_states=3)
    assert np.allclose(ent_cert, 0.0, atol=1e-12)
    assert np.allclose(n_ent_cert, 0.0, atol=1e-12)

    # Case B: Maximum ambiguity uniform belief [1/3, 1/3, 1/3] -> H = ln(3), norm = 1.0
    p_uniform = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    ent_unif, n_ent_unif = compute_entropy_series(p_uniform, n_states=3)
    assert np.allclose(ent_unif, np.log(3.0), atol=1e-12)
    assert np.allclose(n_ent_unif, 1.0, atol=1e-12)

    # Case C: Boundary near-zero [0.9999, 0.0001, 0.0] -> Finite, non-negative, no NaN
    p_edge = np.array([[0.9999, 0.0001, 0.0]])
    ent_edge, n_ent_edge = compute_entropy_series(p_edge, n_states=3)
    assert not np.isnan(ent_edge[0])
    assert not np.isinf(ent_edge[0])
    assert 0.0 < n_ent_edge[0] < 0.1


def test_09_transition_counts_zero_gap_bleed():
    """9. Transition counts do not count across gap boundaries."""
    # 2 bars in State 0 on Friday, then weekend break, then 2 bars in State 1 on Sunday
    ts = [
        pd.Timestamp("2023-05-12 21:30:00+00:00"),  # State 0
        pd.Timestamp("2023-05-12 21:45:00+00:00"),  # State 0
        # Weekend break (48 hours)
        pd.Timestamp("2023-05-14 22:00:00+00:00"),  # State 1
        pd.Timestamp("2023-05-14 22:15:00+00:00"),  # State 1
    ]
    df = pd.DataFrame({"timestamp": ts, "pred_state": [0, 0, 1, 1]})
    stats = compute_transition_and_run_lengths(df, max_gap_seconds=900)

    # Expected: 1 self-transition 0->0, 1 self-transition 1->1.
    # Total adjacent pairs = 2.
    # Total switches = 0 (the transition 0 -> 1 occurred ACROSS the weekend break and must NOT be counted!)
    assert stats["total_adjacent_pairs"] == 2
    assert stats["total_switches"] == 0
    assert stats["transition_counts"][0][1] == 0, "Weekend gap transition was incorrectly counted!"
    assert stats["transition_counts"][0][0] == 1
    assert stats["transition_counts"][1][1] == 1


def test_10_modes_cleanly_segregated(replay_df, full_year_df):
    """10. Retrospective full-year and as-of replay modes are cleanly segregated."""
    # Mode B (Replay): Strictly 8335 post-cutover bars
    assert len(replay_df) == 8335
    assert replay_df["timestamp"].min() == pd.Timestamp("2023-08-24 12:00:00+00:00")
    assert replay_df["timestamp"].max() == pd.Timestamp("2023-12-29 21:30:00+00:00")

    # Mode A (Full-Year): 20836 bars partitioned into Train (12501) and Replay (8335)
    assert len(full_year_df) == 20836
    assert full_year_df["timestamp"].min() == pd.Timestamp("2023-01-01 23:30:00+00:00")
    assert full_year_df["timestamp"].max() == pd.Timestamp("2023-12-29 21:30:00+00:00")

    train_slice = full_year_df[full_year_df["data_partition"] == "train_in_sample"]
    replay_slice = full_year_df[full_year_df["data_partition"] == "replay_out_of_sample"]
    assert len(train_slice) == 12501
    assert len(replay_slice) == 8335

    # Check identical probabilities between full_year replay slice and Mode B replay dataframe
    diff0 = np.max(np.abs(replay_slice["prob_state_0"].values - replay_df["prob_state_0"].values))
    diff1 = np.max(np.abs(replay_slice["prob_state_1"].values - replay_df["prob_state_1"].values))
    diff2 = np.max(np.abs(replay_slice["prob_state_2"].values - replay_df["prob_state_2"].values))
    assert max(diff0, diff1, diff2) < 1e-12, "Replay probabilities mismatch between Mode A and Mode B"
