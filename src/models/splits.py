"""
src.models.splits
~~~~~~~~~~~~~~~~~
Chronological train/validation/test splitting and continuous sequence
identification for Hidden Markov Model training and evaluation.

Adheres strictly to the EURUSD M15 Workplan requirements:
- Strictly chronological: t_train < t_val < t_test.
- Zero future leakage: preprocessing and scalers fitted on train only.
- Gap-aware: identifies breaks in time continuity (weekends, closures,
  or excluded partial bars) so HMM transition probabilities are not
  artificially calculated across non-consecutive intervals.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def identify_continuous_sequences(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_step_minutes: int = 15,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Identify continuous contiguous blocks in a time series where consecutive
    bars are spaced exactly by expected_step_minutes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted chronologically containing a timestamp column.
    timestamp_col : str
        Name of timestamp column.
    expected_step_minutes : int
        Expected delta between consecutive bars (default: 15).

    Returns
    -------
    lengths : List[int]
        Lengths of each continuous sequence (suitable for hmmlearn / multi-sequence EM).
    segments : List[Tuple[int, int]]
        List of (start_idx, end_idx) slice tuples for each continuous segment.
    """
    if len(df) == 0:
        return [], []

    timestamps = pd.to_datetime(df[timestamp_col])
    deltas = timestamps.diff()
    expected_delta = pd.Timedelta(minutes=expected_step_minutes)

    # Break occurs where delta is not 15 minutes (and not the first row)
    breaks = np.where((deltas != expected_delta) & (deltas.notna()))[0]

    segments = []
    start_idx = 0
    for b in breaks:
        segments.append((start_idx, b))
        start_idx = b
    segments.append((start_idx, len(df)))

    lengths = [end - start for start, end in segments]
    return lengths, segments


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    filter_incomplete: bool = True,
    timestamp_col: str = "timestamp",
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset chronologically into train, validation, and test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched M15 features DataFrame.
    train_ratio : float
        Proportion of observations for training (default: 0.60).
    val_ratio : float
        Proportion of observations for validation (default: 0.20).
    test_ratio : float
        Proportion of observations for testing (default: 0.20).
    filter_incomplete : bool
        If True, exclude partial candles (is_complete == False) from the split
        datasets while recording the number of excluded records.
    timestamp_col : str
        Timestamp column name.

    Returns
    -------
    splits : Dict[str, pd.DataFrame]
        Dictionary with keys 'train', 'val', 'test', and 'metadata'.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    df_clean = df.sort_values(timestamp_col).copy()

    total_records = len(df_clean)
    excluded_partial_count = 0

    if filter_incomplete and "is_complete" in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean["is_complete"] == True].copy()
        excluded_partial_count = initial_count - len(df_clean)
        logger.info(
            f"Excluded {excluded_partial_count} partial candles (is_complete == False). "
            f"Remaining complete bars: {len(df_clean)}"
        )

    n = len(df_clean)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remaining goes to test to guarantee exact total
    n_test = n - n_train - n_val

    train_df = df_clean.iloc[:n_train].copy()
    val_df = df_clean.iloc[n_train : n_train + n_val].copy()
    test_df = df_clean.iloc[n_train + n_val :].copy()

    # Verify strict temporal ordering
    assert train_df[timestamp_col].max() < val_df[timestamp_col].min(), (
        "Temporal leak detected: train timestamp overlaps with validation!"
    )
    assert val_df[timestamp_col].max() < test_df[timestamp_col].min(), (
        "Temporal leak detected: validation timestamp overlaps with test!"
    )

    metadata = {
        "total_records_original": total_records,
        "filter_incomplete": filter_incomplete,
        "excluded_partial_count": excluded_partial_count,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_start": str(train_df[timestamp_col].min()),
        "train_end": str(train_df[timestamp_col].max()),
        "val_start": str(val_df[timestamp_col].min()),
        "val_end": str(val_df[timestamp_col].max()),
        "test_start": str(test_df[timestamp_col].min()),
        "test_end": str(test_df[timestamp_col].max()),
    }

    logger.info(
        f"Chronological split completed: Train={len(train_df)} ({train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}) | "
        f"Val={len(val_df)} ({val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}) | "
        f"Test={len(test_df)} ({test_df[timestamp_col].min()} to {test_df[timestamp_col].max()})"
    )

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "metadata": metadata,
    }

