"""Resampling module: strictly aggregates 1-minute (M1) OHLC bars into 15-minute (M15) bars.

Constraints:
- Strictly NO candle fabrication.
- NO forward-filling or price interpolation for inactive/closed intervals.
- Bar timestamp denotes the start (open) of the 15-minute interval in UTC.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def resample_m1_to_m15(df_m1: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLC bars to 15-minute (M15) OHLC bars.

    Only 15-minute intervals with at least one active 1-minute bar are emitted.
    Empty intervals (weekends, holidays, or trading halts) are omitted as gaps
    for downstream quality detection and explicit handling.

    Args:
        df_m1: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
               Timestamp must be timezone-aware (UTC).

    Returns:
        DataFrame of M15 bars with columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'active_m1_bars']
        sorted chronologically.
    """
    if df_m1.empty:
        raise ValueError("Cannot resample an empty DataFrame.")

    required_cols = {"timestamp", "open", "high", "low", "close"}
    missing = required_cols - set(df_m1.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_m1: {missing}")

    logger.info("Resampling %d M1 bars to M15 bars...", len(df_m1))

    # Set timestamp as DatetimeIndex
    df = df_m1.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index("timestamp", inplace=True)

    # 15-minute resampler: left-closed, left-labeled (e.g. 00:00 covers [00:00, 00:15))
    resampled = df.resample("15min", closed="left", label="left")

    m15_open = resampled["open"].first()
    m15_high = resampled["high"].max()
    m15_low = resampled["low"].min()
    m15_close = resampled["close"].last()
    m15_count = resampled["open"].count()

    has_volume = "volume" in df.columns and (df["volume"] > 0).any()
    if has_volume:
        m15_vol = resampled["volume"].sum()
    else:
        m15_vol = resampled["volume"].sum()  # will be 0.0

    df_m15 = pd.DataFrame(
        {
            "open": m15_open,
            "high": m15_high,
            "low": m15_low,
            "close": m15_close,
            "tick_volume": m15_vol,
            "active_m1_bars": m15_count,
            "is_complete": m15_count == 15,
        }
    )

    # CRITICAL: Drop intervals with 0 active bars. Do NOT fabricate or interpolate candles!
    # Partial bars (0 < active_m1_bars < 15) are strictly preserved with is_complete=False.
    before_drop = len(df_m15)
    df_m15 = df_m15[df_m15["active_m1_bars"] > 0].copy()
    omitted_intervals = before_drop - len(df_m15)

    df_m15.reset_index(inplace=True)
    df_m15.rename(columns={"index": "timestamp"}, inplace=True)

    logger.info(
        "Successfully aggregated into %d M15 bars (%d empty intervals preserved as gaps).",
        len(df_m15),
        omitted_intervals,
    )
    return df_m15

