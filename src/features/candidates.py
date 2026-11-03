"""Candidate feature engineering for EURUSD M15 Market Regime Detection.

Implements all candidate features specified in WORK_PLAN.md Phase 2:
1. Close-to-close log return: r_cc = ln(C_t / C_{t-1})
2. Open-to-close log return (body return): r_oc = ln(C_t / O_t)
3. High-low log range: log_range = ln(H_t / L_t)
4. Log volatility: log_vol = ln(max(log_range, eps))
5. Upper and lower wicks:
   - UpperWick = ln(H_t / max(O_t, C_t))
   - LowerWick = ln(min(O_t, C_t) / L_t)
6. Directional body ratio: body_ratio = (C_t - O_t) / (H_t - L_t)
7. Paper 1 benchmark returns: r_oc_pct, r_oh_pct, r_ol_pct
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns_oc(df: pd.DataFrame) -> pd.Series:
    """Compute open-to-close log body return: r_oc = ln(C_t / O_t).

    Financial Meaning:
        Measures the net directional price change within the 15-minute bar.
        Independent of inter-bar gaps or weekend closures.
    """
    return pd.Series(np.log(df["close"] / df["open"]), index=df.index, name="r_oc")


def compute_log_returns_cc(df: pd.DataFrame) -> pd.Series:
    """Compute close-to-close log return: r_cc = ln(C_t / C_{t-1}).

    Financial Meaning:
        Measures inter-bar total return.
    Note:
        First observation is NaN. Spans across session/weekend gaps if not masked.
    """
    return pd.Series(np.log(df["close"] / df["close"].shift(1)), index=df.index, name="r_cc")


def compute_log_range(df: pd.DataFrame) -> pd.Series:
    """Compute high-low log range: log_range = ln(H_t / L_t).

    Financial Meaning:
        Measures total price amplitude / intraday volatility during the bar.
        Strictly non-negative (log_range >= 0).
    """
    return pd.Series(np.log(df["high"] / df["low"]), index=df.index, name="log_range")


def compute_log_volatility(df: pd.DataFrame, eps: float = 1e-6) -> pd.Series:
    """Compute log-transformed range: log_vol = ln(max(log_range, eps)).

    Financial & Statistical Meaning:
        Transforms the heavily right-skewed log_range into an approximately symmetric,
        Gaussian-compatible scale.
    """
    raw_range = np.log(df["high"] / df["low"])
    clipped = np.maximum(raw_range, eps)
    return pd.Series(np.log(clipped), index=df.index, name="log_vol")


def compute_wicks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute Upper and Lower Wick features using OHLC prices:
        UpperWick_t = ln(H_t / max(O_t, C_t))
        LowerWick_t = ln(min(O_t, C_t) / L_t)

    Financial Meaning:
        Measures price rejection pressure at local highs and lows.
        Long upper wick: Bullish momentum rejected by sellers.
        Long lower wick: Bearish pressure rejected by buyers.

    Statistical Limitation:
        Strictly non-negative with ~8% zero-inflation (bars where H == max(O,C)).
    """
    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)

    # Use maximum with 1.0 to prevent minor float imprecision producing negative values
    ratio_upper = np.maximum(df["high"] / max_oc, 1.0)
    ratio_lower = np.maximum(min_oc / df["low"], 1.0)

    upper_wick = pd.Series(np.log(ratio_upper), index=df.index, name="upper_wick")
    lower_wick = pd.Series(np.log(ratio_lower), index=df.index, name="lower_wick")
    return upper_wick, lower_wick


def compute_directional_body_ratio(df: pd.DataFrame, eps: float = 1e-7) -> pd.Series:
    """Compute normalized directional body ratio: body_ratio = (C - O) / (H - L).

    Financial Meaning:
        Measures directional conviction normalized to total candle range.
        Ranges strictly in [-1, +1].
        +1: Pure bullish marubozu (O = L, C = H).
        -1: Pure bearish marubozu (O = H, C = L).
         0: Neutral doji or symmetric consolidation.
    """
    hl_diff = df["high"] - df["low"]
    ratio = np.where(hl_diff > eps, (df["close"] - df["open"]) / hl_diff, 0.0)
    # Clip to [-1.0, 1.0] for absolute numerical safety
    clipped = np.clip(ratio, -1.0, 1.0)
    return pd.Series(clipped, index=df.index, name="body_ratio")


def compute_paper1_returns(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute candidate features directly inherited from Paper 1 (Kijkarncharoensin, 2025):
        r_oc = (C - O) / O
        r_oh = (H - O) / O
        r_ol = (O - L) / O
    """
    r_oc = pd.Series((df["close"] - df["open"]) / df["open"], index=df.index, name="p1_r_oc")
    r_oh = pd.Series((df["high"] - df["open"]) / df["open"], index=df.index, name="p1_r_oh")
    r_ol = pd.Series((df["open"] - df["low"]) / df["open"], index=df.index, name="p1_r_ol")
    return r_oc, r_oh, r_ol


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all candidate features and merge them with existing OHLC and metadata columns.

    Args:
        df: DataFrame containing ['timestamp', 'open', 'high', 'low', 'close']
            and quality metadata ['active_m1_bars', 'is_complete'].

    Returns:
        Enriched DataFrame with all candidate features and metadata.
    """
    logger.info("Computing candidate features for %d M15 bars...", len(df))
    out = df.copy()

    # Log returns
    out["r_oc"] = compute_log_returns_oc(df)
    out["r_cc"] = compute_log_returns_cc(df)

    # Range and volatility
    out["log_range"] = compute_log_range(df)
    out["log_vol"] = compute_log_volatility(df)

    # Wicks
    upper_wick, lower_wick = compute_wicks(df)
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick
    out["total_wick"] = upper_wick + lower_wick

    # Directional conviction
    out["body_ratio"] = compute_directional_body_ratio(df)

    # Paper 1 features
    p1_roc, p1_roh, p1_rol = compute_paper1_returns(df)
    out["p1_r_oc"] = p1_roc
    out["p1_r_oh"] = p1_roh
    out["p1_r_ol"] = p1_rol

    logger.info("Candidate features successfully generated. Columns: %s", list(out.columns))
    return out

