"""Feature evaluation, distribution analysis, and ablation comparisons."""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def evaluate_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Evaluate statistical moments, bounds, and Gaussian suitability for candidate features."""
    records = []

    for col in feature_cols:
        s = df[col].dropna()
        n = len(s)
        mean_val = float(s.mean())
        std_val = float(s.std())
        median_val = float(s.median())
        min_val = float(s.min())
        max_val = float(s.max())
        skew_val = float(s.skew())
        kurt_val = float(s.kurt())
        zero_pct = float((s == 0.0).mean() * 100.0)
        non_neg_pct = float((s >= 0.0).mean() * 100.0)

        # Normality test (D'Agostino-Pearson omnibus test)
        stat_val, p_val = stats.normaltest(s)

        # Compatibility assessment for standard Gaussian emissions
        if zero_pct > 5.0 and non_neg_pct > 99.0:
            suitability = "Low (Zero-inflated & bounded [0, +inf))"
        elif abs(skew_val) > 2.0 or kurt_val > 20.0:
            suitability = "Moderate (Heavy tails / High skew)"
        elif abs(skew_val) < 1.0 and kurt_val < 10.0:
            suitability = "High (Symmetric & moderate tails)"
        else:
            suitability = "Acceptable"

        records.append({
            "feature": col,
            "count": n,
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3),
            "zero_pct": round(zero_pct, 2),
            "non_negative_pct": round(non_neg_pct, 2),
            "dagostino_stat": round(float(stat_val), 1),
            "dagostino_p": float(p_val),
            "gaussian_suitability": suitability,
        })

    return pd.DataFrame(records)


def compute_feature_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices for candidate features."""
    sub = df[feature_cols].dropna()
    pearson_corr = sub.corr(method="pearson")
    spearman_corr = sub.corr(method="spearman")
    return pearson_corr, spearman_corr


def get_candidate_configurations() -> Dict[str, Dict[str, Any]]:
    """Define standard candidate feature configurations for Phase 2 ablation & Phase 3 evaluation."""
    return {
        "Config_1_Paper1": {
            "name": "Paper 1 Direct Benchmark (3D)",
            "features": ["p1_r_oc", "p1_r_oh", "p1_r_ol"],
            "dimension": 3,
            "description": "Direct implementation of Kijkarncharoensin (2025): percentage body return, open-to-high, and low-to-open returns.",
            "theoretical_foundation": "Paper 1 (Risks 2025)",
            "limitations": "Non-negative support for r_oh and r_ol; high correlation between features.",
        },
        "Config_2_MinimalOrthogonal": {
            "name": "Minimal Orthogonal 2D (Body Return + Log-Vol)",
            "features": ["r_oc", "log_vol"],
            "dimension": 2,
            "description": "Decouples signed price direction (r_oc) from unsigned volatility scale (log_vol = ln(ln(H/L))).",
            "theoretical_foundation": "Orthogonal decomposition (zero Pearson correlation r = -0.014).",
            "limitations": "Omits wick rejection geometry.",
        },
        "Config_3_BodyAndWicks": {
            "name": "Candlestick Body & Wicks 3D",
            "features": ["r_oc", "upper_wick", "lower_wick"],
            "dimension": 3,
            "description": "Directional body return combined with upper and lower price rejection wicks.",
            "theoretical_foundation": "Price rejection and intraday candle geometry (WORK_PLAN.md Phase 2).",
            "limitations": "Wicks have ~8% zero-inflation and non-negative support.",
        },
        "Config_4_FullCandle": {
            "name": "Full Candlestick Geometry 4D",
            "features": ["r_oc", "log_vol", "upper_wick", "lower_wick"],
            "dimension": 4,
            "description": "Comprehensive representation including direction, volatility scale, and dual rejection wicks.",
            "theoretical_foundation": "Multi-dimensional price formation.",
            "limitations": "Higher covariance dimensionality (10 covariance parameters per mixture).",
        },
        "Config_5_DiurnalAdjusted": {
            "name": "Diurnal-Adjusted Orthogonal 2D",
            "features": ["r_oc", "range_ratio"],
            "dimension": 2,
            "description": "Directional body return with seasonality-normalized range (scaled by 15-minute time-of-day median).",
            "theoretical_foundation": "Controls for Asian vs London/NY session volatility differences.",
            "limitations": "Requires fitting seasonal baseline on training set without look-ahead leakage.",
        },
        "Config_6_DiurnalConviction": {
            "name": "Diurnal-Adjusted with Conviction 3D",
            "features": ["r_oc", "range_ratio", "body_ratio"],
            "dimension": 3,
            "description": "Direction, seasonality-adjusted range, and normalized directional conviction ((C-O)/(H-L)).",
            "theoretical_foundation": "Combines de-seasonalized volatility with bounded conviction in [-1, +1].",
            "limitations": "Moderate correlation between r_oc and body_ratio.",
        },
    }

