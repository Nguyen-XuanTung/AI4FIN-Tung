"""Intraday Seasonality & Diurnal Volatility Scaling for EURUSD M15.

Provides:
1. Session tagging for FX market sessions (Asian, London, London/NY Overlap, NY Afternoon, Transition).
2. DiurnalVolatilityScaler: A leak-free transformer that learns time-of-day seasonal profiles
   on the training set and transforms out-of-sample data without look-ahead bias.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assign_fx_session(timestamps: pd.Series) -> pd.Series:
    """Classify UTC timestamps into major FX trading sessions.

    Session Definitions (UTC):
        - 'Asian' (00:00 - 07:00 UTC): Tokyo / Sydney liquidity.
        - 'London' (07:00 - 12:00 UTC): European / London morning session.
        - 'Overlap' (12:00 - 16:00 UTC): Peak liquidity London & New York overlap.
        - 'NY_Afternoon' (16:00 - 21:00 UTC): Late US session until London close.
        - 'Transition' (21:00 - 24:00 UTC): End-of-day roll and Pacific transition.
    """
    ts = pd.to_datetime(timestamps)
    hours = ts.dt.hour

    session = pd.Series("Asian", index=timestamps.index, name="session")
    session[hours.between(7, 11)] = "London"
    session[hours.between(12, 15)] = "Overlap"
    session[hours.between(16, 20)] = "NY_Afternoon"
    session[hours.between(21, 23)] = "Transition"
    return session


class DiurnalVolatilityScaler:
    """Leak-free transformer for diurnal (time-of-day) volatility adjustment.

    Estimates 15-minute time-of-day seasonal baselines strictly on training data
    and adjusts candidate volatility features by their expected seasonal scale.

    Mathematical Formulation:
        For bucket b in [0, 95] (hour * 4 + minute // 15):
            seasonal_median_b = median(log_range_{t in b, train})
            seasonal_mad_b    = median(|log_range_{t in b, train} - seasonal_median_b|)
            range_ratio_t     = log_range_t / seasonal_median_b
            range_zscore_t    = (log_range_t - seasonal_median_b) / (1.4826 * seasonal_mad_b)
    """

    def __init__(self, target_col: str = "log_range", eps: float = 1e-7):
        self.target_col = target_col
        self.eps = eps
        self.seasonal_medians: Dict[int, float] = {}
        self.seasonal_mads: Dict[int, float] = {}
        self.global_median: float = 0.0
        self.global_mad: float = 0.0
        self.is_fitted: bool = False

    @staticmethod
    def get_time_bucket(timestamps: pd.Series) -> pd.Series:
        """Map UTC timestamps to 15-minute bucket index [0, 95]."""
        ts = pd.to_datetime(timestamps)
        return ts.dt.hour * 4 + (ts.dt.minute // 15)

    def fit(self, df_train: pd.DataFrame) -> "DiurnalVolatilityScaler":
        """Estimate seasonal volatility profile strictly from training set."""
        if self.target_col not in df_train.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in training DataFrame.")

        logger.info("Fitting DiurnalVolatilityScaler on %d training rows...", len(df_train))
        buckets = self.get_time_bucket(df_train["timestamp"])
        target = df_train[self.target_col]

        self.global_median = float(target.median())
        self.global_mad = float((target - self.global_median).abs().median())
        if self.global_mad < self.eps:
            self.global_mad = self.eps

        self.seasonal_medians = {}
        self.seasonal_mads = {}

        for b in range(96):
            mask = buckets == b
            if mask.sum() > 0:
                b_median = float(target[mask].median())
                b_mad = float((target[mask] - b_median).abs().median())
                self.seasonal_medians[b] = max(b_median, self.eps)
                self.seasonal_mads[b] = max(b_mad, self.eps)
            else:
                self.seasonal_medians[b] = self.global_median
                self.seasonal_mads[b] = self.global_mad

        self.is_fitted = True
        logger.info("DiurnalVolatilityScaler fitted across 96 daily time buckets.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe by adding seasonal baseline and adjusted range features."""
        if not self.is_fitted:
            raise RuntimeError("DiurnalVolatilityScaler must be fitted before transforming data.")

        out = df.copy()
        buckets = self.get_time_bucket(out["timestamp"])
        target = out[self.target_col]

        # Map seasonal baselines
        baseline_median = buckets.map(self.seasonal_medians).fillna(self.global_median)
        baseline_mad = buckets.map(self.seasonal_mads).fillna(self.global_mad)

        out["seasonal_median_range"] = baseline_median
        out["range_ratio"] = target / baseline_median
        out["range_zscore"] = (target - baseline_median) / (1.4826 * baseline_mad)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and transform df."""
        return self.fit(df).transform(df)

