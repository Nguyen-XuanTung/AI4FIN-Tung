"""Feature engineering module for EURUSD M15 regime detection."""

from .candidates import (
    compute_all_features,
    compute_log_returns_oc,
    compute_log_returns_cc,
    compute_log_range,
    compute_log_volatility,
    compute_wicks,
    compute_directional_body_ratio,
)
from .seasonality import DiurnalVolatilityScaler
from .ablation import evaluate_feature_distributions, compute_feature_correlations

__all__ = [
    "compute_all_features",
    "compute_log_returns_oc",
    "compute_log_returns_cc",
    "compute_log_range",
    "compute_log_volatility",
    "compute_wicks",
    "compute_directional_body_ratio",
    "DiurnalVolatilityScaler",
    "evaluate_feature_distributions",
    "compute_feature_correlations",
]

