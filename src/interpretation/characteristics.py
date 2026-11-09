"""
src.interpretation.characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calculates comprehensive regime characteristics on the Training partition
using smoothed posterior responsibilities gamma_t(k) = P(S_t = k | O_{1:T}).

Key Principles:
---------------
1. In-Sample Training Diagnostic Only: All statistics are calculated strictly
   on the 12,501 complete candles of the training partition.
2. Posterior Weighting: Replaces crude argmax hard-assignment with mathematically
   rigorous weighted expectation formulas.
3. Complete Mixture Moment Aggregation: For GMM-HMM, combines component means
   and covariances into state-level moments using total variance formulas.
4. Scale Separation: Clearly distinguishes log-space volatility metrics (log_vol)
   from geometric-scale and original-scale physical candle ranges (basis points and pips).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def weighted_mean_and_std(
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate posterior-weighted mean and sample standard deviation.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
    weights : np.ndarray, shape (N,)

    Returns
    -------
    w_mean : float
    w_std : float
    """
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return 0.0, 0.0

    w_mean = float(np.sum(weights * values) / w_sum)
    variance = float(np.sum(weights * ((values - w_mean) ** 2)) / w_sum)
    w_std = float(np.sqrt(max(0.0, variance)))
    return w_mean, w_std


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: List[float],
) -> Dict[str, float]:
    """
    Compute posterior-weighted quantiles for a 1D continuous array.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
    weights : np.ndarray, shape (N,)
    quantiles : List[float]
        List of quantile levels in [0.0, 1.0].

    Returns
    -------
    q_dict : Dict[str, float]
        Dictionary mapping quantile labels (e.g. 'q05', 'q50') to values.
    """
    w_sum = np.sum(weights)
    if w_sum <= 0 or len(values) == 0:
        return {f"q{int(q*100):02d}": 0.0 for q in quantiles}

    order = np.argsort(values)
    val_sorted = values[order]
    w_sorted = weights[order]

    cum_w = np.cumsum(w_sorted) / w_sum
    # Shift cum_w to midpoint of weight for balanced empirical CDF interpolation
    cum_w_mid = (cum_w + np.concatenate(([0.0], cum_w[:-1]))) / 2.0

    res = {}
    for q in quantiles:
        val_q = float(np.interp(q, cum_w_mid, val_sorted))
        lbl = f"q{int(q * 100):02d}"
        res[lbl] = val_q
    return res


def weighted_higher_moments(
    values: np.ndarray,
    weights: np.ndarray,
    w_mean: float,
    w_std: float,
) -> Tuple[float, float]:
    """
    Calculate posterior-weighted skewness and excess kurtosis.

    Returns
    -------
    skewness : float
    excess_kurtosis : float
    """
    w_sum = float(np.sum(weights))
    if w_sum <= 0 or w_std <= 1e-12:
        return 0.0, 0.0

    diff_norm = (values - w_mean) / w_std
    m3 = float(np.sum(weights * (diff_norm ** 3)) / w_sum)
    m4 = float(np.sum(weights * (diff_norm ** 4)) / w_sum)
    excess_kurt = float(m4 - 3.0)
    return m3, excess_kurt


def aggregate_gmm_state_moments(
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str = "full",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate Gaussian mixture component moments into state-level moments.

    Mathematical Formulation:
    -------------------------
    For state k composed of M Gaussian mixture components:
        mu_k = sum_{m=1}^M w_{km} * mu_{km}
        Sigma_k = sum_{m=1}^M w_{km} * [ Sigma_{km} + (mu_{km} - mu_k)(mu_{km} - mu_k)^T ]

    Parameters
    ----------
    weights : np.ndarray, shape (K, M)
        Component mixture weights per state, sum_m w_{km} = 1.
    means : np.ndarray, shape (K, M, D)
        Component mean vectors.
    covars : np.ndarray, shape (K, M, D, D) or (K, M, D)
        Component covariance matrices.
    covariance_type : str
        'full' or 'diag'.

    Returns
    -------
    state_means : np.ndarray, shape (K, D)
        Aggregated state-level mean vectors.
    state_covars : np.ndarray, shape (K, D, D)
        Aggregated state-level covariance matrices.
    """
    K, M, D = means.shape
    state_means = np.empty((K, D), dtype=np.float64)
    state_covars = np.empty((K, D, D), dtype=np.float64)

    for k in range(K):
        w_k = weights[k]  # (M,)
        # 1. Aggregated Mean: sum_m w_{km} * mu_{km}
        mu_k = np.sum(w_k[:, None] * means[k], axis=0)  # (D,)
        state_means[k] = mu_k

        # 2. Aggregated Covariance: sum_m w_{km} * [ Sigma_{km} + delta * delta^T ]
        cov_k = np.zeros((D, D), dtype=np.float64)
        for m in range(M):
            w_km = float(w_k[m])
            if covariance_type == "full":
                sigma_km = covars[k, m]
            else:
                sigma_km = np.diag(covars[k, m])

            delta_m = (means[k, m] - mu_k)[:, None]  # (D, 1)
            outer_delta = delta_m @ delta_m.T  # (D, D)
            cov_k += w_km * (sigma_km + outer_delta)

        # Ensure exact numerical symmetry
        state_covars[k] = 0.5 * (cov_k + cov_k.T)

    return state_means, state_covars


@dataclass
class StateCharacteristics:
    """Detailed statistical profile of a single latent regime on the Training partition."""
    state_id: int
    soft_occupancy: float
    effective_sample_size: float

    # Return characteristics (r_oc)
    r_oc_mean: float
    r_oc_std: float
    r_oc_median: float
    r_oc_quantiles: Dict[str, float]
    r_oc_skewness: float
    r_oc_kurtosis: float
    prob_positive_return: float
    prob_negative_return: float
    prob_zero_return: float

    # Log-volatility characteristics (log_vol)
    log_vol_mean: float
    log_vol_std: float
    log_vol_median: float
    log_vol_quantiles: Dict[str, float]

    # Original-scale candle range statistics directly from OHLC (basis points: (H-L)/O * 10,000)
    range_bps_mean: float
    range_bps_std: float
    range_bps_median: float
    range_bps_quantiles: Dict[str, float]

    # Pip-scale candle range: (H - L) * 10,000
    range_pips_mean: float
    range_pips_median: float

    # Scale-comparison diagnostic
    # Geometric scale: exp(mean(log_vol)), strictly distinct from arithmetic range
    geometric_range_scale: float

    # Analytical emission moments (from model parameters)
    analytical_mean: List[float]
    analytical_covar: List[List[float]]
    empirical_mean: List[float]
    empirical_covar: List[List[float]]
    moment_discrepancy_mean_l2: float
    moment_discrepancy_covar_frobenius: float


@dataclass
class RegimeCharacteristics:
    """Collection of regime profiles for all hidden states of a fitted model."""
    model_version: str
    model_type: str
    n_components: int
    n_mix: int
    training_samples: int
    feature_names: List[str]
    states: Dict[int, StateCharacteristics]

    def to_dict(self) -> Dict[str, Any]:
        """Convert regime characteristics container to JSON-serializable dictionary."""
        out = {
            "model_version": self.model_version,
            "model_type": self.model_type,
            "n_components": self.n_components,
            "n_mix": self.n_mix,
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "states": {},
        }
        for k, s in self.states.items():
            out["states"][str(k)] = {
                "state_id": s.state_id,
                "soft_occupancy": s.soft_occupancy,
                "effective_sample_size": s.effective_sample_size,
                "return_r_oc": {
                    "mean": s.r_oc_mean,
                    "std": s.r_oc_std,
                    "median": s.r_oc_median,
                    "quantiles": s.r_oc_quantiles,
                    "skewness": s.r_oc_skewness,
                    "kurtosis": s.r_oc_kurtosis,
                    "prob_positive": s.prob_positive_return,
                    "prob_negative": s.prob_negative_return,
                    "prob_zero": s.prob_zero_return,
                },
                "log_volatility": {
                    "mean": s.log_vol_mean,
                    "std": s.log_vol_std,
                    "median": s.log_vol_median,
                    "quantiles": s.log_vol_quantiles,
                    "geometric_scale_exp_mean": s.geometric_range_scale,
                },
                "original_scale_range_basis_points": {
                    "mean_bps": s.range_bps_mean,
                    "std_bps": s.range_bps_std,
                    "median_bps": s.range_bps_median,
                    "quantiles_bps": s.range_bps_quantiles,
                },
                "pip_range": {
                    "mean_pips": s.range_pips_mean,
                    "median_pips": s.range_pips_median,
                },
                "mixture_moment_comparison": {
                    "analytical_mean": s.analytical_mean,
                    "analytical_covar": s.analytical_covar,
                    "empirical_mean": s.empirical_mean,
                    "empirical_covar": s.empirical_covar,
                    "discrepancy_mean_l2": s.moment_discrepancy_mean_l2,
                    "discrepancy_covar_frobenius": s.moment_discrepancy_covar_frobenius,
                },
            }
        return out


def calculate_state_characteristics(
    model: Any,
    train_df: pd.DataFrame,
    gamma: np.ndarray,
    model_version: str,
    feature_cols: Optional[List[str]] = None,
) -> RegimeCharacteristics:
    """
    Compute comprehensive posterior-weighted regime characteristics on Training observations.

    Parameters
    ----------
    model : Union[GaussianHMM, GMMHMM]
        Fitted model instance.
    train_df : pd.DataFrame
        Training partition containing feature_cols and raw price columns (open, high, low, close).
    gamma : np.ndarray, shape (N, K)
        Smoothed posterior responsibilities P(S_t = k | O_{1:T}).
    model_version : str
        Version identifier of the model.
    feature_cols : Optional[List[str]]
        Observation feature column names (defaults to ['r_oc', 'log_vol']).

    Returns
    -------
    regime_chars : RegimeCharacteristics
        Complete statistical characteristics per regime state.
    """
    if feature_cols is None:
        feature_cols = ["r_oc", "log_vol"]

    N, K = gamma.shape
    if len(train_df) != N:
        raise ValueError(f"train_df length ({len(train_df)}) does not match gamma length ({N}).")

    is_gmm = hasattr(model, "weights_") and model.weights_ is not None
    n_mix = getattr(model, "n_mix", 1) if is_gmm else 1
    model_type = "GMMHMM" if is_gmm else "GaussianHMM"

    # Extract observation arrays
    r_oc = train_df["r_oc"].values
    log_vol = train_df["log_vol"].values
    X_obs = train_df[feature_cols].values

    # Compute original-scale physical candle range statistics directly from OHLC
    # 1. Basis points: ((High - Low) / Open) * 10,000
    range_bps = ((train_df["high"] - train_df["low"]) / train_df["open"]).values * 10000.0
    # 2. Pips: (High - Low) * 10,000 (standard EURUSD pip conversion)
    range_pips = (train_df["high"] - train_df["low"]).values * 10000.0

    quantiles_levels = [0.05, 0.25, 0.50, 0.75, 0.95]

    # Calculate analytical state-level emission moments
    if is_gmm:
        analytical_means, analytical_covars = aggregate_gmm_state_moments(
            weights=model.weights_,
            means=model.means_,
            covars=model.covars_,
            covariance_type=model.covariance_type,
        )
    else:
        analytical_means = model.means_
        if model.covariance_type == "full":
            analytical_covars = model.covars_
        else:
            analytical_covars = np.array([np.diag(c) for c in model.covars_])

    state_dict = {}

    for k in range(K):
        w_k = gamma[:, k]
        w_sum = float(np.sum(w_k))
        soft_occ = float(w_sum / N)

        # 1. r_oc statistics
        r_mean, r_std = weighted_mean_and_std(r_oc, w_k)
        r_quantiles = weighted_quantiles(r_oc, w_k, quantiles_levels)
        r_median = r_quantiles["q50"]
        r_skew, r_kurt = weighted_higher_moments(r_oc, w_k, r_mean, r_std)

        prob_pos = float(np.sum(w_k[r_oc > 0.0]) / w_sum) if w_sum > 0 else 0.0
        prob_neg = float(np.sum(w_k[r_oc < 0.0]) / w_sum) if w_sum > 0 else 0.0
        prob_zero = float(np.sum(w_k[r_oc == 0.0]) / w_sum) if w_sum > 0 else 0.0

        # 2. log_vol statistics
        vol_mean, vol_std = weighted_mean_and_std(log_vol, w_k)
        vol_quantiles = weighted_quantiles(log_vol, w_k, quantiles_levels)
        vol_median = vol_quantiles["q50"]

        # Geometric scale summary: exp(mean(log_vol))
        geom_scale = float(np.exp(vol_mean))

        # 3. Original-scale candle range statistics (bps and pips)
        bps_mean, bps_std = weighted_mean_and_std(range_bps, w_k)
        bps_quantiles = weighted_quantiles(range_bps, w_k, quantiles_levels)
        bps_median = bps_quantiles["q50"]

        pip_mean, _ = weighted_mean_and_std(range_pips, w_k)
        pip_quantiles = weighted_quantiles(range_pips, w_k, [0.50])
        pip_median = pip_quantiles["q50"]

        # 4. Empirical posterior-weighted moments on training features
        emp_mean = np.sum(w_k[:, None] * X_obs, axis=0) / max(w_sum, 1e-12)
        diff = X_obs - emp_mean
        emp_cov = (diff.T @ (w_k[:, None] * diff)) / max(w_sum, 1e-12)

        an_mean_k = analytical_means[k]
        an_cov_k = analytical_covars[k]

        disc_mean_l2 = float(np.linalg.norm(an_mean_k - emp_mean))
        disc_cov_fro = float(np.linalg.norm(an_cov_k - emp_cov, ord="fro"))

        state_dict[k] = StateCharacteristics(
            state_id=k,
            soft_occupancy=soft_occ,
            effective_sample_size=w_sum,
            r_oc_mean=r_mean,
            r_oc_std=r_std,
            r_oc_median=r_median,
            r_oc_quantiles=r_quantiles,
            r_oc_skewness=r_skew,
            r_oc_kurtosis=r_kurt,
            prob_positive_return=prob_pos,
            prob_negative_return=prob_neg,
            prob_zero_return=prob_zero,
            log_vol_mean=vol_mean,
            log_vol_std=vol_std,
            log_vol_median=vol_median,
            log_vol_quantiles=vol_quantiles,
            range_bps_mean=bps_mean,
            range_bps_std=bps_std,
            range_bps_median=bps_median,
            range_bps_quantiles=bps_quantiles,
            range_pips_mean=pip_mean,
            range_pips_median=pip_median,
            geometric_range_scale=geom_scale,
            analytical_mean=an_mean_k.tolist(),
            analytical_covar=an_cov_k.tolist(),
            empirical_mean=emp_mean.tolist(),
            empirical_covar=emp_cov.tolist(),
            moment_discrepancy_mean_l2=disc_mean_l2,
            moment_discrepancy_covar_frobenius=disc_cov_fro,
        )

    return RegimeCharacteristics(
        model_version=model_version,
        model_type=model_type,
        n_components=K,
        n_mix=n_mix,
        training_samples=N,
        feature_names=feature_cols,
        states=state_dict,
    )

