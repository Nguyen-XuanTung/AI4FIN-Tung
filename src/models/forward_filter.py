"""
src.models.forward_filter
~~~~~~~~~~~~~~~~~~~~~~~~~
Real-time Forward Filtering engine for Gaussian Hidden Markov Models.

Implements the causal Bayesian recursive update in log-space:
    alpha_t(j) = [sum_i alpha_{t-1}(i) * A_{ij}] * b_j(x_t)
Normalized per timestep:
    P(S_t=j | x_{1:t}) = alpha_t(j) / sum_k alpha_t(k)

Guarantees:
1. Strict causality: Timestep t never accesses future observations x_{t+1:T}.
2. Numerical stability: All calculations executed in log-space with logsumexp.
3. Gap awareness: Explicit policy for weekends, closures, and missing intervals.
4. Equivalence: Sequential (bar-by-bar) and batch filtering yield identical results.
5. High efficiency: Precomputes emission constants and precisions once.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.models.gaussian_hmm import GaussianHMM, _fast_logsumexp_axis0, _fast_logsumexp_1d

logger = logging.getLogger(__name__)


def compute_stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution pi_stat = pi_stat @ transmat
    using eigenvalue decomposition.
    """
    K = transmat.shape[0]
    eigenvals, eigenvecs = np.linalg.eig(transmat.T)
    idx = np.argmin(np.abs(eigenvals - 1.0))
    stat = np.real(eigenvecs[:, idx])
    stat = np.maximum(stat, 0.0)
    if stat.sum() == 0:
        return np.ones(K) / K
    return stat / stat.sum()


class ForwardFilter:
    """
    Online Forward Filtering Engine.

    Parameters
    ----------
    model : GaussianHMM
        Fitted Gaussian HMM instance.
    gap_policy : Literal['reset_prior', 'stationary', 'propagate_matrix']
        Policy when consecutive observations are separated by > max_gap_minutes:
        - 'reset_prior': Reset prior distribution to model.startprob_.
        - 'stationary': Reset prior distribution to stationary distribution of transmat_.
        - 'propagate_matrix': Propagate prior through transmat^n where n = delta_t / 15.
    max_gap_minutes : int
        Threshold above which a gap is declared (default: 15).
    """

    def __init__(
        self,
        model: GaussianHMM,
        gap_policy: Literal["reset_prior", "stationary", "propagate_matrix"] = "reset_prior",
        max_gap_minutes: int = 15,
    ):
        if model.means_ is None or model.covars_ is None or model.transmat_ is None:
            raise ValueError("Provided model must be fitted before running forward filtering.")

        self.model = model
        self.K = model.n_components
        self.D = model.means_.shape[1]
        self.gap_policy = gap_policy
        self.max_gap_minutes = max_gap_minutes

        # Cache log parameters
        self.log_startprob = np.log(np.maximum(model.startprob_, 1e-12))
        self.log_transmat = np.log(np.maximum(model.transmat_, 1e-12))
        self.stat_dist = compute_stationary_distribution(model.transmat_)
        self.log_stat_dist = np.log(np.maximum(self.stat_dist, 1e-12))

        # Precompute emission constants and precision matrices for ultra-fast evaluation
        # Uses model.covars_ directly, consistent with GaussianHMM._compute_log_emissions()
        self._emission_consts = np.empty(self.K, dtype=np.float64)
        self._precision_mats = []
        for k in range(self.K):
            if model.covariance_type == "full":
                cov = 0.5 * (model.covars_[k] + model.covars_[k].T)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0 or not np.isfinite(logdet):
                    cov_safe = cov + np.eye(self.D) * 1e-9
                    sign, logdet = np.linalg.slogdet(cov_safe)
                    inv_cov = np.linalg.pinv(cov_safe)
                else:
                    inv_cov = np.linalg.inv(cov)
                self._precision_mats.append(inv_cov)
                self._emission_consts[k] = -0.5 * self.D * np.log(2.0 * np.pi) - 0.5 * logdet
            else:
                var = np.maximum(model.covars_[k], 1e-12)
                logdet = np.sum(np.log(var))
                self._precision_mats.append(1.0 / var)
                self._emission_consts[k] = -0.5 * self.D * np.log(2.0 * np.pi) - 0.5 * logdet

    def compute_single_log_emission(self, x_t: np.ndarray) -> np.ndarray:
        """
        Compute log b_j(x_t) for state j = 1..K in microseconds.

        Parameters
        ----------
        x_t : np.ndarray, shape (D,)
            Single observation vector.

        Returns
        -------
        log_b : np.ndarray, shape (K,)
        """
        x_t_arr = np.asarray(x_t, dtype=np.float64).ravel()
        log_b = np.empty(self.K, dtype=np.float64)

        for k in range(self.K):
            diff = x_t_arr - self.model.means_[k]
            if self.model.covariance_type == "full":
                inv_cov = self._precision_mats[k]
                mahal = float(diff @ inv_cov @ diff)
            else:
                inv_var = self._precision_mats[k]
                mahal = float(np.sum((diff ** 2) * inv_var))
            log_b[k] = self._emission_consts[k] - 0.5 * mahal

        return np.clip(np.nan_to_num(log_b, nan=-1e12, neginf=-1e12), -1e12, 1e12)

    def step(
        self,
        x_t: np.ndarray,
        prev_filtered_prob: Optional[np.ndarray] = None,
        is_gap: bool = False,
        gap_steps: int = 1,
        precomputed_log_b: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Execute one step of forward filtering in log-space.

        Parameters
        ----------
        x_t : np.ndarray, shape (D,)
            Current observation vector.
        prev_filtered_prob : Optional[np.ndarray], shape (K,)
            Filtered probability distribution from previous timestep P(S_{t-1} | x_{1:t-1}).
            If None, initializes from start of sequence.
        is_gap : bool
            Flag indicating whether a gap occurred between t-1 and t.
        gap_steps : int
            Number of elapsed 15-minute steps during the gap (if propagate_matrix is used).
        precomputed_log_b : Optional[np.ndarray], shape (K,)
            Optional precomputed emission log-likelihood.

        Returns
        -------
        filtered_prob : np.ndarray, shape (K,)
            Updated state probabilities P(S_t=j | x_{1:t}), normalized to sum to 1.0.
        prior_prob : np.ndarray, shape (K,)
            One-step ahead predicted state probabilities P(S_t=j | x_{1:t-1}).
        pred_ll : float
            Predictive log-likelihood ln P(x_t | x_{1:t-1}).
        """
        if precomputed_log_b is not None:
            log_b = precomputed_log_b
        else:
            log_b = self.compute_single_log_emission(x_t)

        # 1. Prior calculation P(S_t | x_{1:t-1})
        if prev_filtered_prob is None:
            log_prior = self.log_startprob.copy()
        elif is_gap:
            if self.gap_policy == "reset_prior":
                log_prior = self.log_startprob.copy()
            elif self.gap_policy == "stationary":
                log_prior = self.log_stat_dist.copy()
            elif self.gap_policy == "propagate_matrix":
                A_pow = np.linalg.matrix_power(self.model.transmat_, max(1, gap_steps))
                prior = prev_filtered_prob @ A_pow
                prior = np.maximum(prior, 1e-12)
                log_prior = np.log(prior / prior.sum())
            else:
                log_prior = self.log_startprob.copy()
        else:
            log_prev = np.log(np.maximum(prev_filtered_prob, 1e-12))
            log_prior = _fast_logsumexp_axis0(log_prev[:, None] + self.log_transmat)

        # 2. Predictive observation log-likelihood ln P(x_t | x_{1:t-1})
        pred_ll = _fast_logsumexp_1d(log_prior + log_b)

        # 3. Bayes update in log-space: log P(S_t=j | x_{1:t}) = log_prior[j] + log_b[j] - pred_ll
        log_filtered = log_prior + log_b - pred_ll

        # 4. Exponentiate and exact normalize
        filtered_prob = np.exp(np.clip(log_filtered, -700, 0))
        filtered_prob /= filtered_prob.sum()

        prior_prob = np.exp(np.clip(log_prior, -700, 0))
        prior_prob /= prior_prob.sum()

        return filtered_prob, prior_prob, pred_ll

    def filter_batch(
        self,
        X: np.ndarray,
        timestamps: Optional[Union[pd.Series, List]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Batch execution of forward filtering across a sequence of observations.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Observation matrix.
        timestamps : Optional[pd.Series or List]
            Timestamps corresponding to observations in X.

        Returns
        -------
        results : Dict[str, np.ndarray]
            - 'filtered_probs': shape (N, K)
            - 'prior_probs': shape (N, K)
            - 'pred_log_likelihoods': shape (N,)
            - 'is_gap': shape (N,) boolean
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        N, D = X_arr.shape

        filtered_probs = np.empty((N, self.K), dtype=np.float64)
        prior_probs = np.empty((N, self.K), dtype=np.float64)
        pred_lls = np.empty(N, dtype=np.float64)
        is_gap_arr = np.zeros(N, dtype=bool)

        # Precompute all log emissions vectorized
        log_b_all = self.model._compute_log_emissions(X_arr)

        # Detect gaps from timestamps if available
        gap_flags = [False] * N
        gap_steps = [1] * N
        if timestamps is not None:
            ts = pd.to_datetime(pd.Series(timestamps))
            deltas = ts.diff()
            threshold = pd.Timedelta(minutes=self.max_gap_minutes)
            for i in range(1, N):
                if pd.notna(deltas.iloc[i]) and deltas.iloc[i] > threshold:
                    gap_flags[i] = True
                    gap_minutes = deltas.iloc[i].total_seconds() / 60.0
                    gap_steps[i] = int(round(gap_minutes / 15.0))

        # Iterative sequential filtering
        prev_filt = None
        for t in range(N):
            filt, prior, pll = self.step(
                x_t=X_arr[t],
                prev_filtered_prob=prev_filt,
                is_gap=gap_flags[t],
                gap_steps=gap_steps[t],
                precomputed_log_b=log_b_all[t],
            )
            filtered_probs[t] = filt
            prior_probs[t] = prior
            pred_lls[t] = pll
            is_gap_arr[t] = gap_flags[t]
            prev_filt = filt

        return {
            "filtered_probs": filtered_probs,
            "prior_probs": prior_probs,
            "pred_log_likelihoods": pred_lls,
            "is_gap": is_gap_arr,
        }


def compute_filtered_probabilities(
    model: GaussianHMM,
    df: pd.DataFrame,
    feature_cols: List[str],
    timestamp_col: str = "timestamp",
    gap_policy: Literal["reset_prior", "stationary", "propagate_matrix"] = "reset_prior",
    max_gap_minutes: int = 15,
) -> pd.DataFrame:
    """
    High-level convenience function: run forward filtering on a DataFrame and
    return an enriched DataFrame with state probabilities and quality metadata.
    """
    df_ordered = df.sort_values(timestamp_col).copy()
    X = df_ordered[feature_cols].values
    timestamps = df_ordered[timestamp_col]

    engine = ForwardFilter(
        model=model,
        gap_policy=gap_policy,
        max_gap_minutes=max_gap_minutes,
    )
    results = engine.filter_batch(X=X, timestamps=timestamps)

    output_df = df_ordered.copy()
    K = model.n_components
    for k in range(K):
        output_df[f"prob_state_{k}"] = results["filtered_probs"][:, k]

    output_df["pred_state"] = np.argmax(results["filtered_probs"], axis=1)
    output_df["pred_log_likelihood"] = results["pred_log_likelihoods"]
    output_df["is_filter_gap"] = results["is_gap"]

    return output_df
