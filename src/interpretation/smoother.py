"""
src.interpretation.smoother
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Log-space multi-sequence Forward-Backward smoother for Gaussian HMM and GMM-HMM.

Computes exact retrospective smoothed state posterior distributions:
    gamma_t(k) = P(S_t = k | O_{1:T})

Scientific Guarantees:
----------------------
1. Strict Sequence Boundary Isolation: Never propagates forward or backward
   messages across independent sequence breaks (weekends, holidays, or missing intervals).
2. Pure Log-Space Arithmetic: Forward and backward recursions are executed
   entirely in log-space using stable logsumexp operations, preventing underflow
   even on long sequences (10,000+ bars) and extreme returns.
3. Architecture Agnostic: Seamlessly processes both single-component GaussianHMM
   and mixture-based GMMHMM models using their native emission density calculations.
4. Normalized Posteriors: Guarantees sum_k gamma_t(k) = 1.0 at every individual timestep.
5. Retrospective Framing: Smoothed posteriors are diagnostic tools for in-sample
   regime characterization; they are NEVER to be conflated with causal real-time
   filtered probabilities P(S_t = k | O_{1:t}).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any
import numpy as np

from src.models.gaussian_hmm import (
    GaussianHMM,
    _fast_logsumexp_axis0,
    _fast_logsumexp_axis1,
    _fast_logsumexp_1d,
)
from src.models.gmm_hmm import GMMHMM


@dataclass
class ForwardBackwardResult:
    """Output container for the Forward-Backward smoother."""
    gamma: np.ndarray                        # Shape (N, K): P(S_t = k | O_{1:T})
    log_alpha: np.ndarray                    # Shape (N, K): forward log-variables
    log_beta: np.ndarray                     # Shape (N, K): backward log-variables
    sequence_log_likelihoods: np.ndarray     # Shape (n_seqs,): ln P(O_{seq})
    total_log_likelihood: float              # Sum of sequence log-likelihoods
    gamma_components: Optional[np.ndarray] = None  # Shape (N, K, M) for GMM-HMM: P(S_t=k, M_t=m | O_{1:T})


def compute_smoothed_posteriors(
    model: Union[GaussianHMM, GMMHMM, Any],
    X: Union[np.ndarray, List[List[float]]],
    lengths: Optional[List[int]] = None,
) -> ForwardBackwardResult:
    """
    Compute exact retrospective smoothed state probabilities using Forward-Backward.

    Parameters
    ----------
    model : Union[GaussianHMM, GMMHMM]
        A fitted Gaussian HMM or GMM-HMM model instance with startprob_ and transmat_.
    X : np.ndarray, shape (N, D)
        Observation matrix.
    lengths : Optional[List[int]]
        Lengths of continuous sequences in X (summing to N).
        If None, X is treated as a single continuous sequence of length N.

    Returns
    -------
    result : ForwardBackwardResult
        Container holding normalized smoothed posteriors gamma of shape (N, K),
        log_alpha, log_beta, and log-likelihood statistics.

    Raises
    ------
    ValueError
        If model parameters are missing, X is empty, or lengths sum does not equal N.
    """
    if model.startprob_ is None or model.transmat_ is None:
        raise ValueError("Provided model must have fitted startprob_ and transmat_ parameters.")

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    N, D = X_arr.shape

    if N == 0:
        raise ValueError("Cannot run smoother on empty observation matrix.")

    if lengths is None:
        lengths = [N]
    elif sum(lengths) != N:
        raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match observation count ({N}).")

    K = model.n_components
    is_gmm = hasattr(model, "weights_") and model.weights_ is not None

    # 1. Compute observation emission log-densities log b_k(x_t)
    if is_gmm:
        log_phi = model._compute_component_log_emissions(X_arr)  # Shape (N, K, M)
        log_b = model._compute_log_emissions(X_arr, log_phi=log_phi)  # Shape (N, K)
    else:
        log_b = model._compute_log_emissions(X_arr)  # Shape (N, K)

    log_startprob = np.log(np.maximum(model.startprob_, 1e-15))
    log_transmat = np.log(np.maximum(model.transmat_, 1e-15))

    log_alpha = np.empty((N, K), dtype=np.float64)
    log_beta = np.empty((N, K), dtype=np.float64)
    n_seqs = len(lengths)
    seq_lls = np.empty(n_seqs, dtype=np.float64)

    # 2. Iterate independently over continuous sequence blocks
    curr_start = 0
    for s_idx, L in enumerate(lengths):
        curr_end = curr_start + L

        # --- A. Forward Pass for sequence s ---
        # t = 0 (relative to sequence start)
        log_alpha[curr_start] = log_startprob + log_b[curr_start]
        for t in range(curr_start + 1, curr_end):
            # log_alpha[t - 1]: shape (K,), log_transmat: shape (K, K)
            # m[i, j] = log_alpha[t-1, i] + log_transmat[i, j]
            m = log_alpha[t - 1, :, None] + log_transmat
            log_alpha[t] = log_b[t] + _fast_logsumexp_axis0(m)

        # Sequence log-likelihood: ln P(O_s) = logsumexp_k(alpha_{L-1}(k))
        seq_ll = _fast_logsumexp_1d(log_alpha[curr_end - 1])
        seq_lls[s_idx] = seq_ll

        # --- B. Backward Pass for sequence s ---
        # t = L - 1 (relative to sequence start): beta_{L-1}(j) = 1.0 => log_beta = 0.0
        log_beta[curr_end - 1] = 0.0
        for t in range(curr_end - 2, curr_start - 1, -1):
            # m[i, j] = log_transmat[i, j] + log_b[t+1, j] + log_beta[t+1, j]
            m = log_transmat + log_b[t + 1, None, :] + log_beta[t + 1, None, :]
            log_beta[t] = _fast_logsumexp_axis1(m)

        curr_start = curr_end

    total_log_likelihood = float(np.sum(seq_lls))

    # 3. Calculate smoothed state posterior responsibilities gamma_t(k)
    # log_gamma_t(k) = log_alpha_t(k) + log_beta_t(k) - seq_ll
    # Map each timestep to its sequence log-likelihood
    seq_ll_per_t = np.empty((N, 1), dtype=np.float64)
    curr_start = 0
    for s_idx, L in enumerate(lengths):
        curr_end = curr_start + L
        seq_ll_per_t[curr_start:curr_end] = seq_lls[s_idx]
        curr_start = curr_end

    log_gamma = log_alpha + log_beta - seq_ll_per_t
    gamma = np.exp(np.clip(log_gamma, -700.0, 0.0))

    # Exact row-wise normalization
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    # Guard against zero sum in degenerate underflow edge cases
    gamma = np.where(gamma_sum > 0, gamma / gamma_sum, 1.0 / K)

    # 4. Optional: Component-level responsibilities for GMM-HMM
    # gamma_{t, km} = gamma_t(k) * [ w_{km} * phi_{km}(x_t) / b_k(x_t) ]
    gamma_components = None
    if is_gmm:
        M = model.n_mix
        log_w = np.log(np.maximum(model.weights_, 1e-15))  # (K, M)
        # component_unnorm: log(w_{km} * phi_{km}(x_t)) = log_w[k, m] + log_phi[t, k, m]
        log_comp_joint = log_phi + log_w[None, :, :]  # (N, K, M)
        # Component posterior given state k:
        # P(M_t = m | S_t = k, x_t) = exp(log_comp_joint - log_b[:, :, None])
        log_comp_cond = log_comp_joint - log_b[:, :, None]
        comp_cond = np.exp(np.clip(log_comp_cond, -700.0, 0.0))
        comp_cond_sum = comp_cond.sum(axis=2, keepdims=True)
        comp_cond = np.where(comp_cond_sum > 0, comp_cond / comp_cond_sum, 1.0 / M)

        # Joint smoothed posterior: P(S_t = k, M_t = m | O_{1:T})
        gamma_components = gamma[:, :, None] * comp_cond  # Shape (N, K, M)

    return ForwardBackwardResult(
        gamma=gamma,
        log_alpha=log_alpha,
        log_beta=log_beta,
        sequence_log_likelihoods=seq_lls,
        total_log_likelihood=total_log_likelihood,
        gamma_components=gamma_components,
    )

