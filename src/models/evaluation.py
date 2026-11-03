"""
src.models.evaluation
~~~~~~~~~~~~~~~~~~~~~
Model evaluation, information criteria (AIC, BIC), sequential out-of-sample
predictive log-likelihood, state persistence, occupancy, and regime stability diagnostics.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.gaussian_hmm import GaussianHMM
from src.models.forward_filter import ForwardFilter


def calculate_num_free_params(
    n_components: int,
    n_features: int,
    covariance_type: str = "full",
) -> int:
    """
    Calculate the exact number of free continuous parameters in a Gaussian HMM.

    Parameters
    ----------
    n_components : int
        Number of hidden states (K).
    n_features : int
        Observation dimension (D).
    covariance_type : str
        'full' or 'diag'.

    Returns
    -------
    num_params : int
    """
    K = n_components
    D = n_features

    # Initial state probabilities: K - 1 free parameters (sum to 1)
    startprob_params = K - 1

    # Transition matrix: K * (K - 1) free parameters (each row sums to 1)
    transmat_params = K * (K - 1)

    # Emission means: K * D
    mean_params = K * D

    # Emission covariances:
    if covariance_type == "full":
        # Each symmetric covariance matrix has D * (D + 1) / 2 unique parameters
        cov_params = K * (D * (D + 1) // 2)
    else:
        # Diagonal has D parameters per state
        cov_params = K * D

    return startprob_params + transmat_params + mean_params + cov_params


def calculate_num_free_params_gmm(
    n_components: int,
    n_mix: int,
    n_features: int,
    covariance_type: str = "full",
) -> int:
    """
    Calculate the exact number of free continuous parameters in a GMM-HMM.

    Parameters
    ----------
    n_components : int
        Number of hidden states (K).
    n_mix : int
        Number of Gaussian mixture components per state (M).
    n_features : int
        Observation dimension (D).
    covariance_type : str
        'full' or 'diag'.

    Returns
    -------
    num_params : int
    """
    K = n_components
    M = n_mix
    D = n_features

    # Initial state probabilities: K - 1
    startprob_params = K - 1

    # Transition matrix: K * (K - 1)
    transmat_params = K * (K - 1)

    # Mixture weights: K * (M - 1) (each state's mixture weights sum to 1)
    weight_params = K * (M - 1)

    # Emission means: K * M * D
    mean_params = K * M * D

    # Emission covariances:
    if covariance_type == "full":
        cov_params = K * M * (D * (D + 1) // 2)
    else:
        cov_params = K * M * D

    return startprob_params + transmat_params + weight_params + mean_params + cov_params


def compute_information_criteria(
    log_likelihood: float,
    n_samples: int,
    n_params: int,
) -> Dict[str, float]:
    """
    Compute AIC and BIC from log-likelihood and sample count.

    Formula:
      AIC = 2*k - 2*ln(L)
      BIC = k*ln(N) - 2*ln(L)
    """
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = n_params * np.log(n_samples) - 2.0 * log_likelihood
    return {"aic": float(aic), "bic": float(bic)}


def evaluate_model_diagnostics(
    model: Union[GaussianHMM, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    train_lengths: Optional[List[int]] = None,
    val_lengths: Optional[List[int]] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_lengths: Optional[List[int]] = None,
    model_name: str = "Model",
) -> Dict:
    """
    Comprehensive diagnostic evaluation of a fitted Gaussian HMM or GMM-HMM.

    Parameters
    ----------
    model : Union[GaussianHMM, GMMHMM]
        Fitted Gaussian HMM or GMM-HMM.
    train_df : pd.DataFrame
        Training dataset (In-sample parameter estimation).
    val_df : pd.DataFrame
        Validation dataset (Hyperparameter tuning and model selection).
    feature_cols : List[str]
        List of feature column names.
    train_lengths : Optional[List[int]]
        Continuous sequence lengths for training data.
    val_lengths : Optional[List[int]]
        Continuous sequence lengths for validation data.
    test_df : Optional[pd.DataFrame]
        Held-out Test dataset (True out-of-sample evaluation).
    test_lengths : Optional[List[int]]
        Continuous sequence lengths for test data.
    model_name : str
        Human-readable model identifier.

    Returns
    -------
    diagnostics : Dict
        Dictionary of all statistical and regime diagnostic metrics.
    """
    K = model.n_components
    D = len(feature_cols)
    N_train = len(train_df)
    N_val = len(val_df)
    N_test = len(test_df) if test_df is not None else N_val

    is_gmm = hasattr(model, "weights_") and model.weights_ is not None
    n_mix = getattr(model, "n_mix", 1) if is_gmm else 1

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values if test_df is not None else X_val

    # 1. Parameter count & Information criteria
    if is_gmm:
        n_params = calculate_num_free_params_gmm(K, n_mix, D, model.covariance_type)
    else:
        n_params = calculate_num_free_params(K, D, model.covariance_type)

    train_ll = model.score(X_train, lengths=train_lengths)
    val_ll = model.score(X_val, lengths=val_lengths)

    ic = compute_information_criteria(train_ll, N_train, n_params)

    # 2. Sequential Validation Log-Likelihood (verifies chain-rule equivalence)
    ff_engine = ForwardFilter(model=model, gap_policy="reset_prior", max_gap_minutes=15)
    val_filter_res = ff_engine.filter_batch(
        X=X_val, timestamps=val_df["timestamp"] if "timestamp" in val_df.columns else None
    )
    val_pred_ll = float(np.sum(val_filter_res["pred_log_likelihoods"]))

    # 3. Sequential Out-of-Sample Predictive Log-Likelihood on held-out Test set
    if test_df is not None:
        test_ll = model.score(X_test, lengths=test_lengths)
        test_filter_res = ff_engine.filter_batch(
            X=X_test, timestamps=test_df["timestamp"] if "timestamp" in test_df.columns else None
        )
        test_oos_pred_ll = float(np.sum(test_filter_res["pred_log_likelihoods"]))
        mean_test_oos_pred_ll = float(np.mean(test_filter_res["pred_log_likelihoods"]))
    else:
        test_ll = val_ll
        test_oos_pred_ll = val_pred_ll
        mean_test_oos_pred_ll = float(np.mean(val_filter_res["pred_log_likelihoods"]))

    # 4. State persistence & Expected stay duration
    # In a discrete-time Markov chain, expected stay duration in state k is E[T_k] = 1 / (1 - A_kk) bars.
    # For M15 bars, duration in hours is E[T_k] * 0.25.
    diag_trans = np.diag(model.transmat_)
    expected_duration_bars = [
        float(1.0 / max(1e-4, 1.0 - p)) for p in diag_trans
    ]
    expected_duration_hours = [b * 0.25 for b in expected_duration_bars]

    # 5. State occupancy (Validation set)
    # Distinguish soft probability occupancy (mean of P(S_t=k|x_{1:t})) from hard argmax occupancy
    val_probs = val_filter_res["filtered_probs"]
    val_pred_states = np.argmax(val_probs, axis=1)
    mean_occupancy_prob = np.mean(val_probs, axis=0).tolist()
    hard_occupancy = [
        float(np.mean(val_pred_states == k)) for k in range(K)
    ]
    min_soft_occupancy = float(np.min(mean_occupancy_prob))
    min_hard_occupancy = float(np.min(hard_occupancy))

    # 6. Switching frequency (fraction of bars where argmax state changes)
    state_switches = np.sum(val_pred_states[1:] != val_pred_states[:-1])
    switch_frequency = float(state_switches / max(1, N_val - 1))

    # 7. Degenerate state detection
    collapsed_states = []
    for k in range(K):
        # Collapse if hard occupancy < 1.0% or soft probability occupancy < 1.0%
        if hard_occupancy[k] < 0.01 or mean_occupancy_prob[k] < 0.01:
            collapsed_states.append(k)

    min_occupancy = min_soft_occupancy
    min_persistence = float(np.min(diag_trans))
    max_persistence = float(np.max(diag_trans))

    # Stability flag
    is_stable = (
        len(collapsed_states) == 0
        and min_persistence >= 0.20
        and max_persistence <= 0.9995
        and switch_frequency <= 0.40  # Less than 40% switching (not hyper-fragmented)
    )

    return {
        "model_name": model_name,
        "n_components": K,
        "n_features": D,
        "features": feature_cols,
        "covariance_type": model.covariance_type,
        "n_params": n_params,
        "train_samples": N_train,
        "val_samples": N_val,
        "test_samples": N_test,
        "train_log_likelihood": float(train_ll),
        "val_log_likelihood": float(val_ll),
        "val_predictive_ll": float(val_pred_ll),
        "test_log_likelihood": float(test_ll),
        "test_oos_predictive_ll": float(test_oos_pred_ll),
        "mean_train_ll": float(train_ll / N_train),
        "mean_val_ll": float(val_ll / N_val),
        "oos_predictive_ll": float(test_oos_pred_ll),
        "mean_oos_predictive_ll": float(mean_test_oos_pred_ll),
        "aic": ic["aic"],
        "bic": ic["bic"],
        "converged": bool(model.converged_),
        "n_iter_done": int(model.n_iter_done_),
        "diag_transmat": diag_trans.tolist(),
        "expected_duration_bars": expected_duration_bars,
        "expected_duration_hours": expected_duration_hours,
        "mean_occupancy_prob": mean_occupancy_prob,
        "hard_occupancy": hard_occupancy,
        "min_soft_occupancy": min_soft_occupancy,
        "min_hard_occupancy": min_hard_occupancy,
        "switch_frequency": switch_frequency,
        "collapsed_states": collapsed_states,
        "min_occupancy": min_occupancy,
        "is_stable": is_stable,
        "startprob": model.startprob_.tolist() if model.startprob_ is not None else None,
        "transmat": model.transmat_.tolist() if model.transmat_ is not None else None,
        "is_gmm": is_gmm,
        "n_mix": n_mix,
        "weights": model.weights_.tolist() if (is_gmm and model.weights_ is not None) else None,
        "state_means": model.means_.tolist() if model.means_ is not None else None,
        "state_covars": model.covars_.tolist() if model.covars_ is not None else None,
    }

