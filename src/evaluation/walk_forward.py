"""
src.evaluation.walk_forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Walk-forward out-of-sample evaluation engine, boundary policy handler,
and statistical hypothesis testing framework for EURUSD M15 HMM models.

Implements:
1. WalkForwardFold partitioning with 100% complete-bar filtering and boundary verification.
2. Sequential forward filtering under Policy A (continuous carry-over with FX calendar gap propagation)
   and Policy B (diagnostic independent sequence starting from model.startprob_).
3. Diebold-Mariano test with Newey-West HAC variance estimation across Bartlett kernel lags [4, 8, 16]
   and Harvey-Leybourne-Newbold (HLN 1997) finite-sample correction.
4. Holm-Bonferroni step-down Family-Wise Error Rate (FWER) multiple-testing adjustment.
5. Deterministic model fitting with seed schedule logging and SHA-256 parameter hashing.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from src.models.gaussian_hmm import GaussianHMM, _fast_logsumexp_1d, _fast_logsumexp_axis0
from src.models.gmm_hmm import GMMHMM
from src.models.forward_filter import ForwardFilter
from src.models.splits import identify_continuous_sequences
from src.inference.calendar import FXCalendarGapHandler, GapClassification
from src.inference.versioning import ModelArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Walk-Forward Fold Data Structure & Generation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """Container for an expanding-window walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    train_lengths: List[int]
    val_lengths: List[int]
    test_lengths: List[int]
    train_bars: int
    val_bars: int
    test_bars: int


# Exact boundary specifications from Section 3.2 of PHASE7_EXPERIMENTAL_DESIGN.md
FOLD_DEFINITIONS = [
    {
        "fold_id": 1,
        "train_start": "2023-01-01 23:30:00+00:00",
        "train_end": "2023-06-30 21:45:00+00:00",
        "expected_train_bars": 9472,
        "val_start": "2023-07-02 23:00:00+00:00",
        "val_end": "2023-07-31 23:30:00+00:00",
        "expected_val_bars": 1414,
        "test_start": "2023-08-01 00:00:00+00:00",
        "test_end": "2023-08-31 23:30:00+00:00",
        "expected_test_bars": 2120,
    },
    {
        "fold_id": 2,
        "train_start": "2023-01-01 23:30:00+00:00",
        "train_end": "2023-07-31 23:30:00+00:00",
        "expected_train_bars": 10886,
        "val_start": "2023-08-01 00:00:00+00:00",
        "val_end": "2023-08-31 23:30:00+00:00",
        "expected_val_bars": 2120,
        "test_start": "2023-09-01 00:00:00+00:00",
        "test_end": "2023-09-29 21:45:00+00:00",
        "expected_test_bars": 1887,
    },
    {
        "fold_id": 3,
        "train_start": "2023-01-01 23:30:00+00:00",
        "train_end": "2023-08-31 23:30:00+00:00",
        "expected_train_bars": 13006,
        "val_start": "2023-09-01 00:00:00+00:00",
        "val_end": "2023-09-29 21:45:00+00:00",
        "expected_val_bars": 1887,
        "test_start": "2023-10-01 22:00:00+00:00",
        "test_end": "2023-10-31 23:45:00+00:00",
        "expected_test_bars": 2056,
    },
    {
        "fold_id": 4,
        "train_start": "2023-01-01 23:30:00+00:00",
        "train_end": "2023-09-29 21:45:00+00:00",
        "expected_train_bars": 14893,
        "val_start": "2023-10-01 22:00:00+00:00",
        "val_end": "2023-10-31 23:45:00+00:00",
        "expected_val_bars": 2056,
        "test_start": "2023-11-01 00:00:00+00:00",
        "test_end": "2023-11-30 23:45:00+00:00",
        "expected_test_bars": 2027,
    },
    {
        "fold_id": 5,
        "train_start": "2023-01-01 23:30:00+00:00",
        "train_end": "2023-10-31 23:45:00+00:00",
        "expected_train_bars": 16949,
        "val_start": "2023-11-01 00:00:00+00:00",
        "val_end": "2023-11-30 23:45:00+00:00",
        "expected_val_bars": 2027,
        "test_start": "2023-12-01 00:00:00+00:00",
        "test_end": "2023-12-29 21:30:00+00:00",
        "expected_test_bars": 1860,
    },
]


def generate_walk_forward_folds(df: pd.DataFrame) -> List[WalkForwardFold]:
    """
    Generate 5 expanding-window walk-forward folds with complete-bar filtering
    and boundary verification.
    """
    # 1. Filter strictly complete candles
    if "is_complete" in df.columns:
        df_complete = df[df["is_complete"] == True].copy()
    else:
        df_complete = df.copy()

    # Ensure timestamp is UTC datetime
    df_complete["timestamp"] = pd.to_datetime(df_complete["timestamp"], utc=True)
    df_complete = df_complete.sort_values("timestamp").reset_index(drop=True)

    folds = []
    for spec in FOLD_DEFINITIONS:
        fid = spec["fold_id"]
        t_tr_start = pd.to_datetime(spec["train_start"])
        t_tr_end = pd.to_datetime(spec["train_end"])
        t_va_start = pd.to_datetime(spec["val_start"])
        t_va_end = pd.to_datetime(spec["val_end"])
        t_te_start = pd.to_datetime(spec["test_start"])
        t_te_end = pd.to_datetime(spec["test_end"])

        df_train = df_complete[(df_complete["timestamp"] >= t_tr_start) & (df_complete["timestamp"] <= t_tr_end)].copy()
        df_val = df_complete[(df_complete["timestamp"] >= t_va_start) & (df_complete["timestamp"] <= t_va_end)].copy()
        df_test = df_complete[(df_complete["timestamp"] >= t_te_start) & (df_complete["timestamp"] <= t_te_end)].copy()

        # Verify bar counts
        assert len(df_train) == spec["expected_train_bars"], (
            f"Fold {fid} train bar count mismatch: expected {spec['expected_train_bars']}, got {len(df_train)}"
        )
        assert len(df_val) == spec["expected_val_bars"], (
            f"Fold {fid} val bar count mismatch: expected {spec['expected_val_bars']}, got {len(df_val)}"
        )
        assert len(df_test) == spec["expected_test_bars"], (
            f"Fold {fid} test bar count mismatch: expected {spec['expected_test_bars']}, got {len(df_test)}"
        )

        # Verify zero overlap between partitions
        assert df_train["timestamp"].max() < df_val["timestamp"].min(), f"Fold {fid} train/val overlap!"
        assert df_val["timestamp"].max() < df_test["timestamp"].min(), f"Fold {fid} val/test overlap!"

        # Identify continuous trading sequences
        tr_lengths, _ = identify_continuous_sequences(df_train, expected_step_minutes=15)
        va_lengths, _ = identify_continuous_sequences(df_val, expected_step_minutes=15)
        te_lengths, _ = identify_continuous_sequences(df_test, expected_step_minutes=15)

        folds.append(
            WalkForwardFold(
                fold_id=fid,
                train_start=t_tr_start,
                train_end=t_tr_end,
                val_start=t_va_start,
                val_end=t_va_end,
                test_start=t_te_start,
                test_end=t_te_end,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                train_lengths=tr_lengths,
                val_lengths=va_lengths,
                test_lengths=te_lengths,
                train_bars=len(df_train),
                val_bars=len(df_val),
                test_bars=len(df_test),
            )
        )

    logger.info("Successfully generated and verified all %d walk-forward folds.", len(folds))
    return folds


# ---------------------------------------------------------------------------
# 2. Sequential Forward Filtering Engine (Policy A & Policy B)
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Results from sequential causal forward filtering."""
    pred_lls: np.ndarray             # (T,) one-step-ahead log P(O_t | O_{1:t-1})
    filtered_probs: np.ndarray       # (T, K) normalized P(S_t = j | O_{1:t})
    prior_probs: np.ndarray          # (T, K) normalized P(S_t = j | O_{1:t-1})
    terminal_posterior: np.ndarray   # (K,) posterior at last observation
    terminal_timestamp: pd.Timestamp # timestamp of last observation
    total_pred_ll: float             # sum of pred_lls
    mean_pred_ll: float              # mean of pred_lls
    switch_frequency: float          # fraction of adjacent bars where argmax state changes
    entropy: np.ndarray              # (T,) Shannon entropy of filtered_probs
    gap_diagnostics: List[Dict[str, Any]] # audit trail of gap classifications


def sequential_forward_filter(
    model: Union[GaussianHMM, GMMHMM],
    df: pd.DataFrame,
    feature_cols: List[str],
    prev_posterior: Optional[np.ndarray] = None,
    prev_timestamp: Optional[pd.Timestamp] = None,
    policy: Literal["A", "B"] = "A",
    gap_handler: Optional[FXCalendarGapHandler] = None,
) -> FilterResult:
    """
    Perform sequential causal forward filtering over observation sequence df.

    Parameters
    ----------
    model : GaussianHMM or GMMHMM
        Fitted HMM instance with frozen parameters.
    df : pd.DataFrame
        Dataframe containing complete observations and 'timestamp'.
    feature_cols : List[str]
        List of feature column names (e.g. ['r_oc', 'log_vol']).
    prev_posterior : Optional[np.ndarray], shape (K,)
        Belief distribution from the last bar of the preceding window.
    prev_timestamp : Optional[pd.Timestamp]
        Timestamp of the last bar of the preceding window.
    policy : Literal['A', 'B']
        - 'A' (Continuous Forward): Carries prev_posterior across the window boundary,
          propagating via A^n during open sessions, and resetting strictly to model.startprob_
          upon market closure.
        - 'B' (Independent Benchmark): Resets unconditionally to model.startprob_ at bar 0.
    gap_handler : Optional[FXCalendarGapHandler]
        FX calendar handler for gap classification. If None, instantiates a new one.

    Returns
    -------
    FilterResult
        Dataclass containing predictive log-likelihoods, filtered posteriors,
        priors, entropy, switching frequency, and gap audit diagnostics.
    """
    if gap_handler is None:
        gap_handler = FXCalendarGapHandler()

    ff = ForwardFilter(model=model)
    K = model.n_components
    T = len(df)

    X = df[feature_cols].values
    timestamps = pd.to_datetime(df["timestamp"], utc=True).values

    pred_lls = np.empty(T, dtype=np.float64)
    filtered_probs = np.empty((T, K), dtype=np.float64)
    prior_probs = np.empty((T, K), dtype=np.float64)
    gap_diagnostics = []

    current_posterior = None
    last_ts = None

    # Handle boundary initialization (Bar 0)
    ts_0 = pd.Timestamp(timestamps[0])
    if policy == "A" and prev_posterior is not None and prev_timestamp is not None:
        # Policy A: Continuous carry-over from preceding window
        prev_ts = pd.Timestamp(prev_timestamp)
        gap_info = gap_handler.classify(t_prev=prev_ts, t_curr=ts_0)

        if gap_info.requires_reset or gap_info.gap_type == "market_closure":
            # Weekend closure or calendar reset: reset strictly to model.startprob_
            log_prior_0 = ff.log_startprob.copy()
            action = "reset_to_startprob_closure"
        elif gap_info.gap_type == "active_missing":
            # Active session gap (e.g. partial bar skipped): propagate via A^n
            n_trans = gap_info.n_transitions
            A_pow = np.linalg.matrix_power(model.transmat_, n_trans)
            prior_prop = prev_posterior @ A_pow
            prior_prop = np.maximum(prior_prop, 1e-15)
            log_prior_0 = np.log(prior_prop / np.sum(prior_prop))
            action = f"propagate_A_pow_{n_trans}"
        elif gap_info.gap_type == "normal":
            # Normal consecutive 15m step
            log_prev = np.log(np.maximum(prev_posterior, 1e-15))
            log_prior_0 = _fast_logsumexp_axis0(log_prev[:, None] + ff.log_transmat)
            action = "propagate_A_normal"
        else:
            log_prior_0 = ff.log_startprob.copy()
            action = "reset_uncertain"

        gap_diagnostics.append({
            "step": 0,
            "timestamp": ts_0.isoformat(),
            "prev_timestamp": prev_ts.isoformat(),
            "gap_type": gap_info.gap_type,
            "n_transitions": gap_info.n_transitions,
            "requires_reset": gap_info.requires_reset,
            "policy_action": action,
            "is_boundary": True,
        })
    else:
        # Policy B or fresh initial start
        log_prior_0 = ff.log_startprob.copy()
        action = "fresh_startprob"
        gap_diagnostics.append({
            "step": 0,
            "timestamp": ts_0.isoformat(),
            "prev_timestamp": None,
            "gap_type": "initial",
            "n_transitions": 0,
            "requires_reset": True,
            "policy_action": action,
            "is_boundary": True,
        })

    # Bar 0 measurement update
    log_b_0 = ff.compute_single_log_emission(X[0])
    pred_ll_0 = float(_fast_logsumexp_1d(log_prior_0 + log_b_0))
    log_post_0 = log_prior_0 + log_b_0 - pred_ll_0
    post_0 = np.exp(np.clip(log_post_0, -700, 0))
    post_0 /= np.sum(post_0)

    prior_0 = np.exp(np.clip(log_prior_0, -700, 0))
    prior_0 /= np.sum(prior_0)

    pred_lls[0] = pred_ll_0
    filtered_probs[0] = post_0
    prior_probs[0] = prior_0
    current_posterior = post_0
    last_ts = ts_0

    # Sequential loop over subsequent bars t = 1 .. T-1
    for t in range(1, T):
        ts_t = pd.Timestamp(timestamps[t])
        x_t = X[t]

        gap_info = gap_handler.classify(t_prev=last_ts, t_curr=ts_t)

        if gap_info.requires_reset or gap_info.gap_type == "market_closure":
            log_prior_t = ff.log_startprob.copy()
            action = "reset_to_startprob_closure"
        elif gap_info.gap_type == "active_missing":
            n_trans = gap_info.n_transitions
            A_pow = np.linalg.matrix_power(model.transmat_, n_trans)
            prior_prop = current_posterior @ A_pow
            prior_prop = np.maximum(prior_prop, 1e-15)
            log_prior_t = np.log(prior_prop / np.sum(prior_prop))
            action = f"propagate_A_pow_{n_trans}"
        elif gap_info.gap_type == "normal":
            log_prev = np.log(np.maximum(current_posterior, 1e-15))
            log_prior_t = _fast_logsumexp_axis0(log_prev[:, None] + ff.log_transmat)
            action = "propagate_A_normal"
        else:
            log_prior_t = ff.log_startprob.copy()
            action = "reset_uncertain"

        if gap_info.gap_type != "normal":
            gap_diagnostics.append({
                "step": t,
                "timestamp": ts_t.isoformat(),
                "prev_timestamp": last_ts.isoformat(),
                "gap_type": gap_info.gap_type,
                "n_transitions": gap_info.n_transitions,
                "requires_reset": gap_info.requires_reset,
                "policy_action": action,
                "is_boundary": False,
            })

        log_b_t = ff.compute_single_log_emission(x_t)
        pred_ll_t = float(_fast_logsumexp_1d(log_prior_t + log_b_t))
        log_post_t = log_prior_t + log_b_t - pred_ll_t
        post_t = np.exp(np.clip(log_post_t, -700, 0))
        post_t /= np.sum(post_t)

        prior_t = np.exp(np.clip(log_prior_t, -700, 0))
        prior_t /= np.sum(prior_t)

        pred_lls[t] = pred_ll_t
        filtered_probs[t] = post_t
        prior_probs[t] = prior_t
        current_posterior = post_t
        last_ts = ts_t

    # Switching frequency
    argmax_states = np.argmax(filtered_probs, axis=1)
    if T > 1:
        sw_freq = float(np.sum(argmax_states[1:] != argmax_states[:-1]) / (T - 1))
    else:
        sw_freq = 0.0

    # Shannon Entropy: -sum_k p_k * log(p_k)
    clipped_probs = np.maximum(filtered_probs, 1e-12)
    entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)

    return FilterResult(
        pred_lls=pred_lls,
        filtered_probs=filtered_probs,
        prior_probs=prior_probs,
        terminal_posterior=current_posterior,
        terminal_timestamp=last_ts,
        total_pred_ll=float(np.sum(pred_lls)),
        mean_pred_ll=float(np.mean(pred_lls)),
        switch_frequency=sw_freq,
        entropy=entropy,
        gap_diagnostics=gap_diagnostics,
    )


# ---------------------------------------------------------------------------
# 3. Statistical Testing: Diebold-Mariano with Newey-West HAC & HLN
# ---------------------------------------------------------------------------

@dataclass
class DieboldMarianoResult:
    """Results of Diebold-Mariano test with Harvey-Leybourne-Newbold adjustment."""
    mean_loss_diff: float
    sample_size: int
    hac_variance: Dict[int, float]
    dm_statistic: Dict[int, float]
    p_value: Dict[int, float]
    is_significant_05: Dict[int, bool]


def diebold_mariano_test(
    loss_diff: np.ndarray,
    lags: List[int] = [4, 8, 16],
) -> DieboldMarianoResult:
    """
    Compute Diebold-Mariano test statistic with Newey-West HAC spectral density
    estimator (triangular Bartlett kernel) and Harvey-Leybourne-Newbold (HLN 1997)
    finite-sample adjustment.

    Parameters
    ----------
    loss_diff : np.ndarray, shape (T,)
        Loss differential sequence d_t = log P_{M_A}(O_t) - log P_{M_B}(O_t).
        Higher values indicate Model A outperforms Model B.
    lags : List[int]
        List of Newey-West truncation lag orders (default [4, 8, 16]).

    Returns
    -------
    DieboldMarianoResult
    """
    d = np.asarray(loss_diff, dtype=np.float64)
    T = len(d)
    if T < 2:
        raise ValueError("Diebold-Mariano test requires at least 2 observations.")

    d_bar = float(np.mean(d))
    dev = d - d_bar

    # Sample autocovariance at lag 0: gamma_0
    gamma_0 = float(np.sum(dev ** 2) / T)

    hac_vars = {}
    dm_stats = {}
    p_vals = {}
    sig_05 = {}

    hln_factor = np.sqrt((T - 1) / T)

    for L in lags:
        # Sample autocovariances gamma_l with Bartlett kernel weight (1 - l / (L + 1))
        weighted_sum = 0.0
        for l in range(1, min(L + 1, T)):
            gamma_l = float(np.sum(dev[l:] * dev[:-l]) / T)
            weight = 1.0 - (l / (L + 1))
            weighted_sum += 2.0 * weight * gamma_l

        # Spectral density estimate at frequency zero
        var_hac = max(gamma_0 + weighted_sum, 1e-12)
        se = np.sqrt(var_hac / T)

        # HLN finite sample corrected statistic
        dm_stat = float((d_bar / se) * hln_factor)

        # Two-sided p-value from standard normal
        p_val = float(2.0 * stats.norm.sf(abs(dm_stat)))

        hac_vars[L] = var_hac
        dm_stats[L] = dm_stat
        p_vals[L] = p_val
        sig_05[L] = bool(p_val < 0.05)

    return DieboldMarianoResult(
        mean_loss_diff=d_bar,
        sample_size=T,
        hac_variance=hac_vars,
        dm_statistic=dm_stats,
        p_value=p_vals,
        is_significant_05=sig_05,
    )


def holm_bonferroni_adjustment(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Apply Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER)
    across m simultaneous hypothesis tests.

    Parameters
    ----------
    p_values : Dict[str, float]
        Mapping from comparison key to raw p-value.

    Returns
    -------
    Dict[str, float]
        Mapping from comparison key to Holm-Bonferroni adjusted p-value.
    """
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    adj_p = {}

    cum_max = 0.0
    for rank, (key, raw_p) in enumerate(items):
        multiplier = m - rank
        val = min(1.0, multiplier * raw_p)
        cum_max = max(cum_max, val)
        adj_p[key] = min(1.0, cum_max)

    # Return in original dictionary order
    return {k: adj_p[k] for k in p_values.keys()}


# ---------------------------------------------------------------------------
# 4. Model Training & Provenance Wrapper
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result and reproducibility record of a model fit."""
    model: Union[GaussianHMM, GMMHMM]
    model_name: str
    fold_id: int
    base_random_state: int
    n_restarts: int
    restart_seed_schedule: List[int]
    parameter_sha256: str
    best_log_likelihood: float
    n_iter_done: int
    converged: bool
    reached_max_iter: bool
    training_time_seconds: float
    parameters: Dict[str, Any]


def fit_model_with_seed_schedule(
    model_class: Union[type(GaussianHMM), type(GMMHMM)],
    init_kwargs: Dict[str, Any],
    X_train: np.ndarray,
    lengths: List[int],
    model_name: str,
    fold_id: int,
    n_restarts: int = 5,
    base_random_state: int = 42,
) -> FitResult:
    """
    Fit an HMM model instance with a deterministic multi-restart seed schedule
    and capture full provenance metadata.
    """
    t0 = time.perf_counter()

    # Determine deterministic restart seed sequence based on codebase formulas
    is_gmm = (model_class == GMMHMM)
    stride = 107 if is_gmm else 101
    seed_schedule = [base_random_state + r * stride for r in range(n_restarts)]

    kwargs = dict(init_kwargs)
    kwargs["random_state"] = base_random_state
    model = model_class(**kwargs)

    # Execute fit
    model.fit(X_train, lengths=lengths, n_restarts=n_restarts)
    elapsed = time.perf_counter() - t0

    # Provenance metadata
    params = model.to_dict()
    serialized = json.dumps(params, sort_keys=True)
    param_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    max_iter = init_kwargs.get("n_iter", 35)
    converged = bool(model.converged_)
    n_iter_done = int(model.n_iter_done_)
    reached_max = bool(n_iter_done >= max_iter and not converged)

    return FitResult(
        model=model,
        model_name=model_name,
        fold_id=fold_id,
        base_random_state=base_random_state,
        n_restarts=n_restarts,
        restart_seed_schedule=seed_schedule,
        parameter_sha256=param_hash,
        best_log_likelihood=float(model.best_log_likelihood_),
        n_iter_done=n_iter_done,
        converged=converged,
        reached_max_iter=reached_max,
        training_time_seconds=elapsed,
        parameters=params,
    )
