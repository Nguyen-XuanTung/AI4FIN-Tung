"""
tests.test_walk_forward
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Phase 7 walk-forward out-of-sample evaluation engine,
boundary propagation policies, and statistical hypothesis testing.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.walk_forward import (
    WalkForwardFold,
    generate_walk_forward_folds,
    sequential_forward_filter,
    diebold_mariano_test,
    holm_bonferroni_adjustment,
    fit_model_with_seed_schedule,
    FOLD_DEFINITIONS,
)
from src.models.gaussian_hmm import GaussianHMM
from src.models.gmm_hmm import GMMHMM
from src.features.candidates import compute_log_returns_oc, compute_log_volatility
from src.inference.calendar import FXCalendarGapHandler


@pytest.fixture(scope="module")
def complete_eurusd_df():
    """Load the preprocessed complete 2023 dataset."""
    df = pd.read_parquet("data/processed/EURUSD_M15_2023.parquet")
    df_c = df[df["is_complete"] == True].copy()
    df_c["r_oc"] = compute_log_returns_oc(df_c)
    df_c["log_vol"] = compute_log_volatility(df_c)
    return df_c


# ---------------------------------------------------------------------------
# 1. Fold Generation & Window Integrity Tests
# ---------------------------------------------------------------------------

def test_generate_walk_forward_folds_integrity(complete_eurusd_df):
    """Verify exact complete bar counts, chronological order, and non-overlap."""
    folds = generate_walk_forward_folds(complete_eurusd_df)
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    expected_counts = [
        (1, 9472, 1414, 2120),
        (2, 10886, 2120, 1887),
        (3, 13006, 1887, 2056),
        (4, 14893, 2056, 2027),
        (5, 16949, 2027, 1860),
    ]

    for fold, (fid, exp_tr, exp_va, exp_te) in zip(folds, expected_counts):
        assert fold.fold_id == fid
        assert fold.train_bars == exp_tr
        assert fold.val_bars == exp_va
        assert fold.test_bars == exp_te

        # Expanding window verification: Train start is constant
        assert fold.train_start == pd.Timestamp("2023-01-01 23:30:00+00:00")

        # Zero accidental overlap
        assert fold.df_train["timestamp"].max() < fold.df_val["timestamp"].min()
        assert fold.df_val["timestamp"].max() < fold.df_test["timestamp"].min()

        # Multi-sequence segmentation
        assert sum(fold.train_lengths) == exp_tr
        assert sum(fold.val_lengths) == exp_va
        assert sum(fold.test_lengths) == exp_te


# ---------------------------------------------------------------------------
# 2. Boundary Policy & Filtering Tests
# ---------------------------------------------------------------------------

def test_boundary_policy_a_active_gap_propagation():
    """
    Test Policy A propagating belief through an active session gap
    via A^2 when 1 incomplete candle is skipped (Folds 1 & 2 boundary).
    """
    # Create simple 2-state Gaussian HMM
    model = GaussianHMM(n_components=2, random_state=42)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.means_ = np.array([[0.0, -7.0], [0.0, -6.0]])
    model.covars_ = np.array([[[1e-6, 0.0], [0.0, 0.1]], [[1e-6, 0.0], [0.0, 0.1]]])

    # Terminal posterior from preceding window
    prev_posterior = np.array([0.9, 0.1])
    prev_ts = pd.Timestamp("2023-07-31 23:30:00+00:00")  # Monday night

    # Incoming test window starting at 00:00 (30 min delta = 2 steps)
    test_df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2023-08-01 00:00:00+00:00"),
            pd.Timestamp("2023-08-01 00:15:00+00:00"),
        ],
        "r_oc": [0.0, 0.0],
        "log_vol": [-7.0, -7.0],
    })

    res = sequential_forward_filter(
        model=model,
        df=test_df,
        feature_cols=["r_oc", "log_vol"],
        prev_posterior=prev_posterior,
        prev_timestamp=prev_ts,
        policy="A",
    )

    # Check that gap diagnostic recorded A^2 propagation
    assert len(res.gap_diagnostics) >= 1
    boundary_diag = res.gap_diagnostics[0]
    assert boundary_diag["gap_type"] == "active_missing"
    assert boundary_diag["n_transitions"] == 2
    assert boundary_diag["requires_reset"] is False
    assert boundary_diag["policy_action"] == "propagate_A_pow_2"

    # Expected prior = prev_posterior @ A^2
    A2 = model.transmat_ @ model.transmat_
    expected_prior = prev_posterior @ A2
    expected_prior /= expected_prior.sum()
    np.testing.assert_allclose(res.prior_probs[0], expected_prior, atol=1e-5)


def test_boundary_policy_a_weekend_reset():
    """
    Test Policy A resetting strictly to model.startprob_
    across a Friday close to Sunday open market closure (Fold 3 boundary).
    """
    model = GaussianHMM(n_components=2, random_state=42)
    model.startprob_ = np.array([0.7, 0.3])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.means_ = np.array([[0.0, -7.0], [0.0, -6.0]])
    model.covars_ = np.array([[[1e-6, 0.0], [0.0, 0.1]], [[1e-6, 0.0], [0.0, 0.1]]])

    prev_posterior = np.array([0.1, 0.9])
    prev_ts = pd.Timestamp("2023-09-29 21:45:00+00:00")  # Friday close

    # First test bar is Sunday evening open
    test_df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2023-10-01 22:00:00+00:00"),
            pd.Timestamp("2023-10-01 22:15:00+00:00"),
        ],
        "r_oc": [0.0, 0.0],
        "log_vol": [-7.0, -7.0],
    })

    res = sequential_forward_filter(
        model=model,
        df=test_df,
        feature_cols=["r_oc", "log_vol"],
        prev_posterior=prev_posterior,
        prev_timestamp=prev_ts,
        policy="A",
    )

    # Check that gap diagnostic recorded weekend closure reset
    boundary_diag = res.gap_diagnostics[0]
    assert boundary_diag["gap_type"] == "market_closure"
    assert boundary_diag["requires_reset"] is True
    assert boundary_diag["policy_action"] == "reset_to_startprob_closure"

    # Prior must match startprob_, NOT prev_posterior @ A
    np.testing.assert_allclose(res.prior_probs[0], model.startprob_, atol=1e-5)


def test_boundary_policy_b_fresh_reset():
    """Test Policy B resetting unconditionally to model.startprob_."""
    model = GaussianHMM(n_components=2, random_state=42)
    model.startprob_ = np.array([0.85, 0.15])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.means_ = np.array([[0.0, -7.0], [0.0, -6.0]])
    model.covars_ = np.array([[[1e-6, 0.0], [0.0, 0.1]], [[1e-6, 0.0], [0.0, 0.1]]])

    prev_posterior = np.array([0.05, 0.95])
    prev_ts = pd.Timestamp("2023-07-31 23:30:00+00:00")

    test_df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2023-08-01 00:00:00+00:00"),
            pd.Timestamp("2023-08-01 00:15:00+00:00"),
        ],
        "r_oc": [0.0, 0.0],
        "log_vol": [-7.0, -7.0],
    })

    res_b = sequential_forward_filter(
        model=model,
        df=test_df,
        feature_cols=["r_oc", "log_vol"],
        prev_posterior=prev_posterior,
        prev_timestamp=prev_ts,
        policy="B",
    )

    np.testing.assert_allclose(res_b.prior_probs[0], model.startprob_, atol=1e-5)


def test_filtering_strict_causality():
    """Verify that altering future observations does not alter past filtered posteriors."""
    model = GaussianHMM(n_components=2, random_state=42)
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
    model.means_ = np.array([[0.0, -7.0], [0.0, -6.0]])
    model.covars_ = np.array([[[1e-6, 0.0], [0.0, 0.1]], [[1e-6, 0.0], [0.0, 0.1]]])

    ts = [pd.Timestamp("2023-08-01 00:00:00+00:00") + pd.Timedelta(minutes=15 * i) for i in range(10)]
    df1 = pd.DataFrame({
        "timestamp": ts,
        "r_oc": [0.0001] * 10,
        "log_vol": [-7.2] * 10,
    })

    # Create perturbed version where bar 9 is violently perturbed
    df2 = df1.copy()
    df2.loc[9, "r_oc"] = 0.05
    df2.loc[9, "log_vol"] = -3.0

    res1 = sequential_forward_filter(model, df1, ["r_oc", "log_vol"])
    res2 = sequential_forward_filter(model, df2, ["r_oc", "log_vol"])

    # Bars 0 to 8 must be bitwise identical
    np.testing.assert_allclose(res1.filtered_probs[:9], res2.filtered_probs[:9], atol=1e-14)
    np.testing.assert_allclose(res1.pred_lls[:9], res2.pred_lls[:9], atol=1e-14)


# ---------------------------------------------------------------------------
# 3. Statistical Testing: Diebold-Mariano & Holm-Bonferroni
# ---------------------------------------------------------------------------

def test_diebold_mariano_identical_forecast():
    """Identical models (d_t = 0 everywhere) must yield DM = 0 and p = 1.0."""
    loss_diff = np.zeros(500)
    res = diebold_mariano_test(loss_diff, lags=[4, 8, 16])
    assert res.mean_loss_diff == 0.0
    for L in [4, 8, 16]:
        assert res.dm_statistic[L] == 0.0
        assert res.p_value[L] == 1.0
        assert res.is_significant_05[L] is False


def test_diebold_mariano_significance_and_bartlett():
    """Consistent positive loss differential must yield positive DM and low p-value."""
    np.random.seed(42)
    # Model A consistently better by +0.1 nats with mild noise
    loss_diff = 0.1 + np.random.normal(0, 0.2, size=1000)
    res = diebold_mariano_test(loss_diff, lags=[4, 8, 16])

    assert res.mean_loss_diff > 0.0
    for L in [4, 8, 16]:
        assert res.dm_statistic[L] > 5.0  # highly significant
        assert res.p_value[L] < 1e-4
        assert res.is_significant_05[L] is True
        assert res.hac_variance[L] > 0.0


def test_holm_bonferroni_adjustment():
    """Test step-down adjustment of p-values across multiple comparisons."""
    raw_p = {
        "M3_vs_M1": 0.01,
        "M2_vs_M1": 0.04,
        "M3_vs_M2": 0.20,
    }
    adj_p = holm_bonferroni_adjustment(raw_p)

    # For m=3:
    # rank 0: min(1, 3 * 0.01) = 0.03
    # rank 1: min(1, max(0.03, 2 * 0.04)) = 0.08
    # rank 2: min(1, max(0.08, 1 * 0.20)) = 0.20
    assert pytest.approx(adj_p["M3_vs_M1"], abs=1e-5) == 0.03
    assert pytest.approx(adj_p["M2_vs_M1"], abs=1e-5) == 0.08
    assert pytest.approx(adj_p["M3_vs_M2"], abs=1e-5) == 0.20


# ---------------------------------------------------------------------------
# 4. Model Training & Seed Schedule
# ---------------------------------------------------------------------------

def test_fit_model_with_seed_schedule_reproducibility():
    """Verify deterministic seed schedule and parameter hashing."""
    np.random.seed(42)
    X = np.random.normal(size=(500, 2))
    lengths = [250, 250]

    fit_res1 = fit_model_with_seed_schedule(
        model_class=GaussianHMM,
        init_kwargs={"n_components": 2, "n_iter": 10},
        X_train=X,
        lengths=lengths,
        model_name="Gaussian_K2",
        fold_id=1,
        n_restarts=3,
        base_random_state=42,
    )

    fit_res2 = fit_model_with_seed_schedule(
        model_class=GaussianHMM,
        init_kwargs={"n_components": 2, "n_iter": 10},
        X_train=X,
        lengths=lengths,
        model_name="Gaussian_K2",
        fold_id=1,
        n_restarts=3,
        base_random_state=42,
    )

    # Exact bitwise reproducibility
    assert fit_res1.parameter_sha256 == fit_res2.parameter_sha256
    assert fit_res1.best_log_likelihood == fit_res2.best_log_likelihood
    assert fit_res1.restart_seed_schedule == [42, 143, 244]
