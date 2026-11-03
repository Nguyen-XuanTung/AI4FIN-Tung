"""
tests.test_gaussian_hmm
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Gaussian HMM and Forward Filtering Engine:
1. Probability normalization: sum_j P(S_t=j | x_{1:t}) == 1.0 at every timestep.
2. Sequential vs batch consistency: identical output within float precision.
3. Temporal causality: altering future observations does not change past filtered probabilities.
4. Numerical stability in log-space: robustness to extreme outliers without underflow/NaN.
5. Prior state initialization: alpha_0 correctly reflects startprob.
6. Data gap handling: policy triggers correctly across breaks.
7. Information criteria & free parameter count calculation.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.gaussian_hmm import GaussianHMM
from src.models.forward_filter import ForwardFilter, compute_filtered_probabilities
from src.models.evaluation import calculate_num_free_params, compute_information_criteria
from src.models.splits import identify_continuous_sequences, split_chronological


@pytest.fixture
def fitted_toy_model():
    """Create a deterministic 2-state 2-D Gaussian HMM for exact mathematical tests."""
    model = GaussianHMM(n_components=2, covariance_type="full", cov_reg=1e-4)
    model.n_features_ = 2
    model.startprob_ = np.array([0.6, 0.4], dtype=np.float64)
    model.transmat_ = np.array([[0.85, 0.15], [0.10, 0.90]], dtype=np.float64)
    # State 0: Low volatility (mean 0, small variance)
    # State 1: High volatility (mean 0, large variance)
    model.means_ = np.array([[0.0, -8.0], [0.0, -6.0]], dtype=np.float64)
    model.covars_ = np.array(
        [
            [[1e-4, 0.0], [0.0, 0.2]],
            [[1e-3, 0.0], [0.0, 0.8]],
        ],
        dtype=np.float64,
    )
    model.converged_ = True
    model.n_iter_done_ = 10
    model.best_log_likelihood_ = -100.0
    return model


def test_probability_normalization(fitted_toy_model):
    """Verify that filtered probabilities sum to exactly 1.0 at every timestep."""
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(100, 2))
    engine = ForwardFilter(model=fitted_toy_model)
    res = engine.filter_batch(X)
    probs = res["filtered_probs"]

    assert probs.shape == (100, 2)
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_sequential_vs_batch_equivalence(fitted_toy_model):
    """Verify that sequential bar-by-bar filtering equals batch filtering."""
    rng = np.random.RandomState(123)
    X = rng.normal(0, 1, size=(50, 2))
    engine = ForwardFilter(model=fitted_toy_model)

    # Batch filtering
    batch_res = engine.filter_batch(X)

    # Sequential bar-by-bar filtering
    seq_probs = []
    seq_plls = []
    prev_prob = None
    for t in range(len(X)):
        prob, prior, pll = engine.step(x_t=X[t], prev_filtered_prob=prev_prob)
        seq_probs.append(prob)
        seq_plls.append(pll)
        prev_prob = prob

    seq_probs_arr = np.array(seq_probs)
    seq_plls_arr = np.array(seq_plls)

    np.testing.assert_allclose(batch_res["filtered_probs"], seq_probs_arr, atol=1e-12)
    np.testing.assert_allclose(batch_res["pred_log_likelihoods"], seq_plls_arr, atol=1e-12)


def test_temporal_causality_invariance(fitted_toy_model):
    """
    Verify strict temporal causality: altering future observations x_{t+1:T}
    must NEVER alter past filtered probabilities P(S_t | x_{1:t}).
    """
    rng = np.random.RandomState(999)
    # Generate observations around state 0 (mean [0.0, -8.0])
    X_original = np.zeros((40, 2))
    X_original[:, 0] = rng.normal(0.0, 0.01, size=40)
    X_original[:, 1] = rng.normal(-8.0, 0.3, size=40)
    engine = ForwardFilter(model=fitted_toy_model)

    res_orig = engine.filter_batch(X_original)
    cutoff = 20

    # Modify future observations after cutoff to sit around state 1 (mean [0.0, -6.0])
    X_modified = X_original.copy()
    X_modified[cutoff:, 1] = rng.normal(-6.0, 0.3, size=20)

    res_mod = engine.filter_batch(X_modified)

    # Past filtered probabilities up to cutoff must be identical
    np.testing.assert_allclose(
        res_orig["filtered_probs"][:cutoff],
        res_mod["filtered_probs"][:cutoff],
        atol=1e-12,
        err_msg="Temporal causality violation: past probabilities changed after future shock!",
    )
    # Future probabilities after cutoff should clearly differ
    diff_future = np.abs(res_orig["filtered_probs"][cutoff:] - res_mod["filtered_probs"][cutoff:])
    assert np.max(diff_future) > 0.10


def test_numerical_stability_extreme_outliers(fitted_toy_model):
    """Verify log-space calculations prevent underflow/NaN under 100-sigma outliers."""
    X_extreme = np.array(
        [
            [0.0, -7.0],
            [1e6, 1e6],       # Extreme positive outlier
            [-1e6, -1e6],     # Extreme negative outlier
            [0.0, -7.0],
        ]
    )
    engine = ForwardFilter(model=fitted_toy_model)
    res = engine.filter_batch(X_extreme)
    probs = res["filtered_probs"]

    assert not np.isnan(probs).any(), "NaN detected in filtered probabilities!"
    assert not np.isinf(probs).any(), "Inf detected in filtered probabilities!"
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)


def test_prior_state_initialization(fitted_toy_model):
    """Verify that the initial forward step correctly utilizes startprob_."""
    engine = ForwardFilter(model=fitted_toy_model)
    x0 = np.array([0.0, -8.0])  # Closer to state 0 mean
    prob, prior, pll = engine.step(x_t=x0, prev_filtered_prob=None)

    # Prior before seeing emission must equal startprob_
    np.testing.assert_allclose(prior, fitted_toy_model.startprob_, atol=1e-12)
    # After update, state 0 probability should be higher because x0 is at state 0 mean
    assert prob[0] > prob[1]


def test_data_gap_handling(fitted_toy_model):
    """Verify that gaps between observations trigger the gap policy appropriately."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00+00:00"),
        pd.Timestamp("2023-01-02 10:15:00+00:00"),
        pd.Timestamp("2023-01-02 10:30:00+00:00"),
        pd.Timestamp("2023-01-05 10:00:00+00:00"),  # 3-day gap!
        pd.Timestamp("2023-01-05 10:15:00+00:00"),
    ]
    X = np.zeros((5, 2))
    engine_reset = ForwardFilter(model=fitted_toy_model, gap_policy="reset_prior", max_gap_minutes=15)
    res = engine_reset.filter_batch(X, timestamps=timestamps)

    assert res["is_gap"][3] == True
    assert res["is_gap"][0] == False
    assert res["is_gap"][1] == False
    assert res["is_gap"][2] == False
    assert res["is_gap"][4] == False

    # Under reset_prior policy, prior at gap index 3 must reset to startprob_
    np.testing.assert_allclose(res["prior_probs"][3], fitted_toy_model.startprob_, atol=1e-12)


def test_calculate_num_free_params():
    """Verify free parameter count calculation for full and diagonal covariance."""
    # K=2, D=2, full cov:
    # startprob: 1
    # transmat: 2 * 1 = 2
    # means: 2 * 2 = 4
    # covars: 2 * (2 * 3 / 2) = 6
    # Total = 1 + 2 + 4 + 6 = 13
    p_full = calculate_num_free_params(n_components=2, n_features=2, covariance_type="full")
    assert p_full == 13

    # K=3, D=3, full cov:
    # startprob: 2
    # transmat: 3 * 2 = 6
    # means: 3 * 3 = 9
    # covars: 3 * (3 * 4 / 2) = 18
    # Total = 2 + 6 + 9 + 18 = 35
    p_full_3 = calculate_num_free_params(n_components=3, n_features=3, covariance_type="full")
    assert p_full_3 == 35


def test_split_chronological_integrity():
    """Verify chronological split ordering, complete-only filtering, and metadata."""
    dates = pd.date_range("2023-01-01", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 1.0,
            "high": 1.01,
            "low": 0.99,
            "close": 1.0,
            "r_oc": 0.0,
            "log_vol": -7.5,
            "is_complete": [True] * 90 + [False] * 10,
        }
    )
    splits = split_chronological(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, filter_incomplete=True)

    assert len(splits["train"]) == 54   # 90 complete * 0.6 = 54
    assert len(splits["val"]) == 18     # 90 complete * 0.2 = 18
    assert len(splits["test"]) == 18    # 90 complete * 0.2 = 18
    assert splits["metadata"]["excluded_partial_count"] == 10

    # Strict chronological order check
    assert splits["train"]["timestamp"].max() < splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() < splits["test"]["timestamp"].min()


def test_continuous_sequences_identification():
    """Verify detection of sequence gaps."""
    timestamps = [
        pd.Timestamp("2023-01-01 00:00:00+00:00"),
        pd.Timestamp("2023-01-01 00:15:00+00:00"),
        pd.Timestamp("2023-01-01 00:30:00+00:00"),
        pd.Timestamp("2023-01-01 02:00:00+00:00"),  # gap
        pd.Timestamp("2023-01-01 02:15:00+00:00"),
    ]
    df = pd.DataFrame({"timestamp": timestamps})
    lengths, segments = identify_continuous_sequences(df, expected_step_minutes=15)

    assert lengths == [3, 2]
    assert segments == [(0, 3), (3, 5)]


def test_gaussian_hmm_fit_synthetic():
    """Verify that Gaussian HMM fits on synthetic two-state data and achieves convergence."""
    rng = np.random.RandomState(42)
    # Generate 300 observations from a clear two-state distribution
    s0 = rng.normal(loc=-1.0, scale=0.2, size=(150, 1))
    s1 = rng.normal(loc=+1.0, scale=0.2, size=(150, 1))
    X = np.vstack([s0, s1])
    lengths = [150, 150]

    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=30, random_state=42)
    model.fit(X, lengths=lengths, n_restarts=3)

    assert model.converged_
    assert len(model.means_) == 2
    # Verify separation of means
    means_sorted = np.sort(model.means_.ravel())
    assert means_sorted[0] < 0.0
    assert means_sorted[1] > 0.0


def test_emission_likelihood_model_and_forward_filter_agree(fitted_toy_model):
    """
    Verify that emission log-likelihoods computed by GaussianHMM and ForwardFilter
    agree numerically across all states and timesteps without double-regularization.
    """
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(50, 2))

    # 1. Model emission log-likelihoods
    hmm_log_b = fitted_toy_model._compute_log_emissions(X)

    # 2. ForwardFilter emission log-likelihoods
    engine = ForwardFilter(model=fitted_toy_model)
    ff_log_b = np.array([engine.compute_single_log_emission(x) for x in X])

    # Check exact numerical agreement
    np.testing.assert_allclose(
        hmm_log_b,
        ff_log_b,
        rtol=1e-10,
        atol=1e-10,
        err_msg="GaussianHMM and ForwardFilter emission likelihoods disagree!",
    )

    # 3. Serialization reproducibility: loaded model computes identical score
    d = fitted_toy_model.to_dict()
    loaded_model = GaussianHMM.from_dict(d)
    np.testing.assert_allclose(
        fitted_toy_model.score(X),
        loaded_model.score(X),
        atol=1e-10,
        err_msg="Deserialized GaussianHMM score does not match original model!",
    )


def test_restart_selection_best_log_likelihood_matches_score():
    """
    Regression test: verify that best_log_likelihood_ matches score(X, lengths)
    for the final parameters saved across random restarts.
    """
    rng = np.random.RandomState(42)
    X = rng.normal(loc=[0.0, -7.0], scale=[0.001, 0.5], size=(200, 2))
    lengths = [100, 100]

    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=15, random_state=42)
    model.fit(X, lengths=lengths, n_restarts=3)

    final_eval_score = model.score(X, lengths=lengths)
    np.testing.assert_allclose(
        model.best_log_likelihood_,
        final_eval_score,
        rtol=1e-6,
        atol=1e-5,
        err_msg="best_log_likelihood_ must match score(X, lengths) for saved parameters!",
    )


def test_expected_duration_and_occupancy_diagnostics():
    """
    Verify mathematical correctness of expected stay duration (1 / (1 - A_kk))
    and distinguish soft probability occupancy from hard argmax occupancy.
    """
    from src.models.evaluation import evaluate_model_diagnostics

    # Construct synthetic model with known transition matrix
    model = GaussianHMM(n_components=2, covariance_type="full")
    model.n_features_ = 1
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.80, 0.20], [0.10, 0.90]])
    model.means_ = np.array([[-1.0], [1.0]])
    model.covars_ = np.array([[[0.1]], [[0.1]]])
    model.converged_ = True
    model.n_iter_done_ = 5

    # Expected durations:
    # State 0: 1 / (1 - 0.8) = 5.0 bars = 1.25 hours
    # State 1: 1 / (1 - 0.9) = 10.0 bars = 2.5 hours
    diag = np.diag(model.transmat_)
    exp_bars = [1.0 / (1.0 - p) for p in diag]
    assert np.isclose(exp_bars[0], 5.0)
    assert np.isclose(exp_bars[1], 10.0)

    # Evaluate on synthetic DataFrame
    rng = np.random.RandomState(123)
    # Generate 60 bars in state 0, 40 bars in state 1
    s0 = rng.normal(-1.0, 0.1, size=(60, 1))
    s1 = rng.normal(1.0, 0.1, size=(40, 1))
    X = np.vstack([s0, s1])
    df = pd.DataFrame({"feat": X.ravel(), "timestamp": pd.date_range("2023-01-01", periods=100, freq="15min", tz="UTC")})

    diag_res = evaluate_model_diagnostics(
        model=model,
        train_df=df,
        val_df=df,
        feature_cols=["feat"],
        model_name="TestDiagModel",
    )

    # Verify expected durations match formula
    np.testing.assert_allclose(diag_res["expected_duration_bars"], [5.0, 10.0])
    np.testing.assert_allclose(diag_res["expected_duration_hours"], [1.25, 2.5])

    # Verify soft and hard occupancies both sum to 1.0
    soft_occ = np.array(diag_res["mean_occupancy_prob"])
    hard_occ = np.array(diag_res["hard_occupancy"])
    assert np.isclose(soft_occ.sum(), 1.0)
    assert np.isclose(hard_occ.sum(), 1.0)
    assert len(soft_occ) == 2
    assert len(hard_occ) == 2
