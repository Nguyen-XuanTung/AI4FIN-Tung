"""
tests.test_gmm_hmm
~~~~~~~~~~~~~~~~~~
Unit tests for Multivariate Gaussian Mixture Hidden Markov Model (GMM-HMM)
and GMM Forward Filtering Engine.

Covers all 10 required verification points:
1. Mixture weights sum to one for every state.
2. Mixture emission likelihood matches an independently calculated reference.
3. Forward probabilities sum to exactly one at every timestep.
4. Sequential and batch filtering yield identical results.
5. Future observations do not change previous filtered probabilities (strict causality).
6. Extreme observations (100-sigma outliers) do not cause underflow or NaN.
7. Covariance matrices remain strictly symmetric and positive definite.
8. Serialization/deserialization preserves exact likelihoods and filtered probabilities.
9. EM convergence and restart selection are internally consistent.
10. Data gap handling follows the documented policy (reset_prior).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from src.models.gmm_hmm import GMMHMM
from src.models.forward_filter import ForwardFilter, compute_filtered_probabilities
from src.models.evaluation import calculate_num_free_params_gmm, evaluate_model_diagnostics


@pytest.fixture
def fitted_toy_gmm_model():
    """Create a deterministic 2-state, 2-mix 2-D GMM-HMM for exact mathematical verification."""
    model = GMMHMM(n_components=2, n_mix=2, covariance_type="full", cov_reg=1e-3)
    model.n_features_ = 2
    model.startprob_ = np.array([0.65, 0.35], dtype=np.float64)
    model.transmat_ = np.array([[0.82, 0.18], [0.12, 0.88]], dtype=np.float64)

    # State 0: Low volatility mixture (two sub-modes)
    # State 1: High volatility mixture (two sub-modes)
    model.weights_ = np.array([
        [0.60, 0.40],
        [0.70, 0.30],
    ], dtype=np.float64)

    model.means_ = np.array([
        # State 0: sub-modes m=0 and m=1
        [[-0.0001, -8.1], [+0.0001, -7.9]],
        # State 1: sub-modes m=0 and m=1
        [[-0.0005, -6.5], [+0.0005, -5.9]],
    ], dtype=np.float64)

    model.covars_ = np.array([
        # State 0 covariances
        [
            [[1e-5, 0.0], [0.0, 0.04]],
            [[2e-5, 0.0], [0.0, 0.06]],
        ],
        # State 1 covariances
        [
            [[1e-4, 0.0], [0.0, 0.15]],
            [[3e-4, 0.0], [0.0, 0.35]],
        ],
    ], dtype=np.float64)

    model.converged_ = True
    model.n_iter_done_ = 12
    model.best_log_likelihood_ = -250.0
    return model


def test_1_mixture_weights_sum_to_one(fitted_toy_gmm_model):
    """Test 1: Verify mixture weights sum to 1.0 for every hidden state."""
    w = fitted_toy_gmm_model.weights_
    assert w.shape == (2, 2)
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_2_mixture_emission_likelihood_reference_match(fitted_toy_gmm_model):
    """
    Test 2: Verify mixture emission likelihood matches an independently calculated
    reference using scipy.stats.multivariate_normal.
    """
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(20, 2)) * np.array([0.001, 0.5]) + np.array([0.0, -7.0])

    # Model evaluation
    log_b = fitted_toy_gmm_model._compute_log_emissions(X)

    # Independent reference calculation
    ref_log_b = np.empty((len(X), fitted_toy_gmm_model.n_components), dtype=np.float64)
    for k in range(fitted_toy_gmm_model.n_components):
        weighted_densities = np.zeros(len(X), dtype=np.float64)
        for m in range(fitted_toy_gmm_model.n_mix):
            w_km = fitted_toy_gmm_model.weights_[k, m]
            mean_km = fitted_toy_gmm_model.means_[k, m]
            cov_km = fitted_toy_gmm_model.covars_[k, m]
            mvn = multivariate_normal(mean=mean_km, cov=cov_km)
            weighted_densities += w_km * mvn.pdf(X)
        ref_log_b[:, k] = np.log(np.maximum(weighted_densities, 1e-300))

    np.testing.assert_allclose(log_b, ref_log_b, rtol=1e-10, atol=1e-10)


def test_3_forward_probabilities_sum_to_one(fitted_toy_gmm_model):
    """Test 3: Verify filtered probabilities sum to exactly 1.0 at every timestep."""
    rng = np.random.RandomState(99)
    X = rng.normal(0, 1, size=(100, 2)) * np.array([0.001, 0.4]) + np.array([0.0, -7.0])

    engine = ForwardFilter(model=fitted_toy_gmm_model)
    res = engine.filter_batch(X)
    probs = res["filtered_probs"]

    assert probs.shape == (100, 2)
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_4_sequential_and_batch_filtering_agree(fitted_toy_gmm_model):
    """Test 4: Verify sequential bar-by-bar filtering exactly equals batch filtering."""
    rng = np.random.RandomState(123)
    X = rng.normal(0, 1, size=(60, 2)) * np.array([0.001, 0.5]) + np.array([0.0, -7.5])
    engine = ForwardFilter(model=fitted_toy_gmm_model)

    batch_res = engine.filter_batch(X)

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


def test_5_future_observations_do_not_change_past(fitted_toy_gmm_model):
    """
    Test 5: Verify strict temporal causality: altering future observations
    must NEVER alter past filtered probabilities.
    """
    rng = np.random.RandomState(777)
    X_orig = np.zeros((50, 2))
    X_orig[:, 0] = rng.normal(0.0, 0.001, size=50)
    X_orig[:, 1] = rng.normal(-8.0, 0.2, size=50)

    engine = ForwardFilter(model=fitted_toy_gmm_model)
    res_orig = engine.filter_batch(X_orig)

    cutoff = 25
    X_mod = X_orig.copy()
    # Shock future observations after cutoff to state 1 regime
    X_mod[cutoff:, 1] = rng.normal(-6.0, 0.3, size=25)

    res_mod = engine.filter_batch(X_mod)

    # Past up to cutoff must be identical to machine precision
    np.testing.assert_allclose(
        res_orig["filtered_probs"][:cutoff],
        res_mod["filtered_probs"][:cutoff],
        atol=1e-12,
        err_msg="Temporal causality violation: future shock altered past filtered probabilities!",
    )
    # Future must reflect the shock
    diff_future = np.abs(res_orig["filtered_probs"][cutoff:] - res_mod["filtered_probs"][cutoff:])
    assert np.max(diff_future) > 0.05


def test_6_extreme_observations_no_numerical_failure(fitted_toy_gmm_model):
    """Test 6: Verify extreme outliers (100-sigma) do not cause NaN or Inf."""
    X_extreme = np.array([
        [0.0, -7.5],
        [1e6, 1e6],
        [-1e6, -1e6],
        [0.0, -7.5],
    ])
    engine = ForwardFilter(model=fitted_toy_gmm_model)
    res = engine.filter_batch(X_extreme)
    probs = res["filtered_probs"]

    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-12)


def test_7_covariance_matrices_remain_positive_definite():
    """Test 7: Verify all fitted GMM covariance matrices are strictly positive definite."""
    rng = np.random.RandomState(42)
    X = rng.normal(loc=[0.0, -7.0], scale=[0.001, 0.5], size=(200, 2))

    model = GMMHMM(n_components=2, n_mix=2, covariance_type="full", n_iter=10, random_state=42)
    model.fit(X, n_restarts=2)

    for k in range(model.n_components):
        for m in range(model.n_mix):
            cov = model.covars_[k, m]
            # Must be symmetric
            np.testing.assert_allclose(cov, cov.T, atol=1e-10)
            # Eigenvalues must be strictly positive
            eigenvals = np.linalg.eigvalsh(cov)
            assert np.all(eigenvals > 0.0), f"Non-positive eigenvalue in state {k} mixture {m}: {eigenvals}"


def test_8_serialization_preserves_likelihoods_and_probabilities(fitted_toy_gmm_model):
    """Test 8: Verify model export to dict/JSON and reload preserves exact outputs."""
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(40, 2)) * np.array([0.001, 0.4]) + np.array([0.0, -7.0])

    orig_score = fitted_toy_gmm_model.score(X)
    d = fitted_toy_gmm_model.to_dict()
    loaded_model = GMMHMM.from_dict(d)
    loaded_score = loaded_model.score(X)

    np.testing.assert_allclose(orig_score, loaded_score, atol=1e-10)

    # Filtered probabilities check
    eng_orig = ForwardFilter(model=fitted_toy_gmm_model)
    eng_loaded = ForwardFilter(model=loaded_model)

    res_orig = eng_orig.filter_batch(X)
    res_loaded = eng_loaded.filter_batch(X)
    np.testing.assert_allclose(res_orig["filtered_probs"], res_loaded["filtered_probs"], atol=1e-12)


def test_9_em_convergence_and_restart_selection():
    """Test 9: Verify best_log_likelihood_ corresponds to score(X, lengths) on saved parameters."""
    rng = np.random.RandomState(42)
    X = rng.normal(loc=[0.0, -7.0], scale=[0.001, 0.4], size=(250, 2))
    lengths = [125, 125]

    model = GMMHMM(n_components=2, n_mix=2, covariance_type="full", n_iter=15, random_state=42)
    model.fit(X, lengths=lengths, n_restarts=3)

    final_score = model.score(X, lengths=lengths)
    np.testing.assert_allclose(
        model.best_log_likelihood_,
        final_score,
        rtol=1e-6,
        atol=1e-5,
        err_msg="best_log_likelihood_ must match score(X, lengths) for saved parameters!",
    )


def test_10_gap_handling_follows_policy(fitted_toy_gmm_model):
    """Test 10: Verify gap handling triggers reset_prior across time breaks."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00+00:00"),
        pd.Timestamp("2023-01-02 10:15:00+00:00"),
        pd.Timestamp("2023-01-02 10:30:00+00:00"),
        pd.Timestamp("2023-01-05 10:00:00+00:00"),  # Weekend/closure gap
        pd.Timestamp("2023-01-05 10:15:00+00:00"),
    ]
    X = np.zeros((5, 2))
    engine = ForwardFilter(model=fitted_toy_gmm_model, gap_policy="reset_prior", max_gap_minutes=15)
    res = engine.filter_batch(X, timestamps=timestamps)

    assert res["is_gap"][3] == True
    assert res["is_gap"][0] == False
    assert res["is_gap"][1] == False
    # Under reset_prior, prior at gap index 3 resets to startprob_
    np.testing.assert_allclose(res["prior_probs"][3], fitted_toy_gmm_model.startprob_, atol=1e-12)
