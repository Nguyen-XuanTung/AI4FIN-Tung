"""
tests.test_interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for Phase 6 interpretation modules:
- Multi-sequence log-space forward-backward smoother (src.interpretation.smoother)
- Posterior-weighted state characteristics & GMM mixture aggregation (src.interpretation.characteristics)
- Persistence, Viterbi multi-sequence decoding, and duration analysis (src.interpretation.duration)
"""

import numpy as np
import pytest
import pandas as pd

from src.models.gaussian_hmm import GaussianHMM
from src.models.gmm_hmm import GMMHMM
from src.models.forward_filter import ForwardFilter
from src.interpretation.smoother import compute_smoothed_posteriors, ForwardBackwardResult
from src.interpretation.characteristics import (
    weighted_mean_and_std,
    weighted_quantiles,
    weighted_higher_moments,
    aggregate_gmm_state_moments,
    calculate_state_characteristics,
)
from src.interpretation.duration import (
    compute_model_implied_duration,
    viterbi_decode_multi_sequence,
    extract_run_lengths,
    summarize_run_length_distribution,
    compute_switching_frequency,
    analyze_regime_durations,
)


class TestSmoother:
    """Tests for multi-sequence log-space forward-backward smoother."""

    @pytest.fixture
    def toy_gaussian_hmm(self):
        """2-state 1D Gaussian HMM with known parameters."""
        model = GaussianHMM(n_components=2, n_iter=1, random_state=42)
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
        model.means_ = np.array([[0.0], [5.0]])
        model.covars_ = np.array([[[1.0]], [[1.0]]])
        model.covariance_type = "full"
        return model

    def test_toy_smoother_probabilities_sum_to_one(self, toy_gaussian_hmm):
        """Verify that smoothed posteriors sum to 1.0 at every timestep."""
        np.random.seed(42)
        X = np.array([[0.1], [0.3], [4.8], [5.2], [0.2]])
        lengths = [len(X)]

        res = compute_smoothed_posteriors(toy_gaussian_hmm, X, lengths)
        assert isinstance(res, ForwardBackwardResult)
        assert res.gamma.shape == (5, 2)

        # Row sums must equal 1.0 within floating point precision
        row_sums = np.sum(res.gamma, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-12)

        # State 0 at t=0,1 (around 0.2) should have high probability
        assert res.gamma[0, 0] > 0.80
        assert res.gamma[1, 0] > 0.70
        # State 1 at t=2,3 (around 5.0) should have high probability
        assert res.gamma[2, 1] > 0.80
        assert res.gamma[3, 1] > 0.80

    def test_smoother_hand_calculated_step(self, toy_gaussian_hmm):
        """Verify log-space smoother matches manual mathematical derivation for T=2."""
        X = np.array([[0.0], [5.0]])
        lengths = [2]

        res = compute_smoothed_posteriors(toy_gaussian_hmm, X, lengths)
        gamma = res.gamma

        # Manual calculation:
        # B_0(0) = N(0|0,1) = 1/sqrt(2pi) = 0.398942
        # B_0(1) = N(0|5,1) = 1/sqrt(2pi) * exp(-12.5) = 1.4867e-6
        # B_1(0) = N(5|0,1) = 1.4867e-6
        # B_1(1) = N(5|5,1) = 0.398942
        # alpha_0 = [0.6 * 0.398942, 0.4 * 1.4867e-6] = [0.239365, 5.9468e-7]
        # alpha_1(0) = (0.239365 * 0.8 + 5.9468e-7 * 0.3) * 1.4867e-6 = 0.191492 * 1.4867e-6 = 2.8469e-7
        # alpha_1(1) = (0.239365 * 0.2 + 5.9468e-7 * 0.7) * 0.398942 = 0.047873 * 0.398942 = 0.019098
        # Since alpha_1(1) >> alpha_1(0), gamma_1(1) must be ~ 0.999985
        assert gamma[1, 1] > 0.9999
        assert gamma[0, 0] > 0.90  # Forward-backward smooths t=0 knowing t=1 is state 1

    def test_multi_sequence_boundary_isolation(self, toy_gaussian_hmm):
        """Verify that sequence boundaries do not leak forward or backward messages."""
        np.random.seed(123)
        X1 = np.random.normal(0.0, 1.0, (20, 1))
        X2 = np.random.normal(5.0, 1.0, (30, 1))
        X_combined = np.vstack([X1, X2])

        # Run combined with multi-sequence lengths
        res_combined = compute_smoothed_posteriors(toy_gaussian_hmm, X_combined, [20, 30])

        # Run each sequence independently
        res_seq1 = compute_smoothed_posteriors(toy_gaussian_hmm, X1, [20])
        res_seq2 = compute_smoothed_posteriors(toy_gaussian_hmm, X2, [30])

        # Exact numerical equality: boundary must guarantee zero leakage
        np.testing.assert_allclose(res_combined.gamma[:20], res_seq1.gamma, atol=1e-12)
        np.testing.assert_allclose(res_combined.gamma[20:], res_seq2.gamma, atol=1e-12)

    def test_extreme_value_numerical_stability(self, toy_gaussian_hmm):
        """Verify smoother does not produce NaNs or Infs under extreme outliers."""
        # Extreme values (-500.0, +1000.0)
        X = np.array([[-500.0], [1000.0], [0.0], [5.0]])
        res = compute_smoothed_posteriors(toy_gaussian_hmm, X, [4])
        assert not np.isnan(res.gamma).any()
        assert not np.isinf(res.gamma).any()
        row_sums = np.sum(res.gamma, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-12)

    def test_gmm_hmm_smoother_compatibility(self):
        """Verify smoother works seamlessly on GMM-HMM."""
        model = GMMHMM(n_components=2, n_mix=2, n_iter=1, random_state=42)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
        model.weights_ = np.array([[0.5, 0.5], [0.7, 0.3]])
        model.means_ = np.array([
            [[-1.0, 0.0], [1.0, 0.0]],
            [[5.0, 5.0], [10.0, 10.0]],
        ])
        model.covars_ = np.array([
            [np.eye(2), np.eye(2)],
            [np.eye(2), np.eye(2)],
        ])
        model.covariance_type = "full"

        X = np.array([
            [0.0, 0.0],
            [0.5, 0.1],
            [5.0, 5.0],
            [10.0, 9.8],
        ])
        res = compute_smoothed_posteriors(model, X, [4])
        assert res.gamma.shape == (4, 2)
        np.testing.assert_allclose(np.sum(res.gamma, axis=1), np.ones(4), atol=1e-12)
        # First two points closer to state 0, last two closer to state 1
        assert res.gamma[0, 0] > res.gamma[0, 1]
        assert res.gamma[3, 1] > res.gamma[3, 0]

    def test_future_observation_perturbation_smoother_vs_causal_filter(self):
        """
        Verify that future observations at t=2 perturb smoothed responsibilities at t=1,
        while causal filtered probabilities at t=1 remain strictly unchanged.
        """
        model = GaussianHMM(n_components=2, n_iter=1, random_state=42)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
        model.means_ = np.array([[0.0], [5.0]])
        model.covars_ = np.array([[[4.0]], [[4.0]]])
        model.covariance_type = "full"

        # Observation sequences identical at t=0, t=1, but differing at future t=2
        X_orig = np.array([[2.5], [2.5], [0.0]])
        X_pert = np.array([[2.5], [2.5], [5.0]])

        # 1. Causal ForwardFilter (P(S_t | O_{1:t}))
        ff1 = ForwardFilter(model)
        filt1 = ff1.filter_batch(X_orig)["filtered_probs"]

        ff2 = ForwardFilter(model)
        filt2 = ff2.filter_batch(X_pert)["filtered_probs"]

        # Filtered probabilities at t=1 must be strictly identical
        np.testing.assert_allclose(filt1[1], filt2[1], atol=1e-12)
        assert np.allclose(filt1[1], [0.5, 0.5])

        # 2. Retrospective Smoother (P(S_t | O_{1:T}))
        res1 = compute_smoothed_posteriors(model, X_orig, [3])
        res2 = compute_smoothed_posteriors(model, X_pert, [3])

        # Smoothed probabilities at t=1 must shift significantly due to backward message
        shift = abs(res1.gamma[1, 0] - res2.gamma[1, 0])
        assert shift > 0.20
        # When future is 0.0, state 0 probability increases; when future is 5.0, state 1 increases
        assert res1.gamma[1, 0] > res2.gamma[1, 0]


class TestCharacteristics:
    """Tests for posterior-weighted state characteristics and mixture moment aggregation."""

    def test_weighted_mean_and_std_known_values(self):
        """Verify weighted mean and std against known analytical formula."""
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 2.0, 1.0])  # sum = 4

        # Mean = (10*1 + 20*2 + 30*1)/4 = 80/4 = 20.0
        # Var = (1*(10-20)^2 + 2*(20-20)^2 + 1*(30-20)^2)/4 = (100 + 0 + 100)/4 = 50.0
        # Std = sqrt(50) = 7.0710678
        w_mean, w_std = weighted_mean_and_std(values, weights)
        assert np.isclose(w_mean, 20.0)
        assert np.isclose(w_std, np.sqrt(50.0))

    def test_weighted_quantiles(self):
        """Verify weighted quantiles compute correct median and percentiles."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        q_dict = weighted_quantiles(values, weights, [0.50])
        assert np.isclose(q_dict["q50"], 3.0, atol=0.1)

    def test_aggregate_gmm_state_moments_total_variance(self):
        """
        Verify GMM mixture moment aggregation against the Law of Total Variance:
        Var(X) = E[Var(X|M)] + Var(E[X|M])
        """
        # 1 state, 2 mixture components, 1D
        weights = np.array([[0.4, 0.6]])  # (1, 2)
        means = np.array([[[2.0], [8.0]]])  # (1, 2, 1)
        covars = np.array([[[[1.0]], [[4.0]]]])  # (1, 2, 1, 1)

        state_means, state_covars = aggregate_gmm_state_moments(
            weights=weights,
            means=means,
            covars=covars,
            covariance_type="full",
        )

        # Expected Mean: 0.4*2.0 + 0.6*8.0 = 0.8 + 4.8 = 5.6
        expected_mu = 5.6
        assert np.isclose(state_means[0, 0], expected_mu)

        # Expected Covariance:
        # E[Var(X|M)] = 0.4*1.0 + 0.6*4.0 = 0.4 + 2.4 = 2.8
        # Var(E[X|M]) = 0.4*(2.0 - 5.6)^2 + 0.6*(8.0 - 5.6)^2
        #             = 0.4*(-3.6)^2 + 0.6*(2.4)^2 = 0.4*12.96 + 0.6*5.76 = 5.184 + 3.456 = 8.64
        # Total Var = 2.8 + 8.64 = 11.44
        expected_var = 11.44
        assert np.isclose(state_covars[0, 0, 0], expected_var)

    def test_scale_separation_in_characteristics(self):
        """Verify that geometric_range_scale strictly differs from arithmetic mean range."""
        # Create synthetic training DataFrame
        N = 100
        np.random.seed(42)
        open_prices = np.full(N, 1.1000)
        # Volatile candle ranges
        ranges = np.random.uniform(0.0005, 0.0030, N)
        high_prices = open_prices + ranges * 0.6
        low_prices = open_prices - ranges * 0.4
        close_prices = open_prices + np.random.uniform(-0.0002, 0.0002, N)

        r_oc = (close_prices - open_prices) / open_prices
        log_vol = np.log(np.maximum(np.log(high_prices / low_prices), 1e-6))

        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "r_oc": r_oc,
            "log_vol": log_vol,
        })

        model = GaussianHMM(n_components=1, n_iter=1, random_state=42)
        model.startprob_ = np.array([1.0])
        model.transmat_ = np.array([[1.0]])
        model.means_ = np.array([[0.0, float(np.mean(log_vol))]])
        model.covars_ = np.array([np.eye(2)])
        model.covariance_type = "full"

        gamma = np.ones((N, 1))

        chars = calculate_state_characteristics(model, df, gamma, "Test_Model")
        state_0 = chars.states[0]

        # Arithmetic range bps: mean of ((H - L) / O) * 10,000 ~ 10 - 25 bps
        # Geometric scale: exp(mean(log_vol)) ~ ln(H/L) ~ 0.001 - 0.002
        assert state_0.range_bps_mean > 5.0
        assert state_0.geometric_range_scale < 0.01
        # They must be on totally different physical scales
        assert abs(state_0.range_bps_mean - state_0.geometric_range_scale) > 1.0


class TestDurationAndPersistence:
    """Tests for duration metrics, multi-sequence Viterbi, and persistence analysis."""

    def test_compute_model_implied_duration(self):
        """Verify theoretical Markov duration formula E[T_k] = 1 / (1 - A_{kk})."""
        # A_00 = 0.90 -> E[T] = 10 bars = 2.5 hours
        # A_11 = 0.75 -> E[T] = 4 bars = 1.0 hour
        # A_22 = 1.00 -> Absorbing state -> inf
        transmat = np.array([
            [0.90, 0.10, 0.00],
            [0.25, 0.75, 0.00],
            [0.00, 0.00, 1.00],
        ])

        res = compute_model_implied_duration(transmat, bar_duration_hours=0.25)
        assert np.isclose(res[0]["expected_bars"], 10.0)
        assert np.isclose(res[0]["expected_hours"], 2.5)
        assert np.isclose(res[1]["expected_bars"], 4.0)
        assert np.isclose(res[1]["expected_hours"], 1.0)
        assert np.isinf(res[2]["expected_bars"])
        assert np.isinf(res[2]["expected_hours"])

    def test_viterbi_multi_sequence_boundary_isolation(self):
        """Verify Viterbi decoding does not propagate across sequence boundaries."""
        model = GaussianHMM(n_components=2, n_iter=1, random_state=42)
        model.startprob_ = np.array([0.9, 0.1])
        model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        model.means_ = np.array([[0.0], [10.0]])
        model.covars_ = np.array([[[1.0]], [[1.0]]])
        model.covariance_type = "full"

        # Seq 1 is clearly state 0, Seq 2 is clearly state 1
        X1 = np.zeros((15, 1))
        X2 = np.full((20, 1), 10.0)
        X_all = np.vstack([X1, X2])

        path = viterbi_decode_multi_sequence(model, X_all, [15, 20])
        assert path.shape == (35,)
        np.testing.assert_array_equal(path[:15], np.zeros(15))
        np.testing.assert_array_equal(path[15:], np.ones(20))

    def test_extract_run_lengths_respects_breaks(self):
        """Verify that identical states across a sequence boundary are NOT merged."""
        # Two sequences of 5 bars each, all in state 0
        states = np.zeros(10, dtype=int)
        lengths = [5, 5]

        runs = extract_run_lengths(states, lengths, n_states=2)
        # Must be two distinct runs of length 5, NOT one run of length 10!
        assert runs[0] == [5, 5]
        assert runs[1] == []

    def test_switching_frequency_excludes_boundary_jumps(self):
        """Verify switches across sequence breaks are excluded from switching frequency."""
        # Seq 1: [0, 0, 0] (0 switches / 2 transitions)
        # Break between Seq 1 and Seq 2
        # Seq 2: [1, 1, 1] (0 switches / 2 transitions)
        states = np.array([0, 0, 0, 1, 1, 1])
        lengths = [3, 3]

        switch_rate, n_switches, n_transitions = compute_switching_frequency(states, lengths)
        assert n_switches == 0
        assert n_transitions == 4  # (3 - 1) + (3 - 1) = 4
        assert switch_rate == 0.0
