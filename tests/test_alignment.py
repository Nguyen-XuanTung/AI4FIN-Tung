"""
tests.test_alignment
~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for Phase 6 label alignment module (src.interpretation.alignment):
- Permutation recovery under synthetic state label switching
- Reference profile standardization and scale floors
- Hungarian assignment with distance rejection threshold
- Ambiguity detection when candidate is near-equidistant to multiple reference states
- Unequal state counts (excess candidate states and partial coverage)
- Financial dimension classification (volatility and drift separation)
"""

import numpy as np
import pytest

from src.interpretation.characteristics import RegimeCharacteristics, StateCharacteristics
from src.interpretation.alignment import (
    ReferenceProfileScaler,
    extract_profile_vector,
    classify_regime_dimensions,
    align_model_states,
)


def _make_dummy_state(
    state_id: int,
    r_mean: float,
    r_std: float,
    vol_mean: float,
    vol_std: float,
    p_pos: float,
) -> StateCharacteristics:
    """Helper to construct a minimal StateCharacteristics for testing."""
    return StateCharacteristics(
        state_id=state_id,
        soft_occupancy=0.33,
        effective_sample_size=1000.0,
        r_oc_mean=r_mean,
        r_oc_std=r_std,
        r_oc_median=r_mean,
        r_oc_quantiles={"q05": r_mean - 2 * r_std, "q50": r_mean, "q95": r_mean + 2 * r_std},
        r_oc_skewness=0.0,
        r_oc_kurtosis=0.0,
        prob_positive_return=p_pos,
        prob_negative_return=1.0 - p_pos,
        prob_zero_return=0.0,
        log_vol_mean=vol_mean,
        log_vol_std=vol_std,
        log_vol_median=vol_mean,
        log_vol_quantiles={"q50": vol_mean},
        range_bps_mean=5.0,
        range_bps_std=2.0,
        range_bps_median=5.0,
        range_bps_quantiles={"q50": 5.0},
        range_pips_mean=5.0,
        range_pips_median=5.0,
        geometric_range_scale=float(np.exp(vol_mean)),
        analytical_mean=[r_mean, vol_mean],
        analytical_covar=[[r_std ** 2, 0.0], [0.0, vol_std ** 2]],
        empirical_mean=[r_mean, vol_mean],
        empirical_covar=[[r_std ** 2, 0.0], [0.0, vol_std ** 2]],
        moment_discrepancy_mean_l2=0.0,
        moment_discrepancy_covar_frobenius=0.0,
    )


@pytest.fixture
def synthetic_reference():
    """Canonical 3-state reference model characteristics."""
    # State 0: Low Vol, Neutral
    s0 = _make_dummy_state(0, r_mean=0.0, r_std=0.0002, vol_mean=-8.2, vol_std=0.35, p_pos=0.50)
    # State 1: High Vol, Neutral
    s1 = _make_dummy_state(1, r_mean=0.0, r_std=0.0009, vol_mean=-6.8, vol_std=0.40, p_pos=0.50)
    # State 2: Med Vol, Neutral
    s2 = _make_dummy_state(2, r_mean=0.0, r_std=0.0004, vol_mean=-7.5, vol_std=0.35, p_pos=0.50)

    return RegimeCharacteristics(
        model_version="Ref_K3_v1",
        model_type="GMMHMM",
        n_components=3,
        n_mix=3,
        training_samples=12501,
        feature_names=["r_oc", "log_vol"],
        states={0: s0, 1: s1, 2: s2},
    )


class TestAlignment:
    """Tests for Hungarian label alignment algorithm."""

    def test_permutation_recovery(self, synthetic_reference):
        """Verify that a pure permutation of reference states is recovered with 0 distance."""
        # Permute reference states: candidate state 0 is ref 2, cand 1 is ref 0, cand 2 is ref 1
        perm = [2, 0, 1]
        cand_states = {}
        for cand_k, ref_k in enumerate(perm):
            s_ref = synthetic_reference.states[ref_k]
            cand_states[cand_k] = _make_dummy_state(
                state_id=cand_k,
                r_mean=s_ref.r_oc_mean,
                r_std=s_ref.r_oc_std,
                vol_mean=s_ref.log_vol_mean,
                vol_std=s_ref.log_vol_std,
                p_pos=s_ref.prob_positive_return,
            )

        candidate = RegimeCharacteristics(
            model_version="Cand_Permuted",
            model_type="GaussianHMM",
            n_components=3,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states=cand_states,
        )

        report = align_model_states(candidate, synthetic_reference)

        assert report.candidate_match_rate == 1.0
        assert report.reference_coverage == 1.0
        assert np.isclose(report.average_matched_cost, 0.0, atol=1e-10)

        # Check matched IDs
        assert report.matches[0].matched_ref_state_id == 2
        assert report.matches[1].matched_ref_state_id == 0
        assert report.matches[2].matched_ref_state_id == 1

        for m in report.matches.values():
            assert m.status == "MATCHED"
            assert not m.is_ambiguous

    def test_reference_profile_scaler_serialization(self, synthetic_reference):
        """Verify ReferenceProfileScaler fit, transform, to_dict, and from_dict."""
        scaler = ReferenceProfileScaler.fit(synthetic_reference)
        d = scaler.to_dict()
        scaler2 = ReferenceProfileScaler.from_dict(d)

        np.testing.assert_allclose(scaler.means, scaler2.means)
        np.testing.assert_allclose(scaler.stds, scaler2.stds)
        assert scaler.feature_names == scaler2.feature_names

        # Verify scale floors prevent zero std
        assert (scaler.stds > 0).all()

    def test_distance_threshold_rejection(self, synthetic_reference):
        """Verify candidate state far from all reference states is rejected."""
        # Candidate with anomalous extreme parameters
        s_wild = _make_dummy_state(0, r_mean=0.05, r_std=0.05, vol_mean=-2.0, vol_std=2.0, p_pos=0.90)
        s_norm0 = synthetic_reference.states[0]
        s_norm1 = synthetic_reference.states[1]

        cand_states = {
            0: s_wild,
            1: _make_dummy_state(1, s_norm0.r_oc_mean, s_norm0.r_oc_std, s_norm0.log_vol_mean, s_norm0.log_vol_std, s_norm0.prob_positive_return),
            2: _make_dummy_state(2, s_norm1.r_oc_mean, s_norm1.r_oc_std, s_norm1.log_vol_mean, s_norm1.log_vol_std, s_norm1.prob_positive_return),
        }
        candidate = RegimeCharacteristics(
            model_version="Cand_Wild",
            model_type="GaussianHMM",
            n_components=3,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states=cand_states,
        )

        report = align_model_states(candidate, synthetic_reference, distance_threshold=2.5)

        # State 0 should be rejected
        assert report.matches[0].status == "REJECTED_DISTANCE"
        assert report.matches[0].matched_ref_state_id is None
        assert report.matches[0].canonical_semantic_label.startswith("Novel_")

        # States 1 and 2 should be matched
        assert report.matches[1].status == "MATCHED"
        assert report.matches[2].status == "MATCHED"
        assert report.unmatched_state_count == 1

    def test_ambiguity_detection(self, synthetic_reference):
        """Verify candidate state near-equidistant between two reference states is flagged."""
        # Reference state 0 is vol=-8.2, state 2 is vol=-7.5.
        # Place candidate right in between at -7.85 with identical std
        s_mid = _make_dummy_state(0, r_mean=0.0, r_std=0.0003, vol_mean=-7.85, vol_std=0.35, p_pos=0.50)
        s1 = synthetic_reference.states[1]
        s2 = synthetic_reference.states[2]

        cand_states = {
            0: s_mid,
            1: _make_dummy_state(1, s1.r_oc_mean, s1.r_oc_std, s1.log_vol_mean, s1.log_vol_std, s1.prob_positive_return),
            2: _make_dummy_state(2, s2.r_oc_mean, s2.r_oc_std, s2.log_vol_mean, s2.log_vol_std, s2.prob_positive_return),
        }
        candidate = RegimeCharacteristics(
            model_version="Cand_Ambiguous",
            model_type="GaussianHMM",
            n_components=3,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states=cand_states,
        )

        # High distance threshold so it's not rejected, margin = 0.50
        report = align_model_states(candidate, synthetic_reference, distance_threshold=5.0, ambiguity_margin=0.50)
        assert report.matches[0].is_ambiguous is True
        assert report.ambiguous_state_count >= 1

    def test_unequal_state_counts(self, synthetic_reference):
        """Verify handling of candidate models with K != K_ref."""
        # 1. K_cand = 4 against K_ref = 3
        cand4_states = {
            k: synthetic_reference.states[min(k, 2)] for k in range(4)
        }
        cand4 = RegimeCharacteristics(
            model_version="Cand_K4",
            model_type="GaussianHMM",
            n_components=4,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states=cand4_states,
        )
        report4 = align_model_states(cand4, synthetic_reference)
        assert report4.k_candidate == 4
        assert report4.k_reference == 3
        # 4th state must be UNMATCHED_EXCESS
        assert report4.matches[3].status == "UNMATCHED_EXCESS"

        # 2. K_cand = 2 against K_ref = 3
        cand2_states = {
            0: synthetic_reference.states[0],
            1: synthetic_reference.states[1],
        }
        cand2 = RegimeCharacteristics(
            model_version="Cand_K2",
            model_type="GaussianHMM",
            n_components=2,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states=cand2_states,
        )
        report2 = align_model_states(cand2, synthetic_reference)
        assert report2.k_candidate == 2
        assert report2.k_reference == 3
        assert report2.candidate_match_rate == 1.0
        assert np.isclose(report2.reference_coverage, 2.0 / 3.0)

    def test_financial_dimension_classification(self):
        """Verify financial classification separates volatility from direction."""
        # High Volatility with Neutral drift
        s_high_vol_neutral = _make_dummy_state(0, r_mean=0.000005, r_std=0.0008, vol_mean=-6.7, vol_std=0.4, p_pos=0.505)
        vol_l, dir_l, comp_l = classify_regime_dimensions(s_high_vol_neutral)
        assert vol_l == "High_Volatility"
        assert dir_l == "Neutral"
        assert "Bear" not in comp_l  # Must never assume High Vol == Bearish

        # High Volatility with genuine negative drift
        s_high_vol_bear = _make_dummy_state(1, r_mean=-0.0001, r_std=0.0008, vol_mean=-6.7, vol_std=0.4, p_pos=0.45)
        vol_l, dir_l, comp_l = classify_regime_dimensions(s_high_vol_bear)
        assert vol_l == "High_Volatility"
        assert dir_l == "Negative_Drift"

        # Low Volatility with Positive drift
        s_low_vol_bull = _make_dummy_state(2, r_mean=0.00005, r_std=0.0002, vol_mean=-8.3, vol_std=0.3, p_pos=0.55)
        vol_l, dir_l, comp_l = classify_regime_dimensions(s_low_vol_bull)
        assert vol_l == "Low_Volatility"
        assert dir_l == "Positive_Drift"

    def test_competing_candidates_ambiguity(self, synthetic_reference):
        """Verify two candidate states competing for the same reference state are flagged as ambiguous."""
        # Both Cand 0 and Cand 1 have profiles nearly identical to Reference State 0
        s_ref0 = synthetic_reference.states[0]
        s_ref2 = synthetic_reference.states[2]

        cand0 = _make_dummy_state(0, s_ref0.r_oc_mean, s_ref0.r_oc_std, s_ref0.log_vol_mean, s_ref0.log_vol_std, s_ref0.prob_positive_return)
        # Cand 1 is only 0.05 pips / slightly shifted from Ref 0
        cand1 = _make_dummy_state(1, s_ref0.r_oc_mean + 1e-6, s_ref0.r_oc_std, s_ref0.log_vol_mean + 0.01, s_ref0.log_vol_std, s_ref0.prob_positive_return)
        # Cand 2 is Ref 2
        cand2 = _make_dummy_state(2, s_ref2.r_oc_mean, s_ref2.r_oc_std, s_ref2.log_vol_mean, s_ref2.log_vol_std, s_ref2.prob_positive_return)

        candidate = RegimeCharacteristics(
            model_version="Cand_Competing",
            model_type="GaussianHMM",
            n_components=3,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states={0: cand0, 1: cand1, 2: cand2},
        )

        report = align_model_states(candidate, synthetic_reference, distance_threshold=5.0, ambiguity_margin=0.20)
        # Cand 0 and Cand 1 are competing for Ref 0 with tiny cost difference
        assert report.matches[0].is_ambiguous or report.matches[1].is_ambiguous
        assert report.ambiguous_state_count >= 1

    def test_global_assignment_gap(self, synthetic_reference):
        """Verify global assignment gap metric is computed and matches expectation."""
        # Permutation matching should have positive global gap
        perm = [2, 0, 1]
        cand_states = {}
        for cand_k, ref_k in enumerate(perm):
            s_ref = synthetic_reference.states[ref_k]
            cand_states[cand_k] = _make_dummy_state(
                state_id=cand_k,
                r_mean=s_ref.r_oc_mean,
                r_std=s_ref.r_oc_std,
                vol_mean=s_ref.log_vol_mean,
                vol_std=s_ref.log_vol_std,
                p_pos=s_ref.prob_positive_return,
            )

        candidate = RegimeCharacteristics(
            model_version="Cand_Permuted",
            model_type="GaussianHMM",
            n_components=3,
            n_mix=1,
            training_samples=12501,
            feature_names=["r_oc", "log_vol"],
            states=cand_states,
        )

        report = align_model_states(candidate, synthetic_reference)
        # For permuted distinct reference states, the second best permutation must have significantly higher cost
        assert report.global_assignment_gap > 0.50
        assert not report.is_globally_ambiguous
        assert report.threshold_type == "operational_heuristic"

