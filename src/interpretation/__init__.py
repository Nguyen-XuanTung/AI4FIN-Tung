"""
src.interpretation
~~~~~~~~~~~~~~~~~~
Phase 6: Regime Interpretation, State Characterization, Duration Analysis,
and Canonical Label Alignment across HMM architectures.
"""

from src.interpretation.smoother import (
    compute_smoothed_posteriors,
    ForwardBackwardResult,
)
from src.interpretation.characteristics import (
    weighted_mean_and_std,
    weighted_quantiles,
    weighted_higher_moments,
    aggregate_gmm_state_moments,
    StateCharacteristics,
    RegimeCharacteristics,
    calculate_state_characteristics,
)
from src.interpretation.duration import (
    compute_model_implied_duration,
    viterbi_decode_multi_sequence,
    extract_run_lengths,
    summarize_run_length_distribution,
    compute_switching_frequency,
    RegimeDurationAnalysis,
    analyze_regime_durations,
)
from src.interpretation.alignment import (
    PROFILE_FEATURE_NAMES,
    extract_profile_vector,
    ReferenceProfileScaler,
    classify_regime_dimensions,
    compute_profile_cost_matrix,
    StateAlignmentMatch,
    ModelAlignmentReport,
    align_model_states,
)

__all__ = [
    # Smoother
    "compute_smoothed_posteriors",
    "ForwardBackwardResult",
    # Characteristics
    "weighted_mean_and_std",
    "weighted_quantiles",
    "weighted_higher_moments",
    "aggregate_gmm_state_moments",
    "StateCharacteristics",
    "RegimeCharacteristics",
    "calculate_state_characteristics",
    # Duration & Persistence
    "compute_model_implied_duration",
    "viterbi_decode_multi_sequence",
    "extract_run_lengths",
    "summarize_run_length_distribution",
    "compute_switching_frequency",
    "RegimeDurationAnalysis",
    "analyze_regime_durations",
    # Alignment
    "PROFILE_FEATURE_NAMES",
    "extract_profile_vector",
    "ReferenceProfileScaler",
    "classify_regime_dimensions",
    "compute_profile_cost_matrix",
    "StateAlignmentMatch",
    "ModelAlignmentReport",
    "align_model_states",
]
