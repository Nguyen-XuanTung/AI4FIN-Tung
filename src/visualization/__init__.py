"""
src.visualization
~~~~~~~~~~~~~~~~~
Phase 8: Historical Market Regime Visualization Module for EURUSD M15.

Provides robust, gap-aware, and publication-grade visualization tools for
causal filtered regime probabilities, price-regime overlays, posterior
entropy, transition dynamics, and statistical characteristic cross-checks.
"""

from src.visualization.data_loader import (
    STATE_LABELS,
    STATE_COLORS,
    STATE_DESCRIPTIONS,
    load_canonical_model,
    load_as_of_replay_data,
    load_full_year_data,
    compute_entropy_series,
    extract_continuous_segments,
    compute_transition_and_run_lengths,
)

from src.visualization.regime_plots import (
    plot_figure_1_price_and_regime_timeline,
    plot_figure_2_filtered_probabilities,
    plot_figure_3_confidence_and_entropy,
    plot_figure_4_transition_and_duration_analysis,
    plot_figure_5_regime_characteristics_crosscheck,
    plot_composite_regime_dashboard,
)

__all__ = [
    "STATE_LABELS",
    "STATE_COLORS",
    "STATE_DESCRIPTIONS",
    "load_canonical_model",
    "load_as_of_replay_data",
    "load_full_year_data",
    "compute_entropy_series",
    "extract_continuous_segments",
    "compute_transition_and_run_lengths",
    "plot_figure_1_price_and_regime_timeline",
    "plot_figure_2_filtered_probabilities",
    "plot_figure_3_confidence_and_entropy",
    "plot_figure_4_transition_and_duration_analysis",
    "plot_figure_5_regime_characteristics_crosscheck",
    "plot_composite_regime_dashboard",
]
