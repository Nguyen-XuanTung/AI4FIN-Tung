"""
src.models
~~~~~~~~~~
Hidden Markov Model modeling, forward filtering, split management, and diagnostics.
"""

from src.models.gaussian_hmm import GaussianHMM
from src.models.forward_filter import (
    ForwardFilter,
    compute_filtered_probabilities,
    compute_stationary_distribution,
)
from src.models.splits import (
    identify_continuous_sequences,
    split_chronological,
)
from src.models.evaluation import (
    calculate_num_free_params,
    compute_information_criteria,
    evaluate_model_diagnostics,
)

__all__ = [
    "GaussianHMM",
    "ForwardFilter",
    "compute_filtered_probabilities",
    "compute_stationary_distribution",
    "identify_continuous_sequences",
    "split_chronological",
    "calculate_num_free_params",
    "compute_information_criteria",
    "evaluate_model_diagnostics",
]

