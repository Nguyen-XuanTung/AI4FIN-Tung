"""
src.interpretation.duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Duration and persistence analysis for latent market regimes on the Training partition.

Methodological Separation:
--------------------------
This module rigorously separates three distinct concepts that must NEVER be conflated:
1. Model-Implied Expected Duration:
   The theoretical duration implied by the Markov transition matrix:
   E[T_k] = 1 / (1 - A_{kk}) bars = 0.25 / (1 - A_{kk}) hours.
   This is purely a property of the learned transition parameters under the geometric
   distribution assumption, independent of actual empirical trajectories.

2. Retrospective Decoded Run Lengths (Viterbi):
   Contiguous runs of identical states decoded retrospectively by the multi-sequence
   Viterbi algorithm on the Training observations.
   Strict Boundary Rule: Runs are bounded by continuous session breaks (weekends, closures).
   A state active before a gap does NOT continue across the gap; each continuous segment
   is decoded independently, and runs terminate at segment boundaries.

3. Causal Filtered-Argmax Run Lengths & Switching Frequency:
   Contiguous runs and switching rate calculated from causal real-time filtered probabilities
   s_t = argmax_k P(S_t = k | O_{1:t}).
   Reflects real-time regime persistence, which may differ significantly from both
   theoretical duration and smoothed/Viterbi retrospective paths due to filtering lag
   and boundary noise.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_model_implied_duration(
    transmat: np.ndarray,
    bar_duration_hours: float = 0.25,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate theoretical expected duration and geometric variance from transition matrix.

    For a discrete-time Markov chain, the time spent in state k before transitioning
    to another state follows a Geometric distribution with success parameter p = 1 - A_{kk}:
        E[T_k] = 1 / (1 - A_{kk})  (in bars)
        Var(T_k) = A_{kk} / (1 - A_{kk})^2  (in bars^2)

    Parameters
    ----------
    transmat : np.ndarray, shape (K, K)
        Row-stochastic transition probability matrix.
    bar_duration_hours : float, default 0.25
        Duration of one bar in hours (0.25 hours = 15 minutes).

    Returns
    -------
    dict : Dict[int, Dict[str, float]]
        Theoretical duration metrics per state k.
    """
    K = transmat.shape[0]
    results = {}

    for k in range(K):
        a_kk = float(transmat[k, k])
        # Probability of exiting state k on next step
        p_exit = 1.0 - a_kk

        if p_exit <= 1e-12:
            expected_bars = float("inf")
            std_bars = float("inf")
            expected_hours = float("inf")
        else:
            expected_bars = 1.0 / p_exit
            var_bars = a_kk / (p_exit ** 2)
            std_bars = float(np.sqrt(max(0.0, var_bars)))
            expected_hours = expected_bars * bar_duration_hours

        results[k] = {
            "self_transition_prob": a_kk,
            "exit_prob": p_exit,
            "expected_bars": float(expected_bars),
            "std_bars": float(std_bars),
            "expected_hours": float(expected_hours),
        }

    return results


def viterbi_decode_multi_sequence(
    model: Any,
    X: np.ndarray,
    lengths: List[int],
) -> np.ndarray:
    """
    Multi-sequence log-space Viterbi algorithm.

    Decodes the globally most probable latent state path for each continuous sequence
    independently. Messages and transitions NEVER propagate across sequence boundaries.

    Parameters
    ----------
    model : Any
        Fitted GaussianHMM or GMMHMM instance with startprob_, transmat_, and _compute_log_emissions.
    X : np.ndarray, shape (N, D)
        Observations.
    lengths : List[int]
        Lengths of continuous segments summing to N.

    Returns
    -------
    best_path : np.ndarray, shape (N,), dtype int
        Decoded hidden state sequence.
    """
    N, D = X.shape
    if sum(lengths) != N:
        raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match X length ({N}).")

    startprob = np.asarray(model.startprob_, dtype=np.float64)
    transmat = np.asarray(model.transmat_, dtype=np.float64)
    K = len(startprob)

    log_startprob = np.log(np.maximum(startprob, 1e-300))
    log_transmat = np.log(np.maximum(transmat, 1e-300))

    best_path = np.empty(N, dtype=np.int64)
    offset = 0

    for seg_idx, T_seg in enumerate(lengths):
        if T_seg <= 0:
            continue

        X_seg = X[offset : offset + T_seg]
        log_B_seg = model._compute_log_emissions(X_seg)  # (T_seg, K)

        # Dynamic programming trellis in log-space
        viterbi_trellis = np.empty((T_seg, K), dtype=np.float64)
        backpointer = np.empty((T_seg, K), dtype=np.int64)

        # Initialization at t = 0
        viterbi_trellis[0] = log_startprob + log_B_seg[0]
        backpointer[0] = 0

        # Induction for t = 1 ... T_seg - 1
        for t in range(1, T_seg):
            # Shape: (K, 1) + (K, K) -> broadcast along previous states
            # transition_scores[j, k] = viterbi_trellis[t-1, j] + log_transmat[j, k]
            prev_scores = viterbi_trellis[t - 1, :, None] + log_transmat  # (K, K)
            backpointer[t] = np.argmax(prev_scores, axis=0)  # (K,)
            viterbi_trellis[t] = np.max(prev_scores, axis=0) + log_B_seg[t]  # (K,)

        # Backtracking
        seg_path = np.empty(T_seg, dtype=np.int64)
        seg_path[T_seg - 1] = int(np.argmax(viterbi_trellis[T_seg - 1]))

        for t in range(T_seg - 2, -1, -1):
            seg_path[t] = backpointer[t + 1, seg_path[t + 1]]

        best_path[offset : offset + T_seg] = seg_path
        offset += T_seg

    return best_path


def extract_run_lengths(
    state_sequence: np.ndarray,
    lengths: List[int],
    n_states: int,
) -> Dict[int, List[int]]:
    """
    Extract contiguous run lengths per state, strictly bounded by sequence boundaries.

    Runs NEVER span across continuous sequence breaks. If a sequence ends in state k
    and the next sequence begins in state k, they are treated as two distinct runs.

    Parameters
    ----------
    state_sequence : np.ndarray, shape (N,)
        Discrete state labels.
    lengths : List[int]
        Continuous segment lengths summing to N.
    n_states : int
        Total number of latent states.

    Returns
    -------
    runs_by_state : Dict[int, List[int]]
        List of integer run lengths for each state k in 0 ... n_states - 1.
    """
    runs_by_state: Dict[int, List[int]] = {k: [] for k in range(n_states)}
    offset = 0

    for T_seg in lengths:
        if T_seg <= 0:
            continue

        seg_states = state_sequence[offset : offset + T_seg]
        curr_state = int(seg_states[0])
        curr_len = 1

        for t in range(1, T_seg):
            s = int(seg_states[t])
            if s == curr_state:
                curr_len += 1
            else:
                runs_by_state[curr_state].append(curr_len)
                curr_state = s
                curr_len = 1

        # Record final run of the segment
        runs_by_state[curr_state].append(curr_len)
        offset += T_seg

    return runs_by_state


def summarize_run_length_distribution(
    runs: List[int],
    bar_duration_hours: float = 0.25,
) -> Dict[str, float]:
    """
    Calculate summary statistics for a list of integer run lengths.

    Parameters
    ----------
    runs : List[int]
    bar_duration_hours : float

    Returns
    -------
    dict with count, mean, std, median, min, max, q25, q75, mean_hours, median_hours.
    """
    if not runs:
        return {
            "count": 0,
            "mean_bars": 0.0,
            "std_bars": 0.0,
            "median_bars": 0.0,
            "min_bars": 0,
            "max_bars": 0,
            "q25_bars": 0.0,
            "q75_bars": 0.0,
            "mean_hours": 0.0,
            "median_hours": 0.0,
        }

    arr = np.array(runs, dtype=np.float64)
    mean_b = float(np.mean(arr))
    std_b = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    med_b = float(np.median(arr))
    min_b = int(np.min(arr))
    max_b = int(np.max(arr))
    q25_b = float(np.percentile(arr, 25))
    q75_b = float(np.percentile(arr, 75))

    return {
        "count": len(runs),
        "mean_bars": mean_b,
        "std_bars": std_b,
        "median_bars": med_b,
        "min_bars": min_b,
        "max_bars": max_b,
        "q25_bars": q25_b,
        "q75_bars": q75_b,
        "mean_hours": mean_b * bar_duration_hours,
        "median_hours": med_b * bar_duration_hours,
    }


def compute_switching_frequency(
    state_sequence: np.ndarray,
    lengths: List[int],
) -> Tuple[float, int, int]:
    """
    Calculate empirical state switching frequency strictly within continuous segments.

    Switches across sequence breaks are excluded because no transition occurred
    during market closures or weekends.

    Parameters
    ----------
    state_sequence : np.ndarray, shape (N,)
    lengths : List[int]

    Returns
    -------
    switch_rate : float
        Proportion of transitions that resulted in a state change: n_switches / n_transitions.
    n_switches : int
        Total number of state changes.
    n_transitions : int
        Total number of active within-sequence transitions: sum(T_seg - 1).
    """
    total_switches = 0
    total_transitions = 0
    offset = 0

    for T_seg in lengths:
        if T_seg <= 1:
            offset += T_seg
            continue

        seg_states = state_sequence[offset : offset + T_seg]
        # Count transitions where state changes
        switches = int(np.sum(seg_states[1:] != seg_states[:-1]))
        transitions = T_seg - 1

        total_switches += switches
        total_transitions += transitions
        offset += T_seg

    switch_rate = float(total_switches / total_transitions) if total_transitions > 0 else 0.0
    return switch_rate, total_switches, total_transitions


@dataclass
class RegimeDurationAnalysis:
    """Comprehensive persistence and duration analysis for a fitted regime model."""
    model_version: str
    n_states: int
    total_samples: int
    n_continuous_sequences: int

    # 1. Theoretical Model-Implied
    model_implied: Dict[int, Dict[str, float]]

    # 2. Retrospective Decoded (Viterbi)
    viterbi_runs_summary: Dict[int, Dict[str, float]]
    viterbi_switch_rate: float
    viterbi_total_switches: int

    # 3. Causal Filtered-Argmax (if provided)
    filtered_runs_summary: Optional[Dict[int, Dict[str, float]]]
    filtered_switch_rate: Optional[float]
    filtered_total_switches: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert duration analysis to serializable dictionary."""
        return {
            "model_version": self.model_version,
            "n_states": self.n_states,
            "total_samples": self.total_samples,
            "n_continuous_sequences": self.n_continuous_sequences,
            "model_implied_expected_duration": self.model_implied,
            "retrospective_viterbi_durations": {
                "per_state_runs": self.viterbi_runs_summary,
                "overall_switching_rate": self.viterbi_switch_rate,
                "total_switches": self.viterbi_total_switches,
            },
            "causal_filtered_durations": {
                "per_state_runs": self.filtered_runs_summary,
                "overall_switching_rate": self.filtered_switch_rate,
                "total_switches": self.filtered_total_switches,
            }
            if self.filtered_runs_summary is not None
            else None,
        }


def analyze_regime_durations(
    model: Any,
    X: np.ndarray,
    lengths: List[int],
    model_version: str,
    filtered_probs: Optional[np.ndarray] = None,
    bar_duration_hours: float = 0.25,
) -> RegimeDurationAnalysis:
    """
    Perform complete duration and persistence analysis for a fitted regime model on Training data.

    Parameters
    ----------
    model : Any
        Fitted GaussianHMM or GMMHMM.
    X : np.ndarray, shape (N, D)
        Training observation matrix.
    lengths : List[int]
        Continuous segment lengths summing to N.
    model_version : str
        Model identifier.
    filtered_probs : Optional[np.ndarray], shape (N, K)
        Causal filtered probabilities P(S_t = k | O_{1:t}) if available.
    bar_duration_hours : float, default 0.25
        Duration of one bar in hours.

    Returns
    -------
    RegimeDurationAnalysis
    """
    K = int(model.n_components)
    N = int(len(X))

    # 1. Model-implied expected duration
    model_implied = compute_model_implied_duration(
        transmat=model.transmat_,
        bar_duration_hours=bar_duration_hours,
    )

    # 2. Retrospective Viterbi decoding
    viterbi_path = viterbi_decode_multi_sequence(model=model, X=X, lengths=lengths)
    viterbi_runs = extract_run_lengths(state_sequence=viterbi_path, lengths=lengths, n_states=K)
    viterbi_summary = {
        k: summarize_run_length_distribution(viterbi_runs[k], bar_duration_hours=bar_duration_hours)
        for k in range(K)
    }
    viterbi_rate, viterbi_switches, _ = compute_switching_frequency(
        state_sequence=viterbi_path, lengths=lengths
    )

    # 3. Causal Filtered-Argmax (if provided)
    filtered_summary = None
    filtered_rate = None
    filtered_switches = None

    if filtered_probs is not None:
        filtered_path = np.argmax(filtered_probs, axis=1)
        filtered_runs = extract_run_lengths(
            state_sequence=filtered_path, lengths=lengths, n_states=K
        )
        filtered_summary = {
            k: summarize_run_length_distribution(
                filtered_runs[k], bar_duration_hours=bar_duration_hours
            )
            for k in range(K)
        }
        filtered_rate, filtered_switches, _ = compute_switching_frequency(
            state_sequence=filtered_path, lengths=lengths
        )

    return RegimeDurationAnalysis(
        model_version=model_version,
        n_states=K,
        total_samples=N,
        n_continuous_sequences=len(lengths),
        model_implied=model_implied,
        viterbi_runs_summary=viterbi_summary,
        viterbi_switch_rate=viterbi_rate,
        viterbi_total_switches=viterbi_switches,
        filtered_runs_summary=filtered_summary,
        filtered_switch_rate=filtered_rate,
        filtered_total_switches=filtered_switches,
    )

