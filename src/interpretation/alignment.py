"""
src.interpretation.alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reproducible label alignment and semantic regime interpretation across model versions.

Methodological Principles:
--------------------------
1. Permutation Invariance & Label Switching:
   Unsupervised latent models (Gaussian HMM, GMM-HMM) have no intrinsic state ordering.
   State indices (0, 1, ..., K-1) are arbitrary and can switch between random seeds,
   re-training runs, or architecture variants (e.g. M=1 vs M=2 vs M=3).

2. Training-Only Canonical Profiling:
   Label alignment and reference scalers are strictly computed from in-sample Training
   observations (12,501 bars). Zero test or validation information is used.

3. Canonical Reference Model:
   The validated GMM-HMM K=3 M=3 model (v1) serves as the canonical baseline.
   Its latent states define the canonical profile reference coordinates.

4. Normalized Profile Distance:
   Profile vectors v_k = [mu_r, sigma_r, mu_vol, sigma_vol, P(r > 0)] are normalized
   using the canonical reference model's feature scales before calculating Euclidean cost:
   C_{i, j} = || (v_i^cand - v_j^ref) / s_ref ||_2.
   To prevent division by near-zero variances on drift/win-rate across states in neutral
   markets, economic scale floors (e.g., 1 pip for return drift) are enforced.

5. Hungarian Bipartite Matching with Rejection:
   Matches are computed via linear sum assignment (scipy.optimize.linear_sum_assignment).
   Robustness safeguards include:
   - Distance threshold rejection: Matches exceeding d_max are rejected as novel/unmatched.
   - Ambiguity detection: Candidates within delta_ambiguity of a second reference state
     are explicitly flagged.
   - Unequal state handling: Candidate states exceeding reference K remain UNMATCHED.

6. Two Independent Descriptive Dimensions:
   Regimes are characterized along two independent axes:
   - Volatility Dimension (Low, Medium, High, Extreme/Turbulence)
   - Directional Dimension (Neutral, Positive Drift, Negative Drift, Mixed)
   High volatility is NEVER automatically labeled as Bearish without empirical negative drift.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.interpretation.characteristics import RegimeCharacteristics, StateCharacteristics


PROFILE_FEATURE_NAMES = [
    "r_oc_mean",
    "r_oc_std",
    "log_vol_mean",
    "log_vol_std",
    "prob_positive_return",
]

# Economic scale floors to prevent near-zero denominator blowing up Euclidean distance
# 1e-4 = 1 pip return, 1e-4 = 1 pip return std, 0.10 = 0.10 log-vol, 0.05 = 0.05 vol std, 0.05 = 5% win rate
DEFAULT_SCALE_FLOORS = [1e-4, 1e-4, 0.10, 0.05, 0.05]


def extract_profile_vector(state_char: StateCharacteristics) -> np.ndarray:
    """
    Extract the 5-dimensional canonical profile vector for a latent state.

    v_k = [mu_r, sigma_r, mu_vol, sigma_vol, P(r > 0)]
    """
    return np.array(
        [
            state_char.r_oc_mean,
            state_char.r_oc_std,
            state_char.log_vol_mean,
            state_char.log_vol_std,
            state_char.prob_positive_return,
        ],
        dtype=np.float64,
    )


@dataclass
class ReferenceProfileScaler:
    """
    Feature-wise standardization fitted strictly on the canonical reference states.

    Prevents any single dimension (e.g. log_vol ~ -8 vs r_mean ~ 1e-5) from dominating
    the Euclidean matching distance.
    """
    feature_names: List[str]
    means: np.ndarray
    stds: np.ndarray
    scale_floors: List[float]

    @classmethod
    def fit(
        cls,
        reference_chars: RegimeCharacteristics,
        scale_floors: Optional[List[float]] = None,
    ) -> "ReferenceProfileScaler":
        """Fit feature scales on the reference model's states."""
        if scale_floors is None:
            scale_floors = list(DEFAULT_SCALE_FLOORS)

        profiles = []
        for k in sorted(reference_chars.states.keys()):
            profiles.append(extract_profile_vector(reference_chars.states[k]))

        arr = np.array(profiles, dtype=np.float64)  # (K_ref, 5)
        means = np.mean(arr, axis=0)
        # Compute population standard deviation across reference states
        stds = np.std(arr, axis=0)
        # Apply economic scale floors to avoid zero-division or over-sensitivity to near-zero drift
        floors_arr = np.array(scale_floors, dtype=np.float64)
        stds = np.maximum(stds, floors_arr)

        return cls(
            feature_names=list(PROFILE_FEATURE_NAMES),
            means=means,
            stds=stds,
            scale_floors=scale_floors,
        )

    def transform(self, profile: np.ndarray) -> np.ndarray:
        """Standardize a profile vector using reference mean and std."""
        return (profile - self.means) / self.stds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "scale_floors": self.scale_floors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReferenceProfileScaler":
        return cls(
            feature_names=d["feature_names"],
            means=np.array(d["means"], dtype=np.float64),
            stds=np.array(d["stds"], dtype=np.float64),
            scale_floors=d.get("scale_floors", list(DEFAULT_SCALE_FLOORS)),
        )


def classify_regime_dimensions(
    state_char: StateCharacteristics,
) -> Tuple[str, str, str]:
    """
    Classify a state into two independent financial dimensions: Volatility and Direction.

    Rules for EURUSD M15:
    ---------------------
    1. Volatility Dimension (based on log_vol_mean and candle range):
       - Low_Volatility: log_vol_mean < -7.80 (typical candle range < 4.5 bps / ~4.5 pips)
       - Medium_Volatility: -7.80 <= log_vol_mean < -7.10 (candle range ~ 4.5 - 9.0 bps)
       - High_Volatility: log_vol_mean >= -7.10 (candle range >= 9.0 bps)

    2. Directional Dimension (based on return mean and win rate):
       - Neutral: |P(r>0) - 0.50| <= 0.02 and |mu_r| <= 2e-5
       - Positive_Drift: P(r>0) > 0.52 and mu_r > 2e-5
       - Negative_Drift: P(r>0) < 0.48 and mu_r < -2e-5
       - Mixed: Neither strongly directional nor tightly neutral

    Returns
    -------
    volatility_label : str
    direction_label : str
    composite_label : str
    """
    vol = state_char.log_vol_mean
    r_mean = state_char.r_oc_mean
    p_pos = state_char.prob_positive_return

    # 1. Volatility Classification
    if vol < -7.80:
        vol_label = "Low_Volatility"
        vol_tag = "LowVol"
    elif vol < -7.10:
        vol_label = "Medium_Volatility"
        vol_tag = "MedVol"
    else:
        vol_label = "High_Volatility"
        vol_tag = "HighVol"

    # 2. Directional Classification
    if abs(p_pos - 0.50) <= 0.02 and abs(r_mean) <= 2e-5:
        dir_label = "Neutral"
        dir_tag = "Neutral"
    elif p_pos > 0.52 and r_mean > 2e-5:
        dir_label = "Positive_Drift"
        dir_tag = "PosDrift"
    elif p_pos < 0.48 and r_mean < -2e-5:
        dir_label = "Negative_Drift"
        dir_tag = "NegDrift"
    else:
        dir_label = "Mixed"
        dir_tag = "Mixed"

    composite_label = f"{vol_tag}_{dir_tag}"
    return vol_label, dir_label, composite_label


def compute_profile_cost_matrix(
    cand_profiles: np.ndarray,
    ref_profiles: np.ndarray,
    scaler: ReferenceProfileScaler,
) -> np.ndarray:
    """
    Compute pairwise normalized Euclidean distance cost matrix.

    Parameters
    ----------
    cand_profiles : np.ndarray, shape (K_cand, D)
    ref_profiles : np.ndarray, shape (K_ref, D)
    scaler : ReferenceProfileScaler

    Returns
    -------
    cost_matrix : np.ndarray, shape (K_cand, K_ref)
    """
    K_cand = len(cand_profiles)
    K_ref = len(ref_profiles)

    # Standardize both candidate and reference using reference scaler
    cand_norm = (cand_profiles - scaler.means) / scaler.stds  # (K_cand, D)
    ref_norm = (ref_profiles - scaler.means) / scaler.stds    # (K_ref, D)

    # Compute Euclidean distance
    cost_matrix = np.empty((K_cand, K_ref), dtype=np.float64)
    for i in range(K_cand):
        diff = ref_norm - cand_norm[i]  # (K_ref, D)
        cost_matrix[i] = np.linalg.norm(diff, axis=1)

    return cost_matrix


@dataclass
class StateAlignmentMatch:
    """Result of matching a single candidate state to a canonical reference state."""
    cand_state_id: int
    matched_ref_state_id: Optional[int]
    canonical_semantic_label: str
    volatility_dimension: str
    direction_dimension: str
    matching_cost: float
    status: str  # 'MATCHED', 'REJECTED_DISTANCE', 'UNMATCHED_EXCESS'
    is_ambiguous: bool
    second_best_ref_id: Optional[int]
    second_best_cost: Optional[float]
    cost_margin: Optional[float]
    competitor_cand_id: Optional[int] = None
    competitor_cost_margin: Optional[float] = None


@dataclass
class ModelAlignmentReport:
    """Complete alignment report mapping candidate model states to canonical reference."""
    candidate_model_version: str
    reference_model_version: str
    k_candidate: int
    k_reference: int
    distance_threshold: float
    ambiguity_margin: float

    cost_matrix: List[List[float]]
    matches: Dict[int, StateAlignmentMatch]

    # Diagnostics
    candidate_match_rate: float
    reference_coverage: float
    average_matched_cost: float
    max_matched_cost: float
    ambiguous_state_count: int
    unmatched_state_count: int

    # Global Assignment Diagnostics
    global_assignment_gap: float = float("inf")
    is_globally_ambiguous: bool = False
    threshold_type: str = "operational_heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_model_version": self.candidate_model_version,
            "reference_model_version": self.reference_model_version,
            "k_candidate": self.k_candidate,
            "k_reference": self.k_reference,
            "distance_threshold": self.distance_threshold,
            "ambiguity_margin": self.ambiguity_margin,
            "threshold_type": self.threshold_type,
            "global_assignment_gap": self.global_assignment_gap,
            "is_globally_ambiguous": self.is_globally_ambiguous,
            "cost_matrix": self.cost_matrix,
            "matches": {
                str(k): {
                    "cand_state_id": m.cand_state_id,
                    "matched_ref_state_id": m.matched_ref_state_id,
                    "canonical_semantic_label": m.canonical_semantic_label,
                    "volatility_dimension": m.volatility_dimension,
                    "direction_dimension": m.direction_dimension,
                    "matching_cost": m.matching_cost,
                    "status": m.status,
                    "is_ambiguous": m.is_ambiguous,
                    "second_best_ref_id": m.second_best_ref_id,
                    "second_best_cost": m.second_best_cost,
                    "cost_margin": m.cost_margin,
                    "competitor_cand_id": m.competitor_cand_id,
                    "competitor_cost_margin": m.competitor_cost_margin,
                }
                for k, m in self.matches.items()
            },
            "diagnostics": {
                "candidate_match_rate": self.candidate_match_rate,
                "reference_coverage": self.reference_coverage,
                "average_matched_cost": self.average_matched_cost,
                "max_matched_cost": self.max_matched_cost,
                "ambiguous_state_count": self.ambiguous_state_count,
                "unmatched_state_count": self.unmatched_state_count,
            },
        }


def compute_global_assignment_gap(
    cost_matrix: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Calculate the second-best global assignment cost and gap using single-edge exclusion.

    Returns
    -------
    opt_cost : float
    second_cost : float
    gap : float
    """
    opt_cost = float(sum(cost_matrix[r, c] for r, c in zip(row_ind, col_ind)))
    second_costs = []

    for r_ex, c_ex in zip(row_ind, col_ind):
        C_mod = cost_matrix.copy()
        C_mod[r_ex, c_ex] = 1e9
        r_sub, c_sub = linear_sum_assignment(C_mod)
        sub_cost = float(sum(cost_matrix[r, c] for r, c in zip(r_sub, c_sub)))
        if sub_cost < 1e8:
            second_costs.append(sub_cost)

    if not second_costs:
        return opt_cost, float("inf"), float("inf")

    best_second = float(min(second_costs))
    gap = float(best_second - opt_cost)
    return opt_cost, best_second, gap


def align_model_states(
    candidate_chars: RegimeCharacteristics,
    reference_chars: RegimeCharacteristics,
    scaler: Optional[ReferenceProfileScaler] = None,
    distance_threshold: float = 2.5,
    ambiguity_margin: float = 0.20,
) -> ModelAlignmentReport:
    """
    Perform Hungarian bipartite matching between candidate model states and reference states.

    Parameters
    ----------
    candidate_chars : RegimeCharacteristics
        Characteristics of candidate model on Training data.
    reference_chars : RegimeCharacteristics
        Characteristics of canonical reference model on Training data.
    scaler : Optional[ReferenceProfileScaler]
        Reference profile scaler. If None, fitted directly on reference_chars.
    distance_threshold : float, default 2.5
        Maximum allowable normalized Euclidean distance. Beyond this, match is rejected.
    ambiguity_margin : float, default 0.20
        Cost margin below which a match is flagged as ambiguous.

    Returns
    -------
    report : ModelAlignmentReport
    """
    K_cand = candidate_chars.n_components
    K_ref = reference_chars.n_components

    if scaler is None:
        scaler = ReferenceProfileScaler.fit(reference_chars)

    # Extract profiles
    cand_order = sorted(candidate_chars.states.keys())
    ref_order = sorted(reference_chars.states.keys())

    cand_profiles = np.array(
        [extract_profile_vector(candidate_chars.states[k]) for k in cand_order],
        dtype=np.float64,
    )
    ref_profiles = np.array(
        [extract_profile_vector(reference_chars.states[k]) for k in ref_order],
        dtype=np.float64,
    )

    # Compute cost matrix
    cost_matrix = compute_profile_cost_matrix(cand_profiles, ref_profiles, scaler)

    # Reference labels for assigning semantic names
    ref_labels = {}
    for ref_k in ref_order:
        ref_state = reference_chars.states[ref_k]
        vol_l, dir_l, comp_l = classify_regime_dimensions(ref_state)
        canonical_name = f"Regime_{ref_k}_{comp_l}"
        ref_labels[ref_k] = (vol_l, dir_l, canonical_name)

    # Solve Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = dict(zip(row_ind, col_ind))

    # Compute global assignment gap (Murty single-edge exclusion)
    opt_cost, second_best_global_cost, global_assignment_gap = compute_global_assignment_gap(
        cost_matrix, row_ind, col_ind
    )
    is_globally_ambiguous = bool(global_assignment_gap < ambiguity_margin)

    matches: Dict[int, StateAlignmentMatch] = {}
    matched_costs = []
    accepted_ref_states = set()
    ambiguous_count = 0
    unmatched_count = 0

    for cand_idx, cand_k in enumerate(cand_order):
        cand_state = candidate_chars.states[cand_k]
        costs_for_cand = cost_matrix[cand_idx]

        if cand_idx in assignment:
            ref_idx = assignment[cand_idx]
            ref_k = ref_order[ref_idx]
            cost = float(costs_for_cand[ref_idx])

            # 1. Reference-side alternative margin:
            # Find closest alternative reference state j != ref_idx
            ref_alt_indices = [j for j in range(K_ref) if j != ref_idx]
            if ref_alt_indices:
                alt_costs = costs_for_cand[ref_alt_indices]
                best_alt_local_idx = int(np.argmin(alt_costs))
                best_alt_ref_idx = ref_alt_indices[best_alt_local_idx]
                second_best_k = ref_order[best_alt_ref_idx]
                second_best_cost = float(costs_for_cand[best_alt_ref_idx])
                ref_margin = float(abs(second_best_cost - cost))
            else:
                second_best_k = None
                second_best_cost = None
                ref_margin = float("inf")

            # 2. Candidate-side competitor margin:
            # Does any other candidate state i' != cand_idx have a similar distance to ref_idx?
            cand_comp_indices = [i for i in range(K_cand) if i != cand_idx]
            if cand_comp_indices:
                cand_costs_to_ref = cost_matrix[cand_comp_indices, ref_idx]
                best_comp_local_idx = int(np.argmin(cand_costs_to_ref))
                best_comp_cand_idx = cand_comp_indices[best_comp_local_idx]
                comp_cand_k = cand_order[best_comp_cand_idx]
                comp_cost = float(cost_matrix[best_comp_cand_idx, ref_idx])
                comp_margin = float(abs(comp_cost - cost))
            else:
                comp_cand_k = None
                comp_cost = None
                comp_margin = float("inf")

            # State is locally ambiguous if either reference-side or competitor-side margin is below threshold
            is_ambig = bool(
                ref_margin < ambiguity_margin or comp_margin < ambiguity_margin
            )
            margin = min(ref_margin, comp_margin)

            if is_ambig:
                ambiguous_count += 1

            # Distance threshold check
            if cost > distance_threshold:
                # Rejected due to excessive distance
                unmatched_count += 1
                cand_vol, cand_dir, cand_comp = classify_regime_dimensions(cand_state)
                matches[cand_k] = StateAlignmentMatch(
                    cand_state_id=cand_k,
                    matched_ref_state_id=None,
                    canonical_semantic_label=f"Novel_{cand_comp}",
                    volatility_dimension=cand_vol,
                    direction_dimension=cand_dir,
                    matching_cost=cost,
                    status="REJECTED_DISTANCE",
                    is_ambiguous=is_ambig,
                    second_best_ref_id=second_best_k,
                    second_best_cost=second_best_cost,
                    cost_margin=margin,
                    competitor_cand_id=comp_cand_k,
                    competitor_cost_margin=comp_margin if comp_cand_k is not None else None,
                )
            else:
                # Successfully matched
                vol_l, dir_l, canonical_name = ref_labels[ref_k]
                matched_costs.append(cost)
                accepted_ref_states.add(ref_k)

                matches[cand_k] = StateAlignmentMatch(
                    cand_state_id=cand_k,
                    matched_ref_state_id=ref_k,
                    canonical_semantic_label=canonical_name,
                    volatility_dimension=vol_l,
                    direction_dimension=dir_l,
                    matching_cost=cost,
                    status="MATCHED",
                    is_ambiguous=is_ambig,
                    second_best_ref_id=second_best_k,
                    second_best_cost=second_best_cost,
                    cost_margin=margin,
                    competitor_cand_id=comp_cand_k,
                    competitor_cost_margin=comp_margin if comp_cand_k is not None else None,
                )
        else:
            # Excess candidate state beyond K_ref
            unmatched_count += 1
            cand_vol, cand_dir, cand_comp = classify_regime_dimensions(cand_state)
            matches[cand_k] = StateAlignmentMatch(
                cand_state_id=cand_k,
                matched_ref_state_id=None,
                canonical_semantic_label=f"Unmatched_Excess_{cand_comp}",
                volatility_dimension=cand_vol,
                direction_dimension=cand_dir,
                matching_cost=float("inf"),
                status="UNMATCHED_EXCESS",
                is_ambiguous=False,
                second_best_ref_id=None,
                second_best_cost=None,
                cost_margin=None,
                competitor_cand_id=None,
                competitor_cost_margin=None,
            )

    cand_match_rate = (
        float(len(accepted_ref_states) / K_cand) if K_cand > 0 else 0.0
    )
    ref_coverage = (
        float(len(accepted_ref_states) / K_ref) if K_ref > 0 else 0.0
    )
    avg_cost = float(np.mean(matched_costs)) if matched_costs else 0.0
    max_cost = float(np.max(matched_costs)) if matched_costs else 0.0

    return ModelAlignmentReport(
        candidate_model_version=candidate_chars.model_version,
        reference_model_version=reference_chars.model_version,
        k_candidate=K_cand,
        k_reference=K_ref,
        distance_threshold=distance_threshold,
        ambiguity_margin=ambiguity_margin,
        cost_matrix=cost_matrix.tolist(),
        matches=matches,
        candidate_match_rate=cand_match_rate,
        reference_coverage=ref_coverage,
        average_matched_cost=avg_cost,
        max_matched_cost=max_cost,
        ambiguous_state_count=ambiguous_count,
        unmatched_state_count=unmatched_count,
        global_assignment_gap=global_assignment_gap,
        is_globally_ambiguous=is_globally_ambiguous,
        threshold_type="operational_heuristic",
    )
