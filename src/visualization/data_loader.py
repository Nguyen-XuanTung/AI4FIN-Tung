"""
src.visualization.data_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data loading, validation, entropy computation, and gap-aware segmentation
for Phase 8: Historical Regime Visualization.

Enforces strict data integrity:
- Zero forward-filling of OHLC.
- Zero synthetic candles.
- Rejection of invalid probability vectors.
- Clear separation between Full-Year Retrospective and As-Of Replay data.
- Session gap segmentation to prevent plotting false lines across closures.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.candidates import compute_log_returns_oc, compute_log_volatility
from src.inference.calendar import FXCalendarGapHandler
from src.inference.engine import RealTimeInferenceEngine
from src.inference.versioning import ModelArtifact

logger = logging.getLogger("VisualizationDataLoader")

# Interpretive labels established in Phase 6 (audited against canonical reference)
STATE_LABELS = {
    0: "LowVol_Neutral",
    1: "HighVol_Neutral",
    2: "MedVol_Neutral",
}

# Neutral, professional palette:
# State 0 (LowVol): Deep Steel Blue (#1f77b4)
# State 2 (MedVol): Warm Ochre Amber (#ff7f0e)
# State 1 (HighVol): Imperial Purple (#7b3294) - avoiding red/green directional bias
STATE_COLORS = {
    0: "#1f77b4",  # Steel Blue (Low Volatility)
    1: "#7b3294",  # Imperial Purple (High Volatility)
    2: "#ff7f0e",  # Ochre Amber (Medium Volatility)
}

STATE_DESCRIPTIONS = {
    0: "State 0: LowVol_Neutral (σ ≈ 1.7 bps, ~3.3 pips range)",
    1: "State 1: HighVol_Neutral (σ ≈ 8.4 bps, ~13.0 pips range)",
    2: "State 2: MedVol_Neutral (σ ≈ 3.9 bps, ~6.6 pips range)",
}


def load_canonical_model(
    model_path: str = "models/GMM_HMM_K3_M3_v1.json",
) -> ModelArtifact:
    """
    Load canonical GMM-HMM K3-M3 model artifact and verify parameter integrity.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Canonical model artifact not found: {model_path}")

    artifact = ModelArtifact.load_json(str(p))
    logger.info("Loaded canonical model artifact: %s (hash: %s)", artifact.model_version, artifact.parameter_hash)
    return artifact


def compute_entropy_series(
    probs: np.ndarray,
    n_states: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Shannon entropy and normalized entropy safely for probability vectors.

    Parameters
    ----------
    probs : np.ndarray of shape (N, n_states)
        Filtered posterior probabilities.
    n_states : int, default 3
        Number of latent states K.

    Returns
    -------
    entropy : np.ndarray of shape (N,)
        Raw Shannon entropy in nats: H_t = -sum_j p_j * ln(p_j).
    norm_entropy : np.ndarray of shape (N,)
        Normalized entropy in [0, 1]: H_t / ln(K).
    """
    probs = np.asarray(probs, dtype=np.float64)
    # Clip probabilities to valid range [0, 1] for robust log evaluation
    probs_safe = np.clip(probs, 0.0, 1.0)
    
    # Safe 0 * log(0) = 0 handling
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs_safe > 0.0, np.log(probs_safe), 0.0)
        entropy = -np.sum(probs_safe * log_p, axis=1)

    max_entropy = np.log(float(n_states))
    norm_entropy = entropy / max_entropy
    norm_entropy = np.clip(norm_entropy, 0.0, 1.0)

    return entropy, norm_entropy


def validate_probability_frame(
    df: pd.DataFrame,
    prob_cols: List[str] = ["prob_state_0", "prob_state_1", "prob_state_2"],
    tol: float = 1e-6,
) -> None:
    """
    Verify probability validity:
    1. Sum of probabilities equals 1.0 within tolerance.
    2. Zero values strictly outside [0, 1].
    3. Monotonic UTC timestamps.
    """
    probs = df[prob_cols].values
    prob_sums = probs.sum(axis=1)

    invalid_sums = np.abs(prob_sums - 1.0) > tol
    if np.any(invalid_sums):
        n_inv = int(np.sum(invalid_sums))
        max_dev = float(np.max(np.abs(prob_sums - 1.0)))
        raise ValueError(f"Found {n_inv} rows where probability sum deviates from 1.0 (max deviation: {max_dev:.2e})")

    if np.any(probs < -tol) or np.any(probs > 1.0 + tol):
        raise ValueError("Probabilities outside valid range [0.0, 1.0] detected.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly monotonically increasing.")


def extract_continuous_segments(
    df: pd.DataFrame,
    max_gap_seconds: int = 900,
) -> List[pd.DataFrame]:
    """
    Split dataframe into continuous trading session segments where adjacent
    bars have elapsed delta <= max_gap_seconds (default 15 minutes = 900s).

    Prevents plotting false continuous diagonal lines across weekend gaps or missing bars.
    """
    if df.empty:
        return []

    timestamps = pd.to_datetime(df["timestamp"])
    deltas = timestamps.diff().dt.total_seconds().fillna(max_gap_seconds)
    
    # New segment where gap > max_gap_seconds
    segment_ids = (deltas > max_gap_seconds).cumsum()
    segments = [group.copy() for _, group in df.groupby(segment_ids)]
    
    return segments


def compute_transition_and_run_lengths(
    df: pd.DataFrame,
    pred_state_col: str = "pred_state",
    max_gap_seconds: int = 900,
    n_states: int = 3,
) -> Dict[str, Any]:
    """
    Compute empirical transition counts, switching frequency, and run length
    distributions across continuous trading segments.

    CRITICAL SAFEGUARD: Never counts transitions across market closures or missing-bar gaps.
    """
    segments = extract_continuous_segments(df, max_gap_seconds=max_gap_seconds)

    trans_counts = np.zeros((n_states, n_states), dtype=np.int64)
    run_lengths: Dict[int, List[int]] = {s: [] for s in range(n_states)}
    total_adjacent_pairs = 0
    total_switches = 0

    for seg in segments:
        if len(seg) < 1:
            continue
        states = seg[pred_state_col].values.astype(int)

        # Count adjacent transitions within this continuous segment
        for t in range(1, len(states)):
            prev_s = states[t - 1]
            curr_s = states[t]
            trans_counts[prev_s, curr_s] += 1
            total_adjacent_pairs += 1
            if prev_s != curr_s:
                total_switches += 1

        # Extract run lengths within this continuous segment
        current_state = states[0]
        current_len = 1
        for t in range(1, len(states)):
            if states[t] == current_state:
                current_len += 1
            else:
                run_lengths[current_state].append(current_len)
                current_state = states[t]
                current_len = 1
        run_lengths[current_state].append(current_len)

    # Normalized transition probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_probs = np.where(row_sums > 0, trans_counts / row_sums, 0.0)

    switching_frequency = (total_switches / total_adjacent_pairs) if total_adjacent_pairs > 0 else 0.0

    run_stats = {}
    for s in range(n_states):
        lens = run_lengths[s]
        if len(lens) > 0:
            run_stats[s] = {
                "count": len(lens),
                "mean_bars": float(np.mean(lens)),
                "median_bars": float(np.median(lens)),
                "p90_bars": float(np.percentile(lens, 90)),
                "max_bars": int(np.max(lens)),
                "mean_hours": float(np.mean(lens) * 0.25),
            }
        else:
            run_stats[s] = {
                "count": 0,
                "mean_bars": 0.0,
                "median_bars": 0.0,
                "p90_bars": 0.0,
                "max_bars": 0,
                "mean_hours": 0.0,
            }

    return {
        "transition_counts": trans_counts.tolist(),
        "transition_probabilities": trans_probs.tolist(),
        "switching_frequency": float(switching_frequency),
        "total_switches": int(total_switches),
        "total_adjacent_pairs": int(total_adjacent_pairs),
        "run_lengths": run_lengths,
        "run_stats": run_stats,
    }


def load_as_of_replay_data(
    replay_path: str = "data/processed/as_of_replay_inference_log.parquet",
    ohlc_path: str = "data/processed/EURUSD_M15_2023.parquet",
) -> pd.DataFrame:
    """
    Load Mode B (As-Of Replay View) data:
    Strictly the 8,335 post-cutover complete bars from Phase 5 (2023-08-24 12:00 to 2023-12-29 21:30 UTC)
    using frozen GMM_HMM_K3_M3_v1.
    """
    p_rep = Path(replay_path)
    p_ohlc = Path(ohlc_path)

    if not p_rep.exists():
        raise FileNotFoundError(f"Replay log file not found: {replay_path}")
    if not p_ohlc.exists():
        raise FileNotFoundError(f"OHLC data file not found: {ohlc_path}")

    df_rep = pd.read_parquet(p_rep)
    df_ohlc = pd.read_parquet(p_ohlc)

    # Filter to valid complete updates
    df_rep = df_rep[df_rep["is_complete"] == True].copy()
    df_rep = df_rep[df_rep["status"] == "valid_update"].copy()

    # Join OHLC price fields
    ohlc_cols = ["timestamp", "open", "high", "low", "close", "tick_volume"]
    merged = pd.merge(df_rep, df_ohlc[ohlc_cols], on="timestamp", how="inner")

    if len(merged) != len(df_rep):
        raise ValueError(f"OHLC merge anomaly: expected {len(df_rep)} rows, got {len(merged)}")

    # Ensure timestamp is UTC datetime
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    validate_probability_frame(merged)

    # Compute Shannon entropy
    probs = merged[["prob_state_0", "prob_state_1", "prob_state_2"]].values
    ent, n_ent = compute_entropy_series(probs, n_states=3)
    merged["entropy"] = ent
    merged["normalized_entropy"] = n_ent

    # State semantic labels
    merged["pred_state_label"] = merged["pred_state"].map(STATE_LABELS)

    # Compute physical candle metrics
    merged["pip_range"] = (merged["high"] - merged["low"]) * 10000.0
    merged["return_bps"] = merged["feat_r_oc"] * 10000.0

    logger.info("Loaded Mode B (As-Of Replay): %d complete bars (%s to %s)", len(merged), merged["timestamp"].min(), merged["timestamp"].max())
    return merged


def load_full_year_data(
    ohlc_path: str = "data/processed/EURUSD_M15_2023.parquet",
    model_path: str = "models/GMM_HMM_K3_M3_v1.json",
    cutover_timestamp: str = "2023-08-24 12:00:00+00:00",
) -> pd.DataFrame:
    """
    Load Mode A (Full-Year Retrospective View) data:
    All 20,836 complete bars of 2023 evaluated under canonical GMM_HMM_K3_M3_v1
    using causal sequential streaming (RealTimeInferenceEngine + FXCalendarGapHandler).

    Guarantees:
    - Train window (first 12,501 bars): In-sample causal filtering conditional on parameter estimation.
    - Post-cutover window (subsequent 8,335 bars): Out-of-sample replay, matching as_of_replay_log exactly.
    """
    p_ohlc = Path(ohlc_path)
    if not p_ohlc.exists():
        raise FileNotFoundError(f"OHLC data file not found: {ohlc_path}")

    artifact = load_canonical_model(model_path)
    df_raw = pd.read_parquet(p_ohlc)
    
    # Strictly complete candles
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df = df_raw.sort_values("timestamp").reset_index(drop=True)

    # Engineer canonical features
    df["r_oc"] = compute_log_returns_oc(df)
    df["log_vol"] = compute_log_volatility(df)

    feature_cols = ["r_oc", "log_vol"]
    gap_handler = FXCalendarGapHandler(max_active_gap_minutes=300)
    engine = RealTimeInferenceEngine(artifact=artifact, gap_handler=gap_handler)

    X_all = df[feature_cols].values
    ts_all = df["timestamp"]
    is_comp_all = df["is_complete"].values

    records = []
    cutover_dt = pd.to_datetime(cutover_timestamp, utc=True)

    for i in range(len(df)):
        x_i = X_all[i]
        ts_i = ts_all.iloc[i]
        is_comp_i = bool(is_comp_all[i])

        step_res = engine.step(x_t=x_i, timestamp=ts_i, is_complete=is_comp_i)

        if is_comp_i:
            p0, p1, p2 = step_res["filtered_probs"]
            partition = "train_in_sample" if ts_i < cutover_dt else "replay_out_of_sample"
            records.append({
                "timestamp": ts_i,
                "open": float(df["open"].iloc[i]),
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "close": float(df["close"].iloc[i]),
                "tick_volume": float(df["tick_volume"].iloc[i]),
                "r_oc": float(x_i[0]),
                "log_vol": float(x_i[1]),
                "prob_state_0": float(p0),
                "prob_state_1": float(p1),
                "prob_state_2": float(p2),
                "pred_state": int(step_res["pred_state"]),
                "pred_log_likelihood": float(step_res["pred_log_likelihood"]),
                "gap_type": step_res["gap_type"],
                "missing_bars": int(step_res["missing_bars"]),
                "data_partition": partition,
                "model_version": artifact.model_version,
                "parameter_hash": artifact.parameter_hash,
            })

    df_full = pd.DataFrame(records)
    validate_probability_frame(df_full)

    # Compute entropy
    probs = df_full[["prob_state_0", "prob_state_1", "prob_state_2"]].values
    ent, n_ent = compute_entropy_series(probs, n_states=3)
    df_full["entropy"] = ent
    df_full["normalized_entropy"] = n_ent

    # Semantic labels
    df_full["pred_state_label"] = df_full["pred_state"].map(STATE_LABELS)

    # Physical metrics
    df_full["pip_range"] = (df_full["high"] - df_full["low"]) * 10000.0
    df_full["return_bps"] = df_full["r_oc"] * 10000.0

    n_train = int(np.sum(df_full["data_partition"] == "train_in_sample"))
    n_replay = int(np.sum(df_full["data_partition"] == "replay_out_of_sample"))

    logger.info("Loaded Mode A (Full-Year Retrospective): %d complete bars (Train: %d, Replay: %d)", len(df_full), n_train, n_replay)
    return df_full
