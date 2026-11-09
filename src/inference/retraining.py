"""
src/inference/retraining.py

Retraining management, model validation gates, warm-up execution,
and model migration tracking for real-time market regime detection.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.models.forward_filter import ForwardFilter
from src.inference.engine import RealTimeInferenceEngine
from src.inference.versioning import (
    ModelArtifact,
    ModelValidationError,
    ModelVersionMismatchError,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report detailing hard validation checks and diagnostic warnings."""

    model_version: str
    is_deployable: bool
    hard_checks_passed: bool
    hard_failures: List[str] = field(default_factory=list)
    diagnostic_warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class RetrainingValidator:
    """
    Two-tiered model validation system:
    1. Hard validation: strict mathematical, structural, and computational criteria.
       Any failure blocks deployment.
    2. Diagnostic warnings: statistical and financial checks (occupancy, switching, durations).
       Informative; does not reject model solely due to an arbitrary threshold.
    """

    def validate_candidate(
        self,
        artifact: ModelArtifact,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> ValidationReport:
        """
        Execute comprehensive validation on a candidate model artifact.

        Parameters
        ----------
        artifact : ModelArtifact
            Candidate model artifact to validate.
        val_df : Optional[pd.DataFrame]
            Optional validation dataset for empirical diagnostic checks.
        feature_cols : Optional[List[str]]
            Expected feature column names.

        Returns
        -------
        ValidationReport
        """
        hard_failures = []
        warnings = []
        diagnostics = {}

        params = artifact.parameters

        # --- TIER 1: HARD VALIDATION GATES (Must all pass) ---
        # 1. Parameter finiteness
        for key in ["startprob", "transmat", "means", "covars"]:
            if key in params:
                arr = np.asarray(params[key])
                if not np.all(np.isfinite(arr)):
                    hard_failures.append(
                        f"Non-finite values (NaN/Inf) detected in parameter '{key}'"
                    )
            else:
                hard_failures.append(f"Missing mandatory parameter matrix '{key}'")

        if "weights" in params and params["weights"] is not None:
            arr = np.asarray(params["weights"])
            if not np.all(np.isfinite(arr)):
                hard_failures.append("Non-finite values in mixture 'weights'")

        # 2. Probability constraints
        if "startprob" in params:
            sp = np.asarray(params["startprob"])
            if np.any(sp < -1e-6) or not np.isclose(np.sum(sp), 1.0, atol=1e-4):
                hard_failures.append(
                    f"Invalid initial state distribution startprob: sum={np.sum(sp):.6f}, min={np.min(sp):.6f}"
                )

        if "transmat" in params:
            tm = np.asarray(params["transmat"])
            if np.any(tm < -1e-6):
                hard_failures.append("Negative probabilities in transition matrix")
            row_sums = np.sum(tm, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-4):
                hard_failures.append(
                    f"Transition matrix rows do not sum to 1.0: {row_sums}"
                )

        if "weights" in params and params["weights"] is not None:
            w = np.asarray(params["weights"])
            if np.any(w < -1e-6):
                hard_failures.append("Negative probabilities in mixture weights")
            w_sums = np.sum(w, axis=1)
            if not np.allclose(w_sums, 1.0, atol=1e-4):
                hard_failures.append(
                    f"Mixture weights do not sum to 1.0 per state: {w_sums}"
                )

        # 3. Covariance positive-definiteness
        if "covars" in params:
            covs = np.asarray(params["covars"])
            K = artifact.n_components
            M = artifact.n_mix
            min_eigs = []
            max_kappas = []
            for k in range(K):
                for m in range(M):
                    cov_mat = covs[k, m] if covs.ndim == 4 else covs[k]
                    eigenvals = np.linalg.eigvalsh(cov_mat)
                    min_eig = float(np.min(eigenvals))
                    max_eig = float(np.max(eigenvals))
                    min_eigs.append(min_eig)
                    kappa = max_eig / max(min_eig, 1e-15)
                    max_kappas.append(kappa)
                    if min_eig <= 0:
                        hard_failures.append(
                            f"Non-positive-definite covariance matrix at state {k}, comp {m}: min_eig={min_eig:.2e}"
                        )
            diagnostics["min_covariance_eigenvalue"] = (
                float(np.min(min_eigs)) if min_eigs else 0.0
            )
            diagnostics["max_condition_number"] = (
                float(np.max(max_kappas)) if max_kappas else 0.0
            )

        # 4. Feature compatibility
        expected_feats = feature_cols or ["r_oc", "log_vol"]
        if list(artifact.features) != list(expected_feats):
            hard_failures.append(
                f"Feature names mismatch: artifact={artifact.features} vs expected={expected_feats}"
            )

        # 5. Functional step test
        try:
            test_model = artifact.to_model_instance()
            ff = ForwardFilter(test_model)
            dummy_x = np.zeros(len(artifact.features), dtype=np.float64)
            filt, prior, pll = ff.step(dummy_x)
            if not (np.isclose(np.sum(filt), 1.0, atol=1e-5) and np.isfinite(pll)):
                hard_failures.append(
                    "Functional step test produced non-normalized or non-finite posterior"
                )
        except Exception as e:
            hard_failures.append(f"Functional step test failed with exception: {e}")

        # --- TIER 2: DIAGNOSTIC WARNINGS (Do not block deployment) ---
        if diagnostics.get("max_condition_number", 0) > 1e8:
            warnings.append(
                f"High covariance condition number: {diagnostics['max_condition_number']:.2e} (> 1e8)"
            )

        if val_df is not None and feature_cols is not None:
            try:
                X_val = val_df[feature_cols].values
                val_res = ff.filter_batch(
                    X_val,
                    timestamps=val_df["timestamp"] if "timestamp" in val_df else None,
                )
                probs = val_res["filtered_probs"]
                states = np.argmax(probs, axis=1)
                soft_occ = np.mean(probs, axis=0)
                hard_occ = [np.mean(states == k) for k in range(artifact.n_components)]
                sw_freq = float(
                    np.sum(states[1:] != states[:-1]) / max(1, len(states) - 1)
                )

                diagnostics["val_min_soft_occupancy"] = float(np.min(soft_occ))
                diagnostics["val_min_hard_occupancy"] = float(np.min(hard_occ))
                diagnostics["val_switch_frequency"] = sw_freq
                diagnostics["val_predictive_ll"] = float(
                    np.sum(val_res["pred_log_likelihoods"])
                )

                if np.min(soft_occ) < 0.05:
                    warnings.append(
                        f"Low minimum soft state occupancy: {np.min(soft_occ)*100:.1f}% (< 5%)"
                    )
                if sw_freq > 0.35:
                    warnings.append(
                        f"High regime switching frequency: {sw_freq*100:.1f}% (> 35%)"
                    )
            except Exception as e:
                warnings.append(
                    f"Could not compute empirical validation diagnostics: {e}"
                )

        hard_passed = len(hard_failures) == 0
        is_deployable = hard_passed

        return ValidationReport(
            model_version=artifact.model_version,
            is_deployable=is_deployable,
            hard_checks_passed=hard_passed,
            hard_failures=hard_failures,
            diagnostic_warnings=warnings,
            diagnostics=diagnostics,
        )


class RetrainingManager:
    """
    Manages the lifecycle of model upgrades:
    - Enforces hard validation gates prior to deployment.
    - Executes warm-up buffer filtering to align initial posterior.
    - Records model migration events in deployment history log.
    """

    def __init__(
        self,
        validator: Optional[RetrainingValidator] = None,
        deployment_history_path: Optional[Union[str, Path]] = None,
    ):
        self.validator = validator or RetrainingValidator()
        self.history_path = (
            Path(deployment_history_path) if deployment_history_path else None
        )
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def validate_and_deploy(
        self,
        engine: RealTimeInferenceEngine,
        new_artifact: ModelArtifact,
        warmup_observations: np.ndarray,
        warmup_timestamps: Optional[Union[pd.Series, List]] = None,
        warmup_length: int = 50,
        warmup_is_complete: Optional[Union[np.ndarray, List[bool], pd.Series]] = None,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Validate candidate model, warm up its posterior, and atomically upgrade the engine.

        Parameters
        ----------
        engine : RealTimeInferenceEngine
            Active running inference engine.
        new_artifact : ModelArtifact
            Candidate new model artifact.
        warmup_observations : np.ndarray, shape (N, D)
            Recent historical observations for warm-up.
        warmup_timestamps : Optional[pd.Series]
            Timestamps of warm-up observations.
        warmup_length : int
            Number of recent bars W to use for warm-up (default 50).
        warmup_is_complete : Optional[Union[np.ndarray, List[bool], pd.Series]]
            Completeness flag for historical observations to enforce exclusion of partial bars.
        val_df : Optional[pd.DataFrame]
            Optional dataset for diagnostic evaluation.

        Returns
        -------
        upgrade_result : Dict[str, Any]

        Raises
        ------
        ModelValidationError
            If new_artifact fails hard validation gates.
        """
        # 1. Hard Validation Gate
        report = self.validator.validate_candidate(
            new_artifact, val_df=val_df, feature_cols=new_artifact.features
        )
        if not report.is_deployable:
            raise ModelValidationError(
                f"Candidate model {new_artifact.model_version} rejected by hard validation: {report.hard_failures}"
            )

        old_version = engine.model_version
        new_version = new_artifact.model_version

        # 2. Extract warm-up slice strictly from complete bars
        if warmup_is_complete is not None:
            comp_mask = np.asarray(warmup_is_complete, dtype=bool)
            warmup_obs_all = np.asarray(warmup_observations)[comp_mask]
            warmup_ts_all = (
                pd.Series(warmup_timestamps)[comp_mask].reset_index(drop=True)
                if warmup_timestamps is not None
                else None
            )
        else:
            warmup_obs_all = np.asarray(warmup_observations)
            warmup_ts_all = (
                pd.Series(warmup_timestamps).reset_index(drop=True)
                if warmup_timestamps is not None
                else None
            )

        W = min(warmup_length, len(warmup_obs_all))
        warmup_obs = warmup_obs_all[-W:]
        warmup_ts = warmup_ts_all.iloc[-W:] if warmup_ts_all is not None else None

        # 3. Create fresh engine instance and warm up exclusively on new model
        warmup_engine = RealTimeInferenceEngine(
            new_artifact, gap_handler=engine.gap_handler
        )
        warmup_summary = warmup_engine.warmup(warmup_obs, timestamps=warmup_ts)

        # 4. Atomic upgrade of active engine
        engine.artifact = new_artifact
        engine.model = warmup_engine.model
        engine._filter = warmup_engine._filter
        engine.current_posterior = warmup_engine.current_posterior
        engine.last_timestamp = warmup_engine.last_timestamp
        engine.last_gap_classification = warmup_engine.last_gap_classification
        engine.step_count = warmup_engine.step_count
        engine.skipped_count = warmup_engine.skipped_count

        # 5. Record deployment event
        deployment_event = {
            "deployed_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_model_version": old_version,
            "new_model_version": new_version,
            "warmup_bars": W,
            "final_warmup_posterior": warmup_summary["final_posterior"],
            "diagnostic_warnings": report.diagnostic_warnings,
            "validation_diagnostics": report.diagnostics,
        }

        if self.history_path:
            events = []
            if self.history_path.exists():
                try:
                    with open(self.history_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                except Exception:
                    events = []
            events.append(deployment_event)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2)

        logger.info(
            f"Successfully upgraded model from {old_version} to {new_version} with W={W} warm-up bars."
        )
        return deployment_event


def evaluate_warmup_sensitivity(
    model: Any,
    observations: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    buffer_lengths: Optional[List[int]] = None,
    asymptotic_length: int = 500,
) -> pd.DataFrame:
    """
    Empirically evaluate warm-up convergence sensitivity.
    Measures the L1 total variation distance of the final posterior as a function of buffer length W.

    Parameters
    ----------
    model : GaussianHMM or GMMHMM
        Model instance.
    observations : np.ndarray, shape (N, D)
        Contiguous historical observation stream.
    timestamps : Optional[pd.Series]
        Observation timestamps.
    buffer_lengths : Optional[List[int]]
        List of buffer lengths W to test (default: [5, 10, 20, 35, 50, 75, 100, 150, 200]).
    asymptotic_length : int
        Buffer length used as reference asymptotic ground-truth posterior.

    Returns
    -------
    sensitivity_df : pd.DataFrame
        Columns: W, l1_distance, max_prob_diff, argmax_match
    """
    lengths = buffer_lengths or [5, 10, 20, 35, 50, 75, 100, 150, 200]
    total_bars = len(observations)
    ref_w = min(asymptotic_length, total_bars)

    ff = ForwardFilter(model=model, gap_policy="reset_prior", max_gap_minutes=15)

    # Reference posterior (long run)
    ref_obs = observations[-ref_w:]
    ref_res = ff.filter_batch(
        ref_obs, timestamps=timestamps.iloc[-ref_w:] if timestamps is not None else None
    )
    ref_final_prob = ref_res["filtered_probs"][-1]
    ref_state = int(np.argmax(ref_final_prob))

    rows = []
    for w in lengths:
        if w > total_bars:
            continue
        sub_obs = observations[-w:]
        sub_ts = timestamps.iloc[-w:] if timestamps is not None else None
        res = ff.filter_batch(sub_obs, timestamps=sub_ts)
        w_final_prob = res["filtered_probs"][-1]
        w_state = int(np.argmax(w_final_prob))

        l1_dist = float(0.5 * np.sum(np.abs(w_final_prob - ref_final_prob)))
        max_diff = float(np.max(np.abs(w_final_prob - ref_final_prob)))
        argmax_match = bool(w_state == ref_state)

        rows.append(
            {
                "warmup_length_W": w,
                "l1_total_variation": l1_dist,
                "max_prob_diff": max_diff,
                "argmax_regime_match": argmax_match,
                "final_posterior": [round(float(p), 6) for p in w_final_prob],
            }
        )

    return pd.DataFrame(rows)
