"""
src/inference/engine.py

Real-time sequential inference engine for market regime detection on EURUSD M15.
Estimates causal filtered regime probabilities P(S_t = j | O_{1:t}) one bar at a time.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.models.forward_filter import ForwardFilter, _fast_logsumexp_1d, _fast_logsumexp_axis0
from src.inference.calendar import FXCalendarGapHandler, GapClassification, UncertainGapError
from src.inference.versioning import ModelArtifact, ModelVersionMismatchError

logger = logging.getLogger(__name__)


class RealTimeInferenceEngine:
    """
    Stateful real-time inference engine for online M15 filtered regime estimation.
    Strictly causal, supports both Gaussian HMM and GMM-HMM models.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        gap_handler: Optional[FXCalendarGapHandler] = None,
    ):
        """
        Parameters
        ----------
        artifact : ModelArtifact
            Validated model artifact bundle.
        gap_handler : Optional[FXCalendarGapHandler]
            FX market session calendar gap evaluator.
        """
        self.artifact = artifact
        self.model = artifact.to_model_instance()
        self.gap_handler = gap_handler or FXCalendarGapHandler()

        # Low-level log-space forward filter
        self._filter = ForwardFilter(model=self.model, gap_policy="reset_prior", max_gap_minutes=15)

        # Engine state variables
        self.current_posterior: Optional[np.ndarray] = None
        self.last_timestamp: Optional[pd.Timestamp] = None
        self.last_gap_classification: Optional[GapClassification] = None
        self.step_count: int = 0
        self.skipped_count: int = 0

    @property
    def model_version(self) -> str:
        return self.artifact.model_version

    @property
    def K(self) -> int:
        return self.artifact.n_components

    def reset_state(self) -> None:
        """Clear active belief state."""
        self.current_posterior = None
        self.last_timestamp = None
        self.last_gap_classification = None
        self.step_count = 0
        self.skipped_count = 0

    def step(
        self,
        x_t: Union[np.ndarray, List[float]],
        timestamp: Union[pd.Timestamp, str],
        is_complete: bool = True,
        strict_gaps: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest a single new M15 bar observation and compute updated regime probabilities.

        Parameters
        ----------
        x_t : np.ndarray or list, shape (D,)
            Observation feature vector (e.g. [r_oc, log_vol]).
        timestamp : pd.Timestamp or ISO string
            Timestamp of the M15 candle (UTC).
        is_complete : bool
            Whether the candle is a complete M15 bar. If False, the candle is treated
            as an incomplete/partial bar: no emission likelihood is computed, the posterior
            belief state is not updated, and last_timestamp is NOT advanced. This ensures
            the subsequent complete candle correctly propagates transitions A^n across the
            unobserved interval.
        strict_gaps : bool
            If True, raises UncertainGapError on ambiguous gaps.

        Returns
        -------
        result : Dict[str, Any]
            Complete inference payload with probabilities, predictive LL, and gap info.
        """
        # 1. Parse and validate input
        x_arr = np.asarray(x_t, dtype=np.float64).ravel()
        if len(x_arr) != len(self.artifact.features):
            raise ValueError(
                f"Observation dimension mismatch: expected {len(self.artifact.features)} features, got {len(x_arr)}"
            )
        if not np.all(np.isfinite(x_arr)):
            raise ValueError(f"Non-finite values detected in observation vector: {x_arr}")

        ts_curr = pd.to_datetime(timestamp)
        if ts_curr.tz is None:
            ts_curr = ts_curr.tz_localize(timezone.utc)
        else:
            ts_curr = ts_curr.tz_convert(timezone.utc)

        # 1b. Partial bar guardrail: skip emission update and preserve state
        if not is_complete:
            self.skipped_count += 1
            return {
                "timestamp_utc": ts_curr.isoformat(),
                "model_version": self.model_version,
                "feature_version": self.artifact.feature_version,
                "filtered_probs": None,
                "prior_probs": None,
                "pred_state": None,
                "pred_log_likelihood": None,
                "gap_type": "partial_bar_skipped",
                "n_transitions": 0,
                "missing_bars": 0,
                "gap_description": "Incomplete M15 candle skipped (no emission measurement update)",
                "step_index": self.step_count,
                "skipped_count": self.skipped_count,
                "is_complete": False,
                "is_valid_prediction": False,
                "status": "skipped_partial",
            }

        # 2. Determine prior state distribution based on gap calendar classification
        if self.current_posterior is None:
            # Initial step of sequence
            log_prior = self._filter.log_startprob.copy()
            gap_info = GapClassification(
                gap_type="initial",
                n_transitions=0,
                missing_bars=0,
                description="Initial sequence start",
                requires_reset=True,
            )
        else:
            gap_info = self.gap_handler.classify(
                t_prev=self.last_timestamp,
                t_curr=ts_curr,
                strict=strict_gaps,
            )

            if gap_info.requires_reset or gap_info.gap_type == "market_closure":
                # Market closure or reset: intraday volatility memory decays
                log_prior = self._filter.log_startprob.copy()
            elif gap_info.gap_type == "normal":
                # Normal consecutive bar: 1 transition through A
                log_prev = np.log(np.maximum(self.current_posterior, 1e-15))
                log_prior = _fast_logsumexp_axis0(log_prev[:, None] + self._filter.log_transmat)
            elif gap_info.gap_type == "active_missing":
                # Active session missing bars: exactly n = missing_bars + 1 transitions through A
                n_steps = gap_info.n_transitions
                A_pow = np.linalg.matrix_power(self.model.transmat_, n_steps)
                prior = self.current_posterior @ A_pow
                prior = np.maximum(prior, 1e-15)
                log_prior = np.log(prior / np.sum(prior))
            else:
                # Uncertain gap in non-strict mode
                log_prior = self._filter.log_startprob.copy()

        # 3. Compute observation emission likelihood in log-space
        log_b = self._filter.compute_single_log_emission(x_arr)

        # 4. Sequential predictive log-likelihood: ln P(x_t | O_{1:t-1})
        pred_ll = float(_fast_logsumexp_1d(log_prior + log_b))

        # 5. Bayes measurement update in log-space
        log_posterior = log_prior + log_b - pred_ll
        posterior = np.exp(log_posterior)
        posterior = posterior / np.sum(posterior)  # Exact unit normalization

        prior_probs = np.exp(log_prior)
        prior_probs = prior_probs / np.sum(prior_probs)

        # 6. Update internal state
        self.current_posterior = posterior.copy()
        self.last_timestamp = ts_curr
        self.last_gap_classification = gap_info
        self.step_count += 1

        pred_state = int(np.argmax(posterior))

        return {
            "timestamp_utc": ts_curr.isoformat(),
            "model_version": self.model_version,
            "feature_version": self.artifact.feature_version,
            "filtered_probs": posterior.tolist(),
            "prior_probs": prior_probs.tolist(),
            "pred_state": pred_state,
            "pred_log_likelihood": pred_ll,
            "gap_type": gap_info.gap_type,
            "n_transitions": gap_info.n_transitions,
            "missing_bars": gap_info.missing_bars,
            "gap_description": gap_info.description,
            "step_index": self.step_count,
            "skipped_count": self.skipped_count,
            "is_complete": True,
            "is_valid_prediction": True,
            "status": "valid_update",
        }

    def warmup(
        self,
        observations: np.ndarray,
        timestamps: Optional[Union[pd.Series, List]] = None,
        is_complete: Optional[Union[np.ndarray, List[bool], pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Execute sequential warm-up across an observation buffer using this model.

        Parameters
        ----------
        observations : np.ndarray, shape (W, D)
            Buffer of W consecutive M15 feature vectors.
        timestamps : Optional list or Series of length W
            Timestamps corresponding to observations.
        is_complete : Optional list, Series, or array of length W
            Completeness flags for each bar.

        Returns
        -------
        warmup_summary : Dict[str, Any]
        """
        obs_arr = np.asarray(observations, dtype=np.float64)
        W, D = obs_arr.shape
        if D != len(self.artifact.features):
            raise ValueError(f"Warmup feature dim {D} != model features {len(self.artifact.features)}")

        self.reset_state()
        ts_list = (
            pd.to_datetime(timestamps).tolist()
            if timestamps is not None
            else [pd.Timestamp("2023-01-01 00:00:00", tz="UTC") + pd.Timedelta(minutes=15 * i) for i in range(W)]
        )
        comp_list = (
            [bool(c) for c in is_complete]
            if is_complete is not None
            else [True] * W
        )

        for i in range(W):
            self.step(obs_arr[i], ts_list[i], is_complete=comp_list[i], strict_gaps=False)

        return {
            "model_version": self.model_version,
            "warmup_bars": W,
            "valid_warmup_steps": self.step_count,
            "skipped_warmup_bars": self.skipped_count,
            "final_posterior": self.current_posterior.tolist() if self.current_posterior is not None else [],
            "last_timestamp_utc": self.last_timestamp.isoformat() if self.last_timestamp else None,
        }

    def save_state(self, file_path: Union[str, Path]) -> None:
        """Persist engine state to a JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state_data = {
            "model_version": self.model_version,
            "parameter_hash": self.artifact.parameter_hash,
            "last_timestamp_utc": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "current_posterior": self.current_posterior.tolist() if self.current_posterior is not None else None,
            "step_count": self.step_count,
            "skipped_count": self.skipped_count,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
        logger.info(f"Saved inference engine state ({self.model_version}) to {p}")

    def load_state(self, file_path: Union[str, Path]) -> None:
        """
        Restore engine state from JSON file with strict version guardrail check.

        Raises
        ------
        ModelVersionMismatchError
            If state file belongs to a different model version or parameter hash.
        """
        p = Path(file_path)
        with open(p, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        if state_data["model_version"] != self.model_version:
            raise ModelVersionMismatchError(
                f"State file model_version '{state_data['model_version']}' does not match active engine '{self.model_version}'"
            )
        if state_data.get("parameter_hash") and state_data["parameter_hash"] != self.artifact.parameter_hash:
            raise ModelVersionMismatchError(
                f"State parameter hash mismatch: state={state_data['parameter_hash']} vs model={self.artifact.parameter_hash}"
            )

        if state_data["current_posterior"] is not None:
            self.current_posterior = np.array(state_data["current_posterior"], dtype=np.float64)
        else:
            self.current_posterior = None

        if state_data["last_timestamp_utc"] is not None:
            self.last_timestamp = pd.to_datetime(state_data["last_timestamp_utc"])
        else:
            self.last_timestamp = None

        self.step_count = state_data.get("step_count", 0)
        self.skipped_count = state_data.get("skipped_count", 0)
        logger.info(f"Restored inference engine state from {p} at step {self.step_count}")

