"""
src/inference/storage.py

Immutable probability storage module for real-time market regime detection.
Implements append-only persistence, strict duplicate rejection for live as-of
probabilities, and clear separation between live as-of streams and retrospective runs.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DuplicateRecordError(ValueError):
    """Raised when an attempt is made to insert a duplicate probability record."""
    pass


class ImmutableProbabilityStore:
    """
    Append-only storage engine for causal regime probability records.
    Ensures complete provenance and prevents accidental record overwrite.
    Maintains strict separation between:
    - 'as_of': Genuine live real-time predictions at wall-clock arrival.
    - 'as_of_replay': Historical causal replay using a frozen model fitted strictly prior to the replay period.
    - 'in_sample_filtered': In-sample causal filtering conditional on parameters estimated from the training partition.
    - 'skipped_audit': Audit records for skipped partial/incomplete candles.
    - 'retrospective': Offline, batch, or counterfactual diagnostic runs.
    """

    def __init__(self, log_file_path: Union[str, Path]):
        """
        Parameters
        ----------
        log_file_path : Union[str, Path]
            Path to the JSON Lines append-only log file.
        """
        self.log_path = Path(log_file_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory index of existing record keys for O(1) duplicate detection
        # Set of tuples: (timestamp_utc, model_version, inference_type, run_id)
        self._index: Set[Tuple[str, str, str, str]] = set()
        # Set of as-of timestamps: strictly one as-of record per timestamp
        self._as_of_timestamps: Set[str] = set()
        # Set of as-of replay timestamps: strictly one as-of replay record per timestamp
        self._as_of_replay_timestamps: Set[str] = set()
        # Set of in-sample filtered timestamps: strictly one in-sample record per timestamp
        self._in_sample_timestamps: Set[str] = set()
        # Set of skipped audit timestamps: strictly one skipped record per timestamp
        self._skipped_timestamps: Set[str] = set()

        self._load_existing_index()

    def __len__(self) -> int:
        """Return total number of unique records in store."""
        return len(self._index)

    def _load_existing_index(self) -> None:
        """Scan existing log file to populate duplicate check index."""
        if not self.log_path.exists():
            return

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    ts_norm = pd.to_datetime(rec["timestamp_utc"])
                    if ts_norm.tz is None:
                        ts_norm = ts_norm.tz_localize(timezone.utc)
                    else:
                        ts_norm = ts_norm.tz_convert(timezone.utc)
                    ts = ts_norm.isoformat()
                    mv = rec["model_version"]
                    it = rec.get("inference_type", "as_of")
                    rid = rec.get("run_id", "default")
                    self._index.add((ts, mv, it, rid))
                    if it == "as_of":
                        self._as_of_timestamps.add(ts)
                    elif it == "as_of_replay":
                        self._as_of_replay_timestamps.add(ts)
                    elif it == "in_sample_filtered":
                        self._in_sample_timestamps.add(ts)
                    elif it == "skipped_audit":
                        self._skipped_timestamps.add(ts)
                except Exception as e:
                    logger.warning(f"Error reading existing log record: {e}")

    @staticmethod
    def compute_record_id(
        timestamp_utc: str,
        model_version: str,
        inference_type: str,
        run_id: str,
    ) -> str:
        """Compute unique deterministic record hash."""
        raw = f"{timestamp_utc}::{model_version}::{inference_type}::{run_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def append(
        self,
        timestamp_utc: str,
        model_version: str,
        feature_version: str,
        filtered_probs: Optional[List[float]],
        pred_state: Optional[int],
        pred_log_likelihood: Optional[float],
        inference_type: str = "as_of",
        run_id: str = "live",
        gap_type: str = "normal",
        missing_bars: int = 0,
        feature_values: Optional[Dict[str, float]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        is_complete: bool = True,
        is_valid_prediction: bool = True,
        status: str = "valid_update",
    ) -> Dict[str, Any]:
        """
        Append a single regime probability record to the immutable log.

        Parameters
        ----------
        timestamp_utc : str
            Bar timestamp in ISO format (UTC).
        model_version : str
            Model version string.
        feature_version : str
            Feature configuration version.
        filtered_probs : Optional[List[float]]
            Filtered regime probability distribution P(S_t = j | O_{1:t}), or None if skipped.
        pred_state : Optional[int]
            Argmax discrete regime state, or None if skipped.
        pred_log_likelihood : Optional[float]
            Predictive log-likelihood ln P(x_t | O_{1:t-1}), or None if skipped.
        inference_type : str
            Category of inference record:
            - 'as_of': Live causal inference at real-time arrival.
            - 'as_of_replay': Historical causal replay with frozen prior model.
            - 'in_sample_filtered': In-sample causal filtering conditional on training parameters.
            - 'skipped_audit': Audit record for skipped incomplete candle.
            - 'retrospective': Offline counterfactual or sensitivity run.
        run_id : str
            Identifier for execution run.
        gap_type : str
            Calendar gap classification.
        missing_bars : int
            Number of missing bars prior to this bar.
        feature_values : Optional[Dict[str, float]]
            Input features observed at this bar.
        extra_metadata : Optional[Dict[str, Any]]
            Additional diagnostic metadata.
        is_complete : bool
            Whether the underlying candle was complete.
        is_valid_prediction : bool
            Whether this record constitutes a valid causal inference prediction.
        status : str
            Status string ('valid_update' or 'skipped_partial').

        Returns
        -------
        record : Dict[str, Any]
            The written record dict.

        Raises
        ------
        DuplicateRecordError
            If an identical record or duplicate as_of timestamp already exists.
        """
        # Normalize timestamp to standardized UTC ISO-8601 string
        ts_norm = pd.to_datetime(timestamp_utc)
        if ts_norm.tz is None:
            ts_norm = ts_norm.tz_localize(timezone.utc)
        else:
            ts_norm = ts_norm.tz_convert(timezone.utc)
        timestamp_utc = ts_norm.isoformat()

        # Strict Duplicate Policy:
        # 1. For as_of records: strictly ONE real-time prediction per timestamp.
        if inference_type == "as_of" and timestamp_utc in self._as_of_timestamps:
            raise DuplicateRecordError(
                f"Duplicate 'as_of' record rejected: timestamp {timestamp_utc} already recorded. "
                "Historical as-of records are immutable and cannot be overwritten."
            )

        # 2. For as_of_replay records: strictly ONE causal replay prediction per timestamp.
        if inference_type == "as_of_replay" and timestamp_utc in self._as_of_replay_timestamps:
            raise DuplicateRecordError(
                f"Duplicate 'as_of_replay' record rejected: timestamp {timestamp_utc} already recorded. "
                "Historical as-of replay records are immutable and cannot be overwritten."
            )

        # 3. For in_sample_filtered records: strictly ONE in-sample filtered record per timestamp.
        if inference_type == "in_sample_filtered" and timestamp_utc in self._in_sample_timestamps:
            raise DuplicateRecordError(
                f"Duplicate 'in_sample_filtered' record rejected: timestamp {timestamp_utc} already recorded."
            )

        # 4. For skipped audit records: strictly one per timestamp per run
        if inference_type == "skipped_audit" and timestamp_utc in self._skipped_timestamps:
            raise DuplicateRecordError(
                f"Duplicate 'skipped_audit' record rejected: timestamp {timestamp_utc} already recorded."
            )

        # 5. For retrospective and general records: reject if exact composite key exists
        composite_key = (timestamp_utc, model_version, inference_type, run_id)
        if composite_key in self._index:
            raise DuplicateRecordError(
                f"Duplicate record rejected: key {composite_key} already exists in store."
            )

        record_id = self.compute_record_id(timestamp_utc, model_version, inference_type, run_id)
        recorded_at_utc = datetime.now(timezone.utc).isoformat()

        record = {
            "record_id": record_id,
            "timestamp_utc": timestamp_utc,
            "model_version": model_version,
            "feature_version": feature_version,
            "inference_type": inference_type,
            "run_id": run_id,
            "status": status,
            "is_complete": is_complete,
            "is_valid_prediction": is_valid_prediction,
            "pred_state": pred_state,
            "pred_log_likelihood": pred_log_likelihood,
            "gap_type": gap_type,
            "missing_bars": missing_bars,
            "filtered_probs": filtered_probs,
            "feature_values": feature_values or {},
            "recorded_at_utc": recorded_at_utc,
            "extra_metadata": extra_metadata or {},
        }

        # Atomically append line to JSONL file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Update in-memory index
        self._index.add(composite_key)
        if inference_type == "as_of":
            self._as_of_timestamps.add(timestamp_utc)
        elif inference_type == "as_of_replay":
            self._as_of_replay_timestamps.add(timestamp_utc)
        elif inference_type == "in_sample_filtered":
            self._in_sample_timestamps.add(timestamp_utc)
        elif inference_type == "skipped_audit":
            self._skipped_timestamps.add(timestamp_utc)

        return record

    def to_dataframe(
        self,
        inference_type: Optional[Union[str, List[str]]] = None,
        model_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read store records into a pandas DataFrame.

        Parameters
        ----------
        inference_type : Optional[Union[str, List[str]]]
            Filter by 'as_of', 'skipped_audit', 'retrospective', or a list of types.
        model_version : Optional[str]
            Filter by model version string.

        Returns
        -------
        df : pd.DataFrame
        """
        if not self.log_path.exists():
            return pd.DataFrame()

        allowed_types: Optional[Set[str]] = None
        if inference_type is not None:
            if isinstance(inference_type, str):
                allowed_types = {inference_type}
            else:
                allowed_types = set(inference_type)

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if allowed_types and rec.get("inference_type") not in allowed_types:
                        continue
                    if model_version and rec.get("model_version") != model_version:
                        continue

                    # Flatten probabilities into columns prob_state_0, ...
                    row = {
                        "record_id": rec["record_id"],
                        "timestamp": pd.to_datetime(rec["timestamp_utc"]),
                        "timestamp_utc": rec["timestamp_utc"],
                        "model_version": rec["model_version"],
                        "feature_version": rec["feature_version"],
                        "inference_type": rec["inference_type"],
                        "run_id": rec["run_id"],
                        "status": rec.get("status", "valid_update"),
                        "is_complete": rec.get("is_complete", True),
                        "is_valid_prediction": rec.get("is_valid_prediction", True),
                        "pred_state": rec["pred_state"],
                        "pred_log_likelihood": rec["pred_log_likelihood"],
                        "gap_type": rec["gap_type"],
                        "missing_bars": rec["missing_bars"],
                        "filtered_probs": rec["filtered_probs"],
                        "recorded_at": pd.to_datetime(rec["recorded_at_utc"]),
                    }
                    probs = rec.get("filtered_probs")
                    if probs is not None:
                        for k, p in enumerate(probs):
                            row[f"prob_state_{k}"] = p
                    else:
                        for k in range(3):
                            row[f"prob_state_{k}"] = np.nan

                    if rec.get("feature_values"):
                        for feat_k, feat_v in rec["feature_values"].items():
                            row[f"feat_{feat_k}"] = feat_v
                    records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df.sort_values("timestamp").reset_index(drop=True)

    def export_parquet(
        self,
        output_path: Union[str, Path],
        inference_type: Optional[Union[str, List[str]]] = None,
    ) -> Path:
        """Export store contents to Parquet for fast analytics."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe(inference_type=inference_type)
        df.to_parquet(p, index=False)
        logger.info(f"Exported {len(df)} records to {p}")
        return p
