"""
src/inference/versioning.py

Model artifact schema, versioning, and validation module.
Guarantees full provenance, reproducible serialization, and prevents
cross-model posterior contamination.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.models.gaussian_hmm import GaussianHMM
from src.models.gmm_hmm import GMMHMM

logger = logging.getLogger(__name__)


class ModelVersionMismatchError(ValueError):
    """Raised when an operation attempts to combine incompatible model versions or posteriors."""
    pass


class ModelValidationError(ValueError):
    """Raised when a candidate model fails hard validation criteria."""
    pass


@dataclass
class ModelArtifact:
    """
    Complete immutable model artifact bundle containing architecture,
    parameters, training provenance, and diagnostic metrics.
    """
    model_version: str
    model_type: str             # 'GaussianHMM' or 'GMMHMM'
    n_components: int          # K
    n_mix: int                 # M (1 for Gaussian)
    features: List[str]        # e.g. ['r_oc', 'log_vol']
    feature_version: str       # e.g. 'Config_2_MinimalOrthogonal_v1'
    created_at_utc: str
    training_metadata: Dict[str, Any]
    validation_diagnostics: Dict[str, Any]
    parameters: Dict[str, Any] # startprob, transmat, weights, means, covars
    parameter_hash: str = ""

    def __post_init__(self):
        if not self.parameter_hash:
            self.parameter_hash = self.compute_parameter_hash()

    def compute_parameter_hash(self) -> str:
        """Compute SHA256 checksum over model numerical parameters."""
        serialized = json.dumps(self.parameters, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def validate_features(self, expected_features: List[str], expected_feature_version: Optional[str] = None) -> None:
        """Ensure observation features match model expectation exactly."""
        if list(self.features) != list(expected_features):
            raise ModelVersionMismatchError(
                f"Feature mismatch: model expects {self.features}, got {expected_features}"
            )
        if expected_feature_version is not None and self.feature_version != expected_feature_version:
            raise ModelVersionMismatchError(
                f"Feature version mismatch: model expects {self.feature_version}, got {expected_feature_version}"
            )

    def to_model_instance(self) -> Union[GaussianHMM, GMMHMM]:
        """Instantiate runnable model instance from serialized parameters."""
        if self.model_type == "GaussianHMM":
            return GaussianHMM.from_dict(self.parameters)
        elif self.model_type == "GMMHMM":
            return GMMHMM.from_dict(self.parameters)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary."""
        return asdict(self)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save artifact to JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved model artifact {self.model_version} to {p}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelArtifact":
        """Load artifact from dictionary."""
        return cls(**data)

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "ModelArtifact":
        """Load artifact from JSON file."""
        p = Path(file_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_model(
        cls,
        model: Union[GaussianHMM, GMMHMM],
        model_version: str,
        features: List[str],
        feature_version: str,
        training_metadata: Optional[Dict[str, Any]] = None,
        validation_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "ModelArtifact":
        """Create a ModelArtifact from an in-memory trained model instance."""
        is_gmm = hasattr(model, "weights_") and model.weights_ is not None
        model_type = "GMMHMM" if is_gmm else "GaussianHMM"
        n_mix = getattr(model, "n_mix", 1) if is_gmm else 1

        parameters = model.to_dict()

        return cls(
            model_version=model_version,
            model_type=model_type,
            n_components=model.n_components,
            n_mix=n_mix,
            features=list(features),
            feature_version=feature_version,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            training_metadata=training_metadata or {},
            validation_diagnostics=validation_diagnostics or {},
            parameters=parameters,
        )

