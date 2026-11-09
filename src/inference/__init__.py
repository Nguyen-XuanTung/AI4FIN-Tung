"""
src/inference/__init__.py

Phase 5: Real-time sequential inference, calendar-aware gap handling,
model versioning, append-only immutable storage, and retraining management.
"""

from src.inference.calendar import (
    FXCalendarGapHandler,
    GapClassification,
    UncertainGapError,
    is_fx_holiday,
)
from src.inference.versioning import (
    ModelArtifact,
    ModelValidationError,
    ModelVersionMismatchError,
)
from src.inference.engine import RealTimeInferenceEngine
from src.inference.storage import (
    DuplicateRecordError,
    ImmutableProbabilityStore,
)
from src.inference.retraining import (
    RetrainingManager,
    RetrainingValidator,
    ValidationReport,
    evaluate_warmup_sensitivity,
)

__all__ = [
    "FXCalendarGapHandler",
    "GapClassification",
    "UncertainGapError",
    "is_fx_holiday",
    "ModelArtifact",
    "ModelValidationError",
    "ModelVersionMismatchError",
    "RealTimeInferenceEngine",
    "DuplicateRecordError",
    "ImmutableProbabilityStore",
    "RetrainingManager",
    "RetrainingValidator",
    "ValidationReport",
    "evaluate_warmup_sensitivity",
]

