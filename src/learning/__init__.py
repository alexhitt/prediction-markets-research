"""
Learning system module for bot improvement.

Provides adaptive learning based on real market outcomes:
- ResolutionWorker: Background job to check resolutions and trigger learning
- CalibrationTracker: Track Brier scores and confidence adjustments
- DomainSpecialist: Per-category performance tracking
- IncrementalUpdater: Signal weight optimization
"""

from .calibration_tracker import CalibrationTracker
from .domain_specialist import DomainSpecialist
from .incremental_updater import IncrementalUpdater
from .resolution_worker import ResolutionWorker

__all__ = [
    "CalibrationTracker",
    "DomainSpecialist",
    "IncrementalUpdater",
    "ResolutionWorker",
]
