"""
Calibration Tracker for Brier score tracking and confidence adjustment.

Tracks predictions by confidence bucket and adjusts confidence multipliers
to improve calibration over time.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import ResearchEstimate, BotMemory


@dataclass
class CalibrationBucket:
    """Statistics for a confidence bucket."""
    bucket_name: str  # e.g., "0.5-0.6"
    lower_bound: float
    upper_bound: float
    count: int
    sum_predicted: float
    sum_actual: float
    sum_brier: float

    @property
    def mean_predicted(self) -> float:
        """Mean predicted probability in this bucket."""
        if self.count == 0:
            return (self.lower_bound + self.upper_bound) / 2
        return self.sum_predicted / self.count

    @property
    def mean_actual(self) -> float:
        """Mean actual outcome rate in this bucket."""
        if self.count == 0:
            return (self.lower_bound + self.upper_bound) / 2
        return self.sum_actual / self.count

    @property
    def mean_brier(self) -> float:
        """Mean Brier score in this bucket."""
        if self.count == 0:
            return 0.25  # Baseline for no predictions
        return self.sum_brier / self.count

    @property
    def calibration_error(self) -> float:
        """
        Difference between predicted and actual.

        Positive = overconfident (predicted > actual)
        Negative = underconfident (predicted < actual)
        """
        return self.mean_predicted - self.mean_actual

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "bucket_name": self.bucket_name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "sum_predicted": self.sum_predicted,
            "sum_actual": self.sum_actual,
            "sum_brier": self.sum_brier,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "mean_brier": self.mean_brier,
            "calibration_error": self.calibration_error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationBucket":
        """Create from dictionary."""
        return cls(
            bucket_name=data["bucket_name"],
            lower_bound=data["lower_bound"],
            upper_bound=data["upper_bound"],
            count=data.get("count", 0),
            sum_predicted=data.get("sum_predicted", 0.0),
            sum_actual=data.get("sum_actual", 0.0),
            sum_brier=data.get("sum_brier", 0.0)
        )


class CalibrationTracker:
    """
    Tracks prediction calibration and adjusts confidence multipliers.

    Superforecaster insight: 50% of accuracy gain comes from noise reduction
    and proper confidence calibration.
    """

    # Default bucket boundaries
    DEFAULT_BUCKETS = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
    ]

    def __init__(self, db_session: Session):
        """
        Initialize the calibration tracker.

        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session

    def get_or_create_buckets(self, bot_id: str) -> Dict[str, CalibrationBucket]:
        """
        Get or create calibration buckets for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Dictionary mapping bucket names to CalibrationBucket objects
        """
        memory = self._get_or_create_memory(bot_id)

        buckets = {}
        stored_buckets = memory.calibration_buckets or {}

        for lower, upper in self.DEFAULT_BUCKETS:
            bucket_name = f"{lower:.1f}-{upper:.1f}"

            if bucket_name in stored_buckets:
                buckets[bucket_name] = CalibrationBucket.from_dict(stored_buckets[bucket_name])
            else:
                buckets[bucket_name] = CalibrationBucket(
                    bucket_name=bucket_name,
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    sum_predicted=0.0,
                    sum_actual=0.0,
                    sum_brier=0.0
                )

        return buckets

    def update_calibration(
        self,
        bot_id: str,
        predicted_probability: float,
        actual_outcome: float,
        brier_score: float
    ) -> CalibrationBucket:
        """
        Update calibration data with a new resolved prediction.

        Args:
            bot_id: Bot identifier
            predicted_probability: The bot's predicted probability (0-1)
            actual_outcome: The actual outcome (1.0 for yes, 0.0 for no)
            brier_score: The calculated Brier score

        Returns:
            The updated CalibrationBucket
        """
        memory = self._get_or_create_memory(bot_id)
        buckets = self.get_or_create_buckets(bot_id)

        # Find the appropriate bucket
        bucket_name = self._get_bucket_name(predicted_probability)
        bucket = buckets[bucket_name]

        # Update bucket statistics (immutable pattern)
        updated_bucket = CalibrationBucket(
            bucket_name=bucket.bucket_name,
            lower_bound=bucket.lower_bound,
            upper_bound=bucket.upper_bound,
            count=bucket.count + 1,
            sum_predicted=bucket.sum_predicted + predicted_probability,
            sum_actual=bucket.sum_actual + actual_outcome,
            sum_brier=bucket.sum_brier + brier_score
        )

        # Update stored buckets
        stored_buckets = memory.calibration_buckets or {}
        stored_buckets[bucket_name] = updated_bucket.to_dict()
        memory.calibration_buckets = stored_buckets

        # Recalculate confidence multiplier
        new_multiplier = self._calculate_confidence_multiplier(buckets)
        memory.confidence_multiplier = new_multiplier

        self.db.commit()

        logger.debug(
            f"Bot {bot_id} calibration updated: bucket={bucket_name}, "
            f"error={updated_bucket.calibration_error:+.3f}, "
            f"multiplier={new_multiplier:.3f}"
        )

        return updated_bucket

    def get_calibration_summary(self, bot_id: str) -> Dict[str, Any]:
        """
        Get a summary of calibration performance.

        Args:
            bot_id: Bot identifier

        Returns:
            Dictionary with calibration statistics
        """
        memory = self._get_or_create_memory(bot_id)
        buckets = self.get_or_create_buckets(bot_id)

        total_count = sum(b.count for b in buckets.values())
        total_brier = sum(b.sum_brier for b in buckets.values())

        # Calculate weighted average calibration error
        weighted_error = 0.0
        if total_count > 0:
            for bucket in buckets.values():
                if bucket.count > 0:
                    weight = bucket.count / total_count
                    weighted_error += abs(bucket.calibration_error) * weight

        return {
            "bot_id": bot_id,
            "total_predictions": total_count,
            "average_brier_score": total_brier / total_count if total_count > 0 else None,
            "confidence_multiplier": memory.confidence_multiplier,
            "weighted_calibration_error": weighted_error,
            "buckets": {name: bucket.to_dict() for name, bucket in buckets.items()},
            "is_overconfident": weighted_error > 0.05,
            "is_underconfident": weighted_error < -0.05,
            "calibration_quality": self._assess_calibration_quality(weighted_error)
        }

    def get_adjusted_confidence(self, bot_id: str, raw_confidence: float) -> float:
        """
        Adjust a confidence score based on historical calibration.

        Args:
            bot_id: Bot identifier
            raw_confidence: The bot's raw confidence score

        Returns:
            Adjusted confidence score
        """
        memory = self._get_or_create_memory(bot_id)
        multiplier = memory.confidence_multiplier or 1.0

        adjusted = raw_confidence * multiplier

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted))

    def _get_or_create_memory(self, bot_id: str) -> BotMemory:
        """Get or create BotMemory record."""
        memory = self.db.query(BotMemory).filter_by(bot_id=bot_id).first()

        if not memory:
            memory = BotMemory(
                bot_id=bot_id,
                signal_weights={},
                calibration_buckets={},
                domain_performance={},
                confidence_multiplier=1.0,
                source_scores={},
                total_predictions=0,
                resolved_predictions=0,
                created_at=datetime.utcnow()
            )
            self.db.add(memory)
            self.db.commit()

        return memory

    def _get_bucket_name(self, probability: float) -> str:
        """Get the bucket name for a probability."""
        for lower, upper in self.DEFAULT_BUCKETS:
            if lower <= probability < upper:
                return f"{lower:.1f}-{upper:.1f}"
        # Handle edge case of exactly 1.0
        return "0.9-1.0"

    def _calculate_confidence_multiplier(
        self,
        buckets: Dict[str, CalibrationBucket]
    ) -> float:
        """
        Calculate the confidence multiplier based on calibration data.

        If bot is overconfident (predicts higher than actual outcomes),
        multiplier < 1.0 to reduce future confidence.
        If bot is underconfident, multiplier > 1.0.
        """
        total_count = sum(b.count for b in buckets.values())

        if total_count < 10:
            # Not enough data to adjust
            return 1.0

        # Calculate weighted average calibration error
        weighted_error = 0.0
        for bucket in buckets.values():
            if bucket.count > 0:
                weight = bucket.count / total_count
                weighted_error += bucket.calibration_error * weight

        # Convert error to multiplier
        # Positive error (overconfident) -> multiplier < 1
        # Negative error (underconfident) -> multiplier > 1
        # Cap adjustment at +/- 30%
        adjustment = -weighted_error * 2  # Scale factor
        multiplier = 1.0 + max(-0.3, min(0.3, adjustment))

        return multiplier

    def _assess_calibration_quality(self, weighted_error: float) -> str:
        """Assess overall calibration quality."""
        abs_error = abs(weighted_error)

        if abs_error < 0.02:
            return "excellent"
        elif abs_error < 0.05:
            return "good"
        elif abs_error < 0.10:
            return "moderate"
        elif abs_error < 0.15:
            return "poor"
        else:
            return "very_poor"
