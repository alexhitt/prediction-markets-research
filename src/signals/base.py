"""
Base Signal Framework

Provides abstract base classes and data structures for signal detection.
All signal detectors inherit from BaseSignalDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalDirection(Enum):
    """Direction of the signal's prediction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Strength classification of a signal."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass(frozen=True)
class SignalResult:
    """
    Immutable result from a signal detector.

    Attributes:
        name: Human-readable name for this signal
        source: Data source (e.g., "mcbroken.com", "google_trends")
        timestamp: When the signal was detected
        value: Numeric value of the signal (interpretation varies by signal type)
        direction: Bullish, bearish, or neutral
        confidence: 0-1 confidence in the signal
        strength: Categorical strength classification
        raw_data: Original data from the source (for debugging/auditing)
        related_markets: List of market keywords/IDs this signal relates to
        metadata: Additional context-specific data
    """
    name: str
    source: str
    timestamp: datetime
    value: float
    direction: SignalDirection
    confidence: float
    strength: SignalStrength = SignalStrength.MODERATE
    raw_data: Optional[Dict[str, Any]] = None
    related_markets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "name": self.name,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "strength": self.strength.value,
            "raw_data": self.raw_data,
            "related_markets": self.related_markets,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalResult":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=data["value"],
            direction=SignalDirection(data["direction"]),
            confidence=data["confidence"],
            strength=SignalStrength(data.get("strength", "moderate")),
            raw_data=data.get("raw_data"),
            related_markets=data.get("related_markets", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class SignalCollectionStats:
    """Statistics from a signal collection run."""
    source: str
    timestamp: datetime
    signals_collected: int
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class BaseSignalDetector(ABC):
    """
    Abstract base class for signal detectors.

    All signal detectors must implement:
    - fetch_data(): Retrieve raw data from the source
    - process_data(): Transform raw data into SignalResult objects
    - run(): Convenience method to fetch and process in one call

    Usage:
        class MySignal(BaseSignalDetector):
            def fetch_data(self):
                return requests.get(...).json()

            def process_data(self, raw_data):
                return [SignalResult(...)]

        detector = MySignal()
        signals = detector.run()
    """

    def __init__(self, name: str, source: str):
        """
        Initialize the detector.

        Args:
            name: Human-readable name for this detector
            source: Data source identifier
        """
        self.name = name
        self.source = source
        self._last_fetch: Optional[datetime] = None
        self._last_data: Optional[Any] = None

    @abstractmethod
    def fetch_data(self) -> Any:
        """
        Fetch raw data from the signal source.

        Returns:
            Raw data from the source (format depends on implementation)

        Raises:
            Exception: If data fetch fails
        """
        pass

    @abstractmethod
    def process_data(self, raw_data: Any) -> List[SignalResult]:
        """
        Process raw data into signal results.

        Args:
            raw_data: Data returned from fetch_data()

        Returns:
            List of SignalResult objects
        """
        pass

    def run(self) -> List[SignalResult]:
        """
        Fetch and process data in one call.

        Returns:
            List of SignalResult objects

        Raises:
            Exception: If fetch or processing fails
        """
        raw_data = self.fetch_data()
        self._last_fetch = datetime.utcnow()
        self._last_data = raw_data
        return self.process_data(raw_data)

    def get_last_fetch_time(self) -> Optional[datetime]:
        """Get the timestamp of the last data fetch."""
        return self._last_fetch

    def get_cached_data(self) -> Optional[Any]:
        """Get the last fetched raw data."""
        return self._last_data


def classify_signal_strength(value: float, thresholds: Dict[str, float]) -> SignalStrength:
    """
    Classify a signal value into strength categories.

    Args:
        value: The signal value to classify
        thresholds: Dict with keys 'weak', 'moderate', 'strong', 'very_strong'
                   Values are the minimum thresholds for each category

    Returns:
        SignalStrength enum value
    """
    if value >= thresholds.get("very_strong", 0.9):
        return SignalStrength.VERY_STRONG
    elif value >= thresholds.get("strong", 0.7):
        return SignalStrength.STRONG
    elif value >= thresholds.get("moderate", 0.4):
        return SignalStrength.MODERATE
    else:
        return SignalStrength.WEAK


def determine_direction(value: float, baseline: float, threshold: float = 0.05) -> SignalDirection:
    """
    Determine signal direction based on deviation from baseline.

    Args:
        value: Current value
        baseline: Expected/normal value
        threshold: Minimum deviation to be non-neutral

    Returns:
        SignalDirection enum value
    """
    deviation = (value - baseline) / baseline if baseline != 0 else 0

    if deviation > threshold:
        return SignalDirection.BULLISH
    elif deviation < -threshold:
        return SignalDirection.BEARISH
    else:
        return SignalDirection.NEUTRAL
