"""
Signal Collector

Scheduled collection of alternative data signals from multiple sources.
Integrates with the existing market collector for comprehensive data gathering.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Type

from loguru import logger

from src.signals.base import (
    BaseSignalDetector,
    SignalResult,
    SignalCollectionStats
)
from src.signals.ice_cream import IceCreamSignalDetector
from src.signals.google_trends import GoogleTrendsSignalDetector
from src.database.db import db_manager, get_session


class SignalCollector:
    """
    Orchestrates signal collection from multiple alternative data sources.

    Integrates with existing market collector for comprehensive data gathering.
    Stores signals to database for historical analysis and hypothesis tracking.

    Usage:
        collector = SignalCollector()

        # Collect once
        stats = collector.collect_all()

        # Run continuously
        collector.run_continuous(interval_seconds=300)
    """

    def __init__(
        self,
        detectors: Optional[List[BaseSignalDetector]] = None,
        trends_keywords: Optional[List[str]] = None
    ):
        """
        Initialize the signal collector.

        Args:
            detectors: List of signal detectors to use (defaults to all available)
            trends_keywords: Keywords for Google Trends detector
        """
        if detectors:
            self.detectors = detectors
        else:
            # Initialize default detectors
            self.detectors = [
                IceCreamSignalDetector(),
            ]

            # Add Google Trends if keywords provided or use defaults
            try:
                trends_detector = GoogleTrendsSignalDetector(
                    keywords=trends_keywords or ["bitcoin", "election", "recession"]
                )
                self.detectors.append(trends_detector)
            except Exception as e:
                logger.warning(f"Could not initialize Google Trends detector: {e}")

    def collect_all(self) -> Dict:
        """
        Collect signals from all configured detectors.

        Returns:
            Dict with collection statistics
        """
        stats = {
            "timestamp": datetime.utcnow(),
            "total_signals": 0,
            "detectors": {},
            "errors": []
        }

        for detector in self.detectors:
            try:
                detector_stats = self._collect_from_detector(detector)
                stats["detectors"][detector.name] = detector_stats
                stats["total_signals"] += detector_stats["signals_collected"]

            except Exception as e:
                error_msg = f"{detector.name}: {str(e)}"
                logger.error(f"Signal collection error - {error_msg}")
                stats["errors"].append(error_msg)
                stats["detectors"][detector.name] = {
                    "signals_collected": 0,
                    "error": str(e)
                }

        logger.info(
            f"Signal collection complete: {stats['total_signals']} signals "
            f"from {len(self.detectors)} sources"
        )
        return stats

    def _collect_from_detector(self, detector: BaseSignalDetector) -> Dict:
        """
        Collect signals from a single detector and store to database.

        Args:
            detector: Signal detector instance

        Returns:
            Dict with collection stats for this detector
        """
        logger.info(f"Collecting signals from {detector.name}...")
        start_time = time.time()

        signals = detector.run()
        signals_stored = 0

        with get_session() as session:
            for signal in signals:
                try:
                    self._store_signal(session, signal)
                    signals_stored += 1
                except Exception as e:
                    logger.warning(f"Error storing signal: {e}")

        duration = time.time() - start_time
        logger.info(
            f"{detector.name}: {signals_stored} signals stored in {duration:.1f}s"
        )

        return {
            "signals_collected": signals_stored,
            "duration_seconds": duration,
            "source": detector.source
        }

    def _store_signal(self, session, signal: SignalResult) -> None:
        """
        Store a signal result to the database.

        Args:
            session: Database session
            signal: SignalResult to store
        """
        db_manager.add_signal(
            session,
            name=signal.name,
            source=signal.source,
            timestamp=signal.timestamp,
            value=signal.value,
            raw_data={
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "strength": signal.strength.value,
                "raw_data": signal.raw_data,
                "related_markets": signal.related_markets,
                "metadata": signal.metadata
            },
            prediction_direction=signal.direction.value,
            prediction_confidence=signal.confidence
        )

    def collect_ice_cream(self) -> Dict:
        """Collect only ice cream machine signals."""
        detector = IceCreamSignalDetector()
        return {
            "timestamp": datetime.utcnow(),
            "ice_cream": self._collect_from_detector(detector)
        }

    def collect_trends(self, keywords: Optional[List[str]] = None) -> Dict:
        """
        Collect only Google Trends signals.

        Args:
            keywords: Keywords to track (max 5)
        """
        detector = GoogleTrendsSignalDetector(keywords=keywords)
        return {
            "timestamp": datetime.utcnow(),
            "google_trends": self._collect_from_detector(detector)
        }

    def run_continuous(
        self,
        interval_seconds: int = 300,
        ice_cream_interval: int = 300,
        trends_interval: int = 3600
    ):
        """
        Run signal collection continuously.

        Different intervals for different sources to avoid rate limiting.

        Args:
            interval_seconds: Base interval between collection runs
            ice_cream_interval: Interval for ice cream data (default: 5 min)
            trends_interval: Interval for Google Trends (default: 1 hour)
        """
        logger.info(
            f"Starting continuous signal collection "
            f"(ice cream: {ice_cream_interval}s, trends: {trends_interval}s)"
        )

        last_ice_cream = 0
        last_trends = 0

        while True:
            try:
                current_time = time.time()
                signals_collected = 0

                # Collect ice cream signals
                if current_time - last_ice_cream >= ice_cream_interval:
                    try:
                        stats = self.collect_ice_cream()
                        signals_collected += stats.get("ice_cream", {}).get(
                            "signals_collected", 0
                        )
                        last_ice_cream = current_time
                    except Exception as e:
                        logger.error(f"Ice cream collection error: {e}")

                # Collect Google Trends signals (less frequently)
                if current_time - last_trends >= trends_interval:
                    try:
                        stats = self.collect_trends()
                        signals_collected += stats.get("google_trends", {}).get(
                            "signals_collected", 0
                        )
                        last_trends = current_time
                    except Exception as e:
                        logger.error(f"Google Trends collection error: {e}")

                if signals_collected > 0:
                    logger.info(f"Collected {signals_collected} signals")

                # Sleep until next check
                time.sleep(min(interval_seconds, ice_cream_interval))

            except KeyboardInterrupt:
                logger.info("Signal collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Signal collection error: {e}")
                time.sleep(interval_seconds)

    def get_recent_signals(
        self,
        source: Optional[str] = None,
        limit: int = 50
    ) -> List[SignalResult]:
        """
        Get recent signals from database.

        Args:
            source: Filter by source (optional)
            limit: Maximum signals to return

        Returns:
            List of SignalResult objects
        """
        from src.database.models import Signal

        with get_session() as session:
            query = session.query(Signal).order_by(Signal.timestamp.desc())

            if source:
                query = query.filter(Signal.source == source)

            db_signals = query.limit(limit).all()

            return [
                self._db_signal_to_result(s) for s in db_signals
            ]

    def _db_signal_to_result(self, db_signal) -> SignalResult:
        """Convert database signal to SignalResult."""
        from src.signals.base import SignalDirection, SignalStrength

        raw_data = db_signal.raw_data or {}

        return SignalResult(
            name=db_signal.name,
            source=db_signal.source,
            timestamp=db_signal.timestamp,
            value=db_signal.value,
            direction=SignalDirection(
                raw_data.get("direction", "neutral")
            ),
            confidence=raw_data.get("confidence", 0.5),
            strength=SignalStrength(
                raw_data.get("strength", "moderate")
            ),
            raw_data=raw_data.get("raw_data"),
            related_markets=raw_data.get("related_markets", []),
            metadata=raw_data.get("metadata", {})
        )


def run_signal_collector():
    """Entry point for running the signal collector."""
    from src.database.db import init_db

    # Initialize database
    init_db()

    # Create and run collector
    collector = SignalCollector()
    collector.run_continuous(
        interval_seconds=60,
        ice_cream_interval=300,  # 5 minutes
        trends_interval=3600     # 1 hour
    )


if __name__ == "__main__":
    # Quick test
    print("Testing Signal Collector")
    print("-" * 50)

    collector = SignalCollector()
    stats = collector.collect_all()

    print(f"\nCollection Stats:")
    print(f"  Timestamp: {stats['timestamp']}")
    print(f"  Total Signals: {stats['total_signals']}")

    for name, detector_stats in stats["detectors"].items():
        print(f"\n  {name}:")
        print(f"    Signals: {detector_stats.get('signals_collected', 0)}")
        if "error" in detector_stats:
            print(f"    Error: {detector_stats['error']}")

    if stats["errors"]:
        print(f"\nErrors: {stats['errors']}")
