"""
McDonald's Ice Cream Machine Signal

Uses data from mcbroken.com to track McDonald's ice cream machine outages
as an alternative data source for economic and political predictions.

Theory:
- High ice cream machine failure rates may correlate with economic stress
- Regional patterns could indicate local economic conditions
- Swing state patterns may have political prediction value

Data source: mcbroken.com/stats.json (free, real-time)
"""

import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.signals.base import (
    BaseSignalDetector,
    SignalResult,
    SignalDirection,
    SignalStrength,
    classify_signal_strength,
    determine_direction
)


# Swing states for political signal tracking
SWING_STATES = [
    "Arizona", "Georgia", "Michigan", "Nevada",
    "North Carolina", "Pennsylvania", "Wisconsin"
]

# Major metro areas to track
MAJOR_METROS = [
    "Washington", "New York", "Los Angeles", "Chicago",
    "Houston", "Phoenix", "Philadelphia", "San Antonio"
]


class IceCreamSignalDetector(BaseSignalDetector):
    """
    Detects signals from McDonald's ice cream machine status data.

    Generates multiple signal types:
    - National broken percentage
    - DC metro area status (political signal)
    - Swing state aggregate
    - Regional breakdowns

    Usage:
        detector = IceCreamSignalDetector()
        signals = detector.run()

        for signal in signals:
            print(f"{signal.name}: {signal.value:.1%}")
    """

    # Normal baseline (historical average is ~10-15% broken)
    BASELINE_BROKEN_PCT = 0.12

    # Thresholds for signal strength classification
    STRENGTH_THRESHOLDS = {
        "weak": 0.10,
        "moderate": 0.15,
        "strong": 0.20,
        "very_strong": 0.30
    }

    def __init__(
        self,
        api_url: str = "https://mcbroken.com/stats.json",
        timeout_seconds: int = 10
    ):
        """
        Initialize the ice cream signal detector.

        Args:
            api_url: URL to fetch mcbroken data
            timeout_seconds: Request timeout
        """
        super().__init__(
            name="Ice Cream Machine Signal",
            source="mcbroken.com"
        )
        self.api_url = api_url
        self.timeout = timeout_seconds

    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch ice cream machine status data from mcbroken.com.

        Returns:
            Dict containing cities with their broken machine stats
        """
        logger.info(f"Fetching ice cream data from {self.api_url}")

        try:
            response = requests.get(
                self.api_url,
                timeout=self.timeout,
                headers={"User-Agent": "PredictionMarketsSignal/1.0"}
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched data for {len(data.get('cities', {}))} cities")
            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ice cream data: {e}")
            raise

    def process_data(self, raw_data: Dict[str, Any]) -> List[SignalResult]:
        """
        Process raw mcbroken data into signal results.

        Args:
            raw_data: Data from mcbroken.com API

        Returns:
            List of SignalResult objects
        """
        signals = []
        timestamp = datetime.utcnow()
        cities_list = raw_data.get("cities", [])

        if not cities_list:
            logger.warning("No city data in mcbroken response")
            return signals

        # Convert list to dict keyed by city name for easier lookup
        cities = {city.get("city", "Unknown"): city for city in cities_list}

        # 1. National broken percentage
        national_signal = self._calculate_national_signal(cities, timestamp, raw_data)
        if national_signal:
            signals.append(national_signal)

        # 2. DC metro signal (political)
        dc_signal = self._calculate_dc_signal(cities, timestamp)
        if dc_signal:
            signals.append(dc_signal)

        # 3. Swing states signal
        swing_signal = self._calculate_swing_states_signal(cities, timestamp)
        if swing_signal:
            signals.append(swing_signal)

        # 4. Major metros breakdown
        metro_signals = self._calculate_metro_signals(cities, timestamp)
        signals.extend(metro_signals)

        logger.info(f"Generated {len(signals)} ice cream signals")
        return signals

    def _calculate_national_signal(
        self,
        cities: Dict[str, Any],
        timestamp: datetime,
        raw_data: Optional[Dict[str, Any]] = None
    ) -> Optional[SignalResult]:
        """Calculate national ice cream machine failure rate."""
        # Try to get national broken % from top-level data
        if raw_data and "broken" in raw_data:
            broken_pct = raw_data["broken"] / 100.0  # Convert from percentage
            total_machines = sum(
                city.get("total_locations", 0) for city in cities.values()
            )
            broken_machines = int(total_machines * broken_pct)
        else:
            # Calculate from city data
            total_machines = 0
            broken_machines = 0

            for city_data in cities.values():
                if isinstance(city_data, dict):
                    total = city_data.get("total_locations", city_data.get("total", 0))
                    broken_pct_city = city_data.get("broken", 0) / 100.0
                    total_machines += total
                    broken_machines += int(total * broken_pct_city)

        if total_machines == 0:
            return None

        broken_pct = broken_machines / total_machines
        direction = determine_direction(
            broken_pct,
            self.BASELINE_BROKEN_PCT,
            threshold=0.03
        )
        strength = classify_signal_strength(broken_pct, self.STRENGTH_THRESHOLDS)

        # High failure = bearish for economy
        if direction == SignalDirection.BULLISH:
            signal_direction = SignalDirection.BEARISH
        elif direction == SignalDirection.BEARISH:
            signal_direction = SignalDirection.BULLISH
        else:
            signal_direction = SignalDirection.NEUTRAL

        return SignalResult(
            name="National Ice Cream Failure Rate",
            source=self.source,
            timestamp=timestamp,
            value=broken_pct,
            direction=signal_direction,
            confidence=0.6,  # Moderate confidence in this alternative data
            strength=strength,
            raw_data={
                "total_machines": total_machines,
                "broken_machines": broken_machines
            },
            related_markets=[
                "recession", "economic", "consumer sentiment", "inflation"
            ],
            metadata={
                "baseline": self.BASELINE_BROKEN_PCT,
                "deviation": broken_pct - self.BASELINE_BROKEN_PCT
            }
        )

    def _calculate_dc_signal(
        self,
        cities: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Calculate DC area signal (political indicator)."""
        dc_data = cities.get("Washington", cities.get("washington", {}))

        if not dc_data or not isinstance(dc_data, dict):
            # Try to find DC by searching
            for city_name, data in cities.items():
                if "washington" in city_name.lower() or "dc" in city_name.lower():
                    dc_data = data
                    break

        if not dc_data or not isinstance(dc_data, dict):
            return None

        total = dc_data.get("total_locations", dc_data.get("total", 0))
        broken_pct_raw = dc_data.get("broken", 0)

        if total == 0:
            return None

        # broken is already a percentage (0-100)
        broken_pct = broken_pct_raw / 100.0
        direction = determine_direction(broken_pct, self.BASELINE_BROKEN_PCT)

        # Invert: high failure = bearish for current administration
        if direction == SignalDirection.BULLISH:
            signal_direction = SignalDirection.BEARISH
        elif direction == SignalDirection.BEARISH:
            signal_direction = SignalDirection.BULLISH
        else:
            signal_direction = SignalDirection.NEUTRAL

        return SignalResult(
            name="DC Metro Ice Cream Status",
            source=self.source,
            timestamp=timestamp,
            value=broken_pct,
            direction=signal_direction,
            confidence=0.4,  # Lower confidence for political signal
            strength=classify_signal_strength(broken_pct, self.STRENGTH_THRESHOLDS),
            raw_data=dc_data,
            related_markets=[
                "election", "president", "politics", "administration"
            ],
            metadata={
                "region": "DC Metro",
                "use_case": "political_indicator"
            }
        )

    def _calculate_swing_states_signal(
        self,
        cities: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Calculate aggregate swing state ice cream signal."""
        swing_total = 0
        swing_broken_weighted = 0

        for city_name, city_data in cities.items():
            if not isinstance(city_data, dict):
                continue

            # Check if city is in a swing state (by city name match)
            for state in SWING_STATES:
                if state.lower() in city_name.lower():
                    total = city_data.get("total_locations", city_data.get("total", 0))
                    broken_pct = city_data.get("broken", 0) / 100.0
                    swing_total += total
                    swing_broken_weighted += total * broken_pct
                    break

        if swing_total == 0:
            return None

        broken_pct = swing_broken_weighted / swing_total
        direction = determine_direction(broken_pct, self.BASELINE_BROKEN_PCT)

        return SignalResult(
            name="Swing States Ice Cream Index",
            source=self.source,
            timestamp=timestamp,
            value=broken_pct,
            direction=direction,
            confidence=0.35,  # Low confidence - experimental
            strength=classify_signal_strength(broken_pct, self.STRENGTH_THRESHOLDS),
            raw_data={
                "swing_total": swing_total,
                "swing_broken": swing_broken,
                "states_tracked": SWING_STATES
            },
            related_markets=[
                "election", "swing state", "battleground"
            ],
            metadata={
                "states": SWING_STATES,
                "use_case": "political_swing_indicator"
            }
        )

    def _calculate_metro_signals(
        self,
        cities: Dict[str, Any],
        timestamp: datetime
    ) -> List[SignalResult]:
        """Calculate individual metro area signals."""
        signals = []

        for metro in MAJOR_METROS:
            metro_data = None

            # Find the metro in cities data
            for city_name, data in cities.items():
                if metro.lower() in city_name.lower():
                    metro_data = data
                    break

            if not metro_data or not isinstance(metro_data, dict):
                continue

            total = metro_data.get("total_locations", metro_data.get("total", 0))
            broken_pct_raw = metro_data.get("broken", 0)

            if total == 0:
                continue

            # broken is already a percentage (0-100)
            broken_pct = broken_pct_raw / 100.0
            direction = determine_direction(broken_pct, self.BASELINE_BROKEN_PCT)

            signals.append(SignalResult(
                name=f"{metro} Ice Cream Status",
                source=self.source,
                timestamp=timestamp,
                value=broken_pct,
                direction=direction,
                confidence=0.3,  # Individual metros have low confidence
                strength=classify_signal_strength(broken_pct, self.STRENGTH_THRESHOLDS),
                raw_data=metro_data,
                related_markets=[
                    f"{metro.lower()} economy", "regional"
                ],
                metadata={
                    "metro": metro,
                    "use_case": "regional_economic_indicator"
                }
            ))

        return signals


def fetch_ice_cream_signals() -> List[SignalResult]:
    """Convenience function to fetch all ice cream signals."""
    detector = IceCreamSignalDetector()
    return detector.run()


if __name__ == "__main__":
    # Test the signal detector
    print("Testing Ice Cream Machine Signal Detector")
    print("-" * 50)

    try:
        detector = IceCreamSignalDetector()
        signals = detector.run()

        print(f"\nFound {len(signals)} signals:\n")

        for signal in signals:
            print(f"  {signal.name}")
            print(f"    Value: {signal.value:.1%}")
            print(f"    Direction: {signal.direction.value}")
            print(f"    Strength: {signal.strength.value}")
            print(f"    Confidence: {signal.confidence:.0%}")
            print(f"    Markets: {', '.join(signal.related_markets[:3])}")
            print()

    except Exception as e:
        print(f"Error: {e}")
