"""
Google Trends Signal Detector

Uses Google Trends data to detect search volume spikes for market-related keywords.
Useful for detecting breaking news, sentiment shifts, and public interest changes.

Data source: pytrends library (free, unofficial Google Trends API)
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not installed. Run: pip install pytrends")

from src.signals.base import (
    BaseSignalDetector,
    SignalResult,
    SignalDirection,
    SignalStrength,
    classify_signal_strength,
)


# Default keywords for prediction market topics
DEFAULT_KEYWORDS = {
    "politics": [
        "election", "president", "congress", "senate", "house"
    ],
    "crypto": [
        "bitcoin", "ethereum", "crypto crash", "bitcoin price"
    ],
    "economics": [
        "recession", "inflation", "federal reserve", "interest rates"
    ],
    "breaking": [
        "breaking news", "just announced", "leaked"
    ]
}


class GoogleTrendsSignalDetector(BaseSignalDetector):
    """
    Detects signals from Google Trends search volume data.

    Generates signals based on:
    - Search volume spikes (relative to 7-day average)
    - Trending topics related to prediction markets
    - Geographic interest for swing states

    Note: Uses pytrends which is an unofficial API and may be rate-limited.

    Usage:
        detector = GoogleTrendsSignalDetector(keywords=["bitcoin", "recession"])
        signals = detector.run()

        for signal in signals:
            print(f"{signal.name}: {signal.value}")
    """

    # Thresholds for spike detection
    SPIKE_THRESHOLD = 1.5  # 50% above average = spike
    STRONG_SPIKE_THRESHOLD = 2.0  # 100% above average = strong spike

    # Signal strength thresholds
    STRENGTH_THRESHOLDS = {
        "weak": 0.10,
        "moderate": 0.30,
        "strong": 0.50,
        "very_strong": 0.75
    }

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        category: Optional[str] = None,
        timeframe: str = "now 7-d",
        geo: str = "US",
        timeout: tuple = (10, 25)
    ):
        """
        Initialize the Google Trends signal detector.

        Args:
            keywords: List of keywords to track (max 5 per request)
            category: Category to pull default keywords from
            timeframe: Trends timeframe (e.g., "now 7-d", "today 1-m")
            geo: Geographic region (e.g., "US", "GB")
            timeout: Request timeout tuple (connect, read)
        """
        super().__init__(
            name="Google Trends Signal",
            source="google_trends"
        )

        # Set keywords
        if keywords:
            self.keywords = keywords[:5]  # Limit to 5
        elif category and category in DEFAULT_KEYWORDS:
            self.keywords = DEFAULT_KEYWORDS[category][:5]
        else:
            # Default: top prediction market keywords
            self.keywords = ["election", "bitcoin", "recession", "inflation", "war"]

        self.timeframe = timeframe
        self.geo = geo
        self.timeout = timeout
        self._pytrends: Optional[Any] = None

    def _get_pytrends(self) -> "TrendReq":
        """Get or create pytrends instance."""
        if not PYTRENDS_AVAILABLE:
            raise ImportError("pytrends is not installed. Run: pip install pytrends")

        if self._pytrends is None:
            self._pytrends = TrendReq(
                hl="en-US",
                tz=360,
                timeout=self.timeout,
                retries=2,
                backoff_factor=0.1
            )
        return self._pytrends

    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch Google Trends data for configured keywords.

        Returns:
            Dict containing interest over time and related data
        """
        logger.info(f"Fetching Google Trends for: {self.keywords}")

        if not PYTRENDS_AVAILABLE:
            logger.error("pytrends not available")
            return {"error": "pytrends not installed", "keywords": self.keywords}

        try:
            pytrends = self._get_pytrends()

            # Build payload
            pytrends.build_payload(
                self.keywords,
                cat=0,
                timeframe=self.timeframe,
                geo=self.geo
            )

            # Get interest over time
            interest_over_time = pytrends.interest_over_time()

            # Get related queries
            related_queries = pytrends.related_queries()

            # Get interest by region (for swing state analysis)
            interest_by_region = pytrends.interest_by_region(
                resolution="REGION",
                inc_low_vol=True,
                inc_geo_code=False
            )

            logger.info(f"Fetched trends data: {len(interest_over_time)} time points")

            return {
                "keywords": self.keywords,
                "interest_over_time": interest_over_time.to_dict() if not interest_over_time.empty else {},
                "related_queries": related_queries,
                "interest_by_region": interest_by_region.to_dict() if not interest_by_region.empty else {},
                "timeframe": self.timeframe,
                "geo": self.geo,
                "fetched_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return {
                "error": str(e),
                "keywords": self.keywords,
                "fetched_at": datetime.utcnow().isoformat()
            }

    def process_data(self, raw_data: Dict[str, Any]) -> List[SignalResult]:
        """
        Process Google Trends data into signal results.

        Args:
            raw_data: Data from fetch_data()

        Returns:
            List of SignalResult objects
        """
        signals = []
        timestamp = datetime.utcnow()

        if "error" in raw_data:
            logger.warning(f"Cannot process trends data: {raw_data['error']}")
            return signals

        # Process interest over time
        interest_data = raw_data.get("interest_over_time", {})
        if interest_data:
            for keyword in self.keywords:
                keyword_signals = self._process_keyword_trend(
                    keyword,
                    interest_data,
                    timestamp
                )
                signals.extend(keyword_signals)

        # Process interest by region
        region_data = raw_data.get("interest_by_region", {})
        if region_data:
            regional_signal = self._process_regional_interest(
                region_data,
                timestamp
            )
            if regional_signal:
                signals.append(regional_signal)

        logger.info(f"Generated {len(signals)} trends signals")
        return signals

    def _process_keyword_trend(
        self,
        keyword: str,
        interest_data: Dict[str, Any],
        timestamp: datetime
    ) -> List[SignalResult]:
        """Process trend data for a single keyword."""
        signals = []

        if keyword not in interest_data:
            return signals

        values = interest_data[keyword]
        if not values:
            return signals

        # Convert to list of values
        value_list = list(values.values())
        if len(value_list) < 2:
            return signals

        # Calculate current vs average
        current = value_list[-1] if value_list else 0
        avg = sum(value_list[:-1]) / len(value_list[:-1]) if len(value_list) > 1 else current

        if avg == 0:
            return signals

        spike_ratio = current / avg

        # Determine if this is a spike
        if spike_ratio >= self.SPIKE_THRESHOLD:
            # Calculate normalized spike value (0-1)
            spike_value = min(1.0, (spike_ratio - 1.0) / 2.0)

            # Determine direction (spike = bullish for related markets)
            if spike_ratio >= self.STRONG_SPIKE_THRESHOLD:
                direction = SignalDirection.BULLISH
                strength = SignalStrength.STRONG
            else:
                direction = SignalDirection.BULLISH
                strength = SignalStrength.MODERATE

            confidence = min(0.8, 0.4 + (spike_value * 0.4))

            signals.append(SignalResult(
                name=f"Google Trends Spike: {keyword}",
                source=self.source,
                timestamp=timestamp,
                value=spike_value,
                direction=direction,
                confidence=confidence,
                strength=strength,
                raw_data={
                    "keyword": keyword,
                    "current_interest": current,
                    "average_interest": avg,
                    "spike_ratio": spike_ratio,
                    "recent_values": value_list[-7:]
                },
                related_markets=self._get_related_markets(keyword),
                metadata={
                    "timeframe": self.timeframe,
                    "geo": self.geo,
                    "is_spike": True,
                    "spike_threshold": self.SPIKE_THRESHOLD
                }
            ))

        return signals

    def _process_regional_interest(
        self,
        region_data: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Process regional interest data for swing states."""
        swing_states = ["Arizona", "Georgia", "Michigan", "Nevada",
                       "North Carolina", "Pennsylvania", "Wisconsin"]

        swing_interest = {}
        total_interest = 0
        count = 0

        for keyword in self.keywords:
            if keyword not in region_data:
                continue

            keyword_regions = region_data[keyword]
            for state in swing_states:
                if state in keyword_regions:
                    interest = keyword_regions[state]
                    if state not in swing_interest:
                        swing_interest[state] = 0
                    swing_interest[state] += interest
                    total_interest += interest
                    count += 1

        if count == 0:
            return None

        avg_swing_interest = total_interest / count

        # Normalize to 0-1 scale
        normalized_value = min(1.0, avg_swing_interest / 100)

        return SignalResult(
            name="Swing State Search Interest",
            source=self.source,
            timestamp=timestamp,
            value=normalized_value,
            direction=SignalDirection.NEUTRAL,
            confidence=0.35,
            strength=classify_signal_strength(normalized_value, self.STRENGTH_THRESHOLDS),
            raw_data={
                "swing_interest": swing_interest,
                "average_interest": avg_swing_interest,
                "keywords_tracked": self.keywords
            },
            related_markets=["election", "swing state", "political"],
            metadata={
                "swing_states": swing_states,
                "use_case": "political_interest_tracking"
            }
        )

    def _get_related_markets(self, keyword: str) -> List[str]:
        """Get related market keywords for a trend keyword."""
        keyword_lower = keyword.lower()

        market_mappings = {
            "election": ["president", "election", "congress", "political"],
            "president": ["president", "election", "administration"],
            "bitcoin": ["bitcoin", "crypto", "btc price"],
            "ethereum": ["ethereum", "eth", "crypto"],
            "crypto": ["crypto", "bitcoin", "blockchain"],
            "recession": ["recession", "economic", "gdp", "unemployment"],
            "inflation": ["inflation", "cpi", "federal reserve", "interest rates"],
            "war": ["geopolitical", "military", "conflict"],
            "fed": ["federal reserve", "interest rates", "monetary policy"],
        }

        for key, markets in market_mappings.items():
            if key in keyword_lower:
                return markets

        return [keyword_lower]


def fetch_trends_signals(
    keywords: Optional[List[str]] = None,
    category: Optional[str] = None
) -> List[SignalResult]:
    """Convenience function to fetch Google Trends signals."""
    detector = GoogleTrendsSignalDetector(keywords=keywords, category=category)
    return detector.run()


if __name__ == "__main__":
    print("Testing Google Trends Signal Detector")
    print("-" * 50)

    if not PYTRENDS_AVAILABLE:
        print("pytrends not installed. Run: pip install pytrends")
    else:
        try:
            detector = GoogleTrendsSignalDetector(
                keywords=["bitcoin", "election", "recession"]
            )
            signals = detector.run()

            print(f"\nFound {len(signals)} signals:\n")

            for signal in signals:
                print(f"  {signal.name}")
                print(f"    Value: {signal.value:.2f}")
                print(f"    Direction: {signal.direction.value}")
                print(f"    Strength: {signal.strength.value}")
                print(f"    Confidence: {signal.confidence:.0%}")
                print(f"    Markets: {', '.join(signal.related_markets[:3])}")
                print()

        except Exception as e:
            print(f"Error: {e}")
