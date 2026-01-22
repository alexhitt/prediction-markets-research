"""
Odds Movement Signal Detector

Tracks rapid price changes and momentum shifts in prediction markets.
Detects both platform-specific movements and cross-platform divergences.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

from src.signals.base import (
    BaseSignalDetector,
    SignalDirection,
    SignalResult,
    SignalStrength,
    classify_signal_strength,
)


# Movement thresholds
RAPID_MOVE_THRESHOLD = 0.05  # 5% move in short period
STRONG_MOVE_THRESHOLD = 0.10  # 10% move
EXTREME_MOVE_THRESHOLD = 0.15  # 15% move

# Time thresholds
RAPID_PERIOD_MINUTES = 60  # 1 hour for rapid detection
MOMENTUM_PERIOD_HOURS = 24  # 24 hours for momentum


class OddsMovementSignalDetector(BaseSignalDetector):
    """
    Detects rapid price movements and momentum in prediction markets.

    Generates signals based on:
    - Rapid price changes (>5% in 1 hour)
    - Momentum (sustained direction over 24h)
    - Cross-platform divergences
    - Volume-weighted movements

    Usage:
        detector = OddsMovementSignalDetector()
        signals = detector.run()
    """

    STRENGTH_THRESHOLDS = {
        "weak": 0.05,
        "moderate": 0.08,
        "strong": 0.12,
        "very_strong": 0.18
    }

    def __init__(
        self,
        rapid_threshold: float = RAPID_MOVE_THRESHOLD,
        momentum_hours: int = MOMENTUM_PERIOD_HOURS,
        min_volume: float = 1000
    ):
        """
        Initialize odds movement detector.

        Args:
            rapid_threshold: Minimum % change for rapid move signal
            momentum_hours: Hours to calculate momentum over
            min_volume: Minimum volume to consider
        """
        super().__init__(
            name="Odds Movement Signal",
            source="market_prices"
        )

        self.rapid_threshold = rapid_threshold
        self.momentum_hours = momentum_hours
        self.min_volume = min_volume

    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch current and historical prices from platforms.

        Returns:
            Dict with platform -> market data mapping
        """
        results = {
            "polymarket": [],
            "kalshi": [],
            "movements": [],
            "error": None
        }

        # Fetch Polymarket data
        polymarket_data = self._fetch_polymarket_prices()
        results["polymarket"] = polymarket_data

        # Fetch Kalshi data
        kalshi_data = self._fetch_kalshi_prices()
        results["kalshi"] = kalshi_data

        logger.info(f"Fetched {len(polymarket_data)} Polymarket and {len(kalshi_data)} Kalshi markets")
        return results

    def _fetch_polymarket_prices(self) -> List[Dict[str, Any]]:
        """Fetch current prices and recent history from Polymarket."""
        markets = []

        try:
            # Get active markets with volume
            response = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "limit": 50},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Polymarket API error: {response.status_code}")
                return markets

            for m in response.json():
                volume = float(m.get("volume", 0) or 0)
                if volume < self.min_volume:
                    continue

                # Get current price - outcomePrices is a JSON string like '["0.5", "0.5"]'
                outcome_prices_raw = m.get("outcomePrices")
                if isinstance(outcome_prices_raw, str):
                    try:
                        outcome_prices = json.loads(outcome_prices_raw)
                        current_price = float(outcome_prices[0]) if outcome_prices else 0.5
                    except (json.JSONDecodeError, IndexError, ValueError):
                        current_price = 0.5
                elif isinstance(outcome_prices_raw, list) and outcome_prices_raw:
                    current_price = float(outcome_prices_raw[0])
                else:
                    current_price = 0.5

                # Get 24h price change if available
                price_change = float(m.get("priceChange24h", 0) or 0)

                markets.append({
                    "market_id": m.get("id") or m.get("conditionId"),
                    "platform": "polymarket",
                    "question": m.get("question", "Unknown"),
                    "category": m.get("category", ""),
                    "current_price": current_price,
                    "price_24h_ago": current_price - price_change if price_change else None,
                    "price_change_24h": price_change,
                    "volume": volume,
                    "liquidity": float(m.get("liquidity", 0) or 0),
                    "end_date": m.get("endDate")
                })

        except requests.RequestException as e:
            logger.error(f"Error fetching Polymarket: {e}")

        return markets

    def _fetch_kalshi_prices(self) -> List[Dict[str, Any]]:
        """Fetch current prices from Kalshi."""
        markets = []

        try:
            response = requests.get(
                "https://api.kalshi.com/trade-api/v2/markets",
                params={"limit": 50, "status": "open"},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Kalshi API error: {response.status_code}")
                return markets

            for m in response.json().get("markets", []):
                volume = int(m.get("volume", 0))
                if volume < self.min_volume:
                    continue

                # Calculate current price from bid/ask
                yes_bid = m.get("yes_bid")
                yes_ask = m.get("yes_ask")
                if yes_bid and yes_ask:
                    current_price = (yes_bid + yes_ask) / 200  # Cents to 0-1
                else:
                    current_price = 0.5

                markets.append({
                    "market_id": m.get("ticker"),
                    "platform": "kalshi",
                    "question": m.get("title", "Unknown"),
                    "category": m.get("category", ""),
                    "current_price": current_price,
                    "price_24h_ago": None,  # Would need historical endpoint
                    "price_change_24h": None,
                    "volume": volume,
                    "liquidity": 0,
                    "end_date": m.get("close_time")
                })

        except requests.RequestException as e:
            logger.error(f"Error fetching Kalshi: {e}")

        return markets

    def process_data(self, raw_data: Dict[str, Any]) -> List[SignalResult]:
        """
        Process price data into movement signals.

        Args:
            raw_data: Dict with platform -> market data

        Returns:
            List of SignalResult objects
        """
        signals = []
        timestamp = datetime.utcnow()

        if raw_data.get("error"):
            return signals

        polymarket_markets = raw_data.get("polymarket", [])
        kalshi_markets = raw_data.get("kalshi", [])
        all_markets = polymarket_markets + kalshi_markets

        # Detect rapid movements
        for market in all_markets:
            rapid_signal = self._detect_rapid_movement(market, timestamp)
            if rapid_signal:
                signals.append(rapid_signal)

        # Detect momentum (24h trends)
        for market in polymarket_markets:  # Only Polymarket has 24h data
            momentum_signal = self._detect_momentum(market, timestamp)
            if momentum_signal:
                signals.append(momentum_signal)

        # Detect cross-platform divergences
        divergence_signals = self._detect_divergences(polymarket_markets, kalshi_markets, timestamp)
        signals.extend(divergence_signals)

        # Overall market momentum
        overall_signal = self._calculate_overall_momentum(all_markets, timestamp)
        if overall_signal:
            signals.append(overall_signal)

        logger.info(f"Generated {len(signals)} odds movement signals")
        return signals

    def _detect_rapid_movement(
        self,
        market: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Detect rapid price movement in a market."""
        price_change = market.get("price_change_24h")
        if price_change is None:
            return None

        abs_change = abs(price_change)
        if abs_change < self.rapid_threshold:
            return None

        current_price = market["current_price"]

        # Determine direction and strength
        if price_change > 0:
            direction = SignalDirection.BULLISH
        else:
            direction = SignalDirection.BEARISH

        # Strength based on magnitude
        if abs_change >= EXTREME_MOVE_THRESHOLD:
            strength = SignalStrength.VERY_STRONG
            signal_type = "extreme_movement"
        elif abs_change >= STRONG_MOVE_THRESHOLD:
            strength = SignalStrength.STRONG
            signal_type = "strong_movement"
        else:
            strength = SignalStrength.MODERATE
            signal_type = "rapid_movement"

        # Confidence based on movement size and volume
        volume_factor = min(1.0, market["volume"] / 100000)
        confidence = min(0.75, 0.4 + abs_change * 2 + volume_factor * 0.2)

        return SignalResult(
            name=f"Rapid Move: {market['question'][:35]}...",
            source=self.source,
            timestamp=timestamp,
            value=price_change,
            direction=direction,
            confidence=confidence,
            strength=strength,
            raw_data={
                "market_id": market["market_id"],
                "platform": market["platform"],
                "current_price": current_price,
                "price_change_24h": price_change,
                "volume": market["volume"]
            },
            related_markets=[market["question"], market.get("category", "")],
            metadata={
                "signal_type": signal_type,
                "urgency": "high" if abs_change >= STRONG_MOVE_THRESHOLD else "medium"
            }
        )

    def _detect_momentum(
        self,
        market: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Detect sustained momentum in a market."""
        price_change = market.get("price_change_24h")
        if price_change is None:
            return None

        current_price = market["current_price"]

        # Check for sustained momentum (price moved significantly AND is continuing)
        # Strong momentum: moved >8% in direction
        abs_change = abs(price_change)
        if abs_change < 0.08:
            return None

        # Additional check: is price near extreme?
        # Bullish momentum stronger if price still below 0.7
        # Bearish momentum stronger if price still above 0.3
        is_continuation_likely = (
            (price_change > 0 and current_price < 0.75) or
            (price_change < 0 and current_price > 0.25)
        )

        if not is_continuation_likely:
            return None

        direction = SignalDirection.BULLISH if price_change > 0 else SignalDirection.BEARISH
        confidence = min(0.65, 0.35 + abs_change * 1.5)

        return SignalResult(
            name=f"Momentum: {market['question'][:35]}...",
            source=self.source,
            timestamp=timestamp,
            value=price_change,
            direction=direction,
            confidence=confidence,
            strength=SignalStrength.MODERATE if abs_change < 0.12 else SignalStrength.STRONG,
            raw_data={
                "market_id": market["market_id"],
                "platform": market["platform"],
                "current_price": current_price,
                "price_change_24h": price_change,
                "continuation_likely": is_continuation_likely
            },
            related_markets=[market["question"]],
            metadata={"signal_type": "momentum"}
        )

    def _detect_divergences(
        self,
        polymarket_markets: List[Dict[str, Any]],
        kalshi_markets: List[Dict[str, Any]],
        timestamp: datetime
    ) -> List[SignalResult]:
        """Detect price divergences between platforms."""
        signals = []

        # Build lookup by question keywords for matching
        kalshi_lookup = {}
        for m in kalshi_markets:
            keywords = self._extract_keywords(m["question"])
            for kw in keywords:
                if kw not in kalshi_lookup:
                    kalshi_lookup[kw] = []
                kalshi_lookup[kw].append(m)

        for poly_market in polymarket_markets:
            poly_keywords = self._extract_keywords(poly_market["question"])

            # Find potential matches
            for kw in poly_keywords:
                for kalshi_market in kalshi_lookup.get(kw, []):
                    # Check price divergence
                    poly_price = poly_market["current_price"]
                    kalshi_price = kalshi_market["current_price"]
                    spread = abs(poly_price - kalshi_price)

                    if spread >= 0.05:  # 5% spread minimum
                        signal = self._create_divergence_signal(
                            poly_market, kalshi_market, spread, timestamp
                        )
                        if signal:
                            signals.append(signal)
                            break  # Only one signal per market pair

        return signals

    def _create_divergence_signal(
        self,
        poly_market: Dict[str, Any],
        kalshi_market: Dict[str, Any],
        spread: float,
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Create signal for cross-platform divergence."""
        poly_price = poly_market["current_price"]
        kalshi_price = kalshi_market["current_price"]

        # Direction: buy on cheaper platform
        if poly_price < kalshi_price:
            direction = SignalDirection.BULLISH  # Bullish on Polymarket
            buy_platform = "polymarket"
            sell_platform = "kalshi"
        else:
            direction = SignalDirection.BEARISH  # Bearish on Polymarket (bullish on Kalshi)
            buy_platform = "kalshi"
            sell_platform = "polymarket"

        # Confidence based on spread size
        confidence = min(0.8, 0.5 + spread * 3)

        return SignalResult(
            name=f"Platform Divergence: {poly_market['question'][:30]}...",
            source=self.source,
            timestamp=timestamp,
            value=spread,
            direction=direction,
            confidence=confidence,
            strength=SignalStrength.STRONG if spread >= 0.10 else SignalStrength.MODERATE,
            raw_data={
                "polymarket_id": poly_market["market_id"],
                "kalshi_id": kalshi_market["market_id"],
                "polymarket_price": poly_price,
                "kalshi_price": kalshi_price,
                "spread": spread,
                "buy_platform": buy_platform,
                "sell_platform": sell_platform
            },
            related_markets=[poly_market["question"], "arbitrage"],
            metadata={
                "signal_type": "platform_divergence",
                "arbitrage_opportunity": True,
                "urgency": "high" if spread >= 0.08 else "medium"
            }
        )

    def _calculate_overall_momentum(
        self,
        markets: List[Dict[str, Any]],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Calculate overall market momentum."""
        markets_with_change = [m for m in markets if m.get("price_change_24h") is not None]

        if len(markets_with_change) < 10:
            return None

        # Calculate aggregate momentum
        total_change = sum(m["price_change_24h"] for m in markets_with_change)
        avg_change = total_change / len(markets_with_change)

        bullish_count = len([m for m in markets_with_change if m["price_change_24h"] > 0.02])
        bearish_count = len([m for m in markets_with_change if m["price_change_24h"] < -0.02])

        # Direction based on majority
        if bullish_count > bearish_count * 1.5:
            direction = SignalDirection.BULLISH
        elif bearish_count > bullish_count * 1.5:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        confidence = min(0.5, 0.25 + abs(bullish_count - bearish_count) / len(markets_with_change))

        return SignalResult(
            name="Overall Market Momentum",
            source=self.source,
            timestamp=timestamp,
            value=avg_change,
            direction=direction,
            confidence=confidence,
            strength=classify_signal_strength(abs(avg_change), self.STRENGTH_THRESHOLDS),
            raw_data={
                "markets_analyzed": len(markets_with_change),
                "avg_change": avg_change,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count
            },
            related_markets=["overall", "market_momentum"],
            metadata={"signal_type": "overall_momentum"}
        )

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from a market question for matching."""
        # Simple keyword extraction
        stopwords = {"will", "the", "be", "to", "in", "on", "at", "for", "of", "a", "an", "and", "or"}
        words = question.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:5]


def get_odds_movement_detector() -> OddsMovementSignalDetector:
    """Get a configured odds movement detector."""
    return OddsMovementSignalDetector()


if __name__ == "__main__":
    # Test the detector
    detector = OddsMovementSignalDetector()

    print("Analyzing odds movements...")
    signals = detector.run()

    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"\n{signal.name}")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Value: {signal.value:.3f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Strength: {signal.strength.value}")
