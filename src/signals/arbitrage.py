"""
Cross-Platform Arbitrage Detection

Identifies arbitrage opportunities between Polymarket and Kalshi
when the same event is priced differently across platforms.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from loguru import logger

from src.clients.polymarket_client import PolymarketClient, PolymarketMarket
from src.clients.kalshi_client import KalshiClient, KalshiMarket


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    detected_at: datetime

    # Platform 1
    platform_1: str
    market_1_id: str
    market_1_question: str
    market_1_yes_price: float  # 0-1 scale

    # Platform 2
    platform_2: str
    market_2_id: str
    market_2_question: str
    market_2_yes_price: float  # 0-1 scale

    # Opportunity metrics
    spread: float  # Absolute price difference
    profit_potential: float  # Profit % if executed
    similarity_score: float  # How confident markets are same event

    # Recommended action
    action: str  # e.g., "Buy Polymarket YES + Kalshi NO"


class ArbitrageDetector:
    """
    Detects arbitrage opportunities between prediction market platforms.

    Usage:
        detector = ArbitrageDetector()
        opportunities = detector.find_opportunities()

        for opp in opportunities:
            print(f"Spread: {opp.spread:.2%} - {opp.action}")
    """

    def __init__(
        self,
        polymarket_client: Optional[PolymarketClient] = None,
        kalshi_client: Optional[KalshiClient] = None,
        min_similarity: float = 0.7,
        min_spread: float = 0.02  # 2% minimum spread
    ):
        self.poly_client = polymarket_client or PolymarketClient()
        self.kalshi_client = kalshi_client or KalshiClient()
        self.min_similarity = min_similarity
        self.min_spread = min_spread

    def find_opportunities(
        self,
        categories: Optional[List[str]] = None,
        limit_per_platform: int = 100
    ) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities across platforms.

        Args:
            categories: Filter by categories (e.g., ["politics", "crypto"])
            limit_per_platform: Max markets to fetch per platform

        Returns:
            List of ArbitrageOpportunity sorted by profit potential
        """
        logger.info("Scanning for arbitrage opportunities...")

        # Fetch markets from both platforms
        poly_markets = self.poly_client.get_markets(limit=limit_per_platform)
        kalshi_markets = self.kalshi_client.get_markets(limit=limit_per_platform)

        logger.info(f"Polymarket: {len(poly_markets)} markets")
        logger.info(f"Kalshi: {len(kalshi_markets)} markets")

        # Find matching markets
        opportunities = []
        matched_pairs = self._find_matching_markets(poly_markets, kalshi_markets)

        for poly_market, kalshi_market, similarity in matched_pairs:
            opp = self._analyze_pair(poly_market, kalshi_market, similarity)
            if opp and opp.spread >= self.min_spread:
                opportunities.append(opp)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x.profit_potential, reverse=True)

        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        return opportunities

    def _find_matching_markets(
        self,
        poly_markets: List[PolymarketMarket],
        kalshi_markets: List[KalshiMarket]
    ) -> List[Tuple[PolymarketMarket, KalshiMarket, float]]:
        """
        Find markets that represent the same underlying event.

        Returns list of (poly_market, kalshi_market, similarity_score) tuples.
        """
        matches = []

        for poly in poly_markets:
            poly_text = self._normalize_text(poly.question)

            for kalshi in kalshi_markets:
                kalshi_text = self._normalize_text(kalshi.title)

                similarity = self._calculate_similarity(poly_text, kalshi_text)

                if similarity >= self.min_similarity:
                    matches.append((poly, kalshi, similarity))

        # Sort by similarity and remove duplicates
        matches.sort(key=lambda x: x[2], reverse=True)

        # Keep only best match for each market
        seen_poly = set()
        seen_kalshi = set()
        unique_matches = []

        for poly, kalshi, sim in matches:
            if poly.id not in seen_poly and kalshi.ticker not in seen_kalshi:
                unique_matches.append((poly, kalshi, sim))
                seen_poly.add(poly.id)
                seen_kalshi.add(kalshi.ticker)

        return unique_matches

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()

        # Remove common prefixes/suffixes
        text = re.sub(r'^will\s+', '', text)
        text = re.sub(r'\?$', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)

        return text.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two market questions."""
        # Use SequenceMatcher for basic similarity
        base_similarity = SequenceMatcher(None, text1, text2).ratio()

        # Boost if key terms match
        keywords1 = set(text1.split())
        keywords2 = set(text2.split())
        common_keywords = keywords1 & keywords2

        # Filter out common words
        stopwords = {'will', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'by'}
        meaningful_common = common_keywords - stopwords

        if len(meaningful_common) >= 3:
            base_similarity = min(1.0, base_similarity + 0.15)

        return base_similarity

    def _analyze_pair(
        self,
        poly_market: PolymarketMarket,
        kalshi_market: KalshiMarket,
        similarity: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Analyze a matched pair for arbitrage opportunity.
        """
        try:
            # Get Polymarket price (already 0-1)
            poly_price = None
            if poly_market.prices:
                # Get YES price
                for outcome, price in poly_market.prices.items():
                    if outcome.lower() in ['yes', 'true', '1']:
                        poly_price = price
                        break
                if poly_price is None:
                    poly_price = list(poly_market.prices.values())[0]

            if poly_price is None:
                return None

            # Get Kalshi price (convert from cents to 0-1)
            kalshi_price = None
            if kalshi_market.yes_bid and kalshi_market.yes_ask:
                kalshi_price = (kalshi_market.yes_bid + kalshi_market.yes_ask) / 200
            elif kalshi_market.yes_bid:
                kalshi_price = kalshi_market.yes_bid / 100
            elif kalshi_market.yes_ask:
                kalshi_price = kalshi_market.yes_ask / 100

            if kalshi_price is None:
                return None

            # Calculate spread
            spread = abs(poly_price - kalshi_price)

            # Determine action based on which is cheaper
            if poly_price < kalshi_price:
                # Polymarket YES is cheaper
                # Buy Polymarket YES + Kalshi NO
                action = "Buy Polymarket YES + Kalshi NO"
                profit = kalshi_price - poly_price
            else:
                # Kalshi YES is cheaper
                # Buy Kalshi YES + Polymarket NO
                action = "Buy Kalshi YES + Polymarket NO"
                profit = poly_price - kalshi_price

            # Calculate profit potential (as percentage)
            # Cost = cheaper YES + (1 - more expensive YES) = total paid
            # Return = $1 guaranteed
            cost = min(poly_price, kalshi_price) + (1 - max(poly_price, kalshi_price))
            profit_potential = (1 - cost) / cost if cost > 0 else 0

            return ArbitrageOpportunity(
                detected_at=datetime.utcnow(),
                platform_1="polymarket",
                market_1_id=poly_market.id,
                market_1_question=poly_market.question,
                market_1_yes_price=poly_price,
                platform_2="kalshi",
                market_2_id=kalshi_market.ticker,
                market_2_question=kalshi_market.title,
                market_2_yes_price=kalshi_price,
                spread=spread,
                profit_potential=profit_potential,
                similarity_score=similarity,
                action=action
            )

        except Exception as e:
            logger.warning(f"Error analyzing pair: {e}")
            return None

    def scan_intra_market(self, platform: str = "polymarket") -> List[Dict]:
        """
        Find intra-market arbitrage (YES + NO < $1).

        This happens when there's inefficient pricing within a single market.
        """
        opportunities = []

        if platform == "polymarket":
            markets = self.poly_client.get_markets(limit=200)

            for market in markets:
                if len(market.prices) >= 2:
                    total = sum(market.prices.values())
                    if total < 0.98:  # 2% minimum profit
                        opportunities.append({
                            "market": market.question,
                            "prices": market.prices,
                            "total": total,
                            "profit": 1 - total,
                            "action": "Buy all outcomes"
                        })

        opportunities.sort(key=lambda x: x["profit"], reverse=True)
        return opportunities


def find_arbitrage() -> List[ArbitrageOpportunity]:
    """Convenience function to find arbitrage opportunities."""
    detector = ArbitrageDetector()
    return detector.find_opportunities()


if __name__ == "__main__":
    # Test arbitrage detection
    print("Scanning for arbitrage opportunities...")
    print("-" * 60)

    detector = ArbitrageDetector(min_spread=0.01)  # 1% minimum for testing
    opportunities = detector.find_opportunities(limit_per_platform=50)

    if opportunities:
        for i, opp in enumerate(opportunities[:10], 1):
            print(f"\n{i}. ARBITRAGE OPPORTUNITY")
            print(f"   Similarity: {opp.similarity_score:.1%}")
            print(f"   Polymarket: {opp.market_1_question[:60]}...")
            print(f"   Price: {opp.market_1_yes_price:.1%}")
            print(f"   Kalshi: {opp.market_2_question[:60]}...")
            print(f"   Price: {opp.market_2_yes_price:.1%}")
            print(f"   SPREAD: {opp.spread:.2%}")
            print(f"   PROFIT: {opp.profit_potential:.2%}")
            print(f"   ACTION: {opp.action}")
    else:
        print("No arbitrage opportunities found.")

    # Also check intra-market
    print("\n" + "=" * 60)
    print("Checking intra-market arbitrage...")
    intra = detector.scan_intra_market()

    if intra:
        for i, opp in enumerate(intra[:5], 1):
            print(f"\n{i}. {opp['market'][:50]}...")
            print(f"   Total price: {opp['total']:.1%}")
            print(f"   Profit: {opp['profit']:.2%}")
