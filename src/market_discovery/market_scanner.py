"""
Market Scanner for real paper trading.

Discovers real markets from Polymarket and Kalshi, applying filters
for liquidity, time to resolution, and categories to find opportunities
suitable for bot research and paper trading.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger


@dataclass(frozen=True)
class DiscoveredMarket:
    """
    A market discovered for potential paper trading.

    Immutable dataclass containing all relevant market information
    needed for research agents to form probability estimates.
    """
    platform: str  # polymarket, kalshi
    market_id: str
    question: str
    description: str
    category: str

    # Current market state
    current_yes_price: float  # 0-1 scale
    current_no_price: float
    spread: float

    # Volume/liquidity
    volume_24h: float
    total_volume: float
    liquidity: float
    open_interest: float

    # Timing
    end_date: Optional[datetime]
    days_to_resolution: Optional[int]

    # Status
    status: str  # open, closed
    resolved: bool

    # Extra data for research
    extra_data: Dict[str, Any]


class MarketScanner:
    """
    Scans Polymarket and Kalshi for markets suitable for paper trading.

    Applies filters to find markets with:
    - Sufficient liquidity (default $10k+)
    - Reasonable time to resolution (default 180 days max)
    - Active trading
    - Binary outcomes
    """

    def __init__(
        self,
        poly_client=None,
        kalshi_client=None,
        db_session=None
    ):
        self.poly_client = poly_client
        self.kalshi_client = kalshi_client
        self.db = db_session

    def scan_for_opportunities(
        self,
        min_liquidity: float = 10000.0,
        max_days_to_resolution: int = 180,
        min_days_to_resolution: int = 1,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
        min_volume_24h: float = 1000.0,
        max_spread: float = 0.10,
        limit_per_platform: int = 100
    ) -> List[DiscoveredMarket]:
        """
        Scan for markets meeting the specified criteria.

        Args:
            min_liquidity: Minimum market liquidity in USD
            max_days_to_resolution: Maximum days until market resolves
            min_days_to_resolution: Minimum days (avoid imminent resolution)
            categories: If provided, only include these categories
            exclude_categories: If provided, exclude these categories
            min_volume_24h: Minimum 24h volume
            max_spread: Maximum bid-ask spread (0.10 = 10%)
            limit_per_platform: Max markets to fetch per platform

        Returns:
            List of DiscoveredMarket objects meeting all criteria
        """
        markets = []

        # Scan Polymarket
        if self.poly_client:
            poly_markets = self._scan_polymarket(
                min_liquidity=min_liquidity,
                max_days_to_resolution=max_days_to_resolution,
                min_days_to_resolution=min_days_to_resolution,
                categories=categories,
                exclude_categories=exclude_categories,
                min_volume_24h=min_volume_24h,
                max_spread=max_spread,
                limit=limit_per_platform
            )
            markets.extend(poly_markets)
            logger.info(f"Found {len(poly_markets)} Polymarket opportunities")

        # Scan Kalshi
        if self.kalshi_client:
            kalshi_markets = self._scan_kalshi(
                min_liquidity=min_liquidity,
                max_days_to_resolution=max_days_to_resolution,
                min_days_to_resolution=min_days_to_resolution,
                categories=categories,
                exclude_categories=exclude_categories,
                min_volume_24h=min_volume_24h,
                max_spread=max_spread,
                limit=limit_per_platform
            )
            markets.extend(kalshi_markets)
            logger.info(f"Found {len(kalshi_markets)} Kalshi opportunities")

        # Sort by liquidity (most liquid first)
        markets.sort(key=lambda m: m.liquidity, reverse=True)

        logger.info(f"Total opportunities found: {len(markets)}")
        return markets

    def _scan_polymarket(
        self,
        min_liquidity: float,
        max_days_to_resolution: int,
        min_days_to_resolution: int,
        categories: Optional[List[str]],
        exclude_categories: Optional[List[str]],
        min_volume_24h: float,
        max_spread: float,
        limit: int
    ) -> List[DiscoveredMarket]:
        """Scan Polymarket for opportunities."""
        discovered = []

        try:
            # Fetch active markets sorted by volume
            raw_markets = self.poly_client.get_markets(
                limit=limit,
                active=True,
                closed=False,
                order="volume",
                ascending=False
            )

            now = datetime.utcnow()

            for market in raw_markets:
                # Skip resolved markets
                if market.resolved:
                    continue

                # Category filters
                if categories and market.category.lower() not in [c.lower() for c in categories]:
                    continue
                if exclude_categories and market.category.lower() in [c.lower() for c in exclude_categories]:
                    continue

                # Liquidity filter
                if market.liquidity < min_liquidity:
                    continue

                # Volume filter
                if market.volume < min_volume_24h:
                    continue

                # Time to resolution filter
                days_to_resolution = None
                if market.end_date:
                    days_to_resolution = (market.end_date - now).days
                    if days_to_resolution < min_days_to_resolution:
                        continue
                    if days_to_resolution > max_days_to_resolution:
                        continue

                # Extract prices
                yes_price = market.prices.get("Yes", market.prices.get("yes", 0.5))
                no_price = market.prices.get("No", market.prices.get("no", 0.5))

                # If prices aren't available from tokens, try to calculate
                if yes_price == 0.5 and no_price == 0.5:
                    tokens = market.extra_data.get("tokens", [])
                    if tokens:
                        for token in tokens:
                            outcome = token.get("outcome", "").lower()
                            price = float(token.get("price", 0.5))
                            if outcome == "yes":
                                yes_price = price
                            elif outcome == "no":
                                no_price = price

                # Calculate spread (approximate from prices)
                spread = abs(1 - yes_price - no_price)
                if spread > max_spread:
                    continue

                discovered.append(DiscoveredMarket(
                    platform="polymarket",
                    market_id=market.id,
                    question=market.question,
                    description=market.description,
                    category=market.category,
                    current_yes_price=yes_price,
                    current_no_price=no_price,
                    spread=spread,
                    volume_24h=market.volume,
                    total_volume=market.volume,  # Polymarket doesn't separate
                    liquidity=market.liquidity,
                    open_interest=0,  # Not provided by Polymarket
                    end_date=market.end_date,
                    days_to_resolution=days_to_resolution,
                    status=market.status,
                    resolved=market.resolved,
                    extra_data={
                        "slug": market.extra_data.get("slug"),
                        "tokens": market.extra_data.get("tokens", [])
                    }
                ))

        except Exception as e:
            logger.error(f"Error scanning Polymarket: {e}")

        return discovered

    def _scan_kalshi(
        self,
        min_liquidity: float,
        max_days_to_resolution: int,
        min_days_to_resolution: int,
        categories: Optional[List[str]],
        exclude_categories: Optional[List[str]],
        min_volume_24h: float,
        max_spread: float,
        limit: int
    ) -> List[DiscoveredMarket]:
        """Scan Kalshi for opportunities."""
        discovered = []

        try:
            # Fetch open markets
            raw_markets = self.kalshi_client.get_markets(
                limit=limit,
                status="open"
            )

            now = datetime.utcnow()

            for market in raw_markets:
                # Category filters
                if categories and market.category.lower() not in [c.lower() for c in categories]:
                    continue
                if exclude_categories and market.category.lower() in [c.lower() for c in exclude_categories]:
                    continue

                # Volume filter (Kalshi volume is in contracts, not USD)
                # Approximate: volume * average_price * 1.00 per contract
                estimated_volume_usd = market.volume * 0.5  # Rough estimate
                if estimated_volume_usd < min_volume_24h:
                    continue

                # Time to resolution filter
                days_to_resolution = None
                if market.close_time:
                    days_to_resolution = (market.close_time - now).days
                    if days_to_resolution < min_days_to_resolution:
                        continue
                    if days_to_resolution > max_days_to_resolution:
                        continue

                # Convert prices from cents to 0-1 scale
                yes_price = (market.yes_bid + market.yes_ask) / 200 if market.yes_bid and market.yes_ask else 0.5
                no_price = 1 - yes_price

                # Calculate spread
                spread = 0
                if market.yes_bid and market.yes_ask:
                    spread = (market.yes_ask - market.yes_bid) / 100
                    if spread > max_spread:
                        continue

                # Estimate liquidity from open interest
                estimated_liquidity = market.open_interest * 0.5  # Rough estimate
                if estimated_liquidity < min_liquidity:
                    continue

                discovered.append(DiscoveredMarket(
                    platform="kalshi",
                    market_id=market.ticker,
                    question=f"{market.title}: {market.subtitle}",
                    description=market.extra_data.get("rules_primary", ""),
                    category=market.category,
                    current_yes_price=yes_price,
                    current_no_price=no_price,
                    spread=spread,
                    volume_24h=estimated_volume_usd,
                    total_volume=estimated_volume_usd,
                    liquidity=estimated_liquidity,
                    open_interest=market.open_interest,
                    end_date=market.close_time,
                    days_to_resolution=days_to_resolution,
                    status=market.status,
                    resolved=market.result is not None,
                    extra_data={
                        "event_ticker": market.event_ticker,
                        "settlement_value": market.extra_data.get("settlement_value"),
                        "rules_secondary": market.extra_data.get("rules_secondary")
                    }
                ))

        except Exception as e:
            logger.error(f"Error scanning Kalshi: {e}")

        return discovered

    def get_market_details(self, platform: str, market_id: str) -> Optional[DiscoveredMarket]:
        """
        Get detailed information for a specific market.

        Args:
            platform: polymarket or kalshi
            market_id: Platform-specific market identifier

        Returns:
            DiscoveredMarket with current data, or None if not found
        """
        if platform == "polymarket" and self.poly_client:
            return self._get_polymarket_details(market_id)
        elif platform == "kalshi" and self.kalshi_client:
            return self._get_kalshi_details(market_id)
        return None

    def _get_polymarket_details(self, market_id: str) -> Optional[DiscoveredMarket]:
        """Get details for a specific Polymarket market."""
        try:
            market = self.poly_client.get_market(market_id)
            if not market:
                return None

            now = datetime.utcnow()
            days_to_resolution = None
            if market.end_date:
                days_to_resolution = (market.end_date - now).days

            yes_price = market.prices.get("Yes", market.prices.get("yes", 0.5))
            no_price = market.prices.get("No", market.prices.get("no", 0.5))
            spread = abs(1 - yes_price - no_price)

            return DiscoveredMarket(
                platform="polymarket",
                market_id=market.id,
                question=market.question,
                description=market.description,
                category=market.category,
                current_yes_price=yes_price,
                current_no_price=no_price,
                spread=spread,
                volume_24h=market.volume,
                total_volume=market.volume,
                liquidity=market.liquidity,
                open_interest=0,
                end_date=market.end_date,
                days_to_resolution=days_to_resolution,
                status=market.status,
                resolved=market.resolved,
                extra_data={
                    "resolution": market.resolution,
                    "tokens": market.extra_data.get("tokens", [])
                }
            )
        except Exception as e:
            logger.error(f"Error getting Polymarket details for {market_id}: {e}")
            return None

    def _get_kalshi_details(self, ticker: str) -> Optional[DiscoveredMarket]:
        """Get details for a specific Kalshi market."""
        try:
            market = self.kalshi_client.get_market(ticker)
            if not market:
                return None

            now = datetime.utcnow()
            days_to_resolution = None
            if market.close_time:
                days_to_resolution = (market.close_time - now).days

            yes_price = (market.yes_bid + market.yes_ask) / 200 if market.yes_bid and market.yes_ask else 0.5
            no_price = 1 - yes_price
            spread = (market.yes_ask - market.yes_bid) / 100 if market.yes_bid and market.yes_ask else 0

            return DiscoveredMarket(
                platform="kalshi",
                market_id=market.ticker,
                question=f"{market.title}: {market.subtitle}",
                description=market.extra_data.get("rules_primary", ""),
                category=market.category,
                current_yes_price=yes_price,
                current_no_price=no_price,
                spread=spread,
                volume_24h=market.volume * 0.5,
                total_volume=market.volume * 0.5,
                liquidity=market.open_interest * 0.5,
                open_interest=market.open_interest,
                end_date=market.close_time,
                days_to_resolution=days_to_resolution,
                status=market.status,
                resolved=market.result is not None,
                extra_data={
                    "result": market.result,
                    "event_ticker": market.event_ticker
                }
            )
        except Exception as e:
            logger.error(f"Error getting Kalshi details for {ticker}: {e}")
            return None

    def check_market_resolution(self, platform: str, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a market has resolved and get the outcome.

        Args:
            platform: polymarket or kalshi
            market_id: Platform-specific market identifier

        Returns:
            Dict with resolution info if resolved, None otherwise
            Example: {"resolved": True, "outcome": "yes", "value": 1.0, "resolved_at": datetime}
        """
        market = self.get_market_details(platform, market_id)

        if not market:
            return None

        if not market.resolved:
            return {"resolved": False}

        # Get resolution outcome
        resolution = market.extra_data.get("resolution") or market.extra_data.get("result")

        if resolution:
            outcome_value = 1.0 if resolution.lower() == "yes" else 0.0
            return {
                "resolved": True,
                "outcome": resolution.lower(),
                "value": outcome_value,
                "resolved_at": datetime.utcnow()  # Approximate
            }

        return {"resolved": False}
