"""
Market Data Collector

Scheduled collection of market data from Polymarket and Kalshi
for historical analysis and signal generation.
"""

import time
from datetime import datetime
from typing import Optional, List
from loguru import logger

from src.clients.polymarket_client import PolymarketClient
from src.clients.kalshi_client import KalshiClient
from src.database.db import db_manager, get_session
from src.database.models import Market, MarketSnapshot
from src.signals.arbitrage import ArbitrageDetector


class MarketCollector:
    """
    Collects and stores market data from prediction market platforms.

    Usage:
        collector = MarketCollector()

        # Run once
        collector.collect_all()

        # Run continuously
        collector.run_continuous(interval_seconds=60)
    """

    def __init__(
        self,
        polymarket_client: Optional[PolymarketClient] = None,
        kalshi_client: Optional[KalshiClient] = None
    ):
        self.poly_client = polymarket_client or PolymarketClient()
        self.kalshi_client = kalshi_client or KalshiClient()
        self.arb_detector = ArbitrageDetector(self.poly_client, self.kalshi_client)

    def collect_all(self) -> dict:
        """
        Collect data from all platforms.

        Returns dict with collection statistics.
        """
        stats = {
            "timestamp": datetime.utcnow(),
            "polymarket_markets": 0,
            "polymarket_snapshots": 0,
            "kalshi_markets": 0,
            "kalshi_snapshots": 0,
            "arbitrage_opportunities": 0,
            "errors": []
        }

        try:
            poly_stats = self.collect_polymarket()
            stats["polymarket_markets"] = poly_stats["markets"]
            stats["polymarket_snapshots"] = poly_stats["snapshots"]
        except Exception as e:
            logger.error(f"Polymarket collection error: {e}")
            stats["errors"].append(f"Polymarket: {str(e)}")

        try:
            kalshi_stats = self.collect_kalshi()
            stats["kalshi_markets"] = kalshi_stats["markets"]
            stats["kalshi_snapshots"] = kalshi_stats["snapshots"]
        except Exception as e:
            logger.error(f"Kalshi collection error: {e}")
            stats["errors"].append(f"Kalshi: {str(e)}")

        try:
            arb_stats = self.collect_arbitrage()
            stats["arbitrage_opportunities"] = arb_stats["opportunities"]
        except Exception as e:
            logger.error(f"Arbitrage detection error: {e}")
            stats["errors"].append(f"Arbitrage: {str(e)}")

        logger.info(f"Collection complete: {stats}")
        return stats

    def collect_polymarket(self, limit: int = 200) -> dict:
        """Collect Polymarket data."""
        logger.info("Collecting Polymarket data...")

        markets = self.poly_client.get_markets(limit=limit)
        stats = {"markets": 0, "snapshots": 0}

        with get_session() as session:
            for market in markets:
                try:
                    # Add/update market
                    db_market = db_manager.add_market(
                        session,
                        platform="polymarket",
                        platform_id=market.id,
                        question=market.question,
                        description=market.description,
                        category=market.category,
                        status=market.status,
                        end_date=market.end_date,
                        resolution=market.resolution if market.resolved else None,
                        extra_data=market.extra_data
                    )
                    session.flush()  # Get the ID
                    stats["markets"] += 1

                    # Add snapshot
                    yes_price = None
                    no_price = None

                    if market.prices:
                        for outcome, price in market.prices.items():
                            if outcome.lower() in ['yes', 'true', '1']:
                                yes_price = price
                            elif outcome.lower() in ['no', 'false', '0']:
                                no_price = price

                        if yes_price is None and len(market.prices) > 0:
                            yes_price = list(market.prices.values())[0]
                        if no_price is None and yes_price is not None:
                            no_price = 1 - yes_price

                    db_manager.add_snapshot(
                        session,
                        market_id=db_market.id,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume_24h=market.volume,
                        liquidity=market.liquidity
                    )
                    stats["snapshots"] += 1

                except Exception as e:
                    logger.warning(f"Error storing Polymarket market: {e}")

        logger.info(f"Polymarket: {stats['markets']} markets, {stats['snapshots']} snapshots")
        return stats

    def collect_kalshi(self, limit: int = 200) -> dict:
        """Collect Kalshi data."""
        logger.info("Collecting Kalshi data...")

        markets = self.kalshi_client.get_markets(limit=limit)
        stats = {"markets": 0, "snapshots": 0}

        with get_session() as session:
            for market in markets:
                try:
                    # Add/update market
                    db_market = db_manager.add_market(
                        session,
                        platform="kalshi",
                        platform_id=market.ticker,
                        question=market.title,
                        description=market.subtitle,
                        category=market.category,
                        status=market.status,
                        end_date=market.close_time,
                        resolution=market.result,
                        extra_data=market.extra_data
                    )
                    session.flush()
                    stats["markets"] += 1

                    # Convert Kalshi cents to 0-1 scale
                    yes_price = market.yes_bid / 100 if market.yes_bid else None
                    yes_ask = market.yes_ask / 100 if market.yes_ask else None
                    no_bid = market.no_bid / 100 if market.no_bid else None
                    no_ask = market.no_ask / 100 if market.no_ask else None

                    spread = None
                    if market.yes_bid and market.yes_ask:
                        spread = (market.yes_ask - market.yes_bid) / 100

                    db_manager.add_snapshot(
                        session,
                        market_id=db_market.id,
                        yes_price=(yes_price + yes_ask) / 2 if yes_price and yes_ask else yes_price,
                        no_price=(no_bid + no_ask) / 2 if no_bid and no_ask else no_bid,
                        yes_bid=yes_price,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        spread=spread,
                        total_volume=market.volume,
                        open_interest=market.open_interest
                    )
                    stats["snapshots"] += 1

                except Exception as e:
                    logger.warning(f"Error storing Kalshi market: {e}")

        logger.info(f"Kalshi: {stats['markets']} markets, {stats['snapshots']} snapshots")
        return stats

    def collect_arbitrage(self, min_spread: float = 0.02) -> dict:
        """Detect and store arbitrage opportunities."""
        logger.info("Scanning for arbitrage...")

        self.arb_detector.min_spread = min_spread
        opportunities = self.arb_detector.find_opportunities(limit_per_platform=100)

        stats = {"opportunities": 0}

        with get_session() as session:
            for opp in opportunities:
                try:
                    db_manager.add_arbitrage(
                        session,
                        market_1_platform=opp.platform_1,
                        market_1_id=opp.market_1_id,
                        market_1_question=opp.market_1_question,
                        market_1_price=opp.market_1_yes_price,
                        market_2_platform=opp.platform_2,
                        market_2_id=opp.market_2_id,
                        market_2_question=opp.market_2_question,
                        market_2_price=opp.market_2_yes_price,
                        spread=opp.spread,
                        profit_potential=opp.profit_potential,
                        confidence=opp.similarity_score
                    )
                    stats["opportunities"] += 1

                except Exception as e:
                    logger.warning(f"Error storing arbitrage: {e}")

        logger.info(f"Arbitrage: {stats['opportunities']} opportunities found")
        return stats

    def run_continuous(self, interval_seconds: int = 60):
        """
        Run collection continuously.

        Args:
            interval_seconds: Seconds between collection runs
        """
        logger.info(f"Starting continuous collection (interval: {interval_seconds}s)")

        while True:
            try:
                start_time = time.time()
                stats = self.collect_all()

                elapsed = time.time() - start_time
                logger.info(f"Collection took {elapsed:.1f}s")

                # Sleep for remaining time
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Collection error: {e}")
                time.sleep(interval_seconds)


def run_collector():
    """Entry point for running the collector."""
    from src.database.db import init_db

    # Initialize database
    init_db()

    # Create and run collector
    collector = MarketCollector()
    collector.run_continuous(interval_seconds=60)


if __name__ == "__main__":
    run_collector()
