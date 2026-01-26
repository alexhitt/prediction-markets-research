#!/usr/bin/env python3
"""
Test script for the real paper trading system.

Runs a single real trading round to verify:
1. Market discovery from Polymarket/Kalshi
2. Research agents produce probability estimates
3. Paper bets are placed when edge is found
4. Pending resolutions are tracked
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

def test_imports():
    """Test that all modules import correctly."""
    logger.info("Testing imports...")

    try:
        from src.database.models import Base, ResearchEstimate, BotMemory, PendingResolution
        logger.info("  Database models: OK")
    except ImportError as e:
        logger.error(f"  Database models: FAILED - {e}")
        return False

    try:
        from src.market_discovery import MarketScanner, ResolutionTracker
        logger.info("  Market discovery: OK")
    except ImportError as e:
        logger.error(f"  Market discovery: FAILED - {e}")
        return False

    try:
        from src.research import get_researcher_for_bot, RESEARCHER_REGISTRY
        logger.info(f"  Research agents: OK ({len(RESEARCHER_REGISTRY)} strategies)")
    except ImportError as e:
        logger.error(f"  Research agents: FAILED - {e}")
        return False

    try:
        from src.learning import ResolutionWorker, CalibrationTracker
        logger.info("  Learning system: OK")
    except ImportError as e:
        logger.error(f"  Learning system: FAILED - {e}")
        return False

    try:
        from src.autonomous.tournament import BotTournament, HAS_REAL_TRADING
        logger.info(f"  Tournament: OK (real trading: {HAS_REAL_TRADING})")
    except ImportError as e:
        logger.error(f"  Tournament: FAILED - {e}")
        return False

    return True


def test_market_scanner():
    """Test market discovery from Polymarket."""
    logger.info("Testing market scanner...")

    try:
        from src.clients.polymarket_client import PolymarketClient
        from src.market_discovery import MarketScanner

        poly_client = PolymarketClient()
        scanner = MarketScanner(poly_client=poly_client)

        # Scan with relaxed filters for testing
        markets = scanner.scan_for_opportunities(
            min_liquidity=1000,  # Lower threshold for testing
            max_days_to_resolution=365,
            min_volume_24h=100,
            limit_per_platform=10
        )

        logger.info(f"  Found {len(markets)} markets")

        if markets:
            for m in markets[:3]:
                logger.info(f"    - {m.question[:60]}... ({m.platform}, ${m.liquidity:,.0f} liq)")
            return markets
        else:
            logger.warning("  No markets found - API may be unavailable")
            return []

    except Exception as e:
        logger.error(f"  Market scanner failed: {e}")
        return []


async def test_researcher(market):
    """Test a researcher on a real market."""
    logger.info("Testing researcher...")

    try:
        from src.research import get_researcher_for_bot

        # Test conservative researcher
        researcher = get_researcher_for_bot("conservative_value")
        logger.info(f"  Using {researcher.researcher_type} researcher")

        estimate = await researcher.research_market(
            market_id=market.market_id,
            platform=market.platform,
            question=market.question,
            description=market.description,
            current_price=market.current_yes_price,
            category=market.category,
            extra_data=market.extra_data
        )

        if estimate:
            logger.info(f"  Estimate: {estimate.estimated_probability:.0%} (conf: {estimate.confidence:.0%})")
            logger.info(f"  Edge: {estimate.edge():.1%}")
            logger.info(f"  Direction: {estimate.direction().upper()}")
            logger.info(f"  Reasoning: {estimate.reasoning[:100]}...")
            return estimate
        else:
            logger.warning("  No estimate produced")
            return None

    except Exception as e:
        logger.error(f"  Researcher failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_real_trading_round():
    """Test a full real trading round."""
    logger.info("Testing real trading round...")

    try:
        from src.clients.polymarket_client import PolymarketClient
        from src.autonomous.tournament import BotTournament
        from src.database.models import Base

        # Create in-memory database for testing
        engine = create_engine("sqlite:///data/test_trading.db")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db_session = Session()

        # Initialize clients
        poly_client = PolymarketClient()

        # Initialize tournament with default bots
        tournament = BotTournament()
        tournament.add_default_bots()

        logger.info(f"  Tournament has {len(tournament.bots)} bots")

        # Run real trading round
        result = await tournament.real_trading_round(
            poly_client=poly_client,
            kalshi_client=None,  # Skip Kalshi for now
            db_session=db_session,
            min_edge=0.03,  # Lower threshold for testing
            min_confidence=0.4,
            max_markets_per_round=5
        )

        logger.info(f"  Mode: {result.get('mode')}")
        logger.info(f"  Markets scanned: {result.get('markets_scanned', 0)}")
        logger.info(f"  Markets researched: {result.get('markets_researched', 0)}")
        logger.info(f"  Bets placed: {result.get('bets_placed', 0)}")
        logger.info(f"  Estimates saved: {result.get('estimates_saved', 0)}")

        # Show bot results
        for bot_id, stats in result.get("bot_results", {}).items():
            if stats.get("bets_placed", 0) > 0:
                logger.info(f"    {bot_id}: {stats['bets_placed']} bets, {stats['total_edge']:.1%} total edge")

        db_session.close()
        return result

    except Exception as e:
        logger.error(f"  Real trading round failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("REAL PAPER TRADING SYSTEM - TEST SUITE")
    logger.info("=" * 60)

    # Test 1: Imports
    if not test_imports():
        logger.error("Import tests failed - fix imports before continuing")
        return

    logger.info("")

    # Test 2: Market scanner
    markets = test_market_scanner()

    logger.info("")

    # Test 3: Researcher (if we have markets)
    if markets:
        await test_researcher(markets[0])
    else:
        logger.warning("Skipping researcher test - no markets available")

    logger.info("")

    # Test 4: Full real trading round
    await test_real_trading_round()

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
