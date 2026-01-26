#!/usr/bin/env python3
"""
Resolution Worker Runner

Standalone script to run the resolution worker that checks for
resolved markets and triggers learning updates for bots.

Can be run as:
1. One-shot check: python run_resolution_worker.py --once
2. Continuous daemon: python run_resolution_worker.py
3. Cron job: */5 * * * * cd /path/to/project && python run_resolution_worker.py --once

Environment variables:
- RESOLUTION_CHECK_INTERVAL: Seconds between checks (default: 300)
- DATABASE_URL: SQLite database path (default: data/paper_trading.db)
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
)
logger.add(
    log_dir / "resolution_worker.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


class ResolutionWorkerRunner:
    """Manages the resolution worker lifecycle."""

    def __init__(self, db_path: str = "data/paper_trading.db"):
        self.db_path = db_path
        self.worker = None
        self._shutdown_requested = False

    def _setup_database(self):
        """Initialize database connection."""
        from src.database.models import Base

        db_file = Path(__file__).parent / self.db_path
        db_file.parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()

    def _setup_scanner(self):
        """Initialize market scanner for resolution checking."""
        from src.market_discovery import MarketScanner

        poly_client = None
        kalshi_client = None

        try:
            from src.clients.polymarket_client import PolymarketClient
            poly_client = PolymarketClient()
            logger.info("Polymarket client initialized")
        except Exception as e:
            logger.warning(f"Polymarket client unavailable: {e}")

        try:
            from src.clients.kalshi_client import KalshiClient
            kalshi_client = KalshiClient()
            logger.info("Kalshi client initialized")
        except Exception as e:
            logger.warning(f"Kalshi client unavailable: {e}")

        scanner = MarketScanner(
            poly_client=poly_client,
            kalshi_client=kalshi_client
        )
        return scanner

    def _setup_worker(self, db_session, scanner):
        """Initialize the resolution worker."""
        from src.market_discovery.resolution_tracker import ResolutionTracker
        from src.learning.resolution_worker import ResolutionWorker

        tracker = ResolutionTracker(
            db_session=db_session,
            market_scanner=scanner
        )

        check_interval = int(os.environ.get("RESOLUTION_CHECK_INTERVAL", 300))

        worker = ResolutionWorker(
            db_session=db_session,
            resolution_tracker=tracker,
            check_interval_seconds=check_interval
        )

        return worker

    async def run_once(self):
        """Run a single resolution check."""
        logger.info("Running one-shot resolution check...")

        db_session = self._setup_database()
        scanner = self._setup_scanner()

        try:
            worker = self._setup_worker(db_session, scanner)
            result = await worker.check_and_learn()

            logger.info(f"Resolution check complete:")
            logger.info(f"  Resolutions found: {result['resolutions_found']}")
            logger.info(f"  Learning updates: {result['learning_updates']}")
            logger.info(f"  Bots updated: {result['bots_updated']}")

            if result['errors']:
                for error in result['errors']:
                    logger.error(f"  Error: {error}")

            return result

        finally:
            db_session.close()

    async def run_continuous(self):
        """Run the worker continuously."""
        check_interval = int(os.environ.get("RESOLUTION_CHECK_INTERVAL", 300))
        logger.info(f"Starting continuous resolution worker (interval: {check_interval}s)")

        db_session = self._setup_database()
        scanner = self._setup_scanner()

        try:
            self.worker = self._setup_worker(db_session, scanner)

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating shutdown...")
                self._shutdown_requested = True
                if self.worker:
                    self.worker.stop()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Start the worker
            await self.worker.start()

        finally:
            db_session.close()
            logger.info("Resolution worker shutdown complete")

    def get_status(self):
        """Get current worker status."""
        if not self.worker:
            return {"status": "not_running"}

        return self.worker.get_worker_stats()


def print_usage():
    """Print usage information."""
    print("""
Resolution Worker Runner

Usage:
    python run_resolution_worker.py [options]

Options:
    --once      Run a single resolution check and exit
    --status    Show pending resolutions status and exit
    --help      Show this help message

Environment Variables:
    RESOLUTION_CHECK_INTERVAL   Seconds between checks (default: 300)
    DATABASE_URL               Database path (default: data/paper_trading.db)

Examples:
    # Run continuous worker
    python run_resolution_worker.py

    # Run single check (for cron)
    python run_resolution_worker.py --once

    # Check status
    python run_resolution_worker.py --status

Cron Setup (check every 5 minutes):
    */5 * * * * cd /path/to/prediction-markets-research && python run_resolution_worker.py --once >> logs/cron.log 2>&1
""")


def show_status():
    """Show current pending resolutions status."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database.models import Base, PendingResolution, ResearchEstimate

    db_file = Path(__file__).parent / "data" / "paper_trading.db"
    if not db_file.exists():
        print("Database not found. Run a trading round first.")
        return

    engine = create_engine(f"sqlite:///{db_file}")
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        pending = db.query(PendingResolution).filter_by(status="pending").all()
        resolved = db.query(PendingResolution).filter_by(status="resolved").all()
        estimates = db.query(ResearchEstimate).count()
        resolved_estimates = db.query(ResearchEstimate).filter(
            ResearchEstimate.actual_outcome.isnot(None)
        ).count()

        print("\n=== Resolution Worker Status ===\n")
        print(f"Pending Resolutions:  {len(pending)}")
        print(f"Resolved Resolutions: {len(resolved)}")
        print(f"Total Estimates:      {estimates}")
        print(f"Resolved Estimates:   {resolved_estimates}")

        if pending:
            print("\n--- Pending Markets ---")
            for p in pending[:10]:
                days_waiting = (datetime.utcnow() - p.created_at).days if p.created_at else 0
                print(f"  [{p.platform}] {p.market_question[:50]}... ({days_waiting}d waiting)")

            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")

    finally:
        db.close()


async def main():
    """Main entry point."""
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_usage()
        return

    if "--status" in args:
        show_status()
        return

    runner = ResolutionWorkerRunner()

    if "--once" in args:
        await runner.run_once()
    else:
        await runner.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
