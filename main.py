#!/usr/bin/env python3
"""
Prediction Markets AI System - Main Entry Point

Usage:
    # Run dashboard
    python main.py dashboard

    # Run data collector
    python main.py collect

    # Run arbitrage scan
    python main.py arbitrage

    # Initialize database
    python main.py init-db

    # Test API connections
    python main.py test
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def run_collector():
    """Run data collector."""
    from src.collectors.market_collector import run_collector
    run_collector()


def run_signals():
    """Run signal collection."""
    from src.collectors.signal_collector import SignalCollector

    print("Collecting alternative data signals...")
    print("-" * 60)

    collector = SignalCollector()
    stats = collector.collect_all()

    print(f"\nCollection completed at {stats['timestamp']}")
    print(f"Total signals: {stats['total_signals']}")

    for name, detector_stats in stats["detectors"].items():
        print(f"\n  {name}:")
        signals_count = detector_stats.get("signals_collected", 0)
        print(f"    Signals: {signals_count}")
        if "duration_seconds" in detector_stats:
            print(f"    Duration: {detector_stats['duration_seconds']:.1f}s")
        if "error" in detector_stats:
            print(f"    Error: {detector_stats['error']}")

    if stats["errors"]:
        print(f"\nErrors: {stats['errors']}")

    # Show recent signals
    print("\n" + "=" * 60)
    print("Recent Signals:")

    recent = collector.get_recent_signals(limit=10)
    for signal in recent:
        print(f"\n  {signal.name}")
        print(f"    Value: {signal.value:.2f}")
        print(f"    Direction: {signal.direction.value}")
        print(f"    Confidence: {signal.confidence:.0%}")


def run_arbitrage():
    """Run arbitrage detection."""
    from src.signals.arbitrage import ArbitrageDetector
    from src.clients.polymarket_client import PolymarketClient
    from src.clients.kalshi_client import KalshiClient

    print("Scanning for arbitrage opportunities...")
    print("-" * 60)

    detector = ArbitrageDetector(
        PolymarketClient(),
        KalshiClient(),
        min_spread=0.01  # 1% minimum
    )

    opportunities = detector.find_opportunities(limit_per_platform=100)

    if opportunities:
        print(f"\nFound {len(opportunities)} opportunities:\n")

        for i, opp in enumerate(opportunities[:10], 1):
            print(f"{i}. SPREAD: {opp.spread:.2%} | PROFIT: {opp.profit_potential:.2%}")
            print(f"   Polymarket: {opp.market_1_question[:50]}... @ {opp.market_1_yes_price:.1%}")
            print(f"   Kalshi: {opp.market_2_question[:50]}... @ {opp.market_2_yes_price:.1%}")
            print(f"   Action: {opp.action}")
            print()
    else:
        print("No arbitrage opportunities found.")

    # Check intra-market
    print("\n" + "=" * 60)
    print("Checking intra-market arbitrage (YES + NO < $1)...")

    intra = detector.scan_intra_market("polymarket")
    if intra:
        print(f"\nFound {len(intra)} intra-market opportunities:\n")
        for opp in intra[:5]:
            print(f"  {opp['market'][:50]}...")
            print(f"  Total: {opp['total']:.1%} | Profit: {opp['profit']:.2%}\n")
    else:
        print("No intra-market arbitrage found.")


def init_database():
    """Initialize the database."""
    from src.database.db import init_db
    init_db()
    print("Database initialized successfully!")


def test_connections():
    """Test API connections."""
    from src.clients.polymarket_client import PolymarketClient
    from src.clients.kalshi_client import KalshiClient

    print("Testing API connections...")
    print("-" * 40)

    # Test Polymarket
    print("\n📊 Polymarket:")
    try:
        client = PolymarketClient()
        markets = client.get_markets(limit=3)
        print(f"  ✅ Connected - {len(markets)} markets fetched")
        for m in markets[:3]:
            print(f"     - {m.question[:50]}...")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test Kalshi
    print("\n📊 Kalshi:")
    try:
        client = KalshiClient()
        markets = client.get_markets(limit=3)
        print(f"  ✅ Connected - {len(markets)} markets fetched")
        for m in markets[:3]:
            print(f"     - {m.title[:50]}...")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print("\n" + "-" * 40)
    print("Connection tests complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prediction Markets AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  dashboard    Launch the Streamlit monitoring dashboard
  collect      Run the data collector (continuous)
  arbitrage    Scan for arbitrage opportunities
  signals      Collect alternative data signals
  init-db      Initialize the database
  test         Test API connections

Examples:
  python main.py dashboard     # Start the dashboard
  python main.py arbitrage     # Quick arbitrage scan
  python main.py signals       # Collect signals
  python main.py test          # Test connections
        """
    )

    parser.add_argument(
        "command",
        choices=["dashboard", "collect", "arbitrage", "signals", "init-db", "test"],
        help="Command to run"
    )

    args = parser.parse_args()

    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "collect":
        run_collector()
    elif args.command == "arbitrage":
        run_arbitrage()
    elif args.command == "signals":
        run_signals()
    elif args.command == "init-db":
        init_database()
    elif args.command == "test":
        test_connections()


if __name__ == "__main__":
    main()
