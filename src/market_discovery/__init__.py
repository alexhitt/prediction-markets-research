"""
Market discovery module for real paper trading.

Provides:
- MarketScanner: Discover real markets from Polymarket/Kalshi
- ResolutionTracker: Monitor pending market resolutions
"""

from .market_scanner import MarketScanner, DiscoveredMarket
from .resolution_tracker import ResolutionTracker

__all__ = ["MarketScanner", "DiscoveredMarket", "ResolutionTracker"]
