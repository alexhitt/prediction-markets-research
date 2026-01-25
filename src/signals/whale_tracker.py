"""
Whale Tracker Signal Detector

Monitors large trades on Polymarket to detect whale activity.
Research shows whale trades often precede price movements.

Features:
- Threshold: >$10k positions considered whale activity
- Copy-trade discovery: Track wallet performance to find smart money
- Wallet leaderboard: Rank wallets by win rate
- Smart money signals: Follow high-performing wallets
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

import requests
from loguru import logger

from src.signals.base import (
    BaseSignalDetector,
    SignalDirection,
    SignalResult,
    SignalStrength,
    classify_signal_strength,
)


# Whale thresholds
WHALE_THRESHOLD_USD = 10000  # $10k minimum for whale
LARGE_WHALE_THRESHOLD_USD = 50000  # $50k for large whale
MEGA_WHALE_THRESHOLD_USD = 100000  # $100k for mega whale

# Data directory for whale tracking persistence
WHALE_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "whales"

# Known whale wallets (curated list - would be populated from research)
KNOWN_WHALES = {
    # Example: "0x1234...": {"label": "Smart Money 1", "track_record": 0.65}
}


@dataclass
class WalletPerformance:
    """Tracks a wallet's trading performance."""
    address: str
    label: Optional[str] = None
    total_trades: int = 0
    resolved_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    first_seen: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    recent_trades: List[Dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.resolved_trades == 0:
            return 0.0
        return self.wins / self.resolved_trades

    @property
    def roi(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.total_pnl / self.total_volume

    @property
    def is_smart_money(self) -> bool:
        """Wallet qualifies as smart money if good win rate with enough trades."""
        return self.resolved_trades >= 10 and self.win_rate >= 0.55

    def to_dict(self) -> Dict:
        return {
            "address": self.address,
            "label": self.label,
            "total_trades": self.total_trades,
            "resolved_trades": self.resolved_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_volume": self.total_volume,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "roi": self.roi,
            "is_smart_money": self.is_smart_money,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
        }


class WalletLeaderboard:
    """Manages wallet performance tracking and rankings."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or WHALE_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wallets: Dict[str, WalletPerformance] = {}
        self._load_data()

    def _load_data(self):
        """Load wallet data from disk."""
        data_file = self.data_dir / "wallet_leaderboard.json"
        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    for wallet_data in data.get("wallets", []):
                        perf = WalletPerformance(
                            address=wallet_data["address"],
                            label=wallet_data.get("label"),
                            total_trades=wallet_data.get("total_trades", 0),
                            resolved_trades=wallet_data.get("resolved_trades", 0),
                            wins=wallet_data.get("wins", 0),
                            losses=wallet_data.get("losses", 0),
                            total_volume=wallet_data.get("total_volume", 0),
                            total_pnl=wallet_data.get("total_pnl", 0),
                        )
                        if wallet_data.get("first_seen"):
                            perf.first_seen = datetime.fromisoformat(wallet_data["first_seen"])
                        if wallet_data.get("last_trade"):
                            perf.last_trade = datetime.fromisoformat(wallet_data["last_trade"])
                        self.wallets[perf.address] = perf
            except Exception as e:
                logger.error(f"Failed to load wallet data: {e}")

    def _save_data(self):
        """Save wallet data to disk."""
        data_file = self.data_dir / "wallet_leaderboard.json"
        try:
            data = {
                "wallets": [w.to_dict() for w in self.wallets.values()],
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save wallet data: {e}")

    def record_trade(
        self,
        address: str,
        usd_value: float,
        side: str,
        market_id: str,
        price: float
    ):
        """Record a new trade for a wallet."""
        if address not in self.wallets:
            self.wallets[address] = WalletPerformance(
                address=address,
                first_seen=datetime.utcnow()
            )

        wallet = self.wallets[address]
        wallet.total_trades += 1
        wallet.total_volume += usd_value
        wallet.last_trade = datetime.utcnow()

        # Keep last 20 trades
        wallet.recent_trades.append({
            "market_id": market_id,
            "side": side,
            "usd_value": usd_value,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
            "resolved": False,
        })
        wallet.recent_trades = wallet.recent_trades[-20:]

        self._save_data()

    def resolve_trade(self, address: str, market_id: str, won: bool, pnl: float):
        """Resolve a trade outcome for a wallet."""
        if address not in self.wallets:
            return

        wallet = self.wallets[address]
        wallet.resolved_trades += 1
        wallet.total_pnl += pnl

        if won:
            wallet.wins += 1
        else:
            wallet.losses += 1

        # Update recent trades
        for trade in wallet.recent_trades:
            if trade.get("market_id") == market_id and not trade.get("resolved"):
                trade["resolved"] = True
                trade["won"] = won
                trade["pnl"] = pnl
                break

        self._save_data()

    def get_top_performers(self, min_trades: int = 10, limit: int = 10) -> List[WalletPerformance]:
        """Get top performing wallets by win rate."""
        qualified = [
            w for w in self.wallets.values()
            if w.resolved_trades >= min_trades
        ]
        return sorted(qualified, key=lambda x: (x.win_rate, x.total_pnl), reverse=True)[:limit]

    def get_smart_money_wallets(self) -> List[WalletPerformance]:
        """Get wallets that qualify as smart money."""
        return [w for w in self.wallets.values() if w.is_smart_money]

    def is_smart_money_trade(self, address: str) -> Tuple[bool, Optional[float]]:
        """Check if a trade is from a smart money wallet."""
        if address not in self.wallets:
            return False, None
        wallet = self.wallets[address]
        return wallet.is_smart_money, wallet.win_rate if wallet.is_smart_money else None


class WhaleTrackerSignalDetector(BaseSignalDetector):
    """
    Detects whale trading activity on Polymarket.

    Generates signals based on:
    - Large individual trades
    - Aggregate whale position changes
    - Known whale wallet activity

    Usage:
        detector = WhaleTrackerSignalDetector()
        signals = detector.run()
    """

    STRENGTH_THRESHOLDS = {
        "weak": 0.3,
        "moderate": 0.5,
        "strong": 0.7,
        "very_strong": 0.9
    }

    def __init__(
        self,
        min_trade_size: float = WHALE_THRESHOLD_USD,
        lookback_hours: int = 24,
        tracked_markets: Optional[List[str]] = None,
        enable_copy_trade: bool = True
    ):
        """
        Initialize whale tracker.

        Args:
            min_trade_size: Minimum USD value to consider whale activity
            lookback_hours: Hours of history to analyze
            tracked_markets: Specific market IDs to track (default: all)
            enable_copy_trade: Enable copy-trade discovery features
        """
        super().__init__(
            name="Whale Tracker Signal",
            source="polymarket_activity"
        )

        self.min_trade_size = min_trade_size
        self.lookback_hours = lookback_hours
        self.tracked_markets = tracked_markets
        self.enable_copy_trade = enable_copy_trade

        # Initialize leaderboard for copy-trade tracking
        self.leaderboard = WalletLeaderboard() if enable_copy_trade else None

    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch recent trade activity from Polymarket.

        Returns:
            Dict with trade data organized by market
        """
        results = {
            "trades": [],
            "markets": {},
            "whale_activity": []
        }

        try:
            # Fetch recent activity via Polymarket's CLOB API
            # Note: Rate limited, be careful
            base_url = "https://clob.polymarket.com"

            # Get recent markets first
            markets_response = requests.get(
                f"{base_url}/markets",
                params={"limit": 50},  # Top 50 by volume
                timeout=30
            )

            if markets_response.status_code != 200:
                logger.warning(f"Could not fetch Polymarket markets: {markets_response.status_code}")
                return results

            markets_data = markets_response.json()
            markets = markets_data.get("data", [])

            # For each market, check for large trades
            for market in markets[:20]:  # Limit to 20 for rate limiting
                market_id = market.get("condition_id") or market.get("id")
                if not market_id:
                    continue

                if self.tracked_markets and market_id not in self.tracked_markets:
                    continue

                # Get recent trades
                trades_response = requests.get(
                    f"{base_url}/trades",
                    params={
                        "market": market_id,
                        "limit": 100
                    },
                    timeout=30
                )

                if trades_response.status_code == 200:
                    trades = trades_response.json()
                    market_trades = self._filter_whale_trades(trades, market)
                    if market_trades:
                        results["markets"][market_id] = {
                            "question": market.get("question", "Unknown"),
                            "trades": market_trades
                        }
                        results["whale_activity"].extend(market_trades)

        except requests.RequestException as e:
            logger.error(f"Error fetching whale data: {e}")
            results["error"] = str(e)

        logger.info(f"Found {len(results['whale_activity'])} whale trades")
        return results

    def _filter_whale_trades(
        self,
        trades: List[Dict],
        market: Dict
    ) -> List[Dict[str, Any]]:
        """Filter trades to whale-sized only."""
        whale_trades = []
        cutoff = datetime.utcnow() - timedelta(hours=self.lookback_hours)

        for trade in trades:
            # Parse trade
            try:
                trade_time_str = trade.get("timestamp") or trade.get("created_at")
                if not trade_time_str:
                    continue

                trade_time = datetime.fromisoformat(trade_time_str.replace("Z", "+00:00")).replace(tzinfo=None)
                if trade_time < cutoff:
                    continue

                # Calculate USD value
                size = float(trade.get("size") or trade.get("amount", 0))
                price = float(trade.get("price", 0.5))
                usd_value = size * price

                if usd_value >= self.min_trade_size:
                    # Check if known whale
                    maker = trade.get("maker_address", "")
                    taker = trade.get("taker_address", "")
                    whale_info = KNOWN_WHALES.get(maker) or KNOWN_WHALES.get(taker)

                    # Check if smart money
                    is_smart_money = False
                    smart_money_win_rate = None
                    primary_address = maker or taker

                    if self.leaderboard and primary_address:
                        is_smart, wr = self.leaderboard.is_smart_money_trade(primary_address)
                        is_smart_money = is_smart
                        smart_money_win_rate = wr

                        # Record trade in leaderboard
                        self.leaderboard.record_trade(
                            address=primary_address,
                            usd_value=usd_value,
                            side=trade.get("side", "unknown"),
                            market_id=market.get("condition_id") or market.get("id"),
                            price=price
                        )

                    whale_trades.append({
                        "timestamp": trade_time,
                        "market_id": market.get("condition_id") or market.get("id"),
                        "question": market.get("question", "Unknown"),
                        "side": trade.get("side", "unknown"),  # yes/no
                        "size": size,
                        "price": price,
                        "usd_value": usd_value,
                        "maker": maker,
                        "taker": taker,
                        "is_known_whale": whale_info is not None,
                        "whale_label": whale_info.get("label") if whale_info else None,
                        "whale_track_record": whale_info.get("track_record") if whale_info else None,
                        "is_smart_money": is_smart_money,
                        "smart_money_win_rate": smart_money_win_rate,
                    })

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Error parsing trade: {e}")
                continue

        return sorted(whale_trades, key=lambda x: x["usd_value"], reverse=True)

    def process_data(self, raw_data: Dict[str, Any]) -> List[SignalResult]:
        """
        Process whale trades into signals.

        Args:
            raw_data: Dict with whale trade data

        Returns:
            List of SignalResult objects
        """
        signals = []
        timestamp = datetime.utcnow()

        if raw_data.get("error"):
            return signals

        whale_activity = raw_data.get("whale_activity", [])
        markets_data = raw_data.get("markets", {})

        if not whale_activity:
            return signals

        # Generate individual whale signals for large trades
        for trade in whale_activity[:10]:  # Top 10 whale trades
            signal = self._create_trade_signal(trade, timestamp)
            if signal:
                signals.append(signal)

        # Generate aggregate signals per market
        for market_id, market_data in markets_data.items():
            market_signal = self._create_market_whale_signal(market_id, market_data, timestamp)
            if market_signal:
                signals.append(market_signal)

        # Overall whale sentiment signal
        overall_signal = self._create_overall_signal(whale_activity, timestamp)
        if overall_signal:
            signals.append(overall_signal)

        # Smart money signals for copy-trade discovery
        if self.enable_copy_trade:
            smart_money_signals = self._create_smart_money_signals(whale_activity, timestamp)
            signals.extend(smart_money_signals)

        logger.info(f"Generated {len(signals)} whale signals")
        return signals

    def _create_trade_signal(
        self,
        trade: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Create signal for individual whale trade."""
        usd_value = trade.get("usd_value", 0)
        side = trade.get("side", "").lower()

        if side not in ["yes", "no"]:
            return None

        # Determine direction based on trade side
        if side == "yes":
            direction = SignalDirection.BULLISH
        else:
            direction = SignalDirection.BEARISH

        # Confidence based on trade size and whale reputation
        base_confidence = 0.4
        if usd_value >= MEGA_WHALE_THRESHOLD_USD:
            base_confidence = 0.7
        elif usd_value >= LARGE_WHALE_THRESHOLD_USD:
            base_confidence = 0.55

        # Boost for known whales
        if trade.get("is_known_whale"):
            track_record = trade.get("whale_track_record", 0.5)
            base_confidence = min(0.85, base_confidence + (track_record - 0.5) * 0.3)

        # Boost for smart money
        if trade.get("is_smart_money"):
            smart_wr = trade.get("smart_money_win_rate", 0.55)
            base_confidence = min(0.9, base_confidence + (smart_wr - 0.5) * 0.4)

        # Strength based on size
        if usd_value >= MEGA_WHALE_THRESHOLD_USD:
            strength = SignalStrength.VERY_STRONG
        elif usd_value >= LARGE_WHALE_THRESHOLD_USD:
            strength = SignalStrength.STRONG
        else:
            strength = SignalStrength.MODERATE

        return SignalResult(
            name=f"Whale Trade: {trade.get('question', 'Unknown')[:40]}...",
            source=self.source,
            timestamp=timestamp,
            value=usd_value,
            direction=direction,
            confidence=base_confidence,
            strength=strength,
            raw_data={
                "market_id": trade.get("market_id"),
                "trade_side": side,
                "usd_value": usd_value,
                "price": trade.get("price"),
                "is_known_whale": trade.get("is_known_whale"),
                "whale_label": trade.get("whale_label"),
                "is_smart_money": trade.get("is_smart_money"),
                "smart_money_win_rate": trade.get("smart_money_win_rate"),
            },
            related_markets=[trade.get("question", "")[:50]],
            metadata={
                "signal_type": "whale_trade",
                "urgency": "high" if usd_value >= LARGE_WHALE_THRESHOLD_USD else "medium"
            }
        )

    def _create_market_whale_signal(
        self,
        market_id: str,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Create aggregate whale signal for a market."""
        trades = market_data.get("trades", [])
        if len(trades) < 2:
            return None

        # Aggregate whale sentiment
        yes_volume = sum(t["usd_value"] for t in trades if t.get("side", "").lower() == "yes")
        no_volume = sum(t["usd_value"] for t in trades if t.get("side", "").lower() == "no")
        total_volume = yes_volume + no_volume

        if total_volume < WHALE_THRESHOLD_USD * 2:
            return None

        # Calculate whale bias
        whale_bias = (yes_volume - no_volume) / total_volume if total_volume > 0 else 0

        if abs(whale_bias) < 0.2:
            return None  # No clear direction

        direction = SignalDirection.BULLISH if whale_bias > 0 else SignalDirection.BEARISH
        confidence = min(0.75, 0.4 + abs(whale_bias) * 0.5)

        return SignalResult(
            name=f"Whale Consensus: {market_data.get('question', 'Unknown')[:35]}...",
            source=self.source,
            timestamp=timestamp,
            value=whale_bias,
            direction=direction,
            confidence=confidence,
            strength=classify_signal_strength(abs(whale_bias), self.STRENGTH_THRESHOLDS),
            raw_data={
                "market_id": market_id,
                "yes_volume": yes_volume,
                "no_volume": no_volume,
                "total_volume": total_volume,
                "whale_count": len(trades),
                "whale_bias": whale_bias
            },
            related_markets=[market_data.get("question", "")],
            metadata={"signal_type": "whale_consensus"}
        )

    def _create_overall_signal(
        self,
        whale_activity: List[Dict[str, Any]],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Create overall whale market sentiment signal."""
        if len(whale_activity) < 5:
            return None

        total_yes = sum(t["usd_value"] for t in whale_activity if t.get("side", "").lower() == "yes")
        total_no = sum(t["usd_value"] for t in whale_activity if t.get("side", "").lower() == "no")
        total = total_yes + total_no

        if total < WHALE_THRESHOLD_USD * 5:
            return None

        market_bias = (total_yes - total_no) / total if total > 0 else 0

        if abs(market_bias) < 0.15:
            direction = SignalDirection.NEUTRAL
        elif market_bias > 0:
            direction = SignalDirection.BULLISH
        else:
            direction = SignalDirection.BEARISH

        confidence = min(0.6, 0.3 + abs(market_bias) * 0.4 + len(whale_activity) / 100)

        return SignalResult(
            name="Overall Whale Sentiment",
            source=self.source,
            timestamp=timestamp,
            value=market_bias,
            direction=direction,
            confidence=confidence,
            strength=classify_signal_strength(abs(market_bias), self.STRENGTH_THRESHOLDS),
            raw_data={
                "total_yes_volume": total_yes,
                "total_no_volume": total_no,
                "whale_count": len(whale_activity),
                "market_bias": market_bias
            },
            related_markets=["prediction_markets", "whale_activity"],
            metadata={"signal_type": "overall_whale_sentiment"}
        )

    def _create_smart_money_signals(
        self,
        whale_activity: List[Dict[str, Any]],
        timestamp: datetime
    ) -> List[SignalResult]:
        """Create signals for smart money (high win-rate wallet) trades."""
        signals = []

        # Filter to smart money trades only
        smart_money_trades = [
            t for t in whale_activity
            if t.get("is_smart_money")
        ]

        if not smart_money_trades:
            return signals

        for trade in smart_money_trades[:5]:  # Top 5 smart money trades
            side = trade.get("side", "").lower()
            if side not in ["yes", "no"]:
                continue

            direction = SignalDirection.BULLISH if side == "yes" else SignalDirection.BEARISH

            # High confidence for smart money
            win_rate = trade.get("smart_money_win_rate", 0.55)
            confidence = min(0.9, 0.5 + win_rate * 0.5)

            signals.append(SignalResult(
                name=f"Smart Money: {trade.get('question', 'Unknown')[:35]}...",
                source=self.source,
                timestamp=timestamp,
                value=trade.get("usd_value", 0),
                direction=direction,
                confidence=confidence,
                strength=SignalStrength.STRONG,
                raw_data={
                    "market_id": trade.get("market_id"),
                    "trade_side": side,
                    "usd_value": trade.get("usd_value"),
                    "smart_money_win_rate": win_rate,
                    "wallet": trade.get("maker", "")[:10] + "...",
                },
                related_markets=[trade.get("question", "")[:50]],
                metadata={
                    "signal_type": "smart_money",
                    "copy_trade_candidate": True,
                    "urgency": "high"
                }
            ))

        return signals

    def get_leaderboard(self, min_trades: int = 10, limit: int = 10) -> List[Dict]:
        """Get the whale leaderboard for dashboard display."""
        if not self.leaderboard:
            return []

        top_wallets = self.leaderboard.get_top_performers(min_trades, limit)
        return [w.to_dict() for w in top_wallets]

    def get_smart_money_summary(self) -> Dict:
        """Get summary of smart money wallets."""
        if not self.leaderboard:
            return {}

        smart_wallets = self.leaderboard.get_smart_money_wallets()
        if not smart_wallets:
            return {"count": 0, "wallets": []}

        return {
            "count": len(smart_wallets),
            "avg_win_rate": sum(w.win_rate for w in smart_wallets) / len(smart_wallets),
            "total_volume": sum(w.total_volume for w in smart_wallets),
            "total_pnl": sum(w.total_pnl for w in smart_wallets),
            "wallets": [w.to_dict() for w in smart_wallets[:10]],
        }


def get_whale_tracker() -> WhaleTrackerSignalDetector:
    """Get a configured whale tracker detector."""
    return WhaleTrackerSignalDetector()


if __name__ == "__main__":
    # Test the detector
    detector = WhaleTrackerSignalDetector()

    print("Tracking whale activity...")
    signals = detector.run()

    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"\n{signal.name}")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Value: ${signal.value:,.2f}" if signal.value > 100 else f"  Value: {signal.value:.3f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Strength: {signal.strength.value}")
