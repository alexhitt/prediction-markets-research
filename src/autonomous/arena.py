"""
Strategy Arena - Multiple Bots Competing Against Each Other

Runs multiple autonomous agents with different strategies simultaneously,
tracking their performance head-to-head to find the best approach.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.autonomous.leaderboard import Leaderboard, StrategyRanking
from src.autonomous.simulator import ContinuousSimulator, MarketScenario


@dataclass
class BotConfig:
    """Configuration for a competing bot."""
    name: str
    strategy_type: str  # "momentum", "value", "contrarian", "ensemble", "aggressive", "conservative", "news_racer"

    # Strategy parameters
    min_confidence: float = 0.5
    min_edge: float = 0.03
    position_size_pct: float = 0.05
    max_position_pct: float = 0.15

    # Signal weights
    momentum_weight: float = 0.20
    sentiment_weight: float = 0.20
    whale_weight: float = 0.20
    odds_weight: float = 0.20
    fast_news_weight: float = 0.20  # Weight for fast news signals

    # Risk
    max_daily_loss: float = 0.05
    stop_loss_pct: float = 0.10


@dataclass
class BotPerformance:
    """Performance metrics for a bot."""
    bot_name: str
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    current_capital: float = 10000.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    daily_pnls: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(1, self.total_trades)

    @property
    def roi(self) -> float:
        return (self.current_capital - 10000) / 10000

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bot_name': self.bot_name,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'current_capital': self.current_capital,
            'roi': self.roi,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'last_updated': self.last_updated.isoformat()
        }


class StrategyArena:
    """
    Runs multiple trading bots in competition.

    Features:
    - Simultaneous execution of multiple strategies
    - Real-time performance comparison
    - Automatic promotion of winning strategies
    - Tournament-style elimination

    Usage:
        arena = StrategyArena()
        arena.add_default_bots()
        arena.start()
        rankings = arena.get_rankings()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the arena."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "arena"
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bots: Dict[str, BotConfig] = {}
        self.performances: Dict[str, BotPerformance] = {}
        self.leaderboard = Leaderboard(data_dir / "leaderboard")
        self.simulator = ContinuousSimulator(data_dir / "simulations")

        self._running = False
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()

        self._load_state()

    def add_bot(self, config: BotConfig):
        """Add a bot to the arena."""
        self.bots[config.name] = config
        if config.name not in self.performances:
            self.performances[config.name] = BotPerformance(bot_name=config.name)
        logger.info(f"Added bot to arena: {config.name} ({config.strategy_type})")

    def add_default_bots(self):
        """Add a diverse set of competing bots."""

        # Aggressive momentum bot - takes bigger risks
        self.add_bot(BotConfig(
            name="Aggressive Alpha",
            strategy_type="momentum",
            min_confidence=0.4,
            min_edge=0.02,
            position_size_pct=0.10,
            max_position_pct=0.25,
            momentum_weight=0.45,
            sentiment_weight=0.15,
            whale_weight=0.15,
            odds_weight=0.10,
            fast_news_weight=0.15,
            max_daily_loss=0.10
        ))

        # Conservative value bot - lower risk, higher confidence required
        self.add_bot(BotConfig(
            name="Conservative Value",
            strategy_type="value",
            min_confidence=0.7,
            min_edge=0.05,
            position_size_pct=0.03,
            max_position_pct=0.10,
            momentum_weight=0.1,
            sentiment_weight=0.25,
            whale_weight=0.25,
            odds_weight=0.30,
            fast_news_weight=0.10,  # Conservative on news
            max_daily_loss=0.03
        ))

        # Whale follower - follows large trades
        self.add_bot(BotConfig(
            name="Whale Watcher",
            strategy_type="whale",
            min_confidence=0.5,
            min_edge=0.03,
            position_size_pct=0.05,
            max_position_pct=0.15,
            momentum_weight=0.1,
            sentiment_weight=0.1,
            whale_weight=0.60,
            odds_weight=0.10,
            fast_news_weight=0.10,
            max_daily_loss=0.05
        ))

        # Sentiment trader - trades on social/news sentiment
        self.add_bot(BotConfig(
            name="Sentiment Surfer",
            strategy_type="sentiment",
            min_confidence=0.5,
            min_edge=0.03,
            position_size_pct=0.05,
            max_position_pct=0.15,
            momentum_weight=0.1,
            sentiment_weight=0.40,
            whale_weight=0.1,
            odds_weight=0.15,
            fast_news_weight=0.25,  # News feeds into sentiment
            max_daily_loss=0.05
        ))

        # Balanced ensemble - equal weights
        self.add_bot(BotConfig(
            name="Balanced Bot",
            strategy_type="ensemble",
            min_confidence=0.55,
            min_edge=0.04,
            position_size_pct=0.05,
            max_position_pct=0.15,
            momentum_weight=0.20,
            sentiment_weight=0.20,
            whale_weight=0.20,
            odds_weight=0.20,
            fast_news_weight=0.20,
            max_daily_loss=0.05
        ))

        # Contrarian - bets against crowd
        self.add_bot(BotConfig(
            name="Contrarian Carl",
            strategy_type="contrarian",
            min_confidence=0.6,
            min_edge=0.05,
            position_size_pct=0.04,
            max_position_pct=0.12,
            momentum_weight=0.0,  # Ignores momentum
            sentiment_weight=0.4,  # Uses sentiment inversely
            whale_weight=0.3,
            odds_weight=0.3,
            fast_news_weight=0.0,
            max_daily_loss=0.04
        ))

        # News Racer - prioritizes fast news for information edge
        # Designed to capture alpha from breaking news before market prices it in
        self.add_bot(BotConfig(
            name="News Racer",
            strategy_type="news_racer",
            min_confidence=0.45,  # Lower threshold for speed
            min_edge=0.02,  # Accept smaller edges for faster execution
            position_size_pct=0.08,  # Larger positions on high-conviction news
            max_position_pct=0.20,
            momentum_weight=0.1,
            sentiment_weight=0.15,
            whale_weight=0.05,
            odds_weight=0.10,
            fast_news_weight=0.60,  # Heavy weight on fast news signals
            max_daily_loss=0.07,  # Higher risk tolerance for speed
            stop_loss_pct=0.08
        ))

        logger.info(f"Added {len(self.bots)} default bots to arena")

    def start(self, simulation_mode: bool = True):
        """
        Start the arena competition.

        Args:
            simulation_mode: If True, runs simulations instead of real paper trading
        """
        if self._running:
            logger.warning("Arena already running")
            return

        self._running = True
        self._stop_event.clear()

        if simulation_mode:
            # Run bots in simulation
            self._thread = threading.Thread(
                target=self._run_simulation_competition,
                daemon=True
            )
            self._thread.start()
        else:
            # Run each bot in its own thread
            for bot_name, config in self.bots.items():
                thread = threading.Thread(
                    target=self._run_bot,
                    args=(config,),
                    daemon=True
                )
                self._threads[bot_name] = thread
                thread.start()

        logger.info(f"Arena started with {len(self.bots)} competing bots")

    def stop(self):
        """Stop the arena competition."""
        self._running = False
        self._stop_event.set()

        for thread in self._threads.values():
            thread.join(timeout=5)

        self._save_state()
        logger.info("Arena stopped")

    def _run_simulation_competition(self):
        """Run competition via simulations."""
        # Add strategies to simulator
        self.simulator.add_default_strategies()

        round_num = 0
        while not self._stop_event.is_set():
            round_num += 1
            logger.info(f"=== Arena Round {round_num} ===")

            # Run simulations for each bot
            for bot_name, config in self.bots.items():
                try:
                    # Map bot strategy to simulator strategy
                    sim_strategy = self._map_to_sim_strategy(config.strategy_type)

                    # Run simulation
                    scenario = MarketScenario.NORMAL
                    result = self.simulator.run_simulation(
                        strategy_name=sim_strategy,
                        scenario=scenario
                    )

                    # Update performance
                    perf = self.performances[bot_name]
                    perf.total_pnl += result.total_pnl
                    perf.total_trades += result.total_trades
                    perf.winning_trades += result.winning_trades
                    perf.current_capital += result.total_pnl
                    perf.daily_pnls.append(result.total_pnl)
                    perf.last_updated = datetime.utcnow()

                    # Calculate Sharpe
                    if len(perf.daily_pnls) > 1:
                        import statistics
                        mean_pnl = statistics.mean(perf.daily_pnls)
                        std_pnl = statistics.stdev(perf.daily_pnls) or 1
                        perf.sharpe_ratio = (mean_pnl / std_pnl) * (252 ** 0.5)

                    # Calculate max drawdown
                    peak = 10000
                    for pnl in perf.daily_pnls:
                        peak = max(peak, peak + pnl)
                        drawdown = (peak - (peak + pnl)) / peak
                        perf.max_drawdown = max(perf.max_drawdown, drawdown)

                    # Record to leaderboard
                    self.leaderboard.record_daily_performance(
                        date=date.today(),
                        strategy=bot_name,
                        trades=result.total_trades,
                        pnl=result.total_pnl,
                        signals=0,
                        opportunities=0,
                        winning_trades=result.winning_trades,
                        sharpe_ratio=perf.sharpe_ratio,
                        roi=perf.roi
                    )

                except Exception as e:
                    logger.error(f"Error simulating {bot_name}: {e}")

            # Print current standings
            self._print_standings()

            # Save state
            self._save_state()

            # Wait between rounds
            time.sleep(60)  # 1 minute between rounds

    def _map_to_sim_strategy(self, strategy_type: str) -> str:
        """Map bot strategy type to simulator strategy."""
        mapping = {
            "momentum": "momentum",
            "value": "value",
            "contrarian": "contrarian",
            "ensemble": "momentum",  # Default to momentum
            "whale": "value",
            "sentiment": "contrarian",
            "aggressive": "momentum",
            "conservative": "value",
            "news_racer": "momentum"  # News-driven trades behave like momentum
        }
        return mapping.get(strategy_type, "momentum")

    def _run_bot(self, config: BotConfig):
        """Run a single bot (for live paper trading mode)."""
        logger.info(f"Starting bot: {config.name}")

        while not self._stop_event.is_set():
            try:
                # This would integrate with the autonomous agent
                # For now, just simulate
                pass

            except Exception as e:
                logger.error(f"Error in bot {config.name}: {e}")

            time.sleep(60)

    def get_rankings(self) -> List[Dict[str, Any]]:
        """Get current bot rankings."""
        rankings = []

        for bot_name, perf in self.performances.items():
            rankings.append({
                **perf.to_dict(),
                'strategy_type': self.bots[bot_name].strategy_type
            })

        # Sort by total P&L
        rankings.sort(key=lambda x: x['total_pnl'], reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings

    def get_head_to_head(self, bot_a: str, bot_b: str) -> Dict[str, Any]:
        """Compare two bots head-to-head."""
        perf_a = self.performances.get(bot_a)
        perf_b = self.performances.get(bot_b)

        if not perf_a or not perf_b:
            return {"error": "Bot not found"}

        return {
            'bot_a': perf_a.to_dict(),
            'bot_b': perf_b.to_dict(),
            'pnl_winner': bot_a if perf_a.total_pnl > perf_b.total_pnl else bot_b,
            'sharpe_winner': bot_a if perf_a.sharpe_ratio > perf_b.sharpe_ratio else bot_b,
            'win_rate_winner': bot_a if perf_a.win_rate > perf_b.win_rate else bot_b,
            'overall_winner': bot_a if perf_a.total_pnl > perf_b.total_pnl else bot_b
        }

    def _print_standings(self):
        """Print current standings."""
        rankings = self.get_rankings()

        logger.info("\n" + "=" * 60)
        logger.info("ARENA STANDINGS")
        logger.info("=" * 60)

        for r in rankings:
            logger.info(
                f"#{r['rank']} {r['bot_name']:20s} | "
                f"P&L: ${r['total_pnl']:+10,.2f} | "
                f"Win: {r['win_rate']:.1%} | "
                f"Sharpe: {r['sharpe_ratio']:.2f}"
            )

        logger.info("=" * 60 + "\n")

    def _save_state(self):
        """Save arena state to disk."""
        try:
            state = {
                'bots': {name: config.__dict__ for name, config in self.bots.items()},
                'performances': {name: perf.to_dict() for name, perf in self.performances.items()},
                'saved_at': datetime.utcnow().isoformat()
            }

            path = self.data_dir / "arena_state.json"
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving arena state: {e}")

    def _load_state(self):
        """Load arena state from disk."""
        try:
            path = self.data_dir / "arena_state.json"
            if path.exists():
                with open(path, 'r') as f:
                    state = json.load(f)

                # Restore performances
                for name, perf_data in state.get('performances', {}).items():
                    self.performances[name] = BotPerformance(
                        bot_name=perf_data['bot_name'],
                        total_pnl=perf_data['total_pnl'],
                        total_trades=perf_data['total_trades'],
                        winning_trades=perf_data['winning_trades'],
                        current_capital=perf_data['current_capital'],
                        sharpe_ratio=perf_data['sharpe_ratio'],
                        max_drawdown=perf_data['max_drawdown'],
                        last_updated=datetime.fromisoformat(perf_data['last_updated'])
                    )

                logger.info(f"Loaded arena state with {len(self.performances)} bots")

        except Exception as e:
            logger.warning(f"Could not load arena state: {e}")


def get_arena() -> StrategyArena:
    """Get or create the global arena instance."""
    return StrategyArena()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Arena")
    parser.add_argument("--start", action="store_true", help="Start the arena")
    parser.add_argument("--rankings", action="store_true", help="Show rankings")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")

    args = parser.parse_args()

    arena = StrategyArena()
    arena.add_default_bots()

    if args.start:
        print("\n" + "=" * 60)
        print("  STRATEGY ARENA - BOT COMPETITION")
        print("=" * 60)
        print(f"\nCompeting bots: {len(arena.bots)}")
        for name, config in arena.bots.items():
            print(f"  - {name} ({config.strategy_type})")
        print("\nStarting competition...\n")

        arena.start(simulation_mode=True)

        try:
            rounds = 0
            while rounds < args.rounds:
                time.sleep(65)
                rounds += 1
                print(f"\nRound {rounds} completed")
                rankings = arena.get_rankings()
                print(f"Leader: {rankings[0]['bot_name']} (${rankings[0]['total_pnl']:+,.2f})")

        except KeyboardInterrupt:
            print("\nStopping arena...")

        arena.stop()
        print("\nFinal Rankings:")
        for r in arena.get_rankings():
            print(f"  #{r['rank']} {r['bot_name']}: ${r['total_pnl']:+,.2f}")

    elif args.rankings:
        rankings = arena.get_rankings()
        print("\nCurrent Rankings:")
        for r in rankings:
            print(f"  #{r['rank']} {r['bot_name']}: ${r['total_pnl']:+,.2f} ({r['strategy_type']})")
