"""
Bot Tournament System

Manages bot competition, evaluation, elimination, and promotion.

Tournament Rules:
- 3-day evaluation period before any bot can advance or be eliminated
- Top performers advance to higher betting tiers
- Bottom performers are eliminated or mutated (strategy tweaked)
- Tracks bot lineage and evolution history

Betting Tiers:
- Tier 0: Simulation only (paper trading)
- Tier 1: Micro bets ($1-5)
- Tier 2: Small bets ($5-25)
- Tier 3: Medium bets ($25-100)
- Tier 4: Large bets ($100+)
"""

import json
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock

from loguru import logger


class BettingTier(Enum):
    """Betting tiers for bot progression."""
    SIMULATION = 0  # Paper trading only
    MICRO = 1       # $1-5 bets
    SMALL = 2       # $5-25 bets
    MEDIUM = 3      # $25-100 bets
    LARGE = 4       # $100+ bets

    @property
    def bet_range(self) -> Tuple[float, float]:
        """Get min/max bet size for this tier."""
        ranges = {
            BettingTier.SIMULATION: (0, 0),
            BettingTier.MICRO: (1, 5),
            BettingTier.SMALL: (5, 25),
            BettingTier.MEDIUM: (25, 100),
            BettingTier.LARGE: (100, 500),
        }
        return ranges[self]

    @property
    def display_name(self) -> str:
        names = {
            BettingTier.SIMULATION: "🎮 Simulation",
            BettingTier.MICRO: "🪙 Micro ($1-5)",
            BettingTier.SMALL: "💵 Small ($5-25)",
            BettingTier.MEDIUM: "💰 Medium ($25-100)",
            BettingTier.LARGE: "🏆 Large ($100+)",
        }
        return names[self]


class BotStatus(Enum):
    """Bot tournament status."""
    EVALUATING = "evaluating"  # In 3-day evaluation
    ACTIVE = "active"          # Passed evaluation, competing
    PROMOTED = "promoted"      # Advanced to next tier
    ELIMINATED = "eliminated"  # Removed from tournament
    MUTATED = "mutated"        # Strategy tweaked, re-evaluating


@dataclass
class SimulatedBet:
    """A single simulated bet."""
    id: str
    bot_id: str
    market_id: str
    market_question: str
    side: str  # "yes" or "no"
    amount: float
    entry_price: float
    placed_at: datetime
    matures_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"  # open, won, lost, expired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bot_id": self.bot_id,
            "market_id": self.market_id,
            "market_question": self.market_question,
            "side": self.side,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "placed_at": self.placed_at.isoformat(),
            "matures_at": self.matures_at.isoformat() if self.matures_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "status": self.status,
        }


@dataclass
class BotProfile:
    """Complete bot profile with tournament data."""
    id: str
    name: str
    strategy_type: str
    tier: BettingTier = BettingTier.SIMULATION
    status: BotStatus = BotStatus.EVALUATING

    # Tournament tracking
    evaluation_start: Optional[datetime] = None
    days_evaluated: int = 0
    tournament_wins: int = 0
    tournament_losses: int = 0

    # Performance
    total_bets: int = 0
    winning_bets: int = 0
    total_pnl: float = 0.0
    current_capital: float = 10000.0
    peak_capital: float = 10000.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Daily tracking
    daily_pnls: List[float] = field(default_factory=list)
    daily_bets: List[int] = field(default_factory=list)

    # Active bets
    open_bets: List[SimulatedBet] = field(default_factory=list)
    bet_history: List[SimulatedBet] = field(default_factory=list)

    # Strategy weights
    weights: Dict[str, float] = field(default_factory=dict)

    # Lineage tracking
    parent_id: Optional[str] = None
    generation: int = 1
    mutation_notes: Optional[str] = None

    @property
    def win_rate(self) -> float:
        return self.winning_bets / max(1, self.total_bets)

    @property
    def roi(self) -> float:
        return (self.current_capital - 10000) / 10000

    @property
    def can_advance(self) -> bool:
        """Check if bot can advance to next tier."""
        return (
            self.days_evaluated >= 3 and
            self.win_rate >= 0.55 and
            self.total_pnl > 0 and
            self.sharpe_ratio > 0.5
        )

    @property
    def should_eliminate(self) -> bool:
        """Check if bot should be eliminated."""
        return (
            self.days_evaluated >= 3 and
            (self.win_rate < 0.40 or self.total_pnl < -500 or self.max_drawdown > 0.25)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "tier": self.tier.value,
            "tier_name": self.tier.display_name,
            "status": self.status.value,
            "evaluation_start": self.evaluation_start.isoformat() if self.evaluation_start else None,
            "days_evaluated": self.days_evaluated,
            "tournament_wins": self.tournament_wins,
            "tournament_losses": self.tournament_losses,
            "total_bets": self.total_bets,
            "winning_bets": self.winning_bets,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "current_capital": self.current_capital,
            "roi": self.roi,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "open_bets_count": len(self.open_bets),
            "can_advance": self.can_advance,
            "should_eliminate": self.should_eliminate,
            "generation": self.generation,
            "weights": self.weights,
        }


class BotTournament:
    """
    Manages the bot tournament and evolution.

    Features:
    - 3-day evaluation period for all bots
    - Automatic advancement/elimination based on performance
    - Bot mutation (strategy tweaking) for underperformers
    - Real-time bet tracking and visualization
    - Tier progression from simulation to real money
    """

    EVALUATION_DAYS = 3
    MIN_BOTS = 5  # Minimum bots to maintain

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "tournament"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bots: Dict[str, BotProfile] = {}
        self.eliminated_bots: List[BotProfile] = []
        self.today_bets: List[SimulatedBet] = []
        self.all_bets: List[SimulatedBet] = []

        self._lock = Lock()
        self._load_state()

    def _load_state(self):
        """Load tournament state from disk."""
        state_file = self.data_dir / "tournament_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    # Reconstruct bots
                    for bot_data in data.get("bots", []):
                        bot = self._dict_to_bot(bot_data)
                        self.bots[bot.id] = bot
                    logger.info(f"Loaded {len(self.bots)} bots from tournament state")
            except Exception as e:
                logger.error(f"Error loading tournament state: {e}")

    def _save_state(self):
        """Save tournament state to disk."""
        state_file = self.data_dir / "tournament_state.json"
        try:
            data = {
                "bots": [bot.to_dict() for bot in self.bots.values()],
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tournament state: {e}")

    def _dict_to_bot(self, data: Dict) -> BotProfile:
        """Reconstruct BotProfile from dict."""
        return BotProfile(
            id=data["id"],
            name=data["name"],
            strategy_type=data["strategy_type"],
            tier=BettingTier(data.get("tier", 0)),
            status=BotStatus(data.get("status", "evaluating")),
            days_evaluated=data.get("days_evaluated", 0),
            total_bets=data.get("total_bets", 0),
            winning_bets=data.get("winning_bets", 0),
            total_pnl=data.get("total_pnl", 0),
            current_capital=data.get("current_capital", 10000),
            sharpe_ratio=data.get("sharpe_ratio", 0),
            max_drawdown=data.get("max_drawdown", 0),
            generation=data.get("generation", 1),
            weights=data.get("weights", {}),
        )

    def add_bot(self, name: str, strategy_type: str, weights: Dict[str, float]) -> BotProfile:
        """Add a new bot to the tournament."""
        with self._lock:
            bot = BotProfile(
                id=str(uuid.uuid4())[:8],
                name=name,
                strategy_type=strategy_type,
                evaluation_start=datetime.utcnow(),
                weights=weights,
            )
            self.bots[bot.id] = bot
            logger.info(f"Added bot to tournament: {name} ({strategy_type})")
            self._save_state()
            return bot

    def add_default_bots(self):
        """Add default competing bots."""
        default_bots = [
            ("Aggressive Alpha", "momentum", {"momentum": 0.45, "fast_news": 0.25, "sentiment": 0.15, "odds": 0.15}),
            ("Conservative Value", "value", {"momentum": 0.10, "fast_news": 0.15, "sentiment": 0.25, "odds": 0.50}),
            ("News Racer", "news_racer", {"momentum": 0.10, "fast_news": 0.60, "sentiment": 0.15, "odds": 0.15}),
            ("Whale Watcher", "whale", {"momentum": 0.10, "fast_news": 0.10, "whale": 0.60, "odds": 0.20}),
            ("Sentiment Surfer", "sentiment", {"momentum": 0.10, "fast_news": 0.25, "sentiment": 0.50, "odds": 0.15}),
            ("Balanced Bot", "ensemble", {"momentum": 0.20, "fast_news": 0.20, "sentiment": 0.20, "odds": 0.20, "whale": 0.20}),
            ("Contrarian Carl", "contrarian", {"momentum": -0.20, "fast_news": 0.10, "sentiment": -0.30, "odds": 0.40}),
        ]

        for name, strategy, weights in default_bots:
            if not any(b.name == name for b in self.bots.values()):
                self.add_bot(name, strategy, weights)

    def record_bet(
        self,
        bot_id: str,
        market_id: str,
        market_question: str,
        side: str,
        amount: float,
        entry_price: float,
        matures_at: Optional[datetime] = None
    ) -> Optional[SimulatedBet]:
        """Record a new bet for a bot."""
        with self._lock:
            if bot_id not in self.bots:
                return None

            bet = SimulatedBet(
                id=str(uuid.uuid4())[:8],
                bot_id=bot_id,
                market_id=market_id,
                market_question=market_question,
                side=side,
                amount=amount,
                entry_price=entry_price,
                placed_at=datetime.utcnow(),
                matures_at=matures_at or datetime.utcnow() + timedelta(hours=random.randint(1, 48)),
            )

            self.bots[bot_id].open_bets.append(bet)
            self.bots[bot_id].total_bets += 1
            self.today_bets.append(bet)
            self.all_bets.append(bet)

            return bet

    def resolve_bet(self, bet_id: str, won: bool, exit_price: float):
        """Resolve a bet with outcome."""
        with self._lock:
            for bot in self.bots.values():
                for bet in bot.open_bets:
                    if bet.id == bet_id:
                        bet.status = "won" if won else "lost"
                        bet.exit_price = exit_price
                        bet.resolved_at = datetime.utcnow()

                        # Calculate P&L
                        if won:
                            bet.pnl = bet.amount * (1 / bet.entry_price - 1)
                            bot.winning_bets += 1
                        else:
                            bet.pnl = -bet.amount

                        bot.total_pnl += bet.pnl
                        bot.current_capital += bet.pnl
                        bot.peak_capital = max(bot.peak_capital, bot.current_capital)

                        # Move to history
                        bot.open_bets.remove(bet)
                        bot.bet_history.append(bet)

                        # Update drawdown
                        drawdown = (bot.peak_capital - bot.current_capital) / bot.peak_capital
                        bot.max_drawdown = max(bot.max_drawdown, drawdown)

                        self._save_state()
                        return

    def run_daily_evaluation(self):
        """Run end-of-day evaluation for all bots."""
        with self._lock:
            today = date.today()
            logger.info(f"Running daily evaluation for {today}")

            for bot in self.bots.values():
                if bot.status in [BotStatus.ELIMINATED]:
                    continue

                bot.days_evaluated += 1

                # Calculate daily stats
                day_pnl = sum(b.pnl or 0 for b in bot.bet_history if b.resolved_at and b.resolved_at.date() == today)
                day_bets = len([b for b in bot.bet_history if b.resolved_at and b.resolved_at.date() == today])

                bot.daily_pnls.append(day_pnl)
                bot.daily_bets.append(day_bets)

                # Update Sharpe ratio
                if len(bot.daily_pnls) > 1:
                    import statistics
                    mean_pnl = statistics.mean(bot.daily_pnls)
                    std_pnl = statistics.stdev(bot.daily_pnls) or 1
                    bot.sharpe_ratio = (mean_pnl / std_pnl) * (252 ** 0.5)

                # Check for advancement/elimination after evaluation period
                if bot.days_evaluated >= self.EVALUATION_DAYS:
                    if bot.can_advance:
                        self._advance_bot(bot)
                    elif bot.should_eliminate:
                        self._handle_underperformer(bot)
                    else:
                        bot.status = BotStatus.ACTIVE

            # Ensure minimum bots
            self._ensure_minimum_bots()

            self._save_state()
            self._log_standings()

    def _advance_bot(self, bot: BotProfile):
        """Advance bot to next tier."""
        if bot.tier.value < BettingTier.LARGE.value:
            old_tier = bot.tier
            bot.tier = BettingTier(bot.tier.value + 1)
            bot.status = BotStatus.PROMOTED
            bot.tournament_wins += 1
            logger.info(f"🎉 {bot.name} ADVANCED: {old_tier.display_name} → {bot.tier.display_name}")

    def _handle_underperformer(self, bot: BotProfile):
        """Handle underperforming bot - eliminate or mutate."""
        if len(self.bots) <= self.MIN_BOTS:
            # Mutate instead of eliminate
            self._mutate_bot(bot)
        else:
            # Eliminate
            bot.status = BotStatus.ELIMINATED
            bot.tournament_losses += 1
            self.eliminated_bots.append(bot)
            logger.info(f"❌ {bot.name} ELIMINATED (Win rate: {bot.win_rate:.1%}, P&L: ${bot.total_pnl:.2f})")

    def _mutate_bot(self, bot: BotProfile):
        """Mutate bot's strategy to try to improve it."""
        bot.status = BotStatus.MUTATED

        # Tweak weights randomly
        for key in bot.weights:
            adjustment = random.gauss(0, 0.1)
            bot.weights[key] = max(0, min(1, bot.weights[key] + adjustment))

        # Normalize weights
        total = sum(bot.weights.values())
        if total > 0:
            bot.weights = {k: v/total for k, v in bot.weights.items()}

        # Reset evaluation
        bot.days_evaluated = 0
        bot.generation += 1
        bot.mutation_notes = f"Mutated on {date.today()} after poor performance"

        logger.info(f"🧬 {bot.name} MUTATED to generation {bot.generation}")

    def _ensure_minimum_bots(self):
        """Ensure we maintain minimum number of active bots."""
        active_count = len([b for b in self.bots.values() if b.status != BotStatus.ELIMINATED])

        while active_count < self.MIN_BOTS:
            # Create a new bot based on best performer
            best_bot = max(
                [b for b in self.bots.values() if b.status != BotStatus.ELIMINATED],
                key=lambda x: x.total_pnl,
                default=None
            )

            if best_bot:
                # Clone and mutate best performer
                new_weights = best_bot.weights.copy()
                for key in new_weights:
                    new_weights[key] += random.gauss(0, 0.05)

                new_name = f"{best_bot.name} Jr. {len(self.bots)+1}"
                new_bot = self.add_bot(new_name, best_bot.strategy_type, new_weights)
                new_bot.parent_id = best_bot.id
                new_bot.generation = best_bot.generation + 1

                active_count += 1

    def _log_standings(self):
        """Log current tournament standings."""
        active_bots = sorted(
            [b for b in self.bots.values() if b.status != BotStatus.ELIMINATED],
            key=lambda x: x.total_pnl,
            reverse=True
        )

        logger.info("=" * 60)
        logger.info("TOURNAMENT STANDINGS")
        logger.info("=" * 60)
        for i, bot in enumerate(active_bots[:10], 1):
            logger.info(
                f"#{i} {bot.name:20s} | {bot.tier.display_name:15s} | "
                f"P&L: ${bot.total_pnl:+10,.2f} | Win: {bot.win_rate:.1%} | "
                f"Day {bot.days_evaluated}/3"
            )

    def get_rankings(self) -> List[Dict[str, Any]]:
        """Get current bot rankings."""
        with self._lock:
            active_bots = sorted(
                [b for b in self.bots.values() if b.status != BotStatus.ELIMINATED],
                key=lambda x: x.total_pnl,
                reverse=True
            )
            return [
                {**bot.to_dict(), "rank": i+1}
                for i, bot in enumerate(active_bots)
            ]

    def get_bot_profile(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed bot profile."""
        with self._lock:
            if bot_id not in self.bots:
                return None

            bot = self.bots[bot_id]
            return {
                **bot.to_dict(),
                "open_bets": [b.to_dict() for b in bot.open_bets],
                "recent_bets": [b.to_dict() for b in bot.bet_history[-20:]],
                "daily_pnls": bot.daily_pnls,
            }

    def get_today_bets(self) -> List[Dict[str, Any]]:
        """Get all bets placed today."""
        with self._lock:
            today = date.today()
            today_bets = [
                b for b in self.all_bets
                if b.placed_at.date() == today
            ]
            return [b.to_dict() for b in sorted(today_bets, key=lambda x: x.placed_at, reverse=True)]

    def get_upcoming_maturities(self) -> List[Dict[str, Any]]:
        """Get bets maturing soon."""
        with self._lock:
            now = datetime.utcnow()
            upcoming = []

            for bot in self.bots.values():
                for bet in bot.open_bets:
                    if bet.matures_at:
                        upcoming.append({
                            **bet.to_dict(),
                            "bot_name": bot.name,
                            "time_to_maturity": (bet.matures_at - now).total_seconds() / 3600,
                        })

            return sorted(upcoming, key=lambda x: x.get("time_to_maturity", 999))

    def simulate_round(self):
        """Simulate a round of betting for all bots."""
        with self._lock:
            for bot in self.bots.values():
                if bot.status == BotStatus.ELIMINATED:
                    continue

                # Simulate 1-3 bets per bot per round
                num_bets = random.randint(1, 3)

                for _ in range(num_bets):
                    # Generate fake market data
                    markets = [
                        ("Will Bitcoin exceed $100k by Feb 2026?", random.uniform(0.3, 0.7)),
                        ("Will Trump win 2028 election?", random.uniform(0.4, 0.6)),
                        ("Will Fed cut rates in Q1 2026?", random.uniform(0.5, 0.8)),
                        ("Will AI regulation pass in 2026?", random.uniform(0.3, 0.5)),
                        ("Will inflation drop below 2% by June?", random.uniform(0.2, 0.4)),
                    ]

                    market_q, base_price = random.choice(markets)
                    side = random.choice(["yes", "no"])
                    price = base_price if side == "yes" else 1 - base_price

                    # Bet size based on tier
                    min_bet, max_bet = bot.tier.bet_range
                    if min_bet == 0:
                        amount = random.uniform(50, 200)  # Simulation amounts
                    else:
                        amount = random.uniform(min_bet, max_bet)

                    self.record_bet(
                        bot_id=bot.id,
                        market_id=f"market_{random.randint(1000, 9999)}",
                        market_question=market_q,
                        side=side,
                        amount=amount,
                        entry_price=price,
                        matures_at=datetime.utcnow() + timedelta(hours=random.randint(1, 72))
                    )

            # Resolve some open bets randomly
            for bot in self.bots.values():
                for bet in list(bot.open_bets):
                    if random.random() < 0.1:  # 10% chance to resolve
                        won = random.random() < (0.5 + (bot.sharpe_ratio * 0.05))  # Slightly skill-based
                        exit_price = random.uniform(0.1, 0.9)
                        self.resolve_bet(bet.id, won, exit_price)

            self._save_state()


def get_tournament() -> BotTournament:
    """Get or create the tournament instance."""
    return BotTournament()


if __name__ == "__main__":
    # Test the tournament
    tournament = BotTournament()
    tournament.add_default_bots()

    print("Running simulation rounds...")
    for i in range(5):
        tournament.simulate_round()
        print(f"Round {i+1}: {len(tournament.get_today_bets())} bets placed")

    print("\nRankings:")
    for r in tournament.get_rankings():
        print(f"#{r['rank']} {r['name']:20s} | {r['tier_name']:15s} | P&L: ${r['total_pnl']:+.2f}")
