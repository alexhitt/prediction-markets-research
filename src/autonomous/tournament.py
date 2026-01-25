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
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import RLock

from loguru import logger

# Import real trading components
try:
    from src.market_discovery import MarketScanner, ResolutionTracker
    from src.research import get_researcher_for_bot, ProbabilityEstimate
    from src.learning import ResolutionWorker, CalibrationTracker, DomainSpecialist
    from src.database.models import ResearchEstimate, PendingResolution, BotMemory
    HAS_REAL_TRADING = True
except ImportError:
    HAS_REAL_TRADING = False
    logger.warning("Real trading modules not available - using simulation only")

# Calendar data directory for daily summaries
CALENDAR_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "calendar"


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
            "open_bets": [b.to_dict() for b in self.open_bets],  # Persist open bets
            "bet_history": [b.to_dict() for b in self.bet_history[-50:]],  # Last 50 bets
            "daily_pnls": self.daily_pnls,
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

        self._lock = RLock()  # Reentrant lock to allow nested calls
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
                    # Restore today_bets and all_bets
                    for bet_data in data.get("today_bets", []):
                        self.today_bets.append(self._dict_to_bet(bet_data))
                    for bet_data in data.get("all_bets", []):
                        self.all_bets.append(self._dict_to_bet(bet_data))
                    logger.info(f"Loaded {len(self.bots)} bots, {len(self.today_bets)} today bets from tournament state")
            except Exception as e:
                logger.error(f"Error loading tournament state: {e}")

    def _save_state(self):
        """Save tournament state to disk."""
        state_file = self.data_dir / "tournament_state.json"
        try:
            data = {
                "bots": [bot.to_dict() for bot in self.bots.values()],
                "today_bets": [b.to_dict() for b in self.today_bets[-100:]],  # Last 100 today's bets
                "all_bets": [b.to_dict() for b in self.all_bets[-500:]],  # Last 500 total bets
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tournament state: {e}")

    def _dict_to_bet(self, data: Dict) -> SimulatedBet:
        """Reconstruct SimulatedBet from dict."""
        return SimulatedBet(
            id=data["id"],
            bot_id=data["bot_id"],
            market_id=data["market_id"],
            market_question=data["market_question"],
            side=data["side"],
            amount=data["amount"],
            entry_price=data["entry_price"],
            placed_at=datetime.fromisoformat(data["placed_at"]) if data.get("placed_at") else datetime.utcnow(),
            matures_at=datetime.fromisoformat(data["matures_at"]) if data.get("matures_at") else None,
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            exit_price=data.get("exit_price"),
            pnl=data.get("pnl", 0),
            status=data.get("status", "pending"),
        )

    def _dict_to_bot(self, data: Dict) -> BotProfile:
        """Reconstruct BotProfile from dict."""
        bot = BotProfile(
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
        # Restore open bets and bet history
        for bet_data in data.get("open_bets", []):
            bot.open_bets.append(self._dict_to_bet(bet_data))
        for bet_data in data.get("bet_history", []):
            bot.bet_history.append(self._dict_to_bet(bet_data))
        bot.daily_pnls = data.get("daily_pnls", [])
        return bot

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
            # === TIER 1: SAFE BOTS (Priority for conservative strategy) ===
            ("Arbitrage Hunter", "arbitrage", {"arbitrage": 1.0}),
            ("High Prob Bond", "high_probability", {"odds": 1.0}),
            ("Smart Money Copy", "smart_money_copier", {"whale": 0.6, "smart_money": 0.4}),

            # === TIER 2: LOW-MEDIUM RISK ===
            ("Conservative Value", "value", {"momentum": 0.10, "fast_news": 0.15, "sentiment": 0.25, "odds": 0.50}),
            ("Whale Watcher", "whale", {"momentum": 0.10, "fast_news": 0.10, "whale": 0.60, "odds": 0.20}),
            ("Balanced Bot", "ensemble", {"momentum": 0.20, "fast_news": 0.20, "sentiment": 0.20, "odds": 0.20, "whale": 0.20}),

            # === TIER 3: MEDIUM-HIGH RISK ===
            ("Aggressive Alpha", "momentum", {"momentum": 0.45, "fast_news": 0.25, "sentiment": 0.15, "odds": 0.15}),
            ("News Racer", "news_racer", {"momentum": 0.10, "fast_news": 0.60, "sentiment": 0.15, "odds": 0.15}),
            ("Sentiment Surfer", "sentiment", {"momentum": 0.10, "fast_news": 0.25, "sentiment": 0.50, "odds": 0.15}),

            # === TIER 4: HIGH RISK ===
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

            # Save daily summary for calendar
            self._save_daily_summary(today)

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

    def _save_daily_summary(self, today: date):
        """Save daily summary for calendar display."""
        CALENDAR_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Calculate today's stats
        today_pnl = 0.0
        bets_placed = 0
        bets_resolved = 0
        wins = 0
        losses = 0

        for bet in self.all_bets:
            if bet.placed_at.date() == today:
                bets_placed += 1
            if bet.resolved_at and bet.resolved_at.date() == today:
                bets_resolved += 1
                today_pnl += bet.pnl or 0
                if bet.status == "won":
                    wins += 1
                elif bet.status == "lost":
                    losses += 1

        # Find best and worst performing bots today
        bot_pnls = {}
        for bot in self.bots.values():
            if bot.status == BotStatus.ELIMINATED:
                continue
            bot_day_pnl = sum(
                b.pnl or 0 for b in bot.bet_history
                if b.resolved_at and b.resolved_at.date() == today
            )
            bot_pnls[bot.name] = bot_day_pnl

        best_bot = max(bot_pnls.items(), key=lambda x: x[1], default=("", 0))[0] if bot_pnls else ""
        worst_bot = min(bot_pnls.items(), key=lambda x: x[1], default=("", 0))[0] if bot_pnls else ""

        summary = {
            "date": today.strftime("%Y-%m-%d"),
            "total_pnl": today_pnl,
            "bets_placed": bets_placed,
            "bets_resolved": bets_resolved,
            "wins": wins,
            "losses": losses,
            "best_bot": best_bot,
            "worst_bot": worst_bot,
            "signals_processed": 0,  # Would be populated by signal collector
            "whale_trades_detected": 0,  # Would be populated by whale tracker
            "tasks_completed": [],
            "notes": "",
        }

        file_path = CALENDAR_DATA_DIR / f"{today.strftime('%Y-%m-%d')}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved daily summary for {today}")
        except Exception as e:
            logger.error(f"Failed to save daily summary: {e}")

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
        """Simulate a round of betting for all bots using their strategies."""
        from src.autonomous.strategies import get_strategy, simulate_market_opportunity

        with self._lock:
            # Generate market opportunities for this round
            num_opportunities = random.randint(5, 10)
            opportunities = [simulate_market_opportunity() for _ in range(num_opportunities)]

            for bot in self.bots.values():
                if bot.status == BotStatus.ELIMINATED:
                    continue

                # Get bot's strategy
                strategy = get_strategy(bot.strategy_type, bot.weights)

                # Evaluate each market opportunity
                for market in opportunities:
                    decision = strategy.evaluate(market)

                    if not decision.should_bet:
                        continue

                    # Check if bot has capital
                    min_bet, max_bet = bot.tier.bet_range
                    if min_bet == 0:
                        base_amount = random.uniform(50, 200)
                    else:
                        base_amount = random.uniform(min_bet, max_bet)

                    # Apply strategy's size multiplier
                    amount = base_amount * decision.size_multiplier

                    # Ensure bot has enough capital
                    if amount > bot.current_capital * 0.25:  # Max 25% of capital per bet
                        amount = bot.current_capital * 0.25

                    if amount < 10:  # Minimum bet size
                        continue

                    price = market.yes_price if decision.side == "yes" else market.no_price

                    self.record_bet(
                        bot_id=bot.id,
                        market_id=market.market_id,
                        market_question=market.question,
                        side=decision.side,
                        amount=amount,
                        entry_price=price,
                        matures_at=datetime.utcnow() + timedelta(hours=random.randint(1, 72))
                    )

                    logger.debug(
                        f"{bot.name} ({strategy.name}): {decision.side.upper()} ${amount:.0f} "
                        f"on '{market.question[:30]}...' (conf: {decision.confidence:.0%})"
                    )

            # Resolve some open bets based on strategy quality
            for bot in self.bots.values():
                strategy = get_strategy(bot.strategy_type, bot.weights)

                for bet in list(bot.open_bets):
                    if random.random() < 0.1:  # 10% chance to resolve per round
                        # Win probability based on strategy risk level and confidence
                        base_win_prob = {
                            "very_low": 0.70,   # Arbitrage, high-prob
                            "low": 0.58,        # Copy-trade, value
                            "medium": 0.52,     # Momentum, ensemble
                            "high": 0.48,       # News racing
                            "very_high": 0.42,  # Contrarian
                        }.get(strategy.risk_level.value, 0.50)

                        # Sharpe ratio bonus
                        win_prob = base_win_prob + (bot.sharpe_ratio * 0.02)

                        won = random.random() < win_prob
                        exit_price = random.uniform(0.1, 0.9)
                        self.resolve_bet(bet.id, won, exit_price)

            self._save_state()

    async def real_trading_round(
        self,
        poly_client=None,
        kalshi_client=None,
        db_session=None,
        min_edge: float = 0.05,
        min_confidence: float = 0.5,
        max_markets_per_round: int = 20
    ) -> Dict[str, Any]:
        """
        Execute a real paper trading round using actual market data.

        Instead of synthetic simulation, this:
        1. Discovers real markets from Polymarket/Kalshi
        2. Each bot researches markets using its specialized methodology
        3. Places paper bets if edge is found
        4. Stores predictions for later resolution tracking

        Args:
            poly_client: PolymarketClient instance
            kalshi_client: KalshiClient instance
            db_session: SQLAlchemy session
            min_edge: Minimum edge required to place bet
            min_confidence: Minimum confidence required
            max_markets_per_round: Limit markets evaluated per round

        Returns:
            Summary of round results
        """
        if not HAS_REAL_TRADING:
            logger.warning("Real trading not available, falling back to simulation")
            self.simulate_round()
            return {"mode": "simulation", "reason": "modules_not_available"}

        if not poly_client and not kalshi_client:
            logger.warning("No API clients provided, falling back to simulation")
            self.simulate_round()
            return {"mode": "simulation", "reason": "no_clients"}

        round_stats = {
            "mode": "real",
            "timestamp": datetime.utcnow().isoformat(),
            "markets_scanned": 0,
            "markets_researched": 0,
            "bets_placed": 0,
            "bots_active": 0,
            "estimates_saved": 0,
            "bot_results": {}
        }

        with self._lock:
            # Step 1: Discover real markets
            scanner = MarketScanner(
                poly_client=poly_client,
                kalshi_client=kalshi_client,
                db_session=db_session
            )

            markets = scanner.scan_for_opportunities(
                min_liquidity=10000,
                max_days_to_resolution=180,
                min_days_to_resolution=1,
                min_volume_24h=1000,
                max_spread=0.10,
                limit_per_platform=max_markets_per_round
            )

            round_stats["markets_scanned"] = len(markets)
            logger.info(f"Real trading round: Found {len(markets)} market opportunities")

            if not markets:
                logger.warning("No markets found meeting criteria")
                return round_stats

            # Step 2: Each bot researches and potentially bets
            active_bots = [b for b in self.bots.values() if b.status != BotStatus.ELIMINATED]
            round_stats["bots_active"] = len(active_bots)

            # Initialize resolution tracker if we have db session
            resolution_tracker = None
            if db_session:
                resolution_tracker = ResolutionTracker(
                    db_session=db_session,
                    market_scanner=scanner
                )

            for bot in active_bots:
                bot_stats = {
                    "markets_evaluated": 0,
                    "estimates_made": 0,
                    "bets_placed": 0,
                    "total_edge": 0
                }

                # Get researcher for this bot's strategy
                researcher = get_researcher_for_bot(bot.strategy_type)

                # Evaluate markets (limit per bot to control API costs)
                markets_for_bot = markets[:max_markets_per_round]

                for market in markets_for_bot:
                    bot_stats["markets_evaluated"] += 1

                    try:
                        # Research the market
                        estimate = await researcher.research_market(
                            market_id=market.market_id,
                            platform=market.platform,
                            question=market.question,
                            description=market.description,
                            current_price=market.current_yes_price,
                            category=market.category,
                            extra_data=market.extra_data
                        )

                        if estimate is None:
                            continue

                        bot_stats["estimates_made"] += 1
                        round_stats["markets_researched"] += 1

                        # Check if edge meets threshold
                        edge = estimate.edge()
                        if edge < min_edge:
                            continue

                        if estimate.confidence < min_confidence:
                            continue

                        bot_stats["total_edge"] += edge

                        # Determine bet size based on tier and confidence
                        min_bet, max_bet = bot.tier.bet_range
                        if min_bet == 0:
                            base_amount = 100  # Simulation default
                        else:
                            base_amount = (min_bet + max_bet) / 2

                        # Scale by confidence and domain performance
                        domain_multiplier = 1.0
                        if db_session:
                            domain_spec = DomainSpecialist(db_session)
                            domain_multiplier = domain_spec.get_domain_multiplier(
                                bot.id, market.category
                            )

                        amount = base_amount * estimate.confidence * domain_multiplier

                        # Cap at 25% of capital
                        if amount > bot.current_capital * 0.25:
                            amount = bot.current_capital * 0.25

                        if amount < 10:
                            continue

                        # Record the bet
                        bet = self.record_bet(
                            bot_id=bot.id,
                            market_id=market.market_id,
                            market_question=market.question,
                            side=estimate.direction(),
                            amount=amount,
                            entry_price=market.current_yes_price if estimate.direction() == "yes" else market.current_no_price,
                            matures_at=market.end_date
                        )

                        if bet:
                            bot_stats["bets_placed"] += 1
                            round_stats["bets_placed"] += 1

                            # Save research estimate to database
                            if db_session:
                                research_record = ResearchEstimate(
                                    platform=market.platform,
                                    market_id=market.market_id,
                                    market_question=market.question,
                                    market_category=market.category,
                                    bot_id=bot.id,
                                    researcher_type=researcher.researcher_type,
                                    estimated_probability=estimate.estimated_probability,
                                    confidence=estimate.confidence,
                                    reasoning=estimate.reasoning,
                                    market_price_at_estimate=market.current_yes_price,
                                    edge_at_estimate=edge,
                                    sources_used=estimate.sources_used,
                                    raw_analysis=estimate.raw_analysis,
                                    paper_trade_id=None,  # Would link if we had trade ID
                                    created_at=datetime.utcnow()
                                )
                                db_session.add(research_record)
                                db_session.flush()
                                round_stats["estimates_saved"] += 1

                                # Add to pending resolution tracker
                                if resolution_tracker:
                                    resolution_tracker.add_pending_resolution(
                                        platform=market.platform,
                                        market_id=market.market_id,
                                        market_question=market.question,
                                        expected_resolution_date=market.end_date,
                                        research_estimate_id=research_record.id
                                    )

                            logger.info(
                                f"{bot.name} bet ${amount:.0f} {estimate.direction().upper()} "
                                f"on '{market.question[:40]}...' "
                                f"(edge: {edge:.1%}, conf: {estimate.confidence:.0%})"
                            )

                    except Exception as e:
                        logger.error(f"Error researching market for {bot.name}: {e}")
                        continue

                round_stats["bot_results"][bot.id] = bot_stats

            # Commit database changes
            if db_session:
                try:
                    db_session.commit()
                except Exception as e:
                    logger.error(f"Error committing database: {e}")
                    db_session.rollback()

            self._save_state()

        logger.info(
            f"Real trading round complete: {round_stats['bets_placed']} bets placed "
            f"across {round_stats['bots_active']} bots"
        )

        return round_stats

    def get_pending_resolutions_summary(self, db_session=None) -> Dict[str, Any]:
        """Get summary of markets awaiting resolution."""
        if not db_session or not HAS_REAL_TRADING:
            return {"available": False}

        try:
            pending = db_session.query(PendingResolution).filter_by(
                status="pending"
            ).all()

            return {
                "available": True,
                "total_pending": len(pending),
                "by_platform": {
                    "polymarket": len([p for p in pending if p.platform == "polymarket"]),
                    "kalshi": len([p for p in pending if p.platform == "kalshi"])
                },
                "upcoming_week": len([
                    p for p in pending
                    if p.expected_resolution_date
                    and (p.expected_resolution_date - datetime.utcnow()).days <= 7
                ]),
                "markets": [
                    {
                        "platform": p.platform,
                        "market_id": p.market_id,
                        "question": p.market_question[:100],
                        "expected_date": p.expected_resolution_date.isoformat() if p.expected_resolution_date else None,
                        "estimates_count": len(p.research_estimate_ids or []),
                        "trades_count": len(p.paper_trade_ids or [])
                    }
                    for p in pending[:20]
                ]
            }
        except Exception as e:
            logger.error(f"Error getting pending resolutions: {e}")
            return {"available": False, "error": str(e)}

    def get_bot_learning_summary(self, bot_id: str, db_session=None) -> Dict[str, Any]:
        """Get learning/calibration summary for a specific bot."""
        if not db_session or not HAS_REAL_TRADING:
            return {"available": False}

        try:
            calibration = CalibrationTracker(db_session)
            domain = DomainSpecialist(db_session)

            return {
                "available": True,
                "bot_id": bot_id,
                "calibration": calibration.get_calibration_summary(bot_id),
                "domains": domain.get_domain_summary(bot_id)
            }
        except Exception as e:
            logger.error(f"Error getting bot learning summary: {e}")
            return {"available": False, "error": str(e)}


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
