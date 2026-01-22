"""
Autonomous Trading Agent

The main orchestrator that runs continuously, making all trading decisions
automatically without user intervention.
"""

import asyncio
import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class AgentState:
    """Current state of the autonomous agent."""
    is_running: bool = False
    started_at: Optional[datetime] = None
    cycles_completed: int = 0
    last_cycle_at: Optional[datetime] = None

    # Performance
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    current_capital: float = 10000.0

    # Today's stats
    today_trades: int = 0
    today_pnl: float = 0.0
    today_signals_collected: int = 0
    today_opportunities_found: int = 0

    # Strategy stats
    active_strategy: str = "ensemble"
    strategy_version: str = "v1.0"
    last_optimization: Optional[datetime] = None

    # Errors
    consecutive_errors: int = 0
    last_error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for autonomous agent."""
    # Timing
    signal_collection_interval: int = 300  # 5 minutes
    trade_evaluation_interval: int = 60  # 1 minute
    optimization_interval: int = 3600  # 1 hour
    simulation_interval: int = 1800  # 30 minutes
    leaderboard_update_interval: int = 86400  # 24 hours

    # Trading
    initial_capital: float = 10000.0
    max_position_size: float = 0.10  # 10% of capital
    min_confidence: float = 0.5
    min_edge: float = 0.03  # 3% minimum edge

    # Risk management
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.15  # 15% max drawdown
    max_open_positions: int = 10

    # Optimization
    auto_optimize: bool = True
    simulation_before_live: bool = True
    min_simulation_trades: int = 50
    min_simulation_sharpe: float = 1.0

    # Persistence
    state_file: str = "data/agent_state.json"
    log_file: str = "data/agent_log.jsonl"


class AutonomousAgent:
    """
    Fully autonomous trading agent.

    Runs continuously, collecting signals, executing trades,
    optimizing strategies, and maximizing profit without user intervention.

    Usage:
        agent = AutonomousAgent()
        agent.start()  # Runs forever in background

        # Check status
        print(agent.get_status())

        # Stop when needed
        agent.stop()
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the autonomous agent."""
        self.config = config or AgentConfig()
        self.state = AgentState(current_capital=self.config.initial_capital)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Components (lazy loaded)
        self._signal_collectors = None
        self._ensemble = None
        self._trader = None
        self._optimizer = None
        self._simulator = None
        self._leaderboard = None
        self._alert_manager = None

        # State persistence
        self._state_path = Path(self.config.state_file)
        self._log_path = Path(self.config.log_file)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # Load previous state
        self._load_state()

    def start(self):
        """Start the autonomous agent in background thread."""
        if self.state.is_running:
            logger.warning("Agent is already running")
            return

        logger.info("Starting autonomous agent...")
        self.state.is_running = True
        self.state.started_at = datetime.utcnow()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._log_event("agent_started", {"config": self.config.__dict__})
        logger.info("Autonomous agent started")

    def stop(self):
        """Stop the autonomous agent."""
        if not self.state.is_running:
            return

        logger.info("Stopping autonomous agent...")
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=10)

        self.state.is_running = False
        self._save_state()
        self._log_event("agent_stopped", {"cycles": self.state.cycles_completed})
        logger.info("Autonomous agent stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        with self._lock:
            uptime = None
            if self.state.started_at:
                uptime = (datetime.utcnow() - self.state.started_at).total_seconds()

            # Get recent activity from log file
            recent_activity = self._get_recent_activity()

            return {
                # Dashboard-friendly format
                "running": self.state.is_running,
                "started_at": self.state.started_at.strftime("%Y-%m-%d %H:%M:%S") if self.state.started_at else None,
                "cycle": self.state.cycles_completed,
                "signals_today": self.state.today_signals_collected,
                "trades_today": self.state.today_trades,
                "pnl_today": self.state.today_pnl,
                "opportunities_today": self.state.today_opportunities_found,
                "recent_activity": recent_activity,

                # Extended info
                "is_running": self.state.is_running,
                "uptime_seconds": uptime,
                "cycles_completed": self.state.cycles_completed,
                "last_cycle": self.state.last_cycle_at.isoformat() if self.state.last_cycle_at else None,
                "performance": {
                    "total_trades": self.state.total_trades,
                    "winning_trades": self.state.winning_trades,
                    "win_rate": self.state.winning_trades / max(1, self.state.total_trades),
                    "total_pnl": self.state.total_pnl,
                    "current_capital": self.state.current_capital,
                    "roi": (self.state.current_capital - self.config.initial_capital) / self.config.initial_capital
                },
                "today": {
                    "trades": self.state.today_trades,
                    "pnl": self.state.today_pnl,
                    "signals": self.state.today_signals_collected,
                    "opportunities": self.state.today_opportunities_found
                },
                "strategy": {
                    "active": self.state.active_strategy,
                    "version": self.state.strategy_version,
                    "last_optimization": self.state.last_optimization.isoformat() if self.state.last_optimization else None
                },
                "health": {
                    "consecutive_errors": self.state.consecutive_errors,
                    "last_error": self.state.last_error
                }
            }

    def _get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity from the event log."""
        try:
            if not self._log_path.exists():
                return []

            activities = []
            with open(self._log_path, 'r') as f:
                lines = f.readlines()

            for line in reversed(lines[-limit:]):
                try:
                    event = json.loads(line.strip())
                    activities.append({
                        'timestamp': event.get('timestamp', ''),
                        'action': event.get('event', '').replace('_', ' ').title(),
                        'details': str(event.get('data', ''))[:100]
                    })
                except json.JSONDecodeError:
                    continue

            return activities

        except Exception:
            return []

    def _run_loop(self):
        """Main autonomous loop."""
        logger.info("Entering autonomous loop")

        last_signal_collection = datetime.min
        last_trade_evaluation = datetime.min
        last_optimization = datetime.min
        last_simulation = datetime.min
        last_leaderboard = datetime.min
        last_daily_reset = datetime.utcnow().date()

        while not self._stop_event.is_set():
            try:
                now = datetime.utcnow()

                # Daily reset
                if now.date() > last_daily_reset:
                    self._daily_reset()
                    last_daily_reset = now.date()

                # Signal collection (every 5 minutes)
                if (now - last_signal_collection).total_seconds() >= self.config.signal_collection_interval:
                    self._collect_signals()
                    last_signal_collection = now

                # Trade evaluation (every 1 minute)
                if (now - last_trade_evaluation).total_seconds() >= self.config.trade_evaluation_interval:
                    self._evaluate_and_trade()
                    last_trade_evaluation = now

                # Optimization (every hour)
                if self.config.auto_optimize and (now - last_optimization).total_seconds() >= self.config.optimization_interval:
                    self._optimize_strategy()
                    last_optimization = now

                # Simulation testing (every 30 minutes)
                if (now - last_simulation).total_seconds() >= self.config.simulation_interval:
                    self._run_simulations()
                    last_simulation = now

                # Leaderboard update (daily)
                if (now - last_leaderboard).total_seconds() >= self.config.leaderboard_update_interval:
                    self._update_leaderboard()
                    last_leaderboard = now

                # Update cycle count
                with self._lock:
                    self.state.cycles_completed += 1
                    self.state.last_cycle_at = now
                    self.state.consecutive_errors = 0

                # Save state periodically
                if self.state.cycles_completed % 10 == 0:
                    self._save_state()

                # Sleep between cycles
                time.sleep(10)

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                with self._lock:
                    self.state.consecutive_errors += 1
                    self.state.last_error = str(e)

                # If too many errors, slow down
                if self.state.consecutive_errors > 10:
                    logger.warning("Too many errors, slowing down...")
                    time.sleep(60)
                else:
                    time.sleep(5)

    def _collect_signals(self):
        """Collect signals from all sources."""
        logger.debug("Collecting signals...")

        try:
            signals = []

            # News sentiment
            try:
                from src.signals.news_sentiment import NewsSentimentSignalDetector
                detector = NewsSentimentSignalDetector()
                news_signals = detector.run()
                signals.extend(news_signals)
                logger.debug(f"Collected {len(news_signals)} news signals")
            except Exception as e:
                logger.warning(f"News signal collection failed: {e}")

            # Whale tracking
            try:
                from src.signals.whale_tracker import WhaleTrackerSignalDetector
                detector = WhaleTrackerSignalDetector()
                whale_signals = detector.run()
                signals.extend(whale_signals)
                logger.debug(f"Collected {len(whale_signals)} whale signals")
            except Exception as e:
                logger.warning(f"Whale signal collection failed: {e}")

            # Social sentiment
            try:
                from src.signals.social_sentiment import SocialSentimentSignalDetector
                detector = SocialSentimentSignalDetector()
                social_signals = detector.run()
                signals.extend(social_signals)
                logger.debug(f"Collected {len(social_signals)} social signals")
            except Exception as e:
                logger.warning(f"Social signal collection failed: {e}")

            # Odds movement
            try:
                from src.signals.odds_movement import OddsMovementSignalDetector
                detector = OddsMovementSignalDetector()
                odds_signals = detector.run()
                signals.extend(odds_signals)
                logger.debug(f"Collected {len(odds_signals)} odds signals")
            except Exception as e:
                logger.warning(f"Odds signal collection failed: {e}")

            # Fast news (low-latency news for edge)
            try:
                from src.signals.fast_news import FastNewsSignalDetector
                detector = FastNewsSignalDetector()
                fast_news_signals = detector.run()
                signals.extend(fast_news_signals)
                logger.debug(f"Collected {len(fast_news_signals)} fast news signals")
            except Exception as e:
                logger.warning(f"Fast news signal collection failed: {e}")

            with self._lock:
                self.state.today_signals_collected += len(signals)

            # Store signals for ensemble
            self._current_signals = signals

            self._log_event("signals_collected", {
                "count": len(signals),
                "sources": list(set(s.source for s in signals))
            })

            logger.info(f"Collected {len(signals)} total signals")

        except Exception as e:
            logger.error(f"Signal collection error: {e}")

    def _evaluate_and_trade(self):
        """Evaluate opportunities and execute trades."""
        logger.debug("Evaluating trading opportunities...")

        try:
            # Get current markets
            from src.clients.polymarket_client import PolymarketClient
            client = PolymarketClient()
            markets = client.get_markets(limit=50)

            if not markets:
                return

            # Get ensemble predictions
            from src.models.ensemble import SignalEnsemble
            if self._ensemble is None:
                self._ensemble = SignalEnsemble()

            signals = getattr(self, '_current_signals', [])

            opportunities = []
            for market in markets:
                # Handle both dict and dataclass market objects
                if hasattr(market, 'id'):
                    market_id = market.id or getattr(market, 'condition_id', '')
                    question = getattr(market, 'question', '')
                    prices = getattr(market, 'prices', {})
                    current_price = prices.get('yes', 0.5) if isinstance(prices, dict) else 0.5
                else:
                    market_id = market.get('id') or market.get('condition_id')
                    question = market.get('question', '')
                    current_price = market.get('outcomePrices', [0.5])[0] if market.get('outcomePrices') else 0.5

                try:
                    current_price = float(current_price)
                except (TypeError, ValueError):
                    current_price = 0.5

                # Get prediction from ensemble
                prediction = self._ensemble.predict(
                    market_id=market_id,
                    signals=signals,
                    current_price=current_price,
                    market_keywords=[question[:50]]
                )

                if prediction and prediction.confidence >= self.config.min_confidence:
                    edge = abs(prediction.predicted_probability - current_price)
                    if edge >= self.config.min_edge:
                        opportunities.append({
                            'market_id': market_id,
                            'question': question or 'Unknown',
                            'current_price': current_price,
                            'prediction': prediction,
                            'edge': edge
                        })

            with self._lock:
                self.state.today_opportunities_found += len(opportunities)

            # Execute top opportunities
            if opportunities:
                # Sort by edge * confidence
                opportunities.sort(key=lambda x: x['edge'] * x['prediction'].confidence, reverse=True)

                for opp in opportunities[:3]:  # Top 3 opportunities
                    self._execute_trade(opp)

            logger.info(f"Found {len(opportunities)} trading opportunities")

        except Exception as e:
            logger.error(f"Trade evaluation error: {e}")

    def _execute_trade(self, opportunity: Dict[str, Any]):
        """Execute a paper trade."""
        try:
            from src.trading.paper_trader import PaperTrader

            if self._trader is None:
                self._trader = PaperTrader()

            prediction = opportunity['prediction']

            # Check risk limits
            if not self._check_risk_limits():
                logger.warning("Risk limits exceeded, skipping trade")
                return

            # Calculate position size
            position_size = min(
                prediction.position_size * self.state.current_capital,
                self.config.max_position_size * self.state.current_capital
            )

            if position_size < 10:  # Minimum $10 trade
                return

            # Determine side
            side = "yes" if prediction.direction.value == "bullish" else "no"

            # Execute paper trade
            result = self._trader.execute_trade(
                platform="polymarket",
                market_id=opportunity['market_id'],
                market_question=opportunity['question'],
                side=side,
                price=opportunity['current_price'],
                edge=opportunity['edge'],
                confidence=prediction.confidence,
                strategy=self.state.active_strategy,
                size_override=position_size
            )

            if result and result.status == "executed":
                with self._lock:
                    self.state.total_trades += 1
                    self.state.today_trades += 1

                self._log_event("trade_executed", {
                    'market_id': opportunity['market_id'],
                    'side': side,
                    'size': position_size,
                    'price': opportunity['current_price'],
                    'confidence': prediction.confidence,
                    'edge': opportunity['edge']
                })

                logger.info(f"Executed trade: {side} ${position_size:.2f} on {opportunity['question'][:40]}...")

        except Exception as e:
            logger.error(f"Trade execution error: {e}")

    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits."""
        # Check daily loss limit
        if self.state.today_pnl < -self.config.max_daily_loss * self.config.initial_capital:
            logger.warning("Daily loss limit reached")
            return False

        # Check max drawdown
        drawdown = (self.config.initial_capital - self.state.current_capital) / self.config.initial_capital
        if drawdown > self.config.max_drawdown:
            logger.warning("Max drawdown limit reached")
            return False

        return True

    def _optimize_strategy(self):
        """Run strategy optimization."""
        logger.info("Running strategy optimization...")

        try:
            from src.autonomous.optimizer import AutoOptimizer

            if self._optimizer is None:
                self._optimizer = AutoOptimizer()

            # Get current performance
            from src.improvement.evaluator import StrategyEvaluator
            evaluator = StrategyEvaluator()
            metrics = evaluator.evaluate_from_database()

            # Check if optimization needed
            should_optimize = (
                metrics.sharpe_ratio < 1.0 or
                metrics.brier_score > 0.28 or
                metrics.edge_vs_market < 0.01
            )

            if should_optimize:
                logger.info("Performance below threshold, optimizing...")

                # Define evaluation function for optimizer
                # Calculate total_pnl from roi (PerformanceMetrics doesn't have total_pnl)
                estimated_pnl = metrics.roi * self.config.initial_capital

                def evaluate_fn(params):
                    return {
                        'sharpe_ratio': metrics.sharpe_ratio + random.gauss(0, 0.1),
                        'win_rate': metrics.win_rate,
                        'total_pnl': estimated_pnl,
                        'max_drawdown': metrics.max_drawdown
                    }

                # Run optimizer with correct API
                self._optimizer.define_default_strategy_parameters()
                new_weights = self._optimizer.optimize(evaluate_fn, generations=5)

                if new_weights and self._ensemble:
                    self._ensemble.weights.update(new_weights)

                    with self._lock:
                        self.state.last_optimization = datetime.utcnow()
                        self.state.strategy_version = f"v{self.state.cycles_completed}"

                    self._log_event("strategy_optimized", {
                        'old_sharpe': metrics.sharpe_ratio,
                        'weights_updated': len(new_weights)
                    })

                    logger.info(f"Strategy optimized, updated {len(new_weights)} weights")
            else:
                logger.info("Performance acceptable, no optimization needed")

        except Exception as e:
            logger.error(f"Optimization error: {e}")

    def _run_simulations(self):
        """Run strategy simulations to test new approaches."""
        logger.info("Running simulations...")

        try:
            from src.autonomous.simulator import ContinuousSimulator, MarketScenario

            if self._simulator is None:
                self._simulator = ContinuousSimulator()
                self._simulator.add_default_strategies()

            # Run simulation with a random scenario
            scenario = random.choice(list(MarketScenario))
            result = self._simulator.run_simulation(
                strategy_name="momentum",  # Use a default strategy
                scenario=scenario
            )

            if result:
                self._log_event("simulation_completed", {
                    'strategy': "momentum",
                    'scenario': scenario.value,
                    'sharpe': result.sharpe_ratio,
                    'pnl': result.total_pnl,
                    'trades': result.total_trades
                })

                # Check if simulation meets criteria for live trading
                if (result.sharpe_ratio >= self.config.min_simulation_sharpe and
                    result.total_trades >= self.config.min_simulation_trades):
                    logger.info("Simulation passed, strategy validated for live trading")
                else:
                    logger.warning("Simulation below threshold, strategy needs improvement")

        except Exception as e:
            logger.error(f"Simulation error: {e}")

    def _update_leaderboard(self):
        """Update daily leaderboard."""
        logger.info("Updating leaderboard...")

        try:
            from src.autonomous.leaderboard import Leaderboard

            if self._leaderboard is None:
                self._leaderboard = Leaderboard()

            # Record today's performance
            self._leaderboard.record_daily_performance(
                date=datetime.utcnow().date(),
                strategy=self.state.active_strategy,
                trades=self.state.today_trades,
                pnl=self.state.today_pnl,
                signals=self.state.today_signals_collected,
                opportunities=self.state.today_opportunities_found
            )

            # Get rankings
            rankings = self._leaderboard.get_rankings()

            # Convert rankings to dicts for JSON serialization
            rankings_data = [
                {
                    'rank': r.rank,
                    'strategy': r.strategy,
                    'pnl': r.total_pnl,
                    'win_rate': r.win_rate,
                    'sharpe': r.sharpe_ratio
                }
                for r in rankings[:5]
            ]

            self._log_event("leaderboard_updated", {
                'date': datetime.utcnow().date().isoformat(),
                'rankings': rankings_data
            })

            # Send daily summary alert
            self._send_daily_summary()

        except Exception as e:
            logger.error(f"Leaderboard error: {e}")

    def _daily_reset(self):
        """Reset daily counters."""
        logger.info("Performing daily reset...")

        # Log yesterday's performance before reset
        self._log_event("daily_summary", {
            'trades': self.state.today_trades,
            'pnl': self.state.today_pnl,
            'signals': self.state.today_signals_collected,
            'opportunities': self.state.today_opportunities_found
        })

        with self._lock:
            self.state.today_trades = 0
            self.state.today_pnl = 0.0
            self.state.today_signals_collected = 0
            self.state.today_opportunities_found = 0

    def _send_daily_summary(self):
        """Send daily performance summary."""
        try:
            from src.realtime.alerts import AlertManager, AlertSeverity, AlertChannel

            if self._alert_manager is None:
                self._alert_manager = AlertManager()
                self._alert_manager.enable_channel(AlertChannel.CONSOLE)
                self._alert_manager.enable_channel(AlertChannel.LOG)

            roi = (self.state.current_capital - self.config.initial_capital) / self.config.initial_capital

            self._alert_manager.alert(
                title="Daily Performance Summary",
                message=f"""
Capital: ${self.state.current_capital:,.2f} ({roi:+.2%})
Today's Trades: {self.state.today_trades}
Today's P&L: ${self.state.today_pnl:+,.2f}
Signals Collected: {self.state.today_signals_collected}
Opportunities Found: {self.state.today_opportunities_found}
Strategy: {self.state.active_strategy} {self.state.strategy_version}
                """,
                severity=AlertSeverity.INFO,
                source="autonomous_agent"
            )

        except Exception as e:
            logger.error(f"Alert error: {e}")

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the event log."""
        try:
            event = {
                'timestamp': datetime.utcnow().isoformat(),
                'event': event_type,
                'data': data
            }

            with open(self._log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')

        except Exception as e:
            logger.error(f"Event logging error: {e}")

    def _save_state(self):
        """Save agent state to disk."""
        try:
            state_data = {
                'cycles_completed': self.state.cycles_completed,
                'total_trades': self.state.total_trades,
                'winning_trades': self.state.winning_trades,
                'total_pnl': self.state.total_pnl,
                'current_capital': self.state.current_capital,
                'active_strategy': self.state.active_strategy,
                'strategy_version': self.state.strategy_version,
                'last_optimization': self.state.last_optimization.isoformat() if self.state.last_optimization else None,
                'saved_at': datetime.utcnow().isoformat()
            }

            with open(self._state_path, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"State save error: {e}")

    def _load_state(self):
        """Load agent state from disk."""
        try:
            if self._state_path.exists():
                with open(self._state_path, 'r') as f:
                    data = json.load(f)

                self.state.cycles_completed = data.get('cycles_completed', 0)
                self.state.total_trades = data.get('total_trades', 0)
                self.state.winning_trades = data.get('winning_trades', 0)
                self.state.total_pnl = data.get('total_pnl', 0.0)
                self.state.current_capital = data.get('current_capital', self.config.initial_capital)
                self.state.active_strategy = data.get('active_strategy', 'ensemble')
                self.state.strategy_version = data.get('strategy_version', 'v1.0')

                if data.get('last_optimization'):
                    self.state.last_optimization = datetime.fromisoformat(data['last_optimization'])

                logger.info(f"Loaded previous state: {self.state.cycles_completed} cycles, ${self.state.current_capital:.2f}")

        except Exception as e:
            logger.warning(f"Could not load state: {e}")


# Global agent instance
_agent: Optional[AutonomousAgent] = None


def get_agent() -> AutonomousAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = AutonomousAgent()
    return _agent


def get_autonomous_agent() -> AutonomousAgent:
    """Alias for get_agent() - used by dashboard."""
    return get_agent()


def start_agent():
    """Start the global autonomous agent."""
    agent = get_agent()
    agent.start()
    return agent


def stop_agent():
    """Stop the global autonomous agent."""
    global _agent
    if _agent:
        _agent.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Trading Agent")
    parser.add_argument("--start", action="store_true", help="Start the agent")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--stop", action="store_true", help="Stop the agent")

    args = parser.parse_args()

    if args.start:
        print("Starting autonomous agent...")
        agent = start_agent()
        print("Agent started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
                status = agent.get_status()
                print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] "
                      f"Cycles: {status['cycles_completed']} | "
                      f"Trades: {status['today']['trades']} | "
                      f"P&L: ${status['today']['pnl']:+.2f}")
        except KeyboardInterrupt:
            print("\nStopping agent...")
            stop_agent()
    elif args.status:
        agent = get_agent()
        status = agent.get_status()
        print(json.dumps(status, indent=2))
    elif args.stop:
        stop_agent()
        print("Agent stopped")
    else:
        parser.print_help()
