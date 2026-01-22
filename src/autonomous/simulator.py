"""
Continuous Simulation Runner

Runs simulations continuously to test strategies before live deployment.
Generates synthetic market scenarios and tracks simulated performance.
"""

import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class MarketScenario(Enum):
    """Types of market scenarios to simulate."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"
    RANGE_BOUND = "range_bound"
    BLACK_SWAN = "black_swan"
    NORMAL = "normal"


@dataclass
class SimulatedMarket:
    """A simulated prediction market."""
    market_id: str
    question: str
    current_price: float
    true_probability: float
    volatility: float
    trend: float  # -1 to 1, direction bias
    volume: float
    scenario: MarketScenario

    def step(self, dt: float = 1.0) -> float:
        """
        Advance market one time step.

        Args:
            dt: Time step size

        Returns:
            New price
        """
        # Random walk with drift
        drift = self.trend * 0.001 * dt
        noise = random.gauss(0, self.volatility * 0.01 * dt)

        # Mean reversion toward true probability
        reversion = (self.true_probability - self.current_price) * 0.01 * dt

        self.current_price += drift + noise + reversion
        self.current_price = max(0.01, min(0.99, self.current_price))

        return self.current_price


@dataclass
class SimulatedTrade:
    """A simulated trade."""
    market_id: str
    direction: str  # "buy" or "sell"
    price: float
    size: float
    timestamp: datetime
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: float = 0.0
    status: str = "open"  # "open", "closed", "expired"


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    simulation_id: str
    strategy_name: str
    scenario: MarketScenario
    duration_hours: int
    total_trades: int
    winning_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float
    start_time: datetime
    end_time: datetime
    parameters: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simulation_id': self.simulation_id,
            'strategy_name': self.strategy_name,
            'scenario': self.scenario.value,
            'duration_hours': self.duration_hours,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_trade_pnl': self.avg_trade_pnl,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'parameters': self.parameters
        }


class ContinuousSimulator:
    """
    Runs continuous simulations to validate strategies.

    Features:
    - Multiple market scenarios
    - Realistic price dynamics
    - Strategy comparison
    - Monte Carlo sampling
    - Performance tracking

    Usage:
        simulator = ContinuousSimulator()
        simulator.add_strategy("momentum", momentum_strategy_fn)
        simulator.start()
        results = simulator.get_results()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize simulator."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "simulations"
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.strategies: Dict[str, Callable] = {}
        self.results: List[SimulationResult] = []
        self.current_simulation: Optional[str] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Simulation settings
        self.simulation_hours = 24  # Each simulation covers 24 hours
        self.time_step_minutes = 5  # 5 minute intervals
        self.initial_capital = 10000
        self.markets_per_sim = 20

        self._load_results()

    def add_strategy(
        self,
        name: str,
        strategy_fn: Callable[[List[SimulatedMarket], Dict], Optional[SimulatedTrade]]
    ):
        """
        Add a strategy to simulate.

        Args:
            name: Strategy name
            strategy_fn: Function that takes (markets, portfolio) and returns trade or None
        """
        self.strategies[name] = strategy_fn
        logger.info(f"Added strategy: {name}")

    def add_default_strategies(self):
        """Add built-in default strategies."""

        def momentum_strategy(markets: List[SimulatedMarket], portfolio: Dict) -> Optional[SimulatedTrade]:
            """Simple momentum strategy."""
            for market in markets:
                # Look for trending markets
                if abs(market.trend) > 0.3 and market.volume > 1000:
                    if market.trend > 0 and market.current_price < 0.7:
                        return SimulatedTrade(
                            market_id=market.market_id,
                            direction="buy",
                            price=market.current_price,
                            size=portfolio['capital'] * 0.05,
                            timestamp=datetime.now()
                        )
                    elif market.trend < 0 and market.current_price > 0.3:
                        return SimulatedTrade(
                            market_id=market.market_id,
                            direction="sell",
                            price=market.current_price,
                            size=portfolio['capital'] * 0.05,
                            timestamp=datetime.now()
                        )
            return None

        def value_strategy(markets: List[SimulatedMarket], portfolio: Dict) -> Optional[SimulatedTrade]:
            """Buy undervalued, sell overvalued."""
            for market in markets:
                diff = market.true_probability - market.current_price
                if abs(diff) > 0.1:
                    direction = "buy" if diff > 0 else "sell"
                    return SimulatedTrade(
                        market_id=market.market_id,
                        direction=direction,
                        price=market.current_price,
                        size=portfolio['capital'] * 0.03,
                        timestamp=datetime.now()
                    )
            return None

        def contrarian_strategy(markets: List[SimulatedMarket], portfolio: Dict) -> Optional[SimulatedTrade]:
            """Bet against the crowd in volatile markets."""
            for market in markets:
                if market.scenario == MarketScenario.VOLATILE:
                    if market.current_price > 0.7:
                        return SimulatedTrade(
                            market_id=market.market_id,
                            direction="sell",
                            price=market.current_price,
                            size=portfolio['capital'] * 0.02,
                            timestamp=datetime.now()
                        )
                    elif market.current_price < 0.3:
                        return SimulatedTrade(
                            market_id=market.market_id,
                            direction="buy",
                            price=market.current_price,
                            size=portfolio['capital'] * 0.02,
                            timestamp=datetime.now()
                        )
            return None

        self.add_strategy("momentum", momentum_strategy)
        self.add_strategy("value", value_strategy)
        self.add_strategy("contrarian", contrarian_strategy)

    def _generate_markets(self, scenario: MarketScenario) -> List[SimulatedMarket]:
        """Generate simulated markets for a scenario."""
        markets = []

        # Scenario-specific parameters
        scenario_params = {
            MarketScenario.TRENDING_UP: {"trend_range": (0.2, 0.8), "vol_range": (0.1, 0.3)},
            MarketScenario.TRENDING_DOWN: {"trend_range": (-0.8, -0.2), "vol_range": (0.1, 0.3)},
            MarketScenario.VOLATILE: {"trend_range": (-0.3, 0.3), "vol_range": (0.4, 0.8)},
            MarketScenario.RANGE_BOUND: {"trend_range": (-0.1, 0.1), "vol_range": (0.05, 0.15)},
            MarketScenario.BLACK_SWAN: {"trend_range": (-1.0, 1.0), "vol_range": (0.6, 1.0)},
            MarketScenario.NORMAL: {"trend_range": (-0.2, 0.2), "vol_range": (0.1, 0.25)},
        }

        params = scenario_params[scenario]

        for i in range(self.markets_per_sim):
            true_prob = random.uniform(0.2, 0.8)
            markets.append(SimulatedMarket(
                market_id=f"sim_{scenario.value}_{i}",
                question=f"Simulated market {i} ({scenario.value})",
                current_price=true_prob + random.gauss(0, 0.1),
                true_probability=true_prob,
                volatility=random.uniform(*params["vol_range"]),
                trend=random.uniform(*params["trend_range"]),
                volume=random.uniform(500, 10000),
                scenario=scenario
            ))

        return markets

    def run_simulation(
        self,
        strategy_name: str,
        scenario: MarketScenario,
        parameters: Optional[Dict[str, float]] = None
    ) -> SimulationResult:
        """
        Run a single simulation.

        Args:
            strategy_name: Name of registered strategy
            scenario: Market scenario to simulate
            parameters: Optional strategy parameters

        Returns:
            SimulationResult with performance metrics
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy_fn = self.strategies[strategy_name]
        markets = self._generate_markets(scenario)

        # Initialize portfolio
        portfolio = {
            'capital': self.initial_capital,
            'positions': {},
            'trades': [],
            'pnl_history': [0.0]
        }

        start_time = datetime.now()
        simulation_id = f"{strategy_name}_{scenario.value}_{start_time.strftime('%Y%m%d%H%M%S')}"

        # Run simulation
        steps = (self.simulation_hours * 60) // self.time_step_minutes

        for step in range(steps):
            # Update market prices
            for market in markets:
                market.step(self.time_step_minutes)

            # Strategy decision
            try:
                trade = strategy_fn(markets, portfolio)
                if trade:
                    portfolio['trades'].append(trade)
                    portfolio['capital'] -= trade.size
            except Exception as e:
                logger.warning(f"Strategy error in {strategy_name}: {e}")

            # Close positions randomly (simulating market resolution)
            for trade in portfolio['trades']:
                if trade.status == "open" and random.random() < 0.01:
                    market = next((m for m in markets if m.market_id == trade.market_id), None)
                    if market:
                        trade.exit_price = market.current_price
                        trade.exit_timestamp = datetime.now()

                        # Calculate P&L
                        if trade.direction == "buy":
                            trade.pnl = (trade.exit_price - trade.price) * trade.size
                        else:
                            trade.pnl = (trade.price - trade.exit_price) * trade.size

                        portfolio['capital'] += trade.size + trade.pnl
                        trade.status = "closed"

            # Track P&L
            total_pnl = sum(t.pnl for t in portfolio['trades'] if t.status == "closed")
            portfolio['pnl_history'].append(total_pnl)

        end_time = datetime.now()

        # Calculate metrics
        closed_trades = [t for t in portfolio['trades'] if t.status == "closed"]
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        total_pnl = sum(t.pnl for t in closed_trades)

        # Sharpe ratio from P&L history
        pnl_changes = [
            portfolio['pnl_history'][i] - portfolio['pnl_history'][i-1]
            for i in range(1, len(portfolio['pnl_history']))
        ]

        if pnl_changes and len(pnl_changes) > 1:
            import statistics
            mean_pnl = statistics.mean(pnl_changes)
            std_pnl = statistics.stdev(pnl_changes) or 1
            sharpe = (mean_pnl / std_pnl) * (252 ** 0.5)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = 0
        max_drawdown = 0
        for pnl in portfolio['pnl_history']:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / max(peak, 1)
            max_drawdown = max(max_drawdown, drawdown)

        result = SimulationResult(
            simulation_id=simulation_id,
            strategy_name=strategy_name,
            scenario=scenario,
            duration_hours=self.simulation_hours,
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=winning_trades / max(1, total_trades),
            avg_trade_pnl=total_pnl / max(1, total_trades),
            start_time=start_time,
            end_time=end_time,
            parameters=parameters or {}
        )

        with self._lock:
            self.results.append(result)
            self._save_results()

        return result

    def run_monte_carlo(
        self,
        strategy_name: str,
        n_simulations: int = 100,
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation across all scenarios.

        Args:
            strategy_name: Strategy to test
            n_simulations: Number of simulations per scenario
            parameters: Optional strategy parameters

        Returns:
            Aggregated statistics
        """
        all_results = []

        for scenario in MarketScenario:
            sims_per_scenario = n_simulations // len(MarketScenario)
            for _ in range(sims_per_scenario):
                result = self.run_simulation(strategy_name, scenario, parameters)
                all_results.append(result)

        # Aggregate statistics
        pnls = [r.total_pnl for r in all_results]
        sharpes = [r.sharpe_ratio for r in all_results]
        win_rates = [r.win_rate for r in all_results]

        import statistics

        return {
            'strategy': strategy_name,
            'n_simulations': len(all_results),
            'mean_pnl': statistics.mean(pnls),
            'std_pnl': statistics.stdev(pnls) if len(pnls) > 1 else 0,
            'min_pnl': min(pnls),
            'max_pnl': max(pnls),
            'mean_sharpe': statistics.mean(sharpes),
            'mean_win_rate': statistics.mean(win_rates),
            'probability_profit': len([p for p in pnls if p > 0]) / len(pnls),
            'var_95': sorted(pnls)[int(len(pnls) * 0.05)] if pnls else 0,  # Value at Risk
        }

    def compare_strategies(self, n_simulations: int = 50) -> Dict[str, Dict[str, Any]]:
        """Compare all registered strategies."""
        comparison = {}

        for strategy_name in self.strategies:
            logger.info(f"Running Monte Carlo for {strategy_name}...")
            comparison[strategy_name] = self.run_monte_carlo(strategy_name, n_simulations)

        # Rank strategies
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1]['mean_sharpe'],
            reverse=True
        )

        for rank, (name, stats) in enumerate(ranked, 1):
            comparison[name]['rank'] = rank

        return comparison

    def start(self, interval_minutes: int = 30):
        """
        Start continuous simulation in background.

        Args:
            interval_minutes: Minutes between simulation runs
        """
        if self._running:
            logger.warning("Simulator already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, args=(interval_minutes,))
        self._thread.daemon = True
        self._thread.start()

        logger.info(f"Started continuous simulator (interval: {interval_minutes} minutes)")

    def stop(self):
        """Stop continuous simulation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped continuous simulator")

    def _run_loop(self, interval_minutes: int):
        """Main simulation loop."""
        while self._running:
            try:
                # Run simulation for each strategy and scenario
                for strategy_name in self.strategies:
                    scenario = random.choice(list(MarketScenario))
                    result = self.run_simulation(strategy_name, scenario)
                    logger.info(
                        f"Simulation: {strategy_name}/{scenario.value} - "
                        f"P&L: ${result.total_pnl:+.2f}, Sharpe: {result.sharpe_ratio:.2f}"
                    )

            except Exception as e:
                logger.error(f"Simulation error: {e}")

            # Wait for next interval
            time.sleep(interval_minutes * 60)

    def get_results(self, strategy: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get simulation results."""
        with self._lock:
            results = self.results

            if strategy:
                results = [r for r in results if r.strategy_name == strategy]

            results = sorted(results, key=lambda r: r.start_time, reverse=True)[:limit]
            return [r.to_dict() for r in results]

    def get_strategy_stats(self, strategy_name: str) -> Dict[str, Any]:
        """Get aggregate stats for a strategy."""
        with self._lock:
            results = [r for r in self.results if r.strategy_name == strategy_name]

            if not results:
                return {'strategy': strategy_name, 'simulations': 0}

            import statistics

            pnls = [r.total_pnl for r in results]
            sharpes = [r.sharpe_ratio for r in results]

            return {
                'strategy': strategy_name,
                'simulations': len(results),
                'mean_pnl': statistics.mean(pnls),
                'std_pnl': statistics.stdev(pnls) if len(pnls) > 1 else 0,
                'mean_sharpe': statistics.mean(sharpes),
                'best_pnl': max(pnls),
                'worst_pnl': min(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls)
            }

    def _save_results(self):
        """Save results to disk."""
        try:
            data = [r.to_dict() for r in self.results[-1000:]]  # Keep last 1000
            path = self.data_dir / "simulation_results.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving simulation results: {e}")

    def _load_results(self):
        """Load results from disk."""
        try:
            path = self.data_dir / "simulation_results.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                self.results = []
                for item in data:
                    self.results.append(SimulationResult(
                        simulation_id=item['simulation_id'],
                        strategy_name=item['strategy_name'],
                        scenario=MarketScenario(item['scenario']),
                        duration_hours=item['duration_hours'],
                        total_trades=item['total_trades'],
                        winning_trades=item['winning_trades'],
                        total_pnl=item['total_pnl'],
                        sharpe_ratio=item['sharpe_ratio'],
                        max_drawdown=item['max_drawdown'],
                        win_rate=item['win_rate'],
                        avg_trade_pnl=item['avg_trade_pnl'],
                        start_time=datetime.fromisoformat(item['start_time']),
                        end_time=datetime.fromisoformat(item['end_time']),
                        parameters=item.get('parameters', {})
                    ))

                logger.info(f"Loaded {len(self.results)} simulation results")

        except Exception as e:
            logger.warning(f"Could not load simulation results: {e}")


def get_simulator() -> ContinuousSimulator:
    """Get or create simulator instance."""
    return ContinuousSimulator()


if __name__ == "__main__":
    # Demo the simulator
    simulator = ContinuousSimulator()
    simulator.add_default_strategies()

    print("\n=== RUNNING STRATEGY COMPARISON ===\n")
    comparison = simulator.compare_strategies(n_simulations=30)

    print("\n=== RESULTS ===\n")
    for strategy_name, stats in sorted(comparison.items(), key=lambda x: x[1]['rank']):
        print(f"#{stats['rank']} {strategy_name}")
        print(f"   Mean P&L: ${stats['mean_pnl']:+,.2f} (±${stats['std_pnl']:,.2f})")
        print(f"   Mean Sharpe: {stats['mean_sharpe']:.2f}")
        print(f"   Win Rate: {stats['mean_win_rate']:.1%}")
        print(f"   P(Profit): {stats['probability_profit']:.1%}")
        print(f"   VaR 95%: ${stats['var_95']:,.2f}")
        print()
