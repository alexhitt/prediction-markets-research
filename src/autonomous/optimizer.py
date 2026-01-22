"""
Auto-Optimization Engine

Automatically adjusts strategy parameters to maximize performance.
Uses genetic algorithms and Bayesian optimization to find optimal settings.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ParameterRange:
    """Defines a parameter's search space."""
    name: str
    min_val: float
    max_val: float
    current: float
    step: float = 0.01
    param_type: str = "continuous"  # "continuous", "discrete", "categorical"
    categories: List[Any] = field(default_factory=list)

    def sample_random(self) -> float:
        """Sample a random value from the range."""
        if self.param_type == "categorical":
            return random.choice(self.categories)
        elif self.param_type == "discrete":
            steps = int((self.max_val - self.min_val) / self.step)
            return self.min_val + random.randint(0, steps) * self.step
        else:
            return random.uniform(self.min_val, self.max_val)

    def mutate(self, value: float, mutation_rate: float = 0.1) -> float:
        """Mutate a value within the range."""
        if self.param_type == "categorical":
            if random.random() < mutation_rate:
                return random.choice(self.categories)
            return value

        delta = (self.max_val - self.min_val) * mutation_rate
        new_val = value + random.uniform(-delta, delta)
        return max(self.min_val, min(self.max_val, new_val))


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    parameters: Dict[str, float]
    fitness: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_pnl: float
    generation: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameters': self.parameters,
            'fitness': self.fitness,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'generation': self.generation,
            'timestamp': self.timestamp.isoformat()
        }


class AutoOptimizer:
    """
    Automatic strategy parameter optimization.

    Uses a hybrid approach:
    - Genetic algorithm for exploration
    - Bayesian-inspired selection for exploitation
    - Periodic random injection to avoid local minima

    Usage:
        optimizer = AutoOptimizer()
        optimizer.define_parameters(strategy_params)
        best_params = optimizer.optimize(evaluate_fn, generations=50)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize optimizer."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "optimization"
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.parameters: Dict[str, ParameterRange] = {}
        self.population: List[Dict[str, float]] = []
        self.history: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

        # Optimization settings
        self.population_size = 20
        self.elite_count = 4
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.random_injection_rate = 0.1

        self._load_history()

    def define_parameters(self, params: Dict[str, Dict[str, Any]]):
        """
        Define parameters to optimize.

        Args:
            params: Dict mapping param names to their definitions
                    Each definition should have: min, max, current, step (optional)

        Example:
            optimizer.define_parameters({
                'momentum_threshold': {'min': 0.01, 'max': 0.1, 'current': 0.05},
                'position_size': {'min': 0.01, 'max': 0.2, 'current': 0.1},
            })
        """
        for name, config in params.items():
            self.parameters[name] = ParameterRange(
                name=name,
                min_val=config.get('min', 0),
                max_val=config.get('max', 1),
                current=config.get('current', 0.5),
                step=config.get('step', 0.01),
                param_type=config.get('type', 'continuous'),
                categories=config.get('categories', [])
            )

        logger.info(f"Defined {len(self.parameters)} parameters for optimization")

    def define_default_strategy_parameters(self):
        """Define default parameters for the trading strategies."""
        self.define_parameters({
            # Signal weights
            'momentum_weight': {'min': 0.0, 'max': 1.0, 'current': 0.25},
            'arbitrage_weight': {'min': 0.0, 'max': 1.0, 'current': 0.3},
            'sentiment_weight': {'min': 0.0, 'max': 1.0, 'current': 0.2},
            'whale_weight': {'min': 0.0, 'max': 1.0, 'current': 0.15},
            'odds_weight': {'min': 0.0, 'max': 1.0, 'current': 0.1},

            # Thresholds
            'min_confidence': {'min': 0.5, 'max': 0.95, 'current': 0.7},
            'position_size_pct': {'min': 0.01, 'max': 0.15, 'current': 0.05},
            'max_position_pct': {'min': 0.1, 'max': 0.5, 'current': 0.2},

            # Timing
            'momentum_lookback': {'min': 3, 'max': 30, 'current': 14, 'type': 'discrete', 'step': 1},
            'rebalance_hours': {'min': 1, 'max': 24, 'current': 6, 'type': 'discrete', 'step': 1},

            # Risk management
            'stop_loss_pct': {'min': 0.02, 'max': 0.15, 'current': 0.05},
            'take_profit_pct': {'min': 0.05, 'max': 0.5, 'current': 0.15},
            'max_drawdown_limit': {'min': 0.1, 'max': 0.3, 'current': 0.2},
        })

    def _initialize_population(self):
        """Initialize random population."""
        self.population = []

        # Add current best if exists
        if self.best_result:
            self.population.append(self.best_result.parameters.copy())

        # Add current values
        current = {name: p.current for name, p in self.parameters.items()}
        self.population.append(current)

        # Fill rest with random
        while len(self.population) < self.population_size:
            individual = {
                name: p.sample_random()
                for name, p in self.parameters.items()
            }
            self.population.append(individual)

    def _evaluate_fitness(
        self,
        params: Dict[str, float],
        evaluate_fn: Callable[[Dict[str, float]], Dict[str, float]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate fitness of a parameter set.

        Fitness combines multiple objectives:
        - Sharpe ratio (risk-adjusted returns)
        - Win rate
        - Total P&L
        - Penalized by drawdown
        """
        try:
            metrics = evaluate_fn(params)

            sharpe = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0)
            pnl = metrics.get('total_pnl', 0)
            drawdown = metrics.get('max_drawdown', 0)

            # Composite fitness (higher is better)
            fitness = (
                sharpe * 0.35 +
                win_rate * 100 * 0.25 +
                (pnl / 1000) * 0.25 +  # Normalize P&L
                (1 - drawdown) * 50 * 0.15  # Penalize drawdown
            )

            # Penalize extreme drawdowns
            if drawdown > 0.25:
                fitness *= 0.5

            return fitness, metrics

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return -1000, {}

    def _select_parents(
        self,
        fitnesses: List[float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Select two parents using tournament selection."""
        tournament_size = 3

        def tournament():
            candidates = random.sample(
                list(zip(self.population, fitnesses)),
                min(tournament_size, len(self.population))
            )
            return max(candidates, key=lambda x: x[1])[0]

        return tournament(), tournament()

    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """Create offspring through crossover."""
        if random.random() > self.crossover_rate:
            return parent1.copy()

        child = {}
        for name in self.parameters:
            # Uniform crossover
            if random.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]

        return child

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation to an individual."""
        mutated = {}
        for name, value in individual.items():
            if name in self.parameters:
                mutated[name] = self.parameters[name].mutate(
                    value, self.mutation_rate
                )
            else:
                mutated[name] = value
        return mutated

    def optimize(
        self,
        evaluate_fn: Callable[[Dict[str, float]], Dict[str, float]],
        generations: int = 30,
        early_stop_generations: int = 10,
        target_sharpe: float = 2.0
    ) -> Dict[str, float]:
        """
        Run optimization to find best parameters.

        Args:
            evaluate_fn: Function that takes params and returns metrics dict
                        Expected keys: sharpe_ratio, win_rate, total_pnl, max_drawdown
            generations: Maximum generations to run
            early_stop_generations: Stop if no improvement for this many generations
            target_sharpe: Stop if Sharpe ratio exceeds this

        Returns:
            Best parameter set found
        """
        logger.info(f"Starting optimization for {generations} generations")

        if not self.parameters:
            self.define_default_strategy_parameters()

        self._initialize_population()

        no_improvement_count = 0
        best_fitness = float('-inf')

        for gen in range(generations):
            # Evaluate all individuals
            results = []
            for individual in self.population:
                fitness, metrics = self._evaluate_fitness(individual, evaluate_fn)
                results.append((individual, fitness, metrics))

            # Sort by fitness
            results.sort(key=lambda x: x[1], reverse=True)
            fitnesses = [r[1] for r in results]

            # Track best
            best_individual, best_gen_fitness, best_metrics = results[0]

            if best_gen_fitness > best_fitness:
                best_fitness = best_gen_fitness
                no_improvement_count = 0

                self.best_result = OptimizationResult(
                    parameters=best_individual.copy(),
                    fitness=best_gen_fitness,
                    sharpe_ratio=best_metrics.get('sharpe_ratio', 0),
                    max_drawdown=best_metrics.get('max_drawdown', 0),
                    win_rate=best_metrics.get('win_rate', 0),
                    total_pnl=best_metrics.get('total_pnl', 0),
                    generation=gen
                )
                self.history.append(self.best_result)

                logger.info(
                    f"Gen {gen}: New best fitness={best_gen_fitness:.2f}, "
                    f"Sharpe={best_metrics.get('sharpe_ratio', 0):.2f}, "
                    f"WinRate={best_metrics.get('win_rate', 0):.1%}"
                )
            else:
                no_improvement_count += 1

            # Early stopping checks
            if no_improvement_count >= early_stop_generations:
                logger.info(f"Early stop: no improvement for {early_stop_generations} generations")
                break

            if best_metrics.get('sharpe_ratio', 0) >= target_sharpe:
                logger.info(f"Target Sharpe {target_sharpe} reached!")
                break

            # Create next generation
            new_population = []

            # Elitism: keep best individuals
            for i in range(self.elite_count):
                new_population.append(results[i][0].copy())

            # Random injection
            num_random = int(self.population_size * self.random_injection_rate)
            for _ in range(num_random):
                random_individual = {
                    name: p.sample_random()
                    for name, p in self.parameters.items()
                }
                new_population.append(random_individual)

            # Fill rest through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(fitnesses)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population

        self._save_history()

        if self.best_result:
            logger.info(
                f"Optimization complete. Best: Sharpe={self.best_result.sharpe_ratio:.2f}, "
                f"P&L=${self.best_result.total_pnl:,.2f}"
            )
            return self.best_result.parameters

        return {name: p.current for name, p in self.parameters.items()}

    def get_best_parameters(self) -> Dict[str, float]:
        """Get current best parameters."""
        if self.best_result:
            return self.best_result.parameters
        return {name: p.current for name, p in self.parameters.items()}

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return [r.to_dict() for r in self.history]

    def suggest_next_experiment(self) -> Dict[str, float]:
        """
        Suggest parameters for next manual experiment.

        Uses acquisition function inspired by Bayesian optimization:
        - Explore areas with high uncertainty
        - Exploit areas near current best
        """
        if not self.best_result:
            return {name: p.sample_random() for name, p in self.parameters.items()}

        # Mix of exploitation (near best) and exploration (random)
        suggestion = {}
        best_params = self.best_result.parameters

        for name, param in self.parameters.items():
            if random.random() < 0.7:  # Exploit
                # Small perturbation around best
                suggestion[name] = param.mutate(best_params.get(name, param.current), 0.05)
            else:  # Explore
                suggestion[name] = param.sample_random()

        return suggestion

    def _save_history(self):
        """Save optimization history to disk."""
        try:
            data = {
                'best_result': self.best_result.to_dict() if self.best_result else None,
                'history': [r.to_dict() for r in self.history[-100:]]  # Keep last 100
            }
            path = self.data_dir / "optimization_history.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimization history: {e}")

    def _load_history(self):
        """Load optimization history from disk."""
        try:
            path = self.data_dir / "optimization_history.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                if data.get('best_result'):
                    br = data['best_result']
                    self.best_result = OptimizationResult(
                        parameters=br['parameters'],
                        fitness=br['fitness'],
                        sharpe_ratio=br['sharpe_ratio'],
                        max_drawdown=br['max_drawdown'],
                        win_rate=br['win_rate'],
                        total_pnl=br['total_pnl'],
                        generation=br['generation'],
                        timestamp=datetime.fromisoformat(br['timestamp'])
                    )

                logger.info("Loaded optimization history")

        except Exception as e:
            logger.warning(f"Could not load optimization history: {e}")


def get_optimizer() -> AutoOptimizer:
    """Get or create optimizer instance."""
    return AutoOptimizer()


if __name__ == "__main__":
    # Demo the optimizer
    optimizer = AutoOptimizer()
    optimizer.define_default_strategy_parameters()

    def mock_evaluate(params: Dict[str, float]) -> Dict[str, float]:
        """Mock evaluation function for testing."""
        import random

        # Simulate that certain parameter combinations are better
        base_sharpe = 1.0
        base_sharpe += params.get('momentum_weight', 0) * 0.5
        base_sharpe += params.get('arbitrage_weight', 0) * 0.8
        base_sharpe -= params.get('position_size_pct', 0.1) * 2  # Smaller positions = better Sharpe

        # Add noise
        sharpe = base_sharpe + random.gauss(0, 0.2)
        win_rate = 0.5 + params.get('min_confidence', 0.7) * 0.2 + random.gauss(0, 0.05)
        pnl = sharpe * 1000 + random.gauss(0, 200)
        drawdown = max(0.05, 0.3 - params.get('stop_loss_pct', 0.05) * 2 + random.gauss(0, 0.02))

        return {
            'sharpe_ratio': sharpe,
            'win_rate': min(1.0, max(0, win_rate)),
            'total_pnl': pnl,
            'max_drawdown': min(1.0, max(0, drawdown))
        }

    print("\n=== RUNNING OPTIMIZATION ===\n")
    best_params = optimizer.optimize(mock_evaluate, generations=10)

    print("\n=== BEST PARAMETERS ===")
    for name, value in sorted(best_params.items()):
        print(f"  {name}: {value:.4f}")

    if optimizer.best_result:
        print(f"\n  Fitness: {optimizer.best_result.fitness:.2f}")
        print(f"  Sharpe Ratio: {optimizer.best_result.sharpe_ratio:.2f}")
        print(f"  Win Rate: {optimizer.best_result.win_rate:.1%}")
        print(f"  Total P&L: ${optimizer.best_result.total_pnl:,.2f}")
