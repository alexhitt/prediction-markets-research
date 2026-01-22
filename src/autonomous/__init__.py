"""
Autonomous Trading Agent

Fully autonomous system that:
- Runs continuously without user intervention
- Collects signals and executes paper trades
- Tests new strategies in simulation
- Tracks performance on daily leaderboard
- Automatically adjusts behavior to maximize profit
"""

from src.autonomous.agent import AutonomousAgent
from src.autonomous.leaderboard import Leaderboard
from src.autonomous.optimizer import AutoOptimizer
from src.autonomous.simulator import ContinuousSimulator
from src.autonomous.arena import StrategyArena

__all__ = [
    "AutonomousAgent",
    "Leaderboard",
    "AutoOptimizer",
    "ContinuousSimulator",
    "StrategyArena",
]
