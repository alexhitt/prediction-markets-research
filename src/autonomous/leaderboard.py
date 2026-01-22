"""
Daily Leaderboard System

Tracks and ranks strategy performance over time.
Provides insights into what's working and what needs improvement.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class DailyPerformance:
    """Performance record for a single day."""
    date: date
    strategy: str
    trades: int
    winning_trades: int
    pnl: float
    signals_collected: int
    opportunities_found: int
    sharpe_ratio: float
    brier_score: float
    roi: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'strategy': self.strategy,
            'trades': self.trades,
            'winning_trades': self.winning_trades,
            'pnl': self.pnl,
            'signals_collected': self.signals_collected,
            'opportunities_found': self.opportunities_found,
            'sharpe_ratio': self.sharpe_ratio,
            'brier_score': self.brier_score,
            'roi': self.roi,
            'win_rate': self.winning_trades / max(1, self.trades)
        }


@dataclass
class StrategyRanking:
    """Ranking entry for a strategy."""
    rank: int
    strategy: str
    total_pnl: float
    total_trades: int
    win_rate: float
    avg_daily_roi: float
    sharpe_ratio: float
    days_active: int
    trend: str  # "up", "down", "stable"
    score: float  # Composite ranking score


class Leaderboard:
    """
    Tracks strategy performance and maintains rankings.

    Features:
    - Daily performance recording
    - Strategy rankings by multiple metrics
    - Trend analysis
    - Best/worst day tracking
    - Strategy comparison

    Usage:
        leaderboard = Leaderboard()
        leaderboard.record_daily_performance(...)
        rankings = leaderboard.get_rankings()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize leaderboard."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "leaderboard"
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._performances: List[DailyPerformance] = []
        self._load_history()

    def record_daily_performance(
        self,
        date: date,
        strategy: str,
        trades: int,
        pnl: float,
        signals: int,
        opportunities: int,
        winning_trades: int = 0,
        sharpe_ratio: float = 0.0,
        brier_score: float = 0.25,
        roi: float = 0.0
    ):
        """
        Record daily performance for a strategy.

        Args:
            date: Performance date
            strategy: Strategy name
            trades: Number of trades
            pnl: Profit/loss
            signals: Signals collected
            opportunities: Opportunities found
            winning_trades: Winning trade count
            sharpe_ratio: Daily Sharpe
            brier_score: Brier score
            roi: Return on investment
        """
        # Auto-calculate winning trades if not provided
        if winning_trades == 0 and trades > 0 and pnl > 0:
            winning_trades = int(trades * 0.55)  # Estimate

        performance = DailyPerformance(
            date=date,
            strategy=strategy,
            trades=trades,
            winning_trades=winning_trades,
            pnl=pnl,
            signals_collected=signals,
            opportunities_found=opportunities,
            sharpe_ratio=sharpe_ratio,
            brier_score=brier_score,
            roi=roi
        )

        # Remove existing entry for same date/strategy
        self._performances = [
            p for p in self._performances
            if not (p.date == date and p.strategy == strategy)
        ]

        self._performances.append(performance)
        self._save_history()

        logger.info(f"Recorded performance for {strategy} on {date}: P&L ${pnl:+.2f}")

    def get_rankings(self, days: int = 30) -> List[StrategyRanking]:
        """
        Get strategy rankings.

        Args:
            days: Number of days to consider

        Returns:
            List of StrategyRanking sorted by score
        """
        cutoff = date.today() - timedelta(days=days)
        recent = [p for p in self._performances if p.date >= cutoff]

        # Group by strategy
        by_strategy: Dict[str, List[DailyPerformance]] = {}
        for p in recent:
            if p.strategy not in by_strategy:
                by_strategy[p.strategy] = []
            by_strategy[p.strategy].append(p)

        rankings = []
        for strategy, performances in by_strategy.items():
            total_pnl = sum(p.pnl for p in performances)
            total_trades = sum(p.trades for p in performances)
            total_wins = sum(p.winning_trades for p in performances)
            win_rate = total_wins / max(1, total_trades)

            # Calculate average daily ROI
            roi_values = [p.roi for p in performances if p.roi != 0]
            avg_daily_roi = sum(roi_values) / max(1, len(roi_values))

            # Calculate Sharpe from daily returns
            pnl_values = [p.pnl for p in performances]
            if len(pnl_values) > 1:
                import statistics
                mean_pnl = statistics.mean(pnl_values)
                std_pnl = statistics.stdev(pnl_values) or 1
                sharpe = (mean_pnl / std_pnl) * (252 ** 0.5)  # Annualized
            else:
                sharpe = 0.0

            # Determine trend (last 7 days vs previous 7)
            trend = self._calculate_trend(performances)

            # Composite score (weighted combination)
            score = (
                total_pnl * 0.3 +
                win_rate * 100 * 0.2 +
                sharpe * 10 * 0.25 +
                avg_daily_roi * 1000 * 0.15 +
                len(performances) * 0.1  # Consistency bonus
            )

            rankings.append(StrategyRanking(
                rank=0,  # Will be set after sorting
                strategy=strategy,
                total_pnl=total_pnl,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_daily_roi=avg_daily_roi,
                sharpe_ratio=sharpe,
                days_active=len(performances),
                trend=trend,
                score=score
            ))

        # Sort by score and assign ranks
        rankings.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1

        return rankings

    def get_daily_history(self, days: int = 30, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get daily performance history.

        Args:
            days: Number of days
            strategy: Filter by strategy (optional)

        Returns:
            List of daily performance dicts
        """
        cutoff = date.today() - timedelta(days=days)
        recent = [p for p in self._performances if p.date >= cutoff]

        if strategy:
            recent = [p for p in recent if p.strategy == strategy]

        recent.sort(key=lambda p: p.date, reverse=True)
        return [p.to_dict() for p in recent]

    def get_best_day(self, strategy: Optional[str] = None) -> Optional[DailyPerformance]:
        """Get the best performing day."""
        performances = self._performances
        if strategy:
            performances = [p for p in performances if p.strategy == strategy]

        if not performances:
            return None

        return max(performances, key=lambda p: p.pnl)

    def get_worst_day(self, strategy: Optional[str] = None) -> Optional[DailyPerformance]:
        """Get the worst performing day."""
        performances = self._performances
        if strategy:
            performances = [p for p in performances if p.strategy == strategy]

        if not performances:
            return None

        return min(performances, key=lambda p: p.pnl)

    def get_streak(self, strategy: str) -> Dict[str, int]:
        """
        Get current winning/losing streak.

        Returns:
            Dict with 'winning_streak' and 'losing_streak'
        """
        performances = [p for p in self._performances if p.strategy == strategy]
        performances.sort(key=lambda p: p.date, reverse=True)

        winning_streak = 0
        losing_streak = 0

        # Count current streak
        if performances:
            streak_type = "winning" if performances[0].pnl > 0 else "losing"
            for p in performances:
                if streak_type == "winning" and p.pnl > 0:
                    winning_streak += 1
                elif streak_type == "losing" and p.pnl < 0:
                    losing_streak += 1
                else:
                    break

        return {
            'winning_streak': winning_streak,
            'losing_streak': losing_streak
        }

    def compare_strategies(self, strategy_a: str, strategy_b: str, days: int = 30) -> Dict[str, Any]:
        """
        Compare two strategies head-to-head.

        Returns:
            Comparison metrics
        """
        cutoff = date.today() - timedelta(days=days)

        perf_a = [p for p in self._performances if p.strategy == strategy_a and p.date >= cutoff]
        perf_b = [p for p in self._performances if p.strategy == strategy_b and p.date >= cutoff]

        def calc_metrics(perfs):
            if not perfs:
                return {'pnl': 0, 'trades': 0, 'win_rate': 0, 'avg_pnl': 0}
            total_pnl = sum(p.pnl for p in perfs)
            total_trades = sum(p.trades for p in perfs)
            total_wins = sum(p.winning_trades for p in perfs)
            return {
                'pnl': total_pnl,
                'trades': total_trades,
                'win_rate': total_wins / max(1, total_trades),
                'avg_pnl': total_pnl / len(perfs)
            }

        metrics_a = calc_metrics(perf_a)
        metrics_b = calc_metrics(perf_b)

        # Determine winner in each category
        pnl_winner = strategy_a if metrics_a['pnl'] > metrics_b['pnl'] else strategy_b
        trades_winner = strategy_a if metrics_a['trades'] > metrics_b['trades'] else strategy_b
        win_rate_winner = strategy_a if metrics_a['win_rate'] > metrics_b['win_rate'] else strategy_b

        return {
            'period_days': days,
            strategy_a: metrics_a,
            strategy_b: metrics_b,
            'winners': {
                'pnl': pnl_winner,
                'trades': trades_winner,
                'win_rate': win_rate_winner
            },
            'overall_winner': pnl_winner  # P&L is primary metric
        }

    def get_insights(self) -> List[str]:
        """
        Generate insights from performance data.

        Returns:
            List of insight strings
        """
        insights = []

        rankings = self.get_rankings(days=30)
        if not rankings:
            return ["No performance data available yet."]

        # Top performer
        top = rankings[0]
        insights.append(f"Top strategy: {top.strategy} with ${top.total_pnl:+,.2f} P&L ({top.win_rate:.1%} win rate)")

        # Trend analysis
        improving = [r for r in rankings if r.trend == "up"]
        declining = [r for r in rankings if r.trend == "down"]

        if improving:
            insights.append(f"Improving strategies: {', '.join(r.strategy for r in improving)}")
        if declining:
            insights.append(f"Declining strategies: {', '.join(r.strategy for r in declining)}")

        # Best/worst day
        best = self.get_best_day()
        worst = self.get_worst_day()

        if best:
            insights.append(f"Best day: {best.date} with ${best.pnl:+,.2f} ({best.strategy})")
        if worst:
            insights.append(f"Worst day: {worst.date} with ${worst.pnl:+,.2f} ({worst.strategy})")

        # Consistency
        most_consistent = max(rankings, key=lambda r: r.days_active) if rankings else None
        if most_consistent:
            insights.append(f"Most consistent: {most_consistent.strategy} ({most_consistent.days_active} days active)")

        return insights

    def _calculate_trend(self, performances: List[DailyPerformance]) -> str:
        """Calculate trend direction."""
        if len(performances) < 7:
            return "stable"

        performances.sort(key=lambda p: p.date)

        # Compare last 7 days vs previous 7
        recent = performances[-7:]
        previous = performances[-14:-7] if len(performances) >= 14 else performances[:-7]

        recent_pnl = sum(p.pnl for p in recent)
        previous_pnl = sum(p.pnl for p in previous) if previous else 0

        if recent_pnl > previous_pnl * 1.1:
            return "up"
        elif recent_pnl < previous_pnl * 0.9:
            return "down"
        return "stable"

    def _save_history(self):
        """Save performance history to disk."""
        try:
            data = [p.to_dict() for p in self._performances]
            path = self.data_dir / "history.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving leaderboard: {e}")

    def _load_history(self):
        """Load performance history from disk."""
        try:
            path = self.data_dir / "history.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                self._performances = []
                for item in data:
                    self._performances.append(DailyPerformance(
                        date=date.fromisoformat(item['date']),
                        strategy=item['strategy'],
                        trades=item['trades'],
                        winning_trades=item['winning_trades'],
                        pnl=item['pnl'],
                        signals_collected=item['signals_collected'],
                        opportunities_found=item['opportunities_found'],
                        sharpe_ratio=item.get('sharpe_ratio', 0),
                        brier_score=item.get('brier_score', 0.25),
                        roi=item.get('roi', 0)
                    ))

                logger.info(f"Loaded {len(self._performances)} performance records")

        except Exception as e:
            logger.warning(f"Could not load leaderboard history: {e}")


def get_leaderboard() -> Leaderboard:
    """Get or create leaderboard instance."""
    return Leaderboard()


if __name__ == "__main__":
    # Demo the leaderboard
    leaderboard = Leaderboard()

    # Add some sample data
    from datetime import date, timedelta

    strategies = ["ensemble", "momentum", "arbitrage"]
    for i in range(30):
        d = date.today() - timedelta(days=i)
        for strategy in strategies:
            import random
            pnl = random.gauss(50 if strategy == "ensemble" else 20, 100)
            trades = random.randint(5, 20)

            leaderboard.record_daily_performance(
                date=d,
                strategy=strategy,
                trades=trades,
                pnl=pnl,
                signals=random.randint(10, 50),
                opportunities=random.randint(1, 10)
            )

    # Show rankings
    print("\n=== STRATEGY LEADERBOARD ===\n")
    rankings = leaderboard.get_rankings()

    for r in rankings:
        trend_icon = "📈" if r.trend == "up" else "📉" if r.trend == "down" else "➡️"
        print(f"#{r.rank} {r.strategy} {trend_icon}")
        print(f"   P&L: ${r.total_pnl:+,.2f} | Win Rate: {r.win_rate:.1%} | Sharpe: {r.sharpe_ratio:.2f}")
        print()

    # Show insights
    print("=== INSIGHTS ===")
    for insight in leaderboard.get_insights():
        print(f"• {insight}")
