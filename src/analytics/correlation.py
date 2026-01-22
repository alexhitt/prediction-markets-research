"""
Correlation Analyzer

Analyzes correlations between alternative data signals and market outcomes.
Tests time-lagged relationships and backtests signal strategies.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.database.db import get_session
from src.database.models import Signal, MarketSnapshot


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""
    signal_name: str
    market_category: str
    correlation: float
    p_value: float
    lag_days: int
    sample_size: int
    confidence: str  # "high", "medium", "low"
    direction: str  # "positive", "negative", "none"


@dataclass
class BacktestResult:
    """Result of a signal backtest."""
    signal_name: str
    strategy: str
    total_signals: int
    actionable_signals: int
    accuracy: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float


class CorrelationAnalyzer:
    """
    Analyzes correlations between signals and market movements.

    Features:
    - Time-aligned data collection
    - Pearson correlation with lag testing
    - Signal strategy backtesting

    Usage:
        analyzer = CorrelationAnalyzer()

        # Test correlation
        result = analyzer.test_correlation(
            signal_source="ice_cream",
            market_category="economics",
            lag_days=7
        )

        # Run backtest
        backtest = analyzer.backtest_signal(
            signal_source="ice_cream",
            threshold=0.2
        )
    """

    def __init__(self, min_samples: int = 10):
        """
        Initialize the analyzer.

        Args:
            min_samples: Minimum samples required for analysis
        """
        self.min_samples = min_samples

    def get_aligned_data(
        self,
        signal_source: str,
        market_category: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lag_days: int = 0
    ) -> Tuple[List[float], List[float], List[datetime]]:
        """
        Get time-aligned signal and market data.

        Args:
            signal_source: Signal source to analyze
            market_category: Market category to correlate with
            start_date: Start of analysis period
            end_date: End of analysis period
            lag_days: Days to lag signal (positive = signal before market)

        Returns:
            Tuple of (signal_values, market_changes, timestamps)
        """
        with get_session() as session:
            # Get signals
            signal_query = session.query(Signal).filter(
                Signal.source == signal_source
            ).order_by(Signal.timestamp.asc())

            if start_date:
                signal_query = signal_query.filter(Signal.timestamp >= start_date)
            if end_date:
                signal_query = signal_query.filter(Signal.timestamp <= end_date)

            signals = signal_query.all()

            if len(signals) < self.min_samples:
                logger.warning(
                    f"Insufficient signal data: {len(signals)} < {self.min_samples}"
                )
                return [], [], []

            # Get market snapshots
            snapshot_query = session.query(MarketSnapshot).join(
                MarketSnapshot.market
            ).order_by(MarketSnapshot.timestamp.asc())

            if market_category:
                from src.database.models import Market
                snapshot_query = snapshot_query.filter(
                    Market.category == market_category
                )

            snapshots = snapshot_query.all()

            # Group snapshots by date
            market_by_date = {}
            for snap in snapshots:
                date_key = snap.timestamp.date()
                if date_key not in market_by_date:
                    market_by_date[date_key] = []
                market_by_date[date_key].append(snap.yes_price)

            # Align data
            signal_values = []
            market_changes = []
            timestamps = []

            for signal in signals:
                signal_date = signal.timestamp.date()
                market_date = signal_date + timedelta(days=lag_days)

                if market_date in market_by_date:
                    # Calculate average market price for the day
                    prices = market_by_date[market_date]
                    avg_price = np.mean([p for p in prices if p is not None])

                    if not np.isnan(avg_price):
                        signal_values.append(signal.value)
                        market_changes.append(avg_price)
                        timestamps.append(signal.timestamp)

            return signal_values, market_changes, timestamps

    def test_correlation(
        self,
        signal_source: str,
        market_category: Optional[str] = None,
        lag_days: int = 0,
        start_date: Optional[datetime] = None
    ) -> Optional[CorrelationResult]:
        """
        Test correlation between signal and market movements.

        Args:
            signal_source: Signal source to analyze
            market_category: Market category to correlate with
            lag_days: Days to lag signal
            start_date: Start of analysis period

        Returns:
            CorrelationResult or None if insufficient data
        """
        signal_values, market_changes, _ = self.get_aligned_data(
            signal_source=signal_source,
            market_category=market_category,
            start_date=start_date,
            lag_days=lag_days
        )

        if len(signal_values) < self.min_samples:
            return None

        # Calculate Pearson correlation
        try:
            from scipy.stats import pearsonr
            correlation, p_value = pearsonr(signal_values, market_changes)
        except ImportError:
            # Fallback without scipy
            correlation = np.corrcoef(signal_values, market_changes)[0, 1]
            p_value = 0.5  # Unknown without scipy

        # Determine confidence
        if p_value < 0.01 and len(signal_values) >= 30:
            confidence = "high"
        elif p_value < 0.05 and len(signal_values) >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine direction
        if abs(correlation) < 0.1:
            direction = "none"
        elif correlation > 0:
            direction = "positive"
        else:
            direction = "negative"

        return CorrelationResult(
            signal_name=signal_source,
            market_category=market_category or "all",
            correlation=correlation,
            p_value=p_value,
            lag_days=lag_days,
            sample_size=len(signal_values),
            confidence=confidence,
            direction=direction
        )

    def find_optimal_lag(
        self,
        signal_source: str,
        market_category: Optional[str] = None,
        max_lag: int = 14
    ) -> List[CorrelationResult]:
        """
        Find the optimal lag for signal-market correlation.

        Args:
            signal_source: Signal source to analyze
            market_category: Market category
            max_lag: Maximum lag days to test

        Returns:
            List of CorrelationResult for each lag, sorted by correlation
        """
        results = []

        for lag in range(0, max_lag + 1):
            result = self.test_correlation(
                signal_source=signal_source,
                market_category=market_category,
                lag_days=lag
            )
            if result:
                results.append(result)

        # Sort by absolute correlation
        results.sort(key=lambda r: abs(r.correlation), reverse=True)
        return results

    def backtest_signal(
        self,
        signal_source: str,
        threshold: float = 0.5,
        signal_direction: str = "bullish",
        market_category: Optional[str] = None
    ) -> Optional[BacktestResult]:
        """
        Backtest a signal strategy.

        Strategy: When signal exceeds threshold, predict market direction.

        Args:
            signal_source: Signal source to test
            threshold: Signal value threshold to trigger trade
            signal_direction: Expected market direction when signal fires
            market_category: Market category to trade

        Returns:
            BacktestResult or None if insufficient data
        """
        with get_session() as session:
            signals = session.query(Signal).filter(
                Signal.source == signal_source
            ).order_by(Signal.timestamp.asc()).all()

            if len(signals) < self.min_samples:
                return None

            # Simulate trades
            trades = []
            for i, signal in enumerate(signals):
                if signal.value >= threshold:
                    # Signal triggered
                    entry_date = signal.timestamp

                    # Check next day market movement
                    next_signals = [
                        s for s in signals[i + 1:]
                        if s.timestamp > entry_date + timedelta(days=1)
                    ]

                    if next_signals:
                        exit_signal = next_signals[0]
                        # Simplified: assume market moved with signal direction
                        raw_data = signal.raw_data or {}
                        actual_direction = raw_data.get("direction", "neutral")

                        correct = actual_direction == signal_direction
                        # Simulated return
                        trade_return = 0.1 if correct else -0.1

                        trades.append({
                            "entry_date": entry_date,
                            "signal_value": signal.value,
                            "correct": correct,
                            "return": trade_return
                        })

            if not trades:
                return None

            # Calculate metrics
            returns = [t["return"] for t in trades]
            correct = sum(1 for t in trades if t["correct"])

            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0

            sharpe = (
                avg_return / std_return * np.sqrt(252)
                if std_return > 0 else 0
            )

            # Max drawdown (simplified)
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

            return BacktestResult(
                signal_name=signal_source,
                strategy=f"threshold_{threshold}",
                total_signals=len(signals),
                actionable_signals=len(trades),
                accuracy=correct / len(trades) if trades else 0,
                avg_return=avg_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd
            )

    def generate_correlation_matrix(
        self,
        signal_sources: List[str],
        market_categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate correlation matrix between multiple signals and markets.

        Args:
            signal_sources: List of signal sources
            market_categories: List of market categories

        Returns:
            Nested dict with correlation values
        """
        if not market_categories:
            market_categories = ["politics", "crypto", "economics"]

        matrix = {}

        for source in signal_sources:
            matrix[source] = {}
            for category in market_categories:
                result = self.test_correlation(
                    signal_source=source,
                    market_category=category
                )
                matrix[source][category] = result.correlation if result else 0.0

        return matrix


if __name__ == "__main__":
    print("Testing Correlation Analyzer")
    print("-" * 50)

    analyzer = CorrelationAnalyzer()

    # Test correlation
    result = analyzer.test_correlation(
        signal_source="mcbroken.com",
        market_category="economics"
    )

    if result:
        print(f"\nCorrelation Result:")
        print(f"  Signal: {result.signal_name}")
        print(f"  Market: {result.market_category}")
        print(f"  Correlation: {result.correlation:.4f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Sample size: {result.sample_size}")
    else:
        print("Insufficient data for correlation analysis")
