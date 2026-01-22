"""
Paper Trading Engine

Simulates trading on prediction markets without real money.
Tracks P&L, implements position sizing via Kelly Criterion.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.database.db import get_session
from src.database.models import PaperPortfolio, PaperTrade


@dataclass
class PortfolioStats:
    """Portfolio performance statistics."""
    total_capital: float
    available_capital: float
    positions_value: float
    total_pnl: float
    pnl_percent: float
    total_trades: int
    winning_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    open_positions: int


@dataclass
class TradeResult:
    """Result of executing a trade."""
    trade_id: int
    status: str
    cost_basis: float
    position_size: float
    error: Optional[str] = None


class PaperTrader:
    """
    Paper trading engine for simulating prediction market trades.

    Features:
    - Position sizing via Kelly Criterion
    - P&L tracking with performance metrics
    - Support for multiple portfolios
    - Risk management (max position size)

    Usage:
        trader = PaperTrader(portfolio_id=1)

        # Execute a trade
        result = trader.execute_trade(
            platform="polymarket",
            market_id="123",
            market_question="Will X happen?",
            side="yes",
            price=0.45,
            edge=0.10,
            strategy="ice_cream_signal"
        )

        # Close trade
        pnl = trader.close_trade(trade_id=result.trade_id, exit_price=0.65)

        # Get portfolio stats
        stats = trader.get_portfolio_stats()
    """

    def __init__(
        self,
        portfolio_id: Optional[int] = None,
        portfolio_name: str = "Default Portfolio"
    ):
        """
        Initialize paper trader.

        Args:
            portfolio_id: Existing portfolio ID (optional)
            portfolio_name: Name for new portfolio if creating
        """
        self.portfolio_id = portfolio_id or self._get_or_create_portfolio(portfolio_name)

    def _get_or_create_portfolio(self, name: str) -> int:
        """Get existing portfolio or create new one."""
        with get_session() as session:
            portfolio = session.query(PaperPortfolio).filter_by(name=name).first()

            if portfolio:
                return portfolio.id

            # Create new portfolio
            portfolio = PaperPortfolio(
                name=name,
                initial_capital=10000.0,
                current_capital=10000.0,
                peak_capital=10000.0
            )
            session.add(portfolio)
            session.flush()
            portfolio_id = portfolio.id

        logger.info(f"Created paper portfolio: {name} (ID: {portfolio_id})")
        return portfolio_id

    def execute_trade(
        self,
        platform: str,
        market_id: str,
        market_question: str,
        side: str,
        price: float,
        edge: Optional[float] = None,
        confidence: Optional[float] = None,
        strategy: Optional[str] = None,
        hypothesis_id: Optional[int] = None,
        notes: Optional[str] = None,
        size_override: Optional[float] = None
    ) -> TradeResult:
        """
        Execute a paper trade.

        Args:
            platform: Market platform (polymarket, kalshi)
            market_id: Platform's market ID
            market_question: Market question text
            side: "yes" or "no"
            price: Entry price (0-1)
            edge: Estimated edge (optional, for Kelly sizing)
            confidence: Confidence in the trade (optional)
            strategy: Strategy name
            hypothesis_id: Associated hypothesis ID
            notes: Trade notes
            size_override: Override calculated position size

        Returns:
            TradeResult with trade details
        """
        with get_session() as session:
            portfolio = session.query(PaperPortfolio).filter_by(
                id=self.portfolio_id
            ).first()

            if not portfolio:
                return TradeResult(
                    trade_id=0,
                    status="error",
                    cost_basis=0,
                    position_size=0,
                    error="Portfolio not found"
                )

            # Calculate position size
            if size_override:
                size = size_override
            else:
                size = self._calculate_position_size(
                    portfolio=portfolio,
                    price=price,
                    edge=edge,
                    confidence=confidence
                )

            cost_basis = size * price

            # Check available capital
            if cost_basis > portfolio.current_capital:
                return TradeResult(
                    trade_id=0,
                    status="rejected",
                    cost_basis=cost_basis,
                    position_size=size,
                    error="Insufficient capital"
                )

            # Create trade
            trade = PaperTrade(
                portfolio_id=self.portfolio_id,
                platform=platform,
                market_id=market_id,
                market_question=market_question,
                side=side,
                entry_price=price,
                size=size,
                cost_basis=cost_basis,
                strategy=strategy,
                hypothesis_id=hypothesis_id,
                notes=notes,
                status="open"
            )
            session.add(trade)

            # Deduct capital
            portfolio.current_capital -= cost_basis
            portfolio.total_trades = (portfolio.total_trades or 0) + 1

            session.flush()
            trade_id = trade.id

        logger.info(
            f"Paper trade executed: {side.upper()} {size:.2f} contracts "
            f"@ {price:.1%} (cost: ${cost_basis:.2f})"
        )

        return TradeResult(
            trade_id=trade_id,
            status="executed",
            cost_basis=cost_basis,
            position_size=size
        )

    def _calculate_position_size(
        self,
        portfolio: PaperPortfolio,
        price: float,
        edge: Optional[float],
        confidence: Optional[float]
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Kelly formula: f* = (bp - q) / b
        Where:
        - f* = fraction of capital to bet
        - b = odds received (1/price - 1 for yes side)
        - p = probability of winning
        - q = probability of losing (1 - p)

        We use fractional Kelly (quarter-Kelly by default) for safety.

        Args:
            portfolio: Portfolio to size for
            price: Entry price
            edge: Estimated edge over market
            confidence: Confidence in estimate

        Returns:
            Number of contracts to buy
        """
        available = portfolio.current_capital
        max_position_pct = portfolio.max_position_size or 0.1
        kelly_fraction = portfolio.kelly_fraction or 0.25

        # Calculate max position value
        max_position_value = available * max_position_pct

        if not edge or edge <= 0:
            # No edge info - use minimum size
            position_value = available * 0.02  # 2% of capital
        elif portfolio.use_kelly:
            # Kelly sizing
            # Assuming we have edge e, our estimated win prob = market_price + edge
            estimated_win_prob = min(0.95, max(0.05, price + edge))
            odds = 1 / price - 1  # Odds for yes side

            # Kelly fraction
            kelly_f = (odds * estimated_win_prob - (1 - estimated_win_prob)) / odds

            if kelly_f <= 0:
                # Negative Kelly = don't bet
                position_value = 0
            else:
                # Apply fractional Kelly
                position_value = available * kelly_f * kelly_fraction

                # Apply confidence adjustment
                if confidence:
                    position_value *= confidence
        else:
            # Fixed fraction
            position_value = available * 0.05  # 5% of capital

        # Cap at max position
        position_value = min(position_value, max_position_value)

        # Convert to number of contracts
        if price > 0:
            size = position_value / price
        else:
            size = 0

        return max(0, size)

    def close_trade(
        self,
        trade_id: int,
        exit_price: Optional[float] = None,
        market_outcome: Optional[str] = None
    ) -> Dict:
        """
        Close a paper trade.

        Args:
            trade_id: ID of trade to close
            exit_price: Exit price (for selling before resolution)
            market_outcome: Market resolution ("yes" or "no")

        Returns:
            Dict with P&L details
        """
        with get_session() as session:
            trade = session.query(PaperTrade).filter_by(id=trade_id).first()

            if not trade:
                return {"error": "Trade not found"}

            if trade.status != "open":
                return {"error": "Trade already closed"}

            portfolio = session.query(PaperPortfolio).filter_by(
                id=trade.portfolio_id
            ).first()

            # Calculate P&L
            if market_outcome:
                # Market resolved
                won = (
                    (trade.side == "yes" and market_outcome == "yes") or
                    (trade.side == "no" and market_outcome == "no")
                )
                if won:
                    exit_value = trade.size * 1.0  # Win = $1 per contract
                else:
                    exit_value = 0.0
                trade.exit_price = 1.0 if won else 0.0
            elif exit_price is not None:
                # Early exit
                exit_value = trade.size * exit_price
                trade.exit_price = exit_price
            else:
                return {"error": "Must provide exit_price or market_outcome"}

            pnl = exit_value - trade.cost_basis
            pnl_percent = pnl / trade.cost_basis if trade.cost_basis > 0 else 0

            # Update trade
            trade.realized_pnl = pnl
            trade.pnl_percent = pnl_percent
            trade.status = "closed"
            trade.closed_at = datetime.utcnow()

            # Update portfolio
            portfolio.current_capital += exit_value
            portfolio.total_pnl = (portfolio.total_pnl or 0) + pnl

            if pnl > 0:
                portfolio.winning_trades = (portfolio.winning_trades or 0) + 1

            # Update peak/drawdown
            if portfolio.current_capital > (portfolio.peak_capital or 0):
                portfolio.peak_capital = portfolio.current_capital

            current_drawdown = (
                (portfolio.peak_capital - portfolio.current_capital) /
                portfolio.peak_capital if portfolio.peak_capital else 0
            )
            if current_drawdown > (portfolio.max_drawdown or 0):
                portfolio.max_drawdown = current_drawdown

        logger.info(
            f"Paper trade closed: P&L ${pnl:+.2f} ({pnl_percent:+.1%})"
        )

        return {
            "trade_id": trade_id,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "exit_value": exit_value
        }

    def get_portfolio_stats(self) -> PortfolioStats:
        """Get current portfolio statistics."""
        with get_session() as session:
            portfolio = session.query(PaperPortfolio).filter_by(
                id=self.portfolio_id
            ).first()

            if not portfolio:
                return PortfolioStats(
                    total_capital=0,
                    available_capital=0,
                    positions_value=0,
                    total_pnl=0,
                    pnl_percent=0,
                    total_trades=0,
                    winning_trades=0,
                    win_rate=0,
                    sharpe_ratio=0,
                    max_drawdown=0,
                    open_positions=0
                )

            # Calculate open positions value
            open_trades = session.query(PaperTrade).filter_by(
                portfolio_id=self.portfolio_id,
                status="open"
            ).all()

            positions_value = sum(t.cost_basis for t in open_trades)

            # Calculate metrics
            total_trades = portfolio.total_trades or 0
            winning_trades = portfolio.winning_trades or 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            total_capital = portfolio.current_capital + positions_value
            initial = portfolio.initial_capital or 10000
            total_pnl = total_capital - initial
            pnl_percent = total_pnl / initial if initial > 0 else 0

            # Estimate Sharpe (simplified)
            sharpe = self._estimate_sharpe(session)

            return PortfolioStats(
                total_capital=total_capital,
                available_capital=portfolio.current_capital,
                positions_value=positions_value,
                total_pnl=total_pnl,
                pnl_percent=pnl_percent,
                total_trades=total_trades,
                winning_trades=winning_trades,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                max_drawdown=portfolio.max_drawdown or 0,
                open_positions=len(open_trades)
            )

    def _estimate_sharpe(self, session) -> float:
        """Estimate Sharpe ratio from closed trades."""
        closed_trades = session.query(PaperTrade).filter_by(
            portfolio_id=self.portfolio_id,
            status="closed"
        ).all()

        if len(closed_trades) < 2:
            return 0.0

        returns = [t.pnl_percent or 0 for t in closed_trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualized (assuming ~250 trading days)
        sharpe = (mean_return * np.sqrt(250)) / std_return
        return float(sharpe)

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        with get_session() as session:
            trades = session.query(PaperTrade).filter_by(
                portfolio_id=self.portfolio_id,
                status="open"
            ).order_by(PaperTrade.opened_at.desc()).all()

            return [
                {
                    "id": t.id,
                    "platform": t.platform,
                    "market_id": t.market_id,
                    "market_question": t.market_question,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "size": t.size,
                    "cost_basis": t.cost_basis,
                    "strategy": t.strategy,
                    "opened_at": t.opened_at.isoformat()
                }
                for t in trades
            ]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get closed trade history."""
        with get_session() as session:
            trades = session.query(PaperTrade).filter_by(
                portfolio_id=self.portfolio_id,
                status="closed"
            ).order_by(PaperTrade.closed_at.desc()).limit(limit).all()

            return [
                {
                    "id": t.id,
                    "platform": t.platform,
                    "market_question": t.market_question[:50],
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "pnl": t.realized_pnl,
                    "pnl_percent": t.pnl_percent,
                    "strategy": t.strategy,
                    "opened_at": t.opened_at.isoformat(),
                    "closed_at": t.closed_at.isoformat() if t.closed_at else None
                }
                for t in trades
            ]

    def get_pnl_curve(self) -> List[Dict]:
        """Get P&L curve over time."""
        with get_session() as session:
            portfolio = session.query(PaperPortfolio).filter_by(
                id=self.portfolio_id
            ).first()

            if not portfolio:
                return []

            trades = session.query(PaperTrade).filter_by(
                portfolio_id=self.portfolio_id,
                status="closed"
            ).order_by(PaperTrade.closed_at.asc()).all()

            curve = []
            cumulative_pnl = 0
            initial = portfolio.initial_capital or 10000

            for trade in trades:
                cumulative_pnl += trade.realized_pnl or 0
                curve.append({
                    "date": trade.closed_at.isoformat() if trade.closed_at else None,
                    "cumulative_pnl": cumulative_pnl,
                    "portfolio_value": initial + cumulative_pnl
                })

            return curve


def create_portfolio(
    name: str,
    initial_capital: float = 10000.0,
    max_position_size: float = 0.1,
    use_kelly: bool = True,
    kelly_fraction: float = 0.25
) -> int:
    """
    Create a new paper trading portfolio.

    Args:
        name: Portfolio name
        initial_capital: Starting capital
        max_position_size: Max position as fraction of capital
        use_kelly: Whether to use Kelly sizing
        kelly_fraction: Fraction of Kelly to use

    Returns:
        Portfolio ID
    """
    with get_session() as session:
        portfolio = PaperPortfolio(
            name=name,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            peak_capital=initial_capital,
            max_position_size=max_position_size,
            use_kelly=use_kelly,
            kelly_fraction=kelly_fraction
        )
        session.add(portfolio)
        session.flush()
        portfolio_id = portfolio.id

    logger.info(f"Created portfolio: {name} with ${initial_capital:.2f}")
    return portfolio_id


if __name__ == "__main__":
    print("Testing Paper Trading Engine")
    print("-" * 50)

    trader = PaperTrader(portfolio_name="Test Portfolio")

    # Get stats
    stats = trader.get_portfolio_stats()
    print(f"\nPortfolio Stats:")
    print(f"  Total Capital: ${stats.total_capital:,.2f}")
    print(f"  Available: ${stats.available_capital:,.2f}")
    print(f"  Total P&L: ${stats.total_pnl:+,.2f} ({stats.pnl_percent:+.1%})")
    print(f"  Win Rate: {stats.win_rate:.1%}")
    print(f"  Open Positions: {stats.open_positions}")
