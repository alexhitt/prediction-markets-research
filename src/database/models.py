"""
Database models for Prediction Markets AI System.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Market(Base):
    """
    Unified market representation across platforms.
    """
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Platform identification
    platform = Column(String(50), nullable=False)  # "polymarket", "kalshi"
    platform_id = Column(String(255), nullable=False)  # Platform's market ID

    # Market details
    question = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)

    # Status
    status = Column(String(50), default="open")  # open, closed, resolved
    resolution = Column(String(50), nullable=True)  # yes, no, null

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Metadata
    tags = Column(JSON, nullable=True)
    extra_data = Column(JSON, nullable=True)

    # Relationships
    snapshots = relationship("MarketSnapshot", back_populates="market")

    __table_args__ = (
        UniqueConstraint("platform", "platform_id", name="uq_platform_market"),
        Index("ix_market_category", "category"),
        Index("ix_market_status", "status"),
    )


class MarketSnapshot(Base):
    """
    Point-in-time snapshot of market prices and volume.
    """
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Prices (0-1 scale)
    yes_price = Column(Float, nullable=True)
    no_price = Column(Float, nullable=True)

    # Best bid/ask
    yes_bid = Column(Float, nullable=True)
    yes_ask = Column(Float, nullable=True)
    no_bid = Column(Float, nullable=True)
    no_ask = Column(Float, nullable=True)

    # Spread
    spread = Column(Float, nullable=True)

    # Volume and liquidity
    volume_24h = Column(Float, nullable=True)
    total_volume = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    open_interest = Column(Float, nullable=True)

    # Relationships
    market = relationship("Market", back_populates="snapshots")

    __table_args__ = (
        Index("ix_snapshot_market_time", "market_id", "timestamp"),
        Index("ix_snapshot_timestamp", "timestamp"),
    )


class ArbitrageOpportunity(Base):
    """
    Detected arbitrage opportunities between platforms.
    """
    __tablename__ = "arbitrage_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timestamp
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expired_at = Column(DateTime, nullable=True)

    # Markets involved
    market_1_platform = Column(String(50), nullable=False)
    market_1_id = Column(String(255), nullable=False)
    market_1_question = Column(Text, nullable=False)
    market_1_price = Column(Float, nullable=False)

    market_2_platform = Column(String(50), nullable=False)
    market_2_id = Column(String(255), nullable=False)
    market_2_question = Column(Text, nullable=False)
    market_2_price = Column(Float, nullable=False)

    # Opportunity details
    spread = Column(Float, nullable=False)  # Price difference
    profit_potential = Column(Float, nullable=False)  # Expected profit %
    confidence = Column(Float, nullable=True)  # How confident markets are same event

    # Status
    status = Column(String(50), default="active")  # active, expired, executed

    # Action taken
    action_taken = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_arb_detected", "detected_at"),
        Index("ix_arb_status", "status"),
    )


class Signal(Base):
    """
    Alternative data signals and their predictions.
    """
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Signal identification
    name = Column(String(255), nullable=False)
    source = Column(String(255), nullable=False)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Signal value
    value = Column(Float, nullable=False)
    raw_data = Column(JSON, nullable=True)

    # Associated prediction
    prediction_market = Column(String(255), nullable=True)
    prediction_direction = Column(String(50), nullable=True)  # bullish, bearish
    prediction_confidence = Column(Float, nullable=True)

    # Outcome tracking
    outcome = Column(String(50), nullable=True)  # correct, incorrect, pending
    outcome_notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_signal_name_time", "name", "timestamp"),
        Index("ix_signal_source", "source"),
    )


class WalletActivity(Base):
    """
    Track whale wallet activity on Polymarket.
    """
    __tablename__ = "wallet_activity"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Wallet
    wallet_address = Column(String(255), nullable=False)
    wallet_label = Column(String(255), nullable=True)  # Known whale names

    # Activity
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    action = Column(String(50), nullable=False)  # buy, sell

    # Market
    market_id = Column(String(255), nullable=False)
    market_question = Column(Text, nullable=True)

    # Trade details
    side = Column(String(10), nullable=False)  # yes, no
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_wallet_address", "wallet_address"),
        Index("ix_wallet_time", "timestamp"),
    )


class PredictionLog(Base):
    """
    Log of system predictions for backtesting.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Market
    platform = Column(String(50), nullable=False)
    market_id = Column(String(255), nullable=False)
    market_question = Column(Text, nullable=False)

    # Prediction
    predicted_outcome = Column(String(50), nullable=False)  # yes, no
    confidence = Column(Float, nullable=False)
    market_price_at_prediction = Column(Float, nullable=False)

    # Reasoning
    strategy = Column(String(100), nullable=True)
    signals_used = Column(JSON, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Outcome
    actual_outcome = Column(String(50), nullable=True)
    profit_loss = Column(Float, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_prediction_market", "platform", "market_id"),
        Index("ix_prediction_strategy", "strategy"),
    )


class Hypothesis(Base):
    """
    A trading hypothesis to track and validate.

    Tracks causal theories about alternative data signals
    and their predictive power for market outcomes.
    """
    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Hypothesis details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    data_source = Column(String(255), nullable=False)  # e.g., "ice_cream", "google_trends"
    causal_theory = Column(Text, nullable=True)  # Why this signal predicts the market

    # Status
    status = Column(String(50), default="active")  # active, validated, invalidated, retired
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Correlation statistics (updated as predictions resolve)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    average_brier_score = Column(Float, nullable=True)
    average_edge = Column(Float, nullable=True)  # Edge vs market

    # Relationships
    predictions = relationship("HypothesisPrediction", back_populates="hypothesis")

    __table_args__ = (
        Index("ix_hypothesis_status", "status"),
        Index("ix_hypothesis_source", "data_source"),
    )


class HypothesisPrediction(Base):
    """
    A prediction made based on a hypothesis.

    Links a hypothesis to a specific market prediction
    and tracks the outcome for Brier score calculation.
    """
    __tablename__ = "hypothesis_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=False)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)

    # Market details
    platform = Column(String(50), nullable=False)
    market_id = Column(String(255), nullable=False)
    market_question = Column(Text, nullable=False)

    # Prediction
    predicted_probability = Column(Float, nullable=False)  # Our prediction (0-1)
    market_probability = Column(Float, nullable=False)  # Market price at prediction time
    prediction_direction = Column(String(10), nullable=False)  # yes, no

    # Signal data at prediction time
    signal_value = Column(Float, nullable=True)
    signal_data = Column(JSON, nullable=True)

    # Outcome
    actual_outcome = Column(Float, nullable=True)  # 1.0 for yes, 0.0 for no
    our_brier_score = Column(Float, nullable=True)  # (predicted - actual)^2
    market_brier_score = Column(Float, nullable=True)  # (market_price - actual)^2
    edge = Column(Float, nullable=True)  # market_brier - our_brier (positive = we beat market)

    # Relationships
    hypothesis = relationship("Hypothesis", back_populates="predictions")

    __table_args__ = (
        Index("ix_hyp_pred_hypothesis", "hypothesis_id"),
        Index("ix_hyp_pred_market", "platform", "market_id"),
        Index("ix_hyp_pred_resolved", "resolved_at"),
    )


class PaperPortfolio(Base):
    """
    Paper trading portfolio for simulated trading.
    """
    __tablename__ = "paper_portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Portfolio details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Capital
    initial_capital = Column(Float, nullable=False, default=10000.0)
    current_capital = Column(Float, nullable=False, default=10000.0)

    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    peak_capital = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)

    # Settings
    max_position_size = Column(Float, default=0.1)  # 10% of capital
    use_kelly = Column(Boolean, default=True)
    kelly_fraction = Column(Float, default=0.25)  # Quarter-Kelly

    # Relationships
    trades = relationship("PaperTrade", back_populates="portfolio")

    __table_args__ = (
        Index("ix_portfolio_name", "name"),
    )


class PaperTrade(Base):
    """
    Individual paper trade record.
    """
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("paper_portfolios.id"), nullable=False)

    # Timing
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)

    # Market details
    platform = Column(String(50), nullable=False)
    market_id = Column(String(255), nullable=False)
    market_question = Column(Text, nullable=False)

    # Trade details
    side = Column(String(10), nullable=False)  # yes, no
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    size = Column(Float, nullable=False)  # Number of contracts
    cost_basis = Column(Float, nullable=False)  # entry_price * size

    # Strategy / Reasoning
    strategy = Column(String(100), nullable=True)
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=True)
    notes = Column(Text, nullable=True)

    # Status
    status = Column(String(50), default="open")  # open, closed, expired

    # P&L
    realized_pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)

    # Relationships
    portfolio = relationship("PaperPortfolio", back_populates="trades")

    __table_args__ = (
        Index("ix_paper_trade_portfolio", "portfolio_id"),
        Index("ix_paper_trade_status", "status"),
        Index("ix_paper_trade_market", "platform", "market_id"),
    )
