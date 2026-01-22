"""
Database connection and session management.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from config.settings import settings, DATA_DIR
from src.database.models import Base


# Create engine
engine = create_engine(
    settings.database_url,
    echo=False,  # Set to True for SQL logging
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    # Ensure data directory exists for SQLite
    if "sqlite" in settings.database_url:
        DATA_DIR.mkdir(exist_ok=True)

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """Get database session (for dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class DatabaseManager:
    """
    High-level database operations.
    """

    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def get_session(self) -> Session:
        """Get a new session."""
        return SessionLocal()

    def add_market(self, session: Session, **kwargs):
        """Add or update a market."""
        from src.database.models import Market

        # Check if market exists
        existing = session.query(Market).filter_by(
            platform=kwargs["platform"],
            platform_id=kwargs["platform_id"]
        ).first()

        if existing:
            # Update existing
            for key, value in kwargs.items():
                setattr(existing, key, value)
            return existing
        else:
            # Create new
            market = Market(**kwargs)
            session.add(market)
            return market

    def add_snapshot(self, session: Session, market_id: int, **kwargs):
        """Add a market snapshot."""
        from src.database.models import MarketSnapshot

        snapshot = MarketSnapshot(market_id=market_id, **kwargs)
        session.add(snapshot)
        return snapshot

    def add_arbitrage(self, session: Session, **kwargs):
        """Record an arbitrage opportunity."""
        from src.database.models import ArbitrageOpportunity

        arb = ArbitrageOpportunity(**kwargs)
        session.add(arb)
        return arb

    def add_signal(self, session: Session, **kwargs):
        """Record a signal."""
        from src.database.models import Signal

        signal = Signal(**kwargs)
        session.add(signal)
        return signal

    def get_market_by_platform_id(self, session: Session, platform: str, platform_id: str):
        """Get market by platform and ID."""
        from src.database.models import Market

        return session.query(Market).filter_by(
            platform=platform,
            platform_id=platform_id
        ).first()

    def get_recent_snapshots(self, session: Session, market_id: int, limit: int = 100):
        """Get recent snapshots for a market."""
        from src.database.models import MarketSnapshot

        return session.query(MarketSnapshot).filter_by(
            market_id=market_id
        ).order_by(MarketSnapshot.timestamp.desc()).limit(limit).all()

    def get_active_arbitrage(self, session: Session):
        """Get active arbitrage opportunities."""
        from src.database.models import ArbitrageOpportunity

        return session.query(ArbitrageOpportunity).filter_by(
            status="active"
        ).order_by(ArbitrageOpportunity.detected_at.desc()).all()


# Global database manager instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
