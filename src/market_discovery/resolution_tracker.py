"""
Resolution Tracker for pending market resolutions.

Monitors markets where we have open predictions/bets and
updates outcomes when markets resolve, triggering the
learning feedback loop.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import (
    PendingResolution, ResearchEstimate, PaperTrade, BotMemory
)


class ResolutionTracker:
    """
    Tracks and resolves pending market predictions.

    Periodically checks markets for resolution and updates:
    - ResearchEstimate records with actual outcomes and Brier scores
    - PaperTrade records with P&L
    - BotMemory with learning updates
    """

    def __init__(
        self,
        db_session: Session,
        market_scanner=None,
        on_resolution: Optional[Callable] = None
    ):
        """
        Initialize the resolution tracker.

        Args:
            db_session: SQLAlchemy session
            market_scanner: MarketScanner instance for fetching market status
            on_resolution: Optional callback when a market resolves
        """
        self.db = db_session
        self.scanner = market_scanner
        self.on_resolution = on_resolution

    def add_pending_resolution(
        self,
        platform: str,
        market_id: str,
        market_question: str,
        expected_resolution_date: Optional[datetime] = None,
        research_estimate_id: Optional[int] = None,
        paper_trade_id: Optional[int] = None
    ) -> PendingResolution:
        """
        Add a market to the pending resolution queue.

        Args:
            platform: polymarket or kalshi
            market_id: Platform-specific market identifier
            market_question: The market question text
            expected_resolution_date: When the market is expected to resolve
            research_estimate_id: Optional linked research estimate ID
            paper_trade_id: Optional linked paper trade ID

        Returns:
            PendingResolution record
        """
        # Check if already tracking this market
        existing = self.db.query(PendingResolution).filter_by(
            platform=platform,
            market_id=market_id
        ).first()

        if existing:
            # Update existing record with new links
            if research_estimate_id:
                estimate_ids = existing.research_estimate_ids or []
                if research_estimate_id not in estimate_ids:
                    existing.research_estimate_ids = [*estimate_ids, research_estimate_id]

            if paper_trade_id:
                trade_ids = existing.paper_trade_ids or []
                if paper_trade_id not in trade_ids:
                    existing.paper_trade_ids = [*trade_ids, paper_trade_id]

            self.db.commit()
            return existing

        # Create new pending resolution
        pending = PendingResolution(
            platform=platform,
            market_id=market_id,
            market_question=market_question,
            expected_resolution_date=expected_resolution_date,
            research_estimate_ids=[research_estimate_id] if research_estimate_id else [],
            paper_trade_ids=[paper_trade_id] if paper_trade_id else [],
            status="pending",
            created_at=datetime.utcnow()
        )
        self.db.add(pending)
        self.db.commit()

        logger.info(f"Added pending resolution: {platform}/{market_id}")
        return pending

    def check_all_pending(self) -> Dict[str, Any]:
        """
        Check all pending resolutions for updates.

        Returns:
            Summary dict with counts of checked, resolved, and errors
        """
        stats = {
            "checked": 0,
            "resolved": 0,
            "errors": 0,
            "resolutions": []
        }

        pending_markets = self.db.query(PendingResolution).filter_by(
            status="pending"
        ).all()

        for pending in pending_markets:
            try:
                result = self._check_single_resolution(pending)
                stats["checked"] += 1

                if result and result.get("resolved"):
                    stats["resolved"] += 1
                    stats["resolutions"].append({
                        "platform": pending.platform,
                        "market_id": pending.market_id,
                        "outcome": result.get("outcome")
                    })

            except Exception as e:
                logger.error(f"Error checking {pending.platform}/{pending.market_id}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Resolution check complete: {stats['checked']} checked, "
            f"{stats['resolved']} resolved, {stats['errors']} errors"
        )

        return stats

    def _check_single_resolution(self, pending: PendingResolution) -> Optional[Dict[str, Any]]:
        """
        Check a single pending resolution for updates.

        Args:
            pending: PendingResolution record to check

        Returns:
            Resolution result dict if resolved, None otherwise
        """
        if not self.scanner:
            logger.warning("No market scanner configured")
            return None

        # Update check tracking
        pending.last_checked = datetime.utcnow()
        pending.check_count = (pending.check_count or 0) + 1

        # Get current market status
        resolution_info = self.scanner.check_market_resolution(
            pending.platform,
            pending.market_id
        )

        if not resolution_info:
            self.db.commit()
            return None

        if not resolution_info.get("resolved"):
            self.db.commit()
            return resolution_info

        # Market has resolved - process the outcome
        return self._process_resolution(pending, resolution_info)

    def _process_resolution(
        self,
        pending: PendingResolution,
        resolution_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a resolved market.

        Updates:
        - PendingResolution status
        - All linked ResearchEstimate records with Brier scores
        - All linked PaperTrade records with P&L
        - Triggers learning updates

        Args:
            pending: The pending resolution record
            resolution_info: Resolution details from market scanner

        Returns:
            Resolution result dict
        """
        outcome = resolution_info.get("outcome", "").lower()
        outcome_value = resolution_info.get("value", 0.0)
        resolved_at = resolution_info.get("resolved_at", datetime.utcnow())

        # Update pending resolution record
        pending.status = "resolved"
        pending.resolution = outcome
        pending.resolution_value = outcome_value
        pending.resolved_at = resolved_at

        logger.info(
            f"Market resolved: {pending.platform}/{pending.market_id} -> {outcome}"
        )

        # Update all linked research estimates
        estimates_updated = 0
        for estimate_id in (pending.research_estimate_ids or []):
            updated = self._update_research_estimate(estimate_id, outcome_value, resolved_at)
            if updated:
                estimates_updated += 1

        # Update all linked paper trades
        trades_updated = 0
        for trade_id in (pending.paper_trade_ids or []):
            updated = self._update_paper_trade(trade_id, outcome, outcome_value, resolved_at)
            if updated:
                trades_updated += 1

        self.db.commit()

        # Trigger callback if provided
        if self.on_resolution:
            try:
                self.on_resolution(pending, resolution_info)
            except Exception as e:
                logger.error(f"Resolution callback error: {e}")

        return {
            "resolved": True,
            "outcome": outcome,
            "value": outcome_value,
            "estimates_updated": estimates_updated,
            "trades_updated": trades_updated
        }

    def _update_research_estimate(
        self,
        estimate_id: int,
        outcome_value: float,
        resolved_at: datetime
    ) -> bool:
        """
        Update a research estimate with resolution outcome.

        Calculates Brier scores for both the bot's estimate and the market price.
        """
        estimate = self.db.query(ResearchEstimate).get(estimate_id)

        if not estimate:
            logger.warning(f"Research estimate {estimate_id} not found")
            return False

        if estimate.actual_outcome is not None:
            # Already resolved
            return False

        # Calculate Brier scores
        # Brier score = (predicted - actual)^2
        # Lower is better
        bot_brier = (estimate.estimated_probability - outcome_value) ** 2
        market_brier = (estimate.market_price_at_estimate - outcome_value) ** 2

        # Edge = market_brier - bot_brier
        # Positive edge means we beat the market
        edge_realized = market_brier - bot_brier

        # Update the estimate
        estimate.actual_outcome = outcome_value
        estimate.brier_score = bot_brier
        estimate.market_brier_score = market_brier
        estimate.edge_realized = edge_realized
        estimate.resolved_at = resolved_at

        logger.debug(
            f"Estimate {estimate_id}: Brier={bot_brier:.4f}, "
            f"Market={market_brier:.4f}, Edge={edge_realized:.4f}"
        )

        return True

    def _update_paper_trade(
        self,
        trade_id: int,
        outcome: str,
        outcome_value: float,
        resolved_at: datetime
    ) -> bool:
        """
        Update a paper trade with resolution outcome.

        Calculates realized P&L based on the outcome.
        """
        trade = self.db.query(PaperTrade).get(trade_id)

        if not trade:
            logger.warning(f"Paper trade {trade_id} not found")
            return False

        if trade.status == "closed":
            # Already resolved
            return False

        # Calculate P&L
        # If we bought YES and outcome is YES (1.0), we profit
        # If we bought NO and outcome is NO (0.0), we profit
        if trade.side.lower() == "yes":
            # Bought YES at entry_price, settles at outcome_value
            exit_price = outcome_value
        else:
            # Bought NO at entry_price (which is 1 - yes_price effectively)
            # If outcome is NO (0.0), YES settles at 0, NO settles at 1
            exit_price = 1.0 - outcome_value

        # P&L per contract
        pnl_per_contract = exit_price - trade.entry_price
        realized_pnl = pnl_per_contract * trade.size

        # Update trade
        trade.exit_price = exit_price
        trade.realized_pnl = realized_pnl
        trade.pnl_percent = (pnl_per_contract / trade.entry_price) * 100 if trade.entry_price > 0 else 0
        trade.status = "closed"
        trade.closed_at = resolved_at

        logger.debug(
            f"Trade {trade_id}: Entry={trade.entry_price:.2f}, "
            f"Exit={exit_price:.2f}, P&L=${realized_pnl:.2f}"
        )

        return True

    def get_pending_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all pending resolutions.

        Returns:
            Summary dict with counts and lists by status
        """
        all_pending = self.db.query(PendingResolution).all()

        summary = {
            "total": len(all_pending),
            "pending": 0,
            "resolved": 0,
            "expired": 0,
            "error": 0,
            "by_platform": {"polymarket": 0, "kalshi": 0},
            "upcoming_resolutions": [],
            "recently_resolved": []
        }

        now = datetime.utcnow()

        for p in all_pending:
            # Count by status
            if p.status == "pending":
                summary["pending"] += 1

                # Track upcoming
                if p.expected_resolution_date:
                    days_until = (p.expected_resolution_date - now).days
                    if 0 <= days_until <= 7:
                        summary["upcoming_resolutions"].append({
                            "platform": p.platform,
                            "market_id": p.market_id,
                            "question": p.market_question[:100],
                            "days_until": days_until
                        })

            elif p.status == "resolved":
                summary["resolved"] += 1

                # Track recently resolved
                if p.resolved_at and (now - p.resolved_at).days <= 7:
                    summary["recently_resolved"].append({
                        "platform": p.platform,
                        "market_id": p.market_id,
                        "resolution": p.resolution,
                        "resolved_at": p.resolved_at.isoformat()
                    })

            elif p.status == "expired":
                summary["expired"] += 1
            else:
                summary["error"] += 1

            # Count by platform
            if p.platform in summary["by_platform"]:
                summary["by_platform"][p.platform] += 1

        # Sort upcoming by days until resolution
        summary["upcoming_resolutions"].sort(key=lambda x: x["days_until"])

        return summary

    def cleanup_old_resolved(self, days_to_keep: int = 90) -> int:
        """
        Clean up old resolved entries.

        Args:
            days_to_keep: Keep resolved entries for this many days

        Returns:
            Number of entries deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)

        old_resolved = self.db.query(PendingResolution).filter(
            PendingResolution.status == "resolved",
            PendingResolution.resolved_at < cutoff
        ).all()

        count = len(old_resolved)

        for entry in old_resolved:
            self.db.delete(entry)

        self.db.commit()

        if count > 0:
            logger.info(f"Cleaned up {count} old resolved entries")

        return count
