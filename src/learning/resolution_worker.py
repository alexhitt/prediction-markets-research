"""
Resolution Worker for outcome feedback loop.

Background worker that periodically checks for resolved markets
and triggers learning updates for all affected bots.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import (
    PendingResolution, ResearchEstimate, PaperTrade, BotMemory
)
from src.market_discovery.resolution_tracker import ResolutionTracker
from .calibration_tracker import CalibrationTracker
from .domain_specialist import DomainSpecialist
from .incremental_updater import IncrementalUpdater


class ResolutionWorker:
    """
    Background worker that checks for market resolutions and triggers learning.

    Runs periodically (default every 5 minutes) to:
    1. Check all pending resolutions for outcomes
    2. Update Brier scores and P&L for resolved predictions
    3. Trigger calibration, domain, and weight learning updates
    4. Track overall bot performance improvements
    """

    def __init__(
        self,
        db_session: Session,
        resolution_tracker: ResolutionTracker,
        check_interval_seconds: int = 300
    ):
        self.db = db_session
        self.tracker = resolution_tracker
        self.check_interval = check_interval_seconds

        self.calibration = CalibrationTracker(db_session)
        self.domain = DomainSpecialist(db_session)
        self.updater = IncrementalUpdater(db_session)

        self._running = False
        self._last_check: Optional[datetime] = None
        self._stats = {
            "total_checks": 0,
            "total_resolutions": 0,
            "total_learning_updates": 0
        }

    async def start(self) -> None:
        """Start the resolution worker loop."""
        self._running = True
        logger.info(f"Resolution worker started (interval: {self.check_interval}s)")

        while self._running:
            try:
                await self.check_and_learn()
            except Exception as e:
                logger.error(f"Resolution worker error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop the resolution worker."""
        self._running = False
        logger.info("Resolution worker stopped")

    async def check_and_learn(self) -> Dict[str, Any]:
        """
        Check for resolutions and trigger learning updates.

        Returns summary of what was processed.
        """
        self._last_check = datetime.utcnow()
        self._stats["total_checks"] += 1

        result = {
            "timestamp": self._last_check.isoformat(),
            "resolutions_found": 0,
            "learning_updates": 0,
            "bots_updated": set(),
            "errors": []
        }

        check_result = self.tracker.check_all_pending()
        result["resolutions_found"] = check_result.get("resolved", 0)

        for resolution in check_result.get("resolutions", []):
            try:
                updates = await self._process_resolution_learning(
                    resolution["platform"],
                    resolution["market_id"],
                    resolution["outcome"]
                )
                result["learning_updates"] += updates["updates_count"]
                result["bots_updated"].update(updates["bots"])
            except Exception as e:
                error_msg = f"Error processing {resolution['market_id']}: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

        self._stats["total_resolutions"] += result["resolutions_found"]
        self._stats["total_learning_updates"] += result["learning_updates"]

        result["bots_updated"] = list(result["bots_updated"])

        if result["resolutions_found"] > 0:
            logger.info(
                f"Processed {result['resolutions_found']} resolutions, "
                f"{result['learning_updates']} learning updates"
            )

        return result

    async def _process_resolution_learning(
        self,
        platform: str,
        market_id: str,
        outcome: str
    ) -> Dict[str, Any]:
        """
        Process learning updates for a resolved market.

        Updates calibration, domain performance, and signal weights
        for all bots that made predictions on this market.
        """
        result = {"updates_count": 0, "bots": []}

        estimates = self.db.query(ResearchEstimate).filter(
            ResearchEstimate.platform == platform,
            ResearchEstimate.market_id == market_id,
            ResearchEstimate.actual_outcome.isnot(None)
        ).all()

        for estimate in estimates:
            if estimate.brier_score is None:
                continue

            bot_id = estimate.bot_id
            result["bots"].append(bot_id)

            self.calibration.update_calibration(
                bot_id=bot_id,
                predicted_probability=estimate.estimated_probability,
                actual_outcome=estimate.actual_outcome,
                brier_score=estimate.brier_score
            )

            if estimate.edge_realized is not None:
                self.domain.update_domain_performance(
                    bot_id=bot_id,
                    domain=estimate.market_category or "unknown",
                    brier_score=estimate.brier_score,
                    edge=estimate.edge_realized
                )

            sources = estimate.sources_used or []
            if sources and estimate.edge_realized is not None:
                self.updater.update_source_performance(
                    bot_id=bot_id,
                    sources_used=sources,
                    edge=estimate.edge_realized
                )

            self._update_bot_aggregate_stats(bot_id)
            result["updates_count"] += 1

        return result

    def _update_bot_aggregate_stats(self, bot_id: str) -> None:
        """Update aggregate statistics for a bot."""
        memory = self.db.query(BotMemory).filter_by(bot_id=bot_id).first()
        if not memory:
            return

        resolved_estimates = self.db.query(ResearchEstimate).filter(
            ResearchEstimate.bot_id == bot_id,
            ResearchEstimate.actual_outcome.isnot(None)
        ).all()

        if not resolved_estimates:
            return

        total_brier = sum(e.brier_score for e in resolved_estimates if e.brier_score)
        total_edge = sum(e.edge_realized for e in resolved_estimates if e.edge_realized)
        count = len(resolved_estimates)

        memory.resolved_predictions = count
        memory.average_brier_score = total_brier / count if count > 0 else None
        memory.average_edge = total_edge / count if count > 0 else None
        memory.updated_at = datetime.utcnow()

        self.db.commit()

    def get_learning_summary(self, bot_id: str) -> Dict[str, Any]:
        """
        Get comprehensive learning summary for a bot.

        Combines calibration, domain, and weight information.
        """
        return {
            "bot_id": bot_id,
            "calibration": self.calibration.get_calibration_summary(bot_id),
            "domains": self.domain.get_domain_summary(bot_id),
            "weights": self.updater.get_update_summary(bot_id),
            "worker_stats": self.get_worker_stats()
        }

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "running": self._running,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "check_interval_seconds": self.check_interval,
            "total_checks": self._stats["total_checks"],
            "total_resolutions": self._stats["total_resolutions"],
            "total_learning_updates": self._stats["total_learning_updates"]
        }

    def force_learning_update(self, bot_id: str) -> Dict[str, Any]:
        """
        Force a learning update for a specific bot.

        Recalculates all learning metrics from resolved predictions.
        """
        resolved = self.db.query(ResearchEstimate).filter(
            ResearchEstimate.bot_id == bot_id,
            ResearchEstimate.actual_outcome.isnot(None)
        ).all()

        updates = 0
        for estimate in resolved:
            if estimate.brier_score is None:
                continue

            self.calibration.update_calibration(
                bot_id=bot_id,
                predicted_probability=estimate.estimated_probability,
                actual_outcome=estimate.actual_outcome,
                brier_score=estimate.brier_score
            )

            if estimate.edge_realized is not None:
                self.domain.update_domain_performance(
                    bot_id=bot_id,
                    domain=estimate.market_category or "unknown",
                    brier_score=estimate.brier_score,
                    edge=estimate.edge_realized
                )

                sources = estimate.sources_used or []
                if sources:
                    self.updater.update_source_performance(
                        bot_id=bot_id,
                        sources_used=sources,
                        edge=estimate.edge_realized
                    )

            updates += 1

        self._update_bot_aggregate_stats(bot_id)

        return {
            "bot_id": bot_id,
            "predictions_processed": updates,
            "summary": self.get_learning_summary(bot_id)
        }

    def get_all_bots_performance(self) -> List[Dict[str, Any]]:
        """Get performance summary for all bots with learning data."""
        memories = self.db.query(BotMemory).all()

        performances = []
        for memory in memories:
            perf = {
                "bot_id": memory.bot_id,
                "total_predictions": memory.total_predictions,
                "resolved_predictions": memory.resolved_predictions,
                "average_brier_score": memory.average_brier_score,
                "average_edge": memory.average_edge,
                "confidence_multiplier": memory.confidence_multiplier,
                "strong_domains": self.domain.get_strong_domains(memory.bot_id),
                "weak_domains": self.domain.get_weak_domains(memory.bot_id),
                "calibration_quality": self.calibration.get_calibration_summary(memory.bot_id).get("calibration_quality"),
                "last_update": memory.updated_at.isoformat() if memory.updated_at else None
            }
            performances.append(perf)

        performances.sort(key=lambda x: x.get("average_edge") or 0, reverse=True)
        return performances
