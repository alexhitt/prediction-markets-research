"""
Domain Specialist for per-category performance tracking.

Tracks how well each bot performs in different categories (politics, crypto, sports, etc.)
and adjusts bet sizing based on domain strength.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import ResearchEstimate, BotMemory


@dataclass
class DomainPerformance:
    """Performance statistics for a specific domain/category."""
    domain: str
    count: int
    sum_brier: float
    sum_edge: float
    wins: int  # Predictions where we beat the market

    @property
    def average_brier(self) -> float:
        """Average Brier score in this domain."""
        if self.count == 0:
            return 0.25  # Baseline
        return self.sum_brier / self.count

    @property
    def average_edge(self) -> float:
        """Average edge vs market in this domain."""
        if self.count == 0:
            return 0.0
        return self.sum_edge / self.count

    @property
    def win_rate(self) -> float:
        """Rate at which we beat the market."""
        if self.count == 0:
            return 0.5
        return self.wins / self.count

    @property
    def skill_score(self) -> float:
        """
        Combined skill score for this domain.

        Higher is better. Combines Brier score (lower is better)
        and edge (higher is better).
        """
        # Brier contribution: 0.25 is random, 0 is perfect
        # Transform so higher is better: (0.25 - brier) / 0.25
        brier_contribution = max(0, (0.25 - self.average_brier) / 0.25)

        # Edge contribution: positive edge is good
        edge_contribution = max(0, min(1, self.average_edge * 10))  # Scale to 0-1

        # Win rate contribution
        win_contribution = self.win_rate

        # Weighted combination
        return (brier_contribution * 0.4 + edge_contribution * 0.3 + win_contribution * 0.3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "domain": self.domain,
            "count": self.count,
            "sum_brier": self.sum_brier,
            "sum_edge": self.sum_edge,
            "wins": self.wins,
            "average_brier": self.average_brier,
            "average_edge": self.average_edge,
            "win_rate": self.win_rate,
            "skill_score": self.skill_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainPerformance":
        """Create from dictionary."""
        return cls(
            domain=data["domain"],
            count=data.get("count", 0),
            sum_brier=data.get("sum_brier", 0.0),
            sum_edge=data.get("sum_edge", 0.0),
            wins=data.get("wins", 0)
        )


class DomainSpecialist:
    """
    Tracks per-domain/category performance for each bot.

    Enables bots to:
    - Know which categories they're good at
    - Adjust bet sizing based on domain strength
    - Skip domains where they consistently underperform
    """

    # Minimum predictions before using domain data
    MIN_PREDICTIONS_FOR_ADJUSTMENT = 5

    # Skill thresholds
    STRONG_DOMAIN_THRESHOLD = 0.6
    WEAK_DOMAIN_THRESHOLD = 0.3

    def __init__(self, db_session: Session):
        """
        Initialize the domain specialist.

        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session

    def get_domain_performance(self, bot_id: str) -> Dict[str, DomainPerformance]:
        """
        Get all domain performance data for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Dictionary mapping domain names to DomainPerformance
        """
        memory = self._get_or_create_memory(bot_id)
        stored_domains = memory.domain_performance or {}

        return {
            domain: DomainPerformance.from_dict(data)
            for domain, data in stored_domains.items()
        }

    def update_domain_performance(
        self,
        bot_id: str,
        domain: str,
        brier_score: float,
        edge: float
    ) -> DomainPerformance:
        """
        Update performance data for a specific domain.

        Args:
            bot_id: Bot identifier
            domain: Category/domain name
            brier_score: Brier score for this prediction
            edge: Edge vs market (positive = beat market)

        Returns:
            Updated DomainPerformance
        """
        memory = self._get_or_create_memory(bot_id)
        domains = self.get_domain_performance(bot_id)

        # Normalize domain name
        domain_key = domain.lower().strip()

        # Get or create domain performance
        if domain_key in domains:
            perf = domains[domain_key]
        else:
            perf = DomainPerformance(
                domain=domain_key,
                count=0,
                sum_brier=0.0,
                sum_edge=0.0,
                wins=0
            )

        # Update with new data (immutable pattern)
        updated_perf = DomainPerformance(
            domain=perf.domain,
            count=perf.count + 1,
            sum_brier=perf.sum_brier + brier_score,
            sum_edge=perf.sum_edge + edge,
            wins=perf.wins + (1 if edge > 0 else 0)
        )

        # Store updated data
        stored_domains = memory.domain_performance or {}
        stored_domains[domain_key] = updated_perf.to_dict()
        memory.domain_performance = stored_domains

        self.db.commit()

        logger.debug(
            f"Bot {bot_id} domain '{domain_key}' updated: "
            f"skill={updated_perf.skill_score:.3f}, "
            f"count={updated_perf.count}"
        )

        return updated_perf

    def get_domain_multiplier(self, bot_id: str, domain: str) -> float:
        """
        Get bet size multiplier based on domain performance.

        Args:
            bot_id: Bot identifier
            domain: Category/domain name

        Returns:
            Multiplier for bet sizing (0.5 to 1.5)
        """
        domains = self.get_domain_performance(bot_id)
        domain_key = domain.lower().strip()

        if domain_key not in domains:
            return 1.0  # No data, use default

        perf = domains[domain_key]

        if perf.count < self.MIN_PREDICTIONS_FOR_ADJUSTMENT:
            return 1.0  # Not enough data

        skill = perf.skill_score

        # Map skill score to multiplier
        # Strong domain (skill > 0.6): multiply up to 1.5x
        # Weak domain (skill < 0.3): multiply down to 0.5x
        if skill >= self.STRONG_DOMAIN_THRESHOLD:
            # Linear scale from 1.0 at 0.6 to 1.5 at 1.0
            return 1.0 + (skill - 0.6) * 1.25
        elif skill <= self.WEAK_DOMAIN_THRESHOLD:
            # Linear scale from 0.5 at 0.0 to 1.0 at 0.3
            return 0.5 + skill * (0.5 / 0.3)
        else:
            return 1.0  # Middle range, no adjustment

    def should_skip_domain(self, bot_id: str, domain: str) -> bool:
        """
        Check if bot should skip betting in this domain.

        Args:
            bot_id: Bot identifier
            domain: Category/domain name

        Returns:
            True if bot consistently underperforms in this domain
        """
        domains = self.get_domain_performance(bot_id)
        domain_key = domain.lower().strip()

        if domain_key not in domains:
            return False

        perf = domains[domain_key]

        # Need sufficient data
        if perf.count < 10:
            return False

        # Skip if very weak with consistent poor performance
        if perf.skill_score < 0.2 and perf.win_rate < 0.4:
            logger.info(f"Bot {bot_id} skipping weak domain '{domain_key}'")
            return True

        return False

    def get_strong_domains(self, bot_id: str) -> List[str]:
        """
        Get list of domains where bot excels.

        Args:
            bot_id: Bot identifier

        Returns:
            List of domain names where skill_score > threshold
        """
        domains = self.get_domain_performance(bot_id)

        strong = [
            domain for domain, perf in domains.items()
            if perf.count >= self.MIN_PREDICTIONS_FOR_ADJUSTMENT
            and perf.skill_score >= self.STRONG_DOMAIN_THRESHOLD
        ]

        return sorted(strong, key=lambda d: domains[d].skill_score, reverse=True)

    def get_weak_domains(self, bot_id: str) -> List[str]:
        """
        Get list of domains where bot struggles.

        Args:
            bot_id: Bot identifier

        Returns:
            List of domain names where skill_score < threshold
        """
        domains = self.get_domain_performance(bot_id)

        weak = [
            domain for domain, perf in domains.items()
            if perf.count >= self.MIN_PREDICTIONS_FOR_ADJUSTMENT
            and perf.skill_score <= self.WEAK_DOMAIN_THRESHOLD
        ]

        return sorted(weak, key=lambda d: domains[d].skill_score)

    def get_domain_summary(self, bot_id: str) -> Dict[str, Any]:
        """
        Get a summary of domain performance.

        Args:
            bot_id: Bot identifier

        Returns:
            Summary dictionary
        """
        domains = self.get_domain_performance(bot_id)

        if not domains:
            return {
                "bot_id": bot_id,
                "total_domains": 0,
                "strong_domains": [],
                "weak_domains": [],
                "domains": {}
            }

        return {
            "bot_id": bot_id,
            "total_domains": len(domains),
            "strong_domains": self.get_strong_domains(bot_id),
            "weak_domains": self.get_weak_domains(bot_id),
            "best_domain": max(domains.items(), key=lambda x: x[1].skill_score)[0] if domains else None,
            "worst_domain": min(domains.items(), key=lambda x: x[1].skill_score)[0] if domains else None,
            "domains": {name: perf.to_dict() for name, perf in domains.items()}
        }

    def _get_or_create_memory(self, bot_id: str) -> BotMemory:
        """Get or create BotMemory record."""
        memory = self.db.query(BotMemory).filter_by(bot_id=bot_id).first()

        if not memory:
            memory = BotMemory(
                bot_id=bot_id,
                signal_weights={},
                calibration_buckets={},
                domain_performance={},
                confidence_multiplier=1.0,
                source_scores={},
                total_predictions=0,
                resolved_predictions=0,
                created_at=datetime.utcnow()
            )
            self.db.add(memory)
            self.db.commit()

        return memory
