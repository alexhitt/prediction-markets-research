"""
Incremental Updater for signal weight optimization.

Adjusts signal weights based on which sources contributed to
winning vs losing predictions, using small incremental updates.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import ResearchEstimate, BotMemory


@dataclass
class SourceScore:
    """Performance score for a signal source."""
    source: str
    total_uses: int
    winning_uses: int
    sum_edge: float

    @property
    def win_rate(self) -> float:
        if self.total_uses == 0:
            return 0.5
        return self.winning_uses / self.total_uses

    @property
    def average_edge(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.sum_edge / self.total_uses

    @property
    def reliability_score(self) -> float:
        if self.total_uses < 3:
            return 0.5
        return (self.win_rate * 0.6 + max(0, min(1, self.average_edge * 5 + 0.5)) * 0.4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "total_uses": self.total_uses,
            "winning_uses": self.winning_uses,
            "sum_edge": self.sum_edge,
            "win_rate": self.win_rate,
            "average_edge": self.average_edge,
            "reliability_score": self.reliability_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceScore":
        return cls(
            source=data["source"],
            total_uses=data.get("total_uses", 0),
            winning_uses=data.get("winning_uses", 0),
            sum_edge=data.get("sum_edge", 0.0)
        )


class IncrementalUpdater:
    """
    Incrementally updates signal weights based on outcomes.

    Uses small, conservative updates to avoid overfitting.
    """

    BASE_LEARNING_RATE = 0.05
    MIN_LEARNING_RATE = 0.01
    DECAY_FACTOR = 0.995
    MIN_SAMPLES_FOR_UPDATE = 5
    MAX_ADJUSTMENT_PER_UPDATE = 0.02

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_source_scores(self, bot_id: str) -> Dict[str, SourceScore]:
        memory = self._get_or_create_memory(bot_id)
        stored_scores = memory.source_scores or {}
        return {
            source: SourceScore.from_dict(data)
            for source, data in stored_scores.items()
        }

    def get_signal_weights(self, bot_id: str) -> Dict[str, float]:
        memory = self._get_or_create_memory(bot_id)
        return memory.signal_weights or {}

    def update_source_performance(
        self,
        bot_id: str,
        sources_used: List[str],
        edge: float
    ) -> Dict[str, SourceScore]:
        memory = self._get_or_create_memory(bot_id)
        scores = self.get_source_scores(bot_id)
        won = edge > 0

        for source in sources_used:
            source_key = source.lower().strip()
            if source_key in scores:
                score = scores[source_key]
            else:
                score = SourceScore(source=source_key, total_uses=0, winning_uses=0, sum_edge=0.0)

            updated_score = SourceScore(
                source=score.source,
                total_uses=score.total_uses + 1,
                winning_uses=score.winning_uses + (1 if won else 0),
                sum_edge=score.sum_edge + edge
            )
            scores[source_key] = updated_score

        stored_scores = {name: score.to_dict() for name, score in scores.items()}
        memory.source_scores = stored_scores
        self._maybe_update_weights(bot_id, memory, scores)
        self.db.commit()
        return scores

    def _maybe_update_weights(
        self,
        bot_id: str,
        memory: BotMemory,
        scores: Dict[str, SourceScore]
    ) -> None:
        total_samples = sum(s.total_uses for s in scores.values())
        if total_samples < self.MIN_SAMPLES_FOR_UPDATE:
            return

        learning_rate = max(
            self.MIN_LEARNING_RATE,
            self.BASE_LEARNING_RATE * (self.DECAY_FACTOR ** total_samples)
        )

        current_weights = memory.signal_weights or {}
        default_weights = {
            "base_rate": 0.15, "momentum": 0.15, "sentiment": 0.15,
            "whale": 0.15, "news": 0.15, "contrarian": 0.10, "technical": 0.15
        }

        new_weights = {}
        for source, score in scores.items():
            if score.total_uses < 3:
                continue
            current = current_weights.get(source, default_weights.get(source, 0.1))
            reliability = score.reliability_score
            target_adjustment = (reliability - 0.5) * learning_rate
            bounded_adjustment = max(-self.MAX_ADJUSTMENT_PER_UPDATE, min(self.MAX_ADJUSTMENT_PER_UPDATE, target_adjustment))
            new_weight = max(0.05, min(0.40, current + bounded_adjustment))
            new_weights[source] = new_weight

        for source, weight in current_weights.items():
            if source not in new_weights:
                new_weights[source] = weight

        if new_weights:
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: v / total for k, v in new_weights.items()}

        memory.signal_weights = new_weights
        memory.last_learning_update = datetime.utcnow()

    def get_weighted_sources(self, bot_id: str) -> List[Tuple[str, float, float]]:
        weights = self.get_signal_weights(bot_id)
        scores = self.get_source_scores(bot_id)
        result = []
        for source, weight in weights.items():
            reliability = scores[source].reliability_score if source in scores else 0.5
            result.append((source, weight, reliability))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def get_update_summary(self, bot_id: str) -> Dict[str, Any]:
        memory = self._get_or_create_memory(bot_id)
        weights = self.get_signal_weights(bot_id)
        scores = self.get_source_scores(bot_id)
        total_samples = sum(s.total_uses for s in scores.values())
        current_lr = max(self.MIN_LEARNING_RATE, self.BASE_LEARNING_RATE * (self.DECAY_FACTOR ** total_samples))

        return {
            "bot_id": bot_id,
            "total_samples": total_samples,
            "current_learning_rate": current_lr,
            "weights": weights,
            "source_scores": {name: score.to_dict() for name, score in scores.items()},
            "top_sources": self.get_weighted_sources(bot_id)[:5],
            "last_update": memory.last_learning_update.isoformat() if memory.last_learning_update else None
        }

    def reset_weights(self, bot_id: str) -> None:
        memory = self._get_or_create_memory(bot_id)
        memory.signal_weights = {}
        memory.source_scores = {}
        self.db.commit()

    def _get_or_create_memory(self, bot_id: str) -> BotMemory:
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
