"""
Hypothesis Tracker

Manages hypotheses about alternative data signals and their predictive power.
Tracks predictions against live markets and calculates Brier scores.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.database.db import get_session
from src.database.models import Hypothesis, HypothesisPrediction
from src.analytics.scoring import (
    calculate_brier_score,
    calculate_edge,
    calculate_aggregate_scores,
    AggregateScores
)


@dataclass
class HypothesisSummary:
    """Summary of a hypothesis for display."""
    id: int
    name: str
    data_source: str
    status: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    average_brier_score: Optional[float]
    average_edge: Optional[float]
    created_at: datetime


class HypothesisTracker:
    """
    Manages hypothesis creation, prediction tracking, and scoring.

    Usage:
        tracker = HypothesisTracker()

        # Create hypothesis
        hyp_id = tracker.create_hypothesis(
            name="Ice Cream Recession Signal",
            data_source="ice_cream",
            causal_theory="High failure rates indicate economic stress"
        )

        # Record prediction
        tracker.record_prediction(
            hypothesis_id=hyp_id,
            platform="polymarket",
            market_id="123",
            market_question="Will there be a recession?",
            predicted_probability=0.65,
            market_probability=0.45,
            prediction_direction="yes",
            signal_value=0.25
        )

        # When market resolves
        tracker.resolve_prediction(prediction_id=1, actual_outcome=1.0)
    """

    def create_hypothesis(
        self,
        name: str,
        data_source: str,
        description: Optional[str] = None,
        causal_theory: Optional[str] = None
    ) -> int:
        """
        Create a new hypothesis.

        Args:
            name: Human-readable name
            data_source: Signal source (e.g., "ice_cream", "google_trends")
            description: Detailed description
            causal_theory: Why this signal predicts market outcomes

        Returns:
            Hypothesis ID
        """
        with get_session() as session:
            hypothesis = Hypothesis(
                name=name,
                data_source=data_source,
                description=description,
                causal_theory=causal_theory,
                status="active"
            )
            session.add(hypothesis)
            session.flush()
            hypothesis_id = hypothesis.id

        logger.info(f"Created hypothesis: {name} (ID: {hypothesis_id})")
        return hypothesis_id

    def update_hypothesis(
        self,
        hypothesis_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        causal_theory: Optional[str] = None,
        status: Optional[str] = None
    ) -> bool:
        """
        Update an existing hypothesis.

        Args:
            hypothesis_id: ID of hypothesis to update
            name: New name (optional)
            description: New description (optional)
            causal_theory: New theory (optional)
            status: New status (optional)

        Returns:
            True if updated successfully
        """
        with get_session() as session:
            hypothesis = session.query(Hypothesis).filter_by(id=hypothesis_id).first()

            if not hypothesis:
                logger.warning(f"Hypothesis not found: {hypothesis_id}")
                return False

            if name is not None:
                hypothesis.name = name
            if description is not None:
                hypothesis.description = description
            if causal_theory is not None:
                hypothesis.causal_theory = causal_theory
            if status is not None:
                hypothesis.status = status

        logger.info(f"Updated hypothesis: {hypothesis_id}")
        return True

    def record_prediction(
        self,
        hypothesis_id: int,
        platform: str,
        market_id: str,
        market_question: str,
        predicted_probability: float,
        market_probability: float,
        prediction_direction: str,
        signal_value: Optional[float] = None,
        signal_data: Optional[Dict] = None
    ) -> int:
        """
        Record a prediction based on a hypothesis.

        Args:
            hypothesis_id: ID of the hypothesis
            platform: Market platform (polymarket, kalshi)
            market_id: Platform's market ID
            market_question: Market question text
            predicted_probability: Our predicted probability (0-1)
            market_probability: Market price at prediction time (0-1)
            prediction_direction: "yes" or "no"
            signal_value: Value of the signal at prediction time
            signal_data: Additional signal data

        Returns:
            Prediction ID
        """
        with get_session() as session:
            prediction = HypothesisPrediction(
                hypothesis_id=hypothesis_id,
                platform=platform,
                market_id=market_id,
                market_question=market_question,
                predicted_probability=predicted_probability,
                market_probability=market_probability,
                prediction_direction=prediction_direction,
                signal_value=signal_value,
                signal_data=signal_data
            )
            session.add(prediction)
            session.flush()
            prediction_id = prediction.id

            # Update hypothesis prediction count
            hypothesis = session.query(Hypothesis).filter_by(id=hypothesis_id).first()
            if hypothesis:
                hypothesis.total_predictions = (hypothesis.total_predictions or 0) + 1

        logger.info(
            f"Recorded prediction for hypothesis {hypothesis_id}: "
            f"{predicted_probability:.0%} {prediction_direction} (market: {market_probability:.0%})"
        )
        return prediction_id

    def resolve_prediction(
        self,
        prediction_id: int,
        actual_outcome: float
    ) -> Dict:
        """
        Resolve a prediction with the actual outcome.

        Calculates Brier score, edge, and updates hypothesis statistics.

        Args:
            prediction_id: ID of the prediction
            actual_outcome: Actual outcome (1.0 for yes, 0.0 for no)

        Returns:
            Dict with scoring results
        """
        with get_session() as session:
            prediction = session.query(HypothesisPrediction).filter_by(
                id=prediction_id
            ).first()

            if not prediction:
                logger.warning(f"Prediction not found: {prediction_id}")
                return {"error": "Prediction not found"}

            if prediction.actual_outcome is not None:
                logger.warning(f"Prediction already resolved: {prediction_id}")
                return {"error": "Already resolved"}

            # Calculate scores
            our_brier = calculate_brier_score(
                prediction.predicted_probability,
                actual_outcome
            )
            market_brier = calculate_brier_score(
                prediction.market_probability,
                actual_outcome
            )
            edge = calculate_edge(our_brier, market_brier)

            # Update prediction
            prediction.actual_outcome = actual_outcome
            prediction.our_brier_score = our_brier
            prediction.market_brier_score = market_brier
            prediction.edge = edge
            prediction.resolved_at = datetime.utcnow()

            # Check if correct
            predicted_yes = prediction.prediction_direction == "yes"
            actual_yes = actual_outcome > 0.5
            is_correct = predicted_yes == actual_yes

            # Update hypothesis statistics
            hypothesis = session.query(Hypothesis).filter_by(
                id=prediction.hypothesis_id
            ).first()

            if hypothesis:
                if is_correct:
                    hypothesis.correct_predictions = (
                        hypothesis.correct_predictions or 0
                    ) + 1

                # Recalculate average scores
                resolved_preds = session.query(HypothesisPrediction).filter(
                    HypothesisPrediction.hypothesis_id == hypothesis.id,
                    HypothesisPrediction.actual_outcome.isnot(None)
                ).all()

                if resolved_preds:
                    hypothesis.average_brier_score = sum(
                        p.our_brier_score for p in resolved_preds
                    ) / len(resolved_preds)
                    hypothesis.average_edge = sum(
                        p.edge for p in resolved_preds
                    ) / len(resolved_preds)

        result = {
            "prediction_id": prediction_id,
            "our_brier_score": our_brier,
            "market_brier_score": market_brier,
            "edge": edge,
            "is_correct": is_correct
        }

        logger.info(
            f"Resolved prediction {prediction_id}: "
            f"Brier={our_brier:.4f}, Edge={edge:+.4f}, Correct={is_correct}"
        )
        return result

    def get_hypothesis(self, hypothesis_id: int) -> Optional[HypothesisSummary]:
        """Get hypothesis summary by ID."""
        with get_session() as session:
            hypothesis = session.query(Hypothesis).filter_by(id=hypothesis_id).first()

            if not hypothesis:
                return None

            accuracy = 0.0
            if hypothesis.total_predictions and hypothesis.total_predictions > 0:
                accuracy = (hypothesis.correct_predictions or 0) / hypothesis.total_predictions

            return HypothesisSummary(
                id=hypothesis.id,
                name=hypothesis.name,
                data_source=hypothesis.data_source,
                status=hypothesis.status,
                total_predictions=hypothesis.total_predictions or 0,
                correct_predictions=hypothesis.correct_predictions or 0,
                accuracy=accuracy,
                average_brier_score=hypothesis.average_brier_score,
                average_edge=hypothesis.average_edge,
                created_at=hypothesis.created_at
            )

    def list_hypotheses(
        self,
        status: Optional[str] = None,
        data_source: Optional[str] = None
    ) -> List[HypothesisSummary]:
        """
        List all hypotheses with optional filtering.

        Args:
            status: Filter by status (optional)
            data_source: Filter by data source (optional)

        Returns:
            List of HypothesisSummary objects
        """
        with get_session() as session:
            query = session.query(Hypothesis)

            if status:
                query = query.filter(Hypothesis.status == status)
            if data_source:
                query = query.filter(Hypothesis.data_source == data_source)

            hypotheses = query.order_by(Hypothesis.created_at.desc()).all()

            return [
                HypothesisSummary(
                    id=h.id,
                    name=h.name,
                    data_source=h.data_source,
                    status=h.status,
                    total_predictions=h.total_predictions or 0,
                    correct_predictions=h.correct_predictions or 0,
                    accuracy=(
                        (h.correct_predictions or 0) / h.total_predictions
                        if h.total_predictions else 0.0
                    ),
                    average_brier_score=h.average_brier_score,
                    average_edge=h.average_edge,
                    created_at=h.created_at
                )
                for h in hypotheses
            ]

    def get_hypothesis_scores(self, hypothesis_id: int) -> Optional[AggregateScores]:
        """
        Get detailed aggregate scores for a hypothesis.

        Args:
            hypothesis_id: ID of the hypothesis

        Returns:
            AggregateScores or None if no resolved predictions
        """
        with get_session() as session:
            predictions = session.query(HypothesisPrediction).filter(
                HypothesisPrediction.hypothesis_id == hypothesis_id,
                HypothesisPrediction.actual_outcome.isnot(None)
            ).all()

            if not predictions:
                return None

            # Convert to tuples for scoring
            pred_tuples = [
                (p.predicted_probability, p.market_probability, p.actual_outcome)
                for p in predictions
            ]

            return calculate_aggregate_scores(pred_tuples)

    def get_pending_predictions(
        self,
        hypothesis_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get predictions that haven't been resolved yet.

        Args:
            hypothesis_id: Filter by hypothesis (optional)

        Returns:
            List of pending prediction dicts
        """
        with get_session() as session:
            query = session.query(HypothesisPrediction).filter(
                HypothesisPrediction.actual_outcome.is_(None)
            )

            if hypothesis_id:
                query = query.filter(
                    HypothesisPrediction.hypothesis_id == hypothesis_id
                )

            predictions = query.order_by(
                HypothesisPrediction.created_at.desc()
            ).all()

            return [
                {
                    "id": p.id,
                    "hypothesis_id": p.hypothesis_id,
                    "platform": p.platform,
                    "market_id": p.market_id,
                    "market_question": p.market_question,
                    "predicted_probability": p.predicted_probability,
                    "market_probability": p.market_probability,
                    "prediction_direction": p.prediction_direction,
                    "created_at": p.created_at.isoformat()
                }
                for p in predictions
            ]

    def get_leaderboard(self, min_predictions: int = 3) -> List[HypothesisSummary]:
        """
        Get hypothesis leaderboard sorted by edge.

        Args:
            min_predictions: Minimum resolved predictions to be included

        Returns:
            List of HypothesisSummary sorted by average edge
        """
        hypotheses = self.list_hypotheses(status="active")

        # Filter by min predictions and sort by edge
        qualified = [
            h for h in hypotheses
            if h.total_predictions >= min_predictions and h.average_edge is not None
        ]

        return sorted(qualified, key=lambda h: h.average_edge or 0, reverse=True)


if __name__ == "__main__":
    print("Testing Hypothesis Tracker")
    print("-" * 50)

    tracker = HypothesisTracker()

    # List existing hypotheses
    hypotheses = tracker.list_hypotheses()
    print(f"\nExisting hypotheses: {len(hypotheses)}")

    for h in hypotheses[:5]:
        print(f"\n  {h.name} ({h.status})")
        print(f"    Source: {h.data_source}")
        print(f"    Predictions: {h.total_predictions}")
        if h.average_brier_score is not None:
            print(f"    Avg Brier: {h.average_brier_score:.4f}")
        if h.average_edge is not None:
            print(f"    Avg Edge: {h.average_edge:+.4f}")
