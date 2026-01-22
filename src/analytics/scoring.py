"""
Scoring Module

Provides Brier score calculation, edge measurement, and prediction performance metrics.
Used for evaluating hypothesis predictions against market outcomes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class PredictionScore:
    """Immutable score for a single prediction."""
    predicted_probability: float
    actual_outcome: float  # 1.0 or 0.0
    market_probability: float
    brier_score: float
    market_brier_score: float
    edge: float  # Positive = we beat the market
    calibration_bucket: str  # e.g., "0.6-0.7"


@dataclass
class AggregateScores:
    """Aggregate scoring metrics across multiple predictions."""
    total_predictions: int
    average_brier_score: float
    average_market_brier_score: float
    average_edge: float
    accuracy: float  # % of correct predictions
    calibration_error: float  # Mean absolute calibration error
    log_loss: float
    roi: float  # Return on investment if betting

    # Breakdown by confidence bucket
    bucket_scores: Dict[str, Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "average_brier_score": self.average_brier_score,
            "average_market_brier_score": self.average_market_brier_score,
            "average_edge": self.average_edge,
            "accuracy": self.accuracy,
            "calibration_error": self.calibration_error,
            "log_loss": self.log_loss,
            "roi": self.roi,
            "bucket_scores": self.bucket_scores
        }


def calculate_brier_score(predicted_probability: float, actual_outcome: float) -> float:
    """
    Calculate Brier score for a single prediction.

    Brier score = (predicted - actual)^2

    Lower is better:
    - 0.0 = perfect prediction
    - 0.25 = random guessing (predicting 0.5)
    - 1.0 = completely wrong

    Args:
        predicted_probability: Our predicted probability (0-1)
        actual_outcome: Actual outcome (1.0 for yes, 0.0 for no)

    Returns:
        Brier score (0-1)
    """
    predicted = max(0.0, min(1.0, predicted_probability))
    actual = 1.0 if actual_outcome > 0.5 else 0.0
    return (predicted - actual) ** 2


def calculate_edge(
    our_brier_score: float,
    market_brier_score: float
) -> float:
    """
    Calculate edge vs market.

    Edge = market_brier - our_brier

    Positive edge means we performed better than the market.

    Args:
        our_brier_score: Our Brier score
        market_brier_score: Market's Brier score

    Returns:
        Edge (positive = we beat market)
    """
    return market_brier_score - our_brier_score


def calculate_log_loss(
    predicted_probability: float,
    actual_outcome: float,
    epsilon: float = 1e-15
) -> float:
    """
    Calculate log loss (cross-entropy) for a single prediction.

    Log loss penalizes confident wrong predictions more heavily.

    Args:
        predicted_probability: Predicted probability (0-1)
        actual_outcome: Actual outcome (1 or 0)
        epsilon: Small value to avoid log(0)

    Returns:
        Log loss value
    """
    p = max(epsilon, min(1 - epsilon, predicted_probability))
    if actual_outcome > 0.5:
        return -np.log(p)
    else:
        return -np.log(1 - p)


def calculate_calibration_error(
    predictions: List[float],
    outcomes: List[float],
    n_bins: int = 10
) -> Tuple[float, Dict[str, Dict]]:
    """
    Calculate calibration error and bucket breakdown.

    Perfect calibration: predictions at 70% are correct 70% of the time.

    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes
        n_bins: Number of calibration bins

    Returns:
        Tuple of (mean_calibration_error, bucket_details)
    """
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        return 0.0, {}

    bins = {}
    for pred, outcome in zip(predictions, outcomes):
        bucket_idx = min(int(pred * n_bins), n_bins - 1)
        bucket_key = f"{bucket_idx / n_bins:.1f}-{(bucket_idx + 1) / n_bins:.1f}"

        if bucket_key not in bins:
            bins[bucket_key] = {"predictions": [], "outcomes": []}

        bins[bucket_key]["predictions"].append(pred)
        bins[bucket_key]["outcomes"].append(outcome)

    # Calculate calibration error per bucket
    total_error = 0.0
    bucket_details = {}

    for bucket_key, data in bins.items():
        avg_pred = np.mean(data["predictions"])
        avg_outcome = np.mean(data["outcomes"])
        error = abs(avg_pred - avg_outcome)

        bucket_details[bucket_key] = {
            "count": len(data["predictions"]),
            "avg_prediction": avg_pred,
            "actual_rate": avg_outcome,
            "calibration_error": error
        }

        total_error += error * len(data["predictions"])

    mean_calibration_error = total_error / len(predictions) if predictions else 0.0

    return mean_calibration_error, bucket_details


def calculate_roi(
    predictions: List[Tuple[float, float, float]]
) -> float:
    """
    Calculate ROI if betting based on edge.

    Assumes:
    - Bet when our prediction differs from market by > 5%
    - Bet size proportional to edge
    - Standard -110 vig (10% edge needed to break even)

    Args:
        predictions: List of (our_prob, market_prob, outcome) tuples

    Returns:
        ROI as a decimal (0.1 = 10% return)
    """
    if not predictions:
        return 0.0

    total_bet = 0.0
    total_return = 0.0

    for our_prob, market_prob, outcome in predictions:
        edge = our_prob - market_prob

        # Only bet if edge > 5%
        if abs(edge) < 0.05:
            continue

        bet_size = abs(edge) * 10  # Scale bet to edge
        total_bet += bet_size

        # Check if we won
        our_side = "yes" if our_prob > market_prob else "no"
        won = (our_side == "yes" and outcome > 0.5) or (
            our_side == "no" and outcome <= 0.5
        )

        if won:
            # Win: return bet + profit (minus vig)
            if our_side == "yes":
                fair_odds = 1 / our_prob
            else:
                fair_odds = 1 / (1 - our_prob)

            # Apply 10% vig
            actual_odds = fair_odds * 0.9
            total_return += bet_size * actual_odds
        # Loss: lose the bet (total_return stays same)

    if total_bet == 0:
        return 0.0

    return (total_return - total_bet) / total_bet


def score_prediction(
    predicted_probability: float,
    actual_outcome: float,
    market_probability: float
) -> PredictionScore:
    """
    Score a single prediction.

    Args:
        predicted_probability: Our prediction (0-1)
        actual_outcome: Actual outcome (1 or 0)
        market_probability: Market price at prediction time (0-1)

    Returns:
        PredictionScore with all metrics
    """
    our_brier = calculate_brier_score(predicted_probability, actual_outcome)
    market_brier = calculate_brier_score(market_probability, actual_outcome)
    edge = calculate_edge(our_brier, market_brier)

    # Determine calibration bucket
    bucket_idx = min(int(predicted_probability * 10), 9)
    bucket = f"{bucket_idx / 10:.1f}-{(bucket_idx + 1) / 10:.1f}"

    return PredictionScore(
        predicted_probability=predicted_probability,
        actual_outcome=actual_outcome,
        market_probability=market_probability,
        brier_score=our_brier,
        market_brier_score=market_brier,
        edge=edge,
        calibration_bucket=bucket
    )


def calculate_aggregate_scores(
    predictions: List[Tuple[float, float, float]]
) -> AggregateScores:
    """
    Calculate aggregate scores across multiple predictions.

    Args:
        predictions: List of (predicted_prob, market_prob, outcome) tuples

    Returns:
        AggregateScores with all metrics
    """
    if not predictions:
        return AggregateScores(
            total_predictions=0,
            average_brier_score=0.0,
            average_market_brier_score=0.0,
            average_edge=0.0,
            accuracy=0.0,
            calibration_error=0.0,
            log_loss=0.0,
            roi=0.0,
            bucket_scores={}
        )

    brier_scores = []
    market_brier_scores = []
    edges = []
    log_losses = []
    correct = 0

    pred_probs = []
    outcomes = []

    for our_prob, market_prob, outcome in predictions:
        our_brier = calculate_brier_score(our_prob, outcome)
        market_brier = calculate_brier_score(market_prob, outcome)
        edge = calculate_edge(our_brier, market_brier)
        ll = calculate_log_loss(our_prob, outcome)

        brier_scores.append(our_brier)
        market_brier_scores.append(market_brier)
        edges.append(edge)
        log_losses.append(ll)

        pred_probs.append(our_prob)
        outcomes.append(outcome)

        # Check if correct (predicted side matches outcome)
        predicted_yes = our_prob > 0.5
        actual_yes = outcome > 0.5
        if predicted_yes == actual_yes:
            correct += 1

    calibration_error, bucket_details = calculate_calibration_error(
        pred_probs, outcomes
    )
    roi = calculate_roi(predictions)

    return AggregateScores(
        total_predictions=len(predictions),
        average_brier_score=np.mean(brier_scores),
        average_market_brier_score=np.mean(market_brier_scores),
        average_edge=np.mean(edges),
        accuracy=correct / len(predictions),
        calibration_error=calibration_error,
        log_loss=np.mean(log_losses),
        roi=roi,
        bucket_scores=bucket_details
    )


if __name__ == "__main__":
    # Test scoring functions
    print("Testing Scoring Module")
    print("-" * 50)

    # Test single prediction
    score = score_prediction(
        predicted_probability=0.7,
        actual_outcome=1.0,
        market_probability=0.55
    )

    print(f"\nSingle Prediction Score:")
    print(f"  Our prediction: 70%")
    print(f"  Market: 55%")
    print(f"  Outcome: Yes")
    print(f"  Our Brier: {score.brier_score:.4f}")
    print(f"  Market Brier: {score.market_brier_score:.4f}")
    print(f"  Edge: {score.edge:.4f}")

    # Test aggregate scores
    test_predictions = [
        (0.7, 0.5, 1.0),  # We said 70%, market 50%, outcome yes
        (0.3, 0.4, 0.0),  # We said 30%, market 40%, outcome no
        (0.8, 0.6, 1.0),  # We said 80%, market 60%, outcome yes
        (0.6, 0.7, 0.0),  # We said 60%, market 70%, outcome no
    ]

    agg = calculate_aggregate_scores(test_predictions)

    print(f"\nAggregate Scores:")
    print(f"  Total predictions: {agg.total_predictions}")
    print(f"  Avg Brier: {agg.average_brier_score:.4f}")
    print(f"  Avg Market Brier: {agg.average_market_brier_score:.4f}")
    print(f"  Avg Edge: {agg.average_edge:.4f}")
    print(f"  Accuracy: {agg.accuracy:.0%}")
    print(f"  Calibration Error: {agg.calibration_error:.4f}")
    print(f"  Log Loss: {agg.log_loss:.4f}")
    print(f"  ROI: {agg.roi:.1%}")
