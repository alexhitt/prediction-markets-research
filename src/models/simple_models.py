"""
Simple ML Models

Provides basic machine learning models for prediction market forecasting.
Designed to be simple, interpretable, and easy to validate.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML models unavailable.")


@dataclass
class ModelPrediction:
    """Prediction from a model."""
    probability: float
    direction: str  # "yes", "no"
    confidence: float
    features_used: List[str]
    model_name: str


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    brier_score: float
    auc_roc: float
    cv_scores: List[float]


class LogisticPredictor:
    """
    Logistic regression model for binary outcome prediction.

    Simple, interpretable baseline model. Works well with small datasets
    and provides probability outputs suitable for Brier score calculation.

    Usage:
        model = LogisticPredictor()
        model.train(X_train, y_train)
        prediction = model.predict(X_test)
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the logistic predictor.

        Args:
            feature_names: Names of features for interpretability
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML models")

        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = feature_names or []
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Train the model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary outcomes (n_samples,)
            feature_names: Feature names (optional)

        Returns:
            Dict with training metrics
        """
        if feature_names:
            self.feature_names = feature_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)

        # Feature importance (coefficients)
        importance = {}
        if self.feature_names:
            for name, coef in zip(self.feature_names, self.model.coef_[0]):
                importance[name] = float(coef)

        return {
            "cv_accuracy": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "feature_importance": importance
        }

    def predict(self, X: np.ndarray) -> List[ModelPrediction]:
        """
        Make predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            List of ModelPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        predictions = []
        for prob in probabilities:
            direction = "yes" if prob >= 0.5 else "no"
            confidence = abs(prob - 0.5) * 2  # Distance from 0.5

            predictions.append(ModelPrediction(
                probability=float(prob),
                direction=direction,
                confidence=float(confidence),
                features_used=self.feature_names,
                model_name="LogisticPredictor"
            ))

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (coefficients)."""
        if not self.is_trained:
            return {}

        importance = {}
        for name, coef in zip(self.feature_names, self.model.coef_[0]):
            importance[name] = float(coef)

        return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))


class SignalEnsemble:
    """
    Ensemble model that combines multiple signals with learned weights.

    Weighted average of signal predictions, with weights optimized
    based on historical performance.

    Usage:
        ensemble = SignalEnsemble()
        ensemble.add_signal("ice_cream", weight=0.3)
        ensemble.add_signal("google_trends", weight=0.5)
        prediction = ensemble.predict(signal_values)
    """

    def __init__(self):
        """Initialize the ensemble."""
        self.signals: Dict[str, float] = {}  # signal_name -> weight
        self.baseline_weight = 0.5  # Weight for market price

    def add_signal(self, signal_name: str, weight: float = 0.2):
        """
        Add a signal to the ensemble.

        Args:
            signal_name: Name of the signal
            weight: Initial weight (will be normalized)
        """
        self.signals[signal_name] = weight
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1 (minus baseline)."""
        total_signal_weight = 1.0 - self.baseline_weight
        current_total = sum(self.signals.values())

        if current_total > 0:
            for name in self.signals:
                self.signals[name] = (
                    self.signals[name] / current_total * total_signal_weight
                )

    def predict(
        self,
        signal_values: Dict[str, float],
        market_price: float
    ) -> ModelPrediction:
        """
        Make prediction using weighted signal ensemble.

        Args:
            signal_values: Dict of signal_name -> signal_value (0-1 scale)
            market_price: Current market price (0-1)

        Returns:
            ModelPrediction
        """
        # Start with market baseline
        weighted_sum = market_price * self.baseline_weight
        features_used = ["market_price"]

        # Add signal contributions
        for signal_name, weight in self.signals.items():
            if signal_name in signal_values:
                value = signal_values[signal_name]
                weighted_sum += value * weight
                features_used.append(signal_name)

        # Clip to valid probability
        probability = max(0.0, min(1.0, weighted_sum))

        direction = "yes" if probability >= 0.5 else "no"
        confidence = abs(probability - 0.5) * 2

        return ModelPrediction(
            probability=probability,
            direction=direction,
            confidence=confidence,
            features_used=features_used,
            model_name="SignalEnsemble"
        )

    def optimize_weights(
        self,
        historical_data: List[Tuple[Dict[str, float], float, float]]
    ):
        """
        Optimize weights based on historical data.

        Uses simple gradient descent to minimize Brier score.

        Args:
            historical_data: List of (signal_values, market_price, outcome) tuples
        """
        if len(historical_data) < 10:
            logger.warning("Insufficient data for weight optimization")
            return

        # Simple grid search for optimal weights
        best_brier = float('inf')
        best_weights = dict(self.signals)

        for baseline in np.arange(0.3, 0.7, 0.1):
            self.baseline_weight = baseline
            self._normalize_weights()

            # Calculate Brier score
            brier_sum = 0
            for signal_values, market_price, outcome in historical_data:
                pred = self.predict(signal_values, market_price)
                brier_sum += (pred.probability - outcome) ** 2

            brier = brier_sum / len(historical_data)

            if brier < best_brier:
                best_brier = brier
                best_weights = dict(self.signals)
                best_baseline = baseline

        self.baseline_weight = best_baseline
        self.signals = best_weights

        logger.info(f"Optimized ensemble: Brier={best_brier:.4f}")


class TrendDetector:
    """
    Detects price momentum and trend patterns.

    Simple technical analysis model for identifying trending markets.

    Usage:
        detector = TrendDetector(window_size=7)
        trend = detector.analyze([0.45, 0.48, 0.52, 0.55, 0.58])
    """

    def __init__(self, window_size: int = 7):
        """
        Initialize trend detector.

        Args:
            window_size: Number of observations for trend calculation
        """
        self.window_size = window_size

    def analyze(self, prices: List[float]) -> Dict:
        """
        Analyze price series for trends.

        Args:
            prices: List of prices (most recent last)

        Returns:
            Dict with trend analysis
        """
        if len(prices) < 2:
            return {
                "trend": "insufficient_data",
                "momentum": 0,
                "volatility": 0,
                "prediction": None
            }

        prices = np.array(prices[-self.window_size:])

        # Calculate momentum (simple linear regression slope)
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]

        # Normalize slope by average price
        avg_price = np.mean(prices)
        momentum = slope / avg_price if avg_price > 0 else 0

        # Calculate volatility
        volatility = np.std(prices) / avg_price if avg_price > 0 else 0

        # Determine trend
        if momentum > 0.01:
            trend = "bullish"
        elif momentum < -0.01:
            trend = "bearish"
        else:
            trend = "neutral"

        # Simple prediction (extrapolate trend)
        current_price = prices[-1]
        predicted_change = slope * 3  # 3-period ahead
        predicted_price = max(0, min(1, current_price + predicted_change))

        return {
            "trend": trend,
            "momentum": float(momentum),
            "volatility": float(volatility),
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "confidence": max(0, 1 - volatility * 5)  # Lower confidence with high volatility
        }

    def get_prediction(self, prices: List[float]) -> ModelPrediction:
        """
        Get prediction from trend analysis.

        Args:
            prices: List of prices

        Returns:
            ModelPrediction
        """
        analysis = self.analyze(prices)

        if analysis["trend"] == "insufficient_data":
            return ModelPrediction(
                probability=0.5,
                direction="neutral",
                confidence=0.0,
                features_used=["price_history"],
                model_name="TrendDetector"
            )

        return ModelPrediction(
            probability=analysis["predicted_price"],
            direction="yes" if analysis["predicted_price"] >= 0.5 else "no",
            confidence=analysis["confidence"],
            features_used=["momentum", "volatility", "price_history"],
            model_name="TrendDetector"
        )


def create_feature_matrix(
    signals: List[Dict],
    market_data: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create feature matrix from signals and market data.

    Args:
        signals: List of signal dicts with 'value', 'source', 'timestamp'
        market_data: List of market dicts with 'price', 'outcome', 'timestamp'

    Returns:
        Tuple of (X, y, feature_names)
    """
    # Group signals by source
    signal_by_source = {}
    for signal in signals:
        source = signal.get("source", "unknown")
        if source not in signal_by_source:
            signal_by_source[source] = []
        signal_by_source[source].append(signal)

    feature_names = list(signal_by_source.keys()) + ["market_price"]

    X = []
    y = []

    for market in market_data:
        if market.get("outcome") is None:
            continue

        row = []

        # Add signal features
        for source in signal_by_source:
            # Find closest signal by time
            market_time = market.get("timestamp")
            source_signals = signal_by_source[source]

            if source_signals and market_time:
                closest = min(
                    source_signals,
                    key=lambda s: abs(
                        (s.get("timestamp") - market_time).total_seconds()
                        if s.get("timestamp") else float('inf')
                    )
                )
                row.append(closest.get("value", 0))
            else:
                row.append(0)

        # Add market price
        row.append(market.get("price", 0.5))

        X.append(row)
        y.append(1.0 if market.get("outcome") == "yes" else 0.0)

    return np.array(X), np.array(y), feature_names


if __name__ == "__main__":
    print("Testing Simple ML Models")
    print("-" * 50)

    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available")
    else:
        # Test LogisticPredictor
        print("\n1. Testing LogisticPredictor")

        # Generate dummy data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticPredictor(feature_names=["signal_a", "signal_b", "signal_c"])
        metrics = model.train(X, y)

        print(f"   CV Accuracy: {metrics['cv_accuracy']:.2%}")
        print(f"   Feature importance: {metrics['feature_importance']}")

        # Test prediction
        preds = model.predict(X[:5])
        print(f"   Sample predictions: {[p.probability for p in preds]}")

    # Test SignalEnsemble
    print("\n2. Testing SignalEnsemble")
    ensemble = SignalEnsemble()
    ensemble.add_signal("ice_cream", weight=0.3)
    ensemble.add_signal("google_trends", weight=0.2)

    pred = ensemble.predict(
        signal_values={"ice_cream": 0.7, "google_trends": 0.6},
        market_price=0.5
    )
    print(f"   Prediction: {pred.probability:.2%} {pred.direction}")
    print(f"   Confidence: {pred.confidence:.2%}")

    # Test TrendDetector
    print("\n3. Testing TrendDetector")
    detector = TrendDetector()
    prices = [0.45, 0.48, 0.50, 0.55, 0.58, 0.60]
    analysis = detector.analyze(prices)
    print(f"   Trend: {analysis['trend']}")
    print(f"   Momentum: {analysis['momentum']:.4f}")
    print(f"   Predicted: {analysis['predicted_price']:.2%}")
