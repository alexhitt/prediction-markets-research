"""
Conservative Researcher using Superforecaster methodology.

Implements the Good Judgment Project's approach:
1. Find base rates for similar events
2. Fermi decomposition into sub-questions
3. Gather evidence from multiple sources
4. Incremental updates from base rate

This is the most rigorous methodology, producing well-calibrated estimates.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger

from .base_researcher import RuleBasedResearcher, ProbabilityEstimate


class ConservativeResearcher(RuleBasedResearcher):
    """
    Superforecaster-style researcher for Conservative Value bot.

    Uses systematic probability estimation with:
    - Base rate anchoring
    - Fermi decomposition
    - Multi-source evidence gathering
    - Calibrated confidence levels
    """

    @property
    def researcher_type(self) -> str:
        return "conservative"

    @property
    def methodology_description(self) -> str:
        return """
        Superforecaster methodology:
        1. BASE RATE: What's the historical frequency of similar events?
        2. FERMI DECOMPOSITION: Break into 3-5 sub-questions
        3. EVIDENCE GATHERING: Collect arguments for YES and NO
        4. INCREMENTAL ADJUSTMENT: Update from base rate based on evidence
        5. CALIBRATION: Apply humility adjustments (move toward 50%)
        """

    async def _claude_based_estimate(
        self,
        market_id: str,
        platform: str,
        question: str,
        description: str,
        current_price: float,
        category: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ProbabilityEstimate]:
        """
        Use Claude for superforecaster analysis.
        """
        prompt = f"""Analyze this prediction market using superforecaster methodology:

MARKET QUESTION: {question}
CURRENT MARKET PRICE: {current_price:.0%} (probability of YES)
CATEGORY: {category}

{f'ADDITIONAL CONTEXT: {description}' if description else ''}

Follow this systematic process:

1. BASE RATE ANALYSIS
   - What category of event is this? (election, policy change, economic indicator, etc.)
   - What is the historical frequency of similar events occurring?
   - Identify the reference class and its base rate

2. FERMI DECOMPOSITION
   - Break this question into 3-5 independent sub-questions
   - For each sub-question, estimate its probability
   - Show how they combine

3. EVIDENCE ASSESSMENT
   - List 3-5 factors supporting YES
   - List 3-5 factors supporting NO
   - Assess the strength and reliability of each

4. PROBABILITY SYNTHESIS
   - Start from your base rate
   - Adjust incrementally based on evidence
   - Apply calibration (if uncertain, move toward 50%)

5. CONFIDENCE ASSESSMENT
   - How much relevant information is available?
   - How stable is this estimate likely to be?
   - What could change your mind?

OUTPUT FORMAT (respond in JSON):
{{
    "base_rate": {{
        "reference_class": "description of comparable events",
        "historical_rate": 0.XX,
        "reasoning": "why this base rate applies"
    }},
    "fermi_components": [
        {{"question": "sub-question 1", "probability": 0.XX}},
        {{"question": "sub-question 2", "probability": 0.XX}},
        ...
    ],
    "evidence_yes": ["factor 1", "factor 2", ...],
    "evidence_no": ["factor 1", "factor 2", ...],
    "final_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence summary of key reasoning"
}}
"""

        system_prompt = """You are a superforecaster trained in rigorous probabilistic reasoning.
Your estimates should be well-calibrated: when you say 70%, similar events should happen 70% of the time.
Apply appropriate humility - extreme probabilities (>90% or <10%) require overwhelming evidence.
Always anchor to base rates and adjust incrementally."""

        response = self._call_claude(prompt, system_prompt)

        if not response:
            return None

        analysis = self._parse_json_from_response(response)

        if not analysis or "final_probability" not in analysis:
            logger.warning(f"Could not parse conservative analysis for {market_id}")
            return None

        # Apply calibration adjustment (move extreme estimates toward 50%)
        raw_prob = float(analysis["final_probability"])
        calibrated_prob = self._apply_calibration(raw_prob)

        # Collect sources
        sources = ["superforecaster_methodology", "base_rate_analysis"]
        if analysis.get("base_rate", {}).get("reference_class"):
            sources.append(f"reference_class:{analysis['base_rate']['reference_class'][:50]}")

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=calibrated_prob,
            confidence=float(analysis.get("confidence", 0.5)),
            reasoning=analysis.get("reasoning", "Superforecaster analysis"),
            sources=sources,
            raw_analysis=analysis
        )

    def _rule_based_estimate(
        self,
        market_id: str,
        platform: str,
        question: str,
        description: str,
        current_price: float,
        category: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ProbabilityEstimate]:
        """
        Simple rule-based estimate when Claude unavailable.

        Uses market price as anchor with conservative adjustments.
        """
        # Start with market price as our estimate
        probability = current_price

        # Apply category-specific base rate adjustments
        base_rate_adjustments = {
            "politics": 0.0,  # Market is usually well-informed
            "crypto": -0.05,  # Markets often overconfident on crypto
            "sports": 0.0,
            "entertainment": 0.0,
            "science": 0.0,
            "economics": -0.03,  # Slight skepticism on economic predictions
        }

        adjustment = base_rate_adjustments.get(category.lower(), 0.0)
        probability = probability + adjustment

        # Apply conservative bias (move toward 50%)
        probability = self._apply_calibration(probability)

        # Low confidence for rule-based estimates
        confidence = 0.35

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=probability,
            confidence=confidence,
            reasoning="Rule-based estimate using market price as anchor with conservative adjustments",
            sources=["market_price", "category_base_rate"],
            raw_analysis={
                "method": "rule_based",
                "base_adjustment": adjustment,
                "original_market_price": current_price
            }
        )

    def _apply_calibration(self, probability: float) -> float:
        """
        Apply calibration adjustment to move extreme probabilities toward 50%.

        Superforecaster insight: Most people are overconfident in extreme predictions.
        This adjustment helps improve calibration.
        """
        # Regression toward 50% - the further from 0.5, the more we pull back
        calibration_factor = 0.15  # 15% regression toward mean

        deviation_from_center = probability - 0.5
        adjusted_probability = probability - (deviation_from_center * calibration_factor)

        return self._clamp_probability(adjusted_probability)
