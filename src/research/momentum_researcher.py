"""
Momentum Researcher for technical signal analysis.

Analyzes price movements, volume patterns, and technical indicators
to identify momentum-based trading opportunities.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger

from .base_researcher import RuleBasedResearcher, ProbabilityEstimate


class MomentumResearcher(RuleBasedResearcher):
    """
    Momentum-focused researcher for Aggressive Alpha bot.

    Methodology:
    - Track price momentum (rate of change)
    - Analyze volume patterns
    - Identify breakouts and reversals
    - Use technical indicators adapted for prediction markets
    """

    @property
    def researcher_type(self) -> str:
        return "momentum"

    @property
    def methodology_description(self) -> str:
        return """
        Momentum methodology:
        1. PRICE MOMENTUM: Track rate and direction of price changes
        2. VOLUME ANALYSIS: Confirm moves with volume
        3. BREAKOUT DETECTION: Identify price breaking key levels
        4. TREND STRENGTH: Measure momentum sustainability
        5. REVERSAL SIGNALS: Watch for momentum exhaustion
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
        Use Claude for momentum analysis.
        """
        # Extract momentum data if available
        momentum = extra_data.get("momentum", 0) if extra_data else 0
        volume_spike = extra_data.get("volume_spike", False) if extra_data else False
        price_history = extra_data.get("price_history", []) if extra_data else []

        prompt = f"""Analyze this prediction market from a momentum/technical perspective:

MARKET QUESTION: {question}
CURRENT MARKET PRICE: {current_price:.0%}
CATEGORY: {category}
MOMENTUM SIGNAL: {momentum:+.2f} (positive = upward momentum)
VOLUME SPIKE: {"YES" if volume_spike else "NO"}
{f'RECENT PRICES: {price_history[-10:]}' if price_history else 'No price history available'}

{f'CONTEXT: {description}' if description else ''}

As a momentum analyst, consider:

1. PRICE MOMENTUM
   - What is the recent price trend (last hour, day, week)?
   - Is momentum accelerating or decelerating?
   - Are there signs of trend exhaustion?

2. VOLUME CONFIRMATION
   - Is the price move supported by volume?
   - Are there unusual volume spikes?
   - What does the volume pattern suggest?

3. KEY LEVELS
   - Has price broken through important levels (25%, 50%, 75%)?
   - Are there support/resistance zones?
   - How did price behave at these levels?

4. TREND STRENGTH
   - How strong and sustainable is the current trend?
   - Is this a new trend or continuation?
   - What would invalidate the trend?

5. TIMING
   - Is this a good entry point for momentum?
   - Should we wait for confirmation?
   - What's the risk of a reversal?

OUTPUT FORMAT (JSON):
{{
    "trend_direction": "bullish/bearish/neutral",
    "trend_strength": 0.XX,
    "momentum_phase": "early/mid/late/exhausted",
    "volume_confirmation": true/false,
    "key_levels": {{
        "resistance": [0.XX, 0.XX],
        "support": [0.XX, 0.XX]
    }},
    "breakout_signal": "none/bullish/bearish",
    "final_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence explanation"
}}
"""

        system_prompt = """You are a technical analyst specializing in prediction market momentum.
You understand that momentum strategies work differently in binary markets than in stocks.
Key levels like 25%, 50%, 75% are psychologically significant.
Be aware that prediction markets can be thin - large orders can move prices without fundamental change."""

        response = self._call_claude(prompt, system_prompt)

        if not response:
            return None

        analysis = self._parse_json_from_response(response)

        if not analysis or "final_probability" not in analysis:
            logger.warning(f"Could not parse momentum analysis for {market_id}")
            return None

        # Adjust probability based on momentum if significant
        probability = float(analysis["final_probability"])
        trend_strength = float(analysis.get("trend_strength", 0.5))

        # Incorporate direct momentum data
        if momentum != 0:
            momentum_adjustment = momentum * 0.08 * trend_strength
            probability = self._clamp_probability(probability + momentum_adjustment)

        # Adjust confidence based on volume confirmation
        base_confidence = float(analysis.get("confidence", 0.5))
        if analysis.get("volume_confirmation"):
            base_confidence += 0.1
        else:
            base_confidence -= 0.05

        # Reduce confidence in late/exhausted momentum
        momentum_phase = analysis.get("momentum_phase", "mid")
        phase_adjustment = {
            "early": 0.05,
            "mid": 0.0,
            "late": -0.1,
            "exhausted": -0.2
        }
        adjusted_confidence = base_confidence + phase_adjustment.get(momentum_phase, 0.0)

        sources = ["momentum_analysis"]
        if analysis.get("breakout_signal") != "none":
            sources.append(f"breakout:{analysis['breakout_signal']}")
        if volume_spike:
            sources.append("volume_spike")

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=probability,
            confidence=self._clamp_confidence(adjusted_confidence),
            reasoning=analysis.get("reasoning", "Momentum analysis"),
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
        Rule-based momentum estimation.
        """
        momentum = 0
        volume_spike = False

        if extra_data:
            momentum = extra_data.get("momentum", 0)
            volume_spike = extra_data.get("volume_spike", False)

        # Apply momentum to current price
        # Momentum typically ranges -1 to 1
        momentum_adjustment = momentum * 0.1
        probability = current_price + momentum_adjustment

        # Boost confidence if volume confirms
        confidence = 0.35
        if volume_spike and abs(momentum) > 0.3:
            confidence = 0.50

        # Detect potential breakouts at key levels
        breakout_signal = None
        if current_price > 0.70 and momentum > 0.3:
            breakout_signal = "bullish_breakout"
            probability += 0.05
            confidence += 0.1
        elif current_price < 0.30 and momentum < -0.3:
            breakout_signal = "bearish_breakout"
            probability -= 0.05
            confidence += 0.1

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=self._clamp_probability(probability),
            confidence=self._clamp_confidence(confidence),
            reasoning=f"Rule-based momentum (signal: {momentum:+.2f}, breakout: {breakout_signal or 'none'})",
            sources=["momentum_signal", "volume_data"] if volume_spike else ["momentum_signal"],
            raw_analysis={
                "method": "rule_based",
                "momentum": momentum,
                "volume_spike": volume_spike,
                "breakout_signal": breakout_signal
            }
        )
