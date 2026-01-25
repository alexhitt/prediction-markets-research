"""
Contrarian Researcher for fading consensus.

Looks for opportunities where the crowd is overconfident and
identifies potential dark horse outcomes that the market undervalues.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger

from .base_researcher import RuleBasedResearcher, ProbabilityEstimate


class ContrarianResearcher(RuleBasedResearcher):
    """
    Contrarian researcher for Contrarian Carl bot.

    Methodology:
    - Identify markets with strong consensus (>75% or <25%)
    - Look for reasons the consensus could be wrong
    - Find dark horse scenarios
    - Bet against overconfident crowds
    """

    @property
    def researcher_type(self) -> str:
        return "contrarian"

    @property
    def methodology_description(self) -> str:
        return """
        Contrarian methodology:
        1. CONSENSUS DETECTION: Find markets with >75% agreement
        2. OVERCONFIDENCE CHECK: Is the consensus justified?
        3. DARK HORSE SEARCH: What scenarios could upset consensus?
        4. HISTORICAL PATTERNS: How often do favorites fail?
        5. SELECTIVE FADING: Only fade when there's a real case
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
        Use Claude for contrarian analysis.
        """
        # Determine if market has strong consensus
        consensus_side = "yes" if current_price > 0.5 else "no"
        consensus_strength = max(current_price, 1 - current_price)
        has_strong_consensus = consensus_strength > 0.75

        prompt = f"""Analyze this prediction market from a contrarian perspective:

MARKET QUESTION: {question}
CURRENT MARKET PRICE: {current_price:.0%} (YES probability)
CATEGORY: {category}
CONSENSUS SIDE: {consensus_side.upper()} at {consensus_strength:.0%}
HAS STRONG CONSENSUS: {"YES - Look for fade opportunity" if has_strong_consensus else "NO - Weaker contrarian signal"}

{f'CONTEXT: {description}' if description else ''}

As a contrarian analyst, systematically consider:

1. CONSENSUS ANALYSIS
   - Why is the market so confident in {consensus_side.upper()}?
   - What would need to happen for the consensus to be wrong?
   - Are there any blind spots in the consensus view?

2. DARK HORSE SCENARIOS
   - What low-probability events could upset the consensus?
   - Are any of these scenarios underpriced?
   - Historical examples of similar consensus being wrong?

3. OVERCONFIDENCE INDICATORS
   - Is there evidence of herding behavior?
   - Are there echo chambers reinforcing the consensus?
   - What information might the crowd be missing?

4. FADE QUALITY ASSESSMENT
   - Is this a good contrarian opportunity or just going against smart money?
   - What's the risk/reward of fading this consensus?
   - Should we fade aggressively or modestly?

5. TIMING
   - Is this the right time to be contrarian?
   - Could the consensus strengthen further before any reversal?

OUTPUT FORMAT (JSON):
{{
    "consensus_quality": "well-founded/somewhat-justified/overconfident/clearly-wrong",
    "fade_recommendation": "strong-fade/mild-fade/no-fade/follow-consensus",
    "dark_horse_scenarios": [
        {{"scenario": "...", "probability": 0.XX, "underpriced_by": 0.XX}}
    ],
    "overconfidence_evidence": ["evidence 1", "evidence 2"],
    "reasons_consensus_could_be_right": ["reason 1", "reason 2"],
    "final_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence explanation"
}}
"""

        system_prompt = """You are a contrarian analyst who specializes in finding overconfident markets.
You understand that being contrarian just for the sake of it loses money.
You only fade consensus when there's a genuine case for the market being wrong.
Most markets with strong consensus ARE correct - you're looking for the exceptions."""

        response = self._call_claude(prompt, system_prompt)

        if not response:
            return None

        analysis = self._parse_json_from_response(response)

        if not analysis or "final_probability" not in analysis:
            logger.warning(f"Could not parse contrarian analysis for {market_id}")
            return None

        # Adjust confidence based on fade quality
        base_confidence = float(analysis.get("confidence", 0.5))
        fade_rec = analysis.get("fade_recommendation", "no-fade")

        # Strong fades get higher confidence, no-fades get lower
        fade_confidence_adjustment = {
            "strong-fade": 0.1,
            "mild-fade": 0.0,
            "no-fade": -0.1,
            "follow-consensus": -0.15
        }
        adjusted_confidence = base_confidence + fade_confidence_adjustment.get(fade_rec, 0.0)

        sources = ["contrarian_analysis"]
        for scenario in analysis.get("dark_horse_scenarios", [])[:2]:
            if scenario.get("scenario"):
                sources.append(f"dark_horse:{scenario['scenario'][:30]}")

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=float(analysis["final_probability"]),
            confidence=self._clamp_confidence(adjusted_confidence),
            reasoning=analysis.get("reasoning", "Contrarian analysis"),
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
        Rule-based contrarian estimation.

        Simple rule: fade extreme prices, follow moderate prices.
        """
        # Calculate how extreme the current price is
        extremeness = abs(current_price - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%

        if extremeness > 0.5:  # Price is >75% or <25%
            # Apply contrarian fade - move probability toward 50%
            fade_amount = (extremeness - 0.5) * 0.15  # Max 7.5% fade
            if current_price > 0.5:
                probability = current_price - fade_amount
            else:
                probability = current_price + fade_amount
            confidence = 0.35 + extremeness * 0.1  # More confident with more extreme prices
            reasoning = f"Rule-based contrarian fade (extremeness: {extremeness:.2f})"
        else:
            # Not extreme enough to fade - stick with market
            probability = current_price
            confidence = 0.25
            reasoning = "No contrarian signal - price not extreme enough"

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=self._clamp_probability(probability),
            confidence=self._clamp_confidence(confidence),
            reasoning=reasoning,
            sources=["contrarian_rule"],
            raw_analysis={
                "method": "rule_based",
                "extremeness": extremeness,
                "fade_applied": extremeness > 0.5
            }
        )
