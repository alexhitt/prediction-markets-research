"""
Whale Researcher for tracking proven traders.

Analyzes whale wallet activity and copies traders with proven track records.
Focuses on skilled traders (high win rate over time) not just lucky ones.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger

from .base_researcher import RuleBasedResearcher, ProbabilityEstimate


class WhaleResearcher(RuleBasedResearcher):
    """
    Whale-tracking researcher for Whale Watcher bot.

    Methodology:
    - Track large wallet positions (>$10k)
    - Filter for proven track record (6+ months, 60%+ win rate)
    - Weight by position size and historical accuracy
    - Distinguish skill from luck using statistical tests
    """

    @property
    def researcher_type(self) -> str:
        return "whale"

    @property
    def methodology_description(self) -> str:
        return """
        Whale tracking methodology:
        1. POSITION TRACKING: Identify large positions on this market
        2. WALLET SCORING: Rate wallets by historical performance
        3. SKILL FILTERING: Only follow wallets with statistically significant edge
        4. AGGREGATION: Combine whale signals weighted by credibility
        5. TIMING: Consider when whales entered vs current price
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
        Use Claude for whale-based analysis.
        """
        # Extract any whale data from extra_data
        whale_data = extra_data.get("whale_activity", []) if extra_data else []
        whale_bias = extra_data.get("whale_bias", 0) if extra_data else 0

        prompt = f"""Analyze this prediction market from a whale-tracking perspective:

MARKET QUESTION: {question}
CURRENT MARKET PRICE: {current_price:.0%}
CATEGORY: {category}
KNOWN WHALE BIAS: {whale_bias:+.2f} (positive = whales favor YES)

{f'CONTEXT: {description}' if description else ''}
{f'WHALE ACTIVITY DATA: {whale_data}' if whale_data else 'No specific whale activity data available.'}

As a whale-tracking analyst, consider:

1. WHALE POSITION ANALYSIS
   - Are smart money traders (high historical win rate) positioned in this market?
   - What side are they favoring (YES or NO)?
   - What is their average entry price vs current price?

2. CREDIBILITY ASSESSMENT
   - Which whales have proven track records in this category?
   - How do we distinguish skill from luck?
   - Should we weight certain wallets more heavily?

3. CONTRARY SIGNALS
   - Are there smart money traders on both sides?
   - Could whale activity be misleading (manipulation, hedging)?
   - What would make us ignore the whale signal?

4. TIMING AND SIZE
   - Are whales accumulating or distributing?
   - Is the position size meaningful relative to market liquidity?
   - How recent is the whale activity?

OUTPUT FORMAT (JSON):
{{
    "whale_signal": "bullish/bearish/neutral",
    "signal_strength": 0.XX,
    "credibility_weighted_bias": 0.XX,
    "key_whales": [
        {{"wallet": "...", "side": "yes/no", "credibility": 0.X, "size": "$XXk"}}
    ],
    "concerns": ["list any concerns about following whales"],
    "final_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence explanation"
}}
"""

        system_prompt = """You are a whale-tracking analyst who follows smart money in prediction markets.
You understand that most large traders are NOT skilled - you only follow wallets with proven,
statistically significant track records.
Be skeptical of whale signals that could be manipulation, hedging, or just luck."""

        response = self._call_claude(prompt, system_prompt)

        if not response:
            return None

        analysis = self._parse_json_from_response(response)

        if not analysis or "final_probability" not in analysis:
            logger.warning(f"Could not parse whale analysis for {market_id}")
            return None

        # Adjust probability based on whale bias if significant
        probability = float(analysis["final_probability"])
        signal_strength = float(analysis.get("signal_strength", 0.5))

        # If we have direct whale bias data, incorporate it
        if whale_bias != 0:
            whale_adjustment = whale_bias * 0.1 * signal_strength
            probability = self._clamp_probability(probability + whale_adjustment)

        sources = ["whale_tracking"]
        for whale in analysis.get("key_whales", [])[:3]:
            if whale.get("wallet"):
                sources.append(f"whale:{whale['wallet'][:10]}...")

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=probability,
            confidence=float(analysis.get("confidence", 0.5)),
            reasoning=analysis.get("reasoning", "Whale tracking analysis"),
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
        Rule-based whale estimation using any provided whale data.
        """
        whale_bias = 0
        whale_data = []

        if extra_data:
            whale_bias = extra_data.get("whale_bias", 0)
            whale_data = extra_data.get("whale_activity", [])

        # Apply whale bias if we have it
        if whale_bias != 0:
            # Whale bias ranges from -1 to 1
            # Apply it as adjustment to market price
            adjustment = whale_bias * 0.08  # Max 8% adjustment
            probability = current_price + adjustment
            confidence = 0.40 + abs(whale_bias) * 0.2  # Higher confidence with stronger signal
        else:
            # No whale data - use market price
            probability = current_price
            confidence = 0.25

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=self._clamp_probability(probability),
            confidence=self._clamp_confidence(confidence),
            reasoning=f"Rule-based whale tracking (bias: {whale_bias:+.2f})",
            sources=["whale_data" if whale_bias != 0 else "no_whale_data"],
            raw_analysis={
                "method": "rule_based",
                "whale_bias": whale_bias,
                "whale_count": len(whale_data)
            }
        )
