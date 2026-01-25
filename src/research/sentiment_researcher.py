"""
Sentiment Researcher for social media aggregation.

Analyzes Twitter, Reddit, and other social platforms to gauge
public sentiment and predict market outcomes.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger

from .base_researcher import RuleBasedResearcher, ProbabilityEstimate


class SentimentResearcher(RuleBasedResearcher):
    """
    Sentiment-focused researcher for Sentiment Surfer bot.

    Methodology:
    - Aggregate sentiment from multiple social platforms
    - Weight by engagement and follower counts
    - Track sentiment momentum (changing vs stable)
    - Identify influencer opinions
    """

    @property
    def researcher_type(self) -> str:
        return "sentiment"

    @property
    def methodology_description(self) -> str:
        return """
        Social sentiment methodology:
        1. AGGREGATION: Collect sentiment from Twitter, Reddit, news comments
        2. WEIGHTING: Weight by engagement, followers, credibility
        3. MOMENTUM: Track if sentiment is changing or stable
        4. INFLUENCERS: Identify key opinion leaders
        5. NOISE FILTERING: Distinguish signal from noise/bots
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
        Use Claude for sentiment analysis.
        """
        # Extract sentiment data if available
        sentiment_score = extra_data.get("sentiment_score", 0) if extra_data else 0
        news_sentiment = extra_data.get("news_sentiment", 0) if extra_data else 0

        prompt = f"""Analyze this prediction market from a social sentiment perspective:

MARKET QUESTION: {question}
CURRENT MARKET PRICE: {current_price:.0%}
CATEGORY: {category}
KNOWN SENTIMENT SCORE: {sentiment_score:+.2f} (positive = bullish sentiment)
NEWS SENTIMENT: {news_sentiment:+.2f}

{f'CONTEXT: {description}' if description else ''}

As a sentiment analyst, consider:

1. PUBLIC OPINION
   - What is the general public sentiment on this topic?
   - Are social media discussions mostly bullish or bearish?
   - Is there consensus or division?

2. PLATFORM ANALYSIS
   - Twitter: What are influencers and general users saying?
   - Reddit: What's the discussion in relevant subreddits?
   - News comments: What's the reader sentiment?

3. SENTIMENT MOMENTUM
   - Is sentiment improving or declining?
   - Are there recent events shifting opinion?
   - Is the sentiment stable or volatile?

4. SIGNAL QUALITY
   - How much of the sentiment is genuine vs bots/manipulation?
   - Are engaged users (high followers, verified) aligned?
   - Is there evidence of coordinated sentiment campaigns?

5. MARKET EFFICIENCY
   - Is the current market price reflecting sentiment?
   - Could sentiment be a leading or lagging indicator here?
   - When does sentiment matter vs fundamentals?

OUTPUT FORMAT (JSON):
{{
    "overall_sentiment": "bullish/bearish/neutral",
    "sentiment_strength": 0.XX,
    "sentiment_momentum": "improving/declining/stable",
    "platform_breakdown": {{
        "twitter": "bullish/bearish/neutral",
        "reddit": "bullish/bearish/neutral",
        "news": "bullish/bearish/neutral"
    }},
    "key_narratives": ["narrative 1", "narrative 2"],
    "noise_level": "high/medium/low",
    "final_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence explanation"
}}
"""

        system_prompt = """You are a social sentiment analyst for prediction markets.
You excel at reading the crowd while filtering out noise and manipulation.
Remember: sentiment can be wrong - crowds make mistakes.
Be especially skeptical when sentiment is extreme or unanimous."""

        response = self._call_claude(prompt, system_prompt)

        if not response:
            return None

        analysis = self._parse_json_from_response(response)

        if not analysis or "final_probability" not in analysis:
            logger.warning(f"Could not parse sentiment analysis for {market_id}")
            return None

        # Adjust confidence based on noise level
        base_confidence = float(analysis.get("confidence", 0.5))
        noise_level = analysis.get("noise_level", "medium")
        noise_adjustment = {"high": -0.15, "medium": 0.0, "low": 0.1}
        adjusted_confidence = base_confidence + noise_adjustment.get(noise_level, 0.0)

        # Incorporate direct sentiment data if available
        probability = float(analysis["final_probability"])
        if sentiment_score != 0:
            sentiment_adjustment = sentiment_score * 0.05
            probability = self._clamp_probability(probability + sentiment_adjustment)

        sources = ["sentiment_analysis"]
        for narrative in analysis.get("key_narratives", [])[:2]:
            sources.append(f"narrative:{narrative[:30]}")

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=probability,
            confidence=self._clamp_confidence(adjusted_confidence),
            reasoning=analysis.get("reasoning", "Sentiment analysis"),
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
        Rule-based sentiment estimation.
        """
        sentiment_score = 0
        news_sentiment = 0

        if extra_data:
            sentiment_score = extra_data.get("sentiment_score", 0)
            news_sentiment = extra_data.get("news_sentiment", 0)

        # Combine sentiment signals
        combined_sentiment = (sentiment_score * 0.6 + news_sentiment * 0.4)

        # Apply to market price
        adjustment = combined_sentiment * 0.08
        probability = current_price + adjustment

        # Confidence based on strength of sentiment
        confidence = 0.30 + abs(combined_sentiment) * 0.15

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=self._clamp_probability(probability),
            confidence=self._clamp_confidence(confidence),
            reasoning=f"Rule-based sentiment (score: {combined_sentiment:+.2f})",
            sources=["sentiment_data"],
            raw_analysis={
                "method": "rule_based",
                "sentiment_score": sentiment_score,
                "news_sentiment": news_sentiment,
                "combined": combined_sentiment
            }
        )
