"""
News Researcher for fast news-based analysis.

Analyzes recent news and events to rapidly assess market implications.
Designed for the News Racer bot that prioritizes speed and news relevance.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger

from .base_researcher import RuleBasedResearcher, ProbabilityEstimate


class NewsResearcher(RuleBasedResearcher):
    """
    News-focused researcher for News Racer bot.

    Methodology:
    - Identify relevant recent news
    - Assess sentiment and implications
    - Weight by recency and source credibility
    - Quick decision-making for time-sensitive opportunities
    """

    @property
    def researcher_type(self) -> str:
        return "news"

    @property
    def methodology_description(self) -> str:
        return """
        News-based rapid analysis:
        1. RELEVANCE: Identify news directly related to market outcome
        2. SENTIMENT: Assess bullish/bearish implications
        3. RECENCY: Weight recent news more heavily
        4. CREDIBILITY: Consider source reliability
        5. SPEED: Prioritize quick, decisive estimates
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
        Use Claude for news-based analysis.
        """
        prompt = f"""Analyze this prediction market from a news and current events perspective:

MARKET QUESTION: {question}
CURRENT MARKET PRICE: {current_price:.0%}
CATEGORY: {category}

{f'CONTEXT: {description}' if description else ''}

As a news-focused analyst, consider:

1. RECENT DEVELOPMENTS
   - What recent news or events are most relevant to this market?
   - Are there any breaking developments that could move this market?
   - What announcements or decisions are expected soon?

2. SENTIMENT ANALYSIS
   - What is the overall news sentiment (bullish/bearish/neutral)?
   - Are major news outlets aligned or divided?
   - Is there momentum in news coverage?

3. INFORMATION EDGE
   - Is the market likely to have fully priced in recent news?
   - Are there events the market may be underweighting?
   - What news could surprise the market?

4. TIMING CONSIDERATIONS
   - How time-sensitive is the current news landscape?
   - Is this a fast-moving story or slow-developing?
   - Should we act quickly or wait for more information?

OUTPUT FORMAT (JSON):
{{
    "relevant_news": [
        {{"headline": "...", "implication": "bullish/bearish", "weight": 0.X}}
    ],
    "overall_sentiment": "bullish/bearish/neutral",
    "sentiment_confidence": 0.XX,
    "information_edge": "description of any edge vs market",
    "urgency": "high/medium/low",
    "final_probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence explanation"
}}
"""

        system_prompt = """You are a news analyst specializing in prediction markets.
You excel at quickly assessing how news events impact market outcomes.
Focus on actionable insights - what does the news mean for YES or NO?
Be decisive but honest about uncertainty when information is limited."""

        response = self._call_claude(prompt, system_prompt)

        if not response:
            return None

        analysis = self._parse_json_from_response(response)

        if not analysis or "final_probability" not in analysis:
            logger.warning(f"Could not parse news analysis for {market_id}")
            return None

        # Adjust confidence based on urgency and sentiment confidence
        base_confidence = float(analysis.get("confidence", 0.5))
        sentiment_confidence = float(analysis.get("sentiment_confidence", 0.5))
        urgency = analysis.get("urgency", "medium")

        # Higher urgency slightly reduces confidence (fast decisions are riskier)
        urgency_adjustment = {"high": -0.1, "medium": 0.0, "low": 0.05}
        adjusted_confidence = base_confidence * sentiment_confidence + urgency_adjustment.get(urgency, 0.0)

        sources = ["news_analysis"]
        for news_item in analysis.get("relevant_news", [])[:3]:
            if news_item.get("headline"):
                sources.append(f"news:{news_item['headline'][:40]}")

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=float(analysis["final_probability"]),
            confidence=self._clamp_confidence(adjusted_confidence),
            reasoning=analysis.get("reasoning", "News-based analysis"),
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
        Rule-based news estimation using keyword analysis.
        """
        question_lower = question.lower()
        description_lower = (description or "").lower()
        combined_text = f"{question_lower} {description_lower}"

        # Detect news-relevant keywords
        bullish_keywords = [
            "will", "expected", "likely", "confirmed", "announced",
            "approved", "passed", "wins", "success", "positive"
        ]
        bearish_keywords = [
            "won't", "unlikely", "rejected", "denied", "fails",
            "loses", "negative", "blocked", "delayed", "cancelled"
        ]

        bullish_count = sum(1 for kw in bullish_keywords if kw in combined_text)
        bearish_count = sum(1 for kw in bearish_keywords if kw in combined_text)

        # Keyword-based adjustment
        keyword_signal = (bullish_count - bearish_count) * 0.03

        # Start with market price
        probability = current_price + keyword_signal

        # Low confidence for rule-based
        confidence = 0.30

        return self._create_estimate(
            market_id=market_id,
            platform=platform,
            question=question,
            current_price=current_price,
            category=category,
            probability=self._clamp_probability(probability),
            confidence=confidence,
            reasoning=f"Rule-based keyword analysis (bullish: {bullish_count}, bearish: {bearish_count})",
            sources=["keyword_analysis"],
            raw_analysis={
                "method": "rule_based",
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "keyword_signal": keyword_signal
            }
        )
