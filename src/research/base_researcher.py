"""
Base researcher class for probability estimation.

Defines the interface that all specialized researchers must implement.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

from loguru import logger

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("anthropic library not installed - Claude research disabled")


@dataclass(frozen=True)
class ProbabilityEstimate:
    """
    Immutable probability estimate from a research agent.

    Contains the estimated probability along with confidence,
    reasoning, and metadata about the research process.
    """
    market_id: str
    platform: str
    estimated_probability: float  # 0-1 scale
    confidence: float  # 0-1 scale
    reasoning: str  # Human-readable explanation
    researcher_type: str  # e.g., "conservative", "news", "whale"

    # Research metadata
    sources_used: List[str]
    raw_analysis: Dict[str, Any]

    # Context
    market_question: str
    market_price_at_estimate: float
    category: str
    timestamp: datetime

    def edge(self) -> float:
        """Calculate edge vs market price."""
        return abs(self.estimated_probability - self.market_price_at_estimate)

    def direction(self) -> str:
        """Return 'yes' if estimate > market, 'no' otherwise."""
        if self.estimated_probability > self.market_price_at_estimate:
            return "yes"
        return "no"

    def should_bet(self, min_edge: float = 0.05, min_confidence: float = 0.5) -> bool:
        """Determine if this estimate warrants a bet."""
        return self.edge() >= min_edge and self.confidence >= min_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "market_id": self.market_id,
            "platform": self.platform,
            "estimated_probability": self.estimated_probability,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "researcher_type": self.researcher_type,
            "sources_used": self.sources_used,
            "raw_analysis": self.raw_analysis,
            "market_question": self.market_question,
            "market_price_at_estimate": self.market_price_at_estimate,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "edge": self.edge(),
            "direction": self.direction()
        }


class BaseResearcher(ABC):
    """
    Abstract base class for research agents.

    Each researcher implements a distinct methodology for
    estimating probabilities. The Claude API is used for
    complex analysis tasks.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        temperature: float = 0.3
    ):
        """
        Initialize the researcher.

        Args:
            model: Claude model to use for analysis
            max_tokens: Maximum tokens for Claude response
            temperature: Temperature for Claude (lower = more deterministic)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client = None
        if HAS_ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)

    @property
    @abstractmethod
    def researcher_type(self) -> str:
        """Return the type identifier for this researcher."""
        pass

    @property
    @abstractmethod
    def methodology_description(self) -> str:
        """Return a description of the research methodology."""
        pass

    @abstractmethod
    async def research_market(
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
        Research a market and produce a probability estimate.

        Args:
            market_id: Platform-specific market identifier
            platform: polymarket or kalshi
            question: The market question
            description: Additional market description
            current_price: Current YES price (0-1)
            category: Market category
            extra_data: Additional market data

        Returns:
            ProbabilityEstimate if research is successful, None otherwise
        """
        pass

    def _call_claude(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Call Claude API for analysis.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Claude's response text, or None if error
        """
        if not self._client:
            logger.warning("Claude client not configured")
            return None

        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = self._client.messages.create(**kwargs)

            if response.content:
                return response.content[0].text

            return None

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None

    def _parse_json_from_response(
        self,
        response: str,
        default: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract JSON from Claude's response.

        Handles responses that include markdown code blocks.
        """
        import json
        import re

        if not response:
            return default or {}

        # Try to extract JSON from code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return default or {}

    def _clamp_probability(self, prob: float) -> float:
        """Clamp probability to valid range."""
        return max(0.01, min(0.99, prob))

    def _clamp_confidence(self, conf: float) -> float:
        """Clamp confidence to valid range."""
        return max(0.0, min(1.0, conf))

    def _create_estimate(
        self,
        market_id: str,
        platform: str,
        question: str,
        current_price: float,
        category: str,
        probability: float,
        confidence: float,
        reasoning: str,
        sources: List[str],
        raw_analysis: Dict[str, Any]
    ) -> ProbabilityEstimate:
        """
        Create a ProbabilityEstimate with validation.

        Helper method to ensure all estimates are properly formatted.
        """
        return ProbabilityEstimate(
            market_id=market_id,
            platform=platform,
            estimated_probability=self._clamp_probability(probability),
            confidence=self._clamp_confidence(confidence),
            reasoning=reasoning,
            researcher_type=self.researcher_type,
            sources_used=sources,
            raw_analysis=raw_analysis,
            market_question=question,
            market_price_at_estimate=current_price,
            category=category,
            timestamp=datetime.utcnow()
        )


class RuleBasedResearcher(BaseResearcher):
    """
    Base class for researchers that can fall back to rule-based heuristics.

    When Claude API is unavailable or for cost efficiency, these researchers
    can produce estimates using simpler rule-based logic.
    """

    @abstractmethod
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
        Produce an estimate using rule-based heuristics.

        Called when Claude API is unavailable.
        """
        pass

    async def research_market(
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
        Research market, falling back to rules if Claude unavailable.
        """
        # Try Claude-based research first
        if self._client:
            try:
                estimate = await self._claude_based_estimate(
                    market_id, platform, question, description,
                    current_price, category, extra_data
                )
                if estimate:
                    return estimate
            except Exception as e:
                logger.warning(f"Claude research failed, falling back to rules: {e}")

        # Fall back to rule-based
        return self._rule_based_estimate(
            market_id, platform, question, description,
            current_price, category, extra_data
        )

    @abstractmethod
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
        Produce an estimate using Claude API.
        """
        pass
