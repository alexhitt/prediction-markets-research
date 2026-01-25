"""
Research agents module for real paper trading.

Provides specialized research agents for each bot type:
- ConservativeResearcher: Superforecaster methodology
- NewsResearcher: Real-time news analysis
- WhaleResearcher: Track proven traders
- SentimentResearcher: Social aggregation
- ContrarianResearcher: Fade consensus
- MomentumResearcher: Technical signals
"""

from .base_researcher import BaseResearcher, ProbabilityEstimate
from .conservative_researcher import ConservativeResearcher
from .news_researcher import NewsResearcher
from .whale_researcher import WhaleResearcher
from .sentiment_researcher import SentimentResearcher
from .contrarian_researcher import ContrarianResearcher
from .momentum_researcher import MomentumResearcher

# Registry mapping bot strategy types to researchers
RESEARCHER_REGISTRY = {
    "conservative_value": ConservativeResearcher,
    "value": ConservativeResearcher,
    "news_racer": NewsResearcher,
    "news": NewsResearcher,
    "whale_watcher": WhaleResearcher,
    "whale": WhaleResearcher,
    "sentiment_surfer": SentimentResearcher,
    "sentiment": SentimentResearcher,
    "contrarian_carl": ContrarianResearcher,
    "contrarian": ContrarianResearcher,
    "aggressive_alpha": MomentumResearcher,
    "momentum": MomentumResearcher,
    "balanced": ConservativeResearcher,  # Default to superforecaster method
}


def get_researcher_for_bot(strategy_type: str, **kwargs) -> BaseResearcher:
    """
    Get the appropriate researcher for a bot's strategy type.

    Args:
        strategy_type: The bot's strategy type (e.g., "conservative_value")
        **kwargs: Additional arguments passed to researcher constructor

    Returns:
        Researcher instance appropriate for the strategy
    """
    researcher_class = RESEARCHER_REGISTRY.get(
        strategy_type.lower(),
        ConservativeResearcher  # Default
    )
    return researcher_class(**kwargs)


__all__ = [
    "BaseResearcher",
    "ProbabilityEstimate",
    "ConservativeResearcher",
    "NewsResearcher",
    "WhaleResearcher",
    "SentimentResearcher",
    "ContrarianResearcher",
    "MomentumResearcher",
    "get_researcher_for_bot",
    "RESEARCHER_REGISTRY",
]
