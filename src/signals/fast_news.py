"""
Fast News Signal Detector

Ultra-low latency news ingestion for prediction market edge.
Prioritizes speed over comprehensiveness - get news before the market prices it in.

Sources (ordered by speed):
1. RSS feeds from major outlets (1-5 min latency)
2. Reddit /new endpoints (near real-time)
3. Direct site monitoring (configurable)

Design principles from HFT blueprint:
- Async I/O for parallel fetching
- Signal decay model (news value degrades over time)
- Pre-computed keyword matching for instant market mapping
"""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from loguru import logger

from src.signals.base import (
    BaseSignalDetector,
    SignalDirection,
    SignalResult,
    SignalStrength,
)


# Fast RSS feeds - sorted by typical update frequency
FAST_RSS_FEEDS = {
    # Breaking news (updates within minutes)
    "reuters_top": "https://feeds.reuters.com/reuters/topNews",
    "reuters_world": "https://feeds.reuters.com/Reuters/worldNews",
    "ap_top": "https://rsshub.app/apnews/topics/apf-topnews",
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",

    # Politics (prediction market relevant)
    "politico": "https://rss.politico.com/politics-news.xml",
    "hill": "https://thehill.com/feed/",

    # Finance/Economics
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",

    # Crypto
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",

    # Tech
    "techcrunch": "https://techcrunch.com/feed/",
    "verge": "https://www.theverge.com/rss/index.xml",
}

# Subreddits for real-time social signals
FAST_SUBREDDITS = [
    "news",
    "worldnews",
    "politics",
    "economics",
    "bitcoin",
    "CryptoCurrency",
]

# Keyword -> Market category mapping for instant classification
KEYWORD_MARKET_MAP = {
    # Politics
    "trump": ["politics", "election"],
    "biden": ["politics", "election"],
    "congress": ["politics", "legislation"],
    "senate": ["politics", "legislation"],
    "supreme court": ["politics", "judicial"],
    "election": ["politics", "election"],
    "impeach": ["politics", "election"],
    "indictment": ["politics", "legal"],

    # Crypto
    "bitcoin": ["crypto", "btc"],
    "ethereum": ["crypto", "eth"],
    "sec crypto": ["crypto", "regulation"],
    "crypto regulation": ["crypto", "regulation"],
    "coinbase": ["crypto", "exchange"],
    "binance": ["crypto", "exchange"],
    "stablecoin": ["crypto", "regulation"],

    # Economics
    "federal reserve": ["economics", "rates"],
    "interest rate": ["economics", "rates"],
    "inflation": ["economics", "inflation"],
    "recession": ["economics", "recession"],
    "unemployment": ["economics", "jobs"],
    "gdp": ["economics", "growth"],
    "cpi": ["economics", "inflation"],

    # Geopolitics
    "russia": ["geopolitics", "russia"],
    "ukraine": ["geopolitics", "ukraine"],
    "china": ["geopolitics", "china"],
    "taiwan": ["geopolitics", "taiwan"],
    "nato": ["geopolitics", "nato"],
    "war": ["geopolitics", "conflict"],
    "nuclear": ["geopolitics", "nuclear"],
    "sanctions": ["geopolitics", "sanctions"],

    # AI/Tech
    "openai": ["tech", "ai"],
    "chatgpt": ["tech", "ai"],
    "artificial intelligence": ["tech", "ai"],
    "ai regulation": ["tech", "ai"],
    "antitrust": ["tech", "regulation"],
}

# Pre-compile regex patterns for speed
KEYWORD_PATTERNS = {
    kw: re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in KEYWORD_MARKET_MAP.keys()
}


@dataclass
class NewsItem:
    """Single news item with metadata."""
    id: str  # Hash of title for dedup
    title: str
    source: str
    url: str
    published: datetime
    fetched_at: datetime
    categories: List[str] = field(default_factory=list)
    keywords_matched: List[str] = field(default_factory=list)
    sentiment_hint: float = 0.0  # Quick sentiment from title keywords

    @property
    def age_seconds(self) -> float:
        """Time since publication in seconds."""
        return (datetime.utcnow() - self.published).total_seconds()

    @property
    def freshness_score(self) -> float:
        """
        Signal decay model - news value degrades over time.

        Returns 1.0 for brand new, decays to 0.1 over 1 hour.
        Based on HFT "Decay Models" concept from the blueprint.
        """
        age_minutes = self.age_seconds / 60

        if age_minutes < 5:
            return 1.0  # Maximum value for very fresh news
        elif age_minutes < 15:
            return 0.9 - (age_minutes - 5) * 0.03  # 0.9 -> 0.6
        elif age_minutes < 30:
            return 0.6 - (age_minutes - 15) * 0.02  # 0.6 -> 0.3
        elif age_minutes < 60:
            return 0.3 - (age_minutes - 30) * 0.007  # 0.3 -> 0.1
        else:
            return 0.1  # Stale but still has some value


class FastNewsSignalDetector(BaseSignalDetector):
    """
    Ultra-fast news ingestion for prediction market alpha.

    Key features:
    - Parallel fetching from multiple sources
    - Signal decay model (freshness matters)
    - Pre-compiled keyword matching
    - Deduplication via content hashing
    - Continuous monitoring mode available

    Usage:
        detector = FastNewsSignalDetector()
        signals = detector.run()  # One-shot

        # Or continuous monitoring:
        detector.start_monitoring(callback=handle_signal)
    """

    def __init__(
        self,
        feeds: Optional[Dict[str, str]] = None,
        subreddits: Optional[List[str]] = None,
        max_age_minutes: int = 30,
        min_freshness: float = 0.3
    ):
        """
        Initialize fast news detector.

        Args:
            feeds: RSS feed name -> URL mapping
            subreddits: List of subreddits to monitor
            max_age_minutes: Ignore news older than this
            min_freshness: Minimum freshness score to generate signal
        """
        super().__init__(
            name="Fast News Signal",
            source="multi_source_realtime"
        )

        self.feeds = feeds or FAST_RSS_FEEDS
        self.subreddits = subreddits or FAST_SUBREDDITS
        self.max_age_minutes = max_age_minutes
        self.min_freshness = min_freshness

        # Deduplication cache
        self._seen_ids: Set[str] = set()
        self._seen_ids_max = 10000

        # Monitoring state
        self._monitoring = False
        self._monitor_callback: Optional[Callable] = None

    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch news from all sources in parallel.

        Uses ThreadPoolExecutor for I/O parallelism.
        """
        start_time = time.time()
        all_items: List[NewsItem] = []

        # Parallel fetch with thread pool
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}

            # Submit RSS feed fetches
            for name, url in self.feeds.items():
                futures[executor.submit(self._fetch_rss, name, url)] = f"rss:{name}"

            # Submit Reddit fetches
            for subreddit in self.subreddits:
                futures[executor.submit(self._fetch_reddit_new, subreddit)] = f"reddit:{subreddit}"

            # Collect results
            for future in as_completed(futures, timeout=15):
                source = futures[future]
                try:
                    items = future.result()
                    all_items.extend(items)
                except Exception as e:
                    logger.debug(f"Fetch failed for {source}: {e}")

        # Deduplicate
        unique_items = self._deduplicate(all_items)

        # Filter by age
        cutoff = datetime.utcnow() - timedelta(minutes=self.max_age_minutes)
        fresh_items = [item for item in unique_items if item.published > cutoff]

        # Sort by freshness (newest first)
        fresh_items.sort(key=lambda x: x.published, reverse=True)

        fetch_time = time.time() - start_time
        logger.info(
            f"Fetched {len(fresh_items)} fresh items from {len(self.feeds) + len(self.subreddits)} "
            f"sources in {fetch_time:.2f}s"
        )

        return {
            "items": fresh_items,
            "fetch_time_ms": fetch_time * 1000,
            "sources_count": len(self.feeds) + len(self.subreddits),
            "total_fetched": len(all_items),
            "after_dedup": len(unique_items),
            "after_age_filter": len(fresh_items)
        }

    def _fetch_rss(self, name: str, url: str) -> List[NewsItem]:
        """Fetch and parse a single RSS feed."""
        items = []

        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:20]:  # Top 20 per feed
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Parse publication time
                published = self._parse_published_time(entry)
                if not published:
                    published = datetime.utcnow()  # Assume fresh if no date

                # Generate ID
                item_id = hashlib.md5(title.encode()).hexdigest()[:12]

                # Quick keyword matching
                keywords, categories = self._match_keywords(title)

                # Quick sentiment from title
                sentiment = self._quick_sentiment(title)

                items.append(NewsItem(
                    id=item_id,
                    title=title,
                    source=f"rss:{name}",
                    url=entry.get("link", ""),
                    published=published,
                    fetched_at=datetime.utcnow(),
                    categories=categories,
                    keywords_matched=keywords,
                    sentiment_hint=sentiment
                ))

        except Exception as e:
            logger.debug(f"RSS fetch error for {name}: {e}")

        return items

    def _fetch_reddit_new(self, subreddit: str) -> List[NewsItem]:
        """Fetch newest posts from a subreddit."""
        items = []

        try:
            response = requests.get(
                f"https://www.reddit.com/r/{subreddit}/new.json",
                headers={"User-Agent": "PredictionBot/1.0"},
                params={"limit": 25},
                timeout=10
            )

            if response.status_code != 200:
                return items

            data = response.json()
            posts = data.get("data", {}).get("children", [])

            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "").strip()

                if not title:
                    continue

                # Reddit uses Unix timestamps
                created_utc = post_data.get("created_utc", 0)
                published = datetime.utcfromtimestamp(created_utc) if created_utc else datetime.utcnow()

                item_id = hashlib.md5(title.encode()).hexdigest()[:12]
                keywords, categories = self._match_keywords(title)
                sentiment = self._quick_sentiment(title)

                # Boost score for highly upvoted posts
                score = post_data.get("score", 0)
                if score > 100:
                    sentiment *= 1.2  # Viral content gets sentiment boost

                items.append(NewsItem(
                    id=item_id,
                    title=title,
                    source=f"reddit:{subreddit}",
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    published=published,
                    fetched_at=datetime.utcnow(),
                    categories=categories,
                    keywords_matched=keywords,
                    sentiment_hint=sentiment
                ))

        except Exception as e:
            logger.debug(f"Reddit fetch error for {subreddit}: {e}")

        return items

    def _parse_published_time(self, entry: Any) -> Optional[datetime]:
        """Parse publication time from RSS entry."""
        for field in ["published_parsed", "updated_parsed", "created_parsed"]:
            parsed = getattr(entry, field, None)
            if parsed:
                try:
                    return datetime(*parsed[:6])
                except (TypeError, ValueError):
                    continue
        return None

    def _match_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Fast keyword matching using pre-compiled patterns.

        Returns (matched_keywords, market_categories)
        """
        keywords = []
        categories = set()

        text_lower = text.lower()

        for keyword, pattern in KEYWORD_PATTERNS.items():
            if pattern.search(text_lower):
                keywords.append(keyword)
                categories.update(KEYWORD_MARKET_MAP[keyword])

        return keywords, list(categories)

    def _quick_sentiment(self, title: str) -> float:
        """
        Ultra-fast sentiment estimation from title keywords.

        Not as accurate as NLP but instant - trade accuracy for speed.
        """
        title_lower = title.lower()

        # Negative indicators
        negative_words = [
            "crash", "plunge", "fall", "drop", "decline", "crisis", "fear",
            "warning", "threat", "attack", "fail", "reject", "ban", "halt",
            "investigation", "lawsuit", "scandal", "fraud", "collapse"
        ]

        # Positive indicators
        positive_words = [
            "surge", "soar", "rise", "gain", "rally", "boost", "approve",
            "pass", "win", "success", "breakthrough", "record", "growth",
            "deal", "agreement", "partnership", "launch", "expand"
        ]

        neg_count = sum(1 for w in negative_words if w in title_lower)
        pos_count = sum(1 for w in positive_words if w in title_lower)

        if neg_count + pos_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count + 1)

    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate items based on ID."""
        unique = []

        for item in items:
            if item.id not in self._seen_ids:
                self._seen_ids.add(item.id)
                unique.append(item)

        # Prune cache if too large
        if len(self._seen_ids) > self._seen_ids_max:
            self._seen_ids = set(list(self._seen_ids)[-5000:])

        return unique

    def process_data(self, raw_data: Dict[str, Any]) -> List[SignalResult]:
        """
        Process news items into trading signals.

        Applies decay model and aggregates by category.
        """
        signals = []
        timestamp = datetime.utcnow()

        items = raw_data.get("items", [])
        if not items:
            return signals

        # Group by category
        category_items: Dict[str, List[NewsItem]] = {}
        for item in items:
            for category in item.categories:
                if category not in category_items:
                    category_items[category] = []
                category_items[category].append(item)

        # Generate signals per category
        for category, cat_items in category_items.items():
            signal = self._generate_category_signal(category, cat_items, timestamp)
            if signal:
                signals.append(signal)

        # Generate breaking news signals for very fresh items
        for item in items[:10]:  # Top 10 freshest
            if item.freshness_score >= 0.8 and item.keywords_matched:
                breaking_signal = self._generate_breaking_signal(item, timestamp)
                if breaking_signal:
                    signals.append(breaking_signal)

        logger.info(f"Generated {len(signals)} fast news signals from {len(items)} items")
        return signals

    def _generate_category_signal(
        self,
        category: str,
        items: List[NewsItem],
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Generate aggregated signal for a category."""
        if len(items) < 2:
            return None

        # Weight by freshness
        weighted_sentiment = sum(
            item.sentiment_hint * item.freshness_score
            for item in items
        )
        total_freshness = sum(item.freshness_score for item in items)

        if total_freshness == 0:
            return None

        avg_sentiment = weighted_sentiment / total_freshness
        avg_freshness = total_freshness / len(items)

        # Skip if not fresh enough
        if avg_freshness < self.min_freshness:
            return None

        # Determine direction
        if avg_sentiment > 0.15:
            direction = SignalDirection.BULLISH
        elif avg_sentiment < -0.15:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        # Confidence based on volume and freshness
        confidence = min(0.75, 0.3 + len(items) * 0.05 + avg_freshness * 0.3)

        # Strength based on sentiment magnitude
        if abs(avg_sentiment) > 0.5:
            strength = SignalStrength.VERY_STRONG
        elif abs(avg_sentiment) > 0.3:
            strength = SignalStrength.STRONG
        elif abs(avg_sentiment) > 0.15:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return SignalResult(
            name=f"Fast News: {category}",
            source=self.source,
            timestamp=timestamp,
            value=avg_sentiment,
            direction=direction,
            confidence=confidence,
            strength=strength,
            raw_data={
                "category": category,
                "item_count": len(items),
                "avg_sentiment": avg_sentiment,
                "avg_freshness": avg_freshness,
                "top_keywords": list(set(
                    kw for item in items for kw in item.keywords_matched
                ))[:5],
                "newest_item_age_sec": min(item.age_seconds for item in items)
            },
            related_markets=[category],
            metadata={
                "signal_type": "fast_news_aggregate",
                "latency_sensitive": True
            }
        )

    def _generate_breaking_signal(
        self,
        item: NewsItem,
        timestamp: datetime
    ) -> Optional[SignalResult]:
        """Generate signal for a single breaking news item."""
        if item.freshness_score < 0.8:
            return None

        # Determine direction from sentiment
        if item.sentiment_hint > 0.2:
            direction = SignalDirection.BULLISH
        elif item.sentiment_hint < -0.2:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        # Higher confidence for fresher news
        confidence = min(0.85, 0.5 + item.freshness_score * 0.3)

        return SignalResult(
            name=f"BREAKING: {item.title[:50]}...",
            source=self.source,
            timestamp=timestamp,
            value=item.sentiment_hint,
            direction=direction,
            confidence=confidence,
            strength=SignalStrength.STRONG,
            raw_data={
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "keywords": item.keywords_matched,
                "categories": item.categories,
                "age_seconds": item.age_seconds,
                "freshness": item.freshness_score
            },
            related_markets=item.categories,
            metadata={
                "signal_type": "breaking_news",
                "latency_sensitive": True,
                "urgent": item.freshness_score > 0.95
            }
        )

    def start_monitoring(
        self,
        callback: Callable[[SignalResult], None],
        interval_seconds: int = 30
    ):
        """
        Start continuous monitoring mode.

        Calls callback with new signals every interval.
        """
        self._monitoring = True
        self._monitor_callback = callback

        logger.info(f"Starting fast news monitoring (interval: {interval_seconds}s)")

        while self._monitoring:
            try:
                signals = self.run()
                for signal in signals:
                    if self._monitor_callback:
                        self._monitor_callback(signal)

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring = False
        logger.info("Stopped fast news monitoring")


def get_fast_news_detector() -> FastNewsSignalDetector:
    """Get a configured fast news detector."""
    return FastNewsSignalDetector()


if __name__ == "__main__":
    # Test the detector
    detector = FastNewsSignalDetector()

    print("Fetching fast news...")
    start = time.time()
    signals = detector.run()
    elapsed = time.time() - start

    print(f"\nFetched in {elapsed:.2f}s")
    print(f"Generated {len(signals)} signals:\n")

    for signal in signals[:10]:
        print(f"{signal.name}")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Freshness: {signal.raw_data.get('avg_freshness', signal.raw_data.get('freshness', 'N/A'))}")
        print()
