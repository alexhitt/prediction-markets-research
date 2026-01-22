"""
Polymarket API Client for data collection and monitoring.

This client supports:
- Market discovery (Gamma API)
- Price and orderbook data (CLOB API)
- Read-only operations (no authentication required)
- Authenticated trading (optional, requires wallet key)
"""

import asyncio
import aiohttp
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class PolymarketMarket:
    """Polymarket market data structure."""
    id: str
    question: str
    description: str
    category: str
    end_date: Optional[datetime]
    volume: float
    liquidity: float
    outcomes: List[str]
    prices: Dict[str, float]
    status: str
    resolved: bool
    resolution: Optional[str]
    extra_data: Dict[str, Any]


@dataclass
class PolymarketOrderbook:
    """Orderbook snapshot."""
    market_id: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{"price": 0.48, "size": 100}, ...]
    asks: List[Dict[str, float]]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]


class PolymarketClient:
    """
    Client for interacting with Polymarket APIs.

    Usage:
        client = PolymarketClient()

        # Get markets
        markets = client.get_markets(limit=10)

        # Get specific market
        market = client.get_market("0x1234...")

        # Get orderbook
        orderbook = client.get_orderbook("token_id")
    """

    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        private_key: Optional[str] = None
    ):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.private_key = private_key
        self._session: Optional[aiohttp.ClientSession] = None

    # =========================================================================
    # Gamma API (Market Discovery)
    # =========================================================================

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume",
        ascending: bool = False
    ) -> List[PolymarketMarket]:
        """
        Get list of markets from Gamma API.

        Args:
            limit: Maximum number of markets to return
            offset: Pagination offset
            active: Include active markets
            closed: Include closed markets
            order: Sort field (volume, liquidity, end_date)
            ascending: Sort direction

        Returns:
            List of PolymarketMarket objects
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower()
        }

        try:
            response = requests.get(
                f"{self.gamma_url}/markets",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            markets = []
            for item in data:
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            logger.info(f"Fetched {len(markets)} markets from Polymarket")
            return markets

        except requests.RequestException as e:
            logger.error(f"Error fetching Polymarket markets: {e}")
            return []

    def get_market(self, market_id: str) -> Optional[PolymarketMarket]:
        """Get a specific market by ID."""
        try:
            response = requests.get(
                f"{self.gamma_url}/markets/{market_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data)

        except requests.RequestException as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get events (groups of related markets)."""
        try:
            response = requests.get(
                f"{self.gamma_url}/events",
                params={"limit": limit},
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def search_markets(self, query: str, limit: int = 20) -> List[PolymarketMarket]:
        """Search markets by query string."""
        try:
            response = requests.get(
                f"{self.gamma_url}/markets",
                params={"search": query, "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            return [self._parse_market(item) for item in data if item]

        except requests.RequestException as e:
            logger.error(f"Error searching markets: {e}")
            return []

    # =========================================================================
    # CLOB API (Prices & Orderbook)
    # =========================================================================

    def get_orderbook(self, token_id: str) -> Optional[PolymarketOrderbook]:
        """
        Get orderbook for a specific token.

        Args:
            token_id: The token ID (outcome token)

        Returns:
            PolymarketOrderbook with bids, asks, and spread
        """
        try:
            response = requests.get(
                f"{self.clob_url}/book",
                params={"token_id": token_id},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            bids = [{"price": float(b["price"]), "size": float(b["size"])}
                    for b in data.get("bids", [])]
            asks = [{"price": float(a["price"]), "size": float(a["size"])}
                    for a in data.get("asks", [])]

            best_bid = bids[0]["price"] if bids else None
            best_ask = asks[0]["price"] if asks else None
            spread = (best_ask - best_bid) if best_bid and best_ask else None

            return PolymarketOrderbook(
                market_id=token_id,
                timestamp=datetime.utcnow(),
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread
            )

        except requests.RequestException as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
            return None

    def get_price(self, token_id: str) -> Optional[float]:
        """Get current mid price for a token."""
        orderbook = self.get_orderbook(token_id)
        if orderbook and orderbook.best_bid and orderbook.best_ask:
            return (orderbook.best_bid + orderbook.best_ask) / 2
        return None

    def get_prices_batch(self, token_ids: List[str]) -> Dict[str, float]:
        """Get prices for multiple tokens."""
        prices = {}
        for token_id in token_ids:
            price = self.get_price(token_id)
            if price:
                prices[token_id] = price
        return prices

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_market(self, data: Dict[str, Any]) -> Optional[PolymarketMarket]:
        """Parse API response into PolymarketMarket object."""
        try:
            # Extract prices from outcomes/tokens
            prices = {}
            outcomes = []

            if "outcomes" in data:
                outcomes = data["outcomes"]
            elif "tokens" in data:
                for token in data.get("tokens", []):
                    outcome = token.get("outcome", "")
                    outcomes.append(outcome)
                    if "price" in token:
                        prices[outcome] = float(token["price"])

            # Parse end date
            end_date = None
            if data.get("end_date_iso"):
                try:
                    end_date = datetime.fromisoformat(
                        data["end_date_iso"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            return PolymarketMarket(
                id=data.get("id", data.get("condition_id", "")),
                question=data.get("question", ""),
                description=data.get("description", ""),
                category=data.get("category", ""),
                end_date=end_date,
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                outcomes=outcomes,
                prices=prices,
                status="closed" if data.get("closed") else "open",
                resolved=data.get("resolved", False),
                resolution=data.get("resolution"),
                extra_data={
                    "slug": data.get("slug"),
                    "image": data.get("image"),
                    "icon": data.get("icon"),
                    "tokens": data.get("tokens", [])
                }
            )

        except Exception as e:
            logger.warning(f"Error parsing market data: {e}")
            return None

    # =========================================================================
    # Async Methods (for high-frequency collection)
    # =========================================================================

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_markets_async(self, limit: int = 100) -> List[PolymarketMarket]:
        """Async version of get_markets."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.gamma_url}/markets",
                params={"limit": limit}
            ) as response:
                data = await response.json()
                return [self._parse_market(item) for item in data if item]
        except Exception as e:
            logger.error(f"Async error fetching markets: {e}")
            return []

    async def get_orderbook_async(self, token_id: str) -> Optional[PolymarketOrderbook]:
        """Async version of get_orderbook."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.clob_url}/book",
                params={"token_id": token_id}
            ) as response:
                data = await response.json()

                bids = [{"price": float(b["price"]), "size": float(b["size"])}
                        for b in data.get("bids", [])]
                asks = [{"price": float(a["price"]), "size": float(a["size"])}
                        for a in data.get("asks", [])]

                best_bid = bids[0]["price"] if bids else None
                best_ask = asks[0]["price"] if asks else None
                spread = (best_ask - best_bid) if best_bid and best_ask else None

                return PolymarketOrderbook(
                    market_id=token_id,
                    timestamp=datetime.utcnow(),
                    bids=bids,
                    asks=asks,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=spread
                )
        except Exception as e:
            logger.error(f"Async error fetching orderbook: {e}")
            return None

    async def close(self):
        """Close async session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Convenience function for quick access
def get_polymarket_client() -> PolymarketClient:
    """Get a configured Polymarket client."""
    from config.settings import settings
    return PolymarketClient(
        gamma_url=settings.polymarket_gamma_url,
        clob_url=settings.polymarket_clob_url,
        private_key=settings.polymarket_private_key
    )


if __name__ == "__main__":
    # Test the client
    client = PolymarketClient()

    print("Fetching top markets by volume...")
    markets = client.get_markets(limit=5, order="volume")

    for market in markets:
        print(f"\n{market.question}")
        print(f"  Volume: ${market.volume:,.0f}")
        print(f"  Category: {market.category}")
        print(f"  Status: {market.status}")
        if market.prices:
            print(f"  Prices: {market.prices}")
