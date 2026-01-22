"""
Kalshi API Client for data collection and monitoring.

This client supports:
- Market data (public, no auth required)
- Orderbook and prices
- Authenticated portfolio operations (optional)
"""

import time
import base64
import hashlib
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography not installed - Kalshi auth disabled")


@dataclass
class KalshiMarket:
    """Kalshi market data structure."""
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    category: str
    status: str  # open, closed, settled
    yes_bid: Optional[int]  # In cents (0-99)
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    volume: int
    open_interest: int
    close_time: Optional[datetime]
    result: Optional[str]  # yes, no, null
    extra_data: Dict[str, Any]


@dataclass
class KalshiOrderbook:
    """Kalshi orderbook snapshot."""
    ticker: str
    timestamp: datetime
    yes_bids: List[Dict[str, Any]]  # [{"price": 65, "count": 100}, ...]
    no_bids: List[Dict[str, Any]]
    best_yes_bid: Optional[int]
    best_yes_ask: Optional[int]
    spread: Optional[int]


class KalshiClient:
    """
    Client for interacting with Kalshi API.

    Usage:
        # Without auth (public data only)
        client = KalshiClient()
        markets = client.get_markets(limit=10)

        # With auth (portfolio access)
        client = KalshiClient(api_key="...", private_key="...")
        balance = client.get_balance()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key: Optional[str] = None,
        use_demo: bool = False
    ):
        self.api_key = api_key
        self.private_key = private_key
        self.use_demo = use_demo

        if use_demo:
            self.base_url = "https://demo-api.kalshi.com/trade-api/v2"
        else:
            self.base_url = "https://api.kalshi.com/trade-api/v2"

    # =========================================================================
    # Authentication
    # =========================================================================

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Sign a request with RSA private key."""
        if not HAS_CRYPTO or not self.private_key:
            raise ValueError("Cryptography library and private key required for signing")

        message = f"{timestamp}{method}{path}{body}"

        # Load private key
        if self.private_key.startswith("-----"):
            # Inline PEM
            key_bytes = self.private_key.encode()
        else:
            # File path
            with open(self.private_key, "rb") as f:
                key_bytes = f.read()

        private_key = serialization.load_pem_private_key(key_bytes, password=None)

        signature = private_key.sign(
            message.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode()

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Get headers for authenticated request."""
        headers = {"Content-Type": "application/json"}

        if self.api_key and self.private_key:
            timestamp = str(int(time.time()))
            signature = self._sign_request(timestamp, method, path, body)

            headers.update({
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp
            })

        return headers

    # =========================================================================
    # Public Market Data
    # =========================================================================

    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None
    ) -> List[KalshiMarket]:
        """
        Get list of markets.

        Args:
            limit: Maximum markets to return (max 200)
            cursor: Pagination cursor
            status: Filter by status (open, closed, settled)
            series_ticker: Filter by series
            event_ticker: Filter by event

        Returns:
            List of KalshiMarket objects
        """
        params = {
            "limit": min(limit, 200),
            "status": status
        }
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker

        try:
            response = requests.get(
                f"{self.base_url}/markets",
                params=params,
                headers=self._get_headers("GET", "/trade-api/v2/markets"),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            markets = []
            for item in data.get("markets", []):
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            logger.info(f"Fetched {len(markets)} markets from Kalshi")
            return markets

        except requests.RequestException as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
            return []

    def get_market(self, ticker: str) -> Optional[KalshiMarket]:
        """Get a specific market by ticker."""
        try:
            response = requests.get(
                f"{self.base_url}/markets/{ticker}",
                headers=self._get_headers("GET", f"/trade-api/v2/markets/{ticker}"),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data.get("market", {}))

        except requests.RequestException as e:
            logger.error(f"Error fetching market {ticker}: {e}")
            return None

    def get_orderbook(self, ticker: str) -> Optional[KalshiOrderbook]:
        """
        Get orderbook for a market.

        Args:
            ticker: Market ticker

        Returns:
            KalshiOrderbook with bids and spread
        """
        try:
            response = requests.get(
                f"{self.base_url}/markets/{ticker}/orderbook",
                headers=self._get_headers("GET", f"/trade-api/v2/markets/{ticker}/orderbook"),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            orderbook = data.get("orderbook", {})
            yes_bids = orderbook.get("yes", [])
            no_bids = orderbook.get("no", [])

            # Best prices
            best_yes_bid = yes_bids[0][0] if yes_bids else None  # [price, count]
            best_no_bid = no_bids[0][0] if no_bids else None
            best_yes_ask = 100 - best_no_bid if best_no_bid else None

            spread = (best_yes_ask - best_yes_bid) if best_yes_bid and best_yes_ask else None

            return KalshiOrderbook(
                ticker=ticker,
                timestamp=datetime.utcnow(),
                yes_bids=[{"price": b[0], "count": b[1]} for b in yes_bids],
                no_bids=[{"price": b[0], "count": b[1]} for b in no_bids],
                best_yes_bid=best_yes_bid,
                best_yes_ask=best_yes_ask,
                spread=spread
            )

        except requests.RequestException as e:
            logger.error(f"Error fetching orderbook for {ticker}: {e}")
            return None

    def get_events(self, limit: int = 50, status: str = "open") -> List[Dict[str, Any]]:
        """Get events (groups of related markets)."""
        try:
            response = requests.get(
                f"{self.base_url}/events",
                params={"limit": limit, "status": status},
                headers=self._get_headers("GET", "/trade-api/v2/events"),
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("events", [])

        except requests.RequestException as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def get_series(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get market series."""
        try:
            response = requests.get(
                f"{self.base_url}/series",
                params={"limit": limit},
                headers=self._get_headers("GET", "/trade-api/v2/series"),
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("series", [])

        except requests.RequestException as e:
            logger.error(f"Error fetching series: {e}")
            return []

    def get_trades(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a market."""
        try:
            response = requests.get(
                f"{self.base_url}/markets/{ticker}/trades",
                params={"limit": limit},
                headers=self._get_headers("GET", f"/trade-api/v2/markets/{ticker}/trades"),
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("trades", [])

        except requests.RequestException as e:
            logger.error(f"Error fetching trades for {ticker}: {e}")
            return []

    # =========================================================================
    # Authenticated Portfolio Operations
    # =========================================================================

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance (requires auth)."""
        if not self.api_key or not self.private_key:
            logger.warning("Auth required for balance")
            return None

        try:
            response = requests.get(
                f"{self.base_url}/portfolio/balance",
                headers=self._get_headers("GET", "/trade-api/v2/portfolio/balance"),
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Error fetching balance: {e}")
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions (requires auth)."""
        if not self.api_key or not self.private_key:
            logger.warning("Auth required for positions")
            return []

        try:
            response = requests.get(
                f"{self.base_url}/portfolio/positions",
                headers=self._get_headers("GET", "/trade-api/v2/portfolio/positions"),
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("market_positions", [])

        except requests.RequestException as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_market(self, data: Dict[str, Any]) -> Optional[KalshiMarket]:
        """Parse API response into KalshiMarket object."""
        try:
            # Parse close time
            close_time = None
            if data.get("close_time"):
                try:
                    close_time = datetime.fromisoformat(
                        data["close_time"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            return KalshiMarket(
                ticker=data.get("ticker", ""),
                event_ticker=data.get("event_ticker", ""),
                title=data.get("title", ""),
                subtitle=data.get("subtitle", ""),
                category=data.get("category", ""),
                status=data.get("status", ""),
                yes_bid=data.get("yes_bid"),
                yes_ask=data.get("yes_ask"),
                no_bid=data.get("no_bid"),
                no_ask=data.get("no_ask"),
                volume=data.get("volume", 0),
                open_interest=data.get("open_interest", 0),
                close_time=close_time,
                result=data.get("result"),
                extra_data={
                    "settlement_value": data.get("settlement_value"),
                    "cap_strike": data.get("cap_strike"),
                    "floor_strike": data.get("floor_strike"),
                    "rules_primary": data.get("rules_primary"),
                    "rules_secondary": data.get("rules_secondary")
                }
            )

        except Exception as e:
            logger.warning(f"Error parsing Kalshi market data: {e}")
            return None

    def get_mid_price(self, market: KalshiMarket) -> Optional[float]:
        """Calculate mid price for a market (0-1 scale)."""
        if market.yes_bid and market.yes_ask:
            return (market.yes_bid + market.yes_ask) / 200  # Convert from cents to 0-1
        return None


# Convenience function for quick access
def get_kalshi_client() -> KalshiClient:
    """Get a configured Kalshi client."""
    from config.settings import settings
    return KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key=settings.kalshi_private_key,
        use_demo=settings.kalshi_use_demo
    )


if __name__ == "__main__":
    # Test the client
    client = KalshiClient()

    print("Fetching markets from Kalshi...")
    markets = client.get_markets(limit=5)

    for market in markets:
        print(f"\n{market.title}")
        print(f"  Ticker: {market.ticker}")
        print(f"  Status: {market.status}")
        print(f"  Volume: {market.volume}")
        if market.yes_bid and market.yes_ask:
            print(f"  Yes: {market.yes_bid}¢ / {market.yes_ask}¢")
