# Kalshi API Reference

## Overview

Kalshi is the first CFTC-regulated prediction market in the United States. The API provides REST and WebSocket interfaces for trading binary event contracts.

---

## API Endpoints

**Base URL:** `https://api.kalshi.com/trade-api/v2`
**Demo URL:** `https://demo-api.kalshi.com/trade-api/v2`

---

## Authentication

Kalshi uses RSA-based API key authentication.

### Generating API Keys
1. Log into Kalshi dashboard
2. Navigate to API settings
3. Generate or upload RSA public key
4. Store private key securely

### Request Signing
```python
import base64
import hashlib
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

def sign_request(private_key_pem, timestamp, method, path, body=""):
    message = f"{timestamp}{method}{path}{body}"
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None
    )
    signature = private_key.sign(
        message.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode()
```

### Headers
```
KALSHI-ACCESS-KEY: your_api_key
KALSHI-ACCESS-SIGNATURE: signature
KALSHI-ACCESS-TIMESTAMP: unix_timestamp
Content-Type: application/json
```

---

## Core Endpoints

### Exchange Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/exchange/status` | GET | Exchange status |
| `/exchange/announcements` | GET | Platform announcements |
| `/exchange/schedule` | GET | Trading schedule |

### Portfolio Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/balance` | GET | Account balance (in cents) |
| `/portfolio/positions` | GET | Current positions |
| `/portfolio/settlements` | GET | Settlement history |
| `/portfolio/fills` | GET | Completed trades |

### Order Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/orders` | GET | List orders |
| `/portfolio/orders` | POST | Create order |
| `/portfolio/orders/{id}` | GET | Get order |
| `/portfolio/orders/{id}` | DELETE | Cancel order |
| `/portfolio/orders/{id}/amend` | POST | Modify order |
| `/portfolio/orders/batched` | POST | Batch orders (up to 20) |

### Market Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/markets` | GET | List all markets |
| `/markets/{ticker}` | GET | Get market details |
| `/markets/{ticker}/orderbook` | GET | Get orderbook |
| `/markets/{ticker}/trades` | GET | Recent trades |
| `/markets/{ticker}/candlesticks` | GET | OHLC data |
| `/series` | GET | List market series |
| `/events` | GET | List events |

---

## WebSocket Streaming

**URL:** `wss://api.kalshi.com/trade-api/ws/v2`

### Channels

| Channel | Description |
|---------|-------------|
| `orderbook` | Orderbook updates |
| `ticker` | Market ticker |
| `trade` | Public trades |
| `fill` | Your fills |
| `position` | Position updates |
| `market_lifecycle` | Market status changes |

### Subscription Example
```python
import asyncio
import websockets
import json

async def subscribe():
    uri = "wss://api.kalshi.com/trade-api/ws/v2"
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp
    }

    async with websockets.connect(uri, extra_headers=headers) as ws:
        # Subscribe to orderbook
        await ws.send(json.dumps({
            "type": "subscribe",
            "channels": ["orderbook"],
            "market_tickers": ["INXD-25JAN21"]
        }))

        async for message in ws:
            data = json.loads(message)
            print(data)

asyncio.run(subscribe())
```

---

## Python SDK Example

```python
import requests
import time

class KalshiClient:
    def __init__(self, api_key, private_key):
        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.api_key = api_key
        self.private_key = private_key

    def _headers(self, method, path, body=""):
        timestamp = str(int(time.time()))
        signature = sign_request(
            self.private_key, timestamp, method, path, body
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }

    def get_markets(self, limit=100, status="open"):
        path = f"/markets?limit={limit}&status={status}"
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers("GET", path)
        )
        return response.json()

    def get_orderbook(self, ticker):
        path = f"/markets/{ticker}/orderbook"
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers("GET", path)
        )
        return response.json()

    def get_balance(self):
        path = "/portfolio/balance"
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers("GET", path)
        )
        return response.json()  # Balance in cents

    def create_order(self, ticker, side, price, count):
        path = "/portfolio/orders"
        body = json.dumps({
            "ticker": ticker,
            "side": side,  # "yes" or "no"
            "type": "limit",
            "action": "buy",
            "count": count,
            "yes_price": price if side == "yes" else None,
            "no_price": price if side == "no" else None
        })
        response = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers("POST", path, body),
            data=body
        )
        return response.json()

    def cancel_order(self, order_id):
        path = f"/portfolio/orders/{order_id}"
        response = requests.delete(
            f"{self.base_url}{path}",
            headers=self._headers("DELETE", path)
        )
        return response.json()

# Usage
client = KalshiClient(API_KEY, PRIVATE_KEY)
markets = client.get_markets(limit=10)
balance = client.get_balance()
```

---

## Data Models

### Market
```json
{
    "ticker": "INXD-25JAN21-T4000",
    "event_ticker": "INXD-25JAN21",
    "title": "S&P 500 above 4000 on Jan 21",
    "subtitle": "Will close above 4000?",
    "status": "open",
    "yes_bid": 65,
    "yes_ask": 67,
    "no_bid": 33,
    "no_ask": 35,
    "volume": 15000,
    "open_interest": 8500,
    "close_time": "2025-01-21T21:00:00Z"
}
```

### Orderbook
```json
{
    "ticker": "INXD-25JAN21-T4000",
    "yes_bids": [
        {"price": 65, "count": 100},
        {"price": 64, "count": 250}
    ],
    "no_bids": [
        {"price": 33, "count": 150},
        {"price": 32, "count": 300}
    ]
}
```

### Order
```json
{
    "order_id": "abc123",
    "ticker": "INXD-25JAN21-T4000",
    "side": "yes",
    "type": "limit",
    "action": "buy",
    "status": "resting",
    "yes_price": 65,
    "count": 100,
    "remaining_count": 100,
    "created_time": "2025-01-21T12:00:00Z"
}
```

### Position
```json
{
    "ticker": "INXD-25JAN21-T4000",
    "position": 100,
    "market_exposure": 6500,
    "realized_pnl": 0,
    "fees_paid": 65
}
```

---

## Candlestick Intervals

| Interval | Parameter |
|----------|-----------|
| 1 minute | `1m` |
| 1 hour | `1h` |
| 1 day | `1d` |

```python
def get_candlesticks(ticker, interval="1h", limit=100):
    path = f"/markets/{ticker}/candlesticks?interval={interval}&limit={limit}"
    # ...
```

---

## Batch Operations

### Batch Order Creation (up to 20)
```python
def batch_create_orders(orders):
    path = "/portfolio/orders/batched"
    body = json.dumps({"orders": orders})
    # ...

orders = [
    {"ticker": "MARKET1", "side": "yes", "yes_price": 50, "count": 10},
    {"ticker": "MARKET2", "side": "no", "no_price": 40, "count": 20}
]
batch_create_orders(orders)
```

---

## Subaccounts

Kalshi supports multiple subaccounts for portfolio segmentation.

```python
# Create subaccount
def create_subaccount(name):
    path = "/portfolio/subaccounts"
    body = json.dumps({"name": name})
    # ...

# Transfer between accounts
def transfer(from_account, to_account, amount):
    path = "/portfolio/subaccounts/transfer"
    body = json.dumps({
        "from": from_account,
        "to": to_account,
        "amount": amount  # In cents
    })
    # ...
```

---

## Fee Structure

| Fee Type | Amount |
|----------|--------|
| Deposit (ACH) | Free |
| Deposit (Wire) | Free |
| Withdrawal | $2.00 |
| Trading Fee | ~1% on expected earnings |
| Maker Fee | Variable (some markets) |

---

## Rate Limits

- Rate limits vary by access tier
- Premier and Market Maker tiers have higher limits
- Check response headers for limit status

---

## Resources

- [Official Documentation](https://docs.kalshi.com/welcome)
- [Fee Schedule](https://kalshi.com/fee-schedule)
- [Developer Discord](https://discord.gg/kalshi)
- [Demo Environment](https://demo-api.kalshi.com/)
