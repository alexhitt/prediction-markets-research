# Polymarket API Reference

## Overview

Polymarket provides a suite of APIs for building prediction market applications. The system is hybrid-decentralized: off-chain order matching with on-chain settlement on Polygon.

---

## API Endpoints

### 1. Gamma API (Market Discovery)
**Base URL:** `https://gamma-api.polymarket.com`

**Purpose:** Fetch events, markets, categories, and resolution data.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events` | GET | List all events |
| `/events/{id}` | GET | Get specific event |
| `/markets` | GET | List all markets |
| `/markets/{id}` | GET | Get specific market |
| `/categories` | GET | List categories |

### 2. CLOB API (Trading)
**Base URL:** `https://clob.polymarket.com`

**Purpose:** Real-time prices, orderbook depth, order placement.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orderbook/{market_id}` | GET | Get orderbook |
| `/prices/{market_id}` | GET | Get current prices |
| `/orders` | POST | Place order |
| `/orders/{id}` | DELETE | Cancel order |
| `/orders` | GET | List your orders |

### 3. Data API (Portfolio)
**Base URL:** `https://data-api.polymarket.com`

**Purpose:** User positions, trade history, portfolio data.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/positions` | GET | Get your positions |
| `/trades` | GET | Trade history |
| `/portfolio` | GET | Portfolio summary |

### 4. WebSocket (Real-Time)
**URL:** `wss://ws-subscriptions-clob.polymarket.com`

**Channels:**
- `orderbook` - Orderbook updates
- `price` - Price changes
- `trades` - New trades
- `orders` - Your order status

---

## SDK Installation

### Python
```bash
pip install py-clob-client
```

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs

# Initialize client
client = ClobClient(
    host="https://clob.polymarket.com",
    key=PRIVATE_KEY,
    chain_id=137  # Polygon
)

# Get markets
markets = client.get_markets()

# Get orderbook
orderbook = client.get_orderbook("market_id")

# Place order
order = client.create_order(
    OrderArgs(
        token_id="token_id",
        price=0.50,
        size=100,
        side="BUY"
    )
)
```

### TypeScript
```bash
npm install @polymarket/clob-client
```

```typescript
import { ClobClient } from "@polymarket/clob-client";

const client = new ClobClient(
  "https://clob.polymarket.com",
  137, // Polygon chain ID
  wallet
);

// Get markets
const markets = await client.getMarkets();

// Get orderbook
const book = await client.getOrderBook("market_id");

// Place order
const order = await client.createOrder({
  tokenId: "token_id",
  price: 0.5,
  size: 100,
  side: "BUY"
});
```

---

## Rate Limits

| Tier | Rate Limit | Cost |
|------|------------|------|
| Free | 1,000 calls/hour | $0 |
| Premium | Higher limits + WebSocket | $99/month |
| Enterprise | Custom | Contact sales |

---

## Authentication

Polymarket uses wallet-based authentication with EIP-712 signatures.

```python
from eth_account import Account
from eth_account.messages import encode_defunct

# Sign message
message = "Login to Polymarket"
signed = Account.sign_message(
    encode_defunct(text=message),
    private_key=PRIVATE_KEY
)
```

---

## Polymarket Agents Framework

### Installation
```bash
git clone https://github.com/Polymarket/agents.git
cd agents
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Setup
```bash
# .env file
POLYGON_WALLET_PRIVATE_KEY="your_key"
OPENAI_API_KEY="your_key"
```

### Architecture Components

| Module | Purpose |
|--------|---------|
| `chroma.py` | Vector database for news/API data indexing |
| `gamma.py` | Polymarket Gamma API interface |
| `polymarket.py` | Order execution and DEX interactions |
| `objects.py` | Pydantic data models |
| `cli.py` | Command-line interface |

### CLI Commands
```bash
# Get markets
python scripts/python/cli.py get-all-markets --limit 5 --sort-by volume

# Execute trade logic
python agents/application/trade.py
```

### Features
- RAG (Retrieval-Augmented Generation) for intelligent decisions
- News and web search integration
- LLM prompt engineering utilities
- Automated order execution

---

## Data Models

### Market
```python
{
    "id": "0x1234...",
    "question": "Will X happen?",
    "description": "...",
    "outcomes": ["Yes", "No"],
    "volume": 1000000,
    "liquidity": 500000,
    "end_date": "2025-12-31T00:00:00Z",
    "resolved": false,
    "resolution": null
}
```

### Orderbook
```python
{
    "market_id": "0x1234...",
    "bids": [
        {"price": 0.48, "size": 1000},
        {"price": 0.47, "size": 2000}
    ],
    "asks": [
        {"price": 0.52, "size": 1500},
        {"price": 0.53, "size": 3000}
    ]
}
```

### Order
```python
{
    "id": "order_123",
    "market_id": "0x1234...",
    "side": "BUY",
    "price": 0.50,
    "size": 100,
    "filled": 50,
    "status": "OPEN",
    "created_at": "2025-01-21T12:00:00Z"
}
```

---

## Useful Queries

### Get High Volume Markets
```python
markets = client.get_markets(
    sort_by="volume",
    order="desc",
    limit=20
)
```

### Get Markets by Category
```python
political = client.get_markets(category="politics")
crypto = client.get_markets(category="crypto")
sports = client.get_markets(category="sports")
```

### Monitor Price Changes
```python
import asyncio
import websockets
import json

async def monitor_prices(market_id):
    uri = "wss://ws-subscriptions-clob.polymarket.com"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "subscribe",
            "channel": "price",
            "market_id": market_id
        }))
        async for message in ws:
            data = json.loads(message)
            print(f"Price update: {data}")

asyncio.run(monitor_prices("market_id"))
```

---

## Resources

- [Official Documentation](https://docs.polymarket.com/)
- [GitHub - Agents Framework](https://github.com/Polymarket/agents)
- [Python SDK](https://pypi.org/project/py-clob-client/)
- [TypeScript SDK](https://www.npmjs.com/package/@polymarket/clob-client)
