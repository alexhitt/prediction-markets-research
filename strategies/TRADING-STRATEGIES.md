# Prediction Market Trading Strategies

## Overview

This document catalogs proven and theoretical trading strategies for prediction markets based on analysis of successful traders and academic research.

---

## Strategy 1: Information Arbitrage (Alpha Generation)

### Concept
Profit from unique data insights that the market hasn't priced in yet.

### Famous Example: The French Whale ($85M)
- Commissioned custom "neighbor effect" polling during 2024 US election
- Asked people who they think their neighbors would vote for (reduces social desirability bias)
- Identified systematic market mispricing
- Result: $85 million profit on Trump victory bet

### Ice Cream Machine Example (From @Argona0x Post)
```
Signal Chain:
Broken machines → Understaffed stores → Franchise cost-cutting
→ Economy weakening → Recession prediction 3-4 weeks early

Additional Signals Found:
- DC machines fixed → Presidential event in 48 hours
- Swing state broken % → Correlates with election outcomes
- 71% broken in swing states → Economy dying, bet against incumbent
```

### Implementation Approach
1. Identify unconventional data sources
2. Establish correlation with market outcomes
3. Build monitoring system
4. Backtest against historical markets
5. Paper trade before real capital

### Data Sources to Explore
| Category | Source | Potential Signal |
|----------|--------|------------------|
| Economic | Job postings | Hiring = growth |
| Economic | Shipping data | Trade activity |
| Political | Lobbyist filings | Policy direction |
| Consumer | Restaurant reservations | Consumer confidence |
| Social | Twitter sentiment | Public opinion shifts |

---

## Strategy 2: Cross-Platform Arbitrage

### Concept
Exploit price differences for identical events across platforms.

### Example
```
Polymarket: "Trump wins 2024" @ $0.52
Kalshi: "Trump wins 2024" @ $0.48
PredictIt: "Trump wins 2024" @ $0.50

Arbitrage:
- Buy Kalshi YES @ $0.48
- Buy Polymarket NO @ $0.48 (100 - 52)
- Guaranteed $0.04 profit per contract (minus fees)
```

### Historical Profits
- Research shows $40M+ extracted through cross-platform arbitrage
- Low-skill but requires capital and attention

### Risks
| Risk | Mitigation |
|------|------------|
| Different settlement rules | Read terms carefully |
| Timing differences | Hedge appropriately |
| Platform counterparty risk | Diversify exposure |
| Fee erosion | Calculate net profit |
| Liquidity constraints | Size positions appropriately |

### Implementation
```python
def find_arbitrage(polymarket_price, kalshi_price):
    """
    Check for arbitrage between platforms.
    Prices should be for same outcome (e.g., YES).
    """
    poly_yes = polymarket_price
    poly_no = 1 - polymarket_price
    kalshi_yes = kalshi_price
    kalshi_no = 1 - kalshi_price

    # Check if buying YES on one and NO on other is profitable
    cost_option_1 = poly_yes + kalshi_no
    cost_option_2 = kalshi_yes + poly_no

    if cost_option_1 < 1.0:
        return {
            "opportunity": True,
            "action": "Buy Polymarket YES + Kalshi NO",
            "cost": cost_option_1,
            "profit": 1.0 - cost_option_1
        }
    elif cost_option_2 < 1.0:
        return {
            "opportunity": True,
            "action": "Buy Kalshi YES + Polymarket NO",
            "cost": cost_option_2,
            "profit": 1.0 - cost_option_2
        }
    return {"opportunity": False}
```

---

## Strategy 3: High-Probability Bonds

### Concept
Buy contracts priced >$0.95 for small, frequent gains on near-certain outcomes.

### Example
```
Market: "Will the sun rise tomorrow?"
Price: YES @ $0.99
Return: 1% in 24 hours = 365% APY (theoretical)
```

### Real Application
- Markets near resolution with clear outcomes
- Events that are 99% certain but not yet settled
- Capture the final 1-5% as probability → certainty

### Risk
- **Black swan events**: The 1% happens
- **Settlement delays**: Capital locked up
- **Opportunity cost**: Could deploy capital elsewhere

### Position Sizing
```python
def kelly_criterion(win_prob, win_return, loss_return):
    """
    Kelly Criterion for optimal bet sizing.
    """
    return (win_prob * win_return - (1 - win_prob) * abs(loss_return)) / win_return

# For a 99% certain bet paying 1%
kelly = kelly_criterion(0.99, 0.01, 1.0)  # ~0.98 (98% of bankroll)
# In practice, use fractional Kelly (1/4 to 1/2)
```

---

## Strategy 4: Speed Trading (News Reaction)

### Concept
React faster than the market to breaking news.

### Famous Example
```
Event: Fed Chair Powell says "We will adjust policy as appropriate"
Timeline:
- T+0: Statement made
- T+8 seconds: "Fed December rate cut" jumps $0.65 → $0.78
- Speed traders captured the move
```

### Requirements
| Requirement | Implementation |
|-------------|----------------|
| Low-latency data | Direct API feeds, WebSocket |
| Pre-set triggers | Keyword detection, price thresholds |
| Automated execution | Pre-signed orders ready to fire |
| Live streams | Fed, Congress, Press conferences |

### Infrastructure Needed
1. **News Monitoring**
   - Multiple live stream sources
   - Speech-to-text transcription
   - Keyword trigger system

2. **Decision Engine**
   - Pre-defined rules for each scenario
   - Confidence thresholds
   - Position sizing logic

3. **Execution Layer**
   - Pre-authenticated API connections
   - Order templates ready
   - Failover systems

### Code Skeleton
```python
import asyncio
from typing import Callable

class SpeedTradingBot:
    def __init__(self, api_client, triggers: dict):
        self.client = api_client
        self.triggers = triggers  # keyword -> action mapping

    async def monitor_stream(self, stream_url):
        async for text in transcribe_stream(stream_url):
            await self.check_triggers(text)

    async def check_triggers(self, text):
        for keyword, action in self.triggers.items():
            if keyword.lower() in text.lower():
                await self.execute_action(action)

    async def execute_action(self, action):
        # Pre-defined order
        order = self.client.create_order(
            market=action["market"],
            side=action["side"],
            price=action["price"],
            size=action["size"]
        )
        print(f"Executed: {order}")

# Example triggers
triggers = {
    "rate cut": {
        "market": "FED-DEC-RATECUT",
        "side": "yes",
        "price": 0.70,
        "size": 100
    },
    "recession": {
        "market": "US-RECESSION-2025",
        "side": "yes",
        "price": 0.30,
        "size": 50
    }
}
```

---

## Strategy 5: Domain Specialization

### Concept
Become the expert in a narrow field with "crushing advantage."

### Evidence
- Most profitable Polymarket traders are specialists
- Generalists underperform focused experts
- Deep knowledge > broad coverage

### Specialization Options

| Domain | Edge Source |
|--------|-------------|
| Baseball | Advanced metrics, injury reports |
| Elections (specific region) | Local polling, campaign intel |
| Crypto prices | On-chain analysis, whale watching |
| Fed policy | Economic modeling, Fed-speak interpretation |
| Tech earnings | Supply chain data, app store rankings |
| Weather events | Meteorological models |

### Building Domain Expertise
1. **Choose narrow domain** (e.g., MLB, not "sports")
2. **Immerse completely** (read everything, follow experts)
3. **Build quantitative models** (data > intuition)
4. **Track your predictions** (measure Brier score)
5. **Refine continuously** (learn from losses)

### Metrics to Track
```python
class SpecializationTracker:
    def __init__(self, domain):
        self.domain = domain
        self.predictions = []

    def add_prediction(self, market_id, my_prob, market_prob, outcome):
        self.predictions.append({
            "market_id": market_id,
            "my_probability": my_prob,
            "market_probability": market_prob,
            "actual_outcome": outcome,
            "my_brier": (my_prob - outcome) ** 2,
            "market_brier": (market_prob - outcome) ** 2
        })

    def performance_summary(self):
        my_avg_brier = sum(p["my_brier"] for p in self.predictions) / len(self.predictions)
        market_avg_brier = sum(p["market_brier"] for p in self.predictions) / len(self.predictions)
        return {
            "my_brier_score": my_avg_brier,
            "market_brier_score": market_avg_brier,
            "edge": market_avg_brier - my_avg_brier,
            "total_predictions": len(self.predictions)
        }
```

---

## Strategy 6: Liquidity Provision + Prediction

### Concept
Act as market maker while also having directional views.

### How It Works
1. Provide liquidity (bid and ask orders)
2. Earn spread on trades
3. Also take directional positions on your predictions
4. Combine spread income + prediction profits

### Example: Domer (Pope Market)
- Provided liquidity in Pope succession market
- Had strong directional view based on research
- Earned spread while waiting for resolution
- Won big on prediction + earned trading fees

### Requirements
- Significant capital
- Understanding of market making
- Strong prediction ability
- Risk management skills

### Risk
- Adverse selection (informed traders pick you off)
- Inventory risk (stuck with losing position)
- Complexity (harder than pure prediction)

---

## Portfolio Management

### Optimal Position Count
| Metric | Recommended |
|--------|-------------|
| Simultaneous positions | 5-12 |
| Sweet spot | 6-10 |
| Capital in reserve | 20-40% |

### Diversification Rules
1. **Across categories**: Don't bet all on politics
2. **Across timeframes**: Mix short and long-term
3. **Across confidence levels**: Some safe, some speculative
4. **Across platforms**: Reduce counterparty risk

### Position Sizing
```python
def position_size(bankroll, confidence, kelly_fraction=0.25):
    """
    Conservative position sizing.

    confidence: Your edge (e.g., 0.05 = 5% better than market)
    kelly_fraction: How much of Kelly to use (0.25 = quarter Kelly)
    """
    # Full Kelly for binary bet
    full_kelly = confidence

    # Fractional Kelly (more conservative)
    position = bankroll * full_kelly * kelly_fraction

    # Cap at 10% of bankroll per position
    max_position = bankroll * 0.10

    return min(position, max_position)
```

---

## Risk Management

### Rules
1. **Never bet more than 10%** of bankroll on single market
2. **Keep 20-40% in reserve** for opportunities
3. **Set stop losses** (exit if confidence drops)
4. **Track all bets** (learn from losses)
5. **Avoid correlated bets** (all eggs in one basket)

### Red Flags to Avoid
- FOMO (fear of missing out)
- Revenge trading (trying to recover losses)
- Overconfidence after wins
- Ignoring fees in calculations
- Betting on unfamiliar domains

---

## Success Metrics

### Key Statistics
- **Wallets with PnL > $1,000**: Only 0.51%
- **Whales (volume > $50,000)**: Only 1.74%
- **Most traders lose money**

### What Separates Winners
1. Systematic identification of market pricing errors
2. Obsessive risk management
3. Patience to build information advantage
4. Domain specialization
5. Emotional discipline

---

## Next Steps

1. [ ] Choose initial specialization domain
2. [ ] Paper trade for 4 weeks minimum
3. [ ] Build prediction tracking system
4. [ ] Measure Brier score vs market
5. [ ] Only deploy real capital with proven edge
