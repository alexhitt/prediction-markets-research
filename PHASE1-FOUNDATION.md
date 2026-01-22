# Phase 1: Prediction Markets Foundation Research

**Research Period:** Week 1-4
**Last Updated:** January 21, 2026
**Status:** In Progress

---

## Table of Contents

1. [Platform Comparison Overview](#platform-comparison-overview)
2. [Polymarket Deep Dive](#polymarket-deep-dive)
3. [Kalshi Deep Dive](#kalshi-deep-dive)
4. [Metaculus Deep Dive](#metaculus-deep-dive)
5. [Successful Trader Strategies](#successful-trader-strategies)
6. [Academic Research Findings](#academic-research-findings)
7. [Arbitrage Opportunities](#arbitrage-opportunities)
8. [Alternative Data Sources](#alternative-data-sources)
9. [Action Items & Next Steps](#action-items--next-steps)

---

## Platform Comparison Overview

| Feature | Polymarket | Kalshi | Metaculus |
|---------|-----------|--------|-----------|
| **Type** | Decentralized (Polygon) | Centralized (CFTC Regulated) | Reputation-based forecasting |
| **Currency** | USDC | USD | Points/Reputation |
| **Regulation** | Unregulated (non-US) | CFTC DCM | N/A |
| **US Access** | Restricted (VPN) | Full access | Full access |
| **Real Money** | Yes | Yes | No (reputation only) |
| **API Access** | Free (rate limited) | Free | Free |
| **Trading Fees** | 0% (most markets) | ~1% taker fee | N/A |
| **Settlement** | UMA Oracle / Pyth | Official sources | Manual resolution |
| **Liquidity** | High ($2B+ weekly) | High ($2.3B weekly) | N/A |

---

## Polymarket Deep Dive

### Platform Overview
- **Blockchain:** Polygon (L2 Ethereum)
- **Currency:** USDC stablecoin
- **Market Type:** Binary outcome contracts ($0.01 - $1.00)
- **Resolution:** Decentralized via UMA optimistic oracle or Pyth (price markets)

### Core Mechanics

#### Price Interpretation
- Share price = implied probability
- $0.75 YES share = 75% market-implied probability
- $0.25 NO share on same market = 25% probability (must sum to ~$1.00)

#### Fundamental Relationship
```
1 YES share + 1 NO share = $1.00 guaranteed payout

Splitting: 1 USDC → 1 YES + 1 NO
Merging: 1 YES + 1 NO → 1 USDC
```

#### Order Book System (CLOB)
- Hybrid-decentralized: off-chain matching, on-chain settlement
- Price-time priority matching
- **Mirroring:** Buy 100 YES @ $0.40 auto-displays Sell 100 NO @ $0.60

### API Architecture

| API | Endpoint | Purpose |
|-----|----------|---------|
| **Gamma API** | `gamma-api.polymarket.com` | Market discovery, events, categories |
| **CLOB API** | `clob.polymarket.com` | Real-time prices, orderbook, trading |
| **Data API** | `data-api.polymarket.com` | Positions, trade history, portfolio |
| **WebSocket** | `ws-subscriptions-clob.polymarket.com` | Real-time orderbook, price updates |

### SDKs Available
```bash
# TypeScript
npm install @polymarket/clob-client

# Python
pip install py-clob-client
```

### Fee Structure

| Market Type | Maker Fee | Taker Fee | Notes |
|-------------|-----------|-----------|-------|
| Political/Events | 0% | 0% | Free trading |
| Long-term crypto | 0% | 0% | Free trading |
| 15-min crypto | 0% | Up to 3.15% | Variable by probability |

**15-Minute Crypto Fee Curve:**
- Highest at 50% probability (~3.15%)
- Decreases toward 0% as odds approach 0% or 100%
- Fees redistributed as maker rebates

### Rate Limits
- Non-trading queries: Up to 1,000 calls/hour (free)
- Premium tiers: Start at $99/month for WebSocket, historical data

### Key Resources
- [Polymarket Documentation](https://docs.polymarket.com/)
- [Polymarket Agents (AI Trading)](https://github.com/Polymarket/agents)
- [Official Leaderboard](https://polymarket.com/leaderboard)

---

## Kalshi Deep Dive

### Platform Overview
- **Regulation:** First CFTC-regulated prediction market (Designated Contract Market)
- **Currency:** USD (bank transfers)
- **Valuation:** $11 billion (2025)
- **Weekly Volume:** $2.3 billion

### Core Mechanics

#### Contract Structure
- Binary contracts: YES/NO positions
- Price range: $0.01 - $0.99
- Settlement: $1.00 (win) or $0.00 (loss)
- Verified by official sources (NCAA.com, government data, etc.)

#### Order Book System
- **Makers:** Place limit orders (first to table)
- **Takers:** Match with existing orders (market orders)
- Price-time priority matching
- YES bid at X = NO ask at (100-X)

### API Architecture

| Feature | Details |
|---------|---------|
| **Protocol** | REST API + WebSocket |
| **Authentication** | RSA-based API keys |
| **Sandbox** | Demo environment available |
| **FIX Protocol** | Full implementation for institutional traders |

#### Key Endpoints
- **Exchange:** Status, announcements, schedule
- **Portfolio:** Balances (in cents), positions, settlements
- **Orders:** Create, amend, cancel, batch (up to 20)
- **Market Data:** Candlesticks (1m, 1h, 1d), orderbooks, trades
- **WebSocket:** Orderbook updates, tickers, trades, fills

### Fee Structure

| Fee Type | Amount | Notes |
|----------|--------|-------|
| ACH Deposit | $0 | Free |
| Wire Deposit | $0 | Free |
| Withdrawal | $2 | Per withdrawal |
| Trading Fee | ~1% | On expected earnings |
| Maker Fees | Variable | Some markets only |

### Unique Features
- **4% APY** on cash balances (including funds in open positions)
- **Subaccounts:** Create, transfer, manage multiple portfolios
- **Order Groups:** Auto-cancel triggers, 15-second windows
- **Subpenny Pricing:** Precision pricing supported
- **RFQ System:** Request for Quote communications

### Key Resources
- [Kalshi API Documentation](https://docs.kalshi.com/welcome)
- [Fee Schedule](https://kalshi.com/fee-schedule)
- [Developer Discord](https://discord.gg/kalshi)

---

## Metaculus Deep Dive

### Platform Overview
- **Type:** Reputation-based forecasting aggregation engine
- **Focus:** Scientific, technological, and global importance topics
- **Payout:** No real money (reputation/points only)

### Prediction Types

| Type | Description |
|------|-------------|
| **Binary** | Yes/No questions |
| **Numerical Range** | Continuous value predictions |
| **Date Range** | Timeline predictions |

### Aggregation Methods

#### 1. Community Prediction
- Time-weighted median of all individual forecasts
- Simple aggregation, treats all forecasters equally

#### 2. Metaculus Prediction (More Accurate)
- **Skill-weighted** forecast
- Leverages historical accuracy of individual forecasters
- Gives more weight to proven performers
- Consistently outperforms community prediction

### Scoring System
- Points awarded for correct predictions
- Points lost for incorrect predictions
- Bonus for outperforming community prediction
- Brier score used for calibration measurement

### API Access
- OpenAPI specification available
- GraphQL interface for queries
- Full forecast data accessible

### Performance Metrics
- Community Brier score on AI questions: 0.207
- Significantly better than chance (0.25)
- Both predictions robustly outperform naive baselines

### AI Forecasting Tools
```bash
# Official forecasting framework
pip install forecasting-tools

# Includes:
# - Metaculus API integration
# - Prediction aggregation
# - Benchmarking interfaces
```

### Key Resources
- [Metaculus API](https://www.metaculus.com/api/)
- [Forecasting Tools GitHub](https://github.com/Metaculus/forecasting-tools)
- [Bot Template](https://github.com/Metaculus/metac-bot-template)

---

## Successful Trader Strategies

### Top Performers Analysis

| Trader | Profit | Specialty | Key Strategy |
|--------|--------|-----------|--------------|
| **French Whale** | $85M | Politics | Commissioned custom polling ("neighbor effect") |
| **HyperLiquid0xb** | $1.4M | Sports (Baseball) | Quantitative modeling + fast news reaction |
| **WindWalk3** | $1.1M | Health Policy (RFK Jr.) | Deep domain expertise |
| **Erasmus** | $1.3M | Politics | Polling analysis + campaign momentum |
| **Domer** | Significant | Various | Liquidity provision + prediction |
| **Axios** | Consistent | Mention Markets | 96% win rate |

### Six Major Profit Strategies

#### 1. Information Arbitrage
- **Definition:** Profiting from unique data insights before market prices
- **Example:** French trader's $85M from custom polling
- **Key:** Access to information others don't have
- **Alpha Source:** Alternative data (like the ice cream machine post)

#### 2. Cross-Platform Arbitrage
- **Definition:** Exploiting price differences between platforms
- **Example:** Same event priced differently on Polymarket vs Kalshi
- **Profit:** $40M+ extracted historically
- **Risk:** Different settlement rules between platforms
- **Skill Required:** Low, but needs capital and attention

#### 3. High-Probability Bond Strategy
- **Definition:** Buying contracts priced >$0.95
- **Returns:** Small but frequent (5% on near-certain outcomes)
- **Risk:** Black swan events
- **Best For:** Capital preservation with slight upside

#### 4. Speed Trading
- **Definition:** Fastest reaction to breaking news
- **Example:** Fed announcement → price moved $0.65→$0.78 in 8 seconds
- **Requirements:** Live streams, preset triggers, low-latency execution
- **Edge:** Milliseconds matter

#### 5. Domain Specialization
- **Definition:** Becoming expert in narrow field
- **Observation:** Most profitable traders are specialists
- **Categories:** Sports (specific leagues), Politics (specific regions), Crypto
- **Advantage:** "Crushing advantage" in narrow domain

#### 6. Liquidity Provision + Prediction
- **Definition:** Market making while also having directional views
- **Example:** Domer on Pope market
- **Advantage:** Earn spread + profit from predictions
- **Complexity:** Highest skill requirement

### Portfolio Best Practices

| Metric | Optimal Range |
|--------|---------------|
| Simultaneous positions | 5-12 |
| Sweet spot | 6-10 |
| Capital reserve | 20-40% |
| Position mix | Short-term + Long-term |

### Success Statistics
- Wallets with PnL > $1,000: Only 0.51%
- Whales (volume > $50,000): Only 1.74%
- **Implication:** Edge is rare, most traders lose

---

## Academic Research Findings

### Key Papers

#### 1. "Unravelling the Probabilistic Forest" (August 2025)
**Source:** [arXiv:2508.03474](https://arxiv.org/abs/2508.03474)

**Key Findings:**
- Two arbitrage types identified:
  1. **Market Rebalancing Arbitrage:** Within single market/condition
  2. **Combinatorial Arbitrage:** Spans multiple related markets
- **$40 million** in realized arbitrage profits extracted
- On-chain arbitrage requires: cross-market execution, speed, noisy information parsing

#### 2. "Price Discovery and Trading in Prediction Markets" (November 2025)
**Source:** [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995)

**Key Findings:**
- First evidence of cross-market information discovery
- **Polymarket leads Kalshi** in price discovery
- Leadership strongest during high relative liquidity
- Significant price disparities create arbitrage opportunities

#### 3. "Microstructure of Wealth Transfer" (January 2026)
**Source:** [jbecker.dev](https://www.jbecker.dev/research/prediction-market-microstructure)

**Key Findings:**
- Maker-taker gaps reveal market efficiency
- **Finance markets:** 0.17 percentage points (near-perfect efficiency)
- **World Events/Media:** >7 percentage points (inefficient)
- Participant selection shapes efficiency

#### 4. "Arbitrage Trade in Prediction Markets" (2012)
**Source:** [Journal of Prediction Markets](https://www.ubplj.org/index.php/jpm/article/view/589)

**Key Findings:**
- Cross-border arbitrage opportunities are rare
- Political markets fairly efficient across countries
- Inter-market opportunities explained by opinion differences

### Theoretical Insights

1. **Arbitrage Paradox:** Symptom of inefficiency AND mechanism for restoring it
2. **Efficiency Factors:**
   - Market microstructure
   - Participant heterogeneity
   - Outcome space design
   - Transaction costs
   - Information flows
3. **EMH Test:** Prediction markets provide purest test of efficient market hypothesis

---

## Arbitrage Opportunities

### Types of Arbitrage

#### 1. Intra-Market (Risk-Free)
```
If YES @ $0.48 + NO @ $0.49 = $0.97
Buy both → Guaranteed $0.03 profit per $1
```

#### 2. Cross-Platform
```
Polymarket: Trump WIN @ $0.52
Kalshi: Trump WIN @ $0.48

Buy Kalshi YES + Polymarket NO
Lock in $0.04 spread (minus fees)
```

**Risks:**
- Different settlement rules
- Timing differences
- Platform counterparty risk

#### 3. Combinatorial
```
Related markets may misprice:
- "Fed cuts in December" @ 60%
- "Inflation above 3%" @ 30%

If these are logically linked, combined odds may exceed 100%
```

### Arbitrage Tracking Tools
- [Polymarket Analytics](https://polymarketanalytics.com/traders)
- [PolyTrack](https://www.polytrackhq.app/)
- [PolyAlertHub](https://polyalerthub.com/traders)
- [FinFeedAPI](https://www.finfeedapi.com/products/prediction-markets-api)

---

## Alternative Data Sources

### Proven Examples

| Data Source | Predictive Signal | Use Case |
|-------------|-------------------|----------|
| **McDonald's Ice Cream Machines** | Broken % → Economy health | Recession 3-4 weeks early |
| **DC Machine Status** | Fixed → Presidential event | 48 hours advance notice |
| **Swing State Machines** | Broken % → Election outcome | Predicted 6/7 states |
| **Custom Polling** | "Neighbor effect" | $85M profit |

### Categories to Explore

#### Real-Time Economic Indicators
- Job posting volumes (Indeed, LinkedIn)
- Shipping container movements
- Electricity consumption
- Credit card transaction aggregates
- Restaurant reservation data (OpenTable)

#### Social Sentiment
- Twitter/X sentiment analysis
- Reddit discussion volume
- Google Trends search patterns
- News headline sentiment

#### Government & Political
- Congressional trading disclosures
- Regulatory filing patterns
- Lobbyist activity
- Campaign finance flows

#### Satellite & Physical
- Parking lot occupancy
- Factory smoke stack activity
- Port shipping activity
- Agricultural field health

#### Platform-Specific
- Polymarket whale wallet movements
- Order flow imbalances
- Sudden liquidity changes
- Cross-platform price divergence

---

## Action Items & Next Steps

### Week 1-2: Platform Setup
- [ ] Create Polymarket account (non-US or via regulated US version)
- [ ] Create Kalshi account
- [ ] Create Metaculus account
- [ ] Set up API keys for all platforms
- [ ] Test basic API calls (read-only)

### Week 2-3: Data Collection Infrastructure
- [ ] Build market data scrapers (prices, volumes, orderbooks)
- [ ] Set up database for historical data
- [ ] Create monitoring dashboard (Streamlit/Grafana)
- [ ] Track whale wallet movements

### Week 3-4: Strategy Research
- [ ] Document 10 potential alternative data sources
- [ ] Paper trade 5 different strategies
- [ ] Build signal hypothesis journal
- [ ] Identify personal specialization domain

### Ongoing
- [ ] Track performance of hypotheses
- [ ] Analyze losing trades for lessons
- [ ] Refine edge identification process
- [ ] Build toward automated recommendations

---

## Resources & Links

### Documentation
- [Polymarket Docs](https://docs.polymarket.com/)
- [Kalshi API Docs](https://docs.kalshi.com/welcome)
- [Metaculus API](https://www.metaculus.com/api/)

### Analytics Tools
- [Polymarket Analytics](https://polymarketanalytics.com/)
- [PolyTrack](https://www.polytrackhq.app/)
- [FinFeedAPI](https://www.finfeedapi.com/)

### Academic
- [arXiv Prediction Markets](https://arxiv.org/search/?searchtype=all&query=prediction+markets)
- [SSRN Prediction Markets](https://papers.ssrn.com/sol3/results.cfm?RequestTimeout=50000000)

### Community
- [Kalshi Discord](https://discord.gg/kalshi)
- [Polymarket Twitter](https://twitter.com/polymarket)
- [Metaculus Community](https://www.metaculus.com/questions/)

---

*This document is part of the Prediction Markets AI System research project.*
*Phase 1 of 4 | Foundation Research*
