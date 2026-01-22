# Prediction Markets AI System - Research Project

## Project Vision

Build a world-class AI-powered prediction system that:
- Monitors Polymarket and Kalshi in real-time
- Uses alternative data sources for predictive edge
- Identifies arbitrage opportunities between markets
- Makes autonomous recommendations
- Features an interactive dashboard with real-time graphs

---

## Project Status

### Phase 1: Foundation Research ✅ COMPLETE
- [x] Platform mechanics analysis (Polymarket, Kalshi, Metaculus)
- [x] API documentation and integration guides
- [x] Fee structures and liquidity patterns
- [x] Academic research review
- [x] Successful trader strategy analysis
- [x] Alternative data source catalog

### Phase 2: Data Infrastructure ✅ COMPLETE
- [x] Set up API connections (read-only)
- [x] Build data collection pipelines
- [x] Create historical database (SQLite/PostgreSQL)
- [x] Implement monitoring dashboard (Streamlit)
- [x] Cross-platform arbitrage detector
- [x] Scheduled data collection jobs

### Phase 3: Signal Development (Next)
- [ ] Test alternative data correlations
- [ ] Build hypothesis tracking system
- [ ] Paper trade strategies
- [ ] Develop AI prediction models

### Phase 4: Production System (Future)
- [ ] Automated recommendations
- [ ] Real-time alerting
- [ ] Whale tracking
- [ ] Advanced analytics

---

## Directory Structure

```
prediction-markets-research/
├── README.md                          # This file
├── PHASE1-FOUNDATION.md               # Complete Phase 1 research
│
├── platforms/                         # Platform-specific docs
│   └── (coming soon)
│
├── apis/                              # API reference docs
│   ├── POLYMARKET-API-REFERENCE.md    # Polymarket API guide
│   └── KALSHI-API-REFERENCE.md        # Kalshi API guide
│
├── strategies/                        # Trading strategies
│   └── TRADING-STRATEGIES.md          # 6 proven strategies
│
├── data-sources/                      # Alternative data
│   └── ALTERNATIVE-DATA-SOURCES.md    # Data source catalog
│
├── papers/                            # Academic research
│   └── (coming soon)
│
└── notes/                             # Working notes
    └── (coming soon)
```

---

## Key Findings

### Platform Comparison

| Platform | Best For | Fee | US Access |
|----------|----------|-----|-----------|
| Polymarket | Liquidity, Crypto | 0% most markets | Restricted |
| Kalshi | Regulation, Safety | ~1% | Full |
| Metaculus | Forecasting practice | Free | Full |

### Top Strategies

1. **Information Arbitrage** - Unique data insights (e.g., ice cream machines)
2. **Cross-Platform Arbitrage** - Price differences between platforms
3. **Domain Specialization** - Deep expertise in narrow field
4. **Speed Trading** - Fast reaction to breaking news
5. **High-Probability Bonds** - Near-certain outcomes
6. **Liquidity Provision** - Market making + prediction

### Key Statistics

- Only 0.51% of wallets have PnL > $1,000
- Only 1.74% are whale accounts (>$50K volume)
- French whale made $85M with custom polling data
- $40M+ extracted through cross-platform arbitrage

---

## Alternative Data Concept

The "Ice Cream Machine" thesis from @Argona0x:

```
Broken machines → Understaffed → Cost cutting → Economy weakening
→ Predicts recession 3-4 weeks early

Additional signals:
- DC machines fixed → Presidential event in 48 hours
- Swing state 71% broken → Bet against incumbent
```

**Philosophy:** Find unconventional data that correlates with predictable events before the market prices it in.

---

## Quick Start

### 1. Install Dependencies
```bash
cd prediction-markets-research
pip3 install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (optional for read-only mode)
```

### 3. Test Connections
```bash
python3 main.py test
```

### 4. Run Commands
```bash
# Launch dashboard
python3 main.py dashboard

# Scan for arbitrage
python3 main.py arbitrage

# Start data collector
python3 main.py collect

# Initialize database
python3 main.py init-db
```

---

## Resources

### Documentation
- [Polymarket Docs](https://docs.polymarket.com/)
- [Kalshi API](https://docs.kalshi.com/)
- [Metaculus API](https://www.metaculus.com/api/)

### Tools
- [Polymarket Analytics](https://polymarketanalytics.com/)
- [PolyTrack](https://www.polytrackhq.app/)
- [Polymarket Agents](https://github.com/Polymarket/agents)

### Research
- [arXiv: Prediction Market Arbitrage](https://arxiv.org/abs/2508.03474)
- [SSRN: Price Discovery](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995)

---

## Next Actions

1. **Week 1-2:** Set up API connections, start data collection
2. **Week 3-4:** Build monitoring dashboard, track whale movements
3. **Week 5-8:** Test alternative data correlations
4. **Week 9+:** Paper trade, measure edge, iterate

---

*Project started: January 21, 2026*
*Phase 1 completed: January 21, 2026*
*Phase 2 completed: January 21, 2026*
