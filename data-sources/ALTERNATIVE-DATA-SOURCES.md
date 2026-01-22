# Alternative Data Sources for Prediction Markets

## The Core Concept

Alternative data refers to non-traditional data sources that can provide predictive signals before conventional indicators or market prices reflect them.

**The Ice Cream Machine Thesis:**
> "The dumbest edge that actually prints" - @Argona0x

```
McDonald's broken ice cream machines → Leading economic indicator

Logic Chain:
1. Machines break → Need staff to fix
2. Understaffed → Machines stay broken
3. Understaffed = Cost cutting
4. Cost cutting = Economy weakening
5. Broken machine % predicts recession 3-4 weeks early
```

---

## Proven Alternative Data Examples

### 1. McDonald's Ice Cream Machines
**Source:** mcbroken.com API, regional franchise data
**Signal:** Broken percentage by geography
**Predictions:**
- Economic health (broken % ↑ = recession coming)
- Political events (DC machines fixed = presidential event in 48 hours)
- Election outcomes (swing state machine status → voting patterns)

### 2. Satellite Imagery
**Source:** Planet Labs, Maxar, Sentinel
**Signals:**
- Parking lot occupancy → Retail earnings
- Oil storage tank shadows → Crude inventory
- Factory activity → Manufacturing output
- Port shipping → Trade data
- Agricultural health → Crop yields

### 3. Web Traffic / App Data
**Source:** SimilarWeb, Sensor Tower, App Annie
**Signals:**
- Company website traffic → Earnings surprise
- App downloads → Product adoption
- Job posting changes → Hiring/layoffs

### 4. Social Sentiment
**Source:** Twitter/X API, Reddit, StockTwits
**Signals:**
- Mention volume → Event importance
- Sentiment score → Public opinion
- Viral spread → Breaking news

### 5. Credit Card Transaction Data
**Source:** Second Measure, Earnest Research
**Signals:**
- Consumer spending patterns
- Retail sales before official reports
- Restaurant/travel industry health

### 6. Shipping & Logistics
**Source:** Marine Traffic, FlightRadar24
**Signals:**
- Container shipping volume → Trade
- Private jet movements → Deal activity
- Trucking data → Goods movement

---

## Categorized Data Source Ideas

### Economic Indicators (Leading)

| Data Source | Signal | Prediction Target |
|-------------|--------|-------------------|
| Job postings (Indeed, LinkedIn) | Hiring velocity | Employment report |
| Unemployment claims (real-time) | Layoff trends | Recession probability |
| Small business sentiment | Main street health | GDP growth |
| Restaurant reservations | Consumer confidence | Retail spending |
| Hotel occupancy | Travel demand | Consumer spending |
| Electricity usage | Industrial activity | Manufacturing PMI |

### Political & Policy

| Data Source | Signal | Prediction Target |
|-------------|--------|-------------------|
| Congressional trading disclosures | Insider knowledge | Policy direction |
| Lobbyist registration | Industry focus | Regulatory changes |
| Campaign finance flows | Fundraising momentum | Election outcomes |
| Betting odds (offshore) | Market consensus | Political events |
| Protest permits | Social unrest | Policy response |
| FOIA request patterns | Investigation activity | Political scandals |

### Corporate & Earnings

| Data Source | Signal | Prediction Target |
|-------------|--------|-------------------|
| Employee reviews (Glassdoor) | Company health | Earnings surprise |
| Patent filings | Innovation pipeline | Product launches |
| SEC 13F filings | Institutional positions | Stock movements |
| Supplier/vendor mentions | Supply chain | Revenue estimates |
| LinkedIn employee count | Headcount changes | Growth trajectory |
| GitHub commit activity | Development velocity | Product releases |

### Weather & Environment

| Data Source | Signal | Prediction Target |
|-------------|--------|-------------------|
| NOAA models | Weather forecasts | Commodity prices |
| Hurricane trackers | Storm paths | Insurance claims |
| Wildfire maps | Fire spread | Utility stocks |
| Air quality sensors | Pollution levels | Health policy |
| Water reservoir levels | Drought severity | Agricultural output |

### Crypto-Specific

| Data Source | Signal | Prediction Target |
|-------------|--------|-------------------|
| Whale wallet movements | Large holder activity | Price direction |
| Exchange inflows/outflows | Selling/accumulation | Volatility |
| Stablecoin minting | New capital entering | Market direction |
| Miner behavior | Hash rate, selling | Bitcoin price |
| DeFi TVL changes | Ecosystem health | Protocol prices |
| Gas prices (Ethereum) | Network activity | Congestion/adoption |

### Sports

| Data Source | Signal | Prediction Target |
|-------------|--------|-------------------|
| Injury reports (beat writers) | Player availability | Game outcomes |
| Weather at venue | Playing conditions | Scoring patterns |
| Travel schedules | Team fatigue | Performance |
| Social media activity | Player focus/distraction | Game results |
| Betting line movements | Sharp money | Outcomes |

---

## Building Your Data Pipeline

### Architecture
```
┌─────────────────┐
│  Data Sources   │
│  (APIs, Scrape) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Collection     │
│  (Schedulers)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Storage        │
│  (TimescaleDB)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Processing     │
│  (Correlation)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Signal Gen     │
│  (Alerts)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dashboard      │
│  (Visualization)│
└─────────────────┘
```

### Example: Ice Cream Machine Monitor
```python
import requests
import schedule
from datetime import datetime

class IceCreamMonitor:
    def __init__(self, db_connection):
        self.db = db_connection
        self.api_url = "https://mcbroken.com/stats.json"

    def fetch_data(self):
        """Fetch current broken machine status."""
        response = requests.get(self.api_url)
        data = response.json()

        return {
            "timestamp": datetime.utcnow(),
            "national_broken_pct": self.calculate_national(data),
            "dc_broken_pct": self.calculate_region(data, "DC"),
            "swing_states": self.calculate_swing_states(data)
        }

    def calculate_national(self, data):
        # Calculate national broken percentage
        total = len(data["locations"])
        broken = sum(1 for loc in data["locations"] if not loc["working"])
        return broken / total * 100

    def calculate_region(self, data, region):
        # Filter to specific region
        regional = [loc for loc in data["locations"] if loc["state"] == region]
        broken = sum(1 for loc in regional if not loc["working"])
        return broken / len(regional) * 100 if regional else 0

    def calculate_swing_states(self, data):
        swing_states = ["PA", "MI", "WI", "AZ", "GA", "NV", "NC"]
        results = {}
        for state in swing_states:
            state_locs = [loc for loc in data["locations"] if loc["state"] == state]
            if state_locs:
                broken = sum(1 for loc in state_locs if not loc["working"])
                results[state] = broken / len(state_locs) * 100
        return results

    def store_data(self, data):
        """Store in database."""
        self.db.insert("ice_cream_data", data)

    def check_signals(self, data):
        """Check for actionable signals."""
        signals = []

        # Signal 1: DC machines suddenly working (political event)
        historical_dc = self.db.get_recent("ice_cream_data", "dc_broken_pct", days=7)
        if historical_dc and data["dc_broken_pct"] < historical_dc[-1] - 20:
            signals.append({
                "type": "DC_MACHINES_FIXED",
                "action": "Check for presidential event in 48 hours",
                "market": "Political events on Polymarket"
            })

        # Signal 2: National broken % spiking (recession)
        if data["national_broken_pct"] > 30:  # Historical threshold
            signals.append({
                "type": "RECESSION_WARNING",
                "action": "Consider bearish economic bets",
                "data": data["national_broken_pct"]
            })

        return signals

    def run(self):
        """Main execution."""
        data = self.fetch_data()
        self.store_data(data)
        signals = self.check_signals(data)
        for signal in signals:
            self.alert(signal)

    def alert(self, signal):
        print(f"SIGNAL: {signal}")
        # Send to Slack, email, etc.

# Schedule hourly checks
monitor = IceCreamMonitor(db)
schedule.every(1).hours.do(monitor.run)
```

---

## Correlation Analysis Framework

### Step 1: Hypothesis Formation
```
"I believe [DATA SOURCE] correlates with [PREDICTION TARGET]
because [CAUSAL REASONING]"

Example:
"I believe McDonald's broken ice cream % correlates with recession timing
because broken machines indicate understaffing from cost-cutting"
```

### Step 2: Data Collection
```python
def collect_historical_data(data_source, target_market, lookback_days=365):
    """
    Collect aligned historical data for both signal and target.
    """
    signal_data = data_source.get_history(days=lookback_days)
    target_data = target_market.get_history(days=lookback_days)

    # Align timestamps
    aligned = align_timeseries(signal_data, target_data)

    return aligned
```

### Step 3: Correlation Testing
```python
import numpy as np
from scipy import stats

def test_correlation(signal_data, target_data, lag_days=0):
    """
    Test correlation with optional time lag.
    """
    # Apply lag (signal leads target)
    if lag_days > 0:
        signal_lagged = signal_data[:-lag_days]
        target_lagged = target_data[lag_days:]
    else:
        signal_lagged = signal_data
        target_lagged = target_data

    correlation, p_value = stats.pearsonr(signal_lagged, target_lagged)

    return {
        "correlation": correlation,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "lag_days": lag_days
    }

def find_optimal_lag(signal_data, target_data, max_lag=30):
    """
    Find the lag that maximizes correlation.
    """
    results = []
    for lag in range(max_lag + 1):
        result = test_correlation(signal_data, target_data, lag)
        results.append(result)

    best = max(results, key=lambda x: abs(x["correlation"]))
    return best
```

### Step 4: Backtesting
```python
def backtest_signal(signal_data, target_outcomes, threshold):
    """
    Backtest trading based on signal threshold.
    """
    trades = []

    for i, signal_value in enumerate(signal_data):
        if signal_value > threshold:
            # Signal triggered - would have bet YES
            outcome = target_outcomes[i]
            profit = 1.0 - entry_price if outcome else -entry_price
            trades.append({
                "signal": signal_value,
                "outcome": outcome,
                "profit": profit
            })

    total_profit = sum(t["profit"] for t in trades)
    win_rate = sum(1 for t in trades if t["profit"] > 0) / len(trades)

    return {
        "total_trades": len(trades),
        "total_profit": total_profit,
        "win_rate": win_rate,
        "avg_profit": total_profit / len(trades) if trades else 0
    }
```

---

## Signal Hypothesis Journal Template

```markdown
## Hypothesis: [NAME]

**Date Created:** YYYY-MM-DD
**Status:** Testing / Validated / Rejected

### Data Source
- **Source:** [Where the data comes from]
- **Update Frequency:** [How often new data available]
- **Access Method:** [API / Scraping / Purchase]
- **Cost:** [Free / Paid]

### Prediction Target
- **Market/Event:** [What you're predicting]
- **Platform:** [Polymarket / Kalshi / Both]
- **Time Horizon:** [How far in advance signal appears]

### Causal Theory
[Why you believe this correlation exists]

### Historical Correlation
- **Correlation coefficient:** X.XX
- **P-value:** X.XX
- **Optimal lag:** X days
- **Backtest profit:** $X,XXX

### Trading Rules
1. [When to enter]
2. [Position size]
3. [When to exit]

### Results Tracking
| Date | Signal | Prediction | Outcome | Profit |
|------|--------|------------|---------|--------|
| | | | | |

### Notes
[Observations, refinements, lessons learned]
```

---

## Data Source APIs & Tools

### Free APIs
| Source | API | Data |
|--------|-----|------|
| Reddit | pushshift.io | Subreddit sentiment |
| Twitter | Academic API | Tweet sentiment |
| Google Trends | pytrends | Search interest |
| Wikipedia | pageviews API | Article traffic |
| Weather | OpenWeatherMap | Forecasts |
| GitHub | REST API | Commit activity |

### Paid APIs
| Source | Provider | Data |
|--------|----------|------|
| Web traffic | SimilarWeb | Site analytics |
| App data | Sensor Tower | Downloads, usage |
| Satellite | Planet Labs | Earth imagery |
| Transactions | Second Measure | Credit card data |
| Jobs | Revelio Labs | Workforce data |

### Scraping Targets
| Site | Data | Notes |
|------|------|-------|
| mcbroken.com | Ice cream status | Already aggregated |
| Glassdoor | Employee reviews | Requires parsing |
| Congressional sites | Trading disclosures | Public but messy |
| Local news | Regional sentiment | NLP required |

---

## Next Steps

1. [ ] Choose 3 data sources to explore
2. [ ] Build collection pipeline for each
3. [ ] Run correlation analysis vs historical markets
4. [ ] Document findings in hypothesis journal
5. [ ] Paper trade promising signals
6. [ ] Iterate and refine
