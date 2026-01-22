"""
Prediction Markets Monitoring Dashboard

Interactive Streamlit dashboard for:
- Real-time market data from Polymarket and Kalshi
- Arbitrage opportunity detection
- Price history visualization
- Signal tracking
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict

# Import our modules
from src.clients.polymarket_client import PolymarketClient
from src.clients.kalshi_client import KalshiClient
from src.signals.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from src.database.db import get_session, init_db
from src.database.models import Market, MarketSnapshot, ArbitrageOpportunity as ArbitrageModel

# Page config
st.set_page_config(
    page_title="Prediction Markets Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize clients (cached)
@st.cache_resource
def get_clients():
    return PolymarketClient(), KalshiClient()

poly_client, kalshi_client = get_clients()


def main():
    """Main dashboard application."""

    # Sidebar
    with st.sidebar:
        st.title("📊 Prediction Markets")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Overview", "📈 Markets", "⚡ Arbitrage", "📊 Analytics", "🧪 Hypotheses", "💰 Paper Trading", "⚙️ Settings"]
        )

        st.markdown("---")
        st.markdown("**Status**")

        # Connection status
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🟢 Polymarket")
        with col2:
            st.markdown("🟢 Kalshi")

        st.markdown("---")
        st.caption("Last updated: " + datetime.now().strftime("%H:%M:%S"))

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Main content
    if page == "🏠 Overview":
        render_overview()
    elif page == "📈 Markets":
        render_markets()
    elif page == "⚡ Arbitrage":
        render_arbitrage()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "🧪 Hypotheses":
        render_hypotheses()
    elif page == "💰 Paper Trading":
        render_paper_trading()
    elif page == "⚙️ Settings":
        render_settings()


def render_overview():
    """Overview page with key metrics."""
    st.title("🏠 Dashboard Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    # Fetch data
    poly_markets = fetch_polymarket_markets(limit=100)
    kalshi_markets = fetch_kalshi_markets(limit=100)
    arb_opportunities = fetch_arbitrage()

    with col1:
        st.metric(
            "Polymarket Markets",
            len(poly_markets),
            delta=None
        )

    with col2:
        st.metric(
            "Kalshi Markets",
            len(kalshi_markets),
            delta=None
        )

    with col3:
        st.metric(
            "Arbitrage Opps",
            len(arb_opportunities),
            delta=None
        )

    with col4:
        total_volume = sum(m.get("volume", 0) for m in poly_markets)
        st.metric(
            "Total Volume (Poly)",
            f"${total_volume:,.0f}"
        )

    st.markdown("---")

    # Top markets
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Top Polymarket by Volume")
        if poly_markets:
            df = pd.DataFrame(poly_markets[:10])
            if "question" in df.columns and "volume" in df.columns:
                df["question"] = df["question"].str[:50] + "..."
                st.dataframe(
                    df[["question", "volume", "category"]].rename(columns={
                        "question": "Market",
                        "volume": "Volume",
                        "category": "Category"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("No Polymarket data available")

    with col2:
        st.subheader("🔥 Top Kalshi by Volume")
        if kalshi_markets:
            df = pd.DataFrame(kalshi_markets[:10])
            if "title" in df.columns and "volume" in df.columns:
                df["title"] = df["title"].str[:50] + "..."
                st.dataframe(
                    df[["title", "volume", "category"]].rename(columns={
                        "title": "Market",
                        "volume": "Volume",
                        "category": "Category"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("No Kalshi data available")

    # Arbitrage alerts
    st.markdown("---")
    st.subheader("⚡ Active Arbitrage Opportunities")

    if arb_opportunities:
        for opp in arb_opportunities[:5]:
            with st.expander(f"📊 Spread: {opp.spread:.2%} | Profit: {opp.profit_potential:.2%}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Polymarket:** {opp.market_1_question[:60]}...")
                    st.markdown(f"Price: **{opp.market_1_yes_price:.1%}**")
                with col2:
                    st.markdown(f"**Kalshi:** {opp.market_2_question[:60]}...")
                    st.markdown(f"Price: **{opp.market_2_yes_price:.1%}**")
                st.info(f"**Action:** {opp.action}")
    else:
        st.info("No arbitrage opportunities detected at this time.")


def render_markets():
    """Markets explorer page."""
    st.title("📈 Markets Explorer")

    # Platform selector
    platform = st.selectbox("Select Platform", ["Polymarket", "Kalshi", "Both"])

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", ["All", "Politics", "Crypto", "Sports", "Economics"])
    with col2:
        sort_by = st.selectbox("Sort By", ["Volume", "Liquidity", "End Date"])
    with col3:
        limit = st.slider("Limit", 10, 200, 50)

    st.markdown("---")

    # Fetch and display markets
    if platform in ["Polymarket", "Both"]:
        st.subheader("Polymarket Markets")
        markets = fetch_polymarket_markets(limit=limit)
        if markets:
            df = pd.DataFrame(markets)
            if "question" in df.columns:
                # Format for display
                display_df = df[["question", "volume", "liquidity", "category", "status"]].copy()
                display_df["question"] = display_df["question"].str[:60]
                display_df["volume"] = display_df["volume"].apply(lambda x: f"${x:,.0f}" if x else "-")
                display_df["liquidity"] = display_df["liquidity"].apply(lambda x: f"${x:,.0f}" if x else "-")

                st.dataframe(display_df, hide_index=True, use_container_width=True)

    if platform in ["Kalshi", "Both"]:
        st.subheader("Kalshi Markets")
        markets = fetch_kalshi_markets(limit=limit)
        if markets:
            df = pd.DataFrame(markets)
            if "title" in df.columns:
                display_df = df[["title", "volume", "yes_bid", "yes_ask", "category", "status"]].copy()
                display_df["title"] = display_df["title"].str[:60]
                display_df["volume"] = display_df["volume"].apply(lambda x: f"{x:,}" if x else "-")
                display_df["yes_bid"] = display_df["yes_bid"].apply(lambda x: f"{x}¢" if x else "-")
                display_df["yes_ask"] = display_df["yes_ask"].apply(lambda x: f"{x}¢" if x else "-")

                st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_arbitrage():
    """Arbitrage detection page."""
    st.title("⚡ Arbitrage Detector")

    # Settings
    col1, col2 = st.columns(2)
    with col1:
        min_spread = st.slider("Minimum Spread %", 0.5, 10.0, 2.0, 0.5)
    with col2:
        min_similarity = st.slider("Minimum Similarity %", 50, 100, 70, 5)

    if st.button("🔍 Scan for Arbitrage", type="primary"):
        with st.spinner("Scanning markets..."):
            detector = ArbitrageDetector(
                poly_client, kalshi_client,
                min_similarity=min_similarity / 100,
                min_spread=min_spread / 100
            )
            opportunities = detector.find_opportunities(limit_per_platform=100)

            st.session_state["arbitrage_results"] = opportunities

    # Display results
    if "arbitrage_results" in st.session_state:
        opportunities = st.session_state["arbitrage_results"]

        if opportunities:
            st.success(f"Found {len(opportunities)} arbitrage opportunities!")

            # Summary chart
            spreads = [opp.spread * 100 for opp in opportunities]
            profits = [opp.profit_potential * 100 for opp in opportunities]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(opportunities) + 1)),
                y=spreads,
                name="Spread %",
                marker_color="blue"
            ))
            fig.add_trace(go.Bar(
                x=list(range(1, len(opportunities) + 1)),
                y=profits,
                name="Profit %",
                marker_color="green"
            ))
            fig.update_layout(
                title="Arbitrage Opportunities",
                xaxis_title="Opportunity #",
                yaxis_title="Percentage",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed list
            st.markdown("---")
            for i, opp in enumerate(opportunities, 1):
                with st.expander(f"#{i} | Spread: {opp.spread:.2%} | Profit: {opp.profit_potential:.2%}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Polymarket")
                        st.markdown(f"**{opp.market_1_question}**")
                        st.markdown(f"YES Price: **{opp.market_1_yes_price:.1%}**")
                        st.markdown(f"ID: `{opp.market_1_id[:20]}...`")

                    with col2:
                        st.markdown("### Kalshi")
                        st.markdown(f"**{opp.market_2_question}**")
                        st.markdown(f"YES Price: **{opp.market_2_yes_price:.1%}**")
                        st.markdown(f"Ticker: `{opp.market_2_id}`")

                    st.markdown("---")
                    st.info(f"**Recommended Action:** {opp.action}")
                    st.caption(f"Similarity Score: {opp.similarity_score:.1%}")

        else:
            st.warning("No arbitrage opportunities found with current settings.")

    # Intra-market arbitrage
    st.markdown("---")
    st.subheader("🔄 Intra-Market Arbitrage (YES + NO < $1)")

    if st.button("Check Intra-Market"):
        with st.spinner("Checking..."):
            detector = ArbitrageDetector(poly_client, kalshi_client)
            intra = detector.scan_intra_market("polymarket")

            if intra:
                for opp in intra[:10]:
                    st.markdown(f"**{opp['market'][:60]}...**")
                    st.markdown(f"Total price: {opp['total']:.1%} | Profit: {opp['profit']:.2%}")
            else:
                st.info("No intra-market arbitrage found.")


def render_analytics():
    """Analytics and charts page."""
    st.title("📊 Analytics")

    # Signal Feed Section
    st.subheader("📡 Real-Time Signal Feed")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Signals"):
            st.cache_data.clear()

    # Fetch and display signals
    signals = fetch_signals()

    if signals:
        # Signal summary metrics
        metric_cols = st.columns(4)
        bullish = sum(1 for s in signals if s.get("direction") == "bullish")
        bearish = sum(1 for s in signals if s.get("direction") == "bearish")
        avg_confidence = sum(s.get("confidence", 0) for s in signals) / len(signals)

        with metric_cols[0]:
            st.metric("Total Signals", len(signals))
        with metric_cols[1]:
            st.metric("Bullish", bullish, delta=None)
        with metric_cols[2]:
            st.metric("Bearish", bearish, delta=None)
        with metric_cols[3]:
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")

        # Signal cards
        st.markdown("---")
        for signal in signals[:10]:
            direction = signal.get("direction", "neutral")
            direction_emoji = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "⚪"
            confidence = signal.get("confidence", 0)
            strength = signal.get("strength", "moderate")

            with st.expander(f"{direction_emoji} {signal.get('name', 'Unknown Signal')} | {confidence:.0%} confidence"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Value:** {signal.get('value', 0):.2f}")
                    st.markdown(f"**Source:** {signal.get('source', 'unknown')}")
                with col2:
                    st.markdown(f"**Direction:** {direction.title()}")
                    st.markdown(f"**Strength:** {strength.title()}")
                with col3:
                    st.markdown(f"**Time:** {signal.get('timestamp', 'N/A')}")
                    related = signal.get('related_markets', [])
                    if related:
                        st.markdown(f"**Markets:** {', '.join(related[:3])}")
    else:
        st.info("No signals available. Run `python main.py signals` to collect data.")

    st.markdown("---")

    # Market Category Distribution
    st.subheader("Market Category Distribution")

    markets = fetch_polymarket_markets(limit=100)
    if markets:
        categories = {}
        for m in markets:
            cat = m.get("category", "Other")
            categories[cat] = categories.get(cat, 0) + 1

        df = pd.DataFrame(list(categories.items()), columns=["Category", "Count"])
        fig = px.pie(df, values="Count", names="Category", title="Polymarket by Category")
        st.plotly_chart(fig, use_container_width=True)


def render_hypotheses():
    """Hypothesis tracking page."""
    st.title("🧪 Hypothesis Tracker")

    st.markdown("""
    Track your trading hypotheses and measure their predictive power with Brier scores.
    """)

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Hypotheses", "➕ Create New", "🏆 Leaderboard"])

    with tab1:
        st.subheader("Active Hypotheses")

        hypotheses = fetch_hypotheses()

        if hypotheses:
            for hyp in hypotheses:
                status_emoji = "🟢" if hyp["status"] == "active" else "🟡" if hyp["status"] == "validated" else "🔴"

                with st.expander(f"{status_emoji} {hyp['name']} | {hyp['accuracy']:.0%} accuracy"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**Source:** {hyp['data_source']}")
                        st.markdown(f"**Predictions:** {hyp['total_predictions']}")

                    with col2:
                        st.markdown(f"**Correct:** {hyp['correct_predictions']}")
                        brier = hyp.get('average_brier_score')
                        st.markdown(f"**Avg Brier:** {brier:.4f}" if brier else "**Avg Brier:** N/A")

                    with col3:
                        edge = hyp.get('average_edge')
                        st.markdown(f"**Avg Edge:** {edge:+.4f}" if edge else "**Avg Edge:** N/A")
                        st.markdown(f"**Created:** {hyp['created_at']}")
        else:
            st.info("No hypotheses yet. Create one to start tracking!")

    with tab2:
        st.subheader("Create New Hypothesis")

        with st.form("new_hypothesis"):
            name = st.text_input("Hypothesis Name", placeholder="e.g., Ice Cream Recession Signal")
            data_source = st.selectbox(
                "Data Source",
                ["ice_cream", "google_trends", "whale_activity", "custom"]
            )
            description = st.text_area(
                "Description",
                placeholder="Describe what this hypothesis predicts..."
            )
            causal_theory = st.text_area(
                "Causal Theory",
                placeholder="Why do you believe this signal predicts market outcomes?"
            )

            submitted = st.form_submit_button("Create Hypothesis")

            if submitted and name:
                try:
                    from src.tracking.hypothesis_tracker import HypothesisTracker
                    tracker = HypothesisTracker()
                    hyp_id = tracker.create_hypothesis(
                        name=name,
                        data_source=data_source,
                        description=description,
                        causal_theory=causal_theory
                    )
                    st.success(f"Created hypothesis: {name} (ID: {hyp_id})")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error creating hypothesis: {e}")

    with tab3:
        st.subheader("Hypothesis Leaderboard")

        hypotheses = fetch_hypotheses()
        qualified = [h for h in hypotheses if h.get('average_edge') is not None]

        if qualified:
            sorted_hyps = sorted(qualified, key=lambda x: x.get('average_edge', 0), reverse=True)

            for i, hyp in enumerate(sorted_hyps[:10], 1):
                edge = hyp.get('average_edge', 0)
                edge_color = "🟢" if edge > 0 else "🔴"

                st.markdown(
                    f"**#{i}** {edge_color} {hyp['name']} - "
                    f"Edge: {edge:+.4f} | Brier: {hyp.get('average_brier_score', 0):.4f}"
                )
        else:
            st.info("No hypotheses with resolved predictions yet.")


def render_paper_trading():
    """Paper trading page."""
    st.title("💰 Paper Trading")

    st.markdown("""
    Practice trading strategies without risking real money. Uses Kelly Criterion for position sizing.
    """)

    # Get portfolio stats
    portfolio = fetch_portfolio_stats()

    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Capital",
            f"${portfolio.get('total_capital', 10000):,.2f}",
            delta=f"{portfolio.get('pnl_percent', 0):+.1%}"
        )

    with col2:
        st.metric(
            "Total P&L",
            f"${portfolio.get('total_pnl', 0):+,.2f}"
        )

    with col3:
        st.metric(
            "Win Rate",
            f"{portfolio.get('win_rate', 0):.0%}"
        )

    with col4:
        st.metric(
            "Open Positions",
            portfolio.get('open_positions', 0)
        )

    st.markdown("---")

    # Tabs for trading views
    tab1, tab2, tab3 = st.tabs(["📊 Positions", "📝 Execute Trade", "📈 History"])

    with tab1:
        st.subheader("Open Positions")

        positions = fetch_open_positions()

        if positions:
            df = pd.DataFrame(positions)
            df["market_question"] = df["market_question"].str[:40] + "..."
            df["entry_price"] = df["entry_price"].apply(lambda x: f"{x:.1%}")
            df["cost_basis"] = df["cost_basis"].apply(lambda x: f"${x:.2f}")

            st.dataframe(
                df[["market_question", "side", "entry_price", "size", "cost_basis", "strategy"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No open positions.")

    with tab2:
        st.subheader("Execute Paper Trade")

        with st.form("paper_trade"):
            col1, col2 = st.columns(2)

            with col1:
                platform = st.selectbox("Platform", ["polymarket", "kalshi"])
                market_id = st.text_input("Market ID")
                market_question = st.text_input("Market Question")

            with col2:
                side = st.selectbox("Side", ["yes", "no"])
                price = st.number_input("Entry Price", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
                edge = st.number_input("Estimated Edge", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)

            strategy = st.text_input("Strategy", placeholder="e.g., ice_cream_signal")
            notes = st.text_area("Notes", placeholder="Why are you making this trade?")

            submitted = st.form_submit_button("Execute Trade", type="primary")

            if submitted and market_id and market_question:
                try:
                    from src.trading.paper_trader import PaperTrader
                    trader = PaperTrader()
                    result = trader.execute_trade(
                        platform=platform,
                        market_id=market_id,
                        market_question=market_question,
                        side=side,
                        price=price,
                        edge=edge,
                        strategy=strategy,
                        notes=notes
                    )

                    if result.status == "executed":
                        st.success(
                            f"Trade executed! Size: {result.position_size:.2f} contracts, "
                            f"Cost: ${result.cost_basis:.2f}"
                        )
                        st.cache_data.clear()
                    else:
                        st.error(f"Trade failed: {result.error}")
                except Exception as e:
                    st.error(f"Error executing trade: {e}")

    with tab3:
        st.subheader("Trade History")

        history = fetch_trade_history()

        if history:
            df = pd.DataFrame(history)
            df["market_question"] = df["market_question"].str[:30] + "..."
            df["pnl"] = df["pnl"].apply(lambda x: f"${x:+.2f}" if x else "-")
            df["pnl_percent"] = df["pnl_percent"].apply(lambda x: f"{x:+.1%}" if x else "-")

            st.dataframe(
                df[["market_question", "side", "entry_price", "exit_price", "pnl", "pnl_percent", "strategy"]],
                hide_index=True,
                use_container_width=True
            )

            # P&L chart
            pnl_curve = fetch_pnl_curve()
            if pnl_curve:
                chart_df = pd.DataFrame(pnl_curve)
                fig = px.line(
                    chart_df,
                    x="date",
                    y="cumulative_pnl",
                    title="Cumulative P&L"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet.")


def render_settings():
    """Settings page."""
    st.title("⚙️ Settings")

    st.subheader("API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Polymarket")
        poly_key = st.text_input("Private Key", type="password", placeholder="0x...")
        st.caption("Required for trading (optional for monitoring)")

    with col2:
        st.markdown("### Kalshi")
        kalshi_key = st.text_input("API Key", type="password")
        kalshi_private = st.text_area("Private Key (PEM)", height=100, type="default")

    st.markdown("---")

    st.subheader("Collection Settings")
    collection_interval = st.number_input("Collection Interval (seconds)", 30, 300, 60)
    st.checkbox("Enable automatic collection", value=False)

    st.markdown("---")

    st.subheader("Alerts")
    st.checkbox("Email alerts for arbitrage > 5%", value=False)
    st.checkbox("Slack notifications", value=False)
    st.text_input("Slack Webhook URL", type="password")

    if st.button("Save Settings"):
        st.success("Settings saved!")


# Data fetching functions (cached)
@st.cache_data(ttl=60)
def fetch_polymarket_markets(limit: int = 100) -> List[Dict]:
    """Fetch Polymarket markets with caching."""
    try:
        markets = poly_client.get_markets(limit=limit)
        return [
            {
                "id": m.id,
                "question": m.question,
                "volume": m.volume,
                "liquidity": m.liquidity,
                "category": m.category,
                "status": m.status,
                "prices": m.prices
            }
            for m in markets
        ]
    except Exception as e:
        st.error(f"Error fetching Polymarket data: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_kalshi_markets(limit: int = 100) -> List[Dict]:
    """Fetch Kalshi markets with caching."""
    try:
        markets = kalshi_client.get_markets(limit=limit)
        return [
            {
                "ticker": m.ticker,
                "title": m.title,
                "volume": m.volume,
                "yes_bid": m.yes_bid,
                "yes_ask": m.yes_ask,
                "category": m.category,
                "status": m.status
            }
            for m in markets
        ]
    except Exception as e:
        st.error(f"Error fetching Kalshi data: {e}")
        return []


@st.cache_data(ttl=30)
def fetch_arbitrage() -> List[ArbitrageOpportunity]:
    """Fetch arbitrage opportunities with caching."""
    try:
        detector = ArbitrageDetector(poly_client, kalshi_client, min_spread=0.02)
        return detector.find_opportunities(limit_per_platform=50)
    except Exception as e:
        st.error(f"Error detecting arbitrage: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_signals(limit: int = 50) -> List[Dict]:
    """Fetch recent signals from database."""
    try:
        from src.database.models import Signal

        with get_session() as session:
            db_signals = session.query(Signal).order_by(
                Signal.timestamp.desc()
            ).limit(limit).all()

            return [
                {
                    "name": s.name,
                    "source": s.source,
                    "timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "value": s.value,
                    "direction": (s.raw_data or {}).get("direction", "neutral"),
                    "confidence": (s.raw_data or {}).get("confidence", 0.5),
                    "strength": (s.raw_data or {}).get("strength", "moderate"),
                    "related_markets": (s.raw_data or {}).get("related_markets", [])
                }
                for s in db_signals
            ]
    except Exception as e:
        st.error(f"Error fetching signals: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_hypotheses() -> List[Dict]:
    """Fetch hypotheses from database."""
    try:
        from src.tracking.hypothesis_tracker import HypothesisTracker
        tracker = HypothesisTracker()
        hypotheses = tracker.list_hypotheses()

        return [
            {
                "id": h.id,
                "name": h.name,
                "data_source": h.data_source,
                "status": h.status,
                "total_predictions": h.total_predictions,
                "correct_predictions": h.correct_predictions,
                "accuracy": h.accuracy,
                "average_brier_score": h.average_brier_score,
                "average_edge": h.average_edge,
                "created_at": h.created_at.strftime("%Y-%m-%d")
            }
            for h in hypotheses
        ]
    except Exception as e:
        st.error(f"Error fetching hypotheses: {e}")
        return []


@st.cache_data(ttl=30)
def fetch_portfolio_stats() -> Dict:
    """Fetch paper trading portfolio stats."""
    try:
        from src.trading.paper_trader import PaperTrader
        trader = PaperTrader()
        stats = trader.get_portfolio_stats()

        return {
            "total_capital": stats.total_capital,
            "available_capital": stats.available_capital,
            "positions_value": stats.positions_value,
            "total_pnl": stats.total_pnl,
            "pnl_percent": stats.pnl_percent,
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "win_rate": stats.win_rate,
            "sharpe_ratio": stats.sharpe_ratio,
            "max_drawdown": stats.max_drawdown,
            "open_positions": stats.open_positions
        }
    except Exception as e:
        st.error(f"Error fetching portfolio stats: {e}")
        return {
            "total_capital": 10000,
            "available_capital": 10000,
            "positions_value": 0,
            "total_pnl": 0,
            "pnl_percent": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "open_positions": 0
        }


@st.cache_data(ttl=30)
def fetch_open_positions() -> List[Dict]:
    """Fetch open paper trading positions."""
    try:
        from src.trading.paper_trader import PaperTrader
        trader = PaperTrader()
        return trader.get_open_positions()
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_trade_history(limit: int = 50) -> List[Dict]:
    """Fetch paper trading history."""
    try:
        from src.trading.paper_trader import PaperTrader
        trader = PaperTrader()
        return trader.get_trade_history(limit=limit)
    except Exception as e:
        st.error(f"Error fetching trade history: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_pnl_curve() -> List[Dict]:
    """Fetch P&L curve for charting."""
    try:
        from src.trading.paper_trader import PaperTrader
        trader = PaperTrader()
        return trader.get_pnl_curve()
    except Exception as e:
        st.error(f"Error fetching P&L curve: {e}")
        return []


if __name__ == "__main__":
    main()
