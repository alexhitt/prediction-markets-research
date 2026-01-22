"""
Main Dashboard - Clean Polymarket/Kalshi Style Interface

A simple, professional interface for monitoring and controlling the prediction bot.
Inspired by Polymarket and Kalshi's clean, intuitive designs.

Key Sections:
1. Portfolio Overview - Your balance, P&L, positions
2. Market Opportunities - Current trading opportunities with probabilities
3. Bot Activity - Real-time view of what the bot is doing
4. News Feed - Breaking news driving market movements
5. Active Positions - Your current open bets
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import requests

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.autonomous.tournament import BotTournament, BettingTier
from src.signals.fast_news import FastNewsSignalDetector

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Prediction Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - Clean, minimal design
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark theme base */
    .stApp {
        background: #0d1117;
    }

    /* Custom header */
    .main-header {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        padding: 20px 30px;
        border-bottom: 1px solid #21262d;
        margin: -1rem -1rem 1rem -1rem;
    }

    /* Portfolio card */
    .portfolio-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Market card */
    .market-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .market-card:hover {
        border-color: #388bfd;
    }

    /* Price display */
    .price-yes {
        background: #238636;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
    }
    .price-no {
        background: #da3633;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
    }

    /* Activity feed item */
    .activity-item {
        background: #161b22;
        border-left: 3px solid #388bfd;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
    }
    .activity-item.buy {
        border-left-color: #238636;
    }
    .activity-item.sell {
        border-left-color: #da3633;
    }
    .activity-item.signal {
        border-left-color: #a371f7;
    }

    /* News item */
    .news-item {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .news-item .source {
        color: #8b949e;
        font-size: 11px;
        text-transform: uppercase;
    }
    .news-item .title {
        color: #c9d1d9;
        font-size: 14px;
        margin: 4px 0;
    }
    .news-item .time {
        color: #6e7681;
        font-size: 11px;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-dot.active { background: #238636; }
    .status-dot.warning { background: #d29922; }
    .status-dot.error { background: #da3633; }

    /* Section header */
    .section-header {
        color: #c9d1d9;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }

    /* Metric value */
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #c9d1d9;
    }
    .metric-value.positive { color: #238636; }
    .metric-value.negative { color: #da3633; }
    .metric-label {
        color: #8b949e;
        font-size: 12px;
        text-transform: uppercase;
    }

    /* Button styling */
    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #2ea043;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #21262d;
        border-radius: 6px;
        color: #8b949e;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #388bfd;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_resource
def get_tournament():
    """Get tournament instance."""
    t = BotTournament()
    if not t.bots:
        t.add_default_bots()
    return t


@st.cache_data(ttl=60)
def fetch_live_markets():
    """Fetch live markets from Polymarket."""
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"closed": "false", "limit": 20},
            timeout=15
        )
        if resp.status_code == 200:
            markets = []
            for m in resp.json():
                # Parse outcome prices
                prices_raw = m.get("outcomePrices", '["0.5", "0.5"]')
                if isinstance(prices_raw, str):
                    import json as json_mod
                    try:
                        prices = json_mod.loads(prices_raw)
                        yes_price = float(prices[0]) if prices else 0.5
                    except:
                        yes_price = 0.5
                else:
                    yes_price = float(prices_raw[0]) if prices_raw else 0.5

                markets.append({
                    "id": m.get("id", ""),
                    "question": m.get("question", "Unknown"),
                    "yes_price": yes_price,
                    "no_price": 1 - yes_price,
                    "volume": float(m.get("volume", 0) or 0),
                    "liquidity": float(m.get("liquidity", 0) or 0),
                    "category": m.get("category", ""),
                })
            return markets
    except Exception as e:
        st.error(f"Error fetching markets: {e}")
    return []


@st.cache_data(ttl=30)
def fetch_news_signals():
    """Fetch fast news signals."""
    try:
        detector = FastNewsSignalDetector()
        data = detector.fetch_data()
        return data.get("items", [])[:15]
    except Exception as e:
        return []


def get_bot_activity(tournament):
    """Get recent bot activity."""
    activities = []

    # Get recent bets
    for bet in tournament.get_today_bets()[:10]:
        bot_name = "Unknown"
        for bot in tournament.bots.values():
            if bot.id == bet['bot_id']:
                bot_name = bot.name
                break

        activities.append({
            "type": "bet",
            "bot": bot_name,
            "action": f"{bet['side'].upper()} ${bet['amount']:.0f}",
            "market": bet['market_question'][:40],
            "time": bet['placed_at'],
            "status": bet['status'],
        })

    return activities


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render the main header with portfolio summary."""
    tournament = get_tournament()

    # Calculate totals
    total_capital = sum(b.current_capital for b in tournament.bots.values())
    total_pnl = sum(b.total_pnl for b in tournament.bots.values())
    total_bets = sum(b.total_bets for b in tournament.bots.values())
    active_bots = len([b for b in tournament.bots.values() if b.status.value != "eliminated"])

    # Best performer
    best_bot = max(tournament.bots.values(), key=lambda x: x.total_pnl, default=None)

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    with col1:
        st.markdown("## 📊 Prediction Bot")
        st.markdown(f"<span class='status-dot active'></span> **{active_bots} bots active** • Last update: {datetime.now().strftime('%H:%M:%S')}", unsafe_allow_html=True)

    with col2:
        pnl_class = "positive" if total_pnl >= 0 else "negative"
        st.markdown(f"""
        <div>
            <div class="metric-label">Total P&L</div>
            <div class="metric-value {pnl_class}">${total_pnl:+,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div>
            <div class="metric-label">Total Bets</div>
            <div class="metric-value">{total_bets:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if best_bot:
            st.markdown(f"""
            <div>
                <div class="metric-label">Top Bot</div>
                <div class="metric-value" style="font-size: 18px;">{best_bot.name}</div>
            </div>
            """, unsafe_allow_html=True)

    with col5:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_market_opportunities():
    """Render live market opportunities."""
    st.markdown("<div class='section-header'>🎯 Market Opportunities</div>", unsafe_allow_html=True)

    markets = fetch_live_markets()

    if not markets:
        st.info("Loading markets...")
        return

    # Filter tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Trending", "💰 High Volume", "⚡ New"])

    with tab1:
        for market in markets[:6]:
            render_market_card(market)

    with tab2:
        sorted_markets = sorted(markets, key=lambda x: x['volume'], reverse=True)
        for market in sorted_markets[:6]:
            render_market_card(market)

    with tab3:
        for market in markets[-6:]:
            render_market_card(market)


def render_market_card(market):
    """Render a single market card."""
    yes_pct = int(market['yes_price'] * 100)
    no_pct = 100 - yes_pct

    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <span style="color: #8b949e; font-size: 11px;">{market.get('category', 'General').upper()}</span>
        </div>
        <div style="color: #c9d1d9; font-size: 14px; font-weight: 500; margin-bottom: 4px;">
            {market['question'][:80]}{'...' if len(market['question']) > 80 else ''}
        </div>
        <div style="color: #6e7681; font-size: 12px;">
            Vol: ${market['volume']:,.0f} • Liq: ${market['liquidity']:,.0f}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="background: #238636; color: white; padding: 8px 12px; border-radius: 6px; font-weight: 600;">
                Yes {yes_pct}¢
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="background: #da3633; color: white; padding: 8px 12px; border-radius: 6px; font-weight: 600;">
                No {no_pct}¢
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #21262d; margin: 12px 0;'>", unsafe_allow_html=True)


def render_bot_activity():
    """Render bot activity feed."""
    st.markdown("<div class='section-header'>🤖 Bot Activity</div>", unsafe_allow_html=True)

    tournament = get_tournament()
    activities = get_bot_activity(tournament)

    if not activities:
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #8b949e;">
            <div style="font-size: 32px; margin-bottom: 10px;">🤖</div>
            <div>No activity yet. Click "Run Simulation" to start.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶️ Run Simulation", use_container_width=True):
            tournament.simulate_round()
            st.rerun()
        return

    for activity in activities:
        status_class = "buy" if "YES" in activity['action'] else "sell" if "NO" in activity['action'] else ""

        # Parse time
        try:
            time_obj = datetime.fromisoformat(activity['time'])
            time_ago = datetime.utcnow() - time_obj
            if time_ago.total_seconds() < 60:
                time_str = "Just now"
            elif time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
        except:
            time_str = "Recently"

        status_icon = "✅" if activity['status'] == 'won' else "❌" if activity['status'] == 'lost' else "⏳"

        st.markdown(f"""
        <div class="activity-item {status_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #c9d1d9; font-weight: 600;">{activity['bot']}</span>
                    <span style="color: #8b949e;"> • {activity['action']}</span>
                </div>
                <span style="color: #6e7681; font-size: 12px;">{time_str} {status_icon}</span>
            </div>
            <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">
                {activity['market']}...
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_news_feed():
    """Render news feed panel."""
    st.markdown("<div class='section-header'>📰 News Feed</div>", unsafe_allow_html=True)

    news_items = fetch_news_signals()

    if not news_items:
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #8b949e;">
            <div style="font-size: 32px; margin-bottom: 10px;">📰</div>
            <div>Loading news...</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for item in news_items[:8]:
        # Calculate freshness
        freshness = item.freshness_score
        if freshness > 0.8:
            fresh_badge = "🔴 BREAKING"
            badge_color = "#da3633"
        elif freshness > 0.5:
            fresh_badge = "🟡 Recent"
            badge_color = "#d29922"
        else:
            fresh_badge = ""
            badge_color = "#8b949e"

        # Sentiment indicator
        if item.sentiment_hint > 0.2:
            sentiment = "📈"
        elif item.sentiment_hint < -0.2:
            sentiment = "📉"
        else:
            sentiment = ""

        source_name = item.source.split(':')[-1] if ':' in item.source else item.source

        st.markdown(f"""
        <div class="news-item">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="source">{source_name}</span>
                <span style="color: {badge_color}; font-size: 10px;">{fresh_badge}</span>
            </div>
            <div class="title">{sentiment} {item.title[:80]}{'...' if len(item.title) > 80 else ''}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="time">{int(item.age_seconds / 60)}m ago</span>
                <span style="color: #8b949e; font-size: 10px;">
                    {', '.join(item.categories[:2]) if item.categories else ''}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_bot_rankings():
    """Render bot performance rankings."""
    st.markdown("<div class='section-header'>🏆 Bot Rankings</div>", unsafe_allow_html=True)

    tournament = get_tournament()
    rankings = tournament.get_rankings()

    for bot in rankings[:5]:
        pnl_color = "#238636" if bot['total_pnl'] >= 0 else "#da3633"
        tier_emoji = ["🎮", "🪙", "💵", "💰", "🏆"][bot['tier']]

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="color: #8b949e; font-weight: 600;">#{bot['rank']}</span>
                <span style="color: #c9d1d9; font-weight: 500;">{bot['name']}</span>
                <span>{tier_emoji}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"<span style='color: {pnl_color}; font-weight: 600;'>${bot['total_pnl']:+,.0f}</span>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<span style='color: #c9d1d9;'>{bot['win_rate']:.0%}</span>", unsafe_allow_html=True)

        with col4:
            day_progress = min(100, bot['days_evaluated'] / 3 * 100)
            st.progress(day_progress / 100, text=f"Day {bot['days_evaluated']}/3")

        st.markdown("<hr style='border-color: #21262d; margin: 8px 0;'>", unsafe_allow_html=True)


def render_controls():
    """Render control panel."""
    st.markdown("<div class='section-header'>⚙️ Controls</div>", unsafe_allow_html=True)

    tournament = get_tournament()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Simulate Round", use_container_width=True):
            tournament.simulate_round()
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("📊 Daily Evaluation", use_container_width=True):
            tournament.run_daily_evaluation()
            st.cache_data.clear()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick stats
    open_bets = sum(len(b.open_bets) for b in tournament.bots.values())
    today_bets = len(tournament.get_today_bets())

    st.markdown(f"""
    <div style="background: #161b22; padding: 12px; border-radius: 8px; border: 1px solid #21262d;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #8b949e;">Open Bets</span>
            <span style="color: #c9d1d9; font-weight: 600;">{open_bets}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #8b949e;">Today's Bets</span>
            <span style="color: #c9d1d9; font-weight: 600;">{today_bets}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #8b949e;">Maturities</span>
            <span style="color: #c9d1d9; font-weight: 600;">{len(tournament.get_upcoming_maturities())}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN LAYOUT
# ============================================================================

def main():
    """Main dashboard layout."""

    # Header
    render_header()

    st.markdown("<hr style='border-color: #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

    # Main content - 3 column layout
    col_left, col_center, col_right = st.columns([2, 3, 2])

    with col_left:
        render_bot_rankings()
        st.markdown("<br>", unsafe_allow_html=True)
        render_controls()

    with col_center:
        render_market_opportunities()

    with col_right:
        render_news_feed()
        st.markdown("<br>", unsafe_allow_html=True)
        render_bot_activity()

    # Auto-refresh option
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()


if __name__ == "__main__":
    main()
