"""
Main Dashboard - Dense, Information-Rich Trading Interface

A professional, data-dense interface for monitoring prediction market bots.
Inspired by Bloomberg Terminal and Polymarket with maximum information density.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
from collections import defaultdict

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.autonomous.tournament import BotTournament, BettingTier, BotStatus
from src.signals.fast_news import FastNewsSignalDetector

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Prediction Bot Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DENSE CSS - Maximum information, minimal chrome
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 0.5rem 1rem !important; max-width: 100% !important;}

    /* Dark terminal theme */
    .stApp { background: #0a0e14; }

    /* Compact metrics row */
    .metrics-bar {
        background: linear-gradient(180deg, #12171d 0%, #0a0e14 100%);
        border-bottom: 1px solid #1a2028;
        padding: 8px 16px;
        display: flex;
        gap: 24px;
        flex-wrap: wrap;
        margin: -0.5rem -1rem 0.5rem -1rem;
    }
    .metric-item {
        display: flex;
        flex-direction: column;
        min-width: 80px;
    }
    .metric-label {
        color: #5a6270;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #e6e8eb;
        font-size: 16px;
        font-weight: 700;
        font-family: 'SF Mono', Monaco, monospace;
    }
    .metric-value.up { color: #00d26a; }
    .metric-value.down { color: #ff4757; }
    .metric-value.warn { color: #ffa502; }

    /* Section panels */
    .panel {
        background: #12171d;
        border: 1px solid #1a2028;
        border-radius: 4px;
        margin-bottom: 8px;
        overflow: hidden;
    }
    .panel-header {
        background: #161c24;
        padding: 6px 10px;
        border-bottom: 1px solid #1a2028;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .panel-title {
        color: #8b95a5;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .panel-body {
        padding: 8px;
    }

    /* Compact table rows */
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 8px;
        border-bottom: 1px solid #1a2028;
        font-size: 11px;
    }
    .data-row:hover { background: #1a2028; }
    .data-row:last-child { border-bottom: none; }

    /* Bot status row */
    .bot-row {
        display: grid;
        grid-template-columns: 20px 100px 60px 50px 50px 40px 60px;
        gap: 8px;
        padding: 5px 8px;
        border-bottom: 1px solid #1a2028;
        font-size: 11px;
        align-items: center;
    }
    .bot-row:hover { background: #1a2028; }

    /* Market row */
    .market-row {
        padding: 6px 8px;
        border-bottom: 1px solid #1a2028;
    }
    .market-row:hover { background: #1a2028; }
    .market-question {
        color: #c9d1d9;
        font-size: 11px;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .market-meta {
        display: flex;
        gap: 12px;
        align-items: center;
    }

    /* Probability bar */
    .prob-bar {
        height: 4px;
        background: #1a2028;
        border-radius: 2px;
        overflow: hidden;
        width: 60px;
    }
    .prob-fill {
        height: 100%;
        background: linear-gradient(90deg, #00d26a 0%, #00a854 100%);
        border-radius: 2px;
    }

    /* Price chips */
    .price-chip {
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .price-yes { background: #0d3321; color: #00d26a; }
    .price-no { background: #3d1a1a; color: #ff4757; }

    /* News item */
    .news-row {
        padding: 5px 8px;
        border-bottom: 1px solid #1a2028;
        font-size: 10px;
    }
    .news-row:hover { background: #1a2028; }
    .news-source {
        color: #5a6270;
        font-size: 9px;
        text-transform: uppercase;
    }
    .news-title {
        color: #c9d1d9;
        font-size: 11px;
        line-height: 1.3;
    }
    .news-time { color: #5a6270; font-size: 9px; }
    .news-fresh { color: #ff4757; font-weight: 600; }
    .news-recent { color: #ffa502; }

    /* Activity item */
    .activity-row {
        display: grid;
        grid-template-columns: 80px 40px 1fr 50px;
        gap: 8px;
        padding: 4px 8px;
        border-bottom: 1px solid #1a2028;
        font-size: 10px;
        align-items: center;
    }

    /* Position row */
    .position-row {
        display: grid;
        grid-template-columns: 1fr 50px 50px 60px;
        gap: 8px;
        padding: 5px 8px;
        border-bottom: 1px solid #1a2028;
        font-size: 10px;
        align-items: center;
    }

    /* Signal indicator */
    .signal-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 4px;
    }
    .signal-strong { background: #00d26a; box-shadow: 0 0 4px #00d26a; }
    .signal-medium { background: #ffa502; }
    .signal-weak { background: #5a6270; }

    /* Status badges */
    .badge {
        font-size: 8px;
        padding: 1px 4px;
        border-radius: 2px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .badge-eval { background: #1a3a5c; color: #58a6ff; }
    .badge-active { background: #0d3321; color: #00d26a; }
    .badge-elim { background: #3d1a1a; color: #ff4757; }
    .badge-promo { background: #3d2e0a; color: #ffa502; }

    /* Tier badges */
    .tier { font-size: 9px; }
    .tier-sim { color: #5a6270; }
    .tier-micro { color: #58a6ff; }
    .tier-small { color: #00d26a; }
    .tier-med { color: #ffa502; }
    .tier-large { color: #ff4757; }

    /* Risk bar */
    .risk-bar {
        height: 3px;
        background: #1a2028;
        border-radius: 2px;
        overflow: hidden;
    }
    .risk-fill {
        height: 100%;
        transition: width 0.3s;
    }
    .risk-low { background: #00d26a; }
    .risk-med { background: #ffa502; }
    .risk-high { background: #ff4757; }

    /* Win streak */
    .streak { font-family: 'SF Mono', Monaco, monospace; font-size: 9px; }
    .streak-win { color: #00d26a; }
    .streak-loss { color: #ff4757; }

    /* Scrollable panel */
    .scroll-panel {
        max-height: 300px;
        overflow-y: auto;
    }
    .scroll-panel::-webkit-scrollbar { width: 4px; }
    .scroll-panel::-webkit-scrollbar-track { background: #12171d; }
    .scroll-panel::-webkit-scrollbar-thumb { background: #2a3441; border-radius: 2px; }

    /* Live indicator */
    .live-dot {
        width: 6px;
        height: 6px;
        background: #00d26a;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Compact buttons */
    .stButton > button {
        background: #1a2028 !important;
        color: #8b95a5 !important;
        border: 1px solid #2a3441 !important;
        border-radius: 3px !important;
        padding: 4px 12px !important;
        font-size: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: #2a3441 !important;
        color: #e6e8eb !important;
    }

    /* Hide default streamlit elements */
    div[data-testid="stMetricValue"] { font-size: 14px !important; }
    .stProgress > div > div { height: 3px !important; }
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


@st.cache_data(ttl=30)
def fetch_live_markets():
    """Fetch live markets from Polymarket."""
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"closed": "false", "limit": 30},
            timeout=10
        )
        if resp.status_code == 200:
            markets = []
            for m in resp.json():
                prices_raw = m.get("outcomePrices", '["0.5", "0.5"]')
                if isinstance(prices_raw, str):
                    try:
                        prices = json.loads(prices_raw)
                        yes_price = float(prices[0]) if prices else 0.5
                    except:
                        yes_price = 0.5
                else:
                    yes_price = float(prices_raw[0]) if prices_raw else 0.5

                markets.append({
                    "id": m.get("id", ""),
                    "question": m.get("question", "Unknown"),
                    "yes_price": yes_price,
                    "volume": float(m.get("volume", 0) or 0),
                    "liquidity": float(m.get("liquidity", 0) or 0),
                    "category": m.get("category", ""),
                    "volume_24h": float(m.get("volume24hr", 0) or 0),
                })
            return sorted(markets, key=lambda x: x['volume_24h'], reverse=True)
    except:
        pass
    return []


@st.cache_data(ttl=20)
def fetch_news():
    """Fetch news signals."""
    try:
        detector = FastNewsSignalDetector()
        data = detector.fetch_data()
        return data.get("items", [])[:20]
    except:
        return []


def load_agent_state():
    """Load agent state from file."""
    try:
        with open("data/agent_state.json") as f:
            return json.load(f)
    except:
        return {}


def load_risk_state():
    """Load risk manager state."""
    try:
        with open("data/risk_state.json") as f:
            return json.load(f)
    except:
        return {
            "is_halted": False,
            "drawdown": 0,
            "daily_pnl": 0,
            "exposure": 0,
            "trades_today": 0
        }


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_top_metrics():
    """Render top metrics bar."""
    tournament = get_tournament()
    agent_state = load_agent_state()
    risk_state = load_risk_state()

    # Calculate metrics
    total_capital = sum(b.current_capital for b in tournament.bots.values())
    initial_capital = sum(1000 for _ in tournament.bots.values())  # Each bot starts with 1000
    total_pnl = total_capital - initial_capital
    pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

    total_bets = sum(b.total_bets for b in tournament.bots.values())
    winning_bets = sum(b.winning_bets for b in tournament.bots.values())
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0

    open_bets = sum(len(b.open_bets) for b in tournament.bots.values())
    active_bots = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])

    # Agent cycle info
    cycle = agent_state.get("cycle", 0)

    pnl_class = "up" if total_pnl >= 0 else "down"

    st.markdown(f"""
    <div class="metrics-bar">
        <div class="metric-item">
            <span class="metric-label">Capital</span>
            <span class="metric-value">${total_capital:,.0f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">P&L</span>
            <span class="metric-value {pnl_class}">{'+' if total_pnl >= 0 else ''}{total_pnl:,.0f} ({pnl_pct:+.1f}%)</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Win Rate</span>
            <span class="metric-value">{win_rate:.1f}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Total Bets</span>
            <span class="metric-value">{total_bets}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Open</span>
            <span class="metric-value">{open_bets}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Bots</span>
            <span class="metric-value">{active_bots}/7</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Cycle</span>
            <span class="metric-value">#{cycle}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Risk</span>
            <span class="metric-value {'down' if risk_state.get('is_halted') else 'up'}">{'HALTED' if risk_state.get('is_halted') else 'OK'}</span>
        </div>
        <div class="metric-item" style="margin-left: auto;">
            <span class="metric-label"><span class="live-dot"></span> Live</span>
            <span class="metric-value" style="font-size: 11px;">{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_bot_panel():
    """Render bot rankings panel."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    # Header row
    header_html = """
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">🤖 Bot Performance</span>
            <span style="color: #5a6270; font-size: 9px;">3-DAY EVAL</span>
        </div>
        <div class="panel-body" style="padding: 0;">
            <div class="bot-row" style="background: #161c24; font-weight: 600; color: #5a6270;">
                <span>#</span>
                <span>Bot</span>
                <span>P&L</span>
                <span>W/L</span>
                <span>WR%</span>
                <span>Day</span>
                <span>Status</span>
            </div>
    """

    rows_html = ""
    for bot in rankings[:7]:
        pnl_class = "up" if bot['total_pnl'] >= 0 else "down"

        # Status badge
        status = bot['status']
        if status == 'evaluating':
            badge = '<span class="badge badge-eval">EVAL</span>'
        elif status == 'active':
            badge = '<span class="badge badge-active">ACTIVE</span>'
        elif status == 'eliminated':
            badge = '<span class="badge badge-elim">ELIM</span>'
        elif status == 'promoted':
            badge = '<span class="badge badge-promo">PROMO</span>'
        else:
            badge = f'<span class="badge">{status[:4].upper()}</span>'

        # Tier indicator
        tier = bot['tier']
        tier_names = ['SIM', 'MICRO', 'SMALL', 'MED', 'LARGE']
        tier_classes = ['tier-sim', 'tier-micro', 'tier-small', 'tier-med', 'tier-large']
        tier_html = f'<span class="tier {tier_classes[tier]}">{tier_names[tier]}</span>'

        # Win/Loss with streak
        streak_html = ""
        total_bets = bot.get('total_bets', 0)
        winning_bets = bot.get('winning_bets', 0)
        losing_bets = total_bets - winning_bets
        if total_bets > 0:
            if winning_bets > losing_bets:
                streak_html = f'<span class="streak streak-win">↑</span>'
            elif losing_bets > winning_bets:
                streak_html = f'<span class="streak streak-loss">↓</span>'

        rows_html += f"""
        <div class="bot-row">
            <span style="color: #5a6270;">{bot['rank']}</span>
            <span style="color: #c9d1d9;">{bot['name'][:12]}</span>
            <span class="metric-value {pnl_class}" style="font-size: 11px;">${bot['total_pnl']:+,.0f}</span>
            <span style="color: #8b95a5;">{winning_bets}/{losing_bets} {streak_html}</span>
            <span style="color: #8b95a5;">{bot.get('win_rate', 0):.0f}%</span>
            <span style="color: #5a6270;">{bot.get('days_evaluated', 0)}/3</span>
            <span>{badge}</span>
        </div>
        """

    st.html(header_html + rows_html + "</div></div>")


def render_markets_panel():
    """Render live markets panel."""
    markets = fetch_live_markets()

    html = """
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">📈 Live Markets</span>
            <span style="color: #5a6270; font-size: 9px;">POLYMARKET</span>
        </div>
        <div class="panel-body scroll-panel" style="padding: 0; max-height: 400px;">
    """

    for m in markets[:15]:
        yes_pct = int(m['yes_price'] * 100)
        no_pct = 100 - yes_pct
        vol_k = m['volume'] / 1000
        vol_24h_k = m['volume_24h'] / 1000

        # Determine signal strength based on volume
        if vol_24h_k > 100:
            signal_class = "signal-strong"
        elif vol_24h_k > 20:
            signal_class = "signal-medium"
        else:
            signal_class = "signal-weak"

        question = m['question'][:70] + ('...' if len(m['question']) > 70 else '')
        category = (m.get('category') or 'general')[:10].upper()

        html += f"""
        <div class="market-row">
            <div class="market-question">
                <span class="signal-dot {signal_class}"></span>
                {question}
            </div>
            <div class="market-meta">
                <span style="color: #5a6270; font-size: 9px;">{category}</span>
                <div class="prob-bar"><div class="prob-fill" style="width: {yes_pct}%;"></div></div>
                <span class="price-chip price-yes">{yes_pct}¢</span>
                <span class="price-chip price-no">{no_pct}¢</span>
                <span style="color: #5a6270; font-size: 9px;">${vol_k:.0f}K</span>
                <span style="color: #8b95a5; font-size: 9px;">24h: ${vol_24h_k:.0f}K</span>
            </div>
        </div>
        """

    html += "</div></div>"
    st.html(html)


def render_news_panel():
    """Render news feed panel."""
    news = fetch_news()

    html = """
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">📰 News Feed</span>
            <span style="color: #5a6270; font-size: 9px;">LIVE</span>
        </div>
        <div class="panel-body scroll-panel" style="padding: 0; max-height: 300px;">
    """

    for item in news[:12]:
        age_min = int(item.age_seconds / 60)
        freshness = item.freshness_score

        if freshness > 0.8:
            time_class = "news-fresh"
            time_str = f"🔴 {age_min}m"
        elif freshness > 0.5:
            time_class = "news-recent"
            time_str = f"{age_min}m"
        else:
            time_class = ""
            if age_min > 60:
                time_str = f"{age_min // 60}h"
            else:
                time_str = f"{age_min}m"

        source = item.source.split(':')[-1][:8].upper()
        title = item.title[:80] + ('...' if len(item.title) > 80 else '')

        # Sentiment indicator
        if item.sentiment_hint > 0.2:
            sentiment = "📈"
        elif item.sentiment_hint < -0.2:
            sentiment = "📉"
        else:
            sentiment = ""

        cats = ', '.join(item.categories[:2]) if item.categories else ''

        html += f"""
        <div class="news-row">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span class="news-source">{source}</span>
                <span class="news-time {time_class}">{time_str}</span>
            </div>
            <div class="news-title">{sentiment} {title}</div>
            <div style="color: #5a6270; font-size: 9px;">{cats}</div>
        </div>
        """

    html += "</div></div>"
    st.html(html)


def render_activity_panel():
    """Render bot activity panel."""
    tournament = get_tournament()
    today_bets = tournament.get_today_bets()[:10]

    html = """
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">⚡ Recent Activity</span>
            <span style="color: #5a6270; font-size: 9px;">TODAY</span>
        </div>
        <div class="panel-body scroll-panel" style="padding: 0; max-height: 200px;">
    """

    if not today_bets:
        html += """
        <div style="padding: 20px; text-align: center; color: #5a6270; font-size: 11px;">
            No activity yet. Run simulation to start.
        </div>
        """
    else:
        for bet in today_bets:
            # Find bot name
            bot_name = "Unknown"
            for bot in tournament.bots.values():
                if bot.id == bet['bot_id']:
                    bot_name = bot.name[:10]
                    break

            side_class = "up" if bet['side'] == 'yes' else "down"
            side_str = bet['side'].upper()

            status_icon = "✅" if bet['status'] == 'won' else "❌" if bet['status'] == 'lost' else "⏳"

            market_short = bet['market_question'][:35] + '...'

            try:
                time_obj = datetime.fromisoformat(bet['placed_at'])
                time_str = time_obj.strftime('%H:%M')
            except:
                time_str = "--:--"

            html += f"""
            <div class="activity-row">
                <span style="color: #8b95a5;">{bot_name}</span>
                <span class="metric-value {side_class}" style="font-size: 10px;">{side_str}</span>
                <span style="color: #c9d1d9; font-size: 10px;">{market_short}</span>
                <span style="color: #5a6270;">{time_str} {status_icon}</span>
            </div>
            """

    html += "</div></div>"
    st.html(html)


def render_positions_panel():
    """Render open positions panel."""
    tournament = get_tournament()

    html = """
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">📊 Open Positions</span>
            <span style="color: #5a6270; font-size: 9px;">PENDING</span>
        </div>
        <div class="panel-body scroll-panel" style="padding: 0; max-height: 180px;">
    """

    # Collect all open bets
    open_positions = []
    for bot in tournament.bots.values():
        for bet in bot.open_bets:
            open_positions.append({
                'bot': bot.name[:8],
                'market': bet.market_question[:30],
                'side': bet.side,
                'amount': bet.amount,
                'maturity': bet.matures_at.isoformat() if bet.matures_at else None,
            })

    if not open_positions:
        html += """
        <div style="padding: 20px; text-align: center; color: #5a6270; font-size: 11px;">
            No open positions
        </div>
        """
    else:
        for pos in open_positions[:8]:
            side_class = "up" if pos['side'] == 'yes' else "down"

            # Calculate time to maturity
            try:
                mat_time = datetime.fromisoformat(pos['maturity'])
                time_left = mat_time - datetime.utcnow()
                if time_left.total_seconds() > 0:
                    hours = int(time_left.total_seconds() / 3600)
                    mins = int((time_left.total_seconds() % 3600) / 60)
                    maturity_str = f"{hours}h {mins}m"
                else:
                    maturity_str = "DONE"
            except:
                maturity_str = "TBD"

            html += f"""
            <div class="position-row">
                <span style="color: #c9d1d9;">{pos['market']}...</span>
                <span class="metric-value {side_class}" style="font-size: 10px;">{pos['side'].upper()}</span>
                <span style="color: #8b95a5;">${pos['amount']:.0f}</span>
                <span style="color: #5a6270;">{maturity_str}</span>
            </div>
            """

    html += "</div></div>"
    st.html(html)


def render_signals_panel():
    """Render active signals panel."""
    news = fetch_news()[:5]

    html = """
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">🎯 Active Signals</span>
        </div>
        <div class="panel-body" style="padding: 4px 8px;">
    """

    # Generate signal summary from news
    signals = []
    for item in news:
        if item.freshness_score > 0.7:
            strength = "STRONG"
            color = "#00d26a"
        elif item.freshness_score > 0.4:
            strength = "MED"
            color = "#ffa502"
        else:
            strength = "WEAK"
            color = "#5a6270"

        if item.categories:
            signals.append({
                'category': item.categories[0],
                'strength': strength,
                'color': color,
                'sentiment': item.sentiment_hint
            })

    # Deduplicate by category
    seen = set()
    unique_signals = []
    for s in signals:
        if s['category'] not in seen:
            seen.add(s['category'])
            unique_signals.append(s)

    if not unique_signals:
        html += '<div style="color: #5a6270; font-size: 10px;">Scanning for signals...</div>'
    else:
        for sig in unique_signals[:4]:
            sent_icon = "↑" if sig['sentiment'] > 0 else "↓" if sig['sentiment'] < 0 else "→"
            html += f"""
            <div style="display: flex; justify-content: space-between; padding: 2px 0; font-size: 10px;">
                <span style="color: #8b95a5;">{sig['category']}</span>
                <span style="color: {sig['color']};">{sig['strength']} {sent_icon}</span>
            </div>
            """

    html += "</div></div>"
    st.html(html)


def render_risk_panel():
    """Render risk metrics panel."""
    risk_state = load_risk_state()
    tournament = get_tournament()

    # Calculate exposure
    total_capital = sum(b.current_capital for b in tournament.bots.values())
    open_amount = sum(
        sum(bet.amount for bet in b.open_bets)
        for b in tournament.bots.values()
    )
    exposure_pct = (open_amount / total_capital * 100) if total_capital > 0 else 0

    # Determine risk level
    if exposure_pct > 40 or risk_state.get('is_halted'):
        risk_class = "risk-high"
        risk_level = "HIGH"
    elif exposure_pct > 20:
        risk_class = "risk-med"
        risk_level = "MEDIUM"
    else:
        risk_class = "risk-low"
        risk_level = "LOW"

    html = f"""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">⚠️ Risk Status</span>
            <span style="color: {'#ff4757' if risk_level == 'HIGH' else '#ffa502' if risk_level == 'MEDIUM' else '#00d26a'}; font-size: 9px;">{risk_level}</span>
        </div>
        <div class="panel-body" style="padding: 8px;">
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 2px;">
                    <span style="color: #5a6270;">Exposure</span>
                    <span style="color: #8b95a5;">{exposure_pct:.1f}%</span>
                </div>
                <div class="risk-bar">
                    <div class="risk-fill {risk_class}" style="width: {min(100, exposure_pct * 2)}%;"></div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                <span style="color: #5a6270;">Open $</span>
                <span style="color: #8b95a5;">${open_amount:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                <span style="color: #5a6270;">Halted</span>
                <span style="color: {'#ff4757' if risk_state.get('is_halted') else '#00d26a'};">{'YES' if risk_state.get('is_halted') else 'NO'}</span>
            </div>
        </div>
    </div>
    """
    st.html(html)


def render_controls():
    """Render control buttons."""
    tournament = get_tournament()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶ Simulate", use_container_width=True):
            tournament.simulate_round()
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("📊 Evaluate", use_container_width=True):
            tournament.run_daily_evaluation()
            st.cache_data.clear()
            st.rerun()

    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with col4:
        auto = st.checkbox("Auto", value=False, help="Auto-refresh every 15s")
        if auto:
            time.sleep(15)
            st.rerun()


# ============================================================================
# CHART VISUALIZATIONS
# ============================================================================

def render_pnl_chart():
    """Render bot P&L comparison chart."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings:
        return

    # Create data for chart
    names = [r['name'][:12] for r in rankings]
    pnls = [r['total_pnl'] for r in rankings]
    colors = ['#00d26a' if p >= 0 else '#ff4757' for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=pnls,
        marker_color=colors,
        text=[f"${p:+,.0f}" for p in pnls],
        textposition='outside',
        textfont=dict(size=10, color='#8b95a5'),
    ))

    fig.update_layout(
        title=dict(text="Bot P&L Comparison", font=dict(size=12, color='#8b95a5')),
        plot_bgcolor='#12171d',
        paper_bgcolor='#12171d',
        font=dict(color='#8b95a5', size=10),
        margin=dict(l=40, r=20, t=40, b=40),
        height=200,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a2028',
            zeroline=True,
            zerolinecolor='#2a3441',
            tickformat='$,.0f',
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_activity_timeline():
    """Render bet activity timeline."""
    tournament = get_tournament()
    all_bets = tournament.all_bets[-50:]  # Last 50 bets

    if not all_bets:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📊 Activity Timeline</span>
            </div>
            <div class="panel-body" style="text-align: center; color: #5a6270; padding: 20px;">
                No activity data yet
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Group bets by hour
    hour_counts = defaultdict(lambda: {'won': 0, 'lost': 0, 'pending': 0})
    for bet in all_bets:
        hour = bet.placed_at.strftime('%H:00')
        status = bet.status
        if status == 'won':
            hour_counts[hour]['won'] += 1
        elif status == 'lost':
            hour_counts[hour]['lost'] += 1
        else:
            hour_counts[hour]['pending'] += 1

    hours = sorted(hour_counts.keys())
    won_counts = [hour_counts[h]['won'] for h in hours]
    lost_counts = [hour_counts[h]['lost'] for h in hours]
    pending_counts = [hour_counts[h]['pending'] for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Won', x=hours, y=won_counts, marker_color='#00d26a'))
    fig.add_trace(go.Bar(name='Lost', x=hours, y=lost_counts, marker_color='#ff4757'))
    fig.add_trace(go.Bar(name='Pending', x=hours, y=pending_counts, marker_color='#ffa502'))

    fig.update_layout(
        title=dict(text="Activity by Hour", font=dict(size=12, color='#8b95a5')),
        barmode='stack',
        plot_bgcolor='#12171d',
        paper_bgcolor='#12171d',
        font=dict(color='#8b95a5', size=10),
        margin=dict(l=40, r=20, t=40, b=40),
        height=180,
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor='#1a2028'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=9)
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_win_rate_chart():
    """Render win rate by bot chart."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings or all(r['total_bets'] == 0 for r in rankings):
        return

    names = [r['name'][:10] for r in rankings if r['total_bets'] > 0]
    win_rates = [r['win_rate'] * 100 for r in rankings if r['total_bets'] > 0]
    total_bets = [r['total_bets'] for r in rankings if r['total_bets'] > 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=names,
        y=win_rates,
        mode='markers+lines',
        marker=dict(
            size=[min(20, max(8, b/2)) for b in total_bets],  # Size by bet count
            color=win_rates,
            colorscale=[[0, '#ff4757'], [0.5, '#ffa502'], [1, '#00d26a']],
            showscale=False,
        ),
        line=dict(color='#2a3441', width=1),
        text=[f"{wr:.0f}% ({tb} bets)" for wr, tb in zip(win_rates, total_bets)],
        hoverinfo='text',
    ))

    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="#5a6270", opacity=0.5)

    fig.update_layout(
        title=dict(text="Win Rate by Bot", font=dict(size=12, color='#8b95a5')),
        plot_bgcolor='#12171d',
        paper_bgcolor='#12171d',
        font=dict(color='#8b95a5', size=10),
        margin=dict(l=40, r=20, t=40, b=40),
        height=180,
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a2028',
            range=[0, 100],
            ticksuffix='%',
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_cumulative_pnl_chart():
    """Render cumulative P&L over time."""
    tournament = get_tournament()
    all_bets = tournament.all_bets

    if len(all_bets) < 5:
        return

    # Sort bets by time and calculate cumulative P&L
    sorted_bets = sorted(all_bets, key=lambda x: x.placed_at)
    cumulative_pnl = []
    running_total = 0

    for bet in sorted_bets:
        if bet.status in ['won', 'lost']:
            running_total += bet.pnl
        cumulative_pnl.append({
            'time': bet.placed_at,
            'pnl': running_total,
            'status': bet.status,
        })

    df = pd.DataFrame(cumulative_pnl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['pnl'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00d26a' if df['pnl'].iloc[-1] >= 0 else '#ff4757', width=2),
        fillcolor='rgba(0, 210, 106, 0.1)' if df['pnl'].iloc[-1] >= 0 else 'rgba(255, 71, 87, 0.1)',
    ))

    fig.update_layout(
        title=dict(text="Cumulative P&L", font=dict(size=12, color='#8b95a5')),
        plot_bgcolor='#12171d',
        paper_bgcolor='#12171d',
        font=dict(color='#8b95a5', size=10),
        margin=dict(l=40, r=20, t=40, b=40),
        height=180,
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a2028',
            zeroline=True,
            zerolinecolor='#2a3441',
            tickformat='$,.0f',
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_bet_distribution_chart():
    """Render bet side distribution (yes vs no)."""
    tournament = get_tournament()
    all_bets = tournament.all_bets

    if not all_bets:
        return

    yes_count = sum(1 for b in all_bets if b.side == 'yes')
    no_count = sum(1 for b in all_bets if b.side == 'no')

    fig = go.Figure(data=[go.Pie(
        labels=['YES Bets', 'NO Bets'],
        values=[yes_count, no_count],
        hole=0.6,
        marker_colors=['#00d26a', '#ff4757'],
        textinfo='percent',
        textfont=dict(size=11, color='white'),
    )])

    fig.update_layout(
        title=dict(text="Bet Distribution", font=dict(size=12, color='#8b95a5')),
        plot_bgcolor='#12171d',
        paper_bgcolor='#12171d',
        font=dict(color='#8b95a5', size=10),
        margin=dict(l=20, r=20, t=40, b=20),
        height=160,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ============================================================================
# MAIN LAYOUT
# ============================================================================

def main():
    """Main dashboard layout - dense 4-column grid with charts."""

    # Top metrics bar
    render_top_metrics()

    # Control bar
    render_controls()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Main grid - 4 columns
    col1, col2, col3, col4 = st.columns([1.2, 1.5, 1.2, 1])

    with col1:
        render_bot_panel()
        render_signals_panel()

    with col2:
        render_markets_panel()

    with col3:
        render_news_panel()
        render_activity_panel()

    with col4:
        render_positions_panel()
        render_risk_panel()

    # Charts section
    st.markdown("<hr style='border-color: #1a2028; margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title' style='margin-bottom: 8px;'>📈 ANALYTICS</div>", unsafe_allow_html=True)

    # Charts row 1 - Performance charts
    chart_col1, chart_col2, chart_col3 = st.columns([1.2, 1, 0.8])

    with chart_col1:
        render_pnl_chart()

    with chart_col2:
        render_cumulative_pnl_chart()

    with chart_col3:
        render_bet_distribution_chart()

    # Charts row 2 - Activity charts
    chart_col4, chart_col5 = st.columns([1, 1])

    with chart_col4:
        render_activity_timeline()

    with chart_col5:
        render_win_rate_chart()


if __name__ == "__main__":
    main()
