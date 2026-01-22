"""
Prediction Markets Command Center
Sharp, data-dense trading interface with real-time monitoring.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.autonomous.tournament import BotTournament, BettingTier, BotStatus
from src.signals.fast_news import FastNewsSignalDetector

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Command Center",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# COMMAND CENTER CSS - Sharp, dense, professional
# ============================================================================
st.markdown("""
<style>
    /* Reset and base */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 0.5rem 0.8rem !important; max-width: 100% !important;}

    /* Dark command center theme */
    .stApp {
        background: #0a0a0f;
        color: #e0e0e0;
    }

    /* Monospace for data */
    * {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
    }

    /* Top bar */
    .top-bar {
        background: linear-gradient(180deg, #12121a 0%, #0d0d12 100%);
        border: 1px solid #1e1e2e;
        border-radius: 2px;
        padding: 12px 20px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .system-title {
        font-size: 14px;
        font-weight: 600;
        color: #00ff88;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .system-status {
        display: flex;
        gap: 30px;
        align-items: center;
    }
    .stat-block {
        text-align: right;
    }
    .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        font-variant-numeric: tabular-nums;
    }
    .stat-value.positive { color: #00ff88; }
    .stat-value.negative { color: #ff4757; }
    .stat-label {
        font-size: 9px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .live-badge {
        background: #ff4757;
        color: #fff;
        padding: 3px 8px;
        border-radius: 2px;
        font-size: 9px;
        font-weight: 700;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Panel styling */
    .panel {
        background: #0d0d12;
        border: 1px solid #1a1a24;
        border-radius: 2px;
        margin-bottom: 8px;
    }
    .panel-header {
        background: #12121a;
        padding: 8px 12px;
        border-bottom: 1px solid #1a1a24;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .panel-title {
        font-size: 10px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .panel-badge {
        font-size: 9px;
        color: #00ff88;
        background: rgba(0,255,136,0.1);
        padding: 2px 6px;
        border-radius: 2px;
    }
    .panel-content {
        padding: 8px;
    }

    /* Data table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
    }
    .data-table th {
        text-align: left;
        padding: 6px 8px;
        color: #555;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 9px;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #1a1a24;
    }
    .data-table td {
        padding: 6px 8px;
        border-bottom: 1px solid #111118;
        color: #ccc;
        font-variant-numeric: tabular-nums;
    }
    .data-table tr:hover {
        background: #111118;
    }
    .rank-1 { color: #ffd700 !important; font-weight: 700; }
    .rank-2 { color: #c0c0c0 !important; }
    .rank-3 { color: #cd7f32 !important; }
    .positive { color: #00ff88 !important; }
    .negative { color: #ff4757 !important; }
    .neutral { color: #888 !important; }
    .hot { color: #ff9f43 !important; }

    /* Mini bar chart in table */
    .mini-bar {
        display: inline-block;
        height: 4px;
        background: #1a1a24;
        border-radius: 1px;
        width: 60px;
        position: relative;
    }
    .mini-bar-fill {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        border-radius: 1px;
    }
    .mini-bar-fill.win { background: #00ff88; }
    .mini-bar-fill.loss { background: #ff4757; }

    /* Market row */
    .market-row {
        padding: 6px 8px;
        border-bottom: 1px solid #111118;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
    }
    .market-row:hover {
        background: #111118;
    }
    .market-q {
        color: #ccc;
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-right: 12px;
    }
    .market-prices {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    .price-tag {
        padding: 2px 6px;
        border-radius: 2px;
        font-size: 10px;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .price-yes {
        background: rgba(0,255,136,0.15);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    .price-no {
        background: rgba(255,71,87,0.15);
        color: #ff4757;
        border: 1px solid rgba(255,71,87,0.3);
    }
    .market-vol {
        color: #555;
        font-size: 9px;
        width: 60px;
        text-align: right;
    }

    /* News item */
    .news-row {
        padding: 5px 8px;
        border-bottom: 1px solid #111118;
        font-size: 10px;
    }
    .news-row:hover {
        background: #111118;
    }
    .news-title {
        color: #bbb;
        line-height: 1.3;
        margin-bottom: 2px;
    }
    .news-meta {
        color: #444;
        font-size: 9px;
        display: flex;
        gap: 8px;
    }
    .news-source {
        color: #666;
        text-transform: uppercase;
    }
    .news-time {
        color: #555;
    }
    .news-breaking {
        color: #ff4757;
        font-weight: 600;
    }
    .news-sentiment-up { color: #00ff88; }
    .news-sentiment-down { color: #ff4757; }

    /* Compact buttons */
    .stButton > button {
        background: #1a1a24 !important;
        color: #888 !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 2px !important;
        padding: 8px 16px !important;
        font-size: 10px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #2a2a3a !important;
        color: #fff !important;
        border-color: #3a3a4a !important;
    }

    /* Checkbox */
    .stCheckbox label {
        font-size: 10px !important;
        color: #666 !important;
    }

    /* Metrics mini cards */
    .metric-mini {
        background: #0d0d12;
        border: 1px solid #1a1a24;
        border-radius: 2px;
        padding: 10px 12px;
        text-align: center;
    }
    .metric-mini-value {
        font-size: 20px;
        font-weight: 700;
        color: #fff;
        font-variant-numeric: tabular-nums;
    }
    .metric-mini-label {
        font-size: 8px;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Strategy compact */
    .strat-row {
        padding: 6px 8px;
        border-bottom: 1px solid #111118;
        font-size: 10px;
    }
    .strat-name {
        color: #ccc;
        font-weight: 500;
        margin-bottom: 2px;
    }
    .strat-desc {
        color: #555;
        font-size: 9px;
    }
    .strat-tags {
        margin-top: 3px;
    }
    .strat-tag {
        display: inline-block;
        background: rgba(0,255,136,0.1);
        color: #00ff88;
        padding: 1px 4px;
        border-radius: 2px;
        font-size: 8px;
        margin-right: 4px;
    }

    /* Activity indicator */
    .activity-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .activity-active { background: #00ff88; }
    .activity-idle { background: #555; }
    .activity-warning { background: #ff9f43; }

    /* Scrollable panel */
    .scroll-panel {
        max-height: 280px;
        overflow-y: auto;
    }
    .scroll-panel::-webkit-scrollbar {
        width: 4px;
    }
    .scroll-panel::-webkit-scrollbar-track {
        background: #0d0d12;
    }
    .scroll-panel::-webkit-scrollbar-thumb {
        background: #2a2a3a;
        border-radius: 2px;
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


@st.cache_data(ttl=30)
def fetch_live_markets():
    """Fetch live markets from Polymarket."""
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"closed": "false", "limit": 25},
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
                    "question": m.get("question", "Unknown"),
                    "yes_price": yes_price,
                    "volume": float(m.get("volume", 0) or 0),
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


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_top_bar():
    """Render command center top bar."""
    tournament = get_tournament()

    total_capital = sum(b.current_capital for b in tournament.bots.values())
    initial = len(tournament.bots) * 1000
    total_pnl = total_capital - initial
    total_bets = sum(b.total_bets for b in tournament.bots.values())
    wins = sum(b.winning_bets for b in tournament.bots.values())
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    active_bots = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])
    open_positions = sum(len(b.open_bets) for b in tournament.bots.values())

    pnl_class = "positive" if total_pnl >= 0 else "negative"

    st.html(f"""
    <div class="top-bar">
        <div>
            <div class="system-title">◉ PREDICTION COMMAND CENTER</div>
        </div>
        <div class="system-status">
            <div class="stat-block">
                <div class="stat-value">${total_capital:,.0f}</div>
                <div class="stat-label">Capital</div>
            </div>
            <div class="stat-block">
                <div class="stat-value {pnl_class}">{'+' if total_pnl >= 0 else ''}{total_pnl:,.0f}</div>
                <div class="stat-label">P&L</div>
            </div>
            <div class="stat-block">
                <div class="stat-value">{win_rate:.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-block">
                <div class="stat-value">{total_bets}</div>
                <div class="stat-label">Bets</div>
            </div>
            <div class="stat-block">
                <div class="stat-value">{open_positions}</div>
                <div class="stat-label">Open</div>
            </div>
            <div class="stat-block">
                <div class="stat-value">{active_bots}/{len(tournament.bots)}</div>
                <div class="stat-label">Bots</div>
            </div>
            <div class="live-badge">● LIVE</div>
        </div>
    </div>
    """)


def render_bot_table():
    """Render compact bot performance table."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    rows = ""
    for i, bot in enumerate(rankings):
        rank = i + 1
        rank_class = f"rank-{rank}" if rank <= 3 else ""

        name = bot['name']
        pnl = bot['total_pnl']
        pnl_class = "positive" if pnl >= 0 else "negative"

        wins = bot.get('winning_bets', 0)
        total = bot.get('total_bets', 0)
        losses = total - wins
        wr = (wins / total * 100) if total > 0 else 0

        capital = bot.get('current_capital', 1000)
        open_count = bot.get('open_bets_count', 0)
        days = bot.get('days_evaluated', 0)

        # Win rate bar
        bar_width = min(wr, 100)
        bar_class = "win" if wr >= 50 else "loss"

        # Activity indicator
        activity = "active" if open_count > 0 else "idle"

        # Hot indicator
        hot = "🔥" if wr > 60 and total >= 5 else ""

        rows += f"""
        <tr>
            <td class="{rank_class}">#{rank}</td>
            <td><span class="activity-dot activity-{activity}"></span>{name[:15]}</td>
            <td class="{pnl_class}">{'+' if pnl >= 0 else ''}{pnl:.0f}</td>
            <td>${capital:,.0f}</td>
            <td>{wins}/{losses}</td>
            <td>
                <span class="mini-bar"><span class="mini-bar-fill {bar_class}" style="width:{bar_width}%"></span></span>
                {wr:.0f}%
            </td>
            <td>{open_count}</td>
            <td>D{days}/3</td>
            <td class="hot">{hot}</td>
        </tr>
        """

    st.html(f"""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Bot Performance</span>
            <span class="panel-badge">{len(rankings)} active</span>
        </div>
        <div class="panel-content">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Bot</th>
                        <th>P&L</th>
                        <th>Capital</th>
                        <th>W/L</th>
                        <th>Win%</th>
                        <th>Open</th>
                        <th>Day</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>
    """)


def render_markets():
    """Render live markets panel."""
    markets = fetch_live_markets()

    rows = ""
    for m in markets[:12]:
        yes = int(m['yes_price'] * 100)
        no = 100 - yes
        vol = m['volume_24h']

        if vol >= 1000000:
            vol_str = f"${vol/1000000:.1f}M"
        elif vol >= 1000:
            vol_str = f"${vol/1000:.0f}K"
        else:
            vol_str = f"${vol:.0f}"

        q = m['question'][:50] + ('...' if len(m['question']) > 50 else '')

        rows += f"""
        <div class="market-row">
            <span class="market-q">{q}</span>
            <div class="market-prices">
                <span class="price-tag price-yes">{yes}¢</span>
                <span class="price-tag price-no">{no}¢</span>
            </div>
            <span class="market-vol">{vol_str}</span>
        </div>
        """

    st.html(f"""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Live Markets</span>
            <span class="panel-badge">{len(markets)} active</span>
        </div>
        <div class="scroll-panel">
            {rows}
        </div>
    </div>
    """)


def render_news():
    """Render news feed panel."""
    news = fetch_news()

    rows = ""
    for item in news[:12]:
        age_min = int(item.age_seconds / 60)
        is_breaking = item.freshness_score > 0.8

        title = item.title[:65] + ('...' if len(item.title) > 65 else '')
        source = item.source.split(':')[-1][:8].upper()

        # Sentiment
        if item.sentiment_hint > 0.2:
            sent_class = "news-sentiment-up"
            sent_icon = "▲"
        elif item.sentiment_hint < -0.2:
            sent_class = "news-sentiment-down"
            sent_icon = "▼"
        else:
            sent_class = ""
            sent_icon = ""

        breaking = '<span class="news-breaking">BREAKING</span>' if is_breaking else ""

        rows += f"""
        <div class="news-row">
            <div class="news-title"><span class="{sent_class}">{sent_icon}</span> {title}</div>
            <div class="news-meta">
                <span class="news-source">{source}</span>
                <span class="news-time">{age_min}m</span>
                {breaking}
            </div>
        </div>
        """

    st.html(f"""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">News Feed</span>
            <span class="panel-badge">{len(news)} items</span>
        </div>
        <div class="scroll-panel">
            {rows}
        </div>
    </div>
    """)


def render_positions():
    """Render open positions panel."""
    tournament = get_tournament()
    all_open = []

    for bot in tournament.bots.values():
        for bet in bot.open_bets:
            all_open.append({
                'bot': bot.name[:10],
                'market': bet.market_question[:35],
                'side': bet.side.upper(),
                'amount': bet.amount,
                'entry': bet.entry_price,
                'placed': bet.placed_at,
            })

    # Sort by placed time
    all_open.sort(key=lambda x: x['placed'], reverse=True)

    rows = ""
    for pos in all_open[:10]:
        side_class = "positive" if pos['side'] == 'YES' else "negative"
        age = (datetime.now() - pos['placed']).total_seconds() / 60

        rows += f"""
        <div class="market-row">
            <span style="width:70px;color:#666;">{pos['bot']}</span>
            <span class="market-q">{pos['market']}...</span>
            <span class="{side_class}" style="width:35px;">{pos['side']}</span>
            <span style="width:50px;text-align:right;">${pos['amount']:.0f}</span>
            <span style="width:40px;text-align:right;color:#666;">{pos['entry']*100:.0f}¢</span>
            <span style="width:40px;text-align:right;color:#444;">{age:.0f}m</span>
        </div>
        """

    if not rows:
        rows = '<div style="padding:20px;color:#444;text-align:center;">No open positions</div>'

    st.html(f"""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Open Positions</span>
            <span class="panel-badge">{len(all_open)} positions</span>
        </div>
        <div class="scroll-panel">
            {rows}
        </div>
    </div>
    """)


def render_recent_bets():
    """Render recent bets panel."""
    tournament = get_tournament()
    recent = tournament.all_bets[-15:]
    recent.reverse()

    rows = ""
    for bet in recent:
        if bet.status == 'won':
            status = '<span class="positive">WIN</span>'
            pnl_str = f'<span class="positive">+{bet.pnl:.0f}</span>' if bet.pnl else ''
        elif bet.status == 'lost':
            status = '<span class="negative">LOSS</span>'
            pnl_str = f'<span class="negative">{bet.pnl:.0f}</span>' if bet.pnl else ''
        else:
            status = '<span class="neutral">OPEN</span>'
            pnl_str = ''

        side_class = "positive" if bet.side == 'yes' else "negative"
        market = bet.market_question[:30] + '...' if len(bet.market_question) > 30 else bet.market_question

        rows += f"""
        <div class="market-row">
            <span style="width:35px;">{status}</span>
            <span style="width:70px;color:#666;">{bet.bot_id[:10]}</span>
            <span class="market-q">{market}</span>
            <span class="{side_class}" style="width:30px;">{bet.side.upper()[:1]}</span>
            <span style="width:45px;text-align:right;">${bet.amount:.0f}</span>
            <span style="width:45px;text-align:right;">{pnl_str}</span>
        </div>
        """

    if not rows:
        rows = '<div style="padding:20px;color:#444;text-align:center;">No bets yet</div>'

    st.html(f"""
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Recent Activity</span>
            <span class="panel-badge">{len(tournament.all_bets)} total</span>
        </div>
        <div class="scroll-panel">
            {rows}
        </div>
    </div>
    """)


def render_performance_chart():
    """Render P&L chart."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings:
        return

    names = [r['name'][:12] for r in rankings]
    pnls = [r['total_pnl'] for r in rankings]
    colors = ['#00ff88' if p >= 0 else '#ff4757' for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=pnls,
        marker_color=colors,
        text=[f"{p:+.0f}" for p in pnls],
        textposition='outside',
        textfont=dict(size=9, color='#888', family='Monaco'),
    ))

    fig.add_hline(y=0, line_color='#2a2a3a', line_width=1)

    fig.update_layout(
        plot_bgcolor='#0d0d12',
        paper_bgcolor='#0d0d12',
        font=dict(color='#555', size=9, family='Monaco'),
        margin=dict(l=40, r=10, t=30, b=50),
        height=180,
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(showgrid=True, gridcolor='#1a1a24', zeroline=False, tickformat='+'),
        showlegend=False,
        title=dict(text='P&L BY BOT', font=dict(size=10, color='#555'), x=0.02),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_winrate_chart():
    """Render win rate chart."""
    tournament = get_tournament()
    rankings = [r for r in tournament.get_rankings() if r.get('total_bets', 0) > 0]

    if not rankings:
        return

    names = [r['name'][:12] for r in rankings]
    wrs = [r.get('win_rate', 0) * 100 for r in rankings]
    colors = ['#00ff88' if w >= 50 else '#ff4757' for w in wrs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=wrs,
        marker_color=colors,
        text=[f"{w:.0f}%" for w in wrs],
        textposition='outside',
        textfont=dict(size=9, color='#888', family='Monaco'),
    ))

    fig.add_hline(y=50, line_color='#2a2a3a', line_width=1, line_dash='dot')

    fig.update_layout(
        plot_bgcolor='#0d0d12',
        paper_bgcolor='#0d0d12',
        font=dict(color='#555', size=9, family='Monaco'),
        margin=dict(l=40, r=10, t=30, b=50),
        height=180,
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(showgrid=True, gridcolor='#1a1a24', range=[0, 100]),
        showlegend=False,
        title=dict(text='WIN RATE', font=dict(size=10, color='#555'), x=0.02),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_activity_chart():
    """Render activity timeline."""
    tournament = get_tournament()
    all_bets = tournament.all_bets[-50:]

    if len(all_bets) < 3:
        return

    hour_data = defaultdict(lambda: {'won': 0, 'lost': 0, 'open': 0})
    for bet in all_bets:
        hour = bet.placed_at.strftime('%H:00')
        if bet.status == 'won':
            hour_data[hour]['won'] += 1
        elif bet.status == 'lost':
            hour_data[hour]['lost'] += 1
        else:
            hour_data[hour]['open'] += 1

    hours = sorted(hour_data.keys())

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Won', x=hours, y=[hour_data[h]['won'] for h in hours], marker_color='#00ff88'))
    fig.add_trace(go.Bar(name='Lost', x=hours, y=[hour_data[h]['lost'] for h in hours], marker_color='#ff4757'))
    fig.add_trace(go.Bar(name='Open', x=hours, y=[hour_data[h]['open'] for h in hours], marker_color='#ff9f43'))

    fig.update_layout(
        barmode='stack',
        plot_bgcolor='#0d0d12',
        paper_bgcolor='#0d0d12',
        font=dict(color='#555', size=9, family='Monaco'),
        margin=dict(l=40, r=10, t=30, b=40),
        height=180,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1a1a24'),
        legend=dict(orientation='h', yanchor='top', y=1.15, xanchor='right', x=1, font=dict(size=8)),
        title=dict(text='ACTIVITY', font=dict(size=10, color='#555'), x=0.02),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_market_sentiment():
    """Render market sentiment gauge."""
    markets = fetch_live_markets()

    if not markets:
        return

    bullish = sum(1 for m in markets if m['yes_price'] > 0.6)
    neutral = sum(1 for m in markets if 0.4 <= m['yes_price'] <= 0.6)
    bearish = sum(1 for m in markets if m['yes_price'] < 0.4)

    total = len(markets)
    sentiment = (bullish - bearish) / total if total > 0 else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment * 100,
        number={'suffix': '%', 'font': {'size': 24, 'color': '#888', 'family': 'Monaco'}},
        gauge={
            'axis': {'range': [-100, 100], 'tickcolor': '#333'},
            'bar': {'color': '#00ff88' if sentiment > 0 else '#ff4757'},
            'bgcolor': '#1a1a24',
            'borderwidth': 0,
            'steps': [
                {'range': [-100, -30], 'color': 'rgba(255,71,87,0.2)'},
                {'range': [-30, 30], 'color': 'rgba(136,136,136,0.1)'},
                {'range': [30, 100], 'color': 'rgba(0,255,136,0.2)'},
            ],
        }
    ))

    fig.update_layout(
        plot_bgcolor='#0d0d12',
        paper_bgcolor='#0d0d12',
        font=dict(color='#555', size=9, family='Monaco'),
        margin=dict(l=20, r=20, t=40, b=10),
        height=160,
        title=dict(text='MARKET SENTIMENT', font=dict(size=10, color='#555'), x=0.5, xanchor='center'),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_controls():
    """Render control buttons."""
    tournament = get_tournament()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶ SIMULATE", use_container_width=True):
            tournament.simulate_round()
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("◉ EVALUATE", use_container_width=True):
            tournament.run_daily_evaluation()
            st.cache_data.clear()
            st.rerun()

    with col3:
        if st.button("↻ REFRESH", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with col4:
        auto = st.checkbox("Auto-refresh", value=False)
        if auto:
            time.sleep(10)
            st.rerun()


# ============================================================================
# MAIN LAYOUT
# ============================================================================

def main():
    """Main dashboard layout."""

    # Top bar
    render_top_bar()

    # Controls
    render_controls()

    # Main grid - 4 columns
    col1, col2, col3, col4 = st.columns([1.4, 1.2, 1.2, 1.2])

    with col1:
        render_bot_table()
        render_positions()

    with col2:
        render_markets()

    with col3:
        render_news()

    with col4:
        render_recent_bets()

    # Charts row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_performance_chart()

    with c2:
        render_winrate_chart()

    with c3:
        render_activity_chart()

    with c4:
        render_market_sentiment()


if __name__ == "__main__":
    main()
