"""
Prediction Bot Arena - Gamified Trading Dashboard

Watch your bots compete in real-time! Features:
- Live bot competition with rankings and streaks
- Strategy explanations and signal breakdowns
- Market sentiment and trend analysis
- System performance tracking
"""

import json
import time
import random
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
    page_title="Bot Arena",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# BOT PERSONALITIES - Makes each bot unique and memorable
# ============================================================================
BOT_PROFILES = {
    "Aggressive Alpha": {
        "emoji": "🦁",
        "color": "#ff6b6b",
        "personality": "Bold & Fearless",
        "strategy": "High-risk, high-reward bets on volatile markets",
        "strengths": ["Fast reactions", "Big wins"],
        "weaknesses": ["Can lose big", "Impulsive"],
    },
    "Conservative Value": {
        "emoji": "🦉",
        "color": "#4ecdc4",
        "personality": "Wise & Patient",
        "strategy": "Steady gains through careful analysis",
        "strengths": ["Consistent", "Low risk"],
        "weaknesses": ["Misses big opportunities", "Slow"],
    },
    "News Racer": {
        "emoji": "⚡",
        "color": "#ffe66d",
        "personality": "Lightning Fast",
        "strategy": "First to react to breaking news",
        "strengths": ["Speed", "News analysis"],
        "weaknesses": ["Can overreact", "False signals"],
    },
    "Whale Watcher": {
        "emoji": "🐋",
        "color": "#6c5ce7",
        "personality": "Strategic Follower",
        "strategy": "Follows big money movements",
        "strengths": ["Smart money tracking", "Trend following"],
        "weaknesses": ["Late entries", "Crowded trades"],
    },
    "Sentiment Surfer": {
        "emoji": "🏄",
        "color": "#fd79a8",
        "personality": "Crowd Reader",
        "strategy": "Rides waves of public sentiment",
        "strengths": ["Social signals", "Momentum"],
        "weaknesses": ["Herd mentality", "Reversals"],
    },
    "Balanced Bot": {
        "emoji": "⚖️",
        "color": "#00b894",
        "personality": "The Diplomat",
        "strategy": "Balanced approach using all signals",
        "strengths": ["Diversified", "Stable"],
        "weaknesses": ["Average returns", "No specialty"],
    },
    "Contrarian Carl": {
        "emoji": "🎭",
        "color": "#e17055",
        "personality": "The Rebel",
        "strategy": "Bets against the crowd",
        "strengths": ["Catches reversals", "Unique edge"],
        "weaknesses": ["Often wrong", "Lonely trades"],
    },
}

# ============================================================================
# CUSTOM CSS - Gamified, engaging design
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 0.5rem 1rem !important; max-width: 100% !important;}

    /* Dark gaming theme */
    .stApp { background: linear-gradient(135deg, #0a0e14 0%, #1a1a2e 100%); }

    /* Glowing header */
    .arena-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    .arena-title {
        font-size: 28px;
        font-weight: 800;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Bot card - gamified */
    .bot-card {
        background: linear-gradient(145deg, #1e1e30 0%, #16213e 100%);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .bot-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .bot-card.rank-1 {
        border-color: #ffd700;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    }
    .bot-card.rank-2 {
        border-color: #c0c0c0;
    }
    .bot-card.rank-3 {
        border-color: #cd7f32;
    }

    /* Bot avatar */
    .bot-avatar {
        font-size: 40px;
        margin-right: 12px;
    }
    .bot-name {
        font-size: 18px;
        font-weight: 700;
        color: #fff;
    }
    .bot-personality {
        font-size: 11px;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Stats badges */
    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .stat-pnl-positive {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    .stat-pnl-negative {
        background: linear-gradient(90deg, #d63031 0%, #e17055 100%);
        color: white;
    }
    .stat-wins {
        background: rgba(0, 184, 148, 0.2);
        color: #00b894;
        border: 1px solid #00b894;
    }
    .stat-streak {
        background: rgba(253, 203, 110, 0.2);
        color: #fdcb6e;
        border: 1px solid #fdcb6e;
    }

    /* Rank badge */
    .rank-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 14px;
        color: #0a0e14;
    }
    .rank-1-badge { background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%); }
    .rank-2-badge { background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%); }
    .rank-3-badge { background: linear-gradient(135deg, #cd7f32 0%, #daa06d 100%); }
    .rank-other-badge { background: #3d3d5c; color: #8b95a5; }

    /* Strategy card */
    .strategy-card {
        background: #1e1e30;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .strategy-title {
        font-size: 13px;
        font-weight: 600;
        color: #fff;
        margin-bottom: 6px;
    }
    .strategy-desc {
        font-size: 11px;
        color: #8b95a5;
        line-height: 1.4;
    }

    /* Market card */
    .market-card {
        background: #1e1e30;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .market-question {
        color: #e0e0e0;
        font-size: 12px;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .market-prices {
        display: flex;
        gap: 8px;
    }
    .price-yes {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    .price-no {
        background: linear-gradient(90deg, #d63031 0%, #e17055 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }

    /* News item */
    .news-item {
        background: #1e1e30;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 6px;
        border-left: 3px solid #667eea;
    }
    .news-breaking {
        border-left-color: #ff6b6b;
        animation: pulse-border 2s infinite;
    }
    @keyframes pulse-border {
        0%, 100% { border-left-color: #ff6b6b; }
        50% { border-left-color: #ff8e8e; }
    }
    .news-title {
        color: #e0e0e0;
        font-size: 12px;
        line-height: 1.3;
    }
    .news-meta {
        color: #6c7a89;
        font-size: 10px;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #2d2d44;
    }
    .section-icon {
        font-size: 24px;
    }
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 107, 107, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        color: #ff6b6b;
        font-weight: 600;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #ff6b6b;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    /* Progress bars */
    .progress-container {
        background: #2d2d44;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 8px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Compact buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e30 0%, #16213e 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #fff;
    }
    .metric-label {
        font-size: 11px;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    .metric-positive { color: #00b894; }
    .metric-negative { color: #d63031; }
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
            params={"closed": "false", "limit": 20},
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
                    "category": m.get("category", ""),
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
        return data.get("items", [])[:15]
    except:
        return []


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render arena header with key stats."""
    tournament = get_tournament()

    total_capital = sum(b.current_capital for b in tournament.bots.values())
    initial = 7000  # 7 bots * 1000 each
    total_pnl = total_capital - initial
    total_bets = sum(b.total_bets for b in tournament.bots.values())
    active_bots = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])

    st.markdown(f"""
    <div class="arena-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="arena-title">🏆 Bot Arena</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 13px;">Watch your bots compete for profit</div>
            </div>
            <div style="display: flex; gap: 30px; align-items: center;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 800; color: white;">${total_capital:,.0f}</div>
                    <div style="font-size: 10px; color: rgba(255,255,255,0.7); text-transform: uppercase;">Total Capital</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 800; color: {'#00ffa3' if total_pnl >= 0 else '#ff6b6b'};">${total_pnl:+,.0f}</div>
                    <div style="font-size: 10px; color: rgba(255,255,255,0.7); text-transform: uppercase;">Total P&L</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 800; color: white;">{total_bets}</div>
                    <div style="font-size: 10px; color: rgba(255,255,255,0.7); text-transform: uppercase;">Total Bets</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 800; color: white;">{active_bots}/7</div>
                    <div style="font-size: 10px; color: rgba(255,255,255,0.7); text-transform: uppercase;">Bots Active</div>
                </div>
                <div class="live-indicator">
                    <span class="live-dot"></span>
                    LIVE
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_bot_leaderboard():
    """Render gamified bot leaderboard."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🏅</span>
        <span class="section-title">Leaderboard</span>
    </div>
    """, unsafe_allow_html=True)

    for i, bot in enumerate(rankings[:7]):
        name = bot['name']
        profile = BOT_PROFILES.get(name, {"emoji": "🤖", "color": "#667eea", "personality": "Bot"})

        rank = i + 1
        rank_class = f"rank-{rank}" if rank <= 3 else ""
        rank_badge_class = f"rank-{rank}-badge" if rank <= 3 else "rank-other-badge"

        pnl = bot['total_pnl']
        pnl_class = "stat-pnl-positive" if pnl >= 0 else "stat-pnl-negative"

        wins = bot.get('winning_bets', 0)
        total = bot.get('total_bets', 0)
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0

        # Calculate streak (simplified)
        streak = "🔥" if wins > losses and total > 5 else ""

        # Day progress
        days = bot.get('days_evaluated', 0)
        day_pct = (days / 3) * 100

        st.markdown(f"""
        <div class="bot-card {rank_class}">
            <div class="rank-badge {rank_badge_class}">#{rank}</div>
            <div style="display: flex; align-items: center;">
                <span class="bot-avatar">{profile['emoji']}</span>
                <div style="flex: 1;">
                    <div class="bot-name">{name}</div>
                    <div class="bot-personality">{profile['personality']}</div>
                </div>
                <div style="text-align: right;">
                    <span class="stat-badge {pnl_class}">${pnl:+,.0f}</span>
                    <span class="stat-badge stat-wins">{wins}W / {losses}L</span>
                    {f'<span class="stat-badge stat-streak">{streak} Hot</span>' if streak else ''}
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 10px; color: #8b95a5;">
                    <span>Evaluation Progress</span>
                    <span>Day {days}/3</span>
                </div>
                <div class="progress-container">
                    <div class="progress-fill" style="width: {day_pct}%; background: linear-gradient(90deg, {profile['color']} 0%, {profile['color']}aa 100%);"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_strategy_cards():
    """Render strategy explanation cards."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🧠</span>
        <span class="section-title">Strategies</span>
    </div>
    """, unsafe_allow_html=True)

    for bot in rankings[:4]:
        name = bot['name']
        profile = BOT_PROFILES.get(name, {"emoji": "🤖", "color": "#667eea", "strategy": "Trading bot", "strengths": [], "weaknesses": []})

        strengths = ", ".join(profile.get('strengths', [])[:2])

        st.markdown(f"""
        <div class="strategy-card" style="border-color: {profile['color']};">
            <div class="strategy-title">{profile['emoji']} {name}</div>
            <div class="strategy-desc">{profile.get('strategy', 'Automated trading')}</div>
            <div style="margin-top: 8px; font-size: 10px;">
                <span style="color: #00b894;">✓ {strengths}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_live_markets():
    """Render live markets panel."""
    markets = fetch_live_markets()

    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📈</span>
        <span class="section-title">Hot Markets</span>
    </div>
    """, unsafe_allow_html=True)

    for m in markets[:8]:
        yes_pct = int(m['yes_price'] * 100)
        no_pct = 100 - yes_pct
        vol_k = m['volume_24h'] / 1000

        question = m['question'][:60] + ('...' if len(m['question']) > 60 else '')

        st.markdown(f"""
        <div class="market-card">
            <div class="market-question">{question}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="market-prices">
                    <span class="price-yes">Yes {yes_pct}¢</span>
                    <span class="price-no">No {no_pct}¢</span>
                </div>
                <span style="color: #6c7a89; font-size: 11px;">${vol_k:.0f}K 24h</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_news_feed():
    """Render news feed."""
    news = fetch_news()

    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📰</span>
        <span class="section-title">News Feed</span>
    </div>
    """, unsafe_allow_html=True)

    for item in news[:8]:
        age_min = int(item.age_seconds / 60)
        is_breaking = item.freshness_score > 0.8

        title = item.title[:70] + ('...' if len(item.title) > 70 else '')
        source = item.source.split(':')[-1][:10].upper()

        # Sentiment indicator
        sent = "📈" if item.sentiment_hint > 0.2 else "📉" if item.sentiment_hint < -0.2 else ""

        st.markdown(f"""
        <div class="news-item {'news-breaking' if is_breaking else ''}">
            <div class="news-title">{sent} {title}</div>
            <div class="news-meta">{source} • {age_min}m ago {'🔴 BREAKING' if is_breaking else ''}</div>
        </div>
        """, unsafe_allow_html=True)


def render_performance_chart():
    """Render bot performance comparison chart."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings:
        return

    names = [r['name'][:10] for r in rankings]
    pnls = [r['total_pnl'] for r in rankings]
    colors = [BOT_PROFILES.get(r['name'], {}).get('color', '#667eea') for r in rankings]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=pnls,
        marker_color=colors,
        text=[f"${p:+,.0f}" for p in pnls],
        textposition='outside',
        textfont=dict(size=11, color='white'),
    ))

    fig.update_layout(
        title=dict(text="💰 Profit & Loss by Bot", font=dict(size=14, color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b95a5'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=250,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#2d2d44', zeroline=True, zerolinecolor='#3d3d5c', tickformat='$,.0f'),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_win_rate_chart():
    """Render win rate comparison."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings or all(r.get('total_bets', 0) == 0 for r in rankings):
        return

    # Filter bots with bets
    active = [r for r in rankings if r.get('total_bets', 0) > 0]
    if not active:
        return

    names = [r['name'][:10] for r in active]
    win_rates = [r.get('win_rate', 0) * 100 for r in active]
    colors = [BOT_PROFILES.get(r['name'], {}).get('color', '#667eea') for r in active]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=win_rates,
        marker_color=colors,
        text=[f"{wr:.0f}%" for wr in win_rates],
        textposition='outside',
        textfont=dict(size=11, color='white'),
    ))

    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="#5a6270", opacity=0.5)

    fig.update_layout(
        title=dict(text="🎯 Win Rate by Bot", font=dict(size=14, color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b95a5'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=250,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#2d2d44', range=[0, 100], ticksuffix='%'),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_activity_chart():
    """Render betting activity over time."""
    tournament = get_tournament()
    all_bets = tournament.all_bets[-50:]

    if len(all_bets) < 3:
        return

    # Group by hour
    hour_data = defaultdict(lambda: {'won': 0, 'lost': 0, 'pending': 0})
    for bet in all_bets:
        hour = bet.placed_at.strftime('%H:00')
        if bet.status == 'won':
            hour_data[hour]['won'] += 1
        elif bet.status == 'lost':
            hour_data[hour]['lost'] += 1
        else:
            hour_data[hour]['pending'] += 1

    hours = sorted(hour_data.keys())

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Won', x=hours, y=[hour_data[h]['won'] for h in hours], marker_color='#00b894'))
    fig.add_trace(go.Bar(name='Lost', x=hours, y=[hour_data[h]['lost'] for h in hours], marker_color='#d63031'))
    fig.add_trace(go.Bar(name='Pending', x=hours, y=[hour_data[h]['pending'] for h in hours], marker_color='#fdcb6e'))

    fig.update_layout(
        title=dict(text="⏰ Activity by Hour", font=dict(size=14, color='white')),
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b95a5'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=200,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#2d2d44'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_market_sentiment():
    """Render market sentiment trend chart."""
    markets = fetch_live_markets()

    if not markets:
        return

    # Calculate market sentiment distribution
    bullish = sum(1 for m in markets if m['yes_price'] > 0.6)
    neutral = sum(1 for m in markets if 0.4 <= m['yes_price'] <= 0.6)
    bearish = sum(1 for m in markets if m['yes_price'] < 0.4)

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=['Bullish (>60%)', 'Neutral (40-60%)', 'Bearish (<40%)'],
        values=[bullish, neutral, bearish],
        hole=0.6,
        marker_colors=['#00b894', '#fdcb6e', '#d63031'],
        textinfo='percent',
        textfont=dict(size=11, color='white'),
    ))

    # Add center text showing overall sentiment
    total = len(markets)
    sentiment_score = (bullish - bearish) / total if total > 0 else 0
    sentiment_emoji = "📈" if sentiment_score > 0.2 else "📉" if sentiment_score < -0.2 else "➡️"

    fig.add_annotation(
        text=f"{sentiment_emoji}<br><b>{abs(sentiment_score)*100:.0f}%</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='white'),
    )

    fig.update_layout(
        title=dict(text="🌡️ Market Sentiment", font=dict(size=14, color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b95a5'),
        margin=dict(l=20, r=20, t=50, b=20),
        height=200,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5, font=dict(size=9)),
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_system_stats():
    """Render system improvement tracking."""
    tournament = get_tournament()
    all_bets = tournament.all_bets

    if len(all_bets) < 5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">--</div>
            <div class="metric-label">More data needed</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Calculate cumulative accuracy over batches of 10 bets
    batch_size = 10
    batches = []
    for i in range(0, len(all_bets), batch_size):
        batch = all_bets[i:i+batch_size]
        resolved = [b for b in batch if b.status in ['won', 'lost']]
        if resolved:
            wins = sum(1 for b in resolved if b.status == 'won')
            accuracy = wins / len(resolved) * 100
            batches.append({'batch': len(batches) + 1, 'accuracy': accuracy})

    if len(batches) < 2:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[b['batch'] for b in batches],
        y=[b['accuracy'] for b in batches],
        mode='lines+markers',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#667eea'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
    ))

    # Add trend line
    if len(batches) >= 3:
        x_vals = list(range(len(batches)))
        y_vals = [b['accuracy'] for b in batches]
        slope = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1) if len(y_vals) > 1 else 0
        trend_text = "📈 Improving" if slope > 1 else "📉 Declining" if slope < -1 else "➡️ Stable"

        fig.add_annotation(
            text=trend_text,
            x=0.95, y=0.95, xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=11, color='white'),
            bgcolor='rgba(102, 126, 234, 0.5)',
            borderpad=4,
        )

    # Add 50% baseline
    fig.add_hline(y=50, line_dash="dash", line_color="#5a6270", opacity=0.5)

    fig.update_layout(
        title=dict(text="🎯 System Accuracy Trend", font=dict(size=14, color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b95a5'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=200,
        xaxis=dict(title='Bet Batch', showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title='Accuracy %', showgrid=True, gridcolor='#2d2d44', range=[0, 100]),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_achievements():
    """Render achievements and streaks panel."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🏆</span>
        <span class="section-title">Achievements</span>
    </div>
    """, unsafe_allow_html=True)

    # Calculate achievements
    achievements = []

    # Top performer
    if rankings:
        top = rankings[0]
        profile = BOT_PROFILES.get(top['name'], {"emoji": "🤖"})
        achievements.append({
            "icon": "👑",
            "title": "Current Champion",
            "desc": f"{profile['emoji']} {top['name'][:12]}",
            "color": "#ffd700"
        })

    # Best win rate
    best_wr = max(rankings, key=lambda x: x.get('win_rate', 0)) if rankings else None
    if best_wr and best_wr.get('total_bets', 0) >= 5:
        wr = best_wr.get('win_rate', 0) * 100
        profile = BOT_PROFILES.get(best_wr['name'], {"emoji": "🤖"})
        achievements.append({
            "icon": "🎯",
            "title": f"Sharpshooter ({wr:.0f}%)",
            "desc": f"{profile['emoji']} {best_wr['name'][:12]}",
            "color": "#00b894"
        })

    # Most active
    most_bets = max(rankings, key=lambda x: x.get('total_bets', 0)) if rankings else None
    if most_bets and most_bets.get('total_bets', 0) > 0:
        profile = BOT_PROFILES.get(most_bets['name'], {"emoji": "🤖"})
        achievements.append({
            "icon": "⚡",
            "title": f"Most Active ({most_bets.get('total_bets', 0)} bets)",
            "desc": f"{profile['emoji']} {most_bets['name'][:12]}",
            "color": "#ffe66d"
        })

    # Big winner
    if rankings:
        big_winner = max(rankings, key=lambda x: x.get('total_pnl', 0))
        if big_winner.get('total_pnl', 0) > 50:
            profile = BOT_PROFILES.get(big_winner['name'], {"emoji": "🤖"})
            achievements.append({
                "icon": "💰",
                "title": f"Big Winner (+${big_winner.get('total_pnl', 0):.0f})",
                "desc": f"{profile['emoji']} {big_winner['name'][:12]}",
                "color": "#00b894"
            })

    # Calculate streaks from recent bets
    all_bets = tournament.all_bets[-30:]
    bot_streaks = defaultdict(int)
    current_streaks = defaultdict(int)

    for bet in reversed(all_bets):
        bot_name = bet.bot_id  # bot_id stores the bot name
        if bet.status == 'won':
            current_streaks[bot_name] += 1
            bot_streaks[bot_name] = max(bot_streaks[bot_name], current_streaks[bot_name])
        else:
            current_streaks[bot_name] = 0

    # Hot streak achievement
    if bot_streaks:
        hot_bot = max(bot_streaks.items(), key=lambda x: x[1])
        if hot_bot[1] >= 3:
            profile = BOT_PROFILES.get(hot_bot[0], {"emoji": "🤖"})
            achievements.append({
                "icon": "🔥",
                "title": f"Hot Streak ({hot_bot[1]} wins)",
                "desc": f"{profile['emoji']} {hot_bot[0][:12]}",
                "color": "#ff6b6b"
            })

    # Render achievements
    for ach in achievements[:5]:
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, rgba(30,30,48,0.9) 0%, rgba(22,33,62,0.9) 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-left: 3px solid {ach['color']};
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <span style="font-size: 24px;">{ach['icon']}</span>
            <div>
                <div style="color: white; font-size: 12px; font-weight: 600;">{ach['title']}</div>
                <div style="color: #8b95a5; font-size: 11px;">{ach['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_capital_flow():
    """Render capital flow chart showing money movement."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings:
        return

    # Create capital comparison - use data from rankings
    names = [r['name'][:8] for r in rankings]
    current = [r.get('current_capital', 1000) for r in rankings]
    colors = [BOT_PROFILES.get(r['name'], {}).get('color', '#667eea') for r in rankings]

    fig = go.Figure()

    # Current capital bars
    fig.add_trace(go.Bar(
        name='Current',
        x=names,
        y=current,
        marker_color=colors,
        text=[f"${c:,.0f}" for c in current],
        textposition='outside',
        textfont=dict(size=10, color='white'),
    ))

    # Initial capital reference line
    fig.add_hline(y=1000, line_dash="dash", line_color="#fdcb6e", opacity=0.7,
                  annotation_text="Start: $1,000", annotation_position="right")

    fig.update_layout(
        title=dict(text="💵 Capital Status", font=dict(size=14, color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b95a5'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=200,
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor='#2d2d44', tickformat='$,.0f'),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_controls():
    """Render control buttons."""
    tournament = get_tournament()

    col1, col2, col3, col4 = st.columns(4)

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

    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
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

    # Header
    render_header()

    # Controls
    render_controls()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Main content - 4 columns for more density
    col1, col2, col3, col4 = st.columns([1.4, 1.2, 1.2, 1.2])

    with col1:
        render_bot_leaderboard()

    with col2:
        render_live_markets()

    with col3:
        render_news_feed()

    with col4:
        render_strategy_cards()
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        render_achievements()

    # Charts row 1 - Performance metrics
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)

    with chart_col1:
        render_performance_chart()

    with chart_col2:
        render_win_rate_chart()

    with chart_col3:
        render_capital_flow()

    with chart_col4:
        render_market_sentiment()

    # Charts row 2 - System metrics
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    sys_col1, sys_col2 = st.columns(2)

    with sys_col1:
        render_activity_chart()

    with sys_col2:
        render_system_stats()


if __name__ == "__main__":
    main()
