"""
Bot Arena Dashboard

Real-time visualization of autonomous trading bots competing.
Watch trades happen live, compare performance, see improvements.

Features:
- P&L Time Series Chart
- Live Market Opportunities
- Signal Sources Status
- Arbitrage Opportunities
- Bot Strategy Cards
- Activity Log
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.dashboard.pnl_history import get_pnl_history

# Page config - dark theme, wide layout
st.set_page_config(
    page_title="Bot Arena",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Bot colors for consistent styling
BOT_COLORS = {
    "Conservative Value": "#00D4AA",
    "Aggressive Alpha": "#FF6B6B",
    "Whale Watcher": "#4ECDC4",
    "Sentiment Surfer": "#FFE66D",
    "Balanced Bot": "#95E1D3",
    "Contrarian Carl": "#DDA0DD",
    "News Racer": "#FF9F43",
}

# Signal configuration
SIGNAL_INFO = {
    "arbitrage": {"name": "Arbitrage", "icon": "🔄", "desc": "Cross-market price discrepancies"},
    "whale_activity": {"name": "Whale Activity", "icon": "🐋", "desc": "Large position movements"},
    "sentiment": {"name": "Sentiment", "icon": "📊", "desc": "Social/news sentiment analysis"},
    "odds_movement": {"name": "Odds Movement", "icon": "📈", "desc": "Significant price changes"},
    "volume_spike": {"name": "Volume Spike", "icon": "📢", "desc": "Unusual trading volume"},
    "news": {"name": "News Impact", "icon": "📰", "desc": "Market-moving news events"},
}

# Bot strategy details
BOT_STRATEGIES = {
    "Conservative Value": {
        "description": "Seeks undervalued positions with high confidence. Prioritizes capital preservation.",
        "personality": "Patient and methodical. The Warren Buffett of prediction markets.",
        "strategy_type": "value",
        "risk_level": 0.2,
        "position_size": 3.0,
        "min_edge": 5.0,
    },
    "Aggressive Alpha": {
        "description": "Takes bigger risks for higher potential returns. Momentum-driven approach.",
        "personality": "Bold and opportunistic. First in, first out.",
        "strategy_type": "momentum",
        "risk_level": 0.8,
        "position_size": 10.0,
        "min_edge": 2.0,
    },
    "Whale Watcher": {
        "description": "Follows large trades from successful wallets. Copy-trade specialist.",
        "personality": "Observant and reactive. Why think when you can follow smart money?",
        "strategy_type": "whale",
        "risk_level": 0.5,
        "position_size": 5.0,
        "min_edge": 3.0,
    },
    "Sentiment Surfer": {
        "description": "Trades on social media and news sentiment signals.",
        "personality": "Tuned into the crowd's emotions. Rides the wave of public opinion.",
        "strategy_type": "sentiment",
        "risk_level": 0.5,
        "position_size": 5.0,
        "min_edge": 3.0,
    },
    "Balanced Bot": {
        "description": "Equal-weighted ensemble of all signal types.",
        "personality": "Steady and diversified. Doesn't put all eggs in one basket.",
        "strategy_type": "ensemble",
        "risk_level": 0.5,
        "position_size": 5.0,
        "min_edge": 4.0,
    },
    "Contrarian Carl": {
        "description": "Bets against the crowd when sentiment is extreme.",
        "personality": "Skeptical contrarian. When everyone zigs, Carl zags.",
        "strategy_type": "contrarian",
        "risk_level": 0.4,
        "position_size": 4.0,
        "min_edge": 5.0,
    },
    "News Racer": {
        "description": "Prioritizes fast news signals for information edge. Speed over caution.",
        "personality": "Lightning fast. Catches alpha before the market prices it in.",
        "strategy_type": "news_racer",
        "risk_level": 0.7,
        "position_size": 8.0,
        "min_edge": 2.0,
    },
}

# Custom CSS for sleek dark theme
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }

    /* Bot rankings */
    .bot-rank-1 { color: #FFD700; font-weight: bold; }
    .bot-rank-2 { color: #C0C0C0; }
    .bot-rank-3 { color: #CD7F32; }

    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff00;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Activity log */
    .activity-item {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #00f0ff;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
    }

    /* Trade card */
    .trade-card {
        background: linear-gradient(135deg, rgba(0,102,255,0.1), rgba(0,240,255,0.1));
        border: 1px solid rgba(0,240,255,0.3);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }

    .trade-buy { border-left: 4px solid #00ff88; }
    .trade-sell { border-left: 4px solid #ff4444; }

    /* Signal card */
    .signal-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        text-align: center;
    }

    /* Arbitrage card */
    .arb-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Strategy card */
    .strategy-quote {
        background: linear-gradient(90deg, rgba(0,212,170,0.2) 0%, transparent 100%);
        border-left: 3px solid #00D4AA;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-style: italic;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def load_arena_data() -> Optional[Dict]:
    """Load arena state and performance data."""
    try:
        arena_path = Path("data/arena/arena_state.json")
        if arena_path.exists():
            with open(arena_path) as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading arena data: {e}")
    return None


def load_agent_status() -> Dict:
    """Load main agent status."""
    try:
        state_path = Path("data/agent_state.json")
        if state_path.exists():
            with open(state_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def load_recent_activity(limit: int = 20) -> List[Dict]:
    """Load recent activity from agent log."""
    activities = []
    try:
        log_path = Path("data/agent_log.jsonl")
        if log_path.exists():
            with open(log_path) as f:
                lines = f.readlines()

            for line in reversed(lines[-limit:]):
                try:
                    event = json.loads(line.strip())
                    activities.append(event)
                except Exception:
                    continue
    except Exception:
        pass
    return activities


def load_signal_states() -> Dict[str, Dict]:
    """Load current signal states from various sources."""
    signals = {}

    # Default inactive state for all signals
    for signal_type in SIGNAL_INFO.keys():
        signals[signal_type] = {
            "strength": 0.0,
            "status": "inactive",
            "status_color": "#666666",
            "description": ""
        }

    try:
        # Try to load from signal state file
        signal_path = Path("data/signal_states.json")
        if signal_path.exists():
            with open(signal_path) as f:
                saved_signals = json.load(f)
                for sig_type, sig_data in saved_signals.items():
                    if sig_type in signals:
                        strength = sig_data.get("strength", 0)
                        signals[sig_type] = {
                            "strength": strength,
                            "status": "active" if strength >= 0.3 else "inactive",
                            "status_color": get_strength_color(strength),
                            "description": sig_data.get("description", "")
                        }
    except Exception:
        pass

    return signals


def get_strength_color(strength: float) -> str:
    """Get color based on signal strength."""
    if strength >= 0.7:
        return "#00D4AA"
    elif strength >= 0.4:
        return "#FFE66D"
    elif strength >= 0.1:
        return "#FF6B6B"
    return "#666666"


def load_arbitrage_opportunities() -> List[Dict]:
    """Load arbitrage opportunities."""
    opportunities = []
    try:
        arb_path = Path("data/arbitrage_opportunities.json")
        if arb_path.exists():
            with open(arb_path) as f:
                opportunities = json.load(f)
    except Exception:
        pass
    return opportunities


def load_pending_resolutions() -> List[Dict]:
    """Load pending market resolutions from database."""
    try:
        pending_path = Path("data/pending_resolutions.json")
        if pending_path.exists():
            with open(pending_path) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def load_research_estimates(bot_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """Load recent research estimates."""
    try:
        estimates_path = Path("data/research_estimates.json")
        if estimates_path.exists():
            with open(estimates_path) as f:
                estimates = json.load(f)
                if bot_id:
                    estimates = [e for e in estimates if e.get("bot_id") == bot_id]
                return estimates[:limit]
    except Exception:
        pass
    return []


def load_bot_learning_data(bot_id: str) -> Dict:
    """Load learning/calibration data for a bot."""
    try:
        learning_path = Path(f"data/learning/{bot_id}.json")
        if learning_path.exists():
            with open(learning_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def format_activity(event: Dict) -> tuple:
    """Format activity event into human-readable text."""
    event_type = event.get('event', '')
    data = event.get('data', {})
    timestamp = event.get('timestamp', '')[:19]

    messages = {
        'agent_started': "🚀 System started up and began autonomous operation",
        'signals_collected': f"📡 Collected {data.get('count', 0)} market signals from {len(data.get('sources', []))} sources",
        'trade_executed': f"💰 Executed trade: {data.get('side', '').upper()} ${data.get('size', 0):,.0f} at {data.get('price', 0):.1%} price",
        'strategy_optimized': f"🧬 Optimized strategy weights - previous Sharpe: {data.get('old_sharpe', 0):.2f}",
        'simulation_completed': f"🧪 Simulation complete: {data.get('strategy', '')} - Sharpe: {data.get('sharpe', 0):.2f}",
        'leaderboard_updated': "🏆 Daily leaderboard rankings updated",
        'daily_summary': f"📊 Daily summary: {data.get('trades', 0)} trades, ${data.get('pnl', 0):+,.2f} P&L",
    }

    message = messages.get(event_type, f"📝 {event_type.replace('_', ' ').title()}")
    return timestamp, message


def create_pnl_time_series_chart(pnl_history: Dict, time_window: str = "24h") -> go.Figure:
    """Create P&L time series chart with Plotly."""
    hours_map = {"24h": 24, "7d": 168, "all": 10000}
    hours = hours_map.get(time_window, 24)
    cutoff = datetime.now() - timedelta(hours=hours)

    fig = go.Figure()

    for bot_id, snapshots in pnl_history.items():
        if not snapshots:
            continue

        filtered = [
            s for s in snapshots
            if datetime.fromisoformat(s["timestamp"]) >= cutoff
        ]

        if not filtered:
            continue

        timestamps = [datetime.fromisoformat(s["timestamp"]) for s in filtered]
        pnl_values = [s["pnl"] for s in filtered]
        display_name = bot_id
        color = BOT_COLORS.get(bot_id, "#FFFFFF")

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=pnl_values,
            mode='lines',
            name=display_name,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{display_name}</b><br>Time: %{{x|%H:%M %m/%d}}<br>P&L: $%{{y:,.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"Bot P&L Performance ({time_window})",
            font=dict(color="#FFFFFF", size=18),
            x=0.5
        ),
        xaxis=dict(
            title="Time",
            gridcolor="#333333",
            tickfont=dict(color="#AAAAAA")
        ),
        yaxis=dict(
            title="P&L ($)",
            gridcolor="#333333",
            tickformat="$,.0f",
            zeroline=True,
            zerolinecolor="#666666",
            tickfont=dict(color="#AAAAAA")
        ),
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#0E0E0E",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#FFFFFF")
        ),
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def render_pnl_chart_section():
    """Render the P&L time series chart section."""
    st.subheader("📈 Bot P&L Performance")

    col1, col2 = st.columns([3, 1])
    with col2:
        time_window = st.selectbox(
            "Time Range",
            ["24h", "7d", "all"],
            key="pnl_time_window"
        )

    pnl_history = get_pnl_history()
    history_data = pnl_history.get_all_history(hours=168 if time_window == "7d" else 24 if time_window == "24h" else 10000)

    if history_data and any(len(v) > 0 for v in history_data.values()):
        fig = create_pnl_time_series_chart(history_data, time_window)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Collecting P&L history... Data will appear after trading begins.")


def render_signal_sources_section():
    """Render the signal sources status section."""
    st.subheader("📡 Signal Sources")

    signals = load_signal_states()
    cols = st.columns(3)

    for i, (signal_type, info) in enumerate(SIGNAL_INFO.items()):
        with cols[i % 3]:
            signal = signals.get(signal_type, {"strength": 0, "status_color": "#666666"})
            strength = signal.get("strength", 0)
            color = signal.get("status_color", "#666666")

            st.markdown(f"""
            <div class='signal-card' style='border: 1px solid {color}66;'>
                <span style='font-size: 24px;'>{info['icon']}</span>
                <h4 style='color: #FFFFFF; margin: 8px 0 4px 0;'>{info['name']}</h4>
                <p style='color: #888; font-size: 12px; margin: 0 0 12px 0;'>{info['desc']}</p>
                <div style='background: #333; border-radius: 4px; height: 8px; margin: 12px 0;'>
                    <div style='background: {color}; width: {strength*100}%; height: 100%; border-radius: 4px;'></div>
                </div>
                <span style='color: {color}; font-weight: bold;'>{strength:.0%}</span>
            </div>
            """, unsafe_allow_html=True)


def render_arbitrage_panel():
    """Render the arbitrage opportunities panel."""
    st.subheader("💹 Arbitrage Opportunities")

    min_spread = st.slider("Min Spread %", 0.5, 10.0, 1.0, key="arb_min_spread")
    opportunities = load_arbitrage_opportunities()
    filtered = [o for o in opportunities if o.get("spread_pct", 0) >= min_spread]

    if filtered:
        total_profit = sum(o.get("potential_profit", 0) for o in filtered)

        st.markdown(f"""
        <div style='display: flex; gap: 24px; padding: 12px; background: #1a1a2e; border-radius: 8px; margin-bottom: 16px;'>
            <div><span style='color: #888;'>Active:</span> <b style='color: #FFF;'>{len(filtered)}</b></div>
            <div><span style='color: #888;'>Potential:</span> <b style='color: #00D4AA;'>${total_profit:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        for opp in filtered[:5]:
            spread = opp.get("spread_pct", 0)
            tier_color = "#00D4AA" if spread >= 5.0 else "#FFE66D" if spread >= 2.0 else "#888888"

            st.markdown(f"""
            <div class='arb-card' style='border-left: 4px solid {tier_color};'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #4ECDC4; font-size: 11px;'>{opp.get('arb_type', 'cross_platform').replace('_',' ').title()}</span>
                    <span style='color: {tier_color}; font-size: 20px; font-weight: bold;'>{spread:.2f}%</span>
                </div>
                <p style='color: #AAA; font-size: 13px; margin: 8px 0;'>{opp.get('question', '')[:60]}...</p>
                <div style='display: flex; gap: 16px; margin-top: 12px;'>
                    <div style='background: #252540; padding: 8px 12px; border-radius: 6px; flex: 1;'>
                        <p style='color: #888; font-size: 10px; margin: 0;'>{opp.get('platform_a', 'Polymarket')}</p>
                        <span style='color: #00D4AA;'>{opp.get('price_a', 0):.1%}</span>
                    </div>
                    <span style='color: #888; align-self: center;'>⟷</span>
                    <div style='background: #252540; padding: 8px 12px; border-radius: 6px; flex: 1;'>
                        <p style='color: #888; font-size: 10px; margin: 0;'>{opp.get('platform_b', 'Kalshi')}</p>
                        <span style='color: #FF6B6B;'>{opp.get('price_b', 0):.1%}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No arbitrage opportunities above the minimum spread threshold.")


def render_bot_strategy_cards(arena_data: Optional[Dict]):
    """Render expandable bot strategy cards."""
    st.subheader("🤖 Bot Strategies")

    performances = arena_data.get("performances", {}) if arena_data else {}

    for bot_name, strategy in BOT_STRATEGIES.items():
        color = BOT_COLORS.get(bot_name, "#888888")
        perf = performances.get(bot_name, {})

        strategy_icon = {
            "value": "💎",
            "momentum": "🚀",
            "whale": "🐋",
            "sentiment": "📊",
            "ensemble": "⚖️",
            "contrarian": "🔮",
            "news_racer": "⚡"
        }.get(strategy.get("strategy_type", ""), "🤖")

        with st.expander(f"{strategy_icon} {bot_name}"):
            st.markdown(f"""
            <div class='strategy-quote' style='border-left-color: {color}; background: linear-gradient(90deg, {color}22 0%, transparent 100%);'>
                <p style='color: #FFFFFF; margin: 0;'>"{strategy.get('personality', '')}"</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**About:** {strategy.get('description', '')}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### ⚙️ Parameters")
                st.write(f"Risk Level: {strategy.get('risk_level', 0):.0%}")
                st.write(f"Position Size: {strategy.get('position_size', 0)}%")
                st.write(f"Min Edge: {strategy.get('min_edge', 0):.1f}%")

            with col2:
                st.markdown("##### 📈 Performance")
                if perf:
                    st.write(f"Win Rate: {perf.get('win_rate', 0):.0%}")
                    st.write(f"Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
                    st.write(f"Max DD: {perf.get('max_drawdown', 0):.0%}")
                    pnl = perf.get('total_pnl', 0)
                    pnl_color = "#00D4AA" if pnl >= 0 else "#FF6B6B"
                    st.markdown(f"<span style='color: {pnl_color}; font-size: 18px; font-weight: bold;'>${pnl:+,.2f}</span>", unsafe_allow_html=True)
                else:
                    st.write("Awaiting data...")


def render_bot_rankings(arena_data: Optional[Dict]):
    """Render bot rankings panel."""
    st.markdown("### 🏆 Bot Rankings")

    if arena_data and 'performances' in arena_data:
        performances = arena_data['performances']

        sorted_bots = sorted(
            performances.items(),
            key=lambda x: x[1].get('total_pnl', 0),
            reverse=True
        )

        for rank, (bot_name, perf) in enumerate(sorted_bots, 1):
            pnl = perf.get('total_pnl', 0)
            win_rate = perf.get('win_rate', 0)
            sharpe = perf.get('sharpe_ratio', 0)

            if rank == 1:
                rank_emoji = "🥇"
                color = "#FFD700"
            elif rank == 2:
                rank_emoji = "🥈"
                color = "#C0C0C0"
            elif rank == 3:
                rank_emoji = "🥉"
                color = "#CD7F32"
            else:
                rank_emoji = f"#{rank}"
                color = "#666"

            pnl_color = "#00ff88" if pnl >= 0 else "#ff4444"

            st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 1.2em; color: {color};'>{rank_emoji}</span>
                    <span style='font-weight: 600;'>{bot_name}</span>
                </div>
                <div style='margin-top: 10px;'>
                    <div style='color: {pnl_color}; font-size: 1.5em; font-weight: bold;'>
                        ${pnl:+,.2f}
                    </div>
                    <div style='color: #888; font-size: 0.9em;'>
                        {win_rate:.1%} win rate • Sharpe {sharpe:.1f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Waiting for arena data...")


def render_live_trades(activities: List[Dict]):
    """Render live trade feed."""
    st.markdown("### ⚡ Live Trade Feed")

    trade_activities = [a for a in activities if a.get('event') == 'trade_executed']

    if trade_activities:
        for trade in trade_activities[:5]:
            data = trade.get('data', {})
            side = data.get('side', 'unknown')
            size = data.get('size', 0)
            price = data.get('price', 0)
            confidence = data.get('confidence', 0)
            market_id = data.get('market_id', '')[:20]

            side_class = 'trade-buy' if side == 'yes' else 'trade-sell'
            side_emoji = '🟢' if side == 'yes' else '🔴'

            st.markdown(f"""
            <div class='trade-card {side_class}'>
                <div style='display: flex; justify-content: space-between;'>
                    <span>{side_emoji} <strong>{side.upper()}</strong></span>
                    <span style='color: #888;'>{trade.get('timestamp', '')[:19]}</span>
                </div>
                <div style='font-size: 1.3em; margin: 8px 0;'>
                    ${size:,.2f} @ {price:.1%}
                </div>
                <div style='color: #888; font-size: 0.85em;'>
                    Confidence: {confidence:.1%} • Market: {market_id}...
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='metric-card' style='text-align: center; color: #888;'>
            <p>Waiting for trades...</p>
            <p style='font-size: 0.85em;'>Bots are analyzing market opportunities</p>
        </div>
        """, unsafe_allow_html=True)


def render_activity_log(activities: List[Dict]):
    """Render activity log."""
    st.markdown("### 📋 Activity Log")

    if activities:
        for event in activities[:10]:
            timestamp, message = format_activity(event)
            st.markdown(f"""
            <div class='activity-item'>
                <div style='color: #888; font-size: 0.8em;'>{timestamp}</div>
                <div style='margin-top: 4px;'>{message}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent activity")


def render_system_stats(agent_status: Dict, arena_data: Optional[Dict]):
    """Render system statistics."""
    st.markdown("### 📊 System Stats")

    cycles = agent_status.get('cycles_completed', 0)
    total_trades = agent_status.get('total_trades', 0)
    capital = agent_status.get('current_capital', 10000)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cycles", f"{cycles:,}")
        st.metric("Capital", f"${capital:,.0f}")
    with col2:
        st.metric("Trades", total_trades)
        st.metric("Bots", len(arena_data.get('performances', {})) if arena_data else 0)


def render_detailed_comparison(arena_data: Optional[Dict]):
    """Render detailed bot comparison table."""
    st.markdown("### 🔬 Detailed Bot Comparison")

    if arena_data and 'performances' in arena_data:
        bot_data = []
        for bot_name, perf in arena_data['performances'].items():
            bot_data.append({
                'Bot': bot_name,
                'P&L': f"${perf.get('total_pnl', 0):+,.2f}",
                'Trades': perf.get('total_trades', 0),
                'Wins': perf.get('winning_trades', 0),
                'Win Rate': f"{perf.get('win_rate', 0):.1%}",
                'Sharpe': f"{perf.get('sharpe_ratio', 0):.2f}",
                'Max DD': f"{perf.get('max_drawdown', 0):.1%}",
                'Capital': f"${perf.get('current_capital', 10000):,.0f}"
            })

        df = pd.DataFrame(bot_data)
        df = df.sort_values('P&L', ascending=False)

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )


def render_pending_resolutions():
    """Render pending market resolutions panel."""
    st.markdown("### ⏳ Pending Resolutions")

    pending = load_pending_resolutions()

    if not pending:
        st.info("No pending market resolutions. Place some bets to start tracking!")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pending", len(pending))
    with col2:
        poly_count = len([p for p in pending if p.get("platform") == "polymarket"])
        st.metric("Polymarket", poly_count)
    with col3:
        kalshi_count = len([p for p in pending if p.get("platform") == "kalshi"])
        st.metric("Kalshi", kalshi_count)

    # Upcoming resolutions timeline
    st.markdown("#### Upcoming This Week")

    upcoming = [p for p in pending if p.get("days_until", 999) <= 7]
    upcoming = sorted(upcoming, key=lambda x: x.get("days_until", 999))

    if upcoming:
        for res in upcoming[:10]:
            days = res.get("days_until", 0)
            platform = res.get("platform", "unknown")
            question = res.get("question", "Unknown market")[:60]
            estimates = res.get("estimates_count", 0)
            trades = res.get("trades_count", 0)

            urgency_color = "#FF6B6B" if days <= 1 else "#FFE66D" if days <= 3 else "#00D4AA"

            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid {urgency_color};'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: {urgency_color}; font-weight: bold;'>
                        {days} day{"s" if days != 1 else ""} left
                    </span>
                    <span style='color: #888; font-size: 12px;'>{platform.title()}</span>
                </div>
                <p style='color: #FFF; margin: 8px 0;'>{question}...</p>
                <div style='color: #888; font-size: 12px;'>
                    {estimates} estimates • {trades} trades
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No resolutions expected this week.")


def render_bot_reasoning(bot_id: Optional[str] = None):
    """Render bot research reasoning panel."""
    st.markdown("### 🧠 Bot Reasoning")

    # Bot selector
    estimates = load_research_estimates(limit=50)

    if not estimates:
        st.info("No research estimates yet. Bots will show their reasoning after analyzing markets.")
        return

    bot_ids = list(set(e.get("bot_id", "") for e in estimates))
    selected_bot = st.selectbox("Select Bot", ["All Bots"] + bot_ids, key="reasoning_bot")

    if selected_bot != "All Bots":
        estimates = [e for e in estimates if e.get("bot_id") == selected_bot]

    for est in estimates[:10]:
        bot = est.get("bot_id", "Unknown")
        researcher = est.get("researcher_type", "unknown")
        question = est.get("market_question", "")[:80]
        prob = est.get("estimated_probability", 0.5)
        confidence = est.get("confidence", 0.5)
        market_price = est.get("market_price_at_estimate", 0.5)
        edge = est.get("edge_at_estimate", 0)
        reasoning = est.get("reasoning", "No reasoning provided")

        # Determine direction
        direction = "YES" if prob > market_price else "NO"
        direction_color = "#00D4AA" if direction == "YES" else "#FF6B6B"

        # Status badge
        outcome = est.get("actual_outcome")
        if outcome is not None:
            brier = est.get("brier_score", 0)
            status_badge = f"<span style='background: {'#00D4AA' if brier < 0.25 else '#FF6B6B'}; padding: 2px 8px; border-radius: 4px; font-size: 11px;'>Resolved - Brier: {brier:.3f}</span>"
        else:
            status_badge = "<span style='background: #FFE66D; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 11px;'>Pending</span>"

        st.markdown(f"""
        <div class='metric-card'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='color: #4ECDC4; font-size: 12px;'>{bot} • {researcher}</span>
                {status_badge}
            </div>
            <p style='color: #FFF; margin: 4px 0;'>{question}...</p>
            <div style='display: flex; gap: 20px; margin: 12px 0;'>
                <div>
                    <span style='color: #888; font-size: 11px;'>Estimate</span><br>
                    <span style='color: {direction_color}; font-size: 18px; font-weight: bold;'>{prob:.0%}</span>
                </div>
                <div>
                    <span style='color: #888; font-size: 11px;'>Market</span><br>
                    <span style='color: #FFF; font-size: 18px;'>{market_price:.0%}</span>
                </div>
                <div>
                    <span style='color: #888; font-size: 11px;'>Edge</span><br>
                    <span style='color: {"#00D4AA" if edge > 0.05 else "#888"}; font-size: 18px;'>{edge:.1%}</span>
                </div>
                <div>
                    <span style='color: #888; font-size: 11px;'>Confidence</span><br>
                    <span style='color: #FFF; font-size: 18px;'>{confidence:.0%}</span>
                </div>
            </div>
            <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-top: 8px;'>
                <span style='color: #888; font-size: 11px;'>Reasoning:</span>
                <p style='color: #AAA; margin: 4px 0 0 0; font-size: 13px;'>{reasoning}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_calibration_chart(bot_id: str):
    """Render calibration chart for a specific bot."""
    learning_data = load_bot_learning_data(bot_id)
    calibration = learning_data.get("calibration", {})
    buckets = calibration.get("buckets", {})

    if not buckets:
        st.info(f"No calibration data for {bot_id} yet.")
        return

    # Prepare data for chart
    bucket_names = []
    predicted = []
    actual = []

    for bucket_name, data in sorted(buckets.items()):
        if data.get("count", 0) > 0:
            bucket_names.append(bucket_name)
            predicted.append(data.get("mean_predicted", 0.5))
            actual.append(data.get("mean_actual", 0.5))

    if not bucket_names:
        st.info("Not enough resolved predictions for calibration chart.")
        return

    # Create calibration chart
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='#666666', dash='dash', width=1)
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=predicted,
        y=actual,
        mode='lines+markers',
        name='Actual Outcomes',
        line=dict(color='#00D4AA', width=2),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title=f"Calibration: {bot_id}",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Outcome Rate",
        xaxis=dict(range=[0, 1], gridcolor='#333'),
        yaxis=dict(range=[0, 1], gridcolor='#333'),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#0E0E0E',
        font=dict(color='#FFFFFF'),
        height=300,
        showlegend=True,
        legend=dict(orientation='h', y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Calibration quality indicator
    quality = calibration.get("calibration_quality", "unknown")
    error = calibration.get("weighted_calibration_error", 0)
    multiplier = calibration.get("confidence_multiplier", 1.0)

    quality_colors = {
        "excellent": "#00D4AA",
        "good": "#95E1D3",
        "moderate": "#FFE66D",
        "poor": "#FF9F43",
        "very_poor": "#FF6B6B"
    }

    st.markdown(f"""
    <div style='display: flex; gap: 20px; margin-top: 12px;'>
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; flex: 1;'>
            <span style='color: #888; font-size: 11px;'>Quality</span><br>
            <span style='color: {quality_colors.get(quality, "#888")}; font-weight: bold;'>{quality.title()}</span>
        </div>
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; flex: 1;'>
            <span style='color: #888; font-size: 11px;'>Calibration Error</span><br>
            <span style='color: #FFF;'>{error:.1%}</span>
        </div>
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; flex: 1;'>
            <span style='color: #888; font-size: 11px;'>Confidence Adj.</span><br>
            <span style='color: {"#FF6B6B" if multiplier < 0.9 else "#00D4AA" if multiplier > 1.1 else "#FFF"};'>{multiplier:.2f}x</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_learning_progress():
    """Render learning progress section with calibration and domain performance."""
    st.markdown("### 📚 Learning Progress")

    # Get all bots with learning data
    learning_dir = Path("data/learning")
    if not learning_dir.exists():
        st.info("No learning data yet. Bots will learn from resolved predictions.")
        return

    bot_files = list(learning_dir.glob("*.json"))
    if not bot_files:
        st.info("No learning data yet. Bots will learn from resolved predictions.")
        return

    bot_ids = [f.stem for f in bot_files]

    # Bot selector for calibration
    selected_bot = st.selectbox("Select Bot for Calibration", bot_ids, key="calibration_bot")

    if selected_bot:
        render_calibration_chart(selected_bot)

        # Domain performance
        st.markdown("#### Domain Performance")
        learning_data = load_bot_learning_data(selected_bot)
        domains = learning_data.get("domains", {}).get("domains", {})

        if domains:
            domain_data = []
            for domain, data in domains.items():
                domain_data.append({
                    "Domain": domain.title(),
                    "Predictions": data.get("count", 0),
                    "Brier Score": f"{data.get('average_brier', 0.25):.3f}",
                    "Edge": f"{data.get('average_edge', 0):+.1%}",
                    "Win Rate": f"{data.get('win_rate', 0.5):.0%}",
                    "Skill": f"{data.get('skill_score', 0.5):.2f}"
                })

            df = pd.DataFrame(domain_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # Strong/weak domains
            strong = learning_data.get("domains", {}).get("strong_domains", [])
            weak = learning_data.get("domains", {}).get("weak_domains", [])

            col1, col2 = st.columns(2)
            with col1:
                if strong:
                    st.markdown("**Strong Domains:** " + ", ".join(d.title() for d in strong))
            with col2:
                if weak:
                    st.markdown("**Weak Domains:** " + ", ".join(d.title() for d in weak))
        else:
            st.write("No domain performance data yet.")


def main():
    # Header
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        st.markdown("# 🤖 Bot Arena")

    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding-top: 10px;'>
            <span class='live-dot'></span>
            <span style='color: #00ff00; font-weight: 500;'>LIVE</span>
            <span style='color: #888; margin-left: 20px;'>
                {datetime.now().strftime('%H:%M:%S')}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()

    st.markdown("---")

    # Load data
    arena_data = load_arena_data()
    agent_status = load_agent_status()
    activities = load_recent_activity(30)

    # Main layout - 3 columns for top section
    left_col, center_col, right_col = st.columns([1, 2, 1])

    # LEFT COLUMN - Bot Rankings
    with left_col:
        render_bot_rankings(arena_data)

    # CENTER COLUMN - P&L Chart & Live Trades
    with center_col:
        render_pnl_chart_section()
        render_live_trades(activities)

    # RIGHT COLUMN - Activity Log & Stats
    with right_col:
        render_activity_log(activities)
        render_system_stats(agent_status, arena_data)

    # Middle section - Signal Sources and Arbitrage
    st.markdown("---")
    sig_col, arb_col = st.columns(2)

    with sig_col:
        render_signal_sources_section()

    with arb_col:
        render_arbitrage_panel()

    # Bot Strategy Cards
    st.markdown("---")
    render_bot_strategy_cards(arena_data)

    # Bottom section - Detailed bot comparison
    st.markdown("---")
    render_detailed_comparison(arena_data)

    # Real Paper Trading Section - Pending Resolutions & Learning
    st.markdown("---")
    st.markdown("## 📊 Real Paper Trading")

    tab1, tab2, tab3 = st.tabs(["⏳ Pending Resolutions", "🧠 Bot Reasoning", "📚 Learning Progress"])

    with tab1:
        render_pending_resolutions()

    with tab2:
        render_bot_reasoning()

    with tab3:
        render_learning_progress()

    # Auto-refresh
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=True)

    if auto_refresh:
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
