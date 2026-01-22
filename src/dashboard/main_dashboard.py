"""
Prediction Markets Trading Dashboard
Professional tabbed interface for bot management and performance tracking.
Inspired by 3Commas, Bitsgap, TraderSync, and TradingView.
"""

import json
import time
from datetime import datetime, timedelta, date
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
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS
# ============================================================================
st.markdown("""
<style>
    /* Base theme */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 1rem 1.5rem !important; max-width: 100% !important;}

    .stApp {
        background: #0f1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #161b22;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #21262d;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background: #238636 !important;
        color: white !important;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 13px;
        margin-top: 4px;
    }
    .positive { color: #3fb950 !important; }
    .negative { color: #f85149 !important; }
    .neutral { color: #8b949e !important; }

    /* Data tables */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .data-table th {
        text-align: left;
        padding: 12px;
        background: #161b22;
        color: #8b949e;
        font-weight: 500;
        border-bottom: 1px solid #21262d;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .data-table td {
        padding: 12px;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }
    .data-table tr:hover {
        background: #161b22;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-success { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
    .badge-danger { background: rgba(248, 81, 73, 0.2); color: #f85149; }
    .badge-warning { background: rgba(210, 153, 34, 0.2); color: #d29922; }
    .badge-info { background: rgba(56, 139, 253, 0.2); color: #388bfd; }

    /* Rank indicators */
    .rank-1 { color: #ffd700 !important; }
    .rank-2 { color: #c0c0c0 !important; }
    .rank-3 { color: #cd7f32 !important; }

    /* Section headers */
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }

    /* Panel */
    .panel {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .panel-title {
        font-size: 14px;
        font-weight: 600;
        color: #f0f6fc;
    }
    .panel-badge {
        font-size: 11px;
        color: #8b949e;
        background: #21262d;
        padding: 2px 8px;
        border-radius: 10px;
    }

    /* Progress bar */
    .progress-bar {
        height: 6px;
        background: #21262d;
        border-radius: 3px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Button overrides */
    .stButton > button {
        background: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: #30363d !important;
        border-color: #8b949e !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #21262d;
        border-color: #30363d;
    }

    /* Small text */
    .text-muted { color: #8b949e; font-size: 12px; }
    .text-sm { font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LAYER
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
        return data.get("items", [])[:25]
    except:
        return []


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with system status and quick actions."""
    tournament = get_tournament()

    with st.sidebar:
        st.markdown("### 📊 Trading Dashboard")
        st.markdown("---")

        # System status
        total_capital = sum(b.current_capital for b in tournament.bots.values())
        initial = len(tournament.bots) * 1000
        total_pnl = total_capital - initial
        pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

        st.markdown("**System Status**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Capital", f"${total_capital:,.0f}")
        with col2:
            st.metric("P&L", f"${total_pnl:+,.0f}", f"{pnl_pct:+.1f}%")

        st.markdown("---")

        # Quick actions
        st.markdown("**Quick Actions**")

        if st.button("▶️ Simulate Round", use_container_width=True):
            tournament.simulate_round()
            st.cache_data.clear()
            st.rerun()

        if st.button("📊 Daily Evaluation", use_container_width=True):
            tournament.run_daily_evaluation()
            st.cache_data.clear()
            st.rerun()

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Auto refresh
        auto_refresh = st.checkbox("Auto-refresh (10s)")
        if auto_refresh:
            time.sleep(10)
            st.rerun()

        st.markdown("---")

        # Bot status summary
        st.markdown("**Bot Status**")
        active = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])
        total_bets = sum(b.total_bets for b in tournament.bots.values())
        open_pos = sum(len(b.open_bets) for b in tournament.bots.values())

        st.markdown(f"""
        - Active Bots: **{active}/{len(tournament.bots)}**
        - Total Bets: **{total_bets}**
        - Open Positions: **{open_pos}**
        """)


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

def render_overview_tab():
    """Overview tab with key metrics and performance summary."""
    tournament = get_tournament()

    # Top metrics row
    total_capital = sum(b.current_capital for b in tournament.bots.values())
    initial = len(tournament.bots) * 1000
    total_pnl = total_capital - initial
    total_bets = sum(b.total_bets for b in tournament.bots.values())
    wins = sum(b.winning_bets for b in tournament.bots.values())
    losses = total_bets - wins
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    open_positions = sum(len(b.open_bets) for b in tournament.bots.values())

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Capital", f"${total_capital:,.0f}")
    with col2:
        delta_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"${total_pnl:+,.0f}", f"{total_pnl/initial*100:+.1f}%", delta_color=delta_color)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}W / {losses}L")
    with col4:
        st.metric("Total Bets", f"{total_bets}")
    with col5:
        st.metric("Open Positions", f"{open_positions}")
    with col6:
        active = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])
        st.metric("Active Bots", f"{active}/{len(tournament.bots)}")

    st.markdown("---")

    # Two column layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # P&L Chart
        st.markdown("#### 📈 Performance by Bot")
        rankings = tournament.get_rankings()

        if rankings:
            df = pd.DataFrame([{
                'Bot': r['name'],
                'P&L': r['total_pnl'],
                'Win Rate': r.get('win_rate', 0) * 100,
                'Bets': r.get('total_bets', 0)
            } for r in rankings])

            fig = go.Figure()
            colors = ['#3fb950' if x >= 0 else '#f85149' for x in df['P&L']]

            fig.add_trace(go.Bar(
                x=df['Bot'],
                y=df['P&L'],
                marker_color=colors,
                text=[f"${x:+,.0f}" for x in df['P&L']],
                textposition='outside',
            ))

            fig.update_layout(
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font=dict(color='#8b949e'),
                margin=dict(l=40, r=20, t=20, b=60),
                height=300,
                xaxis=dict(showgrid=False, tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='#21262d', zeroline=True, zerolinecolor='#30363d'),
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Activity over time
        st.markdown("#### ⏱️ Activity Timeline")
        all_bets = tournament.all_bets[-100:]

        if len(all_bets) >= 3:
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
            fig.add_trace(go.Bar(name='Won', x=hours, y=[hour_data[h]['won'] for h in hours], marker_color='#3fb950'))
            fig.add_trace(go.Bar(name='Lost', x=hours, y=[hour_data[h]['lost'] for h in hours], marker_color='#f85149'))
            fig.add_trace(go.Bar(name='Open', x=hours, y=[hour_data[h]['open'] for h in hours], marker_color='#d29922'))

            fig.update_layout(
                barmode='stack',
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font=dict(color='#8b949e'),
                margin=dict(l=40, r=20, t=20, b=40),
                height=200,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#21262d'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        # Top performers
        st.markdown("#### 🏆 Leaderboard")

        for i, bot in enumerate(rankings[:5]):
            rank = i + 1
            rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            pnl = bot['total_pnl']
            pnl_class = "positive" if pnl >= 0 else "negative"
            wr = bot.get('win_rate', 0) * 100

            st.markdown(f"""
            <div style="background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 18px; margin-right: 8px;">{rank_emoji}</span>
                        <span style="color: #f0f6fc; font-weight: 500;">{bot['name'][:15]}</span>
                    </div>
                    <span class="{pnl_class}" style="font-weight: 600;">${pnl:+,.0f}</span>
                </div>
                <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">
                    Win Rate: {wr:.0f}% • {bot.get('total_bets', 0)} bets
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Recent news
        st.markdown("#### 📰 Latest News")
        news = fetch_news()

        for item in news[:5]:
            age_min = int(item.age_seconds / 60)
            is_breaking = item.freshness_score > 0.8
            breaking_badge = "🔴 " if is_breaking else ""

            st.markdown(f"""
            <div style="background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 10px; margin-bottom: 6px;">
                <div style="color: #c9d1d9; font-size: 12px; line-height: 1.4;">{breaking_badge}{item.title[:60]}...</div>
                <div style="color: #8b949e; font-size: 10px; margin-top: 4px;">{item.source.split(':')[-1][:10]} • {age_min}m ago</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# TAB 2: BOTS
# ============================================================================

def render_bots_tab():
    """Bots tab with detailed bot performance and management."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    st.markdown("#### Bot Performance Overview")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    active = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])
    total_pnl = sum(r['total_pnl'] for r in rankings)
    avg_wr = sum(r.get('win_rate', 0) for r in rankings) / len(rankings) * 100 if rankings else 0

    with col1:
        st.metric("Active Bots", f"{active}/{len(tournament.bots)}")
    with col2:
        st.metric("Combined P&L", f"${total_pnl:+,.0f}")
    with col3:
        st.metric("Avg Win Rate", f"{avg_wr:.1f}%")
    with col4:
        best = rankings[0]['name'] if rankings else "N/A"
        st.metric("Top Performer", best[:12])

    st.markdown("---")

    # Detailed bot table
    st.markdown("#### Detailed Bot Statistics")

    # Create dataframe for display
    bot_data = []
    for i, r in enumerate(rankings):
        bot = tournament.bots.get(r['name'])
        bot_data.append({
            'Rank': i + 1,
            'Bot': r['name'],
            'Status': '🟢 Active' if r.get('status') != 'eliminated' else '🔴 Eliminated',
            'Capital': f"${r.get('current_capital', 1000):,.0f}",
            'P&L': r['total_pnl'],
            'Win Rate': f"{r.get('win_rate', 0)*100:.1f}%",
            'Wins': r.get('winning_bets', 0),
            'Losses': r.get('total_bets', 0) - r.get('winning_bets', 0),
            'Open': r.get('open_bets_count', 0),
            'Day': f"{r.get('days_evaluated', 0)}/3",
            'Sharpe': f"{r.get('sharpe_ratio', 0):.2f}",
            'Drawdown': f"{r.get('max_drawdown', 0)*100:.1f}%",
        })

    df = pd.DataFrame(bot_data)

    # Style the P&L column
    def color_pnl(val):
        if isinstance(val, (int, float)):
            color = '#3fb950' if val >= 0 else '#f85149'
            return f'color: {color}; font-weight: 600'
        return ''

    styled_df = df.style.applymap(color_pnl, subset=['P&L']).format({'P&L': '${:+,.0f}'})

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Individual bot details
    st.markdown("#### Individual Bot Analysis")

    selected_bot = st.selectbox("Select Bot", [r['name'] for r in rankings])

    if selected_bot:
        bot = tournament.bots.get(selected_bot)
        bot_ranking = next((r for r in rankings if r['name'] == selected_bot), None)

        if bot and bot_ranking:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Bot Details**")
                st.markdown(f"""
                - **Name:** {bot.name}
                - **Tier:** {bot.tier.value}
                - **Status:** {bot.status.value}
                - **Generation:** {bot.generation}
                - **Days Evaluated:** {bot.days_evaluated}/3
                """)

                st.markdown("**Performance Metrics**")
                st.markdown(f"""
                - **Capital:** ${bot.current_capital:,.2f}
                - **Peak Capital:** ${bot.peak_capital:,.2f}
                - **Total P&L:** ${bot_ranking['total_pnl']:+,.2f}
                - **ROI:** {bot_ranking.get('roi', 0)*100:+.2f}%
                - **Win Rate:** {bot_ranking.get('win_rate', 0)*100:.1f}%
                - **Sharpe Ratio:** {bot.sharpe_ratio:.3f}
                - **Max Drawdown:** {bot.max_drawdown*100:.2f}%
                """)

            with col2:
                # Bot's recent trades
                st.markdown("**Recent Trades**")
                recent_bets = [b for b in tournament.all_bets if b.bot_id == selected_bot][-10:]

                if recent_bets:
                    trade_data = []
                    for bet in reversed(recent_bets):
                        trade_data.append({
                            'Time': bet.placed_at.strftime('%H:%M'),
                            'Market': bet.market_question[:40] + '...',
                            'Side': bet.side.upper(),
                            'Amount': f"${bet.amount:.0f}",
                            'Entry': f"{bet.entry_price*100:.0f}¢",
                            'Status': bet.status.upper(),
                            'P&L': f"${bet.pnl:+.0f}" if bet.pnl else '-'
                        })

                    st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No trades yet for this bot")


# ============================================================================
# TAB 3: POSITIONS
# ============================================================================

def render_positions_tab():
    """Positions tab showing all open positions."""
    tournament = get_tournament()

    # Gather all open positions
    all_positions = []
    for bot in tournament.bots.values():
        for bet in bot.open_bets:
            age = (datetime.now() - bet.placed_at).total_seconds() / 60
            all_positions.append({
                'bot_name': bot.name,
                'bet': bet,
                'age_min': age
            })

    # Summary
    col1, col2, col3, col4 = st.columns(4)

    total_exposure = sum(p['bet'].amount for p in all_positions)
    yes_positions = len([p for p in all_positions if p['bet'].side == 'yes'])
    no_positions = len([p for p in all_positions if p['bet'].side == 'no'])
    avg_entry = sum(p['bet'].entry_price for p in all_positions) / len(all_positions) * 100 if all_positions else 0

    with col1:
        st.metric("Open Positions", len(all_positions))
    with col2:
        st.metric("Total Exposure", f"${total_exposure:,.0f}")
    with col3:
        st.metric("YES / NO", f"{yes_positions} / {no_positions}")
    with col4:
        st.metric("Avg Entry", f"{avg_entry:.0f}¢")

    st.markdown("---")

    if all_positions:
        # Sort by age
        all_positions.sort(key=lambda x: x['age_min'])

        st.markdown("#### All Open Positions")

        position_data = []
        for p in all_positions:
            bet = p['bet']
            position_data.append({
                'Bot': p['bot_name'][:12],
                'Market': bet.market_question[:50] + '...',
                'Side': bet.side.upper(),
                'Amount': f"${bet.amount:.0f}",
                'Entry Price': f"{bet.entry_price*100:.0f}¢",
                'Age': f"{p['age_min']:.0f}m",
                'Matures': bet.matures_at.strftime('%m/%d %H:%M') if bet.matures_at else '-'
            })

        df = pd.DataFrame(position_data)

        # Color the Side column
        def color_side(val):
            if val == 'YES':
                return 'color: #3fb950; font-weight: 600'
            elif val == 'NO':
                return 'color: #f85149; font-weight: 600'
            return ''

        styled_df = df.style.applymap(color_side, subset=['Side'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Position breakdown by bot
        st.markdown("---")
        st.markdown("#### Positions by Bot")

        bot_positions = defaultdict(list)
        for p in all_positions:
            bot_positions[p['bot_name']].append(p)

        cols = st.columns(min(len(bot_positions), 4))
        for i, (bot_name, positions) in enumerate(bot_positions.items()):
            with cols[i % 4]:
                total = sum(p['bet'].amount for p in positions)
                st.markdown(f"""
                <div style="background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <div style="color: #f0f6fc; font-weight: 600;">{bot_name[:12]}</div>
                    <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">
                        {len(positions)} positions • ${total:.0f} exposure
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No open positions")


# ============================================================================
# TAB 4: HISTORY
# ============================================================================

def render_history_tab():
    """History tab with trade log and filters."""
    tournament = get_tournament()

    st.markdown("#### Trade History")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_filter = st.selectbox("Status", ["All", "Won", "Lost", "Open"])
    with col2:
        bot_names = ["All"] + [b.name for b in tournament.bots.values()]
        bot_filter = st.selectbox("Bot", bot_names)
    with col3:
        side_filter = st.selectbox("Side", ["All", "YES", "NO"])
    with col4:
        limit = st.selectbox("Show", [25, 50, 100, 200], index=1)

    # Get filtered bets
    all_bets = list(tournament.all_bets)
    all_bets.reverse()  # Most recent first

    filtered_bets = []
    for bet in all_bets:
        if status_filter != "All" and bet.status.lower() != status_filter.lower():
            continue
        if bot_filter != "All" and bet.bot_id != bot_filter:
            continue
        if side_filter != "All" and bet.side.upper() != side_filter:
            continue
        filtered_bets.append(bet)
        if len(filtered_bets) >= limit:
            break

    # Summary stats
    wins = len([b for b in filtered_bets if b.status == 'won'])
    losses = len([b for b in filtered_bets if b.status == 'lost'])
    total_pnl = sum(b.pnl or 0 for b in filtered_bets)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Showing", f"{len(filtered_bets)} trades")
    with col2:
        st.metric("Wins / Losses", f"{wins} / {losses}")
    with col3:
        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        st.metric("Win Rate", f"{wr:.1f}%")
    with col4:
        st.metric("Total P&L", f"${total_pnl:+,.0f}")

    st.markdown("---")

    # Trade table
    if filtered_bets:
        trade_data = []
        for bet in filtered_bets:
            status_badge = "🟢 WIN" if bet.status == 'won' else "🔴 LOSS" if bet.status == 'lost' else "🟡 OPEN"
            trade_data.append({
                'Time': bet.placed_at.strftime('%m/%d %H:%M'),
                'Bot': bet.bot_id[:12],
                'Market': bet.market_question[:45] + '...',
                'Side': bet.side.upper(),
                'Amount': f"${bet.amount:.0f}",
                'Entry': f"{bet.entry_price*100:.0f}¢",
                'Exit': f"{bet.exit_price*100:.0f}¢" if bet.exit_price else '-',
                'P&L': bet.pnl if bet.pnl else 0,
                'Status': status_badge
            })

        df = pd.DataFrame(trade_data)

        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #3fb950; font-weight: 600'
                elif val < 0:
                    return 'color: #f85149; font-weight: 600'
            return ''

        styled_df = df.style.applymap(color_pnl, subset=['P&L']).format({'P&L': '${:+,.0f}'})
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No trades matching filters")


# ============================================================================
# TAB 5: MARKETS
# ============================================================================

def render_markets_tab():
    """Markets tab with live market data."""
    markets = fetch_live_markets()

    st.markdown("#### Live Prediction Markets")

    # Summary
    col1, col2, col3, col4 = st.columns(4)

    total_volume = sum(m['volume_24h'] for m in markets)
    bullish = len([m for m in markets if m['yes_price'] > 0.6])
    bearish = len([m for m in markets if m['yes_price'] < 0.4])
    neutral = len(markets) - bullish - bearish

    with col1:
        st.metric("Active Markets", len(markets))
    with col2:
        st.metric("24h Volume", f"${total_volume/1000000:.1f}M")
    with col3:
        st.metric("Bullish / Bearish", f"{bullish} / {bearish}")
    with col4:
        sentiment = (bullish - bearish) / len(markets) * 100 if markets else 0
        st.metric("Market Sentiment", f"{sentiment:+.0f}%")

    st.markdown("---")

    # Category filter
    categories = list(set(m['category'] for m in markets if m['category']))
    categories = ["All"] + sorted(categories)
    category_filter = st.selectbox("Category", categories)

    # Filter markets
    if category_filter != "All":
        markets = [m for m in markets if m['category'] == category_filter]

    # Market table
    if markets:
        market_data = []
        for m in markets[:30]:
            yes_pct = int(m['yes_price'] * 100)
            no_pct = 100 - yes_pct

            vol = m['volume_24h']
            if vol >= 1000000:
                vol_str = f"${vol/1000000:.1f}M"
            elif vol >= 1000:
                vol_str = f"${vol/1000:.0f}K"
            else:
                vol_str = f"${vol:.0f}"

            market_data.append({
                'Market': m['question'][:60] + ('...' if len(m['question']) > 60 else ''),
                'YES': f"{yes_pct}¢",
                'NO': f"{no_pct}¢",
                '24h Volume': vol_str,
                'Category': m['category'] or '-'
            })

        st.dataframe(pd.DataFrame(market_data), use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No markets available")


# ============================================================================
# TAB 6: ANALYTICS
# ============================================================================

def render_analytics_tab():
    """Analytics tab with deep performance analysis."""
    tournament = get_tournament()
    rankings = tournament.get_rankings()

    st.markdown("#### Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Win rate comparison
        st.markdown("##### Win Rate by Bot")

        if rankings:
            active_rankings = [r for r in rankings if r.get('total_bets', 0) > 0]

            if active_rankings:
                fig = go.Figure()

                names = [r['name'][:12] for r in active_rankings]
                wrs = [r.get('win_rate', 0) * 100 for r in active_rankings]
                colors = ['#3fb950' if w >= 50 else '#f85149' for w in wrs]

                fig.add_trace(go.Bar(
                    x=names,
                    y=wrs,
                    marker_color=colors,
                    text=[f"{w:.0f}%" for w in wrs],
                    textposition='outside',
                ))

                fig.add_hline(y=50, line_dash="dash", line_color="#8b949e", opacity=0.5)

                fig.update_layout(
                    plot_bgcolor='#0d1117',
                    paper_bgcolor='#0d1117',
                    font=dict(color='#8b949e'),
                    margin=dict(l=40, r=20, t=20, b=60),
                    height=300,
                    xaxis=dict(showgrid=False, tickangle=45),
                    yaxis=dict(showgrid=True, gridcolor='#21262d', range=[0, 100]),
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        # Risk metrics
        st.markdown("##### Risk Metrics")

        if rankings:
            risk_data = []
            for r in rankings:
                risk_data.append({
                    'Bot': r['name'][:12],
                    'Sharpe': r.get('sharpe_ratio', 0),
                    'Max DD': f"{r.get('max_drawdown', 0)*100:.1f}%",
                    'ROI': f"{r.get('roi', 0)*100:+.1f}%"
                })

            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Cumulative P&L chart
    st.markdown("##### Cumulative P&L Over Time")

    all_bets = tournament.all_bets

    if len(all_bets) >= 5:
        # Group by bot and track cumulative P&L
        bot_pnl_over_time = defaultdict(list)

        for bet in all_bets:
            if bet.pnl is not None:
                bot_pnl_over_time[bet.bot_id].append({
                    'time': bet.resolved_at or bet.placed_at,
                    'pnl': bet.pnl
                })

        fig = go.Figure()

        colors = ['#3fb950', '#388bfd', '#d29922', '#a371f7', '#f85149', '#8b949e', '#f778ba']

        for i, (bot_name, trades) in enumerate(bot_pnl_over_time.items()):
            if trades:
                trades.sort(key=lambda x: x['time'])
                cumulative = 0
                times = []
                values = []
                for t in trades:
                    cumulative += t['pnl']
                    times.append(t['time'])
                    values.append(cumulative)

                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    name=bot_name[:12],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                ))

        fig.add_hline(y=0, line_color='#30363d', line_width=1)

        fig.update_layout(
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            font=dict(color='#8b949e'),
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#21262d', tickformat='$,.0f'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Not enough data for cumulative P&L chart")

    # Side analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Bet Side Distribution")

        all_bets = tournament.all_bets
        yes_bets = len([b for b in all_bets if b.side == 'yes'])
        no_bets = len([b for b in all_bets if b.side == 'no'])

        if yes_bets + no_bets > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['YES', 'NO'],
                values=[yes_bets, no_bets],
                hole=0.5,
                marker_colors=['#3fb950', '#f85149'],
                textinfo='percent+value',
            )])

            fig.update_layout(
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font=dict(color='#8b949e'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=250,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown("##### Outcome Distribution")

        won = len([b for b in all_bets if b.status == 'won'])
        lost = len([b for b in all_bets if b.status == 'lost'])
        pending = len([b for b in all_bets if b.status not in ['won', 'lost']])

        if won + lost + pending > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Won', 'Lost', 'Pending'],
                values=[won, lost, pending],
                hole=0.5,
                marker_colors=['#3fb950', '#f85149', '#d29922'],
                textinfo='percent+value',
            )])

            fig.update_layout(
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font=dict(color='#8b949e'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=250,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application."""

    # Render sidebar
    render_sidebar()

    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🤖 Bots",
        "📈 Positions",
        "📋 History",
        "🌍 Markets",
        "📉 Analytics"
    ])

    with tab1:
        render_overview_tab()

    with tab2:
        render_bots_tab()

    with tab3:
        render_positions_tab()

    with tab4:
        render_history_tab()

    with tab5:
        render_markets_tab()

    with tab6:
        render_analytics_tab()


if __name__ == "__main__":
    main()
