"""
Tournament Dashboard

Real-time visualization of bot competition, bets, and performance.

Features:
- Tournament overview with rankings
- Individual bot profile cards
- Real-time bet feed
- Maturity timeline
- Tier progression tracking
- Daily results and elimination status
"""

import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.autonomous.tournament import BotTournament, BettingTier, BotStatus


# Page config
st.set_page_config(
    page_title="Bot Tournament",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .bot-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #0f3460;
        margin: 10px 0;
    }
    .tier-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .tier-0 { background: #333; color: #888; }
    .tier-1 { background: #cd7f32; color: white; }
    .tier-2 { background: #c0c0c0; color: black; }
    .tier-3 { background: #ffd700; color: black; }
    .tier-4 { background: linear-gradient(135deg, #ffd700, #ff6b6b); color: black; }
    .bet-won { color: #00ff88; }
    .bet-lost { color: #ff4444; }
    .bet-open { color: #ffaa00; }
    .status-evaluating { color: #ffaa00; }
    .status-active { color: #00ff88; }
    .status-promoted { color: #00ffff; }
    .status-eliminated { color: #ff4444; }
    .status-mutated { color: #ff00ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_tournament():
    """Get or create tournament instance."""
    tournament = BotTournament()
    if not tournament.bots:
        tournament.add_default_bots()
    return tournament


def render_header():
    """Render dashboard header."""
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.markdown("# 🏆 Bot Tournament")

    with col2:
        tournament = get_tournament()
        active_bots = len([b for b in tournament.bots.values() if b.status != BotStatus.ELIMINATED])
        total_bets = sum(b.total_bets for b in tournament.bots.values())
        total_pnl = sum(b.total_pnl for b in tournament.bots.values())

        cols = st.columns(4)
        cols[0].metric("Active Bots", active_bots)
        cols[1].metric("Total Bets", f"{total_bets:,}")
        cols[2].metric("Total P&L", f"${total_pnl:+,.2f}")
        cols[3].metric("Today's Bets", len(tournament.get_today_bets()))

    with col3:
        if st.button("🔄 Simulate Round", use_container_width=True):
            tournament = get_tournament()
            tournament.simulate_round()
            st.rerun()

        if st.button("📊 Run Daily Evaluation", use_container_width=True):
            tournament = get_tournament()
            tournament.run_daily_evaluation()
            st.rerun()


def render_tier_progression():
    """Render tier progression visualization."""
    st.markdown("### 📈 Betting Tier Progression")

    tournament = get_tournament()

    # Count bots per tier
    tier_counts = {tier: 0 for tier in BettingTier}
    for bot in tournament.bots.values():
        if bot.status != BotStatus.ELIMINATED:
            tier_counts[bot.tier] += 1

    cols = st.columns(5)
    for i, tier in enumerate(BettingTier):
        with cols[i]:
            color = ["#333", "#cd7f32", "#c0c0c0", "#ffd700", "#ff6b6b"][i]
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {color}20; border-radius: 10px; border: 2px solid {color};">
                <h3 style="margin: 0; color: {color};">{tier_counts[tier]}</h3>
                <p style="margin: 5px 0; font-size: 12px;">{tier.display_name}</p>
            </div>
            """, unsafe_allow_html=True)


def render_rankings():
    """Render bot rankings table."""
    st.markdown("### 🏅 Current Rankings")

    tournament = get_tournament()
    rankings = tournament.get_rankings()

    if not rankings:
        st.info("No bots in tournament yet.")
        return

    # Create DataFrame
    df = pd.DataFrame(rankings)

    # Format columns
    df['P&L'] = df['total_pnl'].apply(lambda x: f"${x:+,.2f}")
    df['Win Rate'] = df['win_rate'].apply(lambda x: f"{x:.1%}")
    df['Sharpe'] = df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    df['Drawdown'] = df['max_drawdown'].apply(lambda x: f"{x:.1%}")
    df['Day'] = df['days_evaluated'].apply(lambda x: f"{x}/3")

    # Status emoji
    status_emoji = {
        'evaluating': '⏳',
        'active': '✅',
        'promoted': '🎉',
        'eliminated': '❌',
        'mutated': '🧬'
    }
    df['Status'] = df['status'].apply(lambda x: status_emoji.get(x, '❓') + ' ' + x.title())

    # Display table
    display_cols = ['rank', 'name', 'tier_name', 'P&L', 'Win Rate', 'Sharpe', 'Day', 'Status', 'open_bets_count']
    display_df = df[display_cols].rename(columns={
        'rank': '#',
        'name': 'Bot',
        'tier_name': 'Tier',
        'open_bets_count': 'Open Bets'
    })

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=400
    )


def render_bot_cards():
    """Render individual bot profile cards."""
    st.markdown("### 🤖 Bot Profiles")

    tournament = get_tournament()
    rankings = tournament.get_rankings()

    # Create columns for bot cards (3 per row)
    for i in range(0, len(rankings), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(rankings):
                bot = rankings[i + j]
                with col:
                    render_single_bot_card(bot, tournament)


def render_single_bot_card(bot: dict, tournament: BotTournament):
    """Render a single bot card."""
    # Determine card color based on status
    status_colors = {
        'evaluating': '#ffaa00',
        'active': '#00ff88',
        'promoted': '#00ffff',
        'eliminated': '#ff4444',
        'mutated': '#ff00ff'
    }
    border_color = status_colors.get(bot['status'], '#0f3460')

    # Performance indicator
    if bot['total_pnl'] > 0:
        pnl_color = '#00ff88'
        pnl_icon = '📈'
    elif bot['total_pnl'] < 0:
        pnl_color = '#ff4444'
        pnl_icon = '📉'
    else:
        pnl_color = '#888'
        pnl_icon = '➡️'

    with st.container():
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    padding: 15px; border-radius: 15px;
                    border: 2px solid {border_color}; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;">#{bot['rank']} {bot['name']}</h4>
                <span style="font-size: 12px; color: {border_color};">{bot['status'].upper()}</span>
            </div>
            <p style="color: #888; margin: 5px 0; font-size: 12px;">{bot['tier_name']}</p>
            <hr style="border-color: #0f3460; margin: 10px 0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <span style="color: #888; font-size: 11px;">P&L</span>
                    <p style="margin: 0; font-size: 18px; color: {pnl_color};">{pnl_icon} ${bot['total_pnl']:+,.2f}</p>
                </div>
                <div>
                    <span style="color: #888; font-size: 11px;">Win Rate</span>
                    <p style="margin: 0; font-size: 18px;">{bot['win_rate']:.1%}</p>
                </div>
                <div>
                    <span style="color: #888; font-size: 11px;">Sharpe</span>
                    <p style="margin: 0; font-size: 14px;">{bot['sharpe_ratio']:.2f}</p>
                </div>
                <div>
                    <span style="color: #888; font-size: 11px;">Open Bets</span>
                    <p style="margin: 0; font-size: 14px;">{bot['open_bets_count']}</p>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <span style="color: #888; font-size: 11px;">Evaluation: Day {bot['days_evaluated']}/3</span>
                <div style="background: #333; border-radius: 5px; height: 6px; margin-top: 5px;">
                    <div style="background: {border_color}; width: {min(100, bot['days_evaluated']/3*100)}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_live_bet_feed():
    """Render real-time bet feed."""
    st.markdown("### 📡 Live Bet Feed")

    tournament = get_tournament()
    today_bets = tournament.get_today_bets()

    if not today_bets:
        st.info("No bets placed today yet. Click 'Simulate Round' to generate bets.")
        return

    # Show last 20 bets
    for bet in today_bets[:20]:
        # Find bot name
        bot_name = "Unknown"
        for bot in tournament.bots.values():
            if bot.id == bet['bot_id']:
                bot_name = bot.name
                break

        # Status styling
        if bet['status'] == 'won':
            status_icon = '✅'
            status_color = '#00ff88'
        elif bet['status'] == 'lost':
            status_icon = '❌'
            status_color = '#ff4444'
        else:
            status_icon = '⏳'
            status_color = '#ffaa00'

        # Time ago
        placed_at = datetime.fromisoformat(bet['placed_at'])
        time_ago = datetime.utcnow() - placed_at
        if time_ago.total_seconds() < 60:
            time_str = f"{int(time_ago.total_seconds())}s ago"
        elif time_ago.total_seconds() < 3600:
            time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
        else:
            time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"

        st.markdown(f"""
        <div style="background: #1a1a2e; padding: 10px 15px; border-radius: 8px;
                    margin: 5px 0; border-left: 3px solid {status_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #888; font-size: 11px;">{bot_name}</span>
                    <p style="margin: 2px 0; font-size: 13px;">
                        {status_icon} <strong>{bet['side'].upper()}</strong> ${bet['amount']:.2f} @ {bet['entry_price']:.2f}
                    </p>
                    <span style="color: #666; font-size: 11px;">{bet['market_question'][:50]}...</span>
                </div>
                <div style="text-align: right;">
                    <span style="color: #666; font-size: 10px;">{time_str}</span>
                    {f"<p style='margin: 0; color: {status_color};'>${bet.get('pnl', 0):+.2f}</p>" if bet['pnl'] else ""}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_maturity_timeline():
    """Render upcoming bet maturities."""
    st.markdown("### ⏰ Upcoming Maturities")

    tournament = get_tournament()
    upcoming = tournament.get_upcoming_maturities()

    if not upcoming:
        st.info("No pending maturities.")
        return

    # Create timeline data
    timeline_data = []
    for bet in upcoming[:15]:
        hours = bet.get('time_to_maturity', 0)
        timeline_data.append({
            'Bot': bet.get('bot_name', 'Unknown'),
            'Market': bet['market_question'][:30] + '...',
            'Side': bet['side'].upper(),
            'Amount': f"${bet['amount']:.2f}",
            'Hours': f"{hours:.1f}h",
            'hours_num': hours
        })

    df = pd.DataFrame(timeline_data)

    # Color code by urgency
    def color_hours(val):
        hours = float(val.replace('h', ''))
        if hours < 1:
            return 'background-color: #ff444444'
        elif hours < 6:
            return 'background-color: #ffaa0044'
        return ''

    styled_df = df.style.applymap(color_hours, subset=['Hours'])
    st.dataframe(df[['Bot', 'Market', 'Side', 'Amount', 'Hours']], hide_index=True, use_container_width=True)


def render_performance_chart():
    """Render cumulative P&L chart for all bots."""
    st.markdown("### 📊 Performance Over Time")

    tournament = get_tournament()

    # Collect data for chart
    chart_data = []
    for bot in tournament.bots.values():
        if bot.status == BotStatus.ELIMINATED:
            continue

        cumulative = 0
        for i, pnl in enumerate(bot.daily_pnls):
            cumulative += pnl
            chart_data.append({
                'Bot': bot.name,
                'Day': i + 1,
                'Cumulative P&L': cumulative
            })

    if not chart_data:
        st.info("Not enough data for performance chart. Run some simulation rounds first.")
        return

    df = pd.DataFrame(chart_data)

    fig = px.line(
        df,
        x='Day',
        y='Cumulative P&L',
        color='Bot',
        title='Bot Performance Over Time'
    )
    fig.update_layout(
        template='plotly_dark',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sidebar():
    """Render sidebar with controls and info."""
    with st.sidebar:
        st.markdown("## 🎮 Controls")

        tournament = get_tournament()

        st.markdown("### Auto-Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=False)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

        st.markdown("---")
        st.markdown("### Tournament Rules")
        st.markdown("""
        - **3-day evaluation** before advancement
        - **Win rate > 55%** to advance
        - **Win rate < 40%** to be eliminated
        - Minimum **5 bots** always maintained
        - Underperformers get **mutated** (strategy tweaked)
        """)

        st.markdown("---")
        st.markdown("### Betting Tiers")
        for tier in BettingTier:
            min_bet, max_bet = tier.bet_range
            if min_bet == 0:
                st.markdown(f"- {tier.display_name}: Simulation")
            else:
                st.markdown(f"- {tier.display_name}: ${min_bet}-${max_bet}")

        st.markdown("---")
        st.markdown("### Quick Actions")

        if st.button("🎲 Simulate 5 Rounds"):
            for _ in range(5):
                tournament.simulate_round()
            st.success("Simulated 5 rounds!")
            st.rerun()

        if st.button("🔄 Reset Tournament"):
            if st.checkbox("Confirm reset"):
                tournament.bots.clear()
                tournament.add_default_bots()
                st.success("Tournament reset!")
                st.rerun()


def main():
    """Main dashboard entry point."""
    render_sidebar()
    render_header()

    st.markdown("---")

    # Tier progression
    render_tier_progression()

    st.markdown("---")

    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings", "🤖 Bot Profiles", "📡 Live Bets", "⏰ Maturities"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            render_rankings()
        with col2:
            render_performance_chart()

    with tab2:
        render_bot_cards()

    with tab3:
        render_live_bet_feed()

    with tab4:
        render_maturity_timeline()


if __name__ == "__main__":
    main()
