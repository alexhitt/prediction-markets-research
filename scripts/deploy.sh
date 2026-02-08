#!/usr/bin/env bash
set -euo pipefail

# Deploy prediction bots to Hetzner server
# Usage: ./scripts/deploy.sh

SERVER="root@157.180.81.173"
REMOTE_DIR="/opt/prediction-bots"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=========================================="
echo "  Deploying Prediction Bots to Hetzner"
echo "=========================================="
echo "Local:  $LOCAL_DIR"
echo "Remote: $SERVER:$REMOTE_DIR"
echo ""

# Step 1: Sync project files
echo "[1/6] Syncing project files..."
rsync -avz --delete \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'data/*.db' \
    --exclude 'data/*.jsonl' \
    --exclude 'data/arena/' \
    --exclude 'data/simulations/' \
    --exclude 'data/tournament/' \
    --exclude 'data/learning/' \
    --exclude 'data/optimization/' \
    --exclude 'data/whales/' \
    --exclude 'data/calendar/' \
    --exclude 'data/weather_trading/' \
    --exclude '.env' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

# Step 2: Set up Python venv and install deps
echo ""
echo "[2/6] Setting up Python environment..."
ssh "$SERVER" "
    cd $REMOTE_DIR
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        echo 'Created new venv'
    fi
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
    echo 'Dependencies installed'
"

# Step 3: Ensure .env exists
echo ""
echo "[3/6] Checking .env..."
ssh "$SERVER" "
    if [ ! -f $REMOTE_DIR/.env ]; then
        cat > $REMOTE_DIR/.env << 'ENVEOF'
# Prediction Bots Environment
# Fill in your Telegram credentials for health alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# API keys (empty = paper trading mode)
POLYMARKET_API_KEY=
KALSHI_API_KEY=
NEWS_API_KEY=
ENVEOF
        echo 'Created .env template — fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID'
    else
        echo '.env already exists'
    fi
"

# Step 4: Ensure data directory exists
echo ""
echo "[4/6] Ensuring data directory..."
ssh "$SERVER" "mkdir -p $REMOTE_DIR/data"

# Step 5: Install systemd units
echo ""
echo "[5/6] Installing systemd services..."
ssh "$SERVER" "
    cp $REMOTE_DIR/scripts/prediction-agent.service /etc/systemd/system/
    cp $REMOTE_DIR/scripts/prediction-health.service /etc/systemd/system/
    cp $REMOTE_DIR/scripts/prediction-health.timer /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable prediction-agent.service
    systemctl enable prediction-health.timer
    echo 'Systemd units installed and enabled'
"

# Step 6: Start/restart services
echo ""
echo "[6/6] Starting services..."
ssh "$SERVER" "
    systemctl restart prediction-agent.service
    systemctl restart prediction-health.timer
    sleep 2
    echo ''
    echo '--- Agent Status ---'
    systemctl status prediction-agent.service --no-pager -l || true
    echo ''
    echo '--- Health Timer Status ---'
    systemctl status prediction-health.timer --no-pager -l || true
"

echo ""
echo "=========================================="
echo "  Deployment complete!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  ssh $SERVER 'systemctl status prediction-agent'"
echo "  ssh $SERVER 'journalctl -u prediction-agent -f'"
echo "  ssh $SERVER 'systemctl restart prediction-agent'"
echo "  ssh $SERVER 'journalctl -u prediction-health -n 20'"
echo ""
echo "Don't forget to fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in:"
echo "  $SERVER:$REMOTE_DIR/.env"
