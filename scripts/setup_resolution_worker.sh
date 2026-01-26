#!/bin/bash
#
# Setup script for the Resolution Worker scheduled job
#
# Usage:
#   ./scripts/setup_resolution_worker.sh install    # Install and start
#   ./scripts/setup_resolution_worker.sh uninstall  # Stop and remove
#   ./scripts/setup_resolution_worker.sh status     # Check status
#   ./scripts/setup_resolution_worker.sh logs       # View logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.prediction-markets.resolution-worker"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
LOG_DIR="$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

install() {
    echo "Installing Resolution Worker scheduled job..."

    # Create logs directory
    mkdir -p "$LOG_DIR"
    print_status "Created logs directory: $LOG_DIR"

    # Create data directory
    mkdir -p "$PROJECT_DIR/data"
    print_status "Created data directory"

    # Update plist with correct paths
    sed "s|/Users/alexhitt/the-great-discovery/prediction-markets-research|$PROJECT_DIR|g" \
        "$PLIST_SRC" > "$PLIST_DEST"
    print_status "Installed plist to: $PLIST_DEST"

    # Load the job
    launchctl load "$PLIST_DEST"
    print_status "Loaded launchd job"

    echo ""
    print_status "Resolution Worker installed successfully!"
    echo ""
    echo "The worker will run every 5 minutes (300 seconds) and check for:"
    echo "  - Resolved markets on Polymarket/Kalshi"
    echo "  - Learning updates for bots based on outcomes"
    echo ""
    echo "Logs: $LOG_DIR/resolution_worker.log"
    echo ""
    echo "Commands:"
    echo "  Check status: ./scripts/setup_resolution_worker.sh status"
    echo "  View logs:    ./scripts/setup_resolution_worker.sh logs"
    echo "  Uninstall:    ./scripts/setup_resolution_worker.sh uninstall"
}

uninstall() {
    echo "Uninstalling Resolution Worker scheduled job..."

    if [ -f "$PLIST_DEST" ]; then
        launchctl unload "$PLIST_DEST" 2>/dev/null || true
        print_status "Unloaded launchd job"

        rm "$PLIST_DEST"
        print_status "Removed plist file"
    else
        print_warning "Plist not found at $PLIST_DEST"
    fi

    print_status "Resolution Worker uninstalled"
}

status() {
    echo "Resolution Worker Status"
    echo "========================"
    echo ""

    # Check if loaded
    if launchctl list | grep -q "$PLIST_NAME"; then
        print_status "Launchd job is loaded"

        # Get more details
        launchctl list "$PLIST_NAME" 2>/dev/null || true
    else
        print_warning "Launchd job is not loaded"
    fi

    echo ""

    # Check database status
    python3 "$PROJECT_DIR/run_resolution_worker.py" --status 2>/dev/null || \
        print_warning "Could not get database status"
}

logs() {
    echo "Resolution Worker Logs (last 50 lines)"
    echo "======================================="
    echo ""

    LOG_FILE="$LOG_DIR/resolution_worker.log"
    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE"
    else
        print_warning "No log file found at $LOG_FILE"
    fi
}

run_now() {
    echo "Running resolution check now..."
    python3 "$PROJECT_DIR/run_resolution_worker.py" --once
}

case "${1:-}" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    run)
        run_now
        ;;
    *)
        echo "Resolution Worker Setup Script"
        echo ""
        echo "Usage: $0 {install|uninstall|status|logs|run}"
        echo ""
        echo "Commands:"
        echo "  install    - Install and start the scheduled job"
        echo "  uninstall  - Stop and remove the scheduled job"
        echo "  status     - Check current status"
        echo "  logs       - View recent logs"
        echo "  run        - Run a check immediately"
        exit 1
        ;;
esac
