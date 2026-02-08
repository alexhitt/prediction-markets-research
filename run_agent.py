#!/usr/bin/env python3
"""
Run the autonomous trading agent with crash recovery.

The outer loop catches any unhandled exception, logs it, and restarts
the agent after a brief delay. This makes the process self-healing
even before systemd kicks in.

Usage:
    python run_agent.py
"""

import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from src.autonomous.agent import AutonomousAgent

agent = None
_shutdown_requested = False


def shutdown(signum, frame):
    """Handle graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print('\nShutting down...')
    if agent:
        agent.stop()
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)


def write_crash_log(error: Exception):
    """Append crash info to a persistent crash log."""
    crash_path = Path("data/crash_log.jsonl")
    crash_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import json
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pid": os.getpid(),
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        with open(crash_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # crash logging should never itself crash the recovery


if __name__ == "__main__":
    print('=' * 60)
    print('  AUTONOMOUS TRADING AGENT')
    print('=' * 60)
    print(f'Starting at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'PID: {os.getpid()}')
    print()

    restart_count = 0
    max_restart_delay = 120  # cap at 2 minutes between restarts

    while not _shutdown_requested:
        try:
            agent = AutonomousAgent()
            agent.start()

            if restart_count == 0:
                print('Agent is now running autonomously.')
                print('It will:')
                print('  - Collect signals every 5 minutes')
                print('  - Evaluate trades every minute')
                print('  - Optimize strategy every hour')
                print('  - Run simulations every 30 minutes')
                print('  - Update leaderboard daily')
                print()
                print('Press Ctrl+C to stop.')
            else:
                print(f'Agent restarted (attempt #{restart_count + 1})')

            print('-' * 60)

            while not _shutdown_requested:
                time.sleep(30)
                status = agent.get_status()

                degraded = ""
                health = status.get("health", {})
                if health.get("consecutive_errors", 0) > 0:
                    degraded = f' | Errors: {health["consecutive_errors"]}'

                print(f'[{datetime.now().strftime("%H:%M:%S")}] '
                      f'Cycle: {status["cycle"]:4d} | '
                      f'Signals: {status["signals_today"]:3d} | '
                      f'Opps: {status["opportunities_today"]:2d} | '
                      f'Trades: {status["trades_today"]:2d} | '
                      f'P&L: ${status["pnl_today"]:+8.2f}{degraded}')

            # Clean exit requested
            break

        except KeyboardInterrupt:
            print('\n' + '-' * 60)
            print('Stopping agent...')
            if agent:
                agent.stop()
            print('Agent stopped.')
            break

        except Exception as e:
            restart_count += 1
            write_crash_log(e)

            print(f'\n{"!" * 60}')
            print(f'CRASH DETECTED: {e}')
            print(f'Restart attempt #{restart_count}')
            print(f'{"!" * 60}')

            # Clean up the crashed agent
            try:
                if agent:
                    agent.stop()
            except Exception:
                pass

            agent = None

            # Exponential backoff for restarts, capped
            delay = min(max_restart_delay, 10 * (2 ** min(restart_count - 1, 4)))
            print(f'Restarting in {delay}s...')
            time.sleep(delay)
