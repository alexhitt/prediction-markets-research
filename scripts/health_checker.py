#!/usr/bin/env python3
"""
Health Checker for Prediction Market Bots.

Reads the heartbeat file written by the agent, checks process health,
and sends Telegram alerts when something is down.

Designed to run as a systemd timer (every 5 minutes) or standalone.

Alert logic:
  - First failure: log warning, no alert (transient)
  - 2 consecutive failures (10 min): send Telegram alert
  - Repeat alert every 30 min while still down
  - Send "recovered" message when system comes back
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

# Resolve project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HEARTBEAT_PATH = DATA_DIR / "heartbeat.json"
HEALTH_STATUS_PATH = DATA_DIR / "health_status.json"
HEALTH_STATE_PATH = DATA_DIR / "health_checker_state.json"

# Thresholds
HEARTBEAT_MAX_AGE_SECONDS = 300  # 5 minutes
CONSECUTIVE_FAILURES_BEFORE_ALERT = 2
ALERT_REPEAT_INTERVAL_SECONDS = 1800  # 30 minutes


def load_json(path: Path) -> dict:
    """Load JSON file, return empty dict on failure."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_json(path: Path, data: dict):
    """Atomically write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def check_heartbeat() -> dict:
    """Check the agent heartbeat file. Returns status dict."""
    result = {
        "heartbeat_exists": False,
        "heartbeat_fresh": False,
        "heartbeat_age_seconds": None,
        "agent_cycle": None,
        "agent_pid": None,
        "agent_pid_alive": False,
        "consecutive_errors": 0,
        "degraded_sources": [],
    }

    heartbeat = load_json(HEARTBEAT_PATH)
    if not heartbeat:
        return result

    result["heartbeat_exists"] = True
    result["agent_cycle"] = heartbeat.get("cycle")
    result["agent_pid"] = heartbeat.get("pid")
    result["consecutive_errors"] = heartbeat.get("consecutive_errors", 0)
    result["degraded_sources"] = heartbeat.get("degraded_sources", [])

    # Check freshness
    agent_ts = heartbeat.get("agent")
    if agent_ts:
        try:
            last_beat = datetime.fromisoformat(agent_ts)
            age = (datetime.utcnow() - last_beat).total_seconds()
            result["heartbeat_age_seconds"] = round(age, 1)
            result["heartbeat_fresh"] = age < HEARTBEAT_MAX_AGE_SECONDS
        except (ValueError, TypeError):
            pass

    # Check if PID is alive
    pid = heartbeat.get("pid")
    if pid:
        try:
            os.kill(pid, 0)  # signal 0 = check existence
            result["agent_pid_alive"] = True
        except (ProcessLookupError, PermissionError):
            result["agent_pid_alive"] = False
        except (TypeError, OSError):
            pass

    return result


def check_api_connectivity() -> dict:
    """Quick connectivity check to Polymarket API."""
    result = {"polymarket_reachable": False}

    try:
        req = urllib.request.Request(
            "https://gamma-api.polymarket.com/markets?limit=1",
            headers={"User-Agent": "prediction-bot-health-checker/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result["polymarket_reachable"] = resp.status == 200
    except Exception:
        pass

    return result


def send_telegram_alert(message: str):
    """Send alert via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print(f"[ALERT] (no Telegram config) {message}")
        return

    payload = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print(f"[TELEGRAM] Alert sent: {message[:80]}...")
            else:
                print(f"[TELEGRAM] Failed with status {resp.status}")
    except Exception as e:
        print(f"[TELEGRAM] Send failed: {e}")


def run_check():
    """Run a single health check cycle."""
    now = datetime.utcnow()
    now_iso = now.isoformat()

    # Load previous state
    state = load_json(HEALTH_STATE_PATH)
    consecutive_failures = state.get("consecutive_failures", 0)
    last_alert_at = state.get("last_alert_at")
    was_down = state.get("is_down", False)

    # Run checks
    hb = check_heartbeat()
    api = check_api_connectivity()

    # Determine overall health
    agent_healthy = hb["heartbeat_exists"] and hb["heartbeat_fresh"] and hb["agent_pid_alive"]
    is_down = not agent_healthy

    # Build status report
    status = {
        "checked_at": now_iso,
        "agent_healthy": agent_healthy,
        "heartbeat": hb,
        "connectivity": api,
    }

    # Save status for dashboard
    save_json(HEALTH_STATUS_PATH, status)

    # Print status
    if agent_healthy:
        age = hb.get("heartbeat_age_seconds", "?")
        degraded = hb.get("degraded_sources", [])
        degraded_str = f" (degraded: {', '.join(degraded)})" if degraded else ""
        print(f"[{now_iso}] OK — Agent alive, heartbeat {age}s old, cycle {hb.get('agent_cycle')}{degraded_str}")
    else:
        reasons = []
        if not hb["heartbeat_exists"]:
            reasons.append("no heartbeat file")
        elif not hb["heartbeat_fresh"]:
            reasons.append(f"heartbeat stale ({hb.get('heartbeat_age_seconds')}s old)")
        if not hb["agent_pid_alive"]:
            reasons.append("PID not running")
        print(f"[{now_iso}] DOWN — {', '.join(reasons)}")

    # Alert logic
    if is_down:
        consecutive_failures += 1

        should_alert = False
        if consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_ALERT:
            if last_alert_at:
                try:
                    last = datetime.fromisoformat(last_alert_at)
                    if (now - last).total_seconds() >= ALERT_REPEAT_INTERVAL_SECONDS:
                        should_alert = True
                except (ValueError, TypeError):
                    should_alert = True
            else:
                should_alert = True

        if should_alert:
            reasons = []
            if not hb["heartbeat_exists"]:
                reasons.append("No heartbeat file found")
            elif not hb["heartbeat_fresh"]:
                reasons.append(f"Heartbeat stale ({hb.get('heartbeat_age_seconds', '?')}s)")
            if not hb["agent_pid_alive"]:
                reasons.append(f"PID {hb.get('agent_pid')} not running")
            if not api["polymarket_reachable"]:
                reasons.append("Polymarket API unreachable")

            msg = (
                f"🚨 *Prediction Bot DOWN*\n\n"
                f"{''.join(f'• {r}' + chr(10) for r in reasons)}"
                f"Down for {consecutive_failures * 5} min\n"
                f"Server: Hetzner (prediction-bots)"
            )
            send_telegram_alert(msg)
            last_alert_at = now_iso

    else:
        # Agent is healthy
        if was_down and consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_ALERT:
            send_telegram_alert(
                f"✅ *Prediction Bot RECOVERED*\n\n"
                f"Agent is back online after {consecutive_failures * 5} min downtime.\n"
                f"Cycle: {hb.get('agent_cycle')}"
            )

        consecutive_failures = 0
        last_alert_at = None

    # Warn about degraded sources (non-alerting)
    degraded = hb.get("degraded_sources", [])
    if degraded and agent_healthy:
        print(f"[{now_iso}] WARN — Degraded sources: {', '.join(degraded)}")

    # Save state
    save_json(HEALTH_STATE_PATH, {
        "consecutive_failures": consecutive_failures,
        "last_alert_at": last_alert_at,
        "is_down": is_down,
        "last_check": now_iso,
    })


def main():
    """Entry point — run once (for systemd timer) or loop (for standalone)."""
    # Load .env if present
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    if "--daemon" in sys.argv:
        print("Health checker running in daemon mode (every 5 min)")
        while True:
            try:
                run_check()
            except Exception as e:
                print(f"[ERROR] Health check failed: {e}")
            time.sleep(300)
    else:
        run_check()


if __name__ == "__main__":
    main()
