#!/usr/bin/env python3
"""
Asian Weather Shadow Tracker — data collection for edge validation.

Shadow-tracks the Asian weather confirmation edge across 8 cities (88 daily bins)
using free data from METAR, Open-Meteo ensemble, and Polymarket Gamma API.
No real trades. No capital at risk.

Usage:
    python run_asian_weather_shadow.py --once          # single invocation
    python run_asian_weather_shadow.py --resolve-only  # check resolutions only
    python run_asian_weather_shadow.py --metrics        # print current metrics
    python run_asian_weather_shadow.py --city seoul     # single city debug

Design: data/asian_weather_shadow/DESIGN.md
Plan:   plan-2026-03-30.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from loguru import logger

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.calibration.emos import (
    CITIES,
    CityConfig,
    EnsembleForecast,
    EMOSParams,
    bayesian_update,
    bin_probabilities,
    compute_naive_gaussian,
    emos_predict,
    fetch_ensemble_forecast,
    fetch_historical_actuals,
    load_emos_params,
    parse_bin_edges_from_questions,
    save_emos_params,
    train_all_cities,
    train_emos,
    TrainingSample,
)

# ── Constants ───────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data" / "asian_weather_shadow"
EMOS_DIR = DATA_DIR / "emos_params"
GAMMA_URL = "https://gamma-api.polymarket.com"
METAR_URL = "https://aviationweather.gov/api/data/metar"
TAF_URL = "https://aviationweather.gov/api/data/taf"
HTTP_TIMEOUT = 15

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

SHADOW_NOTIONAL = 10.0  # $10 per shadow trade

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(DATA_DIR / "shadow.log", rotation="1 day", retention="14 days", level="DEBUG")


# ── Phase 2: Data Fetchers ──────────────────────────────────────────

def build_weather_slug(city_slug: str, target_date: date) -> str:
    return (
        f"highest-temperature-in-{city_slug}-on-"
        f"{MONTHS[target_date.month - 1]}-{target_date.day}-{target_date.year}"
    )


def fetch_polymarket_event(city_slug: str, target_date: date) -> Optional[Dict[str, Any]]:
    """Fetch Polymarket weather event for a city/date. Returns parsed event dict or None."""
    slug = build_weather_slug(city_slug, target_date)
    try:
        resp = requests.get(
            f"{GAMMA_URL}/events",
            params={"slug": slug},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = json.loads(resp.text, strict=False)
        if not data or not isinstance(data, list):
            return None
        event = data[0]
        markets = event.get("markets", [])
        bins = []
        for m in markets:
            prices = json.loads(m.get("outcomePrices", "[]"))
            yes_price = float(prices[0]) if prices else 0.0
            bins.append({
                "market_id": m.get("id"),
                "question": m.get("question", ""),
                "yes_price": yes_price,
                "volume": float(m.get("volume", 0)),
                "active": m.get("active", False),
                "closed": m.get("closed", False),
                "clob_token_id": (
                    json.loads(m.get("clobTokenIds", "[]"))[0]
                    if m.get("clobTokenIds") else None
                ),
            })
        return {
            "slug": slug,
            "event_id": event.get("id"),
            "liquidity": float(event.get("liquidity", 0)),
            "volume": float(event.get("volume", 0)),
            "active": event.get("active", False),
            "closed": event.get("closed", False),
            "end_date": event.get("endDate"),
            "bins": bins,
        }
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Gamma fetch failed for {city_slug}/{target_date}: {e}")
        return None


def fetch_metar(icao_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch METAR for multiple stations in one request.

    Returns {icao: {"temp_c": int, "raw": str, "obs_time": str}}.
    """
    result: Dict[str, Dict[str, Any]] = {}
    try:
        resp = requests.get(
            METAR_URL,
            params={"ids": ",".join(icao_ids), "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        for obs in resp.json():
            icao = obs.get("icaoId", "")
            result[icao] = {
                "temp_c": obs.get("temp"),
                "raw": obs.get("rawOb", ""),
                "obs_time": obs.get("obsTime"),
            }
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.warning(f"METAR fetch failed: {e}")
    return result


def fetch_taf_tx(icao_ids: Sequence[str]) -> Dict[str, Optional[int]]:
    """Fetch TAF maximum temperature (TX) for multiple stations.

    Returns {icao: tx_celsius or None}.
    """
    result: Dict[str, Optional[int]] = {}
    try:
        resp = requests.get(
            TAF_URL,
            params={"ids": ",".join(icao_ids), "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        for taf in resp.json():
            icao = taf.get("icaoId", "")
            raw = taf.get("rawTAF", "")
            # Parse TX group: TX15/0207Z means max 15°C at 07Z on the 2nd
            tx_match = re.search(r"TX(\d{2})/\d{4}Z", raw)
            if tx_match:
                result[icao] = int(tx_match.group(1))
            else:
                # Try TXM format for negative temps
                tx_match = re.search(r"TXM(\d{2})/\d{4}Z", raw)
                if tx_match:
                    result[icao] = -int(tx_match.group(1))
                else:
                    result[icao] = None
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.warning(f"TAF fetch failed: {e}")
    return result


# ── Timestamp Helpers ──────────────────────────────────────────────

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse a timestamp string — ISO 8601 or Unix epoch seconds."""
    if not ts_str or ts_str == "None" or ts_str == "null":
        return None
    try:
        stripped = ts_str.strip()
        if stripped.lstrip("-").isdigit():
            return datetime.fromtimestamp(int(stripped), tz=timezone.utc)
        return datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except (ValueError, OSError, OverflowError):
        return None


# ── Phase 3: Shadow Trade Logic ─────────────────────────────────────

def evaluate_yes_picking(
    bin_probs: Dict[str, float],
    bin_prices: Dict[str, float],
    max_entry_price: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """If EMOS top-1 bin price < EMOS probability AND price < max_entry, return trade."""
    if not bin_probs or not bin_prices:
        return None

    # Find top-1 bin by model probability
    top_bin = max(bin_probs, key=bin_probs.get)  # type: ignore[arg-type]
    top_prob = bin_probs[top_bin]
    market_price = bin_prices.get(top_bin, 1.0)

    if market_price >= max_entry_price:
        return None
    if market_price >= top_prob:
        return None  # no edge

    edge = top_prob - market_price
    shares = SHADOW_NOTIONAL / market_price if market_price > 0 else 0
    return {
        "bin": top_bin,
        "side": "yes",
        "entry_price": market_price,
        "model_prob": round(top_prob, 4),
        "edge": round(edge, 4),
        "strategy": "yes_picking",
        "signal_source": "emos_top1",
        "notional": SHADOW_NOTIONAL,
        "shares": round(shares, 2),
    }


def evaluate_no_harvesting(
    bin_probs: Dict[str, float],
    bin_prices: Dict[str, float],
    prob_threshold: float = 0.01,
    min_yes_price: float = 0.02,
) -> List[Dict[str, Any]]:
    """For bins where EMOS prob < threshold AND YES price > min, return NO trades."""
    trades: List[Dict[str, Any]] = []
    for label, prob in bin_probs.items():
        if prob >= prob_threshold:
            continue
        yes_price = bin_prices.get(label, 0.0)
        if yes_price < min_yes_price:
            continue
        no_price = 1.0 - yes_price
        edge = (1.0 - prob) - no_price  # true NO prob minus NO price
        shares = SHADOW_NOTIONAL / no_price if no_price > 0 else 0
        trades.append({
            "bin": label,
            "side": "no",
            "entry_price": no_price,
            "yes_price_at_entry": yes_price,
            "model_prob_yes": round(prob, 6),
            "edge": round(edge, 4),
            "strategy": "no_harvesting",
            "signal_source": "emos_boundary",
            "notional": SHADOW_NOTIONAL,
            "shares": round(shares, 2),
        })
    return trades


def detect_bin_deaths(
    current_bins: List[Dict[str, Any]],
    previous_bins: List[Dict[str, Any]],
    metar_temp: Optional[int],
    metar_ts: Optional[str],
    ts: str,
    *,
    city: Optional[str] = None,
    target_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Detect bins that transitioned from open (>$0.001) to closed ($0.00).

    Computes two lag metrics:
    - delta_t_seconds: poll time minus METAR observation time (METAR staleness)
    - peak_to_resolution_seconds: poll time minus first observation of the day's
      peak temperature (the real confirmation edge window)
    """
    prev_map = {b["question"]: b for b in previous_bins}
    triggers: List[Dict[str, Any]] = []
    poll_dt = _parse_timestamp(ts)

    # Peak-to-resolution lag (same for all bins in this city/date)
    peak_lag = None
    if city and target_date and poll_dt:
        peak_time = _find_peak_time(city, target_date)
        if peak_time and peak_time < poll_dt:
            peak_lag = int((poll_dt - peak_time).total_seconds())

    for curr in current_bins:
        q = curr["question"]
        prev = prev_map.get(q)
        if prev is None:
            continue
        if curr["closed"] and not prev["closed"] and prev["yes_price"] > 0.001:
            metar_lag = None
            if metar_ts and poll_dt:
                metar_dt = _parse_timestamp(str(metar_ts))
                if metar_dt and metar_dt < poll_dt:
                    metar_lag = int((poll_dt - metar_dt).total_seconds())

            triggers.append({
                "ts": ts,
                "bin_closed": q,
                "previous_price": prev["yes_price"],
                "metar_temp_at_close": metar_temp,
                "metar_ts": metar_ts,
                "delta_t_seconds": metar_lag,
                "peak_to_resolution_seconds": peak_lag,
            })

    return triggers


def detect_confirmation(
    current_bins: List[Dict[str, Any]],
    previous_bins: List[Dict[str, Any]],
    threshold: float = 0.80,
) -> Optional[Dict[str, Any]]:
    """Detect when a bin crosses the confirmation threshold."""
    prev_map = {b["question"]: b for b in previous_bins}
    for curr in current_bins:
        q = curr["question"]
        prev = prev_map.get(q)
        if prev is None:
            continue
        if curr["yes_price"] >= threshold and prev["yes_price"] < 0.50:
            return {
                "bin": q,
                "previous_price": prev["yes_price"],
                "confirmation_price": curr["yes_price"],
            }
    return None


# ── Phase 4: State Management ───────────────────────────────────────

def _ensure_dirs() -> None:
    for subdir in ["snapshots", "oracle_triggers", "forecasts", "shadow_trades", "resolutions"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    EMOS_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _find_peak_time(city: str, target_date: date) -> Optional[datetime]:
    """Find when max_observed_temp_c first reached its final value for a city/date.

    Returns the timestamp of the first snapshot where max_observed_temp_c
    equals the day's final observed maximum, or None if insufficient data.
    """
    snap_path = DATA_DIR / "snapshots" / f"{target_date.isoformat()}.jsonl"
    if not snap_path.exists():
        return None

    records = _read_jsonl(snap_path)
    city_snaps = [r for r in records if r.get("city") == city]
    if not city_snaps:
        return None

    # Find the final max_observed_temp_c (last non-None value)
    final_max = None
    for snap in reversed(city_snaps):
        if snap.get("max_observed_temp_c") is not None:
            final_max = snap["max_observed_temp_c"]
            break

    if final_max is None:
        return None

    # Find the first snapshot where max_observed_temp_c reached final_max
    for snap in city_snaps:
        if snap.get("max_observed_temp_c") == final_max:
            return _parse_timestamp(snap.get("ts", ""))

    return None


def load_previous_snapshot(city: str, target_date: date) -> Optional[Dict[str, Any]]:
    path = DATA_DIR / "snapshots" / f"{target_date.isoformat()}.jsonl"
    records = _read_jsonl(path)
    city_records = [r for r in records if r.get("city") == city]
    return city_records[-1] if city_records else None


def save_snapshot(record: Dict[str, Any], target_date: date) -> None:
    _append_jsonl(DATA_DIR / "snapshots" / f"{target_date.isoformat()}.jsonl", record)


def save_shadow_trade(record: Dict[str, Any], target_date: date) -> None:
    _append_jsonl(DATA_DIR / "shadow_trades" / f"{target_date.isoformat()}.jsonl", record)


def save_oracle_trigger(record: Dict[str, Any], target_date: date) -> None:
    _append_jsonl(DATA_DIR / "oracle_triggers" / f"{target_date.isoformat()}.jsonl", record)


def save_forecast(record: Dict[str, Any], target_date: date) -> None:
    path = DATA_DIR / "forecasts" / f"{target_date.isoformat()}.json"
    # Forecasts are per-day, accumulate as list
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []
    existing.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))


def _load_last_known_forecast(city_slug: str) -> tuple:
    """Load the most recent forecast for a city from prior days' files.

    Returns (ifs_mean, aifs_mean, ifs_var) or (None, None, None) if not found.
    Searches the last 7 days of forecast files.
    """
    forecasts_dir = DATA_DIR / "forecasts"
    today = date.today()
    for days_back in range(1, 8):
        d = today - timedelta(days=days_back)
        fp = forecasts_dir / f"{d.isoformat()}.json"
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text())
            entries = data if isinstance(data, list) else [data]
            city_entries = [e for e in entries if e.get("city") == city_slug]
            if city_entries:
                fc = city_entries[-1]
                return fc["ifs_mean"], fc["aifs_mean"], fc["ifs_var"]
        except (json.JSONDecodeError, KeyError):
            continue
    return None, None, None


# ── Phase 5: Resolution and Metrics ────────────────────────────────

def check_resolved_markets(target_date: date) -> List[Dict[str, Any]]:
    """Check which bins won for a resolved date. Returns list of city resolutions."""
    resolutions: List[Dict[str, Any]] = []

    for city_slug, cfg in CITIES.items():
        event = fetch_polymarket_event(city_slug, target_date)
        if not event or not event.get("closed"):
            continue

        winning_bin = None
        for b in event["bins"]:
            if b["yes_price"] >= 0.95:
                winning_bin = b["question"]
                break

        if not winning_bin:
            continue

        # Load shadow trades for this city/date
        trades_path = DATA_DIR / "shadow_trades" / f"{target_date.isoformat()}.jsonl"
        all_trades = _read_jsonl(trades_path)
        city_trades = [t for t in all_trades if t.get("city") == city_slug]

        shadow_results = []
        daily_pnl = 0.0
        hits = 0
        misses = 0

        for trade in city_trades:
            trade_bin = trade.get("bin", "")
            is_yes = trade.get("strategy") == "yes_picking"
            is_no = trade.get("strategy") == "no_harvesting"
            entry_price = trade.get("entry_price", 0)
            shares = trade.get("shares", 0)

            if is_yes:
                hit = winning_bin.lower().find(trade_bin.lower().replace("°c", "°c")) >= 0 or trade_bin in winning_bin
                exit_price = 1.0 if hit else 0.0
                pnl = (exit_price - entry_price) * shares
            elif is_no:
                # NO trade wins if the bin did NOT win
                hit = trade_bin not in winning_bin
                exit_price = 1.0 if hit else 0.0
                pnl = (exit_price - entry_price) * shares
            else:
                continue

            if hit:
                hits += 1
            else:
                misses += 1
            daily_pnl += pnl

            shadow_results.append({
                "trade_id": trade.get("trade_id", ""),
                "bin": trade_bin,
                "strategy": trade.get("strategy"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "hit": hit,
            })

        resolutions.append({
            "target_date": target_date.isoformat(),
            "city": city_slug,
            "winning_bin": winning_bin,
            "shadow_trades": shadow_results,
            "daily_pnl": round(daily_pnl, 2),
            "total_trades": hits + misses,
            "hits": hits,
            "misses": misses,
        })

    return resolutions


def save_resolutions(resolutions: List[Dict[str, Any]], target_date: date) -> None:
    path = DATA_DIR / "resolutions" / f"{target_date.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolutions, indent=2))


def update_metrics() -> Dict[str, Any]:
    """Aggregate all resolution data into running metrics."""
    res_dir = DATA_DIR / "resolutions"
    if not res_dir.exists():
        return {}

    total_trades = 0
    total_hits = 0
    total_pnl = 0.0
    days_observed = 0
    by_region: Dict[str, Dict[str, Any]] = {
        "east_asia": {"trades": 0, "hits": 0, "pnl": 0.0},
        "south_asia": {"trades": 0, "hits": 0, "pnl": 0.0},
        "europe": {"trades": 0, "hits": 0, "pnl": 0.0},
    }
    east_asia = {"seoul", "hong-kong", "shanghai", "shenzhen", "beijing"}
    south_asia = {"lucknow"}
    europe = {"warsaw", "munich"}

    delta_ts: List[float] = []

    for res_file in sorted(res_dir.glob("*.json")):
        try:
            data = json.loads(res_file.read_text())
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            continue

        days_observed += 1

        for city_res in data:
            city = city_res.get("city", "")
            trades = city_res.get("total_trades", 0)
            hits = city_res.get("hits", 0)
            pnl = city_res.get("daily_pnl", 0.0)

            total_trades += trades
            total_hits += hits
            total_pnl += pnl

            if city in east_asia:
                region = "east_asia"
            elif city in south_asia:
                region = "south_asia"
            else:
                region = "europe"
            by_region[region]["trades"] += trades
            by_region[region]["hits"] += hits
            by_region[region]["pnl"] += pnl

    # Compute confirmation lags: peak temperature observation → oracle resolution
    # Uses snapshot data directly so historical days are included (backfill).
    triggers_dir = DATA_DIR / "oracle_triggers"
    if triggers_dir.exists():
        for trig_file in sorted(triggers_dir.glob("*.jsonl")):
            trig_date = date.fromisoformat(trig_file.stem)
            records = _read_jsonl(trig_file)
            # Group by city to compute one lag per city/date
            cities_seen: Dict[str, datetime] = {}
            for record in records:
                city = record.get("city", "")
                if not city or city in cities_seen:
                    continue
                resolution_dt = _parse_timestamp(record.get("ts", ""))
                if not resolution_dt:
                    continue
                cities_seen[city] = resolution_dt

            for city, resolution_dt in cities_seen.items():
                peak_time = _find_peak_time(city, trig_date)
                if peak_time is None or peak_time >= resolution_dt:
                    continue  # no snapshot data or peak after resolution (gap)
                lag_min = (resolution_dt - peak_time).total_seconds() / 60.0
                if lag_min > 0:
                    delta_ts.append(lag_min)

    accuracy = total_hits / total_trades if total_trades > 0 else 0.0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

    for region in by_region.values():
        region["accuracy"] = (
            region["hits"] / region["trades"] if region["trades"] > 0 else 0.0
        )
        region["pnl"] = round(region["pnl"], 2)

    metrics = {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "days_observed": days_observed,
        "total_shadow_trades": total_trades,
        "hits": total_hits,
        "accuracy": round(accuracy, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "median_delta_t_minutes": round(statistics.median(delta_ts), 1) if delta_ts else None,
        "by_region": by_region,
    }

    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def update_readiness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate kill/promote criteria against metrics."""
    days = metrics.get("days_observed", 0)
    trades = metrics.get("total_shadow_trades", 0)
    accuracy = metrics.get("accuracy", 0)
    avg_pnl = metrics.get("avg_pnl_per_trade", 0)
    delta_t = metrics.get("median_delta_t_minutes")

    status = "collecting"
    decision_status = "pending"

    if days >= 14 and trades >= 50:
        # Kill criteria (any one)
        if accuracy < 0.15:
            status = "kill"
            decision_status = "kill"
        elif avg_pnl < 0:
            status = "kill"
            decision_status = "kill"
        # Promote criteria (all required)
        elif (
            accuracy > 0.25
            and avg_pnl > 1.50
            and trades >= 100
            and delta_t is not None
            and delta_t > 30
        ):
            status = "promote"
            decision_status = "promote"

    readiness = {
        "status": status,
        "decision_status": decision_status,
        "days_observed": days,
        "total_trades": trades,
        "accuracy": accuracy,
        "avg_pnl_per_trade": avg_pnl,
        "median_delta_t_minutes": delta_t,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    (DATA_DIR / "readiness.json").write_text(json.dumps(readiness, indent=2))
    return readiness


# ── Phase 6: Main Loop ──────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_first_run_today(target_date: date) -> bool:
    """Check if we already have a forecast for today."""
    path = DATA_DIR / "forecasts" / f"{target_date.isoformat()}.json"
    return not path.exists()


def process_city(
    city_slug: str,
    cfg: CityConfig,
    target_date: date,
    metar_data: Dict[str, Dict[str, Any]],
    taf_data: Dict[str, Optional[int]],
    is_first_run: bool,
    ts: str,
) -> None:
    """Process one city: fetch market, compute probs, evaluate trades, detect events."""

    event = fetch_polymarket_event(city_slug, target_date)
    if not event:
        logger.info(f"{city_slug}: no market found for {target_date}")
        return

    if event.get("closed"):
        logger.debug(f"{city_slug}: market already closed")
        return

    bins = event["bins"]
    metar = metar_data.get(cfg.icao, {})
    metar_temp = metar.get("temp_c")
    metar_raw = metar.get("raw", "")
    metar_obs_time = metar.get("obs_time")
    taf_tx = taf_data.get(cfg.icao)

    # Parse bin edges from market questions
    questions = [b["question"] for b in bins]
    bin_edges = parse_bin_edges_from_questions(questions)

    # Get or compute bin probabilities
    emos_params = load_emos_params(EMOS_DIR / f"{city_slug}.json")

    if is_first_run:
        # Fetch ensemble forecast
        ifs_members, aifs_members = fetch_ensemble_forecast(cfg.lat, cfg.lon)

        if ifs_members:
            ifs_mean = statistics.mean(ifs_members)
            ifs_var = statistics.variance(ifs_members) if len(ifs_members) > 1 else 1.0
            aifs_mean = statistics.mean(aifs_members) if aifs_members else ifs_mean

            forecast_record = EnsembleForecast(
                city=city_slug,
                target_date=target_date.isoformat(),
                ifs_members=ifs_members,
                aifs_members=aifs_members,
                ifs_mean=round(ifs_mean, 2),
                ifs_var=round(ifs_var, 2),
                aifs_mean=round(aifs_mean, 2),
                fetched_at=ts,
            )
            save_forecast(forecast_record.to_dict(), target_date)

            if emos_params:
                mean, variance = emos_predict(emos_params, ifs_mean, aifs_mean, ifs_var)
            else:
                mean, std = compute_naive_gaussian(ifs_members, aifs_members)
                variance = std * std
        else:
            # Try last-known forecast from previous days
            ifs_mean, aifs_mean, ifs_var = _load_last_known_forecast(city_slug)
            if ifs_mean is not None:
                logger.warning(f"{city_slug}: no ensemble data, using last-known forecast")
                if emos_params:
                    mean, variance = emos_predict(emos_params, ifs_mean, aifs_mean, ifs_var)
                else:
                    mean, variance = ifs_mean, max(ifs_var * 1.3, 2.25)
            else:
                logger.warning(f"{city_slug}: no ensemble data and no prior forecast, using 15°C default")
                mean, variance = 15.0, 9.0
    else:
        # Load today's forecast from file
        forecasts = []
        fp = DATA_DIR / "forecasts" / f"{target_date.isoformat()}.json"
        if fp.exists():
            try:
                loaded = json.loads(fp.read_text())
                forecasts = loaded if isinstance(loaded, list) else [loaded]
            except json.JSONDecodeError:
                pass
        city_forecasts = [f for f in forecasts if f.get("city") == city_slug]
        if city_forecasts:
            fc = city_forecasts[-1]
            ifs_mean = fc["ifs_mean"]
            aifs_mean = fc["aifs_mean"]
            ifs_var = fc["ifs_var"]
            if emos_params:
                mean, variance = emos_predict(emos_params, ifs_mean, aifs_mean, ifs_var)
            else:
                mean, std = compute_naive_gaussian(fc["ifs_members"], fc.get("aifs_members", []))
                variance = std * std
        else:
            # Try last-known forecast from previous days
            ifs_mean, aifs_mean, ifs_var = _load_last_known_forecast(city_slug)
            if ifs_mean is not None:
                logger.warning(f"{city_slug}: no today forecast, using last-known")
                if emos_params:
                    mean, variance = emos_predict(emos_params, ifs_mean, aifs_mean, ifs_var)
                else:
                    mean, variance = ifs_mean, max(ifs_var * 1.3, 2.25)
            else:
                mean, variance = 15.0, 9.0

    # Bayesian update with METAR
    if metar_temp is not None:
        mean, variance = bayesian_update(mean, variance, metar_temp)

    # Compute bin probabilities
    bin_probs = bin_probabilities(mean, variance, bin_edges)

    # Build price map: label → yes_price
    bin_prices: Dict[str, float] = {}
    for b, (lower, upper) in zip(bins, bin_edges):
        if lower is None and upper is not None:
            label = f"{upper}°C or below"
        elif upper is None and lower is not None:
            label = f"{lower}°C or higher"
        elif lower is not None:
            label = f"{lower}°C"
        else:
            continue
        bin_prices[label] = b["yes_price"]

    # Find top bins
    sorted_probs = sorted(bin_probs.items(), key=lambda x: -x[1])
    emos_top1 = sorted_probs[0] if sorted_probs else ("?", 0)
    emos_top2 = sorted_probs[1] if len(sorted_probs) > 1 else ("?", 0)

    # Load previous snapshot for event detection
    prev = load_previous_snapshot(city_slug, target_date)
    prev_bins = prev.get("bins", []) if prev else []

    # Save snapshot
    max_observed = metar_temp
    if prev and prev.get("max_observed_temp_c") is not None and metar_temp is not None:
        max_observed = max(prev["max_observed_temp_c"], metar_temp)

    snapshot = {
        "ts": ts,
        "city": city_slug,
        "target_date": target_date.isoformat(),
        "metar_temp_c": metar_temp,
        "metar_raw": metar_raw,
        "bins": [{"question": b["question"], "yes_price": b["yes_price"], "closed": b["closed"]} for b in bins],
        "emos_top1": {"bin": emos_top1[0], "prob": round(emos_top1[1], 4)},
        "emos_top2": {"bin": emos_top2[0], "prob": round(emos_top2[1], 4)},
        "emos_mean": round(mean, 2),
        "emos_std": round(math.sqrt(variance), 2) if variance > 0 else 0,
        "taf_tx_c": taf_tx,
        "bins_eliminated": sum(1 for b in bins if b["closed"]),
        "bins_active": sum(1 for b in bins if not b["closed"]),
        "max_observed_temp_c": max_observed,
    }
    save_snapshot(snapshot, target_date)

    # Evaluate shadow trades (only on first snapshot per day for YES-picking)
    yes_trade = evaluate_yes_picking(bin_probs, bin_prices)
    if yes_trade and not any(
        t.get("city") == city_slug and t.get("strategy") == "yes_picking"
        for t in _read_jsonl(DATA_DIR / "shadow_trades" / f"{target_date.isoformat()}.jsonl")
    ):
        yes_trade["ts"] = ts
        yes_trade["city"] = city_slug
        yes_trade["target_date"] = target_date.isoformat()
        yes_trade["metar_temp_at_entry"] = metar_temp
        yes_trade["trade_id"] = f"{city_slug}:{target_date}:{yes_trade['bin']}:yes"
        yes_trade["status"] = "open"
        save_shadow_trade(yes_trade, target_date)
        logger.info(f"{city_slug}: YES shadow trade → {yes_trade['bin']} @ ${yes_trade['entry_price']:.3f} (edge {yes_trade['edge']:.3f})")

    # NO-harvesting: log new NO trades for newly identified impossible bins
    no_trades = evaluate_no_harvesting(bin_probs, bin_prices)
    existing_no = [
        t.get("bin") for t in _read_jsonl(DATA_DIR / "shadow_trades" / f"{target_date.isoformat()}.jsonl")
        if t.get("city") == city_slug and t.get("strategy") == "no_harvesting"
    ]
    for nt in no_trades:
        if nt["bin"] not in existing_no:
            nt["ts"] = ts
            nt["city"] = city_slug
            nt["target_date"] = target_date.isoformat()
            nt["metar_temp_at_entry"] = metar_temp
            nt["trade_id"] = f"{city_slug}:{target_date}:{nt['bin']}:no"
            nt["status"] = "open"
            save_shadow_trade(nt, target_date)
            logger.info(f"{city_slug}: NO shadow trade → {nt['bin']} @ YES ${nt['yes_price_at_entry']:.3f}")

    # Detect bin deaths
    if prev_bins:
        deaths = detect_bin_deaths(bins, prev_bins, metar_temp, str(metar_obs_time), ts, city=city_slug, target_date=target_date)
        for death in deaths:
            death["city"] = city_slug
            death["target_date"] = target_date.isoformat()
            save_oracle_trigger(death, target_date)
            logger.info(f"{city_slug}: BIN DEATH → {death['bin_closed']} (was ${death['previous_price']:.3f})")

    # Detect confirmation
    if prev_bins:
        conf = detect_confirmation(bins, prev_bins)
        if conf:
            logger.info(f"{city_slug}: CONFIRMATION → {conf['bin']} (${conf['previous_price']:.3f} → ${conf['confirmation_price']:.3f})")


import math  # ensure available for sqrt


def run_once(target_date: Optional[date] = None, city_filter: Optional[str] = None) -> None:
    """Single invocation of the shadow tracker."""
    _ensure_dirs()
    now = _utcnow()
    ts = now.isoformat().replace("+00:00", "Z")

    if target_date is None:
        target_date = now.date() if now.hour < 12 else now.date()

    is_first_run = _is_first_run_today(target_date)

    cities_to_process = CITIES
    if city_filter:
        if city_filter in CITIES:
            cities_to_process = {city_filter: CITIES[city_filter]}
        else:
            logger.error(f"Unknown city: {city_filter}. Available: {list(CITIES.keys())}")
            return

    # Batch fetch METAR and TAF
    icao_ids = [cfg.icao for cfg in cities_to_process.values()]
    metar_data = fetch_metar(icao_ids)
    taf_data = fetch_taf_tx(icao_ids) if is_first_run else {}

    logger.info(f"Shadow tracker: {target_date} | {len(cities_to_process)} cities | first_run={is_first_run} | METAR: {len(metar_data)} stations")

    for city_slug, cfg in cities_to_process.items():
        try:
            process_city(city_slug, cfg, target_date, metar_data, taf_data, is_first_run, ts)
        except Exception as e:
            logger.error(f"{city_slug}: error — {e}")

    # Auto-retrain EMOS if params are missing (any run) or stale (first run, Sunday)
    _maybe_retrain_emos(check_staleness=is_first_run)


def _maybe_retrain_emos(check_staleness: bool = False) -> None:
    """Retrain EMOS params if any city is missing params, or weekly on Sunday.

    Missing params trigger retrain on every invocation.
    Staleness check (weekly Sunday) only runs when check_staleness=True (first run of day).
    """
    any_missing = any(
        not (EMOS_DIR / f"{city}.json").exists() for city in CITIES
    )

    stale = False
    if check_staleness:
        is_sunday = date.today().weekday() == 6
        oldest_param = None
        for city in CITIES:
            p = EMOS_DIR / f"{city}.json"
            if p.exists():
                age_days = (date.today() - date.fromtimestamp(p.stat().st_mtime)).days
                if oldest_param is None or age_days > oldest_param:
                    oldest_param = age_days
        stale = is_sunday and (oldest_param is None or oldest_param >= 7)

    should_retrain = any_missing or stale

    if not should_retrain:
        return

    reason = "missing params" if any_missing else "weekly retrain"
    logger.info(f"Auto-retrain triggered: {reason}")

    try:
        results = train_all_cities(EMOS_DIR, DATA_DIR / "forecasts", DATA_DIR / "resolutions")
        trained = sum(1 for v in results.values() if v is not None)
        logger.info(f"Auto-retrain complete: {trained}/{len(results)} cities trained")
    except Exception as e:
        logger.error(f"Auto-retrain failed (non-fatal): {e}")


def resolve(target_date: Optional[date] = None) -> None:
    """Check resolutions for yesterday (or specified date)."""
    _ensure_dirs()
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    logger.info(f"Checking resolutions for {target_date}")
    resolutions = check_resolved_markets(target_date)
    if resolutions:
        save_resolutions(resolutions, target_date)
        for r in resolutions:
            logger.info(f"  {r['city']}: winner={r['winning_bin']} | trades={r['total_trades']} | P&L=${r['daily_pnl']:.2f}")
    else:
        logger.info(f"  No resolved markets for {target_date}")

    metrics = update_metrics()
    readiness = update_readiness(metrics)
    logger.info(f"Metrics: {metrics.get('total_shadow_trades', 0)} trades, {metrics.get('accuracy', 0):.1%} accuracy, ${metrics.get('total_pnl', 0):.2f} P&L")
    logger.info(f"Readiness: {readiness.get('status', '?')}")


def print_metrics() -> None:
    """Print current metrics."""
    path = DATA_DIR / "metrics.json"
    if not path.exists():
        print("No metrics yet. Run the tracker first.")
        return
    metrics = json.loads(path.read_text())
    print(json.dumps(metrics, indent=2))


def print_status() -> None:
    """Print formatted terminal dashboard."""
    now = _utcnow()
    today = now.date()
    yesterday = today - timedelta(days=1)

    # Load readiness
    readiness_path = DATA_DIR / "readiness.json"
    readiness = json.loads(readiness_path.read_text()) if readiness_path.exists() else {}
    status = readiness.get("status", "unknown")
    days_obs = readiness.get("days_observed", 0)
    days_left = max(0, 14 - days_obs)

    # Count snapshot files to determine observation days
    snap_dir = DATA_DIR / "snapshots"
    snap_days = len(list(snap_dir.glob("*.jsonl"))) if snap_dir.exists() else 0

    W = 70  # display width
    BAR = "═" * W
    THIN = "─" * W

    print(f"\n{BAR}")
    print(f"  ASIAN WEATHER SHADOW TRACKER — Day {days_obs} of 14")
    print(f"  Status: {status.upper()} | Readiness: {readiness.get('decision_status', 'pending')}")
    print(BAR)

    # ── Today's snapshot data ───────────────────────────────────
    today_snaps = _read_jsonl(DATA_DIR / "snapshots" / f"{today.isoformat()}.jsonl")
    today_trades = _read_jsonl(DATA_DIR / "shadow_trades" / f"{today.isoformat()}.jsonl")
    today_triggers = _read_jsonl(DATA_DIR / "oracle_triggers" / f"{today.isoformat()}.jsonl")

    # Get latest snapshot per city
    latest: Dict[str, Dict[str, Any]] = {}
    for s in today_snaps:
        latest[s.get("city", "")] = s

    yes_count = sum(1 for t in today_trades if t.get("strategy") == "yes_picking")
    no_count = sum(1 for t in today_trades if t.get("strategy") == "no_harvesting")

    print(f"\n  TODAY ({today}) — {len(latest)} cities, {yes_count} YES + {no_count} NO shadow trades")
    print()
    print(f"  {'City':<14} {'METAR':>5}  {'Model Fav':>12}  {'YES Trade':>16}  {'NO':>3}  {'Bins':>7}")
    print(f"  {THIN}")

    city_order = ["seoul", "hong-kong", "shanghai", "shenzhen", "beijing", "lucknow", "warsaw", "munich"]
    for city in city_order:
        snap = latest.get(city)
        if not snap:
            print(f"  {city:<14} {'—':>5}  {'no data':>12}")
            continue

        metar_t = snap.get("metar_temp_c")
        metar_str = f"{metar_t}°C" if metar_t is not None else "—"
        top1 = snap.get("emos_top1", {})
        fav_bin = top1.get("bin", "?")
        fav_prob = top1.get("prob", 0)
        fav_str = f"{fav_bin} @{fav_prob:.2f}"

        # Find this city's YES trade
        city_yes = [t for t in today_trades if t.get("city") == city and t.get("strategy") == "yes_picking"]
        if city_yes:
            t = city_yes[0]
            yes_str = f"{t['bin'][:6]} ${t['entry_price']:.3f}"
        else:
            yes_str = "—"

        city_no = sum(1 for t in today_trades if t.get("city") == city and t.get("strategy") == "no_harvesting")
        elim = snap.get("bins_eliminated", 0)
        active = snap.get("bins_active", 11)

        print(f"  {city:<14} {metar_str:>5}  {fav_str:>12}  {yes_str:>16}  {city_no:>3}  {active:>2}/{active+elim}")

    # ── Bin deaths today ────────────────────────────────────────
    if today_triggers:
        print(f"\n  BIN DEATHS TODAY: {len(today_triggers)}")
        for tr in today_triggers[-5:]:  # last 5
            print(f"    {tr.get('city','?')}: {tr.get('bin_closed','?')} closed (was ${tr.get('previous_price',0):.3f})")

    # ── Yesterday's resolutions ─────────────────────────────────
    res_path = DATA_DIR / "resolutions" / f"{yesterday.isoformat()}.json"
    if res_path.exists():
        try:
            res_data = json.loads(res_path.read_text())
            if not isinstance(res_data, list):
                res_data = [res_data]
        except json.JSONDecodeError:
            res_data = []

        if res_data:
            print(f"\n  YESTERDAY ({yesterday}) — Resolved")
            print()
            print(f"  {'City':<14} {'Winner':>10}  {'YES P&L':>9}  {'NO P&L':>9}  {'Total':>9}")
            print(f"  {THIN}")
            day_total = 0.0
            for r in res_data:
                city = r.get("city", "?")
                winner = r.get("winning_bin", "?")
                # Extract short winner label
                wm = re.search(r"be (.+?) on", winner)
                w_short = wm.group(1) if wm else winner[:10]

                yes_pnl = sum(t["pnl"] for t in r.get("shadow_trades", []) if t.get("strategy") == "yes_picking" or "yes" in str(t.get("hit", "")))
                no_pnl = sum(t["pnl"] for t in r.get("shadow_trades", []) if t.get("strategy") == "no_harvesting")
                # If no strategy field, just use total
                total = r.get("daily_pnl", 0)
                day_total += total
                print(f"  {city:<14} {w_short:>10}  ${yes_pnl:>+8.2f}  ${no_pnl:>+8.2f}  ${total:>+8.2f}")
            print(f"  {'':14} {'':>10}  {'':>9}  {'':>9}  ${day_total:>+8.2f}")

    # ── Running totals ──────────────────────────────────────────
    metrics_path = DATA_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        total_trades = metrics.get("total_shadow_trades", 0)
        total_hits = metrics.get("hits", 0)
        accuracy = metrics.get("accuracy", 0)
        total_pnl = metrics.get("total_pnl", 0)
        avg_pnl = metrics.get("avg_pnl_per_trade", 0)
        delta_t = metrics.get("median_delta_t_minutes")

        # Count YES vs NO from all trade files
        all_yes = 0
        all_no = 0
        all_yes_hits = 0
        all_no_hits = 0
        all_yes_pnl = 0.0
        all_no_pnl = 0.0

        res_dir = DATA_DIR / "resolutions"
        if res_dir.exists():
            for rf in res_dir.glob("*.json"):
                try:
                    rdata = json.loads(rf.read_text())
                    if not isinstance(rdata, list):
                        rdata = [rdata]
                    for city_r in rdata:
                        for st in city_r.get("shadow_trades", []):
                            strat = st.get("strategy", "")
                            if strat == "yes_picking":
                                all_yes += 1
                                if st.get("hit"):
                                    all_yes_hits += 1
                                all_yes_pnl += st.get("pnl", 0)
                            elif strat == "no_harvesting":
                                all_no += 1
                                if st.get("hit"):
                                    all_no_hits += 1
                                all_no_pnl += st.get("pnl", 0)
                except json.JSONDecodeError:
                    pass

        print(f"\n  RUNNING TOTALS ({metrics.get('days_observed', 0)} days)")
        print()
        print(f"  {'Strategy':<18} {'Trades':>7} {'Hits':>6} {'Accuracy':>9} {'Total P&L':>10}")
        print(f"  {THIN}")
        yes_acc = f"{all_yes_hits/all_yes:.1%}" if all_yes > 0 else "—"
        no_acc = f"{all_no_hits/all_no:.1%}" if all_no > 0 else "—"
        comb_acc = f"{accuracy:.1%}" if total_trades > 0 else "—"
        print(f"  {'YES-picking':<18} {all_yes:>7} {all_yes_hits:>6} {yes_acc:>9} ${all_yes_pnl:>+9.2f}")
        print(f"  {'NO-harvesting':<18} {all_no:>7} {all_no_hits:>6} {no_acc:>9} ${all_no_pnl:>+9.2f}")
        print(f"  {'Combined':<18} {total_trades:>7} {total_hits:>6} {comb_acc:>9} ${total_pnl:>+9.2f}")

        # Delta_T
        dt_str = f"median {delta_t:.0f} min" if delta_t else "no data yet"
        print(f"\n  Delta_T (oracle lag): {dt_str}")

        # Kill/promote countdown
        if days_left > 0:
            print(f"\n  Kill/promote decision in {days_left} days")
            print(f"  Kill if: accuracy < 15% OR avg P&L < $0")
            print(f"  Promote if: accuracy > 25% AND avg P&L > $1.50/trade AND Delta_T > 30min")
    else:
        print(f"\n  No metrics yet — run the tracker to start collecting data.")

    print(f"\n{BAR}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Asian Weather Shadow Tracker")
    parser.add_argument("--once", action="store_true", help="Run single invocation")
    parser.add_argument("--resolve-only", action="store_true", help="Check resolutions only")
    parser.add_argument("--metrics", action="store_true", help="Print current metrics (JSON)")
    parser.add_argument("--status", action="store_true", help="Print formatted status dashboard")
    parser.add_argument("--city", type=str, help="Filter to single city")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--train", action="store_true", help="Train EMOS params for all cities")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None

    if args.train:
        _ensure_dirs()
        forecasts_dir = DATA_DIR / "forecasts"
        resolutions_dir = DATA_DIR / "resolutions"
        logger.info("Training EMOS params for all cities...")
        results = train_all_cities(EMOS_DIR, forecasts_dir, resolutions_dir)
        trained = sum(1 for v in results.values() if v is not None)
        skipped = sum(1 for v in results.values() if v is None)
        print(f"\nEMOS Training Complete: {trained} trained, {skipped} skipped")
        for city, params in sorted(results.items()):
            if params:
                print(f"  {city}: a={params.a:.3f} b={params.b:.3f} c={params.c:.3f} d={params.d:.3f} e={params.e:.3f} (n={params.trained_on})")
            else:
                print(f"  {city}: SKIPPED (insufficient data or validation failed)")
        return

    if args.status:
        print_status()
    elif args.metrics:
        print_metrics()
    elif args.resolve_only:
        resolve(target_date)
    elif args.once:
        run_once(target_date, args.city)
        # Also check yesterday's resolutions
        resolve()
    else:
        # Default: run once + resolve
        run_once(target_date, args.city)
        resolve()


if __name__ == "__main__":
    main()
