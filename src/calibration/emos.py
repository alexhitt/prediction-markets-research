"""
EMOS (Ensemble Model Output Statistics) calibration for 1°C temperature bins.

Produces calibrated Gaussian probability distributions from ECMWF ensemble
forecasts. Designed for Polymarket weather markets resolving against Weather
Underground airport station data.

Cold start plan:
  Days 1-7:   Naive Gaussian from pooled ensemble spread (no training data)
  Days 8-14:  Rolling bias correction (a + b*mean)
  Days 15-30: Proper EMOS (5 params, CRPS-minimized)
  Day 30+:    Full EMOS with 30-day rolling window

Sources:
  - Rasp & Lerch (2018): EMOS outperforms BMA at fraction of compute
  - ECMWF Tech Memo 918/931: IFS 48r1/49r1 surface temp improvements
  - Deep Research 2026-03-30: AIFS deterministic unavailable, use AIFS ENS mean
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date, timedelta, timezone, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from scipy.optimize import minimize
from scipy.stats import norm


# ── City Configuration ──────────────────────────────────────────────

@dataclass(frozen=True)
class CityConfig:
    slug: str
    icao: str
    lat: float
    lon: float
    tz: str


CITIES: Dict[str, CityConfig] = {
    "seoul": CityConfig("seoul", "RKSI", 37.4602, 126.4407, "Asia/Seoul"),
    "hong-kong": CityConfig("hong-kong", "VHHH", 22.3080, 113.9185, "Asia/Hong_Kong"),
    "shanghai": CityConfig("shanghai", "ZSPD", 31.1443, 121.8083, "Asia/Shanghai"),
    "shenzhen": CityConfig("shenzhen", "ZGSZ", 22.6393, 113.8107, "Asia/Shanghai"),
    "beijing": CityConfig("beijing", "ZBAA", 40.0801, 116.5846, "Asia/Shanghai"),
    "lucknow": CityConfig("lucknow", "VILK", 26.7606, 80.8893, "Asia/Kolkata"),
    "warsaw": CityConfig("warsaw", "EPWA", 52.1657, 20.9671, "Europe/Warsaw"),
    "munich": CityConfig("munich", "EDDM", 48.3537, 11.7750, "Europe/Berlin"),
}


# ── Ensemble Fetching ───────────────────────────────────────────────

ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
HTTP_TIMEOUT = 20


@dataclass
class EnsembleForecast:
    city: str
    target_date: str
    ifs_members: List[float]
    aifs_members: List[float]
    ifs_mean: float
    ifs_var: float
    aifs_mean: float
    fetched_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fetch_ensemble_forecast(
    lat: float, lon: float, forecast_days: int = 2
) -> Tuple[List[float], List[float]]:
    """Fetch IFS and AIFS ensemble daily-max members from Open-Meteo.

    Returns (ifs_members, aifs_members) for the first forecast day.
    IFS: 50 perturbed members. AIFS: 50 perturbed members.
    Retries once on failure with a 5-second backoff.
    """
    import time

    ifs_members: List[float] = []
    aifs_members: List[float] = []

    for model, target in [("ecmwf_ifs025", ifs_members), ("ecmwf_aifs025", aifs_members)]:
        for attempt in range(2):
            try:
                resp = requests.get(
                    ENSEMBLE_API,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "daily": "temperature_2m_max",
                        "models": model,
                        "forecast_days": forecast_days,
                    },
                    timeout=HTTP_TIMEOUT,
                )
                resp.raise_for_status()
                daily = resp.json().get("daily", {})
                for key, vals in daily.items():
                    if "member" in key and vals:
                        val = vals[0]  # first forecast day
                        if val is not None:
                            target.append(val)
                break  # success — no retry needed
            except (requests.RequestException, KeyError, IndexError) as exc:
                if attempt == 0:
                    logger.warning(
                        "Ensemble fetch failed for %s (attempt 1/2, retrying in 5s): %s",
                        model, exc,
                    )
                    time.sleep(5)
                else:
                    logger.warning(
                        "Ensemble fetch failed for %s (attempt 2/2, giving up): %s",
                        model, exc,
                    )

    return ifs_members, aifs_members


def fetch_historical_actuals(
    lat: float, lon: float, tz: str, days: int = 45
) -> Dict[str, float]:
    """Fetch daily high temps (°C) from Open-Meteo historical archive.

    Returns {date_str: temp_c} for the last N days.
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    try:
        resp = requests.get(
            ARCHIVE_API,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "temperature_2m_max",
                "timezone": tz,
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        dates = data.get("daily", {}).get("time", [])
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        return {d: t for d, t in zip(dates, temps) if t is not None}
    except requests.RequestException:
        return {}


# ── Naive Gaussian (Bootstrap Phase, Days 1-7) ─────────────────────

def compute_naive_gaussian(
    ifs_members: Sequence[float],
    aifs_members: Sequence[float],
) -> Tuple[float, float]:
    """Compute mean and std from pooled IFS + AIFS ensemble members.

    Used during bootstrap phase before EMOS training data is available.
    The spread is intentionally wider than individual model ensembles to
    compensate for known underdispersion.
    """
    pooled = list(ifs_members) + list(aifs_members)
    if len(pooled) < 3:
        # Fallback: use IFS only, or a wide default
        if len(ifs_members) >= 3:
            pooled = list(ifs_members)
        else:
            return (15.0, 3.0)  # wide default, safe fallback

    mean = statistics.mean(pooled)
    std = statistics.stdev(pooled)
    # Inflate std by 1.3x to partially correct for underdispersion
    # (conservative — proper EMOS will replace this after training)
    return mean, max(std * 1.3, 1.0)


# ── EMOS Training ──────────────────────────────────────────────────

@dataclass
class EMOSParams:
    a: float  # mean bias
    b: float  # IFS mean coefficient
    c: float  # AIFS mean coefficient
    d: float  # variance bias
    e: float  # variance scaling
    station: str
    trained_on: int  # number of training samples
    trained_at: str


def _crps_gaussian(mu: float, sigma: float, obs: float) -> float:
    """Closed-form CRPS for a Gaussian distribution.

    CRPS = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (obs - mu) / sigma
    """
    if sigma <= 0:
        return abs(obs - mu)
    z = (obs - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


@dataclass
class TrainingSample:
    ifs_mean: float
    aifs_mean: float
    ifs_var: float
    actual_high: float


def train_emos(
    samples: Sequence[TrainingSample],
    station: str = "unknown",
) -> Optional[EMOSParams]:
    """Fit EMOS parameters by minimizing mean CRPS over training samples.

    EMOS_Mean = a + b * IFS_Mean + c * AIFS_Mean
    EMOS_Var  = d + e * IFS_Var
    (variance must be positive)

    Parameters optimized via L-BFGS-B with bounds.
    Returns None if too few samples (<4).
    """
    if len(samples) < 4:
        return None

    # Regularization strength scales inversely with sample count.
    # With 4 samples, λ=0.5 keeps params near pass-through.
    # With 30+ samples, λ→0.07 and data dominates.
    reg_lambda = 2.0 / len(samples)

    def objective(params: Sequence[float]) -> float:
        a, b, c, d, e = params
        total_crps = 0.0
        for s in samples:
            mu = a + b * s.ifs_mean + c * s.aifs_mean
            var = max(d + e * s.ifs_var, 0.01)
            sigma = math.sqrt(var)
            total_crps += _crps_gaussian(mu, sigma, s.actual_high)
        crps = total_crps / len(samples)

        # Regularize toward pass-through: a≈0, b+c≈1, e≈1
        # Prevents degenerate fits where large |a| compensates for b+c >> 1
        reg = reg_lambda * (a * a + (b + c - 1.0) ** 2 + (e - 1.0) ** 2)
        return crps + reg

    # Initial guess: unbiased pass-through with moderate variance
    x0 = [0.0, 0.7, 0.3, 2.25, 1.0]
    # d lower bound 2.25 = σ floor of 1.5°C (sqrt(2.25))
    # This prevents overfitting on small samples — the optimizer cannot
    # collapse variance below what is physically reasonable for daily high
    # temperature forecasts at 1°C bin resolution.
    bounds = [
        (-10.0, 10.0),   # a: bias can be several degrees
        (0.0, 2.0),      # b: IFS weight
        (0.0, 2.0),      # c: AIFS weight
        (2.25, 10.0),    # d: minimum variance (σ >= 1.5°C)
        (0.01, 5.0),     # e: variance scaling
    ]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        return None

    a, b, c, d, e = result.x
    return EMOSParams(
        a=round(a, 4),
        b=round(b, 4),
        c=round(c, 4),
        d=round(d, 4),
        e=round(e, 4),
        station=station,
        trained_on=len(samples),
        trained_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


# ── EMOS Prediction ────────────────────────────────────────────────

def emos_predict(
    params: EMOSParams,
    ifs_mean: float,
    aifs_mean: float,
    ifs_var: float,
) -> Tuple[float, float]:
    """Predict calibrated (mean, variance) from EMOS parameters."""
    mu = params.a + params.b * ifs_mean + params.c * aifs_mean
    var = max(params.d + params.e * ifs_var, 0.01)
    return mu, var


# ── Bin Probability Computation ────────────────────────────────────

def bin_probabilities(
    mean: float,
    variance: float,
    bin_edges: Sequence[Tuple[Optional[int], Optional[int]]],
) -> Dict[str, float]:
    """Compute probability for each 1°C bin from a Gaussian PDF.

    bin_edges: list of (lower, upper) where None means open-ended.
    Example: [(None, 10), (11, 11), (12, 12), ..., (20, None)]
    For bins like "10°C or below": lower=None, upper=10
    For bins like "16°C": lower=16, upper=16
    For bins like "20°C or higher": lower=20, upper=None

    Returns dict mapping bin label to probability.
    """
    sigma = math.sqrt(max(variance, 0.01))
    probs: Dict[str, float] = {}

    for lower, upper in bin_edges:
        if lower is None and upper is not None:
            # "X°C or below" — P(T <= upper + 0.5)
            p = norm.cdf(upper + 0.5, loc=mean, scale=sigma)
            label = f"{upper}°C or below"
        elif upper is None and lower is not None:
            # "X°C or higher" — P(T > lower - 0.5)
            p = 1.0 - norm.cdf(lower - 0.5, loc=mean, scale=sigma)
            label = f"{lower}°C or higher"
        elif lower is not None and upper is not None:
            # Exact degree bin — P(lower - 0.5 < T <= upper + 0.5)
            p = norm.cdf(upper + 0.5, loc=mean, scale=sigma) - norm.cdf(
                lower - 0.5, loc=mean, scale=sigma
            )
            label = f"{lower}°C"
        else:
            continue
        probs[label] = max(0.0, min(1.0, p))

    return probs


def parse_bin_edges_from_questions(questions: Sequence[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Parse Polymarket bin questions into (lower, upper) edge tuples.

    Handles:
      "Will the highest temperature in Seoul be 16°C on March 30?" → (16, 16)
      "Will the highest temperature in Seoul be 10°C or below on ..." → (None, 10)
      "Will the highest temperature in Seoul be 20°C or higher on ..." → (20, None)
    """
    import re
    edges: List[Tuple[Optional[int], Optional[int]]] = []

    for q in questions:
        q_lower = q.lower()
        # "X°C or below" / "X°C or lower"
        m = re.search(r"(\d+)\s*°?\s*c\s+or\s+(?:below|lower)", q_lower)
        if m:
            edges.append((None, int(m.group(1))))
            continue
        # "X°C or higher" / "X°C or above"
        m = re.search(r"(\d+)\s*°?\s*c\s+or\s+(?:higher|above)", q_lower)
        if m:
            edges.append((int(m.group(1)), None))
            continue
        # "be X°C on" (exact degree)
        m = re.search(r"be\s+(\d+)\s*°?\s*c\s+on", q_lower)
        if m:
            val = int(m.group(1))
            edges.append((val, val))
            continue
        # Fallback: try to find any number followed by °C
        m = re.search(r"(\d+)\s*°?\s*c", q_lower)
        if m:
            val = int(m.group(1))
            edges.append((val, val))

    return edges


# ── Bayesian Intraday Update ───────────────────────────────────────

def bayesian_update(
    emos_mean: float,
    emos_var: float,
    metar_temp: float,
    forecast_temp_at_hour: Optional[float] = None,
) -> Tuple[float, float]:
    """Shift EMOS mean based on METAR observation.

    Simple delta shift: if the current temperature is warmer than expected,
    shift the daily max forecast upward proportionally. The shift is dampened
    because current temp != daily max.

    If forecast_temp_at_hour is not available, use a heuristic: the daily max
    is typically 2-4°C above the morning temperature at mid-latitude stations.
    """
    if forecast_temp_at_hour is not None:
        delta = metar_temp - forecast_temp_at_hour
    else:
        # Heuristic: if METAR temp is already above the EMOS mean,
        # the daily max will likely be even higher
        delta = max(0.0, metar_temp - (emos_mean - 2.0)) * 0.3

    # Dampen the shift — observed temp is not the max yet
    updated_mean = emos_mean + delta * 0.5
    # Slightly reduce variance as we get more observation data
    updated_var = max(emos_var * 0.9, 0.5)

    return updated_mean, updated_var


# ── EMOS Param Persistence ─────────────────────────────────────────

def save_emos_params(params: EMOSParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2))


def load_emos_params(path: Path) -> Optional[EMOSParams]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return EMOSParams(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


# ── Hindcast Fetching (for Training) ─────────────────────────────

import logging

logger = logging.getLogger(__name__)


def fetch_hindcast_ensemble(
    lat: float, lon: float, past_days: int = 10
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Fetch recent hindcast ensemble members from Open-Meteo.

    Returns {date_str: (ifs_members, aifs_members)} for each day with data.
    The ensemble API accepts past_days to return operational forecast data
    from recent model runs (not reanalysis).
    """
    result: Dict[str, Tuple[List[float], List[float]]] = {}

    for model_name, member_idx in [("ecmwf_ifs025", 0), ("ecmwf_aifs025", 1)]:
        try:
            resp = requests.get(
                ENSEMBLE_API,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max",
                    "models": model_name,
                    "past_days": past_days,
                    "forecast_days": 1,
                },
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            dates = daily.get("time", [])

            # Collect member values per date
            member_cols = sorted(
                k for k in daily.keys() if "member" in k and daily[k]
            )
            for i, d in enumerate(dates):
                members = []
                for col in member_cols:
                    vals = daily[col]
                    if i < len(vals) and vals[i] is not None:
                        members.append(vals[i])
                if not members:
                    continue
                if d not in result:
                    result[d] = ([], [])
                if member_idx == 0:
                    result[d] = (members, result[d][1])
                else:
                    result[d] = (result[d][0], members)
        except (requests.RequestException, KeyError, IndexError) as exc:
            logger.warning("Hindcast fetch failed for %s: %s", model_name, exc)

    return result


def assemble_training_data(
    city_slug: str,
    cfg: CityConfig,
    emos_dir: Path,
    forecasts_dir: Path,
    resolutions_dir: Path,
) -> List[TrainingSample]:
    """Assemble (ifs_mean, aifs_mean, ifs_var, actual_high) training pairs.

    Combines three data sources:
    1. Local saved forecasts + resolutions (highest quality — real-time data)
    2. Hindcast ensemble members from Open-Meteo past_days API
    3. Archive actuals from Open-Meteo historical API

    Local data takes priority over hindcast for overlapping dates.
    """
    samples_by_date: Dict[str, TrainingSample] = {}

    # Source 1: local forecast + resolution files
    for res_file in sorted(resolutions_dir.glob("*.json")):
        day = res_file.stem
        fc_file = forecasts_dir / f"{day}.json"
        if not fc_file.exists():
            continue
        try:
            res_data = json.loads(res_file.read_text())
            fc_data = json.loads(fc_file.read_text())
            fc_entries = fc_data if isinstance(fc_data, list) else [fc_data]
        except (json.JSONDecodeError, IOError):
            continue

        fc_map = {e["city"]: e for e in fc_entries}
        if city_slug not in fc_map:
            continue

        # Find actual high from resolution
        city_res = [r for r in res_data if r.get("city") == city_slug]
        if not city_res:
            continue
        winning_bin = city_res[0].get("winning_bin", "")
        # Parse temperature from winning bin question
        import re
        m = re.search(r"(\d+)\s*°?\s*[Cc]", winning_bin)
        if not m:
            continue
        actual_high = float(m.group(1))

        fc = fc_map[city_slug]
        samples_by_date[day] = TrainingSample(
            ifs_mean=fc["ifs_mean"],
            aifs_mean=fc["aifs_mean"],
            ifs_var=fc["ifs_var"],
            actual_high=actual_high,
        )

    # Source 2+3: hindcast ensemble + archive actuals
    hindcast = fetch_hindcast_ensemble(cfg.lat, cfg.lon, past_days=10)
    actuals = fetch_historical_actuals(cfg.lat, cfg.lon, cfg.tz, days=45)

    for day_str, (ifs_members, aifs_members) in hindcast.items():
        if day_str in samples_by_date:
            continue  # local data takes priority
        if day_str not in actuals:
            continue  # no observed actual for this day
        if len(ifs_members) < 3:
            continue

        ifs_mean = statistics.mean(ifs_members)
        ifs_var = statistics.variance(ifs_members) if len(ifs_members) > 1 else 1.0
        aifs_mean = statistics.mean(aifs_members) if aifs_members else ifs_mean

        samples_by_date[day_str] = TrainingSample(
            ifs_mean=ifs_mean,
            aifs_mean=aifs_mean,
            ifs_var=ifs_var,
            actual_high=actuals[day_str],
        )

    return list(samples_by_date.values())


def train_all_cities(
    emos_dir: Path,
    forecasts_dir: Path,
    resolutions_dir: Path,
) -> Dict[str, Optional[EMOSParams]]:
    """Train EMOS parameters for all 8 cities.

    For each city: assemble training data, train EMOS, validate that fitted σ
    is wider than naive σ, save params if valid.

    Returns {city_slug: params_or_None}.
    """
    results: Dict[str, Optional[EMOSParams]] = {}

    for city_slug, cfg in CITIES.items():
        samples = assemble_training_data(
            city_slug, cfg, emos_dir, forecasts_dir, resolutions_dir
        )
        logger.info(
            "Training %s: %d samples assembled", city_slug, len(samples)
        )

        if len(samples) < 4:
            logger.warning(
                "Skipping %s: only %d samples (need >=4)", city_slug, len(samples)
            )
            results[city_slug] = None
            continue

        params = train_emos(samples, station=city_slug)
        if params is None:
            logger.warning("Skipping %s: optimizer did not converge", city_slug)
            results[city_slug] = None
            continue

        # Validation gate: fitted σ must be wider than naive σ
        # naive σ = sqrt(median_ifs_var) * 1.3, matching compute_naive_gaussian()
        median_ifs_var = statistics.median(s.ifs_var for s in samples)
        fitted_var = max(params.d + params.e * median_ifs_var, 0.01)
        fitted_sigma = math.sqrt(fitted_var)

        naive_sigma = max(math.sqrt(median_ifs_var) * 1.3, 1.0)

        if fitted_sigma < naive_sigma:
            logger.warning(
                "Skipping %s: fitted σ=%.2f < naive σ=%.2f (overfit suspected)",
                city_slug, fitted_sigma, naive_sigma,
            )
            results[city_slug] = None
            continue

        # Save
        param_path = emos_dir / f"{city_slug}.json"
        save_emos_params(params, param_path)
        logger.info(
            "Trained %s: %d samples, σ=%.2f (naive=%.2f), params=%s",
            city_slug, len(samples), fitted_sigma, naive_sigma,
            f"a={params.a:.3f} b={params.b:.3f} c={params.c:.3f} d={params.d:.3f} e={params.e:.3f}",
        )
        results[city_slug] = params

    return results
