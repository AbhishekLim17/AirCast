"""
pipeline/fetch_data.py
Fetches current and historical AQI data from the WAQI API for Ahmedabad stations.

Usage (standalone test):
    python pipeline/fetch_data.py
"""

import logging
import json
import time
import requests
from datetime import date, timedelta
from typing import Optional

from config import WAQI_API_TOKEN, WAQI_BASE_URL, AHMEDABAD_STATIONS

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_current_aqi(station: str) -> Optional[dict]:
    """
    Fetch the latest AQI reading for a single station.

    Returns a dict:
        {
            "station":            str,
            "date":               date,
            "aqi":                float,
            "dominant_pollutant": str | None,
            "pm25":               float | None,
            "pm10":               float | None,
            "no2":                float | None,
            "so2":                float | None,
            "co":                 float | None,
            "o3":                 float | None,
            "temperature":        float | None,
            "humidity":           float | None,
            "wind_speed":         float | None,
        }
    Returns None if the station is unreachable or returns an error.
    """
    url = f"{WAQI_BASE_URL}/feed/{station}/"
    params = {"token": WAQI_API_TOKEN}

    last_exc: Exception | None = None
    for attempt in range(1, 4):  # 3 attempts with exponential backoff
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            break  # success
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "HTTP error fetching station '%s' (attempt %d/3): %s",
                station, attempt, exc,
            )
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Invalid JSON response from station '%s': %s", station, exc)
            return None
    else:
        logger.error("All 3 attempts failed for station '%s': %s", station, last_exc)
        return None

    if payload.get("status") != "ok":
        logger.warning("WAQI returned status '%s' for station '%s'", payload.get("status"), station)
        return None

    data = payload["data"]
    iaqi = data.get("iaqi", {})

    # AQI value — can be "-", None, or a numeric string
    raw_aqi = data.get("aqi")
    if raw_aqi == "-" or raw_aqi is None:
        logger.warning("No AQI value for station '%s'", station)
        return None

    try:
        aqi_float = float(raw_aqi)
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid AQI value '%s' for station '%s': %s", raw_aqi, station, exc)
        return None

    # Observation date — WAQI returns "YYYY-MM-DD" or a datetime string
    time_block = data.get("time", {})
    obs_date_str = time_block.get("s", "")[:10]  # take only YYYY-MM-DD part
    try:
        obs_date = date.fromisoformat(obs_date_str)
    except ValueError:
        obs_date = date.today()
        logger.warning("Could not parse date '%s' for station '%s', using today", obs_date_str, station)

    return {
        "station":            station,
        "date":               obs_date,
        "aqi":                aqi_float,
        "dominant_pollutant": data.get("dominentpol"),
        "pm25":               _iaqi_val(iaqi, "pm25"),
        "pm10":               _iaqi_val(iaqi, "pm10"),
        "no2":                _iaqi_val(iaqi, "no2"),
        "so2":                _iaqi_val(iaqi, "so2"),
        "co":                 _iaqi_val(iaqi, "co"),
        "o3":                 _iaqi_val(iaqi, "o3"),
        "temperature":        _iaqi_val(iaqi, "t"),
        "humidity":           _iaqi_val(iaqi, "h"),
        "wind_speed":         _iaqi_val(iaqi, "w"),
    }


def fetch_all_stations() -> list[dict]:
    """
    Fetch current AQI for every station in AHMEDABAD_STATIONS.
    Skips stations that return None and logs a warning.

    Returns a list of reading dicts (may be empty if all stations fail).
    """
    results = []
    for station in AHMEDABAD_STATIONS:
        reading = fetch_current_aqi(station)
        if reading:
            results.append(reading)
            logger.info("Fetched AQI %.0f for '%s' on %s", reading["aqi"], station, reading["date"])
        else:
            logger.warning("Skipped station '%s' — no valid data returned", station)
    return results


def fetch_yesterday_aqi(station: str) -> Optional[dict]:
    """
    Attempt to retrieve yesterday's AQI for a station.
    WAQI free tier returns the most recent reading available.
    If the most recent reading is NOT yesterday's, returns None to skip this read
    (rather than fabricating a historical record).
    """
    reading = fetch_current_aqi(station)
    if reading is None:
        return None

    yesterday = date.today() - timedelta(days=1)
    if reading["date"] != yesterday:
        logger.warning(
            "Most recent reading from '%s' is %s, not %s (yesterday). Skipping.",
            station, reading["date"], yesterday,
        )
        return None  # Don't fabricate historical data

    return reading


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _iaqi_val(iaqi: dict, key: str) -> Optional[float]:
    """Safely extract a numeric value from the WAQI iaqi block."""
    entry = iaqi.get(key)
    if entry is None:
        return None
    try:
        return float(entry.get("v"))
    except (TypeError, ValueError):
        return None


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n── Fetching all Ahmedabad stations ──")
    readings = fetch_all_stations()

    if not readings:
        print("No data returned. Check your WAQI_API_TOKEN in .env")
    else:
        for r in readings:
            print(
                f"  {r['station']:30s}  AQI: {r['aqi']:6.1f}"
                f"  Pollutant: {r['dominant_pollutant'] or 'N/A':8s}"
                f"  Date: {r['date']}"
            )
