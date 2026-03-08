"""
pipeline/fetch_data.py
Fetches current and historical AQI data from the WAQI API for Ahmedabad stations.

Usage (standalone test):
    python pipeline/fetch_data.py
"""

import logging
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

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("HTTP error fetching station '%s': %s", station, exc)
        return None

    payload = response.json()

    if payload.get("status") != "ok":
        logger.warning("WAQI returned status '%s' for station '%s'", payload.get("status"), station)
        return None

    data = payload["data"]
    iaqi = data.get("iaqi", {})

    # AQI value
    raw_aqi = data.get("aqi")
    if raw_aqi == "-" or raw_aqi is None:
        logger.warning("No AQI value for station '%s'", station)
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
        "aqi":                float(raw_aqi),
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
    WAQI free tier returns the most recent reading — if it matches yesterday's date
    it's returned, otherwise we return the current reading as best-effort.
    """
    reading = fetch_current_aqi(station)
    if reading is None:
        return None

    yesterday = date.today() - timedelta(days=1)
    if reading["date"] != yesterday:
        logger.debug(
            "Station '%s' returned date %s, expected %s — using as best-effort",
            station, reading["date"], yesterday,
        )
        # Override the date to yesterday so the DB upsert targets the right row
        reading = {**reading, "date": yesterday}

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
