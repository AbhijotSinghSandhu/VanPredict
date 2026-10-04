"""Download ERA5 liquid rain and build hourly features for the 500 m grid.

Run from the project root. Downloads are cached per source cell and year.
Shared weather is stored once and linked through grid_weather_mapping.csv.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services.spatial_rainfall import (  # noqa: E402
    DATASET_DIR, FEATURES, THRESHOLDS, TIMEZONE, create_features,
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def geometry_centroid(geometry):
    """Area-weighted lon/lat centroid for small WGS84 city grid polygons.

    Subtracting an origin avoids cancellation in the polygon area formula.
    This is a sampling location, not a metric terrain calculation.
    """
    polygons = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
                else geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                else None)
    if polygons is None:
        raise ValueError("Expected Polygon or MultiPolygon grid geometry")
    weighted = np.zeros(2)
    total_area = 0.0
    for polygon in polygons:
        for i, ring in enumerate(polygon):
            points = np.asarray(ring, dtype=float)[:, :2]
            origin = points[0]
            points = points - origin
            a, b = points[:-1], points[1:]
            cross = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
            signed_area = cross.sum() / 2
            if abs(signed_area) < 1e-16:
                continue
            centroid = ((a + b) * cross[:, None]).sum(axis=0) / (6 * signed_area) + origin
            weight = abs(signed_area) * (1 if i == 0 else -1)
            weighted += weight * centroid
            total_area += weight
    if total_area <= 0:
        raise ValueError("Grid geometry has no positive area")
    return weighted / total_area


def grid_mapping(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    crs = data.get("crs", {}).get("properties", {}).get("name", "")
    if crs and not any(x in crs for x in ("CRS84", "4326")):
        raise ValueError("Grid must contain WGS84 longitude/latitude geometries")
    rows = []
    for feature in data["features"]:
        properties = feature["properties"]
        grid_id = properties["grid_id"]
        if not grid_id.startswith("g500_"):
            raise ValueError("Expected existing 500 metre grid IDs")
        lon, lat = geometry_centroid(feature["geometry"])
        # ERA5 is served on a regular 0.25 degree grid. Nearest-cell assignment
        # deliberately preserves source resolution instead of interpolating it.
        source_lat = math.floor(lat * 4 + 0.5) / 4
        source_lon = math.floor(lon * 4 + 0.5) / 4
        rows.append({
            "grid_id": grid_id, "grid_size_m": 500,
            "centroid_latitude": lat, "centroid_longitude": lon,
            "weather_cell_id": f"era5_{source_lat:.2f}_{source_lon:.2f}",
            "weather_latitude": source_lat, "weather_longitude": source_lon,
            "weather_model": "era5", "source_grid_spacing_degrees": 0.25,
        })
    mapping = pd.DataFrame(rows)
    if mapping.empty or mapping.grid_id.duplicated().any():
        raise ValueError("Grid IDs must be nonempty and unique")
    return mapping


def fetch(params, cache):
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = cache / f"{signature}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["request_parameters"] != params:
            raise ValueError("Cached request does not match parameters")
        return stored["response"]
    for attempt in range(5):
        try:
            with urlopen(API_URL + "?" + urlencode(params), timeout=120) as response:
                data = json.load(response)
            break
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            delay = min(float(error.headers.get("Retry-After", 5 * (attempt + 1))), 30)
            print(f"Provider returned {error.code}; retrying in {delay:g}s", flush=True)
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"request_parameters": params, "response": data}),
                         encoding="utf-8")
    temporary.replace(path)
    time.sleep(0.4)
    return data


def download_cell(cell, start_date, end_date, cache):
    # Cover seven completed local days plus a complete trailing 24h window.
    start = pd.Timestamp(start_date).tz_localize(TIMEZONE).tz_convert("UTC")
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(TIMEZONE).tz_convert("UTC")
    download_start = (start - pd.Timedelta(days=9)).date()
    download_end = end.date()
    frames = []
    for year in range(download_start.year, download_end.year + 1):
        first = max(download_start, datetime(year, 1, 1).date())
        last = min(download_end, datetime(year, 12, 31).date())
        print(f"{cell.weather_cell_id}: {first} to {last}", flush=True)
        params = {
            "latitude": cell.weather_latitude, "longitude": cell.weather_longitude,
            "start_date": first.isoformat(), "end_date": last.isoformat(),
            "hourly": "rain,precipitation", "models": "era5",
            "timezone": "UTC", "timeformat": "unixtime",
            "precipitation_unit": "mm", "cell_selection": "nearest", "elevation": "nan",
        }
        data = fetch(params, cache)
        if not (abs(data["latitude"] - cell.weather_latitude) < 1e-5
                and abs(data["longitude"] - cell.weather_longitude) < 1e-5):
            raise ValueError("Returned weather cell differs from nearest ERA5 cell")
        expected = pd.date_range(str(first), f"{last} 23:00:00", freq="h", tz="UTC")
        actual = pd.DatetimeIndex(pd.to_datetime(data["hourly"]["time"], unit="s", utc=True))
        if not actual.equals(expected):
            raise ValueError("Source timestamps are missing, repeated or out of order")
        frame = pd.DataFrame({
            "datetime": actual, "rainfall_mm": data["hourly"]["rain"],
            "precipitation_mm": data["hourly"]["precipitation"],
            "weather_cell_id": cell.weather_cell_id,
        })
        for variable in ("rain", "precipitation"):
            if data["hourly_units"].get(variable) != "mm":
                raise ValueError("Unexpected rainfall units")
        if not np.isfinite(frame[["rainfall_mm", "precipitation_mm"]]).all().all():
            raise ValueError("Source contains missing precipitation values")
        if (frame[["rainfall_mm", "precipitation_mm"]] < 0).any().any():
            raise ValueError("Source contains negative precipitation")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_csv(frame, path):
    temporary = path.with_name(path.name + ".tmp")
    compression = {"method": "gzip", "compresslevel": 1} if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression, float_format="%.4f")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2009-01-01")
    parser.add_argument("--end-date", default=(datetime.now(ZoneInfo(TIMEZONE)).date()
                                              - timedelta(days=7)).isoformat())
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("start-date must not be after end-date")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data/weather/spatial_era5_raw"
    cache.mkdir(parents=True, exist_ok=True)
    mapping = grid_mapping(ROOT / "data/processed/vancouver_grid.geojson")
    cells = mapping.drop_duplicates("weather_cell_id")
    hourly_frames, daily_frames = [], []
    for cell in cells.itertuples(index=False):
        raw = download_cell(cell, args.start_date, args.end_date, cache)
        hourly, daily = create_features(raw)
        hourly = hourly.loc[hourly.local_date.between(args.start_date, args.end_date)]
        daily = daily.loc[daily.local_date.between(args.start_date, args.end_date)]
        if not hourly.features_ready.all() or not daily.features_ready.all():
            raise ValueError("Missing warm-up history or incomplete requested days")
        hourly_frames.append(hourly)
        daily_frames.append(daily)
    hourly = pd.concat(hourly_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    grid_daily = mapping[["grid_id", "weather_cell_id"]].merge(
        daily, on="weather_cell_id", validate="many_to_many"
    )
    if len(grid_daily) != len(mapping) * daily.local_date.nunique():
        raise ValueError("Grid-day coverage is incomplete")
    save_csv(mapping, output / "grid_weather_mapping.csv")
    save_csv(hourly, output / "weather_hourly_features.csv.gz")
    save_csv(daily, output / "weather_daily_features.csv.gz")
    save_csv(grid_daily, output / "grid_daily_features.csv.gz")
    # A concrete hourly table for all grids, covering the first output week.
    sample_end = (pd.Timestamp(args.start_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    sample = mapping[["grid_id", "weather_cell_id"]].merge(
        hourly.loc[hourly.local_date.le(sample_end)], on="weather_cell_id", validate="many_to_many"
    )
    save_csv(sample, output / "grid_hourly_first_week.csv.gz")
    manifest = {
        "start_local_date": args.start_date, "end_local_date": args.end_date,
        "grid_cells": len(mapping), "weather_cells": len(cells),
        "hourly_source_rows": len(hourly), "daily_source_rows": len(daily),
        "grid_daily_rows": len(grid_daily), "grid_hourly_rows_when_joined": len(hourly) // len(cells) * len(mapping),
        "hourly_first_week_rows": len(sample),
        "model": "era5", "provider": "Open-Meteo", "source_spacing_degrees": 0.25,
        "spatial_assignment": "nearest ERA5 grid point to polygon centroid",
        "rainfall_variable": "rain: liquid precipitation, excluding snow",
        "precipitation_mm": "total precipitation including snow water equivalent",
        "timezone": TIMEZONE, "timestamp_semantics": "UTC end of preceding rainfall hour",
        "hourly_feature_semantics": "current-day quantities use only completed hours through datetime",
        "daily_feature_semantics": "complete-day retrospective summaries; not available before day ends",
        "previous_day_semantics": "completed Vancouver calendar days before local_date; excludes today",
        "rain_duration_hours": "count of completed hours with liquid rain > 0 mm",
        "heavy_rain_thresholds_mm": THRESHOLDS,
        "heavy_rain_semantics": "1 if current-day maximum trailing accumulation meets threshold",
        "features": FEATURES,
        "source_documentation": "https://open-meteo.com/en/docs/historical-weather-api",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
