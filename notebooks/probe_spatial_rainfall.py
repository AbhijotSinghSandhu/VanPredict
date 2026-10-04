import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "spatial_weather"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2025-11-01"
END_DATE = "2025-11-30"
MODEL = "ecmwf_ifs"

# A small geographic probe; these are not rain-gauge stations.
LOCATIONS = [
    {"name": f"{row}_{column}", "latitude": latitude, "longitude": longitude}
    for row, latitude in [("south", 49.21), ("central", 49.25), ("north", 49.29)]
    for column, longitude in [("west", -123.20), ("middle", -123.12), ("east", -123.04)]
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected = pd.date_range(
        START_DATE,
        f"{END_DATE} 23:00:00",
        freq="h",
        tz="UTC",
    )

    series_table = pd.DataFrame({"datetime": expected})
    results = []
    response_metadata = []

    with requests.Session() as session:
        for location in LOCATIONS:
            print(f"Checking {location['name']}...", flush=True)

            params = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "start_date": START_DATE,
                "end_date": END_DATE,
                "hourly": "precipitation",
                "models": MODEL,
                "timezone": "UTC",
                "timeformat": "unixtime",
                "precipitation_unit": "mm",
                "cell_selection": "nearest",
                "elevation": "nan",
            }

            response = session.get(
                API_URL,
                params=params,
                timeout=(30, 120),
            )

            if not response.ok:
                raise RuntimeError(
                    f"{location['name']}: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            data = response.json()
            hourly = data["hourly"]

            timestamps = pd.to_datetime(
                hourly["time"],
                unit="s",
                utc=True,
            )

            rain = pd.to_numeric(
                pd.Series(hourly["precipitation"]),
                errors="coerce",
            ).to_numpy(dtype=float)

            if not timestamps.equals(expected):
                raise ValueError(
                    f"{location['name']}: unexpected hourly timestamps."
                )

            if len(rain) != len(expected):
                raise ValueError("Rainfall and timestamp lengths differ.")

            if not np.isfinite(rain).all() or (rain < 0).any():
                raise ValueError(
                    f"{location['name']}: invalid precipitation."
                )

            if data.get("hourly_units", {}).get("precipitation") != "mm":
                raise ValueError("Unexpected precipitation units.")

            fingerprint = hashlib.sha256(
                np.asarray(rain, dtype="<f8").tobytes()
            ).hexdigest()

            weather_cell = (
                round(float(data["latitude"]), 6),
                round(float(data["longitude"]), 6),
            )

            results.append({
                "location": location["name"],
                "requested_latitude": location["latitude"],
                "requested_longitude": location["longitude"],
                "returned_latitude": weather_cell[0],
                "returned_longitude": weather_cell[1],
                "total_precipitation_mm": round(float(rain.sum()), 2),
                "maximum_hourly_mm": float(rain.max()),
                "series_fingerprint": fingerprint,
            })

            response_metadata.append({
                "location": location["name"],
                "request_parameters": params,
                "response_metadata": {
                    key: value
                    for key, value in data.items()
                    if key != "hourly"
                },
            })

            series_table[location["name"]] = rain
            time.sleep(0.5)

    locations = pd.DataFrame(results)

    summary = {
        "model": MODEL,
        "period": {"start": START_DATE, "end": END_DATE},
        "locations_requested": len(results),
        "distinct_returned_weather_cells": int(
            locations[
                ["returned_latitude", "returned_longitude"]
            ].drop_duplicates().shape[0]
        ),
        "distinct_precipitation_series_in_test_month": int(
            locations["series_fingerprint"].nunique()
        ),
        "locations": results,
    }

    locations.to_csv(
        OUTPUT_DIR / "rainfall_probe_locations.csv",
        index=False,
    )

    series_table.to_csv(
        OUTPUT_DIR / "rainfall_probe_hourly.csv",
        index=False,
    )

    (OUTPUT_DIR / "rainfall_probe_metadata.json").write_text(
        json.dumps(response_metadata, indent=2),
        encoding="utf-8",
    )

    serialized = json.dumps(summary, indent=2, allow_nan=False)

    (OUTPUT_DIR / "rainfall_probe_summary.json").write_text(
        serialized,
        encoding="utf-8",
    )

    print(serialized)


if __name__ == "__main__":
    main()