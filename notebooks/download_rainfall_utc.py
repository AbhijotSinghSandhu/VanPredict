from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

LATITUDE = 49.19
LONGITUDE = -123.18

START_YEAR = 2009

# Conservative cutoff for ERA5 availability.
END_DATE = datetime.now(timezone.utc).date() - timedelta(days=7)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# This script lives in VanPredict/notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "weather" / "openmeteo_raw_utc"

OUTPUT_FILE = DATA_DIR / "vancouver_hourly_rainfall_utc.csv"


# ------------------------------------------------------------
# DOWNLOAD ONE YEAR
# ------------------------------------------------------------

def download_year(session, year):
    start_date = datetime(year, 1, 1).date()
    end_date = min(datetime(year, 12, 31).date(), END_DATE)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "precipitation",
        "models": "era5",
        "timezone": "UTC",
        "timeformat": "unixtime",
        "precipitation_unit": "mm",
    }

    print(f"Downloading {start_date} -> {end_date}", flush=True)

    response = session.get(API_URL, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()
    hourly = data.get("hourly", {})

    if "time" not in hourly or "precipitation" not in hourly:
        raise RuntimeError(f"Missing hourly data for {year}")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(
            hourly["time"],
            unit="s",
            utc=True,
            errors="raise",
        ),
        "rainfall_mm": pd.to_numeric(
            pd.Series(hourly["precipitation"]),
            errors="coerce",
        ),
    })

    # Require the complete requested year/partial-year.
    expected = pd.date_range(
        start=start_date.isoformat(),
        end=f"{end_date.isoformat()} 23:00:00",
        freq="h",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(df["datetime"])

    if not actual.equals(expected):
        raise ValueError(
            f"{year}: timestamps are missing, duplicated, "
            "out of order, or outside the requested range."
        )

    rain = df["rainfall_mm"]

    if not np.isfinite(rain).all():
        raise ValueError(f"{year}: missing or nonfinite precipitation.")

    if (rain < 0).any():
        raise ValueError(f"{year}: negative precipitation.")

    # Save each validated year for inspection.
    df.to_csv(RAW_DIR / f"vancouver_{year}_utc.csv", index=False)

    print(
        f"  Validated {len(df):,} hours; "
        f"total precipitation {rain.sum():,.2f} mm",
        flush=True,
    )

    return df


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("VANPREDICT — historical precipitation")
    print(f"Coordinates: {LATITUDE}, {LONGITUDE}")
    print(f"Period: {START_YEAR}-01-01 -> {END_DATE}")
    print("Timezone: UTC")
    print(f"Output: {OUTPUT_FILE}")
    print()

    frames = []
    failed_years = []

    with requests.Session() as session:
        for year in range(START_YEAR, END_DATE.year + 1):
            try:
                frames.append(download_year(session, year))
            except Exception as exc:
                failed_years.append(year)
                print(f"  FAILED {year}: {exc}", flush=True)

            time.sleep(0.5)

    # Never publish an incomplete combined dataset.
    if failed_years:
        raise RuntimeError(
            f"Failed years: {failed_years}. "
            "Final dataset was not saved or replaced."
        )

    weather = (
        pd.concat(frames, ignore_index=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    expected = pd.date_range(
        start=f"{START_YEAR}-01-01",
        end=f"{END_DATE.isoformat()} 23:00:00",
        freq="h",
        tz="UTC",
    )

    if not pd.DatetimeIndex(weather["datetime"]).equals(expected):
        raise ValueError("Combined dataset failed hourly continuity check.")

    rain = weather["rainfall_mm"]

    # Trailing precipitation totals including the current hour.
    # The first rows have partial windows; exclude warm-up rows
    # when assembling the training table.
    for hours in [1, 3, 6, 12, 24, 48]:
        weather[f"rain_{hours}h"] = rain.rolling(
            window=hours,
            min_periods=1,
        ).sum()

    weather["heavy_rain_1h"] = (
        weather["rain_1h"] >= 5
    ).astype(int)

    weather["heavy_rain_6h"] = (
        weather["rain_6h"] >= 15
    ).astype(int)

    weather["heavy_rain_24h"] = (
        weather["rain_24h"] >= 25
    ).astype(int)

    weather["latitude"] = LATITUDE
    weather["longitude"] = LONGITUDE
    weather["weather_source"] = "Open-Meteo ERA5"

    # Write a temporary file first, then replace the final output.
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")
    weather.to_csv(temporary_file, index=False)
    temporary_file.replace(OUTPUT_FILE)

    print()
    print("SUCCESS")
    print(f"Rows: {len(weather):,}")
    print(f"Start: {weather['datetime'].min()}")
    print(f"End: {weather['datetime'].max()}")
    print(f"Missing precipitation: {rain.isna().sum()}")
    print(f"Total precipitation: {rain.sum():,.2f} mm")
    print(f"Maximum hourly precipitation: {rain.max():.2f} mm")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()