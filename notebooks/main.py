import pandas as pd
import requests
from pathlib import Path
import time


# ============================================================
# VANPREDICT
# HOURLY VANCOUVER RAINFALL
# OPEN-METEO ERA5
# ============================================================

LATITUDE = 49.19
LONGITUDE = -123.18

START_YEAR = 2009
END_YEAR = 2026

START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = Path("../data/weather")

RAW_DIR = OUTPUT_DIR / "openmeteo_raw"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "vancouver_hourly_rainfall_2009_2026.csv"
)


# ============================================================
# DOWNLOAD ONE YEAR
# ============================================================

def download_year(year):

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "start_date": start_date,
        "end_date": end_date,

        # IMPORTANT:
        # Hourly precipitation
        "hourly": "precipitation",

        # ERA5
        "models": "era5",

        # Vancouver local time
        "timezone": "America/Vancouver",

        # Millimetres
        "precipitation_unit": "mm",
    }

    print(
        f"      Downloading {start_date} -> {end_date}"
    )

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------
    # Check API response
    # --------------------------------------------------------

    if "hourly" not in data:

        raise RuntimeError(
            f"No hourly data returned for {year}"
        )

    hourly = data["hourly"]

    if "time" not in hourly:

        raise RuntimeError(
            f"No hourly timestamps returned for {year}"
        )

    if "precipitation" not in hourly:

        raise RuntimeError(
            f"No precipitation returned for {year}"
        )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame({
        "datetime": hourly["time"],
        "rainfall_mm": hourly["precipitation"]
    })

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    # --------------------------------------------------------
    # Convert rainfall
    # --------------------------------------------------------

    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Save raw yearly file
    # --------------------------------------------------------

    raw_file = (
        RAW_DIR /
        f"vancouver_{year}.csv"
    )

    df.to_csv(
        raw_file,
        index=False
    )

    return df


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 75)
print("VANPREDICT")
print("VANCOUVER HOURLY RAINFALL")
print("OPEN-METEO ERA5")
print("=" * 75)

print()

print(
    f"Coordinates: "
    f"{LATITUDE}, {LONGITUDE}"
)

print(
    f"Period: "
    f"{START_YEAR} - {END_YEAR}"
)

print(
    "Timezone: America/Vancouver"
)

print(
    "Variable: hourly precipitation"
)

print()


# ============================================================
# DOWNLOAD
# ============================================================

all_data = []

failed_years = []

total_years = (
    END_YEAR -
    START_YEAR +
    1
)


for index, year in enumerate(
    range(
        START_YEAR,
        END_YEAR + 1
    ),
    start=1
):

    print(
        f"[{index:02d}/{total_years:02d}] "
        f"{year}"
    )

    try:

        df = download_year(
            year
        )

        rainfall_count = (
            df["rainfall_mm"]
            .notna()
            .sum()
        )

        non_zero_count = (
            df["rainfall_mm"]
            .fillna(0)
            .gt(0)
            .sum()
        )

        print(
            f"      Rows: "
            f"{len(df):,}"
        )

        print(
            f"      Non-null rainfall: "
            f"{rainfall_count:,}"
        )

        print(
            f"      Non-zero rainfall: "
            f"{non_zero_count:,}"
        )

        all_data.append(
            df
        )

    except Exception as e:

        print(
            "      FAILED"
        )

        print(
            f"      ERROR: {e}"
        )

        failed_years.append(
            year
        )

    # Don't hammer the API
    time.sleep(0.5)


# ============================================================
# DOWNLOAD SUMMARY
# ============================================================

print()
print("=" * 75)
print("DOWNLOAD FINISHED")
print("=" * 75)

print()

print(
    f"Successful years: "
    f"{len(all_data)} / {total_years}"
)

print(
    f"Failed years: "
    f"{len(failed_years)}"
)

if failed_years:

    print()

    print(
        "Failed years:"
    )

    for year in failed_years:

        print(
            f"  - {year}"
        )


# ============================================================
# STOP IF NOTHING DOWNLOADED
# ============================================================

if not all_data:

    raise RuntimeError(
        "No rainfall data was downloaded."
    )


# ============================================================
# COMBINE
# ============================================================

print()

print(
    "Combining yearly datasets..."
)

weather = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

weather = weather.sort_values(
    "datetime"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(weather)

weather = weather.drop_duplicates(
    subset=["datetime"],
    keep="first"
)

after = len(weather)

print(
    f"Duplicate timestamps removed: "
    f"{before - after:,}"
)


# ============================================================
# RESET INDEX
# ============================================================

weather = weather.reset_index(
    drop=True
)


# ============================================================
# RAINFALL CLEANING
# ============================================================

weather["rainfall_mm"] = pd.to_numeric(
    weather["rainfall_mm"],
    errors="coerce"
)

# Negative precipitation should never occur
weather.loc[
    weather["rainfall_mm"] < 0,
    "rainfall_mm"
] = pd.NA


# ============================================================
# RAINFALL FEATURES
# ============================================================

print()

print(
    "Creating rolling rainfall features..."
)

# For rolling calculations:
# missing values are temporarily treated as zero.

rain = (
    weather["rainfall_mm"]
    .fillna(0)
)


# ------------------------------------------------------------
# 1 HOUR
# ------------------------------------------------------------

weather["rain_1h"] = rain


# ------------------------------------------------------------
# 3 HOURS
# ------------------------------------------------------------

weather["rain_3h"] = (
    rain
    .rolling(
        window=3,
        min_periods=1
    )
    .sum()
)


# ------------------------------------------------------------
# 6 HOURS
# ------------------------------------------------------------

weather["rain_6h"] = (
    rain
    .rolling(
        window=6,
        min_periods=1
    )
    .sum()
)


# ------------------------------------------------------------
# 12 HOURS
# ------------------------------------------------------------

weather["rain_12h"] = (
    rain
    .rolling(
        window=12,
        min_periods=1
    )
    .sum()
)


# ------------------------------------------------------------
# 24 HOURS
# ------------------------------------------------------------

weather["rain_24h"] = (
    rain
    .rolling(
        window=24,
        min_periods=1
    )
    .sum()
)


# ------------------------------------------------------------
# 48 HOURS
# ------------------------------------------------------------

weather["rain_48h"] = (
    rain
    .rolling(
        window=48,
        min_periods=1
    )
    .sum()
)


# ============================================================
# HEAVY RAIN FLAGS
# ============================================================

weather["heavy_rain_1h"] = (
    weather["rain_1h"] >= 5
).astype(int)

weather["heavy_rain_6h"] = (
    weather["rain_6h"] >= 15
).astype(int)

weather["heavy_rain_24h"] = (
    weather["rain_24h"] >= 25
).astype(int)


# ============================================================
# ADD LOCATION
# ============================================================

weather["latitude"] = LATITUDE

weather["longitude"] = LONGITUDE

weather["weather_source"] = (
    "Open-Meteo ERA5"
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

weather.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("SUCCESS")
print("=" * 75)

print()

print(
    f"Total hourly records: "
    f"{len(weather):,}"
)

print()

print(
    "Date range:"
)

print(
    f"{weather['datetime'].min()} "
    f"-> "
    f"{weather['datetime'].max()}"
)


# ============================================================
# RAINFALL COVERAGE
# ============================================================

rainfall_records = (
    weather["rainfall_mm"]
    .notna()
    .sum()
)

missing_rainfall = (
    weather["rainfall_mm"]
    .isna()
    .sum()
)

coverage = (
    rainfall_records /
    len(weather)
    * 100
)

print()

print(
    "Rainfall coverage:"
)

print(
    f"  Non-null: "
    f"{rainfall_records:,}"
)

print(
    f"  Missing: "
    f"{missing_rainfall:,}"
)

print(
    f"  Coverage: "
    f"{coverage:.2f}%"
)


# ============================================================
# RAINFALL STATISTICS
# ============================================================

print()

print(
    "Rainfall statistics:"
)

print()

print(
    weather[
        "rainfall_mm"
    ].describe()
)


# ============================================================
# TOTAL RAINFALL
# ============================================================

print()

print(
    f"Total rainfall: "
    f"{weather['rainfall_mm'].sum():.2f} mm"
)

print(
    f"Maximum hourly rainfall: "
    f"{weather['rainfall_mm'].max():.2f} mm"
)


# ============================================================
# HEAVY RAIN EVENTS
# ============================================================

print()

print(
    "Heavy rainfall events:"
)

print(
    f"  >= 5 mm/hour: "
    f"{weather['heavy_rain_1h'].sum():,}"
)

print(
    f"  >= 15 mm/6 hours: "
    f"{weather['heavy_rain_6h'].sum():,}"
)

print(
    f"  >= 25 mm/24 hours: "
    f"{weather['heavy_rain_24h'].sum():,}"
)


# ============================================================
# SAMPLE RAINFALL EVENTS
# ============================================================

print()

print(
    "Sample rainfall events:"
)

rain_events = weather[
    weather["rainfall_mm"] > 0
][
    [
        "datetime",
        "rainfall_mm",
        "rain_3h",
        "rain_6h",
        "rain_12h",
        "rain_24h",
        "rain_48h"
    ]
]

print()

print(
    rain_events
    .head(20)
    .to_string(index=False)
)


# ============================================================
# OUTPUT
# ============================================================

print()

print(
    "Final dataset saved to:"
)

print(
    OUTPUT_FILE.resolve()
)

print()

print("=" * 75)

print(
    "VANPREDICT HOURLY RAINFALL READY"
)

print("=" * 75)