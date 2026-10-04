from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_311


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TIMEZONE = "America/Vancouver"


def parse_timestamp(value):
    """Preserve timezone offsets; assume Vancouver time if no offset exists."""
    try:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return pd.NaT

        if timestamp.tzinfo is None:
            return timestamp.tz_localize(
                TIMEZONE,
                ambiguous="NaT",
                nonexistent="NaT",
            )

        return timestamp.tz_convert(TIMEZONE)

    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def clean_311():
    df = load_311().copy()
    df.columns = df.columns.str.strip()

    required = [
        "service_request_open_timestamp",
        "latitude",
        "longitude",
        "service_request_type",
    ]

    missing_columns = [column for column in required if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    input_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Retain original values for troubleshooting.
    df["original_open_timestamp"] = df["service_request_open_timestamp"]
    df["original_latitude"] = df["latitude"]
    df["original_longitude"] = df["longitude"]

    parsed = df["service_request_open_timestamp"].map(parse_timestamp)

    # Normalize timezone-aware values into a consistent datetime dtype.
    df["service_request_open_timestamp"] = (
        pd.to_datetime(parsed, utc=True, errors="coerce")
        .dt.tz_convert(TIMEZONE)
    )

    for column in ["latitude", "longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_timestamp = df["service_request_open_timestamp"].isna()

    invalid_coordinates = (
        df[["latitude", "longitude"]].isna().any(axis=1)
        | ~np.isfinite(df["latitude"])
        | ~np.isfinite(df["longitude"])
    )

    # Broad screening bounds; exact city-boundary filtering comes with the grid.
    within_bounds = (
        df["latitude"].between(49.15, 49.35)
        & df["longitude"].between(-123.30, -123.00)
    )

    out_of_bounds = ~invalid_coordinates & ~within_bounds

    # Assign one primary exclusion reason per report.
    df["exclusion_reason"] = np.select(
        [invalid_timestamp, invalid_coordinates, out_of_bounds],
        [
            "invalid_open_timestamp",
            "missing_or_invalid_coordinates",
            "outside_screening_bounds",
        ],
        default="",
    )

    clean = df.loc[df["exclusion_reason"].eq("")].copy()
    excluded = df.loc[df["exclusion_reason"].ne("")].copy()

    clean = clean.drop(columns=["exclusion_reason"])

    timestamp = clean["service_request_open_timestamp"]
    clean["date"] = timestamp.dt.strftime("%Y-%m-%d")
    clean["year"] = timestamp.dt.year
    clean["month"] = timestamp.dt.month
    clean["hour"] = timestamp.dt.hour
    clean["day_of_week"] = timestamp.dt.dayofweek

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = OUTPUT_DIR / "311_clean.csv"
    excluded_path = OUTPUT_DIR / "311_excluded.csv"

    clean.to_csv(clean_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    exclusion_counts = {
        str(reason): int(count)
        for reason, count in excluded["exclusion_reason"].value_counts().items()
    }

    return {
        "input_rows": int(input_rows),
        "duplicates_removed": duplicate_rows,
        "retained_rows": int(len(clean)),
        "excluded_rows": int(len(excluded)),
        "exclusion_reasons": exclusion_counts,
        "validation_counts": {
            "invalid_timestamps": int(invalid_timestamp.sum()),
            "missing_or_invalid_coordinates": int(invalid_coordinates.sum()),
            "outside_screening_bounds": int(out_of_bounds.sum()),
        },
        "date_range": {
            "start": timestamp.min().isoformat() if len(clean) else None,
            "end": timestamp.max().isoformat() if len(clean) else None,
        },
        "files": {
            "clean": str(clean_path),
            "excluded": str(excluded_path),
        },
    }