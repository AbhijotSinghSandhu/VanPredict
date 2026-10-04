from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_rainfall


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_PATH = PROJECT_ROOT / "data" / "processed" / "311_gridded.csv"


def audit_rainfall(naive_timezone=None):
    df = load_rainfall().copy()
    df.columns = df.columns.str.strip()

    required = ["datetime", "rainfall_mm"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing rainfall columns: {missing}")

    raw_time = df["datetime"].astype("string").str.strip()
    rain = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # Detect explicit timezone offsets, e.g. Z, +00:00, -07:00.
    has_offset = raw_time.str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})$",
        case=False,
        regex=True,
        na=False,
    )

    nonempty = raw_time.notna() & raw_time.ne("")
    naive_rows = nonempty & ~has_offset

    result = {
        "rows": int(len(df)),
        "timestamp_samples": raw_time.head(5).fillna("").tolist(),
        "rows_with_explicit_timezone": int(has_offset.sum()),
        "rows_without_explicit_timezone": int(naive_rows.sum()),
        "missing_or_nonfinite_rainfall": int(
            (~np.isfinite(rain)).sum()
        ),
        "negative_rainfall_rows": int((rain < 0).sum()),
    }

    # Do not guess the timezone of timestamps without an offset.
    if naive_rows.any() and naive_timezone is None:
        result["status"] = "timezone_confirmation_required"
        result["next_action"] = (
            "Check the rainfall download script's timezone parameter. "
            "Then rerun with naive_timezone=UTC or "
            "naive_timezone=America/Vancouver, as appropriate."
        )
        return result

    timestamps = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )

    timestamps.loc[has_offset] = pd.to_datetime(
        raw_time.loc[has_offset],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    if naive_rows.any():
        parsed = pd.to_datetime(
            raw_time.loc[naive_rows],
            format="mixed",
            errors="coerce",
        )

        timestamps.loc[naive_rows] = (
            parsed.dt.tz_localize(
                naive_timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            ).dt.tz_convert("UTC")
        )

    df["timestamp_utc"] = timestamps
    df["rainfall_mm"] = rain

    result["naive_timezone_used"] = naive_timezone
    result["invalid_or_unresolved_timestamps"] = int(
        timestamps.isna().sum()
    )

    valid = df.loc[timestamps.notna()].copy()
    valid = valid.sort_values("timestamp_utc")

    if valid.empty:
        result["status"] = "no_valid_timestamps"
        return result

    duplicate_count = int(
        valid["timestamp_utc"].duplicated().sum()
    )

    off_hour_count = int(
        valid["timestamp_utc"]
        .ne(valid["timestamp_utc"].dt.floor("h"))
        .sum()
    )

    result["duplicate_timestamp_rows"] = duplicate_count
    result["off_hour_timestamp_rows"] = off_hour_count

    # Unique timestamps for continuity diagnostics only.
    unique = valid.drop_duplicates("timestamp_utc").set_index("timestamp_utc")

    expected_hours = pd.date_range(
        unique.index.min().ceil("h"),
        unique.index.max().floor("h"),
        freq="h",
    )

    missing_hours = expected_hours.difference(unique.index)

    result["continuity"] = {
        "start_utc": unique.index.min().isoformat(),
        "end_utc": unique.index.max().isoformat(),
        "expected_hourly_slots": int(len(expected_hours)),
        "missing_hours": int(len(missing_hours)),
        "first_missing_hours": [
            timestamp.isoformat() for timestamp in missing_hours[:10]
        ],
    }

    values = unique["rainfall_mm"]
    finite_values = values.loc[np.isfinite(values)]

    result["rainfall_statistics"] = {
        "total_mm": (
            round(float(finite_values.sum()), 2)
            if len(finite_values) else None
        ),
        "maximum_hourly_mm": (
            float(finite_values.max())
            if len(finite_values) else None
        ),
    }

    # Recompute rolling totals only when the hourly series is usable.
    can_check_rolling = (
        duplicate_count == 0
        and off_hour_count == 0
        and len(missing_hours) == 0
        and np.isfinite(values).all()
        and (values >= 0).all()
    )

    rolling_checks = {}

    if can_check_rolling:
        for hours in [1, 3, 6, 12, 24, 48]:
            column = f"rain_{hours}h"

            if column not in unique.columns:
                rolling_checks[column] = {"status": "column_missing"}
                continue

            # Assumes each total includes the current row.
            expected = values.rolling(
                window=hours,
                min_periods=1,
            ).sum()

            saved = pd.to_numeric(unique[column], errors="coerce")
            matches = np.isclose(
                saved,
                expected,
                atol=0.011,
                rtol=0,
                equal_nan=False,
            )

            differences = (saved - expected).abs()
            finite_differences = differences.loc[
                np.isfinite(differences)
            ]

            rolling_checks[column] = {
                "mismatched_rows": int((~matches).sum()),
                "maximum_difference_mm": (
                    round(float(finite_differences.max()), 4)
                    if len(finite_differences) else None
                ),
            }

    result["rolling_totals"] = {
        "checked": bool(can_check_rolling),
        "convention": "Trailing rows including the current hour",
        "checks": rolling_checks,
    }

    reports = pd.read_csv(REPORTS_PATH)
    report_times = pd.to_datetime(
        reports["service_request_open_timestamp"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    # Compare report-hour labels with actual rainfall timestamps.
    report_hours = report_times.dt.floor("h")
    matched = report_hours.isin(unique.index)

    result["report_hour_coverage"] = {
        "reports": int(len(reports)),
        "reports_with_matching_rainfall_hour": int(matched.sum()),
        "reports_without_matching_rainfall_hour": int((~matched).sum()),
    }

    result["status"] = "audit_complete"
    return result