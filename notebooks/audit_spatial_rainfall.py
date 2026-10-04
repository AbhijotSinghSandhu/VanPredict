"""Validate saved source features, grid mapping and expanded daily coverage."""

import json
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services.spatial_rainfall import (  # noqa: E402
    DATASET_DIR, FEATURES, THRESHOLDS, TIMEZONE, load_grid_rainfall_panel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="Validate the GitHub source tables without optional expanded exports")
    args = parser.parse_args()
    checksum_path = DATASET_DIR / "checksums.json"
    if checksum_path.exists():
        checksums = json.loads(checksum_path.read_text())
        for name, expected in checksums.items():
            blob = (DATASET_DIR / name).read_bytes()
            assert len(blob) == expected["bytes"], f"Unexpected file length: {name}"
            assert hashlib.sha256(blob).hexdigest() == expected["sha256"], f"Checksum mismatch: {name}"
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    mapping = pd.read_csv(DATASET_DIR / "grid_weather_mapping.csv")
    hourly = pd.read_csv(DATASET_DIR / "weather_hourly_features.csv.gz")
    daily = pd.read_csv(DATASET_DIR / "weather_daily_features.csv.gz")
    assert len(mapping) == manifest["grid_cells"]
    assert not mapping.grid_id.duplicated().any()
    assert not hourly.duplicated(["weather_cell_id", "datetime"]).any()
    assert not daily.duplicated(["weather_cell_id", "local_date"]).any()
    assert np.isfinite(hourly[FEATURES + ["rainfall_mm", "precipitation_mm"]]).all().all()
    assert (hourly[FEATURES] >= 0).all().all()
    assert hourly.features_ready.all() and daily.features_ready.all()
    assert set(hourly.weather_cell_id) == set(mapping.weather_cell_id)
    source_checks = []
    for cell_id, cell in hourly.groupby("weather_cell_id", sort=False):
        timestamps = pd.DatetimeIndex(pd.to_datetime(cell.datetime, utc=True))
        assert timestamps.equals(pd.date_range(timestamps[0], timestamps[-1], freq="h"))
        local_dates = (timestamps - pd.Timedelta(nanoseconds=1)).tz_convert(TIMEZONE).strftime("%Y-%m-%d")
        assert (local_dates == cell.local_date.to_numpy()).all()
        expected_today = cell.groupby("local_date").rainfall_mm.cumsum()
        np.testing.assert_allclose(cell.rainfall_today, expected_today, atol=0.0001)
        expected_duration = cell.rainfall_mm.gt(0).groupby(cell.local_date).cumsum()
        np.testing.assert_array_equal(cell.rain_duration_hours, expected_duration)
        for hours in (1, 6, 24):
            expected = cell.rainfall_mm.rolling(hours, min_periods=hours).sum()
            # First 23 rows need downloaded warm-up data; later rows are self-contained.
            np.testing.assert_allclose(cell[f"rain_{hours}h"].iloc[24:], expected.iloc[24:], atol=0.0001)
            np.testing.assert_array_equal(cell[f"heavy_rain_{hours}h"],
                                          cell[f"max_rain_{hours}h"].ge(THRESHOLDS[hours]).astype(int))
        cell_daily = daily.loc[daily.weather_cell_id.eq(cell_id)].set_index("local_date")
        np.testing.assert_allclose(cell.groupby("local_date").rainfall_mm.sum(), cell_daily.rainfall_today, atol=0.0001)
        for days, name in ((1, "rainfall_previous_day"), (3, "rainfall_previous_3d"), (7, "rainfall_previous_7d")):
            expected = cell_daily.rainfall_today.shift(1).rolling(days, min_periods=days).sum()
            np.testing.assert_allclose(cell_daily[name].iloc[days:], expected.iloc[days:], atol=0.0001)
        source_checks.append({"weather_cell_id": cell_id, "hours": len(cell),
                              "days": len(cell_daily), "grids_assigned": int(mapping.weather_cell_id.eq(cell_id).sum()),
                              "liquid_rain_total_mm": round(float(cell.rainfall_mm.sum()), 1)})
    grid_counts = pd.Series(0, index=mapping.grid_id, dtype="int64")
    expanded_rows = 0
    expanded_path = DATASET_DIR / "grid_daily_features.csv.gz"
    chunks = (pd.read_csv(expanded_path, chunksize=100_000)
              if expanded_path.exists() and not args.compact else [])
    for chunk in chunks:
        expanded_rows += len(chunk)
        assert set(chunk.grid_id).issubset(set(mapping.grid_id))
        counts = chunk.groupby("grid_id").size()
        grid_counts = grid_counts.add(counts, fill_value=0)
        expected = chunk[["grid_id", "local_date"]].merge(
            mapping[["grid_id", "weather_cell_id"]], on="grid_id", validate="many_to_one"
        ).merge(daily, on=["weather_cell_id", "local_date"], validate="many_to_one")
        assert len(expected) == len(chunk)
        np.testing.assert_allclose(chunk[FEATURES].to_numpy(), expected[FEATURES].to_numpy(), atol=0.0001)
    if expanded_rows:
        assert expanded_rows == manifest["grid_daily_rows"]
        assert grid_counts.eq(daily.local_date.nunique()).all()
    assert len(mapping) * daily.local_date.nunique() == manifest["grid_daily_rows"]
    # Exercise the actual grid loader against the materialized hourly sample.
    first = manifest["start_local_date"]
    last = min(manifest["end_local_date"], (pd.Timestamp(first) + pd.Timedelta(days=6)).strftime("%Y-%m-%d"))
    panel = load_grid_rainfall_panel(first, last)
    sample_path = DATASET_DIR / "grid_hourly_first_week.csv.gz"
    sample = pd.read_csv(sample_path) if sample_path.exists() and not args.compact else None
    assert len(panel) == manifest["hourly_first_week_rows"]
    if sample is not None:
        assert len(panel) == len(sample)
        np.testing.assert_allclose(panel[FEATURES], sample[FEATURES], atol=0.0001)
    assert not panel.duplicated(["grid_id", "datetime"]).any()
    report = {"status": "passed", "grid_cells": len(mapping),
              "source_cells": source_checks, "daily_grid_rows_checked": expanded_rows,
              "hourly_grid_sample_rows_checked": len(panel),
              "expanded_exports_checked": bool(expanded_rows),
              "checks": ["published dataset checksums", "hourly continuity and unique keys", "Vancouver interval dates",
                         "as-of cumulative rain and duration", "trailing rainfall windows",
                         "heavy-rain thresholds", "daily totals and antecedent windows",
                         "daily coverage implied by the source tables and grid mapping",
                         "hourly grid loader and unique grid-hour keys"]}
    report_name = "quality_report_compact.json" if args.compact else "quality_report.json"
    (DATASET_DIR / report_name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
