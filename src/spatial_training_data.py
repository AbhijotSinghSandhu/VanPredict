"""Create 500 m training rows from compact spatial rain and gridded reports.

Features describe a completed Vancouver day; the label describes reports
on the next calendar day. This preserves the full-day rain features without
using a later rainfall observation to predict an earlier same-day report.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services.spatial_rainfall import (  # noqa: E402
    DATASET_DIR, FEATURES, TIMEZONE, load_grid_rainfall_panel,
)

TARGET_CATEGORIES = {"Street Surface Water Flooding Case", "Catch Basin Concern Case"}
TRAINING_FEATURES = ["centroid_latitude", "centroid_longitude", "month",
                     "day_of_year", "day_of_week", *FEATURES]


def attach_next_day_labels(panel, reports, coverage_start, coverage_end):
    """Attach next-day *reported* drainage issues within explicit coverage.

    A zero label means no matching 311 report, not proven absence of flooding.
    Caller must state report coverage so unavailable labels are never filled
    with zero. The command-line builder conservatively uses the observed span.
    """
    if panel.empty or panel.duplicated(["grid_id", "local_date"]).any():
        raise ValueError("Expected nonempty, unique grid-day feature rows")
    if not panel[FEATURES].notna().all().all():
        raise ValueError("Training rainfall features are missing")
    frame = panel.copy()
    frame["target_date"] = (pd.to_datetime(frame.local_date) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    if not frame.target_date.between(coverage_start, coverage_end).all():
        raise ValueError("Target dates extend outside report coverage")
    reports = reports.copy()
    timestamps = pd.to_datetime(reports.service_request_open_timestamp, format="mixed", utc=True, errors="raise")
    reports["target_date"] = timestamps.dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")
    selected = reports.loc[reports.service_request_type.isin(TARGET_CATEGORIES)]
    counts = selected.groupby(["grid_id", "target_date"]).size().rename("next_day_report_count").reset_index()
    frame = frame.merge(counts, on=["grid_id", "target_date"], how="left", validate="one_to_one")
    frame["next_day_report_count"] = frame.next_day_report_count.fillna(0).astype(int)
    frame["is_drainage_problem_next_day"] = frame.next_day_report_count.gt(0).astype("int8")
    dates = pd.to_datetime(frame.target_date)
    frame["month"] = dates.dt.month
    frame["day_of_year"] = dates.dt.dayofyear
    frame["day_of_week"] = dates.dt.dayofweek
    return frame


def build_training_table(start_date, end_date, dataset_dir=DATASET_DIR,
                          reports_path=ROOT / "data/processed/311_gridded.csv"):
    mapping = pd.read_csv(dataset_dir / "grid_weather_mapping.csv")
    reports = pd.read_csv(reports_path)
    if not set(reports.grid_id).issubset(set(mapping.grid_id)):
        raise ValueError("Report grid IDs do not match rainfall mapping")
    report_dates = pd.to_datetime(reports.service_request_open_timestamp, format="mixed", utc=True).dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")
    panel = load_grid_rainfall_panel(start_date, end_date, "daily", dataset_dir=dataset_dir)
    expected_days = len(pd.date_range(start_date, end_date, freq="D"))
    if len(panel) != len(mapping) * expected_days or not panel.features_ready.all():
        raise ValueError("Requested training period has incomplete rainfall coverage")
    panel = panel.merge(mapping[["grid_id", "centroid_latitude", "centroid_longitude"]],
                        on="grid_id", validate="many_to_one")
    return attach_next_day_labels(panel, reports, report_dates.min(), report_dates.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="First completed feature day, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Last completed feature day, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/spatial_training/grid_daily_training.csv.gz")
    args = parser.parse_args()
    table = build_training_table(args.start_date, args.end_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    compression = {"method": "gzip", "compresslevel": 1} if args.output.name.endswith(".gz") else None
    table.to_csv(temporary, index=False, compression=compression, float_format="%.4f")
    temporary.replace(args.output)
    print(f"Saved {len(table):,} grid-day training rows to {args.output}")
    print(f"Positive next-day reported drainage labels: {table.is_drainage_problem_next_day.sum():,}")
    print("Feature days and target days are separate. Split chronologically using target_date.")


if __name__ == "__main__":
    main()
