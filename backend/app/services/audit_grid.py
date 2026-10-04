from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def audit_grid():
    grid = gpd.read_file(
        PROCESSED_DIR / "vancouver_grid.geojson"
    ).to_crs("EPSG:32610")

    reports = pd.read_csv(PROCESSED_DIR / "311_gridded.csv")
    unassigned = pd.read_csv(
        PROCESSED_DIR / "311_outside_grid.csv"
    )

    if grid.empty or reports.empty:
        raise ValueError("Grid or assigned reports are empty.")

    timestamps = pd.to_datetime(
        reports["service_request_open_timestamp"],
        format="mixed",
        utc=True,
        errors="coerce",
    ).dt.tz_convert("America/Vancouver")

    if timestamps.isna().any():
        raise ValueError("Gridded reports contain invalid timestamps.")

    reports["report_date"] = timestamps.dt.date
    reports["report_year"] = timestamps.dt.year

    # Include every grid cell when calculating count statistics.
    counts = reports.groupby("grid_id").size()
    cell_counts = grid["grid_id"].map(counts).fillna(0).astype(int)

    count_percentiles = {
        str(percentile): float(value)
        for percentile, value in cell_counts.quantile(
            [0, 0.25, 0.5, 0.75, 0.9, 1]
        ).items()
    }

    # Daily sparsity, including cell-days without reports.
    first_date = reports["report_date"].min()
    last_date = reports["report_date"].max()
    calendar_days = (last_date - first_date).days + 1

    positive_cell_days = len(
        reports[["grid_id", "report_date"]].drop_duplicates()
    )
    total_cell_days = len(grid) * calendar_days

    yearly_counts = reports.groupby("report_year").size()
    category_counts = reports["service_request_type"].value_counts()

    # Check cell overlaps and geometry areas.
    summed_area = float(grid.geometry.area.sum())
    union_area = float(unary_union(grid.geometry.tolist()).area)
    areas = grid.geometry.area

    # Verify that saved report counts agree with the reports file.
    saved_counts_match = (
        grid["historical_report_count"].astype(int).eq(cell_counts).all()
        if "historical_report_count" in grid.columns
        else None
    )

    preview_columns = [
        "latitude",
        "longitude",
        "service_request_type",
        "service_request_open_timestamp",
    ]

    # Convert missing values to JSON-compatible nulls.
    unassigned_preview = (
        unassigned[preview_columns]
        .head(20)
        .astype(object)
        .where(unassigned[preview_columns].head(20).notna(), None)
        .to_dict(orient="records")
    )

    return {
        "geometry_checks": {
            "grid_cells": int(len(grid)),
            "duplicate_grid_ids": int(grid["grid_id"].duplicated().sum()),
            "invalid_geometries": int((~grid.geometry.is_valid).sum()),
            "empty_geometries": int(grid.geometry.is_empty.sum()),
            "summed_cell_area_km2": round(summed_area / 1_000_000, 3),
            "overlap_area_m2": round(
                max(0.0, summed_area - union_area), 3
            ),
            "smallest_cell_area_m2": round(float(areas.min()), 2),
            "largest_cell_area_m2": round(float(areas.max()), 2),
        },
        "assignment_checks": {
            "assigned_reports": int(len(reports)),
            "unassigned_reports": int(len(unassigned)),
            "reports_with_unknown_grid_id": int(
                (~reports["grid_id"].isin(grid["grid_id"])).sum()
            ),
            "saved_cell_counts_match": (
                bool(saved_counts_match)
                if saved_counts_match is not None else None
            ),
        },
        "historical_reports_per_cell": {
            "percentiles": count_percentiles,
            "cells_with_zero_reports": int((cell_counts == 0).sum()),
            "cells_with_fewer_than_10_reports": int(
                (cell_counts < 10).sum()
            ),
        },
        "daily_target_sparsity": {
            "first_date": str(first_date),
            "last_date": str(last_date),
            "calendar_days": int(calendar_days),
            "total_cell_days": int(total_cell_days),
            "cell_days_with_reports": int(positive_cell_days),
            "positive_cell_day_percent": round(
                100 * positive_cell_days / total_cell_days, 4
            ),
            "note": (
                "Assumes complete report coverage across this date span. "
                "Calendar-day counts are a diagnostic, not yet the "
                "rolling next-24-hour model target."
            ),
        },
        "reports_by_year": {
            str(year): int(count)
            for year, count in yearly_counts.items()
        },
        "reports_by_category": {
            str(category): int(count)
            for category, count in category_counts.items()
        },
        "unassigned_report_preview": unassigned_preview,
    }