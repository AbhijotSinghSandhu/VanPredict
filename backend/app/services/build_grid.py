from pathlib import Path
import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import polygonize, unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"

BOUNDARY_PATH = DATA_DIR / "raw" / "city-boundary.geojson"
REPORTS_PATH = OUTPUT_DIR / "311_clean.csv"

GRID_SIZE = 500
METRIC_CRS = "EPSG:32610"  # UTM zone 10N; coordinates in metres.


def build_grid():
    for path in [BOUNDARY_PATH, REPORTS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    boundary = gpd.read_file(BOUNDARY_PATH)

    if boundary.crs is None:
        raise ValueError("Boundary file has no coordinate reference system.")

    boundary = boundary.to_crs(METRIC_CRS)
    geometries = [
        geom for geom in boundary.geometry
        if geom is not None and not geom.is_empty
    ]

    if not geometries:
        raise ValueError("Boundary file has no usable geometry.")

    merged = unary_union(geometries)

    if merged.geom_type in {"Polygon", "MultiPolygon"}:
        city = merged
    else:
        polygons = list(polygonize(merged))

        if not polygons:
            raise ValueError(
                "Boundary lines do not form closed polygons. "
                "Check the downloaded GeoJSON."
            )

        city = unary_union(polygons)

    if not city.is_valid:
        raise ValueError("City boundary polygon is invalid.")

    min_x, min_y, max_x, max_y = city.bounds

    # Align cells to a fixed metre-based origin for stable IDs.
    start_col = math.floor(min_x / GRID_SIZE)
    end_col = math.ceil(max_x / GRID_SIZE)
    start_row = math.floor(min_y / GRID_SIZE)
    end_row = math.ceil(max_y / GRID_SIZE)

    cells = []

    for col in range(start_col, end_col):
        for row in range(start_row, end_row):
            square = box(
                col * GRID_SIZE,
                row * GRID_SIZE,
                (col + 1) * GRID_SIZE,
                (row + 1) * GRID_SIZE,
            )

            clipped = square.intersection(city)

            if clipped.is_empty or clipped.area == 0:
                continue

            cells.append({
              "grid_id": f"g{GRID_SIZE}_{col}_{row}",
                "grid_col": col,
                "grid_row": row,
                "area_m2": clipped.area,
                "geometry": clipped,
            })

    grid = gpd.GeoDataFrame(cells, crs=METRIC_CRS)

    reports = pd.read_csv(REPORTS_PATH)

    points = gpd.GeoDataFrame(
        reports.copy(),
        geometry=gpd.points_from_xy(
            reports["longitude"],
            reports["latitude"],
        ),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    # Intersects includes reports exactly on cell/city boundaries.
    matches = gpd.sjoin(
        points[["geometry"]],
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # A point on a shared edge can match multiple cells.
    # Choose one deterministically to prevent duplicate reports.
    matches["report_row"] = matches.index
    matches = (
        matches.sort_values(["report_row", "grid_id"])
        .drop_duplicates("report_row")
        .set_index("report_row")
    )

    reports["grid_id"] = matches["grid_id"].reindex(reports.index)

    assigned = reports.loc[reports["grid_id"].notna()].copy()
    unassigned = reports.loc[reports["grid_id"].isna()].copy()

    counts = assigned.groupby("grid_id").size()
    grid["historical_report_count"] = (
        grid["grid_id"].map(counts).fillna(0).astype(int)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_path = OUTPUT_DIR / "vancouver_grid.geojson"
    assigned_path = OUTPUT_DIR / "311_gridded.csv"
    unassigned_path = OUTPUT_DIR / "311_outside_grid.csv"

    # Longitude/latitude GeoJSON for web maps.
    grid.to_crs("EPSG:4326").to_file(
        grid_path,
        driver="GeoJSON",
        index=False,
    )

    assigned.to_csv(assigned_path, index=False)
    unassigned.to_csv(unassigned_path, index=False)

    return {
        "grid_size_metres": GRID_SIZE,
        "grid_cells": int(len(grid)),
        "input_reports": int(len(reports)),
        "assigned_reports": int(len(assigned)),
        "unassigned_reports": int(len(unassigned)),
        "cells_with_reports": int(
            (grid["historical_report_count"] > 0).sum()
        ),
        "cells_without_reports": int(
            (grid["historical_report_count"] == 0).sum()
        ),
        "files": {
            "grid": str(grid_path),
            "reports": str(assigned_path),
            "unassigned": str(unassigned_path),
        },
    }