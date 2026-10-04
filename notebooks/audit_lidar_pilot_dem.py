import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "elevation"
GRID_PATH = PROJECT_ROOT / "data" / "processed" / "vancouver_grid.geojson"


def statistics(values, prefix):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    result = {
        f"{prefix}_mean": None,
        f"{prefix}_min": None,
        f"{prefix}_max": None,
        f"{prefix}_p10": None,
        f"{prefix}_median": None,
        f"{prefix}_p90": None,
    }

    if not values.size:
        return result

    result.update({
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_median": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    })

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[2, 5, 10],
        default=5,
    )
    args = parser.parse_args()

    label = str(args.resolution)

    dem_path = OUTPUT_DIR / f"pilot_ground_elevation_{label}m.tif"
    preview_path = OUTPUT_DIR / f"pilot_terrain_audit_{label}m.png"
    csv_path = OUTPUT_DIR / f"pilot_grid_terrain_audit_{label}m.csv"
    summary_path = OUTPUT_DIR / f"pilot_terrain_audit_{label}m.json"

    for path in [dem_path, GRID_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as source:
        elevation = source.read(1, masked=True).astype("float64")
        z = elevation.filled(np.nan)

        transform = source.transform
        crs = source.crs
        bounds = source.bounds

        pixel_x = abs(transform.a)
        pixel_y = abs(transform.e)

        if crs is None or crs.to_epsg() != 26910:
            raise ValueError("Expected terrain CRS EPSG:26910.")

        if (
            transform.a <= 0
            or transform.e >= 0
            or transform.b != 0
            or transform.d != 0
        ):
            raise ValueError("Expected an unrotated, north-up raster.")

        if not (
            np.isclose(pixel_x, args.resolution)
            and np.isclose(pixel_y, args.resolution)
        ):
            raise ValueError(
                f"Requested {args.resolution} m, but raster pixels "
                f"are {pixel_x} × {pixel_y} m."
            )

        if min(z.shape) < 3:
            raise ValueError("Raster is too small for slope calculation.")

        if not np.isfinite(z).any():
            raise ValueError("Raster contains no valid elevations.")

    # Central differences require the centre and four neighbours.
    # Missing neighbours and outside edges remain missing.
    centre = z[1:-1, 1:-1]
    west = z[1:-1, :-2]
    east = z[1:-1, 2:]
    north = z[:-2, 1:-1]
    south = z[2:, 1:-1]

    usable = (
        np.isfinite(centre)
        & np.isfinite(west)
        & np.isfinite(east)
        & np.isfinite(north)
        & np.isfinite(south)
    )

    dz_dx = (east - west) / (2 * pixel_x)
    dz_dy = (south - north) / (2 * pixel_y)

    interior_slope = np.degrees(
        np.arctan(np.hypot(dz_dx, dz_dy))
    )

    slope = np.full(z.shape, np.nan)
    slope[1:-1, 1:-1] = np.where(
        usable,
        interior_slope,
        np.nan,
    )

    # --------------------------------------------------------
    # VISUAL CHECK
    # --------------------------------------------------------

    extent = [
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ]

    terrain_cmap = plt.get_cmap("terrain").copy()
    terrain_cmap.set_bad("white")

    slope_cmap = plt.get_cmap("viridis").copy()
    slope_cmap.set_bad("white")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"LiDAR pilot audit — {label} m pixels")

    terrain_image = axes[0].imshow(
        np.ma.masked_invalid(z),
        extent=extent,
        origin="upper",
        cmap=terrain_cmap,
        interpolation="nearest",
    )
    axes[0].set_title("Ground elevation")
    fig.colorbar(
        terrain_image,
        ax=axes[0],
        label="Elevation (m)",
    )

    axes[1].imshow(
        ~np.isfinite(z),
        extent=extent,
        origin="upper",
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("Missing ground pixels: black")

    slope_image = axes[2].imshow(
        np.ma.masked_invalid(slope),
        extent=extent,
        origin="upper",
        cmap=slope_cmap,
        vmin=0,
        vmax=45,
        interpolation="nearest",
    )
    axes[2].set_title("Slope — missing values unfilled")
    fig.colorbar(
        slope_image,
        ax=axes[2],
        label="Degrees; colours capped at 45°",
    )

    for axis in axes:
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
        axis.ticklabel_format(style="plain", useOffset=False)

    fig.tight_layout()
    fig.savefig(preview_path, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # GRID STATISTICS
    # --------------------------------------------------------

    grid = gpd.read_file(GRID_PATH)

    if grid.crs is None:
        raise ValueError("Grid has no CRS.")

    if grid["grid_id"].duplicated().any():
        raise ValueError("Grid contains duplicate IDs.")

    if (
        grid.geometry.isna().any()
        or grid.geometry.is_empty.any()
        or (~grid.geometry.is_valid).any()
    ):
        raise ValueError("Grid contains missing, empty, or invalid geometry.")

    grid = grid.to_crs(crs)

    footprint = box(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
    )

    rows = []

    for _, cell in grid.iterrows():
        overlap = cell.geometry.intersection(footprint)

        if overlap.is_empty or overlap.area == 0:
            continue

        window = from_bounds(
            *overlap.bounds,
            transform=transform,
        )

        col_start = max(0, int(np.floor(window.col_off)))
        row_start = max(0, int(np.floor(window.row_off)))

        col_end = min(
            z.shape[1],
            int(np.ceil(window.col_off + window.width)),
        )
        row_end = min(
            z.shape[0],
            int(np.ceil(window.row_off + window.height)),
        )

        if col_end <= col_start or row_end <= row_start:
            continue

        cropped_transform = rasterio.windows.transform(
            Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start,
            ),
            transform,
        )

        cropped_z = z[row_start:row_end, col_start:col_end]
        cropped_slope = slope[row_start:row_end, col_start:col_end]

        inside = geometry_mask(
            [overlap.__geo_interface__],
            out_shape=cropped_z.shape,
            transform=cropped_transform,
            invert=True,
            all_touched=False,
        )

        sampled = int(inside.sum())

        if sampled == 0:
            continue

        valid_ground = inside & np.isfinite(cropped_z)
        valid_slope = inside & np.isfinite(cropped_slope)

        tile_coverage = overlap.area / cell.geometry.area

        row = {
            "grid_id": str(cell["grid_id"]),
            "terrain_resolution_metres": args.resolution,
            "pilot_area_coverage_percent": round(
                100 * tile_coverage,
                3,
            ),
            "sampled_pixels": sampled,
            "valid_ground_pixels": int(valid_ground.sum()),
            "valid_slope_pixels": int(valid_slope.sum()),
            "ground_coverage_of_sampled_pixels_percent": round(
                100 * float(valid_ground.sum()) / sampled,
                3,
            ),
            "slope_coverage_of_sampled_pixels_percent": round(
                100 * float(valid_slope.sum()) / sampled,
                3,
            ),
            "coverage_status": (
                "full_cell_footprint"
                if tile_coverage >= 0.999999
                else "partial_cell_footprint"
            ),
        }

        row.update(statistics(cropped_z[inside], "elevation_m"))
        row.update(statistics(cropped_slope[inside], "slope_deg"))

        row["elevation_range_m"] = (
            row["elevation_m_max"] - row["elevation_m_min"]
            if row["elevation_m_min"] is not None
            else None
        )

        rows.append(row)

    if not rows:
        raise ValueError("No grid cells contain sampled raster pixels.")

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    slope_values = slope[np.isfinite(slope)]

    summary = {
        "terrain_resolution_metres": args.resolution,
        "crs": crs.to_string(),
        "width_pixels": int(z.shape[1]),
        "height_pixels": int(z.shape[0]),
        "sampled_grid_cells": len(rows),
        "valid_ground_percent": round(
            100 * float(np.isfinite(z).mean()),
            2,
        ),
        "valid_slope_percent": round(
            100 * float(np.isfinite(slope).mean()),
            2,
        ),
        "elevation_summary_m": statistics(z, "elevation"),
        "slope_percentiles_degrees": (
            {
                str(percentile): round(
                    float(np.percentile(slope_values, percentile)),
                    3,
                )
                for percentile in [50, 90, 95, 99, 100]
            }
            if slope_values.size
            else {}
        ),
        "files": {
            "preview": str(preview_path),
            "grid_statistics": str(csv_path),
            "summary": str(summary_path),
        },
        "cell_statistics": rows,
    }

    serialized = json.dumps(summary, indent=2, allow_nan=False)
    summary_path.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()