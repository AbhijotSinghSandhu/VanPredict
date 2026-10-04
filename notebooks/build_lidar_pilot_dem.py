import argparse
import json
import math
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "486000_5455000"
    / "486000_5455000.las"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "elevation"
NODATA = -9999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[2, 5, 10],
        default=5,
    )
    args = parser.parse_args()

    resolution = float(args.resolution)
    label = str(args.resolution)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dem_path = OUTPUT_DIR / f"pilot_ground_elevation_{label}m.tif"
    counts_path = OUTPUT_DIR / f"pilot_ground_counts_{label}m.tif"

    if not LAS_PATH.exists():
        raise FileNotFoundError(f"LAS file not found: {LAS_PATH}")

    with laspy.open(LAS_PATH) as reader:
        header = reader.header
        crs = header.parse_crs()

        if crs is None or crs.to_epsg() != 26910:
            raise ValueError("Expected pilot CRS EPSG:26910.")

        left = math.floor(header.mins[0] / resolution) * resolution
        bottom = math.floor(header.mins[1] / resolution) * resolution
        right = math.ceil(header.maxs[0] / resolution) * resolution
        top = math.ceil(header.maxs[1] / resolution) * resolution

        width = int(round((right - left) / resolution))
        height = int(round((top - bottom) / resolution))

        if width <= 0 or height <= 0:
            raise ValueError("Invalid raster dimensions.")

        sums = np.zeros(width * height, dtype=np.float64)
        counts = np.zeros(width * height, dtype=np.uint64)
        used_ground_points = 0

        print(f"Building {label} m terrain raster...", flush=True)

        for points in reader.chunk_iterator(1_000_000):
            ground = (
                (np.asarray(points.classification) == 2)
                & (np.asarray(points.withheld) == 0)
            )

            if not ground.any():
                continue

            x = np.asarray(points.x)[ground]
            y = np.asarray(points.y)[ground]
            z = np.asarray(points.z)[ground]

            # Check coordinates before converting them to pixel indices.
            if not (
                np.isfinite(x).all()
                and np.isfinite(y).all()
                and np.isfinite(z).all()
            ):
                raise ValueError("Ground points contain nonfinite coordinates.")

            tolerance = 0.001

            if (
                (x < left - tolerance).any()
                or (x > right + tolerance).any()
                or (y < bottom - tolerance).any()
                or (y > top + tolerance).any()
            ):
                raise ValueError("Ground points fall outside raster bounds.")

            columns = np.floor((x - left) / resolution).astype(int)
            rows = np.floor((top - y) / resolution).astype(int)

            # Handle points exactly on the outside boundary.
            columns = np.clip(columns, 0, width - 1)
            rows = np.clip(rows, 0, height - 1)

            indices = rows * width + columns

            sums += np.bincount(
                indices,
                weights=z,
                minlength=width * height,
            )

            counts += np.bincount(
                indices,
                minlength=width * height,
            ).astype(np.uint64)

            used_ground_points += len(z)

    if used_ground_points == 0:
        raise ValueError("No usable ground points found.")

    valid = counts > 0

    elevation = np.full(
        width * height,
        NODATA,
        dtype=np.float32,
    )
    elevation[valid] = (sums[valid] / counts[valid]).astype(np.float32)

    elevation = elevation.reshape(height, width)
    point_counts = counts.reshape(height, width)

    if point_counts.max() > np.iinfo(np.uint32).max:
        raise ValueError("Pixel point count exceeds uint32 capacity.")

    transform = from_origin(left, top, resolution, resolution)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(
        dem_path,
        "w",
        **profile,
        dtype="float32",
        nodata=NODATA,
    ) as destination:
        destination.write(elevation, 1)

    with rasterio.open(
        counts_path,
        "w",
        **profile,
        dtype="uint32",
    ) as destination:
        destination.write(point_counts.astype(np.uint32), 1)

    # Verify only after both writers have closed.
    with rasterio.open(dem_path) as saved:
        if saved.crs != crs:
            raise ValueError("Saved DEM CRS is incorrect.")

        if saved.transform != transform:
            raise ValueError("Saved DEM transform is incorrect.")

        if saved.shape != (height, width):
            raise ValueError("Saved DEM dimensions are incorrect.")

        if saved.nodata != NODATA:
            raise ValueError("Saved DEM nodata value is incorrect.")

        if not np.array_equal(saved.read(1), elevation):
            raise ValueError("Saved DEM values differ from calculated values.")

    with rasterio.open(counts_path) as saved:
        if (
            saved.crs != crs
            or saved.transform != transform
            or saved.shape != (height, width)
        ):
            raise ValueError("Saved point-count raster metadata is incorrect.")

        saved_counts = saved.read(1)

        if not np.array_equal(saved_counts, point_counts):
            raise ValueError("Saved point counts differ from calculated counts.")

        if int(saved_counts.sum(dtype=np.uint64)) != used_ground_points:
            raise ValueError("Ground-point count conservation failed.")

    print("Saved raster verification passed.", flush=True)

    valid_elevations = elevation[elevation != NODATA]

    print(json.dumps({
        "resolution_metres": resolution,
        "crs": crs.to_string(),
        "width_pixels": width,
        "height_pixels": height,
        "ground_points_used": int(used_ground_points),
        "valid_pixels": int(valid.sum()),
        "missing_pixels": int((~valid).sum()),
        "valid_pixel_percent": round(100 * float(valid.mean()), 2),
        "minimum_pixel_elevation_m": float(valid_elevations.min()),
        "maximum_pixel_elevation_m": float(valid_elevations.max()),
        "files": {
            "ground_elevation": str(dem_path),
            "ground_point_counts": str(counts_path),
        },
    }, indent=2))


if __name__ == "__main__":
    main()