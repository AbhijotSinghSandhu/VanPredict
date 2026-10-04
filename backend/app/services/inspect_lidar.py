import json
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def inspect_lidar():
    tiles_path = DATA_DIR / "raw" / "lidar-2022.geojson"
    grid_path = DATA_DIR / "processed" / "vancouver_grid.geojson"

    for path in [tiles_path, grid_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    tiles = gpd.read_file(tiles_path)
    grid = gpd.read_file(grid_path)

    if tiles.crs is None or grid.crs is None:
        raise ValueError("Tile index or grid has no coordinate system.")

    if tiles.empty or grid.empty:
        raise ValueError("Tile index or grid is empty.")

    if (
        tiles.geometry.isna().any()
        or tiles.geometry.is_empty.any()
        or (~tiles.geometry.is_valid).any()
    ):
        raise ValueError("Tile index contains missing or invalid geometry.")

    if not tiles.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all():
        raise ValueError("Expected tile footprints as polygons.")

    # Use metres to calculate overlap areas.
    tiles = tiles.to_crs("EPSG:32610")
    grid = grid.to_crs("EPSG:32610")

    study_area = unary_union(grid.geometry.tolist())

    overlap_area = tiles.geometry.intersection(study_area).area

    selected = tiles.loc[overlap_area > 0].copy()
    selected["overlap_km2"] = (
        overlap_area.loc[selected.index] / 1_000_000
    ).round(4)

    attributes = selected.drop(columns=[selected.geometry.name])

    return {
        "total_tiles": int(len(tiles)),
        "tiles_overlapping_grid": int(len(selected)),
        "attribute_columns": attributes.columns.tolist(),
        "selected_tile_samples": json.loads(
            attributes.head(5).to_json(orient="records")
        ),
        "note": (
            "These are candidate tiles overlapping the current grid, "
            "including its water areas. No LAS files downloaded yet."
        ),
    }