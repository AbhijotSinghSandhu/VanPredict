"""Generate prototype map assets without changing training grids or reports."""
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.services.build_grid import BOUNDARY_PATH, METRIC_CRS

boundary = gpd.read_file(BOUNDARY_PATH).to_crs(METRIC_CRS)
merged = unary_union(boundary.geometry)
city = merged if merged.geom_type in {'Polygon', 'MultiPolygon'} else unary_union(list(polygonize(merged)))
assert not city.is_empty and city.is_valid
output = ROOT / 'frontend' / 'public' / 'grids'
output.mkdir(parents=True, exist_ok=True)
for size in (200, 500, 1000):
    xmin, ymin, xmax, ymax = city.bounds
    cells = []
    for col in range(math.floor(xmin / size), math.ceil(xmax / size)):
        for row in range(math.floor(ymin / size), math.ceil(ymax / size)):
            clipped = box(col * size, row * size, (col + 1) * size, (row + 1) * size).intersection(city)
            if clipped.is_empty or clipped.area == 0:
                continue
            cells.append({'grid_id': f'g{size}_{col}_{row}', 'grid_size_metres': size, 'area_m2': clipped.area, 'geometry': clipped})
    frame = gpd.GeoDataFrame(cells, crs=METRIC_CRS)
    # Validate coverage before exporting; boundary cells may be smaller than squares.
    assert abs(frame.geometry.area.sum() - city.area) < 1
    assert frame.grid_id.is_unique
    assert frame.area_m2.max() <= size * size + 0.01
    (output / f'vancouver_{size}m.geojson').write_text(frame.to_crs('EPSG:4326').to_json(), encoding='utf-8')
    print(json.dumps({'size_metres': size, 'cells': len(frame), 'coverage_area_m2': round(frame.area_m2.sum())}))
