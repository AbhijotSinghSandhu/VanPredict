"""Publish compact, per-grid inventory points for the selected-area overlay."""
import json
from pathlib import Path
import geopandas as gpd

ROOT=Path(__file__).resolve().parents[2]
raw=ROOT/'data/raw/sewer_catch_basins_snapshot.geojson'
points=gpd.read_file(raw).reset_index(drop=True)
points['record']=points.index
grid=gpd.read_file(ROOT/'frontend/public/grids/vancouver_500m.geojson')[['grid_id','geometry']].to_crs(points.crs)
joined=gpd.sjoin(points,grid,predicate='intersects',how='inner').sort_values(['record','grid_id']).drop_duplicates('record')
cells={str(key):[[round(point.x,6),round(point.y,6)] for point in frame.geometry] for key,frame in joined.groupby('grid_id')}
metadata=json.loads(raw.with_suffix('.metadata.json').read_text())
payload={'cells':cells,'source_modified_at':metadata['source_modified_at'],'source_url':metadata['source_url'],'license':metadata['license'],'locations_approximate':True}
target=ROOT/'frontend/public/infrastructure/catch-basins.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(payload,separators=(',',':')))
print(f'{len(joined):,} records across {len(cells)} grids; {target.stat().st_size:,} bytes')
