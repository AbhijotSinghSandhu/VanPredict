"""Snapshot public catch-basin inventory and summarize infrastructure per grid."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

import geopandas as gpd
import numpy as np
from sklearn.neighbors import KDTree

ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/sewer-catch-basins'
RAW=ROOT/'data/raw/sewer_catch_basins_snapshot.geojson'
FEATURES_PATH=ROOT/'data/processed/spatial_training/catch_basin_grid_features.csv'
DRAINAGE_FEATURES=['catch_basin_count','catch_basin_density_per_km2','catch_basin_nearest_centroid_m']


def download():
    request_headers={'User-Agent':'VanPredict research prototype'}
    with urllib.request.urlopen(urllib.request.Request(SOURCE,headers=request_headers),timeout=60) as response:
        metadata=json.load(response)
    with urllib.request.urlopen(urllib.request.Request(SOURCE+'/exports/geojson',headers=request_headers),timeout=120) as response:
        content=response.read()
    payload=json.loads(content)
    if payload.get('type')!='FeatureCollection' or not payload.get('features'):
        raise ValueError('Expected nonempty catch-basin GeoJSON inventory')
    expected=metadata['metas']['default']['records_count']
    if len(payload['features'])!=expected:raise ValueError('Inventory export count does not match provider metadata')
    RAW.parent.mkdir(parents=True,exist_ok=True)
    temporary=RAW.with_suffix('.geojson.tmp');temporary.write_bytes(content);temporary.replace(RAW)
    snapshot={'source_url':SOURCE,'fetched_at_utc':datetime.now(timezone.utc).isoformat(),
              'source_modified_at':metadata['metas']['default'].get('modified'),
              'records':expected,'sha256':hashlib.sha256(content).hexdigest(),
              'license':metadata['metas']['default'].get('license'),
              'historical_asset_inventory_verified':False,
              'limitation':'Current inventory snapshot does not establish infrastructure as it existed in earlier forecast years.'}
    RAW.with_suffix('.metadata.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(f'Downloaded {expected:,} catch-basin records',flush=True)


def aggregate(grid,basins):
    if grid.empty or basins.empty or grid.crs is None or basins.crs is None:
        raise ValueError('Expected nonempty geospatial grid and inventory with CRS')
    if grid.grid_id.duplicated().any():raise ValueError('Duplicate grid identifiers')
    if not basins.geometry.geom_type.eq('Point').all():raise ValueError('Inventory must contain point geometries')
    for frame in (grid,basins):
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
            raise ValueError('Missing, empty or invalid geometry')
    grid=grid[['grid_id','geometry']].to_crs('EPSG:26910').copy()
    basins=basins[['geometry']].to_crs(grid.crs).copy().reset_index(drop=True)
    basins['_record']=np.arange(len(basins))
    xy=np.column_stack([basins.geometry.x,basins.geometry.y])
    areas=grid.geometry.area.to_numpy()/1_000_000
    if not np.isfinite(xy).all() or not np.isfinite(areas).all() or (areas<=0).any():
        raise ValueError('Invalid projected coordinates or grid area')
    joined=gpd.sjoin(basins,grid,how='left',predicate='intersects')
    matches=joined.loc[joined.grid_id.notna()].copy()
    multiple=int(matches['_record'].duplicated().sum())
    # Assign boundary points exactly once with a deterministic grid-ID tie break.
    assigned=matches.sort_values(['_record','grid_id'],kind='stable').drop_duplicates('_record')
    counts=assigned.groupby('grid_id').size()
    output=grid[['grid_id']].copy()
    output['catch_basin_count']=output.grid_id.map(counts).fillna(0).astype(int)
    output['grid_area_km2']=areas
    output['catch_basin_density_per_km2']=output.catch_basin_count/areas
    centers=grid.geometry.centroid
    center_xy=np.column_stack([centers.x,centers.y])
    distances,_=KDTree(xy).query(center_xy,k=1)
    output['catch_basin_nearest_centroid_m']=distances[:,0]
    if not np.isfinite(output[DRAINAGE_FEATURES].to_numpy()).all():raise ValueError('Nonfinite drainage features')
    assigned_count=int(output.catch_basin_count.sum())
    audit={'grid_cells':len(grid),'inventory_records':len(basins),'assigned_records':assigned_count,
           'outside_grid_records':len(basins)-assigned_count,'boundary_extra_matches_resolved':multiple,
           'co_located_coordinate_records':int(basins.geometry.to_wkb().duplicated().sum()),
           'grids_with_basins':int((output.catch_basin_count>0).sum()),
           'grids_without_basins':int((output.catch_basin_count==0).sum()),
           'basin_count_range':[int(output.catch_basin_count.min()),int(output.catch_basin_count.max())],
           'feature_names':DRAINAGE_FEATURES,'projected_crs':'EPSG:26910',
           'density_denominator':'Area of saved grid polygon, which may be clipped at the city boundary',
           'nearest_distance':'Distance from grid polygon centroid to nearest inventory point, including points outside the grid',
           'limitations':['Inventory records are counted; co-located records are retained because no stable asset ID is supplied.',
                          'Catch-basin presence and count do not establish hydraulic capacity, blockage or maintenance condition.',
                          'Locations may be approximate and centroid distance is not a flow path.',
                          'Zero inventory records in a grid does not establish lack of drainage.',
                          'Current inventory cannot establish historically available infrastructure for 2024-2026 predictions.']}
    if assigned_count+audit['outside_grid_records']!=len(basins):raise ValueError('Inventory assignment count mismatch')
    return output,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    if args.download or not RAW.exists():download()
    snapshot=json.loads(RAW.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(RAW.read_bytes()).hexdigest()!=snapshot['sha256']:raise ValueError('Inventory snapshot checksum mismatch')
    features,audit=aggregate(gpd.read_file(ROOT/'data/processed/vancouver_grid.geojson'),gpd.read_file(RAW))
    FEATURES_PATH.parent.mkdir(parents=True,exist_ok=True)
    features.to_csv(FEATURES_PATH,index=False)
    report={**audit,'snapshot':snapshot}
    (ROOT/'docs/CATCH_BASIN_GRID_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:audit[key] for key in ('grid_cells','inventory_records','assigned_records','outside_grid_records','grids_with_basins','grids_without_basins')},indent=2),flush=True)


if __name__=='__main__':main()
