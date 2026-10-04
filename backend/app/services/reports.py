"""Read-only aggregates of prepared drainage reports; never expose addresses."""
from datetime import date
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_reports():
    frame = pd.read_csv(ROOT / 'data/processed/311_gridded.csv',
                        usecols=['date', 'grid_id', 'latitude', 'longitude'])
    frame['date'] = pd.to_datetime(frame['date'], errors='raise').dt.date
    return frame


@lru_cache(maxsize=3)
def assignments(size):
    frame = load_reports().copy()
    grid = gpd.read_file(ROOT / f'frontend/public/grids/vancouver_{size}m.geojson')
    if size != 500:
        points = gpd.GeoDataFrame(frame.drop(columns='grid_id'),
                                 geometry=gpd.points_from_xy(frame.longitude, frame.latitude), crs='EPSG:4326')
        joined = gpd.sjoin(points, grid[['grid_id', 'geometry']], how='left', predicate='intersects')
        # Boundary ties count each report once, using the same stable-ID rule as preparation.
        joined = joined.sort_values('grid_id').loc[lambda x: ~x.index.duplicated()]
        frame['grid_id'] = joined.grid_id.reindex(frame.index)
    return frame, set(grid.grid_id)


def summarize(size: int, start: date | None = None, end: date | None = None):
    frame, ids = assignments(size)
    coverage_start, coverage_end = frame.date.min(), frame.date.max()
    first, last = start or coverage_start, end or coverage_end
    filtered = frame.loc[(frame.date >= first) & (frame.date <= last)]
    mapped = filtered.loc[filtered.grid_id.isin(ids)]
    return {
        'start': first.isoformat(), 'end': last.isoformat(),
        'coverage_start': coverage_start.isoformat(), 'coverage_end': coverage_end.isoformat(),
        'total_reports': len(filtered), 'mapped_reports': len(mapped),
        'unmapped_reports': len(filtered) - len(mapped),
        'partial_coverage': first < coverage_start or last > coverage_end,
        'grid_size': size,
        'counts': {str(key): int(value) for key, value in mapped.groupby('grid_id').size().items()},
        'source': 'Prepared Vancouver 311 drainage reports',
    }
