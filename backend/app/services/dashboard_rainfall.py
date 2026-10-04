"""Historical liquid rain estimates; missing days are never filled with zero."""
from datetime import date
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from app.services.reports import load_reports

ROOT = Path(__file__).resolve().parents[3]
WEATHER = ROOT / 'data/processed/spatial_weather/era5_500m'


@lru_cache(maxsize=1)
def rainfall_data():
    daily = pd.read_csv(WEATHER / 'weather_daily_features.csv.gz')
    mapping = pd.read_csv(WEATHER / 'grid_weather_mapping.csv')
    grid = gpd.read_file(ROOT / 'frontend/public/grids/vancouver_500m.geojson')
    weights = grid[['grid_id', 'area_m2']].merge(mapping, on='grid_id', validate='one_to_one')
    weights = weights.groupby('weather_cell_id').area_m2.sum()
    weights = weights / weights.sum()
    if daily.duplicated(['local_date', 'weather_cell_id']).any():
        raise ValueError('Duplicate rainfall source days')
    usable = daily.loc[daily.day_complete.eq(True)].copy()
    if not np.isfinite(usable.rainfall_today).all() or (usable.rainfall_today < 0).any():
        raise ValueError('Invalid rainfall totals')
    table = usable.pivot(index='local_date', columns='weather_cell_id', values='rainfall_today').reindex(columns=weights.index)
    city = table.mul(weights).sum(axis=1, min_count=len(weights))
    return daily, mapping, city


@lru_cache(maxsize=3)
def cell_weather(size):
    _, mapping, _ = rainfall_data()
    if size == 500:
        return mapping.set_index('grid_id').weather_cell_id.to_dict()
    grid = gpd.read_file(ROOT / f'frontend/public/grids/vancouver_{size}m.geojson')
    centroids = grid.to_crs('EPSG:32610').centroid.to_crs('EPSG:4326')
    sources = mapping.drop_duplicates('weather_cell_id')
    # Same nearest-source assignment as the existing 500 m mapping; not finer weather.
    distances = (centroids.y.to_numpy()[:, None] - sources.weather_latitude.to_numpy()) ** 2 + (centroids.x.to_numpy()[:, None] - sources.weather_longitude.to_numpy()) ** 2
    return dict(zip(grid.grid_id, sources.weather_cell_id.to_numpy()[distances.argmin(axis=1)]))


def rainfall_summary(start: date | None, end: date | None, size: int, cell_id: str | None):
    daily, _, city = rainfall_data()
    reports = load_reports()
    first, last = start or reports.date.min(), end or reports.date.max()
    dates = pd.date_range(first, last).strftime('%Y-%m-%d')
    series = city.reindex(dates)
    weather_id = None
    if cell_id:
        weather_id = cell_weather(size).get(cell_id)
        if weather_id is None:
            raise KeyError(cell_id)
        source = daily.loc[daily.weather_cell_id.eq(weather_id) & daily.day_complete.eq(True)].set_index('local_date').rainfall_today
        cell_series = source.reindex(dates)
    else:
        cell_series = None

    def stats(values):
        valid = values.dropna()
        return {'total_mm': round(float(valid.sum()), 2) if len(valid) else None,
                'daily_average_mm': round(float(valid.mean()), 2) if len(valid) else None,
                'covered_days': len(valid), 'requested_days': len(dates),
                'complete': len(valid) == len(dates)}

    return {'start': first.isoformat(), 'end': last.isoformat(),
            'coverage_start': str(city.dropna().index.min()), 'coverage_end': str(city.dropna().index.max()),
            'city': stats(series), 'cell': stats(cell_series) if cell_series is not None else None,
            'cell_id': cell_id, 'grid_size': size, 'weather_cell_id': weather_id,
            'daily': [{'date': day, 'mm': round(float(value), 2) if pd.notna(value) else None}
                      for day, value in series.tail(60).items()],
            'source': 'ERA5 via Open-Meteo', 'units': 'mm', 'timezone': 'America/Vancouver'}

