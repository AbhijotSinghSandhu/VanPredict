import sys
import unittest
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.services.dashboard_rainfall import WEATHER, rainfall_summary, cell_weather
from app.routers.rainfall import rainfall


class RainfallTests(unittest.TestCase):
    def test_totals_match_source_and_area_weights(self):
        daily = pd.read_csv(WEATHER / 'weather_daily_features.csv.gz')
        mapping = pd.read_csv(WEATHER / 'grid_weather_mapping.csv')
        grid = gpd.read_file(ROOT / 'frontend/public/grids/vancouver_500m.geojson')
        weights = grid.merge(mapping, on='grid_id').groupby('weather_cell_id').area_m2.sum()
        weights /= weights.sum()
        source = daily.loc[daily.local_date.between('2025-06-04', '2025-07-04')]
        totals = source.groupby('weather_cell_id').rainfall_today.sum()
        result = rainfall_summary(date(2025, 6, 4), date(2025, 7, 4), 500, 'g500_980_10918')
        self.assertEqual(result['city']['total_mm'], round(float((weights * totals).sum()), 2))
        self.assertEqual(result['cell']['total_mm'], round(float(totals[result['weather_cell_id']]), 2))
        self.assertEqual(result['city']['covered_days'], 31)
        self.assertTrue(result['city']['complete'])

    def test_missing_days_are_null_not_zero(self):
        result = rainfall_summary(date(2026, 9, 28), date(2026, 10, 2), 500, None)
        self.assertIsNone(result['city']['total_mm'])
        self.assertEqual(result['city']['covered_days'], 0)
        self.assertTrue(all(day['mm'] is None for day in result['daily']))
        partial = rainfall_summary(date(2026, 9, 27), date(2026, 9, 28), 500, None)
        self.assertEqual(partial['city']['covered_days'], 1)
        self.assertFalse(partial['city']['complete'])

    def test_all_grid_cells_have_a_source(self):
        for size in (200, 500, 1000):
            mapping = cell_weather(size)
            grid = gpd.read_file(ROOT / f'frontend/public/grids/vancouver_{size}m.geojson')
            self.assertEqual(set(mapping), set(grid.grid_id))
            self.assertEqual(len(set(mapping.values())), 2)

    def test_invalid_queries_and_cell(self):
        for args in [(date(2025, 7, 4), date(2025, 6, 4), 500, None),
                     (date(2025, 6, 4), None, 500, None), (None, None, 300, None),
                     (None, None, 500, 'invalid')]:
            with self.assertRaises(HTTPException):
                rainfall(*args)


if __name__ == '__main__':
    unittest.main()
