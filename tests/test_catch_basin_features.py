import unittest
import geopandas as gpd
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from shapely.geometry import Point, box
from src.catch_basin_features import aggregate
from src.train_forecast_drainage_36h import attach_inventory
from src.train_forecast_36h import DATA, predict
from src.catch_basin_features import FEATURES_PATH, DRAINAGE_FEATURES


class CatchBasinTests(unittest.TestCase):
    def test_boundary_outside_and_empty_grid(self):
        grid=gpd.GeoDataFrame({'grid_id':[1,2,3]},geometry=[box(0,0,500,500),box(500,0,1000,500),box(1000,0,1500,500)],crs=26910)
        basins=gpd.GeoDataFrame(geometry=[Point(100,100),Point(500,250),Point(900,250),Point(2000,0)],crs=26910)
        result,audit=aggregate(grid,basins)
        self.assertEqual(result.catch_basin_count.tolist(),[2,1,0])
        self.assertEqual(result.catch_basin_density_per_km2.tolist(),[8,4,0])
        self.assertEqual(audit['outside_grid_records'],1)
        self.assertEqual(audit['boundary_extra_matches_resolved'],1)
        self.assertAlmostEqual(result.catch_basin_nearest_centroid_m.iloc[1],150)

    def test_colocated_records_retained(self):
        grid=gpd.GeoDataFrame({'grid_id':[1]},geometry=[box(0,0,500,500)],crs=26910)
        basins=gpd.GeoDataFrame(geometry=[Point(100,100),Point(100,100)],crs=26910)
        result,audit=aggregate(grid,basins)
        self.assertEqual(result.catch_basin_count.iloc[0],2)
        self.assertEqual(audit['co_located_coordinate_records'],1)

    def test_missing_inventory_is_rejected(self):
        inventory=pd.DataFrame({'grid_id':[1],'catch_basin_count':[0],'catch_basin_density_per_km2':[0],'catch_basin_nearest_centroid_m':[10]})
        with self.assertRaises(ValueError):attach_inventory(pd.DataFrame({'grid_id':[2]}),inventory)

    def test_saved_research_model_contract(self):
        artifact=Path(__file__).resolve().parents[1]/'models/forecast_36h_drainage_v1.joblib'
        if not artifact.exists():self.skipTest('Research artifact not generated')
        model=joblib.load(artifact)
        rows=attach_inventory(pd.read_csv(DATA,nrows=604),pd.read_csv(FEATURES_PATH))
        probabilities=predict(model,rows)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities>=0)&(probabilities<=1)).all())
        self.assertTrue(set(DRAINAGE_FEATURES).issubset(model['features']))
        self.assertEqual(model['metadata']['status'],'research_only')


if __name__=='__main__':unittest.main()
