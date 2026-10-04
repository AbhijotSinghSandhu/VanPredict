import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.prediction import router

app=FastAPI();app.include_router(router)
client=TestClient(app)


class PredictionApiTests(unittest.TestCase):
    def test_prepared_real_model_returns_all_cells_and_ranked_shortlist(self):
        response=client.post('/api/v1/predict',json={'prediction_time':'2025-06-04T06:00:00-07:00','top_k':5})
        self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        self.assertEqual(len(result['areas']),604)
        self.assertEqual(len(result['top_areas']),5)
        self.assertFalse(result['uses_forecast_rainfall'])
        scores=[cell['report_probability'] for cell in result['areas']]
        self.assertEqual(scores,sorted(scores,reverse=True))
        self.assertTrue(all(0<=score<=1 for score in scores))
        self.assertEqual(result['prediction_end'],'2025-06-06T01:00:00+00:00')

    def test_invalid_requests(self):
        for body in [{'grid_size_m':200},{'forecast_hours':24},{'top_k':605},
                     {'prediction_time':'2025-06-04T06:00:00'},
                     {'prediction_time':'2025-06-04T07:00:00-07:00'},
                     {'forecast_rain_mm':10}]:
            self.assertEqual(client.post('/api/v1/predict',json=body).status_code,422)

    def test_current_time_does_not_silently_use_stale_weather(self):
        response=client.post('/api/v1/predict',json={'prediction_time':'2026-10-04T06:00:00-07:00'})
        self.assertEqual(response.status_code,422)
        self.assertIn('coverage',response.json()['detail'])

    def test_missing_artifact_is_service_unavailable(self):
        with patch('app.routers.prediction.prepared_prediction',side_effect=FileNotFoundError):
            self.assertEqual(client.post('/api/v1/predict',json={}).status_code,503)

    def test_caller_features_and_duplicate_rejection(self):
        sample=pd.read_csv(ROOT/'data/processed/spatial_training/history_36h.csv.gz',nrows=1).iloc[0]
        keys=['centroid_latitude','centroid_longitude','rain_past_24h','rain_past_72h','rain_past_168h',
              'reports_past_24h','reports_past_168h','reports_past_720h','historical_reports_per_day']
        cell={k:float(sample[k]) for k in keys};cell['cell_id']=sample.grid_id
        body={'prediction_time':sample.prediction_time,'cells':[cell]}
        response=client.post('/api/v1/predict/features',json=body)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['areas'][0]['cell_id'],sample.grid_id)
        body['cells']=[cell,cell]
        self.assertEqual(client.post('/api/v1/predict/features',json=body).status_code,422)

    def test_inference_builder_does_not_require_or_return_future_labels(self):
        from src.training_36h import build_table
        from app.services.prediction import sources
        mapping,reports,weather=sources()
        issue=pd.Timestamp('2025-06-04T13:00Z')
        frame=build_table(mapping,reports,weather,[issue],'2009-06-24T00:00-07:00',issue,include_targets=False)
        self.assertNotIn('target_36h',frame)
        self.assertNotIn('report_count_36h',frame)


if __name__=='__main__':
    unittest.main()
