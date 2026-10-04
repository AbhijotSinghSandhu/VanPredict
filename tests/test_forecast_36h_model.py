import unittest

import numpy as np
import pandas as pd

from src.train_forecast_36h import split_table,paired_top20_interval,scenario_masks,FORECAST_FEATURES,ROOT,predict


class ForecastModelTests(unittest.TestCase):
    def test_boundary_purge_includes_report_availability_delay(self):
        issues=pd.date_range('2024-12-28T06:00',periods=4,freq='D',tz='America/Vancouver').tz_convert('UTC')
        table=pd.DataFrame({'prediction_time':[stamp.isoformat() for stamp in issues],
                            'prediction_end':[(stamp+pd.Timedelta(hours=36)).isoformat() for stamp in issues]})
        retained=split_table(table,'training')
        self.assertEqual(retained.prediction_time.tolist(),table.prediction_time.iloc[:2].tolist())
        next_start=pd.Timestamp('2025-01-01T00:00-08:00')
        self.assertTrue(((pd.to_datetime(retained.prediction_end,utc=True)+pd.Timedelta(hours=24))<=next_start).all())

    def test_paired_weekly_interval_is_zero_for_identical_rankings(self):
        table=pd.DataFrame({'prediction_time':np.repeat(['2026-01-02T14:00Z','2026-01-10T14:00Z'],25),
                            'grid_id':[f'g{i:02}' for i in range(25)]*2,
                            'target_36h':[1]+[0]*24+[1]+[0]*24})
        scores=np.tile(np.linspace(.9,.1,25),2)
        result=paired_top20_interval(table,scores,scores,draws=100)
        self.assertEqual(result['forecast_minus_history_precision'],0)
        self.assertEqual(result['ci95'],[0,0])
        self.assertEqual(result['week_blocks'],2)

    def test_weather_scenarios_classify_whole_dates_without_weighting_grid_counts(self):
        table=pd.DataFrame({'prediction_time':['a']*4+['b']*4,
                            'weather_cell_id':['west','east','east','east']*2,
                            'rain_36h':[50,0,0,0,0,0,0,0],'showers_36h':[0]*8,
                            'rain_change_second_half_36h':[14,0,0,0,0,0,0,0],
                            'showers_change_second_half_36h':[0]*8})
        masks=scenario_masks(table)
        self.assertEqual(masks['heavy_rain'].tolist(),[True]*4+[False]*4)
        self.assertEqual(masks['increasing_rain'].tolist(),[True]*4+[False]*4)
        self.assertEqual(masks['dry_or_trace'].tolist(),[False]*4+[True]*4)

    @unittest.skipUnless((ROOT/'models/forecast_comparison_forecast_aware_v1.joblib').exists(),'Run experiment first')
    def test_saved_forecast_model_obeys_contract_and_emits_valid_probabilities(self):
        import joblib
        saved=joblib.load(ROOT/'models/forecast_comparison_forecast_aware_v1.joblib')
        self.assertEqual(saved['features'],FORECAST_FEATURES)
        self.assertEqual(saved['metadata']['status'],'research_only')
        table=pd.read_csv(ROOT/'data/processed/spatial_training/forecast_36h_full_v2.csv.gz',nrows=604)
        scores=predict(saved,table)
        self.assertEqual(len(scores),604)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores>=0)&(scores<=1)).all())
        with self.assertRaises(KeyError):predict(saved,table.drop(columns=['rain_36h']))


if __name__=='__main__':unittest.main()
