import unittest

import numpy as np
import pandas as pd

from src.forecast_transitions_36h import TRANSITION_FEATURES,transition_features
from src.train_forecast_36h_v2 import CANDIDATES,ROOT,choose_candidate,partition
from src.train_forecast_36h import predict


def payload():
    times=pd.date_range('2025-06-04T06:00Z',periods=60,freq='h')
    return {'hourly_units':{'rain':'mm','showers':'mm','snowfall':'cm'},
            'hourly':{'time':(times.as_unit('ns').asi8//10**9).tolist(),
                      'rain':[0.]*60,'showers':[0.]*60,'snowfall':[0.]*60,'temperature_2m':[-1.]*60}}


class ForecastV2Tests(unittest.TestCase):
    def test_onset_spell_and_temperature_use_only_prediction_intervals(self):
        entry=payload()
        entry['hourly']['rain'][7]=500  # Interval ending at issue, excluded.
        entry['hourly']['rain'][8:11]=[1,2,3]
        entry['hourly']['rain'][43]=2  # Last included interval.
        entry['hourly']['rain'][44]=500  # Following interval, excluded.
        entry['hourly']['temperature_2m'][43]=2
        result=transition_features(entry,pd.Timestamp('2025-06-04T13:00Z'))
        self.assertEqual(result['forecast_rain_onset_hours'],1)
        self.assertEqual(result['forecast_longest_wet_spell_hours'],3)
        self.assertEqual(result['forecast_longest_dry_spell_hours'],32)
        self.assertEqual(result['forecast_liquid_peak_6h_mm'],6)
        self.assertEqual(result['forecast_heavy_hours_2mm'],3)
        self.assertEqual(result['forecast_temperature_change_c'],3)
        self.assertEqual(result['forecast_freezing_crossings'],1)
        self.assertEqual(result['forecast_below_freezing_hours'],35)

    def test_dry_window_has_explicit_no_onset_sentinel(self):
        result=transition_features(payload(),pd.Timestamp('2025-06-04T13:00Z'))
        self.assertEqual(result['forecast_rain_onset_hours'],37)
        self.assertEqual(result['forecast_longest_wet_spell_hours'],0)
        self.assertEqual(result['forecast_longest_dry_spell_hours'],36)
        self.assertEqual(set(result),set(TRANSITION_FEATURES))

    def test_winner_uses_validation_top20_before_average_precision(self):
        metrics={'better_top20':{'shortlists':{'20':{'precision':.09}},'average_precision':.08,'brier_score':.01},
                 'better_ap':{'shortlists':{'20':{'precision':.08}},'average_precision':.20,'brier_score':.009}}
        self.assertEqual(choose_candidate(metrics),'better_top20')
        metrics['better_ap']['shortlists']['20']['precision']=.09
        self.assertEqual(choose_candidate(metrics),'better_ap')

    def test_extended_training_and_validation_keep_report_delay_embargo(self):
        stamps=pd.date_range('2025-06-27T06:00',periods=5,freq='D',tz='America/Vancouver').tz_convert('UTC')
        table=pd.DataFrame({'prediction_time':[t.isoformat() for t in stamps],
                            'prediction_end':[(t+pd.Timedelta(hours=36)).isoformat() for t in stamps]})
        training=partition(table,'training')
        calibration=partition(table,'calibration')
        self.assertEqual(len(training),2)
        self.assertEqual(len(calibration),1)
        self.assertLess(pd.to_datetime(training.prediction_end,utc=True).max()+pd.Timedelta(hours=24),
                        pd.to_datetime(calibration.prediction_time,utc=True).min())

    @unittest.skipUnless((ROOT/'models/forecast_36h_v2_selected.joblib').exists(),'Run v2 experiment first')
    def test_saved_candidate_contract_and_probability_output(self):
        import joblib
        saved=joblib.load(ROOT/'models/forecast_36h_v2_selected.joblib')
        candidate=saved['metadata']['selected_candidate']
        self.assertEqual(saved['features'],CANDIDATES[candidate]['features'])
        self.assertEqual(saved['metadata']['status'],'research_only')
        sample=pd.read_csv(ROOT/'data/processed/spatial_training/forecast_36h_transitions_v2.csv.gz',nrows=604)
        values=predict(saved,sample)
        self.assertEqual(len(values),604)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values>=0)&(values<=1)).all())


if __name__=='__main__':unittest.main()
