import unittest
import pandas as pd
from src.forecast_training_sample import choose_run,forecast_features


class ForecastSampleTests(unittest.TestCase):
    def setUp(self):
        self.issue=pd.Timestamp('2026-09-20T13:00Z');self.run=choose_run(self.issue)
        times=pd.date_range(self.issue,periods=38,freq='h')
        self.payload={'hourly_units':{'time':'unixtime','rain':'mm','showers':'mm'},
                      'hourly':{'time':[int(t.timestamp()) for t in times],'rain':[1.]*38,'showers':[.5]*38}}

    def test_complete_future_intervals_and_features(self):
        features=forecast_features(self.payload,self.issue,self.run)
        self.assertEqual(features['forecast_rain_36h'],54)
        self.assertEqual(features['forecast_rain_6h'],9)
        self.assertEqual(features['forecast_peak_3h'],4.5)
        self.assertEqual(features['forecast_longest_wet_spell'],36)
        self.assertEqual(features['forecast_second18_minus_first18'],0)

    def test_run_selection_respects_publication_delay(self):
        self.assertEqual(self.run,pd.Timestamp('2026-09-20T00:00Z'))
        with self.assertRaisesRegex(ValueError,'unavailable'):
            forecast_features(self.payload,self.issue,pd.Timestamp('2026-09-20T12:00Z'))

    def test_missing_and_duplicate_hours_reject(self):
        self.payload['hourly']['rain'][10]=None
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            forecast_features(self.payload,self.issue,self.run)
        self.setUp();self.payload['hourly']['time'][10]=self.payload['hourly']['time'][9]
        with self.assertRaisesRegex(ValueError,'timestamps'):
            forecast_features(self.payload,self.issue,self.run)

    def test_outside_window_values_do_not_influence_features(self):
        before=forecast_features(self.payload,self.issue,self.run)
        self.payload['hourly']['rain'][0]=999
        self.payload['hourly']['rain'][-1]=999
        self.assertEqual(before,forecast_features(self.payload,self.issue,self.run))


if __name__=='__main__':
    unittest.main()
