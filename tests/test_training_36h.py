import unittest
import pandas as pd
from src.training_36h import build_table, prediction_times, partition


class Training36hTests(unittest.TestCase):
    def setUp(self):
        self.issue = pd.Timestamp('2025-11-01T13:00:00Z')
        self.mapping = pd.DataFrame([dict(grid_id=g, centroid_latitude=49., centroid_longitude=-123., weather_cell_id='w') for g in ('a','b')])
        self.weather = pd.DataFrame(dict(datetime=pd.date_range(self.issue-pd.Timedelta(hours=167), self.issue+pd.Timedelta(hours=36), freq='h'), rainfall_mm=1., weather_cell_id='w'))
        self.reports = pd.DataFrame([dict(grid_id='a', service_request_open_timestamp=t.isoformat(), service_request_type=category) for t, category in [
            (self.issue-pd.Timedelta(hours=25), 'Catch Basin Concern Case'),
            (self.issue-pd.Timedelta(hours=1), 'Catch Basin Concern Case'),
            (self.issue, 'Catch Basin Concern Case'),
            (self.issue+pd.Timedelta(hours=35), 'Street Surface Water Flooding Case'),
            (self.issue+pd.Timedelta(hours=36), 'Catch Basin Concern Case'),
            (self.issue+pd.Timedelta(hours=2), 'Sewer Backup Case')]])

    def build(self, **kwargs):
        return build_table(self.mapping, self.reports, self.weather, [self.issue],
                           '2025-01-01T00:00:00Z', '2025-12-01T00:00:00Z', **kwargs)

    def test_boundaries_categories_and_grid(self):
        table = self.build().set_index('grid_id')
        self.assertEqual(table.loc['a','report_count_36h'], 2)
        self.assertEqual(table.loc['b','target_36h'], 0)
        self.assertEqual(table.loc['a','reports_past_24h'], 1)
        self.assertEqual(table.loc['a','rain_past_168h'], 168)

    def test_future_reports_and_rain_cannot_change_features(self):
        before = self.build()
        self.weather.loc[self.weather.datetime > self.issue, 'rainfall_mm'] = 1000
        self.reports = self.reports.loc[pd.to_datetime(self.reports.service_request_open_timestamp, utc=True) < self.issue]
        after = self.build()
        from src.training_36h import FEATURES
        pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])

    def test_missing_rain_rejected(self):
        self.weather = self.weather.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.build()

    def test_coverage_and_duplicate_times_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            build_table(self.mapping,self.reports,self.weather,[self.issue],self.issue,self.issue+pd.Timedelta(hours=35))
        with self.assertRaisesRegex(ValueError, 'unique'):
            build_table(self.mapping,self.reports,self.weather,[self.issue,self.issue],'2025-01-01T00:00Z','2025-12-01T00:00Z')

    def test_dst_and_split_purge(self):
        times = prediction_times('2025-11-01','2025-11-03')
        self.assertEqual((times[1]-times[0]).total_seconds()/3600,25)
        self.assertTrue((times.tz_convert('America/Vancouver').hour == 6).all())
        table = self.build()
        self.assertEqual((pd.Timestamp(table.prediction_end.iloc[0])-self.issue).total_seconds()/3600,36)
        self.assertTrue(partition(table,self.issue,self.issue+pd.Timedelta(hours=35)).empty)
        self.assertEqual(len(partition(table,self.issue,self.issue+pd.Timedelta(hours=36))),2)


if __name__ == '__main__':
    unittest.main()
