import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.forecast_training_36h import WEATHER_FEATURES, features_for_location, run_for_issue, fetch_run, RunUnavailable
from src.build_forecast_archive_36h import month_ranges, audit
from src.training_36h import FEATURES


def payload(rain=None,showers=None,snowfall=None):
    times=pd.date_range('2025-06-04T06:00Z',periods=48,freq='h')
    rain=rain if rain is not None else [0.0]*48
    showers=showers if showers is not None else [0.0]*48
    snowfall=snowfall if snowfall is not None else [0.0]*48
    return {'hourly_units':{'rain':'mm','showers':'mm','snowfall':'cm'},
            'hourly':{'time':(times.as_unit('ns').asi8//10**9).tolist(),'rain':rain,'showers':showers,
                      'snowfall':snowfall,'temperature_2m':[10.0]*48}}


class ForecastTrainingTests(unittest.TestCase):
    def test_uses_only_next_36_intervals_and_captures_intensity_and_changes(self):
        rain=[0.0]*48;rain[8]=2;rain[9]=4;rain[10]=1
        showers=[0.0]*48;showers[43]=3
        result=features_for_location(payload(rain,showers),pd.Timestamp('2025-06-04T13:00Z'))
        self.assertEqual(result['rain_6h'],7)
        self.assertEqual(result['rain_36h'],7)
        self.assertEqual(result['rain_peak_hourly_36h'],4)
        self.assertEqual(result['rain_peak_3h_36h'],7)
        self.assertEqual(result['showers_36h'],3)
        self.assertEqual(result['showers_change_second_half_36h'],3)
        self.assertTrue(set(WEATHER_FEATURES).issubset(result))

    def test_rejects_missing_hours_or_values(self):
        entry=payload();entry['hourly']['rain'][8]=None
        with self.assertRaisesRegex(ValueError,'missing values'):
            features_for_location(entry,pd.Timestamp('2025-06-04T13:00Z'))
        entry=payload();entry['hourly']['time'].pop(10)
        for key in ('rain','showers','snowfall','temperature_2m'):entry['hourly'][key].pop(10)
        with self.assertRaisesRegex(ValueError,'continuous'):
            features_for_location(entry,pd.Timestamp('2025-06-04T13:00Z'))

    def test_only_expected_issue_time_has_a_prior_archived_run(self):
        run=run_for_issue(pd.Timestamp('2025-06-04T13:00Z'))
        self.assertEqual(run,pd.Timestamp('2025-06-04T06:00Z'))
        self.assertEqual(run_for_issue(pd.Timestamp('2025-06-04T06:00',tz='America/Vancouver')),run)
        with self.assertRaisesRegex(ValueError,'06:00 Vancouver'):
            run_for_issue(pd.Timestamp('2025-06-04T14:00Z'))

    def test_missing_cycle_falls_back_only_to_earlier_initialization(self):
        issue=pd.Timestamp('2024-06-04T13:00Z')
        earlier=pd.Timestamp('2024-06-04T00:00Z')
        with tempfile.TemporaryDirectory() as folder, patch('src.forecast_training_36h.ARCHIVE',Path(folder)), patch('src.forecast_training_36h._fetch_run_at',side_effect=[RunUnavailable('missing cycle'),({},earlier)]) as fetch:
            _,run=fetch_run(issue,pd.DataFrame())
        self.assertEqual(run,earlier)
        self.assertEqual([call.args[0] for call in fetch.call_args_list],
                         [pd.Timestamp('2024-06-04T06:00Z'),earlier])
        with tempfile.TemporaryDirectory() as folder, patch('src.forecast_training_36h.ARCHIVE',Path(folder)), patch('src.forecast_training_36h._fetch_run_at',side_effect=TimeoutError) as fetch:
            with self.assertRaises(TimeoutError):fetch_run(issue,pd.DataFrame())
            self.assertEqual(fetch.call_count,1)

    def test_month_partitions_cover_every_date_without_gaps(self):
        partitions=list(month_ranges('2024-03-14','2024-05-09'))
        self.assertEqual(partitions,[('2024-03-14','2024-03-31'),('2024-04-01','2024-04-30'),('2024-05-01','2024-05-09')])
        dates=[day for start,end in partitions for day in pd.date_range(start,end)]
        self.assertEqual(dates,list(pd.date_range('2024-03-14','2024-05-09')))

    def test_audit_rejects_missing_grid_and_requested_date(self):
        mapping=pd.DataFrame({'grid_id':['g1','g2'],'weather_cell_id':['w1','w2']})
        rows=[]
        for grid,source in zip(mapping.grid_id,mapping.weather_cell_id):
            rows.append({**{name:0.0 for name in FEATURES+WEATHER_FEATURES},
                         'grid_id':grid,'weather_cell_id':source,
                         'prediction_time':'2025-06-04T13:00:00+00:00',
                         'prediction_end':'2025-06-06T01:00:00+00:00',
                         'history_cutoff':'2025-06-03T13:00:00+00:00',
                         'forecast_run':'2025-06-04T06:00:00+00:00',
                         'forecast_issue_delay_hours':7,'report_count_36h':0,'target_36h':0})
        table=pd.DataFrame(rows)
        valid=audit(table,mapping,'2025-06-04','2025-06-04',[])
        self.assertEqual(valid['status'],'structural_checks_passed')
        invalid=audit(table.iloc[:1],mapping,'2025-06-04','2025-06-05',[])
        self.assertEqual(invalid['status'],'structural_checks_failed')
        self.assertEqual(invalid['checks']['missing_requested_dates'],['2025-06-05'])
        self.assertEqual(len(invalid['checks']['incomplete_grid_dates']),1)
        documented=audit(table,mapping,'2025-06-04','2025-06-05',
                         [{'date':'2025-06-05','error':'Archive unavailable'}])
        self.assertEqual(documented['status'],'structural_checks_passed_with_archive_gaps')
        self.assertEqual(documented['checks']['unexplained_missing_dates'],[])


if __name__=='__main__':unittest.main()
