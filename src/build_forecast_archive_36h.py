"""Resume a complete archived-forecast dataset in calendar-month partitions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.forecast_training_36h import ARCHIVE, fetch_run, features_for_location, _location_records, WEATHER_FEATURES, run_for_issue
from src.training_36h import FEATURES, CATEGORIES, prediction_times


def month_ranges(start,end):
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    if start>end: raise ValueError('Start date must precede end date')
    current=start
    while current<=end:
        last=min(current+pd.offsets.MonthEnd(0),end)
        yield current.strftime('%Y-%m-%d'),last.strftime('%Y-%m-%d')
        current=last+pd.Timedelta(days=1)


def audit(table,mapping,start,end,download_failures):
    expected_dates=set(pd.date_range(start,end).strftime('%Y-%m-%d'))
    actual_dates=set(table.prediction_time.str[:10])
    documented_exclusions={item['date'] for item in download_failures}
    starts=pd.to_datetime(table.prediction_time,utc=True)
    ends=pd.to_datetime(table.prediction_end,utc=True)
    cutoff=pd.to_datetime(table.history_cutoff,utc=True)
    runs=pd.to_datetime(table.forecast_run,utc=True)
    valid_grid=set(mapping.grid_id)
    incomplete=[]
    for stamp,group in table.groupby('prediction_time'):
        if len(group)!=len(mapping) or set(group.grid_id)!=valid_grid:
            incomplete.append(stamp)
    source_groups=table.groupby(['prediction_time','weather_cell_id'])
    duplicate_weather_inconsistencies=int((source_groups[WEATHER_FEATURES].nunique().max(axis=1)>1).sum())
    checks={
        'duplicate_cell_windows':int(table.duplicated(['prediction_time','grid_id']).sum()),
        'missing_requested_dates':sorted(expected_dates-actual_dates),
        'unexplained_missing_dates':sorted(expected_dates-actual_dates-documented_exclusions),
        'unexpected_dates':sorted(actual_dates-expected_dates),
        'incomplete_grid_dates':incomplete,
        'nonfinite_input_values':int((~np.isfinite(table[FEATURES+WEATHER_FEATURES].to_numpy(dtype=float))).sum()),
        'incorrect_horizons':int(((ends-starts)!=pd.Timedelta(hours=36)).sum()),
        'incorrect_history_cutoffs':int(((starts-cutoff)!=pd.Timedelta(hours=24)).sum()),
        'incorrect_local_issue_hours':int((starts.dt.tz_convert('America/Vancouver').dt.hour!=6).sum()),
        'target_count_mismatches':int((table.target_36h!=(table.report_count_36h>0).astype(int)).sum()),
        'invalid_targets':int((~table.target_36h.isin([0,1])).sum()),
        'negative_report_counts':int((table.report_count_36h<0).sum()),
        'forecast_runs_after_issue':int((runs>=starts).sum()),
        'incorrect_forecast_runs':sum(int(pd.Timestamp(run) not in (run_for_issue(pd.Timestamp(stamp)),run_for_issue(pd.Timestamp(stamp))-pd.Timedelta(hours=6)))
                                      for stamp,run in table[['prediction_time','forecast_run']].drop_duplicates().itertuples(index=False,name=None)),
        'incorrect_run_delays':int((table.forecast_issue_delay_hours!=(starts-runs).dt.total_seconds()/3600).sum()),
        'inconsistent_weather_in_same_source_window':duplicate_weather_inconsistencies,
        'incorrect_weather_source_assignments':int((table.weather_cell_id!=table.grid_id.map(mapping.set_index('grid_id').weather_cell_id)).sum()),
    }
    for kind in ('rain','showers'):
        totals=table[[f'{kind}_{h}h' for h in (6,12,24,36)]].to_numpy()
        checks[f'{kind}_negative_or_nonmonotonic_totals']=int(((totals<0).any(axis=1)|(np.diff(totals,axis=1)<-1e-8).any(axis=1)).sum())
    passed=all(not value for key,value in checks.items() if key!='missing_requested_dates')
    daily=table.groupby('prediction_time').agg(rows=('grid_id','size'),positive_cells=('target_36h','sum'),report_occurrences=('report_count_36h','sum'))
    sources=source_groups.first().reset_index()
    sources['liquid_mm']=sources.rain_36h+sources.showers_36h
    sources['change_mm']=sources.rain_change_second_half_36h+sources.showers_change_second_half_36h
    sources['rain_band']=pd.cut(sources.liquid_mm,[-np.inf,.1,5,20,np.inf],labels=['dry_or_trace','light','moderate','heavy'])
    sources['change_band']=pd.cut(sources.change_mm,[-np.inf,-5,5,np.inf],labels=['decreasing','steady','increasing'])
    by_month=[]
    for month,group in table.groupby(table.prediction_time.str[:7]):
        by_month.append({'month':month,'issue_dates':int(group.prediction_time.nunique()),'rows':len(group),
                         'positive_cell_windows':int(group.target_36h.sum()),'positive_rate':float(group.target_36h.mean())})
    status='structural_checks_failed' if not passed else ('structural_checks_passed_with_archive_gaps' if checks['missing_requested_dates'] else 'structural_checks_passed')
    return {'version':'forecast-36h-full-audit-v1','status':status,
            'requested_start':start,'requested_end':end,'issue_dates':len(daily),'rows':len(table),
            'grid_cells':len(mapping),'weather_sources':int(mapping.weather_cell_id.nunique()),
            'source_windows':len(sources),'positive_cell_windows':int(table.target_36h.sum()),
            'positive_cell_window_rate':float(table.target_36h.mean()),
            'issue_dates_with_reports':int((daily.positive_cells>0).sum()),
            'positive_grid_cells':int(table.loc[table.target_36h.eq(1),'grid_id'].nunique()),
            'liquid_forecast_36h_mm':{'minimum':float(sources.liquid_mm.min()),'median':float(sources.liquid_mm.median()),
                                      'p95':float(sources.liquid_mm.quantile(.95)),'maximum':float(sources.liquid_mm.max())},
            'rain_band_source_windows':{str(k):int(v) for k,v in sources.rain_band.value_counts(sort=False).items()},
            'change_band_source_windows':{str(k):int(v) for k,v in sources.change_band.value_counts(sort=False).items()},
            'snow_source_windows':int((sources.snowfall_36h_cm>0).sum()),
            'forecast_delay_hours':sorted(table.forecast_issue_delay_hours.unique().tolist()),
            'checks':checks,'download_failures':download_failures,'months':by_month,
            'limitations':['Targets are drainage concern reports, not verified flooding.',
                           '36-hour windows issued every day overlap by 12 hours; positive windows are not distinct incidents.',
                           'Two weather points provide shared forecasts for 604 grids.',
                           'Forecast run publication timestamps and 24-hour report availability delay remain unaudited assumptions.',
                           'Historical ERA5 rainfall is retrospective; as-issued availability is not verified.',
                           'Observed report timestamp bounds do not establish ingestion completeness.',
                           'Rain intensity and change bands are descriptive thresholds, not hazard thresholds.',
                           'No model performance or weather-to-report causal conclusion has been established.']}


def write_metadata(table,output,start,end):
    """Attach an explicit input/target contract to the combined dataset."""
    metadata={'version':'forecast-history-36h-v2','forecast_model':'ecmwf_ifs',
              'start_date':start,'end_date':end,'rows':len(table),
              'issue_dates':int(table.prediction_time.nunique()),'grid_size_m':500,
              'timezone':'America/Vancouver','issue_hour_local':6,'horizon_hours':36,
              'history_features':FEATURES,'forecast_features':WEATHER_FEATURES,
              'target':'target_36h = at least one matching report in [prediction_time,prediction_end)',
              'report_count_column':'report_count_36h','report_categories':sorted(CATEGORIES),
              'report_publication_lag_hours_assumption':24,
              'forecast_run_rule':'same UTC date 06Z, or earlier 00Z only when the provider reports the 06Z run unavailable',
              'forecast_run_age_hours':sorted(table.forecast_issue_delay_hours.unique().tolist()),
              'forecast_publication_availability_audited':False,'report_ingestion_completeness_audited':False,
              'historical_weather':'retrospective ERA5; as-issued availability not audited',
              'weather_source_count':int(table.weather_cell_id.nunique()),
              'rain_and_showers_unit':'mm','snowfall_unit':'cm','temperature_unit':'degrees Celsius',
              'target_windows_overlap_hours':12,'precipitation_probability_included':False,
              'raw_archive_directory':'data/processed/spatial_training/forecast_archive',
              'audit_path':'docs/FORECAST_36H_DATA_AUDIT.json',
              'attribution':'Open-Meteo / ECMWF, CC BY 4.0'}
    output.with_name(output.name+'.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date',default='2024-03-14')
    parser.add_argument('--end-date',default='2026-09-27')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.workers<=3: raise SystemExit('Use 1 to 3 download workers')
    folder=ROOT/'data/processed/spatial_training/forecast_archive_monthly_v2'
    folder.mkdir(parents=True,exist_ok=True)
    mapping=pd.read_csv(ROOT/'data/processed/spatial_weather/era5_500m/grid_weather_mapping.csv')
    issues=prediction_times(args.start_date,args.end_date)
    request_lock=threading.Lock();next_request=[0.0]
    def download(issue):
        # Revalidate existing cache locally before scheduling any network request.
        preferred=run_for_issue(issue)
        for candidate in (preferred,preferred-pd.Timedelta(hours=6)):
            cache=ARCHIVE/f'{candidate:%Y-%m-%dT%H}.json'
            if not cache.exists():continue
            try:
                payload=json.loads(cache.read_text(encoding='utf-8'))
                for record in _location_records(payload):features_for_location(record,issue)
                return candidate
            except ValueError:
                continue
        # Bound initial requests to at most 80/minute; only public weather-point coordinates are sent.
        with request_lock:
            delay=max(0,next_request[0]-time.monotonic())
            if delay: time.sleep(delay)
            next_request[0]=time.monotonic()+.75
        payload,run=fetch_run(issue,mapping)
        for record in _location_records(payload):features_for_location(record,issue)
        return run
    failures=[];completed=0
    print(f'Downloading/validating {len(issues)} archived issue dates (cached runs reused)',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(download,issue):issue for issue in issues}
        for future in as_completed(futures):
            issue=futures[future]
            try:future.result()
            except Exception as error:
                failures.append({'date':issue.strftime('%Y-%m-%d'),'error':str(error)[:400]})
                print(f'FAILED {issue:%Y-%m-%d}: {error}',flush=True)
            completed+=1
            if completed%20==0 or completed==len(issues):
                print(f'Archive progress: {completed}/{len(issues)}; failures {len(failures)}',flush=True)
    unresolved=[]
    for item in failures:
        if any(marker in item['error'] for marker in ('429','500','502','503','504','timed out','Timeout','urlopen')):
            issue=prediction_times(item['date'],item['date'])[0]
            try:
                download(issue)
                print(f'Recovered temporary download failure: {item["date"]}',flush=True)
                continue
            except Exception as error:item['error']=str(error)[:400]
        unresolved.append(item)
    failures=unresolved
    (folder/'download_audit.json').write_text(json.dumps({'completed':completed,'failures':failures},indent=2),encoding='utf-8')
    paths=[]
    for first,last in month_ranges(args.start_date,args.end_date):
        output=folder/f'forecast_{first[:7]}.csv.gz'
        command=[sys.executable,str(ROOT/'src/forecast_training_36h.py'),'--start-date',first,'--end-date',last,
                 '--output',str(output),'--resume']
        print(f'Building {first} through {last}',flush=True)
        subprocess.run(command,check=True,cwd=ROOT)
        paths.append(output)
    table=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
    output=ROOT/'data/processed/spatial_training/forecast_36h_full_v2.csv.gz'
    temporary=output.with_name(output.name+'.tmp')
    table.to_csv(temporary,index=False,compression='gzip');temporary.replace(output)
    report=audit(table,mapping,args.start_date,args.end_date,failures)
    write_metadata(table,output,args.start_date,args.end_date)
    report['dataset_path']=str(output.relative_to(ROOT))
    (ROOT/'docs/FORECAST_36H_DATA_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Forecast-aware 36-hour dataset audit','',
           f'Status: **{report["status"]}**. {report["issue_dates"]:,} issue dates, {report["rows"]:,} cell-windows, '
           f'{report["positive_cell_windows"]:,} positive cell-windows ({report["positive_cell_window_rate"]:.2%}).','',
           f'Period: {args.start_date} through {args.end_date}. There are {report["source_windows"]:,} weather source-windows '
           f'across {report["weather_sources"]} source points; the grid rows are not independent weather samples.','',
           '## Data checks','',*[f'- {key}: {value}' for key,value in report['checks'].items()],
           '', '## Forecast coverage','',f'36-hour liquid precipitation summary (mm): {report["liquid_forecast_36h_mm"]}.',
           '',f'Rain intensity source-windows: {report["rain_band_source_windows"]}.',
           '',f'Weather change source-windows: {report["change_band_source_windows"]}.','',
           '## Limitations','',*[f'- {item}' for item in report['limitations']],
           '', '## Next experiment','',
           'Freeze date-based train/calibration/evaluation partitions before fitting. Purge target windows crossing boundaries. '
           'Compare history-only and forecast-aware models fitted on the same dates, then add similar-weather retrieval using '
           'only earlier events. Keep this dataset and experiment separate from the live model until evaluation supports promotion.','']
    (ROOT/'docs/FORECAST_36H_DATA_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Dataset complete: {report["rows"]:,} rows, {report["positive_cell_windows"]:,} positives; {report["status"]}',flush=True)
    if report['status']=='structural_checks_failed':raise SystemExit(1)


if __name__=='__main__':main()
