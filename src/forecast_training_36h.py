"""Build forecast-aware 36h training rows from archived ECMWF runs."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
ARCHIVE = ROOT/'data/processed/spatial_training/forecast_archive'
OUTPUT = ROOT/'data/processed/spatial_training/forecast_36h.csv.gz'
MODEL = 'ecmwf_ifs'
HOURLY = ['rain','showers','snowfall','temperature_2m']
WEATHER_FEATURES = [f'{kind}_{window}h' for kind in ('rain','showers') for window in (6,12,24,36)] + [
    'rain_peak_hourly_36h','rain_peak_3h_36h','rain_wet_hours_36h',
    'rain_change_second_half_36h','showers_peak_hourly_36h','showers_peak_3h_36h',
    'showers_wet_hours_36h','showers_change_second_half_36h','snowfall_36h_cm',
    'minimum_temperature_36h_c']


def _location_records(payload):
    return payload if isinstance(payload,list) else [payload]


def features_for_location(payload, issue):
    """Summarize 36 interval-ending forecast hours strictly after issue time."""
    if payload.get('error'):
        raise ValueError(payload.get('reason','Forecast archive error'))
    hourly=payload.get('hourly',{})
    units=payload.get('hourly_units',{})
    if units.get('rain')!='mm' or units.get('showers')!='mm' or units.get('snowfall')!='cm':
        raise ValueError('Unexpected archived forecast units')
    stamps=pd.to_datetime(hourly.get('time',[]),unit='s',utc=True,errors='coerce')
    if stamps.hasnans or stamps.has_duplicates:
        raise ValueError('Invalid archived forecast timestamps')
    frame=pd.DataFrame({'time':stamps})
    for name in HOURLY:
        values=hourly.get(name)
        if values is None or len(values)!=len(frame):
            raise ValueError(f'Missing forecast variable: {name}')
        frame[name]=pd.to_numeric(values,errors='coerce')
    issue=pd.Timestamp(issue)
    if issue.tzinfo is None: raise ValueError('Prediction time requires a timezone')
    issue=issue.tz_convert('UTC'); end=issue+pd.Timedelta(hours=36)
    # Hourly precipitation at t is the preceding interval: include (issue,end].
    selected=frame.loc[(frame.time>issue)&(frame.time<=end)].sort_values('time')
    expected=pd.date_range(issue+pd.Timedelta(hours=1),end,freq='h')
    if not pd.DatetimeIndex(selected.time).as_unit('ns').equals(expected.as_unit('ns')):
        raise ValueError('Archived run does not cover a continuous 36-hour window')
    values=selected[HOURLY]
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError('Archived run has missing values inside the prediction window')
    if (values[['rain','showers','snowfall']]<0).any().any():
        raise ValueError('Archived precipitation cannot be negative')
    out={}
    for kind in ('rain','showers'):
        series=values[kind].to_numpy(dtype=float)
        for hours in (6,12,24,36): out[f'{kind}_{hours}h']=float(series[:hours].sum())
        out[f'{kind}_peak_hourly_36h']=float(series.max())
        out[f'{kind}_peak_3h_36h']=float(pd.Series(series).rolling(3).sum().max())
        out[f'{kind}_wet_hours_36h']=int((series>0).sum())
        out[f'{kind}_change_second_half_36h']=float(series[18:].sum()-series[:18].sum())
    out['rain_peak_hourly_36h']=out.pop('rain_peak_hourly_36h')
    out['rain_peak_3h_36h']=out.pop('rain_peak_3h_36h')
    out['rain_wet_hours_36h']=out.pop('rain_wet_hours_36h')
    out['rain_change_second_half_36h']=out.pop('rain_change_second_half_36h')
    out['showers_peak_hourly_36h']=out.pop('showers_peak_hourly_36h')
    out['showers_peak_3h_36h']=out.pop('showers_peak_3h_36h')
    out['showers_wet_hours_36h']=out.pop('showers_wet_hours_36h')
    out['showers_change_second_half_36h']=out.pop('showers_change_second_half_36h')
    out['snowfall_36h_cm']=float(values.snowfall.sum())
    out['minimum_temperature_36h_c']=float(values.temperature_2m.min())
    return out


def run_for_issue(issue):
    issue=pd.Timestamp(issue)
    if issue.tzinfo is None: raise ValueError('Prediction time requires timezone')
    issue=issue.tz_convert('UTC')
    local=issue.tz_convert('America/Vancouver')
    if (local.hour,local.minute,local.second,local.microsecond)!=(6,0,0,0):
        raise ValueError('Archive builder currently supports 06:00 Vancouver windows')
    run=issue.floor('D')+pd.Timedelta(hours=6)
    if run>issue or (issue-run)>pd.Timedelta(hours=8):
        raise ValueError('No conservative ECMWF run available for this local issue time')
    return run


class RunUnavailable(ValueError):
    """The provider explicitly does not archive this initialization cycle."""


def fetch_run(issue, mapping):
    preferred=run_for_issue(issue)
    fallback=preferred-pd.Timedelta(hours=6)
    # A fallback cached by an earlier unavailable-cycle response is reproducible
    # input; avoid repeating the same unavailable request in every monthly build.
    if not (ARCHIVE/f'{preferred:%Y-%m-%dT%H}.json').exists() and (ARCHIVE/f'{fallback:%Y-%m-%dT%H}.json').exists():
        return _fetch_run_at(fallback,mapping)
    try:
        return _fetch_run_at(preferred,mapping)
    except RunUnavailable:
        # Earlier 2024 archive dates have 00/12Z but not 06/18Z cycles.
        # Fall back only to an earlier run, never to a later forecast.
        return _fetch_run_at(fallback,mapping)


def _fetch_run_at(run, mapping):
    """Download both existing forecast source points once per issue date; cache raw JSON."""
    cache=ARCHIVE/f'{run:%Y-%m-%dT%H}.json'
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8')),run
    points=(mapping[['weather_cell_id','weather_latitude','weather_longitude']]
            .drop_duplicates().sort_values('weather_cell_id'))
    if len(points)!=2: raise ValueError('Expected the two configured Vancouver weather source points')
    params={'latitude':','.join(f'{v:.4f}' for v in points.weather_latitude),
            'longitude':','.join(f'{v:.4f}' for v in points.weather_longitude),
            'hourly':','.join(HOURLY),'forecast_hours':60,'run':run.strftime('%Y-%m-%dT%H:%M'),
            'models':MODEL,'timezone':'GMT','timeformat':'unixtime'}
    url='https://single-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(params)
    request=urllib.request.Request(url,headers={'User-Agent':'VanPredict research prototype'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request,timeout=45) as response: payload=json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            detail=error.read().decode(errors='replace')
            if error.code==400 and 'requested model run is not available' in detail:
                raise RunUnavailable(f'Forecast archive run unavailable: {run.isoformat()}') from error
            if error.code not in (429,500,502,503,504) or attempt==3:
                raise ValueError(f'Forecast archive request failed ({error.code}): {detail[:250]}') from error
            time.sleep(2**attempt)
        except (TimeoutError,urllib.error.URLError):
            if attempt==3: raise
            time.sleep(2**attempt)
    records=_location_records(payload)
    if len(records)!=len(points): raise ValueError('Forecast archive returned an unexpected location count')
    for entry in records:
        if entry.get('error'): raise ValueError(entry.get('reason','Forecast run unavailable'))
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload),encoding='utf-8')
    temporary.replace(cache)
    return payload,run


def build_day(day, mapping, reports, weather, history=None, coverage_start=None, coverage_end=None):
    from src.training_36h import build_table,prediction_times
    issue=prediction_times(day,day)[0]
    payload,run=fetch_run(issue,mapping)
    records=_location_records(payload)
    points=(mapping[['weather_cell_id','weather_latitude','weather_longitude']]
            .drop_duplicates().sort_values('weather_cell_id').reset_index(drop=True))
    per_weather={}
    for (_,source),entry in zip(points.iterrows(),records):
        per_weather[source.weather_cell_id]=features_for_location(entry,issue)
    if history is None:
        report_times=pd.to_datetime(reports.service_request_open_timestamp,format='mixed',utc=True)
        coverage_start=coverage_start or report_times.min()
        coverage_end=coverage_end or report_times.max()
        frame=build_table(mapping,reports,weather,[issue],coverage_start.isoformat(),coverage_end.isoformat())
    else:
        frame=history.loc[history.prediction_time.eq(issue.isoformat())].copy()
        if len(frame)!=len(mapping): raise ValueError('Incomplete precomputed history window')
    forecast=pd.DataFrame([{'grid_id':row.grid_id,'weather_cell_id':row.weather_cell_id,
                           **per_weather[row.weather_cell_id]} for row in mapping.itertuples()])
    frame=frame.merge(forecast,on='grid_id',validate='one_to_one')
    frame['forecast_model']=MODEL
    frame['forecast_run']=run.isoformat()
    frame['forecast_issue_delay_hours']=(issue-run).total_seconds()/3600
    return frame


def audit_table(table, start_date, end_date, skipped, failures, mapping):
    """Summarize date coverage and shared-source feature duplication."""
    date_rows=[]
    if not table.empty:
        for date, group in table.groupby(table.prediction_time.str[:10], sort=True):
            date_rows.append({'date':date,'rows':len(group),
                              'positive_cells':int(group.target_36h.sum()),
                              'positive_grids':int(group.loc[group.target_36h.eq(1),'grid_id'].nunique()),
                              'forecast_run':group.forecast_run.iloc[0],
                              'rain_36h_source_values':int(group.rain_36h.nunique()),
                              'showers_36h_source_values':int(group.showers_36h.nunique())})
    return {'version':'forecast-history-36h-audit-v1','requested_start':start_date,
            'requested_end':end_date,'supported_start':'2024-03-14',
            'grid_cells':int(mapping.grid_id.nunique()),'weather_source_points':int(mapping.weather_cell_id.nunique()),
            'days_requested':len(pd.date_range(start_date,end_date,freq='D')),
            'days_built':len(date_rows),'days_skipped':skipped,'failed_days':failures,
            'rows':len(table),'positive_cell_windows':int(table.target_36h.sum()) if not table.empty else 0,
            'positive_days':sum(1 for row in date_rows if row['positive_cells']>0),
            'dates':date_rows,
            'caveats':['Rows sharing a weather_cell_id receive the same forecast values.',
                       'Daily 36-hour windows overlap by 12 hours.',
                       'A run initialization time is not proof that its forecast was publicly available by issue time.',
                       'Archive coverage and report coverage are different constraints.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date',required=True)
    parser.add_argument('--end-date',required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--coverage-start',default=None,
                        help='Optional explicit coverage start; defaults to earliest actual report timestamp')
    parser.add_argument('--coverage-end',default=None,
                        help='Exclusive end of the report archive; forecast windows must end before this time')
    parser.add_argument('--resume',action='store_true',help='Keep existing output rows and build only missing dates')
    parser.add_argument('--audit-output',type=Path,help='Optional JSON coverage audit path')
    args=parser.parse_args()
    if pd.Timestamp(args.start_date)<pd.Timestamp('2024-03-14'):
        raise SystemExit('ECMWF IFS archived single runs start 2024-03-14')
    folder=ROOT/'data/processed/spatial_weather/era5_500m'
    mapping=pd.read_csv(folder/'grid_weather_mapping.csv')
    reports=pd.read_csv(ROOT/'data/processed/311_gridded.csv')
    weather=pd.read_csv(folder/'weather_hourly_features.csv.gz',usecols=['weather_cell_id','datetime','rainfall_mm'])
    report_timestamps=pd.to_datetime(reports.service_request_open_timestamp,format='mixed',utc=True)
    weather_timestamps=pd.to_datetime(weather.datetime,utc=True)
    # Event min/max provide conservative observed bounds, not proof of complete ingestion.
    coverage_start=pd.Timestamp(args.coverage_start) if args.coverage_start else report_timestamps.min()
    coverage_end=pd.Timestamp(args.coverage_end) if args.coverage_end else report_timestamps.max()
    if coverage_start.tzinfo is None or coverage_end.tzinfo is None:
        raise SystemExit('Coverage timestamps must include timezone')
    days=pd.date_range(args.start_date,args.end_date,freq='D')
    if days.empty: raise SystemExit('Start date must precede end date')
    outputs=[]; skipped=[]; failures=[]
    existing_dates=set()
    if args.resume and args.output.exists():
        existing=pd.read_csv(args.output,compression='infer')
        # Resume only complete dates in this requested range; reject stale schemas.
        required={'weather_cell_id',*WEATHER_FEATURES}
        if not required.issubset(existing.columns):
            raise SystemExit('Existing output has an older schema; use a new output path')
        expected_grids=set(mapping.grid_id)
        for stamp,group in existing.groupby('prediction_time'):
            if len(group)!=len(mapping) or set(group.grid_id)!=expected_grids:
                raise SystemExit(f'Incomplete checkpoint: {stamp}')
        requested_dates=set(days.strftime('%Y-%m-%d'))
        existing=existing.loc[existing.prediction_time.str[:10].isin(requested_dates)]
        existing_dates=set(existing.prediction_time.str[:10])
        outputs.append(existing)
    from src.training_36h import build_table,prediction_times
    valid_days=[]
    for day in days:
        issue=prediction_times(day,day)[0]
        if issue-pd.Timedelta(hours=744)<coverage_start.tz_convert('UTC'):
            skipped.append({'date':day.strftime('%Y-%m-%d'),'reason':'Insufficient 720-hour history before the 24-hour publication cutoff'})
        elif issue+pd.Timedelta(hours=36)>coverage_end.tz_convert('UTC'):
            skipped.append({'date':day.strftime('%Y-%m-%d'),'reason':'Target exceeds observed report timestamp bounds'})
        elif issue>weather_timestamps.max():
            skipped.append({'date':day.strftime('%Y-%m-%d'),'reason':'Past rainfall does not reach issue time'})
        else:
            valid_days.append(day)
    missing=[day for day in valid_days if day.strftime('%Y-%m-%d') not in existing_dates]
    history=build_table(mapping,reports,weather,[prediction_times(day,day)[0] for day in missing],
                        coverage_start.isoformat(),coverage_end.isoformat()) if missing else pd.DataFrame()
    for day in valid_days:
        day_string=day.strftime('%Y-%m-%d')
        if day_string in existing_dates: continue
        try:
            outputs.append(build_day(day,mapping,reports,weather,history,coverage_start,coverage_end))
        except Exception as error:
            failures.append({'date':day_string,'error':str(error)[:400]})
            print(f'FAILED {day_string}: {error}',file=sys.stderr)
        # Persist a checkpoint after each successful day so interruptions are resumable.
        if outputs:
            checkpoint=pd.concat(outputs,ignore_index=True).drop_duplicates(['grid_id','prediction_time'])
            args.output.parent.mkdir(parents=True,exist_ok=True)
            temporary=args.output.with_name(args.output.name+'.tmp')
            checkpoint.to_csv(temporary,index=False,compression='gzip');temporary.replace(args.output)
            outputs=[checkpoint]
    table=pd.concat(outputs,ignore_index=True).drop_duplicates(['grid_id','prediction_time']) if outputs else pd.DataFrame()
    if table.empty: raise SystemExit('No rows were built; check requested dates and archive coverage.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name(args.output.name+'.tmp')
    table.to_csv(temporary,index=False,compression='gzip');temporary.replace(args.output)
    metadata={'version':'forecast-history-36h-v1','model':MODEL,'issue_hour':'06:00 America/Vancouver',
              'horizon_hours':36,'feature_names':WEATHER_FEATURES,'forecast_run_rule':'same UTC date 06Z; earlier 00Z fallback only if 06Z is not archived',
              'forecast_variables':HOURLY,'forecast_probability_available':False,'locations':2,
              'rows':len(table),'days':table.prediction_time.nunique(),'forecast_start':args.start_date,'forecast_end':args.end_date,
              'report_weather_coverage_start':coverage_start.isoformat(),
              'report_coverage_end_exclusive':coverage_end.isoformat(),
              'coverage_completeness_verified':False,
              'historical_weather':'retrospective ERA5; as-issued availability not verified',
              'assumptions':['Archived run initialization is used as an approximate issue-availability proxy; exact publication timestamps are not archived.',
                             'Weather represents two coarse source points, not independent 500 m forecasts.',
                             'Prediction windows overlap by 12 hours for daily 06:00 issues.']}
    args.output.with_name(args.output.name+'.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    audit=audit_table(table,args.start_date,args.end_date,skipped,failures,mapping)
    audit_path=args.audit_output or args.output.with_name(args.output.name+'.audit.json')
    audit_path.parent.mkdir(parents=True,exist_ok=True)
    audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'Saved {len(table):,} cell-window rows across {audit["days_built"]} dates; '
          f'{audit["positive_cell_windows"]:,} positive cell-windows; '
          f'{len(skipped)} skipped, {len(failures)} failed. Audit: {audit_path}')


if __name__=='__main__': main()
