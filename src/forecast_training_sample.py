"""Build a small run-specific forecast dataset; no model or API changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training_36h import build_table, prediction_times, utc, FEATURES

MODEL='ecmwf_ifs'
FORECAST_FEATURES=[*[f'forecast_rain_{h}h' for h in (6,12,24,36)],
                   'forecast_peak_1h','forecast_peak_3h','forecast_wet_hours',
                   'forecast_longest_wet_spell','forecast_second18_minus_first18']


def choose_run(issue,availability_delay_hours=8):
    if availability_delay_hours<6:
        raise ValueError('Use a conservative delay of at least six hours')
    return (utc(issue)-pd.Timedelta(hours=availability_delay_hours)).floor('6h')


def forecast_features(payload,issue,run,availability_delay_hours=8):
    issue,run=utc(issue),utc(run)
    if run+pd.Timedelta(hours=availability_delay_hours)>issue:
        raise ValueError('Forecast run would be unavailable at prediction time')
    if payload.get('hourly_units',{}).get('time')!='unixtime' or any(payload.get('hourly_units',{}).get(v)!='mm' for v in ('rain','showers')):
        raise ValueError('Unexpected forecast units')
    hourly=payload['hourly']
    times=pd.to_datetime(hourly['time'],unit='s',utc=True)
    rain=np.asarray(hourly['rain'],dtype=float)+np.asarray(hourly['showers'],dtype=float)
    if len(times)!=len(rain) or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Invalid forecast timestamps')
    # End-of-interval rainfall: (issue, issue+36h] covers the future window.
    mask=(times>issue)&(times<=issue+pd.Timedelta(hours=36))
    selected=times[mask];values=rain[mask]
    expected=pd.date_range(issue+pd.Timedelta(hours=1),periods=36,freq='h')
    if not selected.equals(expected) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Incomplete or invalid 36-hour forecast')
    longest=current=0
    for value in values:
        current=current+1 if value>0 else 0
        longest=max(longest,current)
    result={f'forecast_rain_{h}h':float(values[:h].sum()) for h in (6,12,24,36)}
    result.update(forecast_peak_1h=float(values.max()),
                  forecast_peak_3h=float(np.convolve(values,np.ones(3),'valid').max()),
                  forecast_wet_hours=int((values>0).sum()),forecast_longest_wet_spell=longest,
                  forecast_second18_minus_first18=float(values[18:].sum()-values[:18].sum()))
    return result


def fetch_run(run,folder):
    params=dict(latitude=49.2827,longitude=-123.1207,hourly='rain,showers',models=MODEL,
                run=run.strftime('%Y-%m-%dT%H:%M'),forecast_days=3,timezone='GMT',timeformat='unixtime')
    url='https://single-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(params)
    path=folder/f'{MODEL}_{run:%Y%m%dT%H%M}.json'
    if path.exists():
        envelope=json.loads(path.read_text(encoding='utf-8'))
        if envelope['url']!=url or envelope['run']!=run.isoformat():
            raise ValueError('Cached forecast provenance mismatch')
        return envelope
    with urllib.request.urlopen(url,timeout=45) as response:
        raw=response.read()
    envelope=dict(url=url,run=run.isoformat(),retrieved_at=pd.Timestamp.now(tz='UTC').isoformat(),
                  sha256=hashlib.sha256(raw).hexdigest(),payload=json.loads(raw))
    folder.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(envelope),encoding='utf-8');temporary.replace(path)
    return envelope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date',default='2026-09-20')
    parser.add_argument('--end-date',default='2026-09-22')
    args=parser.parse_args()
    times=prediction_times(args.start_date,args.end_date)
    if len(times)>7:
        raise ValueError('This verification sample is limited to seven days')
    # Avoid the earlier hindcast archive for this operational-run verification sample.
    if times.min()<pd.Timestamp('2026-05-13T00:00Z'):
        raise ValueError('This sample requires dates after the May 2026 model transition')
    dataset=ROOT/'data/processed/spatial_weather/era5_500m'
    table=build_table(pd.read_csv(dataset/'grid_weather_mapping.csv'),
                      pd.read_csv(ROOT/'data/processed/311_gridded.csv'),
                      pd.read_csv(dataset/'weather_hourly_features.csv.gz',usecols=['weather_cell_id','datetime','rainfall_mm']),
                      times,'2009-06-24T00:00:00-07:00','2026-10-03T00:00:00-07:00')
    records=[]
    for issue in times:
        run=choose_run(issue)
        envelope=fetch_run(run,ROOT/'data/weather/forecast_runs_sample')
        features=forecast_features(envelope['payload'],issue,run)
        records.append(dict(prediction_time=issue.isoformat(),forecast_run=run.isoformat(),
                            assumed_available_at=(run+pd.Timedelta(hours=8)).isoformat(),
                            forecast_retrieved_at=envelope['retrieved_at'],**features))
        print(f'{issue}: run {run}, rain36h={features["forecast_rain_36h"]:.2f} mm',flush=True)
    table=table.merge(pd.DataFrame(records),on='prediction_time',validate='many_to_one')
    if table[FORECAST_FEATURES].isna().any().any():
        raise ValueError('Missing merged forecast inputs')
    output=ROOT/'data/processed/spatial_training/forecast_36h_sample.csv.gz'
    temporary=output.with_name(output.name+'.tmp');table.to_csv(temporary,index=False,compression='gzip');temporary.replace(output)
    metadata=dict(rows=len(table),positive_windows=int(table.target_36h.sum()),model=MODEL,
                  start_date=args.start_date,end_date=args.end_date,features=FEATURES+FORECAST_FEATURES,
                  forecast_location=[49.2827,-123.1207],assumed_publication_delay_hours=8,
                  availability_audited=False,report_publication_delay_hours=24,
                  limitations=['One forecast point shared by every grid cell; not 500m weather detail.',
                               'Run availability uses a conservative assumption, not archived publication timestamps.',
                               'Past weather is retrospective ERA5; report availability and completeness are unaudited.',
                               'Small pipeline sample, not model validation or evidence of skill.'],
                  sources=['https://open-meteo.com/en/docs/single-runs-api'],
                  attribution='Open-Meteo / ECMWF, CC BY 4.0')
    output.with_name(output.name+'.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Saved {len(table)} rows, {table.target_36h.sum()} positive windows: {output}')


if __name__=='__main__':
    main()
