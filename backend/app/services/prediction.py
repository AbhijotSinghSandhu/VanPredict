"""Inference for the saved experimental history-only model; never fit at startup."""
from functools import lru_cache
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.training_36h import FEATURES, build_table, TIMEZONE


@lru_cache(maxsize=1)
def artifact():
    saved = joblib.load(ROOT/'models/history_36h_v1.joblib')
    if saved['features'] != FEATURES or saved['metadata']['horizon_hours'] != 36:
        raise ValueError('Model feature contract mismatch')
    return saved


@lru_cache(maxsize=1)
def sources():
    folder = ROOT/'data/processed/spatial_weather/era5_500m'
    mapping = pd.read_csv(folder/'grid_weather_mapping.csv')
    reports = pd.read_csv(ROOT/'data/processed/311_gridded.csv')
    weather = pd.read_csv(folder/'weather_hourly_features.csv.gz',usecols=['weather_cell_id','datetime','rainfall_mm'])
    weather['datetime'] = pd.to_datetime(weather.datetime,utc=True)
    return mapping,reports,weather


def score(frame,issue,top_k,input_source):
    saved = artifact()
    x = frame[FEATURES].to_numpy(dtype='float32')
    if frame.empty or frame.grid_id.duplicated().any() or not np.isfinite(x).all():
        raise ValueError('Expected unique cells with finite model inputs')
    raw = np.clip(saved['model'].predict_proba(x)[:,1],1e-7,1-1e-7)
    scores = saved['calibration'].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Invalid model outputs')
    ranked = pd.DataFrame({'cell_id':frame.grid_id.to_numpy(),'report_probability':scores})
    ranked = ranked.sort_values(['report_probability','cell_id'],ascending=[False,True])
    areas = [dict(cell_id=row.cell_id,report_probability=float(row.report_probability),rank=i+1)
             for i,row in enumerate(ranked.itertuples(index=False))]
    now = pd.Timestamp.now(tz='UTC')
    return dict(status='experimental',model_version=saved['metadata']['version'],grid_size_m=500,
                target='At least one matching drainage report, not confirmed flooding',
                prediction_start=issue.isoformat(),prediction_end=(issue+pd.Timedelta(hours=36)).isoformat(),
                generated_at=now.isoformat(),input_source=input_source,
                input_age_hours=round((now-issue).total_seconds()/3600,2),
                uses_forecast_rainfall=False,feature_names=FEATURES,areas=areas,top_areas=areas[:top_k],
                warnings=['Experimental probabilities; accuracy and availability assumptions remain under review.',
                          'Forecast rainfall does not influence this history-only model.',
                          'Model trained at 06:00 Vancouver time; arbitrary-hour operation is unsupported.'])


def prepared_prediction(issue,top_k):
    mapping,reports,weather = sources()
    latest = weather.groupby('weather_cell_id').datetime.max().min()
    if issue is None:
        local = latest.tz_convert(TIMEZONE)
        issue = local.normalize()+pd.DateOffset(hours=6)
        if issue > latest:
            issue = (local.normalize()-pd.DateOffset(days=1))+pd.DateOffset(hours=6)
        issue = issue.tz_convert('UTC')
    if issue > latest:
        raise ValueError('Requested time exceeds prepared weather coverage; current live historical inputs are not connected')
    frame = build_table(mapping,reports,weather,[issue],
                        '2009-06-24T00:00:00-07:00','2026-10-03T00:00:00-07:00',include_targets=False)
    result = score(frame,issue,top_k,'prepared_historical_data')
    result['weather_coverage_end'] = latest.isoformat()
    result['history_cutoff'] = frame.history_cutoff.iloc[0]
    result['warnings'].append('Prepared data is a historical snapshot, not a current live prediction. Report publication lag assumes 24 hours.')
    return result
