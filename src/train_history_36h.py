"""Fixed history-only 36-hour development experiment, never live inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training_36h import FEATURES, build_table, prediction_times, partition

SPLITS = {'training': (2019, 2021), 'calibration': (2022, 2022), 'development': (2023, 2025)}


def logits(scores):
    p = np.clip(scores, 1e-7, 1-1e-7)
    return np.log(p/(1-p)).reshape(-1, 1)


def evaluate(y, scores, times, grids):
    if not np.isfinite(scores).all() or ((scores < 0)|(scores > 1)).any():
        raise ValueError('Invalid probabilities')
    frame = pd.DataFrame(dict(y=y, score=scores, time=times, grid=grids))
    ranked = frame.sort_values(['time','score','grid'], ascending=[True,False,True], kind='stable')
    shortlists = {}
    for k in (5,10,20):
        top = ranked.groupby('time').head(k)
        shortlists[str(k)] = {'precision':float(top.y.mean()),
                              'recall':float(top.y.sum()/y.sum()) if y.sum() else 0.,
                              'positive_windows_found':int(top.y.sum())}
    bins = []
    sorted_frame = frame.sort_values('score',kind='stable')
    for indices in np.array_split(np.arange(len(frame)), min(10,len(frame))):
        group = sorted_frame.iloc[indices]
        bins.append({'rows':len(group),'mean_probability':float(group.score.mean()),
                     'observed_rate':float(group.y.mean())})
    return {'rows':len(y), 'positive_windows':int(y.sum()), 'prevalence':float(y.mean()),
            'average_precision':float(average_precision_score(y,scores)),
            'roc_auc':float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
            'brier_score':float(brier_score_loss(y,scores)),
            'mean_probability':float(scores.mean()), 'shortlists':shortlists,
            'calibration_deciles':bins}


def load_part(name, mapping, reports, weather):
    first, last = SPLITS[name]
    frames = []
    for year in range(first,last+1):
        print(f'Building {name}: {year}',flush=True)
        # Year-end windows crossing a split boundary are removed below.
        table = build_table(mapping,reports,weather,prediction_times(f'{year}-01-01',f'{year}-12-31'),
                            '2009-06-24T00:00:00-07:00','2026-10-03T00:00:00-07:00')
        table = partition(table,f'{first}-01-01T00:00:00-08:00',f'{last+1}-01-01T00:00:00-08:00')
        frames.append(table)
    table = pd.concat(frames,ignore_index=True)
    if table.duplicated(['grid_id','prediction_time']).any():
        raise ValueError('Duplicate cell windows')
    return table


def main():
    dataset = ROOT/'data/processed/spatial_weather/era5_500m'
    mapping = pd.read_csv(dataset/'grid_weather_mapping.csv')
    reports = pd.read_csv(ROOT/'data/processed/311_gridded.csv')
    weather = pd.read_csv(dataset/'weather_hourly_features.csv.gz',usecols=['weather_cell_id','datetime','rainfall_mm'])
    train = load_part('training',mapping,reports,weather)
    y = train.target_36h.to_numpy(dtype='int8')
    stats = train.groupby('grid_id').target_36h.agg(['sum','count'])
    prior = float(y.mean())
    baseline = ((stats['sum']+30*prior)/(stats['count']+30)).to_dict()
    model = HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,min_samples_leaf=100,
                                          l2_regularization=10,early_stopping=False,random_state=42)
    print(f'Fitting {len(train):,} training rows',flush=True)
    model.fit(train[FEATURES].to_numpy(dtype='float32'),y)
    summary = {'training_rows':len(train),'training_positive_windows':int(y.sum())}
    del train,y
    cal = load_part('calibration',mapping,reports,weather)
    calibrator = LogisticRegression(C=1,max_iter=200,random_state=42)
    calibrator.fit(logits(model.predict_proba(cal[FEATURES].to_numpy(dtype='float32'))[:,1]),cal.target_36h)
    summary['calibration_rows'] = len(cal)
    del cal
    dev = load_part('development',mapping,reports,weather)
    y = dev.target_36h.to_numpy(dtype='int8')
    scores = calibrator.predict_proba(logits(model.predict_proba(dev[FEATURES].to_numpy(dtype='float32'))[:,1]))[:,1]
    base = dev.grid_id.map(baseline).fillna(prior).to_numpy()
    # Updating location-only rate is a second comparator with the same past-only cutoff.
    dynamic = -np.expm1(-1.5*dev.historical_reports_per_day.to_numpy())
    times, grids = dev.prediction_time.to_numpy(),dev.grid_id.to_numpy()
    results = {label:evaluate(y,p,times,grids) for label,p in
               [('model',scores),('fixed_location_baseline',base),('updating_location_baseline',dynamic)]}
    stability = {}
    for year in (2023,2024,2025):
        for season, months in {'winter':[12,1,2],'spring':[3,4,5],'summer':[6,7,8],'autumn':[9,10,11]}.items():
            mask = dev.prediction_time.str.startswith(str(year)).to_numpy() & dev.month.isin(months).to_numpy()
            stability[f'{year}_{season}'] = {label:evaluate(y[mask],p[mask],times[mask],grids[mask]) for label,p in
                                           [('model',scores),('fixed_location_baseline',base),('updating_location_baseline',dynamic)]}
    report = dict(version='history-36h-v1',status='development_research',horizon_hours=36,
                  grid_size_m=500,issue_hour_local=6,features=FEATURES,splits=SPLITS,
                  summary=summary,results=results,seasonal_results=stability,sklearn_version=sklearn.__version__,
                  assumptions={'coverage_start':'2009-06-24T00:00:00-07:00','coverage_end':'2026-10-03T00:00:00-07:00','publication_lag_hours':24},
                  limitations=['Predicts reported concerns, not confirmed flooding.',
                               'Development periods overlap previously inspected research; no fresh final holdout.',
                               'ERA5 is retrospective, not as-issued weather; coverage and publication lag are unaudited.',
                               'Daily 06:00 windows overlap and are correlated; metrics are descriptive, not uncertainty intervals.',
                               'No forecast inputs, arbitrary-hour validation, or live inference.'])
    (ROOT/'docs/history_36h_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    joblib.dump(dict(model=model,calibration=calibrator,features=FEATURES,baseline=baseline,prior=prior,metadata=report),ROOT/'models/history_36h_v1.joblib')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
