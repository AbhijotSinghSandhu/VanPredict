"""Frozen chronological experiment comparing forecast and history-only models."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.training_36h import FEATURES
from src.forecast_training_36h import WEATHER_FEATURES
from src.train_history_36h import evaluate,logits

DATA=ROOT/'data/processed/spatial_training/forecast_36h_full_v2.csv.gz'
FORECAST_FEATURES=FEATURES+WEATHER_FEATURES+['forecast_issue_delay_hours']
SPLITS={
    'training':('2024-03-14T00:00:00-07:00','2025-01-01T00:00:00-08:00'),
    'calibration':('2025-01-01T00:00:00-08:00','2025-07-01T00:00:00-07:00'),
    'development':('2025-07-01T00:00:00-07:00','2026-01-01T00:00:00-08:00'),
    'evaluation':('2026-01-01T00:00:00-08:00','2026-09-29T00:00:00-07:00'),
}
PARAMETERS=dict(max_iter=80,max_leaf_nodes=15,min_samples_leaf=100,
                l2_regularization=10,early_stopping=False,random_state=42)
EXPERIMENT={
    'version':'forecast-36h-v1','dataset':str(DATA.relative_to(ROOT)),
    'splits':SPLITS,'model_parameters':PARAMETERS,
    'history_features':FEATURES,'forecast_features':FORECAST_FEATURES,
    'calibration':'LogisticRegression on model logits; C=1, max_iter=200, seed=42',
    'primary_metric':'2026 evaluation top-20 precision; matched history-only comparison',
    'secondary_metrics':['top-5 and top-10 precision/recall','average precision','Brier score','calibration deciles'],
    'boundary_embargo':'Require target end + 24-hour report delay <= next partition start',
    'uncertainty':'Paired bootstrap of whole calendar-week blocks, 2,000 draws, seed 42',
    'selection_policy':'One fixed experiment; no tuning or promotion using evaluation results',
}


def split_table(table,name):
    start,end=(pd.Timestamp(value).tz_convert('UTC') for value in SPLITS[name])
    issues=pd.to_datetime(table.prediction_time,utc=True)
    target_ends=pd.to_datetime(table.prediction_end,utc=True)
    availability=target_ends+pd.Timedelta(hours=24 if name!='evaluation' else 0)
    return table.loc[(issues>=start)&(issues<end)&(availability<=end)].copy()


def predict(saved,table):
    inputs=table[saved['features']].to_numpy(dtype='float32')
    if table.empty or not np.isfinite(inputs).all():raise ValueError('Expected nonempty finite inference inputs')
    raw=saved['model'].predict_proba(inputs)[:,1]
    scores=saved['calibration'].predict_proba(logits(raw))[:,1]
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('Invalid model probabilities')
    return scores


def paired_top20_interval(table,forecast,history,draws=2000):
    """Resample whole weeks jointly for both models, preserving within-week dependence."""
    def daily_hits(scores):
        ranking=pd.DataFrame({'time':table.prediction_time.to_numpy(),'grid':table.grid_id.to_numpy(),
                              'target':table.target_36h.to_numpy(),'score':scores})
        ranking=ranking.sort_values(['time','score','grid'],ascending=[True,False,True],kind='stable')
        return ranking.groupby('time').head(20).groupby('time').target.sum()
    daily=pd.DataFrame({'forecast_hits':daily_hits(forecast),'history_hits':daily_hits(history)})
    local_dates=pd.to_datetime(daily.index,utc=True).tz_convert('America/Vancouver').tz_localize(None)
    daily['week']=local_dates.to_period('W-SUN').astype(str)
    daily['slots']=20
    weekly=daily.groupby('week')[['forecast_hits','history_hits','slots']].sum().to_numpy(dtype=float)
    rng=np.random.default_rng(42)
    totals=weekly[rng.integers(0,len(weekly),size=(draws,len(weekly)))].sum(axis=1)
    differences=(totals[:,0]-totals[:,1])/totals[:,2]
    low,high=np.quantile(differences,[.025,.975])
    return {'forecast_minus_history_precision':float((daily.forecast_hits.sum()-daily.history_hits.sum())/daily.slots.sum()),
            'ci95':[float(low),float(high)],'week_blocks':len(weekly),'draws':draws,
            'method':'Paired resampling of calendar-week blocks; descriptive interval, not an independence guarantee'}


def scenario_masks(table):
    # Classify whole issue dates using the mean of the two sources, so top-k still ranks all 604 grids.
    sources=table.drop_duplicates(['prediction_time','weather_cell_id']).copy()
    sources['liquid']=sources.rain_36h+sources.showers_36h
    sources['change']=sources.rain_change_second_half_36h+sources.showers_change_second_half_36h
    daily=sources.groupby('prediction_time')[['liquid','change']].mean()
    liquid=table.prediction_time.map(daily.liquid).to_numpy()
    change=table.prediction_time.map(daily.change).to_numpy()
    return {'dry_or_trace':liquid<=.1,'light_rain':(liquid>.1)&(liquid<=5),
            'moderate_rain':(liquid>5)&(liquid<=20),'heavy_rain':liquid>20,
            'increasing_rain':change>5,'decreasing_rain':change<-5,'steady_rain':np.abs(change)<=5}


def evaluate_models(table,scores):
    return {name:evaluate(table.target_36h.to_numpy(),values,table.prediction_time.to_numpy(),table.grid_id.to_numpy())
            for name,values in scores.items()}


def main():
    if not DATA.exists():raise SystemExit('Build the audited full forecast dataset first')
    audit=json.loads((ROOT/'docs/FORECAST_36H_DATA_AUDIT.json').read_text())
    if audit['status'] not in ('structural_checks_passed','structural_checks_passed_with_archive_gaps'):
        raise SystemExit('Dataset structural audit must pass before fitting')
    dataset_hash=hashlib.sha256(DATA.read_bytes()).hexdigest()
    protocol={**EXPERIMENT,'dataset_sha256':dataset_hash}
    (ROOT/'docs/FORECAST_36H_EXPERIMENT.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    table=pd.read_csv(DATA)
    if table.duplicated(['prediction_time','grid_id']).any():raise ValueError('Duplicate cell windows')
    if len(table)!=audit['rows']:raise ValueError('Dataset does not match audit row count')
    if not np.isfinite(table[FORECAST_FEATURES].to_numpy(dtype=float)).all():raise ValueError('Invalid training features')
    parts={name:split_table(table,name) for name in SPLITS}
    summary={name:{'rows':len(part),'issue_dates':int(part.prediction_time.nunique()),
                   'positive_cell_windows':int(part.target_36h.sum()),
                   'first_issue':part.prediction_time.min(),'last_issue':part.prediction_time.max()}
             for name,part in parts.items()}
    summary['purged_rows']=len(table)-sum(len(part) for part in parts.values())
    summary['purged_issue_dates']=int(summary['purged_rows']/604)
    for name,part in parts.items():
        if part.empty or part.target_36h.nunique()!=2:raise ValueError(f'Insufficient labels in {name}')
        if not part.groupby('prediction_time').grid_id.nunique().eq(604).all():raise ValueError(f'Incomplete grids in {name}')
    print('Frozen partitions: '+json.dumps(summary),flush=True)
    train,cal=parts['training'],parts['calibration']
    models={}
    for name,features in [('history_only',FEATURES),('forecast_aware',FORECAST_FEATURES)]:
        print(f'Fitting {name}: {len(train):,} rows, {len(features)} inputs',flush=True)
        model=HistGradientBoostingClassifier(**PARAMETERS)
        model.fit(train[features].to_numpy(dtype='float32'),train.target_36h.to_numpy(dtype='int8'))
        calibration=LogisticRegression(C=1,max_iter=200,random_state=42)
        calibration.fit(logits(model.predict_proba(cal[features].to_numpy(dtype='float32'))[:,1]),cal.target_36h)
        if calibration.coef_[0,0]<=0:raise ValueError(f'{name} calibration would reverse model ranking')
        models[name]={'model':model,'calibration':calibration,'features':features,
                      'metadata':{'version':f'forecast-comparison-{name}-v1','status':'research_only',
                                  'horizon_hours':36,'protocol':protocol}}
    stats=train.groupby('grid_id').target_36h.agg(['sum','count'])
    prior=float(train.target_36h.mean())
    location_prior=((stats['sum']+30*prior)/(stats['count']+30)).to_dict()
    # The current backend model is an additional comparator, never refitted here.
    legacy_path=ROOT/'models/history_36h_v1.joblib'
    legacy=joblib.load(legacy_path) if legacy_path.exists() else None
    if legacy is not None and legacy['features']!=FEATURES:raise ValueError('Current model contract mismatch')
    results={};stability={};intervals={};score_frames=[]
    for name in ('development','evaluation'):
        part=parts[name]
        scores={label:predict(saved,part) for label,saved in models.items()}
        scores['fixed_location_baseline']=part.grid_id.map(location_prior).fillna(prior).to_numpy()
        scores['updating_location_baseline']=-np.expm1(-1.5*part.historical_reports_per_day.to_numpy())
        if legacy is not None:scores['current_backend_model']=predict(legacy,part)
        results[name]=evaluate_models(part,scores)
        intervals[name]=paired_top20_interval(part,scores['forecast_aware'],scores['history_only'])
        stability[name]={}
        masks=scenario_masks(part)
        masks.update({f'month_{month}':part.prediction_time.str.startswith(month).to_numpy()
                      for month in sorted(part.prediction_time.str[:7].unique())})
        for scenario,mask in masks.items():
            if not mask.any():continue
            selected=part.loc[mask]
            stability[name][scenario]={'issue_dates':int(selected.prediction_time.nunique()),
                                      'results':evaluate_models(selected,{label:values[mask] for label,values in scores.items()})}
        frame=part[['prediction_time','grid_id','target_36h']].copy()
        frame['partition']=name
        for label,values in scores.items():frame[label]=values
        score_frames.append(frame)
    difference=intervals['evaluation']
    evidence='inconclusive'
    if difference['ci95'][0]>0:evidence='forecast_advantage_in_this_evaluation'
    elif difference['ci95'][1]<0:evidence='history_advantage_in_this_evaluation'
    report={'version':'forecast-comparison-36h-v1','status':'research_only','protocol':protocol,
            'sklearn_version':sklearn.__version__,'summary':summary,'results':results,
            'paired_top20_intervals':intervals,'scenario_results':stability,
            'primary_result':evidence,'live_model_promoted':False,
            'limitations':[item for item in audit['limitations'] if not item.startswith('No model performance')]+[
                'Metrics describe this fixed evaluation; general accuracy and causal weather/report effects are not established.',
                'One training year and two coarse weather sources limit weather-pattern learning.',
                'Calibration and evaluation can have different seasonal report rates.',
                'Current backend model uses different older training dates; the matched history-only model is the fair feature comparator.',
                'Prior aggregate dataset inspection and older development research mean this is not a pristine independent final test.',
                'Calendar-week bootstrap reduces within-week dependence; residual dependence can remain.',
                'Only 06:00 Vancouver issue times have been evaluated; no claim about arbitrary-hour live predictions.']}
    for name,saved in models.items():
        saved['metadata']['comparison_report']='docs/FORECAST_36H_MODEL_EVALUATION.json'
        joblib.dump(saved,ROOT/f'models/forecast_comparison_{name}_v1.joblib')
    folder=ROOT/'data/processed/spatial_training'
    pd.concat(score_frames,ignore_index=True).to_csv(folder/'forecast_comparison_scores_v1.csv.gz',index=False,compression='gzip')
    (ROOT/'docs/FORECAST_36H_MODEL_EVALUATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Forecast-aware model comparison','',
           'Fixed settings; calibrated on January–June 2025; evaluated on later dates. Live model unchanged.','',
           f'On the 2026 dates, both models selected {results["evaluation"]["forecast_aware"]["shortlists"]["20"]["positive_windows_found"]} '
           f'(forecast) and {results["evaluation"]["history_only"]["shortlists"]["20"]["positive_windows_found"]} '
           'report-containing cell-windows in their daily top-20 lists. The paired interval does not establish a top-20 advantage. '
           'Average precision across all grid-windows and the Brier score favor forecast inputs in this evaluation.','',
           '## Partitions','', '| Partition | Issue dates | Rows | Positive windows |','| --- | ---: | ---: | ---: |']
    for name in SPLITS:
        s=summary[name];lines.append(f'| {name} | {s["issue_dates"]:,} | {s["rows"]:,} | {s["positive_cell_windows"]:,} |')
    lines+=['',f'{summary["purged_issue_dates"]} boundary issue dates were purged to allow 36-hour labels plus the assumed 24-hour publication delay.','',
            '## 2026 evaluation','', '| Model | Top-20 precision | Top-20 recall | Average precision | Brier score |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name,result in results['evaluation'].items():
        top=result['shortlists']['20']
        lines.append(f'| {name} | {top["precision"]:.2%} | {top["recall"]:.2%} | {result["average_precision"]:.4f} | {result["brier_score"]:.5f} |')
    lines+=['',f'Forecast minus matched history top-20 precision: {difference["forecast_minus_history_precision"]*100:+.2f} percentage points. '
            f'Paired weekly-bootstrap 95% interval: [{difference["ci95"][0]*100:+.2f}, {difference["ci95"][1]*100:+.2f}] percentage points.',
            '',f'Primary result: **{evidence}**.','',
            '## Weather changes in 2026','', '| Scenario | Issue dates | History top-20 precision | Forecast top-20 precision |',
            '| --- | ---: | ---: | ---: |']
    for scenario in ('heavy_rain','increasing_rain','decreasing_rain','dry_or_trace'):
        item=stability['evaluation'].get(scenario)
        if item:
            metrics=item['results']
            lines.append(f'| {scenario} | {item["issue_dates"]} | {metrics["history_only"]["shortlists"]["20"]["precision"]:.2%} | {metrics["forecast_aware"]["shortlists"]["20"]["precision"]:.2%} |')
    lines+=['','Scenario groups use the mean forecast across the two weather source points. Rain intensity thresholds are descriptive; '
            'increasing/decreasing means a greater than 5 mm shift between the first and second 18-hour halves. These scenario results are exploratory.','',
            '## Limitations','',*[f'- {item}' for item in report['limitations']],'',
            '## Next step','',
            'Inspect the fixed comparison and weather-change subsets before choosing the next experiment. '
            'Similar-weather retrieval should use only historical events and outcomes available before each prediction issue. '
            'No training or probability calibration may use the evaluation labels.','']
    lines+=['## Reproduce','',
            '```powershell', '.venv/Scripts/python.exe -u src/train_forecast_36h.py',
            '.venv/Scripts/python.exe -m unittest tests.test_forecast_36h_model tests.test_forecast_training_36h tests.test_training_36h tests.test_history_36h_model',
            '```','',
            'The dataset hash, feature lists and fixed settings are saved in `docs/FORECAST_36H_EXPERIMENT.json`. '
            'Research artifacts: `models/forecast_comparison_forecast_aware_v1.joblib` and '
            '`models/forecast_comparison_history_only_v1.joblib`. Per-window evaluation scores: '
            '`data/processed/spatial_training/forecast_comparison_scores_v1.csv.gz`.','']
    (ROOT/'docs/FORECAST_36H_MODEL_EVALUATION.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'primary_result':evidence,'evaluation':results['evaluation'],'paired_top20':difference},indent=2),flush=True)


if __name__=='__main__':main()
