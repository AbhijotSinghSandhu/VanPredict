"""Small validation-selected experiment with more seasons and weather transitions."""
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
from src.train_forecast_36h import DATA,FORECAST_FEATURES,PARAMETERS,predict,paired_top20_interval,scenario_masks,evaluate_models
from src.train_history_36h import logits
from src.forecast_transitions_36h import TRANSITION_FEATURES,INTERACTION_FEATURES,add_transitions

ENRICHED_FEATURES=FORECAST_FEATURES+TRANSITION_FEATURES+INTERACTION_FEATURES
SPLITS={
    'training':('2024-03-14T00:00:00-07:00','2025-07-01T00:00:00-07:00'),
    'calibration':('2025-07-01T00:00:00-07:00','2025-10-01T00:00:00-07:00'),
    'validation':('2025-10-01T00:00:00-07:00','2026-01-01T00:00:00-08:00'),
    'evaluation':('2026-01-01T00:00:00-08:00','2026-09-29T00:00:00-07:00'),
}
CANDIDATES={
    'more_seasons':{'features':FORECAST_FEATURES,'parameters':PARAMETERS},
    'transitions_80':{'features':ENRICHED_FEATURES,'parameters':PARAMETERS},
    'transitions_160':{'features':ENRICHED_FEATURES,'parameters':{**PARAMETERS,'max_iter':160}},
    'transitions_regularized':{'features':ENRICHED_FEATURES,
                              'parameters':{**PARAMETERS,'max_iter':160,'max_leaf_nodes':31,
                                            'min_samples_leaf':200,'l2_regularization':20,'class_weight':'balanced'}},
}


def partition(table,name):
    start,end=[pd.Timestamp(value).tz_convert('UTC') for value in SPLITS[name]]
    issues=pd.to_datetime(table.prediction_time,utc=True)
    ends=pd.to_datetime(table.prediction_end,utc=True)
    delay=pd.Timedelta(hours=0 if name=='evaluation' else 24)
    return table.loc[(issues>=start)&(issues<end)&(ends+delay<=end)].copy()


def selection_key(result):
    return (result['shortlists']['20']['precision'],result['average_precision'],-result['brier_score'])


def choose_candidate(validation_results):
    if not validation_results:raise ValueError('No validation candidates')
    return max(validation_results,key=lambda name:selection_key(validation_results[name]))


def fit(train,cal,features,parameters,version,protocol):
    model=HistGradientBoostingClassifier(**parameters)
    model.fit(train[features].to_numpy(dtype='float32'),train.target_36h.to_numpy(dtype='int8'))
    calibration=LogisticRegression(C=1,max_iter=200,random_state=42)
    calibration.fit(logits(model.predict_proba(cal[features].to_numpy(dtype='float32'))[:,1]),cal.target_36h)
    if calibration.coef_[0,0]<=0:raise ValueError('Calibration must preserve model ranking')
    return {'model':model,'calibration':calibration,'features':features,
            'metadata':{'version':version,'status':'research_only','horizon_hours':36,
                        'protocol':protocol,'comparison_report':'docs/FORECAST_36H_V2_EVALUATION.json'}}


def main():
    protocol={'version':'forecast-36h-v2','dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'splits':SPLITS,'candidates':CANDIDATES,
              'selection':'2025 validation top-20 precision, then average precision, then lower Brier score',
              'evaluation_policy':'2026 is previously inspected comparison data; never used for candidate selection',
              'boundary_embargo_hours':24,'forecast_source':'Exact initialized runs already cached; no new downloads',
              'feature_definitions':{'wet_hour':'liquid rain plus showers >= 0.1 mm',
                                     'rain_onset':'first wet forecast interval end, 1..36; 37 if no wet interval',
                                     'freezing_crossing':'change between temperature < 0 C and >= 0 C in adjacent forecast hours',
                                     'past_dry':'past 72-hour ERA5 liquid rainfall <= 0.1 mm',
                                     'heavy_hour':'forecast liquid rain plus showers >= 2 mm/hour (descriptive threshold)'}}
    (ROOT/'docs/FORECAST_36H_V2_EXPERIMENT.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    table=pd.read_csv(DATA)
    mapping=pd.read_csv(ROOT/'data/processed/spatial_weather/era5_500m/grid_weather_mapping.csv')
    print('Deriving transitions from cached forecasts',flush=True)
    table=add_transitions(table,mapping)
    enriched=ROOT/'data/processed/spatial_training/forecast_36h_transitions_v2.csv.gz'
    table.to_csv(enriched,index=False,compression='gzip')
    parts={name:partition(table,name) for name in SPLITS}
    summary={name:{'rows':len(part),'issue_dates':int(part.prediction_time.nunique()),
                   'positive_windows':int(part.target_36h.sum()),'first_issue':part.prediction_time.min(),
                   'last_issue':part.prediction_time.max()} for name,part in parts.items()}
    print('Partitions: '+json.dumps(summary),flush=True)
    for name,part in parts.items():
        if part.empty or part.target_36h.nunique()!=2:raise ValueError(f'Insufficient labels: {name}')
    models={};validation_results={}
    for name,candidate in CANDIDATES.items():
        print(f'Fitting candidate: {name}',flush=True)
        saved=fit(parts['training'],parts['calibration'],candidate['features'],candidate['parameters'],name,protocol)
        score=predict(saved,parts['validation'])
        metrics=evaluate_models(parts['validation'],{name:score})[name]
        models[name]=saved;validation_results[name]=metrics
        print(f'Validation {name}: top20={metrics["shortlists"]["20"]["precision"]:.4%}; AP={metrics["average_precision"]:.5f}',flush=True)
    selected=choose_candidate(validation_results)
    selection={'selected_candidate':selected,'validation_results':validation_results,'selection_rule':protocol['selection']}
    # Persist selection before computing any 2026 prediction or metric.
    (ROOT/'docs/FORECAST_36H_V2_SELECTION.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(f'Validation winner frozen: {selected}',flush=True)
    matched=fit(parts['training'],parts['calibration'],FEATURES,CANDIDATES[selected]['parameters'],
                'forecast-36h-v2-matched-history',protocol)
    evaluation=parts['evaluation']
    prior=joblib.load(ROOT/'models/forecast_comparison_forecast_aware_v1.joblib')
    backend=joblib.load(ROOT/'models/history_36h_v1.joblib')
    scores={'selected_v2':predict(models[selected],evaluation),'matched_history_v2':predict(matched,evaluation),
            'previous_forecast_v1':predict(prior,evaluation),'current_backend':predict(backend,evaluation)}
    metrics=evaluate_models(evaluation,scores)
    intervals={label:paired_top20_interval(evaluation,scores['selected_v2'],scores[label])
               for label in ('previous_forecast_v1','matched_history_v2','current_backend')}
    scenarios={}
    for name,mask in scenario_masks(evaluation).items():
        if mask.any():
            subset=evaluation.loc[mask]
            scenarios[name]={'issue_dates':int(subset.prediction_time.nunique()),
                             'results':evaluate_models(subset,{label:values[mask] for label,values in scores.items()})}
    difference=intervals['previous_forecast_v1']
    evidence='inconclusive'
    if difference['ci95'][0]>0:evidence='v2_advantage_on_previously_inspected_dates'
    elif difference['ci95'][1]<0:evidence='v1_advantage_on_previously_inspected_dates'
    report={'version':'forecast-36h-v2-evaluation','status':'research_only','protocol':protocol,
            'sklearn_version':sklearn.__version__,'summary':summary,**selection,
            'evaluation':metrics,'paired_top20_intervals':intervals,'scenarios':scenarios,
            'primary_result':evidence,'live_model_promoted':False,
            'limitations':['2026 results were inspected before this experiment; this is a development comparison, not a fresh final test.',
                           'Only two weather points serve 604 grids; LiDAR is not included in this experiment.',
                           'Targets are reported drainage concerns, not confirmed flooding; missing reports do not establish safety.',
                           'Daily 36-hour target windows overlap and six archive dates remain excluded.',
                           'Report publication delay is assumed to be 24 hours; forecast publication and ingestion completeness are unaudited.',
                           'Retrospective ERA5 past rainfall may differ from as-issued observations.',
                           'Calibration covers July-September 2025; probabilities may shift with season.',
                           'Matched history uses forecast-selected settings; it has not been independently tuned.',
                           'Neither an 80-90% event detection rate nor arbitrary-hour operation has been established.']}
    models[selected]['metadata']['version']='forecast-36h-v2-selected'
    models[selected]['metadata']['selected_candidate']=selected
    joblib.dump(models[selected],ROOT/'models/forecast_36h_v2_selected.joblib')
    joblib.dump(matched,ROOT/'models/forecast_36h_v2_matched_history.joblib')
    output=evaluation[['prediction_time','grid_id','target_36h']].copy()
    for label,values in scores.items():output[label]=values
    output.to_csv(ROOT/'data/processed/spatial_training/forecast_36h_v2_evaluation_scores.csv.gz',index=False,compression='gzip')
    (ROOT/'docs/FORECAST_36H_V2_EVALUATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Forecast model v2 experiment','',f'Validation-selected candidate: **{selected}**. Research model; live backend artifact unchanged.','',
           'Training includes March 2024 through June 2025, with boundary windows purged. '
           'Calibration uses July-September 2025 and selection uses October-December 2025. '
           '2026 is a previously inspected comparison period, not a pristine final test.','',
           '## 2025 validation selection','', '| Candidate | Top-20 precision | Average precision | Brier score |',
           '| --- | ---: | ---: | ---: |']
    for name,result in validation_results.items():
        lines.append(f'| {name} | {result["shortlists"]["20"]["precision"]:.2%} | {result["average_precision"]:.4f} | {result["brier_score"]:.5f} |')
    lines+=['','## 2026 comparison','', '| Model | Top-20 precision | Top-20 recall | Positive windows found | Average precision | Brier score |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,result in metrics.items():
        top=result['shortlists']['20']
        lines.append(f'| {name} | {top["precision"]:.2%} | {top["recall"]:.2%} | {top["positive_windows_found"]} | {result["average_precision"]:.4f} | {result["brier_score"]:.5f} |')
    lines+=['',f'Change against v1: {difference["forecast_minus_history_precision"]*100:+.2f} percentage points top-20 precision. '
            f'Paired weekly-bootstrap 95% interval: [{difference["ci95"][0]*100:+.2f}, {difference["ci95"][1]*100:+.2f}].',
            '',f'Primary result: **{evidence}**.','', '## Weather-change subsets','',
            '| Scenario | Dates | V1 top-20 precision | V2 top-20 precision |','| --- | ---: | ---: | ---: |']
    for name in ('heavy_rain','increasing_rain','decreasing_rain','dry_or_trace'):
        item=scenarios.get(name)
        if item:lines.append(f'| {name} | {item["issue_dates"]} | {item["results"]["previous_forecast_v1"]["shortlists"]["20"]["precision"]:.2%} | {item["results"]["selected_v2"]["shortlists"]["20"]["precision"]:.2%} |')
    lines+=['','## Limitations','',*[f'- {item}' for item in report['limitations']],'',
            '## Reproduce','', '```powershell','.venv/Scripts/python.exe -u src/train_forecast_36h_v2.py','```','',
            'Saved artifact: `models/forecast_36h_v2_selected.joblib`. '
            'Protocol: `docs/FORECAST_36H_V2_EXPERIMENT.json`. Selection before 2026 scoring: `docs/FORECAST_36H_V2_SELECTION.json`.','']
    (ROOT/'docs/FORECAST_36H_V2_EVALUATION.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected':selected,'primary_result':evidence,
                      'top20_precision':{label:value['shortlists']['20']['precision'] for label,value in metrics.items()},
                      'paired_vs_v1':difference},indent=2),flush=True)


if __name__=='__main__':main()
