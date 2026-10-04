"""Controlled catch-basin experiment; never promotes the live backend model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.catch_basin_features import DRAINAGE_FEATURES, FEATURES_PATH
from src.train_forecast_36h import DATA, predict, evaluate_models, paired_top20_interval
from src.train_forecast_36h_v2 import partition, SPLITS, fit, selection_key


def attach_inventory(table,inventory):
    if inventory.grid_id.duplicated().any():raise ValueError('Duplicate inventory grids')
    output=table.merge(inventory[['grid_id']+DRAINAGE_FEATURES],on='grid_id',how='left',validate='many_to_one',sort=False)
    if not np.isfinite(output[DRAINAGE_FEATURES].to_numpy()).all():raise ValueError('Missing or nonfinite infrastructure features')
    return output


def main():
    live=ROOT/'models/history_36h_v1.joblib'
    live_hash=hashlib.sha256(live.read_bytes()).hexdigest()
    base=joblib.load(ROOT/'models/forecast_36h_v2_selected.joblib')
    candidate=base['metadata']['selected_candidate']
    parameters=base['metadata']['protocol']['candidates'][candidate]['parameters']
    audit=json.loads((ROOT/'docs/CATCH_BASIN_GRID_AUDIT.json').read_text())
    table=attach_inventory(pd.read_csv(DATA),pd.read_csv(FEATURES_PATH))
    parts={name:partition(table,name) for name in SPLITS}
    protocol={'splits':SPLITS,'parameters':parameters,'base_candidate':candidate,
              'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'inventory':audit['snapshot'],'features_added':DRAINAGE_FEATURES,
              'selection':'One fixed candidate; compare 2025 validation before scoring previously inspected 2026 dates',
              'historical_inventory_verified':False,'promotion_allowed':False}
    (ROOT/'docs/FORECAST_36H_DRAINAGE_EXPERIMENT.json').write_text(json.dumps(protocol,indent=2))
    print('Training fixed catch-basin candidate',flush=True)
    saved=fit(parts['training'],parts['calibration'],base['features']+DRAINAGE_FEATURES,parameters,'forecast-36h-drainage-v1',protocol)
    saved['metadata']['comparison_report']='docs/FORECAST_36H_DRAINAGE_EVALUATION.json'
    validation=evaluate_models(parts['validation'],{'baseline_v2':predict(base,parts['validation']),'catch_basins':predict(saved,parts['validation'])})
    decision={'validation':validation,'validation_improved':selection_key(validation['catch_basins'])>selection_key(validation['baseline_v2']),
              'promotion_allowed':False}
    (ROOT/'docs/FORECAST_36H_DRAINAGE_SELECTION.json').write_text(json.dumps(decision,indent=2))
    evaluation=parts['evaluation']
    scores={'baseline_v2':predict(base,evaluation),'catch_basins':predict(saved,evaluation),
            'previous_forecast_v1':predict(joblib.load(ROOT/'models/forecast_comparison_forecast_aware_v1.joblib'),evaluation)}
    metrics=evaluate_models(evaluation,scores)
    interval=paired_top20_interval(evaluation,scores['catch_basins'],scores['baseline_v2'])
    report={'protocol':protocol,**decision,'evaluation':metrics,'paired_top20_vs_baseline':interval,'live_model_promoted':False,
            'limitations':['Current inventory may contain assets absent in historical years; this is an exploratory spatial comparison.',
                           '2026 comparison dates were already inspected and are not a fresh final test.',
                           'Targets are reported drainage concerns, not confirmed flooding; count does not establish drainage capacity.',
                           'Only two weather source points cover 604 grids. Calibration uses July-September 2025.']}
    joblib.dump(saved,ROOT/'models/forecast_36h_drainage_v1.joblib')
    (ROOT/'docs/FORECAST_36H_DRAINAGE_EVALUATION.json').write_text(json.dumps(report,indent=2))
    lines=['# Catch-basin model experiment','',
           'Added catch-basin count, density per square kilometre, and nearest distance from grid centroid. Fixed settings and dates match the v2 baseline.',
           '',f'Inventory source modified: {audit["snapshot"]["source_modified_at"]}. Historical infrastructure availability is unverified. Live backend unchanged.','',
           '| Period | Model | Top-20 precision | Average precision | Brier score |','| --- | --- | ---: | ---: | ---: |']
    for period,results in [('2025 validation',validation),('2026 comparison',metrics)]:
        for name,result in results.items():
            lines.append(f'| {period} | {name} | {result["shortlists"]["20"]["precision"]:.2%} | {result["average_precision"]:.4f} | {result["brier_score"]:.5f} |')
    lines+=['',f'Paired weekly-bootstrap 95% interval for top-20 precision change versus v2: [{interval["ci95"][0]*100:+.2f}, {interval["ci95"][1]*100:+.2f}] percentage points.','',*[f'- {item}' for item in report['limitations']],'',
            'Reproduce: `.venv/Scripts/python.exe src/catch_basin_features.py`, then `.venv/Scripts/python.exe src/train_forecast_drainage_36h.py`.','']
    (ROOT/'docs/FORECAST_36H_DRAINAGE_EVALUATION.md').write_text('\n'.join(lines))
    if hashlib.sha256(live.read_bytes()).hexdigest()!=live_hash:raise RuntimeError('Live model changed during experiment')
    print(json.dumps({'validation_improved':decision['validation_improved'],'top20_precision':{name:item['shortlists']['20']['precision'] for name,item in metrics.items()},'interval':interval},indent=2),flush=True)


if __name__=='__main__':main()
