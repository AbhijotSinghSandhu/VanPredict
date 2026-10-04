"""Fit a reproducible experimental 500 m next-day report model.

All grid-days are retained. Calibration and test years follow training years;
no random train/test split, negative subsampling, or current-day labels.
"""
import json
import os
from pathlib import Path
import sys

# Bound CPU use for local dashboard development.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spatial_training_data import build_training_table, TRAINING_FEATURES

SPLITS = {
    'training': ('2019-01-01', '2022-12-31'),
    'calibration': ('2023-01-01', '2023-12-31'),
    'test': ('2024-01-01', '2026-09-28'),
}


def load_split(first, last):
    feature_start = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    feature_end = (pd.Timestamp(last) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    frame = build_training_table(feature_start, feature_end)
    if not frame.target_date.between(first, last).all():
        raise ValueError('Target dates escaped chronological partition')
    x = frame[TRAINING_FEATURES].to_numpy(dtype='float32')
    y = frame.is_drainage_problem_next_day.to_numpy(dtype='int8')
    grids = frame.grid_id.to_numpy()
    dates = frame.target_date.to_numpy()
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite model features')
    return x, y, grids, dates


def logits(probability):
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def metrics(y, scores, dates):
    ranked = pd.DataFrame({'date': dates, 'label': y, 'score': scores})
    top = ranked.sort_values(['date', 'score'], ascending=[True, False], kind='stable').groupby('date').head(20)
    return {
        'rows': len(y), 'positive_grid_days': int(y.sum()), 'prevalence': float(y.mean()),
        'average_precision': float(average_precision_score(y, scores)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'brier_score': float(brier_score_loss(y, scores)),
        'mean_score': float(np.mean(scores)),
        'precision_top20_per_day': float(top.label.mean()),
        'recall_top20_per_day': float(top.label.sum() / y.sum()),
    }


def main():
    print('Loading training targets: 2019–2022', flush=True)
    x, y, grids, _ = load_split(*SPLITS['training'])
    prior = float(y.mean())
    stats = pd.DataFrame({'grid': grids, 'label': y}).groupby('grid').label.agg(['sum', 'count'])
    baseline = ((stats['sum'] + 30 * prior) / (stats['count'] + 30)).to_dict()
    training_summary = {'rows': len(y), 'positive_grid_days': int(y.sum()), 'prevalence': prior}
    print(f'Fitting on {len(y):,} grid-days; {y.sum():,} positives', flush=True)
    model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                                          l2_regularization=10, min_samples_leaf=100,
                                          early_stopping=False, random_state=42)
    model.fit(x, y)
    del x, y, grids
    print('Calibrating with 2023 targets', flush=True)
    x, y, _, _ = load_split(*SPLITS['calibration'])
    calibration = LogisticRegression(C=1, max_iter=200, random_state=42)
    calibration.fit(logits(model.predict_proba(x)[:, 1]), y)
    calibration_summary = {'rows': len(y), 'positive_grid_days': int(y.sum()), 'prevalence': float(y.mean())}
    del x, y
    print('Evaluating untouched 2024–September 2026 targets', flush=True)
    x, y, grids, dates = load_split(*SPLITS['test'])
    scores = calibration.predict_proba(logits(model.predict_proba(x)[:, 1]))[:, 1]
    baseline_scores = np.array([baseline.get(grid, prior) for grid in grids])
    model_metrics, baseline_metrics = metrics(y, scores, dates), metrics(y, baseline_scores, dates)
    # A comparison result, not a production-readiness certificate.
    better = (model_metrics['average_precision'] > baseline_metrics['average_precision']
              and model_metrics['brier_score'] < baseline_metrics['brier_score'])
    report = {
        'version': 'spatial-nextday-v1', 'status': 'experimental',
        'grid_size_metres': 500, 'target': 'At least one matching drainage report on the next Vancouver day',
        'target_categories': ['Street Surface Water Flooding Case', 'Catch Basin Concern Case'],
        'horizon': 'next calendar day after a completed feature day',
        'features': TRAINING_FEATURES, 'splits_by_target_date': SPLITS,
        'training': training_summary, 'calibration': calibration_summary,
        'test': model_metrics, 'location_baseline_test': baseline_metrics,
        'beats_location_baseline_on_ap_and_brier': bool(better),
        'model': 'HistGradientBoostingClassifier', 'calibration_method': 'Sigmoid fitted on 2023 scores',
        'sklearn_version': sklearn.__version__, 'random_seed': 42,
        'limitations': [
            'Predicts reported drainage concerns, not confirmed floods or flood depth.',
            'Report date span is only a proxy for archive completeness; reporting gaps are not audited.',
            'Historical daily ERA5 features are retrospective; no live forecast weather is integrated.',
            'Weather comes from two coarse ERA5 sources; 500 m cells do not have independent rainfall measurements.',
            'This single chronological holdout has not established production reliability or future calibration.',
        ],
    }
    output = ROOT / 'models'
    output.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'calibration': calibration, 'features': TRAINING_FEATURES,
                 'metadata': report}, output / 'spatial_nextday_v1.joblib')
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    (docs / 'model_evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
