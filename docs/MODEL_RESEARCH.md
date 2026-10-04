# Spatial next-day model: first evaluation

The saved experimental model predicts whether a 500 m cell receives at least one Street Surface Water Flooding or Catch Basin Concern report on the next Vancouver calendar day. It does not predict confirmed flooding or flood depth.

Reproduce from the repository root using the project's Python environment:

```sh
python src/train_spatial_model.py
```

The script saves `models/spatial_nextday_v1.joblib` (ignored by Git) and `docs/model_evaluation.json`. The latter contains exact metrics, feature names, software version, and limitations. Training scripts do not run at API startup.

## Chronological partitions

Partitions use target dates, with rainfall features drawn from the preceding completed day:

- Training: 2019-01-01 through 2022-12-31, 882,444 grid-days.
- Sigmoid calibration: 2023-01-01 through 2023-12-31, 220,460 grid-days.
- Untouched test: 2024-01-01 through 2026-09-28, 605,208 grid-days.

All 604 cells and all days are retained, including zero-report days. Training labels are not resampled. Calibration is fitted only on 2023 model scores; the training estimator is not refitted after calibration. The fixed candidate is compared with a location-only baseline: training-period report frequency per cell, smoothed with 30 prior grid-days.

## Results and decision

The test set has 5,405 positive grid-days (0.893%). Average precision is 0.03110 versus 0.02041 for the baseline; ROC-AUC is 0.7803. Brier score is 0.008765 versus 0.008801, a small improvement.

For a practical shortlist of 20 cells per day, model precision is 2.730% and recall is 10.120%. The baseline scores 2.904% precision and 10.768% recall. Most shortlisted cell-days have no matching report. The model does not improve this operating point, despite better overall ranking.

Prediction mode remains disabled. Next research work should evaluate other operating points, temporal stability, calibration drift, and suitable past-report history features, using a new reserved time period or nested chronological validation for further tuning. Repeatedly tuning against this test period would compromise its holdout status. No live forecast-weather pipeline exists; this model uses historical completed-day ERA5 features.

Archive date span is only a proxy for completeness. No matching report does not prove absence of flooding. The two coarse ERA5 weather cells do not provide independent 500 m weather observations. These limitations remain relevant even if future models improve their metrics.
