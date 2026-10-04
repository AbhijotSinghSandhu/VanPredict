# History-only 36-hour development model

Run `.venv/Scripts/python.exe src/train_history_36h.py` from the repository root. This builds examples in memory from the compact weather sources and prepared 311 reports, fits a fixed gradient boosting classifier, and calibrates its scores with a later year. It does not replace the next-day artifact, enable predictions, or change the frontend.

Training: 2019–2021. Sigmoid calibration: 2022. Development evaluation: 2023–2025. Each partition removes any 36-hour label window extending past its boundary. All 604 cells and zero-report windows are retained. No negative sampling or random time split is used. These development years overlap previously inspected research; the results are not a fresh final holdout. No hyperparameter selection was performed for this candidate.

The 12 inputs are explicitly selected from `training_36h.FEATURES`; target counts, target labels and timestamps are excluded. The issue hour is always 06:00 Vancouver time. The report-history availability cutoff is issue minus 24 hours, an assumed delay. The declared report coverage is provisionally the observed archive date span. ERA5 is retrospective rather than an as-issued observation archive. All of these limit claims about prospective performance.

Comparators:

- Fixed location baseline: per-cell 36-hour positive-window frequency in training, smoothed with 30 prior examples. These probabilities stay fixed during development.
- Updating location baseline: past-only per-cell matching report frequency, with the same availability cutoff as model inputs. The probability conversion is `1 - exp(-1.5 * reports_per_day)`, a Poisson assumption; its ranking is simply report frequency. This assumption should not be treated as calibrated probability.

Evaluation reports average precision, ROC-AUC, Brier score, mean probability versus observed prevalence, ten equal-size calibration groups, and shortlist precision/recall at 5/10/20 cells per prediction time. Stable grid IDs resolve score ties. Year-by-season comparisons expose temporal variation. Overlapping windows are correlated and positive-window counts can repeat the same report; no independent-sample confidence intervals are claimed.

Outputs: `docs/history_36h_evaluation.json` and the ignored local artifact `models/history_36h_v1.joblib`, containing estimator, calibration, feature order, baseline mapping, prior and metadata. No live forecast features are used. Further development needs fresh confirmation data, audited availability, forecast snapshots and arbitrary-hour examples before the Predict button can honestly use this model.

Tests: `.venv/Scripts/python.exe -m unittest tests.test_training_36h tests.test_history_36h_model tests.test_spatial_training_data`.

## First development results

Training used 661,380 rows. Development used 661,380 rows with 12,316 positive cell-windows (1.862%). Twelve tests passed, including scoring 604 real rows with the saved artifact.

| Shortlist precision | Model | Fixed frequency | Updating frequency |
|---|---:|---:|---:|
| Top 5 | 8.33% | 7.45% | 7.73% |
| Top 10 | 7.35% | 6.74% | 6.60% |
| Top 20 | 6.44% | 5.69% | 6.05% |

Top-20 model recall is 11.45% versus 10.76% for the updating baseline. Average precision is 0.06432 versus 0.04342; Brier score is 0.017887 versus 0.018064. The mean model probability is 1.872%, close to the overall observed 1.862%, but agreement of overall means does not establish calibration in individual areas. The highest probability decile averages 8.29% versus an observed 6.85% (see exact JSON values); overconfidence remains a concern.

The model exceeds the updating baseline at top 20 in seven of twelve year-season groups, and the fixed baseline in ten of twelve. Overall gain is modest and is not consistent across every season. Most highlighted cell-windows have no matching report. These results support further development, not activation of live flood predictions. Next work should audit availability and collect forecast snapshots, then measure incremental forecast value with fresh confirmation data.
