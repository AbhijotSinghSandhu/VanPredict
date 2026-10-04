# Forecast-aware model comparison

Fixed settings; calibrated on JanuaryÃ¢â‚¬â€œJune 2025; evaluated on later dates. Live model unchanged.

On the 2026 dates, the daily top-20 lists contained 215 report-containing cell-windows for the forecast model and 215 for the matched history-only model. The paired interval does not establish a top-20 advantage. Average precision across all grid-windows and the Brier score favor forecast inputs in this evaluation.

## Partitions

| Partition | Issue dates | Rows | Positive windows |
| --- | ---: | ---: | ---: |
| training | 291 | 175,764 | 3,675 |
| calibration | 179 | 108,116 | 1,161 |
| development | 177 | 106,908 | 1,991 |
| evaluation | 269 | 162,476 | 1,761 |

6 boundary issue dates were purged to allow 36-hour labels plus the assumed 24-hour publication delay.

## 2026 evaluation

| Model | Top-20 precision | Top-20 recall | Average precision | Brier score |
| --- | ---: | ---: | ---: | ---: |
| history_only | 4.00% | 12.21% | 0.0426 | 0.01057 |
| forecast_aware | 4.00% | 12.21% | 0.0626 | 0.01044 |
| fixed_location_baseline | 3.68% | 11.24% | 0.0256 | 0.01087 |
| updating_location_baseline | 3.10% | 9.48% | 0.0279 | 0.01065 |
| current_backend_model | 4.01% | 12.27% | 0.0410 | 0.01060 |

Forecast minus matched history top-20 precision: +0.00 percentage points. Paired weekly-bootstrap 95% interval: [-0.50, +0.51] percentage points.

Primary result: **inconclusive**.

## Weather changes in 2026

| Scenario | Issue dates | History top-20 precision | Forecast top-20 precision |
| --- | ---: | ---: | ---: |
| heavy_rain | 24 | 12.29% | 13.12% |
| increasing_rain | 31 | 6.61% | 6.61% |
| decreasing_rain | 28 | 8.21% | 8.75% |
| dry_or_trace | 104 | 2.36% | 2.31% |

Scenario groups use the mean forecast across the two weather source points. Rain intensity thresholds are descriptive; increasing/decreasing means a greater than 5 mm shift between the first and second 18-hour halves. These scenario results are exploratory.

## Limitations

- Targets are drainage concern reports, not verified flooding.
- 36-hour windows issued every day overlap by 12 hours; positive windows are not distinct incidents.
- Two weather points provide shared forecasts for 604 grids.
- Forecast run publication timestamps and 24-hour report availability delay remain unaudited assumptions.
- Historical ERA5 rainfall is retrospective; as-issued availability is not verified.
- Observed report timestamp bounds do not establish ingestion completeness.
- Rain intensity and change bands are descriptive thresholds, not hazard thresholds.
- Metrics describe this fixed evaluation; general accuracy and causal weather/report effects are not established.
- One training year and two coarse weather sources limit weather-pattern learning.
- Calibration and evaluation can have different seasonal report rates.
- Current backend model uses different older training dates; the matched history-only model is the fair feature comparator.
- Prior aggregate dataset inspection and older development research mean this is not a pristine independent final test.
- Calendar-week bootstrap reduces within-week dependence; residual dependence can remain.
- Only 06:00 Vancouver issue times have been evaluated; no claim about arbitrary-hour live predictions.

## Next step

Inspect the fixed comparison and weather-change subsets before choosing the next experiment. Similar-weather retrieval should use only historical events and outcomes available before each prediction issue. No training or probability calibration may use the evaluation labels.

## Reproduce

```powershell
.venv/Scripts/python.exe -u src/train_forecast_36h.py
.venv/Scripts/python.exe -m unittest tests.test_forecast_36h_model tests.test_forecast_training_36h tests.test_training_36h tests.test_history_36h_model
```

The dataset hash, feature lists and fixed settings are saved in `docs/FORECAST_36H_EXPERIMENT.json`. Research artifacts: `models/forecast_comparison_forecast_aware_v1.joblib` and `models/forecast_comparison_history_only_v1.joblib`. Per-window evaluation scores: `data/processed/spatial_training/forecast_comparison_scores_v1.csv.gz`.
