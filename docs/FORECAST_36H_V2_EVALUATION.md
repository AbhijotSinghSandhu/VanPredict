# Forecast model v2 experiment

Validation-selected candidate: **more_seasons**. Research model; live backend artifact unchanged.

Training includes March 2024 through June 2025, with boundary windows purged. Calibration uses July-September 2025 and selection uses October-December 2025. 2026 is a previously inspected comparison period, not a pristine final test.

## 2025 validation selection

| Candidate | Top-20 precision | Average precision | Brier score |
| --- | ---: | ---: | ---: |
| more_seasons | 10.94% | 0.1159 | 0.02692 |
| transitions_80 | 10.78% | 0.1174 | 0.02947 |
| transitions_160 | 10.89% | 0.1178 | 0.02853 |
| transitions_regularized | 10.61% | 0.1255 | 0.02589 |

## 2026 comparison

| Model | Top-20 precision | Top-20 recall | Positive windows found | Average precision | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected_v2 | 3.75% | 11.47% | 202 | 0.0613 | 0.01068 |
| matched_history_v2 | 3.79% | 11.58% | 204 | 0.0446 | 0.01062 |
| previous_forecast_v1 | 4.00% | 12.21% | 215 | 0.0626 | 0.01044 |
| current_backend | 4.01% | 12.27% | 216 | 0.0410 | 0.01060 |

Change against v1: -0.24 percentage points top-20 precision. Paired weekly-bootstrap 95% interval: [-0.72, +0.24].

Primary result: **inconclusive**.

## Weather-change subsets

| Scenario | Dates | V1 top-20 precision | V2 top-20 precision |
| --- | ---: | ---: | ---: |
| heavy_rain | 24 | 13.12% | 11.88% |
| increasing_rain | 31 | 6.61% | 5.16% |
| decreasing_rain | 28 | 8.75% | 9.11% |
| dry_or_trace | 104 | 2.31% | 2.12% |

## Limitations

- 2026 results were inspected before this experiment; this is a development comparison, not a fresh final test.
- Only two weather points serve 604 grids; LiDAR is not included in this experiment.
- Targets are reported drainage concerns, not confirmed flooding; missing reports do not establish safety.
- Daily 36-hour target windows overlap and six archive dates remain excluded.
- Report publication delay is assumed to be 24 hours; forecast publication and ingestion completeness are unaudited.
- Retrospective ERA5 past rainfall may differ from as-issued observations.
- Calibration covers July-September 2025; probabilities may shift with season.
- Matched history uses forecast-selected settings; it has not been independently tuned.
- Neither an 80-90% event detection rate nor arbitrary-hour operation has been established.

## Reproduce

```powershell
.venv/Scripts/python.exe -u src/train_forecast_36h_v2.py
```

Saved artifact: `models/forecast_36h_v2_selected.joblib`. Protocol: `docs/FORECAST_36H_V2_EXPERIMENT.json`. Selection before 2026 scoring: `docs/FORECAST_36H_V2_SELECTION.json`.
