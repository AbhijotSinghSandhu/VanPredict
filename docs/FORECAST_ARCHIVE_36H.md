# Forecast archive sample for 36-hour training

`src/forecast_training_36h.py` pairs exact initialized ECMWF single runs with each 06:00 Vancouver report window. It requests hourly rain, showers, snowfall and temperature for the two existing weather points, then attaches common weather features to the 604 grid cells. Weather-pattern features include separate 6/12/24/36-hour rain and shower totals, hourly and rolling three-hour peaks, wet-hour counts, first/second-half changes, snowfall and minimum temperature. The target remains at least one matching drainage report in [issue, issue + 36 elapsed hours).

Run one date first:

```powershell
.venv/Scripts/python.exe src/forecast_training_36h.py --start-date 2025-06-04 --end-date 2025-06-04 --output data/processed/spatial_training/forecast_36h.csv.gz
```

Build a small event sample:

```powershell
.venv/Scripts/python.exe src/forecast_training_36h.py --start-date 2025-06-18 --end-date 2025-06-24 --output data/processed/spatial_training/forecast_june_2025_sample.csv.gz
```

The historical sample requests the official [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api), which preserves individual model initialization runs. Its [archive documentation](https://open-meteo.com/en/docs/single-runs-api) lists ECMWF IFS HRES runs from March 2024 and says global model outputs usually take 4–6 hours after initialization to publish. We prefer the 06 UTC run for a 06:00 local prediction: it is 7 hours before the issue in daylight time and 8 hours before in standard time. Some earlier archive dates do not contain the 06 UTC cycle; only an explicit unavailable-run response permits fallback to that date's earlier 00 UTC run (13–14 hours before issue). Both the actual run and its age are retained. The exact historical publication instant is not archived, so this is an approximation that still needs an availability audit. Do not silently reinterpret model initialization as publication time.

The API verified seven June 18–24, 2025 runs at both Vancouver source points. The paired table has 4,228 rows and 25 positive cell-windows. For June 20, the mean 36-hour forecast across the two points was 14.92 mm rain plus 4.66 mm showers; peak hourly rain averaged 2.21 mm and peak rolling 3-hour rain 5.14 mm. June 23 was dry in the forecast at both points. These are distinct forecast patterns for an initial data-pipeline check, not enough rows to establish a weather/report relationship.

In the archived single-run endpoint, rain, showers, snowfall and temperature were available for the sample. Precipitation probability returned an unavailable model-run error for the same date, so it is intentionally omitted. The model uses only two weather points; values repeat among cells assigned to each point and do not imply 500 m meteorological resolution. Rain and showers remain separate features so similar totals with different forecast composition can be evaluated.

Raw API responses are cached under the ignored `data/processed/spatial_training/forecast_archive/`. The paired gzip CSV and metadata sidecar go under ignored `data/processed/spatial_training/`. The builder validates contiguous 36-hour timestamps and missing values, and records model run, issue delay and feature definitions. Daily issue times create overlapping target windows; handle that dependence and purge split-boundary windows during future evaluation.

Tests: `.venv/Scripts/python.exe -m unittest tests.test_forecast_training_36h tests.test_training_36h`.

## Full archive build and audit

```powershell
.venv/Scripts/python.exe -u src/build_forecast_archive_36h.py --start-date 2024-03-14 --end-date 2026-09-27
```

The coordinator defaults to one download worker; `--workers` accepts at most three. New network tasks are spaced at least 0.75 seconds apart, and temporary download failures receive a final serial retry. Raw runs are cached atomically and revalidated locally before contacting the API again; monthly CSV checkpoints resume only complete 604-grid dates. Calendar-month partitions include every date, including the first days of months following a partial start month. History features are computed once per month, rather than reparsing the full report and rainfall archive for every day.

Outputs:

- `data/processed/spatial_training/forecast_archive_monthly_v2/`: monthly datasets, metadata, date-level audits and download failures.
- `data/processed/spatial_training/forecast_36h_full_v2.csv.gz`: combined dataset, including each grid's `weather_cell_id`, with a `.metadata.json` input/target contract.
- `docs/FORECAST_36H_DATA_AUDIT.json` and `docs/FORECAST_36H_DATA_AUDIT.md`: full structural audit and descriptive forecast-pattern coverage.

The audit checks all requested dates, grid coverage, duplicate windows, missing or nonfinite inputs, label/count consistency, 36-hour horizons, report history cutoffs, local issue hours, permissible earlier forecast runs, actual run ages and consistency within each shared weather source. Forecast intensity and change groups are descriptive: liquid precipitation thresholds are 0.1/5/20 mm across 36 hours, and increasing/decreasing means the second 18-hour total differs from the first by more than 5 mm. They are not validated hazard categories.

Unusable forecast dates are excluded in their entirety, with errors recorded in the download audit. `structural_checks_passed_with_archive_gaps` means the included rows pass structural checks and every absent date has a documented archive failure. An unexplained absent date, partial grid window, invalid time alignment or invalid input still fails the structural audit. No missing forecast is filled with zero, and excluded dates must not be treated as no-report dates in subsequent analysis.

Actual report timestamp extremes provide conservative bounds, not proof of complete ingestion. The builder also requires 720 hours of report history before the assumed 24-hour publication cutoff, and all past rainfall through issue time. Future rainfall coverage is not needed to form report targets; requiring it would incorrectly exclude supported final dates. The latest source rainfall timestamp permits a September 27, 2026 06:00 local issue, and the report archive extends past its full 36-hour target window.

If the project virtual environment cannot start, a compatible Python runtime with NumPy and pandas can build and audit this dataset. The existing history-model training tests additionally need joblib and scikit-learn. No model is promoted by running these data builders.
