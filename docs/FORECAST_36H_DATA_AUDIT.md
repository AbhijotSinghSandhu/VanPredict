# Forecast-aware 36-hour dataset audit

Status: **structural_checks_passed_with_archive_gaps**. 922 issue dates, 556,888 cell-windows, 8,637 positive cell-windows (1.55%).

Period: 2024-03-14 through 2026-09-27. There are 1,844 weather source-windows across 2 source points; the grid rows are not independent weather samples.

## Data checks

- duplicate_cell_windows: 0
- missing_requested_dates: ['2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2026-06-11']
- unexplained_missing_dates: []
- unexpected_dates: []
- incomplete_grid_dates: []
- nonfinite_input_values: 0
- incorrect_horizons: 0
- incorrect_history_cutoffs: 0
- incorrect_local_issue_hours: 0
- target_count_mismatches: 0
- invalid_targets: 0
- negative_report_counts: 0
- forecast_runs_after_issue: 0
- incorrect_forecast_runs: 0
- incorrect_run_delays: 0
- inconsistent_weather_in_same_source_window: 0
- incorrect_weather_source_assignments: 0
- rain_negative_or_nonmonotonic_totals: 0
- showers_negative_or_nonmonotonic_totals: 0

## Forecast coverage

36-hour liquid precipitation summary (mm): {'minimum': 0.0, 'median': 1.0, 'p95': 32.88499999999999, 'maximum': 160.50000000000003}.

Rain intensity source-windows: {'dry_or_trace': 694, 'light': 501, 'moderate': 404, 'heavy': 245}.

Weather change source-windows: {'decreasing': 247, 'steady': 1349, 'increasing': 248}.

## Limitations

- Targets are drainage concern reports, not verified flooding.
- 36-hour windows issued every day overlap by 12 hours; positive windows are not distinct incidents.
- Two weather points provide shared forecasts for 604 grids.
- Forecast run publication timestamps and 24-hour report availability delay remain unaudited assumptions.
- Historical ERA5 rainfall is retrospective; as-issued availability is not verified.
- Observed report timestamp bounds do not establish ingestion completeness.
- Rain intensity and change bands are descriptive thresholds, not hazard thresholds.
- No model performance or weather-to-report causal conclusion has been established.

## Next experiment

Freeze date-based train/calibration/evaluation partitions before fitting. Purge target windows crossing boundaries. Compare history-only and forecast-aware models fitted on the same dates, then add similar-weather retrieval using only earlier events. Keep this dataset and experiment separate from the live model until evaluation supports promotion.
