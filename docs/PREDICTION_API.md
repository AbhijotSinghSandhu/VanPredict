# Experimental backend inference

The saved `models/history_36h_v1.joblib` is loaded lazily and cached. No training runs on API startup. Restart the backend after replacing an artifact or prepared data. The existing next-day research model/status is retained; model-status also exposes the separate history_36h backend capability.

## Prepared inputs

`POST /api/v1/predict` accepts:

```json
{"grid_size_m":500,"forecast_hours":36,"prediction_time":"2025-06-04T06:00:00-07:00","top_k":20}
```

Only 500 m, 36 hours and 06:00 Vancouver times are supported by this trained model. Timezone is required. Omit prediction_time to score the latest available 06:00 historical snapshot; this does not mean now. Requests beyond stored weather coverage return 422, rather than substituting stale weather. The backend constructs the exact trained FEATURES, with report-history cutoff 24 hours before issue. Inference never creates or reads future labels. Target coverage is unnecessary for inference, but input-history coverage is required.

## Explicit model inputs

`POST /api/v1/predict/features` accepts the same settings with required prediction_time and a cells array. Each cell has cell_id, centroid_latitude, centroid_longitude, rain_past_24h, rain_past_72h, rain_past_168h, reports_past_24h, reports_past_168h, reports_past_720h and historical_reports_per_day. Calendar features are derived by the backend. IDs and centroids must match the prepared grid. Nonfinite/negative values, inconsistent window totals, duplicate cells, unsupported grid/horizon/time and unknown fields are rejected. Supply one to 604 cells; rankings are relative to supplied cells. Caller-supplied histories must follow the training cutoff and availability assumptions: this endpoint validates structure, not provenance or historical accuracy. It does not accept forecast rainfall as a substitute for past rainfall.

## Outputs

Both endpoints return status=experimental, model_version, grid_size_m, prediction_start/end, generated_at, input_source, input_age_hours, feature_names, uses_forecast_rainfall=false, warnings, areas and top_areas. Each area contains cell_id, calibrated report_probability in [0,1], and rank. All scored cells are returned in descending probability order; stable cell IDs resolve ties. top_areas contains up to top_k (default 20). Probabilities describe matching drainage reports, not flooding. Prepared mode additionally returns weather_coverage_end and history_cutoff.

Missing model/data returns 503. Invalid or unsupported inputs return 422. Full prepared mode scores all 604 cells. The latest prepared weather is older than current time: inspect input_source and input_age_hours before displaying results. Live forecast and current history acquisition, arbitrary-hour support and frontend map integration remain separate work. The API does not pretend this history-only estimator responds to forecast changes.

Tests: `.venv/Scripts/python.exe -m unittest tests.test_prediction_api tests.test_training_36h tests.test_history_36h_model`. Install backend/requirements-dev.txt for the test client. No accuracy tuning was included in this integration.
