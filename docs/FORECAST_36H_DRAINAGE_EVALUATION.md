# Catch-basin model experiment

Added catch-basin count, density per square kilometre, and nearest distance from grid centroid. Fixed settings and dates match the v2 baseline.

Inventory source modified: 2026-09-28T13:36:35+00:00. Historical infrastructure availability is unverified. Live backend unchanged.

| Period | Model | Top-20 precision | Average precision | Brier score |
| --- | --- | ---: | ---: | ---: |
| 2025 validation | baseline_v2 | 10.94% | 0.1159 | 0.02692 |
| 2025 validation | catch_basins | 11.61% | 0.1233 | 0.02712 |
| 2026 comparison | baseline_v2 | 3.75% | 0.0613 | 0.01068 |
| 2026 comparison | catch_basins | 3.75% | 0.0602 | 0.01071 |
| 2026 comparison | previous_forecast_v1 | 4.00% | 0.0626 | 0.01044 |

Paired weekly-bootstrap 95% interval for top-20 precision change versus v2: [-0.31, +0.33] percentage points.

- Current inventory may contain assets absent in historical years; this is an exploratory spatial comparison.
- 2026 comparison dates were already inspected and are not a fresh final test.
- Targets are reported drainage concerns, not confirmed flooding; count does not establish drainage capacity.
- Only two weather source points cover 604 grids. Calibration uses July-September 2025.

Reproduce: `.venv/Scripts/python.exe src/catch_basin_features.py`, then `.venv/Scripts/python.exe src/train_forecast_drainage_36h.py`.
