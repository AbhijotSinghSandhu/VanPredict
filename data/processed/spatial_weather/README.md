# Spatial rainfall for VanPredict's 500 m grid

The `era5_500m` dataset uses hourly **liquid rain**, downloaded as Open-Meteo's
`rain` variable. `precipitation_mm` is retained separately and includes snow
water equivalent. The original city-wide rainfall files are unchanged.

ERA5 is served on a 0.25 degree grid. Each 500 m polygon is linked to the
nearest ERA5 point to its area-weighted longitude/latitude centroid. These
are model estimates, not independent 500 m measurements. Requesting smaller
prediction cells does not add source rainfall detail. Returned source
coordinates are checked against the mapped weather coordinates on download.

## Files in `era5_500m`

| File | Contents |
| --- | --- |
| `grid_weather_mapping.csv` | One row per 500 m grid, with centroid, source coordinates and weather-cell ID |
| `weather_hourly_features.csv.gz` | Full history, one row per source weather cell and completed hour |
| `weather_daily_features.csv.gz` | Full history, completed local-day summaries per weather cell |
| `grid_daily_features.csv.gz` | Full daily table expanded to all 500 m grids |
| `grid_hourly_first_week.csv.gz` | First week's hourly table expanded to all grids |
| `manifest.json` | Coverage, row counts, units, source and feature definitions |
| `checksums.json` | SHA-256 checksums for the published rainfall tables and grid lookup |

## Using the GitHub training dataset

The full rainfall history is versioned as **compact source tables plus the grid
lookup**, about 5.6 MB altogether. The 70 MB expanded daily export, expanded
hourly example, raw download cache and generated map are local outputs. They
are not needed for training: the loader reconstructs each grid's rainfall from
the source tables, including grids with no historical reports.

From Git Bash after cloning or switching to this branch:

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_spatial*.py' -v
python notebooks/audit_spatial_rainfall.py --compact
```

The repository already uses Git LFS for CSV and GeoJSON files; `git lfs pull`
retrieves the grid mapping, 311 reports and existing grid geometry. The compressed
rainfall source tables are ordinary Git files and download with the repository.

Generate a model training table without downloading weather again:

```bash
python src/spatial_training_data.py \
  --start-date 2009-06-24 --end-date 2026-09-27
```

This writes `data/processed/spatial_training/grid_daily_training.csv.gz`.
It combines all 604 grid cells with all eleven requested rainfall fields,
centroid coordinates, calendar fields, and a next-day drainage report label.
Feature dates describe completed Vancouver days; `target_date` is the next
calendar day. The target counts Street Surface Water Flooding and Catch Basin
Concern reports, matching the existing baseline's categories. A negative label
means no matching report, not proven absence of flooding. The builder checks
that all target dates lie within the observed report span, which is only a
proxy for archive coverage, and rejects missing rainfall instead of filling
it with zero. Audit reporting completeness before interpreting zero labels.

For a smaller experiment, pass a shorter date range. Within the repository:

```python
from src.spatial_training_data import build_training_table, TRAINING_FEATURES

table = build_training_table("2025-11-01", "2025-11-30")
X = table[TRAINING_FEATURES]
y = table["is_drainage_problem_next_day"]
```

Split by `target_date` chronologically. This is a new next-day training table;
`src/baseline_model_7.py` still uses the older degree-based grid and same-day
target and must not be assumed to consume this dataset automatically. The
training table contains only rain, location and calendar predictors; terrain
and drainage-history features can be joined separately on the same 500 m IDs.

## Attribution and redistribution

Weather data is provided by [Open-Meteo](https://open-meteo.com/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source is
[ECMWF / Copernicus ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels),
accessed through Open-Meteo's historical API. This dataset derives accumulated
rainfall, maxima, flags, wet-hour counts and antecedent totals from hourly liquid
rain and assigns source grid points to Vancouver prediction cells. Preserve
this attribution when redistributing the rainfall tables or derived features.
See [Open-Meteo's terms](https://open-meteo.com/en/terms) for data attribution and
separate API service conditions. This notice applies to weather data; existing
Vancouver grid geometry and 311 reports retain their original source terms.

The full hourly dataset is stored as two linked tables to avoid repeating
identical weather histories hundreds of times. Join hourly weather to the
mapping on `weather_cell_id` to obtain one row per `grid_id` and `datetime`.
Hourly timestamps are UTC **interval ends**: a value at 10:00 covers 09:00–10:00.
`datetime_local` retains the Vancouver UTC offset to distinguish repeated
hours when daylight saving ends. `local_date` is the date of the rainfall
interval: the hour ending at local midnight belongs to the preceding day.

## Feature definitions

Hourly fields use completed observations through the row's timestamp.
Daily fields use all hours in that completed local day.

| Feature | Definition |
| --- | --- |
| `rainfall_today` | Liquid rain total since the start of the local day, mm |
| `max_rain_1h` | Largest hourly amount encountered in that local day, mm |
| `max_rain_6h` | Largest trailing six-hour total encountered in that local day, mm |
| `max_rain_24h` | Largest trailing 24-hour total encountered in that local day, mm |
| `heavy_rain_1h` | 1 if `max_rain_1h >= 5`, otherwise 0 |
| `heavy_rain_6h` | 1 if `max_rain_6h >= 15`, otherwise 0 |
| `heavy_rain_24h` | 1 if `max_rain_24h >= 25`, otherwise 0 |
| `rain_duration_hours` | Number of completed hours with liquid rain > 0 mm in that local day |
| `rainfall_previous_day` | Total for the preceding completed local calendar day, mm |
| `rainfall_previous_3d` | Total for the preceding three completed local calendar days, mm |
| `rainfall_previous_7d` | Total for the preceding seven completed local calendar days, mm |

Trailing six- and 24-hour windows cross midnight. A new day's six-hour heavy
flag can therefore be positive even if the current hour is dry. Calendar days
can have 23, 24 or 25 hours; previous-day features follow actual local days.
The downloader fetches nine warm-up days before the requested period, verifies
hourly continuity and rejects missing values or incomplete feature history.
Thresholds match the existing project and are not official warning criteria.

For real-time prediction, use the hourly as-of features. Complete-day summaries
are retrospective and would reveal later rainfall if used to predict earlier
events on the same day. Historical reanalysis also differs from forecasts
available at prediction time; a future forecast evaluation needs archived
forecast inputs.

## Loading hourly features for a grid or period

```python
from backend.app.services.spatial_rainfall import (
    load_grid_rainfall, load_grid_rainfall_panel,
)

one_grid = load_grid_rainfall(
    "g500_980_10918", "2025-11-01", "2025-11-30", frequency="hourly"
)
all_grids = load_grid_rainfall_panel(
    "2025-11-01", "2025-11-07", frequency="hourly"
)
```

To rebuild or extend the dataset, run `notebooks/build_spatial_rainfall.py`
with `--start-date` and `--end-date`. Raw requests are cached in
`data/weather/spatial_era5_raw`; identical requests reuse their cache.
The default end date is seven days before the current Vancouver date.

Source: <https://open-meteo.com/en/docs/historical-weather-api>.
Open-Meteo documents `rain` as liquid precipitation of the preceding hour,
`precipitation` as rain plus snow, and ERA5 spatial spacing as 0.25 degrees.
Open-Meteo's ERA5-Land option does not supply finer-resolution precipitation.
