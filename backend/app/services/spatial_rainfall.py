"""Rainfall features from completed hours; days use America/Vancouver."""

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[3] / "data/processed/spatial_weather/era5_500m"
TIMEZONE = "America/Vancouver"
FEATURES = [
    "rainfall_today", "max_rain_1h", "max_rain_6h", "max_rain_24h",
    "heavy_rain_1h", "heavy_rain_6h", "heavy_rain_24h",
    "rain_duration_hours", "rainfall_previous_day", "rainfall_previous_3d",
    "rainfall_previous_7d",
]
THRESHOLDS = {1: 5.0, 6: 15.0, 24: 25.0}


def create_features(raw):
    """Return hourly as-of features and completed-day retrospective summaries.

    Input is one weather cell with hourly UTC interval-end timestamps. An hour
    ending at midnight belongs to the preceding local day. Rolling windows
    cross midnight; maxima and rain duration reset at local midnight.
    """
    frame = raw.copy().sort_values("datetime").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True))
    if frame.empty or timestamps.has_duplicates or not timestamps.equals(
        pd.date_range(timestamps[0], timestamps[-1], freq="h")
    ):
        raise ValueError("Expected unique, continuous hourly timestamps.")
    rain = pd.to_numeric(frame["rainfall_mm"], errors="raise")
    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError("Rainfall must be finite and nonnegative.")
    frame["datetime"] = timestamps
    frame["datetime_local"] = timestamps.tz_convert(TIMEZONE)
    interval_dates = (timestamps - pd.Timedelta(nanoseconds=1)).tz_convert(TIMEZONE)
    frame["local_date"] = interval_dates.strftime("%Y-%m-%d")
    for hours in (1, 6, 24):
        frame[f"rain_{hours}h"] = rain.rolling(hours, min_periods=hours).sum().round(4)
    day_groups = frame.groupby("local_date", sort=True)
    frame["rainfall_today"] = day_groups["rainfall_mm"].cumsum().round(4)
    for hours in (1, 6, 24):
        frame[f"max_rain_{hours}h"] = day_groups[f"rain_{hours}h"].cummax()
        frame[f"heavy_rain_{hours}h"] = (
            frame[f"max_rain_{hours}h"] >= THRESHOLDS[hours]
        ).astype("int8")
    frame["wet_hour"] = (rain > 0).astype("int8")
    frame["rain_duration_hours"] = frame.groupby("local_date")["wet_hour"].cumsum()

    daily = frame.groupby("local_date", sort=True).agg(
        rainfall_today=("rainfall_mm", "sum"),
        max_rain_1h=("rain_1h", "max"),
        max_rain_6h=("rain_6h", "max"),
        max_rain_24h=("rain_24h", "max"),
        rain_duration_hours=("wet_hour", "sum"),
        observed_hours=("rainfall_mm", "size"),
    )
    dates = pd.DatetimeIndex(pd.to_datetime(daily.index))
    midnights = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=1),
                              freq="D", tz=TIMEZONE)
    daily["expected_hours"] = (
        midnights.to_series().diff().iloc[1:].dt.total_seconds().to_numpy() / 3600
    ).astype(int)
    daily["day_complete"] = daily.observed_hours.eq(daily.expected_hours)
    # A partial first/last day must never count as a complete antecedent day.
    totals = daily.rainfall_today.where(daily.day_complete)
    daily["rainfall_previous_day"] = totals.shift(1)
    for days in (3, 7):
        daily[f"rainfall_previous_{days}d"] = totals.shift(1).rolling(
            days, min_periods=days
        ).sum().round(4)
    for hours in (1, 6, 24):
        daily[f"heavy_rain_{hours}h"] = (
            daily[f"max_rain_{hours}h"] >= THRESHOLDS[hours]
        ).astype("int8")
    daily["weather_cell_id"] = frame.weather_cell_id.iloc[0]
    antecedent = ["rainfall_previous_day", "rainfall_previous_3d", "rainfall_previous_7d"]
    frame = frame.merge(daily[antecedent], left_on="local_date", right_index=True,
                        how="left", validate="many_to_one")
    frame["features_ready"] = frame[FEATURES].notna().all(axis=1)
    daily["features_ready"] = daily.day_complete & daily[FEATURES].notna().all(axis=1)
    return frame.drop(columns="wet_hour"), daily.reset_index()


def load_grid_rainfall(grid_id, start_date=None, end_date=None, frequency="hourly",
                       dataset_dir=DATASET_DIR):
    """Materialize one 500 m grid's features without duplicating shared weather.

    Date filters refer to Vancouver local dates. Daily rows contain completed
    day summaries and are retrospective; hourly rows contain as-of features.
    """
    if frequency not in {"hourly", "daily"}:
        raise ValueError("frequency must be hourly or daily")
    mapping = pd.read_csv(Path(dataset_dir) / "grid_weather_mapping.csv")
    match = mapping.loc[mapping.grid_id.eq(grid_id)]
    if len(match) != 1:
        raise ValueError(f"Unknown or duplicate grid ID: {grid_id}")
    frame = pd.read_csv(Path(dataset_dir) / f"weather_{frequency}_features.csv.gz")
    frame = frame.loc[frame.weather_cell_id.eq(match.weather_cell_id.iloc[0])].copy()
    if start_date:
        frame = frame.loc[frame.local_date.ge(start_date)]
    if end_date:
        frame = frame.loc[frame.local_date.le(end_date)]
    frame.insert(0, "grid_id", grid_id)
    return frame.reset_index(drop=True)


def load_grid_rainfall_panel(start_date, end_date, frequency="hourly", grid_ids=None,
                             dataset_dir=DATASET_DIR):
    """Join weather features onto any/all prediction grids for a date range.

    Keep long histories normalized on disk; materialize only the needed period.
    An hourly panel for the entire history has about 94 million grid-hour rows.
    """
    if frequency not in {"hourly", "daily"}:
        raise ValueError("frequency must be hourly or daily")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must not be after end_date")
    mapping = pd.read_csv(Path(dataset_dir) / "grid_weather_mapping.csv")
    if grid_ids is not None:
        missing = set(grid_ids) - set(mapping.grid_id)
        if missing:
            raise ValueError(f"Unknown grid IDs: {sorted(missing)}")
        mapping = mapping.loc[mapping.grid_id.isin(grid_ids)]
    weather = pd.read_csv(Path(dataset_dir) / f"weather_{frequency}_features.csv.gz")
    weather = weather.loc[weather.local_date.between(start_date, end_date)]
    return mapping[["grid_id", "weather_cell_id"]].merge(
        weather, on="weather_cell_id", validate="many_to_many"
    )
