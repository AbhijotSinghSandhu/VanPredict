"""History-only 36-hour report dataset; no future weather or model fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = 'America/Vancouver'
CATEGORIES = {'Street Surface Water Flooding Case', 'Catch Basin Concern Case'}
FEATURES = ['centroid_latitude', 'centroid_longitude', 'month', 'day_of_week',
            'prediction_hour', 'rain_past_24h', 'rain_past_72h', 'rain_past_168h',
            'reports_past_24h', 'reports_past_168h', 'reports_past_720h',
            'historical_reports_per_day']


def utc(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError('Coverage timestamps must include a timezone')
    return timestamp.tz_convert('UTC')


def prediction_times(start, end):
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError('Start date must precede end date')
    # Local dates, rather than 24-hour increments, keep the issue hour at 06:00.
    return pd.DatetimeIndex([pd.Timestamp(f'{day:%Y-%m-%d} 06:00', tz=TIMEZONE)
                             for day in pd.date_range(start, end)]).tz_convert('UTC')


def build_table(mapping, reports, weather, times, coverage_start, coverage_end,
                publication_lag_hours=24, include_targets=True):
    """Coverage is [start,end); lag is an assumed availability delay, not audited.

    Rain hours are interval-end timestamps. Report history is strictly before
    issue time minus lag; labels use [issue, issue+36 elapsed hours).
    """
    first, last = utc(coverage_start), utc(coverage_end)
    if first >= last or publication_lag_hours < 0:
        raise ValueError('Invalid coverage or publication lag')
    if mapping.empty or mapping.grid_id.duplicated().any():
        raise ValueError('Expected unique nonempty grid mapping')
    if not set(reports.grid_id).issubset(set(mapping.grid_id)):
        raise ValueError('Report grid IDs do not match mapping')
    times = pd.DatetimeIndex([utc(t) for t in times])
    if times.empty or times.has_duplicates:
        raise ValueError('Expected unique nonempty prediction times')
    report_times = pd.to_datetime(reports.service_request_open_timestamp,
                                  format='mixed', utc=True, errors='raise')
    if report_times.isna().any():
        raise ValueError('Missing report timestamp')
    chosen = reports.service_request_type.isin(CATEGORIES)
    report_times = report_times.dt.as_unit('ns')
    by_grid = {grid: np.sort(report_times[chosen & reports.grid_id.eq(grid)].astype('int64').to_numpy())
               for grid in mapping.grid_id}
    rain_sources = {}
    for source, group in weather.groupby('weather_cell_id'):
        stamps = pd.DatetimeIndex(pd.to_datetime(group.datetime, utc=True, errors='raise'))
        values = pd.to_numeric(group.rainfall_mm, errors='raise').to_numpy()
        order = np.argsort(stamps.asi8)
        stamps, values = stamps[order], values[order]
        if stamps.has_duplicates or stamps.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid hourly weather')
        rain_sources[source] = pd.Series(values, index=stamps)
    rows = []
    for issue in times:
        end = issue + pd.Timedelta(hours=36)
        cutoff = issue - pd.Timedelta(hours=publication_lag_hours)
        if issue < first or (include_targets and end > last) or cutoff > last or cutoff - pd.Timedelta(hours=720) < first:
            raise ValueError('Prediction window or report history outside declared coverage')
        local = issue.tz_convert(TIMEZONE)
        rain_features = {}
        for source in mapping.weather_cell_id.unique():
            if source not in rain_sources:
                raise ValueError('Missing weather source')
            features = {}
            for hours in (24, 72, 168):
                expected = pd.date_range(issue - pd.Timedelta(hours=hours-1), issue, freq='h')
                values = rain_sources[source].reindex(expected)
                if values.isna().any():
                    raise ValueError('Incomplete past rainfall coverage')
                features[f'rain_past_{hours}h'] = float(values.sum())
            rain_features[source] = features
        for cell in mapping.itertuples(index=False):
            events = by_grid[cell.grid_id]
            count = lambda a, b: int(np.searchsorted(events, b.value, side='left') -
                                     np.searchsorted(events, a.value, side='left'))
            target_count = count(issue, end) if include_targets else None
            history = {f'reports_past_{hours}h': count(cutoff-pd.Timedelta(hours=hours), cutoff)
                       for hours in (24, 168, 720)}
            rows.append(dict(grid_id=cell.grid_id, prediction_time=issue.isoformat(),
                             prediction_end=end.isoformat(), history_cutoff=cutoff.isoformat(),
                             centroid_latitude=cell.centroid_latitude,
                             centroid_longitude=cell.centroid_longitude,
                             month=local.month, day_of_week=local.dayofweek,
                             prediction_hour=local.hour, **rain_features[cell.weather_cell_id],
                             **history, historical_reports_per_day=count(first, cutoff)/((cutoff-first).total_seconds()/86400),
                             **({'report_count_36h':target_count, 'target_36h':int(target_count > 0)} if include_targets else {})))
    table = pd.DataFrame(rows)
    if not np.isfinite(table[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite training inputs')
    return table


def partition(table, start, end):
    """Retain only windows wholly inside a partition, purging crossing labels."""
    starts = pd.to_datetime(table.prediction_time, utc=True)
    ends = pd.to_datetime(table.prediction_end, utc=True)
    return table.loc[(starts >= utc(start)) & (ends <= utc(end))].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--coverage-start', required=True, help='Explicit timezone-aware report coverage start')
    parser.add_argument('--coverage-end', required=True, help='Exclusive report coverage end')
    parser.add_argument('--publication-lag-hours', type=float, default=24)
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/spatial_training/history_36h.csv.gz')
    args = parser.parse_args()
    dataset = ROOT/'data/processed/spatial_weather/era5_500m'
    table = build_table(pd.read_csv(dataset/'grid_weather_mapping.csv'),
                        pd.read_csv(ROOT/'data/processed/311_gridded.csv'),
                        pd.read_csv(dataset/'weather_hourly_features.csv.gz', usecols=['weather_cell_id','datetime','rainfall_mm']),
                        prediction_times(args.start_date, args.end_date), args.coverage_start,
                        args.coverage_end, args.publication_lag_hours)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.tmp')
    table.to_csv(temporary, index=False, compression='gzip')
    temporary.replace(args.output)
    metadata = {'version': 'history-36h-v1', 'grid_size_m': 500,
                'issue_hour_local': 6, 'timezone': TIMEZONE, 'horizon_hours': 36,
                'features': FEATURES, 'categories': sorted(CATEGORIES),
                'coverage_start_assumption': args.coverage_start,
                'coverage_end_assumption': args.coverage_end,
                'publication_lag_hours_assumption': args.publication_lag_hours,
                'historical_weather': 'retrospective ERA5, not archived as-issued observations',
                'future_forecast_features': False, 'rows': len(table),
                'positive_windows': int(table.target_36h.sum()),
                'start_date': args.start_date, 'end_date': args.end_date}
    args.output.with_name(args.output.name+'.metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {len(table):,} rows; {table.target_36h.sum():,} positive windows to {args.output}')
    print('History-only research data. Coverage and publication lag are caller assumptions, not audited availability.')


if __name__ == '__main__':
    main()
