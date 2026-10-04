"""Weather transitions derived only from the exact archived forecast used by a row."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_training_36h import ARCHIVE, _location_records, features_for_location

TRANSITION_FEATURES=['forecast_rain_onset_hours','forecast_longest_wet_spell_hours',
                     'forecast_longest_dry_spell_hours','forecast_liquid_peak_6h_mm',
                     'forecast_heavy_hours_2mm','forecast_temperature_change_c',
                     'forecast_freezing_crossings','forecast_below_freezing_hours']
INTERACTION_FEATURES=['rain_after_dry_72h_mm','rain_6h_after_dry_72h',
                      'past_72h_plus_forecast_36h_mm']


def longest_spell(mask):
    longest=current=0
    for value in mask:
        current=current+1 if value else 0
        longest=max(longest,current)
    return longest


def transition_features(payload,issue):
    features_for_location(payload,issue)  # Require the same complete 36 forecast intervals.
    issue=pd.Timestamp(issue).tz_convert('UTC')
    hourly=payload['hourly']
    times=pd.to_datetime(hourly['time'],unit='s',utc=True)
    indices=np.flatnonzero((times>issue)&(times<=issue+pd.Timedelta(hours=36)))
    indices=indices[np.argsort(times[indices].asi8)]
    rain=np.asarray(hourly['rain'],dtype=float)[indices]+np.asarray(hourly['showers'],dtype=float)[indices]
    temperature=np.asarray(hourly['temperature_2m'],dtype=float)[indices]
    wet=rain>=.1
    onset=np.flatnonzero(wet)
    below=temperature<0
    return {'forecast_rain_onset_hours':int(onset[0]+1) if len(onset) else 37,
            'forecast_longest_wet_spell_hours':longest_spell(wet),
            'forecast_longest_dry_spell_hours':longest_spell(~wet),
            'forecast_liquid_peak_6h_mm':float(pd.Series(rain).rolling(6).sum().max()),
            'forecast_heavy_hours_2mm':int((rain>=2).sum()),
            'forecast_temperature_change_c':float(temperature[-1]-temperature[0]),
            'forecast_freezing_crossings':int((below[1:]!=below[:-1]).sum()),
            'forecast_below_freezing_hours':int(below.sum())}


def add_transitions(table,mapping):
    points=mapping[['weather_cell_id','weather_latitude','weather_longitude']].drop_duplicates().sort_values('weather_cell_id')
    if len(points)!=2:raise ValueError('Expected the configured two forecast source points')
    records=[]
    for (stamp,run),group in table.groupby(['prediction_time','forecast_run'],sort=True):
        run=pd.Timestamp(run)
        cache=ARCHIVE/f'{run:%Y-%m-%dT%H}.json'
        if not cache.exists():raise ValueError(f'Missing exact initialized forecast cache: {cache.name}')
        payloads=_location_records(json.loads(cache.read_text(encoding='utf-8')))
        if len(payloads)!=len(points):raise ValueError('Incorrect archived source count')
        for (_,point),payload in zip(points.iterrows(),payloads):
            values=features_for_location(payload,pd.Timestamp(stamp))
            stored=group.loc[group.weather_cell_id.eq(point.weather_cell_id)]
            if stored.empty:raise ValueError('Missing forecast source in dataset')
            # Catch cache/schema mismatches before attaching new inputs.
            for column in ('rain_36h','showers_36h','snowfall_36h_cm','minimum_temperature_36h_c'):
                if not np.allclose(stored[column],values[column],rtol=1e-7,atol=1e-8):
                    raise ValueError(f'Cached forecast disagrees with dataset: {stamp}, {column}')
            records.append({'prediction_time':stamp,'weather_cell_id':point.weather_cell_id,
                            **transition_features(payload,pd.Timestamp(stamp))})
    derived=pd.DataFrame(records)
    result=table.merge(derived,on=['prediction_time','weather_cell_id'],how='left',validate='many_to_one')
    if len(result)!=len(table):raise ValueError('Transition join changed row count')
    liquid=result.rain_36h+result.showers_36h
    dry=result.rain_past_72h<=.1
    result['rain_after_dry_72h_mm']=np.where(dry,liquid,0.)
    result['rain_6h_after_dry_72h']=(dry & ((result.rain_6h+result.showers_6h)>=1)).astype(int)
    result['past_72h_plus_forecast_36h_mm']=result.rain_past_72h+liquid
    if not np.isfinite(result[TRANSITION_FEATURES+INTERACTION_FEATURES].to_numpy()).all():
        raise ValueError('Nonfinite transition features')
    return result
