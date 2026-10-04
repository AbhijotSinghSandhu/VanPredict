"""Prepare only the model and public historical inputs required for inference."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deployment/runtime'
MODEL='models/history_36h_v1.joblib'
mapping='data/processed/spatial_weather/era5_500m/grid_weather_mapping.csv'
weather='data/processed/spatial_weather/era5_500m/weather_hourly_features.csv.gz'
reports='data/processed/311_gridded.csv'


def main():
    for relative in (MODEL,mapping):
        target=OUT/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    pd.read_csv(ROOT/reports,usecols=['grid_id','service_request_open_timestamp','service_request_type']).to_csv(OUT/reports,index=False)
    pd.read_csv(ROOT/weather,usecols=['weather_cell_id','datetime','rainfall_mm']).to_csv(OUT/weather,index=False,compression='gzip')
    manifest={relative:{'bytes':(OUT/relative).stat().st_size,'sha256':hashlib.sha256((OUT/relative).read_bytes()).hexdigest()} for relative in (MODEL,mapping,weather,reports)}
    (ROOT/'deployment/runtime-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
