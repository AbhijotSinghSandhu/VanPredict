from datetime import datetime
from typing import Literal
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from app.services.prediction import prepared_prediction, score, sources, FEATURES

router = APIRouter(prefix='/api/v1',tags=['Experimental predictions'])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    grid_size_m: Literal[500] = 500
    forecast_hours: Literal[36] = 36
    prediction_time: datetime | None = None
    top_k: int = Field(default=20,ge=1,le=604)

    @model_validator(mode='after')
    def validate_time(self):
        if self.prediction_time is not None:
            validate_issue(self.prediction_time)
        return self


def validate_issue(value):
    if value.tzinfo is None:
        raise ValueError('Prediction time must include timezone')
    issue = pd.Timestamp(value).tz_convert('UTC')
    local = issue.tz_convert('America/Vancouver')
    if (local.hour,local.minute,local.second,local.microsecond) != (6,0,0,0):
        raise ValueError('This model supports 06:00 Vancouver prediction times')
    return issue


class CellInputs(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    cell_id: str
    centroid_latitude: float = Field(ge=-90,le=90)
    centroid_longitude: float = Field(ge=-180,le=180)
    rain_past_24h: float = Field(ge=0)
    rain_past_72h: float = Field(ge=0)
    rain_past_168h: float = Field(ge=0)
    reports_past_24h: int = Field(ge=0)
    reports_past_168h: int = Field(ge=0)
    reports_past_720h: int = Field(ge=0)
    historical_reports_per_day: float = Field(ge=0)


class FeatureRequest(PredictionRequest):
    prediction_time: datetime
    cells: list[CellInputs] = Field(min_length=1,max_length=604)


def run(action):
    try:
        return action()
    except FileNotFoundError:
        raise HTTPException(503,'Saved model or prepared inputs are unavailable')
    except ValueError as error:
        raise HTTPException(422,str(error))


@router.post('/predict')
def predict(request:PredictionRequest):
    return run(lambda:prepared_prediction(validate_issue(request.prediction_time) if request.prediction_time else None,request.top_k))


@router.post('/predict/features')
def predict_features(request:FeatureRequest):
    def action():
        issue = validate_issue(request.prediction_time)
        mapping,_,_ = sources()
        valid = mapping.set_index('grid_id')
        rows=[]
        for cell in request.cells:
            if cell.cell_id not in valid.index:
                raise ValueError('Unknown 500 m cell ID')
            location=valid.loc[cell.cell_id]
            if abs(cell.centroid_latitude-location.centroid_latitude)>1e-6 or abs(cell.centroid_longitude-location.centroid_longitude)>1e-6:
                raise ValueError('Coordinates must match the grid centroid')
            if not cell.rain_past_24h<=cell.rain_past_72h<=cell.rain_past_168h or not cell.reports_past_24h<=cell.reports_past_168h<=cell.reports_past_720h:
                raise ValueError('Past totals must increase with window length')
            row=cell.model_dump();row['grid_id']=row.pop('cell_id')
            local=issue.tz_convert('America/Vancouver')
            row.update(month=local.month,day_of_week=local.dayofweek,prediction_hour=6)
            rows.append(row)
        return score(pd.DataFrame(rows),issue,request.top_k,'caller_supplied_history_features')
    return run(action)
