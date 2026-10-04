"""Prediction-only web service and built frontend, served from one origin."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from app.routers.prediction import router
from app.services.prediction import artifact, sources

ROOT=Path(__file__).resolve().parents[2]
app=FastAPI(title='VanPredict prediction API')
app.include_router(router)


@app.get('/health')
def health():
    try:
        artifact()
        mapping,reports,weather=sources()
        if len(mapping)!=604 or reports.empty or weather.empty:
            raise ValueError('Incomplete prediction data')
    except Exception:
        raise HTTPException(503,'Prediction resources unavailable')
    return {'status':'ready','model_inputs':'prepared_historical_data','uses_live_forecast_for_scoring':False}


app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
