from fastapi import FastAPI

from app.services.data_loader import load_311, load_rainfall
from app.services.data_quality import (
    check_data_quality,
    check_311_quality,
    check_rainfall_quality
)
from app.services.clean_311 import clean_311

from app.services.build_grid import build_grid
from app.services.audit_grid import audit_grid

from fastapi.responses import HTMLResponse
from .services.grid_map import create_grid_map

from .services.audit_rainfall import audit_rainfall

from .services.inspect_lidar import inspect_lidar


app = FastAPI(
    title="VANPREDICT API",
    description="AI-powered Vancouver drainage and flooding prediction system",
    version="1.0.0"
)


# ==============================
# HOME
# ==============================

@app.get("/")
def root():

    return {
        "project": "VANPREDICT",
        "status": "online"
    }


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==============================
# CHECK DATA
# ==============================

@app.get("/data/check")
def check_data():

    df311 = load_311()
    rainfall = load_rainfall()

    return {
        "311_rows": len(df311),
        "rainfall_rows": len(rainfall),

        "311_columns": df311.columns.tolist(),

        "rainfall_columns": rainfall.columns.tolist()
    }

@app.get("/data/quality")
def data_quality():

    df311 = load_311()
    rainfall = load_rainfall()

    report_311 = check_data_quality(
        df311,
        "311 Drainage"
    )

    check_311_quality(df311)

    report_rainfall = check_data_quality(
        rainfall,
        "Rainfall"
    )

    check_rainfall_quality(rainfall)

    return {
        "311": report_311,
        "rainfall": report_rainfall
    }

@app.post("/data/clean-311")
def clean_311_endpoint():
    return clean_311()


@app.post("/data/build-grid")
def build_grid_endpoint():
    return build_grid()

@app.get("/data/audit-grid")
def audit_grid_endpoint():
    return audit_grid()

@app.get("/data/grid-map", response_class=HTMLResponse)
def grid_map_endpoint():
    return HTMLResponse(content=create_grid_map())



@app.get("/data/audit-rainfall")
def audit_rainfall_endpoint(naive_timezone: str | None = None):
    return audit_rainfall(naive_timezone=naive_timezone)



@app.get("/data/inspect-lidar")
def inspect_lidar_endpoint():
    return inspect_lidar()