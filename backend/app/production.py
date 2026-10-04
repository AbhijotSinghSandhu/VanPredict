"""Serve the compiled dashboard and read-only dashboard API together."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routers.reports import router as reports_router
from app.routers.rainfall import router as rainfall_router
from app.routers.model_status import router as model_status_router
from app.routers.prediction import router as prediction_router

ROOT = Path(__file__).resolve().parents[2]
WEATHER = ROOT / "data/processed/spatial_weather/era5_500m"


def create_app(dist: Path = ROOT / "frontend/dist") -> FastAPI:
    app = FastAPI(title="VanPredict", docs_url=None, redoc_url=None, openapi_url=None)
    for router in (reports_router, rainfall_router, model_status_router, prediction_router):
        app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/ready")
    def ready():
        required = [
            dist / "index.html",
            ROOT / "data/processed/311_gridded.csv",
            WEATHER / "grid_weather_mapping.csv",
            WEATHER / "weather_daily_features.csv.gz",
            WEATHER / "weather_hourly_features.csv.gz",
            ROOT / "docs/model_evaluation.json",
            ROOT / "models/history_36h_v1.joblib",
            *(dist / f"grids/vancouver_{size}m.geojson" for size in (200, 500, 1000)),
            *(ROOT / f"frontend/public/grids/vancouver_{size}m.geojson" for size in (200, 500, 1000)),
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return JSONResponse(status_code=503, content={"status": "unavailable", "missing": missing})
        from app.services.prediction import artifact
        try:
            artifact()
        except Exception:
            return JSONResponse(status_code=503, content={"status": "unavailable", "detail": "Model could not load"})
        return {"status": "ready", "model": "experimental_history_36h"}

    # Unknown API requests must remain errors rather than returning the dashboard.
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api(path: str):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return app


app = create_app()
