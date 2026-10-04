# VanPredict

Vancouver drainage dashboard with historical 311 reports, spatial rainfall,
live 36-hour weather forecasts, and an experimental historical model explorer.

## Production deployment

Deploy the root `Dockerfile` as one Render Web Service. It builds the React
frontend, trains the historical model, and serves the dashboard and FastAPI
backend together. Use `/ready` as the health check path and leave Docker Command
empty. The server listens on `0.0.0.0` and honors the hosting provider's `PORT`.

See [Production deployment](docs/PRODUCTION.md) for complete instructions.

## Training data and model

The deployment uses real prepared Vancouver 311 drainage reports and compact
ERA5 hourly rainfall from two weather source cells mapped to 604 city grid cells.
Reports span 2009-06-24 through 2026-10-02; stored weather ends on 2026-09-27.

`src/train_history_36h.py` trains on 2019-2021, calibrates on 2022, and evaluates
on 2023-2025 using chronological partitions. The model uses past rain and report
history to estimate a matching drainage report in the next 36 hours at 06:00
Vancouver time. Target windows crossing partition boundaries are removed.
The image saves `models/history_36h_v1.joblib` during its build, so ignored local
model files are not required for deployment. Training never runs on API requests.

Model scoring is historical, not a current flood forecast. Live Open-Meteo
weather is separate. Probabilities remain experimental; the inspected development
years are not a fresh final holdout. See [Model documentation](docs/HISTORY_36H_MODEL.md).

## Local production server

```powershell
python -m venv .venv-production
.venv-production/Scripts/python.exe -m pip install -r backend/requirements-production.txt
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build
.venv-production/Scripts/python.exe src/train_history_36h.py
.venv-production/Scripts/python.exe backend/serve.py
```

Open http://localhost:8080. Live weather and street-map tiles require internet.

## Earlier baseline

`src/baseline_model_7.py` is a separate degree-grid retrospective baseline:

```powershell
python -m pip install -r requirements.txt
python src/baseline_model_7.py
```

Its same-day rainfall inputs and evaluation do not describe the dashboard's
36-hour model. `_archive/` contains earlier work and is not needed for deployment.
