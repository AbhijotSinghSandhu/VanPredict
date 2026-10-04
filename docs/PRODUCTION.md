# Production deployment

The production server serves the compiled React dashboard and dashboard API from
one origin. Vite is used only to build assets. Data preparation routes are absent.
The historical model remains experimental; hosting it does not make it a live
flood forecast. Live weather and map tiles require browser internet access.

## Run locally on Windows

From the repository root:

```powershell
C:/Python312/python.exe -m venv .venv-production
.venv-production/Scripts/python.exe -m pip install -r backend/requirements-production.txt
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build
.venv-production/Scripts/python.exe src/train_history_36h.py
.venv-production/Scripts/python.exe backend/serve.py
```

Open http://localhost:8080. Training creates the missing model directory and
artifact and updates the history model evaluation. Training runs before serving,
never during server startup or dashboard requests. Set `PORT` to change the port.
One worker limits duplicated in-memory data caches.

In another terminal, test the running service:

```powershell
.venv-production/Scripts/python.exe scripts/smoke_production.py http://localhost:8080
```

`/health` checks server liveness. `/ready` checks required packaged files and loads
the trained model, returning 503 when it is missing or invalid. The smoke test
additionally exercises real data loading and inference.

## Docker

```powershell
docker build -t vanpredict:submission .
docker run --rm -p 8080:8080 --name vanpredict vanpredict:submission
```

The image builds the dashboard and trains the historical model using packaged
compact data. It does not depend on ignored local artifacts. The build can take
several minutes. Python 3.12 and Node 22 are used; production Python dependencies
and pnpm are pinned. The server runs as a non-root user.

## Deploy on Render

1. Commit and push the production files and required compact data to your repository.
2. Create a Render Web Service from that repository, using the Docker runtime
   and the root `Dockerfile`. Keep Docker Command empty to use the image command.
3. Set the health check path to `/ready`. The server honors Render's `PORT`.
4. Choose a plan with enough memory for the cached weather/report data and model.
   Measure memory and latency during the hosted test before deciding capacity.
5. When deployment succeeds, run the smoke script against the assigned HTTPS URL.
6. In the hosted dashboard, test date Apply/Reset, all three grid sizes, cell
   selection, rainfall coverage, forecast refresh, historical scoring and ranked
   area selection. Repeat at mobile width and check browser errors/network failures.

Required repository data:

- `data/processed/311_gridded.csv`
- `data/processed/spatial_weather/era5_500m/grid_weather_mapping.csv`
- `data/processed/spatial_weather/era5_500m/weather_daily_features.csv.gz`
- `data/processed/spatial_weather/era5_500m/weather_hourly_features.csv.gz`
- `frontend/public/grids/vancouver_{200,500,1000}m.geojson`
- `docs/model_evaluation.json`

Model training is deterministic for the fixed code, inputs and dependency versions;
Linux and Windows numerical outputs may differ slightly. Run smoke tests after
each image build. A successful localhost test does not verify hosted HTTPS,
resource limits or external browser requests; those require the deployed test.

Render references: https://render.com/docs/docker,
https://render.com/docs/web-services, https://render.com/docs/health-checks.
