FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
RUN npm install --global pnpm@11.19.0
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /app
COPY backend/requirements-production.txt /app/backend/requirements-production.txt
RUN pip install --no-cache-dir -r backend/requirements-production.txt
COPY backend/app/ /app/backend/app/
COPY backend/serve.py /app/backend/serve.py
COPY src/training_36h.py src/train_history_36h.py /app/src/
COPY docs/model_evaluation.json /app/docs/model_evaluation.json
COPY data/processed/311_gridded.csv /app/data/processed/311_gridded.csv
COPY data/processed/spatial_weather/era5_500m/grid_weather_mapping.csv data/processed/spatial_weather/era5_500m/weather_daily_features.csv.gz data/processed/spatial_weather/era5_500m/weather_hourly_features.csv.gz /app/data/processed/spatial_weather/era5_500m/
# Train at image build time so ignored local model files are not required.
RUN python src/train_history_36h.py
COPY --from=frontend /frontend/dist /app/frontend/dist
COPY --from=frontend /frontend/public/grids /app/frontend/public/grids
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
HEALTHCHECK --start-period=60s --interval=30s --timeout=10s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/ready',timeout=8)"
CMD ["python", "backend/serve.py"]
