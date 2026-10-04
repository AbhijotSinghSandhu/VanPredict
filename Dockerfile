FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
RUN npm install --global pnpm@9
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COPY deployment/requirements.txt ./deployment/requirements.txt
RUN pip install --no-cache-dir -r deployment/requirements.txt
COPY backend/ ./backend/
COPY src/training_36h.py ./src/training_36h.py
COPY deployment/runtime/models/ ./models/
COPY deployment/runtime/data/ ./data/
COPY --from=frontend /build/frontend/dist ./frontend/dist/
RUN useradd --create-home appuser
USER appuser
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn app.production:app --app-dir backend --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
