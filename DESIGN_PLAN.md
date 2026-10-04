# VanPredict — frontend and backend plan

## Product direction

A Vancouver drainage risk dashboard for exploring where drainage incidents have been reported, how rainfall varies across the city, and what a validated model can tell us about future risk.

Start with one polished dashboard. Display historical observations first; enable predictions only after the model's inputs, output meaning, and validation are established. A 311 report is a reported drainage incident, not a confirmed flood measurement.

## Frontend design

Desktop layout:

```text
┌──────────────────────────────────────────────────────────────┐
│ VanPredict     Vancouver drainage risk          Data updated │
├──────────────────────────────────────────────────────────────┤
│ Historical / Prediction     Date range     Layer selector    │
├───────────────────────────────────────┬──────────────────────┤
│                                       │ Selected area        │
│                                       │ Report count         │
│          Vancouver map                │ Rainfall             │
│          Select a grid cell           │ Risk, when available │
│                                       │ Data coverage        │
│          Legend + zoom                │ Trend chart          │
├───────────────────────────────────────┴──────────────────────┤
│ Coverage and limitations • sources • model version           │
└──────────────────────────────────────────────────────────────┘
```

- Use a light neutral background, dark navy text, teal accents, and restrained amber/red for risk. Pair colours with text labels.
- Make the map the primary view, with a collapsible details panel. On mobile, place details beneath the map.
- Layers: drainage report counts, rainfall coverage, and predicted risk when available. Elevation is a later addition once spatial coverage is suitable.
- Clicking a grid cell opens its report history, rainfall summary, and data availability. Clearly show the active date range and units.
- Use labels such as “Historical reports”, “Model estimate”, and “Unavailable”. Do not show invented probabilities or “live” status.
- Include loading, empty, missing coverage, and request failure states from the first version.
- Start with grid selection; add address search later if needed.

## Application structure

Proposed frontend: React, TypeScript, Vite, and a map library such as Leaflet. Reuse the existing Vancouver grid GeoJSON. Confirm dependencies during implementation.

Keep the existing Python/FastAPI backend. Separate HTTP routes, response schemas, data access, aggregation, and model inference. Existing cleaning, grid-building, and training operations should run as explicit preparation tasks rather than on dashboard requests.

```text
frontend/src/
  components/     Map, filters, legend, area details, status states
  pages/          Dashboard
  services/       Typed API client
  styles/         Shared design tokens and responsive layout

backend/app/
  main.py         App configuration and route registration
  routers/        Dashboard, grid, rainfall, predictions
  schemas/        Validated request and response definitions
  services/       Aggregation and inference logic
  repositories/   Access to prepared datasets
```

## Backend contract

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health |
| `GET /api/v1/metadata` | Available date ranges, layers, coverage, sources, and model status |
| `GET /api/v1/grid` | Grid geometry with stable cell identifiers |
| `GET /api/v1/summary?start=...&end=...` | Counts and coverage for the selected period |
| `GET /api/v1/cells/{id}?start=...&end=...` | Selected cell aggregates and history |
| `GET /api/v1/rainfall?start=...&end=...` | Rainfall summaries with units and coverage |
| `POST /api/v1/predictions` | Later: validated inference request with an explicit time horizon |

Separate mostly static geometry from date-dependent values so map interactions do not repeatedly transfer the full grid. Validate dates and cell identifiers, limit query sizes, and return consistent errors. Use Vancouver time for displayed periods and explicit timezone-aware timestamps in API responses.

Prediction responses must include model version, input timestamp, horizon, output definition, and coverage limitations. Do not interpret classifier scores as calibrated probabilities without validation.

## Data and storage

For the first local version, reuse prepared CSV/GeoJSON and spatial datasets through a single data access layer. Cache suitable read-only aggregates and avoid loading large datasets on every request. No database is needed for the first dashboard.

Introduce PostgreSQL/PostGIS when deployment, data volume, or query performance warrants it. Do not load training scripts from application startup: the current model scripts execute training at module level.

## Build sequence

1. **Confirm the data contract.** Inspect prepared schemas, grid identifiers, date coverage, spatial rainfall coverage, and available model artifacts. Define what each metric means.
2. **Create the frontend shell.** Build the dashboard layout, responsive map, filters, details panel, and state handling using explicitly labelled sample data.
3. **Connect historical data.** Implement metadata, geometry, summary, and cell detail endpoints. Replace sample data with actual prepared datasets and verify counts against source files.
4. **Add rainfall.** Connect rainfall summaries, dates, units, and coverage indicators. Verify the frontend and backend use the same time boundaries.
5. **Integrate a validated model.** Extract inference into a service, match training-time feature construction, evaluate output meaning and calibration, and expose prediction mode only when ready.
6. **Prepare for use.** Add configuration, permitted frontend origins, logging, setup instructions, meaningful API checks, and an end-to-end dashboard check.

## First milestone

A locally running frontend and backend where the user can view the Vancouver grid, select a date range and grid cell, and inspect real historical drainage reports and data coverage. Prediction mode is visibly unavailable until inference is supported.

Excluded from the first milestone: accounts, administration screens, alerts, automatic data ingestion, and production deployment.
