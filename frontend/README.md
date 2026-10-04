# VanPredict frontend

Prototype milestones 1–2: responsive dashboard shell and interactive Vancouver grid map. The map loads 604 real cells from a local GeoJSON snapshot. Drainage and rainfall metrics remain unconnected; no backend is required yet.

## Local development

Requires Node.js 20.19+ or 22.12+ and a package manager.

```sh
pnpm install
pnpm dev
```

Open the local URL printed by Vite. Run `pnpm build` to type-check and create a production build, then `pnpm preview` to serve it.

## Milestone checks

- Desktop and mobile layouts: no horizontal overflow or clipped content.
- Keyboard: skip link, overview link, about buttons, and visible focus.
- About dialog: opens, closes with its buttons or Escape, and returns focus to the trigger.
- Placeholder map and metrics clearly identified; no invented observations.

Area selection is implemented: click a grid cell or focus it and press Enter/Space to highlight it and show its ID, grid size, and clipped area. Clear selection restores the empty panel. Changing grid size clears the old selection. Report history and rainfall remain unconnected.

Next milestone: date filters.

Step 3 checks passed: production build, mouse selection, Enter-key selection, single highlighted cell when switching selection, clear selection, and reset after changing grid size. Verified a full 500 m cell displays 250,000 square metres.

The cell-size selector loads pre-generated 200 m, 500 m, or 1,000 m grids from `public/grids/` and caches them in memory. Regenerate these assets with the project's Python environment: `python frontend/scripts/generate_grids.py`. This uses the existing city boundary and UTM metre-based grid alignment without modifying training data. Street-map tiles come from OpenStreetMap and require internet. Teal shading indicates grid coverage, not risk or report counts. Zoom, keyboard panning, and Fit Vancouver are available; mouse-wheel zoom is disabled to keep page scrolling comfortable.

Grid-size checks passed: 200 m renders 3,581 cells, 500 m renders 604, and 1,000 m renders 168. All sizes cover the same boundary area with unique cell IDs. Switching replaces the previous layer; returning to a loaded size uses the cache. Custom sizes and report aggregation are deferred.

## Verification completed

- TypeScript check and production build passed.
- Browser preview rendered successfully; no console warnings or errors were captured.
- No horizontal page overflow at desktop, 390px, or 320px viewport widths.
- Both About triggers opened the dialog. Escape and the return button closed it; focus returned to the opening trigger.
- Desktop and 390px mobile layouts visually reviewed.

### Step 2 checks

- All 604 grid polygons rendered in the browser; loaded street-map tiles visually verified.
- Zoom in changed the scale from 3 km to 2 km; zoom out restored 3 km.
- Fit Vancouver works, including after resizing to mobile.
- 390px mobile viewport: 340px map height and no horizontal overflow.

### Step 4: date controls

Start/end dates support Apply and Reset, with valid calendar dates and chronological order required. Applied periods persist locally between visits. Draft edits are labelled until applied. Dates represent inclusive Vancouver calendar days; actual report filtering and available-date limits will be connected with historical data next.

Verified: TypeScript and production build; empty/reversed range errors; leap-day range; saved period after reload; pending edits; Reset clears inputs and saved range; 390px mobile layout without horizontal overflow. Data counts remain unconnected.

### Step 5: historical drainage data

Run the backend from the repository root with `python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001`, then run the frontend as above. Vite proxies `/api` to this local backend. A deployed frontend must configure an equivalent API route.

The report card and selected-cell panel now show real prepared 311 counts. Apply filters by inclusive Vancouver calendar dates; Reset uses all available dates. The source contains 40,560 reports from 2009-06-24 to 2026-10-02. This is the span of report dates, not a guarantee of complete reporting coverage. Counts are reassigned spatially for 200m and 1000m grids; 500m uses prepared assignments. API responses contain aggregates only. Prepared files are cached until backend restart.

Verification: four source-based backend tests passed (all grid sizes, same-day counts, empty periods and invalid query parameters). Browser confirmed 40,560 total reports, one report on 2009-06-24 and one in its selected cell, zero for an earlier period, and a coverage notice. TypeScript and production build passed. Loading and unavailable-backend states were verified.

Next milestone: rainfall data and coverage.

### Step 6: historical rainfall

The `/api/v1/rainfall` endpoint connects the compact ERA5 daily source tables. Rainfall is liquid rain in millimetres, excluding snow. City totals sum daily averages weighted by the areas of the 500m polygons; selected cells use the nearest ERA5 source point. All three map grid sizes reuse the same two weather sources and do not imply finer rainfall resolution.

Only complete Vancouver days count as covered. Missing dates remain null, totals are labelled as partial when coverage is incomplete, and wholly unavailable periods display no data. The chart displays up to the last 60 days of the requested period with a keyboard-accessible table. Rainfall uses the same date range as reports; Reset uses the report date span. Weather ends on 2026-09-27, before the final report date.

Verified: four backend tests reconcile area-weighted totals and selected-source totals, check all grid assignments, reject invalid requests, and preserve missing days. Browser checks confirm 35.21mm city average and 31/31 covered days for 2025-06-04 through 2025-07-04; source cell g500_980_10918 has 28mm. An unavailable five-day period shows no data and 0/5 coverage. TypeScript and production build passed. Mobile width checked at 390px.

Next: choose the next map layer or evaluate model readiness before exposing predictions.

### Historical report map layer

Choose Drainage reports or Grid coverage in the map toolbar. Fixed bands show 0, 1, 2–4, 5–9, and 10+ historical reports per cell. Counts are not area-normalized and do not represent flood risk. Hover reveals the exact count; keyboard cell descriptions include counts. A dark selection outline preserves the count colour. While counts are unavailable, the map falls back to uniform geometry with an unavailable label.

Verified legend boundaries; production build and TypeScript; 2025-06-04 to 2025-07-04 reconciles 82 reports across 69 nonzero cells at 500m and 75 at 200m. Switching layers resets shading; selection outline persists; an empty date period has zero shading; 390px mobile controls do not overflow.

Prediction readiness: no saved model artifact currently exists in models/ or the baseline output location. The v0.7 script uses a degree-based grid and same-day target, while the spatial training table uses 500m cells and next-day targets. Train and validate a matching spatial model before exposing prediction mode.

### Spatial model research milestone

A reproducible 500m next-day report model is saved locally in models/spatial_nextday_v1.joblib. Train it with `python src/train_spatial_model.py`; see docs/MODEL_RESEARCH.md for the experiment and limitations. Model artifacts are ignored by Git; the evaluation JSON and training script are versioned.

The dashboard fetches `/api/v1/model-status` and displays the fixed historical holdout comparison and limitations. The model improves average precision, but its daily top-20 shortlist underperforms the location-only baseline. Prediction mode remains unavailable. This is historical research, not a live flood forecast.

Verification: three existing next-day label tests, three new partition/artifact tests, TypeScript and production build. Browser confirmed evaluation values, expandable limitations, and mobile layout without horizontal overflow at 390px.

### Live 36-hour weather forecast

The forecast button requests Open-Meteo directly for central Vancouver (49.2827, -123.1207). Refresh requests the latest available forecast; there is no automatic polling. Historical filters do not change the forecast. No credentials or backend changes are required.

The displayed window starts at the next full hour and includes 36 complete hourly intervals. Rain plus showers gives liquid rainfall in millimetres; timestamps mark interval ends. UTC processing preserves continuity across daylight-saving changes, with Vancouver time used for display. The panel shows hourly rain, precipitation chance, total liquid rain, current temperature estimate, retrieval time, and coverage. Retrieval time is not the weather model issue time.

Missing rain, invalid units, and incomplete/discontinuous coverage produce an error rather than invented zero values. Requests have a 20-second timeout; grid changes and clear cancel pending requests.

This replaces the simulated forecast and removes fixture risk rankings and map highlights. Weather represents one central-city point. Model inference and area risk scores remain unconnected; the map continues to show historical reports or grid coverage.

Verification: four forecast parser tests cover interval selection, rain plus showers, missing data, units, continuity and daylight-saving transitions. Run with node --test frontend/tests/liveForecast.test.mjs. TypeScript, production build, and an actual public API request passed.

### Historical model frontend integration

The separate Model explorer calls POST /api/v1/predict at 06:00 Vancouver time for the chosen date. Scores all 604 cells and highlights the top five; ranked rows select matching map areas. Experimental report estimates, historical window and input age are displayed separately from live weather. Historical filters are independent. Changing the grid or date clears results; only 500 m supports predictions. Requests beyond weather coverage display an error with retry. Map layer switching retains results until cleared.

Verified real backend scores, five highlights, ranked selection, date error/recovery, unsupported grid disabling and mobile layout without overflow. TypeScript and production build passed.
