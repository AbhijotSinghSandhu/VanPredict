# Render deployment

The Docker service serves the built frontend and prediction API from one URL.
It exposes prediction routes and `/health`; development and data-processing
routes are excluded. No training or download runs during startup.

1. Run `python deployment/prepare.py` in the project environment. This copies the
   saved model, grid mapping, and minimal public historical inputs to
   `deployment/runtime`. It excludes request addresses and other unused columns.
2. Include `deployment/runtime`, `runtime-manifest.json`, deployment files,
   frontend changes and assets, and `backend/app/production.py` in the repository
   revision used by Render. Check the manifest before publishing.
3. In Render create a Web Service from the repository, using Docker at the root.
   Choose the free instance for a demo and health check `/health`. `render.yaml`
   contains equivalent Blueprint settings. Do not attach a disk; these packaged
   inputs are read-only snapshots.
4. Verify `/health`, load the page, request weather, run the historical model,
   select an area, inspect catch basins, and export a checklist.

The free service sleeps during inactivity. Live weather remains a browser
request to Open-Meteo and does not influence the historical model. This release
does not claim current live risk scoring. Inspection records remain local to
each browser; local localhost records do not transfer to the deployed origin.

Local image test, when Docker is installed:

```sh
docker build -t vanpredict .
docker run --rm -p 10000:10000 vanpredict
```

The image has not been validated until these commands complete. Python inference
dependencies match the environment that created the saved artifact.
