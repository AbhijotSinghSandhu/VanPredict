from datetime import date
from fastapi import APIRouter, HTTPException, Query
from app.services.reports import summarize

router = APIRouter(prefix='/api/v1', tags=['Historical drainage'])


@router.get('/reports')
def reports(start: date | None = None, end: date | None = None,
            grid_size: int = Query(default=500)):
    if grid_size not in (200, 500, 1000):
        raise HTTPException(422, 'Choose a 200, 500, or 1000 metre grid.')
    if (start is None) != (end is None):
        raise HTTPException(422, 'Provide both start and end dates, or neither.')
    if start and end and start > end:
        raise HTTPException(422, 'End date must be on or after start date.')
    try:
        return summarize(grid_size, start, end)
    except FileNotFoundError:
        raise HTTPException(503, 'Prepared drainage data is unavailable.')
