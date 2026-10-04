from datetime import date
from fastapi import APIRouter, HTTPException
from app.services.dashboard_rainfall import rainfall_summary

router = APIRouter(prefix='/api/v1', tags=['Historical rainfall'])


@router.get('/rainfall')
def rainfall(start: date | None = None, end: date | None = None, grid_size: int = 500, cell_id: str | None = None):
    if grid_size not in (200, 500, 1000) or (start is None) != (end is None):
        raise HTTPException(422, 'Provide a supported grid size and both dates, or neither date.')
    if start and end and (start > end or (end - start).days > 36525):
        raise HTTPException(422, 'Choose an ordered date range of at most 100 years.')
    try:
        return rainfall_summary(start, end, grid_size, cell_id)
    except KeyError:
        raise HTTPException(404, 'Cell not found in this grid.')
    except (FileNotFoundError, ValueError):
        raise HTTPException(503, 'Prepared rainfall data is unavailable.')
