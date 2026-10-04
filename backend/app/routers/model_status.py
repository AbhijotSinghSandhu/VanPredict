import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix='/api/v1', tags=['Model research'])


@router.get('/model-status')
def model_status():
    try:
        report = json.loads((ROOT / 'docs/model_evaluation.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(503, 'Model evaluation is unavailable.')
    return {**report, 'artifact_available': (ROOT / 'models/spatial_nextday_v1.joblib').exists(),
            'history_36h': {'artifact_available': (ROOT / 'models/history_36h_v1.joblib').exists(),
                           'backend_inference_enabled': (ROOT / 'models/history_36h_v1.joblib').exists(),
                           'endpoint': '/api/v1/predict', 'status': 'experimental',
                           'uses_forecast_rainfall': False, 'frontend_connected': True},
            'prediction_enabled': False,
            'readiness_note': 'The daily top-20 list underperforms the location baseline. Further evaluation and forecast inputs are needed.'}
