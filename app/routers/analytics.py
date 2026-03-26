# backend/app/routers/analytics.py
from fastapi import APIRouter, HTTPException
from app.services.prediction_engine import SurfacePredictor
from app.services.traffic_control import TrafficController

router = APIRouter()

@router.get("/predict/{camera_id}")
async def get_surface_prediction(camera_id: str):
    """
    Get the predicted time-to-failure for a specific road segment.
    """
    try:
        prediction = await SurfacePredictor.forecast_degradation(camera_id)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/traffic/override/{camera_id}")
async def request_traffic_override(camera_id: str, incident_type: str = "MANUAL"):
    """
    Force clear a green wave for the current lane.
    """
    return TrafficController.request_emergency_override(camera_id, incident_type)

@router.post("/traffic/release/{camera_id}")
async def release_traffic_override(camera_id: str):
    """
    Return the intersection to automatic mode.
    """
    return TrafficController.release_override(camera_id)
