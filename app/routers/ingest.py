# app/routers/ingest.py
from fastapi import APIRouter, Depends, BackgroundTasks
from app.models import IngestPayload
from app.middleware.auth import verify_api_key
from app.services.event_processor import process_frame_detections

router = APIRouter()


@router.post("/", summary="Ingest a processed camera frame")
async def ingest_frame(
    payload: IngestPayload,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    """
    Accepts a processed frame payload from an edge inference node.
    Processing (event generation, DB writes, WebSocket broadcast) is handled
    asynchronously in a background task so the edge node gets an immediate response.
    """
    background_tasks.add_task(process_frame_detections, payload)
    return {"status": "accepted", "camera_id": payload.camera_id}
