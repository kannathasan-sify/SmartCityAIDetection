# app/routers/events.py
from fastapi import APIRouter, Query
from app.services.supabase_client import supabase

router = APIRouter()

VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_TYPES = {"congestion", "incident", "pedestrian_surge", "parking", "infrastructure"}


@router.get("/", summary="List road events with optional filters")
async def list_events(
    camera_id: str | None = Query(None, description="Filter by camera ID"),
    event_type: str | None = Query(None, description="Filter by event type"),
    severity: str | None = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200),
    resolved: bool | None = Query(None, description="Filter by resolved status"),
):
    query = supabase.table("road_events").select("*").order("created_at", desc=True).limit(limit)

    if camera_id:
        query = query.eq("camera_id", camera_id)
    if event_type and event_type in VALID_TYPES:
        query = query.eq("event_type", event_type)
    if severity and severity in VALID_SEVERITIES:
        query = query.eq("severity", severity)
    if resolved is not None:
        query = query.eq("resolved", resolved)

    result = query.execute()
    return result.data or []


@router.get("/{event_id}", summary="Get a single event by ID")
async def get_event(event_id: str):
    result = supabase.table("road_events").select("*").eq("id", event_id).single().execute()
    return result.data


@router.patch("/{event_id}/resolve", summary="Mark an event as resolved")
async def resolve_event(event_id: str):
    result = (
        supabase.table("road_events")
        .update({"resolved": True})
        .eq("id", event_id)
        .execute()
    )
    return {"status": "resolved", "event_id": event_id}
