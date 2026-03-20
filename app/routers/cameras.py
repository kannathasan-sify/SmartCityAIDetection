# app/routers/cameras.py
from fastapi import APIRouter
from app.services.supabase_client import supabase

router = APIRouter()


@router.get("/", summary="List all cameras")
async def list_cameras():
    result = supabase.table("cameras").select("*").order("name").execute()
    return result.data or []


@router.get("/{camera_id}", summary="Get a single camera by ID")
async def get_camera(camera_id: str):
    result = (
        supabase.table("cameras").select("*").eq("id", camera_id).single().execute()
    )
    return result.data


@router.get("/{camera_id}/measurements", summary="Get recent road measurements for a camera")
async def get_camera_measurements(camera_id: str, limit: int = 20):
    result = (
        supabase.table("road_measurements")
        .select("*")
        .eq("camera_id", camera_id)
        .order("measured_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


@router.get("/{camera_id}/assets", summary="Get infrastructure assets detected by a camera")
async def get_camera_assets(camera_id: str):
    result = (
        supabase.table("infrastructure_assets")
        .select("*")
        .eq("camera_id", camera_id)
        .order("last_seen_at", desc=True)
        .execute()
    )
    return result.data or []
