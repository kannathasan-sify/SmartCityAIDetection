# app/routers/heatmap.py
from fastapi import APIRouter, Query
from app.services.supabase_client import supabase
from app.services.heatmap_builder import generate_heatmap_snapshot

router = APIRouter()


@router.get("/", summary="Get the latest heatmap grid")
async def get_heatmap(
    window_minutes: int = Query(15, ge=1, le=60),
    mode: str = Query("traffic", regex="^(traffic|infrastructure)$")
):
    """
    Returns the most recent heatmap snapshot for the requested mode.
    """
    grid = await generate_heatmap_snapshot(window_minutes, mode)
    return {"window_minutes": window_minutes, "mode": mode, "points": grid, "count": len(grid)}


@router.get("/snapshots", summary="List recent heatmap snapshots")
async def list_snapshots(limit: int = Query(10, ge=1, le=50)):
    result = (
        supabase.table("heatmap_snapshots")
        .select("id, snapshot_at")
        .order("snapshot_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


@router.get("/snapshots/{snapshot_id}", summary="Get a specific heatmap snapshot")
async def get_snapshot(snapshot_id: str):
    result = (
        supabase.table("heatmap_snapshots")
        .select("*")
        .eq("id", snapshot_id)
        .single()
        .execute()
    )
    return result.data
