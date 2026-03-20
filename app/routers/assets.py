# app/routers/assets.py
from fastapi import APIRouter, HTTPException
from app.services.supabase_client import supabase
from typing import List, Dict, Any

router = APIRouter()

@router.get("/", response_model=List[Dict[str, Any]])
async def get_assets(camera_id: str = None):
    """Fetch persistent infrastructure assets for the map."""
    query = supabase.table("infrastructure_assets").select("*")
    if camera_id:
        query = query.eq("camera_id", camera_id)
    
    res = query.execute()
    return res.data

@router.get("/{asset_id}", response_model=Dict[str, Any])
async def get_asset(asset_id: str):
    res = supabase.table("infrastructure_assets").select("*").eq("id", asset_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Asset not found")
    return res.data
