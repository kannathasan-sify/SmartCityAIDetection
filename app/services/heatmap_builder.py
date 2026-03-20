# app/services/heatmap_builder.py
from app.services.supabase_client import supabase
from datetime import datetime, timezone, timedelta


async def generate_heatmap_snapshot(window_minutes: int = 15, mode: str = "traffic") -> list[dict]:
    """
    Aggregate events (traffic) or assets (infrastructure) into a heatmap grid.
    """
    if mode == "infrastructure":
        # Aggregate all persistent assets for infrastructure density
        result = supabase.table("infrastructure_assets").select("lat, lng").execute()
        grid = [
            {"lat": row["lat"] or 13.0827, "lng": row["lng"] or 80.2707, "intensity": 0.5}
            for row in (result.data or [])
        ]
    else:
        # Default: Traffic events based on severity
        since = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        result = (
            supabase.table("road_events")
            .select("lat, lng, severity")
            .gte("created_at", since)
            .execute()
        )

        severity_weight = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        grid = [
            {
                "lat": row["lat"],
                "lng": row["lng"],
                "intensity": severity_weight.get(row["severity"], 0.3),
            }
            for row in (result.data or [])
        ]

    if grid and mode == "traffic":
        try:
            supabase.table("heatmap_snapshots").insert(
                {
                    "snapshot_at": datetime.now(timezone.utc).isoformat(),
                    "grid": grid,
                }
            ).execute()
        except: pass

    return grid
