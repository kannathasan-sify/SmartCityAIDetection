# backend/app/services/prediction_engine.py
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from app.services.supabase_client import supabase

class SurfacePredictor:
    """
    Implements v2.0: Predictive surface degradation forecasting.
    Analyzes historical road_measurements to predict future condition drops.
    """

    @staticmethod
    async def forecast_degradation(camera_id: str) -> Dict[str, Any]:
        # 1. Fetch historical measurements (Last 30 days)
        response = supabase.table("road_measurements") \
            .select("surface_score, measured_at") \
            .eq("camera_id", camera_id) \
            .order("measured_at", { "ascending": False }) \
            .limit(10) \
            .execute()

        history = response.data
        if not history or len(history) < 2:
            return {
                "status": "INSUFFICIENT_DATA",
                "days_until_failure": None,
                "current_score": history[0]["surface_score"] if history else None,
                "trend": "STABLE"
            }

        # 2. Calculate degradation rate (Simple Linear Regression logic)
        latest = history[0]
        oldest = history[-1]
        
        latest_at = datetime.fromisoformat(latest["measured_at"].replace("Z", "+00:00"))
        oldest_at = datetime.fromisoformat(oldest["measured_at"].replace("Z", "+00:00"))
        
        days_delta = (latest_at - oldest_at).days or 1
        score_delta = oldest["surface_score"] - latest["surface_score"]
        
        # Degradatiton per day
        rate_per_day = score_delta / days_delta
        
        # 3. Forecast
        # Threshold for "POOR" is usually < 40
        current_score = latest["surface_score"]
        if rate_per_day <= 0:
            days_until_failure = 365 # Stable or Improving
            trend = "IMPROVING" if rate_per_day < -0.1 else "STABLE"
        else:
            remaining_buffer = current_score - 40
            days_until_failure = max(0, int(remaining_buffer / rate_per_day))
            trend = "DECLINING" if rate_per_day > 0.5 else "GRADUAL_DECAY"

        return {
            "camera_id": camera_id,
            "current_score": current_score,
            "trend": trend,
            "degradation_rate_daily": round(rate_per_day, 3),
            "days_until_critical": days_until_failure,
            "forecast_date": (latest_at + timedelta(days=days_until_failure)).isoformat(),
            "confidence": 0.85 if len(history) > 5 else 0.5
        }
