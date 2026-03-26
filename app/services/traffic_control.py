# backend/app/services/traffic_control.py
from typing import Dict, Any, List
from datetime import datetime, timezone
import random

class TrafficController:
    """
    Implements v2.1: Adaptive Infrastructure & Signal Integration.
    Mocks the communication with a SCATS/SCOOT-like traffic control system.
    """
    
    _signal_states: Dict[str, str] = {} # Mock in-memory state

    @staticmethod
    def get_signal_state(camera_id: str) -> str:
        return TrafficController._signal_states.get(camera_id, "AUTO")

    @staticmethod
    def request_emergency_override(camera_id: str, incident_type: str) -> Dict[str, Any]:
        """
        Force all approaches to RED except the priority lane.
        """
        TrafficController._signal_states[camera_id] = "EMERGENCY_OVERRIDE"
        
        return {
            "camera_id": camera_id,
            "status": "OVERRIDE_ACTIVE",
            "mode": "GREEN_WAVE",
            "reason": incident_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expires_in_sec": 120
        }

    @staticmethod
    def release_override(camera_id: str):
        TrafficController._signal_states[camera_id] = "AUTO"
        return {"camera_id": camera_id, "status": "RELEASED"}

    @staticmethod
    def compute_adaptive_timing(vehicle_count: int, congestion_level: str) -> Dict[str, Any]:
        """
        Mocks the v2.1 adaptive signal timing logic.
        """
        base_green = 30
        if congestion_level == "high":
            green_time = base_green + (vehicle_count * 2)
        else:
            green_time = base_green
            
        return {
            "recommended_green_sec": min(green_time, 90),
            "cycle_adjustment_pct": 15 if congestion_level == "high" else 0
        }
