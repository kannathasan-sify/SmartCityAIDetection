# backend/app/services/reasoning.py
import math
from typing import List, Dict, Any, Optional
from app.models import Detection, IngestPayload, RoadEvent

class IncidentReasoner:
    """
    Implements Section 6.25: Situational Awareness & AI Reasoning.
    Converts raw object detections into classified high-priority incidents.
    """

    @staticmethod
    def classify_incidents(payload: IngestPayload, lat: float, lng: float) -> List[RoadEvent]:
        events = []
        dets = payload.detections
        
        # 1. Traffic Blockage (Stopped vehicle in main road area)
        # We assume y > 0.4 (lower half of frame) is the main road segment in perspective
        blocking = [d for d in dets if d.class_id == 8 and d.bbox[1] > 400] # class 8 = stopped_vehicle
        if blocking:
            events.append(RoadEvent(
                camera_id=payload.camera_id,
                event_type="incident",
                severity="high" if len(blocking) > 1 else "medium",
                lat=lat,
                lng=lng,
                description=f"TRAFFIC BLOCKAGE: {len(blocking)} vehicle(s) stalled in active travel lane."
            ))

        # 2. Collision Detection (Overlap of vehicles with 0 speed)
        # Check for multiple vehicles (class 0, 1, 2) that have high overlap and 0 speed
        vehicles = [d for d in dets if d.class_id in [0, 1, 2]]
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if IncidentReasoner.check_overlap(vehicles[i].bbox, vehicles[j].bbox, threshold=0.3):
                    if (vehicles[i].speed_kmh or 0) < 5 and (vehicles[j].speed_kmh or 0) < 5:
                        events.append(RoadEvent(
                            camera_id=payload.camera_id,
                            event_type="incident",
                            severity="critical",
                            lat=lat,
                            lng=lng,
                            description=f"POTENTIAL COLLISION: High-confidence overlap between {vehicles[i].class_name} and {vehicles[j].class_name}."
                        ))
                        break # Only report once per pair group
        
        # 3. Construction Zone vs Hazard
        cones = [d for d in dets if d.class_id in [70, 71, 72]] # cones, barrels, barriers
        if len(cones) >= 5:
            # This is likely a construction zone, not just a hazard
            events.append(RoadEvent(
                camera_id=payload.camera_id,
                event_type="infrastructure",
                severity="medium",
                lat=lat,
                lng=lng,
                description=f"CONSTRUCTION ZONE: {len(cones)} temporary markers detected. Narrowing expected."
            ))

        return events

    @staticmethod
    def check_overlap(boxA: List[float], boxB: List[float], threshold: float = 0.5) -> bool:
        """Determines if two bounding boxes overlap significantly (IoU-like)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou > threshold
