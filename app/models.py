# app/models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    bbox: list[float]  # [x1, y1, x2, y2]
    speed_kmh: Optional[float] = None


class SegmentMask(BaseModel):
    class_id: int
    polygon: list[list[float]]  # list of [x, y] points


class PoseKeypoints(BaseModel):
    keypoints: list[list[float]]  # 17 keypoints as [x, y] pairs


class PoleAsset(BaseModel):
    class_name: str
    bbox: list[float]
    confidence: float
    est_height_m: Optional[float] = None
    est_diameter_m: Optional[float] = None
    est_distance_m: Optional[float] = None
    pixel_centre: list[float]


class LightAsset(BaseModel):
    class_name: str
    bbox: list[float]
    confidence: float
    light_status: str  # ON | OFF | FLICKERING | DAMAGED
    est_distance_m: Optional[float] = None
    pixel_centre: list[float]


class SurfaceCondition(BaseModel):
    condition_score: int  # 0-100
    condition_label: str  # GOOD | FAIR | POOR | CRITICAL
    defects_detected: dict  # {pothole: 2, road_crack: 1, ...}


class SignAsset(BaseModel):
    class_name: str
    bbox: list[float]
    confidence: float
    est_distance_m: Optional[float] = None
    pixel_centre: list[float]


class VegetationAsset(BaseModel):
    class_name: str
    class_id: int
    bbox: list[float]
    confidence: float
    est_height_m: Optional[float] = None
    est_canopy_width_m: Optional[float] = None
    est_distance_m: Optional[float] = None
    is_hazard: bool = False


class InfrastructureAsset(BaseModel):
    class_name: str
    class_id: int
    bbox: list[float]
    confidence: float
    est_distance_m: Optional[float] = None
    pixel_centre: list[float]


class IngestPayload(BaseModel):
    camera_id: str
    timestamp: datetime
    frame_count: int
    # ── Traffic ──────────────────────────────────────
    detections: list[Detection]
    segments: list[SegmentMask] = []
    poses: list[PoseKeypoints] = []
    # ── Road infrastructure measurements ─────────────
    road_width_m: Optional[float] = None
    lane_width_m: Optional[float] = None
    lane_count: Optional[int] = None
    poles: list[PoleAsset] = []
    lights: list[LightAsset] = []
    signs: list[SignAsset] = []
    vegetation: list[VegetationAsset] = []
    street_furniture: list[InfrastructureAsset] = []
    temp_objects: list[InfrastructureAsset] = []
    hazards: list[dict] = []  # Detailed hazard breakdown
    pole_spacings_m: list[float] = []
    tree_coverage_pct: Optional[float] = None
    surface_condition: Optional[SurfaceCondition] = None
    lane_speeds: Optional[Dict[str, Any]] = None


class RoadEvent(BaseModel):
    id: Optional[str] = None
    camera_id: str
    event_type: str  # congestion | incident | pedestrian_surge | parking | infrastructure
    severity: str  # low | medium | high | critical
    lat: float
    lng: float
    description: str
    snapshot_url: Optional[str] = None
    created_at: Optional[datetime] = None


class HeatmapPoint(BaseModel):
    lat: float
    lng: float
    intensity: float = Field(ge=0.0, le=1.0)


class Camera(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    stream_url: Optional[str] = None
    location: Optional[str] = None
    active: bool = True
    created_at: Optional[datetime] = None
