# app/services/event_processor.py
from app.models import IngestPayload, RoadEvent
from app.services.supabase_client import supabase
from app.services.ws_manager import manager
from app.services.reasoning import IncidentReasoner
from datetime import datetime, timezone

CONGESTION_THRESHOLD = 15
STOPPED_CLASS_ID = 8
EMERGENCY_CLASS_ID = 6
PEDESTRIAN_CLASS_ID = 5


async def process_frame_detections(payload: IngestPayload):
    """
    Core business logic: analyse an ingested frame payload and generate
    road events (traffic + infrastructure), persist them to Supabase,
    and broadcast them to all WebSocket clients.
    """
    vehicle_count = sum(1 for d in payload.detections if d.class_id in [0, 1, 2, 3, 4])
    stopped_vehicles = [d for d in payload.detections if d.class_id == STOPPED_CLASS_ID]
    emergency = [d for d in payload.detections if d.class_id == EMERGENCY_CLASS_ID]
    pedestrians = [d for d in payload.detections if d.class_id == PEDESTRIAN_CLASS_ID]

    # Detect fallen pedestrians via YOLO26 pose estimation
    fallen_pedestrians = []
    for pose in payload.poses:
        kps = pose.keypoints
        if len(kps) >= 15:
            left_hip_y = kps[11][1] if len(kps) > 11 else 0
            left_knee_y = kps[13][1] if len(kps) > 13 else 0
            if left_hip_y > left_knee_y:
                fallen_pedestrians.append(pose)

    # Fetch camera GPS location
    try:
        cam = (
            supabase.table("cameras")
            .select("lat, lng")
            .eq("id", payload.camera_id)
            .single()
            .execute()
        )
        lat, lng = cam.data["lat"], cam.data["lng"]
    except Exception:
        lat, lng = 0.0, 0.0

    events: list[RoadEvent] = []

    # ── AI Reasoning (Section 6.25: situational awareness) ───────────────────
    ai_incidents = IncidentReasoner.classify_incidents(payload, lat, lng)
    events.extend(ai_incidents)

    # ── Traffic events ────────────────────────────────────────────────────────
    if vehicle_count >= CONGESTION_THRESHOLD:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="congestion",
                severity="high" if vehicle_count > 25 else "medium",
                lat=lat,
                lng=lng,
                description=f"{vehicle_count} vehicles — {'heavy' if vehicle_count > 25 else 'moderate'} congestion",
            )
        )

    if stopped_vehicles:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="incident",
                severity="medium",
                lat=lat,
                lng=lng,
                description=f"{len(stopped_vehicles)} stopped vehicle(s) detected",
            )
        )

    if emergency:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="incident",
                severity="critical",
                lat=lat,
                lng=lng,
                description="Emergency vehicle detected — clear the road",
            )
        )

    if fallen_pedestrians:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="incident",
                severity="critical",
                lat=lat,
                lng=lng,
                description="Fallen pedestrian detected via YOLO26 pose estimation",
            )
        )

    if len(pedestrians) > 10:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="pedestrian_surge",
                severity="medium",
                lat=lat,
                lng=lng,
                description=f"{len(pedestrians)} pedestrians — high crossing density",
            )
        )

    # ── Speeding Detection (Section 6.9) ──────────────────────────────────────
    speeding_vehicles = [d for d in payload.detections if (d.speed_kmh or 0) > 60.0]
    if speeding_vehicles:
        top_speed = max(d.speed_kmh for d in speeding_vehicles)
        fastest_track = next(d for d in speeding_vehicles if d.speed_kmh == top_speed)
        
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="incident",
                severity="critical" if top_speed > 100 else "high",
                lat=lat,
                lng=lng,
                description=(
                    f"Speeding detected! {len(speeding_vehicles)} vehicle(s). "
                    f"Fastest: {top_speed} km/h (Track ID: {fastest_track.track_id})"
                )
            )
        )

    # ── Persistent Speed Logging (Section 6.10) ───────────────────────────────
    for det in payload.detections:
        if det.speed_kmh and det.speed_kmh > 0:
            try:
                supabase.table("vehicle_speeds").insert({
                    "camera_id": payload.camera_id,
                    "track_id": det.track_id,
                    "class_name": det.class_name,
                    "speed_kmh": det.speed_kmh,
                    "metadata": {"bbox": det.bbox, "confidence": det.confidence}
                }).execute()
            except Exception as e:
                print(f"[event_processor] Speed log failed: {e}")

    if payload.lane_speeds:
        try:
            # Aggregate snapshots (Section 6.10 requirement)
            # Structure: {"lane_breakdown": {"lane_1": {"avg": 50, "max": 65}, ...}}
            breakdown = payload.lane_speeds.get("lane_breakdown", {})
            for lane_key, metrics in breakdown.items():
                lane_id = int(lane_key.split("_")[1])
                supabase.table("lane_speed_snapshots").insert({
                    "camera_id": payload.camera_id,
                    "lane_id": lane_id,
                    "avg_speed_kmh": metrics["avg"],
                    "max_speed_kmh": metrics["max"]
                }).execute()
        except Exception as e:
            print(f"[event_processor] Lane snapshot failed: {e}")

    # ── Infrastructure events ─────────────────────────────────────────────────
    off_lights = [l for l in payload.lights if l.light_status in ("OFF", "DAMAGED")]
    flickering = [l for l in payload.lights if l.light_status == "FLICKERING"]

    if off_lights:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="infrastructure",
                severity="high" if len(off_lights) > 2 else "medium",
                lat=lat,
                lng=lng,
                description=f"{len(off_lights)} street light(s) OFF or DAMAGED — maintenance required",
            )
        )

    if flickering:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="infrastructure",
                severity="low",
                lat=lat,
                lng=lng,
                description=f"{len(flickering)} street light(s) FLICKERING",
            )
        )

    # Vegetation hazards — fallen trees
    fallen_trees = [v for v in payload.vegetation if v.is_hazard]
    if fallen_trees:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="hazard",
                severity="critical",
                lat=lat,
                lng=lng,
                description="Fallen tree detected on road — immediate clearance required",
            )
        )

    # Flood water and construction zones
    for hazard in payload.hazards:
        h_type = hazard.get("hazard_type", "unknown")
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="hazard",
                severity="critical" if h_type == "flood_water" else "medium",
                lat=lat,
                lng=lng,
                description=f"Hazard detected: {h_type.replace('_', ' ').title()}",
            )
        )
    
    if payload.is_narrowed:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="infrastructure",
                severity="high",
                lat=lat,
                lng=lng,
                description=f"Road narrowing detected! Lane width: {payload.lane_width_m}m. Potential obstruction.",
            )
        )

    if len(payload.temp_objects) >= 5:
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="infrastructure",
                severity="medium",
                lat=lat,
                lng=lng,
                description=f"Active construction zone detected — {len(payload.temp_objects)} temporary objects (cones/barriers) in road",
            )
        )

    if payload.surface_condition and payload.surface_condition.condition_label in (
        "POOR",
        "CRITICAL",
    ):
        events.append(
            RoadEvent(
                camera_id=payload.camera_id,
                event_type="infrastructure",
                severity=(
                    "critical"
                    if payload.surface_condition.condition_label == "CRITICAL"
                    else "high"
                ),
                lat=lat,
                lng=lng,
                description=(
                    f"Road surface {payload.surface_condition.condition_label} — "
                    f"score {payload.surface_condition.condition_score}/100. "
                    f"Defects: {payload.surface_condition.defects_detected}"
                ),
            )
        )

    # ── Persist events ────────────────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    for event in events:
        row = event.model_dump()
        row["created_at"] = now_iso
        try:
            supabase.table("road_events").insert(row).execute()
            await manager.broadcast({"type": "new_event", "data": row})
        except Exception as exc:
            print(f"[event_processor] Failed to persist event: {exc}")

    # ── Persist road measurement snapshot ─────────────────────────────────────
    measurement_row = {
        "camera_id": payload.camera_id,
        "measured_at": now_iso,
        "road_width_m": payload.road_width_m,
        "lane_width_m": payload.lane_width_m,
        "lane_count": payload.lane_count,
        "poles_detected": len(payload.poles),
        "lights_on": sum(1 for l in payload.lights if l.light_status == "ON"),
        "lights_off": sum(1 for l in payload.lights if l.light_status == "OFF"),
        "lights_damaged": sum(1 for l in payload.lights if l.light_status == "DAMAGED"),
        "pole_spacings_m": payload.pole_spacings_m,
        "tree_coverage_pct": payload.tree_coverage_pct,
        "surface_score": (
            payload.surface_condition.condition_score
            if payload.surface_condition
            else None
        ),
        "surface_label": (
            payload.surface_condition.condition_label
            if payload.surface_condition
            else None
        ),
        "defects": (
            payload.surface_condition.defects_detected
            if payload.surface_condition
            else None
        ),
    }
    try:
        supabase.table("road_measurements").insert(measurement_row).execute()
    except Exception as exc:
        print(f"[event_processor] Failed to persist measurement: {exc}")

    # ── Upsert infrastructure asset registry ──────────────────────────────────
    assets_to_upsert = []
    
    for pole in payload.poles:
        assets_to_upsert.append({
            "camera_id": payload.camera_id,
            "asset_type": pole.class_name,
            "est_height_m": pole.est_height_m,
            "est_diameter_m": pole.est_diameter_m,
            "last_seen_at": now_iso,
            "raw_detection": pole.model_dump(),
        })

    for light in payload.lights:
        assets_to_upsert.append({
            "camera_id": payload.camera_id,
            "asset_type": light.class_name,
            "light_status": light.light_status,
            "last_seen_at": now_iso,
            "raw_detection": light.model_dump(),
        })

    for veg in payload.vegetation:
        assets_to_upsert.append({
            "camera_id": payload.camera_id,
            "asset_type": veg.class_name,
            "est_height_m": veg.est_height_m,
            "last_seen_at": now_iso,
            "raw_detection": veg.model_dump(),
        })

    for item in payload.street_furniture + payload.temp_objects:
        assets_to_upsert.append({
            "camera_id": payload.camera_id,
            "asset_type": item.class_name,
            "last_seen_at": now_iso,
            "raw_detection": item.model_dump(),
        })

    for asset in assets_to_upsert:
        try:
            supabase.table("infrastructure_assets").upsert(asset).execute()
        except Exception as exc:
            print(f"[event_processor] Failed to upsert asset: {exc}")
