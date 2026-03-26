# backend/app/services/measure/infrastructure.py
import cv2
import numpy as np

# Infrastructure class IDs (matching 76-class data.yaml)
POLE_CLASSES  = {14, 15, 16, 17, 18, 19, 20}   # all pole / vertical structure types
LIGHT_CLASSES = {21}                             # street_light_head
SIGNAL_CLASSES = {22, 23}                        # traffic / pedestrian signal heads
VMS_CLASSES = {24}                               # variable_message_sign
SIGN_CLASSES = {25}                              # road_sign
VEGETATION = {56, 57, 58, 59, 60, 61}            # trees, shrubs, verges
STREET_FURNITURE = {51, 52, 53, 54, 55}          # bench, bin, bus stop, etc.
HAZARD_CLASSES = {58, 69, 44}                    # fallen_tree, flood_water, construction_zone
TEMP_OBJECTS = {70, 71, 72, 73, 74, 75}          # cones, barrels, barriers, etc.

def detect_infrastructure(
    frame: np.ndarray,
    detections: list[dict],
    depth_map: np.ndarray,
    px_per_metre: float,
    camera_height_m: float = 6.0,
) -> dict:
    """
    Extract real-world measurements for all detected infrastructure assets.
    """
    h, w = frame.shape[:2]
    poles = []
    lights = []
    signals = []
    signs = []
    vegetation = []
    furniture = []
    temp_objects = []
    hazards = []

    for det in detections:
        cid = det["class_id"]
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        depth_val = float(depth_map[int(cy), int(cx)]) if depth_map is not None else 0.5
        est_distance_m = round(camera_height_m / max(depth_val, 0.01) * 0.15, 1)

        base = {
            "class_name": det["class_name"],
            "class_id": cid,
            "bbox": bbox,
            "confidence": det["confidence"],
            "est_distance_m": est_distance_m,
            "pixel_centre": [round(cx, 1), round(cy, 1)],
        }

        if cid in POLE_CLASSES:
            poles.append({
                **base,
                "est_height_m": round((y2 - y1) / px_per_metre, 1) if px_per_metre else None,
                "est_diameter_m": round((x2 - x1) / px_per_metre, 2) if px_per_metre else None,
            })
        elif cid in LIGHT_CLASSES:
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            lights.append({**base, "light_status": classify_light_status(roi)})
        elif cid in SIGNAL_CLASSES:
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            signals.append({**base, "signal_state": classify_signal_state(roi)})
        elif cid in SIGN_CLASSES or cid in VMS_CLASSES:
            signs.append(base)
        elif cid in VEGETATION:
            veg = {**base}
            if px_per_metre:
                veg["est_height_m"] = round((y2 - y1) / px_per_metre, 1)
                veg["est_canopy_width_m"] = round((x2 - x1) / px_per_metre, 1)
            veg["is_hazard"] = (cid == 58)  # fallen_tree
            vegetation.append(veg)
        elif cid in STREET_FURNITURE:
            furniture.append(base)
        elif cid in TEMP_OBJECTS:
            temp_objects.append(base)
        elif cid in HAZARD_CLASSES:
            hazards.append({"hazard_type": det["class_name"], "raw_det": base})

    pole_spacings = compute_pole_spacing(poles, px_per_metre)
    
    # Tree coverage
    tree_px = sum((t["bbox"][2]-t["bbox"][0])*(t["bbox"][3]-t["bbox"][1]) 
                  for t in vegetation if t["class_id"] == 56)
    tree_coverage_pct = round((tree_px / (h * w)) * 100, 1) if h * w else 0

    return {
        "poles": poles,
        "lights": lights,
        "signals": signals,
        "signs": signs,
        "vegetation": vegetation,
        "street_furniture": furniture,
        "temp_objects": temp_objects,
        "hazards": hazards,
        "pole_spacings_m": pole_spacings,
        "tree_coverage_pct": tree_coverage_pct
    }

def classify_light_status(roi: np.ndarray) -> str:
    """Enhanced glow detection for maintenance scheduling (v1.3)."""
    if roi.size == 0: return "UNKNOWN"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    avg_v = float(np.mean(hsv[:, :, 2]))
    # Use Laplacian variance to detect "glow" (active light source)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if avg_v > 200 and laplacian_var > 100: 
        return "ON"
    elif avg_v > 150: 
        return "DIM"
    elif avg_v > 50: 
        return "OFF"
    else: 
        return "DAMAGED"

def classify_signal_state(roi: np.ndarray) -> str:
    """Maintenance check for traffic signal heads."""
    if roi.size == 0: return "UNKNOWN"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red range
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    red_px = cv2.countNonZero(red1) + cv2.countNonZero(red2)
    
    # Amber/Yellow
    amber_px = cv2.countNonZero(cv2.inRange(hsv, (15, 70, 50), (35, 255, 255)))
    
    # Green
    green_px = cv2.countNonZero(cv2.inRange(hsv, (40, 70, 50), (95, 255, 255)))
    
    total = red_px + amber_px + green_px
    if total < 5: return "OFF" # No active signal detected
    
    if red_px > amber_px and red_px > green_px: return "RED"
    if amber_px > red_px and amber_px > green_px: return "AMBER"
    if green_px > red_px and green_px > amber_px: return "GREEN"
    return "TRANSITION"

def compute_pole_spacing(poles: list[dict], px_per_metre: float) -> list[float]:
    if len(poles) < 2 or px_per_metre is None: return []
    sorted_poles = sorted(poles, key=lambda p: p["pixel_centre"][0])
    spacings = []
    for i in range(1, len(sorted_poles)):
        dx = sorted_poles[i]["pixel_centre"][0] - sorted_poles[i-1]["pixel_centre"][0]
        spacings.append(round(dx / px_per_metre, 1))
    return spacings
