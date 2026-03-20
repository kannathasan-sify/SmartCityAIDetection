# backend/app/services/measure/surface.py
import cv2
import numpy as np

# Detailed defect classes (from 76-class set)
DEFECT_CLASSES = {
    35: "pothole",
    36: "road_crack",
    37: "road_patch",
    38: "speed_bump",
    39: "speed_table",
    40: "rumble_strip",
    41: "manhole_cover",
    42: "drain_grate",
    43: "road_staining",
    44: "construction_zone",
}

def assess_road_surface(detections: list[dict], frame: np.ndarray) -> dict:
    """
    Aggregate road surface defect detections into a condition score (0–100).
    """
    defects = [d for d in detections if d["class_id"] in DEFECT_CLASSES]

    defect_summary = {}
    for d in defects:
        label = DEFECT_CLASSES[d["class_id"]]
        defect_summary[label] = defect_summary.get(label, 0) + 1

    # Penalties for road health
    deductions = (
        defect_summary.get("pothole",           0) * 20 +
        defect_summary.get("road_crack",        0) * 8  +
        defect_summary.get("road_staining",     0) * 5  +
        defect_summary.get("construction_zone", 0) * 10 +
        defect_summary.get("manhole_cover",     0) * 2  +
        defect_summary.get("drain_grate",       0) * 1
    )
    condition_score = max(0, 100 - deductions)

    if condition_score >= 80:   condition_label = "GOOD"
    elif condition_score >= 50: condition_label = "FAIR"
    elif condition_score >= 20: condition_label = "POOR"
    else:                       condition_label = "CRITICAL"

    return {
        "condition_score":  condition_score,
        "condition_label":  condition_label,
        "defects_detected": defect_summary,
    }
