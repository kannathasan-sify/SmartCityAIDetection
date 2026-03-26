# backend/app/services/measure/road_width.py
import cv2
import numpy as np

# Known average widths (metres) used as pixel calibration references
REFERENCE_WIDTHS = {
    "car":   1.8,    # average passenger car width
    "truck": 2.5,
    "bus":   2.5,
    "van":   2.0,
}

def estimate_pixel_per_metre(detections: list[dict], frame_width: int) -> float:
    """
    Use detected vehicles as known-width references to compute
    a pixel-per-metre scale factor for the frame.
    """
    scales = []
    for det in detections:
        cls = det.get("class_name")
        if cls in REFERENCE_WIDTHS:
            # Assuming bbox is [x1, y1, x2, y2]
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                pixel_width = x2 - x1
                real_width  = REFERENCE_WIDTHS[cls]
                scale = pixel_width / real_width   # pixels per metre
                scales.append(scale)

    return float(np.median(scales)) if scales else None


def measure_road_width(
    road_segment_mask: list[list[float]],
    px_per_metre: float,
    frame_shape: tuple,
) -> dict:
    """
    Given the road surface segmentation polygon and pixel-per-metre scale,
    measure the road width in metres at multiple cross-section points.
    """
    if px_per_metre is None or not road_segment_mask:
        return {"road_width_m": None, "lane_width_m": None, "lane_count": None}

    pts = np.array(road_segment_mask, dtype=np.float32)
    h, w = frame_shape[:2]

    # Sample horizontal cross-sections at 25%, 50%, 75% of frame height
    widths_px = []
    for y_frac in [0.25, 0.50, 0.75]:
        y = int(h * y_frac)
        # Find all polygon points near this y level
        nearby = pts[np.abs(pts[:, 1] - y) < 20]
        if len(nearby) >= 2:
            width_px = nearby[:, 0].max() - nearby[:, 0].min()
            widths_px.append(width_px)

    if not widths_px:
        return {"road_width_m": None, "lane_width_m": None, "lane_count": None}

    avg_width_px = float(np.mean(widths_px))
    road_width_m = round(avg_width_px / px_per_metre, 2)

    # Estimate lane count: standard lane = 3.5m
    lane_count = max(1, round(road_width_m / 3.5))
    lane_width_m = round(road_width_m / lane_count, 2)
    
    # Section 6.18: Narrowing Anomaly detection (v1.4 roadmap)
    # If lane is < 2.8m, it's considered restricted/narrrowed
    narrowing_alert = lane_width_m < 2.8

    return {
        "road_width_m":  road_width_m,
        "lane_width_m":  lane_width_m,
        "lane_count":    lane_count,
        "is_narrowed":   narrowing_alert
    }
