# app/services/measure/speed.py
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# ── Persistent State Management ───────────────────────────────────────────
# camera_states = { camera_id: { "prev_gray": ..., "prev_time": ..., "tracks": { track_id: [history] } } }
camera_states: Dict[str, Dict] = {}

# Hyperparameters
HISTORY_WINDOW = 8
SPEED_LIMIT_KMH = 250
OUTLIER_THRESHOLD = 1.5  # Z-score-ish threshold for trimmed mean

class AdvancedSpeedEstimator:
    """Implements Section 6: Advanced Multi-Vector Speed Estimation."""

    @staticmethod
    def get_cam_state(camera_id: str):
        if camera_id not in camera_states:
            camera_states[camera_id] = {
                "prev_gray": None,
                "prev_time": None,
                "tracks": {} # track_id -> deque of (speed_val, timestamp)
            }
        return camera_states[camera_id]

    @staticmethod
    def calculate_centroid(bbox: list) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    @staticmethod
    def estimate_optical_flow(prev_gray, curr_gray, bbox) -> float:
        """Section 6.4: Lucas-Kanade pyramidal tracking of features inside bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        # ROI inside vehicle
        roi_prev = prev_gray[y1:y2, x1:x2]
        roi_curr = curr_gray[y1:y2, x1:x2]
        if roi_prev.size == 0 or roi_curr.size == 0:
            return 0.0

        # Find features to track (Shi-Tomasi)
        p0 = cv2.goodFeaturesToTrack(roi_prev, maxCorners=20, qualityLevel=0.3, minDistance=7)
        if p0 is None:
            return 0.0

        # Lucas-Kanade
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_prev, roi_curr, p0, None, **lk_params)

        if p1 is None:
            return 0.0

        # Calculate median displacement (Section 6.4 requirement: median displacement suppress noise)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) == 0:
            return 0.0

        displacements = np.linalg.norm(good_new - good_old, axis=1)
        return float(np.median(displacements))

    @staticmethod
    def fuse_speeds(track_id: int, flow_px: float, cent_px: float, dt: float, px_per_m: Optional[float], history: deque) -> float:
        """Section 6.6: SpeedFuser with trimmed mean and 8-frame history."""
        if dt <= 0 or px_per_m is None or px_per_m == 0:
            return 0.0

        # Primary: Optical Flow, Fallback: Centroid
        raw_px = flow_px if flow_px > 0 else cent_px
        raw_speed_ms = (raw_px / px_per_m) / dt
        raw_speed_kmh = raw_speed_ms * 3.6

        # Add to history
        history.append(raw_speed_kmh)
        if len(history) < 2:
            return raw_speed_kmh

        # Trimmed Mean / Outlier Removal (Section 6.6)
        speeds = sorted(list(history))
        if len(speeds) >= 4:
            # Remove top/bottom 25%
            trim = len(speeds) // 4
            trimmed = speeds[trim:-trim]
            return float(np.mean(trimmed))
        
        return float(np.mean(speeds))

def compute_v6_speed(camera_id: str, frame: np.ndarray, detections: List[Dict], px_per_metre: Optional[float]) -> List[Dict]:
    """Section 6.8: Main entry point for SpeedFuser in inference loop."""
    state = AdvancedSpeedEstimator.get_cam_state(camera_id)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_time = time.time()
    
    # Persistent Scale Calibration (Section 6.3)
    if px_per_metre is not None:
        state["last_px_per_m"] = px_per_metre
    else:
        px_per_metre = state.get("last_px_per_m")
    
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_time = time.time()
    
    # Section 6.14: Resolution Change Protection (LK Pyramid Sync)
    prev_gray = state.get("prev_gray")
    if prev_gray is not None and prev_gray.shape != curr_gray.shape:
        print(f"[media] Resolution change: {prev_gray.shape} -> {curr_gray.shape}. Resetting state.")
        state["prev_gray"] = curr_gray
        state["prev_time"] = curr_time
        state["tracks"] = {} # Full reset for new geometry
        return detections

    prev_time = state.get("prev_time")
    dt = curr_time - prev_time if prev_time else 0
    
    for det in detections:
        track_id = det.get("track_id")
        if track_id is None: continue
        
        if track_id not in state["tracks"]:
            state["tracks"][track_id] = {
                "history": deque(maxlen=HISTORY_WINDOW),
                "last_centroid": None
            }
        
        track_data = state["tracks"][track_id]
        bbox = det["bbox"]
        curr_centroid = AdvancedSpeedEstimator.calculate_centroid(bbox)
        
        # 1. Centroid Displacement (px)
        cent_px = 0.0
        if track_data["last_centroid"]:
            prev_c = track_data["last_centroid"]
            cent_px = math.sqrt((curr_centroid[0] - prev_c[0])**2 + (curr_centroid[1] - prev_c[1])**2)
        
        # 2. Optical Flow (px) - With Exception Shield
        flow_px = 0.0
        if prev_gray is not None:
            try:
                flow_px = AdvancedSpeedEstimator.estimate_optical_flow(prev_gray, curr_gray, bbox)
            except Exception as e:
                print(f"[media] Error in Optical Flow: {e}")
                flow_px = 0.0

        # 3. Fuse & Convert
        speed_kmh = AdvancedSpeedEstimator.fuse_speeds(
            track_id, flow_px, cent_px, dt, px_per_metre, track_data["history"]
        )
        
        # Sanity check
        det["speed_kmh"] = round(speed_kmh, 1) if speed_kmh < SPEED_LIMIT_KMH else 0.0
        
        # Update track state
        track_data["last_centroid"] = curr_centroid

    # Store frame/time for next cycle
    state["prev_gray"] = curr_gray
    state["prev_time"] = curr_time
    return detections

import math

def compute_lane_speeds(detections: List[Dict], road_width_m: float, lane_count: int) -> Dict[str, Any]:
    """Section 6.7: divide road into lane bands and bucket vehicle speeds."""
    if not lane_count or not road_width_m or road_width_m <= 0:
        return {}

    lane_data = {i: [] for i in range(1, lane_count + 1)}
    
    # We estimate lane by horizontal position. 
    # In a more advanced version, we'd use the road mask segmentation boundary.
    # We assume detections are in a 1920-wide or similar perspective.
    # For now, we'll use a relative 0.0-1.0 horizontal position.
    
    for det in detections:
        speed = det.get("speed_kmh")
        if speed is None or speed <= 0: continue
        
        # Calculate normalized horizontal center (0 = left, 1 = right)
        # Using bbox coords [x1, y1, x2, y2]
        # We need the frame width to normalize properly. 
        # For simplicity, we'll assume the frame is 1280 or we use the max x in detections.
        max_x = max([d["bbox"][2] for d in detections]) if detections else 1280
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        norm_x = cx / max_x
        
        # Assign to lane (1-indexed)
        lane_idx = int(norm_x * lane_count) + 1
        lane_idx = min(lane_idx, lane_count)
        lane_data[lane_idx].append(speed)

    # Calculate metrics
    lane_breakdown = {}
    all_speeds = []
    for i in range(1, lane_count + 1):
        speeds = lane_data[i]
        all_speeds.extend(speeds)
        lane_breakdown[f"lane_{i}"] = {
            "avg": round(sum(speeds)/len(speeds), 1) if speeds else 0.0,
            "max": round(max(speeds), 1) if speeds else 0.0,
            "count": len(speeds)
        }
    
    return {
        "avg_speed_kmh": round(sum(all_speeds)/len(all_speeds), 1) if all_speeds else 0.0,
        "lane_breakdown": lane_breakdown
    }
