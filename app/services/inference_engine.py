# app/services/inference_engine.py
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import os
from datetime import datetime, timezone

# Load models once on startup
# Use yolo11 if yolo26 is not available locally
# Section 6.22: Ultra-Fast Nano Model for Real-time Response
MODEL_PATH = os.getenv("MODEL_DET", "yolo11n.pt")
model = YOLO(MODEL_PATH)

# Import measurement services
from app.services.measure.depth import get_depth_map
from app.services.measure.road_width import estimate_pixel_per_metre, measure_road_width
from app.services.measure.infrastructure import detect_infrastructure
from app.services.measure.surface import assess_road_surface
from app.services.measure.speed import compute_v6_speed, compute_lane_speeds

def run_inference(file_path: str, camera_id: str = "default") -> Dict[str, Any]:
    """Runs YOLO v4.0 inference with Section 6 Advanced Speed Estimation.
    Now supports video files by processing 1-2 second clips and baking results.
    """
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in [".mp4", ".mov", ".avi", ".mkv"]
    processed_video_url = None
    processed_image_url = None
    frame_idx = 0
    fps = 0
    final_lane_speeds = [] # Section 6.20: Safe Initialization
    
    if is_video:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Setup VideoWriter for baking detections (Section 6.9: Robust Encoding)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        output_path = file_path.replace(ext, f"_processed.mp4") # Force .mp4 for compatibility
        
        # Try multiple codecs to find one that works in the local environment
        codecs = ['avc1', 'mp4v', 'XVID', 'MJPG']
        out = None
        for c in codecs:
            fourcc = cv2.VideoWriter_fourcc(*c)
            temp_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if temp_out.isOpened():
                out = temp_out
                print(f"[media] Successfully opened VideoWriter with codec: {c}")
                break
            else:
                 print(f"[media] Falling back from codec: {c}")
                 
        if out is None or not out.isOpened():
            print(f"[media] ERROR: All video codecs failed for {output_path}")
            # Non-blocking: continue without baking if possible, or return error
        
        max_frames = 30
        frame_idx = 0
        final_detections = []
        temporal_detections = []
        video_gallery_map = {} # Track unique objects for the horizontal gallery
        last_frame = None
        
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            last_frame = frame
            # Section 6.13/6.19: Increased Sensitivity (Conf 0.15 for V4.0 Intelligence)
            results = model.track(frame, persist=True, conf=0.15, verbose=False)
            first_result = results[0] if results else None
            
            if frame_idx % 5 == 0:
                 print(f"[media] Processing Frame {frame_idx}/{max_frames}...")
                 
            detections = []
            if first_result:
                for box in first_result.boxes:
                    track_id = int(box.id[0]) if box.id is not None else None
                    detections.append({
                        "track_id":   track_id,
                        "class_id":   int(box.cls),
                        "class_name": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox":       box.xyxy[0].tolist(),
                    })
            
            px_per_metre = estimate_pixel_per_metre(detections, width)
            detections = compute_v6_speed(camera_id, frame, detections, px_per_metre)
            
            # Baking logic (Section 6.9: Visual Feedback)
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                color = (74, 222, 128) if det["confidence"] > 0.7 else (245, 158, 11) # BGR (Green/Orange)
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label with background for readability
                label = f"{det['class_name'].upper()} {det.get('speed_kmh', 0)}km/h"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if out is not None:
                out.write(frame)
            temporal_detections.append(detections)
            
            # Detailed Logging (Section 6.19)
            if detections:
                asset_str = ", ".join([f'{d.get("class_name", "unknown").upper()}({d.get("confidence", 0):.2f})' for d in detections])
                print(f"[media] Frame {frame_idx} Assets: {asset_str}")
            
            # Aggregate for the Gallery (Section 6.16)
            for det in detections:
                tid = det.get("track_id")
                if tid is not None and tid not in video_gallery_map:
                    # Save a high-quality crop for this specific object
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        base_path = os.path.splitext(file_path)[0]
                        crop_path = f"{base_path}_track_{tid}.jpg"
                        cv2.imwrite(crop_path, crop)
                        det["crop_url"] = os.path.basename(crop_path)
                        video_gallery_map[tid] = det

            frame_idx += 1
            
        cap.release()
        out.release()
        
        processed_video_url = os.path.basename(output_path)
        
        road_masks = [r.masks.xy[i].tolist() for i, r in enumerate(results) if r.masks is not None]
        road_metrics = measure_road_width(road_masks[0] if road_masks else [], px_per_metre, (height, width))
        final_detections = list(video_gallery_map.values())
        final_lane_speeds = compute_lane_speeds(final_detections, road_metrics["road_width_m"], road_metrics["lane_count"])
        frame = last_frame
        
    else:
        # Standard Single Image Logic - Section 6.13 Sensitivity Sync
        frame = cv2.imread(file_path)
        if frame is None:
            return {"error": "Could not read image"}

        # Use 0.15 threshold for maximum "V4.0" Intelligence sensitivity
        results = model.track(frame, persist=True, conf=0.15, verbose=False)
        first_result = results[0] if results else None
        
        # Fallback to standard predict if track yields nothing (common for high-noise stills)
        if not first_result or len(first_result.boxes) == 0:
            print("[media] Tracking failed to find assets, falling back to standard prediction...")
            results = model(frame, conf=0.15, verbose=False)
            first_result = results[0] if results else None
            
        print(f"[media] Image Detection... Found {len(first_result.boxes) if first_result else 0} boxes.")

        detections = []
        if first_result:
            for box in first_result.boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                cls_id = int(box.cls[0])
                detections.append(det_obj := {
                    "track_id":   track_id,
                    "class_id":   cls_id,
                    "class_name": model.names[cls_id] if cls_id in model.names else "unknown",
                    "confidence": float(box.conf[0]),
                    "bbox":       box.xyxy[0].tolist(),
                })
                print(f"[media]  - Asset: {det_obj['class_name'].upper()} (Conf: {det_obj['confidence']:.4f})")

        px_per_metre = estimate_pixel_per_metre(detections, frame.shape[1])
        final_detections = compute_v6_speed(camera_id, frame, detections, px_per_metre)
        
        # Section 6.15: Image Baking (Consistency with Video)
        processed_image_path = file_path.replace(ext, f"_processed.jpg")
        baked_frame = frame.copy()
        for det in final_detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = (74, 222, 128) if det["confidence"] > 0.7 else (245, 158, 11) # BGR
            cv2.rectangle(baked_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name'].upper()} {det.get('speed_kmh', 0)}km/h"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(baked_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(baked_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        

        cv2.imwrite(processed_image_path, baked_frame)
        processed_image_url = os.path.basename(processed_image_path)
        
        road_masks = [r.masks.xy[i].tolist() for i, r in enumerate(results) if r.masks is not None]
        road_metrics = measure_road_width(road_masks[0] if road_masks else [], px_per_metre, frame.shape)
        final_lane_speeds = compute_lane_speeds(final_detections, road_metrics["road_width_m"], road_metrics["lane_count"])

    # Section 6.16: Shared logic for Images (Videos already aggregated above)
    if not is_video:
        for i, det in enumerate(final_detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                base_path = os.path.splitext(file_path)[0]
                crop_path = f"{base_path}_crop_{i}.jpg"
                cv2.imwrite(crop_path, crop)
                det["crop_url"] = os.path.basename(crop_path)

    # (Infrastructure & Surface shared)
    # Section 6.22: Speed optimization - Skip depth for static images unless requested
    depth_map = get_depth_map(frame) if is_video else np.zeros(frame.shape[:2], dtype=np.float32)
    infra_data = detect_infrastructure(frame, final_detections, depth_map, px_per_metre)
    surface_data = assess_road_surface(final_detections, frame)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "is_video": is_video,
        "processed_video": processed_video_url,
        "processed_image": processed_image_url,
        "temporal_detections": temporal_detections if is_video else None,
        "video_metadata": {
            "fps": fps if is_video else 0,
            "frame_count": frame_idx if is_video else 1
        } if is_video else None,
        "detections": final_detections,
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "measurements": {
            "road_width_m": road_metrics["road_width_m"],
            "lane_width_m": road_metrics.get("lane_width_m", 0),
            "lane_count": road_metrics.get("lane_count", 0),
            "lane_speeds": final_lane_speeds,
            "surface": surface_data,
            "infrastructure": infra_data
        }
    }
