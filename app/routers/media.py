# app/routers/media.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.middleware.auth import verify_api_key
from app.services.inference_engine import run_inference
import shutil
import os
import uuid

router = APIRouter()

from fastapi.responses import FileResponse
import os

import os
import json
from datetime import datetime

UPLOAD_DIR = "uploads"
LOG_FILE = os.path.join(UPLOAD_DIR, "url_logs.json") # Section 6.21: Persistent Analytics Log
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/download/{filename}")
async def download_media(filename: str):
    """Serves processed media files."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@router.post("/process")
def process_media(
    camera_id: str = "default",
    file: UploadFile = File(...),
):
    """Processes an uploaded image or video for road detection."""
    print(f"\n[media] Received file: {file.filename} (content_type: {file.content_type})")
    
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".jpg" # Fallback
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    print(f"[media] Saving to: {file_path}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[media] Running inference on {file_path}...")
        results = run_inference(file_path, camera_id)
        print(f"[media] Inference complete. Found {len(results['detections'])} detections.")
        
        # Section 6.21: Permanent URL Logging
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "file_id": file_id,
                "original_name": file.filename,
                "processed_url": f"/media/download/{results.get('processed_image') or results.get('processed_video')}",
                "detections": len(results['detections']),
                "camera": camera_id
            }
            with open(LOG_FILE, "a") as logf:
                logf.write(json.dumps(log_data) + "\n")
            print(f"[media] Saved URL log to {LOG_FILE}")
        except Exception as log_err:
            print(f"[media] WARNING: Failed to write URL log: {log_err}")

        return {
            "id": file_id,
            "filename": file.filename,
            "results": results
        }
    except Exception as e:
        print(f"[media] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Keep small files for now or delete them
        # os.remove(file_path)
        pass
