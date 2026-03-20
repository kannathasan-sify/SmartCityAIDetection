# backend/app/services/measure/depth.py
import torch
import cv2
import numpy as np

# Load MiDaS model (run once at startup)
# Using small transform for edge compatibility as per blueprint
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
except Exception as e:
    print(f"Error loading MiDaS: {e}. Falling back to CPU/Dummy.")
    midas = None

def get_depth_map(frame: np.ndarray) -> np.ndarray:
    """
    Run MiDaS on a single frame and return a normalised depth map (0–1).
    0 = far, 1 = close.
    """
    if midas is None:
        return np.zeros(frame.shape[:2], dtype=np.float32)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_t = midas_transforms(img_rgb).to(device)

    with torch.no_grad():
        depth = midas(input_t)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_np = depth.cpu().numpy()
    # Normalise to 0–1
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-6)
    return depth_norm.astype(np.float32)
