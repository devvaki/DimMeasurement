# Data loading and preprocessing

from __future__ import annotations
import pathlib
import time
import numpy as np
import open3d as o3d
import cv2
from typing import Tuple, Optional

def _sanitize_points(P: np.ndarray) -> np.ndarray:
    
    if P.size == 0:
        return P
    
    # Remove non-finite points
    mask = np.isfinite(P).all(axis=1)
    P = P[mask]
    
    # Remove exact duplicates to help clustering
    if len(P) > 0:
        P = np.unique(P, axis=0)
    
    return P

def _auto_scale_to_mm(P: np.ndarray) -> Tuple[np.ndarray, str]:
    
    if P.size == 0:
        return P, "empty"
    
    span = float(np.max(np.ptp(P, axis=0)))
    
    if 1.0 <= span <= 20.0:  # Likely in meters
        return P * 1000.0, "m_to_mm"
    elif 0.1 <= span <= 2.0:  # Likely in decimeters
        return P * 100.0, "dm_to_mm"
    else:  # Assume already in mm or reasonable units
        return P, "assumed_mm"

def load_pointcloud_and_image(
    ply_path: str | pathlib.Path,
    png_path: str | pathlib.Path
) -> Tuple[o3d.geometry.PointCloud, Optional[np.ndarray], str, int]:
    
    ply_path = pathlib.Path(ply_path)
    png_path = pathlib.Path(png_path)
    
    # Extract parcel ID from filename
    parcel_id = ply_path.stem.split("-")[0]
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    P = _sanitize_points(np.asarray(pcd.points))
    P, units = _auto_scale_to_mm(P)
    pcd_mm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    
    # Load image (optional)
    img_bgr = None
    if png_path.exists():
        img_bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    
    # Generate timestamp from file modification time
    try:
        timestamp_ms = int(ply_path.stat().st_mtime * 1000)
    except Exception:
        timestamp_ms = int(time.time() * 1000)
    
    print(f"Loaded {parcel_id}: {len(P)} points, units: {units}")
    
    return pcd_mm, img_bgr, parcel_id, timestamp_ms