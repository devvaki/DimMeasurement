# Dimension Measurement and Volume Estimation

from __future__ import annotations
import numpy as np
import open3d as o3d
import trimesh
from typing import Tuple, Dict, Any

def classify_parcel_type(
    parcel_pcd: o3d.geometry.PointCloud, 
    frame: Dict[str, Any],
    rigidity_threshold: float = 0.15
) -> str:
    
    if parcel_pcd.is_empty():
        return "Unknown"
    
    try:
        P = np.asarray(parcel_pcd.points)
        n = frame["n"]
        
        # Calculate height variation
        heights = P @ n
        height_std = np.std(heights)
        height_range = np.ptp(heights)
        
        # Estimate surface normal consistency
        parcel_pcd.estimate_normals()
        normals = np.asarray(parcel_pcd.normals)
        normal_consistency = 1.0 - np.std(normals, axis=0).mean()
        
        # Rigidity score: boxes are more rigid (consistent normals, less height variation)
        rigidity_score = normal_consistency - (height_std / max(height_range, 1.0))
        
        return "Box" if rigidity_score > rigidity_threshold else "Polybag"
        
    except Exception:
        return "Unknown"

def measure_box_dimensions(
    parcel_pcd: o3d.geometry.PointCloud,
    frame: Dict[str, Any],
    bottom_trim_mm: float = 12.0,
    top_frac: float = 0.30
) -> Tuple[float, float, float, float, o3d.geometry.PointCloud]:
    
    #OBB

    n, h0 = frame["n"], float(frame["h0"])
    P = np.asarray(parcel_pcd.points)
    
    if P.shape[0] == 0:
        empty_pcd = o3d.geometry.PointCloud()
        return 0.0, 0.0, 0.0, 0.0, empty_pcd

    # Calculate heights relative to table
    heights_rel = (P @ n) - h0

    # Trim points too close to table surface
    keep_mask = heights_rel > bottom_trim_mm
    P_trimmed = P[keep_mask]
    heights_trimmed = heights_rel[keep_mask]
    
    trimmed_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_trimmed))
    
    if P_trimmed.shape[0] < 20:
        # Fallback: use simple OBB on available points
        obb = trimmed_pcd.get_oriented_bounding_box()
        L, W, H = sorted(obb.extent, reverse=True)
        return float(L), float(W), float(H), float(L*W*H), trimmed_pcd

    # Robust height calculation using percentiles
    h5, h95 = np.percentile(heights_trimmed, [5, 95])
    height_mm = float(h95 - h5)

    # Use top slice for length/width measurement (more stable for irregular shapes)
    height_cutoff = np.percentile(heights_trimmed, 100 * (1.0 - top_frac))
    top_points = P_trimmed[heights_trimmed >= height_cutoff]
    
    if top_points.shape[0] < 20:
        # Fallback: use full trimmed cloud
        obb = trimmed_pcd.get_oriented_bounding_box()
        L, W, _ = sorted(obb.extent, reverse=True)
        box_volume = float(L * W * height_mm)
        return float(L), float(W), height_mm, box_volume, trimmed_pcd

    # Calculate L/W from oriented bounding box of top slice
    top_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top_points))
    obb_top = top_pcd.get_oriented_bounding_box()
    length_mm, width_mm, _ = sorted(obb_top.extent, reverse=True)
    
    box_volume = float(length_mm * width_mm * height_mm)
    
    return float(length_mm), float(width_mm), height_mm, box_volume, trimmed_pcd

def calculate_true_volume(parcel_pcd: o3d.geometry.PointCloud) -> Tuple[float, o3d.geometry.TriangleMesh]:
    
    # Hull

    if parcel_pcd.is_empty():
        return 0.0, o3d.geometry.TriangleMesh()
    
    try:
        # Compute convex hull
        hull_mesh, _ = parcel_pcd.compute_convex_hull()
        hull_mesh = hull_mesh.remove_duplicated_vertices()
        
        # Extract vertices and faces
        vertices = np.asarray(hull_mesh.vertices)
        faces = np.asarray(hull_mesh.triangles)
        
        if len(vertices) < 4 or len(faces) < 4:
            return 0.0, hull_mesh
        
        # Use trimesh for robust volume calculation
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        volume = float(abs(trimesh_obj.volume))
        
        return volume, hull_mesh
        
    except Exception as e:
        print(f"Warning: Hull volume calculation failed: {e}")
        return 0.0, o3d.geometry.TriangleMesh()

def get_obb_dimensions(pcd: o3d.geometry.PointCloud) -> Tuple[float, float, float, float]:
    
    if pcd.is_empty():
        return 0.0, 0.0, 0.0, 0.0
    
    obb = pcd.get_oriented_bounding_box()
    L, W, H = sorted(obb.extent, reverse=True)
    volume = float(L * W * H)
    
    return float(L), float(W), float(H), volume