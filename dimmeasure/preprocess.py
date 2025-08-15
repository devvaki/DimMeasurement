# Table frame estimation and parcel isolation

from __future__ import annotations
import numpy as np
import open3d as o3d
from typing import Tuple, Dict, Any

def estimate_table_frame(pcd_mm: o3d.geometry.PointCloud) -> Tuple[np.ndarray, float]:
    
    P = np.asarray(pcd_mm.points)
    if P.shape[0] == 0:
        return np.array([0, 0, 1.0], dtype=float), 0.0
    
    # Center points and perform PCA via SVD
    Pc = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    
    # Last principal component is the table normal ('up' direction)
    n = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
    
    # Project all points onto this direction to get heights
    heights = P @ n
    h0 = float(np.percentile(heights, 5))  # 5th percentile as table surface
    
    return n, h0

def isolate_parcel(
    pcd_mm: o3d.geometry.PointCloud,
    delta_low: float = 14.0,     # Min height above table (mm)
    delta_high: float = 180.0,   # Max height above table (mm)
    voxel_mm: float = 3.0,       # Voxel downsampling size (mm)
    eps_xy_mm: float = 24.0,     # DBSCAN epsilon (mm)
    min_pts: int = 110,          # Min points for cluster
    z_weight: float = 0.25,      # Z weight in DBSCAN
    top_frac: float = 0.20       # Top fraction for fallback
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any], Dict[str, Any]]:
   
    P = np.asarray(pcd_mm.points)
    n, h0 = estimate_table_frame(pcd_mm)
    
    # Project points to table normal to get heights
    heights = P @ n

    # Step 1: Filter by height window above table
    height_mask = (heights - h0 > delta_low) & (heights - h0 < delta_high)
    P_filtered = P[height_mask]
    
    if P_filtered.size == 0:
        return (o3d.geometry.PointCloud(), 
               {"status": "failed", "reason": "empty_after_height_filter"}, 
               {"n": n, "h0": h0})

    # Step 2: Create cloud and downsample
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_filtered))
    if voxel_mm and not cloud.is_empty():
        cloud = cloud.voxel_down_sample(voxel_size=voxel_mm)
    
    P_voxel = np.asarray(cloud.points)
    if P_voxel.shape[0] < min_pts:
        return (cloud, 
               {"status": "warning", "reason": "few_points_after_voxel", "n_points": int(P_voxel.shape[0])}, 
               {"n": n, "h0": h0})

    # Step 3: DBSCAN clustering with reduced Z weight
    P_scaled = P_voxel.copy()
    P_scaled[:, 2] *= z_weight  # Reduce Z influence for XY-dominant clustering
    
    labels = np.array(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_scaled))
        .cluster_dbscan(eps=eps_xy_mm, min_points=min_pts, print_progress=False)
    )

    # Step 4: Select best cluster
    if labels.max() >= 0:
        best_cluster_idx, best_score = None, -1e9
        heights_voxel = P_voxel @ n
        
        for cluster_id in range(labels.max() + 1):
            cluster_indices = np.where(labels == cluster_id)[0]
            if cluster_indices.size < min_pts:
                continue
                
            cluster_points = P_voxel[cluster_indices]
            
            # Score based on median height and compactness
            median_height = float(np.median(heights_voxel[cluster_indices] - h0))
            xy_extent = np.ptp(cluster_points[:, :2], axis=0)
            footprint_area = float(xy_extent[0] * xy_extent[1])
            
            # Prefer higher objects, slightly penalize very large footprints
            score = median_height - 0.0005 * footprint_area
            
            if score > best_score:
                best_score, best_cluster_idx = score, cluster_indices
        
        if best_cluster_idx is not None:
            parcel_pcd = cloud.select_by_index(best_cluster_idx)
            return (parcel_pcd, 
                   {"status": "success", "method": "dbscan", "n_points": int(len(best_cluster_idx))}, 
                   {"n": n, "h0": h0})

    # Step 5: Fallback method - keep top fraction by height
    heights_voxel = P_voxel @ n
    height_cutoff = np.percentile(heights_voxel - h0, 100 * (1.0 - top_frac))
    top_indices = np.where((heights_voxel - h0) >= height_cutoff)[0]
    
    if top_indices.size == 0:
        return (o3d.geometry.PointCloud(), 
               {"status": "failed", "reason": "fallback_empty"}, 
               {"n": n, "h0": h0})
    
    parcel_pcd = cloud.select_by_index(top_indices)
    if voxel_mm and not parcel_pcd.is_empty():
        parcel_pcd = parcel_pcd.voxel_down_sample(voxel_size=voxel_mm)
    
    return (parcel_pcd, 
           {"status": "success", "method": "fallback_top", "n_points": int(len(parcel_pcd.points))}, 
           {"n": n, "h0": h0})