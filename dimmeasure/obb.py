# dimmeasure/obb.py
import numpy as np

def obb_dims_and_volume(pcd_mm):
    """
    pcd_mm: Open3D PointCloud in **millimeters** (after preprocess).
    Returns (L_mm, W_mm, H_mm, obb, box_volume_mm3).
    """
    obb = pcd_mm.get_oriented_bounding_box()
    L, W, H = sorted(obb.extent, reverse=True)  # mm
    vol = float(L * W * H)                      # mm^3
    return float(L), float(W), float(H), obb, vol
