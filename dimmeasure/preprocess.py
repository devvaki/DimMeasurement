import numpy as np
import open3d as o3d

# ---- Step 1: Denoise ----
def denoise(pcd: o3d.geometry.PointCloud,
            nb_neighbors: int = 30,
            std_ratio: float = 1.5) -> o3d.geometry.PointCloud:
    """
    Statistical outlier removal (assumes coordinates are in millimeters).
    """
    clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return clean

# ---- Step 2: Plane → crop above → cluster → pick parcel ----
def segment_plane_and_crop_above(
    pcd_mm: o3d.geometry.PointCloud,
    delta_mm: float = 6.0,
    voxel_mm: float = 3.0,
) -> o3d.geometry.PointCloud:
    """
    Isolate the parcel from the AGV/table.

    1) Fit plane in **millimeters**.
    2) Keep points above plane by delta_mm.
    3) DBSCAN cluster and choose the cluster with the greatest height over plane,
       while rejecting very flat/huge rims.
    4) Voxel downsample (mm).
    """
    if pcd_mm.is_empty():
        return pcd_mm

    # --- Plane segmentation in mm ---
    plane_model, inliers = pcd_mm.segment_plane(
        distance_threshold=1.5,  # ~1.5 mm tolerance
        ransac_n=3,
        num_iterations=2000
    )
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)

    pts = np.asarray(pcd_mm.points)
    signed = (pts @ n + d)  # + above plane, in mm

    # --- Crop: keep only points at least delta_mm above plane ---
    keep_idx = np.where(signed > float(delta_mm))[0]
    if keep_idx.size == 0:
        return o3d.geometry.PointCloud()
    crop = pcd_mm.select_by_index(keep_idx)

    # --- Cluster in mm and pick the tallest valid cluster ---
    labels = np.array(crop.cluster_dbscan(eps=8.0, min_points=30))  # eps ~ 8 mm
    if labels.size > 0 and labels.max() >= 0:
        crop_pts = np.asarray(crop.points)
        crop_signed = (crop_pts @ n + d)

        best_lab = None
        best_height = -1.0
        for lab in range(labels.max() + 1):
            idx = np.where(labels == lab)[0]
            if idx.size < 30:
                continue

            cluster = crop.select_by_index(idx)

            # height above plane for this cluster (95th percentile is robust)
            h95 = float(np.percentile(crop_signed[idx], 95))

            # reject super-flat, huge-footprint rims
            aabb = cluster.get_axis_aligned_bounding_box()
            ex = aabb.get_extent()  # [ex, ey, ez] in mm
            footprint_area = float(ex[0] * ex[1])  # mm^2
            if ex.min() < 8.0 and footprint_area > (600.0 * 600.0):
                continue

            if h95 > best_height:
                best_height = h95
                best_lab = lab

        if best_lab is not None:
            crop = crop.select_by_index(np.where(labels == best_lab)[0])

    # --- Voxel downsample for stability/perf ---
    if voxel_mm and voxel_mm > 0:
        crop = crop.voxel_down_sample(voxel_size=float(voxel_mm))

    return crop
