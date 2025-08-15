import numpy as np
import open3d as o3d

def load_and_scale_mm(ply_path):
    #loading a ply file and ensure coord are in mm
    pcd = o3d.io.read_point_cloud(str(ply_path))
    P = np.asarray(pcd.points)
    P = P[np.isfinite(P).all(axis=1)] #remove NaN/Inf

    span = float(np.max(np.ptp(P, axis=0)))

    #if data is in m -> mm
    if 1.0 <= span <= 20.0:  # assume meters, convert to mm
        P *= 1000.0
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))

#estimate vertical axis using PCA and return heights above table/slab
def pca_height(P):
    Pc = P - P.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    n = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12) #vertical unit vector
    h = P @ n
    h0 = float(np.percentile(h, 5)) #table height level
    return n, h, h0

def isolate_parcel(pcd_mm,
                   delta_low=10.0, delta_high=300.0,
                   voxel_mm=3.0,
                   eps_xy_mm=30.0, min_pts=60,
                   z_weight=0.15, top_frac=0.20):
    

    P = np.asarray(pcd_mm.points)
    n, h, h0 = pca_height(P)

    #1 height filter (above table)
    mask = (h - h0 > delta_low) & (h - h0 < delta_high)
    Pw = P[mask]
    if Pw.size == 0:
        return o3d.geometry.PointCloud(), {"note": "empty after height window"}

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw))
    if voxel_mm:
        cloud = cloud.voxel_down_sample(voxel_size=voxel_mm)
    P1 = np.asarray(cloud.points)
    if len(P1) < min_pts:
        return cloud, {"note": "too few after voxel", "n": len(P1)}

    #2 DBSCAN to get parcel footprint
    Psc = P1.copy()
    Psc[:, 2] *= z_weight
    labs = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Psc))
                    .cluster_dbscan(eps=eps_xy_mm, min_points=min_pts, print_progress=False))

    if labs.max() >= 0:
        h1 = P1 @ n
        best, score = None, -1e9
        for k in range(labs.max() + 1):
            idx = np.where(labs == k)[0]
            if idx.size < min_pts:
                continue
            pts = P1[idx]
            med_h = float(np.median(h1[idx] - h0))
            ex = np.ptp(pts[:, :2], axis=0)
            area = float(ex[0] * ex[1])
            sc = med_h - 0.0005 * area
            if sc > score:
                score, best = sc, idx
        if best is not None:
            return cloud.select_by_index(best), {"note": "dbscan_ok", "n": int(len(best))}

    #3 fallback to top height band
    h1 = P1 @ n
    cutoff = np.percentile(h1 - h0, 100 * (1.0 - top_frac))
    keep = np.where((h1 - h0) >= cutoff)[0]
    if keep.size == 0:
        return o3d.geometry.PointCloud(), {"note": "fallback_empty"}
    pc = cloud.select_by_index(keep)
    if voxel_mm:
        pc = pc.voxel_down_sample(voxel_size=voxel_mm)
    return pc, {"note": "fallback_top_band", "n": int(len(pc.points))}
