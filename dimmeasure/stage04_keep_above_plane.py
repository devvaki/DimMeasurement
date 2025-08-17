# dimmeasure/stage04_keep_above_plane.py
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import open3d as o3d

def load_points_mm(path: pathlib.Path, assume_units="m") -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty or unreadable: {path}")
    pts = np.asarray(pcd.points, dtype=np.float64)
    if assume_units == "m":   pts *= 1000.0
    elif assume_units == "cm": pts *= 10.0
    elif assume_units == "mm": pass
    else: raise ValueError("assume_units must be m/cm/mm")
    return pts

def remove_nans(pts: np.ndarray) -> np.ndarray:
    return pts[np.isfinite(pts).all(axis=1)]

def ransac_plane(pts: np.ndarray, dist_mm: float, iters: int):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = pcd.segment_plane(distance_threshold=dist_mm,
                                       ransac_n=3,
                                       num_iterations=iters)
    a,b,c,d = model
    n = np.array([a,b,c], dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)
    # Ensure plane normal points "up" (positive Z)
    if n[2] < 0:
        n = -n
        a,b,c,d = -a,-b,-c,-d
    inliers = np.asarray(inliers, dtype=np.int64)
    return (a,b,c,d), n, inliers

def signed_distance_mm(pts: np.ndarray, plane):
    a,b,c,d = plane
    # ax+by+cz+d=0 → distance = (a x + b y + c z + d) / ||n|| ; ||n||=1 after normalization above
    return pts[:,0]*a + pts[:,1]*b + pts[:,2]*c + d

def main():
    ap = argparse.ArgumentParser("Stage04: Keep points above the detected plane (height band)")
    ap.add_argument("--in_glob", required=True, help="e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m","cm","mm"])
    ap.add_argument("--ransac-dist", type=float, default=2.0, help="RANSAC inlier distance (mm)")
    ap.add_argument("--ransac-iters", type=int, default=1200, help="RANSAC iterations")
    ap.add_argument("--h-min", type=float, default=5.0, help="Keep points with height ≥ h_min (mm) above plane")
    ap.add_argument("--h-max", type=float, default=800.0, help="Keep points with height ≤ h_max (mm) above plane")
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr); sys.exit(1)

    print(f"[Stage04] Filtering to height band [{args.h_min}, {args.h_max}] mm above plane")
    for f in files:
        try:
            pts = remove_nans(load_points_mm(f, assume_units=args.assume_units))
            N0 = pts.shape[0]

            plane, n, inliers = ransac_plane(pts, args.ransac_dist, args.ransac_iters)

            # Remove the plane inliers first (optional; band filter also excludes them)
            mask_not_plane = np.ones(N0, dtype=bool)
            mask_not_plane[inliers] = False
            pts_np = pts[mask_not_plane]

            # Compute signed distance for the remaining points
            sd = signed_distance_mm(pts_np, plane)
            keep = (sd >= args.h_min) & (sd <= args.h_max)
            kept = pts_np[keep]
            Nk = kept.shape[0]

            print(f"{f} :: kept {Nk}/{N0} ({100.0*Nk/max(N0,1):.1f}%) | "
                  f"plane inliers removed={len(inliers)} | "
                  f"h_min={args.h_min} h_max={args.h_max} mm")
        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
