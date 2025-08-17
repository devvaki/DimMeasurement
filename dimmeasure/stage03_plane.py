# dimmeasure/stage03_plane.py
from __future__ import annotations
import argparse, pathlib, sys, numpy as np, open3d as o3d
from math import acos, degrees

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
    # ax + by + cz + d = 0
    model, inliers = pcd.segment_plane(distance_threshold=dist_mm,
                                       ransac_n=3,
                                       num_iterations=iters)
    a, b, c, d = model
    n = np.array([a, b, c], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    angle_to_z = degrees(acos(np.clip(abs(n[2]), 0.0, 1.0)))  # 0° if parallel to Z
    return (a, b, c, d), np.array(inliers, dtype=np.int64), n, angle_to_z

def main():
    ap = argparse.ArgumentParser("Stage03: Detect dominant plane (RANSAC)")
    ap.add_argument("--in_glob", required=True, help="e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m","cm","mm"])
    ap.add_argument("--dist-mm", type=float, default=2.0, help="RANSAC inlier distance (mm)")
    ap.add_argument("--iters", type=int, default=1200, help="RANSAC iterations")
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr); sys.exit(1)

    print(f"[Stage03] Plane detection on {len(files)} file(s)  (dist={args.dist_mm} mm, iters={args.iters})")
    for f in files:
        try:
            pts = remove_nans(load_points_mm(f, assume_units=args.assume_units))
            N = pts.shape[0]
            model, inliers, n, ang = ransac_plane(pts, args.dist_mm, args.iters)
            frac = 100.0 * len(inliers) / max(N, 1)
            a,b,c,d = model
            print(f"{f} :: inliers={len(inliers)}/{N} ({frac:.1f}%) | "
                  f"plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.1f}=0 | "
                  f"|n|=[{n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}] | angle_to_Z≈{ang:.1f}°")
        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
