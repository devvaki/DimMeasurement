# dimmeasure/stage06_measure_obb.py
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import open3d as o3d

def load_points_mm(path: pathlib.Path, assume_units="m") -> np.ndarray:
    p = o3d.io.read_point_cloud(str(path))
    if p.is_empty():
        raise ValueError(f"Empty or unreadable: {path}")
    P = np.asarray(p.points, dtype=np.float64)
    if assume_units == "m":   P *= 1000.0
    elif assume_units == "cm": P *= 10.0
    elif assume_units == "mm": pass
    else: raise ValueError("assume_units must be m/cm/mm")
    return P

def remove_nans(P: np.ndarray) -> np.ndarray:
    return P[np.isfinite(P).all(axis=1)]

def ransac_plane(P: np.ndarray, dist_mm: float, iters: int):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    model, inliers = pcd.segment_plane(distance_threshold=dist_mm, ransac_n=3, num_iterations=iters)
    a,b,c,d = model
    n = np.array([a,b,c], dtype=float); n /= (np.linalg.norm(n)+1e-12)
    if n[2] < 0:  # make “up”
        a,b,c,d,n = -a,-b,-c,-d,-n
    return (a,b,c,d), np.asarray(inliers, dtype=int), n

def sd_to_plane(P: np.ndarray, plane):
    a,b,c,d = plane
    return P[:,0]*a + P[:,1]*b + P[:,2]*c + d  # unit normal assumed

def dbscan_cluster(P: np.ndarray, eps_mm: float, min_pts: int):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    labels = np.array(pc.cluster_dbscan(eps=eps_mm, min_points=min_pts, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return None, None
    k = labels.max() + 1
    sizes = [int((labels == i).sum()) for i in range(k)]
    best = int(np.argmax(sizes))
    return best, labels

def obb_lwh(P: np.ndarray):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    obb = pc.get_oriented_bounding_box()
    L, W, H = obb.extent  # already length/width/height in mm because points are mm
    # Sort so L >= W >= H for consistency
    dims = sorted([L, W, H], reverse=True)
    return dims[0], dims[1], dims[2], obb

def main():
    ap = argparse.ArgumentParser("Stage06: OBB on largest cluster (above-plane band)")
    ap.add_argument("--in_glob", required=True, help="e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m","cm","mm"])
    ap.add_argument("--ransac-dist", type=float, default=2.0)
    ap.add_argument("--ransac-iters", type=int, default=1200)
    ap.add_argument("--h-min", type=float, default=5.0)
    ap.add_argument("--h-max", type=float, default=800.0)
    ap.add_argument("--eps", type=float, default=25.0)
    ap.add_argument("--min-pts", type=int, default=60)
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr); sys.exit(1)

    print(f"[Stage06] OBB on largest DBSCAN cluster; band [{args.h_min},{args.h_max}] mm; eps={args.eps}, min_pts={args.min_pts}")
    for f in files:
        try:
            P = remove_nans(load_points_mm(f, assume_units=args.assume_units))
            N = P.shape[0]
            plane, inliers, _ = ransac_plane(P, args.ransac_dist, args.ransac_iters)

            # remove plane, height-band
            mask = np.ones(N, dtype=bool); mask[inliers] = False
            P2 = P[mask]
            sd = sd_to_plane(P2, plane)
            band = (sd >= args.h_min) & (sd <= args.h_max)
            A = P2[band]
            if A.shape[0] == 0:
                print(f"{f} :: 0 points after band — try larger h_max"); continue

            # cluster & select largest
            best, labels = dbscan_cluster(A, eps_mm=args.eps, min_pts=args.min_pts)
            if best is None:
                print(f"{f} :: 0 clusters — try larger eps or smaller min-pts"); continue
            G = A[labels == best]

            # OBB dims
            L, W, H, _ = obb_lwh(G)
            print(f"{f} :: cluster_size={G.shape[0]} | OBB (mm): L={L:.1f}, W={W:.1f}, H={H:.1f} | BoxVol={L*W*H:.0f} mm^3")
        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
