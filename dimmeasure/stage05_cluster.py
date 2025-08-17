# dimmeasure/stage05_cluster.py
from __future__ import annotations
import argparse, pathlib, sys, numpy as np, open3d as o3d

def load_points_mm(path: pathlib.Path, assume_units="m") -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty or unreadable: {path}")
    P = np.asarray(pcd.points, dtype=np.float64)
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
    if n[2] < 0:  # make normal point upward
        a,b,c,d,n = -a,-b,-c,-d,-n
    return (a,b,c,d), np.asarray(inliers, dtype=int), n

def sd_to_plane(P: np.ndarray, plane):
    a,b,c,d = plane
    return P[:,0]*a + P[:,1]*b + P[:,2]*c + d  # normal is unit-length

def cluster_points(P: np.ndarray, eps_mm: float, min_pts: int):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    labels = np.array(p.cluster_dbscan(eps=eps_mm, min_points=min_pts, print_progress=False))
    return labels

def main():
    ap = argparse.ArgumentParser("Stage05: Cluster points above plane and select parcel")
    ap.add_argument("--in_glob", required=True, help="e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m","cm","mm"])
    ap.add_argument("--ransac-dist", type=float, default=2.0)
    ap.add_argument("--ransac-iters", type=int, default=1200)
    ap.add_argument("--h-min", type=float, default=5.0)
    ap.add_argument("--h-max", type=float, default=800.0)
    ap.add_argument("--eps", type=float, default=10.0, help="DBSCAN eps (mm)")
    ap.add_argument("--min-pts", type=int, default=200, help="DBSCAN min points per cluster")
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr); sys.exit(1)

    print(f"[Stage05] DBSCAN(eps={args.eps} mm, min_pts={args.min_pts}) on above-plane band [{args.h_min},{args.h_max}] mm")
    for f in files:
        try:
            P = remove_nans(load_points_mm(f, assume_units=args.assume_units))
            N0 = P.shape[0]

            plane, inliers, n = ransac_plane(P, args.ransac_dist, args.ransac_iters)
            mask = np.ones(N0, dtype=bool); mask[inliers] = False
            P_np = P[mask]
            sd = sd_to_plane(P_np, plane)
            band = (sd >= args.h_min) & (sd <= args.h_max)
            C = P_np[band]
            if C.shape[0] == 0:
                print(f"{f} :: kept 0 after band; try larger h_max"); continue

            labels = cluster_points(C, eps_mm=args.eps, min_pts=args.min_pts)
            n_clusters = labels.max() + 1 if labels.size and labels.max() >= 0 else 0

            if n_clusters == 0:
                print(f"{f} :: 0 clusters (all noise). Try larger eps or smaller min-pts.")
                continue

            sizes = [int((labels == i).sum()) for i in range(n_clusters)]
            best_id = int(np.argmax(sizes))
            best_sz = sizes[best_id]

            print(f"{f} :: kept={C.shape[0]} pts | clusters={n_clusters} | "
                  f"largest_id={best_id} size={best_sz} ({best_sz*100.0/max(C.shape[0],1):.1f}%)")
        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
