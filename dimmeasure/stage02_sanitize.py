# dimmeasure/stage02_sanitize.py
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
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]

def sor(pts: np.ndarray, mean_k: int = 40, std_mul: float = 1.2) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=mean_k, std_ratio=std_mul)
    return np.asarray(pcd.points, dtype=np.float64)

def ror(pts: np.ndarray, radius_mm: float = 5.0, min_n: int = 5) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, _ = pcd.remove_radius_outlier(nb_points=min_n, radius=radius_mm)
    return np.asarray(pcd.points, dtype=np.float64)

def summarize(pts: np.ndarray) -> str:
    if pts.size == 0:
        return "N=0"
    p5, p95 = np.nanpercentile(pts, 5, axis=0), np.nanpercentile(pts, 95, axis=0)
    span = p95 - p5
    return f"N={pts.shape[0]:7d} | span(mm) X={span[0]:.1f} Y={span[1]:.1f} Z={span[2]:.1f}"

def main():
    ap = argparse.ArgumentParser("Stage02: Sanitize (NaNs + outliers)")
    ap.add_argument("--in_glob", required=True, help="e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m","cm","mm"])
    ap.add_argument("--no-sor", action="store_true", help="disable Statistical Outlier Removal")
    ap.add_argument("--no-ror", action="store_true", help="disable Radius Outlier Removal")
    ap.add_argument("--sor-k", type=int, default=40)
    ap.add_argument("--sor-std", type=float, default=1.2)
    ap.add_argument("--ror-radius", type=float, default=5.0)
    ap.add_argument("--ror-min", type=int, default=5)
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr)
        sys.exit(1)

    print(f"[Stage02] Sanitizing {len(files)} file(s)")
    for f in files:
        try:
            raw = load_points_mm(f, assume_units=args.assume_units)
            n0 = raw.shape[0]

            # Remove NaNs/Inf
            pts = remove_nans(raw)
            n_nan_removed = n0 - pts.shape[0]

            # Outlier removal (optional but recommended)
            if not args.no_sor and pts.shape[0] >= 10:
                pts = sor(pts, mean_k=args.sor_k, std_mul=args.sor_std)
            if not args.no_ror and pts.shape[0] >= 10:
                pts = ror(pts, radius_mm=args.ror_radius, min_n=args.ror_min)

            print(f"{f} :: {summarize(pts)} | drop NaN={n_nan_removed}")
        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
