# dimmeasure/stage01_load.py
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import open3d as o3d

def load_points_mm(path: pathlib.Path, assume_units="m") -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty or unreadable: {path}")
    pts = np.asarray(pcd.points, dtype=np.float64)
    # scale to millimetres
    if assume_units == "m":   pts *= 1000.0
    elif assume_units == "cm": pts *= 10.0
    elif assume_units == "mm": pass
    else: raise ValueError("assume_units must be m/cm/mm")
    return pts

def summarize(pts: np.ndarray) -> str:
    n = pts.shape[0]
    # robust 5–95% ranges to avoid outliers skewing
    p5, p95 = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
    span = p95 - p5
    return f"N={n:7d} | span(mm) X={span[0]:.1f} Y={span[1]:.1f} Z={span[2]:.1f}"

def main():
    ap = argparse.ArgumentParser("Stage01: Load PLY and print basic stats")
    ap.add_argument("--in_glob", required=True, help="Glob for files, e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m","cm","mm"])
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr)
        sys.exit(1)

    print(f"[Stage01] Loading {len(files)} file(s) with units={args.assume_units} → mm")
    for f in files:
        try:
            pts = load_points_mm(f, assume_units=args.assume_units)
            print(f"{f} :: {summarize(pts)}")
        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
