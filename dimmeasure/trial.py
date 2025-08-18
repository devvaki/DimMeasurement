from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import open3d as o3d

# --- Matplotlib (headless) for PNGs ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_points_mm(path: pathlib.Path, assume_units: str = "m") -> np.ndarray:
    """Load a PLY point cloud and scale to millimetres."""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty or unreadable: {path}")
    pts = np.asarray(pcd.points, dtype=np.float64)
    if assume_units == "m":
        pts *= 1000.0
    elif assume_units == "cm":
        pts *= 10.0
    elif assume_units == "mm":
        pass
    else:
        raise ValueError("assume_units must be m/cm/mm")
    # keep only finite rows (defensive)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def robust_box(pts: np.ndarray):
    """Return p5, p95 corners and span along each axis (robust to outliers)."""
    p5  = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    span = p95 - p5
    return p5, p95, span


def aabb_box(pts: np.ndarray):
    """Return min, max corners and span of the AABB."""
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    return mn, mx, span


def obb_box_pca(
    pts: np.ndarray,
    trim_q: float = 0.05,
):
    """
    Compute an Oriented Bounding Box via PCA.
    Steps:
      1) Trim extremes per axis (keep [q, 1-q] quantile slab) to reduce outlier influence.
      2) Compute PCA on trimmed set -> eigenvectors (columns of R) and center (mean).
      3) Transform all points to PCA frame, get min/max in that frame -> box extents.
      4) Return world-space corners and spans (lengths along principal axes).
    Returns:
      (corners_world_8x3, span_xyz, R, center)
    """
    if pts.shape[0] < 4:
        raise ValueError("Too few points for OBB")

    # Trim to reduce outlier impact on PCA
    if trim_q > 0:
        lo = np.percentile(pts, 100 * trim_q, axis=0)
        hi = np.percentile(pts, 100 * (1.0 - trim_q), axis=0)
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        trimmed = pts[mask]
        if trimmed.shape[0] >= 16:  # need enough for stable PCA
            pca_pts = trimmed
        else:
            pca_pts = pts
    else:
        pca_pts = pts

    # PCA
    mu = pca_pts.mean(axis=0)
    X = pca_pts - mu
    # covariance & eigen-decomposition
    C = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(C)  # eigvecs: columns are eigenvectors for ascending eigvals
    # sort by descending variance
    idx = np.argsort(eigvals)[::-1]
    R = eigvecs[:, idx]
    # ensure right-handed basis (det > 0)
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # project ALL points to PCA frame to bound the whole cloud
    Y = (pts - mu) @ R  # local PCA coords
    lo = Y.min(axis=0)
    hi = Y.max(axis=0)
    span = hi - lo

    # 8 corners in local PCA frame
    corners_local = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ])

    # back to world
    corners_world = corners_local @ R.T + mu
    return corners_world, span, R, mu


def format_span(tag: str, span: np.ndarray, n: int) -> str:
    return f"{tag} N={n:7d} | span(mm) X={span[0]:.1f} Y={span[1]:.1f} Z={span[2]:.1f}"


def edge_segments_from_corners(corners8: np.ndarray):
    """
    Build the 12 edges from 8 corners in the standard indexing used above:
      0: (lo,lo,lo)  1:(hi,lo,lo)  2:(hi,hi,lo)  3:(lo,hi,lo)
      4: (lo,lo,hi)  5:(hi,lo,hi)  6:(hi,hi,hi)  7:(lo,hi,hi)
    """
    idx_pairs = [
        (0,1),(1,2),(2,3),(3,0),  # bottom rectangle
        (4,5),(5,6),(6,7),(7,4),  # top rectangle
        (0,4),(1,5),(2,6),(3,7)   # verticals
    ]
    return [(corners8[i], corners8[j]) for (i,j) in idx_pairs]


def edge_segments(c0, c1):
    """12 edges of an axis-aligned box given opposite corners c0=(x0,y0,z0), c1=(x1,y1,z1)."""
    x0, y0, z0 = c0
    x1, y1, z1 = c1
    return [
        [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
        [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
        [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)],
    ]


def quick_plot(
    pts: np.ndarray,
    aabb_c0, aabb_c1,
    r_c0, r_c1,
    out_path: pathlib.Path,
    title: str = "",
    max_points: int = 80000,
    stride: int = 10,
    obb_corners: np.ndarray | None = None
):
    """Save a 3D scatter with AABB (red), robust 5–95 (blue), and optional OBB (green),
    plus a legend and on-plot labels for each box."""
    n = pts.shape[0]
    if n == 0:
        return
    # downsample (random) then stride render for speed
    ds = pts if n <= max_points else pts[np.random.choice(n, max_points, replace=False)]
    show = ds[::stride]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(show[:, 0], show[:, 1], show[:, 2], s=0.2, c="gray")

    # --- helpers
    def draw_box_edges(edges, color):
        for p, q in edges:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c=color, linewidth=1, label="_nolegend_")

    def box_label_anchor_from_corners(c0, c1):
        # pick the "top, far, right" corner for a stable label anchor
        x = max(c0[0], c1[0]); y = max(c0[1], c1[1]); z = max(c0[2], c1[2])
        return np.array([x, y, z], dtype=float)

    def label_box(text, anchor_xyz, color):
        # small offset so text doesn't sit on the edge line
        offset = np.array([0.02, 0.02, 0.02]) * np.linalg.norm(anchor_xyz) if np.linalg.norm(anchor_xyz) > 0 else np.array([10,10,10])
        pos = anchor_xyz + offset
        ax.text(pos[0], pos[1], pos[2], text, color=color, fontsize=9, zorder=10)

    # --- AABB (red)
    draw_box_edges(edge_segments(aabb_c0, aabb_c1), "red")
    label_box("AABB (min–max)", box_label_anchor_from_corners(aabb_c0, aabb_c1), "red")

    # --- Robust 5–95 (blue)
    draw_box_edges(edge_segments(r_c0, r_c1), "blue")
    label_box("AABB 5–95%", box_label_anchor_from_corners(r_c0, r_c1), "blue")

    # --- OBB (green), if provided
    if obb_corners is not None:
        from numpy import array
        # build edges from 8 OBB corners
        for p, q in edge_segments_from_corners(obb_corners):
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c="green", linewidth=1, label="_nolegend_")
        # label near the max-corner in world space
        obb_anchor = obb_corners[np.argmax(np.sum(obb_corners, axis=1))]
        label_box("OBB (PCA)", obb_anchor, "green")

    # Legend stubs
    ax.plot([], [], [], c="red",  label="AABB (min–max)")
    ax.plot([], [], [], c="blue", label="AABB 5–95%")
    if obb_corners is not None:
        ax.plot([], [], [], c="green", label="OBB (PCA)")
    ax.legend(loc="upper right", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(elev=18, azim=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Stage01: Load PLY, print stats, and (optionally) save plots")
    ap.add_argument("--in_glob", required=True, help="Glob for files, e.g. 'data/*.ply'")
    ap.add_argument("--assume-units", default="m", choices=["m", "cm", "mm"])
    ap.add_argument("--save-plots", type=str, help="Directory to save quick PNG plots (optional)")
    ap.add_argument("--max-points", type=int, default=80000, help="Max points to keep before stride plotting")
    ap.add_argument("--stride", type=int, default=10, help="Render every k-th point for scatter")
    ap.add_argument("--with-obb", action="store_true", help="Compute and report OBB via PCA")
    ap.add_argument("--obb-trim", type=float, default=0.05, help="Quantile trim for PCA (e.g., 0.05 keeps 5–95%%). Set 0 to disable.")
    args = ap.parse_args()

    files = sorted(pathlib.Path().glob(args.in_glob))
    if not files:
        print(f"No files match: {args.in_glob}", file=sys.stderr)
        sys.exit(1)

    out_dir = None
    if args.save_plots:
        out_dir = pathlib.Path(args.save_plots)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage01] Loading {len(files)} file(s) with units={args.assume_units} → mm")
    for f in files:
        try:
            pts = load_points_mm(f, assume_units=args.assume_units)
            n = pts.shape[0]

            # robust 5–95
            r_c0, r_c1, r_span = robust_box(pts)
            # AABB min–max
            a_c0, a_c1, a_span = aabb_box(pts)

            # console output (AABB + 5–95)
            out_line = (
                f"{f} :: "
                f"{format_span('[AABB]   ', a_span, n)} | "
                f"{format_span('[5-95%] ', r_span, n)}"
            )

            obb_corners = None
            if args.with_obb:
                try:
                    obb_corners, obb_span, R, mu = obb_box_pca(pts, trim_q=args.obb_trim)
                    out_line += " | " + format_span("[OBB]    ", obb_span, n)
                except Exception as e:
                    out_line += f" | [OBB] ERROR: {e}"

            print(out_line)

            # optional plot
            if out_dir is not None:
                out_png = out_dir / f"{f.stem}_quick.png"
                quick_plot(
                    pts,
                    a_c0, a_c1,
                    r_c0, r_c1,
                    out_png,
                    title=f.stem,
                    max_points=args.max_points,
                    stride=args.stride,
                    obb_corners=obb_corners
                )

        except Exception as e:
            print(f"{f} :: ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
