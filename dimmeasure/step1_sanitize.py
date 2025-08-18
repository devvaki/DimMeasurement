# Step 1 — Pair files, basic sanity stats, and quick visualization (no geometry math)
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import re

DATA_DIR = Path("./data")                  # <- change if needed
CSV_PATH = Path("./eval_data.csv")
OUT_SUMMARY = Path("./step1_file_check.csv")
PLOTS_DIR = Path("./step1_plots"); PLOTS_DIR.mkdir(exist_ok=True)

def parse_dims(s):
    # "50x33x14.4" -> [50,33,14.4]
    return [float(x) for x in re.split(r"[xX]", str(s)) if x]

def load_points_ply(ply_path):
    obj = trimesh.load(ply_path)
    if hasattr(obj, "vertices"):
        pts = np.asarray(obj.vertices)
    elif isinstance(obj, trimesh.Scene):
        pts = None
        for g in obj.geometry.values():
            if hasattr(g, "vertices"):
                pts = np.asarray(g.vertices); break
        if pts is None: raise ValueError("No vertices in scene")
    else:
        raise ValueError(f"Unsupported trimesh object: {type(obj)}")
    # keep only finite rows
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

# 1) Pair .ply with .png
rows = []
for ply in sorted(DATA_DIR.glob("*-pointcloud.ply")):
    stem = ply.stem.replace("-pointcloud","")
    png = DATA_DIR / f"{stem}-rgba.png"
    rows.append({"id": stem, "ply_file": str(ply), "png_file": str(png), "png_exists": png.exists()})
pairs_df = pd.DataFrame(rows)

# 2) Join CSV dims (if present)
if CSV_PATH.exists():
    csv = pd.read_csv(CSV_PATH)
    # Normalize eval_data.csv columns so the script can find id + dims
    csv.columns = [c.strip() for c in csv.columns]  # trim spaces


    # Create an 'id' column (e.g., from "3163484-rgba.png")
    if "id" not in csv.columns:
        if "File Name" in csv.columns:
            csv["id"] = csv["File Name"].str.extract(r"(\d{6,})")[0]
        elif "file_name" in csv.columns:
            csv["id"] = csv["file_name"].str.extract(r"(\d{6,})")[0]

    # Ensure a 'dims' column exists (text like "50x33x14.4" in cm)
    if "dims" not in csv.columns:
        if "Actual Box volume (LxBxH)" in csv.columns:
            csv = csv.rename(columns={"Actual Box volume (LxBxH)": "dims"})
        elif {"Length (cm)", "Width (cm)", "Height (cm)"} <= set(csv.columns):
            csv["dims"] = (
                csv["Length (cm)"].astype(str) + "x" +
                csv["Width (cm)"].astype(str)  + "x" +
                csv["Height (cm)"].astype(str)
            )
        else:
            # no usable dims columns; leave missing so step1 just warns
            csv["dims"] = None

    csv_norm = csv[["id", "dims"]].copy()
    pairs_df = pairs_df.merge(csv_norm, on="id", how="left")
else:
    pairs_df["dims"] = np.nan

# 3) Basic stats per PLY + save a simple visualization
stats = []
for _, r in pairs_df.iterrows():
    ply_path = r["ply_file"]
    png_ok = r["png_exists"]
    dims = r["dims"]
    try:
        pts = load_points_ply(ply_path)
        n = len(pts)
        mins = pts.min(axis=0) if n else [np.nan]*3
        maxs = pts.max(axis=0) if n else [np.nan]*3

        # --- quick visualization (downsample & AABB box) ---
        ds = pts if n <= 80000 else pts[np.random.choice(n, 80000, replace=False)]
        aabb_min, aabb_max = ds.min(axis=0), ds.max(axis=0)
        # corners of AABB
        x0,y0,z0 = aabb_min; x1,y1,z1 = aabb_max
        box_edges = [
            [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)], [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
            [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)], [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
            [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)], [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)]
        ]

        import matplotlib
        matplotlib.use("Agg")  # non-interactive
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(ds[::10,0], ds[::10,1], ds[::10,2], s=0.2, c="gray")  # sparse scatter
        for (p,q) in box_edges:
            ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], c="red", linewidth=1)
        ax.set_title(f"{r['id']}  |  points={n}  |  dims(csv)={dims}")
        ax.set_xlabel("X (cloud units)"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=18, azim=35)
        fig.tight_layout()
        out_png = PLOTS_DIR / f"{r['id']}_quick.png"
        fig.savefig(out_png, dpi=140); plt.close(fig)

        stats.append({
            "id": r["id"], "png_exists": png_ok, "csv_dims": dims,
            "n_points": n,
            "min_x": float(mins[0]), "min_y": float(mins[1]), "min_z": float(mins[2]),
            "max_x": float(maxs[0]), "max_y": float(maxs[1]), "max_z": float(maxs[2]),
            "quick_plot": str(out_png)
        })
    except Exception as e:
        stats.append({
            "id": r["id"], "png_exists": png_ok, "csv_dims": dims,
            "n_points": 0, "min_x": np.nan, "min_y": np.nan, "min_z": np.nan,
            "max_x": np.nan, "max_y": np.nan, "max_z": np.nan,
            "quick_plot": "", "error": str(e)
        })

summary = pd.DataFrame(stats).sort_values("id")
summary.to_csv(OUT_SUMMARY, index=False)

print("=== Step 1 summary ===")
print(summary[["id","png_exists","csv_dims","n_points","quick_plot"]].to_string(index=False))
print(f"\nSaved CSV: {OUT_SUMMARY}")
print(f"Saved plots in: {PLOTS_DIR.resolve()}")
missing = summary["csv_dims"].isna().sum()
if missing:
    print(f"WARNING: {missing} item(s) missing CSV dims.")
