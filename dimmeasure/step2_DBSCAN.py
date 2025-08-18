# Step 2 — Fit deck plane (RANSAC), remove it, isolate parcel, and save visuals
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save images to disk
import matplotlib.pyplot as plt
import re, os, trimesh
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN

# ====== CONFIG ======
DATA_DIR = Path("./data")
CSV_PATH = Path("./eval_data_normalized.csv")   # from previous step (optional)
PLOTS_DIR = Path("./step2_plots"); PLOTS_DIR.mkdir(exist_ok=True)
OUT_CSV   = Path("./step2_parcel_stats.csv")

PLANE_THRESH = 0.02   # points within this |z - z_pred| are deck (in cloud units)
Z_ABOVE      = 0.05   # keep only points this much above deck (in cloud units)
DBSCAN_EPS   = 0.05   # clustering radius (cloud units)
DBSCAN_MIN   = 80     # minimum points per cluster

# ====== helpers ======
def parse_dims(s):
    parts = [p for p in re.split(r"[xX]", str(s)) if p]
    return [float(p) for p in parts[:3]] if parts else [np.nan, np.nan, np.nan]

def load_pts_ply(p):
    obj = trimesh.load(p)
    if hasattr(obj, "vertices"):
        pts = np.asarray(obj.vertices)
    elif isinstance(obj, trimesh.Scene):
        pts = None
        for g in obj.geometry.values():
            if hasattr(g, "vertices"):
                pts = np.asarray(g.vertices); break
        if pts is None: raise ValueError("no vertices in scene")
    else:
        raise ValueError("unsupported trimesh object")
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

# optional CSV (for labeling)
csv = None
if CSV_PATH.exists():
    csv = pd.read_csv(CSV_PATH)
    if "id" not in csv.columns:
        csv["id"] = csv["file_name"].str.replace("-rgba.png","", regex=False)

rows = []
for ply in sorted(DATA_DIR.glob("*-pointcloud.ply")):
    stem = ply.stem.replace("-pointcloud","")
    print(f"[INFO] {stem}")

    # --- load
    pts = load_pts_ply(ply)
    n_all = len(pts)

    # --- RANSAC plane: z ~ f(x,y)
    X, z = pts[:, :2], pts[:, 2]
    ransac = RANSACRegressor().fit(X, z)
    z_pred = ransac.predict(X)
    deck_mask = np.abs(z - z_pred) < PLANE_THRESH
    deck_z = np.median(z[deck_mask]) if deck_mask.any() else np.median(z)

    # align so deck is z=0
    z_aligned = z - deck_z
    pts_aligned = pts.copy()
    pts_aligned[:,2] = z_aligned

    # --- keep points above deck
    above = pts_aligned[pts_aligned[:,2] > Z_ABOVE]
    n_above = len(above)

    # --- DBSCAN to isolate parcel (largest cluster)
    if n_above > 0:
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit(above[:,:3])
        labels = clustering.labels_
        if np.any(labels >= 0):
            counts = np.bincount(labels[labels>=0])
            main = np.argmax(counts)
            parcel = above[labels==main]
        else:
            parcel = above
    else:
        parcel = above

    n_parcel = len(parcel)

    # --- quick visualization: deck (light gray), above (silver), parcel (blue), parcel AABB (red)
    # downsample for speed
    deck_ds   = pts_aligned[deck_mask]
    if len(deck_ds) > 50000: deck_ds = deck_ds[np.random.choice(len(deck_ds), 50000, replace=False)]
    above_ds  = above if len(above) <= 50000 else above[np.random.choice(len(above), 50000, replace=False)]
    parcel_ds = parcel if len(parcel) <= 50000 else parcel[np.random.choice(len(parcel), 50000, replace=False)]

    # parcel AABB box edges (in aligned cloud units)
    if n_parcel > 0:
        mn = parcel_ds.min(axis=0); mx = parcel_ds.max(axis=0)
        x0,y0,z0 = mn; x1,y1,z1 = mx
        edges = [
            [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)], [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
            [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)], [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
            [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)], [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)],
        ]
    else:
        edges = []

    title_dims = ""
    if csv is not None:
        row = csv[csv["id"]==stem]
        if not row.empty:
            L,B,H = parse_dims(row.iloc[0]["dims"])
            title_dims = f" | dims(csv)={L}x{B}x{H}"

    fig = plt.figure(figsize=(7.5,6.5))
    ax = fig.add_subplot(111, projection="3d")
    if len(deck_ds):   ax.scatter(deck_ds[::20,0], deck_ds[::20,1], deck_ds[::20,2], s=0.2, c="#cfcfcf", label="deck")
    if len(above_ds):  ax.scatter(above_ds[::25,0], above_ds[::25,1], above_ds[::25,2], s=0.2, c="#aaaaaa", label="above deck")
    if len(parcel_ds): ax.scatter(parcel_ds[::10,0], parcel_ds[::10,1], parcel_ds[::10,2], s=0.3, c="#2a6fdb", label="parcel")

    for p,q in edges:
        ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], c="red", linewidth=1.2)

    ax.set_title(f"{stem} | pts={n_all} | deck={deck_mask.sum()} | parcel={n_parcel}{title_dims}")
    ax.set_xlabel("X (cloud units)"); ax.set_ylabel("Y"); ax.set_zlabel("Z (aligned)")
    ax.view_init(elev=18, azim=35); fig.tight_layout()
    out_png = PLOTS_DIR / f"{stem}_parcel.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)

    # simple stats
    parcel_aabb = {"px_min": np.nan, "py_min": np.nan, "pz_min": np.nan,
                   "px_max": np.nan, "py_max": np.nan, "pz_max": np.nan}
    if n_parcel:
        mn = parcel.min(axis=0); mx = parcel.max(axis=0)
        parcel_aabb = {"px_min": mn[0], "py_min": mn[1], "pz_min": mn[2],
                       "px_max": mx[0], "py_max": mx[1], "pz_max": mx[2]}
    rows.append({
        "id": stem, "n_all": n_all, "n_deck": int(deck_mask.sum()),
        "n_above": int(n_above), "n_parcel": int(n_parcel),
        **parcel_aabb, "plot": str(out_png)
    })

# save stats
out_df = pd.DataFrame(rows).sort_values("id")
out_df.to_csv(OUT_CSV, index=False)
print("\nSaved images to:", PLOTS_DIR.resolve())
print("Saved stats CSV:", OUT_CSV)
print(out_df[["id","n_all","n_deck","n_parcel","plot"]].to_string(index=False))
