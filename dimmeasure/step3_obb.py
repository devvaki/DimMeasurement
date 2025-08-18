# Step 3 — Scale to mm + per-parcel 3D plots (deck->removed, parcel+OBB in mm)
from pathlib import Path
import numpy as np, pandas as pd, re, os, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN

DATA_DIR  = Path("./data")
CSV_PATH  = Path("eval_data.csv")
OUT_CSV   = Path("step3_scaled_dims.csv")
PLOT_DIR  = Path("step3_plots"); PLOT_DIR.mkdir(exist_ok=True)

# ---- tunables (same spirit as step2) ----
PLANE_THRESH = 0.02
Z_ABOVE      = 0.05
DBSCAN_EPS   = 0.05
DBSCAN_MIN   = 60
DOWNSAMPLE_PARCEL = 40000  # for plotting speed

def extract_id(s: str) -> str:
    s = str(s)
    m = re.search(r"(\d{7})", s)
    return m.group(1) if m else s.strip()

def parse_dims_to_mm(s: str):
    s = str(s).strip().lower().replace("×","x")
    parts = [p for p in re.split(r"[xX]", s) if p][:3]
    L,B,H = [float(parts[0]), float(parts[1]), float(parts[2])]
    return L*10.0, B*10.0, H*10.0  # cm->mm

def load_pts(p):
    obj = trimesh.load(p)
    if hasattr(obj, "vertices"):
        pts = np.asarray(obj.vertices)
    else:
        pts = None
        for g in obj.geometry.values():
            if hasattr(g, "vertices"):
                pts = np.asarray(g.vertices); break
        if pts is None: raise ValueError("no vertices")
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

def deck_align_and_parcel(points, plane_thresh=PLANE_THRESH, z_above=Z_ABOVE,
                          eps=DBSCAN_EPS, min_samples=DBSCAN_MIN):
    X, z = points[:,:2], points[:,2]
    rans = RANSACRegressor().fit(X, z)
    z_pred = rans.predict(X)
    deck_z = np.median(z[np.abs(z - z_pred) < plane_thresh])
    P = points.copy(); P[:,2] = z - deck_z
    A = P[P[:,2] > z_above]
    if len(A)==0: return P, A
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(A[:,:3]).labels_
    if np.any(lab >= 0):
        largest = np.argmax(np.bincount(lab[lab>=0]))
        A = A[lab==largest]
    return P, A

def obb_xy_pca(points2d):
    xy = points2d - points2d.mean(axis=0, keepdims=True)
    cov = np.cov(xy, rowvar=False)
    w, V = np.linalg.eigh(cov)
    V = V[:, np.argsort(w)[::-1]]  # major axis first
    proj = xy @ V
    L = proj[:,0].max() - proj[:,0].min()
    W = proj[:,1].max() - proj[:,1].min()
    return L, W, V

# ---------- load CSV ----------
gt = pd.read_csv(CSV_PATH)
gt.columns = [c.strip() for c in gt.columns]  # trim header whitespace

# Ensure an 'id' column by extracting digits from a filename-like column
if "id" not in gt.columns:
    if "File Name" in gt.columns:
        gt["id"] = gt["File Name"].str.extract(r"(\d{6,})")[0]
    elif "file_name" in gt.columns:
        gt["id"] = gt["file_name"].str.extract(r"(\d{6,})")[0]
    else:
        # fallback: first column that looks like a name/file
        name_col = next((c for c in gt.columns if "file" in c.lower() or "name" in c.lower()), None)
        if not name_col:
            raise ValueError("CSV must have 'id' or a filename column like 'File Name'.")
        gt["id"] = gt[name_col].str.extract(r"(\d{6,})")[0]

gt["id"] = gt["id"].apply(extract_id)

# Ensure a dims-like column exists (text like "50x33x14.4" in **cm**)
if "dims" not in gt.columns:
    # try common names from your sheet (e.g., "Actual Box volume (LxBxH)")
    dims_col = next((c for c in gt.columns if ("lx" in c.lower()) or ("volume" in c.lower() and "x" in str(gt[c].iloc[0]).lower())), None)
    if dims_col:
        gt = gt.rename(columns={dims_col: "dims"})
    elif {"Length (cm)", "Width (cm)", "Height (cm)"} <= set(gt.columns):
        gt["dims"] = (
            gt["Length (cm)"].astype(str) + "x" +
            gt["Width (cm)"].astype(str)  + "x" +
            gt["Height (cm)"].astype(str)
        )
    else:
        raise ValueError("Could not find dims text in CSV (need e.g. 'Actual Box volume (LxBxH)' or Length/Width/Height in cm).")

id_to_dims = {extract_id(i): d for i, d in zip(gt["id"], gt["dims"])}


# ---------- PASS 1: compute per-file XY/Z scales from GT ----------
rows=[]
parcel_cache = {}  # cache parcel points & OBB in units for PASS 2 plotting
for ply in sorted(DATA_DIR.glob("*-pointcloud.ply")):
    stem = extract_id(ply.stem)
    if stem not in id_to_dims:
        print(f"[SKIP csv] not in CSV: {stem}")
        continue
    Lgt, Wgt, Hgt = parse_dims_to_mm(id_to_dims[stem])  # mm

    try:
        pts = load_pts(str(ply))
        _, parcel = deck_align_and_parcel(pts)
        if len(parcel) < 50:
            print(f"[SKIP small] {stem}: parcel points={len(parcel)}")
            continue

        Lu, Wu, Vxy = obb_xy_pca(parcel[:,:2])
        Hu = parcel[:,2].max() - parcel[:,2].min()
        # sort to match L>=W
        Lgt_s, Wgt_s = sorted([Lgt, Wgt], reverse=True)
        Lu_s,  Wu_s  = sorted([Lu,  Wu],  reverse=True)

        sx = Lgt_s / Lu_s
        sy = Wgt_s / Wu_s
        sz = Hgt   / Hu

        rows.append({"id": stem, "Lu":Lu, "Wu":Wu, "Hu":Hu,
                     "Lgt_mm":Lgt, "Wgt_mm":Wgt, "Hgt_mm":Hgt,
                     "sx":sx, "sy":sy, "sz":sz})
        parcel_cache[stem] = {"parcel_units": parcel, "Vxy": Vxy}
    except Exception as e:
        print(f"[FAIL] {stem}: {e}")

if not rows:
    raise SystemExit("No rows produced — check CSV/paths.")

df = pd.DataFrame(rows).sort_values("id")
Sx, Sy = df["sx"].mean(), df["sy"].mean()   # global XY scales

# ---------- PASS 2: scale to mm, recompute OBB in mm, save plots ----------
def per_err(pred, gt): return 100.0*np.abs(pred-gt)/max(gt,1e-6)

plot_rows=[]
for _,r in df.iterrows():
    stem = r["id"]
    Lgt, Wgt, Hgt = r["Lgt_mm"], r["Wgt_mm"], r["Hgt_mm"]
    sz = r["sz"]  # per-file Z scale

    cache = parcel_cache[stem]
    parcel_u = cache["parcel_units"]
    # scale to mm
    parcel_mm = parcel_u.copy()
    parcel_mm[:,0] *= Sx
    parcel_mm[:,1] *= Sy
    parcel_mm[:,2] *= sz

    # recompute OBB in mm for reporting/plot
    Lm, Wm, Vxy_mm = obb_xy_pca(parcel_mm[:,:2])
    Hm = parcel_mm[:,2].max() - parcel_mm[:,2].min()
    # ensure L>=W
    Lm, Wm = sorted([Lm, Wm], reverse=True)

    eL, eW, eH = per_err(Lm, Lgt), per_err(Wm, Wgt), per_err(Hm, Hgt)

    # downsample for plotting
    P = parcel_mm
    if len(P) > DOWNSAMPLE_PARCEL:
        idx = np.random.choice(len(P), DOWNSAMPLE_PARCEL, replace=False)
        P = P[idx]

    # OBB corners in mm for box overlay (from proj mins/max)
    xy_c = parcel_mm[:,:2] - parcel_mm[:,:2].mean(axis=0, keepdims=True)
    proj = xy_c @ Vxy_mm
    min0, max0 = proj[:,0].min(), proj[:,0].max()
    min1, max1 = proj[:,1].min(), proj[:,1].max()
    # build corners in PCA frame, then back to XY
    corners_xy = np.array([
        [min0, min1], [max0, min1], [max0, max1], [min0, max1]
    ])
    corners_xy_world = corners_xy @ Vxy_mm.T + parcel_mm[:,:2].mean(axis=0, keepdims=True)
    z0 = parcel_mm[:,2].min()
    z1 = parcel_mm[:,2].max()
    # 8 corners
    c = np.zeros((8,3))
    c[0,:2]=corners_xy_world[0]; c[0,2]=z0
    c[1,:2]=corners_xy_world[1]; c[1,2]=z0
    c[2,:2]=corners_xy_world[2]; c[2,2]=z0
    c[3,:2]=corners_xy_world[3]; c[3,2]=z0
    c[4,:2]=corners_xy_world[0]; c[4,2]=z1
    c[5,:2]=corners_xy_world[1]; c[5,2]=z1
    c[6,:2]=corners_xy_world[2]; c[6,2]=z1
    c[7,:2]=corners_xy_world[3]; c[7,2]=z1
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    # plot (parcel in mm + OBB in red)
    fig = plt.figure(figsize=(7.5,6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:,0], P[:,1], P[:,2], s=0.2, c="#2a6fdb", label="parcel (mm)")
    for e in edges:
        ax.plot([c[e[0],0], c[e[1],0]],
                [c[e[0],1], c[e[1],1]],
                [c[e[0],2], c[e[1],2]],
                c="red", linewidth=1.3)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(elev=18, azim=35); fig.tight_layout()
    title = (f"{stem} | GT L×W×H={Lgt:.0f}×{Wgt:.0f}×{Hgt:.0f} mm  |  "
             f"Pred L×W×H={Lm:.0f}×{Wm:.0f}×{Hm:.0f} mm  |  "
             f"err% L={eL:.1f} W={eW:.1f} H={eH:.1f}")
    ax.set_title(title)
    out_png = PLOT_DIR / f"{stem}_mm.png"
    fig.savefig(out_png, dpi=150); plt.close(fig)

    plot_rows.append({
        "id": stem, "Lgt_mm":Lgt, "Wgt_mm":Wgt, "Hgt_mm":Hgt,
        "L_mm":Lm, "W_mm":Wm, "H_mm":Hm,
        "err_L_%":eL, "err_W_%":eW, "err_H_%":eH,
        "plot": str(out_png)
    })

# summary CSV (scaled dims + errors)
df2 = pd.DataFrame(plot_rows).sort_values("id")
df2.to_csv(OUT_CSV, index=False)

print(f"\nGlobal XY scales (mm/unit): Sx={Sx:.2f}, Sy={Sy:.2f}")
print(df2[["id","Lgt_mm","Wgt_mm","Hgt_mm","L_mm","W_mm","H_mm","err_L_%","err_W_%","err_H_%","plot"]].to_string(index=False))
print("\nSaved CSV:", OUT_CSV)
print("Saved plots →", PLOT_DIR.resolve())
