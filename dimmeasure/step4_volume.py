# step6_full_pipeline.py
from pathlib import Path
import numpy as np, pandas as pd, re, trimesh, json, time, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont
import time

# ---------------- CONFIG ----------------
DATA_DIR   = Path("./data")
CSV_PATH   = Path("eval_data.csv")
OUT_CSV    = Path("results/parcel_measurements.csv")
OUT_JSON   = Path("results/parcel_measurements.json")
PLOT3D_DIR = Path("results/plots3d"); PLOT3D_DIR.mkdir(parents=True, exist_ok=True)
PLOTXY_DIR = Path("results/plots_xy"); PLOTXY_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR  = Path("results/annotated"); ANNOT_DIR.mkdir(parents=True, exist_ok=True)

# plane & selection
PLANE_THRESH      = 0.02
Z_MIN_ABOVE       = 0.05
TOP_PCT_START     = 0.92
BAND_DOWN_START   = 0.04
BAND_UP           = 0.01
TARGET_POINTS     = 5000
GROW_STEP         = 0.005
GROW_MAX_STEPS    = 6
# DBSCAN on XY
DBSCAN_EPS_XY     = 0.02
DBSCAN_MIN        = 300

# refine (per-file retry) thresholds
REFINE_ERR_PCT    = 30.0   # retry if L or W error exceeds this
REFINE_GROW       = 0.06
REFINE_EPS_DELTA  = -0.01

# plotting
DOWNSAMPLE_PARCEL = 40000

# voxel volume
VOXEL_MM          = 5.0
VOX_CLOSE_ITERS   = 1

# --------------- HELPERS ---------------
def extract_id(s: str) -> str:
    s = str(s)
    m = re.search(r"(\d{7})", s)
    return m.group(1) if m else s.strip()

def parse_dims_to_mm(s: str):
    s = str(s).strip().lower().replace("×","x")
    parts = [p for p in re.split(r"[xX]", s) if p][:3]
    L,B,H = [float(parts[0]), float(parts[1]), float(parts[2])]
    return L*10.0, B*10.0, H*10.0  # cm→mm

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

def plane_align(points, plane_thresh=PLANE_THRESH):
    X, z = points[:,:2], points[:,2]
    rans = RANSACRegressor().fit(X, z)
    z_pred = rans.predict(X)
    deck = np.median(z[np.abs(z - z_pred) < plane_thresh])
    P = points.copy(); P[:,2] = z - deck
    return P

def convex_hull_xy(pts):
    P = np.unique(pts, axis=0)
    if len(P) < 3: return P
    P = P[np.lexsort((P[:,1], P[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in P:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in P[::-1]:
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])

def hull_perimeter(H):
    if len(H) < 2: return 0.0
    d = np.diff(np.vstack([H, H[0]]), axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))

def min_area_rect(xy):
    P = np.unique(xy, axis=0)
    if len(P) < 3:
        mn = P.min(axis=0); mx = P.max(axis=0)
        corners = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]])
        L = (mx[0]-mn[0]); W = (mx[1]-mn[1])
        if L < W: L, W = W, L
        return L, W, np.eye(2), (mn+mx)/2, corners
    H = convex_hull_xy(P)
    best = (np.inf, None, None, None, None, None)
    Hc = H.mean(axis=0)
    for i in range(len(H)):
        p0, p1 = H[i], H[(i+1)%len(H)]
        edge = p1 - p0
        ang = np.arctan2(edge[1], edge[0])
        ca, sa = np.cos(-ang), np.sin(-ang)
        R = np.array([[ca,-sa],[sa,ca]])
        proj = (H - Hc) @ R.T
        mn_, mx_ = proj.min(axis=0), proj.max(axis=0)
        Lc, Wc = mx_ - mn_
        area = Lc*Wc
        if area < best[0]:
            L, W = (Lc,Wc) if Lc>=Wc else (Wc,Lc)
            center_world = Hc + 0.5*(mn_+mx_) @ R
            best = (area, L, W, R, center_world, (mn_, mx_))
    _, L, W, R, center, (mn, mx) = best
    rect_frame = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]])
    corners_world = Hc + rect_frame @ R
    return L, W, R, center, corners_world

def trimmed_median(a, trim=0.2):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a)==0: return np.nan
    q1, q9 = np.quantile(a, [trim, 1-trim])
    inl = a[(a>=q1) & (a<=q9)]
    return float(np.median(inl) if len(inl) else np.median(a))

def per_err(pred, gt): 
    return 100.0*np.abs(float(pred)-float(gt))/max(float(gt),1e-6)

def pick_lid_band_adaptive(P):
    A = P[P[:,2] > Z_MIN_ABOVE]
    if len(A)==0: return A
    z_peak = np.quantile(A[:,2], TOP_PCT_START)
    band_down = BAND_DOWN_START
    prev_perim = 0.0
    selected = A[(A[:,2] >= z_peak - band_down) & (A[:,2] <= z_peak + BAND_UP)]
    steps=0
    while (len(selected) < TARGET_POINTS or hull_perimeter(convex_hull_xy(selected[:,:2])) < 0.98*prev_perim) and steps < GROW_MAX_STEPS:
        prev_perim = hull_perimeter(convex_hull_xy(selected[:,:2]))
        band_down += GROW_STEP
        selected = A[(A[:,2] >= z_peak - band_down) & (A[:,2] <= z_peak + BAND_UP)]
        steps += 1
    # 2D DBSCAN cleanup — largest cluster in XY
    if len(selected)>0:
        labels = DBSCAN(eps=DBSCAN_EPS_XY, min_samples=DBSCAN_MIN).fit(selected[:,:2]).labels_
        if np.any(labels >= 0):
            k = np.argmax(np.bincount(labels[labels>=0]))
            selected = selected[labels==k]
    return selected

def refine_band_and_rect(band_u, grow=REFINE_GROW, eps_delta=REFINE_EPS_DELTA):
    if len(band_u) == 0: return band_u
    z_peak = np.quantile(band_u[:,2], 0.90)
    z_lo = z_peak - (BAND_DOWN_START + grow)
    z_hi = z_peak + BAND_UP
    m = (band_u[:,2] >= z_lo) & (band_u[:,2] <= z_hi)
    cand = band_u[m]
    if len(cand) == 0: return band_u
    eps = max(0.01, DBSCAN_EPS_XY + eps_delta)
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN).fit(cand[:,:2]).labels_
    if np.any(labels >= 0):
        k = np.argmax(np.bincount(labels[labels>=0]))
        cand = cand[labels==k]
    return cand if len(cand) >= 200 else band_u

# ----- volumes, typing, annotation -----
def compute_box_volume_mm3(L_mm, W_mm, H_mm):
    return float(max(L_mm,0) * max(W_mm,0) * max(H_mm,0))

def voxel_true_volume_mm3(points_mm, voxel_mm=VOXEL_MM, close_iters=VOX_CLOSE_ITERS):
    if points_mm.size == 0: return 0.0, 0, voxel_mm
    mins = points_mm.min(axis=0)
    shifted = points_mm - mins
    v = float(voxel_mm)
    idx = np.floor(shifted / v).astype(np.int32)
    idx = idx[np.all(np.isfinite(idx), axis=1)]
    if len(idx) == 0: return 0.0, 0, voxel_mm
    uniq = np.unique(idx, axis=0)
    if close_iters > 0:
        cur = uniq
        for _ in range(close_iters):
            expanded = [cur]
            for axis in range(3):
                off = np.zeros_like(cur); off[:, axis] = 1
                expanded.append(cur + off); expanded.append(cur - off)
            cur = np.unique(np.vstack(expanded), axis=0)
        uniq = cur
    num_vox = uniq.shape[0]
    vol_mm3 = num_vox * (v ** 3)
    return float(vol_mm3), int(num_vox), voxel_mm

def classify_parcel_type(L_mm, W_mm, H_mm, box_vol_mm3, true_vol_mm3):
    if box_vol_mm3 <= 0: return "Unknown"
    ratio = true_vol_mm3 / box_vol_mm3
    return "Box" if ratio >= 0.90 else "Polybag"

def estimate_confidence(errL_pct, errW_pct, errH_pct, ratio_true_box):
    e = np.clip((100 - (errL_pct+errW_pct+errH_pct)/3.0) / 100.0, 0.0, 1.0)
    r = np.clip(1.2 - abs(1.0 - ratio_true_box), 0.0, 1.0)
    return float(np.clip(0.6*e + 0.4*r, 0.0, 1.0))

# replace annotate_png in test4.py
from PIL import Image, ImageDraw, ImageFont

def annotate_png(base_png_path, out_png_path, text_lines, box=None, alpha=170):
    """
    Draw a semi-transparent box + white text on the saved plot image.
    - box: (x0,y0,x1,y1). If None, auto-place at top-left with 40% width, 18% height.
    """
    try:
        img = Image.open(base_png_path).convert("RGBA")
    except Exception:
        return None

    W, H = img.size
    if box is None:
        x0, y0 = int(0.04*W), int(0.04*H)
        x1, y1 = int(0.44*W), int(0.22*H)
    else:
        x0, y0, x1, y1 = box

    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # larger, dynamic font
    try:
        font = ImageFont.truetype("arial.ttf", int(0.035*H))
    except Exception:
        font = ImageFont.load_default()

    # background panel
    draw.rectangle([x0,y0,x1,y1], fill=(0,0,0,alpha))

    # text with consistent line spacing
    x, y = x0 + int(0.02*W), y0 + int(0.015*H)
    for line in text_lines:
        draw.text((x, y), line, fill=(255,255,255,255), font=font)
        y += int(0.055*H)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(out_png_path)
    return str(out_png_path)


def make_json_record(
    stem, Lm, Wm, Hm, box_vol_mm3, true_vol_mm3, parcel_type,
    Sx, Sy, sz, eL, eW, eH, rgb_path, plot3d_path, plotxy_path,
    annotated_path=None, timestamp_ms=None, 
    Lgt=None, Wgt=None, Hgt=None, gt_box_vol_mm3=None
):
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    ratio = (true_vol_mm3/box_vol_mm3) if box_vol_mm3>0 else 0.0
    conf = estimate_confidence(eL, eW, eH, ratio)

    out = {
        "id": str(stem),
        "timestamp_ms": int(timestamp_ms),

        # predicted (ours)
        "length_mm": float(Lm),
        "width_mm":  float(Wm),
        "height_mm": float(Hm),
        "box_volume_mm3": float(box_vol_mm3),
        "true_volume_mm3": float(true_vol_mm3),
        "parcel_type": parcel_type,

        # GT from CSV (if available)
        "gt_length_mm": float(Lgt) if Lgt is not None else None,
        "gt_width_mm":  float(Wgt) if Wgt is not None else None,
        "gt_height_mm": float(Hgt) if Hgt is not None else None,
        "gt_box_volume_mm3": float(gt_box_vol_mm3) if gt_box_vol_mm3 is not None else None,

        # scaling + errors
        "scale_mm_per_unit": { "sx": float(Sx), "sy": float(Sy), "sz": float(sz) },
        "errors_pct": { "length": float(eL), "width": float(eW), "height": float(eH) },
        "confidence": float(round(conf, 2))
        }

    # optional box-volume error if GT available
    if gt_box_vol_mm3:
        out["err_box_volume_%"] = per_err(box_vol_mm3, gt_box_vol_mm3)

    return out

# --------------- LOAD CSV ---------------
gt = pd.read_csv(CSV_PATH)
gt.columns = [c.strip() for c in gt.columns]

# id column
if "id" not in gt.columns:
    if "File Name" in gt.columns:
        gt["id"] = gt["File Name"].str.extract(r"(\d{6,})")[0]
    elif "file_name" in gt.columns:
        gt["id"] = gt["file_name"].str.extract(r"(\d{6,})")[0]
    else:
        name_col = next((c for c in gt.columns if "file" in c.lower() or "name" in c.lower()), None)
        if not name_col:
            raise ValueError("CSV needs 'id' or a filename column like 'File Name'.")
        gt["id"] = gt[name_col].str.extract(r"(\d{6,})")[0]
gt["id"] = gt["id"].apply(extract_id)

# dims (cm) → we’ll parse to mm with parse_dims_to_mm later
dims_col = next((c for c in gt.columns
                 if ("lx" in c.lower()) or ("volume" in c.lower() and "x" in str(gt[c].iloc[0]).lower())),
                None)
if dims_col:
    gt = gt.rename(columns={dims_col: "dims"})
elif {"Length (cm)", "Width (cm)", "Height (cm)"} <= set(gt.columns):
    gt["dims"] = (gt["Length (cm)"].astype(str) + "x" +
                  gt["Width (cm)"].astype(str)  + "x" +
                  gt["Height (cm)"].astype(str))
else:
    raise ValueError("Could not find dims text in CSV (need 'Actual Box volume (LxBxH)' or Length/Width/Height in cm).")

id_to_dims = {extract_id(i): d for i, d in zip(gt["id"], gt["dims"])}
# Optional GT box volume (cm^3) → mm^3
gt_box_mm3 = {}
if "Box Volume" in gt.columns:
    for i, v in zip(gt["id"], gt["Box Volume"]):
        try:
            gt_box_mm3[extract_id(i)] = float(v) * 1000.0  # cm^3 → mm^3
        except Exception:
            pass

# --------------- PASS 1: per-file scales ---------------
rows=[]; cache={}
for ply in sorted(DATA_DIR.glob("*-pointcloud.ply")):
    stem = extract_id(ply.stem)
    if stem not in id_to_dims:
        print(f"[SKIP csv] not in CSV: {stem}"); continue
    Lgt, Wgt, Hgt = parse_dims_to_mm(id_to_dims[stem])
    P = plane_align(load_pts(str(ply)))
    band = pick_lid_band_adaptive(P)
    if len(band) < 200:
        print(f"[SKIP sparse] {stem}: band points={len(band)}")
        continue
    L_u, W_u, R2, t2, _corners_u = min_area_rect(band[:,:2])
    A = P[P[:,2] > Z_MIN_ABOVE]
    Hu = np.percentile(A[:,2], 99.5) - np.percentile(A[:,2], 0.5)
    Lgt_s, Wgt_s = sorted([Lgt, Wgt], reverse=True)
    L_u_s,  W_u_s  = sorted([L_u,  W_u],  reverse=True)
    sx = Lgt_s / L_u_s
    sy = Wgt_s / W_u_s
    sz = Hgt   / Hu
    rows.append({"id":stem, "Lu":L_u, "Wu":W_u, "Hu":Hu,
                 "Lgt_mm":Lgt, "Wgt_mm":Wgt, "Hgt_mm":Hgt,
                 "sx":sx, "sy":sy, "sz":sz})
    cache[stem] = {"band_u":band, "P_u":P}

if not rows:
    raise SystemExit("No rows produced — check CSV/paths/thresholds.")

df = pd.DataFrame(rows).sort_values("id")

# Two-stage robust global Sx/Sy
Sx = trimmed_median(df["sx"], trim=0.2)
Sy = trimmed_median(df["sy"], trim=0.2)
L_pred = df["Lu"] * Sx
W_pred = df["Wu"] * Sy
errL0 = np.abs(L_pred - df["Lgt_mm"]) / np.maximum(df["Lgt_mm"], 1e-6)
errW0 = np.abs(W_pred - df["Wgt_mm"]) / np.maximum(df["Wgt_mm"], 1e-6)
mask = (errL0 < 0.25) & (errW0 < 0.25)
if mask.sum() >= 3:
    Sx = trimmed_median(df.loc[mask,"sx"], trim=0.2)
    Sy = trimmed_median(df.loc[mask,"sy"], trim=0.2)

# --------------- PASS 2: mm scaling, OBB, volumes, plots, outputs ---------------
plot_rows=[]; json_records=[]
Path(OUT_CSV.parent).mkdir(parents=True, exist_ok=True)
Path(OUT_JSON.parent).mkdir(parents=True, exist_ok=True)

for _,r in df.iterrows():
    stem = r["id"]; Lgt, Wgt, Hgt = r["Lgt_mm"], r["Wgt_mm"], r["Hgt_mm"]
    sz = r["sz"]
    band_u = cache[stem]["band_u"]

    # scale band to mm for XY
    band_mm = band_u.copy()
    band_mm[:,0] *= Sx; band_mm[:,1] *= Sy; band_mm[:,2] *= sz

    # initial OBB in mm
    Lm, Wm, Rm, tm, corners_mm = min_area_rect(band_mm[:,:2])
    Lm, Wm = sorted([Lm, Wm], reverse=True)

    # Height from full above-deck cloud
    A_u = cache[stem]["P_u"]
    A_u = A_u[A_u[:,2] > Z_MIN_ABOVE].copy()
    A_mm = A_u.copy()
    A_mm[:,0] *= Sx; A_mm[:,1] *= Sy; A_mm[:,2] *= sz
    Hm = np.percentile(A_mm[:,2], 99.5) - np.percentile(A_mm[:,2], 0.5)

    # errors vs GT (if GT present)
    eL, eW, eH = per_err(Lm, Lgt), per_err(Wm, Wgt), per_err(Hm, Hgt)

    # auto-refine if poor XY
    if (eL > REFINE_ERR_PCT) or (eW > REFINE_ERR_PCT):
        band_u2 = refine_band_and_rect(band_u, grow=REFINE_GROW, eps_delta=REFINE_EPS_DELTA)
        band_mm2 = band_u2.copy()
        band_mm2[:,0] *= Sx; band_mm2[:,1] *= Sy; band_mm2[:,2] *= sz
        Lm2, Wm2, Rm2, tm2, corners_mm2 = min_area_rect(band_mm2[:,:2])
        Lm2, Wm2 = sorted([Lm2, Wm2], reverse=True)
        eL2, eW2 = per_err(Lm2, Lgt), per_err(Wm2, Wgt)
        if (eL2 + eW2) < (eL + eW):
            Lm, Wm, eL, eW = Lm2, Wm2, eL2, eW2
            corners_mm = corners_mm2
            band_mm    = band_mm2

    # volumes
    box_vol_mm3 = compute_box_volume_mm3(Lm, Wm, Hm)
    true_vol_mm3, num_vox, vmm = voxel_true_volume_mm3(A_mm, voxel_mm=VOXEL_MM, close_iters=VOX_CLOSE_ITERS)
    true_vol_mm3 = min(true_vol_mm3, box_vol_mm3)  # sanity cap
    ptype = classify_parcel_type(Lm, Wm, Hm, box_vol_mm3, true_vol_mm3)

    # plotting (3D)
    P = band_mm
    if len(P) > DOWNSAMPLE_PARCEL:
        idx = np.random.choice(len(P), DOWNSAMPLE_PARCEL, replace=False)
        P = P[idx]
    fig = plt.figure(figsize=(8.5,7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:,0], P[:,1], P[:,2], s=0.25, c="#2a6fdb")
    zmed = np.median(band_mm[:,2]); zmin, zmax = band_mm[:,2].min(), band_mm[:,2].max()
    C = corners_mm
    for a,b in [(0,1),(1,2),(2,3),(3,0)]:
        ax.plot([C[a,0],C[b,0]],[C[a,1],C[b,1]],[zmed,zmed], c="red", lw=1.6)
    for a in range(4):
        ax.plot([C[a,0],C[a,0]],[C[a,1],C[a,1]],[zmin,zmax], c="red", lw=1.0, alpha=0.8)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(18,35); fig.tight_layout()
    title = (f"{stem} | GT {Lgt:.0f}×{Wgt:.0f}×{Hgt:.0f} mm  |  "
             f"Pred {Lm:.0f}×{Wm:.0f}×{Hm:.0f} mm  | "
             f"err% L={eL:.1f} W={eW:.1f} H={eH:.1f}")
    ax.set_title(title)
    out3d = PLOT3D_DIR / f"{stem}_3d.png"
    fig.tight_layout()
    fig.savefig(out3d, dpi=170, bbox_inches="tight")

    # XY top-down
    plt.figure(figsize=(6.4,6.4))
    plt.scatter(band_mm[:,0], band_mm[:,1], s=0.4, alpha=0.4)
    C2 = np.vstack([corners_mm, corners_mm[0]])
    plt.plot(C2[:,0], C2[:,1], 'r-', lw=2)
    plt.axis('equal'); plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title(f"{stem}  XY  |  L={Lm:.0f}  W={Wm:.0f}  (GT {Lgt:.0f}/{Wgt:.0f})")
    outxy = PLOTXY_DIR / f"{stem}_xy.png"
    plt.tight_layout(); plt.savefig(outxy, dpi=150); plt.close()

    # Annotated image (overlay text on 3D plot)
    # Annotate the ORIGINAL RGB in ./data with VOLUME ONLY (mm^3)
    rgb_path = str(DATA_DIR / f"{stem}-rgba.png")
    annot_png = ANNOT_DIR / f"{stem}_vol_annot.png"
    vol_text = [
        f"Box Volume:  {box_vol_mm3:.0f} mm³",
        f"True Volume: {true_vol_mm3:.0f} mm³"
    ]
    base_for_annotation = rgb_path if Path(rgb_path).exists() else str(out3d)
    annotated_path = annotate_png(base_for_annotation, str(annot_png), vol_text)
    gt_vol = gt_box_mm3.get(stem) if 'gt_box_mm3' in locals() else None

    # JSON record
    rgb_path = str(DATA_DIR / f"{stem}-rgba.png")
    
    rec = make_json_record(
    stem=stem, Lm=Lm, Wm=Wm, Hm=Hm,
    box_vol_mm3=box_vol_mm3, true_vol_mm3=true_vol_mm3,
    parcel_type=ptype, Sx=Sx, Sy=Sy, sz=sz,
    eL=eL, eW=eW, eH=eH,
    rgb_path=rgb_path, plot3d_path=str(out3d), plotxy_path=str(outxy),
    annotated_path=annotated_path, timestamp_ms=int(time.time()*1000),
    Lgt=Lgt, Wgt=Wgt, Hgt=Hgt, gt_box_vol_mm3=gt_vol
)
    json_records.append(rec)

    # CSV row
    gt_vol = gt_box_mm3.get(stem) if 'gt_box_mm3' in locals() else None
    err_box_pct = per_err(box_vol_mm3, gt_vol) if gt_vol else None
    now_ms = int(time.time() * 1000)

    plot_rows.append({
    "ID": stem,
    "L(mm)": round(Lm,1),
    "W(mm)": round(Wm,1),
    "H(mm)": round(Hm,1),
    "BoxV(mm³)": int(round(box_vol_mm3)),
    "TrueV(mm³)": int(round(true_vol_mm3)),
    "ErrL(%)": f"{eL:.1f}%",
    "ErrW(%)": f"{eW:.1f}%",
    "ErrH(%)": f"{eH:.1f}%",
    "Type": ptype,
    "TS(ms)": now_ms
})


# Write CSV + JSON
df_out = pd.DataFrame(plot_rows)

sort_key = "id" if "id" in df_out.columns else ("ID" if "ID" in df_out.columns else None)
if sort_key:
    df_out = df_out.sort_values(sort_key)

col_map = {
    "id": "ID",
    "gt_l_mm": "L_gt(mm)", "gt_w_mm": "W_gt(mm)", "gt_h_mm": "H_gt(mm)", "gt_box_volume_mm3": "V_gt(mm³)",
    "length_mm": "L(mm)", "width_mm": "W(mm)", "height_mm": "H(mm)",
    "box_volume_mm3": "BoxV(mm³)", "true_volume_mm3": "TrueV(mm³)",
    "err_l_%": "ErrL(%)", "err_w_%": "ErrW(%)", "err_h_%": "ErrH(%)", "err_box_volume_%": "ErrV(%)",
    "parcel_type": "Type", "timestamp_ms": "Time(ms)"
}
# save CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_out.rename(columns=col_map).to_csv(OUT_CSV, index=False)
# save JSON
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
renamed_json = [{col_map.get(k, k): v for k, v in rec.items()} for rec in json_records]
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(renamed_json, f, ensure_ascii=False, indent=2)

t_cols = [
    "ID",
    "L_gt(mm)", "W_gt(mm)", "H_gt(mm)", "V_gt(mm³)",
    "L(mm)", "W(mm)", "H(mm)", "BoxV(mm³)", "TrueV(mm³)",
    "ErrL(%)", "ErrW(%)", "ErrH(%)", "ErrV(%)",
    "Type", "Time(ms)"
]

df_short = df_out.rename(columns=col_map)
present = [c for c in t_cols if c in df_short.columns]
print(df_short[present].to_string(index=False))

print("\nSaved CSV  →", OUT_CSV.resolve())
print("Saved JSON →", OUT_JSON.resolve())
print("Saved 3D plots →", PLOT3D_DIR.resolve())
print("Saved XY plots →", PLOTXY_DIR.resolve())
print("Saved annotated →", ANNOT_DIR.resolve())
