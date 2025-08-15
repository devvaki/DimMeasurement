import pathlib
import json
import open3d as o3d
from dimmeasure.preprocess import load_and_scale_mm, isolate_parcel
from dimmeasure.obb import obb_dims_and_volume

PID = "3163484"
PLY = pathlib.Path(f"data/{PID}-pointcloud.ply")

pcd_mm = load_and_scale_mm(PLY)
parcel, info = isolate_parcel(
    pcd_mm,
    delta_low=10.0,     
    delta_high=300.0,   
    voxel_mm=3.0,
    eps_xy_mm=30.0, min_pts=60,
    z_weight=0.15, top_frac=0.20
)

print("isolation:", info)

if parcel.is_empty():
    print("→ tweak: delta_low=6..15, delta_high=200..500, eps_xy=25..40, min_pts=40..80, z_weight=0.1..0.2")
else:
    L, W, H, Vbox = obb_dims_and_volume(parcel)

    # mm → cm for comparison
    Lc, Wc, Hc = L / 10.0, W / 10.0, H / 10.0
    Vbox_cm3 = Vbox / 1000.0

    print(f"{PID}  LxWxH(mm) = {L:.1f} x {W:.1f} x {H:.1f} | Vbox(mm³) = {Vbox:.0f}")
    print(f"{PID}  LxWxH(cm) = {Lc:.1f} x {Wc:.1f} x {Hc:.1f} | Vbox(cm³) = {Vbox_cm3:.3f}")

    row = {
        "File Name": f"{PID}-rgba.png",
        "Actual Box volume (LxBxH)": f"{Lc:.1f}x{Wc:.1f}x{Hc:.1f}",
        "Box Volume (cm³)": round(Vbox_cm3, 3),
    }
    pathlib.Path("results/table").mkdir(parents=True, exist_ok=True)
    with open(f"results/table/{PID}.json", "w") as f:
        json.dump(row, f, indent=2)

    out = pathlib.Path("results/debug")
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out / f"{PID}_parcel_mm.ply"), parcel)
