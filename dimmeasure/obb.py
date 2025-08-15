import open3d as o3d

def obb_dims_and_volume(pcd_mm):
    obb = pcd_mm.get_oriented_bounding_box()
    L, W, H = sorted(obb.extent, reverse=True)
    return float(L), float(W), float(H), float(L * W * H)
