import os
import time
import cv2
import open3d as o3d

def load_pointcloud_and_image(ply_path: str, img_path: str):
    """
    Load the point cloud (.ply) and matching image (.png).
    Returns: (pcd, img_rgb, parcel_id:str, timestamp_ms:int)
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    parcel_id = os.path.basename(ply_path).split("-")[0]
    timestamp_ms = int(time.time() * 1000)

    return pcd, img_rgb, parcel_id, timestamp_ms
