# Parcel Dimension Measurement from PCD
## Introduction
Measuring a box without actually having a physical contact? Yes!
Handling `.ply` files — 3D **point clouds** , measuring to find their dimensions and volume.

- **.ply**: Polygon File Format, often used for storing 3D point cloud geometry.
- **Point Cloud**: Essentially the simplest form of a 3d model. It is a collection of individual points plotted in a 3d space each point contains several measurements including its coordinate along X, Y, Z directions, RGB (color value) and luminance (brightness).

## Understanding the Raw Data
<p align="center">
  <img width="650" height="650" alt="pcl1" src="https://github.com/user-attachments/assets/1eff8dff-2a87-423d-903b-78d9c6ca88e8" alt="Point Cloud PCS Analysis" />
</p>

So what is the above graphs?
1. **Z distribution** — how points are spread along the scanner's vertical axis.
2. **PCA height plot** — same data but aligned to the table plane using PCA (red line = estimated table height).
3. **Top-down view** — X–Y plane colored by height above the table (yellow = top surfaces, purple = low).

From this, we can **spot the parcel’s footprint** and guess its height range.

## Problem Statemen
Given: The dataset includes representative photos showing:
- Variation in parcel sizes
- Common shapes and surface textures
- Reflective or glossy packaging materials
- Color variations and potential overhangs or deformations
- Real-world placement on AGV top surfaces


## Challenges Faced
1. **Data Scale Issues**  
   - Raw point clouds were in **meters**, but our measurements were in **millimeters**.  
   - **Solution**: Added an automatic scaling check in `load_and_scale_mm()`.

2. **Parcel Lost During Filtering**  
   - Certain `delta` height thresholds returned zero points.  
   - **Solution**: Tuned `delta_low`/`delta_high` window and added a **fallback top-band filter**.

3. **DBSCAN Not Detecting Parcel Cluster**  
   - Parameter sensitivity caused no clusters to be detected.  
   - **Solution**: Adjusted `eps_xy_mm`, `min_pts`, and added **Z-weighting** for better XY clustering.

4. **NumPy 2.0 Compatibility**  
   - `.ptp()` method removal caused errors.  
   - **Solution**: Replaced with `np.ptp()` calls.
  
## Expected Output
For each `.ply` file, we now get:
- **Isolation Info**: Parcel found or fallback method used
- **Dimensions in mm and cm**
- **Bounding Box Volume in mm³ and cm³**
- Debug `.ply` file of the isolated parcel
- JSON row for comparison with ground truth

