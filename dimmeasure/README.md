<<<<<<< HEAD
# Parcel Dimension Measurement from PCD
=======
# Parcel Dimension Measurement – Status Update

>>>>>>> 629617d (Initial parcel measurement pipeline with OBB, refined box, and true volume calculation. Added README with current status, known issues, and next steps.)
## Overview
Measuring a box without actually having a physical contact? Yes!
Handling `.ply` files — 3D **point clouds** , measuring to find their dimensions and volume.

- **.ply**: Polygon File Format, often used for storing 3D point cloud geometry.
- **Point Cloud**: Essentially the simplest form of a 3d model. It is a collection of individual points plotted in a 3d space each point contains several measurements including its coordinate along X, Y, Z directions, RGB (color value) and luminance (brightness).

<<<<<<< HEAD
## Understanding the Raw Data

So what is the above graph say? From this, we can **spot the parcel’s footprint** and guess its height range.
1. **Z distribution** — how points are spread along the scanner's vertical axis.
2. **PCA height plot** — same data but aligned to the table plane using PCA (red line = estimated table height).
3. **Top-down view** — X–Y plane colored by height above the table (yellow = top surfaces, purple = low).

=======
>>>>>>> 629617d (Initial parcel measurement pipeline with OBB, refined box, and true volume calculation. Added README with current status, known issues, and next steps.)
## Problem Statement
Given: The dataset includes representative photos showing:
- Variation in parcel sizes
- Common shapes and surface textures
- Reflective or glossy packaging materials
- Color variations and potential overhangs or deformations
- Real-world placement on AGV top surfaces

## Data to be delivered per parcel:
- Length(mm)
- Width (mm)
- Height (mm)
- Timestamp in millisecond
- Parcel type (Box, polybag, etc.)
- Image with embossed measurements
- Box Volume and True Volume.

<<<<<<< HEAD
## Challenges Faced
1. **Data Scale Issues**  
   - Raw point clouds were in **meters**, but our measurements were in **millimeters**.  
   - **Solution**: Added an automatic scaling check in `load_and_scale_mm()`.
=======
## Understanding the Raw Data
<p align="center">
    <img src="/home/dev/Pictures/Screenshots/plc1.png" alt="Point cloud PCS Analysis" width="500" height="500">
</p>
So what is the above graph say? From this, we can **spot the parcel’s footprint** and guess its height range.
1. **Z distribution** — how points are spread along the scanner's vertical axis.
2. **PCA height plot** — same data but aligned to the table plane using PCA (red line = estimated table height).
3. **Top-down view** — X–Y plane colored by height above the table (yellow = top surfaces, purple = low).

## Challenges Faced
1. **Data Scale Issues**  
   - Raw point clouds were in **meters**, but my measurements were in **millimeters**.  
   - **Solution**: Added an automatic scaling check
>>>>>>> 629617d (Initial parcel measurement pipeline with OBB, refined box, and true volume calculation. Added README with current status, known issues, and next steps.)

2. **Parcel Lost During Filtering**  
   - Certain `delta` height thresholds returned zero points.  
   - **Solution**: Tuned `delta_low`/`delta_high` window and added a **fallback top-band filter**.

3. **DBSCAN Not Detecting Parcel Cluster**  
   - Parameter sensitivity caused no clusters to be detected.  
   - **Solution**: Adjusted `eps_xy_mm`, `min_pts`, and added **Z-weighting** for better XY clustering.
<<<<<<< HEAD
  
## Expected vs Current Results

**Expected Output** (sample from the sheet):
| File | Expected Dimensions (cm) | Expected Volume (cm³) |
|------|---------------------------|------------------------|
| 3163484-rgba.png | 50×33×14.4 | 23760 |

**Current Output**:
| File | Measured Dimensions (cm) | Measured Volume (cm³) |
|------|---------------------------|------------------------|
| 3163484-rgba.png | 68.1×59.5×15.0 | 60940.702 |

## Next Steps:
- Fine-tune DBSCAN parameters per parcel
- Implement table-plane subtraction for better accuracy
- Automate batch processing + CSV export
=======

## Current Status

**Implemented so far**  
- `load_and_scale_mm()` – Loads `.ply` and scales to millimeters.  
- `isolate_parcel()` – DBSCAN-based isolation of the parcel from background.  
- Basic **Oriented Bounding Box (OBB)** measurement.  
- **Refined bounding box** after top/bottom trimming.  
- **Convex hull true volume** calculation.  
- Saving debug `.ply` outputs for visualization.  
- JSON output with measured dimensions & classification.

**Work in progress / Issues**  
- Measurements differ sign
>>>>>>> 629617d (Initial parcel measurement pipeline with OBB, refined box, and true volume calculation. Added README with current status, known issues, and next steps.)
