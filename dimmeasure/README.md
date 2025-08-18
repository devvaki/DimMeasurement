# Parcel Dimension Measurement Pipeline
## Problem Statement
This project builds an **automated pipeline** that processes 3D point cloud data of parcels to estimate:
- **L, W, H (mm)**
- **Box Volume (V = L × W × H)**
- **True Volume (point cloud reconstruction)**
- **Parcel Type** (Box / Polybag)

The goal is to achieve **high accuracy** (±10 mm for dimensions, ±50 mm³ for volume).

## Given
The dataset includes representative photos showing:
- Variation in parcel sizes
- Common shapes and surface textures
- Reflective or glossy packaging materials
- Color variations and potential overhangs or deformations
- Real-world placement on AGV top surfaces

## Pipeline Overview

### 1. Step 1 – Sanitize (`step1_sanitize.py`)
- Cleans raw point clouds (removes NaNs, noise, outliers).
- Normalizes to a consistent coordinate frame.
- **Output:** cleaned `.ply` point clouds.

### 2. Step 2 – DBSCAN (`step2_DBSCAN.py`)
- Applies clustering (DBSCAN) to separate parcel from background/plane.
- Filters points belonging only to the parcel.
- **Output:** clustered parcel point clouds.

### 3. Step 3 – OBB (Oriented Bounding Box) (`step3_obb.py`)
- Fits an **Oriented Bounding Box** (OBB) around the parcel.
- Extracts preliminary L, W, H dimensions.
- **Output:** annotated plots with bounding box dimensions.

### 4. Step 4 – Volume Calculation (`step4_volume.py`)
- Calculates:
  - **Box Volume (L × W × H)**  
  - **True Volume (voxelized / convex hull approximation)**  
- Compares against Ground Truth (GT).
- Saves results in **CSV, JSON, and annotated images**.
- **Output:** Measurement reports, error percentages.

## Techniques Used
- **RANSAC Plane Segmentation** – ground removal.  
- **DBSCAN Clustering** – parcel isolation.  
- **Oriented Bounding Box (OBB)** – L, W, H extraction.  
- **Convex Hull / Voxelization** – true volume estimation.  
- **Pandas + JSON** – structured logging.  
- **Matplotlib 3D** – visualization and annotation.

## Key Parameters to Tune

Several parameters have major impact on accuracy:

- **RANSAC Distance (`--ransac-dist`)** → Plane fitting tolerance.  
  Too high → background leaks in, too low → parcel points lost.
- **RANSAC Iterations (`--ransac-iters`)** → Number of iterations for stable plane removal.
- **Height Thresholds (`--h-min`, `--h-max`)** → Cropping range along Z-axis.
- **DBSCAN Parameters:**
  - `eps` (neighborhood size)  
  - `min_pts` (minimum cluster size)  
  Controls how parcel is segmented vs. noise.
- **Hull Margin (`--hull-margin-mm`)** → Extra buffer around convex hull for true volume estimation.

👉 These parameters need dataset-specific tuning to balance **box vs. polybag cases**.

## Challenges Faced & Fixes

1. **Background interference**  
   - Issue: Slab or noise points included in bounding box.  
   - Fix: Tuned `ransac-dist` + cropping thresholds.

2. **Over/under segmentation in DBSCAN**  
   - Issue: Polybags often produced multiple small clusters.  
   - Fix: Adjusted `eps` and `min_pts` dynamically.

3. **Accuracy mismatch (GT vs. measured)**  
   - Issue: Major deviations in Length/Width due to slab projection (mostly occured for polybags) 
   - Fix: Focused tuning on **X & Y dimensions** (major error contributors).

4. **Verbose logs**  
   - Fix: Shortened column headers (e.g., `ErrV(%)` instead of `err_box_volume_%`).

## Accuracy Insights

- **Boxes:**  
  Dimensions stable (±10 mm). Errors from **slab leakage** & **OBB misalignment**.

- **Polybags:**  
  True Volume estimation is unstable. Convex hull inflates volume → errors up to **100%+**.  
  Needs better modeling (voxel carving / multi-view).

- **Summary:**  
  - **Length & Width**: Main error sources.  
  - **Height**: More stable.  
  - **Box Volume**: Acceptable for rigid boxes.  
  - **True Volume**: Requires advanced methods.

## Outputs

- **Step 1 Plots (`step1_plots/`)** → Cleaning & noise removal.  
- **Step 2 Plots (`step2_plots/`)** → DBSCAN clustering; helps debug segmentation.  
- **Step 3 Plots (`step3_plots/`)** → OBB visualization with L, W, H.  
- **Step 4 Results (`results/`)**  
  - **CSV/JSON**: Structured measurement records.  
  - **3D Plots**: 3 Dimension overlays.  
  - **Annotated PNGs**: RGB image + embossed volumes.

## Next Steps
- Improve **true volume** estimation for polybags.  
- Explore **multi-view heightmap fusion**.  
- Build **auto-parameter tuning** per parcel type.  
- Move towards **real-time deployment**.
