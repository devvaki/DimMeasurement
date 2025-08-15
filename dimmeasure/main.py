# Processing pipeline

from __future__ import annotations
import argparse
import json
import pathlib
from typing import Dict, Any

from dimmeasure.io import load_pointcloud_and_image
from dimmeasure.preprocess import isolate_parcel
from dimmeasure.measure import classify_parcel_type, measure_box_dimensions, calculate_true_volume

def process_parcel(ply_path: str, png_path: str, verbose: bool = True) -> Dict[str, Any]:

    try:
        # Step 1: Load data
        if verbose:
            print(f"Loading data from {ply_path}")
        pcd_mm, img_bgr, parcel_id, timestamp_ms = load_pointcloud_and_image(ply_path, png_path)
        
        # Step 2: Isolate parcel from table
        if verbose:
            print("Isolating parcel from table...")
        parcel_pcd, isolation_info, frame = isolate_parcel(pcd_mm)
        
        if parcel_pcd.is_empty() or isolation_info["status"] == "failed":
            return {
                'parcel_id': parcel_id,
                'timestamp_ms': timestamp_ms,
                'success': False,
                'error': f"Parcel isolation failed: {isolation_info.get('reason', 'unknown')}",
                'length_mm': 0.0,
                'width_mm': 0.0,
                'height_mm': 0.0,
                'parcel_type': 'Unknown',
                'box_volume_mm3': 0.0,
                'true_volume_mm3': 0.0,
                'n_points': 0
            }
        
        if verbose:
            print(f"Parcel isolated: {isolation_info['n_points']} points using {isolation_info['method']}")
        
        # Step 3: Classify parcel type
        parcel_type = classify_parcel_type(parcel_pcd, frame)
        if verbose:
            print(f"Parcel type: {parcel_type}")
        
        # Step 4: Measure dimensions
        if verbose:
            print("Measuring dimensions...")
        length_mm, width_mm, height_mm, box_volume_mm3, _ = measure_box_dimensions(parcel_pcd, frame)
        
        # Step 5: Calculate true volume
        if verbose:
            print("Calculating true volume...")
        true_volume_mm3, _ = calculate_true_volume(parcel_pcd)
        
        # Compile results
        result = {
            'parcel_id': parcel_id,
            'timestamp_ms': timestamp_ms,
            'success': True,
            'length_mm': round(float(length_mm), 2),
            'width_mm': round(float(width_mm), 2),
            'height_mm': round(float(height_mm), 2),
            'parcel_type': parcel_type,
            'box_volume_mm3': round(float(box_volume_mm3), 0),
            'true_volume_mm3': round(float(true_volume_mm3), 0),
            'n_points': int(len(parcel_pcd.points)),
            'isolation_method': isolation_info['method']
        }
        
        if verbose:
            print_results(result)
        
        return result
        
    except Exception as e:
        return {
            'parcel_id': 'unknown',
            'timestamp_ms': 0,
            'success': False,
            'error': f"Processing failed: {str(e)}",
            'length_mm': 0.0,
            'width_mm': 0.0,
            'height_mm': 0.0,
            'parcel_type': 'Unknown',
            'box_volume_mm3': 0.0,
            'true_volume_mm3': 0.0,
            'n_points': 0
        }

def print_results(result: Dict[str, Any]) -> None:
    """Print formatted results"""
    print("\n" + "="*50)
    print("PARCEL MEASUREMENT RESULTS")
    print("="*50)
    print(f"Parcel ID: {result['parcel_id']}")
    print(f"Timestamp: {result['timestamp_ms']}")
    print(f"Type: {result['parcel_type']}")
    print(f"Points: {result['n_points']}")
    print(f"Method: {result.get('isolation_method', 'unknown')}")
    print("-"*50)
    print(f"Length:  {result['length_mm']:8.1f} mm")
    print(f"Width:   {result['width_mm']:8.1f} mm") 
    print(f"Height:  {result['height_mm']:8.1f} mm")
    print(f"Box Vol: {result['box_volume_mm3']:8.0f} mm³")
    print(f"True Vol: {result['true_volume_mm3']:8.0f} mm³")
    print("="*50)

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="Parcel dimension measurement")
    parser.add_argument("--ply", required=True, help="Path to PLY point cloud file")
    parser.add_argument("--png", required=True, help="Path to PNG image file")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Validate input files
    ply_path = pathlib.Path(args.ply)
    png_path = pathlib.Path(args.png)
    
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        return 1
    
    # Process parcel
    result = process_parcel(str(ply_path), str(png_path), verbose=not args.quiet)
    
    # Save results if requested
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return 0 if result['success'] else 1

if __name__ == "__main__":
    exit(main())