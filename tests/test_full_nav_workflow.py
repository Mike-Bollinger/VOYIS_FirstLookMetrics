#!/usr/bin/env python3
"""
End-to-end test script for navigation plotting with heave data
"""

import pandas as pd
import numpy as np
import os
import tempfile
from src.models.nav_plotter import NavPlotter

def create_realistic_test_data():
    """Create realistic test data that mimics actual NAV_STATE and PHINS data"""
    
    # Create realistic navigation data (50 points over ~1 minute)
    n_points = 50
    start_time = "5:53:22"
    
    # Generate time series
    times = []
    for i in range(n_points):
        seconds = 22 + i * 1.5  # 1.5 second intervals
        mins = 53 + int(seconds // 60)
        secs = seconds % 60
        times.append(f"5:{mins:02d}:{secs:06.3f}")
    
    # Generate realistic lat/lon track (small movements)
    base_lat = 28.092587  # Convert 28N05.55220 to decimal degrees
    base_lon = -90.940955  # Convert 90W56.45729 to decimal degrees
    
    # Create a curved track
    track_progress = np.linspace(0, 2*np.pi, n_points)
    lat_offsets = np.cos(track_progress) * 0.0001  # Small movements
    lon_offsets = np.sin(track_progress) * 0.0001
    
    latitudes = base_lat + lat_offsets
    longitudes = base_lon + lon_offsets
    
    # Generate realistic attitude data
    np.random.seed(42)
    headings = 240 + 5 * np.sin(track_progress) + np.random.normal(0, 1, n_points)
    pitches = 2 * np.sin(track_progress * 3) + np.random.normal(0, 0.5, n_points)
    rolls = 3 * np.cos(track_progress * 2) + np.random.normal(0, 1, n_points)
    
    # Generate depth profile (diving and ascending)
    depths = 5 + 15 * np.sin(track_progress * 0.5) + np.random.normal(0, 0.5, n_points)
    depths = np.abs(depths)  # Ensure positive depths
    
    nav_data = {
        'Time': times,
        'Date': ['06/27/2024'] * n_points,
        'latitude': latitudes,
        'longitude': longitudes,
        'depth': depths,
        'heading': headings,
        'pitch': pitches,
        'roll': rolls,
        'altitude': 80 + np.random.normal(0, 2, n_points)
    }
    
    # Create PHINS heave data (realistic heave motion)
    heave_data = {
        'time': times,
        'heave': 0.2 * np.sin(track_progress * 4) + np.random.normal(0, 0.05, n_points),  # Realistic heave
        'pitch_phins': pitches + np.random.normal(0, 0.1, n_points),  # Slightly different from nav
        'roll_phins': rolls + np.random.normal(0, 0.1, n_points)
    }
    
    return pd.DataFrame(nav_data), pd.DataFrame(heave_data)

def test_full_plotting_workflow():
    """Test the complete navigation plotting workflow with heave data"""
    print("Testing complete navigation plotting workflow...")
    
    # Create realistic test data
    nav_df, phins_df = create_realistic_test_data()
    
    print(f"Created test data:")
    print(f"  - Navigation data: {len(nav_df)} points")
    print(f"  - PHINS data: {len(phins_df)} points")
    print(f"  - Heave range: {phins_df['heave'].min():.3f} to {phins_df['heave'].max():.3f} m")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_NAV_STATE.txt', delete=False) as nav_file:
        nav_df.to_csv(nav_file.name, index=False)
        nav_file_path = nav_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_PHINS_INS.txt', delete=False) as phins_file:
        phins_df.to_csv(phins_file.name, index=False)
        phins_file_path = phins_file.name
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            print(f"\nTesting navigation processing...")
            print(f"  NAV file: {nav_file_path}")
            print(f"  PHINS file: {phins_file_path}")
            print(f"  Output dir: {output_dir}")
            
            # Test the complete workflow
            plotter = NavPlotter()
            
            def test_log_callback(message):
                print(f"  LOG: {message}")
            
            # Process navigation file with PHINS heave data
            plotter.process_navigation_file(
                nav_file_path, 
                output_dir, 
                dive_name="TEST_DIVE", 
                log_callback=test_log_callback,
                phins_file_path=phins_file_path
            )
            
            # Check what files were created
            print(f"\nChecking output files...")
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.startswith('Nav_') and file.endswith('.png'):
                        output_files.append(file)
            
            output_files.sort()
            print(f"Created {len(output_files)} plot files:")
            for file in output_files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file} ({file_size:,} bytes)")
            
            # Verify expected plots were created
            expected_plots = [
                'Nav_Motion_Analysis.png',           # Comprehensive plot
                'Nav_Heave_Map.png',                # Individual heave map
                'Nav_Heave_Timeseries.png',         # Individual heave time series
                'Nav_Heave_Histogram.png',          # Individual heave histogram
                'Nav_Pitch_Map.png',                # Individual pitch plots
                'Nav_Pitch_Timeseries.png',
                'Nav_Pitch_Histogram.png',
                'Nav_Roll_Map.png',                 # Individual roll plots
                'Nav_Roll_Timeseries.png',
                'Nav_Roll_Histogram.png',
                'Nav_Motion_Timeseries.png',        # Combined motion time series
                'Nav_Depth_Profile.png'             # Depth profile
            ]
            
            missing_plots = []
            for expected in expected_plots:
                if expected not in output_files:
                    missing_plots.append(expected)
            
            if missing_plots:
                print(f"\n⚠ Missing expected plots: {missing_plots}")
            else:
                print(f"\n✓ All expected plots were created successfully!")
            
            # Check if heave plots contain data
            heave_plots = [f for f in output_files if 'Heave' in f]
            if heave_plots:
                print(f"✓ Heave plots created: {heave_plots}")
            else:
                print(f"✗ No heave plots found")
            
            print(f"\n✓ Full workflow test completed successfully!")
            
        except Exception as e:
            print(f"\nError during full workflow test: {e}")
            import traceback
            traceback.print_exc()
            
    # Clean up temporary files
    try:
        os.unlink(nav_file_path)
        os.unlink(phins_file_path)
    except:
        pass

if __name__ == "__main__":
    test_full_plotting_workflow()
