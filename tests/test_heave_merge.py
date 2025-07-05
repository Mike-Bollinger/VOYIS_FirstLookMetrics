#!/usr/bin/env python3
"""
Test script for heave data merging functionality
"""

import pandas as pd
import numpy as np
import os
import tempfile
from src.models.nav_plotter import NavPlotter, load_phins_ins_text

def create_test_nav_file():
    """Create a test NAV_STATE.txt file similar to the real format"""
    # Create test navigation data based on the real NAV_STATE format
    data = {
        'Time': ['5:53:22.730', '5:53:23.170', '5:53:23.607', '5:53:24.049', '5:53:24.490'],
        'Date': ['06/27/2024', '06/27/2024', '06/27/2024', '06/27/2024', '06/27/2024'],
        'Lon/Lat': ['28N05.55220  90W56.45729', '28N05.55225  90W56.45725', 
                   '28N05.55230  90W56.45721', '28N05.55228  90W56.45716', 
                   '28N05.55224  90W56.45712'],
        'Altitude': [79.230003357, 79.230003357, 84.982498169, 84.982498169, 84.982498169],
        'Depth of vehicle': [-0.127831459, -0.089560509, -0.083024025, -0.077548981, -0.046422005],
        'Heading': [242.774047852, 243.152191162, 242.545700073, 242.528945923, 243.160522461],
        'Pitch': [1.225421429, 0.184756935, -1.377160668, -0.672525287, 0.032464668],
        'Roll': [3.098049879, 17.894330978, 3.201517582, -13.359998703, -4.870718956]
    }
    
    df = pd.DataFrame(data)
    return df

def create_test_phins_file():
    """Create a test PHINS INS text file with heave data"""
    # Create test heave data
    times = ['5:53:22.730', '5:53:23.170', '5:53:23.607', '5:53:24.049', '5:53:24.490']
    heave_values = [0.1, -0.05, 0.15, -0.1, 0.08]  # Realistic heave values in meters
    
    data = {
        'time': times,
        'heave': heave_values,
        'pitch': [1.2, 0.2, -1.4, -0.7, 0.0],
        'roll': [3.1, 17.9, 3.2, -13.4, -4.9]
    }
    
    df = pd.DataFrame(data)
    return df

def test_heave_merging():
    """Test the heave merging functionality"""
    print("Testing heave data merging...")
    
    # Create test data
    nav_df = create_test_nav_file()
    phins_df = create_test_phins_file()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as nav_file:
        nav_df.to_csv(nav_file.name, index=False)
        nav_file_path = nav_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as phins_file:
        phins_df.to_csv(phins_file.name, index=False)
        phins_file_path = phins_file.name
    
    try:
        # Test PHINS INS text loading
        print(f"\n1. Testing PHINS INS text loading...")
        phins_data = load_phins_ins_text(phins_file_path)
        print(f"   PHINS data keys: {list(phins_data.keys())}")
        
        if 'HEAVE_' in phins_data:
            heave_data = phins_data['HEAVE_']
            print(f"   Found {len(heave_data)} heave measurements")
            print(f"   Heave range: {heave_data['Heave'].min():.3f} to {heave_data['Heave'].max():.3f} m")
        
        # Test navigation data loading and merging
        print(f"\n2. Testing navigation data loading and heave merging...")
        plotter = NavPlotter()
        nav_data = plotter.load_nav_data(nav_file_path)
        
        if nav_data is not None:
            print(f"   Loaded {len(nav_data)} navigation data points")
            print(f"   Navigation columns: {list(nav_data.columns)}")
            
            # Test heave merging
            print(f"\n3. Testing heave merging...")
            merged_data = plotter.merge_phins_heave(nav_data, phins_file_path)
            
            if 'heave' in merged_data.columns:
                print(f"   ✓ Heave column successfully added to navigation data")
                print(f"   Heave values: {merged_data['heave'].tolist()}")
                print(f"   Heave statistics:")
                print(f"     - Mean: {merged_data['heave'].mean():.3f} m")
                print(f"     - Std:  {merged_data['heave'].std():.3f} m")
                print(f"     - Min:  {merged_data['heave'].min():.3f} m")
                print(f"     - Max:  {merged_data['heave'].max():.3f} m")
            else:
                print(f"   ✗ Failed to add heave column")
            
            # Test individual plot creation (without actually saving)
            print(f"\n4. Testing plot capabilities...")
            available_motion_cols = []
            for motion in ['heave', 'pitch', 'roll']:
                if motion in merged_data.columns:
                    available_motion_cols.append(motion)
            
            print(f"   Available motion columns: {available_motion_cols}")
            
            if 'heave' in available_motion_cols:
                print(f"   ✓ Heave data ready for plotting")
            else:
                print(f"   ✗ Heave data not available for plotting")
        
        print(f"\n5. Test completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(nav_file_path)
            os.unlink(phins_file_path)
        except:
            pass

if __name__ == "__main__":
    test_heave_merging()
