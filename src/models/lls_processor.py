import os
import sys
import traceback
import tempfile
import shutil
import glob
from typing import Optional, Callable

# Add models directory to path for importing Ship_LLS_Read_Plot_V2 and read_phinsdata
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from Ship_LLS_Read_Plot_V2 import Step01_Find_Good_Data
    import read_phinsdata as phins
except ImportError as e:
    print(f"Error importing LLS processing modules: {e}")
    Step01_Find_Good_Data = None
    phins = None

class LLSProcessor:
    """Wrapper class for LLS processing functionality"""
    
    def __init__(self, log_callback: Optional[Callable] = None, 
                 progress_callback: Optional[Callable] = None):
        """
        Initialize LLS processor
        
        Args:
            log_callback: Function to call for logging messages
            progress_callback: Function to call for progress updates
        """
        self.log_callback = log_callback or print
        self.progress_callback = progress_callback or (lambda x, msg: None)
        
        # Default processing parameters
        self.min_intensity_threshold = 100
        self.bad_point_threshold = 70
        self.radius = 4.0
        
    def log_message(self, message: str):
        """Log a message using the callback"""
        if self.log_callback:
            self.log_callback(message)
    
    def update_progress(self, value: int, message: str = "Processing..."):
        """Update progress using the callback"""
        if self.progress_callback:
            self.progress_callback(value, message)
    
    def process_lls_data(self, lls_folder: str, phins_nav_file: str, 
                        output_folder: str) -> bool:
        """
        Process LLS data using the Ship_LLS_Read_Plot_V2 functionality
        
        Args:
            lls_folder: Path to folder containing LLS .xyz files
            phins_nav_file: Path to Phins navigation file (.bin, .txt, or .csv)
            output_folder: Output directory for processed results
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        if Step01_Find_Good_Data is None:
            self.log_message("Error: LLS processing modules not available")
            return False
        
        try:
            # Validate inputs
            if not os.path.exists(lls_folder):
                self.log_message(f"Error: LLS folder does not exist: {lls_folder}")
                return False
                
            if not os.path.exists(phins_nav_file):
                self.log_message(f"Error: Phins navigation file does not exist: {phins_nav_file}")
                return False
            
            # Find LLS files
            xyz_files = glob.glob(os.path.join(lls_folder, 'LLS_*.xyz'))
            if not xyz_files:
                self.log_message(f"No LLS_*.xyz files found in {lls_folder}")
                return False
            
            self.log_message(f"Found {len(xyz_files)} LLS files to process")
            self.update_progress(10, "Setting up LLS processing...")
            
            # Create temporary directory structure that matches expected format
            temp_base_dir = self.setup_temp_directory(lls_folder, phins_nav_file, output_folder)
            
            self.update_progress(20, "Processing LLS data...")
            
            # Call the original processing function with log callback
            Step01_Find_Good_Data(
                BaseDir=temp_base_dir,
                MIN_INTENSITY_THRESHOLD=self.min_intensity_threshold,
                BAD_POINT_THRESHOLD=self.bad_point_threshold,
                RADIUS=self.radius,
                gui_output_dir=output_folder,  # Pass the GUI output directory
                xyz_files=None,  # Let it find all LLS_*.xyz files
                log_callback=self.log_message  # Pass the log callback
            )
            
            self.update_progress(90, "Copying results to output folder...")
            
            # Copy results to final output location
            self.copy_results_to_output(temp_base_dir, output_folder)
            
            self.update_progress(100, "LLS processing complete")
            self.log_message("LLS data processing completed successfully")
            
            return True
            
        except Exception as e:
            self.log_message(f"Error during LLS processing: {str(e)}")
            self.log_message(f"Details: {traceback.format_exc()}")
            return False
    
    def setup_temp_directory(self, lls_folder: str, phins_nav_file: str, 
                           output_folder: str) -> str:
        """
        Set up temporary directory structure expected by Ship_LLS_Read_Plot_V2
        
        Args:
            lls_folder: Source LLS folder
            phins_nav_file: Phins navigation file
            output_folder: Output folder
            
        Returns:
            str: Path to temporary base directory
        """
        # Create temporary base directory
        temp_dir = tempfile.mkdtemp(prefix="lls_processing_")
        
        # Create expected subdirectories
        lls_dir = os.path.join(temp_dir, 'LLS')
        vehicle_data_dir = os.path.join(temp_dir, 'Vehicle_Data')
        
        os.makedirs(lls_dir, exist_ok=True)
        os.makedirs(vehicle_data_dir, exist_ok=True)
        
        # Copy LLS files
        self.log_message("Copying LLS files...")
        for file in os.listdir(lls_folder):
            if file.endswith('.xyz') and file.startswith('LLS_'):
                src = os.path.join(lls_folder, file)
                dst = os.path.join(lls_dir, file)
                shutil.copy2(src, dst)
        
        # Handle Phins navigation file
        self.log_message("Setting up navigation data...")
        nav_file_name = os.path.basename(phins_nav_file)
        nav_file_ext = os.path.splitext(nav_file_name)[1].lower()
        
        if nav_file_ext == '.txt':
            # For text files that look like the test file, we need to convert them
            self.process_text_nav_file(phins_nav_file, vehicle_data_dir)
        elif nav_file_ext == '.bin':
            # Copy binary file directly
            dst = os.path.join(vehicle_data_dir, nav_file_name)
            shutil.copy2(phins_nav_file, dst)
        else:
            # For other formats, copy as-is and let the phins module handle it
            dst = os.path.join(vehicle_data_dir, nav_file_name)
            shutil.copy2(phins_nav_file, dst)
        
        return temp_dir
    
    def process_text_nav_file(self, nav_file_path: str, vehicle_data_dir: str):
        """
        Process text navigation file that matches the format in tests/NavFileTest_DIVE003_NAV.txt
        """
        try:
            self.log_message("Processing text navigation file...")
            
            # Read the navigation file and parse it
            import pandas as pd
            from datetime import datetime
            import numpy as np
            
            # The format appears to be:
            # Time, Date, GPS_Position, Easting, Northing, Heading, Depth, Altitude, Pitch, Roll, ?
            
            nav_data = []
            with open(nav_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 11:  # Ensure we have enough columns
                        try:
                            time_str = parts[0].strip()
                            date_str = parts[1].strip()
                            
                            # Parse GPS position (format like "28N05.60197  90W56.77129")
                            gps_parts = parts[2].strip().split()
                            if len(gps_parts) >= 2:
                                # Extract latitude and longitude
                                lat_str = gps_parts[0]  # e.g., "28N05.60197"
                                lon_str = gps_parts[1]  # e.g., "90W56.77129"
                                
                                # Convert to decimal degrees (simplified conversion)
                                # This is a basic conversion - you may need to adjust based on exact format
                                
                                # Parse other values
                                easting = float(parts[3])
                                northing = float(parts[4])
                                heading = float(parts[5])
                                depth = float(parts[6])
                                altitude = float(parts[7])
                                pitch = float(parts[8])
                                roll = float(parts[9])
                                
                                # Create datetime
                                datetime_str = f"{date_str} {time_str}"
                                dt = datetime.strptime(datetime_str, "%m/%d/%Y %H:%M:%S.%f")
                                
                                nav_data.append({
                                    'DateTime': dt,
                                    'Timestamp': dt.timestamp(),
                                    'Easting': easting,
                                    'Northing': northing,
                                    'Heading': heading,
                                    'Depth': depth,
                                    'Altitude': altitude,
                                    'Pitch': pitch,
                                    'Roll': roll
                                })
                        except (ValueError, IndexError) as e:
                            continue  # Skip invalid lines
            
            if not nav_data:
                self.log_message("Warning: No valid navigation data found in text file")
                return
            
            # Create DataFrame and save as CSV files expected by the phins module
            df = pd.DataFrame(nav_data)
            
            # Create the CSV files that read_phinsdata expects
            # This is a simplified version - you may need to adjust based on actual requirements
            
            # UTMWGS84 file (position data)
            utmwgs_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'AUV_Easting': df['Easting'],
                'AUV_Northing': df['Northing']
            })
            utmwgs_df.to_csv(os.path.join(vehicle_data_dir, 'UTMWGS84.csv'), index=False)
            
            # HEHDT file (heading data)
            hehdt_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'Heading': df['Heading']
            })
            hehdt_df.to_csv(os.path.join(vehicle_data_dir, 'HEHDT_.csv'), index=False)
            
            # Attitude file (pitch/roll data)
            attitude_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'Pitch': df['Pitch'],
                'Roll': df['Roll']
            })
            attitude_df.to_csv(os.path.join(vehicle_data_dir, 'Atitude.csv'), index=False)
            
            # DEPIN file (depth data)
            depin_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'Depth': df['Depth']
            })
            depin_df.to_csv(os.path.join(vehicle_data_dir, 'DEPIN_.csv'), index=False)
            
            # LOGDVL file (altitude data)
            logdvl_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'DVL_Distance_2btm': df['Altitude']
            })
            logdvl_df.to_csv(os.path.join(vehicle_data_dir, 'LOGDVL.csv'), index=False)
            
            # SPEED file (velocity data - create dummy data if not available)
            speed_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'Speed': np.ones(len(df)) * 1.0  # Default speed
            })
            speed_df.to_csv(os.path.join(vehicle_data_dir, 'SPEED_.csv'), index=False)
            
            # POSITI file (position data)
            positi_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'Latitude': np.zeros(len(df)),  # Placeholder
                'Longitude': np.zeros(len(df))  # Placeholder
            })
            positi_df.to_csv(os.path.join(vehicle_data_dir, 'POSITI.csv'), index=False)
            
            self.log_message(f"Created navigation CSV files from text file with {len(df)} records")
            
        except Exception as e:
            self.log_message(f"Error processing text navigation file: {str(e)}")
            self.log_message(f"Details: {traceback.format_exc()}")
    
    def copy_results_to_output(self, temp_dir: str, output_folder: str):
        """Copy processing results from temporary directory to final output"""
        # Since we're now passing gui_output_dir directly to Step01_Find_Good_Data,
        # the files should already be in the correct output location.
        # We only need to copy any files that might still be in the temp directory.
        
        files_copied = 0
        
        # Check if there are any LLS_Output files in temp that weren't copied directly
        temp_lls_output = os.path.join(temp_dir, 'LLS_Output')
        if os.path.exists(temp_lls_output):
            for file in os.listdir(temp_lls_output):
                src = os.path.join(temp_lls_output, file)
                dst = os.path.join(output_folder, file)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    self.log_message(f"Copied {file} to output directory")
                    files_copied += 1
        
        # Check if there are any Vehicle_Output files in temp that weren't copied directly
        temp_vehicle_output = os.path.join(temp_dir, 'Vehicle_Output')
        if os.path.exists(temp_vehicle_output):
            for file in os.listdir(temp_vehicle_output):
                src = os.path.join(temp_vehicle_output, file)
                dst = os.path.join(output_folder, file)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    self.log_message(f"Copied {file} to output directory")
                    files_copied += 1
        
        if files_copied == 0:
            self.log_message("All files already in target output directory")
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            self.log_message("Cleaned up temporary processing directory")
        except Exception as e:
            self.log_message(f"Warning: Could not clean up temp directory: {e}")
    
    def set_processing_parameters(self, min_intensity: float = 100, 
                                bad_point_threshold: float = 70, 
                                radius: float = 4.0):
        """Set processing parameters"""
        self.min_intensity_threshold = min_intensity
        self.bad_point_threshold = bad_point_threshold
        self.radius = radius
        
        self.log_message(f"LLS processing parameters set:")
        self.log_message(f"  Min Intensity: {min_intensity}")
        self.log_message(f"  Bad Point Threshold: {bad_point_threshold}%")
        self.log_message(f"  Radius: {radius}m")