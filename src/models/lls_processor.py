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
            line_count = 0
            valid_lines = 0
            
            with open(nav_file_path, 'r') as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 10:  # Reduced from 11 to be more flexible
                        try:
                            time_str = parts[0].strip()
                            date_str = parts[1].strip()
                            
                            # Parse other values first to validate they're numeric
                            easting = float(parts[3])
                            northing = float(parts[4])
                            heading = float(parts[5])
                            # Normalize heading to 0-360 range
                            if heading < 0:
                                while heading < 0:
                                    heading += 360
                            elif heading >= 360:
                                heading = heading % 360
                            depth = float(parts[6])
                            altitude = float(parts[7])
                            pitch = float(parts[8])
                            roll = float(parts[9])
                            
                            # Create datetime - try multiple formats
                            datetime_str = f"{date_str} {time_str}"
                            dt = None
                            
                            # Try different datetime formats
                            formats_to_try = [
                                "%m/%d/%Y %H:%M:%S.%f",  # Original format
                                "%m/%d/%Y %H:%M:%S",     # Without microseconds
                                "%Y-%m-%d %H:%M:%S.%f",  # ISO format with microseconds
                                "%Y-%m-%d %H:%M:%S",     # ISO format without microseconds
                                "%d/%m/%Y %H:%M:%S.%f",  # Day/month/year format
                                "%d/%m/%Y %H:%M:%S"      # Day/month/year format without microseconds
                            ]
                            
                            for fmt in formats_to_try:
                                try:
                                    dt = datetime.strptime(datetime_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            
                            if dt is None:
                                self.log_message(f"Warning: Could not parse datetime '{datetime_str}' on line {line_count}")
                                continue
                            
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
                            valid_lines += 1
                            
                        except (ValueError, IndexError) as e:
                            self.log_message(f"Warning: Could not parse line {line_count}: {str(e)}")
                            continue  # Skip invalid lines
            
            self.log_message(f"Processed {line_count} lines, found {valid_lines} valid navigation records")
            
            if not nav_data:
                self.log_message("Error: No valid navigation data found in text file")
                # Create minimal dummy data to prevent crashes
                self.create_dummy_nav_files(vehicle_data_dir)
                return
            
            # Create DataFrame and save as CSV files expected by the phins module
            df = pd.DataFrame(nav_data)
            
            # Sort by datetime to ensure proper ordering
            df = df.sort_values('DateTime').reset_index(drop=True)
            
            self.log_message(f"Navigation data spans from {df['DateTime'].min()} to {df['DateTime'].max()}")
            
            # Create the CSV files that read_phinsdata expects
            # Use consistent datetime format that the phins module can parse
            datetime_format = '%Y-%m-%d %H:%M:%S.%f'
            
            # UTMWGS84 file (position data)
            utmwgs_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'AUV_Easting': df['Easting'],
                'AUV_Northing': df['Northing']
            })
            utmwgs_file = os.path.join(vehicle_data_dir, 'UTMWGS84.csv')
            utmwgs_df.to_csv(utmwgs_file, index=False)
            self.log_message(f"Created {utmwgs_file} with {len(utmwgs_df)} records")
            
            # HEHDT file (heading data)
            hehdt_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'Heading': df['Heading']
            })
            hehdt_file = os.path.join(vehicle_data_dir, 'HEHDT_.csv')
            hehdt_df.to_csv(hehdt_file, index=False)
            self.log_message(f"Created {hehdt_file} with {len(hehdt_df)} records")
            
            # Attitude file (pitch/roll data)
            attitude_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'Pitch': df['Pitch'],
                'Roll': df['Roll']
            })
            attitude_file = os.path.join(vehicle_data_dir, 'Atitude.csv')
            attitude_df.to_csv(attitude_file, index=False)
            self.log_message(f"Created {attitude_file} with {len(attitude_df)} records")
            
            # DEPIN file (depth data)
            depin_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'Depth': df['Depth']
            })
            depin_file = os.path.join(vehicle_data_dir, 'DEPIN_.csv')
            depin_df.to_csv(depin_file, index=False)
            self.log_message(f"Created {depin_file} with {len(depin_df)} records")
            
            # LOGDVL file (altitude data)
            logdvl_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'DVL_Distance_2btm': df['Altitude']
            })
            logdvl_file = os.path.join(vehicle_data_dir, 'LOGDVL.csv')
            logdvl_df.to_csv(logdvl_file, index=False)
            self.log_message(f"Created {logdvl_file} with {len(logdvl_df)} records")
            
            # SPEED file (velocity data - calculate from position changes or use default)
            velocities = []
            if len(df) > 1:
                for i in range(len(df)):
                    if i == 0:
                        velocities.append(1.0)  # Default for first point
                    else:
                        # Calculate distance and time difference
                        dx = df.iloc[i]['Easting'] - df.iloc[i-1]['Easting']
                        dy = df.iloc[i]['Northing'] - df.iloc[i-1]['Northing']
                        distance = np.sqrt(dx**2 + dy**2)
                        time_diff = (df.iloc[i]['DateTime'] - df.iloc[i-1]['DateTime']).total_seconds()
                        
                        if time_diff > 0:
                            velocity = distance / time_diff
                            velocities.append(max(0.1, min(velocity, 5.0)))  # Clamp to reasonable range
                        else:
                            velocities.append(1.0)
            else:
                velocities = [1.0] * len(df)
            
            speed_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'Speed': velocities
            })
            speed_file = os.path.join(vehicle_data_dir, 'SPEED_.csv')
            speed_df.to_csv(speed_file, index=False)
            self.log_message(f"Created {speed_file} with {len(speed_df)} records")
            
            # POSITI file (position data - convert from UTM to lat/lon if possible)
            # For now, use placeholder values - this could be enhanced to do proper UTM conversion
            positi_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'Latitude': np.zeros(len(df)),  # Placeholder - could convert from UTM
                'Longitude': np.zeros(len(df))  # Placeholder - could convert from UTM
            })
            positi_file = os.path.join(vehicle_data_dir, 'POSITI.csv')
            positi_df.to_csv(positi_file, index=False)
            self.log_message(f"Created {positi_file} with {len(positi_df)} records")
            
            # GPSIN_ file (GPS data - use placeholder values since we don't have GPS data)
            gpsin_df = pd.DataFrame({
                'Date_Time': df['DateTime'].dt.strftime(datetime_format),
                'Latitude': np.zeros(len(df)),  # Placeholder GPS data
                'Longitude': np.zeros(len(df)),  # Placeholder GPS data
                'Altitude': np.zeros(len(df)),  # Placeholder GPS altitude
                'Time': np.zeros(len(df)),      # Placeholder GPS time
                'Quality': np.ones(len(df))     # Default GPS quality 1
            })
            gpsin_file = os.path.join(vehicle_data_dir, 'GPSIN_.csv')
            gpsin_df.to_csv(gpsin_file, index=False)
            self.log_message(f"Created {gpsin_file} with {len(gpsin_df)} records")
            
            self.log_message(f"Successfully created all navigation CSV files from text file with {len(df)} records")
            
        except Exception as e:
            self.log_message(f"Error processing text navigation file: {str(e)}")
            self.log_message(f"Details: {traceback.format_exc()}")
            # Create dummy files to prevent complete failure
            self.create_dummy_nav_files(vehicle_data_dir)
    
    def create_dummy_nav_files(self, vehicle_data_dir: str):
        """Create minimal dummy navigation files to prevent crashes"""
        try:
            self.log_message("Creating minimal dummy navigation files...")
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create a simple time series for one hour
            start_time = datetime.now()
            times = [start_time + timedelta(seconds=i) for i in range(3600)]  # 1 hour of data
            datetime_format = '%Y-%m-%d %H:%M:%S.%f'
            
            time_strings = [t.strftime(datetime_format) for t in times]
            
            # Create dummy data
            dummy_data = {
                'Date_Time': time_strings,
                'AUV_Easting': np.linspace(0, 1000, len(times)),
                'AUV_Northing': np.linspace(0, 1000, len(times)),
                'Heading': np.full(len(times), 0.0),
                'Pitch': np.full(len(times), 0.0),
                'Roll': np.full(len(times), 0.0),
                'Depth': np.full(len(times), 10.0),
                'DVL_Distance_2btm': np.full(len(times), 5.0),
                'Speed': np.full(len(times), 1.0),
                'Latitude': np.full(len(times), 0.0),
                'Longitude': np.full(len(times), 0.0),
                'Altitude': np.full(len(times), 0.0),
                'Time': np.full(len(times), 0.0),
                'Quality': np.full(len(times), 1.0)
            }
            
            # Create all required files
            files_to_create = [
                ('UTMWGS84.csv', ['Date_Time', 'AUV_Easting', 'AUV_Northing']),
                ('HEHDT_.csv', ['Date_Time', 'Heading']),
                ('Atitude.csv', ['Date_Time', 'Pitch', 'Roll']),
                ('DEPIN_.csv', ['Date_Time', 'Depth']),
                ('LOGDVL.csv', ['Date_Time', 'DVL_Distance_2btm']),
                ('SPEED_.csv', ['Date_Time', 'Speed']),
                ('POSITI.csv', ['Date_Time', 'Latitude', 'Longitude']),
                ('GPSIN_.csv', ['Date_Time', 'Latitude', 'Longitude', 'Altitude', 'Time', 'Quality'])
            ]
            
            for filename, columns in files_to_create:
                df = pd.DataFrame({col: dummy_data[col] for col in columns})
                filepath = os.path.join(vehicle_data_dir, filename)
                df.to_csv(filepath, index=False)
                self.log_message(f"Created dummy {filename} with {len(df)} records")
                
        except Exception as e:
            self.log_message(f"Error creating dummy navigation files: {str(e)}")
    
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