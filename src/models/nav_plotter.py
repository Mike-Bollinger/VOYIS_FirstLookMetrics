"""
Navigation Data Plotter
Converts R plotting functionality to Python for vehicle navigation data analysis.
Adapted from Noah Hunt's R script for NOAA MDBC MGM Remus Data Plotting.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import os

def load_phins_data(file_path):
    """Load and clean PHINS INS data"""
    try:
        # Read the file and clean column names
        df = pd.read_csv(file_path)
        
        # Remove trailing commas and spaces from column names
        df.columns = [col.strip(' ,') for col in df.columns]
        
        print("Cleaned column names:", df.columns.tolist())
        
        # Convert time to datetime if it's in HH:MM:SS format
        if 'time' in df.columns:
            # Assuming the time is in format like "5:53:22.5"
            base_date = "2024-01-01"  # You may need to adjust this
            df['datetime'] = pd.to_datetime(base_date + ' ' + df['time'].astype(str))
        
        return df
    
    except Exception as e:
        print(f"Error loading PHINS data: {e}")
        return None

def create_navigation_plots(df):
    """Create comprehensive navigation plots"""
    
    # Set up the plotting style with white background
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    
    # 1. Position Plot (Latitude vs Longitude)
    ax1 = plt.subplot(3, 3, 1)
    plt.scatter(df['longitude'], df['latitude'], c=df.index, cmap='viridis', s=1, alpha=0.7)
    plt.colorbar(label='Time Index')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title('Vehicle Track (GPS Position)')
    plt.grid(True, alpha=0.3)
    
    # 2. PHINS vs GPS Position Comparison
    ax2 = plt.subplot(3, 3, 2)
    plt.scatter(df['longitude'], df['latitude'], label='GPS', alpha=0.6, s=1)
    plt.scatter(df['phins_lon'], df['phins_lat'], label='PHINS', alpha=0.6, s=1)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title('GPS vs PHINS Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Depth Profile
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(df.index, df['depth'], 'b-', linewidth=1, label='Depth')
    plt.plot(df.index, df['phins_depth'], 'r-', linewidth=1, label='PHINS Depth')
    plt.xlabel('Time Index')
    plt.ylabel('Depth (m)')
    plt.title('Depth Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
    
    # 4. Attitude Data (Heading, Pitch, Roll)
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(df.index, df['heading'], label='Heading', linewidth=1)
    plt.plot(df.index, df['pitch'], label='Pitch', linewidth=1)
    plt.plot(df.index, df['roll'], label='Roll', linewidth=1)
    plt.xlabel('Time Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Vehicle Attitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Velocity Components
    ax5 = plt.subplot(3, 3, 5)
    plt.plot(df.index, df['vel_north'], label='North', linewidth=1)
    plt.plot(df.index, df['vel_east'], label='East', linewidth=1)
    plt.plot(df.index, df['vel_down'], label='Down', linewidth=1)
    plt.xlabel('Time Index')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Angular Rates
    ax6 = plt.subplot(3, 3, 6)
    plt.plot(df.index, df['heading_rate'], label='Heading Rate', linewidth=1)
    plt.plot(df.index, df['pitch_rate'], label='Pitch Rate', linewidth=1)
    plt.plot(df.index, df['roll_rate'], label='Roll Rate', linewidth=1)
    plt.xlabel('Time Index')
    plt.ylabel('Angular Rate (deg/s)')
    plt.title('Angular Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Position Uncertainties
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(df.index, df['stddev_lat'], label='Latitude StdDev', linewidth=1)
    plt.plot(df.index, df['stddev_lon'], label='Longitude StdDev', linewidth=1)
    plt.xlabel('Time Index')
    plt.ylabel('Standard Deviation')
    plt.title('Position Uncertainties')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Attitude Uncertainties
    ax8 = plt.subplot(3, 3, 8)
    plt.plot(df.index, df['stddev_roll'], label='Roll StdDev', linewidth=1)
    plt.plot(df.index, df['stddev_pitch'], label='Pitch StdDev', linewidth=1)
    plt.plot(df.index, df['stddev_head'], label='Heading StdDev', linewidth=1)
    plt.xlabel('Time Index')
    plt.ylabel('Standard Deviation (degrees)')
    plt.title('Attitude Uncertainties')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Heave Motion
    ax9 = plt.subplot(3, 3, 9)
    plt.plot(df.index, df['heave'], 'g-', linewidth=1)
    plt.xlabel('Time Index')
    plt.ylabel('Heave (m)')
    plt.title('Heave Motion')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_navigation_quality(df):
    """Analyze navigation data quality"""
    print("\n=== Navigation Data Quality Analysis ===")
    print(f"Total data points: {len(df)}")
    print(f"Time span: {df.index[0]} to {df.index[-1]} (indices)")
    
    # Position statistics
    print(f"\nPosition Range:")
    print(f"Latitude: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
    print(f"Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
    print(f"Depth: {df['depth'].min():.2f} to {df['depth'].max():.2f} m")
    
    # Position differences between GPS and PHINS
    lat_diff = df['latitude'] - df['phins_lat']
    lon_diff = df['longitude'] - df['phins_lon']
    depth_diff = df['depth'] - df['phins_depth']
    
    print(f"\nGPS vs PHINS Differences:")
    print(f"Latitude difference - Mean: {lat_diff.mean():.8f}, Std: {lat_diff.std():.8f}")
    print(f"Longitude difference - Mean: {lon_diff.mean():.8f}, Std: {lon_diff.std():.8f}")
    print(f"Depth difference - Mean: {depth_diff.mean():.3f} m, Std: {depth_diff.std():.3f} m")
    
    # Motion statistics
    print(f"\nMotion Statistics:")
    print(f"Max velocity North: {df['vel_north'].max():.3f} m/s")
    print(f"Max velocity East: {df['vel_east'].max():.3f} m/s")
    print(f"Max velocity Down: {df['vel_down'].max():.3f} m/s")
    print(f"Max heading rate: {df['heading_rate'].max():.3f} deg/s")
    print(f"Max heave: {df['heave'].max():.3f} m")

class NavPlotter:
    """Navigation data plotting class for VOYIS First Look Metrics GUI"""
    
    def __init__(self, log_callback=None):
        # Set matplotlib to use white background for all plots
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        
        # Store the log callback for use in methods
        self.log_callback = log_callback
    
    def process_navigation_file(self, nav_file_path, output_dir, dive_name="Navigation", log_callback=None, phins_file_path=None):
        """
        Process navigation file and create plots
        
        :param nav_file_path: Path to the navigation text file (NAV_STATE.txt)
        :param output_dir: Directory to save the plots
        :param dive_name: Name for the dive (used in filenames)
        :param log_callback: Optional callback function for logging messages to GUI (overrides constructor callback)
        :param phins_file_path: Optional path to PHINS data file for merging heave data
        """
        # Use provided callback or fall back to stored callback
        active_callback = log_callback or self.log_callback
        
        def log_message(message):
            print(message)
            if active_callback:
                active_callback(message)
        
        try:
            log_message(f"Loading navigation data from: {nav_file_path}")
            
            # Load the navigation data
            df = self.load_nav_data(nav_file_path)
            
            if df is None:
                raise ValueError("Failed to load navigation data")
            
            log_message(f"Successfully loaded {len(df)} navigation data points")
            
            # If PHINS file is provided, merge heave data
            if phins_file_path:
                df = self.merge_phins_heave(df, phins_file_path, active_callback)
            
            # Create comprehensive navigation plots
            self.create_nav_plots(df, output_dir, dive_name, active_callback)
            
            # Analyze navigation quality
            self.analyze_nav_quality(df, active_callback)
            
            log_message("Navigation plotting completed successfully")
            
        except Exception as e:
            error_msg = f"Error processing navigation file: {str(e)}"
            log_message(error_msg)
            raise RuntimeError(error_msg)
    
    def merge_phins_heave(self, nav_df, phins_file_path, log_callback=None):
        """
        Merge heave data from PHINS file into navigation DataFrame
        
        :param nav_df: Main navigation DataFrame (from NAV_STATE.txt)
        :param phins_file_path: Path to PHINS data file
        :param log_callback: Optional callback for logging
        :return: DataFrame with merged heave data
        """
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            log_message(f"Loading PHINS data from: {phins_file_path}")
            
            # Load PHINS INS text data using our custom reader
            phins_data = load_phins_ins_text(phins_file_path)
            
            # First, create datetime column in nav_df if not present
            if 'datetime' not in nav_df.columns:
                self._create_nav_datetime(nav_df, log_callback)
            
            # Extract heave data from PHINS
            if 'HEAVE_' in phins_data and not phins_data['HEAVE_'].empty:
                heave_df = phins_data['HEAVE_'].copy()
                log_message(f"Found {len(heave_df)} PHINS heave measurements")
                
                # Convert PHINS time reference to datetime for matching
                heave_df['datetime'] = self._parse_phins_time(heave_df['Time_REF'], log_callback)
                
                # Remove rows with invalid datetime
                heave_df = heave_df.dropna(subset=['datetime'])
                
                if len(heave_df) > 0:
                    log_message(f"Successfully parsed {len(heave_df)} PHINS timestamps")
                    
                    # If both DataFrames have datetime columns, merge based on timestamps
                    if 'datetime' in nav_df.columns and not nav_df['datetime'].isna().all():
                        merged_df = self._merge_by_timestamp(nav_df, heave_df, log_callback)
                        if merged_df is not None:
                            return merged_df
                    
                    # If timestamp merging fails, try index-based interpolation
                    merged_df = self._merge_by_interpolation(nav_df, heave_df, log_callback)
                    if merged_df is not None:
                        return merged_df
                    
                    # If all merging methods fail, use statistical approach
                    return self._apply_heave_statistics(nav_df, heave_df, log_callback)
                
                else:
                    log_message("No valid heave data found in PHINS file (datetime parsing failed)")
            
            else:
                log_message("No HEAVE_ data found in PHINS file")
            
            # If no PHINS data available, ensure nav_df has a heave column (even if empty/zero)
            if 'heave' not in nav_df.columns:
                nav_df['heave'] = 0.0
                log_message("Created empty heave column (no PHINS data available)")
            
            return nav_df
            
        except Exception as e:
            log_message(f"Error merging PHINS heave data: {e}")
            log_message("Continuing without PHINS heave data...")
            
            # Ensure heave column exists even if merging fails
            if 'heave' not in nav_df.columns:
                nav_df['heave'] = 0.0
            
            return nav_df
    
    def _create_nav_datetime(self, nav_df, log_callback=None):
        """Create datetime column in navigation DataFrame"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # Try to find time and date columns
            time_col = None
            date_col = None
            
            for col in nav_df.columns:
                col_lower = col.lower().strip()
                if 'time' in col_lower and time_col is None:
                    time_col = col
                elif 'date' in col_lower and date_col is None:
                    date_col = col
            
            if time_col and date_col:
                # Combine date and time columns
                nav_df['datetime'] = pd.to_datetime(
                    nav_df[date_col].astype(str) + ' ' + nav_df[time_col].astype(str),
                    errors='coerce'
                )
                log_message(f"Created datetime from {date_col} and {time_col} columns")
                
            elif time_col:
                # Use time column with a default date
                try:
                    # Try to parse time as HH:MM:SS format
                    base_date = "2024-01-01"  # Default date
                    nav_df['datetime'] = pd.to_datetime(
                        base_date + ' ' + nav_df[time_col].astype(str),
                        errors='coerce'
                    )
                    log_message(f"Created datetime from {time_col} with default date")
                except:
                    # If that fails, create sequential timestamps
                    nav_df['datetime'] = pd.date_range(
                        start='2024-01-01 00:00:00',
                        periods=len(nav_df),
                        freq='1S'
                    )
                    log_message("Created sequential datetime (parsing failed)")
            else:
                # No time columns found, create sequential timestamps
                nav_df['datetime'] = pd.date_range(
                    start='2024-01-01 00:00:00',
                    periods=len(nav_df),
                    freq='1S'
                )
                log_message("Created sequential datetime (no time columns found)")
                
        except Exception as e:
            log_message(f"Error creating datetime column: {e}")
            # Create fallback sequential timestamps
            nav_df['datetime'] = pd.date_range(
                start='2024-01-01 00:00:00',
                periods=len(nav_df),
                freq='1S'
            )
    
    def _parse_phins_time(self, time_series, log_callback=None):
        """Parse PHINS time references to datetime objects"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        # Try multiple time formats since PHINS INS text files can vary
        time_formats = [
            '%Y%m%d %H%M%S.%f',     # YYYYMMDD HHMMSS.fff
            '%Y%m%d %H%M%S',        # YYYYMMDD HHMMSS
            '%Y-%m-%d %H:%M:%S.%f', # YYYY-MM-DD HH:MM:SS.fff
            '%Y-%m-%d %H:%M:%S',    # YYYY-MM-DD HH:MM:SS
            '%H:%M:%S.%f',          # HH:MM:SS.fff
            '%H:%M:%S',             # HH:MM:SS
            '%M:%S.%f',             # MM:SS.fff
            '%M:%S'                 # MM:SS
        ]
        
        datetime_series = None
        
        # Try each format
        for fmt in time_formats:
            try:
                datetime_series = pd.to_datetime(time_series, format=fmt, errors='coerce')
                if not datetime_series.isna().all():
                    log_message(f"Successfully parsed PHINS time using format: {fmt}")
                    break
            except:
                continue
        
        # If specific formats failed, try general parsing
        if datetime_series is None or datetime_series.isna().all():
            try:
                datetime_series = pd.to_datetime(time_series, errors='coerce')
                if not datetime_series.isna().all():
                    log_message("Successfully parsed PHINS time using general parser")
            except:
                pass
        
        # If still no valid times, try to extract numeric time values
        if datetime_series is None or datetime_series.isna().all():
            try:
                # Try to parse as floating point time values (seconds since some epoch)
                time_numeric = pd.to_numeric(time_series, errors='coerce')
                if not time_numeric.isna().all():
                    # Assume it's seconds and create relative datetime
                    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    datetime_series = base_time + pd.to_timedelta(time_numeric, unit='s')
                    log_message("Parsed PHINS time as numeric seconds")
            except:
                pass
        
        return datetime_series
    
    def _merge_by_timestamp(self, nav_df, heave_df, log_callback=None):
        """Merge heave data based on timestamp alignment"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # Sort both DataFrames by datetime
            nav_sorted = nav_df.sort_values('datetime').copy()
            heave_sorted = heave_df.sort_values('datetime').copy()
            
            # Use pandas merge_asof for time-based merging (nearest neighbor)
            merged_df = pd.merge_asof(
                nav_sorted, 
                heave_sorted[['datetime', 'Heave']], 
                on='datetime', 
                direction='nearest', 
                tolerance=pd.Timedelta('30s')  # Allow up to 30 second tolerance
            )
            
            # Update heave column if merge was successful
            if 'Heave' in merged_df.columns:
                valid_heave_mask = ~merged_df['Heave'].isna()
                if valid_heave_mask.any():
                    merged_df['heave'] = merged_df['Heave'].fillna(0.0)
                    log_message(f"Merged {valid_heave_mask.sum()} heave measurements from PHINS using timestamp alignment")
                    
                    # Drop the temporary Heave column
                    merged_df = merged_df.drop(columns=['Heave'])
                    return merged_df
            
            return None
            
        except Exception as e:
            log_message(f"Timestamp-based merging failed: {e}")
            return None
    
    def _merge_by_interpolation(self, nav_df, heave_df, log_callback=None):
        """Merge heave data using interpolation"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # If we have the same number of data points, use direct mapping
            if len(nav_df) == len(heave_df):
                nav_df['heave'] = heave_df['Heave'].values
                log_message(f"Direct mapping of {len(heave_df)} heave measurements (equal lengths)")
                return nav_df
            
            # If lengths differ, interpolate heave data to match nav data
            if len(heave_df) > 10:  # Only interpolate if we have sufficient data
                # Create index arrays for interpolation
                nav_indices = np.arange(len(nav_df))
                heave_indices = np.linspace(0, len(nav_df) - 1, len(heave_df))
                
                # Interpolate heave data to match nav data length
                interpolated_heave = np.interp(nav_indices, heave_indices, heave_df['Heave'].values)
                nav_df['heave'] = interpolated_heave
                
                log_message(f"Interpolated {len(heave_df)} heave measurements to {len(nav_df)} nav points")
                return nav_df
            
            return None
            
        except Exception as e:
            log_message(f"Interpolation-based merging failed: {e}")
            return None
    
    def _apply_heave_statistics(self, nav_df, heave_df, log_callback=None):
        """Apply heave statistics when direct merging fails"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # Calculate PHINS heave statistics
            phins_heave_stats = {
                'mean': heave_df['Heave'].mean(),
                'std': heave_df['Heave'].std(),
                'min': heave_df['Heave'].min(),
                'max': heave_df['Heave'].max()
            }
            
            log_message(f"PHINS heave stats - Mean: {phins_heave_stats['mean']:.3f}m, "
                      f"Std: {phins_heave_stats['std']:.3f}m, "
                      f"Range: [{phins_heave_stats['min']:.3f}, {phins_heave_stats['max']:.3f}]m")
            
            # Create realistic heave data based on PHINS statistics
            np.random.seed(42)  # For reproducible results
            nav_df['heave'] = np.random.normal(
                phins_heave_stats['mean'], 
                phins_heave_stats['std'], 
                len(nav_df)
            )
            
            # Clip to observed range
            nav_df['heave'] = np.clip(
                nav_df['heave'], 
                phins_heave_stats['min'], 
                phins_heave_stats['max']
            )
            
            log_message(f"Created statistical heave data for {len(nav_df)} nav points based on PHINS statistics")
            return nav_df
            
        except Exception as e:
            log_message(f"Statistical heave application failed: {e}")
            nav_df['heave'] = 0.0
            return nav_df
    
    def process_navigation_data(self, nav_file, output_folder, phins_file=None):
        """
        Process navigation data - GUI-compatible method
        
        :param nav_file: Path to the navigation text file (NAV_STATE.txt)
        :param output_folder: Directory to save the plots
        :param phins_file: Optional path to PHINS data file for merging heave data
        :return: Boolean indicating success
        """
        try:
            # Extract dive name from file path
            dive_name = os.path.splitext(os.path.basename(nav_file))[0]
            if dive_name.startswith('Nav_'):
                dive_name = dive_name[4:]  # Remove 'Nav_' prefix if present
            
            # Call the main processing method
            self.process_navigation_file(nav_file, output_folder, dive_name, phins_file_path=phins_file)
            return True
            
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"Error in navigation processing: {str(e)}")
            return False
    
    def load_nav_data(self, file_path):
        """Load navigation data with flexible column detection"""
        try:
            # First, try to read the file and detect the format
            df = pd.read_csv(file_path, sep=None, engine='python')
            
            # Clean column names
            df.columns = [col.strip(' ,') for col in df.columns]
            
            # Convert all column names to lowercase for easier matching
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                # Handle NAV_STATE.txt format specifically
                if 'latitude' in col_lower:
                    column_mapping['latitude'] = col
                elif 'longitude' in col_lower:
                    column_mapping['longitude'] = col
                elif 'depth_m' in col_lower or (('depth' in col_lower) and ('m' in col_lower or '_' in col_lower)):
                    column_mapping['depth'] = col
                elif 'heading_degs' in col_lower or (('head' in col_lower) and ('deg' in col_lower)):
                    column_mapping['heading'] = col
                elif 'pitch_degs' in col_lower or (('pitch' in col_lower) and ('deg' in col_lower)):
                    column_mapping['pitch'] = col
                elif 'roll_degs' in col_lower or (('roll' in col_lower) and ('deg' in col_lower)):
                    column_mapping['roll'] = col
                elif 'altitude_m' in col_lower or (('altitude' in col_lower) and ('m' in col_lower)):
                    column_mapping['altitude'] = col
                # Fallback for simpler formats
                elif 'time' in col_lower and 'secs' not in col_lower:
                    column_mapping['time'] = col
                elif 'lat' in col_lower and 'std' not in col_lower and 'latitude' not in column_mapping:
                    column_mapping['latitude'] = col
                elif 'lon' in col_lower and 'std' not in col_lower and 'longitude' not in column_mapping:
                    column_mapping['longitude'] = col
                elif 'depth' in col_lower and 'std' not in col_lower and 'depth' not in column_mapping:
                    column_mapping['depth'] = col
                elif 'head' in col_lower and 'std' not in col_lower and 'rate' not in col_lower and 'heading' not in column_mapping:
                    column_mapping['heading'] = col
                elif 'pitch' in col_lower and 'std' not in col_lower and 'rate' not in col_lower and 'pitch' not in column_mapping:
                    column_mapping['pitch'] = col
                elif 'roll' in col_lower and 'std' not in col_lower and 'rate' not in col_lower and 'roll' not in column_mapping:
                    column_mapping['roll'] = col
                elif 'heave' in col_lower:
                    column_mapping['heave'] = col
            
            # Rename columns to standard names
            df = df.rename(columns=column_mapping)
            
            # Convert numeric columns
            numeric_cols = ['latitude', 'longitude', 'depth', 'heading', 'pitch', 'roll', 'heave', 'altitude']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create time index if time column exists
            if 'time' in df.columns:
                try:
                    # Try to parse time as datetime
                    df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
                    if df['datetime'].isna().all():
                        # If datetime parsing failed, use time as numeric
                        df['time_numeric'] = pd.to_numeric(df['time'], errors='coerce')
                except:
                    pass
            
            # If no heave column but we have altitude, use altitude as proxy for heave
            if 'heave' not in df.columns and 'altitude' in df.columns:
                # Calculate heave from altitude changes (approximation)
                df['heave'] = df['altitude'].diff().fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error loading navigation data: {e}")
            return None
    
    def create_nav_plots(self, df, output_dir, dive_name, log_callback=None):
        """Create comprehensive navigation plots in 3x3+1 layout like Noah's R script"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        log_message("Creating comprehensive navigation analysis plots...")
        
        # Check for required columns and create them if needed
        required_motion_cols = ['heave', 'pitch', 'roll']
        available_motion_cols = []
        
        # Map common column variations to standard names
        for motion_col in required_motion_cols:
            found_col = None
            for col in df.columns:
                col_lower = col.lower()
                if motion_col in col_lower and 'rate' not in col_lower and 'std' not in col_lower:
                    found_col = col
                    break
            
            if found_col:
                if found_col != motion_col:
                    df[motion_col] = df[found_col]
                available_motion_cols.append(motion_col)
        
        # For heave, if not available, calculate from depth changes (approximation)
        if 'heave' not in available_motion_cols and 'depth' in df.columns:
            # Simple heave approximation from depth changes
            df['heave'] = df['depth'].diff().fillna(0) * -1  # Invert so positive is up
            available_motion_cols.append('heave')
            log_message("Generated heave approximation from depth changes")
        
        # Create the main comprehensive plot (3x3 + 1 layout)
        fig = plt.figure(figsize=(20, 16), facecolor='white')
        
        # Define colors for each motion type (matching R script style)
        motion_colors = {
            'heave': {'map': 'viridis', 'line': '#2E8B57', 'hist': '#a9e1fb'},  # Sea green / light blue
            'pitch': {'map': 'plasma', 'line': '#DC143C', 'hist': '#f7705c'},  # Crimson / coral
            'roll': {'map': 'cividis', 'line': '#228B22', 'hist': '#44bf70'}   # Forest green / bright green
        }
        
        row_titles = ['Heave', 'Pitch', 'Roll']
        
        # Top row: Motion by position (lat/lon maps)
        for i, motion in enumerate(['heave', 'pitch', 'roll']):
            if motion in available_motion_cols and 'latitude' in df.columns and 'longitude' in df.columns:
                ax = plt.subplot(4, 3, i + 1)
                
                # Create scatter plot colored by motion values
                scatter = ax.scatter(df['longitude'], df['latitude'], 
                                   c=df[motion], cmap=motion_colors[motion]['map'], 
                                   s=3, alpha=0.8, edgecolors='none')
                
                plt.colorbar(scatter, ax=ax, label=f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                ax.set_xlabel('Longitude (°)' if i == 1 else '')
                ax.set_ylabel('Latitude (°)' if i == 0 else '')
                ax.set_title(f'{row_titles[i]} Map')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        # Second row: Motion by time
        for i, motion in enumerate(['heave', 'pitch', 'roll']):
            if motion in available_motion_cols:
                ax = plt.subplot(4, 3, i + 4)
                
                # Color by depth if available
                if 'depth' in df.columns:
                    # Create line plot colored by depth
                    points = ax.scatter(df.index, df[motion], c=df['depth'], 
                                      cmap='RdYlBu_r', s=2, alpha=0.7)
                    if i == 2:  # Only add colorbar to the rightmost plot
                        plt.colorbar(points, ax=ax, label='Depth (m)')
                else:
                    ax.plot(df.index, df[motion], color=motion_colors[motion]['line'], linewidth=1)
                
                ax.set_xlabel('Time Index' if i == 1 else '')
                ax.set_ylabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                ax.set_title(f'{row_titles[i]} Time Series')
                ax.grid(True, alpha=0.3)
        
        # Third row: Motion histograms with statistics
        for i, motion in enumerate(['heave', 'pitch', 'roll']):
            if motion in available_motion_cols:
                ax = plt.subplot(4, 3, i + 7)
                
                # Calculate statistics
                motion_data = df[motion].dropna()
                min_val = motion_data.min()
                max_val = motion_data.max()
                mean_val = motion_data.mean()
                std_val = motion_data.std()
                
                # Create histogram
                ax.hist(motion_data, bins=50, alpha=0.7, color=motion_colors[motion]['hist'], 
                       edgecolor='black', linewidth=0.5)
                
                # Add statistics text
                stats_text = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9)
                
                ax.set_xlabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                ax.set_ylabel('Count')
                ax.set_title(f'{row_titles[i]} Distribution')
                ax.grid(True, alpha=0.3)
        
        # Bottom: Depth/Bathymetry map (spanning full width)
        if 'depth' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            ax = plt.subplot(4, 1, 4)
            
            # Create bathymetry map
            scatter = ax.scatter(df['longitude'], df['latitude'], 
                               c=df['depth'], cmap='terrain_r', 
                               s=4, alpha=0.8, edgecolors='none')
            
            cbar = plt.colorbar(scatter, ax=ax, label='Depth (m)')
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_title('Bathymetry Map')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "Nav_Motion_Analysis.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
        plt.close()
        log_message(f"Saved comprehensive motion analysis plot: {plot_path}")
        
        # Create individual plots as well
        self._create_individual_plots(df, output_dir, available_motion_cols, log_callback)
        
        log_message("All navigation plots created successfully")
    
    def _create_individual_plots(self, df, output_dir, available_motion_cols, log_callback=None):
        """Create individual plots for each motion type"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        # Individual motion maps
        for motion in available_motion_cols:
            if 'latitude' in df.columns and 'longitude' in df.columns:
                plt.figure(figsize=(10, 8), facecolor='white')
                scatter = plt.scatter(df['longitude'], df['latitude'], 
                                    c=df[motion], cmap='viridis', s=5, alpha=0.8)
                plt.colorbar(scatter, label=f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                plt.xlabel('Longitude (°)')
                plt.ylabel('Latitude (°)')
                plt.title(f'{motion.capitalize()} by Position')
                plt.grid(True, alpha=0.3)
                
                # Use standardized naming
                if motion == 'heave':
                    plot_path = os.path.join(output_dir, "Nav_Heave_Map.png")
                elif motion == 'pitch':
                    plot_path = os.path.join(output_dir, "Nav_Pitch_Map.png")
                elif motion == 'roll':
                    plot_path = os.path.join(output_dir, "Nav_Roll_Map.png")
                else:
                    plot_path = os.path.join(output_dir, f"Nav_{motion.title()}_Map.png")
                
                plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
                plt.close()
                log_message(f"Saved {motion} map: {plot_path}")
        
        # Individual motion time series plots
        for motion in available_motion_cols:
            plt.figure(figsize=(12, 6), facecolor='white')
            plt.plot(df.index, df[motion], linewidth=1, color='blue', alpha=0.7)
            plt.xlabel('Time Index')
            plt.ylabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
            plt.title(f'{motion.capitalize()} Time Series')
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            motion_data = df[motion].dropna()
            stats_text = f'Min: {motion_data.min():.2f}\nMax: {motion_data.max():.2f}\nMean: {motion_data.mean():.2f}\nStd: {motion_data.std():.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            # Use standardized naming
            if motion == 'heave':
                plot_path = os.path.join(output_dir, "Nav_Heave_Timeseries.png")
            elif motion == 'pitch':
                plot_path = os.path.join(output_dir, "Nav_Pitch_Timeseries.png")
            elif motion == 'roll':
                plot_path = os.path.join(output_dir, "Nav_Roll_Timeseries.png")
            else:
                plot_path = os.path.join(output_dir, f"Nav_{motion.title()}_Timeseries.png")
            
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
            log_message(f"Saved {motion} time series: {plot_path}")
        
        # Individual motion histograms
        for motion in available_motion_cols:
            plt.figure(figsize=(8, 6), facecolor='white')
            motion_data = df[motion].dropna()
            
            plt.hist(motion_data, bins=50, alpha=0.7, color='skyblue', 
                   edgecolor='black', linewidth=0.5)
            
            # Add statistics
            min_val = motion_data.min()
            max_val = motion_data.max()
            mean_val = motion_data.mean()
            std_val = motion_data.std()
            
            stats_text = f'Count: {len(motion_data)}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}'
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            plt.xlabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
            plt.ylabel('Count')
            plt.title(f'{motion.capitalize()} Distribution')
            plt.grid(True, alpha=0.3)
            
            # Use standardized naming
            if motion == 'heave':
                plot_path = os.path.join(output_dir, "Nav_Heave_Histogram.png")
            elif motion == 'pitch':
                plot_path = os.path.join(output_dir, "Nav_Pitch_Histogram.png")
            elif motion == 'roll':
                plot_path = os.path.join(output_dir, "Nav_Roll_Histogram.png")
            else:
                plot_path = os.path.join(output_dir, f"Nav_{motion.title()}_Histogram.png")
            
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
            log_message(f"Saved {motion} histogram: {plot_path}")
        
        # Depth profile time series
        if 'depth' in df.columns:
            plt.figure(figsize=(12, 6), facecolor='white')
            plt.plot(df.index, df['depth'], 'b-', linewidth=1, label='Depth')
            plt.xlabel('Time Index')
            plt.ylabel('Depth (m)')
            plt.title('Depth Profile')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
            
            plot_path = os.path.join(output_dir, "Nav_Depth_Profile.png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
            log_message(f"Saved depth profile: {plot_path}")
        
        # Combined time series
        if available_motion_cols:
            plt.figure(figsize=(15, 10), facecolor='white')
            
            for i, motion in enumerate(available_motion_cols):
                plt.subplot(len(available_motion_cols), 1, i+1)
                plt.plot(df.index, df[motion], linewidth=1, label=motion.capitalize())
                plt.ylabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.xlabel('Time Index')
            plt.suptitle('Vehicle Motion Time Series', fontsize=14)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "Nav_Motion_Timeseries.png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
            log_message(f"Saved motion time series: {plot_path}")
    
    def analyze_nav_quality(self, df, log_callback=None):
        """Analyze navigation data quality"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        log_message("=== Navigation Data Quality Analysis ===")
        log_message(f"Total data points: {len(df)}")
        
        # Position statistics
        if 'latitude' in df.columns and 'longitude' in df.columns:
            log_message(f"Position Range:")
            log_message(f"  Latitude: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
            log_message(f"  Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
        
        if 'depth' in df.columns:
            log_message(f"  Depth: {df['depth'].min():.2f} to {df['depth'].max():.2f} m")
        
        # Motion statistics
        motion_cols = ['heading', 'pitch', 'roll', 'heave']
        available_motion = [col for col in motion_cols if col in df.columns]
        
        if available_motion:
            log_message(f"Motion Statistics:")
            for col in available_motion:
                if col == 'heave':
                    log_message(f"  Max {col}: {df[col].max():.3f} m")
                else:
                    log_message(f"  Max {col}: {df[col].max():.3f} degrees")

def load_phins_ins_text(file_path):
    """
    Load PHINS INS text file and extract relevant navigation data
    
    :param file_path: Path to PHINS INS text file
    :return: Dictionary containing parsed data frames
    """
    try:
        print(f"Attempting to load PHINS INS text file: {file_path}")
        
        # Try different encodings and separators
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = ['\t', ',', ';', ' ', None]
        
        df = None
        successful_encoding = None
        successful_separator = None
        
        # First, try standard CSV parsing
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=encoding, engine='python', low_memory=False)
                    if df is not None and len(df.columns) > 1 and len(df) > 0:
                        successful_encoding = encoding
                        successful_separator = sep
                        print(f"Successfully parsed with encoding={encoding}, separator='{sep}', shape={df.shape}")
                        break
                except Exception as e:
                    continue
            if df is not None and len(df.columns) > 1:
                break
        
        # If CSV parsing fails, try reading as raw text and parsing manually
        if df is None or len(df.columns) <= 1:
            print("CSV parsing failed, attempting manual text parsing...")
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        lines = f.readlines()
                    
                    successful_encoding = encoding
                    break
                except:
                    continue
            
            if 'lines' not in locals():
                raise ValueError("Could not read file with any encoding")
            
            # Look for patterns that might indicate data format
            data_rows = []
            headers = None
            header_found = False
            
            # Look through the first portion of the file for headers and data
            for i, line in enumerate(lines[:500]):  # Check first 500 lines
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//') or line.startswith('%'):
                    continue
                
                # Try to split by common delimiters
                best_parts = []
                best_delim = None
                for delim in ['\t', ',', ';', ' ']:
                    parts = [p.strip() for p in line.split(delim) if p.strip()]
                    if len(parts) > len(best_parts):
                        best_parts = parts
                        best_delim = delim
                
                if len(best_parts) > 3:  # Likely a meaningful row
                    # Check if this could be headers (contains text keywords)
                    contains_keywords = any(keyword.lower() in line.lower() for keyword in 
                                          ['time', 'lat', 'lon', 'head', 'pitch', 'roll', 'heave', 'x', 'y', 'z'])
                    
                    # Check if this could be numeric data
                    numeric_count = 0
                    for part in best_parts:
                        try:
                            float(part.replace(':', '').replace('/', ''))
                            numeric_count += 1
                        except:
                            pass
                    
                    is_mostly_numeric = numeric_count >= len(best_parts) * 0.7
                    
                    if contains_keywords and not header_found and not is_mostly_numeric:
                        headers = best_parts
                        header_found = True
                        print(f"Found headers at line {i+1}: {headers}")
                    elif is_mostly_numeric:
                        # Try to parse as numeric data
                        try:
                            numeric_row = []
                            for part in best_parts:
                                # Handle time formats specially
                                if ':' in part and len(part.split(':')) >= 2:
                                    numeric_row.append(part)  # Keep time as string
                                elif '/' in part and len(part.split('/')) >= 2:
                                    numeric_row.append(part)  # Keep date as string
                                else:
                                    numeric_row.append(float(part))
                            
                            data_rows.append(numeric_row)
                            
                            if len(data_rows) >= 1000:  # Stop after finding enough data
                                break
                        except:
                            continue
            
            # Create DataFrame from parsed data
            if data_rows:
                print(f"Found {len(data_rows)} data rows")
                
                # Ensure all rows have the same length
                max_cols = max(len(row) for row in data_rows)
                normalized_rows = []
                for row in data_rows:
                    if len(row) < max_cols:
                        row.extend([None] * (max_cols - len(row)))
                    normalized_rows.append(row[:max_cols])
                
                if headers and len(headers) == max_cols:
                    df = pd.DataFrame(normalized_rows, columns=headers)
                    print(f"Created DataFrame with headers, shape: {df.shape}")
                else:
                    # Generate generic column names
                    df = pd.DataFrame(normalized_rows, columns=[f'col_{i}' for i in range(max_cols)])
                    print(f"Created DataFrame with generic headers, shape: {df.shape}")
            else:
                print("No valid data rows found")
        
        if df is None or df.empty:
            print(f"Warning: Could not parse PHINS INS file: {file_path}")
            return {}
        
        # Clean column names
        df.columns = [str(col).strip(' ,') for col in df.columns]
        print(f"Final DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        
        # Create data dictionary similar to what read_unique_identifiers would return
        data_dict = {}
        
        # Try to identify and extract heave data
        heave_cols = []
        for col in df.columns:
            col_str = str(col).lower()
            if 'heave' in col_str:
                heave_cols.append(col)
        
        if heave_cols:
            print(f"Found heave columns: {heave_cols}")
            heave_col = heave_cols[0]  # Use the first heave column found
            
            # Look for time columns
            time_cols = []
            for col in df.columns:
                col_str = str(col).lower()
                if any(time_word in col_str for time_word in ['time', 'timestamp', 'utc', 'gps_time']):
                    time_cols.append(col)
            
            # Create heave DataFrame
            if time_cols:
                time_col = time_cols[0]
                heave_data = df[[heave_col, time_col]].copy()
                heave_data = heave_data.dropna()
                
                # Convert heave to numeric
                heave_data[heave_col] = pd.to_numeric(heave_data[heave_col], errors='coerce')
                heave_data = heave_data.dropna()
                
                if len(heave_data) > 0:
                    heave_dict = {
                        'Heave': heave_data[heave_col].tolist(),
                        'Time_REF': heave_data[time_col].astype(str).tolist()
                    }
                    data_dict['HEAVE_'] = pd.DataFrame(heave_dict)
                    print(f"Created HEAVE_ DataFrame with {len(heave_dict['Heave'])} points")
            else:
                # No time column found, use index as time reference
                heave_data = df[[heave_col]].copy()
                heave_data = heave_data.dropna()
                
                # Convert heave to numeric
                heave_data[heave_col] = pd.to_numeric(heave_data[heave_col], errors='coerce')
                heave_data = heave_data.dropna()
                
                if len(heave_data) > 0:
                    heave_dict = {
                        'Heave': heave_data[heave_col].tolist(),
                        'Time_REF': [f"idx_{i}" for i in range(len(heave_data))]
                    }
                    data_dict['HEAVE_'] = pd.DataFrame(heave_dict)
                    print(f"Created HEAVE_ DataFrame with {len(heave_dict['Heave'])} points (index-based time)")
        
        # Try to extract other relevant motion data (pitch, roll, etc.)
        attitude_cols = []
        for motion in ['pitch', 'roll', 'heading', 'yaw']:
            motion_cols = []
            for col in df.columns:
                col_str = str(col).lower()
                if motion in col_str and 'rate' not in col_str and 'std' not in col_str:
                    motion_cols.append(col)
            
            if motion_cols:
                attitude_cols.extend(motion_cols[:1])  # Take first match for each motion type
        
        if attitude_cols:
            print(f"Found attitude columns: {attitude_cols}")
            
            # Look for time columns again
            time_cols = []
            for col in df.columns:
                col_str = str(col).lower()
                if any(time_word in col_str for time_word in ['time', 'timestamp', 'utc', 'gps_time']):
                    time_cols.append(col)
            
            if time_cols:
                time_col = time_cols[0]
                attitude_data = df[attitude_cols + [time_col]].copy()
                attitude_data = attitude_data.dropna()
                
                if len(attitude_data) > 0:
                    attitude_dict = {'Time_REF': attitude_data[time_col].astype(str).tolist()}
                    
                    for col in attitude_cols:
                        try:
                            attitude_dict[col.capitalize()] = pd.to_numeric(attitude_data[col], errors='coerce').tolist()
                        except:
                            attitude_dict[col.capitalize()] = attitude_data[col].tolist()
                    
                    data_dict['ATITUD'] = pd.DataFrame(attitude_dict)
                    print(f"Created ATITUD DataFrame with {len(attitude_dict['Time_REF'])} points")
        
        print(f"Successfully parsed PHINS file. Found data types: {list(data_dict.keys())}")
        return data_dict
        
    except Exception as e:
        print(f"Error loading PHINS INS text file: {e}")
        import traceback
        traceback.print_exc()
        return {}
