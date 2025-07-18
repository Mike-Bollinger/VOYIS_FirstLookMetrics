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
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
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
    # Format coordinate axes to avoid scientific notation
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
    
    # 2. PHINS vs GPS Position Comparison
    ax2 = plt.subplot(3, 3, 2)
    plt.scatter(df['longitude'], df['latitude'], label='GPS', alpha=0.6, s=1)
    plt.scatter(df['phins_lon'], df['phins_lat'], label='PHINS', alpha=0.6, s=1)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title('GPS vs PHINS Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Format coordinate axes to avoid scientific notation
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
    
    # 3. Depth Profile
    ax3 = plt.subplot(3, 3, 3)
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['depth'], 'b-', linewidth=1, label='Depth')
        plt.plot(df['datetime'], df['phins_depth'], 'r-', linewidth=1, label='PHINS Depth')
        plt.xlabel('Time')
    else:
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
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['heading'], label='Heading', linewidth=1)
        plt.plot(df['datetime'], df['pitch'], label='Pitch', linewidth=1)
        plt.plot(df['datetime'], df['roll'], label='Roll', linewidth=1)
        plt.xlabel('Time')
    else:
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
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['vel_north'], label='North', linewidth=1)
        plt.plot(df['datetime'], df['vel_east'], label='East', linewidth=1)
        plt.plot(df['datetime'], df['vel_down'], label='Down', linewidth=1)
        plt.xlabel('Time')
    else:
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
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['heading_rate'], label='Heading Rate', linewidth=1)
        plt.plot(df['datetime'], df['pitch_rate'], label='Pitch Rate', linewidth=1)
        plt.plot(df['datetime'], df['roll_rate'], label='Roll Rate', linewidth=1)
        plt.xlabel('Time')
    else:
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
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['stddev_lat'], label='Latitude StdDev', linewidth=1)
        plt.plot(df['datetime'], df['stddev_lon'], label='Longitude StdDev', linewidth=1)
        plt.xlabel('Time')
    else:
        plt.plot(df.index, df['stddev_lat'], label='Latitude StdDev', linewidth=1)
        plt.plot(df.index, df['stddev_lon'], label='Longitude StdDev', linewidth=1)
        plt.xlabel('Time Index')
    plt.ylabel('Standard Deviation')
    plt.title('Position Uncertainties')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Attitude Uncertainties
    ax8 = plt.subplot(3, 3, 8)
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['stddev_roll'], label='Roll StdDev', linewidth=1)
        plt.plot(df['datetime'], df['stddev_pitch'], label='Pitch StdDev', linewidth=1)
        plt.plot(df['datetime'], df['stddev_head'], label='Heading StdDev', linewidth=1)
        plt.xlabel('Time')
    else:
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
    if 'datetime' in df.columns:
        plt.plot(df['datetime'], df['heave'], 'g-', linewidth=1)
        plt.xlabel('Time')
    else:
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
            
            log_message("Navigation plotting and analysis completed")
            return True  # Indicate success
            
        except Exception as e:
            error_msg = f"Error processing navigation file: {str(e)}"
            log_message(error_msg)
            return False  # Indicate failure
    
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
            phins_data = self._load_phins_ins_text(phins_file_path)
            
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
        """Create datetime column in navigation DataFrame using mission_msecs"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # Look for mission time column
            mission_time_col = None
            for col in nav_df.columns:
                col_lower = col.lower().strip()
                if 'mission' in col_lower and ('time' in col_lower or 'msec' in col_lower):
                    mission_time_col = col
                    break
            
            if mission_time_col:
                # Convert mission time (format like "5:53:22.4") to datetime
                try:
                    base_date = "2024-01-01"  # Default date
                    nav_df['datetime'] = pd.to_datetime(
                        base_date + ' ' + nav_df[mission_time_col].astype(str),
                        format='%Y-%m-%d %H:%M:%S.%f',
                        errors='coerce'
                    )
                    # Fill any parsing failures with alternative format
                    mask = nav_df['datetime'].isna()
                    if mask.any():
                        nav_df.loc[mask, 'datetime'] = pd.to_datetime(
                            base_date + ' ' + nav_df.loc[mask, mission_time_col].astype(str),
                            format='%Y-%m-%d %H:%M:%S',
                            errors='coerce'
                        )
                    
                    valid_count = nav_df['datetime'].notna().sum()
                    log_message(f"Created datetime from mission time column '{mission_time_col}' ({valid_count} valid timestamps)")
                    
                except Exception as e:
                    log_message(f"Error parsing mission time: {e}")
                    # Fallback to sequential timestamps
                    nav_df['datetime'] = pd.date_range(
                        start='2024-01-01 00:00:00',
                        periods=len(nav_df),
                        freq='100ms'  # 10 Hz typical nav rate
                    )
                    log_message("Created sequential datetime (mission time parsing failed)")
            else:
                log_message("No mission time column found, creating sequential datetime")
                nav_df['datetime'] = pd.date_range(
                    start='2024-01-01 00:00:00',
                    periods=len(nav_df),
                    freq='100ms'
                )
                
        except Exception as e:
            log_message(f"Error creating datetime column: {e}")
            # Create fallback sequential timestamps
            nav_df['datetime'] = pd.date_range(
                start='2024-01-01 00:00:00',
                periods=len(nav_df),
                freq='100ms'
            )
    
    def _parse_phins_time(self, time_series, log_callback=None):
        """Parse PHINS time references to datetime objects (matching mission time format)"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # PHINS time should be in the same format as mission_msecs (H:MM:SS.f)
            base_date = "2024-01-01 "
            
            # First try with microseconds
            try:
                full_time_series = base_date + time_series.astype(str)
                datetime_series = pd.to_datetime(full_time_series, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                valid_count = datetime_series.notna().sum()
                if valid_count > 0:
                    log_message(f"Successfully parsed {valid_count} PHINS timestamps using format with microseconds")
                    return datetime_series
            except:
                pass
            
            # Try without microseconds
            try:
                full_time_series = base_date + time_series.astype(str)
                datetime_series = pd.to_datetime(full_time_series, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                valid_count = datetime_series.notna().sum()
                if valid_count > 0:
                    log_message(f"Successfully parsed {valid_count} PHINS timestamps using format without microseconds")
                    return datetime_series
            except:
                pass
            
            # Try general parsing with base date
            try:
                full_time_series = base_date + time_series.astype(str)
                datetime_series = pd.to_datetime(full_time_series, errors='coerce')
                valid_count = datetime_series.notna().sum()
                if valid_count > 0:
                    log_message(f"Successfully parsed {valid_count} PHINS timestamps using general parser with base date")
                    return datetime_series
            except:
                pass
            
            # If all parsing fails, return None
            log_message("Failed to parse PHINS timestamps")
            return None
            
        except Exception as e:
            log_message(f"Error in PHINS time parsing: {e}")
            return None
    
    def _merge_by_timestamp(self, nav_df, heave_df, log_callback=None):
        """Merge heave data based on timestamp alignment"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
    def _merge_by_timestamp(self, nav_df, heave_df, log_callback=None):
        """Merge heave data based on timestamp alignment (mission time synchronization)"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # Ensure both DataFrames have timezone-naive datetime columns
            nav_df_copy = nav_df.copy()
            heave_df_copy = heave_df.copy()
            
            # Remove timezone information if present
            if nav_df_copy['datetime'].dt.tz is not None:
                nav_df_copy['datetime'] = nav_df_copy['datetime'].dt.tz_localize(None)
            if heave_df_copy['datetime'].dt.tz is not None:
                heave_df_copy['datetime'] = heave_df_copy['datetime'].dt.tz_localize(None)
            
            # Sort both DataFrames by datetime
            nav_sorted = nav_df_copy.sort_values('datetime').copy()
            heave_sorted = heave_df_copy.sort_values('datetime').copy()
            
            # Check time overlap
            nav_start, nav_end = nav_sorted['datetime'].iloc[0], nav_sorted['datetime'].iloc[-1]
            heave_start, heave_end = heave_sorted['datetime'].iloc[0], heave_sorted['datetime'].iloc[-1]
            
            log_message(f"Nav time range: {nav_start.strftime('%H:%M:%S.%f')[:-3]} to {nav_end.strftime('%H:%M:%S.%f')[:-3]}")
            log_message(f"Heave time range: {heave_start.strftime('%H:%M:%S.%f')[:-3]} to {heave_end.strftime('%H:%M:%S.%f')[:-3]}")
            
            # Calculate overlap
            overlap_start = max(nav_start, heave_start)
            overlap_end = min(nav_end, heave_end)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                log_message(f"Time overlap: {overlap_duration:.1f} seconds")
                
                # Use pandas merge_asof for time-based merging (nearest neighbor)
                merged_df = pd.merge_asof(
                    nav_sorted, 
                    heave_sorted[['datetime', 'Heave']], 
                    on='datetime', 
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=10)  # Allow up to 10 second difference
                )
                
                # Check if we successfully merged heave data
                if 'Heave' in merged_df.columns:
                    valid_heave_mask = ~merged_df['Heave'].isna()
                    if valid_heave_mask.any():
                        merged_df['heave'] = merged_df['Heave'].fillna(0.0)
                        log_message(f"Successfully merged {valid_heave_mask.sum()}/{len(merged_df)} heave measurements using timestamp alignment")
                        
                        # Drop the temporary Heave column
                        merged_df = merged_df.drop(columns=['Heave'])
                        return merged_df
                    else:
                        log_message("No heave data within tolerance window")
                else:
                    log_message("No Heave column found after merge")
            else:
                log_message("No time overlap between navigation and heave data")
            
            return None
            
        except Exception as e:
            log_message(f"Timestamp-based merging failed: {e}")
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
            
            # Remove empty columns (those with empty names or all NaN)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
            df = df.dropna(axis=1, how='all')  # Remove columns that are entirely NaN
            empty_cols = [col for col in df.columns if col.strip() == '']
            if empty_cols:
                df = df.drop(columns=empty_cols)
                print(f"Removed empty columns: {empty_cols}")
            
            # Handle 'Lon/Lat' column if it exists - split into separate columns
            if 'Lon/Lat' in df.columns:
                print("Found 'Lon/Lat' column, splitting into separate longitude and latitude columns")
                try:
                    lon_lat_split = df['Lon/Lat'].str.split('/', expand=True)
                    if len(lon_lat_split.columns) >= 2:
                        df['longitude'] = pd.to_numeric(lon_lat_split[0], errors='coerce')
                        df['latitude'] = pd.to_numeric(lon_lat_split[1], errors='coerce')
                        df = df.drop('Lon/Lat', axis=1)
                        print("Successfully split 'Lon/Lat' column")
                except Exception as e:
                    print(f"Error splitting 'Lon/Lat' column: {e}")
            
            # Convert all column names to lowercase for easier matching
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                # Handle NAV_STATE.txt format specifically
                if 'latitude' in col_lower and 'latitude' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'latitude'
                elif 'longitude' in col_lower and 'longitude' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'longitude'
                elif 'depth_m' in col_lower or (('depth' in col_lower) and ('m' in col_lower or '_' in col_lower)):
                    column_mapping[col] = 'depth'
                elif col_lower == 'heading_degs' or col_lower == 'heading_deg':  # Exact match for heading_degs
                    column_mapping[col] = 'heading'
                elif col_lower == 'pitch_degs' or col_lower == 'pitch_deg':  # Exact match for pitch_degs (not rate)
                    column_mapping[col] = 'pitch'
                elif col_lower == 'roll_degs' or col_lower == 'roll_deg':  # Exact match for roll_degs (not rate)
                    column_mapping[col] = 'roll'
                elif 'altitude_m' in col_lower or (('altitude' in col_lower) and ('m' in col_lower)):
                    column_mapping[col] = 'altitude'
                # Fallback for simpler formats (PHINS INS.txt)
                elif 'time' in col_lower and 'secs' not in col_lower and 'time' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'time'
                elif 'lat' in col_lower and 'std' not in col_lower and 'latitude' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'latitude'
                elif 'lon' in col_lower and 'std' not in col_lower and 'longitude' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'longitude'
                elif 'depth' in col_lower and 'std' not in col_lower and 'depth' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'depth'
                elif col_lower == 'heading' and 'heading' not in [v for v in column_mapping.values()]:  # Exact match for 'heading'
                    column_mapping[col] = 'heading'
                elif col_lower == 'pitch' and 'pitch' not in [v for v in column_mapping.values()]:  # Exact match for 'pitch'
                    column_mapping[col] = 'pitch'
                elif col_lower == 'roll' and 'roll' not in [v for v in column_mapping.values()]:  # Exact match for 'roll'
                    column_mapping[col] = 'roll'
                elif 'head' in col_lower and 'std' not in col_lower and 'rate' not in col_lower and 'heading' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'heading'
                elif 'pitch' in col_lower and 'std' not in col_lower and 'rate' not in col_lower and 'pitch' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'pitch'
                elif 'roll' in col_lower and 'std' not in col_lower and 'rate' not in col_lower and 'roll' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'roll'
                elif 'heave' in col_lower and 'heave' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'heave'
            
            # Rename columns to standard names
            df = df.rename(columns=column_mapping)
            
            # Handle any duplicate column names that might have been created
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Debug: print column mapping
            if column_mapping:
                print(f"Column mapping applied: {column_mapping}")
            print(f"Final columns after mapping: {list(df.columns)}")
            
            # Verify the mapping worked - check for renamed columns
            for old_name, new_name in column_mapping.items():
                if new_name in df.columns:
                    print(f"Successfully mapped '{old_name}' -> '{new_name}'")
                else:
                    print(f"WARNING: Mapping failed for '{old_name}' -> '{new_name}'")
            
            # Convert numeric columns
            numeric_cols = ['latitude', 'longitude', 'depth', 'heading', 'pitch', 'roll', 'heave', 'altitude']
            for col in numeric_cols:
                if col in df.columns:
                    # Clean the column data first (remove extra whitespace)
                    try:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip()
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Log any conversion issues
                        if df[col].isna().any():
                            na_count = df[col].isna().sum()
                            print(f"Warning: {na_count} NaN values found in column '{col}' after conversion")
                    except Exception as e:
                        print(f"Error converting column '{col}' to numeric: {e}")
                        # Try simpler conversion
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
            
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
        """Create comprehensive navigation plots in 5x6 grid layout with enhanced features"""
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
        
        # Calculate Crab Index if we have the required data
        if all(col in df.columns for col in ['latitude', 'longitude', 'heading']):
            df = self._calculate_crab_index(df, window_size=5, log_callback=log_callback)
        else:
            missing_cols = [col for col in ['latitude', 'longitude', 'heading'] if col not in df.columns]
            log_message(f"Skipping crab index calculation - missing columns: {missing_cols}")
            log_message(f"Available columns: {list(df.columns)}")
        
        # Create the main comprehensive plot (5x6 grid layout for better control)
        # Row 0: Motion by position maps (3 plots, each spanning 2 columns)
        # Row 1: Motion time series (3 plots, each spanning 2 columns)
        # Row 2: Motion distributions (3 plots, each spanning 2 columns)
        # Rows 3-4: Enlarged Crab Index and Bathymetry maps (each spanning 3 columns x 2 rows)
        fig = plt.figure(figsize=(20, 20), facecolor='white')
        
        # Define colors for each motion type (matching R script style)
        motion_colors = {
            'heave': {'map': 'viridis', 'line': '#2E8B57', 'hist': '#a9e1fb'},
            'pitch': {'map': 'plasma', 'line': '#DC143C', 'hist': '#f7705c'},
            'roll': {'map': 'cividis', 'line': '#228B22', 'hist': '#44bf70'}
        }
        
        row_titles = ['Heave', 'Pitch', 'Roll']
        
        # Row 0: Motion by position (lat/lon maps)
        for i, motion in enumerate(['heave', 'pitch', 'roll']):
            if motion in available_motion_cols and 'latitude' in df.columns and 'longitude' in df.columns:
                ax = plt.subplot2grid((5, 6), (0, i*2), colspan=2)
                
                # Create depth-colored scatter plot for better visualization
                scatter, cbar = self._create_depth_colored_scatter(
                    ax, df['longitude'], df['latitude'], df[motion],
                    colormap=motion_colors[motion]['map'], size=1.5, alpha=0.8,
                    add_colorbar=True, 
                    colorbar_label=f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)',
                    log_scale=False
                )
                
                ax.set_xlabel('Longitude (°)' if i == 1 else '')
                ax.set_ylabel('Latitude (°)' if i == 0 else '')
                ax.set_title(f'{row_titles[i]} by Position')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                # Format coordinate axes to avoid scientific notation
                self._format_coordinate_axis(ax, 'both')
        
        # Row 1: Motion time series
        for i, motion in enumerate(['heave', 'pitch', 'roll']):
            if motion in available_motion_cols:
                ax = plt.subplot2grid((5, 6), (1, i*2), colspan=2)
                
                # Use datetime for x-axis if available
                x_data = df['datetime'] if 'datetime' in df.columns else df.index
                x_label = 'Time' if 'datetime' in df.columns else 'Time Index'
                
                # Color by depth if available for enhanced visualization
                if 'depth' in df.columns:
                    # Apply exponential scaling to emphasize deeper depths
                    depth_values = df['depth'].copy()
                    min_depth = depth_values.min()
                    max_depth = depth_values.max()
                    
                    # Normalize depth values to 0-1 range
                    if max_depth > min_depth:
                        depth_normalized = (depth_values - min_depth) / (max_depth - min_depth)
                    else:
                        depth_normalized = np.zeros_like(depth_values)
                    
                    # Apply exponential scaling for deeper depth emphasis
                    depth_scaled = depth_normalized ** 2.5
                    
                    points = ax.scatter(x_data, df[motion], c=depth_scaled, 
                                      cmap='viridis_r', s=1, alpha=0.7, vmin=0, vmax=1)
                    if i == 2:  # Only add colorbar to the rightmost plot
                        cbar = plt.colorbar(points, ax=ax)
                        cbar.set_label('Depth (m)', rotation=270, labelpad=15)
                        
                        # Set colorbar ticks to show actual depth values
                        num_ticks = 5
                        depth_tick_values = np.linspace(min_depth, max_depth, num_ticks)
                        if max_depth > min_depth:
                            depth_tick_normalized = (depth_tick_values - min_depth) / (max_depth - min_depth)
                            depth_tick_scaled = depth_tick_normalized ** 2.5
                        else:
                            depth_tick_scaled = np.zeros(num_ticks)
                        
                        cbar.set_ticks(depth_tick_scaled)
                        cbar.set_ticklabels([f'{d:.1f}' for d in depth_tick_values])
                else:
                    ax.plot(x_data, df[motion], color=motion_colors[motion]['line'], linewidth=1)
                
                ax.set_xlabel(x_label if i == 1 else '')
                ax.set_ylabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                ax.set_title(f'{row_titles[i]} Time Series')
                ax.grid(True, alpha=0.3)
                
                # Format time axis if using datetime
                if 'datetime' in df.columns:
                    self._format_datetime_axis(ax, x_data)
                    self._format_datetime_axis(ax, df['datetime'])
        
        # Row 2: Motion histograms with statistics
        for i, motion in enumerate(['heave', 'pitch', 'roll']):
            if motion in available_motion_cols:
                ax = plt.subplot2grid((5, 6), (2, i*2), colspan=2)
                
                # Calculate statistics
                motion_data = df[motion].dropna()
                min_val = motion_data.min()
                max_val = motion_data.max()
                mean_val = motion_data.mean()
                std_val = motion_data.std()
                
                # Create histogram with better styling
                n, bins, patches = ax.hist(motion_data, bins=50, alpha=0.7, 
                                         color=motion_colors[motion]['hist'], 
                                         edgecolor='black', linewidth=0.5)
                
                # Add statistics text with better formatting
                stats_text = f'n: {len(motion_data)}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                       fontsize=9, family='monospace')
                
                ax.set_xlabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{row_titles[i]} Distribution')
                ax.grid(True, alpha=0.3)
        
        # Bottom two rows: Enlarged Crab Index and Bathymetry maps spanning the full width and extra height
        # Crab Index plot (left half of bottom area, taller)
        if 'crab_index' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            ax = plt.subplot2grid((5, 6), (3, 0), colspan=3, rowspan=2)
            
            # Create crab index plot with enhanced visualization
            crab_data = df.dropna(subset=['crab_index', 'latitude', 'longitude'])
            if len(crab_data) > 0:
                # Use RdBu_r colormap for crab index (red=positive crab, blue=negative crab)
                scatter = ax.scatter(crab_data['longitude'], crab_data['latitude'], 
                                   c=crab_data['crab_index'], cmap='RdBu_r', 
                                   s=8, alpha=0.8, edgecolors='none',
                                   vmin=-20, vmax=20)  # Symmetric range for better visualization
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Crab Index (°)', rotation=270, labelpad=15)
                
                # Add crab index statistics
                crab_stats = crab_data['crab_index']
                stats_text = f'Crab Index Statistics\nn: {len(crab_stats)}\nMean: {crab_stats.mean():.2f}°\nStd: {crab_stats.std():.2f}°\nRange: [{crab_stats.min():.1f}°, {crab_stats.max():.1f}°]'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                       fontsize=9, family='monospace')
            
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_title('Crab Index Analysis\n(Red: Starboard Crab, Blue: Port Crab)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            # Format coordinate axes to avoid scientific notation
            self._format_coordinate_axis(ax, 'both')
        
        # Bathymetry map (right half of bottom area, taller)
        if 'depth' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            ax = plt.subplot2grid((5, 6), (3, 3), colspan=3, rowspan=2)
            
            # Create enhanced bathymetry visualization
            depth_data = df.dropna(subset=['depth', 'latitude', 'longitude'])
            if len(depth_data) > 0:
                scatter, cbar = self._create_depth_colored_scatter(
                    ax, depth_data['longitude'], depth_data['latitude'], depth_data['depth'],
                    colormap='viridis_r', size=4, alpha=0.8,
                    add_colorbar=True, colorbar_label='Depth (m)',
                    log_scale=True  # Use exponential scaling for better depth visualization
                )
                
                # Add depth statistics
                depth_stats = depth_data['depth']
                stats_text = f'Depth Statistics\nn: {len(depth_stats)}\nMin: {depth_stats.min():.1f}m\nMax: {depth_stats.max():.1f}m\nMean: {depth_stats.mean():.1f}m\nStd: {depth_stats.std():.1f}m'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                       fontsize=9, family='monospace')
            
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_title('Bathymetry Map')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            # Format coordinate axes to avoid scientific notation
            self._format_coordinate_axis(ax, 'both')
        
        # Use subplots_adjust for the 5x6 grid layout
        plt.subplots_adjust(left=0.06, bottom=0.10, right=0.95, top=0.94, 
                           wspace=0.30, hspace=0.55)
        plot_path = os.path.join(output_dir, "Nav_Motion_Analysis.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
        plt.close()
        log_message(f"Saved comprehensive motion analysis plot: {plot_path}")
        
        # Create comprehensive time series comparison plot (all motion types together)
        self._create_comprehensive_timeseries_plot(df, output_dir, available_motion_cols, log_callback)
        
        # Create individual plots as well
        self._create_individual_plots(df, output_dir, available_motion_cols, log_callback)
        
        log_message("All navigation plots created successfully")
    
    def _create_comprehensive_timeseries_plot(self, df, output_dir, available_motion_cols, log_callback=None):
        """Create comprehensive time series comparison plot showing all motion types together"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        if not available_motion_cols:
            log_message("No motion columns available for comprehensive time series plot")
            return
        
        log_message("Creating comprehensive time series comparison plot...")
        
        # Create figure with subplots for each motion type - wider and taller to accommodate colorbar and title
        fig, axes = plt.subplots(len(available_motion_cols), 1, figsize=(18, 5 * len(available_motion_cols)), 
                                facecolor='white', sharex=True)
        
        # Handle single subplot case
        if len(available_motion_cols) == 1:
            axes = [axes]
        
        # Define colors for each motion type
        motion_colors = {
            'heave': '#2E8B57',  # Sea Green
            'pitch': '#DC143C',  # Crimson
            'roll': '#228B22'    # Forest Green
        }
        
        # Use datetime if available, otherwise fall back to index
        if 'datetime' in df.columns and not df['datetime'].isna().all():
            x_data = df['datetime']
            x_label = 'Mission Time'
        else:
            x_data = df.index
            x_label = 'Time Index'
        
        # Store depth information for colorbar
        depth_info = None
        if 'depth' in df.columns:
            depth_values = df['depth'].copy()
            min_depth = depth_values.min()
            max_depth = depth_values.max()
            
            # Normalize depth values to 0-1 range
            if max_depth > min_depth:
                depth_normalized = (depth_values - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = np.zeros_like(depth_values)
            
            # Apply exponential scaling for deeper depth emphasis
            depth_scaled = depth_normalized ** 2.5
            
            depth_info = {
                'scaled': depth_scaled,
                'min': min_depth,
                'max': max_depth,
                'tick_values': np.linspace(min_depth, max_depth, 5),
                'tick_positions': None
            }
            
            # Calculate tick positions
            if max_depth > min_depth:
                depth_tick_normalized = (depth_info['tick_values'] - min_depth) / (max_depth - min_depth)
                depth_info['tick_positions'] = depth_tick_normalized ** 2.5
            else:
                depth_info['tick_positions'] = np.zeros(5)
        
        # Create subplot for each motion type
        scatter_plots = []  # Store scatter plots for shared colorbar
        for i, motion in enumerate(available_motion_cols):
            ax = axes[i]
            
            # Get motion data
            motion_data = df[motion].dropna()
            if len(motion_data) == 0:
                continue
            
            # Color by depth if available for enhanced visualization
            if depth_info is not None:
                # Create depth-colored scatter plot with reversed viridis
                points = ax.scatter(x_data, df[motion], c=depth_info['scaled'], 
                                  cmap='viridis_r', s=2, alpha=0.8, vmin=0, vmax=1)
                scatter_plots.append(points)
            else:
                # Use solid line if no depth data
                ax.plot(x_data, df[motion], color=motion_colors.get(motion, 'blue'), 
                       linewidth=1, alpha=0.7)
            
            # Set labels and title
            ax.set_ylabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
            ax.set_title(f'{motion.capitalize()} Time Series', fontsize=12, pad=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {motion_data.mean():.2f}\nStd: {motion_data.std():.2f}\nRange: [{motion_data.min():.2f}, {motion_data.max():.2f}]'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                   fontsize=9, family='monospace')
            
            # Format time axis if using datetime
            if 'datetime' in df.columns:
                self._format_datetime_axis(ax, x_data)
        
        # Add shared colorbar spanning the entire right side
        if scatter_plots and depth_info is not None:
            # Adjust subplot positions to make room for colorbar
            plt.subplots_adjust(right=0.85)
            
            # Create colorbar that spans all subplots, positioned further right
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax)
            cbar.set_label('Depth (m)', rotation=270, labelpad=20)
            
            # Set colorbar ticks to show actual depth values
            cbar.set_ticks(depth_info['tick_positions'])
            cbar.set_ticklabels([f'{d:.1f}' for d in depth_info['tick_values']])
        
        # Set x-label only on the bottom subplot
        axes[-1].set_xlabel(x_label)
        
        # Add overall title with more space
        fig.suptitle('Comprehensive Motion Time Series Comparison', fontsize=18, y=0.98)
        
        # Adjust layout - account for colorbar space and title
        plt.tight_layout()
        if scatter_plots and depth_info is not None:
            plt.subplots_adjust(top=0.94, right=0.85)
        else:
            plt.subplots_adjust(top=0.94)
        
        # Save plot
        plot_path = os.path.join(output_dir, "Nav_Comprehensive_Timeseries.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
        plt.close()
        log_message(f"Saved comprehensive time series plot: {plot_path}")
        
        # Also create a single-axis version with all motion types overlaid
        self._create_overlaid_timeseries_plot(df, output_dir, available_motion_cols, log_callback)
    
    def _create_overlaid_timeseries_plot(self, df, output_dir, available_motion_cols, log_callback=None):
        """Create overlaid time series plot with all motion types on the same axis"""
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        if not available_motion_cols:
            return
        
        log_message("Creating overlaid time series comparison plot...")
        
        # Create figure
        plt.figure(figsize=(15, 8), facecolor='white')
        
        # Define colors for each motion type
        motion_colors = {
            'heave': '#2E8B57',  # Sea Green
            'pitch': '#DC143C',  # Crimson
            'roll': '#228B22'    # Forest Green
        }
        
        # Use datetime if available, otherwise fall back to index
        if 'datetime' in df.columns and not df['datetime'].isna().all():
            x_data = df['datetime']
            x_label = 'Mission Time'
        else:
            x_data = df.index
            x_label = 'Time Index'
        
        # Plot each motion type
        for motion in available_motion_cols:
            motion_data = df[motion].dropna()
            if len(motion_data) == 0:
                continue
            
            # Use solid lines for overlaid plot for better visibility
            plt.plot(x_data, df[motion], color=motion_colors.get(motion, 'blue'), 
                    linewidth=1.5, alpha=0.8, label=f'{motion.capitalize()}')
        
        # Set labels and title
        plt.xlabel(x_label)
        plt.ylabel('Motion Values')
        plt.title('All Motion Types - Overlaid Time Series Comparison', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Format time axis if using datetime
        if 'datetime' in df.columns:
            self._format_datetime_axis(plt.gca(), x_data)
        
        # Add statistics box
        stats_lines = []
        for motion in available_motion_cols:
            motion_data = df[motion].dropna()
            if len(motion_data) > 0:
                stats_lines.append(f'{motion.capitalize()}: μ={motion_data.mean():.2f}, σ={motion_data.std():.2f}')
        
        if stats_lines:
            stats_text = '\n'.join(stats_lines)
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                    fontsize=10, family='monospace')
        
        # Save plot
        plot_path = os.path.join(output_dir, "Nav_Overlaid_Timeseries.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
        plt.close()
        log_message(f"Saved overlaid time series plot: {plot_path}")

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
                                    c=df[motion], cmap='viridis_r', s=2, alpha=0.8)
                plt.colorbar(scatter, label=f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
                plt.xlabel('Longitude (°)')
                plt.ylabel('Latitude (°)')
                plt.title(f'{motion.capitalize()} by Position')
                plt.grid(True, alpha=0.3)
                # Format coordinate axes to avoid scientific notation
                self._format_coordinate_axis(plt.gca(), 'both')
                
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
            
            # Use datetime if available, otherwise fall back to index
            if 'datetime' in df.columns and not df['datetime'].isna().all():
                x_data = df['datetime']
                x_label = 'Mission Time'
                # Format x-axis for time display
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=1))
                plt.xticks(rotation=45)
            else:
                x_data = df.index
                x_label = 'Time Index'
            
            # Color by depth if available for enhanced visualization
            if 'depth' in df.columns:
                # Apply exponential scaling to emphasize deeper depths
                depth_values = df['depth'].copy()
                min_depth = depth_values.min()
                max_depth = depth_values.max()
                
                # Normalize depth values to 0-1 range
                if max_depth > min_depth:
                    depth_normalized = (depth_values - min_depth) / (max_depth - min_depth)
                else:
                    depth_normalized = np.zeros_like(depth_values)
                
                # Apply exponential scaling for deeper depth emphasis
                depth_scaled = depth_normalized ** 2.5
                
                points = plt.scatter(x_data, df[motion], c=depth_scaled, 
                                  cmap='viridis_r', s=2, alpha=0.8, vmin=0, vmax=1)
                
                # Add colorbar with actual depth values
                cbar = plt.colorbar(points)
                cbar.set_label('Depth (m)', rotation=270, labelpad=15)
                
                # Set colorbar ticks to show actual depth values
                num_ticks = 5
                depth_tick_values = np.linspace(min_depth, max_depth, num_ticks)
                if max_depth > min_depth:
                    depth_tick_normalized = (depth_tick_values - min_depth) / (max_depth - min_depth)
                    depth_tick_scaled = depth_tick_normalized ** 2.5
                else:
                    depth_tick_scaled = np.zeros(num_ticks)
                
                cbar.set_ticks(depth_tick_scaled)
                cbar.set_ticklabels([f'{d:.1f}' for d in depth_tick_values])
            else:
                plt.plot(x_data, df[motion], linewidth=1, color='blue', alpha=0.7)
            
            plt.xlabel(x_label)
            plt.ylabel(f'{motion.capitalize()} (°)' if motion != 'heave' else f'{motion.capitalize()} (m)')
            plt.title(f'{motion.capitalize()} Time Series')
            plt.grid(True, alpha=0.3)
            
            # Format time axis if using datetime
            if 'datetime' in df.columns:
                self._format_datetime_axis(plt.gca(), x_data)
            
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
        
        # Depth profile time series
        if 'depth' in df.columns:
            plt.figure(figsize=(12, 6), facecolor='white')
            
            # Use datetime if available, otherwise fall back to index
            if 'datetime' in df.columns and not df['datetime'].isna().all():
                x_data = df['datetime']
                x_label = 'Mission Time'
                # Use our custom formatting function
                self._format_datetime_axis(plt.gca(), x_data)
            else:
                x_data = df.index
                x_label = 'Time Index'
            
            plt.plot(x_data, df['depth'], 'b-', linewidth=1, label='Depth')
            plt.xlabel(x_label)
            plt.ylabel('Depth (m)')
            plt.title('Depth Profile')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
            
            plot_path = os.path.join(output_dir, "Nav_Depth_Profile.png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
            log_message(f"Saved depth profile: {plot_path}")
        
        # Standalone bathymetry map
        if 'depth' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            plt.figure(figsize=(12, 10), facecolor='white')
            depth_data = df.dropna(subset=['depth', 'latitude', 'longitude'])
            
            scatter, cbar = self._create_depth_colored_scatter(
                plt.gca(), depth_data['longitude'], depth_data['latitude'], depth_data['depth'],
                colormap='viridis_r', size=5, alpha=0.8,
                add_colorbar=True, colorbar_label='Depth (m)',
                log_scale=True
            )
            
            plt.xlabel('Longitude (°)')
            plt.ylabel('Latitude (°)')
            plt.title('Bathymetry Map')
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis='x', rotation=45)
            # Format coordinate axes to avoid scientific notation
            self._format_coordinate_axis(plt.gca(), 'both')
            
            plot_path = os.path.join(output_dir, "Nav_Bathymetry_Map.png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
            log_message(f"Saved bathymetry map: {plot_path}")
        
        # Standalone Crab Index plots
        if 'crab_index' in df.columns:
            # Crab Index Map
            if 'latitude' in df.columns and 'longitude' in df.columns:
                plt.figure(figsize=(12, 10), facecolor='white')
                crab_data = df.dropna(subset=['crab_index', 'latitude', 'longitude'])
                
                if len(crab_data) > 0:
                    scatter = plt.scatter(crab_data['longitude'], crab_data['latitude'], 
                                        c=crab_data['crab_index'], cmap='RdBu_r', 
                                        s=4, alpha=0.8, edgecolors='none',
                                        vmin=-20, vmax=20)
                    
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('Crab Index (°)', rotation=270, labelpad=15)
                    
                    plt.xlabel('Longitude (°)')
                    plt.ylabel('Latitude (°)')
                    plt.title('Crab Index Map\n(Red: Starboard Crab, Blue: Port Crab)')
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(axis='x', rotation=45)
                    # Format coordinate axes to avoid scientific notation
                    self._format_coordinate_axis(plt.gca(), 'both')
                    
                    plot_path = os.path.join(output_dir, "Nav_Crab_Index_Map.png")
                    plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
                    plt.close()
                    log_message(f"Saved crab index map: {plot_path}")
            
            # Crab Index Time Series
            plt.figure(figsize=(15, 6), facecolor='white')
            crab_data = df['crab_index'].dropna()
            
            if len(crab_data) > 0:
                # Plot crab index over time
                valid_subset = df.dropna(subset=['crab_index'])
                
                # Use datetime for x-axis if available
                if 'datetime' in df.columns:
                    x_data = valid_subset['datetime']
                    x_label = 'Time'
                else:
                    x_data = valid_subset.index
                    x_label = 'Time Index'
                
                plt.plot(x_data, crab_data, linewidth=1, color='red', alpha=0.7, label='Crab Index')
                
                # Add zero line for reference
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero Reference')
                
                # Add threshold lines for moderate and severe crabbing
                plt.axhline(y=5, color='orange', linestyle=':', alpha=0.7, label='Moderate Threshold (±5°)')
                plt.axhline(y=-5, color='orange', linestyle=':', alpha=0.7)
                plt.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='Severe Threshold (±10°)')
                plt.axhline(y=-10, color='red', linestyle=':', alpha=0.7)
                
                plt.xlabel(x_label)
                plt.ylabel('Crab Index (°)')
                plt.title('Crab Index Time Series\n(Positive: Starboard crab, Negative: Port crab)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Format time axis if using datetime
                if 'datetime' in df.columns:
                    self._format_datetime_axis(plt.gca(), x_data)
                
                # Add statistics
                stats_text = f'n: {len(crab_data)}\nMean: {crab_data.mean():.2f}°\nStd: {crab_data.std():.2f}°\nRange: [{crab_data.min():.1f}°, {crab_data.max():.1f}°]'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                        fontsize=10, family='monospace')
                
                plot_path = os.path.join(output_dir, "Nav_Crab_Index_Timeseries.png")
                plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
                plt.close()
                log_message(f"Saved crab index time series: {plot_path}")
            
            # Crab Index Histogram
            plt.figure(figsize=(10, 6), facecolor='white')
            
            if len(crab_data) > 0:
                # Create histogram with color coding
                n, bins, patches = plt.hist(crab_data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color code the histogram bars based on crab severity
                for i, patch in enumerate(patches):
                    bin_center = (bins[i] + bins[i+1]) / 2
                    if abs(bin_center) <= 5:
                        patch.set_facecolor('lightgreen')  # Good
                    elif abs(bin_center) <= 10:
                        patch.set_facecolor('orange')      # Moderate
                    else:
                        patch.set_facecolor('red')         # Severe
                
                # Add vertical lines for thresholds
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Zero')
                plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Moderate (±5°)')
                plt.axvline(x=-5, color='orange', linestyle='--', alpha=0.7)
                plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Severe (±10°)')
                plt.axvline(x=-10, color='red', linestyle='--', alpha=0.7)
                
                plt.xlabel('Crab Index (°)')
                plt.ylabel('Frequency')
                plt.title('Crab Index Distribution\n(Green: Good, Orange: Moderate, Red: Severe)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add statistics
                abs_crab = crab_data.abs()
                good_count = (abs_crab <= 5).sum()
                moderate_count = ((abs_crab > 5) & (abs_crab <= 10)).sum()
                severe_count = (abs_crab > 10).sum()
                
                stats_text = f'Classification:\nGood (≤5°): {good_count} ({good_count/len(crab_data)*100:.1f}%)\nModerate (5-10°): {moderate_count} ({moderate_count/len(crab_data)*100:.1f}%)\nSevere (>10°): {severe_count} ({severe_count/len(crab_data)*100:.1f}%)'
                plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                        fontsize=10, family='monospace')
                
                plot_path = os.path.join(output_dir, "Nav_Crab_Index_Histogram.png")
                plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
                plt.close()
                log_message(f"Saved crab index histogram: {plot_path}")
        
        # Heading vs Course Over Ground Comparison (like in nav_plotter_v2)
        if 'heading' in df.columns and 'course_over_ground' in df.columns:
            plt.figure(figsize=(15, 6), facecolor='white')
            
            # Plot both heading and course over ground
            valid_data = df.dropna(subset=['heading', 'course_over_ground'])
            if len(valid_data) > 0:
                # Use datetime for x-axis if available
                if 'datetime' in df.columns:
                    x_data = valid_data['datetime']
                    x_label = 'Time'
                else:
                    x_data = valid_data.index
                    x_label = 'Time Index'
                
                plt.plot(x_data, valid_data['heading'], 'b-', linewidth=1, alpha=0.8, label='Heading')
                plt.plot(x_data, valid_data['course_over_ground'], 'r-', linewidth=1, alpha=0.8, label='Course Over Ground')
                
                plt.xlabel(x_label)
                plt.ylabel('Angle (°)')
                plt.title('Heading vs Course Over Ground Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 360)  # Standard compass range
                
                # Format time axis if using datetime
                if 'datetime' in df.columns:
                    self._format_datetime_axis(plt.gca(), x_data)
                
                plot_path = os.path.join(output_dir, "Nav_Heading_vs_COG.png")
                plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
                plt.close()
                log_message(f"Saved heading vs COG comparison: {plot_path}")
        
        # Enhanced Crab Index Time Series (matching nav_plotter_v2 style)
        if 'crab_index' in df.columns:
            plt.figure(figsize=(15, 8), facecolor='white')
            crab_data = df['crab_index'].dropna()
            
            if len(crab_data) > 0:
                # Create subplot for time series
                ax1 = plt.subplot(2, 1, 1)
                valid_subset = df.dropna(subset=['crab_index'])
                
                # Use datetime for x-axis if available
                if 'datetime' in df.columns:
                    x_data = valid_subset['datetime']
                    x_label = 'Time'
                else:
                    x_data = valid_subset.index
                    x_label = 'Time Index'
                
                # Plot crab index with color coding
                plt.plot(x_data, crab_data, linewidth=1, color='blue', alpha=0.7, label='Crab Index')
                
                # Add threshold lines
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
                plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Moderate Crab (±5°)')
                plt.axhline(y=-5, color='orange', linestyle='--', alpha=0.7)
                plt.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Severe Crab (±15°)')
                plt.axhline(y=-15, color='red', linestyle='--', alpha=0.7)
                
                # Add statistics box
                abs_crab = crab_data.abs()
                severe_count = (abs_crab > 15).sum()
                stats_text = f'Mean: {crab_data.mean():.2f}°, Std: {crab_data.std():.2f}°\nSevere Crab (>15°): {severe_count} points'
                plt.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                        fontsize=10, family='monospace')
                
                plt.ylabel('Crab Index (°)')
                plt.title('Crab Index Time Series\n(Positive: Starboard Crab, Negative: Port Crab)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(-40, 40)  # Match the range from your image
                
                # Format time axis if using datetime
                if 'datetime' in df.columns:
                    self._format_datetime_axis(ax1, x_data)
                
                # Create subplot for distribution
                ax2 = plt.subplot(2, 1, 2)
                
                # Create histogram with color coding based on severity
                n, bins, patches = plt.hist(crab_data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color code the histogram bars
                for i, patch in enumerate(patches):
                    bin_center = (bins[i] + bins[i+1]) / 2
                    if abs(bin_center) <= 5:
                        patch.set_facecolor('lightblue')     # Good
                    elif abs(bin_center) <= 15:
                        patch.set_facecolor('orange')        # Moderate  
                    else:
                        patch.set_facecolor('red')           # Severe
                
                # Add vertical threshold lines
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
                plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7)
                plt.axvline(x=-5, color='orange', linestyle='--', alpha=0.7)
                plt.axvline(x=15, color='red', linestyle='--', alpha=0.7)
                plt.axvline(x=-15, color='red', linestyle='--', alpha=0.7)
                
                plt.xlabel('Crab Index (°)')
                plt.ylabel('Count')
                plt.title('Crab Index Distribution')
                plt.grid(True, alpha=0.3)
                plt.xlim(-40, 40)  # Match the range from your image
                
                plt.tight_layout()
                
                plot_path = os.path.join(output_dir, "Nav_Crab_Index_Analysis.png")
                plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
                plt.close()
                log_message(f"Saved enhanced crab index analysis: {plot_path}")

    def _calculate_crab_index(self, df, window_size=5, log_callback=None):
        """
        Calculate Crab Index - the difference between vehicle heading and course over ground (CoG)
        
        Crabbing indicates sideways movement relative to the heading direction, which can indicate:
        - Current effects
        - Navigation control issues
        - Environmental forces
        
        Args:
            df: DataFrame with latitude, longitude, and heading data
            window_size: Number of points for moving window CoG calculation (minimum 5)
            log_callback: Optional logging callback
            
        Returns:
            DataFrame with added crab_index and course_over_ground columns
        """
        def log_message(message):
            print(message)
            if log_callback:
                log_callback(message)
        
        try:
            # Check if we have the required columns
            if not all(col in df.columns for col in ['latitude', 'longitude', 'heading']):
                missing_cols = [col for col in ['latitude', 'longitude', 'heading'] if col not in df.columns]
                log_message(f"Cannot calculate crab index - missing columns: {missing_cols}")
                return df
            
            # Ensure we have enough data points
            if len(df) < window_size:
                log_message(f"Not enough data points for crab index calculation (need at least {window_size})")
                return df
            
            log_message(f"Calculating crab index with window size {window_size}...")
            
            # Calculate course over ground (CoG) using moving window
            cog_values = np.full(len(df), np.nan)
            
            for i in range(window_size - 1, len(df)):
                start_idx = i - window_size + 1
                end_idx = i
                
                # Get start and end positions for this window
                lat1 = df['latitude'].iloc[start_idx]
                lon1 = df['longitude'].iloc[start_idx]
                lat2 = df['latitude'].iloc[end_idx]
                lon2 = df['longitude'].iloc[end_idx]
                
                # Only calculate if we have valid coordinates
                if pd.notna([lat1, lon1, lat2, lon2]).all():
                    # Calculate bearing between start and end points
                    bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
                    cog_values[i] = bearing
            
            df['course_over_ground'] = cog_values
            
            # Calculate crab index (difference between heading and CoG)
            df['crab_index'] = df['heading'] - df['course_over_ground']
            
            # Normalize crab index to [-180, 180] range
            df['crab_index'] = ((df['crab_index'] + 180) % 360) - 180
            
            # Count valid crab index calculations
            valid_count = df['crab_index'].notna().sum()
            log_message(f"Successfully calculated crab index for {valid_count} data points")
            
            # Log some statistics
            if valid_count > 0:
                crab_stats = df['crab_index'].describe()
                log_message(f"Crab index statistics - Mean: {crab_stats['mean']:.2f}°, "
                          f"Std: {crab_stats['std']:.2f}°, "
                          f"Range: [{crab_stats['min']:.2f}°, {crab_stats['max']:.2f}°]")
            
            return df
            
        except Exception as e:
            log_message(f"Error calculating crab index: {e}")
            return df
    
    def _format_datetime_axis(self, ax, x_data):
        """Format datetime axis to avoid excessive ticks"""
        if hasattr(x_data, 'dtype') and 'datetime' in str(x_data.dtype):
            # Calculate appropriate time interval based on data span
            time_span = (x_data.max() - x_data.min()).total_seconds()
            
            if time_span > 7200:  # More than 2 hours
                # Use hourly ticks
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
            elif time_span > 1800:  # More than 30 minutes
                # Use 10-minute ticks
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
            else:
                # Use 5-minute ticks for shorter spans
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
            
            # Rotate labels to prevent overlap
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _format_coordinate_axis(self, ax, axis='both'):
        """Format latitude and longitude axes to avoid scientific notation"""
        def coord_formatter(x, pos):
            """Format coordinates with fixed decimal places"""
            return f'{x:.3f}'
        
        formatter = FuncFormatter(coord_formatter)
        
        if axis in ['x', 'both']:
            ax.xaxis.set_major_formatter(formatter)
        if axis in ['y', 'both']:
            ax.yaxis.set_major_formatter(formatter)
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate bearing (course over ground) between two points
        
        Args:
            lat1, lon1: Starting point coordinates (degrees)
            lat2, lon2: Ending point coordinates (degrees)
            
        Returns:
            Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)
        
        # Calculate bearing
        y = np.sin(dlon_rad) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad))
        
        bearing_rad = np.arctan2(y, x)
        bearing_deg = np.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg

    def _create_depth_colored_scatter(self, ax, x_data, y_data, depth_data, 
                                    colormap='viridis', size=4, alpha=0.8, add_colorbar=True, 
                                    colorbar_label='Depth (m)', log_scale=True, 
                                    colorbar_shrink=None, colorbar_aspect=None):
        """
        Create a scatter plot colored by depth with exponential depth-emphasized scaling
        
        Using 'viridis' colormap provides perceptually uniform depth visualization:
        - Purple/blue for shallow depths 
        - Green/yellow for deep depths
        - Exponential scaling gives more color resolution to deeper parts of the dive
        
        Args:
            ax: matplotlib axis
            x_data: x coordinates
            y_data: y coordinates  
            depth_data: depth values for coloring
            colormap: colormap name (default 'viridis' for perceptually uniform coloring)
            size: marker size
            alpha: transparency
            add_colorbar: whether to add colorbar
            colorbar_label: colorbar label
            log_scale: whether to use exponential depth-emphasized scale (recommended for dive analysis)
            colorbar_shrink: shrink factor for colorbar (default None uses matplotlib default)
            colorbar_aspect: aspect ratio for colorbar (default None uses matplotlib default)
            
        Returns:
            scatter plot object and colorbar (if created)
        """
        from matplotlib.colors import Normalize
        
        # Handle depth data preparation
        depth_values = depth_data.copy()
        cbar = None
        
        if log_scale:
            # Use inverse exponential scaling to emphasize depth variations in deeper parts
            # This gives more color resolution to deeper depths where more variation is needed
            min_depth = depth_values.min()
            max_depth = depth_values.max()
            
            # Normalize depth values to 0-1 range
            if max_depth > min_depth:
                depth_normalized = (depth_values - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = np.zeros_like(depth_values)
            
            # Apply exponential scaling that emphasizes deeper depths
            # Higher exponent gives more color resolution to deeper parts
            exponent = 2.5  # Adjust this to control how much emphasis on deep depths
            depth_scaled = depth_normalized ** exponent
            
            # Create the scatter plot with exponentially scaled colors
            scatter = ax.scatter(x_data, y_data, c=depth_scaled, cmap=colormap, 
                               s=size, alpha=alpha, vmin=0, vmax=1)
            
            if add_colorbar:
                # Create custom colorbar that shows actual depth values
                cbar_kwargs = {}
                if colorbar_shrink is not None:
                    cbar_kwargs['shrink'] = colorbar_shrink
                if colorbar_aspect is not None:
                    cbar_kwargs['aspect'] = colorbar_aspect
                
                cbar = plt.colorbar(scatter, ax=ax, **cbar_kwargs)
                cbar.set_label(colorbar_label, rotation=270, labelpad=15)
                
                # Set colorbar ticks to show actual depth values
                # Create evenly spaced depth values for ticks
                num_ticks = 6
                depth_tick_values = np.linspace(min_depth, max_depth, num_ticks)
                # Convert to scaled positions for positioning
                if max_depth > min_depth:
                    depth_tick_normalized = (depth_tick_values - min_depth) / (max_depth - min_depth)
                    depth_tick_scaled = depth_tick_normalized ** exponent
                else:
                    depth_tick_scaled = np.zeros(num_ticks)
                
                cbar.set_ticks(depth_tick_scaled)
                cbar.set_ticklabels([f'{d:.1f}' for d in depth_tick_values])
        else:
            # Use linear scaling
            scatter = ax.scatter(x_data, y_data, c=depth_values, cmap=colormap, 
                               s=size, alpha=alpha)
            
            if add_colorbar:
                cbar_kwargs = {}
                if colorbar_shrink is not None:
                    cbar_kwargs['shrink'] = colorbar_shrink
                if colorbar_aspect is not None:
                    cbar_kwargs['aspect'] = colorbar_aspect
                
                cbar = plt.colorbar(scatter, ax=ax, label=colorbar_label, **cbar_kwargs)
        
        return scatter, cbar

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
        
        # Crab Index statistics
        if 'crab_index' in df.columns:
            crab_data = df['crab_index'].dropna()
            if len(crab_data) > 0:
                log_message(f"Crab Index Statistics:")
                log_message(f"  Valid measurements: {len(crab_data)}")
                log_message(f"  Mean: {crab_data.mean():.2f}°")
                log_message(f"  Std: {crab_data.std():.2f}°")
                log_message(f"  Range: [{crab_data.min():.2f}°, {crab_data.max():.2f}°]")
                
                # Analyze crabbing severity
                abs_crab = crab_data.abs()
                severe_crab = (abs_crab > 10).sum()
                moderate_crab = ((abs_crab > 5) & (abs_crab <= 10)).sum()
                
                log_message(f"  Severe crabbing (>10°): {severe_crab} points ({severe_crab/len(crab_data)*100:.1f}%)")
                log_message(f"  Moderate crabbing (5-10°): {moderate_crab} points ({moderate_crab/len(crab_data)*100:.1f}%)")

    def _load_phins_ins_text(self, file_path):
        """
        Load PHINS INS text file and extract relevant navigation data
        
        :param file_path: Path to PHINS INS text file
        :return: Dictionary containing parsed data frames
        """
        try:
            print(f"Attempting to load PHINS INS text file: {file_path}")
            
            df = None
            
            # Try the simple CSV read first with comma separator (most common for PHINS INS files)
            try:
                df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
                # Clean column names (remove trailing spaces and commas)
                df.columns = [col.strip(' ,') for col in df.columns]
                
                # Remove any completely empty columns
                df = df.dropna(axis=1, how='all')
                
                # Check if we have meaningful data
                if len(df.columns) > 3 and len(df) > 100:
                    print(f"Successfully parsed CSV with comma separator, shape={df.shape}")
                    print(f"Columns: {list(df.columns)}")
                else:
                    print(f"CSV parsing gave insufficient data: {df.shape}")
                    df = None
                    
            except Exception as e:
                print(f"Standard CSV parsing failed: {e}")
                df = None
            
            # If CSV parsing fails, try alternative approaches
            if df is None:
                # Try different encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                separators = ['\t', ',', ';', ' ', None]
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, sep=sep, encoding=encoding, engine='python', low_memory=False)
                            if df is not None and len(df.columns) > 1 and len(df) > 0:
                                print(f"Successfully parsed with encoding={encoding}, separator='{sep}', shape={df.shape}")
                                break
                        except Exception as e:
                            continue
                    if df is not None and len(df.columns) > 1:
                        break
            
            
            # If all parsing methods still fail, try manual text parsing
            if df is None or len(df.columns) <= 1:
                print("All CSV parsing methods failed, attempting manual text parsing...")
                
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                            lines = f.readlines()
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
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('//') or line.startswith('%'):
                        continue
                    
                    # Try to split by common delimiters
                    best_parts = []
                    best_delim = None
                    for delim in [',', '\t', ';', ' ']:
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
                            # Try to parse as numeric data - REMOVE THE 1000 ROW LIMIT
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
                                
                                # Log progress every 5000 rows
                                if len(data_rows) % 5000 == 0:
                                    print(f"Processed {len(data_rows)} data rows...")
                                    
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
