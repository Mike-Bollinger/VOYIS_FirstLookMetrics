"""
Navigation Data Merger
Multi-file navigation data merging system for VOYIS First Look Metrics.

This module provides comprehensive navigation data merging capabilities for combining
data from multiple legacy and modern navigation file formats including:
- PHINS INS files (highest priority)
- NAV_STATE files  
- STATE files
- ADCP files
- *_Veh_Data files
- Other navigation files (lowest priority)

Handles time-based merging and column standardization across different file formats.
"""

import pandas as pd
import numpy as np
import os
import traceback
from .metrics import Metrics


class NavigationDataMerger:
    """
    Comprehensive navigation data merger for VOYIS First Look Metrics.
    
    Merges navigation data from multiple sources in priority order:
    1. PHINS INS files (highest priority)
    2. NAV_STATE files  
    3. STATE files
    4. ADCP files
    5. *_Veh_Data files
    6. Other navigation files (lowest priority)
    
    Handles time-based merging and column standardization across different file formats.
    """
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        
        # Track whether any midnight crossover was detected during processing
        self.midnight_crossover_detected = False
        
        # Initialize altitude filter parameters with sensible defaults
        self.altitude_filter_params = {
            'method': 'combined',
            'z_threshold': 3.0,
            'iqr_multiplier': 2.0,
            'rolling_window': 50,
            'rolling_threshold': 3.0
        }
        
        # Define column mappings for each file type based on actual sample data
        self.column_mappings = {
            'phins_ins': {
                'time_cols': ['time', 'phins_time', 'mission_msecs'],
                'mappings': {
                    # Primary GPS coordinates
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'depth': 'depth',
                    'time': 'time',
                    # PHINS-specific coordinates (often more accurate)
                    'phins_lat': 'phins_latitude',
                    'phins_lon': 'phins_longitude', 
                    'phins_depth': 'phins_depth',
                    'phins_heave': 'heave',  # Will be standardized by _standardize_heave_columns
                    'heave': 'heave',        # Direct mapping, will be standardized
                    'phins_heading': 'heading',
                    'phins_pitch': 'pitch', 
                    'phins_roll': 'roll',
                    'phins_time': 'phins_time'
                }
            },
            'nav_state': {
                'time_cols': ['mission_msecs', 'time'],
                'mappings': {
                    'latitude_deg': 'latitude',
                    'longitude_deg': 'longitude',
                    'depth_m': 'depth',
                    'heading_degs': 'heading',
                    'pitch_degs': 'pitch',
                    'roll_degs': 'roll',
                    'altitude_m': 'altitude',
                    'mission_msecs': 'time'
                }
            },
            'state': {
                'time_cols': ['mission_msecs', 'time'],
                'mappings': {
                    'latitude': 'latitude',
                    'longitude': 'longitude', 
                    'depth': 'depth',
                    'compass_heading': 'heading',
                    'pitch': 'pitch',
                    'roll': 'roll',
                    'heading_rate': 'heading_rate',
                    'pitch_rate': 'pitch_rate',
                    'roll_rate': 'roll_rate',
                    'mission_msecs': 'time'
                }
            },
            'adcp': {
                'time_cols': ['mission_msecs', 'time'],
                'mappings': {
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'depth': 'depth', 
                    'heading': 'heading',
                    'pitch': 'pitch',
                    'roll': 'roll',
                    'altitude': 'altitude',
                    'mission_msecs': 'time'
                }
            },
            'veh_data': {
                'time_cols': ['Time', 'mission_msecs'],
                'mappings': {
                    'Lon/Lat': 'lon_lat_combined',  # Special handling needed
                    'Altitude': 'altitude',
                    'Depth of vehicle': 'depth',
                    'Heading': 'heading', 
                    'Pitch': 'pitch',
                    'Roll': 'roll',
                    'Heading rate': 'heading_rate',
                    'Pitch rate': 'pitch_rate',
                    'Roll rate': 'roll_rate',
                    'Time': 'time'
                }
            },
            'ctd': {
                'time_cols': ['time', 'mission_msecs'],
                'mappings': {
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'depth': 'depth',
                    'time': 'time',
                    'mission_msecs': 'time',
                    'temperature': 'temperature',
                    'conductivity': 'conductivity',
                    'salinity': 'salinity'
                }
            }
        }
        
        # Define required columns for navigation data
        self.required_columns = ['time', 'latitude', 'longitude', 'depth']
        self.optional_columns = ['heading', 'pitch', 'roll', 'heave', 'altitude']
        
        # Default altitude filter parameters
        self.altitude_filter_params = {
            'method': 'combined',
            'z_threshold': 3.0,
            'iqr_multiplier': 2.0,
            'rolling_window': 50,
            'rolling_threshold': 3.0
        }
        
    def log_message(self, message):
        """Log message with optional callback"""
        print(message)
        if self.log_callback:
            self.log_callback(message)
    
    def identify_file_type(self, file_path):
        """Identify the navigation file type based on filename and content"""
        filename = os.path.basename(file_path).lower()
        
        # Check filename patterns first
        if 'phins' in filename and 'ins' in filename:
            return 'phins_ins'
        elif 'nav_state' in filename or filename == 'nav_state.txt':
            return 'nav_state'
        elif filename == 'state.txt' or filename.endswith('_state.txt'):
            return 'state'
        elif 'adcp' in filename:
            return 'adcp'
        elif 'veh' in filename and ('data' in filename or 'nav' in filename):
            return 'veh_data'
        elif 'ctd' in filename:
            return 'ctd'  # CTD files are low priority environmental data
        
        # If filename doesn't match, check column headers for more robust identification
        try:
            # Read first few lines to check headers
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().lower().strip()
                second_line = f.readline().lower().strip()  # Sometimes headers are on second line
                
            combined_header = f"{first_line} {second_line}"
            
            # Check for PHINS INS format (has phins-specific columns)
            if any(col in combined_header for col in ['phins_lat', 'phins_lon', 'phins_depth', 'phins_heave']):
                return 'phins_ins'
            
            # Check for CTD format first (has CTD-specific columns like conductivity, salinity)
            elif any(col in combined_header for col in ['conductivity', 'salinity', 'temperature', 'sound_speed']):
                return 'ctd'
            
            # Check for NAV_STATE format (has specific naming pattern)
            elif any(col in combined_header for col in ['latitude_deg', 'longitude_deg', 'depth_m']):
                return 'nav_state'
            
            # Check for STATE format (has vehicle control columns)
            elif all(col in combined_header for col in ['compass_heading', 'thruster_rpm', 'software_revision']):
                return 'state'
            
            # Check for Veh_Data format (has specific column format)
            elif 'lon/lat' in combined_header and 'depth of vehicle' in combined_header:
                return 'veh_data'
            
            # Check for ADCP format (has altitude and velocity data)
            elif 'altitude' in combined_header and any(col in combined_header for col in ['forward_velocity', 'starboard_velocity']):
                return 'adcp'
            
            # Additional checks for variations
            elif 'mission_msecs' in combined_header and 'latitude' in combined_header:
                # Could be STATE format without full header detection
                if 'heading_rate' in combined_header or 'vehicle_temperature' in combined_header:
                    return 'state'
                # Could be basic navigation format
                else:
                    return 'nav_state'
                    
        except Exception as e:
            self.log_message(f"Warning: Could not analyze file headers for {file_path}: {e}")
        
        # Default fallback
        self.log_message(f"Warning: Could not identify file type for {file_path}, using 'unknown'")
        return 'unknown'
    
    def load_and_standardize_file(self, file_path, file_type=None):
        """Load a navigation file and standardize its columns"""
        if file_type is None:
            file_type = self.identify_file_type(file_path)
        
        self.log_message(f"Loading {file_type} file: {os.path.basename(file_path)}")
        
        try:
            # Load the file based on type
            if file_type == 'phins_ins':
                # For PHINS INS files, use standard CSV loading
                # The complex binary PHINS loading is not needed for your text files
                df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
                
                # Clean column names
                df.columns = [col.strip(' ,') for col in df.columns]
                
                # Remove empty columns
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df.dropna(axis=1, how='all')
            else:
                # Standard CSV loading for other formats
                df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
                
                # Clean column names
                df.columns = [col.strip(' ,') for col in df.columns]
                
                # Remove empty columns
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df.dropna(axis=1, how='all')
            
            if df is None or df.empty:
                self.log_message(f"Warning: No data loaded from {file_path}")
                return None
                
            # Apply file-type specific processing
            df = self._apply_file_specific_processing(df, file_type)
            
            # Standardize columns using mappings
            df = self._standardize_columns(df, file_type)
            
            # Standardize heave columns (handle both 'heave' and 'phins_heave')
            df = self._standardize_heave_columns(df)
            
            # Convert time column to standardized format
            df = self._standardize_time_column(df, file_type)
            
            # Convert numeric columns
            df = self._convert_numeric_columns(df)
            
            # Add source information
            df['source_file'] = os.path.basename(file_path)
            df['source_type'] = file_type
            
            self.log_message(f"Successfully loaded {len(df)} records from {file_type} file")
            self.log_message(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.log_message(f"Error loading file {file_path}: {e}")
            traceback.print_exc()
            return None
    
    def _apply_file_specific_processing(self, df, file_type):
        """Apply file-type specific processing"""
        
        if file_type == 'veh_data':
            # Handle special 'Lon/Lat' column format like "29N53.37270  87W14.82942"
            if 'Lon/Lat' in df.columns:
                self.log_message("Processing Lon/Lat column from Veh_Data format")
                try:
                    # Parse the coordinate format: "29N53.37270  87W14.82942"
                    coord_pattern = r'(\d+)N(\d+\.\d+)\s+(\d+)W(\d+\.\d+)'
                    coord_parts = df['Lon/Lat'].str.extract(coord_pattern)
                    
                    if not coord_parts.empty and not coord_parts.isna().all().all():
                        # Convert to decimal degrees
                        lat_deg = coord_parts[0].astype(float, errors='ignore')
                        lat_min = coord_parts[1].astype(float, errors='ignore')
                        lon_deg = coord_parts[2].astype(float, errors='ignore') 
                        lon_min = coord_parts[3].astype(float, errors='ignore')
                        
                        df['latitude'] = lat_deg + lat_min / 60.0
                        df['longitude'] = -(lon_deg + lon_min / 60.0)  # West is negative
                        
                        valid_coords = (~df['latitude'].isna() & ~df['longitude'].isna()).sum()
                        self.log_message(f"Successfully parsed {valid_coords} coordinates from Lon/Lat column")
                        self.log_message(f"Lat range: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
                        self.log_message(f"Lon range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
                        
                except Exception as e:
                    self.log_message(f"Warning: Could not parse Lon/Lat column: {e}")
        
        elif file_type == 'phins_ins':
            # For PHINS INS files, prefer phins-specific data over GPS data when available
            phins_priority_mapping = {
                'phins_lat': 'latitude',
                'phins_lon': 'longitude',
                'phins_depth': 'depth',
                'phins_heading': 'heading',
                'phins_pitch': 'pitch',
                'phins_roll': 'roll'
            }
            
            for phins_col, standard_col in phins_priority_mapping.items():
                if phins_col in df.columns and standard_col in df.columns:
                    # Use PHINS data to fill missing GPS data
                    mask = df[standard_col].isna() & df[phins_col].notna()
                    filled_count = mask.sum()
                    if filled_count > 0:
                        df.loc[mask, standard_col] = df.loc[mask, phins_col]
                        self.log_message(f"Filled {filled_count} missing {standard_col} values with PHINS data")
                elif phins_col in df.columns and standard_col not in df.columns:
                    # Create standard column from PHINS data
                    df[standard_col] = df[phins_col]
                    self.log_message(f"Created {standard_col} column from PHINS {phins_col}")
        
        elif file_type == 'state':
            # For STATE files, ensure we use proper heading data
            if 'compass_heading' in df.columns and 'heading' not in df.columns:
                df['heading'] = df['compass_heading'].apply(lambda x: Metrics.normalize_heading(x) if pd.notna(x) else x)
                self.log_message("Using compass_heading as heading for STATE file with normalization")
                
        elif file_type == 'adcp':
            # ADCP files should have altitude data - this is crucial for bathymetry calculations
            if 'altitude' in df.columns:
                valid_altitude = (~df['altitude'].isna()).sum()
                self.log_message(f"ADCP file contains {valid_altitude} valid altitude measurements")
        
        return df
    
    def _standardize_columns(self, df, file_type):
        """Standardize column names based on file type mappings"""
        if file_type not in self.column_mappings:
            self.log_message(f"Warning: No column mapping defined for file type {file_type}")
            return df
            
        mappings = self.column_mappings[file_type]['mappings']
        
        # Apply mappings, handling duplicate columns
        rename_dict = {}
        columns_to_drop = []
        
        for old_col, new_col in mappings.items():
            if old_col in df.columns:
                # Check if the new column name already exists
                if new_col in df.columns and old_col != new_col:
                    # If target column already exists, we may have a duplicate
                    # Keep the existing one and mark the mapped one for dropping later
                    if old_col != new_col:  # Only if they're actually different
                        columns_to_drop.append(old_col)
                        self.log_message(f"Skipping mapping {old_col} -> {new_col} (target already exists)")
                        continue
                
                rename_dict[old_col] = new_col
        
        # Apply renames
        if rename_dict:
            df = df.rename(columns=rename_dict)
            self.log_message(f"Applied column mappings: {rename_dict}")
        
        # Drop duplicate columns that couldn't be mapped
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            self.log_message(f"Dropped duplicate columns: {columns_to_drop}")
        
        # Handle remaining duplicate column names by adding suffixes
        if df.columns.duplicated().any():
            self.log_message("Found duplicate column names, adding suffixes...")
            new_columns = []
            col_counts = {}
            
            for col in df.columns:
                if col in col_counts:
                    col_counts[col] += 1
                    new_col = f"{col}_{col_counts[col]}"
                    new_columns.append(new_col)
                    self.log_message(f"Renamed duplicate column {col} to {new_col}")
                else:
                    col_counts[col] = 0
                    new_columns.append(col)
            
            df.columns = new_columns
        
        return df
    
    def _standardize_time_column(self, df, file_type):
        """Standardize time column to a consistent format"""
        time_cols = self.column_mappings.get(file_type, {}).get('time_cols', ['time'])
        
        # Special handling for PHINS INS files: prioritize phins_time if it looks like seconds-since-midnight
        if file_type == 'phins_ins' and 'phins_time' in df.columns:
            # Check if phins_time looks like seconds-since-midnight (0-86400 range)
            phins_time_values = pd.to_numeric(df['phins_time'], errors='coerce').dropna()
            if len(phins_time_values) > 0:
                max_val = phins_time_values.max()
                min_val = phins_time_values.min()
                
                # If phins_time is in seconds-since-midnight range, use it preferentially
                if 0 <= min_val and max_val <= 90000:  # Allow some margin above 86400
                    self.log_message(f"PHINS INS: Using phins_time (seconds-since-midnight format) as primary time column")
                    # Convert seconds to milliseconds for consistency
                    time_in_ms = phins_time_values * 1000
                    # Apply midnight crossover handling
                    time_in_ms = self._handle_midnight_crossover(time_in_ms)
                    df['time'] = time_in_ms
                    return df
                else:
                    self.log_message(f"PHINS INS: phins_time range ({min_val:.1f} to {max_val:.1f}) not in seconds-since-midnight format, using mission time")
        
        # Find the first available time column
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            self.log_message(f"Warning: No time column found in {file_type} file")
            # Try to create time from other sources
            if file_type == 'veh_data':
                # For Veh_Data, try to parse the Time column (HH:MM:SS.SSS format)
                if 'Time' in df.columns:
                    try:
                        df['time'] = self._parse_time_string(df['Time'], file_type)
                        self.log_message(f"Converted Time column to milliseconds for {file_type}")
                    except Exception as e:
                        # Fallback: create sequential time
                        df['time'] = range(len(df))
                        self.log_message(f"Created sequential time for {file_type} (Error: {e})")
                else:
                    # Create sequential time
                    df['time'] = range(len(df))
                    self.log_message(f"Created sequential time for {file_type} (no Time column)")
            else:
                # For other files, create sequential time
                df['time'] = range(len(df))
                self.log_message(f"Created sequential time for {file_type}")
            
            return df
        
        # Rename to standard 'time' if needed
        if time_col != 'time':
            df['time'] = df[time_col]
        
        # Convert time to numeric if it's not already
        # Check for non-numeric dtype (covers both 'object' and pandas StringDtype)
        if not pd.api.types.is_numeric_dtype(df['time']):
            try:
                # Try parsing as time string first (legacy WHOI format)
                parsed_time = self._parse_time_string(df['time'], file_type)
                df['time'] = parsed_time
                self.log_message(f"Converted time strings to milliseconds in {file_type}")
            except:
                try:
                    # Try to convert time strings to numeric (modern mission_msecs format)
                    df['time'] = pd.to_numeric(df['time'], errors='coerce')
                    self.log_message(f"Converted time column to numeric in {file_type}")
                except:
                    self.log_message(f"Warning: Could not convert time column to numeric in {file_type}")
        
        return df
    
    def _parse_time_string(self, time_series, file_type):
        """Parse time strings in various formats to continuous milliseconds handling midnight crossover"""
        time_str = time_series.astype(str).str.strip()
        
        # Try different time patterns
        patterns = [
            # H:MM:SS.S or HH:MM:SS.S (PHINS INS WHOI format)
            r'^\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d+)\s*$',
            # H:MM:SS.SSS or HH:MM:SS.SSS (NavFileTest WHOI format) 
            r'^\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,3})\s*$',
            # H:MM:SS or HH:MM:SS (basic time format)
            r'^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$'
        ]
        
        for pattern in patterns:
            try:
                time_parts = time_str.str.extract(pattern)
                if not time_parts.empty and time_parts.iloc[:, 0].notna().any():
                    hours = pd.to_numeric(time_parts.iloc[:, 0], errors='coerce').fillna(0)
                    minutes = pd.to_numeric(time_parts.iloc[:, 1], errors='coerce').fillna(0) 
                    seconds = pd.to_numeric(time_parts.iloc[:, 2], errors='coerce').fillna(0)
                    
                    # Handle fractional seconds
                    if time_parts.shape[1] > 3:
                        frac_str = time_parts.iloc[:, 3].fillna('0')
                        # Convert fractional seconds to milliseconds based on digit count
                        milliseconds = frac_str.apply(lambda x: self._convert_fractional_seconds(str(x)))
                    else:
                        milliseconds = pd.Series([0] * len(time_parts))
                    
                    # Convert to milliseconds since midnight
                    total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
                    
                    # Handle midnight crossover for continuous time
                    total_milliseconds = self._handle_midnight_crossover(total_milliseconds)
                    
                    self.log_message(f"Successfully parsed time format for {file_type}: {pattern}")
                    return total_milliseconds
                    
            except Exception as e:
                self.log_message(f"Pattern {pattern} failed for {file_type}: {e}")
                continue
        
        # If all patterns fail, create sequential time
        self.log_message(f"Warning: Could not parse time strings for {file_type}, using sequential time")
        return pd.Series(range(len(time_series)))
    
    def _convert_fractional_seconds(self, frac_str):
        """Convert fractional seconds string to milliseconds"""
        if not frac_str or frac_str == 'nan':
            return 0
        
        try:
            # Remove any non-digit characters
            digits = ''.join(filter(str.isdigit, str(frac_str)))
            if not digits:
                return 0
                
            # Convert based on number of digits
            if len(digits) == 1:
                # .5 -> 500ms
                return int(digits) * 100
            elif len(digits) == 2:
                # .55 -> 550ms  
                return int(digits) * 10
            elif len(digits) == 3:
                # .555 -> 555ms
                return int(digits)
            else:
                # More than 3 digits, truncate to 3
                return int(digits[:3])
                
        except:
            return 0
    
    def _handle_midnight_crossover(self, time_ms):
        """Handle midnight crossover to create continuous time series"""
        if len(time_ms) <= 1:
            return time_ms
        
        # Make a copy to avoid modifying original
        continuous_time = time_ms.copy()
        
        # Detect different types of midnight crossover:
        # 1. For seconds-since-midnight data (0-86400 range): look for backward jumps
        # 2. For time string data: look for hour wrapping from 23:xx to 0x:xx
        
        # Check if this looks like seconds since midnight (typical range 0-86400)
        max_time = continuous_time.max()
        min_time = continuous_time.min()
        
        # If we have times that look like milliseconds since midnight (0-86400000 range)
        if max_time < 90000000 and min_time >= 0:
            self.log_message("Detected milliseconds-since-midnight format, checking for midnight wraparound")
            
            # Detect midnight crossover: significant backward jumps (> 20 hours = 72,000,000 ms)
            time_diffs = continuous_time.diff()
            midnight_crossover_threshold = -20 * 3600 * 1000  # -20 hours in milliseconds
            
            crossover_indices = time_diffs < midnight_crossover_threshold
            
            if crossover_indices.any():
                # Add 24 hours (86,400,000 ms) for each midnight crossover
                cumulative_days = crossover_indices.cumsum()
                day_offset_ms = cumulative_days * 24 * 3600 * 1000
                continuous_time = continuous_time + day_offset_ms
                
                crossover_count = crossover_indices.sum()
                self.log_message(f"Detected and corrected {crossover_count} midnight crossovers in milliseconds-since-midnight data")
                
                # Set the midnight crossover flag
                self.midnight_crossover_detected = True
                
                # Log the time range for debugging
                start_time_str = self._ms_to_time_string(time_ms.iloc[0])
                end_time_str = self._ms_to_time_string(time_ms.iloc[-1])
                duration_hours = (continuous_time.iloc[-1] - continuous_time.iloc[0]) / (3600 * 1000)
                self.log_message(f"Mission time span: {start_time_str} to {end_time_str} (duration: {duration_hours:.1f} hours)")
        
        # If we have times in the seconds-since-midnight range (0-90000)
        elif max_time < 90000 and min_time >= 0:
            self.log_message("Detected seconds-since-midnight format, checking for midnight wraparound")
            
            # Detect midnight crossover: significant backward jumps (> 20 hours = 72,000 seconds)
            time_diffs = continuous_time.diff()
            midnight_crossover_threshold = -20 * 3600  # -20 hours in seconds
            
            crossover_indices = time_diffs < midnight_crossover_threshold
            
            if crossover_indices.any():
                # Add 24 hours (86,400 seconds) for each midnight crossover
                cumulative_days = crossover_indices.cumsum()
                day_offset_seconds = cumulative_days * 24 * 3600
                continuous_time = continuous_time + day_offset_seconds
                
                crossover_count = crossover_indices.sum()
                self.log_message(f"Detected and corrected {crossover_count} midnight crossovers in seconds-since-midnight data")
                
                # Set the midnight crossover flag
                self.midnight_crossover_detected = True
                
                # Log the time range for debugging
                start_time_str = self._seconds_to_time_string(time_ms.iloc[0])
                end_time_str = self._seconds_to_time_string(time_ms.iloc[-1])
                duration_hours = (continuous_time.iloc[-1] - continuous_time.iloc[0]) / 3600
                self.log_message(f"Mission time span: {start_time_str} to {end_time_str} (duration: {duration_hours:.1f} hours)")
        
        else:
            # For mission time data, check if the time span suggests crossing midnight
            # by examining the datetime conversion and looking for day changes
            self.log_message("Mission time format detected, checking for day boundary crossover")
            
            # Try to detect if this is a long mission that could cross midnight
            duration_ms = max_time - min_time
            duration_hours = duration_ms / (3600 * 1000)  # Convert to hours assuming milliseconds
            
            if duration_hours > 18:  # Mission longer than 18 hours likely crosses midnight
                self.log_message(f"Long mission detected ({duration_hours:.1f} hours), flagging as midnight crossover")
                self.midnight_crossover_detected = True
        
        return continuous_time
    
    def _ms_to_time_string(self, ms):
        """Convert milliseconds since midnight to HH:MM:SS.SSS format for logging"""
        try:
            total_seconds = int(ms / 1000)
            milliseconds = int(ms % 1000)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        except:
            return str(ms)
    
    def _seconds_to_time_string(self, seconds):
        """Convert seconds since midnight to HH:MM:SS.SSS format for logging"""
        try:
            total_seconds = int(seconds)
            fractional_seconds = int((seconds - total_seconds) * 1000)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{fractional_seconds:03d}"
        except:
            return str(seconds)
    
    def _convert_numeric_columns(self, df):
        """Convert navigation columns to numeric types"""
        numeric_cols = ['latitude', 'longitude', 'depth', 'heading', 'pitch', 'roll', 'heave', 'altitude', 'time']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Check if column is already numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    
                    # Convert to numeric, handling potential hex values or other formats
                    if col == 'heading' and df[col].dtype == 'object':
                        # Special handling for heading which might have hex values
                        # Try to convert, coercing errors to NaN
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        # If most values are NaN, it might be hex format
                        if numeric_values.isna().sum() > len(df) * 0.8:
                            self.log_message(f"Warning: {col} appears to contain non-numeric data, keeping as-is")
                            continue
                        else:
                            df[col] = numeric_values
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Log conversion results
                    valid_count = df[col].notna().sum()
                    total_count = len(df)
                    self.log_message(f"Converted {col} to numeric: {valid_count}/{total_count} valid values")
                    
                except Exception as e:
                    self.log_message(f"Warning: Could not convert {col} to numeric: {e}")
        
        return df
    
    def merge_navigation_files(self, file_paths):
        """
        Efficiently merge navigation files by loading only what's needed.
        Stops loading files once all required attributes are found.
        
        Args:
            file_paths: List of file paths or dict with file_type: path mappings
            
        Returns:
            Merged DataFrame with navigation data
        """
        self.log_message("=== Starting Optimized Navigation Data Merge ===")
        
        # If file_paths is a list, convert to priority-ordered processing
        if isinstance(file_paths, list):
            file_data = []
            for path in file_paths:
                if path and os.path.exists(path):
                    file_type = self.identify_file_type(path)
                    file_data.append((path, file_type))
        else:
            # If it's a dict, use the provided mappings
            file_data = [(path, ftype) for ftype, path in file_paths.items() if path and os.path.exists(path)]
        
        # Sort by priority (PHINS INS highest, CTD and unknown lowest)
        priority_order = ['phins_ins', 'nav_state', 'state', 'adcp', 'veh_data', 'ctd', 'unknown']
        file_data.sort(key=lambda x: priority_order.index(x[1]) if x[1] in priority_order else len(priority_order))
        
        self.log_message(f"Found {len(file_data)} files in priority order:")
        for path, ftype in file_data:
            self.log_message(f"  {ftype}: {os.path.basename(path)}")
        
        # Define required and optional attributes
        required_attrs = {'time', 'latitude', 'longitude', 'depth'}
        important_attrs = {'heading', 'pitch', 'roll', 'heave', 'altitude'}
        all_attrs = required_attrs | important_attrs
        
        # Track which attributes we've found
        found_attrs = set()
        loaded_dfs = []
        
        # Load files in priority order until we have all needed attributes
        for file_path, file_type in file_data:
            self.log_message(f"\nProcessing {file_type}: {os.path.basename(file_path)}")
            
            # Quick check: scan file columns to see what it offers
            try:
                # Read just the header to see what columns are available
                header_df = pd.read_csv(file_path, nrows=1, sep=None, engine='python')
                header_df.columns = [col.strip(' ,') for col in header_df.columns]
                available_cols = set(header_df.columns)
                
                # Map to standardized column names based on file type
                if file_type in self.column_mappings:
                    mappings = self.column_mappings[file_type]['mappings']
                    mapped_cols = set()
                    for old_col, new_col in mappings.items():
                        if old_col in available_cols:
                            mapped_cols.add(new_col)
                    # Also include direct matches
                    direct_matches = available_cols & all_attrs
                    potential_attrs = mapped_cols | direct_matches
                else:
                    potential_attrs = available_cols & all_attrs
                
                self.log_message(f"  Available attributes: {sorted(potential_attrs)}")
                
                # Check if this file would add any new attributes we need
                new_attrs = potential_attrs - found_attrs
                missing_required = required_attrs - found_attrs
                missing_important = important_attrs - found_attrs
                
                if new_attrs or missing_required:
                    # Load this file as it provides needed data
                    df = self.load_and_standardize_file(file_path, file_type)
                    if df is not None and not df.empty:
                        loaded_dfs.append(df)
                        
                        # Update found attributes
                        actual_cols = set(df.columns)
                        newly_found = actual_cols & all_attrs
                        found_attrs.update(newly_found)
                        
                        self.log_message(f"  ✓ Loaded - added attributes: {sorted(newly_found)}")
                        self.log_message(f"  Total found so far: {sorted(found_attrs)}")
                        
                        # Check if we have all required attributes
                        missing_required = required_attrs - found_attrs
                        if not missing_required:
                            # We have all required attributes
                            missing_important = important_attrs - found_attrs
                            if not missing_important:
                                self.log_message(f"  ✅ All required and important attributes found! Stopping file loading.")
                                break
                            else:
                                self.log_message(f"  📍 All required attributes found. Still missing: {sorted(missing_important)}")
                        else:
                            self.log_message(f"  ⚠️  Still missing required: {sorted(missing_required)}")
                    else:
                        self.log_message(f"  ❌ Failed to load file")
                else:
                    self.log_message(f"  ⏭️  Skipping - no new attributes needed")
                    
            except Exception as e:
                self.log_message(f"  ❌ Error checking file: {e}")
                continue
        
        # Final summary
        final_missing_required = required_attrs - found_attrs
        final_missing_important = important_attrs - found_attrs
        
        self.log_message(f"\n=== Loading Summary ===")
        self.log_message(f"Loaded {len(loaded_dfs)} files out of {len(file_data)} available")
        self.log_message(f"Found attributes: {sorted(found_attrs)}")
        if final_missing_required:
            self.log_message(f"⚠️  Missing REQUIRED: {sorted(final_missing_required)}")
        if final_missing_important:
            self.log_message(f"Missing optional: {sorted(final_missing_important)}")
        
        if not loaded_dfs:
            raise ValueError("No valid navigation files could be loaded")
        
        if final_missing_required:
            raise ValueError(f"Could not find required navigation attributes: {sorted(final_missing_required)}")
        
        # Merge the dataframes by time
        merged_df = self._merge_by_time(loaded_dfs)
        
        # Post-process the merged data
        merged_df = self._post_process_merged_data(merged_df)
        
        if merged_df is None:
            raise ValueError("Merge produced no valid data — all time values may be null or incompatible")
        
        self.log_message(f"\n=== Merge Complete ===")
        self.log_message(f"Final dataset: {len(merged_df)} records")
        self.log_message(f"Available columns: {list(merged_df.columns)}")
        self.log_message(f"Time range: {merged_df['time'].min()} to {merged_df['time'].max()}")
        
        return merged_df
    
    def _merge_by_time(self, dataframes):
        """Merge dataframes based on time with priority-based column selection"""
        self.log_message("Merging dataframes by time with priority-based selection...")
        
        # First, ensure all time columns are the same data type (float64)
        for df in dataframes:
            if 'time' in df.columns:
                df['time'] = pd.to_numeric(df['time'], errors='coerce').astype('float64')
        
        # Filter out rows with null time values from all dataframes
        clean_dataframes = []
        for i, df in enumerate(dataframes):
            source_type = df['source_type'].iloc[0] if len(df) > 0 else f"dataset_{i}"
            original_count = len(df)
            
            # Remove rows where time is null
            clean_df = df.dropna(subset=['time']).copy()
            dropped_count = original_count - len(clean_df)
            
            if dropped_count > 0:
                self.log_message(f"Dropped {dropped_count} rows with null time values from {source_type}")
            
            if len(clean_df) == 0:
                self.log_message(f"Warning: No valid time data remaining in {source_type} after filtering")
                continue
                
            clean_dataframes.append(clean_df)
        
        if len(clean_dataframes) == 0:
            self.log_message("Error: No dataframes have valid time data for merging")
            return None
        
        # Start with the highest priority dataframe as base
        base_df = clean_dataframes[0].copy()
        base_source = base_df['source_type'].iloc[0]
        self.log_message(f"Base dataset from {base_source}: {len(base_df)} records")
        
        # Merge additional dataframes in priority order
        for i, df in enumerate(clean_dataframes[1:], 1):
            merge_source = df['source_type'].iloc[0]
            self.log_message(f"Merging dataset {i+1} from {merge_source}: {len(df)} records")
            
            # Sort both dataframes by time for merge_asof
            base_sorted = base_df.sort_values('time').reset_index(drop=True)
            merge_sorted = df.sort_values('time').reset_index(drop=True)
            
            # Ensure consistent data types for time columns
            base_sorted['time'] = pd.to_numeric(base_sorted['time'], errors='coerce')
            merge_sorted['time'] = pd.to_numeric(merge_sorted['time'], errors='coerce')
            
            # Double-check for any remaining null values
            if base_sorted['time'].isna().any():
                self.log_message(f"Warning: Base dataset still has {base_sorted['time'].isna().sum()} null time values")
                base_sorted = base_sorted.dropna(subset=['time'])
            
            if merge_sorted['time'].isna().any():
                self.log_message(f"Warning: Merge dataset still has {merge_sorted['time'].isna().sum()} null time values")
                merge_sorted = merge_sorted.dropna(subset=['time'])
            
            if len(base_sorted) == 0 or len(merge_sorted) == 0:
                self.log_message(f"Skipping merge with {merge_source} - no valid time data")
                continue
            
            # Perform time-based merge with tolerance for slight time differences
            merged = pd.merge_asof(
                base_sorted, 
                merge_sorted,
                on='time',
                direction='nearest',
                tolerance=2000,  # 2 second tolerance for time matching (in milliseconds)
                suffixes=('', f'_src{i+1}')
            )
            
            # Priority-based column filling
            nav_columns = ['latitude', 'longitude', 'depth', 'heading', 'pitch', 'roll', 'heave', 'altitude']
            
            for col in nav_columns:
                base_col = col
                merge_col = f"{col}_src{i+1}"
                
                if base_col in merged.columns and merge_col in merged.columns:
                    # Fill NaN values in base column with values from merge column
                    mask = merged[base_col].isna() & merged[merge_col].notna()
                    filled_count = mask.sum()
                    if filled_count > 0:
                        merged.loc[mask, base_col] = merged.loc[mask, merge_col]
                        self.log_message(f"  Filled {filled_count} missing {col} values from {merge_source}")
                    
                    # For critical navigation data, also check for outliers and replace if needed
                    if col in ['latitude', 'longitude'] and filled_count < len(merged) * 0.5:
                        # If we have both datasets and one seems more complete, prefer it
                        base_valid = merged[base_col].notna().sum()
                        merge_valid = merged[merge_col].notna().sum()
                        
                        if merge_valid > base_valid * 1.5:  # Merge source has significantly more data
                            # Replace base data with merge data where merge is valid
                            replacement_mask = merged[merge_col].notna()
                            replaced_count = replacement_mask.sum()
                            merged.loc[replacement_mask, base_col] = merged.loc[replacement_mask, merge_col]
                            self.log_message(f"  Replaced {replaced_count} {col} values with higher quality data from {merge_source}")
                    
                    # Drop the temporary merge column
                    merged = merged.drop(columns=[merge_col])
                    
                elif merge_col in merged.columns and base_col not in merged.columns:
                    # Add new column if it doesn't exist in base
                    merged[base_col] = merged[merge_col]
                    merged = merged.drop(columns=[merge_col])
                    self.log_message(f"  Added new column {col} from {merge_source}")
            
            # Clean up any remaining merge columns
            merge_cols_to_drop = [col for col in merged.columns if col.endswith(f'_src{i+1}')]
            if merge_cols_to_drop:
                merged = merged.drop(columns=merge_cols_to_drop)
            
            base_df = merged
        
        # Log final data quality
        nav_columns = ['latitude', 'longitude', 'depth', 'heading', 'pitch', 'roll', 'heave', 'altitude']
        for col in nav_columns:
            if col in base_df.columns:
                valid_count = base_df[col].notna().sum()
                total_count = len(base_df)
                coverage = (valid_count / total_count) * 100
                self.log_message(f"Final {col} coverage: {valid_count}/{total_count} ({coverage:.1f}%)")
        
        return base_df
    
    def _post_process_merged_data(self, df):
        """Post-process merged navigation data"""
        if df is None:
            self.log_message("Warning: _post_process_merged_data received None — skipping post-processing")
            return None

        # Create datetime column from time
        df = self._create_datetime_column(df)
        
        # Filter altitude outliers before calculating derived metrics
        df = self._filter_altitude_outliers(df, **self.altitude_filter_params)
        
        # Calculate derived metrics (using filtered altitude data)
        df = self._calculate_derived_metrics(df)
        
        # Add midnight crossover flag as metadata
        if hasattr(self, 'midnight_crossover_detected') and self.midnight_crossover_detected:
            df.attrs['midnight_crossover'] = True
            self.log_message("Added midnight crossover flag to dataset metadata")
        else:
            df.attrs['midnight_crossover'] = False
        
        # Clean up source columns
        source_cols = [col for col in df.columns if col.startswith('source_') and col.endswith(('_src2', '_src3', '_src4', '_src5'))]
        if source_cols:
            df = df.drop(columns=source_cols)
        
        # Remove duplicate time entries
        df = df.drop_duplicates(subset=['time'], keep='first')
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def _create_datetime_column(self, df):
        """Create datetime column from time data, handling midnight crossover"""
        try:
            if 'time' in df.columns:
                # Time is now in continuous milliseconds (handles midnight crossover)
                # Use a reasonable base date and time
                base_date = pd.Timestamp('2024-01-01 08:00:00')  # Start at 8 AM
                df['datetime'] = base_date + pd.to_timedelta(df['time'], unit='ms')
                self.log_message("Created datetime column from continuous time data")
        except Exception as e:
            self.log_message(f"Warning: Could not create datetime column: {e}")
        
        return df
    
    def _calculate_derived_metrics(self, df):
        """Calculate derived navigation metrics"""
        # Calculate bathymetry if both depth and altitude are available
        if 'depth' in df.columns and 'altitude' in df.columns:
            valid_mask = df['depth'].notna() & df['altitude'].notna() 
            df.loc[valid_mask, 'bathymetry'] = df.loc[valid_mask, 'depth'] + df.loc[valid_mask, 'altitude']
            bathy_count = valid_mask.sum()
            if bathy_count > 0:
                self.log_message(f"Calculated bathymetry for {bathy_count} data points")
        
        return df
    
    def _standardize_heave_columns(self, df):
        """Standardize heave columns to handle both 'heave' and 'phins_heave' flexibly"""
        heave_variations = ['heave', 'phins_heave', 'Heave', 'PHINS_heave', 'phins_Heave']
        
        # Find available heave columns
        available_heave_cols = [col for col in df.columns if col in heave_variations]
        
        if available_heave_cols:
            # Use the first available heave column and standardize the name to 'heave'
            primary_heave_col = available_heave_cols[0]
            
            if primary_heave_col != 'heave':
                df = df.rename(columns={primary_heave_col: 'heave'})
                self.log_message(f"Standardized heave column: {primary_heave_col} -> heave")
            
            # If there are multiple heave columns, drop the others to avoid confusion
            other_heave_cols = [col for col in available_heave_cols[1:] if col in df.columns]
            if other_heave_cols:
                df = df.drop(columns=other_heave_cols)
                self.log_message(f"Dropped duplicate heave columns: {other_heave_cols}")
        
        return df

    def scan_directory_for_navigation_files(self, directory_path, log_callback=None):
        """
        Scan a directory for navigation files and automatically identify their types
        
        Args:
            directory_path: Path to directory containing navigation files
            log_callback: Optional callback function for logging
            
        Returns:
            Dict mapping file types to file paths, sorted by priority
        """
        if log_callback:
            def log_message(message):
                print(message)
                log_callback(message)
        else:
            def log_message(message):
                print(message)
        
        if not os.path.exists(directory_path):
            log_message(f"Error: Directory does not exist: {directory_path}")
            return {}
        
        if not os.path.isdir(directory_path):
            log_message(f"Error: Path is not a directory: {directory_path}")
            return {}
        
        log_message(f"Scanning directory for navigation files: {directory_path}")
        
        # Common navigation file extensions
        nav_extensions = ['.txt', '.csv', '.dat', '.log', '.nav', '.phins']

        # Subdirectory names to skip — these contain config, raw bags, or system files
        # rather than exported navigation data. Comparison is case-insensitive.
        skip_dirs = frozenset({
            'config', 'bags', 'missions', 'temp', 'logs',
        })
        
        # Find all potential navigation files, recursing into subdirectories but
        # pruning known non-nav folders so we don't pick up ADCP config files,
        # MCAP bags, system logs, etc.
        potential_files = []
        for root, dirs, files in os.walk(directory_path):
            # Prune dirs in-place so os.walk won't descend into them
            dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in nav_extensions):
                    file_path = os.path.join(root, file)
                    potential_files.append(file_path)

        # Sort so files in "export"-style subdirectories (EXPORTED, export, nav,
        # navigation, output) are processed before files in other subdirectories.
        # Top-level files (depth == 0) are always first.
        norm_root = os.path.normpath(directory_path)
        export_dirs = frozenset({'exported', 'export', 'nav', 'navigation', 'output'})

        def _file_priority(path):
            parent = os.path.normpath(os.path.dirname(path))
            if parent == norm_root:
                return 0  # flat / top-level — highest priority
            if os.path.basename(parent).lower() in export_dirs:
                return 1  # recognised export subdirectory
            return 2      # other subdirectory

        potential_files.sort(key=_file_priority)
        
        log_message(f"Found {len(potential_files)} potential navigation files")
        
        # Identify file types for each potential navigation file
        identified_files = {}
        file_type_counts = {}
        
        for file_path in potential_files:
            try:
                file_type = self.identify_file_type(file_path)
                filename = os.path.basename(file_path)
                
                if file_type != 'unknown':
                    log_message(f"  {file_type}: {filename}")
                    
                    # Keep track of multiple files of the same type
                    if file_type in file_type_counts:
                        file_type_counts[file_type] += 1
                        # Use suffix for duplicates
                        unique_key = f"{file_type}_{file_type_counts[file_type]}"
                    else:
                        file_type_counts[file_type] = 1
                        unique_key = file_type
                    
                    identified_files[unique_key] = file_path
                else:
                    log_message(f"  unknown: {filename} (skipped)")
            
            except Exception as e:
                log_message(f"  Error identifying {os.path.basename(file_path)}: {e}")
        
        # Sort by priority and create final mapping
        priority_order = ['phins_ins', 'nav_state', 'state', 'adcp', 'veh_data', 'ctd']
        final_files = {}
        
        # First, add files in priority order (without suffixes)
        for file_type in priority_order:
            if file_type in identified_files:
                final_files[file_type] = identified_files[file_type]
        
        # Then add any additional files of the same type with suffixes
        for key, path in identified_files.items():
            if key not in final_files:
                final_files[key] = path
        if final_files:
            log_message(f"Selected navigation files in priority order:")
            for file_type, file_path in final_files.items():
                log_message(f"  {file_type}: {os.path.basename(file_path)}")
        else:
            log_message("No valid navigation files found in directory")
        
        return final_files
    
    def merge_navigation_directory(self, directory_path, log_callback=None):
        """
        Scan directory for navigation files and merge them automatically
        
        Args:
            directory_path: Path to directory containing navigation files
            log_callback: Optional callback function for logging
            
        Returns:
            Merged DataFrame with navigation data
        """
        # Scan directory for navigation files
        nav_files = self.scan_directory_for_navigation_files(directory_path, log_callback)
        
        if not nav_files:
            raise ValueError(f"No valid navigation files found in directory: {directory_path}")
        
        # Merge the identified files
        return self.merge_navigation_files(nav_files)
    
    def set_altitude_filter_parameters(self, method='combined', z_threshold=3.0, iqr_multiplier=2.0, rolling_window=50, rolling_threshold=3.0):
        """
        Configure altitude outlier filter parameters.
        
        Args:
            method: Outlier detection method ('z_score', 'iqr', 'rolling_median', 'combined')
            z_threshold: Z-score threshold for outlier detection (default: 3.0)
            iqr_multiplier: IQR multiplier for outlier detection (default: 2.0)
            rolling_window: Window size for rolling median filter (default: 50)
            rolling_threshold: Threshold for rolling median filter (default: 3.0)
        """
        self.altitude_filter_params = {
            'method': method,
            'z_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier,
            'rolling_window': rolling_window,
            'rolling_threshold': rolling_threshold
        }
        self.log_message(f"Altitude filter parameters updated: method={method}, z_threshold={z_threshold}")

    def _filter_altitude_outliers(self, df, method='combined', z_threshold=3.0, iqr_multiplier=2.0, rolling_window=50, rolling_threshold=3.0):
        """
        Filter altitude outliers using robust statistical methods.
        
        When an AUV's altimeter loses bottom lock (e.g., at the surface), it can report 
        spurious altitude values that need to be filtered out for accurate bathymetry calculations.
        
        Args:
            df: DataFrame with altitude data
            method: Outlier detection method ('z_score', 'iqr', 'rolling_median', 'combined')
            z_threshold: Z-score threshold for outlier detection (default: 3.0)
            iqr_multiplier: IQR multiplier for outlier detection (default: 2.0) 
            rolling_window: Window size for rolling median filter (default: 50)
            rolling_threshold: Threshold for rolling median filter (default: 3.0)
            
        Returns:
            DataFrame with filtered altitude data (outliers set to NaN)
        """
        if 'altitude' not in df.columns:
            return df
            
        # Get original altitude data
        original_altitude = df['altitude'].copy()
        valid_mask = original_altitude.notna()
        
        if valid_mask.sum() < 10:  # Need at least 10 valid points for meaningful statistics
            self.log_message("Insufficient altitude data for outlier filtering (< 10 valid points)")
            return df
            
        valid_altitudes = original_altitude[valid_mask]
        outlier_mask = pd.Series(False, index=df.index)
        
        self.log_message(f"Filtering altitude outliers using {method} method...")
        self.log_message(f"Original altitude range: {valid_altitudes.min():.2f}m to {valid_altitudes.max():.2f}m")
        
        # Method 1: Z-score based outlier detection
        if method in ['z_score', 'combined']:
            mean_alt = valid_altitudes.mean()
            std_alt = valid_altitudes.std()
            
            if std_alt > 0:  # Avoid division by zero
                z_scores = np.abs((original_altitude - mean_alt) / std_alt)
                z_outliers = z_scores > z_threshold
                outlier_mask |= z_outliers
                z_outlier_count = z_outliers.sum()
                self.log_message(f"  Z-score method: detected {z_outlier_count} outliers (threshold: {z_threshold})")
        
        # Method 2: Interquartile Range (IQR) based detection
        if method in ['iqr', 'combined']:
            q1 = valid_altitudes.quantile(0.25)
            q3 = valid_altitudes.quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:  # Avoid edge case where all values are the same
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
                
                iqr_outliers = (original_altitude < lower_bound) | (original_altitude > upper_bound)
                outlier_mask |= iqr_outliers
                iqr_outlier_count = iqr_outliers.sum()
                self.log_message(f"  IQR method: detected {iqr_outlier_count} outliers (bounds: {lower_bound:.2f}m to {upper_bound:.2f}m)")
        
        # Method 3: Rolling median based detection (good for temporal outliers)
        if method in ['rolling_median', 'combined']:
            # Calculate rolling median and standard deviation
            rolling_median = original_altitude.rolling(window=rolling_window, center=True, min_periods=5).median()
            rolling_std = original_altitude.rolling(window=rolling_window, center=True, min_periods=5).std()
            
            # Detect points that deviate significantly from local trend
            deviation = np.abs(original_altitude - rolling_median)
            rolling_outliers = (deviation > rolling_threshold * rolling_std) & rolling_std.notna()
            outlier_mask |= rolling_outliers
            rolling_outlier_count = rolling_outliers.sum()
            self.log_message(f"  Rolling median method: detected {rolling_outlier_count} outliers (window: {rolling_window}, threshold: {rolling_threshold})")
        
        # Apply additional physics-based filters for extreme cases
        # Filter extremely high altitudes (likely altimeter errors at surface)
        max_reasonable_altitude = valid_altitudes.quantile(0.95) + 3 * valid_altitudes.std()
        max_reasonable_altitude = max(max_reasonable_altitude, 100.0)  # At least 100m ceiling
        extreme_high = original_altitude > max_reasonable_altitude
        outlier_mask |= extreme_high
        extreme_high_count = extreme_high.sum()
        if extreme_high_count > 0:
            self.log_message(f"  Physics filter: detected {extreme_high_count} extremely high altitude values (>{max_reasonable_altitude:.1f}m)")
        
        # Filter negative altitudes (physically impossible for bottom-looking altimeter)
        negative_altitudes = original_altitude < 0
        outlier_mask |= negative_altitudes  
        negative_count = negative_altitudes.sum()
        if negative_count > 0:
            self.log_message(f"  Physics filter: detected {negative_count} negative altitude values")
        
        # Apply outlier mask by setting outliers to NaN
        total_outliers = outlier_mask.sum()
        if total_outliers > 0:
            df.loc[outlier_mask, 'altitude'] = np.nan
            remaining_valid = df['altitude'].notna().sum()
            outlier_percentage = (total_outliers / len(df)) * 100
            self.log_message(f"  Total outliers filtered: {total_outliers} ({outlier_percentage:.1f}% of data)")
            self.log_message(f"  Remaining valid altitude points: {remaining_valid}")
            
            # Log altitude range after filtering
            if remaining_valid > 0:
                filtered_altitudes = df['altitude'].dropna()
                self.log_message(f"  Filtered altitude range: {filtered_altitudes.min():.2f}m to {filtered_altitudes.max():.2f}m")
        else:
            self.log_message("  No altitude outliers detected")
            
        return df


def merge_navigation_files(file_paths, log_callback=None):
    """
    Convenience function to merge navigation files
    
    Args:
        file_paths: List of file paths or dict with file_type: path mappings
        log_callback: Optional callback function for logging
        
    Returns:
        Merged DataFrame with navigation data
    """
    merger = NavigationDataMerger(log_callback)
    return merger.merge_navigation_files(file_paths)


def identify_navigation_file_type(file_path, log_callback=None):
    """
    Convenience function to identify navigation file type
    
    Args:
        file_path: Path to navigation file
        log_callback: Optional callback function for logging
        
    Returns:
        String indicating file type ('phins_ins', 'nav_state', 'state', 'adcp', 'veh_data', 'unknown')
    """
    merger = NavigationDataMerger(log_callback)
    return merger.identify_file_type(file_path)


def load_single_navigation_file(file_path, file_type=None, log_callback=None):
    """
    Convenience function to load and standardize a single navigation file
    
    Args:
        file_path: Path to navigation file
        file_type: Optional file type (will be auto-detected if None)
        log_callback: Optional callback function for logging
        
    Returns:
        Standardized DataFrame with navigation data
    """
    merger = NavigationDataMerger(log_callback)
    return merger.load_and_standardize_file(file_path, file_type)


def merge_navigation_directory(directory_path, log_callback=None):
    """
    Convenience function to scan a directory and merge navigation files
    
    Args:
        directory_path: Path to directory containing navigation files
        log_callback: Optional callback function for logging
        
    Returns:
        Merged DataFrame with navigation data
    """
    merger = NavigationDataMerger(log_callback)
    return merger.merge_navigation_directory(directory_path)


def scan_navigation_directory(directory_path, log_callback=None):
    """
    Convenience function to scan a directory for navigation files
    
    Args:
        directory_path: Path to directory containing navigation files
        log_callback: Optional callback function for logging
        
    Returns:
        Dict mapping file types to file paths
    """
    merger = NavigationDataMerger(log_callback)
    return merger.scan_directory_for_navigation_files(directory_path, log_callback)
