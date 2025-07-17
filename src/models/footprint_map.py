import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import warnings
import datetime
from dateutil import parser
import traceback

# Try importing geopandas, but handle the case where it might not be installed
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("geopandas or shapely not available. Shapefile export will be disabled.")

class FootprintMap:
    def __init__(self, altitude_threshold=5.0):
        """
        Initialize the footprint map generator
        
        Args:
            altitude_threshold: Maximum altitude to include in the footprint map (in meters)
        """
        # VOYIS camera parameters from FoV Overlap Calculator.R
        self.hfov = 56  # Horizontal Field of View in degrees
        self.vfov = 42  # Vertical Field of View in degrees
        self.nav_data = None
        self.altitude_threshold = altitude_threshold  # Add threshold parameter
        
    def load_nav_data(self, nav_file_path: str) -> bool:
        """
        Load navigation data from a text file, focusing only on time and heading
        
        Args:
            nav_file_path: Path to the navigation file
            
        Returns:
            Boolean indicating success
        """
        try:
            print(f"Loading navigation data from {nav_file_path}")
            
            # First, examine the file structure to determine format
            with open(nav_file_path, 'r') as f:
                # Read the first few lines to analyze structure
                header_lines = [f.readline() for _ in range(5)]
            
            # Look for header line with "Time" and "Heading"
            header_line_index = -1
            for i, line in enumerate(header_lines):
                if 'Time' in line and ('Heading' in line or 'heading' in line.lower()):
                    header_line_index = i
                    break
            
            if header_line_index == -1:
                print("Could not identify header line with 'Time' and 'Heading' columns")
                # Try to find any line with commas as potential header
                for i, line in enumerate(header_lines):
                    if ',' in line and line.strip():
                        header_line_index = i
                        break
            
            # Read the file with pandas
            try:
                # Skip identified header lines
                if header_line_index >= 0:
                    self.nav_data = pd.read_csv(nav_file_path, skiprows=header_line_index)
                else:
                    # Try reading with default settings
                    self.nav_data = pd.read_csv(nav_file_path)
                
                # Clean column names
                self.nav_data.columns = self.nav_data.columns.str.strip()
                
                # Print available columns for debugging
                print(f"Available columns in nav file: {list(self.nav_data.columns)}")
                
                # Find time column (could be named Time, time, timestamp, etc.)
                time_col = None
                date_col = None
                
                for col in self.nav_data.columns:
                    if col.lower() == 'time':
                        time_col = col
                    elif col.lower() == 'date':
                        date_col = col
                
                if time_col is None:
                    # Try to find any column with 'time' in the name
                    time_candidates = [col for col in self.nav_data.columns if 'time' in col.lower()]
                    if time_candidates:
                        time_col = time_candidates[0]
                        print(f"Using '{time_col}' as time column")
                
                # Find heading column
                heading_col = None
                for col in self.nav_data.columns:
                    if col.lower() == 'heading':
                        heading_col = col
                
                if heading_col is None:
                    # Try to find any column with 'head' in the name
                    heading_candidates = [col for col in self.nav_data.columns if 'head' in col.lower()]
                    if heading_candidates:
                        heading_col = heading_candidates[0]
                        print(f"Using '{heading_col}' as heading column")
                
                # Extract and process only the columns we need
                if heading_col:
                    # Ensure heading is numeric
                    self.nav_data['Heading'] = pd.to_numeric(self.nav_data[heading_col], errors='coerce')
                    print(f"Processed heading data from column: {heading_col}")
                    # Remove rows with missing heading
                    self.nav_data = self.nav_data.dropna(subset=['Heading'])
                else:
                    print("Warning: No heading column found. Using default heading of 0.")
                    self.nav_data['Heading'] = 0.0
                
                # Extract datetime from time and date columns
                if time_col and date_col:
                    # Both time and date columns available
                    print(f"Using time column '{time_col}' and date column '{date_col}'")
                    
                    # Create datetime column
                    try:
                        # Show sample data
                        print(f"Sample time: {self.nav_data[time_col].iloc[0]}")
                        print(f"Sample date: {self.nav_data[date_col].iloc[0]}")
                        
                        # Convert to datetime - try different formats
                        self.nav_data['Datetime'] = pd.to_datetime(
                            self.nav_data[date_col].astype(str) + ' ' + self.nav_data[time_col].astype(str),
                            errors='coerce'
                        )
                    except Exception as e:
                        print(f"Error converting date and time to datetime: {e}")
                        return False
                elif time_col:
                    # Only time column available
                    print(f"Only time column '{time_col}' available, no date column")
                    try:
                        # Try to parse the time column directly
                        self.nav_data['Datetime'] = pd.to_datetime(self.nav_data[time_col], errors='coerce')
                    except Exception as e:
                        print(f"Error converting time to datetime: {e}")
                        return False
                else:
                    print("Warning: No time column found. Navigation data timing won't be available.")
                    return False
                
                # Remove rows with invalid datetime
                self.nav_data = self.nav_data.dropna(subset=['Datetime'])
                
                print(f"Successfully loaded {len(self.nav_data)} navigation records with datetime and heading")
                if not self.nav_data.empty:
                    print("First navigation record:")
                    print(f"  Datetime: {self.nav_data['Datetime'].iloc[0]}")
                    print(f"  Heading: {self.nav_data['Heading'].iloc[0]}")
                    print("Last navigation record:")
                    print(f"  Datetime: {self.nav_data['Datetime'].iloc[-1]}")
                    print(f"  Heading: {self.nav_data['Heading'].iloc[-1]}")
                
                return True
                
            except Exception as e:
                print(f"Error reading navigation file with pandas: {e}")
                return False
                
        except Exception as e:
            print(f"Error loading navigation data: {e}")
            import traceback
            print(traceback.format_exc())
            self.nav_data = None
            return False
    
    def _parse_nav_datetime(self, date_str, time_str):
        """Parse navigation date and time into datetime object"""
        try:
            # Print raw input for debugging
            print(f"Parsing date: '{date_str}' and time: '{time_str}'")
            
            # Convert MM/DD/YYYY to YYYY-MM-DD
            if isinstance(date_str, str) and '/' in date_str:
                date_parts = date_str.strip().split('/')
                if len(date_parts) == 3:
                    # Check if year is in position 2 (MM/DD/YYYY)
                    if len(date_parts[2]) == 4:  # Likely YYYY
                        formatted_date = f"{date_parts[2]}-{date_parts[0].zfill(2)}-{date_parts[1].zfill(2)}"
                    # Or if it's in position 0 (YYYY/MM/DD)
                    elif len(date_parts[0]) == 4:  # Likely YYYY
                        formatted_date = f"{date_parts[0]}-{date_parts[1].zfill(2)}-{date_parts[2].zfill(2)}"
                    else:
                        # Default to MM/DD/YYYY and assume current century
                        year = date_parts[2]
                        if len(year) == 2:
                            year = f"20{year}"
                        formatted_date = f"{year}-{date_parts[0].zfill(2)}-{date_parts[1].zfill(2)}"
                else:
                    # Couldn't parse, use as-is
                    formatted_date = date_str.strip()
            else:
                formatted_date = str(date_str).strip()
            
            # Handle time format HH:MM:SS.SSS or similar
            if isinstance(time_str, str):
                time_str = time_str.strip()
            else:
                time_str = str(time_str).strip()
            
            # Create datetime string
            datetime_str = f"{formatted_date} {time_str}"
            print(f"Attempting to parse: '{datetime_str}'")
            
            # Try to parse
            return parser.parse(datetime_str)
            
        except Exception as e:
            print(f"Error parsing datetime: {date_str, time_str, {e}}")
            
            # Try even more basic approach - just combine and try to parse
            try:
                datetime_str = f"{str(date_str).strip()} {str(time_str).strip()}"
                return parser.parse(datetime_str)
            except:
                return None
    
    def _extract_latitude(self, lon_lat_str):
        """Extract latitude from combined Lon/Lat string"""
        try:
            # Format: 28N05.40726  90W56.15713
            parts = lon_lat_str.strip().split()
            if len(parts) >= 1:
                lat_str = parts[0]
                
                # Check if format is like 28N05.40726
                if 'N' in lat_str or 'S' in lat_str:
                    direction = 1 if 'N' in lat_str else -1
                    lat_parts = lat_str.replace('N', ' ').replace('S', ' ').split()
                    degrees = float(lat_parts[0])
                    minutes = float(lat_parts[1])
                    return direction * (degrees + minutes / 60)
            
            return None
            
        except Exception as e:
            print(f"Error extracting latitude from {lon_lat_str}: {e}")
            return None
    
    def _extract_longitude(self, lon_lat_str):
        """Extract longitude from combined Lon/Lat string"""
        try:
            # Format: 28N05.40726  90W56.15713
            parts = lon_lat_str.strip().split()
            if len(parts) >= 2:
                lon_str = parts[1]
                
                # Check if format is like 90W56.15713
                if 'E' in lon_str or 'W' in lon_str:
                    direction = 1 if 'E' in lon_str else -1
                    lon_parts = lon_str.replace('E', ' ').replace('W', ' ').split()
                    degrees = float(lon_parts[0])
                    minutes = float(lon_parts[1])
                    return direction * (degrees + minutes / 60)
            
            return None
            
        except Exception as e:
            print(f"Error extracting longitude from {lon_lat_str}: {e}")
            return None
    
    def _calculate_footprint(self, altitude: float) -> Tuple[float, float]:
        """
        Calculate image footprint dimensions based on altitude
        
        Args:
            altitude: Distance from seafloor in meters
            
        Returns:
            Tuple of (width, height) in meters
        """
        # Calculate width using horizontal field of view
        width = 2 * altitude * np.tan(np.radians(self.hfov / 2))
        
        # Calculate height using vertical field of view
        height = 2 * altitude * np.tan(np.radians(self.vfov / 2))
        
        return width, height
    
    def _calculate_polygon_vertices(self, lat: float, lon: float, altitude: float, 
                                   heading: float) -> List[Tuple[float, float]]:
        """
        Calculate the four corners of the image footprint on the seafloor
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            altitude: Distance from seafloor in meters
            heading: Vehicle heading in degrees (0=North, 90=East)
            
        Returns:
            List of (lon, lat) coordinates for the four corners
        """
        # Calculate footprint dimensions
        width, height = self._calculate_footprint(altitude)
        
        # For VOYIS camera, width is typically larger than height (horizontal FOV > vertical FOV)
        # To align the shorter edge with the heading direction, we need to orient the rectangle correctly
        
        # Convert heading to radians
        heading_rad = np.radians(heading)
        
        # Calculate the offset vectors in the heading direction and perpendicular to it
        # The heading direction corresponds to the short edge (height)
        # The perpendicular direction corresponds to the long edge (width)
        
        # Unit vector in heading direction
        dx_heading = np.sin(heading_rad)  # x component of heading vector
        dy_heading = np.cos(heading_rad)  # y component of heading vector
        
        # Unit vector perpendicular to heading (90 degrees clockwise from heading)
        dx_perp = np.sin(heading_rad + np.pi/2)  # x component of perpendicular vector
        dy_perp = np.cos(heading_rad + np.pi/2)  # y component of perpendicular vector
        
        # Calculate half-dimensions for convenience
        half_width = width / 2
        half_height = height / 2
        
        # Approximate meters per degree of latitude and longitude at this location
        meters_per_lat = 111320  # approximate meters per degree latitude
        meters_per_lon = 111320 * np.cos(np.radians(lat))  # approximate meters per degree longitude
        
        # Calculate the four corners:
        # - along the heading vector (use the height dimension for forward/backward)
        # - along the perpendicular vector (use the width dimension for left/right)
        corners = [
            # Bottom left: back-left corner (-height/2 along heading, -width/2 perpendicular)
            (lon + (-half_height * dx_heading - half_width * dx_perp) / meters_per_lon, 
             lat + (-half_height * dy_heading - half_width * dy_perp) / meters_per_lat),
            
            # Bottom right: back-right corner (-height/2 along heading, +width/2 perpendicular)
            (lon + (-half_height * dx_heading + half_width * dx_perp) / meters_per_lon,
             lat + (-half_height * dy_heading + half_width * dy_perp) / meters_per_lat),
            
            # Top right: front-right corner (+height/2 along heading, +width/2 perpendicular)
            (lon + (half_height * dx_heading + half_width * dx_perp) / meters_per_lon,
             lat + (half_height * dy_heading + half_width * dy_perp) / meters_per_lat),
            
            # Top left: front-left corner (+height/2 along heading, -width/2 perpendicular)
            (lon + (half_height * dx_heading - half_width * dx_perp) / meters_per_lon,
             lat + (half_height * dy_heading - half_width * dy_perp) / meters_per_lat)
        ]
        
        return corners
    
    def _match_image_to_nav(self, image_datetime, tolerance_seconds=60):
        """
        Find matching navigation record for an image timestamp
        
        Args:
            image_datetime: Image timestamp
            tolerance_seconds: Max allowable time difference
        
        Returns:
            Closest matching navigation record's heading or None
        """
        if self.nav_data is None or 'Datetime' not in self.nav_data.columns:
            return None
        
        try:
            # Check if we need to adjust the year (in case nav data and images have different years)
            if not hasattr(self, '_year_difference_checked'):
                self._year_difference_checked = True
                
                # Get a sample image datetime
                img_year = image_datetime.year
                nav_year = self.nav_data['Datetime'].iloc[0].year
                
                if img_year != nav_year:
                    print(f"Warning: Image year ({img_year}) differs from navigation data year ({nav_year})")
                    print(f"Will adjust image datetime year to match navigation data")
                    
                    # Store the year difference for future adjustments
                    self._year_difference = nav_year - img_year
                else:
                    self._year_difference = 0
            
            # Adjust the image datetime year if needed
            if hasattr(self, '_year_difference') and self._year_difference != 0:
                adjusted_datetime = image_datetime.replace(year=image_datetime.year + self._year_difference)
            else:
                adjusted_datetime = image_datetime
                
            # Convert to pandas timestamp for comparison
            img_time = pd.Timestamp(adjusted_datetime)
            
            # Calculate time differences
            time_diffs = abs(self.nav_data['Datetime'] - img_time)
            
            # Find closest match
            if not time_diffs.empty:
                min_diff_seconds = time_diffs.min().total_seconds()
                
                if min_diff_seconds <= tolerance_seconds:
                    closest_idx = time_diffs.idxmin()
                    closest_match = self.nav_data.iloc[closest_idx]
                    
                    # Only print for the first few matches to avoid flooding the console
                    if not hasattr(self, '_match_count'):
                        self._match_count = 0
                        
                    if self._match_count < 10:
                        print(f"Matched image at {img_time} to nav record at {closest_match['Datetime']}, "
                             f"diff: {min_diff_seconds:.2f}s, heading: {closest_match['Heading']}")
                        self._match_count += 1
                    elif self._match_count == 10:
                        print("(Suppressing further match messages)")
                        self._match_count += 1
                    
                    # Just return the heading
                    return closest_match['Heading']
                else:
                    if not hasattr(self, '_nomatch_count'):
                        self._nomatch_count = 0
                        
                    if self._nomatch_count < 5:
                        print(f"No match within tolerance: Image time {img_time}, "
                             f"closest nav diff: {min_diff_seconds:.2f}s (tolerance: {tolerance_seconds}s)")
                        self._nomatch_count += 1
                    elif self._nomatch_count == 5:
                        print("(Suppressing further no-match messages)")
                        self._nomatch_count += 1
            
        except Exception as e:
            print(f"Error matching image to nav: {e}")
            
        return None
    
    def _parse_datetime_from_filename(self, filename):
        """Extract datetime from filename format like ESC_stills_processed_PPS_2024-06-27T074938.458700_2170.jpg"""
        try:
            if '_' not in filename or 'T' not in filename:
                return None
                
            # Extract the date-time part
            parts = filename.split('_')
            datetime_part = None
            
            for part in parts:
                if 'T' in part and len(part.split('T')) == 2:
                    datetime_part = part
                    break
                    
            if not datetime_part:
                return None
                
            # Split into date and time
            date_str, time_str = datetime_part.split('T')
            
            # Remove milliseconds if present
            if '.' in time_str:
                time_str = time_str.split('.')[0]
                
            # Format time with colons if needed
            if len(time_str) >= 6 and ':' not in time_str:
                time_str = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                
            # Create datetime string and parse it
            datetime_str = f"{date_str} {time_str}"
            return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            print(f"Error parsing datetime from filename '{filename}': {e}")
            return None
    
    def create_footprint_map(self, gps_data: List[Dict], output_path: str, 
                            nav_file_path: str = None,
                            filename: str = "Image_Footprints_Map.png") -> Optional[str]:
        """
        Create a map showing image footprints on the seafloor
        
        Args:
            gps_data: List of dictionaries with GPS and EXIF data
            output_path: Directory to save the output files
            nav_file_path: Path to vehicle navigation file
            filename: Filename for the generated map
            
        Returns:
            Path to the saved plot file, or None if no data or error
"""
        if not gps_data:
            print("No GPS data available")
            return None
        
        # Detect outliers before filtering
        outliers = self._detect_outliers(gps_data)
        
        # Create non-outlier subset for plotting
        plot_data = [point for point, is_outlier in zip(gps_data, outliers) if not is_outlier]
        
        print(f"\nExcluding {sum(outliers)} outlier points from plots (but keeping them in export files)")
        
        # Use plot_data for creating the map, but keep original gps_data for exports
        # Rest of the existing function code, but use plot_data instead of gps_data for plotting
        
        # Load navigation data if provided
        if nav_file_path and os.path.exists(nav_file_path):
            self.load_nav_data(nav_file_path)
        
        print(f"Processing {len(plot_data)} GPS data points")
        
        # Debug: Print the keys available in the first few GPS points
        if plot_data:
            print("First GPS point keys:", list(plot_data[0].keys()))
        
        # Extract valid points - filter by altitude threshold
        valid_points = []
        filtered_out_count = 0
        
        for point in plot_data:
            # Check if required fields are present
            if not all(key in point for key in ['latitude', 'longitude', 'altitude']):
                continue
                
            # Apply altitude threshold filter
            if point['altitude'] > self.altitude_threshold:
                filtered_out_count += 1
                continue
                
            # Add to valid points
            valid_points.append(point)
                
        if not valid_points:
            print("No valid GPS points with altitude below threshold found")
            return None
            
        print(f"Found {len(valid_points)} valid points with altitude below threshold ({filtered_out_count} points filtered out)")
        
        # Add footprint data to the points
        footprints = []
        
        # Keep track of points with/without datetime for logging
        datetime_available_count = 0
        match_found_count = 0
        
        for point in valid_points:
            # Default heading - will be used if we can't match with nav data
            heading = 0
            image_datetime = None
            
            try:
                # Try to extract datetime from the point for nav matching
                
                # Try to get DateTime from EXIF
                if 'DateTime' in point and point['DateTime'] and point['DateTime'] != 'N/A':
                    try:
                        # Handle common EXIF datetime format: "2024:06:27 07:49:38"
                        if isinstance(point['DateTime'], str):
                            if ':' in point['DateTime']:
                                # Replace colons in date part with dashes for parsing
                                date_time = point['DateTime'].replace(':', '-', 2)
                                image_datetime = parser.parse(date_time)
                            else:
                                image_datetime = parser.parse(point['DateTime'])
                        else:
                            # If it's already a datetime object
                            image_datetime = point['DateTime']
                    except Exception as e:
                        print(f"Error parsing DateTime from EXIF: {e}")
                        
                # Try from filename if DateTime not available from EXIF
                if image_datetime is None and 'filename' in point:
                    image_datetime = self._parse_datetime_from_filename(point['filename'])
                    if image_datetime:
                        print(f"Parsed datetime from filename: {image_datetime}")
                
                # If we have a datetime, count it for logging
                if image_datetime:
                    datetime_available_count += 1
                    
                    # Get heading from nav data
                    if self.nav_data is not None:
                        matched_heading = self._match_image_to_nav(image_datetime)
                        if matched_heading is not None:
                            heading = matched_heading
                            match_found_count += 1
                
                # Calculate footprint dimensions
                width, height = self._calculate_footprint(point['altitude'])
                
                # Calculate polygon vertices
                vertices = self._calculate_polygon_vertices(
                    point['latitude'], 
                    point['longitude'], 
                    point['altitude'], 
                    heading
                )
                
                # Add footprint data to the point
                footprint_data = {
                    'latitude': point['latitude'],
                    'longitude': point['longitude'],
                    'altitude': point['altitude'],
                    'heading': heading,
                    'width': width,
                    'height': height,
                    'vertices': vertices,
                    'filename': point.get('filename', 'unknown'),
                    'DateTime': str(image_datetime) if image_datetime else 'unknown'
                }
                
                footprints.append(footprint_data)
                
            except Exception as e:
                print(f"Error processing point: {e}")
                continue
        
        print(f"Created {len(footprints)} footprints for mapping")
        print(f"Points with datetime: {datetime_available_count} of {len(valid_points)}")
        print(f"Points with matched heading from nav: {match_found_count}")
        
        if not footprints:
            print("No valid footprints could be calculated")
            return None
            
        # Create figure for plotting
        print("Creating plot...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each footprint
        for i, fp in enumerate(footprints):
            try:
                # Plot polygon with low alpha for overlap visibility
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.3, edgecolor='blue', facecolor='skyblue')
                
                # Only plot every Nth center point to reduce clutter for large datasets
                if len(footprints) > 500:
                    if i % 20 == 0:  # Plot every 20th point for large datasets
                        ax.plot(fp['longitude'], fp['latitude'], 'r.', markersize=2)
                else:
                    ax.plot(fp['longitude'], fp['latitude'], 'r.', markersize=2)
            except Exception as e:
                print(f"Error plotting footprint: {e}")
                continue
        
        # Set labels and title
        ax.set_xlabel('Longitude (Decimal Degrees)')
        ax.set_ylabel('Latitude (Decimal Degrees)')
        ax.set_title('Image Footprints on Seafloor')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add statistics text box with overlap info if available
        stats_text = 'Total Images: {}\nAvg. Footprint: {:.2f}m × {:.2f}m\nAvg. Altitude: {:.2f}m'.format(
            len(footprints),
            np.mean([fp["width"] for fp in footprints]),
            np.mean([fp["height"] for fp in footprints]),
            np.mean([fp["altitude"] for fp in footprints])
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Configure axis formatting to show full coordinate values
        # Round to 2 decimal places
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
        
        # Increase font sizes
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(item.get_fontsize() + 2)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to accommodate the longer tick labels
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_path, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate all types of overlap first
        
        # Calculate vertical overlap
        if len(footprints) >= 2:
            print("\nCalculating vertical overlap between sequential images...")
            vertical_overlap_stats = self.calculate_vertical_overlap(footprints)
            print("Vertical overlap calculation complete")
            # Store overlap stats for later use in exports
            self.vertical_overlap_stats = vertical_overlap_stats


        
        # Calculate horizontal overlap
        if len(footprints) >= 21:  # Need at least min_separation+1 images
            print("\nCalculating horizontal overlap...")
            horizontal_overlap_stats = self.calculate_horizontal_overlap(footprints)
            print("Horizontal overlap calculation complete")
            # Store overlap stats for later use in exports
            self.horizontal_overlap_stats = horizontal_overlap_stats
        
        # Calculate overall overlap
        if len(footprints) >= 2:
            print("\nCalculating overall overlap...")
            overall_overlap_stats = self.calculate_overall_overlap(footprints)
            print("Overall overlap calculation complete")
            # Store overlap stats for later use in exports
            self.overall_overlap_stats = overall_overlap_stats


        # Export to CSV and shapefile AFTER all overlap metrics are calculated
        print("\nExporting to CSV and shapefile...")
        self.export_footprints(footprints, output_path) 
        
        # Export metrics to individual text files
        print("\nExporting metrics to text files...")
        self.export_metrics_to_text_files(footprints, output_path)
        
        # Create various overlap maps
        
        # Create vertical overlap map
        if len(footprints) >= 2:
            print("\nCreating vertical overlap map...")
            self.create_vertical_overlap_map(footprints, output_path)
            print("Vertical overlap map created")
        
        # Create horizontal overlap map
        if len(footprints) >= 21:  # Need at least min_separation+1 images
            print("\nCreating horizontal overlap map...")
            self.create_horizontal_overlap_map(footprints, output_path)
            print("Horizontal overlap map created")
        
        # Create overall overlap map
        if len(footprints) >= 2:
            print("\nCreating overall overlap map...")
            self.create_overall_overlap_map(footprints, output_path)
            print("Overall overlap map created")
        
        # Identify meaningful subsets and create all zoomed maps in one place
        subsets = self.identify_subsets(footprints)
        if subsets:
            self.create_zoomed_maps(footprints, subsets, output_path)
        

        
        return output_file

    def calculate_vertical_overlap(self, footprints: List[Dict]) -> Dict:
        """
        Calculate overlap between sequential images in a survey line
        
        Args:
            footprints: List of footprint data dictionaries
            
        Returns:
            Dictionary with overlap statistics and detailed overlap data
        """
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Cannot calculate vertical overlap.")
            return None
            
        if len(footprints) < 2:
            print("Not enough images to calculate vertical overlap.")
            return None
        
        try:
            # Sort footprints by filename (which should put them in sequence)
            sorted_footprints = sorted(footprints, key=lambda x: x['filename'])
            
            # Calculate overlaps between sequential images
            overlap_data = []
            total_overlap_area = 0
            total_overlap_count = 0
            
            # Counters for overlap categories
            low_overlap = 0
            medium_overlap = 0
            high_overlap = 0
            
            for i in range(len(sorted_footprints) - 1):
                current_fp = sorted_footprints[i]
                next_fp = sorted_footprints[i + 1]
                
                # Create Shapely polygons
                current_poly = Polygon(current_fp['vertices'])
                next_poly = Polygon(next_fp['vertices'])
                
                # Calculate intersection
                if current_poly.intersects(next_poly):
                    intersection = current_poly.intersection(next_poly)
                    
                    # Calculate overlap area and percentage
                    current_area = current_poly.area
                    next_area = next_poly.area
                    intersection_area = intersection.area
                    
                    # Overlap as percentage of the smaller footprint
                    overlap_percent = (intersection_area / min(current_area, next_area)) * 100.0
                    
                    # Calculate center point of intersection
                    if intersection.geom_type == 'Polygon':
                        midpoint = (intersection.centroid.x, intersection.centroid.y)
                    elif intersection.geom_type == 'MultiPolygon':
                        # Find the largest polygon in the collection
                        largest_poly = max(intersection.geoms, key=lambda p: p.area)
                        midpoint = (largest_poly.centroid.x, largest_poly.centroid.y)
                    else:
                        # Default to midpoint between the two footprints
                        midpoint = ((current_fp['longitude'] + next_fp['longitude']) / 2,
                                   (current_fp['latitude'] + next_fp['latitude']) / 2)
                    
                    # Categorize overlap
                    if overlap_percent < 40:
                        overlap_category = 'low'
                        low_overlap += 1
                    elif overlap_percent < 70:
                        overlap_category = 'medium'
                        medium_overlap += 1
                    else:
                        overlap_category = 'high'
                        high_overlap += 1
                    
                    # Store overlap data for this pair
                    overlap_info = {
                        'current_fp': current_fp,
                        'next_fp': next_fp,
                        'overlap_area': intersection_area,
                        'overlap_percent': overlap_percent,
                        'category': overlap_category,
                        'midpoint': midpoint,
                        'intersection': intersection
                    }
                    
                    overlap_data.append(overlap_info)
                    total_overlap_area += intersection_area
                    total_overlap_count += 1
            # Calculate statistics
            if total_overlap_count > 0:
                overlap_percentages = [d['overlap_percent'] for d in overlap_data]
                avg_overlap = sum(overlap_percentages) / len(overlap_percentages)
                median_overlap = np.median(overlap_percentages)
                min_overlap = min(overlap_percentages)
                max_overlap = max(overlap_percentages)
            else:
                avg_overlap = 0
                median_overlap = 0
                min_overlap = 0
                max_overlap = 0
            
            print(f"\nVertical Overlap Statistics:")
            print(f"  Total overlapping pairs: {total_overlap_count}")
            print(f"  Average overlap: {avg_overlap:.2f}%")
            print(f"  Median overlap: {median_overlap:.2f}%")
            print(f"  Range: {min_overlap:.2f}% - {max_overlap:.2f}%")
            print(f"  Low overlap (<40%): {low_overlap}")
            print(f"  Medium overlap (40-70%): {medium_overlap}")
            print(f"  High overlap (>70%): {high_overlap}")
            
            return {
                'overlap_data': overlap_data,
                'total_overlap_count': total_overlap_count,
                'avg_overlap': avg_overlap,
                'median_overlap': median_overlap,
                'min_overlap': min_overlap,
                'max_overlap': max_overlap,
                'low_overlap': low_overlap,
                'medium_overlap': medium_overlap,
                'high_overlap': high_overlap
            }
        
        except Exception as e:
            print(f"Error calculating vertical overlap: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def calculate_horizontal_overlap(self, footprints: List[Dict], min_separation: int = 20) -> Dict:
        """
        Calculate overlap between non-sequential images (likely in adjacent survey lines)
        
        Args:
            footprints: List of footprint data dictionaries
            min_separation: Minimum separation in sequence to consider non-sequential
            
        Returns:
            Dictionary with overlap statistics and detailed overlap data
        """
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Cannot calculate horizontal overlap.")
            return None
            
        if len(footprints) < min_separation + 1:
            print(f"Not enough images to calculate horizontal overlap (need at least {min_separation + 1}).")
            return None
        
        try:
            print(f"Calculating horizontal overlap for {len(footprints)} images...")
            
            # Sort footprints by filename for consistent ordering
            sorted_footprints = [(i, fp) for i, fp in enumerate(sorted(footprints, key=lambda x: x['filename']))]
            
            # Create an RTtree spatial index for fast intersection checks
            try:
                from rtree import index
                
                # Set up the spatial index
                idx = index.Index()
                poly_lookup = {}
                
                # Add polygons to the spatial index
                for i, (idx_i, fp) in enumerate(sorted_footprints):
                    poly = Polygon(fp['vertices'])
                    # Store the polygon and index in the lookup dictionary
                    poly_lookup[i] = (idx_i, fp, poly)
                    # Add to spatial index using the polygon's bounds (minx, miny, maxx, maxy)
                    idx.insert(i, poly.bounds)
                
                print("Using RTree spatial index for efficient overlap calculation")
            except ImportError:
                print("RTree not available, falling back to brute force comparison (slower)")
                idx = None
                
            # Calculate overlaps between non-sequential images
            overlap_data = []
            total_overlap_count = 0
            
            # Counters for overlap categories
            low_overlap = 0    # < 10%
            medium_overlap = 0 # 10-40%
            high_overlap = 0   # > 40%
            
            # Process in chunks to show progress
            total_checks = 0
            chunk_size = min(1000, len(sorted_footprints) // 10 or 1)
            
            # If we have RTree, use spatial index to speed up calculations
            if idx is not None:
                for i, (idx_i, fp) in enumerate(sorted_footprints):
                    # Create polygon for current footprint
                    poly = Polygon(fp['vertices'])
                    
                    # Show progress for large datasets
                    if i % chunk_size == 0 and i > 0:
                        print(f"Processed {i}/{len(sorted_footprints)} footprints, found {total_overlap_count} horizontal overlaps")
                    
                    # Find potential intersections using the spatial index
                    potential_matches = list(idx.intersection(poly.bounds))
                    
                    for j in potential_matches:
                        idx_j, distant_fp, distant_poly = poly_lookup[j]
                        
                        # Skip self-comparison
                        if idx_i == idx_j:
                            continue
                            
                        # Skip if the images are close in sequence (likely same survey line)
                        if abs(idx_i - idx_j) < min_separation:
                            continue
                        
                        total_checks += 1
                        
                        # Now do the exact intersection check
                        if poly.intersects(distant_poly):
                            intersection = poly.intersection(distant_poly)
                            
                            # Calculate overlap area and percentage
                            current_area = poly.area
                            distant_area = distant_poly.area
                            intersection_area = intersection.area
                            
                            # Overlap as percentage of the smaller footprint
                            overlap_percent = (intersection_area / min(current_area, distant_area)) * 100.0
                            
                            # Calculate center point of intersection
                            if intersection.geom_type == 'Polygon':
                                midpoint = (intersection.centroid.x, intersection.centroid.y)
                            elif intersection.geom_type == 'MultiPolygon':
                                largest_poly = max(intersection.geoms, key=lambda p: p.area)
                                midpoint = (largest_poly.centroid.x, largest_poly.centroid.y)
                            else:
                                midpoint = ((fp['longitude'] + distant_fp['longitude']) / 2,
                                           (fp['latitude'] + distant_fp['latitude']) / 2)
                            
                            # Categorize overlap - horizontal overlaps are typically lower than vertical
                            if overlap_percent < 10:
                                overlap_category = 'low'
                                low_overlap += 1
                            elif overlap_percent < 40:
                                overlap_category = 'medium'
                                medium_overlap += 1
                            else:
                                overlap_category = 'high'
                                high_overlap += 1
                            
                            # Store overlap data for this pair
                            overlap_info = {
                                'current_fp': fp,
                                'next_fp': distant_fp,
                                'overlap_area': intersection_area,
                                'overlap_percent': overlap_percent,
                                'category': overlap_category,
                                'midpoint': midpoint,
                                'intersection': intersection,
                                'sequence_distance': abs(idx_i - idx_j)
                            }
                            
                            overlap_data.append(overlap_info)
                            total_overlap_count += 1
            else:
                # Fallback to brute force comparison if RTree is not available
                for i in range(len(sorted_footprints)):
                    idx_i, current_fp = sorted_footprints[i]
                    current_poly = Polygon(current_fp['vertices'])
                    
                    # Show progress for large datasets
                    if i % chunk_size == 0 and i > 0:
                        print(f"Processed {i}/{len(sorted_footprints)} footprints, found {total_overlap_count} horizontal overlaps")
                    
                    for j in range(i + 1, len(sorted_footprints)):
                        idx_j, distant_fp = sorted_footprints[j]
                        
                        # Skip if the images are close in sequence (likely same survey line)
                        if abs(idx_i - idx_j) < min_separation:
                            continue
                        
                        total_checks += 1
                        
                        # Create Shapely polygon
                        distant_poly = Polygon(distant_fp['vertices'])
                        
                        # Check for intersection
                        if current_poly.intersects(distant_poly):
                            intersection = current_poly.intersection(distant_poly)
                            
                            # Calculate overlap area and percentage
                            current_area = current_poly.area
                            distant_area = distant_poly.area
                            intersection_area = intersection.area
                            
                            # Overlap as percentage of the smaller footprint
                            overlap_percent = (intersection_area / min(current_area, distant_area)) * 100.0
                            
                            # Calculate center point of intersection
                            if intersection.geom_type == 'Polygon':
                                midpoint = (intersection.centroid.x, intersection.centroid.y)
                            elif intersection.geom_type == 'MultiPolygon':
                                largest_poly = max(intersection.geoms, key=lambda p: p.area)
                                midpoint = (largest_poly.centroid.x, largest_poly.centroid.y)
                            else:
                                midpoint = ((current_fp['longitude'] + distant_fp['longitude']) / 2,
                                           (current_fp['latitude'] + distant_fp['latitude']) / 2)
                            
                            # Categorize overlap - horizontal overlaps are typically lower than vertical
                            if overlap_percent < 10:
                                overlap_category = 'low'
                                low_overlap += 1
                            elif overlap_percent < 40:
                                overlap_category = 'medium'
                                medium_overlap += 1
                            else:
                                overlap_category = 'high'
                                high_overlap += 1
                            
                            # Store overlap data for this pair
                            overlap_info = {
                                'current_fp': current_fp,
                                'next_fp': distant_fp,
                                'overlap_area': intersection_area,
                                'overlap_percent': overlap_percent,
                                'category': overlap_category,
                                'midpoint': midpoint,
                                'intersection': intersection,
                                'sequence_distance': abs(idx_i - idx_j)
                            }
                            
                            overlap_data.append(overlap_info)
                            total_overlap_count += 1
            
            # Calculate statistics
            if total_overlap_count > 0:
                overlap_percentages = [d['overlap_percent'] for d in overlap_data]
                avg_overlap = sum(overlap_percentages) / len(overlap_percentages)
                median_overlap = np.median(overlap_percentages)
                min_overlap = min(overlap_percentages)
                max_overlap = max(overlap_percentages)
                avg_sequence_distance = np.mean([d['sequence_distance'] for d in overlap_data])
            else:
                avg_overlap = 0
                median_overlap = 0
                min_overlap = 0
                max_overlap = 0
                avg_sequence_distance = 0
            
            print(f"\nHorizontal Overlap Statistics:")
            print(f"  Total pairs checked: {total_checks}")
            print(f"  Total overlapping pairs: {total_overlap_count}")
            print(f"  Average overlap: {avg_overlap:.2f}%")
            print(f"  Median overlap: {median_overlap:.2f}%")
            print(f"  Range: {min_overlap:.2f}% - {max_overlap:.2f}%")
            print(f"  Low overlap (<10%): {low_overlap}")
            print(f"  Medium overlap (10-40%): {medium_overlap}")
            print(f"  High overlap (>40%): {high_overlap}")
            print(f"  Average sequence distance: {avg_sequence_distance:.1f} images")
            
            return {
                'overlap_data': overlap_data,
                'total_overlap_count': total_overlap_count,
                'total_checks': total_checks,
                'avg_overlap': avg_overlap,
                'median_overlap': median_overlap,
                'min_overlap': min_overlap,
                'max_overlap': max_overlap,
                'low_overlap': low_overlap,
                'medium_overlap': medium_overlap,
                'high_overlap': high_overlap,
                'avg_sequence_distance': avg_sequence_distance
            }
        
        except Exception as e:
            print(f"Error calculating horizontal overlap: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def calculate_overall_overlap(self, footprints: List[Dict]) -> Dict:
        """Calculate overlap between all image footprints, counting how many overlaps each footprint has"""
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Cannot calculate overall overlap.")
            return None
            
        if len(footprints) < 2:
            print("Not enough images to calculate overall overlap.")
            return None
        
        try:
            print(f"Calculating overall overlap for {len(footprints)} images")
            
            # Create Shapely polygons from vertices
            polygons = []
            for i, fp in enumerate(footprints):
                poly = Polygon(fp['vertices'])
                polygons.append({
                    'index': i,
                    'polygon': poly, 
                    'footprint': fp,
                    'overlap_count': 0,  # Will track number of other footprints overlapping this one
                    'overlapping_indices': []  # Will track which footprints overlap with this one
                })
            
            # Variables for tracking overlap statistics
            total_checks = 0
            total_overlaps = 0
            
            # Process in chunks to show progress
            chunk_size = min(1000, len(polygons) // 10 or 1)
            
            # Try to use RTree spatial index for faster overlap calculations
            rtree_available = False
            try:
                from rtree import index
                
                # Set up the spatial index
                idx = index.Index()
                
                # Add polygons to the spatial index
                for i, poly_data in enumerate(polygons):
                    # Add to spatial index using the polygon's bounds (minx, miny, maxx, maxy)
                    idx.insert(i, poly_data['polygon'].bounds)
                    
                print("Using RTree spatial index for efficient overlap calculation")
                rtree_available = True
                
                # Calculate all pairwise overlaps using the spatial index
                for i, current in enumerate(polygons):
                    # Find potential intersections using the spatial index
                    potential_matches = list(idx.intersection(current['polygon'].bounds))
                    
                    # Skip self-comparison and check actual intersections
                    for j in potential_matches:
                        if i == j:
                            continue
                            
                        total_checks += 1
                        other = polygons[j]
                        
                        # Calculate intersection
                        if current['polygon'].intersects(other['polygon']):
                            current['overlap_count'] += 1
                            current['overlapping_indices'].append(j)
                            total_overlaps += 1
                            
                    # Show progress for large datasets
                    if i % chunk_size == 0 and i > 0:
                        print(f"Processed {i}/{len(polygons)} images, found {total_overlaps} overlaps so far")
                        
            except ImportError:
                print("RTree not available, falling back to brute force comparison (slower)")
                
            # If RTree fails or isn't available, use brute force comparison
            if not rtree_available:
                # Calculate all pairwise overlaps using brute force
                for i in range(len(polygons)):
                    current = polygons[i]
                    
                    # Compare with all other footprints
                    for j in range(len(polygons)):
                        if i == j:
                            continue
                            
                        total_checks += 1
                        other = polygons[j]
                        
                        # Calculate intersection
                        if current['polygon'].intersects(other['polygon']):
                            current['overlap_count'] += 1
                            current['overlapping_indices'].append(j)
                            total_overlaps += 1
            
            # Extract overlap counts and statistics
            overlap_counts = [p['overlap_count'] for p in polygons]
            
            # Calculate statistics
            avg_count = np.mean(overlap_counts)
            median_count = np.median(overlap_counts)
            min_count = np.min(overlap_counts)
            max_count = np.max(overlap_counts)
            
            print(f"\nOverall Overlap Statistics:")
            print(f"  Total pairs checked: {total_checks}")
            print(f"  Total overlapping pairs: {total_overlaps}")
            print(f"  Average overlap count per image: {avg_count:.2f}")
            print(f"  Median overlap count: {median_count:.2f}")
            print(f"  Minimum overlap count: {min_count}")
            print(f"  Maximum overlap count: {max_count}")
            
            # Store the results
            return {
                'avg_count': avg_count,
                'median_count': median_count,
                'min_count': min_count,
                'max_count': max_count,
                'total_checks': total_checks,
                'total_overlaps': total_overlaps,
                'polygons': polygons  # Store for mapping
            }
            
        except Exception as e:
            print(f"Error calculating overall overlap: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def create_vertical_overlap_map(self, footprints: List[Dict], output_path: str, 
                                  filename: str = "Image_Vertical_Overlap_Map.png") -> Optional[str]:
        """
        Create a map showing vertical overlap between sequential images
        
        Args:
            footprints: List of dictionaries with footprint data
            output_path: Directory to save the output files
            filename: Filename for the generated map
            
        Returns:
            Path to the saved plot file, or None if no data or error
        """
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Cannot create vertical overlap map.")
            return None
            
        if not hasattr(self, 'vertical_overlap_stats') or not self.vertical_overlap_stats:
            print("No vertical overlap data available.")
            return None
        
        try:
            # Use non-interactive backend to avoid threading issues
            import matplotlib
            matplotlib.use('Agg')
            
            # Create figure for plotting
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Setup colors
            low_color = '#FF4040'      # Red
            medium_color = '#FFFF40'   # Yellow
            high_color = '#40FF40'     # Green
            
            # Plot all footprints with very low alpha for context
            for fp in footprints:
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.05, edgecolor='gray', facecolor='whitesmoke')
            
            # Plot overlap regions with colors by category
            for overlap in self.vertical_overlap_stats['overlap_data']:
                intersection = overlap['intersection']
                category = overlap['category']
                
                # Set color based on overlap category
                if category == 'low':
                    color = low_color
                elif category == 'medium':
                    color = medium_color
                else:
                    color = high_color
                
                # Plot intersection
                if intersection.geom_type == 'Polygon':
                    x, y = intersection.exterior.xy
                    ax.fill(x, y, alpha=0.7, edgecolor='none', facecolor=color)
                    
                    # Add percentage label for larger polygons
                    if intersection.area > 0.0001:
                        centroid = intersection.centroid
                        ax.text(centroid.x, centroid.y, f"{overlap['overlap_percent']:.0f}%", 
                               fontsize=10,  # Increase from 8 to 10
                               ha='center', va='center', color='black',
                               bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
                    
                elif intersection.geom_type == 'MultiPolygon':
                    for poly in intersection.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, alpha=0.7, edgecolor='none', facecolor=color)
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title('Vertical Overlap Between Sequential Images')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Create legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=low_color, edgecolor='none', alpha=0.7, label='Low (<40%)'),
                Patch(facecolor=medium_color, edgecolor='none', alpha=0.7, label='Medium (40-70%)'),
                Patch(facecolor=high_color, edgecolor='none', alpha=0.7, label='High (>70%)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add statistics
            stats = self.vertical_overlap_stats
            stats_text = 'Vertical Overlap Statistics:\n'
            stats_text += f'Avg: {stats["avg_overlap"]:.1f}%\n'
            stats_text += f'Median: {stats["median_overlap"]:.1f}%\n'
            stats_text += f'Range: {stats["min_overlap"]:.1f}% - {stats["max_overlap"]:.1f}%\n'
            stats_text += f'Low: {stats["low_overlap"]} pairs\n'
            stats_text += f'Medium: {stats["medium_overlap"]} pairs\n'
            stats_text += f'High: {stats["high_overlap"]} pairs'
            
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, fontsize=11, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Configure axis formatting
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            plt.xticks(rotation=45, ha='right')
            
            # Increase font sizes
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(item.get_fontsize() + 2)
            
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return output_file
            
        except Exception as e:
            print(f"Error creating vertical overlap map: {e}")
            print(traceback.format_exc())
            return None

    def create_horizontal_overlap_map(self, footprints: List[Dict], output_path: str, 
                                   filename: str = "Image_Horizontal_Overlap_Map.png") -> Optional[str]:
        """
        Create a map showing horizontal overlap between non-sequential images
        
        Args:
            footprints: List of dictionaries with footprint data
            output_path: Directory to save the output files
            filename: Filename for the generated map
            
        Returns:
            Path to the saved plot file, or None if no data or error
        """
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Cannot create horizontal overlap map.")
            return None
            
        if not hasattr(self, 'horizontal_overlap_stats') or not self.horizontal_overlap_stats:
            print("No horizontal overlap data available.")
            return None
        
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Create figure for plotting
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Setup colors - different color scheme than vertical to distinguish
            low_color = '#FF4040'      # Red
            medium_color = '#FFA500'   # Orange
            high_color = '#40FF40'     # Green
            
            # Plot all footprints with very low alpha for context
            for fp in footprints:
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.05, edgecolor='gray', facecolor='whitesmoke')
            
            # Plot overlap regions with colors by category
            for overlap in self.horizontal_overlap_stats['overlap_data']:
                intersection = overlap['intersection']
                category = overlap['category']
                
                # Set color based on overlap category
                if category == 'low':
                    color = low_color
                elif category == 'medium':
                    color = medium_color
                else:
                    color = high_color
                
                # Plot intersection
                if intersection.geom_type == 'Polygon':
                    x, y = intersection.exterior.xy
                    ax.fill(x, y, alpha=0.7, edgecolor='none', facecolor=color)
                    
                    # Add percentage label for larger polygons
                    if intersection.area > 0.0001:
                        centroid = intersection.centroid
                        ax.text(centroid.x, centroid.y, f"{overlap['overlap_percent']:.0f}%", 
                               fontsize=10,  # Increase from 8 to 10
                               ha='center', va='center', color='black',
                               bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
                    
                elif intersection.geom_type == 'MultiPolygon':
                    for poly in intersection.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, alpha=0.7, edgecolor='none', facecolor=color)
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title('Horizontal Overlap Between Non-Sequential Images')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Create legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=low_color, edgecolor='none', alpha=0.7, label='Low (<10%)'),
                Patch(facecolor=medium_color, edgecolor='none', alpha=0.7, label='Medium (10-40%)'),
                Patch(facecolor=high_color, edgecolor='none', alpha=0.7, label='High (>40%)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add statistics
            stats = self.horizontal_overlap_stats
            stats_text = 'Horizontal Overlap Statistics:\n'
            stats_text += f'Avg: {stats["avg_overlap"]:.1f}%\n'
            stats_text += f'Median: {stats["median_overlap"]:.1f}%\n'
            stats_text += f'Range: {stats["min_overlap"]:.1f}% - {stats["max_overlap"]:.1f}%\n'
            stats_text += f'Low: {stats["low_overlap"]} pairs\n'
            stats_text += f'Medium: {stats["medium_overlap"]} pairs\n'
            stats_text += f'High: {stats["high_overlap"]} pairs'
            
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, fontsize=11, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Configure axis formatting
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            plt.xticks(rotation=45, ha='right')
            
            # Increase font sizes
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(item.get_fontsize() + 2)
            
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return output_file
            
        except Exception as e:
            print(f"Error creating horizontal overlap map: {e}")
            print(traceback.format_exc())
            return None

    def create_overall_overlap_map(self, footprints: List[Dict], output_path: str, 
                                filename: str = "Image_Overall_Overlap_Map.png") -> Optional[str]:
        """
        Create a map showing overall overlap counts with rainbow color gradient
        
        Args:
            footprints: List of dictionaries with footprint data
            output_path: Directory to save the output files
            filename: Filename for the generated map
            
        Returns:
            Path to the saved plot file, or None if no data or error
        """
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Cannot create overall overlap map.")
            return None
        
        # Calculate overall overlap if not already done
        if not hasattr(self, 'overall_overlap_stats') or not self.overall_overlap_stats:
            print("No overall overlap data available.")
            return None
        
        try:
            # Use non-interactive backend to avoid threading issues
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.colors as mcolors
            
            # Create figure for plotting
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Get polygons with overlap counts
            polygons = self.overall_overlap_stats['polygons']
            
            # Create a rainbow colormap from blue/purple (high) to red (low)
            # This reverses the usual rainbow order to have purple as highest
            cmap = plt.cm.rainbow_r
            
            # Determine color normalization based on overlap count range
            min_count = self.overall_overlap_stats['min_count']
            max_count = self.overall_overlap_stats['max_count']
            
            # Create a normalization that handles the case where min=max
            if min_count == max_count:
                norm = mcolors.Normalize(vmin=min_count-0.5, vmax=max_count+0.5)
            else:
                norm = mcolors.Normalize(vmin=min_count, vmax=max_count)
            
            # Plot each footprint with color based on overlap count
            for poly_data in polygons:
                count = poly_data['overlap_count']
                polygon = poly_data['polygon']
                
                # Get color from colormap
                color = cmap(norm(count))
                
                # Plot polygon
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.7, edgecolor='none', linewidth=0, facecolor=color)
                
                # Add count label for larger polygons
                if polygon.area > 0.0001:
                    centroid = polygon.centroid
                    ax.text(centroid.x, centroid.y, str(count), 
                           fontsize=10,  # Increase from 8 to 10
                           ha='center', va='center', color='white',
                           bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round,pad=0.2'))
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title('Overall Image Overlap Count')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # ScalarMappable needs an array for the colorbar
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Number of Overlapping Images')
            cbar.ax.tick_params(labelsize=10)  # Increase tick label size
            
            # Add overlap statistics
            stats_text = 'Overall Overlap Statistics:\n'
            stats_text += f'Avg overlaps: {self.overall_overlap_stats["avg_count"]:.1f} images\n'
            stats_text += f'Median: {self.overall_overlap_stats["median_count"]:.1f} images\n'
            stats_text += f'Range: {self.overall_overlap_stats["min_count"]} to {self.overall_overlap_stats["max_count"]} images\n'
            
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, fontsize=11, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Configure axis formatting to show full coordinate values
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            
            # Increase font sizes
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(item.get_fontsize() + 2)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout to accommodate the longer tick labels
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Explicitly close figure
            
            print(f"Overall overlap map saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating overall overlap map: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def export_footprints(self, footprints: List[Dict], output_path: str) -> Dict:
        """
        Export footprint data to CSV and shapefile formats
        
        Args:
            footprints: List of footprint data dictionaries
            output_path: Directory to save the output files
            
        Returns:
            Dictionary with paths to the exported files
        """
        result_files = {}
        
        try:
            # Create DataFrame from footprints
            export_data = []
            for i, fp in enumerate(footprints):
                # Extract the essential data
                data = {
                    'index': i,
                    'filename': fp.get('filename', ''),
                    'latitude': fp.get('latitude', 0),
                    'longitude': fp.get('longitude', 0),
                    'altitude': fp.get('altitude', 0),
                    'heading': fp.get('heading', 0),
                    'footprint_width': fp.get('width', 0),
                    'footprint_height': fp.get('height', 0),
                    'datetime': fp.get('DateTime', '')
                }
                
                # Add statistics from overlap calculations if available
                if hasattr(self, 'overall_overlap_stats') and self.overall_overlap_stats:
                    # Find matching polygon in overall stats
                    for poly_data in self.overall_overlap_stats.get('polygons', []):
                        if poly_data['index'] == i:
                            data['overlap_count'] = poly_data.get('overlap_count', 0)
                            break
                
                export_data.append(data)
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Export to CSV
            csv_file = os.path.join(output_path, "footprints.csv")
            df.to_csv(csv_file, index=False)
            result_files['csv'] = csv_file
            print(f"Footprint data exported to CSV: {csv_file}")
            
            # Export to shapefile if geopandas is available
            if GEOPANDAS_AVAILABLE:
                try:
                    # Create points for each footprint center
                    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
                    
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(df, geometry=geometry)
                    
                    # Set coordinate reference system (WGS84)
                    gdf.crs = "EPSG:4326"
                    
                    # Export to shapefile
                    shapefile = os.path.join(output_path, "Image_Footprints.shp")
                    gdf.to_file(shapefile)
                    result_files['shapefile'] = shapefile
                    print(f"Footprint data exported to shapefile: {shapefile}")
                    
                    # Create additional shapefile with footprint polygons if we have vertices
                    if all('vertices' in fp for fp in footprints):
                        # Create polygons
                        poly_geoms = [Polygon(fp['vertices']) for fp in footprints]
                        
                        # Create GeoDataFrame with polygons
                        poly_gdf = gpd.GeoDataFrame(df, geometry=poly_geoms)
                        poly_gdf.crs = "EPSG:4326"
                        
                        # Export polygon shapefile
                        poly_shapefile = os.path.join(output_path, "Image_Footprint_Polygons.shp")
                        poly_gdf.to_file(poly_shapefile)
                        result_files['polygon_shapefile'] = poly_shapefile
                        print(f"Footprint polygons exported to shapefile: {poly_shapefile}")
                
                except Exception as e:
                    print(f"Error exporting to shapefile: {e}")
                    print(traceback.format_exc())
            
            return result_files
            
        except Exception as e:
            print(f"Error exporting footprints: {e}")
            print(traceback.format_exc())
            return result_files

    def export_metrics_to_text_files(self, footprints: List[Dict], output_path: str) -> None:
        """
        Export all calculated metrics to individual text files
        
        Args:
            footprints: List of footprint data dictionaries
            output_path: Directory to save the output files
        """
        try:
            # Export basic footprint metrics
            self._export_footprint_metrics(footprints, output_path)
            
            # Export vertical overlap metrics
            if hasattr(self, 'vertical_overlap_stats') and self.vertical_overlap_stats:
                self._export_vertical_overlap_metrics(output_path)
            
            # Export horizontal overlap metrics  
            if hasattr(self, 'horizontal_overlap_stats') and self.horizontal_overlap_stats:
                self._export_horizontal_overlap_metrics(output_path)
            
            # Export overall overlap metrics
            if hasattr(self, 'overall_overlap_stats') and self.overall_overlap_stats:
                self._export_overall_overlap_metrics(output_path)
            
            print("All metrics exported to text files successfully")
            
        except Exception as e:
            print(f"Error exporting metrics to text files: {e}")
            import traceback
            print(traceback.format_exc())

    def _export_footprint_metrics(self, footprints: List[Dict], output_path: str) -> None:
        """Export basic footprint metrics to text file"""
        try:
            metrics_file = os.path.join(output_path, "Image_Footprint_Metrics.txt")
            
            with open(metrics_file, 'w') as f:
                f.write("FOOTPRINT ANALYSIS METRICS\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic statistics
                f.write(f"Total Images Analyzed: {len(footprints)}\n\n")
                
                # Altitude statistics
                altitudes = [fp['altitude'] for fp in footprints]
                f.write("ALTITUDE STATISTICS:\n")
                f.write(f"  Average Altitude: {np.mean(altitudes):.2f} m\n")
                f.write(f"  Median Altitude: {np.median(altitudes):.2f} m\n")
                f.write(f"  Min Altitude: {np.min(altitudes):.2f} m\n")
                f.write(f"  Max Altitude: {np.max(altitudes):.2f} m\n")
                f.write(f"  Std Deviation: {np.std(altitudes):.2f} m\n\n")
                
                # Footprint size statistics
                widths = [fp['width'] for fp in footprints]
                heights = [fp['height'] for fp in footprints]
                
                f.write("FOOTPRINT SIZE STATISTICS:\n")
                f.write(f"  Average Width: {np.mean(widths):.2f} m\n")
                f.write(f"  Average Height: {np.mean(heights):.2f} m\n")
                f.write(f"  Min Width: {np.min(widths):.2f} m\n")
                f.write(f"  Max Width: {np.max(widths):.2f} m\n")
                f.write(f"  Min Height: {np.min(heights):.2f} m\n")
                f.write(f"  Max Height: {np.max(heights):.2f} m\n\n")
                
                # Coverage area estimation
                areas = [fp['width'] * fp['height'] for fp in footprints]
                total_area = sum(areas)
                
                f.write("COVERAGE AREA ESTIMATION:\n")
                f.write(f"  Total Footprint Area: {total_area:.2f} m²\n")
                f.write(f"  Average Footprint Area: {np.mean(areas):.2f} m²\n")
                f.write(f"  Total Coverage (hectares): {total_area / 10000:.2f} ha\n\n")
                
                # Coordinate bounds
                lats = [fp['latitude'] for fp in footprints]
                lons = [fp['longitude'] for fp in footprints]
                
                f.write("SPATIAL BOUNDS:\n")
                f.write(f"  Latitude Range: {min(lats):.6f} to {max(lats):.6f}\n")
                f.write(f"  Longitude Range: {min(lons):.6f} to {max(lons):.6f}\n")
                f.write(f"  Latitude Span: {max(lats) - min(lats):.6f} degrees\n")
                f.write(f"  Longitude Span: {max(lons) - min(lons):.6f} degrees\n\n")
            
            print(f"Footprint metrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"Error exporting footprint metrics: {e}")

    def _export_vertical_overlap_metrics(self, output_path: str) -> None:
        """Export vertical overlap metrics to text file"""
        try:
            metrics_file = os.path.join(output_path, "Image_Vertical_Overlap_Metrics.txt")
            stats = self.vertical_overlap_stats
            
            with open(metrics_file, 'w') as f:
                f.write("VERTICAL OVERLAP ANALYSIS METRICS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Sequential image overlap statistics (along survey lines)\n\n")
                
                f.write(f"Total Overlapping Pairs: {stats['total_overlap_count']}\n")
                f.write(f"Average Overlap: {stats['avg_overlap']:.2f}%\n")
                f.write(f"Median Overlap: {stats['median_overlap']:.2f}%\n")
                f.write(f"Minimum Overlap: {stats['min_overlap']:.2f}%\n")
                f.write(f"Maximum Overlap: {stats['max_overlap']:.2f}%\n\n")
                
                f.write("OVERLAP DISTRIBUTION:\n")
                f.write(f"  Low Overlap (<40%): {stats['low_overlap']} pairs\n")
                f.write(f"  Medium Overlap (40-70%): {stats['medium_overlap']} pairs\n")
                f.write(f"  High Overlap (>70%): {stats['high_overlap']} pairs\n\n")
                
                total_pairs = stats['total_overlap_count']
                if total_pairs > 0:
                    f.write("OVERLAP PERCENTAGES:\n")
                    f.write(f"  Low Overlap: {(stats['low_overlap']/total_pairs*100):.1f}%\n")
                    f.write(f"  Medium Overlap: {(stats['medium_overlap']/total_pairs*100):.1f}%\n")
                    f.write(f"  High Overlap: {(stats['high_overlap']/total_pairs*100):.1f}%\n\n")
            
            print(f"Vertical overlap metrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"Error exporting vertical overlap metrics: {e}")

    def _export_horizontal_overlap_metrics(self, output_path: str) -> None:
        """Export horizontal overlap metrics to text file"""
        try:
            metrics_file = os.path.join(output_path, "Image_Horizontal_Overlap_Metrics.txt")
            stats = self.horizontal_overlap_stats
            
            with open(metrics_file, 'w') as f:
                f.write("HORIZONTAL OVERLAP ANALYSIS METRICS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Non-sequential image overlap statistics (between survey lines)\n\n")
                
                f.write(f"Total Pairs Checked: {stats['total_checks']}\n")
                f.write(f"Total Overlapping Pairs: {stats['total_overlap_count']}\n")
                f.write(f"Average Overlap: {stats['avg_overlap']:.2f}%\n")
                f.write(f"Median Overlap: {stats['median_overlap']:.2f}%\n")
                f.write(f"Minimum Overlap: {stats['min_overlap']:.2f}%\n")
                f.write(f"Maximum Overlap: {stats['max_overlap']:.2f}%\n")
                f.write(f"Average Sequence Distance: {stats['avg_sequence_distance']:.1f} images\n\n")
                
                f.write("OVERLAP DISTRIBUTION:\n")
                f.write(f"  Low Overlap (<10%): {stats['low_overlap']} pairs\n")
                f.write(f"  Medium Overlap (10-40%): {stats['medium_overlap']} pairs\n")
                f.write(f"  High Overlap (>40%): {stats['high_overlap']} pairs\n\n")
                
                total_pairs = stats['total_overlap_count']
                if total_pairs > 0:
                    f.write("OVERLAP PERCENTAGES:\n")
                    f.write(f"  Low Overlap: {(stats['low_overlap']/total_pairs*100):.1f}%\n")
                    f.write(f"  Medium Overlap: {(stats['medium_overlap']/total_pairs*100):.1f}%\n")
                    f.write(f"  High Overlap: {(stats['high_overlap']/total_pairs*100):.1f}%\n\n")
            
            print(f"Horizontal overlap metrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"Error exporting horizontal overlap metrics: {e}")

    def _export_overall_overlap_metrics(self, output_path: str) -> None:
        """Export overall overlap metrics to text file"""
        try:
            metrics_file = os.path.join(output_path, "Image_Overall_Overlap_Metrics.txt")
            stats = self.overall_overlap_stats
            
            with open(metrics_file, 'w') as f:
                f.write("OVERALL OVERLAP ANALYSIS METRICS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Overall overlap counts per image (all overlap types)\n\n")
                
                f.write(f"Total Pairs Checked: {stats['total_checks']}\n")
                f.write(f"Total Overlapping Pairs: {stats['total_overlaps']}\n")
                f.write(f"Average Overlap Count per Image: {stats['avg_count']:.2f}\n")
                f.write(f"Median Overlap Count: {stats['median_count']:.2f}\n")
                f.write(f"Minimum Overlap Count: {stats['min_count']}\n")
                f.write(f"Maximum Overlap Count: {stats['max_count']}\n\n")
                
                # Calculate distribution of overlap counts
                overlap_counts = [p['overlap_count'] for p in stats['polygons']]
                f.write("OVERLAP COUNT DISTRIBUTION:\n")
                
                # Create histogram bins
                bins = [0, 1, 2, 5, 10, 20, 50, max(overlap_counts)+1] if overlap_counts else [0, 1]
                for i in range(len(bins)-1):
                    count = sum(1 for x in overlap_counts if bins[i] <= x < bins[i+1])
                    if i == len(bins)-2:  # Last bin is inclusive of upper bound
                        count = sum(1 for x in overlap_counts if x >= bins[i])
                        f.write(f"  {bins[i]}+ overlaps: {count} images\n")
                    else:
                        f.write(f"  {bins[i]}-{bins[i+1]-1} overlaps: {count} images\n")
                
            print(f"Overall overlap metrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"Error exporting overall overlap metrics: {e}")

    def _detect_outliers(self, points: List[Dict], std_threshold: float = 3.0) -> List[bool]:
        """
        Detect outliers in GPS coordinates using the z-score method
        
        Args:
            points: List of dictionaries containing GPS data
            std_threshold: Number of standard deviations to use as threshold (default: 3.0)
            
        Returns:
            List of boolean values indicating if each point is an outlier
        """
        try:
            # Extract coordinates
            lats = np.array([p['latitude'] for p in points])
            lons = np.array([p['longitude'] for p in points])
            
            # Calculate z-scores for both latitude and longitude
            lat_mean = np.mean(lats)
            lon_mean = np.mean(lons)
            lat_std = np.std(lats)
            lon_std = np.std(lons)
            
            lat_z_scores = np.abs((lats - lat_mean) / lat_std)
            lon_z_scores = np.abs((lons - lon_mean) / lon_std)
            
            # Mark points as outliers if either latitude or longitude exceeds threshold
            outliers = (lat_z_scores > std_threshold) | (lon_z_scores > std_threshold)
            
            # Log outliers found
            if any(outliers):
                print("\nOutliers detected:")
                for i, (is_outlier, point) in enumerate(zip(outliers, points)):
                    if is_outlier:
                        print(f"Point {i}: lat={point['latitude']}, lon={point['longitude']}")
                        print(f"Z-scores: lat={lat_z_scores[i]:.2f}, lon={lon_z_scores[i]:.2f}")
            
            return outliers.tolist()
            
        except Exception as e:
            print(f"Error detecting outliers: {e}")
            return [False] * len(points)

    def identify_subsets(self, footprints: List[Dict], min_subset_size: int = 50) -> List[Dict]:
        """
        Identify meaningful subsets of the survey for creating zoomed maps
        
        Args:
            footprints: List of footprint data dictionaries
            min_subset_size: Minimum number of footprints to consider as a subset
            
        Returns:
            List of subset dictionaries with bounds and descriptions
        """
        if len(footprints) < min_subset_size:
            print(f"Dataset too small ({len(footprints)} images) for subset analysis (min: {min_subset_size})")
            return []
        
        try:
            # Extract coordinates for analysis
            lats = [fp['latitude'] for fp in footprints]
            lons = [fp['longitude'] for fp in footprints]
            
            # Calculate data bounds
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            
            print(f"Survey area: {lat_range:.6f}° lat × {lon_range:.6f}° lon")
            
            subsets = []
            
            # If the dataset is large enough, create meaningful subsets
            if len(footprints) >= min_subset_size * 2:
                
                # Create geographic quadrants if the survey area is large enough
                if lat_range > 0.001 and lon_range > 0.001:  # Only if area is reasonably large
                    
                    mid_lat = (min_lat + max_lat) / 2
                    mid_lon = (min_lon + max_lon) / 2
                    
                    # Define quadrant boundaries
                    quadrants = [
                        {'name': 'Northwest', 'lat_range': (mid_lat, max_lat), 'lon_range': (min_lon, mid_lon)},
                        {'name': 'Northeast', 'lat_range': (mid_lat, max_lat), 'lon_range': (mid_lon, max_lon)},
                        {'name': 'Southwest', 'lat_range': (min_lat, mid_lat), 'lon_range': (min_lon, mid_lon)},
                        {'name': 'Southeast', 'lat_range': (min_lat, mid_lat), 'lon_range': (mid_lon, max_lon)}
                    ]
                    
                    for quad in quadrants:
                        # Find footprints in this quadrant
                        quad_footprints = []
                        for fp in footprints:
                            if (quad['lat_range'][0] <= fp['latitude'] <= quad['lat_range'][1] and
                                quad['lon_range'][0] <= fp['longitude'] <= quad['lon_range'][1]):
                                quad_footprints.append(fp)
                        
                        # Only create subset if it has enough data
                        if len(quad_footprints) >= min_subset_size:
                            subset_lats = [fp['latitude'] for fp in quad_footprints]
                            subset_lons = [fp['longitude'] for fp in quad_footprints]
                            
                            subsets.append({
                                'name': f"{quad['name']} Quadrant",
                                'footprints': quad_footprints,
                                'bounds': {
                                    'min_lat': min(subset_lats),
                                    'max_lat': max(subset_lats),
                                    'min_lon': min(subset_lons),
                                    'max_lon': max(subset_lons)
                                },
                                'count': len(quad_footprints)
                            })
                
                # Create temporal subsets (first half, second half)
                if len(footprints) >= min_subset_size * 2:
                    sorted_footprints = sorted(footprints, key=lambda x: x['filename'])
                    
                    mid_point = len(sorted_footprints) // 2
                    
                    # First half
                    first_half = sorted_footprints[:mid_point + min_subset_size//2]
                    if len(first_half) >= min_subset_size:
                        first_lats = [fp['latitude'] for fp in first_half]
                        first_lons = [fp['longitude'] for fp in first_half]
                        
                        subsets.append({
                            'name': 'First Half (Temporal)',
                            'footprints': first_half,
                            'bounds': {
                                'min_lat': min(first_lats),
                                'max_lat': max(first_lats),
                                'min_lon': min(first_lons),
                                'max_lon': max(first_lons)
                            },
                            'count': len(first_half)
                        })
                    
                    # Second half
                    second_half = sorted_footprints[mid_point - min_subset_size//2:]
                    if len(second_half) >= min_subset_size:
                        second_lats = [fp['latitude'] for fp in second_half]
                        second_lons = [fp['longitude'] for fp in second_half]
                        
                        subsets.append({
                            'name': 'Second Half (Temporal)',
                            'footprints': second_half,
                            'bounds': {
                                'min_lat': min(second_lats),
                                'max_lat': max(second_lats),
                                'min_lon': min(second_lons),
                                'max_lon': max(second_lons)
                            },
                            'count': len(second_half)
                        })
            
            # If no meaningful subsets were created, create a single subset of the densest area
            if not subsets and len(footprints) >= min_subset_size:
                # Find the center of the survey
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)
                
                # Calculate distances from center
                distances = []
                for i, fp in enumerate(footprints):
                    dist = ((fp['latitude'] - center_lat)**2 + (fp['longitude'] - center_lon)**2)**0.5
                    distances.append((dist, i, fp))
                
                # Sort by distance and take the closest points
                distances.sort()
                central_footprints = [fp for _, _, fp in distances[:min_subset_size * 2]]
                
                if central_footprints:
                    central_lats = [fp['latitude'] for fp in central_footprints]
                    central_lons = [fp['longitude'] for fp in central_footprints]
                    
                    subsets.append({
                        'name': 'Central Dense Area',
                        'footprints': central_footprints,
                        'bounds': {
                            'min_lat': min(central_lats),
                            'max_lat': max(central_lats),
                            'min_lon': min(central_lons),
                            'max_lon': max(central_lons)
                        },
                        'count': len(central_footprints)
                    })
            
            print(f"Identified {len(subsets)} subsets for zoomed analysis:")
            for subset in subsets:
                print(f"  - {subset['name']}: {subset['count']} images")
            
            return subsets
            
        except Exception as e:
            print(f"Error identifying subsets: {e}")
            print(traceback.format_exc())
            return []

    def create_zoomed_maps(self, footprints: List[Dict], subsets: List[Dict], output_path: str):
        """
        Create zoomed maps for identified subsets
        
        Args:
            footprints: Full list of footprint data
            subsets: List of subset dictionaries from identify_subsets
            output_path: Directory to save output files
        """
        if not subsets:
            print("No subsets available for zoomed maps")
            return
        
        try:
            print(f"Creating zoomed maps for {len(subsets)} subsets...")
            
            for subset in subsets:
                subset_name = subset['name'].replace(' ', '_').replace('(', '').replace(')', '')
                
                print(f"Creating zoomed maps for {subset['name']} ({subset['count']} images)")
                
                # Create zoomed footprint map
                self._create_zoomed_footprint_map(subset, output_path, f"Image_Footprint_Map_Zoomed_{subset_name}.png")
                
                # Create zoomed overlap maps if we have overlap data
                if hasattr(self, 'vertical_overlap_stats') and self.vertical_overlap_stats:
                    self._create_zoomed_vertical_overlap_map(subset, output_path, f"Image_Vertical_Overlap_Map_Zoomed_{subset_name}.png")
                
                if hasattr(self, 'horizontal_overlap_stats') and self.horizontal_overlap_stats:
                    self._create_zoomed_horizontal_overlap_map(subset, output_path, f"Image_Horizontal_Overlap_Map_Zoomed_{subset_name}.png")
                
                if hasattr(self, 'overall_overlap_stats') and self.overall_overlap_stats:
                    self._create_zoomed_overall_overlap_map(subset, output_path, f"Image_Overall_Overlap_Map_Zoomed_{subset_name}.png")
        
        except Exception as e:
            print(f"Error creating zoomed maps: {e}")
            print(traceback.format_exc())

    def _create_zoomed_footprint_map(self, subset: Dict, output_path: str, filename: str):
        """Create a zoomed footprint map for a subset"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot footprints for this subset
            for fp in subset['footprints']:
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.3, edgecolor='blue', facecolor='skyblue')
                ax.plot(fp['longitude'], fp['latitude'], 'r.', markersize=3)
            
            # Set appropriate bounds with some padding
            bounds = subset['bounds']
            lat_range = bounds['max_lat'] - bounds['min_lat']
            lon_range = bounds['max_lon'] - bounds['min_lon']
            padding = max(lat_range, lon_range) * 0.1
            
            ax.set_xlim(bounds['min_lon'] - padding, bounds['max_lon'] + padding)
            ax.set_ylim(bounds['min_lat'] - padding, bounds['max_lat'] + padding)
            
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Image Footprints - {subset["name"]} ({subset["count"]} images)')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Zoomed footprint map saved: {filename}")
            
        except Exception as e:
            print(f"Error creating zoomed footprint map: {e}")

    def _create_zoomed_vertical_overlap_map(self, subset: Dict, output_path: str, filename: str):
        """Create a zoomed vertical overlap map for a subset"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset filenames for filtering overlaps
            subset_filenames = set(fp['filename'] for fp in subset['footprints'])
            
            # Plot subset footprints with low alpha
            for fp in subset['footprints']:
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.1, edgecolor='gray', facecolor='lightgray')
            
            # Colors for overlap categories
            colors = {'low': '#FF4040', 'medium': '#FFFF40', 'high': '#40FF40'}
            
            # Plot overlaps that involve footprints in this subset
            overlap_count = 0
            for overlap in self.vertical_overlap_stats['overlap_data']:
                if (overlap['current_fp']['filename'] in subset_filenames or 
                    overlap['next_fp']['filename'] in subset_filenames):
                    
                    if hasattr(overlap['intersection'], 'geoms'):
                        for geom in overlap['intersection'].geoms:
                            if geom.geom_type == 'Polygon':
                                coords = list(geom.exterior.coords)
                                polygon = np.array(coords)
                                ax.fill(polygon[:, 0], polygon[:, 1], 
                                       color=colors[overlap['category']], alpha=0.7)
                    else:
                        coords = list(overlap['intersection'].exterior.coords)
                        polygon = np.array(coords)
                        ax.fill(polygon[:, 0], polygon[:, 1], 
                               color=colors[overlap['category']], alpha=0.7)
                    overlap_count += 1
            
            # Set bounds
            bounds = subset['bounds']
            lat_range = bounds['max_lat'] - bounds['min_lat']
            lon_range = bounds['max_lon'] - bounds['min_lon']
            padding = max(lat_range, lon_range) * 0.1
            
            ax.set_xlim(bounds['min_lon'] - padding, bounds['max_lon'] + padding)
            ax.set_ylim(bounds['min_lat'] - padding, bounds['max_lat'] + padding)
            
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f' ve rtical Overlap - {subset["name"]} ({overlap_count} overlaps)')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Zoomed vertical overlap map saved: {filename}")
            
        except Exception as e:
            print(f"Error creating zoomed vertical overlap map: {e}")

    def _create_zoomed_horizontal_overlap_map(self, subset: Dict, output_path: str, filename: str):
        """Create a zoomed horizontal overlap map for a subset"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset filenames for filtering overlaps
            subset_filenames = set(fp['filename'] for fp in subset['footprints'])
            
            # Plot subset footprints with low alpha
            for fp in subset['footprints']:
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.1, edgecolor='gray', facecolor='lightgray')
            
            # Colors for overlap categories
            colors = {'low': '#FF8080', 'medium': '#FFFF80', 'high': '#80FF80'}
            
            # Plot overlaps that involve footprints in this subset
            overlap_count = 0
            for overlap in self.horizontal_overlap_stats['overlap_data']:
                if (overlap['current_fp']['filename'] in subset_filenames or 
                    overlap['next_fp']['filename'] in subset_filenames):
                    
                    if hasattr(overlap['intersection'], 'geoms'):
                        for geom in overlap['intersection'].geoms:
                            if geom.geom_type == 'Polygon':
                                coords = list(geom.exterior.coords)
                                polygon = np.array(coords)
                                ax.fill(polygon[:, 0], polygon[:, 1], 
                                       color=colors[overlap['category']], alpha=0.7)
                    else:
                        coords = list(overlap['intersection'].exterior.coords)
                        polygon = np.array(coords)
                        ax.fill(polygon[:, 0], polygon[:, 1], 
                               color=colors[overlap['category']], alpha=0.7)
                    overlap_count += 1
            
            # Set bounds
            bounds = subset['bounds']
            lat_range = bounds['max_lat'] - bounds['min_lat']
            lon_range = bounds['max_lon'] - bounds['min_lon']
            padding = max(lat_range, lon_range) * 0.1
            
            ax.set_xlim(bounds['min_lon'] - padding, bounds['max_lon'] + padding)
            ax.set_ylim(bounds['min_lat'] - padding, bounds['max_lat'] + padding)
            
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Horizontal Overlap - {subset["name"]} ({overlap_count} overlaps)')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Zoomed horizontal overlap map saved: {filename}")
            
        except Exception as e:
            print(f"Error creating zoomed horizontal overlap map: {e}")

    def _create_zoomed_overall_overlap_map(self, subset: Dict, output_path: str, filename: str):
        """Create a zoomed overall overlap map for a subset"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset filenames for filtering
            subset_filenames = set(fp['filename'] for fp in subset['footprints'])
            
            # Find subset polygons from overall overlap data
            subset_polygons = []
            if hasattr(self, 'overall_overlap_stats') and self.overall_overlap_stats and 'polygons' in self.overall_overlap_stats:
                for poly_data in self.overall_overlap_stats['polygons']:
                    if poly_data['footprint']['filename'] in subset_filenames:
                        subset_polygons.append(poly_data)
            
            # Create color map based on overlap counts
            if subset_polygons:
                overlap_counts = [p['overlap_count'] for p in subset_polygons]
                max_count = max(overlap_counts) if overlap_counts else 1
                
                # Plot each footprint colored by its overlap count
                for poly_data in subset_polygons:
                    # Normalize color intensity by overlap count
                    intensity = poly_data['overlap_count'] / max_count if max_count > 0 else 0
                    color = plt.cm.viridis(intensity)
                    
                    # Plot the polygon
                    if poly_data['polygon'].geom_type == 'Polygon':
                        coords = list(poly_data['polygon'].exterior.coords)
                        polygon = np.array(coords)
                        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Set bounds
            bounds = subset['bounds']
            lat_range = bounds['max_lat'] - bounds['min_lat']
            lon_range = bounds['max_lon'] - bounds['min_lon']
            padding = max(lat_range, lon_range) * 0.1
            
            ax.set_xlim(bounds['min_lon'] - padding, bounds['max_lon'] + padding)
            ax.set_ylim(bounds['min_lat'] - padding, bounds['max_lat'] + padding)
            
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Overall Overlap Count - {subset["name"]} ({len(subset_polygons)} images)')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Add colorbar
            if subset_polygons and max_count > 0:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_count))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Number of Overlapping Images')
            
            plt.tight_layout()
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Zoomed overall overlap map saved: {filename}")
            
        except Exception as e:
            print(f"Error creating zoomed overall overlap map: {e}")

    def create_footprint_map_from_csv(self, csv_path: str, output_folder: str, 
                                     filename: str = "Image_Footprints_Map.png") -> Optional[str]:
        """
        Create footprint map from the master CSV file
        
        Args:
            csv_path: Path to the master Image_Metrics.csv file
            output_folder: Directory to save the footprint map
            filename: Name for the output file
            
        Returns:
            Path to created footprint map file, or None if failed
        """
        try:
            import pandas as pd
            
            # Check if CSV exists, if not try to create it
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                print("Attempting to create CSV file...")
                
                # Try to create the CSV by finding input folder from CSV path
                csv_dir = os.path.dirname(csv_path)
                possible_input_folders = [
                    os.path.join(csv_dir, '..', 'input'),
                    os.path.join(csv_dir, 'input'),
                    os.path.join(csv_dir, '..', 'test_images_auv_proc'),
                    os.path.join(csv_dir, 'test_images_auv_proc'),
                    csv_dir
                ]
                
                input_folder = None
                for folder in possible_input_folders:
                    folder = os.path.normpath(folder)
                    if os.path.exists(folder) and any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) 
                                                    for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))):
                        input_folder = folder
                        break
                
                if input_folder:
                    from models.metrics import Metrics
                    metrics = Metrics()
                    csv_created = metrics.create_image_metrics_csv(input_folder, csv_dir)
                    if not csv_created:
                        print("Failed to create CSV file")
                        return None
                else:
                    print("Could not find input folder to create CSV")
                    return None
            
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_columns = ['filename', 'latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing required columns in CSV: {missing_columns}")
                return None
            
            # Filter out rows with missing GPS data
            df_filtered = df.dropna(subset=['latitude', 'longitude'])
            
            if len(df_filtered) == 0:
                print("No valid GPS data found in CSV")
                return None
            
            # Convert DataFrame to GPS data format expected by existing methods
            gps_data = []
            for _, row in df_filtered.iterrows():
                gps_point = {
                    'filename': row['filename'],
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'altitude': float(row['altitude']) if pd.notna(row.get('altitude')) else 5.0,  # Default altitude if missing
                    'heading': float(row['heading']) if pd.notna(row.get('heading')) else 0.0,
                    'datetime': row.get('datetime_original', ''),
                    'processing_type': row.get('processing_type', 'unknown')
                }
                gps_data.append(gps_point)
            
            print(f"Loaded {len(gps_data)} GPS points from CSV for footprint analysis")
            
            # Use the existing create_footprint_map method
            result = self.create_footprint_map(
                gps_data=gps_data,
                output_path=output_folder,
                nav_file_path=None,  # CSV already contains heading data
                filename=filename
            )
            
            # Update the master CSV with footprint results if successful
            if result:
                self._update_master_csv_with_footprint_results(csv_path, gps_data)
            
            return result
            
        except Exception as e:
            print(f"Error creating footprint map from CSV: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _update_master_csv_with_footprint_results(self, csv_path: str, gps_data: list) -> None:
        """
        Update the master CSV file with footprint calculation results
        
        Args:
            csv_path: Path to the master CSV file
            gps_data: List of GPS data points with footprint information
        """
        try:
            import pandas as pd
            
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            # Create mapping from filename to footprint data
            footprint_data = {}
            
            # Add footprint dimensions columns if they don't exist
            new_columns = ['footprint_width', 'footprint_height', 'footprint_area', 
                          'vertical_overlap', 'horizontal_overlap', 'overall_overlap']
            
            for col in new_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Update footprint data from GPS data
            for point in gps_data:
                filename = point['filename']
                if filename not in footprint_data:
                    footprint_data[filename] = {}
                
                # Calculate footprint dimensions based on altitude
                altitude = point.get('altitude', 5.0)
                
                # Standard camera parameters for AUV systems
                sensor_width = 13.2  # mm (typical APS-C sensor)
                sensor_height = 8.8  # mm
                focal_length = 12.0  # mm
                
                # Calculate footprint size
                footprint_width = (sensor_width * altitude) / focal_length
                footprint_height = (sensor_height * altitude) / focal_length
                footprint_area = footprint_width * footprint_height
                
                footprint_data[filename]['footprint_width'] = footprint_width
                footprint_data[filename]['footprint_height'] = footprint_height
                footprint_data[filename]['footprint_area'] = footprint_area
            
            # Add overlap data if available
            if hasattr(self, 'vertical_overlap_stats') and self.vertical_overlap_stats:
                for overlap in self.vertical_overlap_stats.get('overlap_data', []):
                    filename1 = overlap.get('filename1', '')
                    filename2 = overlap.get('filename2', '')
                    overlap_pct = overlap.get('overlap_percentage', 0.0)
                    
                    if filename1 in footprint_data:
                        footprint_data[filename1]['vertical_overlap'] = overlap_pct
                    if filename2 in footprint_data:
                        footprint_data[filename2]['vertical_overlap'] = overlap_pct
            
            if hasattr(self, 'horizontal_overlap_stats') and self.horizontal_overlap_stats:
                for overlap in self.horizontal_overlap_stats.get('overlap_data', []):
                    filename1 = overlap.get('filename1', '')
                    filename2 = overlap.get('filename2', '')
                    overlap_pct = overlap.get('overlap_percentage', 0.0)
                    
                    if filename1 in footprint_data:
                        footprint_data[filename1]['horizontal_overlap'] = overlap_pct
                    if filename2 in footprint_data:
                        footprint_data[filename2]['horizontal_overlap'] = overlap_pct
            
            if hasattr(self, 'overall_overlap_stats') and self.overall_overlap_stats:
                for overlap in self.overall_overlap_stats.get('overlap_data', []):
                    filename = overlap.get('filename', '')
                    overlap_count = overlap.get('overlap_count', 0)
                    
                    if filename in footprint_data:
                        footprint_data[filename]['overall_overlap'] = overlap_count
            
            # Update the DataFrame
            for idx, row in df.iterrows():
                filename = row['filename']
                if filename in footprint_data:
                    for col, value in footprint_data[filename].items():
                        if value is not None:
                            df.at[idx, col] = value
            
            # Save updated CSV
            df.to_csv(csv_path, index=False)
            print(f"Updated master CSV with footprint results: {csv_path}")
            
        except Exception as e:
            print(f"Error updating master CSV with footprint results: {e}")
            import traceback
            traceback.print_exc()