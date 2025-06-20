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
                            filename: str = "image_footprints_map.png") -> Optional[str]:
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
                
            # If RTree failed or wasn't available, use brute force comparison
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
                            
                    # Show progress for large datasets
                    if i % chunk_size == 0 and i > 0:
                        print(f"Processed {i}/{len(polygons)} images, found {total_overlaps} overlaps so far")
            
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
                                  filename: str = "vertical_overlap_map.png") -> Optional[str]:
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
                                   filename: str = "horizontal_overlap_map.png") -> Optional[str]:
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
                                filename: str = "overall_overlap_map.png") -> Optional[str]:
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
                    shapefile = os.path.join(output_path, "footprints.shp")
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
                        poly_shapefile = os.path.join(output_path, "footprint_polygons.shp")
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

    def identify_subsets(self, footprints: List[Dict]) -> List[Dict]:
        """
        Identify meaningful subsets of data for zoomed maps
        
        Args:
            footprints: List of dictionaries with footprint data
            
        Returns:
            List of subset definitions
        """
        if len(footprints) < 10:
            print("Not enough footprints to identify meaningful subsets")
            return []
            
        try:
            print("Identifying meaningful subsets for zoomed maps...")
            
            # Get all coordinates
            lats = [fp['latitude'] for fp in footprints]
            lons = [fp['longitude'] for fp in footprints]
            
            # Calculate overall bounds
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Calculate range
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            
            # Identify clusters using DBSCAN if we have enough points
            try:
                from sklearn.cluster import DBSCAN
                import numpy as np
                
                # Prepare coordinates for clustering
                coords = np.array([[lat, lon] for lat, lon in zip(lats, lons)])
                
                # Epsilon is a fraction of the coordinate range
                eps = min(lat_range, lon_range) * 0.05
                
                # Run DBSCAN clustering
                clustering = DBSCAN(eps=eps, min_samples=5).fit(coords)
                
                # Get cluster labels (-1 means noise)
                labels = clustering.labels_
                
                # Count number of clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                print(f"DBSCAN identified {n_clusters} clusters")
                
                # Create subset for each cluster with at least 10 points
                subsets = []
                
                for cluster_id in range(n_clusters):
                    # Get indices of points in this cluster
                    indices = [i for i, label in enumerate(labels) if label == cluster_id]
                    
                    if len(indices) >= 10:
                        # Calculate bounds for this cluster
                        cluster_lats = [lats[i] for i in indices]
                        cluster_lons = [lons[i] for i in indices]
                        
                        # Add padding around cluster
                        padding = 0.05  # 5% padding
                        c_min_lat = min(cluster_lats) - lat_range * padding
                        c_max_lat = max(cluster_lats) + lat_range * padding
                        c_min_lon = min(cluster_lons) - lon_range * padding
                        c_max_lon = max(cluster_lons) + lon_range * padding
                        
                        # Create subset definition
                        subset = {
                            'name': f"Cluster_{cluster_id}",
                            'indices': indices,
                            'bounds': [c_min_lon, c_min_lat, c_max_lon, c_max_lat],
                            'count': len(indices)
                        }
                        subsets.append(subset)
                
                return subsets
                
            except ImportError:
                print("scikit-learn not available, falling back to grid-based subsets")
            
            # If DBSCAN fails or isn't available, fall back to simple grid-based subsetting
            
            # Divide into quadrants - four equal parts
            quadrants = [
                {
                    'name': "Northwest",
                    'bounds': [min_lon, (min_lat + max_lat)/2, (min_lon + max_lon)/2, max_lat]
                },
                {
                    'name': "Northeast",
                    'bounds': [(min_lon + max_lon)/2, (min_lat + max_lat)/2, max_lon, max_lat]
                },
                {
                    'name': "Southwest",
                    'bounds': [min_lon, min_lat, (min_lon + max_lon)/2, (min_lat + max_lat)/2]
                },
                {
                    'name': "Southeast",
                    'bounds': [(min_lon + max_lon)/2, min_lat, max_lon, (min_lat + max_lat)/2]
                }
            ]
            
            # Filter quadrants to only include those with at least 10 footprints
            valid_quadrants = []
            
            for q in quadrants:
                # Count footprints in this quadrant
                bounds = q['bounds']
                min_lon, min_lat, max_lon, max_lat = bounds
                
                # Find footprint indices in this quadrant
                indices = [
                    i for i, fp in enumerate(footprints)
                    if min_lon <= fp['longitude'] <= max_lon and min_lat <= fp['latitude'] <= max_lat
                ]
                
                if len(indices) >= 10:
                    q['indices'] = indices
                    q['count'] = len(indices)
                    valid_quadrants.append(q)
            print(f"Identified {len(valid_quadrants)} valid quadrant subsets")
            
            return valid_quadrants
            
        except Exception as e:
            print(f"Error identifying subsets: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    def create_zoomed_maps(self, footprints: List[Dict], subsets: List[Dict], output_path: str) -> None:
        """
        Create zoomed maps for subsets of data
        
        Args:
            footprints: List of dictionaries with footprint data
            subsets: List of subset definitions
            output_path: Directory to save the output files
        """
        if not subsets:
            return
        
        print(f"Creating {len(subsets)} zoomed maps...")
        
        # Create zoomed footprint maps
        for i, subset in enumerate(subsets):
            try:
                subset_name = subset['name']
                subset_indices = subset['indices']
                bounds = subset['bounds']
                
                print(f"Creating zoomed map for subset: {subset_name} with {len(subset_indices)} footprints")
                
                # Create zoomed footprint map
                self._create_zoomed_footprint_map(footprints, subset, output_path)
                
                # Create zoomed overlap maps if data is available
                if hasattr(self, 'vertical_overlap_stats') and self.vertical_overlap_stats:
                    self._create_zoomed_vertical_overlap_map(footprints, subset, output_path)
                
                if hasattr(self, 'horizontal_overlap_stats') and self.horizontal_overlap_stats:
                    self._create_zoomed_horizontal_overlap_map(footprints, subset, output_path)
                    
                if hasattr(self, 'overall_overlap_stats') and self.overall_overlap_stats:
                    self._create_zoomed_overall_overlap_map(footprints, subset, output_path)
                    
            except Exception as e:
                print(f"Error creating zoomed map for subset {i}: {e}")
                import traceback
                print(traceback.format_exc())

    def _create_zoomed_footprint_map(self, footprints: List[Dict], subset: Dict, output_path: str) -> Optional[str]:
        """Create zoomed footprint map for a data subset"""
        try:
            # Use non-interactive backend to avoid threading issues
            import matplotlib
            matplotlib.use('Agg')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset info
            subset_name = subset['name']
            subset_indices = subset['indices']
            bounds = subset['bounds']
            
            # Plot each footprint in the subset
            for idx in subset_indices:
                fp = footprints[idx]
                
                # Plot polygon with low alpha for overlap visibility
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.3, edgecolor='blue', facecolor='skyblue')
                
                # Plot center point
                ax.plot(fp['longitude'], fp['latitude'], 'r.', markersize=2)
            
            # Set the plot bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Image Footprints - {subset_name}')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Round to 2 decimal places
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            
            # Increase font sizes
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(item.get_fontsize() + 2)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(output_path, f"footprints_zoom_{subset_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Zoomed footprint map saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating zoomed footprint map: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _create_zoomed_vertical_overlap_map(self, footprints: List[Dict], subset: Dict, output_path: str) -> Optional[str]:
        """Create zoomed vertical overlap map for a data subset"""
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset info
            subset_name = subset['name']
            subset_indices = set(subset['indices'])  # Convert to set for faster lookups
            bounds = subset['bounds']
            
            # Setup colors
            low_color = '#FF4040'      # Red
            medium_color = '#FFFF40'   # Yellow
            high_color = '#40FF40'     # Green
            
            # Plot footprints in subset with very low alpha for context
            for idx in subset_indices:
                fp = footprints[idx]
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.05, edgecolor='gray', facecolor='whitesmoke')
            
            # Plot relevant overlaps (where both images are in the subset)
            overlap_in_subset = []
            
            for overlap in self.vertical_overlap_stats['overlap_data']:
                # Find the indices of both footprints
                current_idx = footprints.index(overlap['current_fp'])
                next_idx = footprints.index(overlap['next_fp'])
                
                # Only include if both are in the subset
                if current_idx in subset_indices and next_idx in subset_indices:
                    overlap_in_subset.append(overlap)
                    
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
                                fontsize=10,
                                ha='center', va='center', color='black',
                                bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
                        
                    elif intersection.geom_type == 'MultiPolygon':
                        for poly in intersection.geoms:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, alpha=0.7, edgecolor='none', facecolor=color)
            
            # Set the plot bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Vertical Overlap - {subset_name}')
            
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
            output_file = os.path.join(output_path, f"vertical_overlap_zoom_{subset_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Zoomed vertical overlap map saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating zoomed vertical overlap map: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _create_zoomed_horizontal_overlap_map(self, footprints: List[Dict], subset: Dict, output_path: str) -> Optional[str]:
        """Create zoomed horizontal overlap map for a data subset"""
        # Implementation very similar to _create_zoomed_vertical_overlap_map
        # but using horizontal_overlap_stats and appropriate colors/labels
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset info
            subset_name = subset['name']
            subset_indices = set(subset['indices'])
            bounds = subset['bounds']
            
            # Setup colors
            low_color = '#FF4040'      # Red
            medium_color = '#FFA500'   # Orange
            high_color = '#40FF40'     # Green
            
            # Plot footprints in subset with very low alpha for context
            for idx in subset_indices:
                fp = footprints[idx]
                polygon = np.array(fp['vertices'])
                ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.05, edgecolor='gray', facecolor='whitesmoke')
            
            # Plot relevant overlaps (where both images are in the subset)
            overlap_in_subset = []
            
            for overlap in self.horizontal_overlap_stats['overlap_data']:
                # Find the indices of both footprints
                current_idx = footprints.index(overlap['current_fp'])
                next_idx = footprints.index(overlap['next_fp'])
                
                # Only include if both are in the subset
                if current_idx in subset_indices and next_idx in subset_indices:
                    overlap_in_subset.append(overlap)
                    
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
                                fontsize=10,
                                ha='center', va='center', color='black',
                                bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round,pad=0.2'))
                        
                    elif intersection.geom_type == 'MultiPolygon':
                        for poly in intersection.geoms:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, alpha=0.7, edgecolor='none', facecolor=color)
            
            # Set the plot bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Horizontal Overlap - {subset_name}')
            
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
            output_file = os.path.join(output_path, f"horizontal_overlap_zoom_{subset_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Zoomed horizontal overlap map saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating zoomed horizontal overlap map: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _create_zoomed_overall_overlap_map(self, footprints: List[Dict], subset: Dict, output_path: str) -> Optional[str]:
        """Create zoomed overall overlap map for a data subset"""
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.colors as mcolors
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get subset info
            subset_name = subset['name']
            subset_indices = set(subset['indices'])
            bounds = subset['bounds']
            
            # Create a rainbow colormap from blue/purple (high) to red (low)
            cmap = plt.cm.rainbow_r
            
            # Filter polygons to just those in the subset
            subset_polygons = [p for p in self.overall_overlap_stats['polygons'] if p['index'] in subset_indices]
            
            # Get min and max overlap count in the subset
            overlap_counts = [p['overlap_count'] for p in subset_polygons]
            min_count = min(overlap_counts) if overlap_counts else 0
            max_count = max(overlap_counts) if overlap_counts else 0
            
            # Create a normalization that handles the case where min=max
            if min_count == max_count:
                norm = mcolors.Normalize(vmin=min_count-0.5, vmax=max_count+0.5)
            else:
                norm = mcolors.Normalize(vmin=min_count, vmax=max_count)
            
            # Plot each footprint with color based on overlap count
            for poly_data in subset_polygons:
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
                        fontsize=10,
                        ha='center', va='center', color='white',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round,pad=0.2'))
            
            # Set the plot bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            
            # Set labels and title
            ax.set_xlabel('Longitude (Decimal Degrees)')
            ax.set_ylabel('Latitude (Decimal Degrees)')
            ax.set_title(f'Overall Image Overlap - {subset_name}')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # ScalarMappable needs an array for the colorbar
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Number of Overlapping Images')
            cbar.ax.tick_params(labelsize=10)  # Increase tick label size
            
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
            output_file = os.path.join(output_path, f"overall_overlap_zoom_{subset_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Zoomed overall overlap map saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating zoomed overall overlap map: {e}")
            import traceback
            print(traceback.format_exc())
            return None

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