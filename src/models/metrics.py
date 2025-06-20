import os
import re
from typing import Dict, List, Tuple, Callable, Optional
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class Metrics:
    def __init__(self, altitude_threshold=9.0):
        # Initialize counters
        self.processed_count = 0
        self.processed_size = 0
        self.raw_count = 0
        self.raw_size = 0
        self.other_count = 0
        self.other_size = 0
        
        # Store GPS data for mapping
        self.gps_data = []
        
        # Save the altitude threshold
        self.altitude_threshold = altitude_threshold
        
        # Initialize overlap statistics (will be set from footprint_map if available)
        self.vertical_overlap_stats = None
        self.horizontal_overlap_stats = None
        self.overall_overlap_stats = None
        
        # Track images with issues
        self.no_gps_images = []
        self.above_threshold_images = []
        self.above_threshold_values = {}
        
        # Patterns for recognizing files
        self.raw_pattern = re.compile(r'ESC_stills_raw_PPS_')
        self.processed_pattern = re.compile(r'ESC_stills_processed_PPS_')
    
    def analyze_directory(self, input_folder: str, progress_callback: Callable = None, extract_gps: bool = False) -> Tuple[int, List[str]]:
        """
        Analyze the input directory and count/measure image files
        
        Args:
            input_folder: Path to input directory
            progress_callback: Function to call with progress updates (receives percentage and message)
            extract_gps: Whether to extract GPS data for mapping
            
        Returns:
            Tuple of (total_files_processed, results_list)
        """
        # Reset metrics
        self.processed_count = 0
        self.processed_size = 0
        self.raw_count = 0
        self.raw_size = 0
        self.other_count = 0
        self.other_size = 0
        self.gps_data = []
        
        # Check if PIL is available for GPS extraction
        if extract_gps and not PIL_AVAILABLE:
            if progress_callback:
                progress_callback(0, "Warning: PIL/Pillow library not available. GPS data extraction disabled.")
            extract_gps = False
        
        # Count total files for progress tracking
        total_files = 0
        for root, _, files in os.walk(input_folder):
            total_files += len(files)
            
        if progress_callback:
            progress_callback(0, f"Found {total_files} files to analyze...")
        
        processed_files = 0
        
        # Walk through all files in the input folder
        for root, _, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                # Update progress
                processed_files += 1
                if progress_callback:
                    progress = (processed_files / total_files) * 100 if total_files > 0 else 0
                    
                    # Only report progress on milestones to avoid GUI flooding
                    if processed_files == 1 or processed_files == total_files or processed_files % 1000 == 0:
                        progress_callback(progress, f"Analyzed {processed_files} of {total_files} files...")
                
                # Determine file type
                file_type = None
                if self.processed_pattern.search(file):
                    self.processed_count += 1
                    self.processed_size += file_size
                    file_type = "processed"
                    # Report milestones
                    if self.processed_count % 100 == 0 and progress_callback:
                        progress_callback(progress, f"Found {self.processed_count} processed images so far...")
                    
                    # Extract GPS data from processed images only
                    if extract_gps and file.lower().endswith(('.jpg', '.jpeg', '.tiff', '.tif', '.png')):
                        try:
                            gps_info = self.extract_gps_from_image(file_path)
                            if gps_info:
                                gps_info['file_type'] = file_type
                                gps_info['filename'] = file
                                self.gps_data.append(gps_info)
                                if len(self.gps_data) % 10 == 0 and progress_callback:
                                    progress_callback(progress, f"Extracted GPS data from {len(self.gps_data)} processed images so far...")
                        except Exception as e:
                            if progress_callback:
                                progress_callback(progress, f"Could not extract GPS data from {file}: {str(e)}")
                
                elif self.raw_pattern.search(file):
                    self.raw_count += 1
                    self.raw_size += file_size
                    file_type = "raw"
                    # Report milestones
                    if self.raw_count % 100 == 0 and progress_callback:
                        progress_callback(progress, f"Found {self.raw_count} raw images so far...")
                    # Note: We no longer extract GPS data from raw images
                
                else:
                    self.other_count += 1
                    self.other_size += file_size
                    file_type = "other"
        
        # Generate results
        results = self.get_summary_report()
        return processed_files, results
                
    def get_summary(self) -> Dict:
        """Return metrics dictionary"""
        return {
            "Processed Images": self.processed_count,
            "Processed Size": self.processed_size,
            "Raw Images": self.raw_count,
            "Raw Size": self.raw_size,
            "Other Files": self.other_count,
            "Other Size": self.other_size,
            "Total Images": self.processed_count + self.raw_count,
            "Total Files": self.processed_count + self.raw_count + self.other_count,
            "Total Size": self.processed_size + self.raw_size + self.other_size,
            "Images with GPS data": len(self.gps_data)
        }
    
    def get_summary_report(self) -> List[str]:
        """Return formatted summary as list of strings"""
        # Format sizes for display
        processed_size_hr = self.format_size(self.processed_size)
        raw_size_hr = self.format_size(self.raw_size)
        other_size_hr = self.format_size(self.other_size)
        total_size_hr = self.format_size(self.processed_size + self.raw_size + self.other_size)
        
        report = [
            "Image Analysis Summary:",
            "-----------------------",
            f"Processed Still Images: {self.processed_count}",
            f"Processed Stills Size: {processed_size_hr}",
            f"Raw Images: {self.raw_count}",
            f"Raw Images Size: {raw_size_hr}",
            f"Other Files: {self.other_count}",
            f"Other Files Size: {other_size_hr}",
            f"Total Files: {self.processed_count + self.raw_count + self.other_count}",
            f"Total Size: {total_size_hr}"
        ]
        
        # Add GPS data summary if available
        if self.gps_data:
            altitude_stats = self.get_altitude_statistics(threshold=self.altitude_threshold)
            
            if altitude_stats:
                # Add altitude statistics to the report
                report.extend([
                    "",
                    "Altitude Analysis (Processed Images Only):",
                    "----------------------------------------",
                    f"Images with GPS data: {len(self.gps_data)} of {self.processed_count} processed images",
                    f"Images below {self.altitude_threshold:.1f}m: {altitude_stats['below_threshold']} ({altitude_stats['below_percent']:.1f}%)",
                    f"Images above {self.altitude_threshold:.1f}m: {altitude_stats['above_threshold']} ({altitude_stats['above_percent']:.1f}%)",
                    f"Altitude range: {altitude_stats['min']:.2f}m to {altitude_stats['max']:.2f}m",
                    f"Average altitude: {altitude_stats['avg']:.2f}m",
                    f"Median altitude: {altitude_stats['median']:.2f}m",
                    f"Standard deviation: {altitude_stats['std_dev']:.2f}m"
                ])
            else:
                report.extend([
                    "",
                    f"Processed Images with GPS data: {len(self.gps_data)}",
                    "(No altitude data available in the GPS information)"
                ])
        
             
        return report
    
    def extract_gps_from_image(self, image_path: str) -> Optional[Dict]:
        """
        Extract GPS information and camera parameters from image EXIF data
        and/or navigation data
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with location and camera parameters if available, None otherwise
        """
        if not PIL_AVAILABLE:
            return None
            
        try:
            with Image.open(image_path) as img:
                # Get image dimensions directly from the image
                width, height = img.size
                
                # Start with base info including dimensions and filename
                result = {
                    'width': width,
                    'height': height,
                    'filename': os.path.basename(image_path),
                    'file_path': image_path
                }
                
                # Extract date/time for navigation data matching
                datetime_str = None
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif_data = {
                        TAGS.get(tag, tag): value
                        for tag, value in img._getexif().items()
                    }
                    
                    # Extract camera parameters
                    for tag in ['ExposureTime', 'FNumber', 'FocalLength', 'SubjectDistance']:
                        if tag in exif_data:
                            value = exif_data[tag]
                            if isinstance(value, tuple) and len(value) == 2:
                                result[tag] = round(value[0] / value[1], 1)
                            else:
                                result[tag] = value
                
                    # DateTime - extract for navigation matching
                    for dt_tag in ['DateTimeOriginal', 'DateTime']:
                        if dt_tag in exif_data and exif_data[dt_tag]:
                            result['DateTime'] = exif_data[dt_tag]
                            datetime_str = exif_data[dt_tag]
                            break
                
                # Process GPS data from EXIF if available - GET LAT/LON ONLY
                if 'GPSInfo' in exif_data:
                    gps_info = {
                        GPSTAGS.get(tag, tag): value
                        for tag, value in exif_data['GPSInfo'].items()
                    }
                    
                    # Extract latitude
                    if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                        lat = self._convert_to_degrees(gps_info['GPSLatitude'])
                        if gps_info['GPSLatitudeRef'] == 'S':
                            lat = -lat
                        result['latitude'] = lat
                    
                    # Extract longitude
                    if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                        lon = self._convert_to_degrees(gps_info['GPSLongitude'])
                        if gps_info['GPSLongitudeRef'] == 'W':
                            lon = -lon
                        result['longitude'] = lon
                    
                    # DO NOT extract altitude from EXIF - Explicitly omit this code
                # ALWAYS try to get altitude from nav data if available
                if hasattr(self, 'nav_timestamps'):
                    # First try with the datetime from EXIF
                    if datetime_str:
                        nav_altitude = self.get_altitude_from_nav(datetime_str)
                        if nav_altitude is not None:
                            result['altitude'] = nav_altitude
                            result['altitude_source'] = 'nav'
                    
                    # If no match by EXIF time, try filename
                    if 'altitude' not in result:
                        nav_altitude = self.get_altitude_from_nav(image_path)
                        if nav_altitude is not None:
                            result['altitude'] = nav_altitude
                            result['altitude_source'] = 'nav'
                            
                return result
                
        except Exception as e:
            print(f"Error extracting data from {image_path}: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _convert_to_degrees(self, value):
        """Helper function to convert GPS coordinates from EXIF format to decimal degrees"""
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

    def get_altitude_statistics(self, threshold: float = 9.0) -> Dict:
        """
        Calculate altitude statistics from the GPS data
        
        Args:
            threshold: Altitude threshold in meters (default 9.0m)
            
        Returns:
            Dictionary with altitude statistics
        """
        if not self.gps_data:
            return {}
            
        altitudes = [point['altitude'] for point in self.gps_data if 'altitude' in point]
        
        if not altitudes:
            return {}
            
        below_threshold = sum(1 for alt in altitudes if alt < threshold)
        above_threshold = len(altitudes) - below_threshold
        
        min_altitude = min(altitudes)
        max_altitude = max(altitudes)
        avg_altitude = sum(altitudes) / len(altitudes)
        
        # Calculate standard deviation
        variance = sum((alt - avg_altitude) ** 2 for alt in altitudes) / len(altitudes)
        std_dev = variance ** 0.5
        
        # Calculate percentiles
        sorted_altitudes = sorted(altitudes)
        percentile_25 = sorted_altitudes[int(len(sorted_altitudes) * 0.25)]
        percentile_50 = sorted_altitudes[int(len(sorted_altitudes) * 0.50)]  # median
        percentile_75 = sorted_altitudes[int(len(sorted_altitudes) * 0.75)]
        
        return {
            "count": len(altitudes),
            "below_threshold": below_threshold,
            "above_threshold": above_threshold,
            "below_percent": (below_threshold / len(altitudes) * 100) if altitudes else 0,
            "above_percent": (above_threshold / len(altitudes) * 100) if altitudes else 0,
            "min": min_altitude,
            "max": max_altitude,
            "range": max_altitude - min_altitude,
            "avg": avg_altitude,
            "median": percentile_50,
            "std_dev": std_dev,
            "percentile_25": percentile_25,
            "percentile_50": percentile_50,
            "percentile_75": percentile_75
        }
    
    def load_nav_data(self, nav_file_path: str) -> bool:
        """
        Load navigation data from a text file, extracting altitude data
        
        Args:
            nav_file_path: Path to the navigation file
            
        Returns:
            Boolean indicating success
        """
        try:
            print(f"Loading navigation data from {nav_file_path}")
            if not os.path.exists(nav_file_path):
                print(f"Navigation file not found: {nav_file_path}")
                return False
            
            # Store nav file path for reference
            self.nav_file_path = nav_file_path
            
            # Initialize nav data storage
            self.nav_timestamps = []
            
            # Read the navigation file
            with open(nav_file_path, 'r') as f:
                lines = f.readlines()
                
            # Skip header lines (first two lines typically)
            data_lines = lines[2:]
            
            total_entries = 0
            for line in data_lines:
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Split the line by commas
                    parts = line.strip().split(',')
                    
                    # Check if we have enough parts for time, date, and altitude
                    if len(parts) >= 4:
                        # Format: "HH:MM:SS.SSS", "MM/DD/YYYY", "Lat/Lon", "Altitude"
                        time_str = parts[0].strip()
                        date_str = parts[1].strip()
                        lon_lat_str = parts[2].strip()
                        altitude_str = parts[3].strip()
                        
                        # Combine date and time
                        timestamp = self._parse_nav_datetime(date_str, time_str)
                        
                        if timestamp:
                            try:
                                altitude = float(altitude_str)
                                # Store timestamp and altitude as a tuple for direct comparison
                                self.nav_timestamps.append((timestamp, altitude))
                                total_entries += 1
                            except ValueError:
                                print(f"Invalid altitude value: {altitude_str}")
                                
                except Exception as e:
                    print(f"Error parsing navigation line: {str(e)}")
                    continue
            
            print(f"Loaded {total_entries} altitude points from navigation file")
            
            # Sort timestamps chronologically for faster searching
            if self.nav_timestamps:
                self.nav_timestamps.sort(key=lambda x: x[0])
                
            return total_entries > 0
                
        except Exception as e:
            print(f"Error loading navigation data: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def _parse_nav_datetime(self, date_str, time_str):
        """
        Parse navigation timestamp from separate date and time strings
        
        Args:
            date_str: Date string in format MM/DD/YYYY
            time_str: Time string in format HH:MM:SS.SSS
            
        Returns:
            Datetime object or None if parsing failed
        """
        try:
            import datetime
            
            # Clean up the strings
            date_str = date_str.strip()
            time_str = time_str.strip()
            
            # Parse date components
            month, day, year = map(int, date_str.split('/'))
            
            # Parse time components
            if '.' in time_str:  # Has milliseconds
                time_parts = time_str.split('.')
                hour, minute, second = map(int, time_parts[0].split(':'))
                millisecond = int(time_parts[1])
                microsecond = millisecond * 1000  # Convert to microseconds
            else:
                hour, minute, second = map(int, time_str.split(':'))
                microsecond = 0
                
            # Create datetime object
            dt = datetime.datetime(year, month, day, hour, minute, second, microsecond)
            return dt
                
        except Exception as e:
            print(f"Error parsing nav datetime - date: {date_str}, time: {time_str}, error: {e}")
            return None

    def get_altitude_from_nav(self, image_path_or_timestamp):
        """
        Get altitude from navigation data based on image path or timestamp
        
        Args:
            image_path_or_timestamp: Image path or timestamp string
            
        Returns:
            Altitude value or None if no match found
        """
        if not hasattr(self, 'nav_timestamps') or not self.nav_timestamps:
            return None
            
        try:
            import datetime
            from dateutil import parser
        except ImportError as e:
            print(f"Required modules not available: {e}")
            return None
        from pathlib import Path
        
        # If we're passed a full path, extract the filename
        if isinstance(image_path_or_timestamp, str) and ('\\' in image_path_or_timestamp or '/' in image_path_or_timestamp):
            filename = Path(image_path_or_timestamp).name
        else:
            filename = image_path_or_timestamp
        
        # First try to parse it as a timestamp directly
        dt = None
        if isinstance(filename, str):
            # Try to extract timestamp from VOYIS filename pattern
            # Example: ESC_stills_processed_PPS_2024-06-27T074938.458700_2170.jpg
            if 'T' in filename and '_' in filename:
                try:
                    parts = filename.split('_')
                    # Look for the part with a T in it
                    for part in parts:
                        if 'T' in part:
                            # Format: 2024-06-27T074938.458700
                            date_part, time_part = part.split('T')
                            
                            # Handle time with or without milliseconds
                            if '.' in time_part:
                                time_base = time_part.split('.')[0]
                            else:
                                time_base = time_part
                                
                            # Format time with proper separators (HH:MM:SS)
                            if len(time_base) >= 6:  # At least HHMMSS
                                hours = time_base[0:2]
                                minutes = time_base[2:4]
                                seconds = time_base[4:6]
                                formatted_time = f"{hours}:{minutes}:{seconds}"
                                dt = parser.parse(f"{date_part} {formatted_time}")
                                break
                except Exception as e:
                    # Try direct parsing if filename processing failed
                    try:
                        dt = parser.parse(filename)
                    except:
                        dt = None
        
            # Direct timestamp parsing if format extraction failed
            if dt is None and isinstance(filename, str) and (':' in filename or '-' in filename):
                try:
                    dt = parser.parse(filename)
                except:
                    dt = None
        
            # If we have a datetime object, find the closest match in navigation data
            if dt:
                # Find closest timestamp within tolerance (3 seconds)
                tolerance_seconds = 3.0  # Set to 3 seconds as requested
                tolerance = datetime.timedelta(seconds=tolerance_seconds)
                closest_timestamp = None
                closest_altitude = None
                closest_diff = tolerance
                
                for timestamp, altitude in self.nav_timestamps:
                    diff = abs(timestamp - dt)
                    
                    # If this timestamp is closer than our previous best match
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_timestamp = timestamp
                        closest_altitude = altitude
                        
                        # If very close match, we can stop early
                        if diff.total_seconds() < 0.1:
                            break
                
                # Only return if within tolerance
                if closest_timestamp and closest_diff <= tolerance:
                    return closest_altitude
                
            # If we get here, no altitude was found
            return None

