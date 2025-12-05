def extract_dive_number(path):
    """
    Extract dive number from a file or directory path.
    
    Searches for patterns like DIVE###, Dive###, dive### in the path.
    Returns formatted as DIVE### with at least 3 digits (zero-padded if needed).
    
    Args:
        path: File or directory path string
        
    Returns:
        Formatted dive string (e.g., "DIVE015") or None if not found
        
    Examples:
        "Q:\\EN2501\\Image_LLS\\DIVE015_Stokey\\image_raw" -> "DIVE015"
        "Q:\\EN2501\\Image_LLS\\dive15_Stokey\\image_raw" -> "DIVE015"
        "Q:\\EN2501\\Image_LLS\\Dive2\\image_raw" -> "DIVE002"
    """
    import re
    
    if not path:
        return None
    
    # Pattern matches: DIVE, Dive, or dive followed by 1-4 digits
    # Case insensitive search for dive followed by numbers
    pattern = r'[Dd][Ii][Vv][Ee](\d{1,4})'
    
    match = re.search(pattern, str(path))
    if match:
        dive_num = match.group(1)
        # Pad with zeros to ensure at least 3 digits
        dive_num_padded = dive_num.zfill(3)
        return f"DIVE{dive_num_padded}"
    
    return None


def get_output_prefix(path, module_type):
    """
    Get output file prefix based on dive number and module type.
    
    Args:
        path: Input file or directory path
        module_type: Type of module ('Image', 'Nav', 'LLS')
        
    Returns:
        Prefix string (e.g., "DIVE015_Image_" or "Image_" if no dive found)
        
    Examples:
        get_output_prefix("Q:\\EN2501\\DIVE015\\images", "Image") -> "DIVE015_Image_"
        get_output_prefix("Q:\\EN2501\\data\\images", "Image") -> "Image_"
    """
    dive_number = extract_dive_number(path)
    
    if dive_number:
        return f"{dive_number}_{module_type}_"
    else:
        return f"{module_type}_"


def list_files_in_directory(directory):
    import os
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def check_image_file_type(file_name):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(file_name.lower().endswith(ext) for ext in valid_extensions)

def convert_csv_to_shapefile(csv_path, output_shapefile_path=None, log_callback=None):
    """
    Convert Image_Metrics.csv to ESRI Shapefile format
    
    Args:
        csv_path: Path to the Image_Metrics.csv file
        output_shapefile_path: Optional custom output path for shapefile (defaults to same directory as CSV)
        log_callback: Optional callback function for logging messages
        
    Returns:
        Path to created shapefile if successful, None otherwise
    """
    import os
    import pandas as pd
    
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        # Import geopandas - check if available
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            log("⚠ Warning: geopandas not available. Cannot create shapefile.")
            log("   Install with: pip install geopandas")
            return None
        
        # Check if CSV exists
        if not os.path.exists(csv_path):
            log(f"⚠ Warning: CSV file not found: {csv_path}")
            return None
        
        # Read CSV
        log(f"Reading Image_Metrics.csv from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Check if latitude and longitude columns exist
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            log("⚠ Warning: CSV does not contain latitude/longitude columns. Cannot create shapefile.")
            return None
        
        # Filter out rows with missing coordinates
        original_count = len(df)
        df = df.dropna(subset=['latitude', 'longitude'])
        valid_count = len(df)
        
        if valid_count == 0:
            log("⚠ Warning: No valid coordinates found in CSV. Cannot create shapefile.")
            return None
        
        if valid_count < original_count:
            log(f"   Note: {original_count - valid_count} rows without coordinates excluded from shapefile")
        
        # Create geometry column
        log(f"Creating shapefile with {valid_count} points...")
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        
        # Create GeoDataFrame with WGS84 coordinate system (EPSG:4326)
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Determine output path
        if output_shapefile_path is None:
            csv_dir = os.path.dirname(csv_path)
            # Use the same base name as the CSV file
            csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
            output_shapefile_path = os.path.join(csv_dir, f"{csv_basename}.shp")
        
        # Export to shapefile
        gdf.to_file(output_shapefile_path)
        log(f"✓ Shapefile created successfully: {output_shapefile_path}")
        log(f"   Contains {valid_count} image locations with {len(df.columns)} attributes")
        
        return output_shapefile_path
        
    except Exception as e:
        log(f"✗ Error creating shapefile: {str(e)}")
        import traceback
        log(f"   Traceback: {traceback.format_exc()}")
        return None