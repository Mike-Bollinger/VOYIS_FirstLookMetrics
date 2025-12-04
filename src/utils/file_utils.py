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
            output_shapefile_path = os.path.join(csv_dir, "Image_Metrics.shp")
        
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