import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

# Try importing geopandas, but handle the case where it might not be installed
try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("geopandas or shapely not available. Shapefile export will be disabled.")

class AltitudeMap:
    def __init__(self):
        """Initialize the altitude map generator"""
        self.altitude_threshold = 8.0  # Default threshold in meters
        self.low_altitude_threshold = 4.0  # Default low threshold in meters
    
    def set_altitude_thresholds(self, high_threshold: float, low_threshold: float = None) -> None:
        """
        Set the altitude thresholds for altitude classification
        
        Args:
            high_threshold: Upper threshold for altitude classification
            low_threshold: Lower threshold for altitude classification (default: high_threshold/2)
        """
        self.altitude_threshold = high_threshold
        if low_threshold is not None:
            self.low_altitude_threshold = low_threshold
        else:
            # Default low threshold is half of high threshold
            self.low_altitude_threshold = high_threshold / 2.0
    
    def _detect_outliers(self, gps_data: List[Dict], std_threshold: float = 3.0) -> List[bool]:
        """Same outlier detection logic as FootprintMap"""
        if not gps_data or len(gps_data) < 2:
            return [False] * len(gps_data)
        
        # Extract latitude, longitude, and altitude for all points
        lats = np.array([point['latitude'] for point in gps_data])
        lons = np.array([point['longitude'] for point in gps_data])
        alts = np.array([point['altitude'] for point in gps_data])
        
        # Calculate mean and standard deviation for altitude
        alt_mean = np.mean(alts)
        alt_std = np.std(alts)
        
        # Detect outliers based on altitude z-score
        altitude_outliers = np.abs((alts - alt_mean) / alt_std) > std_threshold
        
        # For latitude and longitude, we can use a simple range check based on IQR
        lat_q1, lat_q3 = np.percentile(lats, [25, 75])
        lon_q1, lon_q3 = np.percentile(lons, [25, 75])
        lat_iqr = lat_q3 - lat_q1
        lon_iqr = lon_q3 - lon_q1
        
        # Define bounds for latitude and longitude (1.5*IQR rule)
        lat_lower_bound = lat_q1 - 1.5 * lat_iqr
        lat_upper_bound = lat_q3 + 1.5 * lat_iqr
        lon_lower_bound = lon_q1 - 1.5 * lon_iqr
        lon_upper_bound = lon_q3 + 1.5 * lon_iqr
        
        # Detect spatial outliers
        spatial_outliers = (lats < lat_lower_bound) | (lats > lat_upper_bound) | \
                           (lons < lon_lower_bound) | (lons > lon_upper_bound)
        
        # Combine altitude and spatial outliers
        combined_outliers = altitude_outliers | spatial_outliers
        
        return combined_outliers.tolist()
    
    def create_location_map(self, gps_data: List[Dict], output_path: str, 
                       filename: str = None,
                       metrics=None) -> Optional[str]:
        """Create location map with outlier handling"""
        if not gps_data:
            print("No GPS data available")
            return None
        
        try:
            # Validate data points first
            valid_data = []
            invalid_data = []
            
            # Use gps_data instead of plot_data here
            for i, point in enumerate(gps_data):
                required_fields = ['latitude', 'longitude', 'altitude']
                missing_fields = [field for field in required_fields if field not in point]
                
                if missing_fields:
                    invalid_data.append({
                        'index': i,
                        'missing_fields': missing_fields,
                        'filename': point.get('filename', 'unknown')
                    })
                else:
                    valid_data.append(point)
            
            if invalid_data:
                print("\nWarning: Found invalid GPS data points:")
                for item in invalid_data[:5]:
                    print(f"Index {item['index']}: Missing {', '.join(item['missing_fields'])} for file: {item['filename']}")
                if len(invalid_data) > 5:
                    print(f"...and {len(invalid_data) - 5} more invalid points")
            
            if not valid_data:
                print("No valid GPS points found after filtering invalid data")
                return None
            
            # Detect outliers
            outliers = self._detect_outliers(valid_data)
            
            # Create separate lists for plotting and export
            plot_data = [point for point, is_outlier in zip(valid_data, outliers) if not is_outlier]
            export_data = valid_data  # Keep all valid data for export
            
            print(f"\nProcessing {len(plot_data)} points for visualization ({len(invalid_data)} invalid, {sum(outliers)} outliers)")
            
            # Export data to GIS formats first
            try:
                self.export_to_gis_formats(export_data, output_path)
            except Exception as e:
                print(f"Warning: Failed to export GIS formats: {e}")
            
            # Create visualization
            return self._create_location_plot(plot_data, output_path, filename, metrics)
            
        except Exception as e:
            print(f"Error in create_location_map: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _create_location_plot(self, plot_data: List[Dict], output_path: str, 
                         filename: str, metrics=None) -> Optional[str]:
        """Create the actual plot using cleaned data"""
        # ... existing plotting code moved here ...
        if not plot_data:
            print("No GPS data available")
            return None
        
        try:
            # Validate data points first
            valid_data = []
            invalid_data = []
            
            for i, point in enumerate(plot_data):
                required_fields = ['latitude', 'longitude', 'altitude']
                missing_fields = [field for field in required_fields if field not in point]
                
                if missing_fields:
                    invalid_data.append({
                        'index': i,
                        'missing_fields': missing_fields,
                        'filename': point.get('filename', 'unknown')
                    })
                else:
                    valid_data.append(point)
            
            if invalid_data:
                print("\nWarning: Found invalid GPS data points:")
                for item in invalid_data[:5]:
                    print(f"Index {item['index']}: Missing {', '.join(item['missing_fields'])} for file: {item['filename']}")
                if len(invalid_data) > 5:
                    print(f"...and {len(invalid_data) - 5} more invalid points")
            
            if not valid_data:
                print("No valid GPS points found after filtering invalid data")
                return None
                
            print(f"\nProcessing {len(valid_data)} valid GPS points ({len(invalid_data)} points excluded)")
            
            # Create non-outlier subset for plotting
            outliers = self._detect_outliers(valid_data)
            plot_data = [point for point, is_outlier in zip(valid_data, outliers) if not is_outlier]
            
            print(f"\nExcluding {sum(outliers)} outlier points from plots (but keeping them in export files)")
            
            # Use plot_data for visualization but keep original data for exports
            # Rest of the existing function code, but use plot_data instead of gps_data for plotting
            try:
                # Set up matplotlib for non-interactive use
                import matplotlib
                matplotlib.use('Agg')
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Extract coordinates and altitudes
                lats = [p['latitude'] for p in plot_data]
                lons = [p['longitude'] for p in plot_data]
                alts = [p['altitude'] for p in plot_data]
                
                # Use class variable for low altitude threshold
                low_altitude_threshold = self.low_altitude_threshold
                
                # Separate points into three altitude categories
                optimal_mask = [(a >= low_altitude_threshold) and (a < self.altitude_threshold) for a in alts]
                too_high_mask = [a >= self.altitude_threshold for a in alts]
                too_low_mask = [a < low_altitude_threshold for a in alts]
                
                # Count points in each category
                optimal_count = sum(optimal_mask)
                too_high_count = sum(too_high_mask)
                too_low_count = sum(too_low_mask)
                total_count = len(alts)
                
                # Create scatter plots with different colors for each category
                sc1 = ax.scatter([lons[i] for i in range(len(lons)) if optimal_mask[i]],
                               [lats[i] for i in range(len(lats)) if optimal_mask[i]],
                               c='green', label=f'Optimal altitude ({low_altitude_threshold}-{self.altitude_threshold}m): {optimal_count} images', 
                               alpha=0.6, s=50)
                               
                sc2 = ax.scatter([lons[i] for i in range(len(lons)) if too_high_mask[i]],
                               [lats[i] for i in range(len(lats)) if too_high_mask[i]],
                               c='red', label=f'Too high (>{self.altitude_threshold}m): {too_high_count} images', 
                               alpha=0.6, s=50)
                               
                sc3 = ax.scatter([lons[i] for i in range(len(lons)) if too_low_mask[i]],
                               [lats[i] for i in range(len(lats)) if too_low_mask[i]],
                               c='orange', label=f'Too low (<{low_altitude_threshold}m): {too_low_count} images', 
                               alpha=0.6, s=50)
                
                # Calculate percentages
                optimal_pct = (optimal_count / total_count) * 100
                too_high_pct = (too_high_count / total_count) * 100
                too_low_pct = (too_low_count / total_count) * 100
                
                # Add labels and title
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.set_title('Image Locations by Altitude', fontsize=14)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend
                legend = ax.legend(loc='upper right', fontsize=10)
                
                # Add statistics annotation
                stats_text = f'Total Images: {total_count}\n'
                stats_text += f'Optimal ({low_altitude_threshold}-{self.altitude_threshold}m): {optimal_count} ({optimal_pct:.1f}%)\n'
                stats_text += f'Too high (>{self.altitude_threshold}m): {too_high_count} ({too_high_pct:.1f}%)\n'
                stats_text += f'Too low (<{low_altitude_threshold}m): {too_low_count} ({too_low_pct:.1f}%)'
                
                # Add stats box
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                      verticalalignment='bottom', horizontalalignment='left',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add scale bar if metrics object with scale info is available
                if metrics and hasattr(metrics, 'scale_info') and metrics.scale_info:
                    self._add_scale_bar(ax, metrics.scale_info)
                
                plt.tight_layout()
                
                # Save plot with standardized naming
                if filename is None:
                    filename = "Image_Locations_Map.png"
                output_file = os.path.join(output_path, filename)
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory
                
                print(f"Location map saved to: {output_file}")
                return output_file
                
            except Exception as e:
                print(f"Error creating location map: {e}")
                import traceback
                print(traceback.format_exc())
                return None
        except Exception as e:
            print(f"Error processing GPS data: {e}")
            return None
    
    def create_altitude_histogram(self, gps_data: List[Dict], output_path: str, 
                         filename: str = None,
                         max_display_altitude: float = 50.0) -> Optional[str]:
        """
        Create a histogram of image altitudes
        
        Args:
            gps_data: List of dictionaries with altitude data
            output_path: Directory to save the plot
            filename: Filename for the generated histogram
            max_display_altitude: Maximum altitude to display (higher values are clipped)
            
        Returns:
            Path to the saved histogram file, or None if no data or error
        """
        if not gps_data:
            return None
        
        try:
            # Extract altitudes, filtering out missing values
            all_altitudes = [point['altitude'] for point in gps_data if 'altitude' in point]
            
            if not all_altitudes:
                print("No altitude data found")
                return None
            
            # Calculate how many values are above max_display_altitude for stats
            above_max_count = sum(1 for alt in all_altitudes if alt > max_display_altitude)
            above_max_pct = (above_max_count / len(all_altitudes)) * 100
            
            # Filter the data to include only values within display range
            altitudes = [alt for alt in all_altitudes if alt <= max_display_altitude]
            
            if not altitudes:
                print(f"No altitudes below {max_display_altitude}m found")
                return None
            
            # Set up matplotlib for non-interactive use
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import gridspec
            from matplotlib.ticker import MaxNLocator
            
            # Create a figure with GridSpec for a broken y-axis
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
            
            # Create top and bottom axes
            ax_top = fig.add_subplot(gs[0])
            ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
            
            # Calculate range of altitudes within display range
            min_alt = min(altitudes)
            max_alt = max(altitudes)
            
            # Dynamically calculate number of bins based on dataset size
            if len(altitudes) > 10000:
                n_bins = min(100, len(altitudes) // 50)
            elif len(altitudes) > 1000:
                n_bins = min(50, len(altitudes) // 25)
            elif len(altitudes) > 100:
                n_bins = min(30, len(altitudes) // 5)
            else:
                n_bins = min(20, len(altitudes) // 5)
                
            # Ensure minimum 10 bins
            n_bins = max(10, n_bins)
            
            print(f"Using {n_bins} bins for altitude histogram with {len(altitudes)} data points")
            
            # Define bin edges from 0 to max_display_altitude
            bin_edges = np.linspace(0, max_display_altitude, n_bins)
            
            # Create the histograms with filtered altitudes
            counts_top, bins_top, patches_top = ax_top.hist(altitudes, bins=bin_edges, color='skyblue', 
                                                          edgecolor='black', alpha=0.7)
            counts_bottom, bins_bottom, patches_bottom = ax_bottom.hist(altitudes, bins=bin_edges, color='skyblue', 
                                                                      edgecolor='black', alpha=0.7)
            
            # Calculate where to break the y-axis
            max_count = max(counts_top) if len(counts_top) > 0 else 1
            
            # Set top plot to only show the highest peaks
            top_min = max_count * 0.2  # Show only the top 80% of the highest peak
            top_max = max_count * 1.05  # Add a little headroom at the top
            
            # Set bottom plot to focus on lower counts
            bottom_max = max_count * 0.15  # Show up to 15% of max in bottom plot
            
            # Set the y-axis limits for both subplots
            ax_top.set_ylim(top_min, top_max)
            ax_bottom.set_ylim(0, bottom_max)
            
            # Add broken axis indicators
            d = .015  # size of diagonal lines
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)        # bottom-left diagonal
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom-right diagonal
            
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top-left diagonal
            ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal
            
            # Use class variable instead of local variable
            low_altitude_threshold = self.low_altitude_threshold
            
            # Add high altitude threshold line (red)
            ax_top.axvline(x=self.altitude_threshold, color='red', linestyle='--', 
                         linewidth=2, label=f'High Threshold: {self.altitude_threshold}m')
            ax_bottom.axvline(x=self.altitude_threshold, color='red', linestyle='--', 
                            linewidth=2)
            
            # Add low altitude threshold line (orange)
            ax_top.axvline(x=low_altitude_threshold, color='orange', linestyle='--', 
                         linewidth=2, label=f'Low Threshold: {low_altitude_threshold}m')
            ax_bottom.axvline(x=low_altitude_threshold, color='orange', linestyle='--', 
                            linewidth=2)
            
            # Calculate the percentages of images in each category
            below_low = sum(1 for alt in all_altitudes if alt < self.low_altitude_threshold)
            optimal = sum(1 for alt in all_altitudes if self.low_altitude_threshold <= alt < self.altitude_threshold)
            above_high = sum(1 for alt in all_altitudes if alt >= self.altitude_threshold)
            
            below_low_pct = (below_low / len(all_altitudes)) * 100
            optimal_pct = (optimal / len(all_altitudes)) * 100
            above_high_pct = (above_high / len(all_altitudes)) * 100
            
            # Improve plot appearance
            ax_bottom.set_xlabel('Altitude (meters)', fontsize=12)
            ax_top.set_title('Image Altitude Distribution', fontsize=14)
            
            # Set x-axis limit
            ax_bottom.set_xlim(0, max_display_altitude)
            
            # Add subtitle showing full altitude range
            overall_min_alt = min(all_altitudes)
            overall_max_alt = max(all_altitudes)
            
            if above_max_count > 0:
                subtitle = f"Display limited to {max_display_altitude}m (full range: {overall_min_alt:.1f}m - {overall_max_alt:.1f}m)"
                subtitle += f"\n{above_max_count} images ({above_max_pct:.1f}%) above {max_display_altitude}m excluded from plot"
                ax_top.text(0.5, 0.95, subtitle, transform=ax_top.transAxes, 
                         fontsize=10, ha='center', va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add y-label only to the left side (once between the two plots)
            fig.text(0.04, 0.5, 'Number of Images', va='center', rotation='vertical', fontsize=12)
            
            # Set x-ticks for bottom plot only
            ax_top.xaxis.set_tick_params(labelbottom=False)  # Hide x-axis labels for top plot
            ax_bottom.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit number of x-ticks
            
            # Add statistics annotation to top plot
            stats_text = f'Total Images: {len(all_altitudes)}\n'
            stats_text += f'Altitude Range: {overall_min_alt:.2f}m - {overall_max_alt:.2f}m\n'
            stats_text += f'Mean: {np.mean(all_altitudes):.2f}m\n'
            stats_text += f'Median: {np.median(all_altitudes):.2f}m\n'
            stats_text += f'Too Low (<{self.low_altitude_threshold}m): {below_low} images ({below_low_pct:.1f}%)\n'
            stats_text += f'Optimal ({self.low_altitude_threshold}-{self.altitude_threshold}m): {optimal} images ({optimal_pct:.1f}%)\n'
            stats_text += f'Too High (>{self.altitude_threshold}m): {above_high} images ({above_high_pct:.1f}%)'
            
            # Add stats box to top plot
            ax_top.text(0.98, 0.98, stats_text, transform=ax_top.transAxes, 
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add legend only to top plot
            ax_top.legend()
            
            # Add grid to both plots
            ax_top.grid(True, linestyle='--', alpha=0.7)
            ax_bottom.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust spacing manually
            fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.05)
            
            # Save plot with standardized naming
            if filename is None:
                filename = "Image_Altitude_Histogram.png"
            output_file = os.path.join(output_path, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            
            print(f"Altitude histogram saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating altitude histogram: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def export_to_gis_formats(self, gps_data: List[Dict], output_path: str, 
                             csv_filename: str = "Image_Locations.csv",
                             shapefile_filename: str = "Image_Locations.shp") -> Dict[str, str]:
        """Export GPS data to CSV and Shapefile formats"""
        result_files = {}
        print("\n--- STARTING GIS EXPORT ---")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Create DataFrame
            df = pd.DataFrame(gps_data)
            
            # Export CSV
            csv_path = os.path.join(output_path, csv_filename)
            df.to_csv(csv_path, index=False)
            result_files['csv'] = csv_path
            print(f"CSV exported successfully: {csv_path}")
            
            # Export Shapefile if geopandas is available
            if GEOPANDAS_AVAILABLE:
                try:
                    # Create geometry column
                    geometry = [Point(x['longitude'], x['latitude']) 
                              for x in gps_data]
                    
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                    
                    # Export to shapefile
                    shapefile_path = os.path.join(output_path, shapefile_filename)
                    gdf.to_file(shapefile_path)
                    result_files['shapefile'] = shapefile_path
                    print(f"Shapefile exported successfully: {shapefile_path}")
                    
                except Exception as e:
                    print(f"Error exporting shapefile: {str(e)}")
            
            return result_files
            
        except Exception as e:
            print(f"Error in GIS export: {str(e)}")
            return result_files

    def extract_exif_data(self, image_path: str) -> Dict:
        """Extract EXIF data and image dimensions from an image file"""
        try:
            from PIL import Image, ExifTags
            
            if not os.path.exists(image_path):
                return {}
                
            with Image.open(image_path) as img:
                # Get image dimensions
                width, height = img.size
                
                # Store dimensions regardless of EXIF availability
                result = {
                    'ImageWidth': width,
                    'ImageHeight': height
                }
                
                # Get EXIF data
                exif_data = img._getexif()
                if not exif_data:
                    return result  # Return just the dimensions if no EXIF
                    
                # Convert EXIF tag numbers to readable names
                exif = {
                    ExifTags.TAGS.get(tag, tag): value
                    for tag, value in exif_data.items()
                    if tag in ExifTags.TAGS
                }
                
                # Extract specific EXIF values
                # DateTime
                if 'DateTimeOriginal' in exif:
                    result['DateTime'] = exif['DateTimeOriginal']
                elif 'DateTime' in exif:
                    result['DateTime'] = exif['DateTime']
                    
                # ExposureTime
                if 'ExposureTime' in exif:
                    exposure = exif['ExposureTime']
                    if isinstance(exposure, tuple):
                        result['ExposureTime'] = f"{exposure[0]}/{exposure[1]}"
                    else:
                        result['ExposureTime'] = str(exposure)
                
                # FNumber
                if 'FNumber' in exif:
                    fnumber = exif['FNumber']
                    if isinstance(fnumber, tuple):
                        result['FNumber'] = round(fnumber[0] / fnumber[1], 1)
                    else:
                        result['FNumber'] = fnumber
                
                # FocalLength
                if 'FocalLength' in exif:
                    focal = exif['FocalLength']
                    if isinstance(focal, tuple):
                        result['FocalLength'] = round(focal[0] / focal[1], 1)
                    else:
                        result['FocalLength'] = focal
                        
                # SubjectDistance
                if 'SubjectDistance' in exif:
                    distance = exif['SubjectDistance']
                    if isinstance(distance, tuple):
                        result['SubjectDistance'] = round(distance[0] / distance[1], 1)
                    else:
                        result['SubjectDistance'] = distance
                
                return result
        except Exception as e:
            print(f"Error extracting EXIF data from {image_path}: {e}")
            return {}


    def _add_scale_bar(self, ax, scale_info=None):
        """
        Add a scale bar to the map
        
        Args:
            ax: Matplotlib axis to add the scale bar to
            scale_info: Optional dictionary with scale information
        """
        if not scale_info:
            return
            
        # Get scale factor (meters per degree)
        meters_per_degree = scale_info.get('meters_per_degree', 111000)
        
        # Get axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Calculate scale bar length in degrees
        # Aim for a scale bar around 1/5 of the map width
        map_width_degrees = x_max - x_min
        map_width_meters = map_width_degrees * meters_per_degree
        
        # Choose a nice round number for scale bar in meters
        scale_bar_meters = 100  # Default 100m scale
        
        if map_width_meters > 10000:  # > 10km
            scale_bar_meters = 1000  # 1km
        elif map_width_meters > 1000:  # > 1km
            scale_bar_meters = 200   # 200m
        
        # Convert back to degrees
        scale_bar_degrees = scale_bar_meters / meters_per_degree
        
        # Scale bar position (lower right)
        bar_x = x_max - scale_bar_degrees * 1.5  # Offset from right edge
        bar_y = y_min + (y_max - y_min) * 0.05   # Offset from bottom edge
        
        # Draw scale bar line
        ax.plot([bar_x, bar_x + scale_bar_degrees], [bar_y, bar_y], 'k-', 
               linewidth=2)
        
        # Add tick marks at each end
        tick_height = (y_max - y_min) * 0.01
        ax.plot([bar_x, bar_x], [bar_y, bar_y + tick_height], 'k-', 
               linewidth=2)
        ax.plot([bar_x + scale_bar_degrees, bar_x + scale_bar_degrees], 
               [bar_y, bar_y + tick_height], 'k-', linewidth=2)
        
        # Add label
        if scale_bar_meters >= 1000:
            label = f"{scale_bar_meters/1000:.0f} km"
        else:
            label = f"{scale_bar_meters:.0f} m"
            
        ax.text(bar_x + scale_bar_degrees/2, bar_y + tick_height*2, 
               label, ha='center', va='bottom')
