import os
import sys
import traceback
import time
from typing import Dict, List, Optional, Callable, Tuple
from pandas import DataFrame
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import concurrent.futures  # For thread-based parallel processing
import threading
from pathlib import Path

class HighlightSelector:
    """
    Class to select highlight images from a dataset based on image metrics
    Can work independently or integrate with visibility analysis
    """
    
    def __init__(self):
        """Initialize the highlight selector"""
        # Libraries will be imported on demand to avoid startup issues
        self._np = None
        self._plt = None
        self._pd = None
        
        # Configuration
        self.metrics_weight = {
            'contrast': 0.35,       # Weight for contrast score
            'texture': 0.30,        # Weight for texture/detail score
            'color_variance': 0.20, # Weight for color diversity
            'entropy': 0.15,        # Weight for information content
            'visibility': 0.50,     # Used only if visibility results available
        }
        
        # Add metrics cache
        self.metrics_cache = {}
        self.cache_path = None  # Will be set during select_highlights
    
    def _ensure_imports(self):
        """Import necessary libraries on demand"""
        if self._np is None:
            try:
                import numpy as np
                self._np = np
            except ImportError:
                raise ImportError("NumPy is required for highlight image selection")
                
        # Use PIL for faster image loading if available
        if not hasattr(self, '_pil'):
            try:
                from PIL import Image, ImageStat
                self._pil = Image
                self._imagestat = ImageStat
                self._has_pil = True
            except ImportError:
                self._has_pil = False
                # Fall back to matplotlib
                if self._plt is None:
                    try:
                        import matplotlib
                        # Use Agg backend to avoid GUI requirements
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        self._plt = plt
                    except ImportError:
                        self._plt = None  # Mark as attempted but failed
                        print("Warning: Neither PIL nor Matplotlib is available. Some features will be disabled.")
        
        if self._pd is None:
            try:
                import pandas as pd
                self._pd = pd
            except ImportError:
                raise ImportError("Pandas is required for highlight image selection")
                
        # Check for optional libraries
        try:
            import scipy
            self._has_scipy = True
        except ImportError:
            self._has_scipy = False
            print("Warning: SciPy not found. Some advanced metrics will be disabled.")
                
    def select_highlights(self,
                          input_folder: str,
                          output_folder: str,
                          count: int = 10,
                          progress_callback: Optional[Callable] = None,
                          altitude_threshold: Optional[float] = None,
                          min_altitude_threshold: Optional[float] = 2.0,
                          visibility_results: Optional[Dict] = None) -> List[str]:
        """
        Select highlight images based on image metrics
        
        Args:
            input_folder: Folder with images to analyze
            output_folder: Directory to save highlight images
            count: Number of highlight images to select (default: 10)
            progress_callback: Optional callback for progress updates
            altitude_threshold: Optional maximum altitude threshold to filter images
            min_altitude_threshold: Optional minimum altitude threshold (default: 2.0m)
            visibility_results: Optional visibility analysis results to integrate
            
        Returns:
            List of paths to selected highlight images
        """
        try:
            self._ensure_imports()
            
            # Create highlight images directory
            highlight_dir = os.path.join(output_folder, "highlight_images")
            os.makedirs(highlight_dir, exist_ok=True)
            
            # Setup logging function
            def log_message(msg, progress=None):
                print(f"HighlightSelector: {msg}")
                if progress_callback and progress is not None:
                    try:
                        progress_callback(progress, msg)
                    except Exception:
                        pass  # Ignore progress callback errors
            
            # If no visibility results were provided, try to find them in the output folder
            if visibility_results is None:
                log_message("Checking for existing visibility analysis results in output folder...", progress=5)
                try:
                    # Look in multiple potential locations
                    vis_csv_paths = [
                        os.path.join(output_folder, "Image_Visibility_Results.csv"),
                        os.path.join(os.path.dirname(output_folder), "Image_Visibility_Results.csv"),
                        os.path.join(input_folder, "Image_Visibility_Results.csv"),
                        os.path.join(os.path.dirname(input_folder), "visibility_results.csv")
                    ]
                    
                    vis_csv_path = None
                    for path in vis_csv_paths:
                        if os.path.exists(path):
                            vis_csv_path = path
                            log_message(f"Found visibility results at: {vis_csv_path}", progress=6)
                            break
                            
                    if vis_csv_path:
                        # Load the CSV file
                        vis_df = self._pd.read_csv(vis_csv_path)
                        
                        # Check if it has the expected columns
                        if 'image' in vis_df.columns:
                            # Create a visibility results structure
                            visibility_results = {
                                'analyzed_images': []
                            }
                            
                            # Extract data from each row
                            for _, row in vis_df.iterrows():
                                result = {
                                    'image': row['image'],
                                    'visibility': row.get('visibility', 'unknown'),
                                    'confidence': row.get('confidence', 0.0)
                                }
                                visibility_results['analyzed_images'].append(result)
                                
                            log_message(f"Found existing visibility analysis for {len(visibility_results['analyzed_images'])} images", progress=6)
                        else:
                            log_message("Found visibility results file but missing required columns", progress=6)
                
                except Exception as e:
                    log_message(f"Could not load existing visibility results: {e}", progress=6)
                    # Continue without visibility data
                
            log_message(f"Selecting {count} highlight images based on image metrics...", progress=10)
            
            # Check if metrics already exist in Image_Metrics.csv
            log_message("Checking for existing metrics in Image_Metrics.csv...", progress=12)
            existing_metrics_df = self.load_metrics_from_csv(output_folder, input_folder)
            
            # Set up visibility map
            visibility_map = {}
            has_visibility_data = False
            
            # STEP 1: Find all images
            log_message("Step 1: Finding all images in dataset...", progress=15)
            
            all_image_paths = []
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            
            # First, try the input_folder directly
            log_message(f"Scanning for images in: {input_folder}")
            
            if not os.path.exists(input_folder):
                log_message(f"ERROR: Input folder does not exist: {input_folder}")
                return []
            
            # Scan for images in the input folder and all subdirectories
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        # Don't skip raw images anymore - include ALL images for selection
                        full_path = os.path.join(root, file)
                        all_image_paths.append(full_path)
            
            log_message(f"Found {len(all_image_paths)} images in dataset", progress=20)
            
            # Log some sample paths for debugging
            if len(all_image_paths) > 0:
                log_message(f"Sample image paths:")
                for i, path in enumerate(all_image_paths[:5]):  # Show first 5
                    log_message(f"  {i+1}: {path}")
                if len(all_image_paths) > 5:
                    log_message(f"  ... and {len(all_image_paths) - 5} more images")
            
            # If we found no images in input_folder, try alternative scanning
            if len(all_image_paths) == 0:
                log_message("No images found in primary input folder, trying alternative locations...", progress=21)
                
                # Try parent directory
                parent_dir = os.path.dirname(input_folder)
                log_message(f"Trying parent directory: {parent_dir}")
                
                for root, dirs, files in os.walk(parent_dir):
                    # Skip the output folder itself
                    if root == output_folder:
                        continue
                    for file in files:
                        if file.lower().endswith(image_extensions):
                            full_path = os.path.join(root, file)
                            all_image_paths.append(full_path)
                
                log_message(f"After alternative scanning: Found {len(all_image_paths)} images", progress=22)
            
            if len(all_image_paths) == 0:
                log_message("No images found in any location", progress=100)
                return []
            
            # Step 2: Filter by altitude if altitude data is available
            if altitude_threshold is not None or min_altitude_threshold is not None:
                log_message("Step 2: Filtering images by altitude...", progress=25)
                
                # Load altitude data from CSV if available
                altitude_map = {}
                try:
                    csv_path = os.path.join(output_folder, "Image_Metrics.csv")
                    if os.path.exists(csv_path):
                        df = self._pd.read_csv(csv_path)
                        if 'filename' in df.columns and 'altitude' in df.columns:
                            for _, row in df.iterrows():
                                filename = os.path.basename(row['filename'])
                                altitude_map[filename] = row['altitude']
                            log_message(f"Loaded altitude data for {len(altitude_map)} images")
                except Exception as e:
                    log_message(f"Could not load altitude data: {e}")
                
                # Filter images by altitude
                if altitude_map:
                    filtered_images = []
                    for img_path in all_image_paths:
                        filename = os.path.basename(img_path)
                        if filename in altitude_map:
                            altitude = altitude_map[filename]
                            # Check altitude thresholds
                            passes_max = altitude_threshold is None or altitude <= altitude_threshold
                            passes_min = min_altitude_threshold is None or altitude >= min_altitude_threshold
                            
                            if passes_max and passes_min:
                                filtered_images.append(img_path)
                    
                    log_message(f"Selected {len(filtered_images)} of {len(all_image_paths)} images within altitude range", progress=40)
                    all_image_paths = filtered_images
            
            log_message(f"After filtering: {len(all_image_paths)} images remain")
            
            # Step 3: Check for visibility analysis results
            if visibility_results:
                log_message("Step 3: Integrating visibility analysis results...", progress=45)
                source = "current session" if progress_callback else "previous analysis"
                
                try:
                    # Extract visibility results for each image
                    if 'analyzed_images' in visibility_results:
                        for img_data in visibility_results['analyzed_images']:
                            filename = img_data['image']
                            visibility_map[filename] = {
                                'category': img_data['visibility'],
                                'confidence': img_data['confidence']
                            }
                        has_visibility_data = True
                        
                        log_message(f"Found visibility data for {len(visibility_map)} images from {source}", progress=50)
                except Exception as e:
                    log_message(f"Error integrating visibility results: {e}", progress=50)
            else:
                log_message("Step 3: No visibility analysis results available", progress=45)
            
            # Step 4: Check for existing metrics and decide whether to use them or calculate new ones
            has_sufficient_metrics = False
            if not existing_metrics_df.empty:
                # Check if we have metrics for a reasonable percentage of our images
                existing_filenames = set(existing_metrics_df['filename'].apply(os.path.basename))
                current_filenames = set(os.path.basename(path) for path in all_image_paths)
                overlap_count = len(existing_filenames.intersection(current_filenames))
                coverage_percentage = overlap_count / len(current_filenames) if current_filenames else 0
                
                log_message(f"Found existing metrics for {overlap_count}/{len(current_filenames)} images ({coverage_percentage:.1%} coverage)")
                
                # Use existing metrics if we have good coverage (>= 75%)
                if coverage_percentage >= 0.75:
                    has_sufficient_metrics = True
                    log_message(f"Using existing metrics (sufficient coverage)", progress=55)
                    metrics_df = existing_metrics_df
                    
                    # Filter to only include images we found
                    metrics_df = metrics_df[metrics_df['filename'].apply(os.path.basename).isin(current_filenames)]
                    
                    # Add visibility data if available and not already present
                    if has_visibility_data and 'visibility_score' not in metrics_df.columns:
                        log_message("Adding visibility data to existing metrics...", progress=60)
                        for i, row in metrics_df.iterrows():
                            filename = os.path.basename(row['filename'])
                            if filename in visibility_map:
                                vis_data = visibility_map[filename]
                                metrics_df.at[i, 'visibility_category'] = vis_data['category']
                                metrics_df.at[i, 'visibility_confidence'] = vis_data['confidence']
                                
                                # Calculate visibility score
                                vis_category_score = 0.0
                                if vis_data['category'] == 'great_visibility':
                                    vis_category_score = 1.0
                                elif vis_data['category'] == 'good_visibility':
                                    vis_category_score = 0.75
                                elif vis_data['category'] == 'low_visibility':
                                    vis_category_score = 0.3
                                
                                metrics_df.at[i, 'visibility_score'] = vis_category_score * vis_data['confidence']
            
            # Step 5: Calculate metrics if we don't have sufficient existing ones
            if not has_sufficient_metrics:
                log_message(f"Calculating metrics for {len(all_image_paths)} images...", progress=50)
                
                # Process images for metrics
                image_metrics = []
                start_time = time.time()
                
                for i, img_path in enumerate(all_image_paths):
                    try:
                        # Log progress every 100 images for large datasets
                        if i % 100 == 0 or i < 50:  # Log first 50 and then every 100th
                            elapsed = time.time() - start_time
                            rate = i / elapsed if elapsed > 0 and i > 0 else 0
                            eta = (len(all_image_paths) - i) / rate if rate > 0 else 0
                            log_message(f"Processing image {i+1}/{len(all_image_paths)} - {os.path.basename(img_path)} (Rate: {rate:.1f} img/s, ETA: {eta/60:.1f}min)")
                        
                        # Calculate basic metrics
                        metrics = self.calculate_image_metrics(img_path)
                        
                        if metrics:
                            filename = os.path.basename(img_path)
                            
                            # Get altitude if available
                            if 'altitude_map' in locals() and altitude_map and filename in altitude_map:
                                metrics['altitude'] = altitude_map[filename]
                            
                            # Get visibility data if available
                            if has_visibility_data and visibility_map and filename in visibility_map:
                                vis_data = visibility_map[filename]
                                metrics['visibility_category'] = vis_data['category']
                                metrics['visibility_confidence'] = vis_data['confidence']
                                
                                # Calculate visibility score
                                vis_category_score = 0.0
                                if vis_data['category'] == 'great_visibility':
                                    vis_category_score = 1.0
                                elif vis_data['category'] == 'good_visibility':
                                    vis_category_score = 0.75
                                elif vis_data['category'] == 'low_visibility':
                                    vis_category_score = 0.3
                                
                                metrics['visibility_score'] = vis_category_score * vis_data['confidence']
                            else:
                                metrics['visibility_score'] = None
                                
                            metrics['path'] = img_path
                            metrics['filename'] = filename
                            
                            image_metrics.append(metrics)
                            
                        # Update progress more frequently
                        if progress_callback and (i % 50 == 0 or i == len(all_image_paths) - 1):
                            progress = 50 + int((i / len(all_image_paths)) * 35)  # Progress from 50% to 85%
                            log_message(f"Processed {i+1}/{len(all_image_paths)} images", progress=progress)
                            
                    except Exception as e:
                        log_message(f"Error processing {img_path}: {e}")

                total_time = time.time() - start_time
                log_message(f"Calculated metrics for {len(image_metrics)} images in {total_time:.1f} seconds", progress=85)
                
                # Convert to DataFrame for easier filtering and sorting
                metrics_df = self._pd.DataFrame(image_metrics)
                
                # Export metrics to CSV for future use
                try:
                    self.export_metrics_to_csv(output_folder, metrics_df)
                    log_message("Exported calculated metrics to CSV for future use")
                except Exception as e:
                    log_message(f"Warning: Could not save metrics to CSV: {e}")
            
            # Check if we have any metrics
            if len(metrics_df) == 0:
                log_message("No valid images found with proper metrics", progress=90)
                return []
            
            # Step 6: Calculate combined scores and select highlights
            log_message("Calculating combined scores for image selection...", progress=87)
            
            # Weights for different metrics (can be adjusted)
            contrast_weight = 0.3
            texture_weight = 0.3
            color_weight = 0.2
            entropy_weight = 0.2
            visibility_weight = 1.0  # Visibility gets full weight if available
            
            combined_scores = []
            for _, row in metrics_df.iterrows():
                score = 0.0
                
                # Add metric scores
                score += contrast_weight * row.get('contrast', 0)
                score += texture_weight * row.get('texture', 0) 
                score += color_weight * row.get('color_variance', 0)
                score += entropy_weight * row.get('entropy', 0)
                
                # Add visibility score if available
                if row.get('visibility_score') is not None:
                    score += visibility_weight * row.get('visibility_score', 0)
                
                combined_scores.append(score)
            
            metrics_df['combined_score'] = combined_scores
            
            # Debug info
            log_message(f"Combined scores calculated. Range: {min(combined_scores):.3f} to {max(combined_scores):.3f}, "
                       f"mean={metrics_df['combined_score'].mean():.3f}")
            
            # Filter out zero_visibility images if we have visibility data
            if has_visibility_data:
                log_message("Filtering out zero visibility images...", progress=88)
                zero_vis_count = 0
                
                # Create a filter to exclude zero_visibility images
                if 'visibility_category' in metrics_df.columns:
                    zero_vis_mask = metrics_df['visibility_category'] == 'zero_visibility'
                    zero_vis_count = zero_vis_mask.sum()
                    
                    if zero_vis_count > 0:
                        # Filter out zero visibility images
                        metrics_df = metrics_df[~zero_vis_mask]
                        log_message(f"Filtered out {zero_vis_count} images with zero visibility", progress=89)
            
            # Sort by combined score
            metrics_df = metrics_df.sort_values('combined_score', ascending=False)
            
            # Select top images
            selected_count = min(count, len(metrics_df))
            top_images = metrics_df.head(selected_count)
            
            log_message(f"Selected top {selected_count} images based on combined scores", progress=90)
            
            # Copy selected images to highlight directory
            highlight_paths = []
            for i, (_, row) in enumerate(top_images.iterrows()):
                try:
                    src_path = row['path']
                    filename = row['filename']
                    dest_path = os.path.join(highlight_dir, filename)
                    
                    import shutil
                    shutil.copy2(src_path, dest_path)
                    highlight_paths.append(dest_path)
                    
                    log_message(f"✓ Copied highlight {i+1}/{selected_count}: {filename}")
                    
                except Exception as e:
                    log_message(f"ERROR copying {filename}: {e}")
            
            if len(highlight_paths) > 0:
                # Create HTML gallery
                try:
                    has_visibility = 'visibility_category' in top_images.columns
                    html_path = self._create_highlights_html(highlight_dir, top_images, has_visibility)
                    log_message(f"Created highlights gallery: {os.path.basename(html_path)}")
                except Exception as e:
                    log_message(f"Warning: Could not create HTML gallery: {e}")
                
                # Create highlight panel in the main output directory
                try:
                    panel_path = self.create_highlight_panel(highlight_dir, output_folder, top_images, top_count=4)
                    if panel_path:
                        log_message(f"Created highlight panel: {os.path.basename(panel_path)}")
                except Exception as e:
                    log_message(f"Warning: Could not create highlight panel: {e}")
                
                log_message(f"✓ Selected {len(highlight_paths)} highlight images. Saved to {highlight_dir}", progress=100)
            else:
                log_message("No highlight images were selected", progress=100)
            
            return highlight_paths
            
        except Exception as e:
            log_message(f"ERROR in highlight selection: {e}", progress=100)
            print(f"HighlightSelector traceback: {traceback.format_exc()}")
            return []
    
    def _calculate_image_metrics(self, img_path: str) -> Dict[str, float]:
        """Calculate various metrics for image quality assessment"""
        try:
            # Ensure imports are available
            self._ensure_imports()
            
            # Initialize metrics dictionary
            metrics = {
                'contrast': 0.0,
                'texture': 0.0,
                'color_variance': 0.0,
                'entropy': 0.0
            }
            
            # Use PIL for faster loading if available
            if hasattr(self, '_has_pil') and self._has_pil:
                # Open image with PIL
                img = self._pil.open(img_path)
                
                # Resize large images
                max_dimension = 800
                if max(img.width, img.height) > max_dimension:
                    if img.width > img.height:
                        new_size = (max_dimension, int(img.height * max_dimension / img.width))
                    else:
                        new_size = (int(img.width * max_dimension / img.height), max_dimension)
                    img = img.resize(new_size, self._pil.LANCZOS)
                
                # Convert to numpy array for processing
                img_array = self._np.array(img)
            else:
                # Fall back to matplotlib
                img_array = self._plt.imread(img_path)
            
            # 1. Contrast: Standard deviation of pixel values
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # Convert RGB to grayscale
                gray = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
            else:
                gray = img_array
                
            contrast = self._np.std(gray)
            
            # Normalize based on data type
            if img_array.dtype == self._np.uint8:
                max_std = 127.5  # Max possible std for uint8
                contrast = min(contrast / max_std, 1.0)
            elif img_array.dtype == self._np.float32 or img_array.dtype == self._np.float64:
                max_std = 0.5    # Max possible std for float [0,1]
                contrast = min(contrast / max_std, 1.0)
            else:
                # Just normalize to [0,1] range if unknown
                contrast = min(contrast / 255.0, 1.0)
                
            metrics['contrast'] = contrast
            
            # 2. Texture: Use average gradient magnitude
            # Calculate gradients using simple difference
            dx = self._np.diff(gray, axis=1)
            dy = self._np.diff(gray, axis=0)
            
            # Pad to maintain size
            dx = self._np.pad(dx, ((0, 0), (0, 1)), mode='constant')
            dy = self._np.pad(dy, ((0, 1), (0, 0)), mode='constant')
            
            # Calculate magnitude
            gradient_magnitude = self._np.sqrt(dx**2 + dy**2)
            
            # Normalize
            if img_array.dtype == self._np.uint8:
                texture = self._np.mean(gradient_magnitude) / 255.0
            else:
                texture = self._np.mean(gradient_magnitude)
                
            texture = min(texture * 5.0, 1.0)  # Scale up for better distribution
            metrics['texture'] = texture
            
            # 3. Color variance: Variance across color channels
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # Calculate mean variance across all three channels
                r_var = self._np.var(img_array[:,:,0])
                g_var = self._np.var(img_array[:,:,1])
                b_var = self._np.var(img_array[:,:,2])
                
                color_variance = (r_var + g_var + b_var) / 3.0
                
                # Normalize
                if img_array.dtype == self._np.uint8:
                    color_variance = min(color_variance / (255.0 * 255.0 / 4.0), 1.0)
                else:
                    color_variance = min(color_variance * 4.0, 1.0)
            else:
                color_variance = 0.0
                
            metrics['color_variance'] = color_variance
            
            # 4. Simple entropy: Histogram variety
            # Use histogram spread as a simple entropy measure
            if len(img_array.shape) == 3:
                # Calculate histograms for each channel
                hist_r, _ = self._np.histogram(img_array[:,:,0], bins=32)
                hist_g, _ = self._np.histogram(img_array[:,:,1], bins=32)
                hist_b, _ = self._np.histogram(img_array[:,:,2], bins=32)
                
                # Count non-empty bins
                non_empty_bins = (self._np.count_nonzero(hist_r) + 
                                self._np.count_nonzero(hist_g) + 
                                self._np.count_nonzero(hist_b)) / 3.0
                
                # Normalize
                entropy = min(non_empty_bins / 32.0, 1.0)
            else:
                hist, _ = self._np.histogram(gray, bins=32)
                non_empty_bins = self._np.count_nonzero(hist)
                entropy = min(non_empty_bins / 32.0, 1.0)
                
            metrics['entropy'] = entropy

            return metrics
            
        except Exception as e:
            print(f"Error calculating image metrics for {img_path}: {e}")
            # Return default values
            return {
                'contrast': 0.0,
                'texture': 0.0,
                'color_variance': 0.0,
                'entropy': 0.0
            }
    
    def _get_image_metrics(self, img_path: str) -> Dict[str, float]:
        """
        Get image metrics with caching support
        
        Args:
            img_path: Path to the image
            
        Returns:
            Dictionary of metrics
        """
        # Create cache key based on filename and modification time
        try:
            if img_path in self.metrics_cache:
                # If we have it in memory cache, use that
                return self.metrics_cache[img_path].copy()
                
            # Otherwise calculate metrics
            metrics = self._calculate_image_metrics(img_path)
            
            # Store in cache
            self.metrics_cache[img_path] = metrics.copy()
            
            return metrics
        except Exception as e:
            print(f"Error in cached metrics: {e}")
            # Fall back to direct calculation
            return self._calculate_image_metrics(img_path)
    
    def _process_image_batch(self, args):
        """
        Process a batch of images to calculate metrics
        
        Args:
            args: Tuple containing (img_paths, altitude_map, visibility_map, has_visibility_data)
            
        Returns:
            List of image metrics dictionaries
        """
        img_paths, altitude_map, visibility_map, has_visibility_data = args
        
        # Import libraries locally in this function (so they don't need to be pickled)
        import numpy as np
        try:
            from PIL import Image, ImageStat
            has_pil = True
        except ImportError:
            has_pil = False
            import matplotlib.pyplot as plt
        
        results = []
        for img_path in img_paths:
            try:
                # Initialize metrics with default values
                metrics = {
                    'contrast': 0.0,
                    'texture': 0.0,
                    'color_variance': 0.0,
                    'entropy': 0.0,
                    'path': img_path,
                    'filename': os.path.basename(img_path)
                }
                
                # Use PIL for faster loading if available
                if has_pil:
                    # Open image with PIL
                    img = Image.open(img_path)
                    
                    # Resize large images
                    max_dimension = 800
                    if max(img.width, img.height) > max_dimension:
                        if img.width > img.height:
                            new_size = (max_dimension, int(img.height * max_dimension / img.width))
                        else:
                            new_size = (int(img.width * max_dimension / img.height), max_dimension)
                        img = img.resize(new_size, Image.LANCZOS)
                    
                    # Convert to numpy array for processing
                    img_array = np.array(img)
                else:
                    # Fall back to matplotlib
                    img_array = plt.imread(img_path)
                
                # Calculate metrics from img_array
                # 1. Contrast: Standard deviation of pixel values
                if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                    # Convert RGB to grayscale
                    gray = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
                else:
                    gray = img_array
                    
                contrast = np.std(gray)
                
                # Normalize based on data type
                if img_array.dtype == np.uint8:
                    max_std = 127.5  # Max possible std for uint8
                    contrast = min(contrast / max_std, 1.0)
                elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    max_std = 0.5    # Max possible std for float [0,1]
                    contrast = min(contrast / max_std, 1.0)
                else:
                    # Just normalize to [0,1] range if unknown
                    contrast = min(contrast / 255.0, 1.0)
                    
                metrics['contrast'] = contrast
                
                # Continue with other metrics calculations...
                # Add each metric to the metrics dict
                
                # Get altitude if available
                filename = os.path.basename(img_path)
                if altitude_map and filename in altitude_map:
                    metrics['altitude'] = altitude_map[filename]
                
                # Get visibility data if available
                if has_visibility_data and visibility_map and filename in visibility_map:
                    vis_data = visibility_map[filename]
                    metrics['visibility_category'] = vis_data['category']
                    metrics['visibility_confidence'] = vis_data['confidence']
                    
                    # Calculate visibility score
                    # ...
                    
                results.append(metrics)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add the image with default metrics
                default_metrics = {
                    'contrast': 0.0,
                    'texture': 0.0,
                    'color_variance': 0.0,
                    'entropy': 0.0,
                    'path': img_path,
                    'filename': os.path.basename(img_path)
                }
                
                # Still add visibility and altitude if available
                filename = os.path.basename(img_path)
                if altitude_map and filename in altitude_map:
                    default_metrics['altitude'] = altitude_map[filename]
                    
                if has_visibility_data and visibility_map and filename in visibility_map:
                    vis_data = visibility_map[filename]
                    default_metrics['visibility_category'] = vis_data['category']
                    default_metrics['visibility_confidence'] = vis_data['confidence']
                    default_metrics['visibility_score'] = 0.0
                    
                results.append(default_metrics)
        
        return results

    def _process_batch_in_thread(self, img_paths, altitude_map, visibility_map, has_visibility_data):
        """
        Process a batch of images to calculate metrics - designed for ThreadPoolExecutor
        
        Args:
            img_paths: List of image paths to process
            altitude_map: Dictionary mapping filenames to altitudes
            visibility_map: Dictionary mapping filenames to visibility data
            has_visibility_data: Whether visibility data is available
            
        Returns:
            List of image metrics dictionaries
        """
        results = []
        for img_path in img_paths:
            try:
                # Calculate metrics
                metrics = self._calculate_image_metrics(img_path)
                
                # Get filename
                filename = os.path.basename(img_path)
                
                # Add metadata
                metrics['path'] = img_path
                metrics['filename'] = filename
                
                # Add altitude if available
                if altitude_map and filename in altitude_map:
                    metrics['altitude'] = altitude_map[filename]
                
                # Add visibility data if available
                if has_visibility_data and visibility_map and filename in visibility_map:
                    vis_data = visibility_map[filename]
                    metrics['visibility_category'] = vis_data['category']
                    metrics['visibility_confidence'] = vis_data['confidence']
                    
                    # Calculate visibility score
                    vis_category_score = 0.0
                    if vis_data['category'] == 'great_visibility':
                        vis_category_score = 1.0
                    elif vis_data['category'] == 'good_visibility':
                        vis_category_score = 0.75
                    elif vis_data['category'] == 'low_visibility':
                        vis_category_score = 0.3
                    
                    metrics['visibility_score'] = vis_category_score * vis_data['confidence']
                else:
                    metrics['visibility_score'] = None
                
                results.append(metrics)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return results

    def _select_diverse_highlights(self, metrics_df, count: int):
        """
        Select a diverse set of highlight images by using clustering and top scores
        
        Args:
            metrics_df: DataFrame with image metrics
            count: Number of images to select
            
        Returns:
            DataFrame with selected images
        """
        try:
            # If we have very few images, just return top ones
            if len(metrics_df) <= count:
                return metrics_df
                
            # Try to use clustering to ensure diversity
            try:
                from sklearn.cluster import KMeans
                
                # Get metrics columns for clustering
                cluster_columns = ['contrast', 'texture', 'color_variance', 'entropy']
                
                # Add visibility score if available
                if 'visibility_score' in metrics_df.columns:
                    non_null = metrics_df['visibility_score'].notna()
                    if non_null.any():
                        cluster_columns.append('visibility_score')
                
                # Get cluster data excluding NaN
                cluster_data = metrics_df[cluster_columns].fillna(0).values
                
                # Determine number of clusters based on count
                n_clusters = min(count, len(metrics_df) // 2)
                n_clusters = max(n_clusters, 2)  # At least 2 clusters
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                metrics_df['cluster'] = kmeans.fit_predict(cluster_data)
                
                # Select top images from each cluster
                selected = []
                
                # Calculate how many to take from each cluster
                images_per_cluster = count // n_clusters
                remainder = count % n_clusters
                
                for i in range(n_clusters):
                    cluster_df = metrics_df[metrics_df['cluster'] == i].sort_values('combined_score', ascending=False)
                    
                    # Take extra image from highest scoring clusters if needed
                    take_count = images_per_cluster
                    if i < remainder:
                        take_count += 1
                        
                    selected.append(cluster_df.head(take_count))
                
                # Combine all selected images
                return self._pd.concat(selected).sort_values('combined_score', ascending=False)
                
            except ImportError:
                # If scikit-learn is not available, use a simpler approach
                # Just return top images
                return metrics_df.head(count)
                
        except Exception as e:
            print(f"Error in diverse selection: {e}")
            # Safety check before returning top images
            if len(metrics_df) == 0:
                print("No valid metrics found for any images!")
                return metrics_df
            else:
                # Sort by combined score and ensure at least one image has a non-zero score
                if metrics_df['combined_score'].max() <= 0:
                    print("Warning: All images have zero scores")
                    # In this case, just select randomly to provide some results
                    return metrics_df.sample(min(count, len(metrics_df)))
                else:
                    # Return top images by score
                    return metrics_df.head(count)
    
    def _create_highlights_html(self, highlight_dir: str, metrics_df, has_visibility: bool) -> str:
        """
        Create an HTML summary of highlight images
        
        Args:
            highlight_dir: Directory with highlight images
            metrics_df: DataFrame with selected images and their metrics
            has_visibility: Whether visibility data is available
            
        Returns:
            Path to the created HTML file
        """
        try:
            html_path = os.path.join(highlight_dir, "highlights.html")
            
            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Highlight Images</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    h1 { color: #333; }
                    .gallery { display: flex; flex-wrap: wrap; justify-content: center; }
                    .item { 
                        margin: 15px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
                        padding: 10px; 
                        background-color: white;
                        border-radius: 5px;
                        transition: transform 0.2s;
                    }
                    .item:hover { transform: scale(1.03); }
                    img { 
                        max-width: 300px; 
                        max-height: 300px; 
                        display: block;
                        margin: 0 auto;
                    }
                    .info { 
                        margin-top: 10px; 
                        font-size: 14px;
                        color: #555;
                    }
                    .filename { 
                        font-weight: bold; 
                        margin-bottom: 5px;
                        color: #333;
                        font-size: 16px;
                    }
                    .metrics {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }
                    .metric {
                        margin: 3px 0;
                        width: 48%;
                    }
                    .score {
                        font-weight: bold;
                        color: #0066cc;
                        font-size: 15px;
                        margin-bottom: 8px;
                    }
                    .category {
                        font-weight: bold;
                    }
                    .zero { color: #cc0000; }
                    .low { color: #ff6600; }
                    .good { color: #339933; }
                    .great { color: #0066cc; }
                </style>
            </head>
            <body>
                <h1>Highlight Images</h1>
                <p>Selected representative images from the dataset based on image metrics</p>
                <div class="gallery">
            """
            
            # Add each highlight image
            for _, row in metrics_df.iterrows():
                filename = row['filename']
                
                # Format metrics
                combined_score = row['combined_score']
                contrast = row['contrast']
                texture = row['texture']
                color_var = row['color_variance']
                entropy = row['entropy']
                
                # Add visibility info if available
                visibility_html = ""
                category_class = ""
                
                if has_visibility and 'visibility_category' in row and not self._pd.isna(row['visibility_category']):
                    visibility = row['visibility_category']
                    confidence = row['visibility_confidence']
                    
                    # Determine category CSS class
                    if visibility == "zero_visibility":
                        category_class = "zero"
                    elif visibility == "low_visibility":
                        category_class = "low"
                    elif visibility == "good_visibility":
                        category_class = "good"
                    elif visibility == "great_visibility":
                        category_class = "great"
                    
                    visibility_html = f"""
                    <div class="metric category {category_class}">Visibility Category: {visibility}</div>
                    <div class="metric">Visibility Model Confidence: {confidence:.2f}</div>
                    """
                
                # Add altitude if available
                altitude_html = ""
                if 'altitude' in row and not self._pd.isna(row['altitude']):
                    altitude_html = f"<div class='metric'>Altitude: {row['altitude']:.2f}m</div>"
                
                html_content += f"""
                    <div class="item">
                        <img src="{filename}" alt="{filename}">
                        <div class="info">
                            <div class="filename">{filename}</div>
                            <div class="score">Combined Score: {combined_score:.2f}</div>
                            <div class="metrics">
                                <div class="metric">Contrast: {contrast:.2f}</div>
                                <div class="metric">Texture: {texture:.2f}</div>
                                <div class="metric">Color Variance: {color_var:.2f}</div>
                                <div class="metric">Entropy: {entropy:.2f}</div>
                                {altitude_html}
                                {visibility_html}
                            </div>
                        </div>
                    </div>
                """
            
            # Close HTML
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Write HTML file
            with open(html_path, 'w') as f:
                f.write(html_content)
                
            return html_path
            
        except Exception as e:
            print(f"Error creating highlights HTML: {e}")
            return os.path.join(highlight_dir, "highlights.html")  # Return path even if failed

    def create_highlight_panel(self, highlight_dir: str, output_folder: str, metrics_df, top_count: int = 4) -> str:
        """
        Create a multi-panel image with the top highlights for report inclusion
        
        Args:
            highlight_dir: Directory with highlight images
            output_folder: Main output directory where the panel will be saved
            metrics_df: DataFrame with selected images and their metrics
            top_count: Number of top images to include (default: 4)
            
        Returns:
            Path to the created panel image
        """
        try:
            # Get required imports
            self._ensure_imports()
            
            # Explicitly import matplotlib and PIL here if not already available
            if self._plt is None:
                try:
                    import matplotlib.pyplot as plt
                    self._plt = plt
                except ImportError:
                    print("Error: Matplotlib is required for creating highlight panels")
                    return None
                    
            # Import ExifTags and PIL modules for EXIF extraction
            try:
                from PIL import Image, ExifTags
                GPSTAGS = {
                    0: "GPSVersionID",
                    1: "GPSLatitudeRef",
                    2: "GPSLatitude",
                    3: "GPSLongitudeRef",
                    4: "GPSLongitude",
                    5: "GPSAltitudeRef",
                    6: "GPSAltitude"
                }
            except ImportError:
                print("Error: PIL is required for EXIF data extraction")
                return None
            
            # Import gridspec
            try:
                import matplotlib.gridspec as gridspec
            except ImportError:
                print("Error: Matplotlib gridspec is required for creating highlight panels")
                return None
            
            # Helper function to convert DMS (Degrees, Minutes, Seconds) to decimal degrees
            def dms_to_dd(degrees, minutes, seconds, direction):
                dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
                if direction == 'S' or direction == 'W':
                    dd *= -1
                return dd

            # Helper function to convert GPS coordinates - defined BEFORE it's used
            def _convert_to_degrees(value):
                try:
                    if isinstance(value, tuple) and len(value) == 3:
                        degrees = float(value[0][0]) / float(value[0][1])
                        minutes = float(value[1][0]) / float(value[1][1])
                        seconds = float(value[2][0]) / float(value[2][1])
                        
                        return degrees + (minutes / 60.0) + (seconds / 3600.0)
                    else:
                        # Handle rationals stored differently
                        d, m, s = 0, 0, 0
                        if len(value) > 0:
                            d = float(value[0].numerator) / float(value[0].denominator) if hasattr(value[0], 'numerator') else float(value[0][0]) / float(value[0][1])
                        if len(value) > 1:
                            m = float(value[1].numerator) / float(value[1].denominator) if hasattr(value[1], 'numerator') else float(value[1][0]) / float(value[1][1])
                        if len(value) > 2:
                            s = float(value[2].numerator) / float(value[2].denominator) if hasattr(value[2], 'numerator') else float(value[2][0]) / float(value[2][1])
                        return d + (m / 60.0) + (s / 3600.0)
                except Exception as e:
                    print(f"Error converting GPS value: {e}, value: {value}")
                    return 0.0
            
            # Helper function to extract GPS data from EXIF
            def extract_gps_data(img_path):
                try:
                    with self._pil.open(img_path) as img:
                        exif_data = img._getexif()
                        if exif_data is None:
                            return None
                        
                        gps_info = {}
                        for tag, value in exif_data.items():
                            tag_name = self._pil.ExifTags.TAGS.get(tag, tag)
                            if tag_name == 'GPSInfo':
                                for gps_tag, gps_value in value.items():
                                    gps_info[self._pil.ExifTags.GPSTAGS.get(gps_tag, gps_tag)] = gps_value
                        
                        if not gps_info:
                            return None
                            
                        result = {}
                        
                        # Handle Latitude
                        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                            try:
                                lat_dms = gps_info['GPSLatitude']
                                lat_ref = gps_info['GPSLatitudeRef']
                                
                                # Check if we have a tuple/list of components (DMS format)
                                if isinstance(lat_dms, tuple) and len(lat_dms) == 3:
                                    # Convert each component (which might be a fraction) to float
                                    lat_d = float(lat_dms[0].numerator) / float(lat_dms[0].denominator) if hasattr(lat_dms[0], 'numerator') else float(lat_dms[0])
                                    lat_m = float(lat_dms[1].numerator) / float(lat_dms[1].denominator) if hasattr(lat_dms[1], 'numerator') else float(lat_dms[1])
                                    lat_s = float(lat_dms[2].numerator) / float(lat_dms[2].denominator) if hasattr(lat_dms[2], 'numerator') else float(lat_dms[2])
                                    
                                    # Convert to decimal degrees
                                    latitude = dms_to_dd(lat_d, lat_m, lat_s, lat_ref)
                                    result['latitude'] = latitude
                                else:
                                    # Already in decimal degrees
                                    result['latitude'] = float(lat_dms)
                            except (ValueError, TypeError, ZeroDivisionError) as e:
                                print(f"Error parsing latitude: {e}, value: {gps_info['GPSLatitude']}")
                        
                        # Handle Longitude
                        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                            try:
                                lon_dms = gps_info['GPSLongitude']
                                lon_ref = gps_info['GPSLongitudeRef']
                                
                                # Check if we have a tuple/list of components (DMS format)
                                if isinstance(lon_dms, tuple) and len(lon_dms) == 3:
                                    # Convert each component (which might be a fraction) to float
                                    lon_d = float(lon_dms[0].numerator) / float(lon_dms[0].denominator) if hasattr(lon_dms[0], 'numerator') else float(lon_dms[0])
                                    lon_m = float(lon_dms[1].numerator) / float(lon_dms[1].denominator) if hasattr(lon_dms[1], 'numerator') else float(lon_dms[1])
                                    lon_s = float(lon_dms[2].numerator) / float(lon_dms[2].denominator) if hasattr(lon_dms[2], 'numerator') else float(lon_dms[2])
                                    
                                    # Convert to decimal degrees
                                    longitude = dms_to_dd(lon_d, lon_m, lon_s, lon_ref)
                                    result['longitude'] = longitude
                                else:
                                    # Already in decimal degrees
                                    result['longitude'] = float(lon_dms)
                            except (ValueError, TypeError, ZeroDivisionError) as e:
                                print(f"Error parsing longitude: {e}, value: {gps_info['GPSLongitude']}")
                        
                        # Handle Altitude
                        if 'GPSAltitude' in gps_info:
                            try:
                                if hasattr(gps_info['GPSAltitude'], 'numerator'):
                                    altitude = float(gps_info['GPSAltitude'].numerator) / float(gps_info['GPSAltitude'].denominator)
                                else:
                                    altitude = float(gps_info['GPSAltitude'])
                                    
                                if 'GPSAltitudeRef' in gps_info and gps_info['GPSAltitudeRef'] == 1:
                                    altitude = -altitude
                                
                                result['altitude'] = altitude
                            except (ValueError, TypeError, ZeroDivisionError) as e:
                                print(f"Error parsing altitude: {e}, value: {gps_info['GPSAltitude']}")
                        
                        # If we found any GPS data, return it
                        if result:
                            return result
                        return None
                        
                except Exception as e:
                    print(f"Error extracting GPS data from {img_path}: {e}")
                    return None
            
            # Get the top N images sorted by combined score
            top_df = metrics_df.sort_values('combined_score', ascending=False).head(top_count)
            
            # Create figure with 2x2 grid for 4 images (more compact, less whitespace)
            fig = self._plt.figure(figsize=(10, 8))
            
            # Use tight layout and reduce spacing between subplots
            gs = self._plt.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)
            
            # Add each image to the grid
            for i, (_, row) in enumerate(top_df.iterrows()):
                if i >= top_count:
                    break
                    
                # Calculate grid position
                row_idx = i // 2
                col_idx = i % 2
                
                # Create subplot
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                # Get image path
                filename = row['filename']
                img_path = os.path.join(highlight_dir, filename)
                
                try:
                    # Read and display image
                    img = self._plt.imread(img_path)
                    ax.imshow(img)
                    
                    # Use only filename without extension as title
                    title = f"{os.path.splitext(os.path.basename(filename))[0]}"
                    ax.set_title(title, fontsize=8, pad=2)  # Reduced padding
                    
                    # Get GPS data from image
                    gps_data = extract_gps_data(img_path)
                    
                    # Format text with ONLY coordinates and altitude (no filename)
                    if gps_data and 'latitude' in gps_data and 'longitude' in gps_data and 'altitude' in gps_data:
                        text = f"Lat: {gps_data['latitude']:.6f}\nLon: {gps_data['longitude']:.6f}\nAlt: {gps_data['altitude']:.2f}m"
                    elif 'latitude' in row and 'longitude' in row and 'altitude' in row:
                        # Fall back to DataFrame values if available
                        text = f"Lat: {row['latitude']:.6f}\nLon: {row['longitude']:.6f}\nAlt: {row['altitude']:.2f}m"
                    else:
                        # Provide empty text if no coordinates are available
                        text = ""
                    
                    # Add text box with metrics - adjust position if needed
                    ax.text(0.02, 0.02, text, 
                            transform=ax.transAxes, 
                            fontsize=7,  # Reduced font size
                            verticalalignment='bottom', 
                            bbox=dict(boxstyle='round,pad=0.3',  # Reduced padding
                                      facecolor='white', 
                                      alpha=0.7))
                    
                except Exception as e:
                    print(f"Error adding image to panel: {e}")
                    ax.text(0.5, 0.5, f"Error loading image:\n{filename}", 
                            ha='center', va='center')
                    
                # Turn off axis labels
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Save the panel image with tight bbox to eliminate extra margins
            # Save to the main output directory with standardized name
            panel_path = os.path.join(output_folder, "Image_Top_Highlights.png")
            fig.savefig(panel_path, dpi=300, bbox_inches='tight')
            self._plt.close(fig)
            
            return panel_path
                
        except Exception as e:
            print(f"Error creating highlight panel: {e}")
            return None

    def export_metrics_to_csv(self, output_folder: str, metrics_df) -> bool:
        """
        Export image metrics to the Image_Metrics.csv file
        
        Args:
            output_folder: Directory containing Image_Metrics.csv
            metrics_df: DataFrame with calculated metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_imports()  # This will import pandas as self._pd
            csv_path = os.path.join(output_folder, "Image_Metrics.csv")
            
            # Check if the CSV file exists
            if os.path.exists(csv_path):
                # Load existing CSV
                try:
                    locations_df = self._pd.read_csv(csv_path)
                    print(f"Loaded existing Image_Metrics.csv with {len(locations_df)} entries")
                    
                    # Create a mapping of filenames to metrics
                    metrics_dict = {}
                    for _, row in metrics_df.iterrows():
                        if 'filename' in row and isinstance(row['filename'], str):
                            metrics_dict[row['filename']] = row.to_dict()
                    
                    # Columns to add to the CSV
                    metric_columns = ['contrast', 'texture', 'color_variance', 'entropy', 'combined_score']
                    
                    # Initialize new columns if they don't exist
                    for col in metric_columns:
                        if col not in locations_df.columns:
                            locations_df[col] = None
                    
                    # Update the entries with metrics
                    updated_count = 0
                    for i, row in locations_df.iterrows():
                        filename = None
                        
                        # Determine filename column
                        if 'filename' in locations_df.columns:
                            filename = row['filename']
                        elif locations_df.shape[1] >= 12:  # Assuming the filename is in the 12th column
                            filename = row.iloc[11]  # 0-indexed, so 11 is the 12th column
                        
                        if filename and filename in metrics_dict:
                            for col in metric_columns:
                                if col in metrics_dict[filename]:
                                    locations_df.at[i, col] = metrics_dict[filename][col]
                            updated_count += 1
                    
                    print(f"Updated {updated_count} entries with metrics")
                    
                    # Save the updated CSV
                    locations_df.to_csv(csv_path, index=False)
                    print(f"Saved updated Image_Metrics.csv with metrics to {csv_path}")
                    return True
                    
                except Exception as e:
                    print(f"Error updating existing CSV: {e}")
                    print(traceback.format_exc())
                    return False
            else:
                # If no existing CSV, we should not create a metrics-only file
                # The Image_Metrics.csv should be created by the main processing pipeline first
                # with GPS/EXIF data, and then we add metrics to it
                print("Warning: Image_Metrics.csv not found. Cannot add metrics without location data.")
                print("The location CSV should be created by the main processing pipeline first.")
                return False
                
        except Exception as e:
            print(f"Error exporting metrics to CSV: {e}")
            print(traceback.format_exc())
            return False

    def load_metrics_from_csv(self, output_folder: str, input_folder: str):
        """
        Load pre-calculated metrics from Image_Metrics.csv if available
        
        Args:
            output_folder: Directory containing Image_Metrics.csv
            input_folder: Directory with input images to match with metrics
            
        Returns:
            DataFrame with loaded metrics or empty DataFrame if not available
        """
        try:
            self._ensure_imports()  # This will import pandas as self._pd
            csv_path = os.path.join(output_folder, "Image_Metrics.csv")
            
            if not os.path.exists(csv_path):
                print("No existing Image_Metrics.csv found")
                return self._pd.DataFrame()
            
            # Load the CSV
            locations_df = self._pd.read_csv(csv_path)
            
            # Check if it has metrics columns
            metric_columns = ['contrast', 'texture', 'color_variance', 'entropy']
            has_metrics = all(col in locations_df.columns for col in metric_columns)
            
            if not has_metrics:
                print("Existing CSV doesn't contain metrics columns")
                return self._pd.DataFrame()
            
            print(f"Found existing metrics for {len(locations_df)} images")
            
            # Determine filename column
            filename_col = None
            if 'filename' in locations_df.columns:
                filename_col = 'filename'
            elif locations_df.shape[1] >= 12:  # Assuming the filename is in the 12th column
                # Rename the 12th column to 'filename' for consistency
                locations_df = locations_df.rename(columns={locations_df.columns[11]: 'filename'})
                filename_col = 'filename'
            
            if not filename_col:
                print("Could not determine filename column in CSV")
                return self._pd.DataFrame()
            
            # Add path information for each file
            paths_found = []
            for filename in locations_df[filename_col].unique():
                # Look for the file in the input folder
                for root, _, files in os.walk(input_folder):
                    if filename in files:
                        path = os.path.join(root, filename)
                        idx = locations_df[locations_df[filename_col] == filename].index
                        locations_df.loc[idx, 'path'] = path
                        paths_found.append(filename)
                        break
            
            # Filter to only include entries where a file was found
            metrics_df = locations_df[locations_df[filename_col].isin(paths_found)].copy()
            
            # If we have combined_score, use it; otherwise, calculate it
            if 'combined_score' not in metrics_df.columns:
                # Calculate combined score based on available metrics
                metrics_df['combined_score'] = (
                    metrics_df['contrast'] * self.metrics_weight['contrast'] +
                    metrics_df['texture'] * self.metrics_weight['texture'] +
                    metrics_df['color_variance'] * self.metrics_weight['color_variance'] +
                    metrics_df['entropy'] * self.metrics_weight['entropy']
                )
            
            print(f"Successfully loaded metrics for {len(metrics_df)} images")
            return metrics_df
            
        except Exception as e:
            print(f"Error loading metrics from CSV: {e}")
            print(traceback.format_exc())
            return self._pd.DataFrame()
    
    def calculate_image_metrics(self, img_path: str) -> Optional[Dict]:
        """
        Calculate quality metrics for a single image
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Dictionary with calculated metrics or None if error
        """
        try:
            # Import libraries locally
            import numpy as np
            from PIL import Image, ImageStat
            
            # Check if file exists and is readable
            if not os.path.exists(img_path):
                print(f"Warning: Image file does not exist: {img_path}")
                return None
                      
            # Load image
            with Image.open(img_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize large images for faster processing
                max_dimension = 800
                if max(img.width, img.height) > max_dimension:
                    if img.width > img.height:
                        new_size = (max_dimension, int(img.height * max_dimension / img.width))
                    else:
                        new_size = (int(img.width * max_dimension / img.height), max_dimension)
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert to numpy array for calculations
                img_array = np.array(img)
                
                # Calculate contrast (standard deviation of luminance)
                gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
                contrast = np.std(gray) / 255.0
                
                # Calculate texture (Laplacian variance)
                try:
                    from scipy import ndimage
                    laplacian = ndimage.laplace(gray)
                    texture = np.var(laplacian) / 10000.0  # Normalize
                except ImportError:
                    # Fallback: use simple gradient if scipy not available
                    dx = np.diff(gray, axis=1)
                    dy = np.diff(gray, axis=0)
                    gradient_magnitude = np.sqrt(
                        np.pad(dx, ((0, 0), (0, 1)), mode='constant')**2 + 
                        np.pad(dy, ((0, 1), (0, 0)), mode='constant')**2
                    )
                    texture = np.mean(gradient_magnitude) / 255.0
                
                # Calculate color variance
                color_variance = np.var(img_array) / 65535.0  # Normalize for RGB
                
                # Calculate entropy
                hist, _ = np.histogram(gray, bins=256, range=(0, 256))
                hist = hist[hist > 0]  # Remove zeros
                if len(hist) > 0:
                    prob = hist / hist.sum()
                    entropy = -np.sum(prob * np.log2(prob)) / 8.0  # Normalize
                else:
                    entropy = 0.0
                
                return {
                    'contrast': float(min(contrast, 1.0)),
                    'texture': float(min(texture, 1.0)),
                    'color_variance': float(min(color_variance, 1.0)),
                    'entropy': float(min(entropy, 1.0))
                }
                
        except Exception as e:
            print(f"Error calculating metrics for {img_path}: {e}")
            return {
                'contrast': 0.0,
                'texture': 0.0,
                'color_variance': 0.0,
                'entropy': 0.0
            }
    
    def select_highlights_from_csv(self, csv_path: str, output_folder: str, 
                                  count: int = 10, progress_callback: Optional[Callable] = None,
                                  altitude_threshold: Optional[float] = None,
                                  min_altitude_threshold: Optional[float] = 2.0) -> List[str]:
        """
        Select highlight images using the master CSV file
        
        Args:
            csv_path: Path to the master Image_Metrics.csv file
            output_folder: Directory to save highlight images
            count: Number of highlight images to select
            progress_callback: Optional callback for progress updates
            altitude_threshold: Optional maximum altitude threshold to filter images
            min_altitude_threshold: Optional minimum altitude threshold
            
        Returns:
            List of paths to selected highlight images
        """
        try:
            self._ensure_imports()
            
            # Check if CSV exists, if not try to create it
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                print("Attempting to create master CSV file...")
                
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
                        print("Failed to create master CSV file")
                        return []
                else:
                    print("Could not find input folder to create CSV")
                    return []
            
            # Load the CSV file
            if not self._pd:
                import pandas as pd
                self._pd = pd
            
            df = self._pd.read_csv(csv_path)
            
            # Check required columns
            required_columns = ['filename', 'file_path']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing required columns in CSV: {missing_columns}")
                return []
            
            # Filter out rows with missing path information
            df_filtered = df.dropna(subset=['file_path'])
            
            if len(df_filtered) == 0:
                print("No valid image paths found in CSV")
                return []
            
            # Apply altitude filtering if specified
            if altitude_threshold is not None and 'altitude' in df_filtered.columns:
                initial_count = len(df_filtered)
                # Filter out rows with missing altitude values first
                df_with_altitude = df_filtered.dropna(subset=['altitude'])
                if len(df_with_altitude) > 0:
                    df_filtered = df_with_altitude[df_with_altitude['altitude'] <= altitude_threshold]
                    print(f"Filtered by altitude <= {altitude_threshold}m: {initial_count} -> {len(df_filtered)} images")
                else:
                    print(f"No altitude data available for filtering")
            
            if min_altitude_threshold is not None and 'altitude' in df_filtered.columns:
                initial_count = len(df_filtered)
                # Filter out rows with missing altitude values first
                df_with_altitude = df_filtered.dropna(subset=['altitude'])
                if len(df_with_altitude) > 0:
                    df_filtered = df_with_altitude[df_with_altitude['altitude'] >= min_altitude_threshold]
                    print(f"Filtered by altitude >= {min_altitude_threshold}m: {initial_count} -> {len(df_filtered)} images")
                else:
                    print(f"No altitude data available for filtering, using all images")
                    # Keep all images if no altitude data
            
            # Convert DataFrame to image paths list
            image_paths = df_filtered['file_path'].tolist()
            
            print(f"Loaded {len(image_paths)} image paths from CSV for highlight selection")
            
            # Prepare visibility results from CSV if available
            visibility_results = None
            if 'visibility' in df_filtered.columns and 'visibility_confidence' in df_filtered.columns:
                visibility_results = {
                    'analyzed_images': []
                }
                
                for _, row in df_filtered.iterrows():
                    if self._pd.notna(row.get('visibility')):
                        result = {
                            'image': row['filename'],
                            'visibility': row['visibility'],
                            'confidence': row.get('visibility_confidence', 0.0)
                        }
                        visibility_results['analyzed_images'].append(result)
                
                if visibility_results['analyzed_images']:
                    print(f"Found visibility data for {len(visibility_results['analyzed_images'])} images in CSV")
                else:
                    visibility_results = None
            
            # Find input folder from first image path
            if image_paths:
                first_image_path = image_paths[0]
                input_folder = os.path.dirname(first_image_path)
                
                # Try to find a common parent folder if paths are diverse
                all_dirs = [os.path.dirname(path) for path in image_paths]
                if len(set(all_dirs)) > 1:
                    # Find common parent directory
                    common_parent = os.path.commonpath(all_dirs)
                    input_folder = common_parent
                
                print(f"Using input folder: {input_folder}")
            else:
                print("No image paths found")
                return []
            
            # Use the existing select_highlights method
            highlight_paths = self.select_highlights(
                input_folder=input_folder,
                output_folder=output_folder,
                count=count,
                progress_callback=progress_callback,
                altitude_threshold=altitude_threshold,
                min_altitude_threshold=min_altitude_threshold,
                visibility_results=visibility_results
            )
            
            if highlight_paths:
                # Update the master CSV with highlight information
                self._update_master_csv_with_highlights(csv_path, highlight_paths)
                
            return highlight_paths
            
        except Exception as e:
            print(f"Error in select_highlights_from_csv: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    def _update_master_csv_with_highlights(self, csv_path: str, highlight_paths: List[str]) -> None:
        """
        Update the master CSV file with highlight selection results
        
        Args:
            csv_path: Path to the master CSV file
            highlight_paths: List of paths to selected highlight images
        """
        try:
            # Ensure pandas is imported
            self._ensure_imports()
            
            # Load the CSV file
            df = self._pd.read_csv(csv_path)
            
            # Add highlight column if it doesn't exist
            if 'is_highlight' not in df.columns:
                df['is_highlight'] = False
            
            # Convert the column to boolean to avoid dtype warnings
            df['is_highlight'] = df['is_highlight'].astype(bool)
            
            # Create a set of highlight filenames for quick lookup
            highlight_filenames = set()
            for path in highlight_paths:
                filename = os.path.basename(path)
                highlight_filenames.add(filename)
            
            # Update master CSV with highlight information
            for idx, row in df.iterrows():
                filename = row['filename']
                if filename in highlight_filenames:
                    df.at[idx, 'is_highlight'] = True
                else:
                    df.at[idx, 'is_highlight'] = False
            
            # Save updated CSV
            df.to_csv(csv_path, index=False)
            print(f"Updated master CSV with highlight information: {csv_path}")
            
        except Exception as e:
            print(f"Error updating master CSV with highlight information: {e}")
