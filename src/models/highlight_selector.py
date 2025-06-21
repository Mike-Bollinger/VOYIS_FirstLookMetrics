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
                print(msg)
                if progress_callback and progress is not None:
                    progress_callback(progress, msg)
            
            # If no visibility results were provided, try to find them in the output folder
            if visibility_results is None:
                log_message("Checking for existing visibility analysis results in output folder...", progress=5)
                try:
                    # Look in multiple potential locations
                    vis_csv_paths = [
                        os.path.join(output_folder, "visibility_results.csv"),
                        os.path.join(os.path.dirname(output_folder), "visibility_results.csv"),
                        os.path.join(input_folder, "visibility_results.csv"),
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
                            
                            # Extract data from each row - handle different column naming conventions
                            confidence_col = next((col for col in ['confidence', 'conf'] if col in vis_df.columns), None)
                            visibility_col = next((col for col in ['visibility', 'category', 'vis_category'] if col in vis_df.columns), None)
                            
                            if visibility_col and confidence_col:
                                for _, row in vis_df.iterrows():
                                    result = {
                                        'image': os.path.basename(row['image']),  # Ensure basename only
                                        'visibility': row[visibility_col],
                                        'confidence': row[confidence_col]
                                    }
                                    
                                    # Add altitude if available
                                    if 'altitude' in row:
                                        result['altitude'] = row['altitude']
                                        
                                    visibility_results['analyzed_images'].append(result)
                                
                                log_message(f"Found existing visibility analysis for {len(visibility_results['analyzed_images'])} images", progress=6)
                            else:
                                log_message("Found visibility results file but missing required columns", progress=6)
                
                except Exception as e:
                    log_message(f"Could not load existing visibility results: {e}", progress=6)
                    # Continue without visibility data
                
            log_message(f"Selecting {count} highlight images based on image metrics...", progress=10)
            
            # Check if metrics already exist in CSV
            log_message("Checking for existing metrics in image_locations.csv...", progress=12)
            existing_metrics_df = self.load_metrics_from_csv(output_folder, input_folder)
            
            # Set up visibility map
            visibility_map = {}
            has_visibility_data = False
            
            # Step 1: Find all image files in the input folder
            log_message("Step 1: Finding all images in dataset...", progress=15)
            
            all_image_paths = []
            for root, _, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        # Skip raw images
                        if "raw" not in root.lower() and "raw" not in file.lower():
                            full_path = os.path.join(root, file)
                            all_image_paths.append(full_path)
            
            log_message(f"Found {len(all_image_paths)} images in dataset", progress=20)
            
            if len(all_image_paths) == 0:
                log_message("No images found in the input folder", progress=100)
                return []
            
            # Step 2: Load altitude data if available
            altitude_map = {}
            has_altitude_data = False
            
            if altitude_threshold is not None:
                log_message(f"Step 2: Loading altitude data (threshold: {altitude_threshold}m)...", progress=25)
                try:
                    csv_path = os.path.join(output_folder, "image_locations.csv")
                    if os.path.exists(csv_path):
                        df = self._pd.read_csv(csv_path)
                        
                        if 'filename' in df.columns and 'altitude' in df.columns:
                            for _, row in df.iterrows():
                                filename = os.path.basename(row['filename'])
                                altitude_map[filename] = row['altitude']
                            has_altitude_data = True
                            
                            log_message(f"Found altitude data for {len(altitude_map)} images", progress=30)
                except Exception as e:
                    log_message(f"Error loading altitude data: {e}", progress=30)
                    # Continue without altitude data
            
            # Check altitude data distribution
            if has_altitude_data:
                altitudes = list(altitude_map.values())
                log_message(f"Altitude stats: min={min(altitudes):.2f}m, max={max(altitudes):.2f}m, mean={sum(altitudes)/len(altitudes):.2f}m", progress=31)
                
                # Check for suspicious values (too many zeros)
                zero_count = sum(1 for a in altitudes if a == 0.0)
                if zero_count > len(altitudes) * 0.5:  # If more than 50% are zero
                    log_message(f"Warning: {zero_count} of {len(altitudes)} altitudes are exactly zero. Data may be incorrect.", progress=32)
            
            # Step 3: Filter by altitude if threshold is provided
            if altitude_threshold is not None and has_altitude_data:
                log_message(f"Step 3: Filtering images by altitude threshold ({min_altitude_threshold}m - {altitude_threshold}m)...", progress=35)
                
                filtered_images = []
                filtered_too_high = 0
                filtered_too_low = 0
                
                for img_path in all_image_paths:
                    filename = os.path.basename(img_path)
                    if filename in altitude_map:
                        altitude = altitude_map[filename]
                        if min_altitude_threshold <= altitude <= altitude_threshold:
                            filtered_images.append(img_path)
                        elif altitude > altitude_threshold:
                            filtered_too_high += 1
                        else:
                            filtered_too_low += 1
                    else:
                        # Include if no altitude data
                        filtered_images.append(img_path)
                
                log_message(f"Filtered out {filtered_too_high} images above {altitude_threshold}m threshold", progress=37)
                log_message(f"Filtered out {filtered_too_low} images below {min_altitude_threshold}m threshold", progress=38)
                log_message(f"Selected {len(filtered_images)} of {len(all_image_paths)} images within altitude range", progress=40)
                all_image_paths = filtered_images
            
            log_message(f"After filtering: {len(all_image_paths)} images remain")
            
            # Step 4: Check for visibility analysis results
            if visibility_results:
                log_message("Step 4: Integrating visibility analysis results...", progress=45)
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
                log_message("Step 4: No visibility analysis results available", progress=45)
            
            # Decide whether to use existing metrics or calculate new ones
            if not existing_metrics_df.empty and len(existing_metrics_df) >= len(all_image_paths) * 0.75:  # If we have metrics for at least 75% of images
                log_message(f"Using {len(existing_metrics_df)} existing metrics from image_locations.csv", progress=55)
                metrics_df = existing_metrics_df
                
                # We still need to add visibility data if it wasn't in the CSV
                if has_visibility_data:
                    log_message("Adding visibility data to existing metrics...", progress=60)
                    for i, row in metrics_df.iterrows():
                        filename = row['filename']
                        if filename in visibility_map and 'visibility_score' not in metrics_df.columns:
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
                
                # Skip to the selection phase
                log_message("Using existing metrics, skipping to scoring phase", progress=65)
                
            else:
                # Process ALL images regardless of dataset size
                log_message(f"Processing all {len(all_image_paths)} images in the dataset", progress=45)

                # Just a note about large datasets to inform the user
                if len(all_image_paths) > 1000:
                    log_message(f"Large dataset detected with {len(all_image_paths)} images - this may take some time", progress=46)
                    if len(all_image_paths) > 5000:
                        log_message(f"Very large dataset ({len(all_image_paths)} images) - using parallel processing", progress=47)
                
                # Step 5: Calculate metrics for each image using parallel processing if dataset is large
                log_message("Step 5: Calculating image metrics...", progress=55)

                # Determine if we should use parallel processing based on dataset size
                use_parallel = len(all_image_paths) >= 100  # Use parallel for datasets with 100+ images

                if use_parallel:
                    # Import multiprocessing locally to avoid potential pickle issues
                    import multiprocessing
                    from concurrent.futures import ThreadPoolExecutor
                    
                    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
                    # ThreadPoolExecutor is still much faster than serial processing and doesn't have pickle issues
                    num_cores = multiprocessing.cpu_count()
                    num_workers = max(1, min(num_cores - 1, 8))  # Use at most 8 workers, leave one core free
                    
                    log_message(f"Using parallel processing with {num_workers} workers", progress=56)
                    
                    # Calculate batch size - smaller batches for more frequent progress updates
                    batch_size = max(10, len(all_image_paths) // (num_workers * 10))
                    
                    # Create batches
                    batches = []
                    for i in range(0, len(all_image_paths), batch_size):
                        batch = all_image_paths[i:i + batch_size]
                        batches.append(batch)
                    
                    log_message(f"Processing {len(batches)} batches of approximately {batch_size} images each", progress=57)
                    
                    # Track progress
                    processed_images = 0
                    image_metrics = []
                    
                    # Process batches using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        # Submit all batch tasks
                        batch_futures = []
                        for batch in batches:
                            future = executor.submit(self._process_batch_in_thread, batch, altitude_map, visibility_map, has_visibility_data)
                            batch_futures.append(future)
                        
                        # Process results as they complete
                        for i, future in enumerate(as_completed(batch_futures)):
                            try:
                                batch_results = future.result()
                                image_metrics.extend(batch_results)
                                
                                # Update progress
                                processed_images += len(batch_results)
                                progress_value = 60 + int((processed_images / len(all_image_paths)) * 25)
                                log_message(f"Processed {processed_images}/{len(all_image_paths)} images", progress=progress_value)
                                
                            except Exception as e:
                                log_message(f"Error processing batch: {e}", progress=60)
                else:
                    # Simple single-process approach for smaller datasets
                    log_message("Processing images sequentially", progress=56)
                    
                    # Process directly
                    image_metrics = []
                    total_images = len(all_image_paths)
                    
                    for i, img_path in enumerate(all_image_paths):
                        try:
                            # Update progress periodically
                            if i % 10 == 0 or i == total_images - 1:
                                progress_value = 55 + int((i / total_images) * 30)
                                log_message(f"Processing image {i+1}/{total_images}", progress=progress_value)
                            
                            # Calculate metrics
                            metrics = self._calculate_image_metrics(img_path)
                            
                            # Add metadata
                            filename = os.path.basename(img_path)
                            
                            # Get altitude if available
                            if altitude_map and filename in altitude_map:
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
                            
                        except Exception as e:
                            log_message(f"Error processing {img_path}: {e}")

                log_message(f"After metrics calculation: {len(image_metrics)} valid images")
                log_message(f"Calculated metrics for {len(image_metrics)} images", progress=85)
                
                # Convert to DataFrame for easier filtering and sorting
                metrics_df = self._pd.DataFrame(image_metrics)
                
                # Check if we have any metrics
                if len(metrics_df) == 0:
                    log_message("No valid images found with proper metrics", progress=90)
                    return []
                
                # Check if required columns exist
                required_columns = ['contrast', 'texture', 'color_variance', 'entropy']
                missing_columns = [col for col in required_columns if col not in metrics_df.columns]
                if missing_columns:
                    log_message(f"Error: Missing required metrics columns: {', '.join(missing_columns)}", progress=88)
                    
                    # Add missing columns with default values
                    for col in missing_columns:
                        metrics_df[col] = 0.0
                    log_message("Added missing columns with default values", progress=89)
            
            # Step 6: Score and select top images
            log_message("Step 6: Scoring and selecting top images...", progress=87)
            
            # Calculate combined score based on metrics if needed
            if 'combined_score' not in metrics_df.columns:
                def calculate_combined_score(row):
                    # Base metrics score - only use image quality metrics
                    metrics_score = (
                        row['contrast'] * self.metrics_weight['contrast'] +
                        row['texture'] * self.metrics_weight['texture'] +
                        row['color_variance'] * self.metrics_weight['color_variance'] +
                        row['entropy'] * self.metrics_weight['entropy']
                    )
                    
                    # Add visibility score if available
                    if has_visibility_data and 'visibility_score' in row and not self._pd.isna(row['visibility_score']):
                        # Combine with visibility score
                        return (metrics_score * (1 - self.metrics_weight['visibility']) + 
                               row['visibility_score'] * self.metrics_weight['visibility'])
                    else:
                        # Just use metrics score
                        return metrics_score
                
                # Add combined score column
                metrics_df['combined_score'] = metrics_df.apply(calculate_combined_score, axis=1)
            
            # Export metrics to CSV if we calculated new ones
            if not existing_metrics_df.empty:
                log_message("Using existing metrics from CSV", progress=88)
            else:
                log_message("Exporting metrics to image_locations.csv...", progress=88)
                self.export_metrics_to_csv(output_folder, metrics_df)
            
            # Print statistics to help debug
            print(f"Score statistics: min={metrics_df['combined_score'].min():.3f}, "
                  f"max={metrics_df['combined_score'].max():.3f}, "
                  f"mean={metrics_df['combined_score'].mean():.3f}")
            print(f"Images with non-zero scores: {(metrics_df['combined_score'] > 0).sum()} of {len(metrics_df)}")

            if (metrics_df['combined_score'] > 0).sum() == 0:
                print("WARNING: All images have zero scores. Check metric calculation.")

            log_message(f"Images with combined_score > 0: {len(metrics_df[metrics_df['combined_score'] > 0])}")
            
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
            
            # Get diverse set of top images
            selected_df = self._select_diverse_highlights(metrics_df, count)
            
            # Ensure highlight directory exists
            os.makedirs(highlight_dir, exist_ok=True)
            
            # Copy selected images to highlight directory
            log_message("Step 7: Copying selected highlight images...", progress=90)
            
            highlight_paths = []
            copy_errors = 0
            
            for idx, (_, row) in enumerate(selected_df.iterrows()):
                source_path = row['path']
                filename = row['filename']
                dest_path = os.path.join(highlight_dir, filename)
                
                try:
                    # Verify source file exists
                    if not os.path.exists(source_path):
                        log_message(f"ERROR: Source file not found: {source_path}")
                        copy_errors += 1
                        continue
                        
                    # Copy file
                    shutil.copy2(source_path, dest_path)
                    
                    # Verify copy was successful
                    if os.path.exists(dest_path):
                        highlight_paths.append(dest_path)
                        
                        # Log selection with relevant metrics
                        metrics_str = f"score: {row['combined_score']:.2f}, contrast: {row['contrast']:.2f}"
                        
                        # Add visibility info if available
                        if 'visibility_category' in row and not self._pd.isna(row['visibility_category']):
                            metrics_str += f", visibility: {row['visibility_category']}"
                        
                        # Add altitude info if available
                        if 'altitude' in row and not self._pd.isna(row['altitude']):
                            metrics_str += f", altitude: {row['altitude']:.2f}m"
                        
                        log_message(f"✓ Copied highlight {idx+1}/{len(selected_df)}: {filename} ({metrics_str})")
                    else:
                        log_message(f"ERROR: Failed to copy {filename} - destination file not found")
                        copy_errors += 1
                        
                except Exception as e:
                    log_message(f"ERROR copying {filename}: {e}")
                    copy_errors += 1
                    
            # Report copy results
            if copy_errors > 0:
                log_message(f"WARNING: {copy_errors} files failed to copy")
            else:
                log_message(f"✓ Successfully copied all {len(highlight_paths)} highlight images")
            
            if len(highlight_paths) == 0:
                log_message("ERROR: No highlight images were successfully copied!")
                return []
            
            # Create HTML summary with error handling
            log_message("Step 8: Creating HTML summary...", progress=95)
            try:
                html_path = self._create_highlights_html(highlight_dir, selected_df, has_visibility_data)
                if html_path and os.path.exists(html_path):
                    log_message(f"✓ HTML summary created: {html_path}")
                else:
                    log_message("WARNING: Failed to create HTML summary")
            except Exception as html_error:
                log_message(f"ERROR creating HTML summary: {html_error}")
                log_message(traceback.format_exc())
            
            # Create panel image with top 4 highlights
            log_message("Creating 4-panel summary image for reports...", progress=97)
            try:
                panel_path = self.create_highlight_panel(highlight_dir, selected_df, top_count=4)
                if panel_path and os.path.exists(panel_path):
                    log_message(f"✓ Created highlight panel image: {panel_path}")
                else:
                    log_message("WARNING: Could not create highlight panel image (matplotlib may be missing)")
            except Exception as panel_error:
                log_message(f"ERROR creating panel image: {panel_error}")
                log_message(traceback.format_exc())
            
            # Export metrics to CSV
            try:
                log_message("Exporting metrics to image_locations.csv...", progress=98)
                if self.export_metrics_to_csv(output_folder, selected_df):
                    log_message("✓ Metrics exported to CSV successfully")
                else:
                    log_message("WARNING: Failed to export metrics to CSV")
            except Exception as csv_error:
                log_message(f"ERROR exporting metrics to CSV: {csv_error}")
                log_message(traceback.format_exc())
            
            log_message(f"✓ Selected {len(highlight_paths)} highlight images. Saved to {highlight_dir}", progress=100)
            
            return highlight_paths
            
        except Exception as e:
            log_message(f"ERROR in highlight selection: {e}", progress=100)
            log_message(traceback.format_exc())
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

    def create_highlight_panel(self, highlight_dir: str, metrics_df, top_count: int = 4) -> str:
        """
        Create a multi-panel image with the top highlights for report inclusion
        
        Args:
            highlight_dir: Directory with highlight images
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
            panel_path = os.path.join(highlight_dir, "top_highlights_panel.png")
            fig.savefig(panel_path, dpi=300, bbox_inches='tight')
            self._plt.close(fig)
            
            return panel_path
                
        except Exception as e:
            print(f"Error creating highlight panel: {e}")
            return None

    def export_metrics_to_csv(self, output_folder: str, metrics_df) -> bool:
        """
        Export image metrics to the image_locations.csv file
        
        Args:
            output_folder: Directory containing image_locations.csv
            metrics_df: DataFrame with calculated metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_imports()  # This will import pandas as self._pd
            csv_path = os.path.join(output_folder, "image_locations.csv")
            
            # Check if the CSV file exists
            if os.path.exists(csv_path):
                # Load existing CSV
                try:
                    locations_df = self._pd.read_csv(csv_path)
                    print(f"Loaded existing image_locations.csv with {len(locations_df)} entries")
                    
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
                    print(f"Saved updated image_locations.csv with metrics to {csv_path}")
                    return True
                    
                except Exception as e:
                    print(f"Error updating existing CSV: {e}")
                    print(traceback.format_exc())
                    return False
            else:
                # Create a new CSV with only the metrics data
                # This will be a simplified version with just the essential columns
                output_df = metrics_df.copy()
                
                # Keep only essential columns
                essential_cols = ['filename', 'contrast', 'texture', 'color_variance', 
                                'entropy', 'combined_score']
                
                # Add altitude if available
                if 'altitude' in output_df.columns:
                    essential_cols.append('altitude')
                
                # Filter columns
                output_cols = [col for col in essential_cols if col in output_df.columns]
                output_df = output_df[output_cols]
                
                # Save to CSV
                output_df.to_csv(csv_path, index=False)
                print(f"Created new image_locations.csv with metrics at {csv_path}")
                return True
                
        except Exception as e:
            print(f"Error exporting metrics to CSV: {e}")
            print(traceback.format_exc())
            return False

    def load_metrics_from_csv(self, output_folder: str, input_folder: str):
        """
        Load pre-calculated metrics from image_locations.csv if available
        
        Args:
            output_folder: Directory containing image_locations.csv
            input_folder: Directory with input images to match with metrics
            
        Returns:
            DataFrame with loaded metrics or empty DataFrame if not available
        """
        try:
            self._ensure_imports()  # This will import pandas as self._pd
            csv_path = os.path.join(output_folder, "image_locations.csv")
            
            if not os.path.exists(csv_path):
                print("No existing image_locations.csv found")
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