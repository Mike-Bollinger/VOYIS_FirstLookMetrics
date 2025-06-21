import os
import sys 
from typing import Dict, List, Optional, Callable, Tuple
import time
import traceback

# Default directory for saving pre-trained models
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "v_a_pre-trained_models")

class VisibilityAnalyzer:
    """
    Class to analyze image visibility in underwater imagery.
    Uses a pre-trained deep learning model or can train a new one.
    """
    
    def __init__(self, log_callback: Optional[Callable] = None, altitude_threshold=9.0):
        """
        Initialize the VisibilityAnalyzer
        
        Args:
            log_callback: Optional function to call for logging messages
            altitude_threshold: Altitude threshold for filtering images (same as used in metrics)
        """
        self.model = None
        self.tf = None
        self._cv2 = None
        self._plt = None
        self._pd = None
        self._np = None
        self._folium = None
        self._geopandas = None
        self.using_gpu = False
        self.log_callback = log_callback
        
        # Initialize categories
        self.categories = ['good', 'fair', 'poor']  # Default categories
    
    def log_message(self, message: str, progress: Optional[int] = None):
        """
        Log a message using the callback if available, otherwise print
        
        Args:
            message: Message to log
            progress: Optional progress percentage
        """
        if self.log_callback:
            if progress is not None:
                self.log_callback(message, progress=progress)
            else:
                self.log_callback(message)
        else:
            print(f"VisibilityAnalyzer: {message}")
    
    def set_log_callback(self, log_callback: Callable):
        """Set or update the log callback function"""
        self.log_callback = log_callback
    
    def _import_required_libraries(self):
        """Import required libraries on demand"""
        try:
            if self._pd is None:
                import pandas as pd
                self._pd = pd
            
            if self._np is None:
                import numpy as np
                self._np = np
                
            if self._plt is None:
                import matplotlib.pyplot as plt
                self._plt = plt
                
            if self._sns is None:
                import seaborn as sns
                self._sns = sns
                
            return True
        except ImportError as e:
            if hasattr(self, 'log_message'):
                self.log_message(f"Error importing required libraries: {e}")
            return False
    
    def _import_tensorflow(self):
        """Try to import TensorFlow and set it up"""
        if self.tf_available:
            return True
            
        try:
            import tensorflow as tf
            self.tf = tf
            
            # Configure GPU if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.using_gpu = True
                    self.gpu_devices = [gpu.name for gpu in gpus]
                    if hasattr(self, 'log_message'):
                        self.log_message(f"TensorFlow {tf.__version__} initialized with GPU support ({len(gpus)} GPU(s))")
                except RuntimeError as e:
                    if hasattr(self, 'log_message'):
                        self.log_message(f"Error configuring GPU: {e}")
            else:
                if hasattr(self, 'log_message'):
                    self.log_message(f"TensorFlow {tf.__version__} initialized (CPU only)")
            
            self.tf_available = True
            return True
            
        except ImportError:
            if hasattr(self, 'log_message'):
                self.log_message("TensorFlow not available - visibility analysis will be disabled")
            return False
        except Exception as e:
            if hasattr(self, 'log_message'):
                self.log_message(f"Error setting up TensorFlow: {str(e)}")
            return False
    
    def _load_img(self, path, target_size=(224, 224)):
        """Load and preprocess an image"""
        return self.tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    
    def _img_to_array(self, img):
        """Convert image to array"""
        return self.tf.keras.preprocessing.image.img_to_array(img)
    
    def _preprocess_input(self, img_array):
        """Preprocess image array for VGG16"""
        return self.tf.keras.applications.vgg16.preprocess_input(img_array)
    
    def _load_model(self, model_path: str) -> object:
        """
        Load a saved Keras model from disk
        
        Args:
            model_path: Path to the .h5 model file
            
        Returns:
            Loaded Keras model
        """
        try:
            if not os.path.exists(model_path):
                self.log_message(f"Model file not found: {model_path}")
                return None
                
            self.log_message(f"Loading model from: {model_path}")
            model = self.tf.keras.models.load_model(model_path)
            self.log_message("Model loaded successfully")
            return model
        except Exception as e:
            self.log_message(f"Error loading model: {e}")
            self.log_message(traceback.format_exc())
            return None
    
    def load_or_train_model(self, path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Load an existing model or train a new one if the path is a directory
        
        Args:
            path: Path to the model file (.h5) or training data directory
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Boolean indicating success
        """
        # Import required libraries
        if not self._import_tensorflow():
            if progress_callback:
                progress_callback(0, "TensorFlow not available. Cannot load or train model.")
            return False
            
        if not self._import_required_libraries():
            if progress_callback:
                progress_callback(0, "Required libraries not available.")
            return False
            
        try:
            # If path is a file, try to load it as a model
            if os.path.isfile(path) and path.lower().endswith('.h5'):
                if progress_callback:
                    progress_callback(10, f"Loading model from {path}...")
                self.model = self._load_model(path)
                if self.model is not None:
                    if progress_callback:
                        progress_callback(100, "Model loaded successfully.")
                    return True
                else:
                    if progress_callback:
                        progress_callback(0, "Failed to load model.")
                    return False
                
            # If path is a directory, assume it's training data
            elif os.path.isdir(path):
                if progress_callback:
                    progress_callback(5, "Verifying training data directory structure...")
                
                # First verify the training data structure
                if not self.verify_training_data(path, progress_callback):
                    # Try to look for nested structure
                    found_valid_subdir = False
                    
                    for subdir in os.listdir(path):
                        potential_training_dir = os.path.join(path, subdir)
                        if os.path.isdir(potential_training_dir):
                            if progress_callback:
                                progress_callback(10, f"Checking subdirectory: {subdir}")
                            
                            if self.verify_training_data(potential_training_dir, progress_callback):
                                if progress_callback:
                                    progress_callback(15, f"Found valid training data in subdirectory: {subdir}")
                                path = potential_training_dir
                                found_valid_subdir = True
                                break
                    
                    if not found_valid_subdir:
                        if progress_callback:
                            progress_callback(0, "Invalid training data structure. Please ensure your training directory contains subdirectories named exactly: " + ", ".join(self.categories))
                        return False
                
                # If we got here, we have valid training data
                if progress_callback:
                    progress_callback(20, "Training data verified. Preparing to train model...")
                    
                return self._train_model(path, progress_callback)
                
            else:
                if progress_callback:
                    progress_callback(0, "Invalid path. Must be a .h5 model file or a training data directory.")
                return False
                
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error loading/training model: {str(e)}")
            self.log_message(f"Error loading/training model: {e}")
            self.log_message(traceback.format_exc())
            return False
    
    def _train_model(self, training_data_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Train the model on the provided dataset
        
        Args:
            training_data_path: Path to training data directory (with subdirectories for each category)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Boolean indicating success
        """
        try:
            if progress_callback:
                progress_callback(5, "Loading training data...")
            
            # Define categories from folder names
            category_folders = [d for d in os.listdir(training_data_path) 
                               if os.path.isdir(os.path.join(training_data_path, d))]
            
            if not category_folders:
                if progress_callback:
                    progress_callback(0, "Error: No category folders found in training data path")
                return False
            
            # Save categories for prediction
            self.categories = sorted(category_folders)
            self.log_message(f"Found categories: {self.categories}")
            
            # Load all images and labels
            images = []
            labels = []
            
            for i, category in enumerate(self.categories):
                if progress_callback:
                    progress_callback(10 + int((i / len(self.categories)) * 20), 
                                     f"Loading images for '{category}'...")
                
                category_path = os.path.join(training_data_path, category)
                image_files = [f for f in os.listdir(category_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                
                if not image_files:
                    if progress_callback:
                        progress_callback(0, f"Error: No images found in {category} folder")
                    return False
                
                # Load images
                for j, img_file in enumerate(image_files):
                    try:
                        img_path = os.path.join(category_path, img_file)
                        img = self._load_img(img_path, target_size=(224, 224))
                        img_array = self._img_to_array(img)
                        img_array = self._preprocess_input(img_array)
                        
                        images.append(img_array)
                        labels.append(i)
                        
                        # Update progress periodically
                        if progress_callback and j % 10 == 0:
                            progress_callback(
                                10 + int((i / len(self.categories)) * 20), 
                                f"Loading {category} images: {j+1}/{len(image_files)}"
                            )
                    except Exception as e:
                        self.log_message(f"Error loading image {img_file}: {e}")
            
            # Convert to arrays
            if progress_callback:
                progress_callback(30, "Converting data to training format...")
            
            images_array = self._np.array(images)
            labels_array = self._np.array(labels)
            
            # One-hot encode labels
            y = self.tf.keras.utils.to_categorical(labels_array, num_classes=len(self.categories))
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                images_array, y, test_size=0.2, random_state=42
            )
            
            if progress_callback:
                progress_callback(40, "Creating model architecture...")
            
            # Build model
            base_model = self.tf.keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            model = self.tf.keras.models.Sequential([
                base_model,
                self.tf.keras.layers.GlobalAveragePooling2D(),
                self.tf.keras.layers.Dense(512, activation='relu'),
                self.tf.keras.layers.Dropout(0.5),
                self.tf.keras.layers.Dense(len(self.categories), activation='softmax')
            ])
            
            model.compile(
                optimizer=self.tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            if progress_callback:
                progress_callback(45, "Training model (this may take several minutes)...")
            
            # Train model
            callbacks = [
                self.tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    patience=5, 
                    restore_best_weights=True
                )
            ]
            
            batch_size = 64 if self.using_gpu else 32
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=15,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            self.model = model
            
            # Evaluate model
            if progress_callback:
                progress_callback(90, "Evaluating model performance...")
            
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            
            if progress_callback:
                progress_callback(95, f"Training complete. Validation accuracy: {accuracy*100:.1f}%")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error training model: {str(e)}")
            self.log_message(f"Error training model: {e}")
            self.log_message(traceback.format_exc())
            return False
    
    def analyze_images(self, image_paths: List[str], output_folder: str, 
                      progress_callback: Optional[Callable] = None,
                      altitude_threshold: float = 8.0) -> Tuple[bool, Dict]:
        """
        Analyze the visibility of a set of images
        
        Args:
            image_paths: List of paths to images to analyze
            output_folder: Output directory for reports
            progress_callback: Optional callback for progress updates
            altitude_threshold: Maximum altitude to analyze (in meters)
            
        Returns:
            Tuple of (success, statistics_dict)
        """
        def log_message(msg, progress=None):
            if progress_callback and progress is not None:
                progress_callback(progress, msg)
            print(msg)
        
        # Store the log function
        self.log_message = log_message

        log_message("\n" + "=" * 80)
        log_message(f"VISIBILITY ANALYSIS STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"Altitude threshold: {altitude_threshold}m")
        log_message("=" * 80)

        # Import required libraries
        if not self._import_tensorflow():
            log_message("TensorFlow not available. Cannot perform analysis.", progress=0)
            return False, {}
        
        if not self._import_required_libraries():
            log_message("Required libraries not available. Cannot perform analysis.", progress=0)
            return False, {}
        
        log_message(f"✓ TensorFlow successfully loaded (GPU: {self.using_gpu})")

        if not self.model:
            log_message("Error: No model loaded. Please load or train a model first.", progress=0)
            return False, {}
        
        log_message("✓ Visibility model loaded and ready")

        try:
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            log_message(f"✓ Output directory confirmed: {output_folder}")
            
            # Initialize tracking variables
            self.all_image_locations = {}
            self.selected_image_locations = {}
            self.selected_images = set()
            self.all_images = set()
            
            # STEP 1: Load altitude and location data
            log_message("\nSTEP 1: Loading altitude and location data", progress=5)
            
            altitude_map = {}
            lat_lon_map = {}
            has_altitude_data = False
            has_location_data = False
            
            # Try to load existing CSV data
            try:
                csv_path = os.path.join(output_folder, "image_locations.csv")
                if os.path.exists(csv_path):
                    df = self._pd.read_csv(csv_path)
                    
                    required_columns = ['filename', 'altitude', 'latitude', 'longitude']
                    has_required_columns = all(col in df.columns for col in required_columns)
                    
                    if has_required_columns:
                        for _, row in df.iterrows():
                            filename = os.path.basename(row['filename'])
                            altitude_map[filename] = row['altitude']
                            lat_lon_map[filename] = (row['latitude'], row['longitude'])
                        has_altitude_data = True
                        has_location_data = True
                        log_message(f"✓ Found altitude and location data for {len(altitude_map)} images", progress=10)
                    elif 'altitude' in df.columns and 'filename' in df.columns:
                        for _, row in df.iterrows():
                            filename = os.path.basename(row['filename'])
                            altitude_map[filename] = row['altitude']
                        has_altitude_data = True
                        log_message(f"✓ Loaded altitude data for {len(altitude_map)} images (no location data)", progress=10)
            except Exception as e:
                log_message(f"Warning: Could not load altitude/location data from CSV: {e}", progress=10)
            
            # STEP 2: Scan for images
            log_message("\nSTEP 2: Scanning for processed images in dataset", progress=15)
            
            all_images_paths = []
            
            if not image_paths:
                # Scan directory
                input_folder = os.path.dirname(output_folder)
                if not os.path.isdir(input_folder) or input_folder == output_folder:
                    input_folder = os.path.dirname(input_folder)
                
                if os.path.isdir(input_folder):
                    log_message(f"Scanning for processed images in {input_folder}...", progress=20)
                    
                    for root, _, files in os.walk(input_folder):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg')):
                                full_path = os.path.join(root, file)
                                if "raw" not in root.lower() and "raw" not in file.lower():
                                    if os.path.isfile(full_path):
                                        all_images_paths.append(full_path)
            else:
                all_images_paths = [path for path in image_paths 
                                  if os.path.isfile(path) and
                                  path.lower().endswith(('.jpg', '.jpeg')) and
                                  "raw" not in path.lower()]
            
            total_images_found = len(all_images_paths)
            
            if total_images_found == 0:
                log_message("Error: No images found in dataset", progress=0)
                return False, {}
                
            log_message(f"✓ Found {total_images_found} total images in dataset", progress=25)
            
            # STEP 3: Filter by altitude
            log_message(f"\nSTEP 3: Filtering images by altitude threshold ({altitude_threshold}m)", progress=30)
            
            below_threshold_images = []
            filtered_by_altitude = 0
            
            for i, img_path in enumerate(all_images_paths):
                filename = os.path.basename(img_path)
                
                # Store all image info
                self.all_images.add(filename)
                if filename in lat_lon_map:
                    self.all_image_locations[filename] = lat_lon_map[filename]
                
                # Check altitude threshold
                if has_altitude_data and filename in altitude_map:
                    altitude = altitude_map[filename]
                    if altitude <= altitude_threshold:
                        below_threshold_images.append(img_path)
                    else:
                        filtered_by_altitude += 1
                else:
                    below_threshold_images.append(img_path)
                
                # Update progress periodically
                if progress_callback and (i % 1000 == 0 or i == total_images_found - 1):
                    log_message(
                        f"Filtered {i+1}/{total_images_found} images, {filtered_by_altitude} above threshold",
                        progress=30 + int((i / total_images_found) * 10)
                    )
            log_message(f"✓ Altitude filtering complete: {filtered_by_altitude} images excluded", progress=40)
            
            # STEP 4: Select images for analysis
            log_message("\nSTEP 4: Selecting images for visibility analysis", progress=45)
            
            below_threshold_count = len(below_threshold_images)
            
            if below_threshold_count == 0:
                log_message(f"Error: No images found below altitude threshold of {altitude_threshold}m", progress=0)
                return False, {}
            
            # Random sample if too many images
            images_to_analyze = below_threshold_images
            if below_threshold_count > 5000:
                log_message(f"Randomly selecting 5000 images from {below_threshold_count} below threshold...", progress=47)
                
                import random
                random.seed(42)
                images_to_analyze = random.sample(below_threshold_images, 5000)
                
                log_message(f"✓ Selected 5000 random images for analysis", progress=50)
            else:
                log_message(f"✓ Using all {below_threshold_count} images below threshold for analysis", progress=50)
            
            # STEP 5: Analyze images
            log_message(f"\nSTEP 5: Analyzing visibility of {len(images_to_analyze)} images", progress=55)
            
            # Store selected image info
            for img_path in images_to_analyze:
                filename = os.path.basename(img_path)
                self.selected_images.add(filename)
                if filename in lat_lon_map:
                    self.selected_image_locations[filename] = lat_lon_map[filename]
            
            # Initialize statistics
            self.visibility_stats = {
                'total_images_found': total_images_found,
                'filtered_by_altitude': filtered_by_altitude,
                'below_threshold_count': below_threshold_count,
                'images_analyzed': len(images_to_analyze),
                'by_category': {cat: 0 for cat in self.categories},
                'percentages': {},
                'analyzed_images': []
            }
            
            # Process images
            results = []
            for i, img_path in enumerate(images_to_analyze):
                try:
                    # Load and preprocess image
                    img = self._load_img(img_path, target_size=(224, 224))
                    img_array = self._img_to_array(img)
                    img_array = self._preprocess_input(img_array)
                    img_array = self._np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    predictions = self.model.predict(img_array, verbose=0)
                    predicted_class = self._np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class])
                    
                    result = {
                        'image': os.path.basename(img_path),
                        'visibility': self.categories[predicted_class],
                        'confidence': confidence
                    }
                    
                    # Add altitude if available
                    if os.path.basename(img_path) in altitude_map:
                        result['altitude'] = altitude_map[os.path.basename(img_path)]
                    
                    # Update statistics
                    self.visibility_stats['by_category'][self.categories[predicted_class]] += 1
                    self.visibility_stats['analyzed_images'].append(result)
                    results.append(result)
                    
                    # Update progress
                    if progress_callback and (i % 50 == 0 or i == len(images_to_analyze) - 1):
                        log_message(
                            f"Analyzed {i+1}/{len(images_to_analyze)} images",
                            progress=55 + int((i / len(images_to_analyze)) * 35)
                        )
                    
                except Exception as e:
                    log_message(f"Error processing image {img_path}: {e}")
            
            # Calculate percentages
            total_analyzed = len(self.visibility_stats['analyzed_images'])
            for category in self.categories:
                count = self.visibility_stats['by_category'].get(category, 0)
                self.visibility_stats['percentages'][category] = (count / total_analyzed * 100) if total_analyzed > 0 else 0
            
            log_message("\nAnalysis complete. Results by category:")
            for category in self.categories:
                count = self.visibility_stats['by_category'].get(category, 0)
                percentage = self.visibility_stats['percentages'].get(category, 0)
                log_message(f"  {category}: {count} images ({percentage:.1f}%)")
            
            # STEP 6: Save results and create visualizations
            log_message("\nSTEP 6: Saving results and creating visualizations...", progress=90)
            
            if results:
                # Save CSV
                csv_path = os.path.join(output_folder, "visibility_results.csv")
                try:
                    df = self._pd.DataFrame(results)
                    
                    # Add location data if available
                    if has_location_data:
                        for i, row in df.iterrows():
                            img_filename = row['image']
                            if img_filename in lat_lon_map:
                                df.at[i, 'latitude'] = lat_lon_map[img_filename][0]
                                df.at[i, 'longitude'] = lat_lon_map[img_filename][1]
                    
                    # Save CSV
                    cols = ['image', 'visibility', 'confidence']
                    if has_location_data:
                        cols.extend(['latitude', 'longitude'])
                    if 'altitude' in df.columns:
                        cols.append('altitude')
                    
                    # Ensure all columns exist
                    for col in cols:
                        if col not in df.columns:
                            df[col] = None
                    
                    df[cols].to_csv(csv_path, index=False)
                    log_message(f"✓ Results saved to CSV: {csv_path}")
                    
                    # Create visualizations
                    try:
                        chart_path = self.create_visibility_chart(csv_path, output_folder)
                        if chart_path and os.path.exists(chart_path):
                            log_message(f"✓ Visibility chart created: {chart_path}")
                        
                        enhanced_chart_path = self.create_enhanced_visibility_chart(csv_path, output_folder)
                        if enhanced_chart_path and os.path.exists(enhanced_chart_path):
                            log_message(f"✓ Enhanced visibility chart created: {enhanced_chart_path}")
                            
                    except Exception as chart_error:
                        log_message(f"Warning: Error creating charts: {chart_error}")
                    
                    # Create map if location data available
                    if has_location_data:
                        try:
                            map_path = self.create_location_map(output_folder)
                            if map_path and os.path.exists(map_path):
                                log_message(f"✓ Location map created: {map_path}")
                            
                            shapefile_path = self.create_shapefile(output_folder)
                            if shapefile_path and os.path.exists(shapefile_path):
                                log_message(f"✓ Shapefile created: {shapefile_path}")
                                
                        except Exception as map_error:
                            log_message(f"Warning: Error creating maps/shapefiles: {map_error}")
                    
                    log_message("✓ Visibility analysis completed successfully!", progress=100)
                    return True, self.visibility_stats
                    
                except Exception as e:
                    log_message(f"Error saving results to CSV: {e}", progress=0)
                    return False, {}
            else:
                log_message("WARNING: No results to save!", progress=0)
                return False, {}
        except Exception as e:
            log_message(f"Error during analysis: {e}", progress=0)
            return False, {}