import os
import sys 
from typing import Dict, List, Optional, Callable, Tuple
import time
import traceback

# Default directory for saving pre-trained models
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "v_a_pre-trained_models")

# Flag to track if TensorFlow has been imported
TF_IMPORTED = False
TF_AVAILABLE = None

class VisibilityAnalyzer:
    """
    Class to analyze image visibility in underwater imagery.
    Uses a pre-trained deep learning model or can train a new one.
    """
    
    def __init__(self, altitude_threshold=9.0):
        """
        Initialize the visibility analyzer
        
        Args:
            altitude_threshold: Altitude threshold for filtering images (same as used in metrics)
        """
        self.altitude_threshold = altitude_threshold
        
        # Default image categories for visibility
        self.categories = ['zero_visibility', 'low_visibility', 'good_visibility', 'great_visibility']
        self.model = None
        
        # Statistics from analysis
        self.visibility_stats = {
            'total_images': 0,
            'by_category': {},
            'percentages': {},
            'analyzed_images': []
        }
        
        # Location data for visualization
        self.all_image_locations = {}
        self.selected_image_locations = {}
        self.selected_images = set()
        self.all_images = set()
        
        # Don't check for TensorFlow until we need it
        self.tf_available = None
        
        # Ensure global variables are defined
        global TF_IMPORTED, TF_AVAILABLE
        if 'TF_IMPORTED' not in globals():
            global TF_IMPORTED
            TF_IMPORTED = False
        if 'TF_AVAILABLE' not in globals():
            global TF_AVAILABLE
            TF_AVAILABLE = None
        
        # Create default models directory
        try:
            os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
        except Exception:
            # Don't fail initialization if we can't create the directory
            pass
    
    def _import_tensorflow(self):
        """Try to import TensorFlow with GPU support, falling back to CPU if needed"""
        if hasattr(self, 'tf') and self.tf is not None:
            return True
            
        try:
            # Save original environment variables to restore them later if needed
            original_env = os.environ.copy()
            
            # First, try to setup TensorFlow optimally for GPU use
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging noise
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Avoid taking all GPU memory
            
            import tensorflow as tf
            
            # Try to load and configure GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            using_gpu = False
            
            if gpus:
                try:
                    # Configure GPU to use memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    using_gpu = True
                    print(f"Using GPU acceleration with {len(gpus)} GPU(s): {[gpu.name.decode('utf-8') for gpu in gpus]}")
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            
            # If no GPUs or configuration failed, explicitly configure for CPU use
            if not using_gpu:
                # Restore original environment variables
                for key, value in original_env.items():
                    os.environ[key] = value
                    
                print("No compatible GPU found, using CPU for processing")
                    
            # Minimize TensorFlow verbosity regardless of device
            tf.get_logger().setLevel('ERROR')
            
            # Import other required modules
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Store modules for later use
            self.tf = tf
            self._np = np
            self._pd = pd
            self._plt = plt
            self._sns = sns
            
            # Define helper functions that use tensorflow
            def load_img(img_path, target_size):
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                return img
                
            def img_to_array(img):
                return tf.keras.preprocessing.image.img_to_array(img)
                
            def preprocess_input(x):
                return tf.keras.applications.vgg16.preprocess_input(x)
            
            # Store helper functions
            self._load_img = load_img
            self._img_to_array = img_to_array
            self._preprocess_input = preprocess_input
            
            # Store GPU status for reference
            self.using_gpu = using_gpu
            self.gpu_devices = [gpu.name.decode('utf-8') for gpu in gpus] if gpus else []
            
            self.tf_available = True
            return True
            
        except ImportError as e:
            print(f"TensorFlow import error: {e}")
            self.tf_available = False
            return False
        except Exception as e:
            print(f"Error initializing TensorFlow: {e}")
            print(traceback.format_exc())
            self.tf_available = False
            return False
    
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
                print(f"Model file not found: {model_path}")
                return None
                
            print(f"Loading model from: {model_path}")
            model = self.tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
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
        # Only import TensorFlow when we actually need it
        if not self._import_tensorflow():
            if progress_callback:
                progress_callback(0, "TensorFlow not available. Cannot load or train model.")
            return False
            
        try:
            # If path is a file, try to load it as a model
            if os.path.isfile(path) and path.lower().endswith('.h5'):
                if progress_callback:
                    progress_callback(10, f"Loading model from {path}...")
                self.model = self._load_model(path)
                if progress_callback:
                    progress_callback(100, "Model loaded successfully.")
                return True
                
            # If path is a directory, assume it's training data
            elif os.path.isdir(path):
                if progress_callback:
                    progress_callback(5, "Verifying training data directory structure...")
                
                # First verify the training data structure
                if not self.verify_training_data(path, progress_callback):
                    # Try to look for nested structure - sometimes people put training data
                    # in a subdirectory with the date or other identifying info
                    found_valid_subdir = False
                    
                    for subdir in os.listdir(path):
                        potential_training_dir = os.path.join(path, subdir)
                        if os.path.isdir(potential_training_dir):
                            if progress_callback:
                                progress_callback(10, f"Checking subdirectory: {subdir}")
                            
                            if self.verify_training_data(potential_training_dir, progress_callback):
                                if progress_callback:
                                    progress_callback(15, f"Found valid training data in subdirectory: {subdir}")
                                path = potential_training_dir  # Update the path
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
            print(f"Error loading/training model: {e}")
            print(traceback.format_exc())
            return False
    
    def _build_model(self) -> 'Sequential':  # type: ignore
        """
        Build the CNN model for visibility classification
        
        Returns:
            Compiled Keras Sequential model
        """
        # Use VGG16 as base model with pre-trained weights
        base_model = self._VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        model = self._Sequential([
            base_model,
            self._Flatten(),
            self._Dense(256, activation='relu'),
            self._Dropout(0.5),
            self._Dense(len(self.categories), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
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
            
            # Define categories from folder names, ensuring consistent ordering
            category_folders = [d for d in os.listdir(training_data_path) 
                               if os.path.isdir(os.path.join(training_data_path, d))]
            
            if not category_folders:
                if progress_callback:
                    progress_callback(0, "Error: No category folders found in training data path")
                return False
            
            # Save categories for prediction
            self.categories = sorted(category_folders)
            print(f"Found categories: {self.categories}")
            
            # Prepare for data loading
            images = []
            labels = []
            
            # Load all images and labels
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
                
                # Load images with a progress update every few images
                for j, img_file in enumerate(image_files):
                    try:
                        img_path = os.path.join(category_path, img_file)
                        img = self._load_img(img_path, target_size=(224, 224))
                        img_array = self._img_to_array(img)
                        img_array = self._preprocess_input(img_array)
                        
                        images.append(img_array)
                        labels.append(i)  # Use index as label
                        
                        # Update progress periodically
                        if progress_callback and j % 10 == 0:
                            progress_callback(
                                10 + int((i / len(self.categories)) * 20), 
                                f"Loading {category} images: {j+1}/{len(image_files)}"
                            )
                    except Exception as e:
                        print(f"Error loading image {img_file}: {e}")
            
            # Convert lists to arrays
            if progress_callback:
                progress_callback(30, "Converting data to training format...")
            
            # Use GPU-optimized arrays if available
            images_array = self._np.array(images)
            labels_array = self._np.array(labels)
            
            # One-hot encode the labels for multi-class classification
            y = self.tf.keras.utils.to_categorical(labels_array, num_classes=len(self.categories))
            
            # Split data into training and validation sets
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                images_array, y, test_size=0.2, random_state=42
            )
            
            if progress_callback:
                progress_callback(40, "Creating model architecture...")
            
            # Optimize batch size based on available memory
            batch_size = 32
            if self.using_gpu:
                # Larger batch size for GPU
                batch_size = 64
            
            # Build the model with optimal settings for GPU or CPU
            base_model = self.tf.keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Freeze base model layers to speed up training
            base_model.trainable = False
            
            model = self.tf.keras.models.Sequential([
                base_model,
                self.tf.keras.layers.GlobalAveragePooling2D(),
                self.tf.keras.layers.Dense(512, activation='relu'),
                self.tf.keras.layers.Dropout(0.5),
                self.tf.keras.layers.Dense(len(self.categories), activation='softmax')
            ])
            
            # Compile with efficient settings
            model.compile(
                optimizer=self.tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            if progress_callback:
                progress_callback(45, "Training model (this may take several minutes)...")
            
            # Use efficient callbacks to avoid overfitting and save time
            callbacks = [
                self.tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    patience=5, 
                    restore_best_weights=True
                )
            ]
            
            # Train the model with efficient settings
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
                
                # Add GPU info to the progress message if applicable
                if self.using_gpu:
                    gpu_info = f"Used GPU acceleration ({', '.join(self.gpu_devices)})"
                    progress_callback(98, gpu_info)
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error training model: {str(e)}")
            print(f"Error training model: {e}")
            print(traceback.format_exc())
            return False
    
    def save_model(self, model_path: Optional[str] = None) -> Optional[str]:
        """
        Save the current model to disk
        
        Args:
            model_path: Path to save the model file, or None to use default location
            
        Returns:
            Path to saved model, or None if error
        """
        if not self.tf_available or self.model is None:
            return None
        
        # If no path provided, use default location with timestamp
        if model_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"visibility_model_{timestamp}.h5"
            
            # Create default directory if it doesn't exist
            os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
            model_path = os.path.join(DEFAULT_MODELS_DIR, model_filename)
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            
            # Try saving the model
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            print(f"Error saving model: {e}")
            print(traceback.format_exc())
            return None
    
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
        # Store original log_message if exists
        self._original_log_message = getattr(self, 'log_message', None)

        # Create a direct function that handles both 'prog' and 'progress' arguments
        def adapter_log_message(msg, progress=None, prog=None):
            # Use either progress or prog, with progress taking precedence
            progress_value = progress if progress is not None else prog
            
            if progress_callback and progress_value is not None:
                progress_callback(progress_value, msg)
            print(msg)  # Also print to console
        
        # Set the adapter as the log_message function
        self.log_message = adapter_log_message

        self.log_message("\n" + "=" * 80)
        self.log_message(f"VISIBILITY ANALYSIS STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Altitude threshold: {altitude_threshold}m")
        self.log_message("=" * 80)

        if not self._import_tensorflow():
            self.log_message("TensorFlow not available. Cannot perform analysis.", progress=0)
            return False, {}
        
        self.log_message(f"✓ TensorFlow successfully loaded (GPU: {hasattr(self, 'using_gpu') and self.using_gpu})")

        if not self.model:
            self.log_message("Error: No model loaded. Please load or train a model first.", progress=0)
            return False, {}
        
        self.log_message("✓ Visibility model loaded and ready")

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            self.log_message(f"✓ Output directory confirmed: {output_folder}")
            
            # Initialize location tracking
            self.all_image_locations = {}
            self.selected_image_locations = {}
            self.selected_images = set()
            self.all_images = set()
            
            # STEP 1: Gather altitude data and locations for the ENTIRE dataset
            self.log_message("\nSTEP 1: Loading altitude and location data")
            self.log_message("-" * 50)
            
            self.log_message("Step 1: Loading altitude and location data for dataset...", progress=5)
            
            altitude_map = {}
            lat_lon_map = {}  # Map to store lat/lon for each image
            has_altitude_data = False
            has_location_data = False
            
            # Try to use existing CSV data for altitude information
            try:
                csv_path = os.path.join(output_folder, "image_locations.csv")
                self.log_message(f"Looking for altitude/location data in: {csv_path}")
                if os.path.exists(csv_path):
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # Check if we have all necessary columns
                    required_columns = ['filename', 'altitude', 'latitude', 'longitude']
                    has_required_columns = all(col in df.columns for col in required_columns)
                    
                    if has_required_columns:
                        for _, row in df.iterrows():
                            filename = os.path.basename(row['filename'])
                            altitude_map[filename] = row['altitude']
                            lat_lon_map[filename] = (row['latitude'], row['longitude'])
                        has_altitude_data = True
                        has_location_data = True
                        
                        self.log_message(f"✓ Found CSV file with {len(df)} entries")
                        self.log_message(f"CSV columns: {', '.join(df.columns)}")
                        self.log_message(f"Has all required location columns: {has_required_columns}")
                        
                        self.log_message(f"Found altitude and location data for {len(altitude_map)} images", progress=10)
                    elif 'altitude' in df.columns and 'filename' in df.columns:
                        # Just altitude data is available
                        for _, row in df.iterrows():
                            filename = os.path.basename(row['filename'])
                            altitude_map[filename] = row['altitude']
                        has_altitude_data = True
                        
                        self.log_message(f"✓ Loaded altitude data for {len(altitude_map)} images (no location data)")
                        
                        self.log_message(f"Found altitude data for {len(altitude_map)} images (no location data)", progress=10)
            except Exception as e:
                self.log_message(f"Error loading altitude/location data: {e}")
                self.log_message("⚠ CSV file not found. No altitude or location data available.")
                self.log_message("Warning: Could not load altitude/location data from CSV", progress=10)
            
            # STEP 2: Scan for all processed images in the dataset
            self.log_message("\nSTEP 2: Scanning for processed images in dataset")
            self.log_message("-" * 50)
            
            self.log_message("Step 2: Scanning for processed images in dataset...", progress=15)
            
            all_images_paths = []
            
            # If no image paths provided or empty list, scan the directory
            if not image_paths:
                # Try to extract parent folder from the output path for input images
                input_folder = os.path.dirname(output_folder)
                self.log_message(f"Initial input folder path: {input_folder}")
                if not os.path.isdir(input_folder) or input_folder == output_folder:
                    # If the output folder is directly in the directory we want to scan,
                    # go up one level to avoid scanning output files
                    input_folder = os.path.dirname(input_folder)
                
                self.log_message(f"Adjusted input folder path: {input_folder}")
                if os.path.isdir(input_folder):
                    self.log_message(f"Scanning for processed images in: {input_folder}")
                    self.log_message(f"Scanning for processed images in {input_folder}...", progress=20)
                    
                    # First pass: collect only processed image paths
                    for root, _, files in os.walk(input_folder):
                        for file in files:
                            # Only include jpg/jpeg files (processed images)
                            # Exclude any files in RAW directories or with RAW in the filename
                            if file.lower().endswith(('.jpg', '.jpeg')):
                                full_path = os.path.join(root, file)
                                # Skip files in RAW directories or with RAW in the filename
                                if "raw" not in root.lower() and "raw" not in file.lower():
                                    if os.path.isfile(full_path):
                                        all_images_paths.append(full_path)
            else:
                # Use the provided image paths but filter to only include processed images
                self.log_message(f"Using {len(image_paths)} provided image paths")
                all_images_paths = [path for path in image_paths 
                                  if os.path.isfile(path) and
                                  path.lower().endswith(('.jpg', '.jpeg')) and
                                  "raw" not in path.lower()]
                self.log_message(f"✓ {len(all_images_paths)} valid processed image paths found")
            
            total_images_found = len(all_images_paths)
            
            if total_images_found == 0:
                self.log_message("Error: No images found in dataset", progress=0)
                return False, {}
                
            self.log_message(f"✓ Found {total_images_found} total images in dataset")
            self.log_message(f"Example image paths:")
            for i, path in enumerate(all_images_paths[:3]):
                self.log_message(f"  {i+1}. {path}")
            if total_images_found > 3:
                self.log_message(f"  ... and {total_images_found - 3} more")
            
            self.log_message(f"Found {total_images_found} total images in dataset", progress=25)
            
            # STEP 3: Filter by altitude threshold
            self.log_message(f"\nSTEP 3: Filtering images by altitude threshold ({altitude_threshold}m)")
            self.log_message("-" * 50)
            self.log_message(f"Starting altitude filtering of {total_images_found} images...")
            if has_altitude_data:
                self.log_message(f"Using altitude data from CSV for filtering")
            else:
                self.log_message("⚠ No altitude data available - all images will be included")
            
            self.log_message(f"Step 3: Filtering images by altitude threshold ({altitude_threshold}m)...", progress=30)
            
            below_threshold_images = []
            filtered_by_altitude = 0
            
            for i, img_path in enumerate(all_images_paths):
                filename = os.path.basename(img_path)
                
                # Store all image filenames for visualization
                self.all_images.add(filename)
                if filename in lat_lon_map:
                    self.all_image_locations[filename] = lat_lon_map[filename]
                
                # Check if we have altitude data for this image
                if has_altitude_data and filename in altitude_map:
                    altitude = altitude_map[filename]
                    if altitude <= altitude_threshold:
                        below_threshold_images.append(img_path)
                    else:
                        filtered_by_altitude += 1
                else:
                    # If no altitude data, include the image (better to include than exclude)
                    below_threshold_images.append(img_path)
                
                # Update progress periodically
                if progress_callback and (i % 1000 == 0 or i == total_images_found - 1):
                    self.log_message(
                        f"Filtered {i+1}/{total_images_found} images, {filtered_by_altitude} above threshold",
                        progress=30 + int((i / total_images_found) * 10)  # Progress from 30% to 40%
                    )
                if i % 5000 == 0 and i > 0:
                    self.log_message(f"Processed {i}/{total_images_found} images, {filtered_by_altitude} filtered out so far")
            
            self.log_message(f"✓ Altitude filtering complete: {filtered_by_altitude} images excluded (above {altitude_threshold}m)")
            self.log_message(f"✓ {len(below_threshold_images)} images below altitude threshold")
            
            self.log_message(
                f"Altitude filtering complete: {filtered_by_altitude} of {total_images_found} images excluded (above {altitude_threshold}m)",
                progress=40
            )
            
            # STEP 4: Randomly select up to 5000 images from those below threshold
            self.log_message("\nSTEP 4: Selecting images for visibility analysis")
            self.log_message("-" * 50)
            
            self.log_message("Step 4: Selecting images for visibility analysis...", progress=45)
            
            below_threshold_count = len(below_threshold_images)
            
            if below_threshold_count == 0:
                self.log_message(f"Error: No images found below altitude threshold of {altitude_threshold}m", progress=0)
                return False, {}
            
            # If more than 5000 images, randomly select 5000
            images_to_analyze = below_threshold_images
            if below_threshold_count > 5000:
                self.log_message(f"Too many images below threshold ({below_threshold_count}), randomly selecting 5000...")
                self.log_message(f"Using random seed: 42 for reproducible sampling")
                self.log_message(f"Randomly selecting 5000 images from {below_threshold_count} below threshold...", progress=47)
                
                # Use random sampling without replacement
                import random
                random.seed(42)  # For reproducibility
                images_to_analyze = random.sample(below_threshold_images, 5000)
                
                self.log_message(f"✓ Randomly selected 5000 images from {below_threshold_count} total below threshold")
                self.log_message(f"Selected 5000 random images for analysis", progress=50)
            else:
                self.log_message(f"Using all {below_threshold_count} images below threshold (no sampling needed)")
                self.log_message(f"Using all {below_threshold_count} images below threshold for analysis", progress=50)
            
            # STEP 5: Analyze the selected images
            self.log_message("\nSTEP 5: Analyzing visibility of selected images")
            self.log_message("-" * 50)
            gpu_msg = " with GPU acceleration" if hasattr(self, 'using_gpu') and self.using_gpu else ""
            self.log_message(f"Starting visibility analysis of {len(images_to_analyze)} images{gpu_msg}")
            self.log_message(f"Location data available for {len(self.selected_image_locations)} of {len(self.selected_images)} selected images")
            self.log_message("Beginning image visibility analysis...")
            self.log_message(f"Categories: {', '.join(self.categories)}")
            
            self.log_message(f"Step 5: Analyzing visibility of {len(images_to_analyze)} images{gpu_msg}...", progress=55)
            
            # Store selected image information for visualization
            for img_path in images_to_analyze:
                filename = os.path.basename(img_path)
                self.selected_images.add(filename)
                if filename in lat_lon_map:
                    self.selected_image_locations[filename] = lat_lon_map[filename]
            
            # Reset statistics
            self.visibility_stats = {
                'total_images_found': total_images_found,
                'filtered_by_altitude': filtered_by_altitude,
                'below_threshold_count': below_threshold_count,
                'images_analyzed': len(images_to_analyze),
                'by_category': {cat: 0 for cat in self.categories},
                'percentages': {cat: 0 for cat in self.categories},
                'analyzed_images': []
            }
            
            # Process images and collect results
            results = []
            for i, img_path in enumerate(images_to_analyze):
                try:
                    # Process image
                    img = self._load_img(img_path, target_size=(224, 224))
                    img_array = self._img_to_array(img)
                    img_array = self._preprocess_input(img_array)
                    img_array = self._np.expand_dims(img_array, axis=0)
                    
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
                        self.log_message(
                            f"Analyzed {i+1}/{len(images_to_analyze)} images",
                            progress=55 + int((i / len(images_to_analyze)) * 35)  # Progress from 55% to 90%
                        )
                    if i % 500 == 0 and i > 0:
                        # Print status update with category breakdown
                        category_counts = ", ".join([f"{cat}: {count}" for cat, count in self.visibility_stats['by_category'].items() if count > 0])
                        self.log_message(f"Processed {i}/{len(images_to_analyze)} images. Counts so far: {category_counts}")
                    
                except Exception as e:
                    self.log_message(f"Error processing image {img_path}: {e}")
                    self.log_message(traceback.format_exc())
            
            self.log_message("\nAnalysis complete. Results by category:")
            for category in self.categories:
                count = self.visibility_stats['by_category'].get(category, 0)
                self.log_message(f"  {category}: {count} images")
            
            # Save results to CSV with additional location data
            self.log_message("\nSaving results and creating visualizations...")
            if results:
                csv_path = os.path.join(output_folder, "visibility_results.csv")
                try:
                    df = self._pd.DataFrame(results)
                    
                    # Include altitude and location in output if we have it
                    cols = ['image', 'visibility', 'confidence']
                    
                    # Add location data if available
                    self.log_message("Adding location data to CSV output...")
                    location_count = 0
                    for i, row in df.iterrows():
                        img_filename = row['image']
                        if img_filename in lat_lon_map:
                            df.at[i, 'latitude'] = lat_lon_map[img_filename][0]
                            df.at[i, 'longitude'] = lat_lon_map[img_filename][1]
                            location_count += 1
                    
                    self.log_message(f"✓ Added location data for {location_count} of {len(df)} images")
                    
                    if has_location_data:
                        cols.extend(['latitude', 'longitude'])
                    
                    if 'altitude' in df.columns:
                        cols.append('altitude')
                        self.log_message(f"✓ Including altitude data in CSV output")
                    
                    # Ensure columns exist in the DataFrame
                    for col in cols:
                        if col not in df.columns:
                            if col == 'latitude' or col == 'longitude':
                                df[col] = None
                    
                    df = df[cols]  # Keep only these columns for output
                    df.to_csv(csv_path, index=False)
                    
                    self.log_message(f"✓ Results saved to CSV: {csv_path}")
                    
                    self.log_message("Creating visualizations...", progress=95)
                    
                    # Create visualizations
                    self.create_visibility_chart(csv_path, output_folder)
                    self.create_enhanced_visibility_chart(csv_path, output_folder)

                    
                    # Create map and shapefile if we have location data
                    if has_location_data:
                        self.log_message("Creating location map...")
                        self.log_message("Creating location map...", progress=96)
                        map_path = self.create_location_map(output_folder)
                        self.log_message(f"✓ Location map created: {map_path}")
                        
                        self.log_message("Creating shapefile of selected image locations...")
                        self.log_message("Creating shapefile...", progress=97)
                        shapefile_path = self.create_shapefile(output_folder)
                        self.log_message(f"✓ Shapefile created: {shapefile_path}")
                except Exception as e:
                    self.log_message(f"Error saving results to CSV: {e}")
                    self.log_message(traceback.format_exc())
            
            # Calculate percentages for statistics
            self.log_message("\nCalculating category percentages...")
            total_analyzed = len(results)
            if total_analyzed > 0:
                for category in self.categories:
                    count = self.visibility_stats['by_category'].get(category, 0)
                    percentage = (count / total_analyzed) * 100
                    self.visibility_stats['percentages'][category] = percentage
                    self.log_message(f"  {category}: {count} images ({percentage:.1f}%)")
            
            gpu_info = ""
            if hasattr(self, 'using_gpu') and self.using_gpu:
                gpu_info = f" using GPU acceleration ({', '.join(self.gpu_devices)})"
            
            summary = (
                f"Analysis complete{gpu_info}.\n"
                f"Total images in dataset: {total_images_found}\n"
                f"Images filtered out (above {altitude_threshold}m threshold): {filtered_by_altitude}\n"
                f"Images below threshold: {below_threshold_count}\n"
                f"Images analyzed: {total_analyzed}"
            )
            
            self.log_message(summary, progress=100)
            
            self.log_message("\n" + "=" * 80)
            self.log_message(f"VISIBILITY ANALYSIS COMPLETED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_message(f"Total images in dataset: {total_images_found}")
            self.log_message(f"Images filtered out (above {altitude_threshold}m threshold): {filtered_by_altitude}")
            self.log_message(f"Images below threshold: {below_threshold_count}")
            self.log_message(f"Images analyzed: {total_analyzed}")
            self.log_message("=" * 80)
            
            # Restore original log_message if it existed
            if self._original_log_message:
                self.log_message = self._original_log_message
            
            return True, self.visibility_stats
            
        except Exception as e:
            self.log_message("\n" + "=" * 80)
            self.log_message("ERROR DURING VISIBILITY ANALYSIS:")
            self.log_message(str(e))
            self.log_message(traceback.format_exc())
            self.log_message("=" * 80)
            self.log_message(f"Error during visibility analysis: {str(e)}", progress=0)
            
            # Restore original log_message if it existed
            if self._original_log_message:
                self.log_message = self._original_log_message
            
            return False, {}
    
    def create_visibility_chart(self, csv_path: str, output_folder: str) -> Optional[str]:
        """
        Create bar chart visualization of visibility categories
        
        Args:
            csv_path: Path to the CSV file with analysis results
            output_folder: Directory to save the chart
            
        Returns:
            Path to the saved chart, or None if error
        """
        try:
            # Read the CSV file
            df = self._pd.read_csv(csv_path)
            
            # Count occurrences of each visibility category
            visibility_counts = df['visibility'].value_counts()
            
            # Convert to a DataFrame for seaborn plotting
            plot_df = self._pd.DataFrame({
                'category': visibility_counts.index,
                'count': visibility_counts.values
            })
            
            # Sort by visibility category in a logical order
            category_order = ['zero_visibility', 'low_visibility', 'good_visibility', 'great_visibility']
            plot_df['order'] = plot_df['category'].map(lambda x: 
                                                     category_order.index(x) if x in category_order else 999)
            plot_df = plot_df.sort_values('order').drop('order', axis=1)
            
            # Set up the plot style
            fig, ax = self._plt.subplots(figsize=(10, 6))
            self._sns.set_style("whitegrid")
            
            # Color mapping for different visibility categories
            colors = {
                'zero_visibility': 'red',
                'low_visibility': 'orange',
                'good_visibility': 'green',
                'great_visibility': 'blue'
            }
            
            # Create the bar chart with the new syntax
            chart = self._sns.barplot(
                data=plot_df,
                x='category',  # This becomes hue as well
                y='count',
                hue='category',  # Add hue parameter
                palette=colors,  # Keep your custom colors
                legend=False,    # No legend needed
                ax=ax
            )
            
            # Add count labels on top of each bar
            for i, row in enumerate(plot_df.itertuples()):
                ax.text(i, row.count + 5, str(row.count), ha='center', fontweight='bold')
            
            # Add labels and title
            self._plt.title('Visibility Analysis Results', fontsize=16)
            self._plt.xlabel('Visibility Category', fontsize=14)
            self._plt.ylabel('Number of Images', fontsize=14)
            self._plt.xticks(rotation=15)
            
            # Calculate percentages for each category
            total = plot_df['count'].sum()
            
            # Add percentage annotations
            for i, row in enumerate(plot_df.itertuples()):
                percentage = (row.count/total)*100
                ax.text(i, row.count/2, f'{percentage:.1f}%', 
                        ha='center', color='white', fontweight='bold')
            
            self._plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(output_folder, "visibility_analysis.png")
            self._plt.savefig(chart_path, dpi=300)
            self._plt.close()
            
            self.log_message(f"Visibility analysis chart saved to: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.log_message(f"Error creating visibility chart: {e}")
            self.log_message(traceback.format_exc())
            return None
    
    def create_enhanced_visibility_chart(self, csv_path: str, output_folder: str) -> Optional[str]:
        """
        Create an enhanced bar chart visualization with example images for each visibility category
        
        Args:
            csv_path: Path to the CSV file with analysis results
            output_folder: Directory to save the chart
            
        Returns:
            Path to the saved chart, or None if error
        """
        try:
            # Read the CSV file
            df = self._pd.read_csv(csv_path)
            
            # Count occurrences of each visibility category
            visibility_counts = df['visibility'].value_counts()
            
            # Convert to a DataFrame for plotting
            plot_df = self._pd.DataFrame({
                'category': visibility_counts.index,
                'count': visibility_counts.values
            })
            
            # Sort by visibility category in a logical order
            category_order = ['zero_visibility', 'low_visibility', 'good_visibility', 'great_visibility']
            plot_df['order'] = plot_df['category'].map(lambda x: 
                                                     category_order.index(x) if x in category_order else 999)
            plot_df = plot_df.sort_values('order').drop('order', axis=1)
            
            # Create a larger figure to accommodate images
            fig = self._plt.figure(figsize=(14, 10))
            
            # Create gridspec for layout: bar chart on top, images on bottom
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3)
            
            # Top subplot for bar chart
            ax_bar = fig.add_subplot(gs[0])
            
            # Color mapping for different visibility categories
            colors = {
                'zero_visibility': 'red',
                'low_visibility': 'orange',
                'good_visibility': 'green',
                'great_visibility': 'blue'
            }
            
            # Create the bar chart
            self._sns.barplot(
                data=plot_df,
                x='category',
                y='count',
                hue='category',
                palette=colors,
                legend=False,
                ax=ax_bar
            )
            
            # Add count labels on top of each bar
            for i, row in enumerate(plot_df.itertuples()):
                ax_bar.text(i, row.count + 5, str(row.count), ha='center', fontweight='bold')
            
            # Add labels and title
            ax_bar.set_title('Visibility Analysis Results', fontsize=16)
            ax_bar.set_xlabel('Visibility Category', fontsize=14)
            ax_bar.set_ylabel('Number of Images', fontsize=14)
            ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=0)  # Keep labels horizontal
            
            # Calculate percentages for each category
            total = plot_df['count'].sum()
            
            # Add percentage annotations
            for i, row in enumerate(plot_df.itertuples()):
                percentage = (row.count/total)*100
                ax_bar.text(i, row.count/2, f'{percentage:.1f}%', 
                        ha='center', color='white', fontweight='bold')
            
            # Bottom subplot for example images
            ax_img = fig.add_subplot(gs[1])
            ax_img.axis('off')  # Turn off axis for image display
            
            # Find an example image for each category with highest confidence
            example_images = {}
            for category in plot_df['category']:
                # Filter dataset for this category
                category_df = df[df['visibility'] == category]
                
                # Skip if no images in this category
                if len(category_df) == 0:
                    continue
                    
                # Sort by confidence and get the highest confidence image
                if 'confidence' in category_df.columns:
                    category_df = category_df.sort_values('confidence', ascending=False)
                
                # Get the full path to the image (we need to reconstruct it)
                img_filename = category_df.iloc[0]['image']
                
                # Look for the image first in the parent directory of the output folder
                input_folder = os.path.dirname(output_folder)
                example_paths = []
                
                # Search for the image in various locations
                for root, _, files in os.walk(input_folder):
                    if img_filename in files:
                        example_paths.append(os.path.join(root, img_filename))
                
                if example_paths:
                    example_images[category] = example_paths[0]
                    self.log_message(f"Found example image for {category}: {example_paths[0]}")
                else:
                    self.log_message(f"Could not find example image for {category}: {img_filename}")
            
            # If we found some example images, display them
            if example_images:
                # Number of categories with example images
                num_categories = len(example_images)
                
                # Create grid of axes for images
                img_axes = []
                # Use reduced margins (10% on each side instead of default)
                margin = 0.05
                usable_width = 1.0 - (2 * margin)
                col_width = usable_width / num_categories
                
                for i, (category, img_path) in enumerate(example_images.items()):
                    # Create axis for this image - with reduced margins
                    img_ax = ax_img.inset_axes([margin + (i * col_width), 0.05, col_width, 0.9])
                    img_axes.append(img_ax)
                    
                    try:
                        # Read and display image
                        img = self._plt.imread(img_path)
                        img_ax.imshow(img)
                        img_ax.set_title(f"{category}", fontsize=12)
                        img_ax.axis('off')
                    except Exception as e:
                        self.log_message(f"Error displaying example image for {category}: {e}")
                        img_ax.text(0.5, 0.5, f"Error loading image", 
                                   ha='center', va='center', transform=img_ax.transAxes)
                        img_ax.axis('off')
            else:
                # If we couldn't find any example images, show a message
                ax_img.text(0.5, 0.5, "Example images not found. Check image paths and permissions.", 
                          ha='center', va='center', fontsize=14, transform=ax_img.transAxes)
                
            # Add overall title
            #fig.suptitle('Visibility Analysis Results', fontsize=18, y=0.98)
            
            # Use tighter margins for the overall figure - reduced left, right, bottom margins
            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])  # [left, bottom, right, top]
            
            # Save enhanced chart
            chart_path = os.path.join(output_folder, "visibility_analysis_with_examples.png")
            fig.savefig(chart_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' reduces margins further
            self._plt.close(fig)
            
            self.log_message(f"Enhanced visibility analysis chart with examples saved to: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.log_message(f"Error creating enhanced visibility chart: {e}")
            self.log_message(traceback.format_exc())
            return None

    def verify_training_data(self, training_data_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Verify and fix training data directory structure if needed
        
        Args:
            training_data_path: Path to training data
            progress_callback: Optional callback function
            
        Returns:
            Boolean indicating if the structure is valid
        """
        try:
            # Check if the directory exists
            if not os.path.isdir(training_data_path):
                self.log_message(f"Training path doesn't exist: {training_data_path}", progress=0)
                return False
            
            # List all subdirectories
            subdirs = [d for d in os.listdir(training_data_path) 
                      if os.path.isdir(os.path.join(training_data_path, d))]
            
            if not subdirs:
                self.log_message(f"No subdirectories found in {training_data_path}", progress=0)
                return False
            
            # Check if all category directories exist
            missing_categories = [cat for cat in self.categories 
                                if cat not in subdirs]
            
            if missing_categories:
                msg = f"Missing category directories: {', '.join(missing_categories)}"
                self.log_message(msg, progress=0)
                self.log_message("Each visibility category must have its own subfolder.", progress=5)
                self.log_message(f"Expected structure: {', '.join(self.categories)}", progress=10)
                return False
            
            # Check each category directory for images
            valid_categories = 0
            for cat in self.categories:
                cat_dir = os.path.join(training_data_path, cat)
                image_files = [f for f in os.listdir(cat_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                
                if image_files:
                    valid_categories += 1
                    self.log_message(f"Found {len(image_files)} images in {cat} category", progress=15 + (valid_categories * 5))
                else:
                    self.log_message(f"Warning: No images found in {cat} category", progress=15 + (valid_categories * 5))
            
            # Success if at least one category has images
            return valid_categories > 0
        
        except Exception as e:
            self.log_message(f"Error checking training data: {str(e)}", progress=0)
            return False
    
    def get_summary_report(self) -> List[str]:
        """
        Generate a summary report of visibility analysis
        
        Returns:
            List of strings with report lines
        """
        if not self.visibility_stats.get('images_analyzed', 0):
            return ["No visibility analysis performed."]
            
        report = []
        
        report.append("Visibility Analysis Results:")
        report.append("--------------------------")
        
        # Add GPU usage information if available
        if hasattr(self, 'using_gpu') and self.using_gpu:
            report.append(f"Processed using GPU acceleration: {', '.join(self.gpu_devices)}")
        
        # Include dataset statistics
        if 'total_images_found' in self.visibility_stats:
            total_found = self.visibility_stats['total_images_found']
            report.append(f"Total images in dataset: {total_found}")
        
        # Include altitude filtering statistics
        if 'filtered_by_altitude' in self.visibility_stats and self.visibility_stats['filtered_by_altitude'] > 0:
            filtered = self.visibility_stats['filtered_by_altitude']
            threshold = self.altitude_threshold
            filtered_pct = (filtered / total_found) * 100 if total_found else 0
            report.append(f"Images excluded (above {threshold}m altitude threshold): {filtered} ({filtered_pct:.1f}%)")
        
        # Include selection statistics
        if 'below_threshold_count' in self.visibility_stats:
            below_threshold = self.visibility_stats['below_threshold_count']
            below_pct = (below_threshold / total_found) * 100 if total_found else 0
            report.append(f"Images below threshold: {below_threshold} ({below_pct:.1f}%)")
        
        # Number of images actually analyzed
        analyzed = len(self.visibility_stats['analyzed_images'])
        if 'below_threshold_count' in self.visibility_stats and self.visibility_stats['below_threshold_count'] > 0:
            analyzed_pct = (analyzed / self.visibility_stats['below_threshold_count']) * 100
            report.append(f"Images analyzed: {analyzed} ({analyzed_pct:.1f}% of below-threshold images)")
        else:
            report.append(f"Images analyzed: {analyzed}")
        
        report.append("\nVisibility Category Results:")
        
        # Sort categories by visibility quality (assuming certain ordering)
        sorted_cats = sorted(self.categories, 
                             key=lambda x: ['zero_visibility', 'low_visibility', 'good_visibility', 'great_visibility'].index(x) 
                             if x in ['zero_visibility', 'low_visibility', 'good_visibility', 'great_visibility'] else 999)
        
        for category in sorted_cats:
            count = self.visibility_stats['by_category'].get(category, 0)
            percentage = self.visibility_stats['percentages'].get(category, 0)
            report.append(f"{category}: {count} images ({percentage:.1f}%)")
        
        report.append("\nSummary assessment:")
        
        # Calculate the percentage of good or better visibility
        good_or_better = (self.visibility_stats['by_category'].get('good_visibility', 0) + 
                         self.visibility_stats['by_category'].get('great_visibility', 0))
        
        good_or_better_pct = (good_or_better / len(self.visibility_stats['analyzed_images'])) * 100 if self.visibility_stats['analyzed_images'] else 0
        
        if good_or_better_pct >= 80:
            report.append("Excellent overall visibility in this dataset.")
        elif good_or_better_pct >= 60:
            report.append("Good overall visibility in this dataset.")
        elif good_or_better_pct >= 40:
            report.append("Moderate overall visibility in this dataset.")
        elif good_or_better_pct >= 20:
            report.append("Poor overall visibility in this dataset.")
        else:
            report.append("Very poor overall visibility in this dataset.")
        
        return report
    
    def create_location_map(self, output_folder: str) -> Optional[str]:
        """
        Create a map showing the locations of all images and selected images
        
        Args:
            output_folder: Directory to save the map
            
        Returns:
            Path to the saved map, or None if error
        """
        try:
            if not hasattr(self, 'all_image_locations') or not self.all_image_locations:
                self.log_message("No location data available to create map")
                return None
                
            # Create figure
            fig, ax = self._plt.subplots(figsize=(12, 8))
            
            # Extract coordinates
            all_lats = []
            all_lons = []
            selected_lats = []
            selected_lons = []
            
            # Get coordinates for all images
            for filename, (lat, lon) in self.all_image_locations.items():
                all_lats.append(lat)
                all_lons.append(lon)
                
                # Add to selected if in selected images
                if filename in self.selected_images:
                    selected_lats.append(lat)
                    selected_lons.append(lon)
            
            # Plot all images as black-bordered circles
            ax.scatter(all_lons, all_lats, s=30, marker='o', edgecolors='black', 
                      facecolors='none', alpha=0.5, label='All Images')
            
            # Plot selected images as filled red circles
            ax.scatter(selected_lons, selected_lats, s=20, marker='o', color='red', 
                      label='Selected Images', alpha=0.7)
            
            # Add title and labels
            ax.set_title('Image Locations Map - Selected vs. All Images', fontsize=16)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add legend
            ax.legend(loc='best')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Save map
            map_path = os.path.join(output_folder, "visibility_analysis_map.png")
            fig.tight_layout()
            self._plt.savefig(map_path, dpi=300)
            self._plt.close()
            
            self.log_message(f"Image locations map saved to: {map_path}")
            return map_path
            
        except Exception as e:
            self.log_message(f"Error creating location map: {e}")
            self.log_message(traceback.format_exc())
            return None

    def create_shapefile(self, output_folder: str) -> Optional[str]:
        """
        Create a shapefile of the selected image locations
        
        Args:
            output_folder: Directory to save the shapefile
            
        Returns:
            Path to the saved shapefile, or None if error
        """
        try:
            if not hasattr(self, 'selected_image_locations') or not self.selected_image_locations:
                self.log_message("No location data available to create shapefile")
                return None
                
            # Attempt to import shapefile libraries
            try:
                import geopandas as gpd
                from shapely.geometry import Point
            except ImportError:
                self.log_message("Geopandas or Shapely not available. Installing required packages...")
                import pip
                pip.main(['install', 'geopandas', 'shapely'])
                
                # Try importing again
                import geopandas as gpd
                from shapely.geometry import Point
            
            # Create a list of points and attributes
            points = []
            properties = []
            
            for filename, (lat, lon) in self.selected_image_locations.items():
                points.append(Point(lon, lat))
                
                # Find visibility category if available
                visibility = "unknown"
                confidence = 0.0
                altitude = None
                
                for result in self.visibility_stats.get('analyzed_images', []):
                    if result['image'] == filename:
                        visibility = result['visibility']
                        confidence = result['confidence']
                        if 'altitude' in result:
                            altitude = result['altitude']
                        break
                
                properties.append({
                    'filename': filename,
                    'visibility': visibility,
                    'confidence': confidence,
                    'altitude': altitude,
                    'latitude': lat,
                    'longitude': lon
                })
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(properties, geometry=points, crs="EPSG:4326")
            
            # Save to shapefile directly in the output directory with updated name
            shapefile_path = os.path.join(output_folder, "visibility_analysis.shp")
            gdf.to_file(shapefile_path)
            
            # Create a CSV backup with consistent name
            csv_path = os.path.join(output_folder, "visibility_analysis.csv")
            gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
            
            self.log_message(f"Shapefile saved to: {shapefile_path}")
            self.log_message(f"CSV locations saved to: {csv_path}")
            return shapefile_path
            
        except Exception as e:
            self.log_message(f"Error creating shapefile: {e}")
            self.log_message(traceback.format_exc())
            
            # Try creating just a CSV file as a fallback
            try:
                # Save CSV directly to output folder with consistent name
                csv_path = os.path.join(output_folder, "visibility_analysis.csv")
                
                data = []
                for filename, (lat, lon) in self.selected_image_locations.items():
                    visibility = "unknown"
                    confidence = 0.0
                    altitude = None
                    
                    for result in self.visibility_stats.get('analyzed_images', []):
                        if result['image'] == filename:
                            visibility = result['visibility']
                            confidence = result['confidence']
                            if 'altitude' in result:
                                altitude = result['altitude']
                            break
                    
                    data.append({
                        'filename': filename,
                        'visibility': visibility,
                        'confidence': confidence,
                        'altitude': altitude,
                        'latitude': lat,
                        'longitude': lon
                    })
                
                df = self._pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
                self.log_message(f"Fallback CSV locations saved to: {csv_path}")
                return csv_path
            except Exception as csv_error:
                self.log_message(f"Error creating fallback CSV: {csv_error}")
                return None

    def log_message(self, message: str, progress_callback: Optional[Callable] = None, progress: Optional[int] = None):
        """
        Log a message to both console and GUI (if progress_callback is provided)
        
        Args:
            message: Message to log
            progress_callback: Optional callback to update GUI
            progress: Optional progress percentage to update
        """
        # Always print to console
        print(message)
        
        # If we have a progress callback, also send the message there
        if progress_callback:
            if progress is not None:
                # If progress is provided, update both progress and status message
                progress_callback(progress, message)
            else:
                # If no progress is provided, just update the status message
                # This assumes the progress_callback can handle None as the progress value
                progress_callback(None, message)