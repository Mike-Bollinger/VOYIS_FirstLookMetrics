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
    
    def __init__(self, log_callback: Optional[Callable] = None):
        """
        Initialize the VisibilityAnalyzer
        
        Args:
            log_callback: Optional function to call for logging messages
        """
        self.model = None
        self.tf = None
        self._cv2 = None
        self._plt = None
        self._pd = None
        self._np = None
        self._folium = None
        self._geopandas = None
        self._sns = None  # Add seaborn for enhanced charts
        self.using_gpu = False
        self.tf_available = False
        self.log_callback = log_callback
        
        # Initialize categories
        self.categories = ['zero_visibility', 'low_visibility', 'good_visibility', 'great_visibility']
        
        # Initialize statistics containers
        self.visibility_stats = {}
        self.all_images = set()
        self.all_image_locations = {}
        self.selected_images = set()
        self.selected_image_locations = {}
    
    def log_message(self, message: str, progress: Optional[int] = None):
        """
        Log a message using the callback if available, otherwise print
        
        Args:
            message: Message to log
            progress: Optional progress percentage
        """
        if self.log_callback:
            try:
                # Try calling with progress first
                if progress is not None:
                    self.log_callback(message, progress=progress)
                else:
                    self.log_callback(message)
            except TypeError:
                # Fallback if the callback doesn't accept progress
                self.log_callback(message)
        else:
            print(f"VisibilityAnalyzer: {message}")
    
    def set_log_callback(self, log_callback: Callable):
        """Set or update the log callback function"""
        self.log_callback = log_callback
    
    def _import_required_libraries(self):
        """Import required libraries on demand"""
        try:
            if self._cv2 is None:
                import cv2
                self._cv2 = cv2
                
            if self._plt is None:
                import matplotlib.pyplot as plt
                self._plt = plt
                
            if self._pd is None:
                import pandas as pd
                self._pd = pd
                
            if self._np is None:
                import numpy as np
                self._np = np
                
            # Try to import seaborn for enhanced plotting
            if self._sns is None:
                try:
                    import seaborn as sns
                    self._sns = sns
                except ImportError:
                    # Seaborn is optional
                    pass
                
            return True
            
        except ImportError as e:
            self.log_message(f"Required libraries not available: {e}")
            return False
    
    def _import_optional_libraries(self) -> bool:
        """Import optional libraries for mapping features"""
        try:
            if self._folium is None:
                import folium
                self._folium = folium
                
            if self._geopandas is None:
                import geopandas as gpd
                self._geopandas = gpd
                
            return True
            
        except ImportError:
            self.log_message("Optional mapping libraries not available (folium, geopandas)")
            return False
    
    def _import_tensorflow(self) -> bool:
        """Import TensorFlow with proper error handling"""
        try:
            if self.tf is None:
                import tensorflow as tf
                self.tf = tf
                
                # Check for GPU
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        # Enable memory growth for GPU
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        self.using_gpu = True
                        self.log_message(f"✓ TensorFlow loaded with GPU support ({len(gpus)} GPU(s) found)")
                    except Exception as e:
                        self.log_message(f"GPU setup failed, using CPU: {e}")
                        self.using_gpu = False
                else:
                    self.using_gpu = False
                    self.log_message("✓ TensorFlow loaded (CPU only)")
            
            self.tf_available = True
            return True
            
        except ImportError as e:
            self.log_message(f"TensorFlow not available: {e}")
            self.tf_available = False
            return False
        except Exception as e:
            self.log_message(f"Error importing TensorFlow: {e}")
            self.tf_available = False
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
        Analyze images for visibility and generate reports
        
        Args:
            image_paths: List of image paths to analyze (if empty, will scan output folder's parent)
            output_folder: Output directory for results
            progress_callback: Optional callback for progress updates
            altitude_threshold: Altitude threshold for filtering
            
        Returns:
            Tuple of (success: bool, results: dict)
        """
        def log_message(msg, progress=None):
            # Always call self.log_message first
            self.log_message(msg)
            # Then call progress callback if available
            if progress_callback and progress is not None:
                try:
                    progress_callback(progress, msg)
                except Exception:
                    pass  # Ignore progress callback errors
        
        log_message("Starting visibility analysis...", progress=0)
        
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
            
            # If no image paths provided, scan for images
            if not image_paths:
                # Look in the input folder that should be the parent of output folder
                parent_dir = os.path.dirname(output_folder)
                
                # Check if parent directory name suggests it's an image folder
                if any(keyword in parent_dir.lower() for keyword in ['image', 'photo', 'jpg', 'png']):
                    scan_dir = parent_dir
                else:
                    # Look for common image folder patterns
                    possible_dirs = []
                    parent_parent = os.path.dirname(parent_dir)
                    
                    for item in os.listdir(parent_parent):
                        item_path = os.path.join(parent_parent, item)
                        if os.path.isdir(item_path):
                            if any(keyword in item.lower() for keyword in ['image', 'jpg', 'raw', 'processed', 'advanced']):
                                possible_dirs.append(item_path)
                    
                    # Use the first matching directory or fall back to parent
                    scan_dir = possible_dirs[0] if possible_dirs else parent_dir
                
                log_message(f"Scanning for images in: {scan_dir}")
                
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
                image_paths = []
                
                if os.path.exists(scan_dir):
                    for root, dirs, files in os.walk(scan_dir):
                        # Skip the output folder itself
                        if root == output_folder:
                            continue
                        for file in files:
                            if file.lower().endswith(image_extensions):
                                image_paths.append(os.path.join(root, file))
                
                if not image_paths:
                    log_message(f"No images found in {scan_dir}", progress=0)
                    return False, {}
            
            log_message(f"Found {len(image_paths)} images to analyze", progress=5)
            
            # Initialize tracking variables
            results = []
            processed_count = 0
            error_count = 0
            
            # Process images in batches for efficiency
            batch_size = 32 if self.using_gpu else 16
            total_batches = (len(image_paths) + batch_size - 1) // batch_size
            
            log_message(f"Processing {len(image_paths)} images in {total_batches} batches", progress=15)
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]
                
                # Load and preprocess batch
                batch_images = []
                batch_info = []
                
                for img_path in batch_paths:
                    try:
                        # Use tf.keras.preprocessing.image.load_img for compatibility
                        img = self.tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                        img_array = self.tf.keras.preprocessing.image.img_to_array(img)
                        img_normalized = img_array.astype('float32') / 255.0
                        
                        batch_images.append(img_normalized)
                        batch_info.append({
                            'path': img_path,
                            'filename': os.path.basename(img_path)
                        })
                    except Exception as e:
                        error_count += 1
                        log_message(f"Could not load image: {img_path} - {e}")
                
                if batch_images:
                    # Run prediction
                    batch_array = self._np.array(batch_images)
                    predictions = self.model.predict(batch_array, verbose=0)
                    
                    # Process predictions
                    for i, (pred, info) in enumerate(zip(predictions, batch_info)):
                        try:
                            # Get predicted class and confidence
                            class_idx = self._np.argmax(pred)
                            confidence = float(pred[class_idx])
                            
                            # Map class index to visibility category
                            if class_idx < len(self.categories):
                                visibility = self.categories[class_idx]
                            else:
                                visibility = f"class_{class_idx}"
                            
                            result = {
                                'image': info['filename'],
                                'visibility': visibility,
                                'confidence': confidence
                            }
                            
                            results.append(result)
                            processed_count += 1
                            
                        except Exception as e:
                            error_count += 1
                            log_message(f"Error processing prediction for {info['filename']}: {e}")
                
                # Update progress
                progress = 15 + int((batch_idx + 1) / total_batches * 70)
                log_message(f"Processed batch {batch_idx + 1}/{total_batches}", progress=progress)
            
            log_message(f"✓ Analysis complete: {processed_count} images processed, {error_count} errors", progress=85)
            
            if not results:
                log_message("No results to save!", progress=0)
                return False, {}
            
            # Save results to CSV
            log_message("Saving results and creating visualizations...", progress=90)
            
            csv_path = os.path.join(output_folder, "visibility_results.csv")
            try:
                df = self._pd.DataFrame(results)
                
                # Save basic CSV
                df.to_csv(csv_path, index=False)
                log_message(f"✓ Results saved to CSV: {csv_path}")
                
                # Verify file was created
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    log_message(f"✓ CSV file created successfully: {file_size} bytes")
                else:
                    log_message("ERROR: CSV file was not created!")
                    return False, {}
                
                # Create basic visualization if matplotlib is available
                if self._plt:
                    try:
                        log_message("Creating visualizations...", progress=95)
                        chart_path = self.create_visibility_chart(csv_path, output_folder)
                        if chart_path and os.path.exists(chart_path):
                            log_message(f"✓ Visibility chart created: {chart_path}")
                    except Exception as chart_error:
                        log_message(f"Warning: Error creating charts: {chart_error}")
                
                log_message("✓ Visibility analysis completed successfully!", progress=100)
                
                # Return results summary
                summary = {
                    'total_images_found': len(image_paths),
                    'images_analyzed': processed_count,
                    'errors': error_count,
                    'csv_path': csv_path,
                    'by_category': df['visibility'].value_counts().to_dict() if not df.empty else {}
                }
                
                return True, summary
                        
            except Exception as e:
                log_message(f"Error saving results to CSV: {e}")
                return False, {}
                
        except Exception as e:
            log_message(f"Error during visibility analysis: {e}")
            return False, {}

    def create_visibility_chart(self, csv_path: str, output_folder: str) -> Optional[str]:
        """Create a basic visibility distribution chart"""
        try:
            if not self._plt:
                return None
                
            df = self._pd.read_csv(csv_path)
            
            # Create chart
            visibility_counts = df['visibility'].value_counts()
            
            fig, ax = self._plt.subplots(figsize=(10, 6))
            bars = ax.bar(visibility_counts.index, visibility_counts.values)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
            
            ax.set_title('Visibility Distribution')
            ax.set_xlabel('Visibility Category')
            ax.set_ylabel('Number of Images')
            
            chart_path = os.path.join(output_folder, 'visibility_analysis.png')
            self._plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            self._plt.close()
            
            return chart_path
            
        except Exception as e:
            self.log_message(f"Error creating visibility chart: {e}")
            return None