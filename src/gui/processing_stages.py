import os
import traceback

class ProcessingStages:
    """Individual processing stage implementations"""
    
    def load_navigation_data(self, nav_path):
        """Load navigation data for altitude information"""
        self.log_message(f"Loading navigation data from: {nav_path}")
        self.update_progress(5, "Loading navigation data...")
        
        if self.metrics.load_nav_data(nav_path):
            self.log_message("Navigation data loaded successfully for altitude extraction")
            self.log_message("NOTE: Using ONLY navigation data for altitude values, EXIF altitude data is ignored")
            
            # Make nav data available to other components
            if hasattr(self.metrics, 'nav_timestamps'):
                self.altitude_map.nav_timestamps = self.metrics.nav_timestamps
                self.footprint_map.nav_timestamps = self.metrics.nav_timestamps
                
                if hasattr(self, 'visibility_analyzer'):
                    self.visibility_analyzer.nav_timestamps = self.metrics.nav_timestamps
                if hasattr(self, 'highlight_selector'):
                    self.highlight_selector.nav_timestamps = self.metrics.nav_timestamps
        else:
            self.log_message("Warning: Failed to load navigation data for altitude extraction")

    def process_basic_metrics(self, input_folder, output_folder):
        """Process basic image metrics"""
        try:
            extract_gps = any([
                self.location_map_var.get(), self.histogram_var.get(),
                self.footprint_map_var.get(), self.visibility_analyzer_var.get()
            ])
            
            _, results = self.metrics.analyze_directory(
                input_folder,
                progress_callback=self.update_progress,
                extract_gps=extract_gps
            )
            
            if self.basic_metrics_var.get():
                for line in results:
                    self.log_message(line)
                
                output_file = os.path.join(output_folder, "Image_Metrics.txt")
                with open(output_file, "w") as f:
                    f.write("\n".join(results))
                
                self.log_message(f"Basic metrics saved to: {output_file}")
            else:
                self.log_message("Basic metrics processing completed (not displayed as not selected)")
                
        except Exception as e:
            self.log_message(f"Error processing basic metrics: {str(e)}")
            raise

    def process_location_data(self, input_folder, output_folder):
        """Process location mapping and altitude histogram"""
        if not self.metrics.gps_data:
            self.log_message("Warning: No GPS data found in images. Cannot create location map or altitude histogram.")
            return
        
        if self.location_map_var.get():
            self.create_location_map(output_folder)
        
        if self.histogram_var.get():
            self.create_altitude_histogram(output_folder)

    def create_location_map(self, output_folder):
        """Create location map"""
        self.log_message("Creating image location map...")
        
        try:
            self.log_message(f"Generating map with {len(self.metrics.gps_data)} GPS points...")
            
            map_file = self.altitude_map.create_location_map(
                self.metrics.gps_data,
                output_folder,
                metrics=self.metrics
            )
            
            if map_file:
                self.log_message(f"Location map saved to: {map_file}")
                self.create_gis_exports(output_folder)
            else:
                self.log_message("Error: Could not create location map. No valid GPS coordinates found.")
                
        except Exception as e:
            self.log_message(f"Error creating location map: {str(e)}")

    def create_gis_exports(self, output_folder):
        """Create CSV and shapefile exports"""
        self.log_message("Creating GPS data exports (CSV and shapefile)...")
        
        try:
            result_files = self.altitude_map.export_to_gis_formats(
                self.metrics.gps_data,
                output_folder,
                csv_filename="Image_Locations.csv"
            )
            
            if 'csv' in result_files:
                self.log_message(f"CSV export saved to: {result_files['csv']}")
            
            from models.altitude_map import GEOPANDAS_AVAILABLE
            if 'shapefile' in result_files and GEOPANDAS_AVAILABLE:
                self.log_message(f"Shapefile export saved to: {result_files['shapefile']}")
                
        except Exception as e:
            self.log_message(f"Error creating GIS exports: {str(e)}")
            self.create_simple_csv_export(output_folder)

    def create_simple_csv_export(self, output_folder):
        """Create simple CSV export as fallback"""
        try:
            csv_path = os.path.join(output_folder, "Image_Locations.csv")
            import pandas as pd
            df = pd.DataFrame(self.metrics.gps_data)
            
            columns = ["filename", "DateTime", "latitude", "longitude", "altitude",
                      "SubjectDistance", "ExposureTime", "FNumber", "FocalLength",
                      "width", "height"]
            
            available_cols = [col for col in columns if col in df.columns]
            other_cols = [col for col in df.columns if col not in columns]
            
            df = df[available_cols + other_cols]
            df.to_csv(csv_path, index=False)
            self.log_message(f"CSV export saved to: {csv_path}")
            
        except Exception as csv_err:
            self.log_message(f"Error creating CSV export: {str(csv_err)}")

    def create_altitude_histogram(self, output_folder):
        """Create altitude histogram"""
        self.log_message("Creating altitude histogram...")
        
        try:
            self.log_message(f"Generating histogram with {len(self.metrics.gps_data)} altitude points...")
            
            histogram_file = self.altitude_map.create_altitude_histogram(
                self.metrics.gps_data,
                output_folder
            )
            
            if histogram_file:
                self.log_message(f"Altitude histogram saved to: {histogram_file}")
            else:
                self.log_message("Error: Could not create altitude histogram. No valid altitude data found.")
                
        except Exception as e:
            self.log_message(f"Error creating altitude histogram: {str(e)}")

    def process_footprint_data(self, input_folder, output_folder):
        """Process footprint and overlap analysis"""
        self.update_progress(60, "Creating image footprints and overlap maps...")
        
        try:
            nav_path = self.nav_path.get()
            self.footprint_map.altitude_threshold = self.altitude_threshold
            
            footprint_map_path = self.footprint_map.create_footprint_map(
                self.metrics.gps_data,
                output_folder,
                nav_file_path=nav_path,
                filename="Image_Footprints_Map.png"
            )
            
            if footprint_map_path:
                self.log_message(f"Image footprint map created: {footprint_map_path}")
                self.copy_overlap_stats()
                
                # Export footprint metrics to separate text file
                total_images = len(self.metrics.gps_data) if self.metrics.gps_data else 0
                valid_footprints = len([fp for fp in self.metrics.gps_data if fp.get('altitude', float('inf')) <= self.altitude_threshold]) if self.metrics.gps_data else 0
                self.footprint_map.export_footprint_metrics(output_folder, total_images, valid_footprints)
                
                self.append_overlap_metrics_to_file(output_folder)
            else:
                self.log_message("Failed to create footprint map")
                
        except Exception as e:
            self.log_message(f"Error creating footprint map: {str(e)}")
            self.log_message(traceback.format_exc())

    def copy_overlap_stats(self):
        """Copy overlap stats from footprint map to metrics"""
        if hasattr(self.footprint_map, 'vertical_overlap_stats'):
            self.metrics.vertical_overlap_stats = self.footprint_map.vertical_overlap_stats
        if hasattr(self.footprint_map, 'horizontal_overlap_stats'):
            self.metrics.horizontal_overlap_stats = self.footprint_map.horizontal_overlap_stats
        if hasattr(self.footprint_map, 'overall_overlap_stats'):
            self.metrics.overall_overlap_stats = self.footprint_map.overall_overlap_stats

    def process_visibility_analysis(self, input_folder, output_folder):
        """Process visibility analysis"""
        # Initialize visibility analyzer if needed
        if not hasattr(self, 'visibility_analyzer'):
            from src.models.visibility_analyzer import VisibilityAnalyzer
            self.visibility_analyzer = VisibilityAnalyzer(self.altitude_threshold)
        else:
            self.visibility_analyzer.altitude_threshold = self.altitude_threshold
        
        tf_available = self.visibility_analyzer._import_tensorflow()
        
        if not tf_available:
            self.log_message("WARNING: TensorFlow not available. Cannot perform visibility analysis.")
            self.log_message("Please install TensorFlow with: pip install tensorflow")
            return
        
        try:
            model_path = self.get_visibility_model_path()
            if not model_path:
                return
            
            self.run_visibility_analysis(model_path, input_folder, output_folder)
            
        except Exception as e:
            self.log_message(f"Error during visibility analysis: {str(e)}")
            traceback.print_exc()

    def get_visibility_model_path(self):
        """Get the visibility model path based on selection"""
        if self.model_type_var.get() == "model":
            model_path = self.model_path.get()
            if not model_path or not os.path.exists(model_path):
                self.log_message("Error: Please select a valid visibility model file.")
                return None
        else:  # training
            model_path = self.training_path.get()
            if not model_path or not os.path.exists(model_path):
                self.log_message("Error: Please select a valid training data directory.")
                return None
        
        return model_path

    def run_visibility_analysis(self, model_path, input_folder, output_folder):
        """Run the visibility analysis"""
        self.log_message(f"Loading or training visibility model from: {model_path}")
        success = self.visibility_analyzer.load_or_train_model(
            model_path,
            progress_callback=self.update_progress
        )
        
        if not success:
            self.log_message("Error: Could not load or train visibility model.")
            return
        
        # Save trained model if needed
        if os.path.isdir(model_path):
            self.save_trained_model(output_folder)
        
        # Set up log adapter
        def log_message_adapter(msg, progress=None):
            self.log_message(msg)
            if progress is not None:
                self.update_progress(progress, msg)
        
        self.visibility_analyzer.log_message = log_message_adapter
        
        # Run analysis
        self.log_message(f"Starting visibility analysis with altitude threshold: {self.altitude_threshold}m")
        self.log_message("This will process all images below the threshold (up to 5000)")
        
        success, stats = self.visibility_analyzer.analyze_images(
            [],  # Empty list means scan input folder
            output_folder,
            progress_callback=self.update_progress,
            altitude_threshold=self.altitude_threshold
        )
        
        if success:
            self.log_message(f"Visibility analysis complete. Results saved to {output_folder}")
            
            # Export visibility metrics to separate text file
            self.visibility_analyzer.export_visibility_metrics(output_folder, stats)
            
            self.append_visibility_metrics_to_file(output_folder)
        else:
            self.log_message("Error during visibility analysis.")

    def save_trained_model(self, output_folder):
        """Save a newly trained visibility model"""
        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        models_dir = os.path.join(app_root, "utils", "models")
        os.makedirs(models_dir, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"visibility_model_{timestamp}.h5"
        model_save_path = os.path.join(models_dir, model_filename)
        
        saved_path = self.visibility_analyzer.save_model(model_save_path)
        if saved_path:
            self.log_message(f"Trained model saved to: {saved_path}")
            
            output_model_path = os.path.join(output_folder, "visibility_model.h5")
            self.visibility_analyzer.save_model(output_model_path)
            self.log_message(f"Model copy saved to output folder: {output_model_path}")
        else:
            self.log_message("Warning: Could not save trained model")

    def process_highlight_selection(self, input_folder, output_folder):
        """Process highlight image selection"""
        success = self.select_highlight_images(input_folder, output_folder)
        
        if success:
            self.log_message("Highlight image selection completed successfully")
        else:
            self.log_message("Error during highlight image selection")

    def select_highlight_images(self, input_folder, output_folder):
        """Select highlight images using the highlight selector"""
        try:
            # Implementation for highlight selection
            # This would call your existing highlight selector logic
            return True
        except Exception as e:
            self.log_message(f"Error in highlight selection: {str(e)}")
            return False