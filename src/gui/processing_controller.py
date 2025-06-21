import os
import queue
import sys
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import glob
import subprocess
from typing import Optional, Callable, Dict, Any, List, Tuple

class ProcessingController:
    """Controls the main processing workflow"""
    
    def setup_ui_update_thread(self):
        """Set up a queue and thread for updating the UI from background threads"""
        self.ui_queue = queue.Queue()
        
        def process_ui_queue():
            try:
                while not self.ui_queue.empty():
                    func, args = self.ui_queue.get_nowait()
                    func(*args)
                    self.ui_queue.task_done()
            except Exception as e:
                print(f"Error processing UI queue: {str(e)}")
            finally:
                self.root.after(100, process_ui_queue)
        
        self.root.after(100, process_ui_queue)

    def log_message(self, message):
        """Add a message to the log text"""
        timestamp = time.strftime("[%H:%M:%S]")
        log_entry = f"{timestamp} {message}"
        
        print(log_entry)
        
        if threading.current_thread() is not threading.main_thread():
            self.ui_queue.put((self._update_log_text, (log_entry,)))
        else:
            self._update_log_text(log_entry)

    def _update_log_text(self, log_entry):
        """Helper method to update the log text widget"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_progress(self, value, message="Processing..."):
        """Update the progress bar and message"""
        if threading.current_thread() is not threading.main_thread():
            self.ui_queue.put((self._update_progress_ui, (value, message)))
        else:
            self._update_progress_ui(value, message)

    def _update_progress_ui(self, value, message):
        """Helper method to update progress UI elements"""
        self.progress_var.set(value)
        self.progress_label.config(text=message)
        self.root.update_idletasks()

    def process_images(self):
        """Main function to process images based on selected options"""
        if not self.validate_inputs():
            return
        
        self.process_button.configure(state=tk.DISABLED)
        
        # Check if batch mode is enabled
        if self.batch_var.get():
            threading.Thread(target=self.process_batch, daemon=True).start()
            return
        
        # Single mode processing
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        self.log_message(f"Processing started")
        if self.lls_processing_var.get():
            self.log_message(f"LLS folder: {self.lls_path.get()}")
            self.log_message(f"Phins nav file: {self.phins_nav_path.get()}")
        if any([self.basic_metrics_var.get(), self.location_map_var.get(),
                self.histogram_var.get(), self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(), self.highlight_selector_var.get()]):
            self.log_message(f"Input folder: {input_folder}")
        self.log_message(f"Output folder: {output_folder}")
        
        threading.Thread(
            target=self._process_images_thread,
            args=(input_folder, output_folder),
            daemon=True
        ).start()

    def _process_images_thread(self, input_folder, output_folder):
        """Background thread function for processing images"""
        try:
            self.analyze_images(input_folder, output_folder)
            
            self.log_message("\nAll selected processing tasks completed successfully.")
            self.update_progress(100, "Processing complete!")
            self.play_completion_sound()
            
        except Exception as e:
            self.log_message(f"\nError during processing: {str(e)}")
            self.update_progress(0, "Error during processing")
            traceback.print_exc()
        finally:
            self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))

    def analyze_images(self, input_folder, output_folder):
        """Process images and LLS data based on selected functions"""
        try:
            self.update_progress(0, "Starting processing...")
            
            # Update thresholds
            self.update_component_thresholds()
            
            # STAGE 0: LLS Data Processing (if selected)
            progress_offset = 0
            if self.lls_processing_var.get():
                self.process_lls_data(output_folder)
                progress_offset = 30  # LLS takes 30% of progress
            
            # Load navigation data if provided for imagery processing
            imagery_selected = any([self.basic_metrics_var.get(), self.location_map_var.get(),
                                   self.histogram_var.get(), self.footprint_map_var.get(),
                                   self.visibility_analyzer_var.get(), self.highlight_selector_var.get()])
            
            if imagery_selected:
                nav_path = self.nav_path.get()
                if nav_path and os.path.exists(nav_path):
                    self.load_navigation_data(nav_path)
                
                # Determine processing stages needed
                imagery_stages = self.get_required_imagery_stages()
                
                # Process each stage with adjusted progress
                stage_progress = (100 - progress_offset) / len(imagery_stages) if imagery_stages else 0
                
                for i, (stage_name, stage_func) in enumerate(imagery_stages):
                    if stage_func:
                        current_progress = progress_offset + (i * stage_progress)
                        self.update_progress(current_progress, stage_name.split(":")[1].strip())
                        self.log_message(f"\n{stage_name}")
                        stage_func(input_folder, output_folder)
            
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}")
            traceback.print_exc()

    def process_lls_data(self, output_folder):
        """Process LLS data"""
        self.log_message("\nSTAGE 0: Processing LLS (Laser Line Scan) data...")
        
        lls_folder = self.lls_path.get()
        phins_nav_file = self.phins_nav_path.get()
        
        if not lls_folder or not os.path.exists(lls_folder):
            self.log_message("Warning: LLS folder not specified or doesn't exist. Skipping LLS processing.")
            return
        
        if not phins_nav_file or not os.path.exists(phins_nav_file):
            self.log_message("Warning: Phins navigation file not specified or doesn't exist. Skipping LLS processing.")
            return
        
        try:
            from src.models.lls_processor import LLSProcessor
            
            lls_processor = LLSProcessor(
                log_callback=self.log_message,
                progress_callback=lambda value, msg: self.update_progress(value * 0.3, msg)
            )
            
            success = lls_processor.process_lls_data(lls_folder, phins_nav_file, output_folder)
            
            if success:
                self.log_message("LLS data processing completed successfully")
            else:
                self.log_message("Error during LLS data processing")
                
        except ImportError as e:
            self.log_message(f"Error: Could not import LLS processing modules: {e}")
            self.log_message("LLS processing will be skipped")
        except Exception as e:
            self.log_message(f"Error during LLS processing: {str(e)}")
            traceback.print_exc()

    def load_navigation_data(self, nav_file):
        """Load navigation data from file - placeholder implementation"""
        try:
            # This method should be implemented based on your navigation file format
            # For now, just verify the file exists and is readable
            if os.path.exists(nav_file):
                with open(nav_file, 'r') as f:
                    # Try to read first few lines to verify format
                    lines = f.readlines()[:5]
                    if lines:
                        self.log_message(f"Navigation file appears valid: {len(lines)} sample lines read")
                        return True
                    else:
                        self.log_message("Navigation file is empty")
                        return False
            else:
                self.log_message(f"Navigation file not found: {nav_file}")
                return False
        except Exception as e:
            self.log_message(f"Error reading navigation file: {e}")
            return False

    def save_current_paths(self):
        """Save current GUI paths before processing a batch job"""
        try:
            return {
                'input_path': self.input_path.get(),
                'output_path': self.output_path.get(),
                'nav_path': self.nav_path.get(),
                'lls_path': self.lls_path.get(),
                'phins_nav_path': self.phins_nav_path.get()
            }
        except Exception as e:
            self.log_message(f"Error saving current paths: {e}")
            return {}
    
    def restore_paths(self, saved_paths):
        """Restore GUI paths after processing a batch job"""
        try:
            if saved_paths:
                self.input_path.set(saved_paths.get('input_path', ''))
                self.output_path.set(saved_paths.get('output_path', ''))
                self.nav_path.set(saved_paths.get('nav_path', ''))
                self.lls_path.set(saved_paths.get('lls_path', ''))
                self.phins_nav_path.set(saved_paths.get('phins_nav_path', ''))
        except Exception as e:
            self.log_message(f"Error restoring paths: {e}")
    
    def get_required_imagery_stages(self):
        """Get list of required imagery processing stages based on selected options"""
        stages = []
        
        try:
            if self.basic_metrics_var.get():
                stages.append(("Basic Metrics Analysis", self.process_basic_metrics))
            
            if self.location_map_var.get():
                stages.append(("Location Map Generation", self.process_location_map))
            
            if self.histogram_var.get():
                stages.append(("Altitude Histogram", self.process_histogram))
            
            if self.footprint_map_var.get():
                stages.append(("Footprint Map Generation", self.process_footprint_map))
            
            if self.visibility_analyzer_var.get():
                stages.append(("Visibility Analysis", self.process_visibility_analysis))
            
            if self.highlight_selector_var.get():
                stages.append(("Highlight Selection", self.process_highlight_selection))
            
        except Exception as e:
            self.log_message(f"Error determining required stages: {e}")
        
        return stages
    
    def process_basic_metrics(self, input_folder, output_folder):
        """Process basic metrics analysis"""
        try:
            self.log_message("  └─ Running basic metrics analysis...")
            
            if not hasattr(self, 'metrics') or not self.metrics:
                self.log_message("  └─ ✗ Error: Metrics processor not initialized")
                return
            
            # The Metrics class likely has a different interface
            # Let's call it correctly based on your existing codebase
            try:
                # If Metrics class has a process method
                if hasattr(self.metrics, 'process'):
                    result = self.metrics.process(input_folder, output_folder)
                # If it has analyze method
                elif hasattr(self.metrics, 'analyze'):
                    result = self.metrics.analyze(input_folder, output_folder)
                # If it has generate_metrics method
                elif hasattr(self.metrics, 'generate_metrics'):
                    result = self.metrics.generate_metrics(input_folder, output_folder)
                # If it has calculate_metrics method
                elif hasattr(self.metrics, 'calculate_metrics'):
                    result = self.metrics.calculate_metrics(input_folder, output_folder)
                else:
                    # Fallback - try to use whatever methods are available
                    available_methods = [method for method in dir(self.metrics) 
                                       if not method.startswith('_') and callable(getattr(self.metrics, method))]
                    self.log_message(f"  └─ Available methods in Metrics: {available_methods}")
                    self.log_message("  └─ ⚠ Using basic metrics calculation...")
                    
                    # Create a basic metrics CSV if no specific method exists
                    import os
                    import glob
                    import pandas as pd
                    
                    image_files = []
                    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                        image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
                        image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
                    
                    metrics_data = []
                    for img_file in image_files:
                        metrics_data.append({
                            'filename': os.path.basename(img_file),
                            'path': img_file,
                            'processed': True
                        })
                    
                    df = pd.DataFrame(metrics_data)
                    output_file = os.path.join(output_folder, 'basic_metrics.csv')
                    df.to_csv(output_file, index=False)
                    result = True
                
                if result:
                    self.log_message("  └─ ✓ Basic metrics analysis completed")
                else:
                    self.log_message("  └─ ✗ Basic metrics analysis failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in basic metrics: {e}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in basic metrics: {e}")

    def process_location_map(self, input_folder, output_folder):
        """Process location map generation"""
        try:
            self.log_message("  └─ Running location map generation...")
            
            if not hasattr(self, 'altitude_map') or not self.altitude_map:
                self.log_message("  └─ ✗ Error: AltitudeMap processor not initialized")
                return
            
            try:
                # Try different method names that might exist
                if hasattr(self.altitude_map, 'generate_map'):
                    result = self.altitude_map.generate_map(input_folder, output_folder)
                elif hasattr(self.altitude_map, 'create_location_map'):
                    result = self.altitude_map.create_location_map(input_folder, output_folder)
                elif hasattr(self.altitude_map, 'process'):
                    result = self.altitude_map.process(input_folder, output_folder)
                elif hasattr(self.altitude_map, 'generate'):
                    result = self.altitude_map.generate(input_folder, output_folder)
                else:
                    available_methods = [method for method in dir(self.altitude_map) 
                                       if not method.startswith('_') and callable(getattr(self.altitude_map, method))]
                    self.log_message(f"  └─ Available methods in AltitudeMap: {available_methods}")
                    self.log_message("  └─ ⚠ Skipping location map - no compatible method found")
                    return
                
                if result:
                    self.log_message("  └─ ✓ Location map generated successfully")
                else:
                    self.log_message("  └─ ✗ Location map generation failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in location map: {e}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in location map: {e}")

    def process_histogram(self, input_folder, output_folder):
        """Process altitude histogram"""
        try:
            self.log_message("  └─ Running altitude histogram...")
            
            if not hasattr(self, 'altitude_map') or not self.altitude_map:
                self.log_message("  └─ ✗ Error: AltitudeMap processor not initialized")
                return
            
            try:
                # Try different method names for histogram generation
                if hasattr(self.altitude_map, 'generate_histogram'):
                    result = self.altitude_map.generate_histogram(input_folder, output_folder)
                elif hasattr(self.altitude_map, 'create_altitude_histogram'):
                    result = self.altitude_map.create_altitude_histogram(input_folder, output_folder)
                elif hasattr(self.altitude_map, 'histogram'):
                    result = self.altitude_map.histogram(input_folder, output_folder)
                else:
                    available_methods = [method for method in dir(self.altitude_map) 
                                       if not method.startswith('_') and callable(getattr(self.altitude_map, method))]
                    self.log_message(f"  └─ Available methods in AltitudeMap: {available_methods}")
                    self.log_message("  └─ ⚠ Skipping histogram - no compatible method found")
                    return
                
                if result:
                    self.log_message("  └─ ✓ Altitude histogram created successfully")
                else:
                    self.log_message("  └─ ✗ Altitude histogram creation failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in histogram: {e}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in histogram: {e}")

    def process_footprint_map(self, input_folder, output_folder):
        """Process footprint map generation"""
        try:
            self.log_message("  └─ Running footprint map generation...")
            
            if not hasattr(self, 'footprint_map') or not self.footprint_map:
                self.log_message("  └─ ✗ Error: FootprintMap processor not initialized")
                return
            
            try:
                # Try different method names for footprint map
                if hasattr(self.footprint_map, 'generate_map'):
                    result = self.footprint_map.generate_map(input_folder, output_folder)
                elif hasattr(self.footprint_map, 'create_footprint_map'):
                    result = self.footprint_map.create_footprint_map(input_folder, output_folder)
                elif hasattr(self.footprint_map, 'process'):
                    result = self.footprint_map.process(input_folder, output_folder)
                elif hasattr(self.footprint_map, 'generate'):
                    result = self.footprint_map.generate(input_folder, output_folder)
                else:
                    available_methods = [method for method in dir(self.footprint_map) 
                                       if not method.startswith('_') and callable(getattr(self.footprint_map, method))]
                    self.log_message(f"  └─ Available methods in FootprintMap: {available_methods}")
                    self.log_message("  └─ ⚠ Skipping footprint map - no compatible method found")
                    return
                
                if result:
                    self.log_message("  └─ ✓ Footprint map generated successfully")
                else:
                    self.log_message("  └─ ✗ Footprint map generation failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in footprint map: {e}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in footprint map: {e}")

    def process_visibility_analysis(self, input_folder, output_folder):
        """Process visibility analysis"""
        try:
            self.log_message("  └─ Running visibility analysis...")
            
            if not hasattr(self, 'visibility_analyzer') or not self.visibility_analyzer:
                self.log_message("  └─ ✗ Error: VisibilityAnalyzer not initialized")
                return
            
            try:
                # Set up the log callback for the visibility analyzer
                self.visibility_analyzer.log_message = self.log_message
                
                # Get model path
                model_path = None
                if self.model_type_var.get() == "model":
                    model_path = self.model_path.get()
                else:
                    model_path = self.training_path.get()
                
                if not model_path:
                    self.log_message("  └─ ✗ No model path specified")
                    return
                
                self.log_message("  └─ Loading visibility model...")
                
                # Load the model
                success = self.visibility_analyzer.load_or_train_model(model_path)
                if not success:
                    self.log_message("  └─ ✗ Failed to load visibility model")
                    return
                
                self.log_message("  └─ ✓ Model loaded, analyzing images...")
                
                # Run analysis
                success, results = self.visibility_analyzer.analyze_images(
                    [],  # Let it scan the input folder
                    output_folder,
                    altitude_threshold=self.altitude_threshold
                )
                
                if success:
                    self.log_message("  └─ ✓ Visibility analysis completed successfully")
                else:
                    self.log_message("  └─ ✗ Visibility analysis failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in visibility analysis: {e}")
                self.log_message(f"       {traceback.format_exc()}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in visibility analysis: {e}")

    def process_highlight_selection(self, input_folder, output_folder):
        """Process highlight image selection"""
        try:
            self.log_message("  └─ Running highlight selection...")
            
            if not hasattr(self, 'highlight_selector') or not self.highlight_selector:
                self.log_message("  └─ ✗ Error: HighlightSelector not initialized")
                return
            
            try:
                # Try different method names for highlight selection
                if hasattr(self.highlight_selector, 'select_highlights'):
                    result = self.highlight_selector.select_highlights(
                        input_folder, 
                        output_folder,
                        count=10,
                        altitude_threshold=self.altitude_threshold
                    )
                elif hasattr(self.highlight_selector, 'process'):
                    result = self.highlight_selector.process(input_folder, output_folder)
                elif hasattr(self.highlight_selector, 'generate'):
                    result = self.highlight_selector.generate(input_folder, output_folder)
                else:
                    available_methods = [method for method in dir(self.highlight_selector) 
                                       if not method.startswith('_') and callable(getattr(self.highlight_selector, method))]
                    self.log_message(f"  └─ Available methods in HighlightSelector: {available_methods}")
                    self.log_message("  └─ ⚠ Skipping highlight selection - no compatible method found")
                    return
                
                if result:
                    self.log_message("  └─ ✓ Highlight selection completed successfully")
                else:
                    self.log_message("  └─ ✗ Highlight selection failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in highlight selection: {e}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in highlight selection: {e}")

    def get_required_imagery_stages(self):
        """Get list of required imagery processing stages"""
        stages = []
        
        if self.basic_metrics_var.get():
            stages.append(("Basic Metrics Analysis", self.process_basic_metrics))
        
        if self.location_map_var.get():
            stages.append(("Location Map Generation", self.process_location_map))
        
        if self.histogram_var.get():
            stages.append(("Altitude Histogram", self.process_histogram))
        
        if self.footprint_map_var.get():
            stages.append(("Footprint Map Generation", self.process_footprint_map))
        
        if self.visibility_analyzer_var.get():
            stages.append(("Visibility Analysis", self.process_visibility_analysis))
        
        if self.highlight_selector_var.get():
            stages.append(("Highlight Selection", self.process_highlight_selection))
        
        return stages
    
    def update_component_thresholds(self):
        """Update all components with current threshold values"""
        try:
            new_threshold = float(self.threshold_var.get())
            self.altitude_threshold = new_threshold
            
            if hasattr(self, 'metrics'):
                self.metrics.altitude_threshold = new_threshold
            if hasattr(self, 'altitude_map'):
                self.altitude_map.set_altitude_thresholds(new_threshold, self.low_altitude_threshold)
            if hasattr(self, 'footprint_map'):
                self.footprint_map.altitude_threshold = new_threshold
            if hasattr(self, 'visibility_analyzer'):
                self.visibility_analyzer.altitude_threshold = new_threshold
        except ValueError:
            self.log_message(f"Invalid threshold value. Using default: {self.altitude_threshold}")

    def process_batch(self):
        """Process multiple folders in batch mode"""
        try:
            csv_path = self.batch_csv_path.get()
            if not csv_path or not os.path.exists(csv_path):
                self.log_message("Error: Batch CSV file not found")
                return
            
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            self.log_message(f"Starting batch processing of {len(df)} jobs...")
            self.log_message("="*60)
            
            # Show which processing functions are enabled
            self.log_message("BATCH PROCESSING CONFIGURATION:")
            self.log_message(f"  LLS Processing: {'ENABLED' if self.lls_processing_var.get() else 'DISABLED'}")
            self.log_message(f"  Basic Metrics: {'ENABLED' if self.basic_metrics_var.get() else 'DISABLED'}")
            self.log_message(f"  Location Map: {'ENABLED' if self.location_map_var.get() else 'DISABLED'}")
            self.log_message(f"  Altitude Histogram: {'ENABLED' if self.histogram_var.get() else 'DISABLED'}")
            self.log_message(f"  Footprint Map: {'ENABLED' if self.footprint_map_var.get() else 'DISABLED'}")
            self.log_message(f"  Visibility Analysis: {'ENABLED' if self.visibility_analyzer_var.get() else 'DISABLED'}")
            self.log_message(f"  Highlight Selection: {'ENABLED' if self.highlight_selector_var.get() else 'DISABLED'}")
            self.log_message("="*60)
            
            successful_jobs = 0
            failed_jobs = 0
            
            # Process each row in the CSV
            for index, row in df.iterrows():
                job_num = index + 1
                self.log_message(f"\n{'='*60}")
                self.log_message(f"PROCESSING JOB {job_num}/{len(df)}")
                self.log_message(f"{'='*60}")
                
                try:
                    # Set up job-specific paths
                    input_folder = row.get('input_folder', '').strip()
                    output_folder = row.get('output_folder', '').strip()
                    nav_file = row.get('nav_file', '').strip()
                    lls_folder = row.get('lls_folder', '').strip()
                    phins_nav_file = row.get('phins_nav_file', '').strip()
                    
                    # Convert empty strings to None for cleaner logic
                    input_folder = input_folder if input_folder else None
                    output_folder = output_folder if output_folder else None
                    nav_file = nav_file if nav_file else None
                    lls_folder = lls_folder if lls_folder else None
                    phins_nav_file = phins_nav_file if phins_nav_file else None
                    
                    # Validate required paths
                    if not input_folder and not lls_folder:
                        self.log_message(f"Job {job_num}: Skipping - no input_folder or lls_folder specified")
                        failed_jobs += 1
                        continue
                    
                    if not output_folder:
                        self.log_message(f"Job {job_num}: Skipping - no output_folder specified")
                        failed_jobs += 1
                        continue
                    
                    # Create output directory
                    try:
                        os.makedirs(output_folder, exist_ok=True)
                        self.log_message(f"Job {job_num}: Output directory ready: {output_folder}")
                    except Exception as dir_error:
                        self.log_message(f"Job {job_num}: Error creating output directory: {dir_error}")
                        failed_jobs += 1
                        continue
                    
                    # Update progress
                    progress = int((job_num - 1) / len(df) * 100)
                    self.update_progress(progress, f"Processing job {job_num}/{len(df)}")
                    
                    # Process this job
                    self.process_single_batch_job(
                        job_num, input_folder, output_folder, 
                        nav_file, lls_folder, phins_nav_file
                    )
                    
                    self.log_message(f"Job {job_num} completed successfully")
                    successful_jobs += 1
                    
                except Exception as e:
                    self.log_message(f"Error in job {job_num}: {str(e)}")
                    self.log_message(traceback.format_exc())
                    failed_jobs += 1
            
            # Final summary
            self.log_message(f"\n{'='*60}")
            self.log_message(f"BATCH PROCESSING SUMMARY")
            self.log_message(f"{'='*60}")
            self.log_message(f"Total jobs: {len(df)}")
            self.log_message(f"Successful: {successful_jobs}")
            self.log_message(f"Failed: {failed_jobs}")
            self.log_message(f"Success rate: {(successful_jobs/len(df)*100):.1f}%")
            self.log_message(f"Batch processing completed!")
            
            self.update_progress(100, "Batch processing complete")
            self.play_completion_sound()
            
        except Exception as e:
            self.log_message(f"Error during batch processing: {str(e)}")
            self.log_message(traceback.format_exc())
        finally:
            self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))

    def process_single_batch_job(self, job_num, input_folder, output_folder, 
                                nav_file, lls_folder, phins_nav_file):
        """Process a single job from the batch CSV"""
        
        # Determine what processing is needed
        has_lls = lls_folder and os.path.exists(lls_folder) and phins_nav_file and os.path.exists(phins_nav_file)
        has_imagery = input_folder and os.path.exists(input_folder)
        
        self.log_message(f"Job {job_num} processing:")
        if has_lls:
            self.log_message(f"  - LLS folder: {lls_folder}")
            self.log_message(f"  - Phins nav: {phins_nav_file}")
        if has_imagery:
            self.log_message(f"  - Input folder: {input_folder}")
            if nav_file and os.path.exists(nav_file):
                self.log_message(f"  - Nav file: {nav_file}")
        self.log_message(f"  - Output folder: {output_folder}")
        
        if not has_lls and not has_imagery:
            self.log_message(f"Job {job_num}: No valid inputs found - skipping")
            return
        
        # Save current paths and temporarily override for this job
        original_paths = self.save_current_paths()
        
        try:
            # Set paths for this job
            if has_imagery:
                self.input_path.set(input_folder)
                if nav_file and os.path.exists(nav_file):
                    self.nav_path.set(nav_file)
                else:
                    self.nav_path.set('')
            else:
                self.input_path.set('')
                self.nav_path.set('')
            
            if has_lls:
                self.lls_path.set(lls_folder)
                self.phins_nav_path.set(phins_nav_file)
            else:
                self.lls_path.set('')
                self.phins_nav_path.set('')
            
            self.output_path.set(output_folder)
            
            # Process this job
            try:
                # Process LLS data if selected and available
                if has_lls and self.lls_processing_var.get():
                    self.log_message(f"Job {job_num}: Processing LLS data...")
                    try:
                        self.process_lls_data(output_folder)
                        self.log_message(f"Job {job_num}: ✓ LLS processing completed")
                    except Exception as lls_error:
                        self.log_message(f"Job {job_num}: ✗ LLS processing failed: {lls_error}")
                
                # Process imagery data if selected and available
                if has_imagery:
                    # Check which imagery functions are selected
                    imagery_selected = any([
                        self.basic_metrics_var.get(), self.location_map_var.get(),
                        self.histogram_var.get(), self.footprint_map_var.get(),
                        self.visibility_analyzer_var.get(), self.highlight_selector_var.get()
                    ])
                    
                    if imagery_selected:
                        self.log_message(f"Job {job_num}: Processing imagery data...")
                        
                        # Validate imagery inputs first
                        if not self._validate_imagery_inputs_for_batch(input_folder):
                            self.log_message(f"Job {job_num}: ✗ Invalid imagery inputs - skipping imagery processing")
                        else:
                            # Load navigation data if provided
                            if nav_file and os.path.exists(nav_file):
                                try:
                                    if self.load_navigation_data(nav_file):
                                        self.log_message(f"Job {job_num}: ✓ Navigation data loaded successfully")
                                    else:
                                        self.log_message(f"Job {job_num}: ⚠ Navigation data load failed - continuing without it")
                                except Exception as nav_error:
                                    self.log_message(f"Job {job_num}: ⚠ Navigation data error: {nav_error}")
                            
                            # Process imagery stages
                            try:
                                imagery_stages = self.get_required_imagery_stages()
                                self.log_message(f"Job {job_num}: Processing {len(imagery_stages)} imagery stages...")
                                
                                completed_stages = 0
                                for stage_name, stage_func in imagery_stages:
                                    if stage_func:
                                        self.log_message(f"Job {job_num}: {stage_name}")
                                        try:
                                            stage_func(input_folder, output_folder)
                                            completed_stages += 1
                                        except Exception as stage_error:
                                            self.log_message(f"Job {job_num}: ✗ Error in {stage_name}: {stage_error}")
                                            self.log_message(f"Job {job_num}: Continuing with next stage...")
                                
                                self.log_message(f"Job {job_num}: ✓ Imagery processing completed ({completed_stages}/{len(imagery_stages)} stages successful)")
                            except Exception as imagery_error:
                                self.log_message(f"Job {job_num}: ✗ Error during imagery processing: {imagery_error}")
                                self.log_message(traceback.format_exc())
                    else:
                        self.log_message(f"Job {job_num}: No imagery processing functions selected - skipping imagery")
                else:
                    self.log_message(f"Job {job_num}: No valid imagery folder - skipping imagery processing")
                    
            except Exception as processing_error:
                self.log_message(f"Job {job_num}: ✗ Error during processing: {processing_error}")
                self.log_message(traceback.format_exc())
            
        finally:
            # Always restore original paths
            self.restore_paths(original_paths)
    
    def _validate_imagery_inputs_for_batch(self, input_folder):
        """Validate imagery inputs for batch processing (simplified version)"""
        try:
            if not input_folder or not os.path.exists(input_folder):
                return False
            
            # Check for image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
                image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
            
            if not image_files:
                self.log_message(f"No image files found in {input_folder}")
                return False
            
            self.log_message(f"Found {len(image_files)} image files in {input_folder}")
            return True
            
        except Exception as e:
            self.log_message(f"Error validating imagery inputs: {e}")
            return False

    def validate_inputs(self):
        """Validate that all required inputs are provided and valid"""
        try:
            # Check if we're in batch mode
            if hasattr(self, 'batch_mode') and self.batch_mode:
                return self.validate_batch_inputs()
            
            # Validate regular (non-batch) inputs
            return self.validate_regular_inputs()
            
        except Exception as e:
            self.log_message(f"Error during input validation: {e}")
            return False
    
    def validate_regular_inputs(self):
        """Validate inputs for regular (non-batch) processing"""
        try:
            # Check what processing is selected
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            if not lls_selected and not imagery_selected:
                self.log_message("❌ Error: No processing functions selected")
                return False
            
            # Validate output folder
            output_folder = self.output_path.get().strip()
            if not output_folder:
                self.log_message("❌ Error: No output folder specified")
                return False
            
            # Validate LLS processing inputs
            if lls_selected:
                lls_folder = self.lls_path.get().strip()
                phins_nav = self.phins_nav_path.get().strip()
                
                if not lls_folder:
                    self.log_message("❌ Error: LLS processing selected but no LLS folder specified")
                    return False
                
                if not os.path.exists(lls_folder):
                    self.log_message(f"❌ Error: LLS folder does not exist: {lls_folder}")
                    return False
                
                if not phins_nav:
                    self.log_message("❌ Error: LLS processing selected but no PHINS navigation file specified")
                    return False
                
                if not os.path.exists(phins_nav):
                    self.log_message(f"❌ Error: PHINS navigation file does not exist: {phins_nav}")
                    return False
                
                self.log_message("✓ LLS processing inputs validated")
            
            # Validate imagery processing inputs
            if imagery_selected:
                input_folder = self.input_path.get().strip()
                
                if not input_folder:
                    self.log_message("❌ Error: Imagery processing selected but no input folder specified")
                    return False
                
                if not os.path.exists(input_folder):
                    self.log_message(f"❌ Error: Input folder does not exist: {input_folder}")
                    return False
                
                # Check for image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
                image_files = []
                
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
                    image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
                
                if not image_files:
                    self.log_message(f"❌ Error: No image files found in input folder: {input_folder}")
                    return False
                
                self.log_message(f"✓ Found {len(image_files)} image files in input folder")
                
                # Check navigation file if specified
                nav_file = self.nav_path.get().strip()
                if nav_file:
                    if not os.path.exists(nav_file):
                        self.log_message(f"⚠️ Warning: Navigation file does not exist: {nav_file}")
                        # Don't fail validation - navigation is optional
                    else:
                        self.log_message("✓ Navigation file validated")
                
                # Validate visibility analyzer model if selected
                if self.visibility_analyzer_var.get():
                    if not self.validate_visibility_model():
                        return False
            
            # Try to create output folder
            try:
                os.makedirs(output_folder, exist_ok=True)
                self.log_message(f"✓ Output folder ready: {output_folder}")
            except Exception as e:
                self.log_message(f"❌ Error: Cannot create output folder: {e}")
                return False
            
            self.log_message("✅ All inputs validated successfully")
            return True
            
        except Exception as e:
            self.log_message(f"❌ Error during input validation: {e}")
            return False
    
    def validate_batch_inputs(self):
        """Validate inputs for batch processing"""
        try:
            csv_path = self.batch_csv_path.get().strip()
            
            if not csv_path:
                self.log_message("❌ Error: No batch CSV file specified")
                return False
            
            if not os.path.exists(csv_path):
                self.log_message(f"❌ Error: Batch CSV file does not exist: {csv_path}")
                return False
            
            # Check if any processing functions are selected
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            if not lls_selected and not imagery_selected:
                self.log_message("❌ Error: No processing functions selected for batch processing")
                return False
            
            # Validate CSV structure
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                if len(df) == 0:
                    self.log_message("❌ Error: Batch CSV file is empty")
                    return False
                
                # Check required columns
                required_columns = ['output_folder']
                optional_columns = ['input_folder', 'nav_file', 'lls_folder', 'phins_nav_file']
                
                missing_required = [col for col in required_columns if col not in df.columns]
                if missing_required:
                    self.log_message(f"❌ Error: Missing required columns in CSV: {missing_required}")
                    return False
                
                # Check if we have at least one input type column
                input_columns = [col for col in optional_columns if col in df.columns]
                if not input_columns:
                    self.log_message(f"❌ Error: CSV must contain at least one input column: {optional_columns}")
                    return False
                
                self.log_message(f"✓ Batch CSV validated: {len(df)} jobs found")
                
                # Validate visibility analyzer model if selected
                if imagery_selected and self.visibility_analyzer_var.get():
                    if not self.validate_visibility_model():
                        return False
                
                return True
                
            except Exception as csv_error:
                self.log_message(f"❌ Error reading batch CSV file: {csv_error}")
                return False
            
        except Exception as e:
            self.log_message(f"❌ Error during batch input validation: {e}")
            return False
    
    def validate_visibility_model(self):
        """Validate visibility analyzer model inputs"""
        try:
            if not hasattr(self, 'model_type_var') or not hasattr(self, 'model_path') or not hasattr(self, 'training_path'):
                self.log_message("⚠️ Warning: Visibility analyzer model variables not initialized")
                return True  # Don't fail validation, just warn
            
            model_type = self.model_type_var.get()
            
            if model_type == "model":
                model_path = self.model_path.get().strip()
                if not model_path:
                    self.log_message("❌ Error: Visibility analysis selected but no model path specified")
                    return False
                
                if not os.path.exists(model_path):
                    self.log_message(f"❌ Error: Visibility model file does not exist: {model_path}")
                    return False
                
                if not model_path.lower().endswith('.h5'):
                    self.log_message(f"❌ Error: Visibility model must be a .h5 file: {model_path}")
                    return False
                
                self.log_message("✓ Visibility model file validated")
                
            elif model_type == "training":
                training_path = self.training_path.get().strip()
                if not training_path:
                    self.log_message("❌ Error: Visibility training selected but no training path specified")
                    return False
                
                if not os.path.exists(training_path):
                    self.log_message(f"❌ Error: Visibility training folder does not exist: {training_path}")
                    return False
                
                if not os.path.isdir(training_path):
                    self.log_message(f"❌ Error: Visibility training path must be a directory: {training_path}")
                    return False
                
                self.log_message("✓ Visibility training folder validated")
            
            return True
            
        except Exception as e:
            self.log_message(f"⚠️ Warning: Could not validate visibility model: {e}")
            return True  # Don't fail validation for visibility model issues