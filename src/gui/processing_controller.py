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
                       
            # Check what processing is selected
            nav_selected = self.nav_processing_var.get()
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            # PhinsData manager has been deprecated and removed
            # All modules now use CSV-based processing for consistency
            
            # Process Navigation data first if selected
            if nav_selected:
                self.log_message("Processing Navigation data...")
                try:
                    self.process_navigation_data(output_folder)
                    self.log_message("✓ Navigation processing completed")
                except Exception as nav_error:
                    self.log_message(f"✗ Navigation processing failed: {nav_error}")
                    self.log_message(f"Traceback: {traceback.format_exc()}")
            
            # Process LLS data if selected
            if lls_selected:
                self.log_message("Processing LLS data...")
                try:
                    self.process_lls_data(output_folder)
                    self.log_message("✓ LLS processing completed")
                except Exception as lls_error:
                    self.log_message(f"✗ LLS processing failed: {lls_error}")
                    self.log_message(f"Traceback: {traceback.format_exc()}")
            
            # Process imagery data if selected
            if imagery_selected:
                self.log_message("Processing Imagery data...")
                self.log_message("       ⚠ Note: All modules now use CSV-based processing for consistency")
                
                # Load navigation data from dive nav text file ONLY for imagery
                nav_file = None
                # Only use nav_path for imagery processing (Dive Nav text file)
                if hasattr(self, 'nav_path') and self.nav_path.get():
                    file_path = self.nav_path.get()
                    if file_path and os.path.exists(file_path):
                        nav_file = file_path
                        self.log_message(f"       Navigation source for imagery: {os.path.basename(nav_file)} (Dive Nav text file)")
                    else:
                        self.log_message(f"       ⚠ Dive Nav text file not found: {file_path}")
                else:
                    self.log_message("       ⚠ No Dive Nav text file specified for imagery processing")
                
                if nav_file:
                    try:
                        if self.load_navigation_data_for_imagery_only(nav_file):
                            self.log_message("✓ Navigation data loaded successfully from text file for imagery processing")
                        else:
                            self.log_message("⚠ Navigation data load failed - continuing without it")
                    except Exception as nav_error:
                        self.log_message(f"⚠ Navigation data error: {nav_error}")
                else:
                    self.log_message("⚠ No navigation file specified - continuing without navigation data")
                
                # IMPORTANT: Process basic metrics FIRST to populate GPS data
                try:
                    self.log_message("Extracting image metadata...")
                    
                    # Always extract GPS data for other stages
                    extract_gps = True
                    
                    # Create a progress callback that reports to the main log
                    def metadata_progress(progress_pct, message):
                        # Update progress every 10% or on important messages
                        if progress_pct % 10 == 0 or "GPS data from" in message or "files" in message:
                            self.log_message(f"[{progress_pct:.0f}%] {message}")
                    
                    processed_files, results = self.metrics.analyze_directory(
                        input_folder,
                        progress_callback=metadata_progress,
                        extract_gps=extract_gps
                    )
                    
                    self.log_message(f"✓ Processed {processed_files} files, extracted GPS from {len(self.metrics.gps_data)} images")
                    
                    # Image_Metrics.csv already contains all GPS and EXIF data
                    self.log_message("✓ Image_Metrics.csv contains all required GPS and EXIF data")
                    
                except Exception as metadata_error:
                    self.log_message(f"✗ Error extracting metadata: {metadata_error}")
                    self.log_message("Cannot proceed without image metadata")
                    return
                
                # Get list of processing stages
                processing_stages = []
                
                if self.basic_metrics_var.get():
                    processing_stages.append(("Basic Metrics Analysis", self.process_basic_metrics))
                
                if self.location_map_var.get():
                    processing_stages.append(("Location Map Generation", self.process_location_map))
                
                if self.histogram_var.get():
                    processing_stages.append(("Altitude Histogram", self.process_histogram))
                
                if self.footprint_map_var.get():
                    processing_stages.append(("Footprint Map Generation", self.process_footprint_map))
                
                if self.visibility_analyzer_var.get():
                    processing_stages.append(("Visibility Analysis", self.process_visibility_analysis))
                
                if self.highlight_selector_var.get():
                    processing_stages.append(("Highlight Selection", self.process_highlight_selection))
                
                if not processing_stages:
                    self.log_message("No imagery processing functions selected")
                else:
                    self.log_message(f"Processing {len(processing_stages)} imagery stages...")
                    
                    # STEP 1: Create/update the master Image_Metrics.csv as the first step
                    self.log_message("STEP 1: Creating/updating master Image_Metrics.csv...")
                    try:
                        # Use the PhinsData file path for navigation integration
                        nav_file = None
                        for var_name in ['nav_path', 'phins_nav_path', 'nav_file_path']:
                            if hasattr(self, var_name):
                                file_path = getattr(self, var_name).get()
                                if file_path and os.path.exists(file_path):
                                    nav_file = file_path
                                    break
                        
                        if hasattr(self.metrics, 'create_image_metrics_csv_parallel'):
                            csv_path = self.metrics.create_image_metrics_csv_parallel(
                                input_folder, 
                                output_folder, 
                                nav_file, 
                                progress_callback=lambda p, msg="Creating master CSV...": self.update_progress(p, msg)
                            )
                        else:
                            # Fallback to original method
                            csv_path = self.metrics.create_image_metrics_csv(
                                input_folder, 
                                output_folder, 
                                nav_file, 
                                progress_callback=lambda p, msg="Creating master CSV...": self.update_progress(p, msg)
                            )
                        
                        if csv_path:
                            self.log_message(f"✓ Created Image_Metrics.csv: {os.path.basename(csv_path)}")
                        else:
                            self.log_message("⚠ Failed to create Image_Metrics.csv")
                        
                    except Exception as e:
                        self.log_message(f"⚠ Error creating Image_Metrics.csv: {e}")
                        self.log_message("Processing will continue without master CSV")
                    
                    # STEP 2: Process each stage
                    self.log_message("STEP 2: Processing individual analysis stages...")
                    completed_stages = 0
                    total_stages = len(processing_stages)
                    
                    for stage_idx, (stage_name, stage_func) in enumerate(processing_stages):
                        try:
                            # Calculate progress
                            base_progress = 40 if lls_selected else 20  # Account for LLS processing
                            current_progress = int(base_progress + (stage_idx / total_stages * 50))  # 40-90% or 20-70% for stages
                            
                            # Update progress with stage name
                            progress_text = stage_name
                            self.update_progress(current_progress, progress_text)
                            
                            self.log_message(f"STAGE {stage_idx + 1}/{total_stages}: {stage_name}")
                            
                            # Execute the stage
                            stage_func(input_folder, output_folder)
                            completed_stages += 1
                            
                            self.log_message(f"✓ {stage_name} completed")
                            
                        except Exception as stage_error:
                            self.log_message(f"✗ Error in {stage_name}: {stage_error}")
                            self.log_message(f"Traceback: {traceback.format_exc()}")
                            self.log_message(f"Continuing with next stage...")
                    
                    # Final summary for imagery processing
                    self.log_message(f"\n{'='*60}")
                    self.log_message(f"IMAGERY PROCESSING SUMMARY")
                    self.log_message(f"{'='*60}")
                    self.log_message(f"Total stages: {total_stages}")
                    self.log_message(f"Completed successfully: {completed_stages}")
                    self.log_message(f"Failed: {total_stages - completed_stages}")
                    self.log_message(f"Success rate: {(completed_stages/total_stages*100):.1f}%")
                    
                    if completed_stages == total_stages:
                        self.log_message("✓ All imagery processing completed successfully!")
                    else:
                        self.log_message(f"⚠ Imagery processing completed with {total_stages - completed_stages} errors")
            
            # Final overall summary
            total_processes = (1 if nav_selected else 0) + (1 if lls_selected else 0) + (1 if imagery_selected else 0)
            self.log_message(f"\n{'='*60}")
            self.log_message(f"OVERALL PROCESSING SUMMARY")
            self.log_message(f"{'='*60}")
            
            if nav_selected:
                self.log_message("Navigation Processing: Completed")
            if lls_selected:
                self.log_message("LLS Processing: Completed")
            if imagery_selected:
                self.log_message("Imagery Processing: Completed")
            
            if total_processes > 0:
                self.log_message("✓ All processing completed successfully!")
                self.update_progress(100, "All processing completed")
            else:
                self.log_message("⚠ No processing was performed")
                self.update_progress(100, "No processing performed")
            
            # Play completion sound
            self.play_completion_sound()
            
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.update_progress(0, "Error during processing")
        finally:
            # Re-enable the process button
            if hasattr(self, 'process_button'):
                self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))

    def process_navigation_data(self, output_folder):
        """Process navigation data for plotting using nav_plotter.py"""
        self.log_message("STAGE 1: Processing Navigation data for plotting...")
        
        # Get navigation files
        nav_file = self.nav_plot_file_path.get()  # NAV_STATE.txt file
        phins_file = self.phins_ins_path.get()    # PHINS INS file (optional, for navigation plotting)
        
        if not nav_file or not os.path.exists(nav_file):
            self.log_message("⚠ Navigation file not specified or doesn't exist. Skipping navigation plotting.")
            self.log_message("   Please select a NAV_STATE.txt file for navigation plotting")
            return
        
        self.log_message(f"       Using navigation file: {os.path.basename(nav_file)}")
        if phins_file and os.path.exists(phins_file):
            self.log_message(f"       Using PHINS file: {os.path.basename(phins_file)}")
        
        try:
            from src.models.nav_plotter import NavPlotter
            
            # Create nav plotter instance
            nav_plotter = NavPlotter(log_callback=self.log_message)
            
            # Process navigation data
            success = nav_plotter.process_navigation_data(
                nav_file=nav_file,
                output_folder=output_folder,
                phins_file=phins_file if phins_file and os.path.exists(phins_file) else None
            )
            
            if success:
                self.log_message("✓ Navigation plotting completed successfully")
                self.update_progress(20, "Navigation plotting completed")
            else:
                self.log_message("✗ Navigation plotting failed")
                
        except ImportError as e:
            self.log_message(f"Error: Could not import navigation plotting modules: {e}")
            self.log_message("Navigation plotting will be skipped")
        except Exception as e:
            self.log_message(f"Error during navigation plotting: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")

    def process_lls_data(self, output_folder):
        """Process LLS data using navigation file"""
        self.log_message("STAGE 2: Processing LLS (Laser Line Scan) data...")
        
        lls_folder = self.lls_path.get()
        
        # Use the phins navigation file path for LLS processing 
        nav_file = self.phins_nav_path.get()
        
        if not lls_folder or not os.path.exists(lls_folder):
            self.log_message("Warning: LLS folder not specified or doesn't exist. Skipping LLS processing.")
            return
        
        if not nav_file or not os.path.exists(nav_file):
            self.log_message("Warning: Navigation file not specified or doesn't exist. Skipping LLS processing.")
            return
        
        self.log_message(f"       Using navigation file: {os.path.basename(nav_file)}")
        
        try:
            from src.models.lls_processor import LLSProcessor
            
            # Calculate progress offset for LLS (use 0-30% for LLS processing)
            progress_offset = 0
            progress_scale = 30
            
            lls_processor = LLSProcessor(
                log_callback=self.log_message,
                progress_callback=lambda value, msg: self.update_progress(
                    progress_offset + (value * progress_scale / 100), msg
                )
            )
            
            success = lls_processor.process_lls_data(lls_folder, nav_file, output_folder)
            
            if success:
                self.log_message("✓ LLS data processing completed successfully")
                self.update_progress(30, "LLS processing completed")
            else:
                self.log_message("✗ Error during LLS data processing")
                
        except ImportError as e:
            self.log_message(f"Error: Could not import LLS processing modules: {e}")
            self.log_message("LLS processing will be skipped")
        except Exception as e:
            self.log_message(f"Error during LLS processing: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")

    def load_navigation_data(self, nav_file):
        """Load navigation data for altitude information (general method, may contain PhinsData influence)"""
        try:
            if not hasattr(self, 'metrics') or not self.metrics:
                return False
                
            success = self.metrics.load_nav_data(nav_file)
            if success:
                self.log_message(f"       ✓ Navigation data loaded for altitude extraction")
                        
                return True
            else:
                self.log_message(f"       ⚠ Failed to load navigation data")
                return False
                
        except Exception as e:
            self.log_message(f"       ⚠ Navigation data error: {e}")
            return False

    def save_current_paths(self):
        """Save current GUI paths before processing a batch job"""
        try:
            return {
                'input_path': self.input_path.get(),
                'output_path': self.output_path.get(),
                'nav_path': self.nav_path.get(),
                'lls_path': self.lls_path.get(),
                'phins_nav_path': self.phins_nav_path.get(),
                'nav_plot_file_path': self.nav_plot_file_path.get(),
                'phins_ins_path': self.phins_ins_path.get()
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
                nav_plot_path = saved_paths.get('nav_plot_file_path', '')
                self.nav_plot_file_path.set(nav_plot_path)
                self.nav_state_file_path.set(nav_plot_path)  # Keep synchronized
                self.phins_ins_path.set(saved_paths.get('phins_ins_path', ''))
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
            
            try:
                # Call the correct method that actually processes the directory
                extract_gps = any([
                    self.location_map_var.get(), 
                    self.histogram_var.get(),
                    self.footprint_map_var.get(), 
                    self.visibility_analyzer_var.get()
                ])
                
                processed_files, results = self.metrics.analyze_directory(
                    input_folder,
                    progress_callback=None,  # Skip progress for batch processing
                    extract_gps=extract_gps
                )
                
                # Log the summary results
                for line in results:
                    self.log_message(f"       {line}")
                
                # Save results to file
                metrics_file = os.path.join(output_folder, "Image_Metrics.txt")
                with open(metrics_file, "w") as f:
                    f.write("\n".join(results))
                
                self.log_message(f"  └─ ✓ Basic metrics analysis completed - {processed_files} files processed")
                self.log_message(f"       Results saved to: {metrics_file}")
                
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
                
            if not hasattr(self, 'metrics') or not self.metrics or not self.metrics.gps_data:
                self.log_message("  └─ ✗ Error: No GPS data available for location map")
                return
            
            try:
                # Call the correct method with GPS data from metrics
                map_file = self.altitude_map.create_location_map(
                    self.metrics.gps_data,
                    output_folder,
                    metrics=self.metrics
                )
                
                if map_file and os.path.exists(map_file):
                    self.log_message(f"  └─ ✓ Location map created: {os.path.basename(map_file)}")
                    
                    # Also create GIS exports
                    try:
                        result_files = self.altitude_map.export_to_gis_formats(
                            self.metrics.gps_data,
                            output_folder
                        )
                        
                        if 'csv' in result_files:
                            self.log_message(f"       CSV export: {os.path.basename(result_files['csv'])}")
                        
                        if 'shapefile' in result_files:
                            self.log_message(f"       Shapefile export: {os.path.basename(result_files['shapefile'])}")
                            
                    except Exception as export_error:
                        self.log_message(f"       Warning: GIS export failed: {export_error}")
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
                
            if not hasattr(self, 'metrics') or not self.metrics or not self.metrics.gps_data:
                self.log_message("  └─ ✗ Error: No GPS data available for histogram")
                return
            
            try:
                # Call the correct method with GPS data from metrics
                histogram_file = self.altitude_map.create_altitude_histogram(
                    self.metrics.gps_data,
                    output_folder
                )
                
                if histogram_file and os.path.exists(histogram_file):
                    self.log_message(f"  └─ ✓ Altitude histogram created: {os.path.basename(histogram_file)}")
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
                
            if not hasattr(self, 'metrics') or not self.metrics or not self.metrics.gps_data:
                self.log_message("  └─ ✗ Error: No GPS data available for footprint map")
                return
            
            try:
                # Set altitude threshold
                self.footprint_map.altitude_threshold = self.altitude_threshold
                
                # First try to use the CSV data if available
                csv_path = os.path.join(output_folder, "Image_Metrics.csv")
                footprint_file = None
                
                if os.path.exists(csv_path):
                    self.log_message("  └─ Using Image_Metrics.csv for footprint analysis...")
                    # Use the new CSV-based method that includes heading data
                    footprint_file = self.footprint_map.create_footprint_map_from_csv(
                        csv_path,
                        output_folder,
                        filename="Image_Footprints_Map.png"
                    )
                else:
                    self.log_message("  └─ Using legacy GPS data for footprint analysis...")
                    # Get navigation file path if available
                    nav_file_path = None
                    if hasattr(self, 'nav_path') and self.nav_path.get():
                        nav_file_path = self.nav_path.get()
                    
                    # Call the legacy method with GPS data from metrics
                    footprint_file = self.footprint_map.create_footprint_map(
                        self.metrics.gps_data,
                        output_folder,
                        nav_file_path=nav_file_path,
                        filename="Image_Footprints_Map.png"
                    )
                
                if footprint_file and os.path.exists(footprint_file):
                    self.log_message(f"  └─ ✓ Footprint map created: {os.path.basename(footprint_file)}")
                    
                    # Copy overlap statistics if available
                    if hasattr(self.footprint_map, 'vertical_overlap_stats'):
                        self.metrics.vertical_overlap_stats = self.footprint_map.vertical_overlap_stats
                    if hasattr(self.footprint_map, 'horizontal_overlap_stats'):
                        self.metrics.horizontal_overlap_stats = self.footprint_map.horizontal_overlap_stats
                    if hasattr(self.footprint_map, 'overall_overlap_stats'):
                        self.metrics.overall_overlap_stats = self.footprint_map.overall_overlap_stats
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
            
            # Verify input folder exists and has images
            if not input_folder or not os.path.exists(input_folder):
                self.log_message(f"  └─ ✗ Error: Input folder does not exist: {input_folder}")
                return
            
            # Count images in input folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            image_count = 0
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_count += 1
            
            if image_count == 0:
                self.log_message(f"  └─ ✗ Error: No images found in input folder: {input_folder}")
                return
            
            self.log_message(f"       Found {image_count} images in input folder")
            
            try:
                # Get model path
                model_path = None
                if hasattr(self, 'model_type_var') and self.model_type_var.get() == "model":
                    if hasattr(self, 'model_path'):
                        model_path = self.model_path.get()
                else:
                    if hasattr(self, 'training_path'):
                        model_path = self.training_path.get()
                
                if not model_path:
                    # Use default model path
                    default_model = "v_a_pre-trained_models/visibility_model_20250402.h5"
                    if os.path.exists(default_model):
                        model_path = default_model
                        self.log_message(f"       Using default model: {default_model}")
                    else:
                        self.log_message("  └─ ✗ No model path specified and default model not found")
                        return
                
                self.log_message("       Loading visibility model...")
                
                # Load the model
                success = self.visibility_analyzer.load_or_train_model(model_path)
                if not success:
                    self.log_message("  └─ ✗ Failed to load visibility model")
                    return
                
                self.log_message("       ✓ Model loaded, analyzing images...")
                
                # Create the master CSV path
                master_csv = os.path.join(output_folder, "Image_Metrics.csv")
                if not os.path.exists(master_csv):
                    self.log_message("  └─ ⚠ Image_Metrics.csv not found, visibility analysis requires existing CSV")
                    return
                
                # Run analysis using the CSV method to update the master CSV
                success = self.visibility_analyzer.analyze_images_from_csv(
                    master_csv,    # Path to master CSV file
                    output_folder  # Output folder for any additional files
                )
                
                if success:
                    self.log_message("  └─ ✓ Visibility analysis completed successfully")
                    self.log_message("       Results updated in Image_Metrics.csv")
                else:
                    self.log_message("  └─ ✗ Visibility analysis failed")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in visibility analysis: {e}")
                self.log_message(f"       Traceback: {traceback.format_exc()}")
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in visibility analysis: {e}")

    def process_highlight_selection(self, input_folder, output_folder):
        """Process highlight image selection"""
        try:
            self.log_message("  └─ Running highlight selection...")
            
            if not hasattr(self, 'highlight_selector') or not self.highlight_selector:
                self.log_message("  └─ ✗ Error: HighlightSelector not initialized")
                return
            
            # Verify input folder exists and has images
            if not input_folder or not os.path.exists(input_folder):
                self.log_message(f"  └─ ✗ Error: Input folder does not exist: {input_folder}")
                return
            
            # Count images in input folder to verify we have something to work with
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            image_count = 0
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_count += 1
            
            self.log_message(f"       Found {image_count} images in input folder")
            
            if image_count == 0:
                self.log_message(f"  └─ ✗ Error: No images found in input folder")
                return
            
            try:
                # The highlight selector will automatically check for visibility results
                # in the output folder, so we don't need to pass them explicitly
                # This maintains the dual functionality you want
                
                self.log_message(f"       Processing images from master CSV...")
                
                # Create the master CSV path
                master_csv = os.path.join(output_folder, "Image_Metrics.csv")
                if not os.path.exists(master_csv):
                    self.log_message("  └─ ⚠ Image_Metrics.csv not found, highlight selection requires existing CSV")
                    return
                
                # Call the CSV-based highlight selector to update the master CSV
                highlight_paths = self.highlight_selector.select_highlights_from_csv(
                    master_csv,       # Path to master CSV file
                    output_folder,    # Output folder for highlight images
                    count=10,         # Number of highlights to select
                    progress_callback=None,  # Skip progress for batch processing
                    altitude_threshold=self.altitude_threshold,
                    min_altitude_threshold=2.0
                )
                
                if highlight_paths and len(highlight_paths) > 0:
                    self.log_message(f"  └─ ✓ Selected {len(highlight_paths)} highlight images")
                    self.log_message(f"       Highlights saved to: highlight_images/")
                    
                    # Log what mode was used based on the selector's internal logic
                    vis_csv_path = os.path.join(output_folder, "Image_Visibility_Results.csv")
                    if os.path.exists(vis_csv_path):
                        self.log_message(f"       Used visibility analysis + image metrics mode")
                    else:
                        self.log_message(f"       Used image metrics only mode")
                else:
                    self.log_message("  └─ ✗ No highlight images were selected")
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in highlight selection: {e}")
                self.log_message(f"       Traceback: {traceback.format_exc()}")
                
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
            self.log_message(f"  Navigation Processing: {'ENABLED' if self.nav_processing_var.get() else 'DISABLED'}")
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
                    # Set up job-specific paths using standardized column names
                    input_folder = str(row.get('Image_Input', '')).strip() if pd.notna(row.get('Image_Input', '')) else ''
                    output_folder = str(row.get('Output_folder', '')).strip() if pd.notna(row.get('Output_folder', '')) else ''
                    nav_file = str(row.get('Dive_Nav_file', '')).strip() if pd.notna(row.get('Dive_Nav_file', '')) else ''
                    lls_folder = str(row.get('LLS_Input', '')).strip() if pd.notna(row.get('LLS_Input', '')) else ''
                    phins_nav_file = str(row.get('PhinsData_Bin_file', '')).strip() if pd.notna(row.get('PhinsData_Bin_file', '')) else ''
                    phins_data_nav_file = str(row.get('PhinsData_Nav_file', '')).strip() if pd.notna(row.get('PhinsData_Nav_file', '')) else ''
                    
                    # Navigation module files
                    nav_state_file = str(row.get('NAV_STATE_file', '')).strip() if pd.notna(row.get('NAV_STATE_file', '')) else ''
                    phins_ins_file = str(row.get('PHINS_INS_file', '')).strip() if pd.notna(row.get('PHINS_INS_file', '')) else ''
                    
                    # Convert empty strings to None for cleaner logic
                    input_folder = input_folder if input_folder else None
                    output_folder = output_folder if output_folder else None
                    nav_file = nav_file if nav_file else None
                    lls_folder = lls_folder if lls_folder else None
                    phins_nav_file = phins_nav_file if phins_nav_file else None
                    phins_data_nav_file = phins_data_nav_file if phins_data_nav_file else None
                    nav_state_file = nav_state_file if nav_state_file else None
                    phins_ins_file = phins_ins_file if phins_ins_file else None
                    
                    # Validate required paths - only output is always required
                    if not output_folder:
                        self.log_message(f"Job {job_num}: Skipping - no Output_folder specified")
                        failed_jobs += 1
                        continue
                    
                    # Check if we have inputs for at least one processing module
                    has_nav_module = nav_state_file  # Navigation module needs NAV_STATE_file
                    has_image_module = input_folder
                    has_lls_module = lls_folder and phins_nav_file
                    
                    if not (has_nav_module or has_image_module or has_lls_module):
                        self.log_message(f"Job {job_num}: Skipping - no valid processing module inputs specified")
                        self.log_message(f"  Navigation module needs: NAV_STATE_file")
                        self.log_message(f"  Image module needs: Image_Input")
                        self.log_message(f"  LLS module needs: LLS_Input and PhinsData_Bin_file")
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
                        nav_file, lls_folder, phins_nav_file, phins_data_nav_file,
                        nav_state_file, phins_ins_file
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
                                nav_file, lls_folder, phins_nav_file, phins_data_nav_file,
                                nav_state_file, phins_ins_file):
        """Process a single job from the batch CSV - mirrors single dive processing"""
        
        # Determine what processing is needed
        nav_selected = self.nav_processing_var.get()
        lls_selected = self.lls_processing_var.get()
        imagery_selected = any([
            self.basic_metrics_var.get(),
            self.location_map_var.get(),
            self.histogram_var.get(),
            self.footprint_map_var.get(),
            self.visibility_analyzer_var.get(),
            self.highlight_selector_var.get()
        ])
        
        self.log_message(f"Job {job_num} processing:")
        if nav_selected:
            self.log_message(f"  - Navigation processing: ENABLED")
            if nav_state_file:
                self.log_message(f"  - Nav state file: {nav_state_file}")
            if phins_ins_file:
                self.log_message(f"  - PHINS INS file: {phins_ins_file}")
        if lls_selected:
            self.log_message(f"  - LLS folder: {lls_folder}")
            self.log_message(f"  - Phins nav: {phins_nav_file}")
        if imagery_selected:
            self.log_message(f"  - Input folder: {input_folder}")
            if nav_file and os.path.exists(nav_file):
                self.log_message(f"  - Nav file: {nav_file}")
        self.log_message(f"  - Output folder: {output_folder}")
        
        if not nav_selected and not lls_selected and not imagery_selected:
            self.log_message(f"Job {job_num}: No processing functions selected - skipping")
            return
        
        # Save current paths and temporarily override for this job
        original_paths = self.save_current_paths()
        
        try:
            # Set paths for this job
            if imagery_selected:
                self.input_path.set(input_folder if input_folder else '')
                self.nav_path.set(nav_file if nav_file and os.path.exists(nav_file) else '')
            else:
                self.input_path.set('')
                self.nav_path.set('')
            
            if lls_selected:
                self.lls_path.set(lls_folder if lls_folder else '')
                self.phins_nav_path.set(phins_nav_file if phins_nav_file else '')
            else:
                self.lls_path.set('')
                self.phins_nav_path.set('')
                
            if nav_selected:
                self.nav_plot_file_path.set(nav_state_file if nav_state_file and os.path.exists(nav_state_file) else '')
                # Set PHINS INS file path for navigation processing (separate from LLS)
                self.phins_ins_path.set(phins_ins_file if phins_ins_file and os.path.exists(phins_ins_file) else '')
            else:
                self.nav_plot_file_path.set('')
                self.phins_ins_path.set('')
                
            self.output_path.set(output_folder)
            
            # Process this job using the same structure as single processing
            try:
                # Process Navigation data first if selected
                if nav_selected:
                    self.log_message(f"Job {job_num}: Processing Navigation data...")
                    try:
                        self.process_navigation_data(output_folder)
                        self.log_message(f"Job {job_num}: ✓ Navigation processing completed")
                    except Exception as nav_error:
                        self.log_message(f"Job {job_num}: ✗ Navigation processing failed: {nav_error}")
                        self.log_message(f"Job {job_num}: Navigation Traceback: {traceback.format_exc()}")
                
                # Process LLS data if selected
                if lls_selected:
                    self.log_message(f"Job {job_num}: Processing LLS data...")
                    if not lls_folder or not os.path.exists(lls_folder):
                        self.log_message(f"Job {job_num}: ✗ LLS folder not found: {lls_folder}")
                    elif not phins_nav_file or not os.path.exists(phins_nav_file):
                        self.log_message(f"Job {job_num}: ✗ Phins nav file not found: {phins_nav_file}")
                    else:
                        try:
                            self.process_lls_data(output_folder)
                            self.log_message(f"Job {job_num}: ✓ LLS processing completed")
                        except Exception as lls_error:
                            self.log_message(f"Job {job_num}: ✗ LLS processing failed: {lls_error}")
                            self.log_message(f"Job {job_num}: LLS Traceback: {traceback.format_exc()}")
                
                # Process imagery data if selected
                if imagery_selected:
                    if not input_folder or not os.path.exists(input_folder):
                        self.log_message(f"Job {job_num}: ✗ Input folder not found: {input_folder}")
                    else:
                        # Mirror the single processing structure exactly
                        self.log_message(f"Job {job_num}: Processing Imagery data...")
                        self.log_message(f"Job {job_num}: ⚠ Note: All modules now use CSV-based processing for consistency")
                        
                        # Load navigation data from dive nav text file ONLY for imagery
                        nav_file_for_imagery = None
                        if nav_file and os.path.exists(nav_file):
                            nav_file_for_imagery = nav_file
                            self.log_message(f"Job {job_num}: Navigation source for imagery: {os.path.basename(nav_file_for_imagery)} (Dive Nav text file)")
                        else:
                            self.log_message(f"Job {job_num}: ⚠ No Dive Nav text file specified for imagery processing")
                        
                        if nav_file_for_imagery:
                            try:
                                if self.load_navigation_data_for_imagery_only(nav_file_for_imagery):
                                    self.log_message(f"Job {job_num}: ✓ Navigation data loaded successfully from text file for imagery processing")
                                else:
                                    self.log_message(f"Job {job_num}: ⚠ Navigation data load failed - continuing without it")
                            except Exception as nav_error:
                                self.log_message(f"Job {job_num}: ⚠ Navigation data error: {nav_error}")
                        else:
                            self.log_message(f"Job {job_num}: ⚠ No navigation file specified - continuing without navigation data")
                        
                        # Call the main imagery processing method
                        self.analyze_images(input_folder, output_folder)
                        self.log_message(f"Job {job_num}: ✓ Imagery processing completed")
                
                self.log_message(f"Job {job_num}: ✓ All processing completed successfully")
                
            except Exception as processing_error:
                self.log_message(f"Job {job_num}: ✗ Processing error: {processing_error}")
                self.log_message(f"Job {job_num}: Traceback: {traceback.format_exc()}")
                raise processing_error
            
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
            nav_selected = self.nav_processing_var.get()
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            if not nav_selected and not lls_selected and not imagery_selected:
                self.log_message("❌ Error: No processing functions selected")
                return False
            
            # Validate output folder
            output_folder = self.output_path.get().strip()
            if not output_folder:
                self.log_message("❌ Error: No output folder specified")
                return False
            
            # Validate Navigation processing inputs
            if nav_selected:
                nav_file = self.nav_plot_file_path.get().strip()
                
                if not nav_file:
                    self.log_message("❌ Error: Navigation processing selected but no navigation file specified")
                    return False
                
                if not os.path.exists(nav_file):
                    self.log_message(f"❌ Error: Navigation file does not exist: {nav_file}")
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
            nav_selected = self.nav_processing_var.get()
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            if not nav_selected and not lls_selected and not imagery_selected:
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
                required_columns = ['Output_folder']
                optional_columns = ['Image_Input', 'Dive_Nav_file', 'LLS_Input', 'PhinsData_Bin_file']
                
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
    
    def load_navigation_data_for_imagery_only(self, nav_file):
        """Load navigation data specifically for imagery processing, ensuring no PhinsData contamination"""
        try:
            if not hasattr(self, 'metrics') or not self.metrics:
                return False
                
            # Force reload of navigation data from text file
            success = self.metrics.load_nav_data(nav_file)
            if success:
                self.log_message(f"       ✓ Navigation data loaded for imagery altitude and heading extraction")
                
                # ONLY share navigation data from text file (not PhinsData) with imagery modules
                if hasattr(self.metrics, 'nav_timestamps'):
                    # For footprint map, we don't use nav_timestamps, it loads its own nav_data
                    # Just ensure it loads from the correct file
                    if hasattr(self, 'footprint_map') and self.footprint_map:
                        # Make sure footprint map uses the text file navigation data
                        footprint_nav_success = self.footprint_map.load_nav_data(nav_file)
                        if footprint_nav_success:
                            self.log_message(f"       ✓ Footprint map loaded navigation data from text file")
                        else:
                            self.log_message(f"       ⚠ Footprint map failed to load navigation data")
                        
                return True
            else:
                self.log_message(f"       ⚠ Failed to load navigation data for imagery")
                return False
                
        except Exception as e:
            self.log_message(f"       ⚠ Navigation data error for imagery: {e}")
            return False