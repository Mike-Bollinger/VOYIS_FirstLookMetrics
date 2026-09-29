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
from src.utils.file_utils import extract_dive_number, get_output_prefix

class ProcessingController:
    """Controls the main processing workflow"""
    
    def __init__(self):
        # Add stop processing flag
        self.stop_processing_flag = False
        self.current_processing_thread = None
        # Initialize dive prefix attributes
        self.dive_prefix_image = None
        self.dive_prefix_nav = None
        self.dive_prefix_lls = None
    
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

    def stop_processing(self):
        """Stop the current processing operation"""
        if self.current_processing_thread and self.current_processing_thread.is_alive():
            self.stop_processing_flag = True
            self.log_message("⚠️ Stop request received - processing will halt at next checkpoint...")
            self.update_progress(0, "Stopping processing...")
            
            # Disable stop button immediately
            if hasattr(self, 'stop_button'):
                self.stop_button.configure(state=tk.DISABLED)
        else:
            self.log_message("No active processing to stop")

    def check_stop_flag(self):
        """Check if processing should be stopped. Returns True if should stop."""
        if self.stop_processing_flag:
            # Only log the stop message once
            if not hasattr(self, '_stop_message_logged'):
                self.log_message("🛑 Processing stopped by user request")
                self.update_progress(0, "Processing stopped")
                self._stop_message_logged = True
            
            # Re-enable process button and disable stop button
            self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))
            if hasattr(self, 'stop_button'):
                self.root.after(0, lambda: self.stop_button.configure(state=tk.DISABLED))
            
            return True
        return False
    
    def play_completion_sound(self):
        """Play a sound when processing completes"""
        try:
            import winsound
            sound_file = os.path.join(os.path.dirname(__file__), '..', 'utils', 'sounds', 'beer_open.wav')
            if os.path.exists(sound_file):
                # Play the custom WAV file asynchronously
                winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Use system sound as fallback if custom sound not found
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception as e:
            # Silently fail if sound can't play
            pass

    def process_images(self):
        """Main function to process images based on selected options"""
        if not self.validate_inputs():
            return
        
        # Reset stop flag and stop message tracking
        self.stop_processing_flag = False
        if hasattr(self, '_stop_message_logged'):
            delattr(self, '_stop_message_logged')
        
        self.process_button.configure(state=tk.DISABLED)
        
        # Enable stop button if it exists
        if hasattr(self, 'stop_button'):
            self.stop_button.configure(state=tk.NORMAL)
        
        # Check if batch mode is enabled
        if self.batch_var.get():
            self.current_processing_thread = threading.Thread(target=self.process_batch, daemon=True)
            self.current_processing_thread.start()
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
        
        self.current_processing_thread = threading.Thread(
            target=self._process_images_thread,
            args=(input_folder, output_folder),
            daemon=True
        )
        self.current_processing_thread.start()

    def _process_images_thread(self, input_folder, output_folder):
        """Background thread function for processing images"""
        try:
            # Check stop flag before starting
            if self.check_stop_flag():
                return
                
            self.analyze_images(input_folder, output_folder)
            
            # Only show completion if not stopped
            if not self.stop_processing_flag:
                self.log_message("\nAll selected processing tasks completed successfully.")
                self.update_progress(100, "Processing complete!")
                self.play_completion_sound()
            
        except Exception as e:
            if not self.stop_processing_flag:  # Only log errors if not intentionally stopped
                self.log_message(f"\nError during processing: {str(e)}")
                self.update_progress(0, "Error during processing")
                traceback.print_exc()
        finally:
            # Always clean up thread reference and reset UI state
            self.current_processing_thread = None
            
            # Reset stop flag and message tracking for next run
            self.stop_processing_flag = False
            if hasattr(self, '_stop_message_logged'):
                delattr(self, '_stop_message_logged')
            
            # Always re-enable process button and disable stop button
            self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))
            if hasattr(self, 'stop_button'):
                self.root.after(0, lambda: self.stop_button.configure(state=tk.DISABLED))

    def extract_dive_prefixes(self, input_folder=None):
        """Extract dive number prefixes from input paths for all modules"""
        # Extract from image input path
        if input_folder:
            self.dive_prefix_image = get_output_prefix(input_folder, "Image")
            dive_num = extract_dive_number(input_folder)
            if dive_num:
                self.log_message(f"Detected dive number from image path: {dive_num}")
        else:
            self.dive_prefix_image = "Image_"
        
        # Extract from navigation directory path
        if hasattr(self, 'nav_directory_path') and self.nav_directory_path.get():
            nav_path = self.nav_directory_path.get()
            self.dive_prefix_nav = get_output_prefix(nav_path, "Nav")
            dive_num = extract_dive_number(nav_path)
            if dive_num:
                self.log_message(f"Detected dive number from nav path: {dive_num}")
        else:
            self.dive_prefix_nav = "Nav_"
        
        # Extract from LLS input path
        if hasattr(self, 'lls_path') and self.lls_path.get():
            lls_path = self.lls_path.get()
            self.dive_prefix_lls = get_output_prefix(lls_path, "LLS")
            dive_num = extract_dive_number(lls_path)
            if dive_num:
                self.log_message(f"Detected dive number from LLS path: {dive_num}")
        else:
            self.dive_prefix_lls = "LLS_"

    def analyze_images(self, input_folder, output_folder, skip_nav_processing=False):
        """Process images and LLS data based on selected functions
        
        Args:
            input_folder: Path to input folder containing images
            output_folder: Path to output folder for results
            skip_nav_processing: If True, skip navigation processing (used in batch mode when nav already processed)
        """
        try:
            self.update_progress(0, "Starting processing...")
            
            # Extract dive prefixes from input paths
            self.extract_dive_prefixes(input_folder)
                       
            # Check what processing is selected
            nav_selected = self.nav_processing_var.get()
            turbidity_selected = (
                hasattr(self, 'turbidity_plot_var')
                and self.turbidity_plot_var.get()
            )
            nav_to_shp_selected = (
                hasattr(self, 'nav_to_shp_var')
                and self.nav_to_shp_var.get()
            )
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            # Process Navigation data first if selected (but skip if already done in batch processing)
            if nav_selected and not skip_nav_processing:
                if self.check_stop_flag():
                    return
                    
                self.log_message("Processing Navigation data...")
                try:
                    self.process_navigation_data(output_folder)
                    self.log_message("✓ Navigation processing completed")
                except Exception as nav_error:
                    self.log_message(f"✗ Navigation processing failed: {nav_error}")
                    self.log_message(f"Traceback: {traceback.format_exc()}")
            elif nav_selected and skip_nav_processing:
                self.log_message("⚬ Skipping navigation processing (already completed in batch mode)")

            # Export nav track to shapefile if selected
            if nav_to_shp_selected and not skip_nav_processing:
                if self.check_stop_flag():
                    return
                self.log_message("Exporting Nav Track to Shapefile...")
                try:
                    self.process_nav_shapefile(output_folder)
                    self.log_message("✓ Nav track shapefile export completed")
                except Exception as shp_error:
                    self.log_message(f"✗ Nav track shapefile export failed: {shp_error}")
                    self.log_message(f"Traceback: {traceback.format_exc()}")

            # Process Turbidity data if selected (requires nav_directory to be set)
            if turbidity_selected and not skip_nav_processing:
                if self.check_stop_flag():
                    return
                self.log_message("Processing Turbidity data...")
                try:
                    self.process_turbidity_data(output_folder)
                    self.log_message("✓ Turbidity processing completed")
                except Exception as turb_error:
                    self.log_message(f"✗ Turbidity processing failed: {turb_error}")
                    self.log_message(f"Traceback: {traceback.format_exc()}")

            # Process LLS data if selected
            if lls_selected:
                if self.check_stop_flag():
                    return
                    
                self.log_message("Processing LLS data...")
                try:
                    self.process_lls_data(output_folder)
                    self.log_message("✓ LLS processing completed")
                except Exception as lls_error:
                    self.log_message(f"✗ LLS processing failed: {lls_error}")
                    self.log_message(f"Traceback: {traceback.format_exc()}")
            
            # Track imagery failure count for the overall summary
            imagery_failed_stages = 0
            self._last_imagery_failed_stages = 0

            # Process imagery data if selected
            if imagery_selected:
                if self.check_stop_flag():
                    return
                    
                self.log_message("Processing Imagery data...")
                self.log_message("       ⚠ Note: All modules now use CSV-based processing for consistency")
                
                # Wire nav directory into footprint_map so it can self-load PHINS_INS + ADCP.
                nav_dir = (self.nav_directory_path.get()
                           if hasattr(self, 'nav_directory_path') else '')
                if nav_dir and os.path.isdir(nav_dir):
                    if hasattr(self, 'footprint_map') and self.footprint_map:
                        self.footprint_map.nav_directory = nav_dir
                        self.footprint_map.nav_data = None   # force reload on first use
                        self.log_message(f"       ✓ Footprint map will use nav directory: {os.path.basename(nav_dir)}")
                else:
                    self.log_message("       ⚠ No navigation directory set – footprint map will rely on EXIF/CSV data only")
                
                # Check stop flag before starting metadata extraction
                if self.check_stop_flag():
                    return
                
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
                        return True  # Continue processing (don't check stop flag here to avoid repeated messages)
                    
                    processed_files, results = self.metrics.analyze_directory(
                        input_folder,
                        progress_callback=metadata_progress,
                        extract_gps=extract_gps
                    )
                    
                    gps_count = len(self.metrics.gps_data)
                    self.log_message(f"✓ Processed {processed_files} files, extracted GPS from {gps_count} images")
                    
                    if gps_count > 0:
                        self.log_message("✓ Metrics CSV will contain GPS and EXIF data")
                    else:
                        self.log_message(f"⚠ No GPS data found in: {os.path.basename(input_folder)}")
                        self.log_message(f"   Found: {self.metrics.processed_count} processed stills, {self.metrics.raw_count} raw images, {self.metrics.other_count} other files")
                        self.log_message("   Verify Image_Input points to the folder containing processed JPEG images (ESC_stills_processed_*)")
                    
                except Exception as metadata_error:
                    self.log_message(f"✗ Error extracting metadata: {metadata_error}")
                    self.log_message("Cannot proceed without image metadata")
                    return
                
                # Check stop flag after metadata extraction
                if self.check_stop_flag():
                    return
                
                # Get list of processing stages
                processing_stages = []
                
                if self.basic_metrics_var.get():
                    processing_stages.append(("Basic Metrics Analysis", self.process_basic_metrics))
                
                if self.location_map_var.get():
                    processing_stages.append(("Location Map Generation", self.process_location_map))
                
                if self.histogram_var.get():
                    processing_stages.append(("Altitude Histogram", self.process_histogram))

                if hasattr(self, 'turbidity_merge_var') and self.turbidity_merge_var.get():
                    processing_stages.append(("Turbidity Data Merge", self.process_turbidity_merge))

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
                    if self.check_stop_flag():
                        return
                        
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

                        nav_directory = None
                        if hasattr(self, 'nav_directory_path'):
                            dir_path = self.nav_directory_path.get()
                            if dir_path and os.path.isdir(dir_path):
                                nav_directory = dir_path
                        
                        # Use dive prefix if available
                        dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "Image_"
                        
                        if hasattr(self.metrics, 'create_image_metrics_csv_parallel'):
                            csv_path = self.metrics.create_image_metrics_csv_parallel(
                                input_folder, 
                                output_folder, 
                                nav_file, 
                                progress_callback=lambda p, msg="Creating master CSV...": self.update_progress(p, msg),
                                file_prefix=dive_prefix
                            )
                        else:
                            # Fallback to original method
                            csv_path = self.metrics.create_image_metrics_csv(
                                input_folder, 
                                output_folder, 
                                nav_file,
                                nav_directory=nav_directory,
                                progress_callback=lambda p, msg="Creating master CSV...": self.update_progress(p, msg),
                                file_prefix=dive_prefix
                            )
                        
                        if csv_path:
                            self.log_message(f"✓ Created Image_Metrics.csv: {os.path.basename(csv_path)}")
                            
                            # Check for old non-prefixed CSV and delete it
                            if dive_prefix != "Image_":
                                old_csv = os.path.join(output_folder, "Image_Metrics.csv")
                                if os.path.exists(old_csv) and old_csv != csv_path:
                                    try:
                                        os.remove(old_csv)
                                        self.log_message(f"✓ Removed old non-prefixed Image_Metrics.csv")
                                        self.log_message(f"   Now using: {os.path.basename(csv_path)}")
                                    except Exception as del_error:
                                        self.log_message(f"⚠ Could not delete old Image_Metrics.csv: {del_error}")
                                        self.log_message(f"   You can manually delete: {old_csv}")
                        else:
                            self.log_message("⚠ Failed to create Image_Metrics.csv")
                        
                    except Exception as e:
                        self.log_message(f"⚠ Error creating Image_Metrics.csv: {e}")
                        self.log_message("Processing will continue without master CSV")
                    
                    # Check stop flag before processing stages
                    if self.check_stop_flag():
                        return
                    
                    # STEP 2: Process each stage
                    self.log_message("STEP 2: Processing individual analysis stages...")
                    completed_stages = 0
                    failed_stages = 0
                    total_stages = len(processing_stages)
                    
                    for stage_idx, (stage_name, stage_func) in enumerate(processing_stages):
                        # Check stop flag before each stage
                        if self.check_stop_flag():
                            return
                            
                        try:
                            # Calculate progress
                            base_progress = 40 if lls_selected else 20  # Account for LLS processing
                            current_progress = int(base_progress + (stage_idx / total_stages * 50))  # 40-90% or 20-70% for stages
                            
                            # Update progress with stage name
                            progress_text = stage_name
                            self.update_progress(current_progress, progress_text)
                            
                            self.log_message(f"STAGE {stage_idx + 1}/{total_stages}: {stage_name}")
                            
                            # Execute the stage and track explicit success/failure.
                            stage_result = stage_func(input_folder, output_folder)
                            stage_success = True if stage_result is None else bool(stage_result)

                            if stage_success:
                                completed_stages += 1
                                self.log_message(f"✓ {stage_name} completed")
                            else:
                                failed_stages += 1
                                self.log_message(f"✗ {stage_name} failed")
                            
                        except Exception as stage_error:
                            failed_stages += 1
                            self.log_message(f"✗ Error in {stage_name}: {stage_error}")
                            self.log_message(f"Traceback: {traceback.format_exc()}")
                            self.log_message(f"Continuing with next stage...")
                    
                    # Final summary for imagery processing
                    self.log_message(f"\n{'='*60}")
                    self.log_message(f"IMAGERY PROCESSING SUMMARY")
                    self.log_message(f"{'='*60}")
                    self.log_message(f"Total stages: {total_stages}")
                    self.log_message(f"Completed successfully: {completed_stages}")
                    self.log_message(f"Failed: {failed_stages}")
                    self.log_message(f"Success rate: {(completed_stages/total_stages*100):.1f}%")
                    
                    imagery_failed_stages = failed_stages
                    self._last_imagery_failed_stages = failed_stages

                    if failed_stages == 0:
                        self.log_message("✓ All imagery processing completed successfully!")
                    else:
                        self.log_message(f"⚠ Imagery processing completed with {failed_stages} errors")
            
            # Final check before completion
            if self.check_stop_flag():
                return
            
            # Regenerate shapefiles for every Image_Metrics CSV found in the
            # output folder.  This runs regardless of which modules were selected
            # so the SHP always stays in sync with the CSV.
            self.log_message("\nFINAL STEP: Syncing Image_Metrics shapefiles...")
            try:
                import glob
                from src.utils.csv_to_shapefile import csv_to_shp

                metrics_csvs = glob.glob(os.path.join(output_folder, "*Image_Metrics.csv"))
                if metrics_csvs:
                    for csv_path in metrics_csvs:
                        success, msg = csv_to_shp(csv_path, log_fn=self.log_message)
                        if success:
                            self.log_message(f"✓ Shapefile updated: {os.path.basename(msg)}")
                        else:
                            self.log_message(f"⚠ Shapefile export skipped: {msg}")
                else:
                    self.log_message("   No Image_Metrics CSV found in output folder – shapefile step skipped")

            except Exception as shapefile_error:
                self.log_message(f"⚠ Error during shapefile sync: {shapefile_error}")
                self.log_message("   Processing completed successfully, but shapefile was not updated")
            
            # Final overall summary
            total_processes = (1 if nav_selected else 0) + (1 if nav_to_shp_selected else 0) + (1 if turbidity_selected else 0) + (1 if lls_selected else 0) + (1 if imagery_selected else 0)
            self.log_message(f"\n{'='*60}")
            self.log_message(f"OVERALL PROCESSING SUMMARY")
            self.log_message(f"{'='*60}")
            
            if nav_selected:
                nav_status = "⚬ Skipped (batch)" if skip_nav_processing else "✓ Completed"
                self.log_message(f"Navigation Processing: {nav_status}")
            if nav_to_shp_selected:
                shp_status = "⚬ Skipped (batch)" if skip_nav_processing else "✓ Completed"
                self.log_message(f"Nav Track Shapefile: {shp_status}")
            if turbidity_selected:
                turb_status = "⚬ Skipped (batch)" if skip_nav_processing else "✓ Completed"
                self.log_message(f"Turbidity Processing: {turb_status}")
            if lls_selected:
                self.log_message("LLS Processing: ✓ Completed")
            if imagery_selected:
                if imagery_failed_stages == 0:
                    self.log_message("Imagery Processing: ✓ Completed")
                else:
                    self.log_message(f"Imagery Processing: ✗ Failed ({imagery_failed_stages} stage(s) failed)")
            
            if total_processes > 0:
                if imagery_selected and imagery_failed_stages > 0:
                    self.log_message(f"⚠ Processing completed with errors – imagery had {imagery_failed_stages} failed stage(s)")
                    self.update_progress(100, f"Completed with {imagery_failed_stages} imagery error(s)")
                else:
                    self.log_message("✓ All processing completed successfully!")
                    self.update_progress(100, "All processing completed")
            else:
                self.log_message("⚠ No processing was performed")
                self.update_progress(100, "No processing performed")
            
            # Play completion sound
            self.play_completion_sound()
            
        except Exception as e:
            if not self.stop_processing_flag:  # Only log errors if not intentionally stopped
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
        
        try:
            from src.models.nav_plotter import NavPlotter
            
            # Create nav plotter instance
            nav_plotter = NavPlotter(log_callback=self.log_message)
            
            # Directory mode (the only mode now - automatic file discovery)
            nav_directory = self.nav_directory_path.get()
            
            if not nav_directory or not os.path.exists(nav_directory):
                self.log_message("⚠ Navigation directory not specified or doesn't exist. Skipping navigation plotting.")
                self.log_message("   Please select a directory containing navigation files")
                return
            
            self.log_message(f"       Using navigation directory: {nav_directory}")
            
            # Use dive prefix if available, otherwise use default
            dive_prefix = self.dive_prefix_nav if hasattr(self, 'dive_prefix_nav') and self.dive_prefix_nav else "nav_"
            
            # Process using directory method
            success = nav_plotter.process_navigation_directory(
                nav_directory=nav_directory,
                output_dir=output_folder,
                dive_name="Navigation",
                log_callback=self.log_message,
                file_prefix=dive_prefix
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

    def process_turbidity_data(self, output_folder):
        """Process turbidity data from MCAP and/or TURBIDITY.txt sources."""
        self.log_message("STAGE 1b: Processing Turbidity data from nav sources...")

        nav_directory = self.nav_directory_path.get() if hasattr(self, 'nav_directory_path') else ""

        if not nav_directory or not os.path.exists(nav_directory):
            self.log_message("⚠ Navigation directory not specified or doesn't exist – turbidity skipped.")
            return

        try:
            from src.models.turbidity_processor import TurbidityProcessor

            dive_prefix = (
                self.dive_prefix_nav
                if hasattr(self, 'dive_prefix_nav') and self.dive_prefix_nav
                else ""
            )

            processor = TurbidityProcessor(log_callback=self.log_message)
            success = processor.process(
                nav_directory=nav_directory,
                output_dir=output_folder,
                file_prefix=dive_prefix,
            )

            if success:
                self.update_progress(22, "Turbidity processing completed")
            else:
                self.log_message("⚠ No turbidity data found – plots not generated")

        except ImportError as e:
            self.log_message(f"Error: Could not import TurbidityProcessor: {e}")
        except Exception as e:
            self.log_message(f"Error during turbidity processing: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")

    def process_nav_shapefile(self, output_folder):
        """Export navigation track as a dissolved LineString shapefile."""
        self.log_message("STAGE 1c: Exporting Nav Track to Shapefile...")

        nav_directory = self.nav_directory_path.get() if hasattr(self, 'nav_directory_path') else ""

        if not nav_directory or not os.path.exists(nav_directory):
            self.log_message("⚠ Navigation directory not specified or doesn't exist – shapefile export skipped.")
            return

        try:
            from src.utils.nav_to_shapefile import nav_to_shapefile

            dive_prefix = (
                self.dive_prefix_nav
                if hasattr(self, 'dive_prefix_nav') and self.dive_prefix_nav
                else ""
            )

            success = nav_to_shapefile(
                nav_directory=nav_directory,
                output_dir=output_folder,
                file_prefix=dive_prefix,
                log_fn=self.log_message,
            )

            if success:
                self.update_progress(21, "Nav shapefile export completed")
            else:
                self.log_message("⚠ Nav track shapefile was not created")

        except ImportError as e:
            self.log_message(f"Error: Could not import nav_to_shapefile: {e}")
        except Exception as e:
            self.log_message(f"Error during nav shapefile export: {str(e)}")
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
            
            # Use dive prefix if available, otherwise use default
            dive_prefix = self.dive_prefix_lls if hasattr(self, 'dive_prefix_lls') and self.dive_prefix_lls else "lls_"
            
            # Get processing mode from GUI
            processing_mode = getattr(self, 'lls_processing_mode', None)
            use_inplace = processing_mode.get() == 'inplace' if processing_mode else False
            
            lls_processor = LLSProcessor(
                log_callback=self.log_message,
                progress_callback=lambda value, msg: self.update_progress(
                    progress_offset + (value * progress_scale / 100), msg
                ),
                stop_check_callback=self.check_stop_flag
            )
            
            success = lls_processor.process_lls_data(lls_folder, nav_file, output_folder, 
                                                     file_prefix=dive_prefix, 
                                                     use_inplace=use_inplace)
            
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

            if hasattr(self, 'turbidity_merge_var') and self.turbidity_merge_var.get():
                stages.append(("Turbidity Data Merge", self.process_turbidity_merge))

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
                return False
            
            try:
                # OPTIMIZATION: Skip GPS extraction if we already have it from CSV creation
                # This prevents redundant EXIF reading of 70k+ images
                already_have_gps = hasattr(self.metrics, 'gps_data') and len(self.metrics.gps_data) > 0
                
                # Only extract GPS if we need it AND don't already have it
                extract_gps = False
                if not already_have_gps:
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
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "Image_"
                metrics_file = os.path.join(output_folder, f"{dive_prefix}Metrics.txt")
                with open(metrics_file, "w") as f:
                    f.write("\n".join(results))
                
                self.log_message(f"  └─ ✓ Basic metrics analysis completed - {processed_files} files processed")
                self.log_message(f"       Results saved to: {metrics_file}")
                return True
                
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in basic metrics: {e}")
                return False
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in basic metrics: {e}")
            return False

    def process_location_map(self, input_folder, output_folder):
        """Process location map generation"""
        try:
            self.log_message("  └─ Running location map generation...")
            
            if not hasattr(self, 'altitude_map') or not self.altitude_map:
                self.log_message("  └─ ✗ Error: AltitudeMap processor not initialized")
                return False
                
            if not hasattr(self, 'metrics') or not self.metrics or not self.metrics.gps_data:
                self.log_message("  └─ ✗ Error: No GPS data available for location map")
                return False
            
            try:
                # Use dive prefix if available
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                
                # Call the correct method with GPS data from metrics
                map_file = self.altitude_map.create_location_map(
                    self.metrics.gps_data,
                    output_folder,
                    metrics=self.metrics,
                    file_prefix=dive_prefix
                )
                
                if map_file and os.path.exists(map_file):
                    self.log_message(f"  └─ ✓ Location map created: {os.path.basename(map_file)}")
                    return True
                else:
                    self.log_message("  └─ ✗ Location map generation failed")
                    return False
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in location map: {e}")
                return False
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in location map: {e}")
            return False

    def process_histogram(self, input_folder, output_folder):
        """Process altitude histogram"""
        try:
            self.log_message("  └─ Running altitude histogram...")
            
            if not hasattr(self, 'altitude_map') or not self.altitude_map:
                self.log_message("  └─ ✗ Error: AltitudeMap processor not initialized")
                return False
                
            if not hasattr(self, 'metrics') or not self.metrics or not self.metrics.gps_data:
                self.log_message("  └─ ✗ Error: No GPS data available for histogram")
                return False
            
            try:
                # Use dive prefix if available
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                
                # Call the correct method with GPS data from metrics
                histogram_file = self.altitude_map.create_altitude_histogram(
                    self.metrics.gps_data,
                    output_folder,
                    file_prefix=dive_prefix
                )
                
                if histogram_file and os.path.exists(histogram_file):
                    self.log_message(f"  └─ ✓ Altitude histogram created: {os.path.basename(histogram_file)}")
                    return True
                else:
                    self.log_message("  └─ ✗ Altitude histogram creation failed")
                    return False
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in histogram: {e}")
                return False
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in histogram: {e}")
            return False

    def process_turbidity_merge(self, input_folder, output_folder):
        """Merge turbidity NTU from TURBIDITY.txt (or fallback to MCAP bags) into Image_Metrics CSV.
        
        Matches by nearest time (±10s tolerance) using existing depth/heading/altitude from CSV.
        """
        self.log_message("  └─ Running turbidity data merge...")

        import glob as _glob
        import pandas as pd
        from pathlib import Path

        # ── Locate Image_Metrics CSV ─────────────────────────────────────────
        dive_prefix = (
            self.dive_prefix_image
            if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image
            else "Image_"
        )
        csv_path = os.path.join(output_folder, f"{dive_prefix}Metrics.csv")
        if not os.path.exists(csv_path):
            candidates = _glob.glob(os.path.join(output_folder, "*Metrics.csv"))
            if candidates:
                csv_path = candidates[0]
            else:
                self.log_message("  └─ ⚠ Image_Metrics CSV not found – turbidity merge skipped")
                return False

        # ── Locate navigation directory ──────────────────────────────────────
        nav_directory = (
            self.nav_directory_path.get()
            if hasattr(self, 'nav_directory_path')
            else ""
        )
        if not nav_directory or not os.path.exists(nav_directory):
            self.log_message("  └─ ⚠ Navigation directory not set – turbidity merge skipped")
            return False

        try:
            # ── Try to read TURBIDITY.txt first (simpler, preferred) ─────────
            turb_files = list(Path(nav_directory).rglob("TURBIDITY.txt"))
            turb_df = None
            dive_date = None
            
            if turb_files:
                self.log_message(f"  └─ Found {len(turb_files)} TURBIDITY.txt file(s)")
                try:
                    # Read TURBIDITY.txt – has mission_msecs, lat, lon, depth, data1 (turbidity)
                    turb_df = pd.read_csv(turb_files[0], skipinitialspace=True)
                    self.log_message(f"  └─ Loaded {len(turb_df)} turbidity rows from TURBIDITY.txt")
                    
                    # Expect columns: mission_msecs, latitude, longitude, depth, type, index, faults, data1, data2
                    # data1 is turbidity in NTU
                    if 'data1' not in turb_df.columns:
                        turb_df = None
                except Exception as e:
                    self.log_message(f"  └─ ⚠ Failed to parse TURBIDITY.txt: {e}")
                    turb_df = None
            
            # Fallback: try MCAP bags if TURBIDITY.txt unavailable or failed
            if turb_df is None or turb_df.empty:
                self.log_message("  └─ Falling back to MCAP bag parsing...")
                from src.models.turbidity_processor import TurbidityProcessor
                processor = TurbidityProcessor(log_callback=self.log_message)
                load_result = processor.load_turbidity_and_status(nav_directory)
                if isinstance(load_result, tuple) and len(load_result) >= 1:
                    turb_df = load_result[0]
                else:
                    turb_df = pd.DataFrame()
            
            if turb_df is None or turb_df.empty:
                self.log_message("  └─ ⚠ No turbidity data found from any source")
                return False

            self.log_message(f"  └─ Loaded {len(turb_df):,} turbidity readings")

            # ── Load Image_Metrics CSV ────────────────────────────────────────
            img_df = pd.read_csv(csv_path)
            if img_df.empty or 'datetime_original' not in img_df.columns:
                self.log_message("  └─ ⚠ Image_Metrics CSV missing datetime_original column")
                return False

            # ── Infer dive_date from image timestamps if needed ───────────────
            if 'mission_msecs' in turb_df.columns and dive_date is None:
                # Extract dive date from first valid image EXIF timestamp
                for idx, row in img_df.iterrows():
                    if pd.notna(row['datetime_original']):
                        try:
                            from dateutil import parser
                            dt = parser.parse(str(row['datetime_original']).replace(':', '-', 2))
                            dive_date = dt.date()
                            break
                        except Exception:
                            continue
                if dive_date is None:
                    dive_date = pd.Timestamp.now().date()
                self.log_message(f"  └─ Using dive date: {dive_date}")

            # ── Convert timestamps for matching ──────────────────────────────
            def _parse_exif_seconds(dt_str):
                """Parse EXIF 'YYYY:MM:DD HH:MM:SS' → seconds since midnight UTC"""
                if not dt_str or pd.isna(dt_str):
                    return None
                try:
                    from dateutil import parser
                    dt = parser.parse(str(dt_str).replace(':', '-', 2))
                    # Return seconds since start of day
                    return dt.hour * 3600 + dt.minute * 60 + dt.second
                except Exception:
                    return None

            def _mission_msecs_to_seconds(msecs):
                """Convert mission milliseconds to seconds"""
                if pd.isna(msecs):
                    return None
                return float(msecs) / 1000.0

            # Add time columns for matching
            img_df['_img_seconds'] = pd.to_numeric(
                img_df['datetime_original'].apply(_parse_exif_seconds), errors='coerce'
            ).astype('float64')
            if 'mission_msecs' in turb_df.columns:
                turb_df['_turb_seconds'] = pd.to_numeric(
                    turb_df['mission_msecs'].apply(_mission_msecs_to_seconds), errors='coerce'
                ).astype('float64')
            elif 'timestamp_ns' in turb_df.columns:
                # Use nanosecond timestamps - convert to seconds
                turb_df['_turb_seconds'] = pd.to_numeric(
                    (turb_df['timestamp_ns'] / 1e9) % 86400, errors='coerce'
                ).astype('float64')  # seconds into day
            else:
                turb_df['_turb_seconds'] = pd.Series(dtype='float64')

            # Prepare for merge
            img_valid = img_df[img_df['_img_seconds'].notna()].copy()
            img_valid['_orig_index'] = img_valid.index
            turb_valid = turb_df[turb_df['_turb_seconds'].notna()].copy().reset_index(drop=True)

            if img_valid.empty or turb_valid.empty:
                self.log_message("  └─ ⚠ No valid timestamps for matching")
                return False

            # ── Merge by nearest time (±10s) + depth proximity ─────────────────
            # Sort both by time for merge_asof
            img_valid['_img_seconds'] = pd.to_numeric(img_valid['_img_seconds'], errors='coerce').astype('float64')
            turb_valid['_turb_seconds'] = pd.to_numeric(turb_valid['_turb_seconds'], errors='coerce').astype('float64')
            img_valid = img_valid.sort_values('_img_seconds').reset_index(drop=True)
            turb_valid = turb_valid.sort_values('_turb_seconds').reset_index(drop=True)

            # Select turbidity column (could be data1, turbidity_ntu, etc)
            turb_col = 'data1' if 'data1' in turb_valid.columns else 'turbidity_ntu'
            if turb_col not in turb_valid.columns:
                self.log_message(f"  └─ ⚠ No turbidity column found (expected {turb_col})")
                return False

            merged = pd.merge_asof(
                img_valid[['_img_seconds', 'depth']],
                turb_valid[['_turb_seconds', turb_col, 'depth']].rename(
                    columns={'depth': 'turb_depth', turb_col: 'turbidity_ntu'}
                ),
                left_on='_img_seconds',
                right_on='_turb_seconds',
                direction='nearest',
                tolerance=10.0,  # ±10 seconds
                suffixes=('_img', '_turb')
            )

            matched = merged['turbidity_ntu'].notna().sum()
            self.log_message(f"  └─ Matched {matched}/{len(img_valid)} images by time (±10s)")

            # ── Write turbidity back to original CSV ─────────────────────────
            if 'turbidity_ntu' not in img_df.columns:
                img_df['turbidity_ntu'] = float('nan')

            img_df.loc[img_valid['_orig_index'].values, 'turbidity_ntu'] = merged['turbidity_ntu'].values
            img_df.drop(columns=['_img_seconds'], inplace=True, errors='ignore')
            img_df.to_csv(csv_path, index=False)

            self.log_message(
                f"  └─ ✓ Turbidity merged: {matched} rows populated in {os.path.basename(csv_path)}"
            )
            return True

        except ImportError as e:
            self.log_message(f"  └─ ✗ Missing dependency for turbidity merge: {e}")
            return False
        except Exception as e:
            self.log_message(f"  └─ ✗ Turbidity merge error: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            return False

    def process_footprint_map(self, input_folder, output_folder):
        """Process footprint map generation"""
        try:
            self.log_message("  └─ Running footprint map generation...")
            if not hasattr(self, 'footprint_map') or not self.footprint_map:
                self.log_message("  └─ ✗ Error: FootprintMap processor not initialized")
                return False
                
            if not hasattr(self, 'metrics') or not self.metrics or not self.metrics.gps_data:
                self.log_message("  └─ ✗ Error: No GPS data available for footprint map")
                return False
            
            try:
                # Set altitude threshold
                self.footprint_map.altitude_threshold = self.altitude_threshold
                
                # First try to use the CSV data if available
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                csv_filename = f"{dive_prefix}Metrics.csv"
                csv_path = os.path.join(output_folder, csv_filename)
                footprint_file = None
                
                if os.path.exists(csv_path):
                    self.log_message("  └─ Using Image_Metrics.csv for footprint analysis...")
                    # Use dive prefix if available
                    dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                    # Use the new CSV-based method that includes heading data
                    footprint_file = self.footprint_map.create_footprint_map_from_csv(
                        csv_path,
                        output_folder,
                        file_prefix=dive_prefix
                    )
                else:
                    self.log_message("  └─ Using legacy GPS data for footprint analysis...")
                    # Use dive prefix if available
                    dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"

                    # Call the legacy method – nav_directory already set on footprint_map
                    footprint_file = self.footprint_map.create_footprint_map(
                        self.metrics.gps_data,
                        output_folder,
                        file_prefix=dive_prefix
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
                    return True
                else:
                    self.log_message("  └─ ✗ Footprint map generation failed")
                    return False
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in footprint map: {e}")
                return False
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in footprint map: {e}")
            return False

    def process_visibility_analysis(self, input_folder, output_folder):
        """Process visibility analysis"""
        try:
            self.log_message("  └─ Running visibility analysis...")
            
            if not hasattr(self, 'visibility_analyzer') or not self.visibility_analyzer:
                self.log_message("  └─ ✗ Error: VisibilityAnalyzer not initialized")
                return False
            
            # Verify input folder exists and has images
            if not input_folder or not os.path.exists(input_folder):
                self.log_message(f"  └─ ✗ Error: Input folder does not exist: {input_folder}")
                return False
            
            # Count images in input folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            image_count = 0
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_count += 1
            
            if image_count == 0:
                self.log_message(f"  └─ ✗ Error: No images found in input folder: {input_folder}")
                return False
            
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
                        return False
                
                self.log_message("       Loading visibility model...")
                
                # Load the model
                success = self.visibility_analyzer.load_or_train_model(model_path)
                if not success:
                    self.log_message("  └─ ✗ Failed to load visibility model")
                    return False
                
                self.log_message("       ✓ Model loaded, analyzing images...")
                
                # Create the master CSV path with dive prefix
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                csv_filename = f"{dive_prefix}Metrics.csv"
                master_csv = os.path.join(output_folder, csv_filename)
                if not os.path.exists(master_csv):
                    self.log_message(f"  └─ ⚠ {csv_filename} not found, visibility analysis requires existing CSV")
                    return False
                
                # Use dive prefix if available
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                
                # Run analysis using the CSV method to update the master CSV
                success = self.visibility_analyzer.analyze_images_from_csv(
                    master_csv,    # Path to master CSV file
                    output_folder,  # Output folder for any additional files
                    file_prefix=dive_prefix
                )
                
                if success:
                    self.log_message("  └─ ✓ Visibility analysis completed successfully")
                    self.log_message("       Results updated in Image_Metrics.csv")
                    return True
                else:
                    self.log_message("  └─ ✗ Visibility analysis failed")
                    return False
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in visibility analysis: {e}")
                self.log_message(f"       Traceback: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in visibility analysis: {e}")
            return False

    def process_highlight_selection(self, input_folder, output_folder):
        """Process highlight image selection"""
        try:
            self.log_message("  └─ Running highlight selection...")
            
            if not hasattr(self, 'highlight_selector') or not self.highlight_selector:
                self.log_message("  └─ ✗ Error: HighlightSelector not initialized")
                return False
            
            # Verify input folder exists and has images
            if not input_folder or not os.path.exists(input_folder):
                self.log_message(f"  └─ ✗ Error: Input folder does not exist: {input_folder}")
                return False
            
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
                return False
            
            try:
                # The highlight selector will automatically check for visibility results
                # in the output folder, so we don't need to pass them explicitly
                # This maintains the dual functionality you want
                
                self.log_message(f"       Processing images from master CSV...")
                
                # Create the master CSV path with dive prefix
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                csv_filename = f"{dive_prefix}Metrics.csv"
                master_csv = os.path.join(output_folder, csv_filename)
                if not os.path.exists(master_csv):
                    self.log_message(f"  └─ ⚠ {csv_filename} not found, highlight selection requires existing CSV")
                    return False
                
                # Use dive prefix if available
                dive_prefix = self.dive_prefix_image if hasattr(self, 'dive_prefix_image') and self.dive_prefix_image else "image_"
                
                # Call the CSV-based highlight selector to update the master CSV
                highlight_paths = self.highlight_selector.select_highlights_from_csv(
                    master_csv,       # Path to master CSV file
                    output_folder,    # Output folder for highlight images
                    count=10,         # Number of highlights to select
                    file_prefix=dive_prefix,
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
                    return True
                else:
                    self.log_message("  └─ ✗ No highlight images were selected")
                    return False
                    
            except Exception as e:
                self.log_message(f"  └─ ✗ Error in highlight selection: {e}")
                self.log_message(f"       Traceback: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            self.log_message(f"  └─ ✗ Error in highlight selection: {e}")
            return False

    def get_required_imagery_stages(self):
        """Get list of required imagery processing stages"""
        stages = []
        
        if self.basic_metrics_var.get():
            stages.append(("Basic Metrics Analysis", self.process_basic_metrics))
        
        if self.location_map_var.get():
            stages.append(("Location Map Generation", self.process_location_map))
        
        if self.histogram_var.get():
            stages.append(("Altitude Histogram", self.process_histogram))

        if hasattr(self, 'turbidity_merge_var') and self.turbidity_merge_var.get():
            stages.append(("Turbidity Data Merge", self.process_turbidity_merge))
        
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
                # Check stop flag before each job
                if self.check_stop_flag():
                    self.log_message(f"Batch processing stopped at job {index + 1}/{len(df)}")
                    break
                
                job_num = index + 1
                self.log_message(f"\n{'='*60}")
                self.log_message(f"PROCESSING JOB {job_num}/{len(df)}")
                self.log_message(f"{'='*60}")
                
                try:
                    # Set up job-specific paths
                    input_folder = str(row.get('Image_Input', '')).strip() if pd.notna(row.get('Image_Input', '')) else ''
                    output_folder = str(row.get('Output_folder', '')).strip() if pd.notna(row.get('Output_folder', '')) else ''
                    
                    # Navigation directory for nav processing module
                    nav_directory = str(row.get('nav_directory', '')).strip() if pd.notna(row.get('nav_directory', '')) else ''
                    
                    lls_folder = str(row.get('LLS_Input', '')).strip() if pd.notna(row.get('LLS_Input', '')) else ''
                    phins_nav_file = str(row.get('PhinsData_Bin_file', '')).strip() if pd.notna(row.get('PhinsData_Bin_file', '')) else ''
                    phins_data_nav_file = str(row.get('PhinsData_Nav_file', '')).strip() if pd.notna(row.get('PhinsData_Nav_file', '')) else ''
                    
                    # Navigation module files (may not be present in all CSV formats)
                    nav_state_file = str(row.get('NAV_STATE_file', '')).strip() if pd.notna(row.get('NAV_STATE_file', '')) else ''
                    phins_ins_file = str(row.get('PHINS_INS_file', '')).strip() if pd.notna(row.get('PHINS_INS_file', '')) else ''
                    
                    # Convert empty strings to None for cleaner logic
                    input_folder = input_folder if input_folder else None
                    output_folder = output_folder if output_folder else None
                    nav_directory = nav_directory if nav_directory else None
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
                    has_nav_module = nav_directory or nav_state_file  # Navigation module needs nav_directory OR NAV_STATE_file
                    has_image_module = input_folder
                    has_lls_module = lls_folder and phins_nav_file
                    
                    if not (has_nav_module or has_image_module or has_lls_module):
                        self.log_message(f"Job {job_num}: Skipping - no valid processing module inputs specified")
                        self.log_message(f"  Navigation module needs: nav_directory OR NAV_STATE_file")
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
                    job_success = self.process_single_batch_job(
                        job_num, input_folder, output_folder,
                        lls_folder, phins_nav_file, phins_data_nav_file,
                        nav_state_file, phins_ins_file, nav_directory
                    )
                    
                    # Check if processing was stopped during the job
                    if self.stop_processing_flag:
                        break
                    
                    if job_success:
                        self.log_message(f"Job {job_num} completed successfully")
                        successful_jobs += 1
                    else:
                        self.log_message(f"Job {job_num} failed")
                        failed_jobs += 1
                    
                except Exception as e:
                    self.log_message(f"Error in job {job_num}: {str(e)}")
                    self.log_message(traceback.format_exc())
                    failed_jobs += 1
            
            # Final summary
            self.log_message(f"\n{'='*60}")
            if self.stop_processing_flag:
                self.log_message(f"BATCH PROCESSING STOPPED BY USER")
            else:
                self.log_message(f"BATCH PROCESSING SUMMARY")
            self.log_message(f"{'='*60}")
            self.log_message(f"Total jobs: {len(df)}")
            self.log_message(f"Successful: {successful_jobs}")
            self.log_message(f"Failed: {failed_jobs}")
            if successful_jobs + failed_jobs > 0:
                self.log_message(f"Success rate: {(successful_jobs/(successful_jobs + failed_jobs)*100):.1f}%")
            
            if not self.stop_processing_flag:
                self.log_message(f"Batch processing completed!")
                self.update_progress(100, "Batch processing complete")
                self.play_completion_sound()
            
        except Exception as e:
            if not self.stop_processing_flag:  # Only log errors if not intentionally stopped
                self.log_message(f"Error during batch processing: {str(e)}")
                self.log_message(traceback.format_exc())
        finally:
            # Always clean up thread reference and reset UI state
            self.current_processing_thread = None
            
            # Reset stop flag and message tracking for next run
            self.stop_processing_flag = False
            if hasattr(self, '_stop_message_logged'):
                delattr(self, '_stop_message_logged')
            
            # Always re-enable process button and disable stop button
            self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))
            if hasattr(self, 'stop_button'):
                self.root.after(0, lambda: self.stop_button.configure(state=tk.DISABLED))

    def process_single_batch_job(self, job_num, input_folder, output_folder,
                                lls_folder, phins_nav_file, phins_data_nav_file,
                                nav_state_file, phins_ins_file, nav_directory=None):
        """Process a single job from the batch CSV - mirrors single dive processing
        Returns True if successful, False if failed or stopped"""
        
        # Check stop flag at start of each job
        if self.check_stop_flag():
            return False
        
        # Determine what processing is needed
        nav_selected = self.nav_processing_var.get()
        turbidity_selected = (
            hasattr(self, 'turbidity_plot_var')
            and self.turbidity_plot_var.get()
        )
        nav_to_shp_selected = (
            hasattr(self, 'nav_to_shp_var')
            and self.nav_to_shp_var.get()
        )
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
            if nav_directory:
                self.log_message(f"  - Nav directory: {nav_directory}")
            if nav_state_file:
                self.log_message(f"  - Nav state file: {nav_state_file}")
            if phins_ins_file:
                self.log_message(f"  - PHINS INS file: {phins_ins_file}")
        if nav_to_shp_selected:
            self.log_message(f"  - Nav Track Shapefile export: ENABLED")
            if nav_directory:
                self.log_message(f"  - Nav directory: {nav_directory}")
        if turbidity_selected:
            self.log_message(f"  - Turbidity plotting: ENABLED")
            if nav_directory:
                self.log_message(f"  - Nav directory: {nav_directory}")
        if lls_selected:
            self.log_message(f"  - LLS processing: ENABLED")
            self.log_message(f"  - LLS folder: {lls_folder}")
            self.log_message(f"  - Phins nav: {phins_nav_file}")
        if imagery_selected:
            self.log_message(f"  - Imagery processing: ENABLED")
            self.log_message(f"  - Input folder: {input_folder}")
            if nav_directory:
                self.log_message(f"  - Nav directory for imagery: {nav_directory}")
        self.log_message(f"  - Output folder: {output_folder}")
        
        if not nav_selected and not nav_to_shp_selected and not turbidity_selected and not lls_selected and not imagery_selected:
            self.log_message(f"Job {job_num}: No processing functions selected - skipping")
            return False
        
        # Save current paths and temporarily override for this job
        original_paths = self.save_current_paths()
        
        try:
            # Extract dive prefixes from ALL input paths for this batch job
            # This must be done BEFORE any processing so all modules use correct prefixes
            self.log_message(f"Job {job_num}: Extracting dive prefixes from input paths...")
            
            # Extract from image input path
            if input_folder:
                self.dive_prefix_image = get_output_prefix(input_folder, "Image")
                dive_num = extract_dive_number(input_folder)
                if dive_num:
                    self.log_message(f"  → Image prefix: {self.dive_prefix_image}")
                else:
                    self.dive_prefix_image = "Image_"
                    self.log_message(f"  → Image prefix defaulted to: {self.dive_prefix_image}")
            else:
                self.dive_prefix_image = "Image_"
            
            # Extract from navigation directory path
            if nav_directory:
                self.dive_prefix_nav = get_output_prefix(nav_directory, "Nav")
                dive_num = extract_dive_number(nav_directory)
                if dive_num:
                    self.log_message(f"  → Nav prefix: {self.dive_prefix_nav}")
                else:
                    self.dive_prefix_nav = "Nav_"
                    self.log_message(f"  → Nav prefix defaulted to: {self.dive_prefix_nav}")
            else:
                self.dive_prefix_nav = "Nav_"
            
            # Extract from LLS input path
            if lls_folder:
                self.dive_prefix_lls = get_output_prefix(lls_folder, "LLS")
                dive_num = extract_dive_number(lls_folder)
                if dive_num:
                    self.log_message(f"  → LLS prefix: {self.dive_prefix_lls}")
                else:
                    self.dive_prefix_lls = "LLS_"
                    self.log_message(f"  → LLS prefix defaulted to: {self.dive_prefix_lls}")
            else:
                self.dive_prefix_lls = "LLS_"
            
            # Set paths for this job
            if imagery_selected:
                self.input_path.set(input_folder if input_folder else '')

                # Wire nav_directory into footprint_map (replaces Dive_Nav_file)
                if nav_directory and os.path.isdir(nav_directory):
                    if hasattr(self, 'footprint_map') and self.footprint_map:
                        self.footprint_map.nav_directory = nav_directory
                        self.footprint_map.nav_data = None
                else:
                    if hasattr(self, 'footprint_map') and self.footprint_map:
                        self.footprint_map.nav_directory = None
            else:
                self.input_path.set('')
                self.nav_path.set('')
            
            if lls_selected:
                self.lls_path.set(lls_folder if lls_folder else '')
                self.phins_nav_path.set(phins_nav_file if phins_nav_file else '')
            else:
                self.lls_path.set('')
                self.phins_nav_path.set('')
                
            if nav_selected or nav_to_shp_selected or turbidity_selected:
                # Set navigation directory from CSV if provided
                if nav_directory and os.path.exists(nav_directory):
                    self.nav_directory_path.set(nav_directory)
                    self.log_message(f"Job {job_num}: Navigation directory: {nav_directory}")
                else:
                    self.nav_directory_path.set('')
                    if nav_directory:
                        self.log_message(f"Job {job_num}: ⚠ Navigation directory not found: {nav_directory}")
                
                self.nav_plot_file_path.set(nav_state_file if nav_state_file and os.path.exists(nav_state_file) else '')
                # Set PHINS INS file path for navigation processing (separate from LLS)
                self.phins_ins_path.set(phins_ins_file if phins_ins_file and os.path.exists(phins_ins_file) else '')
            else:
                self.nav_directory_path.set('')
                self.nav_plot_file_path.set('')
                self.phins_ins_path.set('')
                
            self.output_path.set(output_folder)
            
            # Process this job using the same structure as single processing
            try:
                # Process Navigation data first if selected
                if nav_selected:
                    if self.check_stop_flag():
                        return False
                        
                    self.log_message(f"Job {job_num}: Processing Navigation data...")
                    try:
                        self.process_navigation_data(output_folder)
                        self.log_message(f"Job {job_num}: ✓ Navigation processing completed")
                    except Exception as nav_error:
                        self.log_message(f"Job {job_num}: ✗ Navigation processing failed: {nav_error}")
                        self.log_message(f"Job {job_num}: Navigation Traceback: {traceback.format_exc()}")

                # Export nav track to shapefile if selected
                if nav_to_shp_selected:
                    if self.check_stop_flag():
                        return False

                    self.log_message(f"Job {job_num}: Exporting Nav Track to Shapefile...")
                    try:
                        self.process_nav_shapefile(output_folder)
                        self.log_message(f"Job {job_num}: ✓ Nav track shapefile export completed")
                    except Exception as shp_error:
                        self.log_message(f"Job {job_num}: ✗ Nav track shapefile export failed: {shp_error}")
                        self.log_message(f"Job {job_num}: Shapefile Traceback: {traceback.format_exc()}")

                # Process turbidity data if selected
                if turbidity_selected:
                    if self.check_stop_flag():
                        return False

                    self.log_message(f"Job {job_num}: Processing Turbidity data...")
                    try:
                        self.process_turbidity_data(output_folder)
                        self.log_message(f"Job {job_num}: ✓ Turbidity processing completed")
                    except Exception as turb_error:
                        self.log_message(f"Job {job_num}: ✗ Turbidity processing failed: {turb_error}")
                        self.log_message(f"Job {job_num}: Turbidity Traceback: {traceback.format_exc()}")
                
                # Process LLS data if selected
                if lls_selected:
                    if self.check_stop_flag():
                        return False
                        
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
                    if self.check_stop_flag():
                        return False
                        
                    if not input_folder or not os.path.exists(input_folder):
                        self.log_message(f"Job {job_num}: ✗ Input folder not found: {input_folder}")
                    else:
                        # Process imagery processing
                        self.log_message(f"Job {job_num}: Processing Imagery data...")
                        
                        # Don't load navigation data here - let analyze_images handle it
                        # Just ensure the nav_path is set correctly (which we did earlier)
                        
                        # Call the main imagery processing method
                        # Pass flag to skip navigation processing since we already did it in batch mode
                        self.analyze_images(
                            input_folder,
                            output_folder,
                            skip_nav_processing=(nav_selected or nav_to_shp_selected or turbidity_selected),
                        )
                        
                        # Check if processing was stopped during imagery processing
                        if self.check_stop_flag():
                            return False

                        imagery_fails = getattr(self, '_last_imagery_failed_stages', 0)
                        if imagery_fails == 0:
                            self.log_message(f"Job {job_num}: ✓ Imagery processing completed")
                        else:
                            self.log_message(f"Job {job_num}: ⚠ Imagery processing completed with {imagery_fails} stage failure(s)")
                
                # Check final stop flag
                if self.check_stop_flag():
                    return False

                imagery_fails = getattr(self, '_last_imagery_failed_stages', 0)
                if imagery_fails == 0:
                    self.log_message(f"Job {job_num}: ✓ All processing completed successfully")
                    return True
                else:
                    self.log_message(f"Job {job_num}: ⚠ Processing completed with {imagery_fails} imagery stage failure(s)")
                    return False
                
            except Exception as processing_error:
                self.log_message(f"Job {job_num}: ✗ Processing error: {processing_error}")
                self.log_message(f"Job {job_num}: Traceback: {traceback.format_exc()}")
                return False
            
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
            turbidity_selected = (
                hasattr(self, 'turbidity_plot_var')
                and self.turbidity_plot_var.get()
            )
            nav_to_shp_selected = (
                hasattr(self, 'nav_to_shp_var')
                and self.nav_to_shp_var.get()
            )
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            if not nav_selected and not turbidity_selected and not nav_to_shp_selected and not lls_selected and not imagery_selected:
                self.log_message("❌ Error: No processing functions selected")
                return False
            
            # Validate output folder
            output_folder = self.output_path.get().strip()
            if not output_folder:
                self.log_message("❌ Error: No output folder specified")
                return False
            
            # Validate navigation-directory-driven inputs
            nav_dir_tasks_selected = nav_selected or turbidity_selected or nav_to_shp_selected
            if nav_dir_tasks_selected:
                nav_mode = self.nav_merge_mode.get()

                if nav_mode == 'directory':
                    # Directory mode validation
                    nav_directory = self.nav_directory_path.get().strip()

                    if not nav_directory:
                        self.log_message("❌ Error: Navigation-directory processing selected but no navigation directory specified")
                        self.log_message("   Please select a directory containing navigation files")
                        return False

                    if not os.path.exists(nav_directory):
                        self.log_message(f"❌ Error: Navigation directory does not exist: {nav_directory}")
                        return False

                    if not os.path.isdir(nav_directory):
                        self.log_message(f"❌ Error: Navigation path is not a directory: {nav_directory}")
                        return False

                    # Full nav plotting requires mergeable nav files; turbidity/shapefile can run
                    # from any supported nav-directory sources.
                    if nav_selected:
                        try:
                            from src.models.nav_merger import scan_navigation_directory
                            nav_files = scan_navigation_directory(nav_directory)

                            if not nav_files:
                                self.log_message("❌ Error: No valid navigation files found in selected directory")
                                self.log_message(f"   Directory: {nav_directory}")
                                self.log_message("   Expected files: PHINS INS, NAV_STATE, STATE, ADCP, or *_Veh_Data files")
                                return False

                            file_types = list(nav_files.keys())
                            self.log_message(f"✅ Navigation directory contains: {', '.join(file_types).upper()}")

                        except Exception as e:
                            self.log_message(f"❌ Error: Could not scan navigation directory: {e}")
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
            turbidity_selected = (
                hasattr(self, 'turbidity_plot_var')
                and self.turbidity_plot_var.get()
            )
            nav_to_shp_selected = (
                hasattr(self, 'nav_to_shp_var')
                and self.nav_to_shp_var.get()
            )
            lls_selected = self.lls_processing_var.get()
            imagery_selected = any([
                self.basic_metrics_var.get(),
                self.location_map_var.get(),
                self.histogram_var.get(),
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
            ])
            
            if not nav_selected and not turbidity_selected and not nav_to_shp_selected and not lls_selected and not imagery_selected:
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
                optional_columns = [
                    'Image_Input',
                    'LLS_Input',
                    'PhinsData_Bin_file',
                    'nav_directory',
                    'NAV_STATE_file',
                    'PHINS_INS_file',
                ]
                
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
    
    def load_navigation_data_for_imagery_only(self, nav_file=None):
        """Wire the nav directory into imagery processors (footprint_map).
        The nav_file argument is kept for backward compatibility but is no
        longer used; the Navigation Directory drives everything.
        """
        try:
            nav_dir = (self.nav_directory_path.get()
                       if hasattr(self, 'nav_directory_path') else '')

            if nav_dir and os.path.isdir(nav_dir):
                if hasattr(self, 'footprint_map') and self.footprint_map:
                    self.footprint_map.nav_directory = nav_dir
                    self.footprint_map.nav_data = None   # will load on first use
                    self.log_message(f"       ✓ Footprint map wired to nav directory: {os.path.basename(nav_dir)}")
                return True
            else:
                self.log_message("       ⚠ No navigation directory set for imagery")
                return False
        except Exception as e:
            self.log_message(f"       ⚠ Navigation wiring error: {e}")
            return False