import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import queue
import builtins
from tkinter import scrolledtext
from typing import Optional
import traceback
from models.metrics import Metrics
from models.altitude_map import GEOPANDAS_AVAILABLE, AltitudeMap
from src.models.footprint_map import FootprintMap
from src.models.visibility_analyzer import VisibilityAnalyzer
from src.models.highlight_selector import HighlightSelector
from datetime import datetime
import numpy as np
import time
import pandas as pd
import csv
import platform
import subprocess

class AppWindow:
    def __init__(self, root):
        self.root = root
        self.master = root  # For compatibility with some tkinter code
        
        # Create a custom style for the process button
        style = ttk.Style()
        style.configure("AccentButton.TButton", font=('', 10, 'bold'))
        
        # Initialize AltitudeMap
        self.altitude_map = AltitudeMap()
        
        # Make sure metrics object exists
        self.metrics = None  # Will be initialized later
        
        # Set default thresholds
        self.altitude_threshold = 8.0
        self.low_altitude_threshold = 4.0
        
        # Initialize variables
        self.all_var = tk.BooleanVar(value=True)
        self.basic_metrics_var = tk.BooleanVar(value=True)
        self.location_map_var = tk.BooleanVar(value=True)
        self.histogram_var = tk.BooleanVar(value=True)
        self.footprint_map_var = tk.BooleanVar(value=True)
        self.visibility_analyzer_var = tk.BooleanVar(value=True)
        self.highlight_selector_var = tk.BooleanVar(value=True)
        self.nav_path = tk.StringVar()
        self.threshold_var = tk.StringVar(value="8.0")  # String for text entry
        self.batch_mode = False
        self.batch_csv_path = tk.StringVar()
        
        # Set all functions to True if all individual options are True
        if all([self.basic_metrics_var.get(), self.location_map_var.get(),
                self.histogram_var.get(), self.footprint_map_var.get()]):
            self.all_var.set(True)
        else:
            self.all_var.set(False)
        
        # Create GUI elements
        self.create_frames()
        
        # Initialize components
        self.metrics = Metrics(self.altitude_threshold)
        self.altitude_map = AltitudeMap()
        self.altitude_map.set_altitude_thresholds(self.altitude_threshold, 4.0)
        self.footprint_map = FootprintMap(self.altitude_threshold)
        
        # Initialize highlight selector
        self.highlight_selector = HighlightSelector()
        
        # Create metrics, altitude map and footprint map objects with the threshold
        self.metrics = Metrics(self.altitude_threshold)
        self.altitude_map.set_altitude_thresholds(self.altitude_threshold, 4.0)  # Set both thresholds
        self.footprint_map = FootprintMap()
        self.footprint_map.altitude_threshold = self.altitude_threshold
        
        # Add visibility analyzer
        self.visibility_analyzer = VisibilityAnalyzer()
        self.visibility_analyzer.altitude_threshold = self.altitude_threshold
        
        # Create the UI components in the correct order
        # First create the log section so log_message works
        self.create_log_section()
        
        # Then create the rest of the UI
        self.create_input_section()
        self.create_functions_section()
        
        # Initialize the UI update queue and thread
        self.setup_ui_update_thread()

    def create_frames(self):
        """Create the main frames for the application with scrolling support"""
        # Set width for left frame with increased width
        LEFT_FRAME_WIDTH = 550  # Increased from 500 to 550
        
        # Create main container frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame that contains the canvas and scrollbar
        canvas_container = ttk.Frame(main_frame)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Create Canvas with scrollbar
        self.left_canvas = tk.Canvas(canvas_container, width=LEFT_FRAME_WIDTH, borderwidth=0)
        left_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.left_canvas.yview)
        
        # Pack canvas and scrollbar
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Create the main frame inside the canvas
        self.left_frame = ttk.Frame(self.left_canvas, width=LEFT_FRAME_WIDTH)
        
        # Create window in canvas that contains the frame
        self.left_canvas_window = self.left_canvas.create_window(
            (0, 0),  # Coordinates
            window=self.left_frame,
            anchor="nw",
            width=LEFT_FRAME_WIDTH  # Important: ensure the frame has a fixed width
        )
        
        # Configure bindings to handle resizing
        self.left_frame.bind("<Configure>", self._on_frame_configure)
        self.left_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Allow mousewheel scrolling
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Right frame for log output and progress (no scroll needed)
        self.right_frame = ttk.Frame(main_frame, padding="5")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create log frame inside the right frame
        self.log_frame = ttk.Frame(self.right_frame)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create sub-frames in left frame with additional padding
        self.input_frame = ttk.LabelFrame(self.left_frame, text="Input/Output", padding="10")  # Increased from 5 to 10
        self.input_frame.pack(fill=tk.X, pady=(5, 10), padx=10)  # Added padx=10
        
        self.functions_frame = ttk.LabelFrame(self.left_frame, text="Analysis Functions", padding="10")  # Increased from 5 to 10
        self.functions_frame.pack(fill=tk.X, pady=(0, 10), padx=10)  # Added padx=10

    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Update the width of the window to fit the canvas"""
        # Update the width to fill the canvas
        canvas_width = event.width
        self.left_canvas.itemconfig(self.left_canvas_window, width=canvas_width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        # Get the pointer position relative to the screen
        x_screen, y_screen = self.root.winfo_pointerxy()
        
        # Find the widget under the pointer
        widget_under_pointer = self.root.winfo_containing(x_screen, y_screen)
        
        # Only scroll if mouse is over the canvas or its children
        if widget_under_pointer and (widget_under_pointer == self.left_canvas or 
                                    self.left_canvas.winfo_ismapped() and
                                    widget_under_pointer.winfo_toplevel() == self.root):
            # Platform specific scrolling (direction and speed)
            if event.delta > 0:
                self.left_canvas.yview_scroll(-1, "units")
            else:
                self.left_canvas.yview_scroll(1, "units")

    def toggle_all_functions(self):
        """Toggle all function checkboxes based on 'All Functions' checkbox"""
        state = self.all_var.get()
        self.basic_metrics_var.set(state)
        self.location_map_var.set(state)
        self.histogram_var.set(state)
        self.footprint_map_var.set(state)
        self.visibility_analyzer_var.set(state)
        self.highlight_selector_var.set(state)
        
        # Show/hide visibility options based on checkbox state
        if state:
            self.toggle_visibility_options()
        else:
            self.visibility_model_frame.grid_remove()

    def update_all_checkbox(self):
        """Update 'All Functions' checkbox based on individual checkboxes"""
        if all([self.basic_metrics_var.get(), 
                self.location_map_var.get(), 
                self.histogram_var.get(), 
                self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(),
                self.highlight_selector_var.get()
               ]):
            self.all_var.set(True)
        else:
            self.all_var.set(False)

    def select_input_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.input_path.set(folder_path)
            self.log_message(f"Input folder set to: {folder_path}")

    def select_output_folder(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_path.set(folder_path)
            self.log_message(f"Output folder set to: {folder_path}")

    def select_nav_file(self):
        """Select vehicle navigation file"""
        file_path = filedialog.askopenfilename(
            title="Select Vehicle Navigation File",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.nav_path.set(file_path)
            self.log_message(f"Vehicle navigation file set to: {file_path}")

    def select_visibility_file(self, file_type):
        """Select model file or training data directory for visibility analyzer"""
        if file_type == "model":
            # Get the default models directory
            default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                       "v_a_pre-trained_models")
            os.makedirs(default_dir, exist_ok=True)
            
            file_path = filedialog.askopenfilename(
                title="Select Pre-trained Visibility Model",
                initialdir=default_dir,
                filetypes=[("Model files", "*.h5"), ("All files", "*.*")]
            )
            if file_path:
                self.model_path.set(file_path)
                self.training_path.set("")  # Clear the other path
                self.model_type_var.set("model")
                self.log_message(f"Visibility model file set to: {file_path}")
        else:  # training data
            folder_path = filedialog.askdirectory(
                title="Select Visibility Training Data Directory"
            )
            if folder_path:
                self.training_path.set(folder_path)
                self.model_path.set("")  # Clear the other path
                self.model_type_var.set("training")
                self.log_message(f"Visibility training data directory set to: {folder_path}")

    def update_nav_file_visibility(self):
        """Navigation file is now always visible in the input section"""
        pass

    def toggle_nav_options(self):
        """Navigation file is now always visible in the input section"""
        pass

    def toggle_visibility_options(self):
        """Show or hide visibility model options based on checkbox state"""
        if self.visibility_analyzer_var.get():
            # Position it below the note, but before highlight selector checkbox
            self.visibility_model_frame.grid(row=self.visibility_model_row, column=0, 
                                             columnspan=3, sticky='ew', padx=45, pady=(0, 5))
        else:
            # Hide the options when unchecked
            self.visibility_model_frame.grid_remove()
        
        # Update the "All" checkbox state
        self.update_all_checkbox()

    def setup_ui_update_thread(self):
        """Set up a queue and thread for updating the UI from background threads"""
        self.ui_queue = queue.Queue()
        
        # Define the function to process the queue
        def process_ui_queue():
            try:
                # Process all items currently in the queue
                while not self.ui_queue.empty():
                    func, args = self.ui_queue.get_nowait()
                    func(*args)
                    self.ui_queue.task_done()
            except Exception as e:
                print(f"Error processing UI queue: {str(e)}")
            finally:
                # Schedule to run again
                self.root.after(100, process_ui_queue)
        
        # Start the queue processing
        self.root.after(100, process_ui_queue)

    def log_message(self, message):
        """
        Add a message to the log text
        
        Args:
            message: Message to log
        """
        # Format with timestamp
        timestamp = time.strftime("[%H:%M:%S]")
        log_entry = f"{timestamp} {message}"
        
        # Print to console immediately
        print(log_entry)
        
        # Add to UI queue if we're in a background thread
        if threading.current_thread() is not threading.main_thread():
            self.ui_queue.put((self._update_log_text, (log_entry,)))
        else:
            # We're in the main thread, update directly
            self._update_log_text(log_entry)

    def _update_log_text(self, log_entry):
        """Helper method to update the log text widget"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry + "\n")
        self.log_text.see(tk.END)  # Scroll to the end
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
        # Disable the process button to prevent multiple clicks
        self.process_button.configure(state=tk.DISABLED)
        
        if self.batch_mode:
            # Use batch processing
            threading.Thread(target=self.process_batch, daemon=True).start()
            return
        
        # Single mode processing
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()
        
        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Please select a valid input folder.")
            self.process_button.configure(state=tk.NORMAL)
            return
            
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            self.process_button.configure(state=tk.NORMAL)
            return
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Log the start of processing
        self.log_message(f"Processing images from {input_folder}")
        self.log_message(f"Output folder: {output_folder}")
        
        # Process images in a background thread
        threading.Thread(
            target=self._process_images_thread,
            args=(input_folder, output_folder),
            daemon=True
        ).start()

    def _process_images_thread(self, input_folder, output_folder):
        """Background thread function for processing images"""
        try:
            self.analyze_images(input_folder, output_folder)
            
            # Show completion message
            self.log_message("\nAll selected processing tasks completed successfully.")
            self.update_progress(100, "Processing complete!")
            
            # Play completion sound
            try:
                # Try using winsound first
                try:
                    import winsound
                    sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                            "utils", "sounds", "beer_open.wav")
                    if os.path.exists(sound_path):
                        winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                    else:
                        # Fall back to mp3 with os.system
                        sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                "utils", "sounds", "beer_open.mp3")
                        if os.path.exists(sound_path):
                            os.system(f'start "" "{sound_path}"')
                except Exception as sound_err:
                    # Fall back to original method
                    sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                            "utils", "sounds", "beer_open.mp3")
                    if os.path.exists(sound_path):
                        os.system(f'start "" "{sound_path}"')
            except Exception as e:
                print(f"Could not play sound: {str(e)}")
                
        except Exception as e:
            self.log_message(f"\nError during processing: {str(e)}")
            self.update_progress(0, "Error during processing")
            traceback.print_exc()

    def analyze_images(self, input_folder, output_folder):
        """Process images based on selected functions"""
        try:
            self.update_progress(0, "Starting processing...")
            
            # Update the threshold from the entry field
            try:
                new_threshold = float(self.threshold_var.get())
                self.altitude_threshold = new_threshold
                
                # Update all components with the new threshold
                self.metrics.altitude_threshold = new_threshold
                self.altitude_map.set_altitude_thresholds(new_threshold, self.low_altitude_threshold)
                self.footprint_map.altitude_threshold = new_threshold
                if hasattr(self, 'visibility_analyzer'):
                    self.visibility_analyzer.altitude_threshold = new_threshold
            except ValueError:
                self.log_message(f"Invalid threshold value. Using default: {self.altitude_threshold}")
            
            # Load navigation data first if provided - this is used by multiple functions
            nav_path = self.nav_path.get()
            if nav_path and os.path.exists(nav_path):
                self.log_message(f"Loading navigation data from: {nav_path}")
                self.update_progress(5, "Loading navigation data...")
                
                # Load navigation data into metrics object for altitude information
                if self.metrics.load_nav_data(nav_path):
                    self.log_message("Navigation data loaded successfully for altitude extraction")
                    self.log_message("NOTE: Using ONLY navigation data for altitude values, EXIF altitude data is ignored")
                    
                    # Make the nav data available to other components
                    if hasattr(self.metrics, 'nav_timestamps'):
                        self.altitude_map.nav_timestamps = self.metrics.nav_timestamps
                        self.footprint_map.nav_timestamps = self.metrics.nav_timestamps
                        
                        if hasattr(self, 'visibility_analyzer'):
                            self.visibility_analyzer.nav_timestamps = self.metrics.nav_timestamps
                        if hasattr(self, 'highlight_selector'):
                            self.highlight_selector.nav_timestamps = self.metrics.nav_timestamps
                else:
                    self.log_message("Warning: Failed to load navigation data for altitude extraction")
        
            # Determine if we need to extract GPS data
            extract_gps = self.location_map_var.get() or self.histogram_var.get() or self.footprint_map_var.get() or self.visibility_analyzer_var.get()
            
            # STAGE 1: Basic Metrics (Always run this first to extract GPS data)
            self.log_message("STAGE 1: Processing basic metrics...")
            
            try:
                # Calculate metrics with progress updates
                _, results = self.metrics.analyze_directory(
                    input_folder, 
                    progress_callback=self.update_progress,
                    extract_gps=extract_gps  # Extract GPS data if needed for later
                )
                
                # If basic metrics was selected, show the results
                if self.basic_metrics_var.get():
                    # Log results to the GUI
                    for line in results:
                        self.log_message(line)
                    
                    # Write results to output file
                    output_file = os.path.join(output_folder, "image_metrics.txt")
                    with open(output_file, "w") as f:
                        f.write("\n".join(results))
                    
                    self.log_message(f"Basic metrics saved to: {output_file}")
                else:
                    self.log_message("Basic metrics processing completed (not displayed as not selected)")
            except Exception as e:
                self.log_message(f"Error processing basic metrics: {str(e)}")
                raise  # Re-raise to be caught by outer try-except
            
            # STAGE 2: Location Map and Altitude Histogram
            if self.location_map_var.get() or self.histogram_var.get():
                self.log_message("\nSTAGE 2: Creating location map and altitude histogram...")

                # Check if we have GPS data
                if not self.metrics.gps_data:
                    self.log_message("Warning: No GPS data found in images. Cannot create location map or altitude histogram.")
                else:
                    # Step 2A: Create location map if selected
                    if self.location_map_var.get():
                        self.log_message("Creating image location map...")
                        
                        try:
                            # Generate the location map
                            self.log_message(f"Generating map with {len(self.metrics.gps_data)} GPS points...")
                            
                            map_file = self.altitude_map.create_location_map(
                                self.metrics.gps_data,
                                output_folder,
                                metrics=self.metrics
                            )
                            
                            if map_file:
                                self.log_message(f"Location map saved to: {map_file}")
                            else:
                                self.log_message("Error: Could not create location map. No valid GPS coordinates found.")
                                
                            # Create CSV and shapefile export using AltitudeMap's export_to_gis_formats method
                            self.log_message("Creating GPS data exports (CSV and shapefile)...")

                            try:
                                # Use the export_to_gis_formats method from AltitudeMap
                                result_files = self.altitude_map.export_to_gis_formats(
                                    self.metrics.gps_data,
                                    output_folder
                                )
                                
                                if 'csv' in result_files:
                                    self.log_message(f"CSV export saved to: {result_files['csv']}")
                                    
                                if 'shapefile' in result_files and GEOPANDAS_AVAILABLE:
                                    self.log_message(f"Shapefile export saved to: {result_files['shapefile']}")
                                    
                            except Exception as e:
                                # Fall back to the simple CSV export if the GIS export fails
                                self.log_message(f"Error creating GIS exports: {str(e)}")
                                
                                try:
                                    csv_path = os.path.join(output_folder, "image_locations.csv")
                                    import pandas as pd
                                    df = pd.DataFrame(self.metrics.gps_data)
                                    
                                    # Define column order before saving
                                    columns = ["filename", "DateTime", "latitude", "longitude", "altitude", 
                                              "SubjectDistance", "ExposureTime", "FNumber", "FocalLength",
                                              "width", "height"]
                                              
                                    # Only include columns that exist in the dataframe
                                    available_cols = [col for col in columns if col in df.columns]
                                    other_cols = [col for col in df.columns if col not in columns]
                                    
                                    # Reorder columns
                                    df = df[available_cols + other_cols]
                                    
                                    df.to_csv(csv_path, index=False)
                                    self.log_message(f"CSV export saved to: {csv_path}")
                                except Exception as csv_err:
                                    self.log_message(f"Error creating CSV export: {str(csv_err)}")
                        except Exception as e:
                            self.log_message(f"Error creating location map: {str(e)}")
                    
                    # Step 2B: Create altitude histogram if selected
                    if self.histogram_var.get():
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
            
            # STAGE 3: Footprint and Overlap Analysis
            if self.footprint_map_var.get():
                self.log_message("\nSTAGE 3: Creating image footprints and overlap maps...")
                self.update_progress(60, "Creating image footprints and overlap maps...")
                
                try:
                    # Pass the same navigation data to footprint map
                    nav_path = self.nav_path.get()
                    self.footprint_map.altitude_threshold = self.altitude_threshold
                    
                    footprint_map_path = self.footprint_map.create_footprint_map(
                        self.metrics.gps_data, 
                        output_folder,
                        nav_file_path=nav_path,  # Use the nav path from the input section
                        filename="image_footprints_map.png"
                    )
                    
                    if footprint_map_path:
                        self.log_message(f"Image footprint map created: {footprint_map_path}")
                        
                        # Copy overlap stats from footprint map to metrics for report generation
                        if hasattr(self.footprint_map, 'vertical_overlap_stats'):
                            self.metrics.vertical_overlap_stats = self.footprint_map.vertical_overlap_stats
                        if hasattr(self.footprint_map, 'horizontal_overlap_stats'):
                            self.metrics.horizontal_overlap_stats = self.footprint_map.horizontal_overlap_stats
                        if hasattr(self.footprint_map, 'overall_overlap_stats'):
                            self.metrics.overall_overlap_stats = self.footprint_map.overall_overlap_stats
                            
                        # Append overlap metrics to the metrics file
                        self.append_overlap_metrics_to_file(output_folder)
                    else:
                        self.log_message("Failed to create footprint map")
                        
                except Exception as e:
                    self.log_message(f"Error creating footprint map: {str(e)}")
                    self.log_message(traceback.format_exc())
            
            # STAGE 4: Visibility Analysis (last step)
            if self.visibility_analyzer_var.get():
                self.log_message("\nSTAGE 4: Running visibility analysis...")
                
                # Initialize visibility analyzer if needed
                if not hasattr(self, 'visibility_analyzer'):
                    from src.models.visibility_analyzer import VisibilityAnalyzer
                    self.visibility_analyzer = VisibilityAnalyzer(self.altitude_threshold)
                else:
                    # Update the altitude threshold in case it was changed
                    self.visibility_analyzer.altitude_threshold = self.altitude_threshold
                
                # Try to check TensorFlow without importing
                tf_available = self.visibility_analyzer._import_tensorflow()
                            
                if not tf_available:
                    self.log_message("WARNING: TensorFlow not available. Cannot perform visibility analysis.")
                    self.log_message("Please install TensorFlow with: pip install tensorflow")
                else:
                    try:
                        # Get the path based on radio button selection
                        if self.model_type_var.get() == "model":
                            model_path = self.model_path.get()
                            if not model_path or not os.path.exists(model_path):
                                self.log_message("Error: Please select a valid visibility model file.")
                                return
                        else:  # training
                            model_path = self.training_path.get()
                            if not model_path or not os.path.exists(model_path):
                                self.log_message("Error: Please select a valid training data directory.")
                                return
                        
                        # Load or train model
                        self.log_message(f"Loading or training visibility model from: {model_path}")
                        success = self.visibility_analyzer.load_or_train_model(
                            model_path, 
                            progress_callback=self.update_progress
                        )
                        
                        if not success:
                            self.log_message("Error: Could not load or train visibility model.")
                        else:
                            # If we trained a new model, save it
                            if os.path.isdir(model_path):  # model_path is a training data directory
                                # Create a models directory in the application root
                                app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                                models_dir = os.path.join(app_root, "utils", "models")
                                os.makedirs(models_dir, exist_ok=True)
                                
                                # Create a timestamped filename for the model
                                import datetime
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                model_filename = f"visibility_model_{timestamp}.h5"
                                model_save_path = os.path.join(models_dir, model_filename)
                                
                                # Save the model
                                saved_path = self.visibility_analyzer.save_model(model_save_path)
                                if saved_path:
                                    self.log_message(f"Trained model saved to: {saved_path}")
                                    
                                    # Also save a copy to the output folder for this specific run
                                    output_model_path = os.path.join(output_folder, "visibility_model.h5")
                                    self.visibility_analyzer.save_model(output_model_path)
                                    self.log_message(f"Model copy saved to output folder: {output_model_path}")
                                else:
                                    self.log_message("Warning: Could not save trained model")
                            
                            # Patch the log_message method for visibility analyzer
                            # Store original log_message if exists
                            if hasattr(self.visibility_analyzer, 'log_message'):
                                original_log_message = self.visibility_analyzer.log_message
                            
                            # Create a properly-scoped log message function
                            def log_message_adapter(msg, progress=None):
                                self.log_message(msg)
                                if progress is not None:
                                    self.update_progress(progress, msg)
                            
                            # Replace the log_message in visibility_analyzer
                            self.visibility_analyzer.log_message = log_message_adapter
                            
                            # The input folder contains the images to analyze
                            self.log_message(f"Starting visibility analysis with altitude threshold: {self.altitude_threshold}m")
                            self.log_message("This will process all images below the threshold (up to 5000)")
                            
                            # Let the analyzer find all images - don't pre-filter or limit here
                            success, stats = self.visibility_analyzer.analyze_images(
                                [],  # Empty list means it will scan the input folder
                                output_folder,
                                progress_callback=self.update_progress,
                                altitude_threshold=self.altitude_threshold
                            )
                            
                            # Restore original log_message if it existed
                            if hasattr(self.visibility_analyzer, 'original_log_message'):
                                self.visibility_analyzer.log_message = original_log_message
                            
                            if not success:
                                self.log_message("Error during visibility analysis.")
                            else:
                                # Show results
                                self.log_message(f"Visibility analysis complete. Results saved to {output_folder}")
                                
                                # Display summary in log
                                for line in self.visibility_analyzer.get_summary_report():
                                    self.log_message(line)
                                
                                # Append to metrics file
                                self.append_visibility_metrics_to_file(output_folder)
                    except Exception as e:
                        self.log_message(f"Error during visibility analysis: {str(e)}")
                        import traceback
                        self.log_message(traceback.format_exc())
            
            # STAGE 5: Highlight Image Selection
            if self.highlight_selector_var.get():
                self.log_message("\nSTAGE 5: Selecting highlight images...")
                
                success = self.select_highlight_images(input_folder, output_folder)
                
                if success:
                    self.log_message("Highlight image selection completed successfully")
                else:
                    self.log_message("Error during highlight image selection")
            
            self.log_message("\nAll selected processing tasks completed successfully.")
            
            # Play completion sound
            self.play_completion_sound()
            
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}")
            import traceback
            self.log_message(f"Details: {traceback.format_exc()}")
        finally:
            # Re-enable the process button
            self.master.after(0, lambda: self.process_button.configure(state=tk.NORMAL))

    def append_overlap_metrics_to_file(self, output_folder: str):
        """
        Append overlap metrics to the existing metrics text file
        
        Args:
            output_folder: Directory where the metrics file is located
        """
        metrics_file = os.path.join(output_folder, "image_metrics.txt")
        
        if not os.path.exists(metrics_file):
            self.log_message("Warning: Metrics file not found. Creating new file with overlap metrics.")
            
        try:
            # Get only the overlap-related metrics
            overlap_metrics = []
            
            if hasattr(self.metrics, 'vertical_overlap_stats') and self.metrics.vertical_overlap_stats:
                overlap_metrics.append("\n\n--- OVERLAP METRICS ---\n")
                overlap_metrics.append("\nVertical Overlap Statistics (Sequential Images):")
                overlap_metrics.append("-" * 45)
                overlap_metrics.append(f"Average overlap: {self.metrics.vertical_overlap_stats['avg_overlap']:.1f}%")
                overlap_metrics.append(f"Median overlap: {self.metrics.vertical_overlap_stats['median_overlap']:.1f}%")
                overlap_metrics.append(f"Overlap range: {self.metrics.vertical_overlap_stats['min_overlap']:.1f}% to {self.metrics.vertical_overlap_stats['max_overlap']:.1f}%")
                overlap_metrics.append(f"Low overlap (<40%): {self.metrics.vertical_overlap_stats['low_overlap']} image pairs")
                overlap_metrics.append(f"Medium overlap (40-70%): {self.metrics.vertical_overlap_stats['medium_overlap']} image pairs")
                overlap_metrics.append(f"High overlap (>70%): {self.metrics.vertical_overlap_stats['high_overlap']} image pairs")
                overlap_metrics.append(f"Total image pairs analyzed: {self.metrics.vertical_overlap_stats['total_overlap_count']}")
            
            if hasattr(self.metrics, 'horizontal_overlap_stats') and self.metrics.horizontal_overlap_stats:
                overlap_metrics.append("\nHorizontal Overlap Statistics (Between Survey Lines):")
                overlap_metrics.append("-" * 50)
                overlap_metrics.append(f"Average overlap: {self.metrics.horizontal_overlap_stats['avg_overlap']:.1f}%")
                overlap_metrics.append(f"Median overlap: {self.metrics.horizontal_overlap_stats['median_overlap']:.1f}%")
                overlap_metrics.append(f"Overlap range: {self.metrics.horizontal_overlap_stats['min_overlap']:.1f}% to {self.metrics.horizontal_overlap_stats['max_overlap']:.1f}%")
                overlap_metrics.append(f"Low overlap (<10%): {self.metrics.horizontal_overlap_stats['low_overlap']} image pairs")
                overlap_metrics.append(f"Medium overlap (10-40%): {self.metrics.horizontal_overlap_stats['medium_overlap']} image pairs")
                overlap_metrics.append(f"High overlap (>40%): {self.metrics.horizontal_overlap_stats['high_overlap']} image pairs")
                overlap_metrics.append(f"Total image pairs with overlap: {self.metrics.horizontal_overlap_stats['total_overlap_count']} (of {self.metrics.horizontal_overlap_stats.get('total_checks', 'N/A')} possible pairs)")
            
            if hasattr(self.metrics, 'overall_overlap_stats') and self.metrics.overall_overlap_stats:
                overlap_metrics.append("\nOverall Overlap Statistics (All Overlapping Images):")
                overlap_metrics.append("-" * 49)
                overlap_metrics.append(f"Average overlapping images: {self.metrics.overall_overlap_stats['avg_count']:.1f} per footprint")
                overlap_metrics.append(f"Median overlapping images: {self.metrics.overall_overlap_stats['median_count']:.1f} per footprint")
                overlap_metrics.append(f"Range: {self.metrics.overall_overlap_stats['min_count']} to {self.metrics.overall_overlap_stats['max_count']} overlapping images")
                overlap_metrics.append(f"Total overlapping pairs: {self.metrics.overall_overlap_stats['total_overlaps']} (of {self.metrics.overall_overlap_stats.get('total_checks', 'N/A')} possible pairs)")
            
            # Only proceed if we have overlap metrics to append
            if overlap_metrics:
                with open(metrics_file, 'a') as f:
                    f.write("\n".join(overlap_metrics))
                self.log_message("Overlap metrics appended to image_metrics.txt")
            
        except Exception as e:
            self.log_message(f"Error appending overlap metrics to file: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())

    def append_visibility_metrics_to_file(self, output_folder: str):
        """
        Append visibility metrics to the existing metrics text file
        
        Args:
            output_folder: Directory where the metrics file is located
        """
        metrics_file = os.path.join(output_folder, "image_metrics.txt")
        
        if not os.path.exists(metrics_file):
            self.log_message("Warning: Metrics file not found. Creating new file with visibility metrics.")
            
        try:
            # Get visibility metrics
            visibility_metrics = self.visibility_analyzer.get_summary_report()
            
            # Only proceed if we have metrics to append
            if visibility_metrics:
                with open(metrics_file, 'a') as f:
                    f.write("\n\n--- VISIBILITY METRICS ---\n")
                    f.write("\n".join(visibility_metrics))
                self.log_message("Visibility metrics appended to image_metrics.txt")
            
        except Exception as e:
            self.log_message(f"Error appending visibility metrics to file: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())

    def threshold_changed(self, *args):
        """Update the altitude threshold when the slider is moved"""
        try:
            # Get the new threshold value
            new_threshold = float(self.threshold_var.get())
            
            # Update threshold value
            self.altitude_threshold = new_threshold
            
            # Update both map objects with the new threshold
            if hasattr(self.altitude_map, 'set_altitude_thresholds'):
                self.altitude_map.set_altitude_thresholds(new_threshold, 4.0)  # Keep low threshold at 4m
                
            if hasattr(self.footprint_map, 'altitude_threshold'):
                self.footprint_map.altitude_threshold = new_threshold
            
            # Update the threshold label
            self.threshold_label.config(text=f"Altitude Threshold: {new_threshold:.1f}m")
        except Exception as e:
            print(f"Error updating threshold: {e}")

    def create_input_section(self):
        """Create the input/output file selection widgets"""
        # Input folder selection
        self.input_path = tk.StringVar()
        input_frame = ttk.Frame(self.input_frame)
        input_frame.grid(row=0, column=0, sticky='ew', pady=2)
        
        ttk.Label(input_frame, text="Input Folder:").grid(row=0, column=0, sticky='w')
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=40)
        self.input_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.input_button = ttk.Button(input_frame, text="Browse...", command=self.select_input_folder)
        self.input_button.grid(row=0, column=2)
        
        # Output folder selection
        self.output_path = tk.StringVar()
        output_frame = ttk.Frame(self.input_frame)
        output_frame.grid(row=1, column=0, sticky='ew', pady=2)
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky='w')
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=40)
        self.output_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.output_button = ttk.Button(output_frame, text="Browse...", command=self.select_output_folder)
        self.output_button.grid(row=0, column=2)
        
        # Navigation file selection - moved here from functions section
        self.nav_path = tk.StringVar()
        nav_frame = ttk.Frame(self.input_frame)
        nav_frame.grid(row=2, column=0, sticky='ew', pady=2)
        
        ttk.Label(nav_frame, text="Vehicle Nav File:").grid(row=0, column=0, sticky='w')
        self.nav_entry = ttk.Entry(nav_frame, textvariable=self.nav_path, width=40)
        self.nav_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.nav_button = ttk.Button(nav_frame, text="Browse...", command=self.select_nav_file)
        self.nav_button.grid(row=0, column=2)
        
        # Add batch mode toggle button - moved to row 3
        self.batch_button = ttk.Button(
            self.input_frame,
            text="Switch to Batch Mode",
            command=self.toggle_batch_mode
        )
        self.batch_button.grid(row=3, column=0, columnspan=3, pady=(10, 0))

    def create_functions_section(self):
        """Create the functions selection checkbox widgets"""
        # Initialize the current row counter
        self.current_row = 0
        
        # "All Functions" checkbox
        self.all_check = ttk.Checkbutton(
            self.functions_frame,
            text="All Functions",
            variable=self.all_var,
            command=self.toggle_all_functions
        )
        self.all_check.grid(row=self.current_row, column=0, sticky='w', pady=2)
        self.current_row += 1
        
        # Individual functions checkboxes
        self.basic_metrics_check = ttk.Checkbutton(
            self.functions_frame,
            text="Basic Metrics",
            variable=self.basic_metrics_var,
            command=self.update_all_checkbox
        )
        self.basic_metrics_check.grid(row=self.current_row, column=0, sticky='w', padx=25, pady=2)
        self.current_row += 1
        
        self.location_map_check = ttk.Checkbutton(
            self.functions_frame,
            text="Image Location Map",
            variable=self.location_map_var,
            command=self.update_all_checkbox
        )
        self.location_map_check.grid(row=self.current_row, column=0, sticky='w', padx=25, pady=2)
        self.current_row += 1
        
        self.histogram_check = ttk.Checkbutton(
            self.functions_frame,
            text="Image Altitude Histogram",
            variable=self.histogram_var,
            command=self.update_all_checkbox
        )
        self.histogram_check.grid(row=self.current_row, column=0, sticky='w', padx=25, pady=2)
        self.current_row += 1
        
        self.footprint_map_check = ttk.Checkbutton(
            self.functions_frame,
            text="Image Footprint and Overlap Maps",
            variable=self.footprint_map_var,
            command=self.update_all_checkbox
        )
        self.footprint_map_check.grid(row=self.current_row, column=0, sticky='w', padx=25, pady=2)
        self.current_row += 1
        
        # Visibility analyzer checkbox
        self.visibility_analyzer_check = ttk.Checkbutton(
            self.functions_frame,
            text="Image Visibility Analysis",
            variable=self.visibility_analyzer_var,
            command=self.toggle_visibility_options
        )
        self.visibility_analyzer_check.grid(row=self.current_row, column=0, sticky='w', padx=25, pady=2)
        self.current_row += 1

        # Add a note about computational intensity below the visibility checkbox
        self.vis_note_label = ttk.Label(
            self.functions_frame,
            text="(Visibility analysis is computationally intensive and may take a long time)",
            font=("", 8, "italic")
        )
        self.vis_note_label.grid(row=self.current_row, column=0, sticky='w', padx=45, pady=(0, 5))
        self.current_row += 1

        # Make the visibility frame wider to show all options
        # Use a LabelFrame for better visual separation and more space
        self.visibility_model_frame = ttk.LabelFrame(self.functions_frame, text="Visibility Analysis Options")
        # Note: We don't grid it here - it will be positioned by toggle_visibility_options
        self.visibility_model_row = self.current_row  # Store the correct position
        self.current_row += 1  # Still increment for spacing

        # Add highlight image selector checkbox
        self.highlight_selector_check = ttk.Checkbutton(
            self.functions_frame, 
            text="Select Highlight Images", 
            variable=self.highlight_selector_var
        )
        self.highlight_selector_check.grid(row=self.current_row, column=0, sticky='w', padx=25, pady=2)
        self.current_row += 1
        
        # Configure column weights to make the frame expand properly
        self.visibility_model_frame.columnconfigure(1, weight=1)
        
        # Model selection radio buttons
        self.model_type_var = tk.StringVar(value="model")
        
        # Model radio button
        self.model_radio = ttk.Radiobutton(
            self.visibility_model_frame,
            text="Pre-trained Model",
            variable=self.model_type_var,
            value="model",
            command=self.on_model_radio_select
        )
        self.model_radio.grid(row=0, column=0, sticky='w', padx=(10, 5), pady=(5, 2))
        
        # Model path - make entry wider
        self.model_path = tk.StringVar()
        self.model_entry = ttk.Entry(self.visibility_model_frame, textvariable=self.model_path, width=40)
        self.model_entry.grid(row=0, column=1, padx=(5, 5), pady=(5, 2), sticky='ew')
        self.model_button = ttk.Button(
            self.visibility_model_frame, 
            text="Browse...", 
            command=lambda: self.select_visibility_file("model")
        )
        self.model_button.grid(row=0, column=2, padx=(0, 10), pady=(5, 2))
        
        # Training data selection - increase padding for better spacing
        self.training_radio = ttk.Radiobutton(
            self.visibility_model_frame,
            text="Training Data",
            variable=self.model_type_var,
            value="training",
            command=self.on_training_radio_select
        )
        self.training_radio.grid(row=1, column=0, sticky='w', padx=(10, 5), pady=(2, 5))
        
        # Training path - make entry wider
        self.training_path = tk.StringVar()
        self.training_entry = ttk.Entry(self.visibility_model_frame, textvariable=self.training_path, width=40)
        self.training_entry.grid(row=1, column=1, padx=(5, 5), pady=(2, 5), sticky='ew')
        self.training_button = ttk.Button(
            self.visibility_model_frame, 
            text="Browse...", 
            command=lambda: self.select_visibility_file("training")
        )
        self.training_button.grid(row=1, column=2, padx=(0, 10), pady=(2, 5))
        
        # Add Training Guide button inside the visibility model frame - center it better
        self.help_button = ttk.Button(
            self.visibility_model_frame,
            text="Training Guide",
            command=self.show_training_guide
        )
        self.help_button.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Create a separate LabelFrame for the altitude threshold - CLEARLY SEPARATED FROM FUNCTIONS
        # Create threshold frame after functions frame
        self.threshold_frame = ttk.LabelFrame(self.left_frame, text="Altitude Threshold", padding="10")
        self.threshold_frame.pack(fill=tk.X, pady=(10, 10), padx=10)  # Added padx=10
        
        # Create a frame inside the LabelFrame for better layout
        threshold_content = ttk.Frame(self.threshold_frame)
        threshold_content.pack(fill=tk.X, padx=10, pady=5)
        
        # Add altitude threshold controls
        ttk.Label(threshold_content, text="Altitude Threshold (m):").grid(row=0, column=0, sticky='w', padx=(0, 5))
        
        self.threshold_var = tk.StringVar(value=str(self.altitude_threshold))
        threshold_entry = ttk.Entry(threshold_content, textvariable=self.threshold_var, width=6)
        threshold_entry.grid(row=0, column=1, padx=5, sticky='w')
        
        apply_button = ttk.Button(threshold_content, text="Apply", command=self.update_threshold)
        apply_button.grid(row=0, column=2, padx=5, sticky='w')
        
        # Add explanation text for altitude threshold
        ttk.Label(
            threshold_content, 
            text="(Filters out images that are too high to see the seabed)",
            font=("", 8, "italic")
        ).grid(row=1, column=0, columnspan=3, sticky='w', pady=(5, 0))
        
        # Add process button as the last element in the left frame (below everything else)
        process_button_frame = ttk.Frame(self.left_frame)
        process_button_frame.pack(fill=tk.X, pady=(10, 0), padx=10)  # Added padx=10
        
        self.process_button = ttk.Button(
            process_button_frame,
            text="Process Images",
            command=self.process_images,
            style="AccentButton.TButton"  # Apply a custom style to make it stand out
        )
        self.process_button.pack(fill=tk.X, padx=10, pady=10, ipady=5)  # Make button taller with ipady

        # Since visibility analysis is checked by default, show the options immediately
        if self.visibility_analyzer_var.get():
            self.toggle_visibility_options()

        # Since footprint map is checked by default, show nav file selection immediately  
        if self.footprint_map_var.get():
            self.update_nav_file_visibility()

    def create_log_section(self):
        """Create the log output and progress widgets"""
        # Create log text widget within the log frame
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)  # Make read-only
        
        # Create progress bar
        self.progress_frame = ttk.Frame(self.right_frame, padding="5")
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            length=300,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.pack(side=tk.RIGHT, padx=5)
        
        # Add initial log message
        self.log_message("Welcome to VOYIS First Look Metrics")
        self.log_message("Select an input folder with VOYIS images and an output folder for results")
        self.log_message("Then check the desired functions and click 'Process Images'")

    def create_threshold_control(self):
        """Create the altitude threshold entry box and label"""
        threshold_frame = ttk.Frame(self.functions_frame)
        threshold_frame.grid(row=8, column=0, sticky='ew', pady=(10, 5))
        
        ttk.Label(threshold_frame, text="Altitude Threshold (m):").grid(row=0, column=0, sticky='w')
        
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=6)
        threshold_entry.grid(row=0, column=1, padx=5, sticky='w')
        
        # Add a button to apply the threshold
        apply_button = ttk.Button(threshold_frame, text="Apply", command=self.update_threshold)
        apply_button.grid(row=0, column=2, padx=5, sticky='w')
        
        # Add explanation text
        ttk.Label(
            self.functions_frame, 
            text="(Altitude threshold filters out images that are too high to see the seabed)",
            font=("", 8, "italic")
        ).grid(row=9, column=0, sticky='w', pady=(0, 10), padx=25)

    def update_threshold(self):
        """Update altitude threshold when Apply button is clicked"""
        try:
            new_threshold = float(self.threshold_var.get())
            
            if new_threshold <= 0:
                messagebox.showwarning("Invalid Threshold", "Altitude threshold must be a positive number.")
                return
                
            # Update the threshold value
            self.altitude_threshold = new_threshold
            
            # Update all components
            self.metrics.altitude_threshold = new_threshold
            self.altitude_map.set_altitude_thresholds(new_threshold, 4.0)  # Keep low threshold at 4m
            
            # Update the footprint map if it exists
            if hasattr(self, 'footprint_map'):
                self.footprint_map.altitude_threshold = new_threshold
                
            # Update the visibility analyzer if it exists
            if hasattr(self, 'visibility_analyzer'):
                self.visibility_analyzer.altitude_threshold = new_threshold
                
            # Log the change
            self.log_message(f"Altitude threshold set to {new_threshold:.1f}m")
            
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number for the altitude threshold.")
        except Exception as e:
            self.log_message(f"Error updating threshold: {str(e)}")

    def _load_image(self, image_path: str, target_size=(224, 224)) -> Optional[np.ndarray]:
        """Load and preprocess an image for the model"""
        try:
            # Use TensorFlow's method to load the image
            img = self._load_img(image_path, target_size=target_size)
            img_array = self._img_to_array(img)
            # Apply VGG16 preprocessing (normalize for the model)
            img_array = self._preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def show_training_guide(self):
        """Display a guide for setting up training data"""
        # Lazy-load the visibility analyzer if needed
        if not hasattr(self, 'visibility_analyzer'):
            from src.models.visibility_analyzer import VisibilityAnalyzer
            self.visibility_analyzer = VisibilityAnalyzer(self.altitude_threshold)
        
        # Create a dialog window
        guide_window = tk.Toplevel(self.master)
        guide_window.title("Training Data Structure Guide")
        guide_window.geometry("650x450")
        
        # Add text widget for the guide content
        guide_text = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        guide_text.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(guide_text, command=guide_text.yview)
        guide_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add the guide content
        guide_content = """
TRAINING DATA DIRECTORY STRUCTURE GUIDE

Your training data folder should have this structure:

Training_Data/
  ├── zero_visibility/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── low_visibility/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── good_visibility/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── great_visibility/
      ├── image1.jpg
      ├── image2.jpg
      └── ...

IMPORTANT NOTES:

1. Each category subfolder MUST be named exactly as shown above
2. Each category should contain example images for that visibility level
3. Supported image formats: .jpg, .jpeg, .png, .tif, .tiff
4. For best results, include at least 20-30 images per category
5. Images should be representative of the category (clear examples)
6. Avoid using the same or very similar images across categories

HOW TO PREPARE YOUR DATA:

1. Create a main training folder (e.g., "Training_Data")
2. Inside it, create the four subfolders with exact names shown above
3. Sort your example images into the appropriate folders
4. Browse to the main folder (not the subfolders) when selecting training data

The visibility categories represent:
- zero_visibility: No visible features, complete turbidity
- low_visibility: Minimal features visible, heavy turbidity
- good_visibility: Clear features, moderate turbidity
- great_visibility: Very clear features, minimal turbidity
"""
        
        guide_text.insert(tk.END, guide_content)
        guide_text.config(state=tk.DISABLED)  # Make read-only
        
        # Add a close button
        close_button = ttk.Button(guide_window, text="Close", command=guide_window.destroy)
        close_button.pack(pady=10)
        
        # Make dialog modal
        guide_window.transient(self.master)
        guide_window.grab_set()
        self.master.wait_window(guide_window)

    def on_training_radio_select(self):
        """Called when training radio button is selected"""
        path = self.training_path.get()
        if path and os.path.isdir(path):
            # Check training directory structure
            if hasattr(self, 'visibility_analyzer'):
                message = self.visibility_analyzer.verify_and_provide_training_guidance(path)
                if "Error" in message or "Warning" in message:
                    self.log_message(message)
        
        # Update UI state
        self.training_entry.config(state=tk.NORMAL)
        self.training_button.config(state=tk.NORMAL)
        self.help_button.config(state=tk.NORMAL)
        self.model_entry.config(state=tk.DISABLED)
        self.model_button.config(state=tk.DISABLED)

    def on_model_radio_select(self):
        """Called when model radio button is selected"""
        # Update UI state
        self.model_entry.config(state=tk.NORMAL)
        self.model_button.config(state=tk.NORMAL)
        self.training_entry.config(state=tk.DISABLED)
        self.training_button.config(state=tk.DISABLED)
        self.help_button.config(state=tk.DISABLED)

    def select_highlight_images(self, input_folder, output_folder):
        """Select highlight images from the dataset"""
        self.log_message("Selecting highlight images from the dataset...")
        
        try:
            # Get visibility results if available from a previous analysis
            visibility_results = None
            if hasattr(self, 'visibility_analyzer') and hasattr(self.visibility_analyzer, 'visibility_stats'):
                visibility_results = self.visibility_analyzer.visibility_stats
            
            # Define min altitude threshold 
            min_altitude_threshold = 4.0
            
            # Call the highlight selector with our app's altitude threshold
            highlight_paths = self.highlight_selector.select_highlights(
                input_folder=input_folder,
                output_folder=output_folder,
                count=10,  # Select 10 highlight images
                progress_callback=self.update_progress,
                altitude_threshold=self.altitude_threshold,
                min_altitude_threshold=min_altitude_threshold,
                visibility_results=visibility_results
            )
            
            if highlight_paths:
                self.log_message(f"Successfully selected {len(highlight_paths)} highlight images")
                self.log_message(f"Highlight images saved to: {os.path.join(output_folder, 'highlight_images')}")
            else:
                self.log_message("No highlight images were selected")
                
            return True
                
        except Exception as e:
            self.log_message(f"Error during highlight image selection: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return False

    def toggle_batch_mode(self):
        """Toggle between single processing and batch processing modes"""
        self.batch_mode = not self.batch_mode
        
        if self.batch_mode:
            # Switch to batch mode
            self.batch_button.configure(text="Switch to Single Mode")
            self.show_batch_ui()
        else:
            # Switch back to single mode
            self.batch_button.configure(text="Switch to Batch Mode")
            self.hide_batch_ui()

    def show_batch_ui(self):
        """Show batch processing interface"""
        # Hide standard input/output rows but keep the grid structure intact
        if hasattr(self, 'input_entry'):
            # Store current values before hiding
            self.stored_input_path = self.input_path.get()
            self.stored_output_path = self.output_path.get()
            self.stored_nav_path = self.nav_path.get()  # Add this line
            
            # Hide the elements
            self.input_entry.grid_remove()
            self.input_button.grid_remove()
        
        if hasattr(self, 'output_entry'):
            self.output_entry.grid_remove()
            self.output_button.grid_remove()
        
        # Hide nav file selection elements
        if hasattr(self, 'nav_entry'):
            self.nav_entry.grid_remove()
            self.nav_button.grid_remove()
    
        # Create batch processing frame if it doesn't exist
        if not hasattr(self, 'batch_frame'):
            self.batch_frame = ttk.LabelFrame(self.input_frame, text="Batch Processing")
            
            # CSV file selection
            csv_frame = ttk.Frame(self.batch_frame)
            csv_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Label(csv_frame, text="Batch CSV File:").grid(row=0, column=0, sticky='w')
            self.batch_csv_entry = ttk.Entry(csv_frame, textvariable=self.batch_csv_path)
            self.batch_csv_entry.grid(row=0, column=1, padx=5, sticky='ew')
            self.batch_browse_button = ttk.Button(csv_frame, text="Browse...", command=self.select_batch_csv)
            self.batch_browse_button.grid(row=0, column=2)
            csv_frame.columnconfigure(1, weight=1)
            
            # Template button
            template_frame = ttk.Frame(self.batch_frame)
            template_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            self.save_template_button = ttk.Button(
                template_frame,
                text="Save Example Template CSV",
                command=self.save_csv_template
            )
            self.save_template_button.pack(anchor='w')
            
            # Instructions
            instructions_frame = ttk.LabelFrame(self.batch_frame, text="Batch Processing Instructions")
            instructions_frame.pack(fill=tk.X, padx=10, pady=5)
            
            instructions_text = """
1. Create a CSV with these columns: Input_Path, Output_Path, Nav_File_Path (optional)
2. Each row represents one processing job
3. Use the checkboxes to select functions to run on all entries
4. Click 'Process Images' to process all entries in the CSV
            """
            ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT).pack(padx=10, pady=5, anchor='w')
        
        # Show batch frame in the right position
        self.batch_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=5)
        
        # Update process button text
        self.process_button.configure(text="Process All Batches")
        
        # Note: We keep the visibility model entries intact to preserve their values

    def hide_batch_ui(self):
        """Hide batch processing interface and show standard UI"""
        # Hide batch frame
        if hasattr(self, 'batch_frame'):
            self.batch_frame.grid_remove()
        
        # Show standard input/output fields
        if hasattr(self, 'input_entry'):
            # Restore saved values if they exist
            if hasattr(self, 'stored_input_path'):
                self.input_path.set(self.stored_input_path)
            if hasattr(self, 'stored_output_path'):
                self.output_path.set(self.stored_output_path)
            if hasattr(self, 'stored_nav_path'):  # Add this block
                self.nav_path.set(self.stored_nav_path)
                
            # Make elements visible again
            self.input_entry.grid()
            self.input_button.grid()
        
        if hasattr(self, 'output_entry'):
            self.output_entry.grid()
            self.output_button.grid()
        
        # Show nav file selection elements
        if hasattr(self, 'nav_entry'):
            self.nav_entry.grid()
            self.nav_button.grid()
    
        # Reset process button text
        self.process_button.configure(text="Process Images")

    def select_batch_csv(self):
        """Browse for batch CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Batch CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.batch_csv_path.set(file_path)
            self.log_message(f"Batch CSV file set to: {file_path}")

    def save_csv_template(self):
        """Save an example CSV template file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Example CSV Template",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            try:
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Input_Path', 'Output_Path', 'Nav_File_Path'])
                    
                    # Add example rows (with placeholder paths)
                    example_paths = [
                        ["D:\\Data\\DIVE001", "D:\\Results\\DIVE001_Imagery_Outputs", "D:\\Data\\DIVE001\\Vehicle_Data\\DIVE001_Nav.txt"],
                        ["D:\\Data\\DIVE002", "D:\\Results\\DIVE002_Imagery_Outputs", "D:\\Data\\DIVE002\\Vehicle_Data\\DIVE002_Nav.txt"]
                    ]
                    writer.writerows(example_paths)
                
                self.log_message(f"Example batch CSV template saved to: {file_path}")
                
                # Automatically set the path
                self.batch_csv_path.set(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save template: {str(e)}")

    def process_batch(self):
        """Process all entries in the batch CSV file"""
        batch_csv = self.batch_csv_path.get()
        
        if not batch_csv or not os.path.exists(batch_csv):
            messagebox.showerror("Error", "Please select a valid batch CSV file.")
            return
        
        try:
            # Read the CSV file
            df = pd.read_csv(batch_csv)
            
            # Check required columns
            required_columns = ['Input_Path', 'Output_Path']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                messagebox.showerror(
                    "Invalid CSV Format",
                    f"The CSV is missing required column(s): {', '.join(missing_columns)}\n\n"
                    f"Required columns are: {', '.join(required_columns)}"
                )
                return
            
            # Check if all input paths exist
            missing_inputs = [path for path in df['Input_Path'] if not os.path.exists(path)]
            if missing_inputs:
                msg = "The following input paths do not exist:\n\n"
                for path in missing_inputs[:5]:  # Show first 5 to avoid excessive message
                    msg += f"- {path}\n"
                
                if len(missing_inputs) > 5:
                    msg += f"\nand {len(missing_inputs) - 5} more..."
                    
                if not messagebox.askyesno("Missing Input Folders", msg + "\n\nDo you want to continue anyway?"):
                    return

            # Check nav files if column exists and required for selected functions
            if (self.footprint_map_var.get() or self.location_map_var.get()) and 'Nav_File_Path' not in df.columns:
                messagebox.showwarning(
                    "Nav File Required",
                    "Navigation file path column (Nav_File_Path) is missing from CSV.\n"
                    "This is required for location maps and footprint analysis."
                )
                return

            # Prepare output directories
            for output_path in df['Output_Path']:
                os.makedirs(output_path, exist_ok=True)
            
            # Disable the process button during processing
            self.process_button.configure(state=tk.DISABLED)
            
            # Set both thresholds in the AltitudeMap object
            self.altitude_map.set_altitude_thresholds(self.altitude_threshold, self.low_altitude_threshold)
            
            # Start batch processing in a thread
            thread = threading.Thread(
                target=self.run_batch_processing,
                args=(df,)
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process batch CSV: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def run_batch_processing(self, df):
        """Run processing for all rows in the dataframe"""
        try:
            total_rows = len(df)
            self.log_message(f"Starting batch processing of {total_rows} folders...")
            
            for index, row in df.iterrows():
                try:
                    self.log_message(f"\nProcessing entry {index + 1} of {total_rows}")
                    self.log_message(f"Input: {row['Input_Path']}")
                    self.log_message(f"Output: {row['Output_Path']}")
                    
                    # Reset analysis state for this run
                    self.reset_analysis_state()
                    
                    # Set paths for this batch entry
                    self.input_path.set(row['Input_Path'])
                    self.output_path.set(row['Output_Path'])
                    
                    # Set nav path if available in CSV
                    if 'Nav_File_Path' in df.columns:
                        self.nav_path.set(row['Nav_File_Path'])
                    else:
                        self.nav_path.set('')  # Clear nav path if not in CSV
                
                    # Process this entry
                    self.analyze_images(row['Input_Path'], row['Output_Path'])
                
                except Exception as e:
                    self.log_message(f"Error processing entry {index + 1}: {str(e)}")
                    continue  # Continue with next entry even if this one fails
            
            # Re-enable the process button
            self.master.after(0, lambda: self.process_button.configure(state=tk.NORMAL))
            self.log_message("\nAll batch processing completed.")
            
        except Exception as e:
            self.log_message(f"Fatal error in batch processing: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            # Re-enable the process button
            self.master.after(0, lambda: self.process_button.configure(state=tk.NORMAL))

    def reset_analysis_state(self):
        """Reset analysis state between batch runs to prevent cross-contamination"""
        # Reset metrics state
        if hasattr(self, 'metrics'):
            self.metrics.gps_data = []
            if hasattr(self.metrics, 'vertical_overlap_stats'):
                self.metrics.vertical_overlap_stats = None
            if hasattr(self.metrics, 'horizontal_overlap_stats'):
                self.metrics.horizontal_overlap_stats = None
            if hasattr(self.metrics, 'overall_overlap_stats'):
                self.metrics.overall_overlap_stats = None
        
        # Reset footprint map state
        if hasattr(self, 'footprint_map'):
            if hasattr(self.footprint_map, 'vertical_overlap_stats'):
                self.footprint_map.vertical_overlap_stats = None
            if hasattr(self.footprint_map, 'horizontal_overlap_stats'):
                self.footprint_map.horizontal_overlap_stats = None
            if hasattr(self.footprint_map, 'overall_overlap_stats'):
                self.footprint_map.overall_overlap_stats = None
        
        # Reset visibility analyzer state
        if hasattr(self, 'visibility_analyzer'):
            if hasattr(self.visibility_analyzer, 'visibility_stats'):
                self.visibility_analyzer.visibility_stats = None

    def play_completion_sound(self):
        """Play an MP3 sound when processing is complete"""
        try:
            from playsound import playsound
            
            # Get path to MP3 sound file
            sound_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "utils", "sounds", "beer_open.mp3")
            
            # Create sounds directory if it doesn't exist
            os.makedirs(os.path.dirname(sound_file), exist_ok=True)
            
            # Play sound if file exists
            if os.path.exists(sound_file):
                playsound(sound_file)
            else:
                print('\a')  # Console bell as fallback
            
            self.log_message("Processing complete!")
            
        except Exception as e:
            print(f"Error playing completion sound: {str(e)}")
            print('\a')  # Console bell as fallback
            self.log_message("Processing complete!")

    def log_metrics_summary(self, metrics):
        """Log a summary of image metrics"""
        try:
            # Get image counts and size stats
            image_counts = metrics.get_image_counts()
            
            # Format file sizes as human-readable
            processed_size_str = self.format_file_size(image_counts.get('processed_size', 0))
            raw_size_str = self.format_file_size(image_counts.get('raw_size', 0))
            other_size_str = self.format_file_size(image_counts.get('other_size', 0))
            total_size_str = self.format_file_size(image_counts.get('total_size', 0))
            
            # Log image counts and sizes
            self.log_message("Image Analysis Summary:")
            self.log_message("-----------------------")
            self.log_message(f"Processed Still Images: {image_counts.get('processed_count', 0)}")
            self.log_message(f"Processed Stills Size: {processed_size_str}")
            self.log_message(f"Raw Images: {image_counts.get('raw_count', 0)}")
            self.log_message(f"Raw Images Size: {raw_size_str}")
            self.log_message(f"Other Files: {image_counts.get('other_count', 0)}")
            self.log_message(f"Other Files Size: {other_size_str}")
            self.log_message(f"Total Files: {image_counts.get('total_count', 0)}")
            self.log_message(f"Total Size: {total_size_str}")
            self.log_message("")
            
            # Get altitude statistics
            altitude_stats = metrics.get_altitude_statistics(self.altitude_threshold)
            if altitude_stats and altitude_stats.get('count', 0) > 0:
                # Log altitude stats
                self.log_message("Altitude Analysis (Navigation Data):")
                self.log_message("----------------------------------------")
                self.log_message(f"Images with navigation altitude data: {altitude_stats['count']} of {image_counts.get('processed_count', 0)} processed images")
                self.log_message(f"Images below {self.altitude_threshold}m: {altitude_stats['below_threshold']} ({altitude_stats['below_percent']:.1f}%)")
                self.log_message(f"Images above {self.altitude_threshold}m: {altitude_stats['above_threshold']} ({altitude_stats['above_percent']:.1f}%)")
                self.log_message(f"Altitude range: {altitude_stats['min']:.2f}m to {altitude_stats['max']:.2f}m")
                self.log_message(f"Average altitude: {altitude_stats['avg']:.2f}m")
                self.log_message(f"Median altitude: {altitude_stats['median']:.2f}m")
                self.log_message(f"Standard deviation: {altitude_stats['std_dev']:.2f}m")
                if metrics.metrics_file:
                    self.log_message(f"Basic metrics saved to: {metrics.metrics_file}")
                self.log_message("")
        except Exception as e:
            self.log_message(f"Error logging metrics summary: {str(e)}")


