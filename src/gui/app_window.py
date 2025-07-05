import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import time
import traceback
import pandas as pd

# Import base classes
from src.gui.ui_components import UIComponents
from src.gui.processing_controller import ProcessingController

class AppWindow(UIComponents, ProcessingController):
    def __init__(self, root):
        self.root = root
        self.root.title("VOYIS First Look Metrics")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.setup_variables()
        
        # Create UI FIRST (so log_text exists)
        self.create_ui()
        
        # Set up UI update thread
        self.setup_ui_update_thread()
        
        # Initialize processors AFTER UI is created (so logging works)
        self.initialize_processors()
        
        # Set the window icon AFTER everything is created
        self.root.after(100, self.set_window_icon)  # Delay by 100ms

    def set_window_icon(self):
        """Set the window icon using the NOAA logo"""
        try:
            # Get the path to the NOAA logo
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(current_dir, "..", "..", "src", "utils", "NOAA_VOYIS_Logo.ico")
            logo_path = os.path.abspath(logo_path)
            
            if os.path.exists(logo_path):
                self.root.iconbitmap(logo_path)
            else:
                print(f"Warning: NOAA logo not found at {logo_path}")
        except Exception as e:
            print(f"Warning: Could not set window icon: {str(e)}")

    def setup_variables(self):
        """Initialize all tkinter variables"""
        # Input/Output paths
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.nav_path = tk.StringVar()
        
        # LLS processing paths
        self.lls_path = tk.StringVar()
        self.phins_nav_path = tk.StringVar()
        
        # Navigation processing paths (for plotting - text files with heave data)
        self.nav_processing_var = tk.BooleanVar(value=False)
        self.nav_plot_file_path = tk.StringVar()  # PHINS file
        self.nav_state_file_path = tk.StringVar()  # NAV_STATE file
        
        # Batch processing
        self.batch_mode = False
        self.batch_csv_path = tk.StringVar()
        self.batch_var = tk.BooleanVar()  # Add this for ProcessingController compatibility
        
        # Processing function variables
        self.lls_processing_var = tk.BooleanVar(value=True)
        self.basic_metrics_var = tk.BooleanVar(value=True)
        self.location_map_var = tk.BooleanVar(value=True)
        self.histogram_var = tk.BooleanVar(value=True)
        self.footprint_map_var = tk.BooleanVar(value=True)
        self.visibility_analyzer_var = tk.BooleanVar(value=True)
        self.highlight_selector_var = tk.BooleanVar(value=True)
        self.all_var = tk.BooleanVar(value=True)
        
        # Visibility model variables
        self.model_type_var = tk.StringVar(value="model")
        self.model_path = tk.StringVar()
        self.training_path = tk.StringVar()
        
        # Set default pre-trained model path
        try:
            app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            default_model = os.path.join(app_root, "v_a_pre-trained_models", "visibility_model_20250402.h5")
            if os.path.exists(default_model):
                self.model_path.set(default_model)
        except:
            pass
        
        # Control variables - DEFAULT TO 9.0 METERS
        self.altitude_threshold = 9.0
        self.low_altitude_threshold = 4.0
        self.threshold_var = tk.StringVar(value=str(self.altitude_threshold))
        self.progress_var = tk.DoubleVar()
        
        # Initialize lists for widgets
        self.input_widgets = []
        self.single_mode_frames = []

    def initialize_processors(self):
        """Initialize all processing components"""
        try:
            # Only import what actually exists in your workspace
            from src.models.metrics import Metrics
            from src.models.altitude_map import AltitudeMap
            from src.models.footprint_map import FootprintMap
            from src.models.visibility_analyzer import VisibilityAnalyzer
            from src.models.highlight_selector import HighlightSelector
            
            # Initialize components with thresholds and callbacks
            self.metrics = Metrics(self.altitude_threshold)
            self.altitude_map = AltitudeMap()
            if hasattr(self.altitude_map, 'set_altitude_thresholds'):
                self.altitude_map.set_altitude_thresholds(self.altitude_threshold, self.low_altitude_threshold)
            
            self.footprint_map = FootprintMap()
            if hasattr(self.footprint_map, 'altitude_threshold'):
                self.footprint_map.altitude_threshold = self.altitude_threshold
            
            # Initialize visibility analyzer with log callback
            self.visibility_analyzer = VisibilityAnalyzer(log_callback=self.log_message)
            
            # Initialize highlight selector
            self.highlight_selector = HighlightSelector()
            
            # Debug: Show available methods for each processor
            if hasattr(self, 'log_text'):
                self.log_message("Processor initialization complete:")
                
                for name, processor in [
                    ("Metrics", self.metrics),
                    ("AltitudeMap", self.altitude_map), 
                    ("FootprintMap", self.footprint_map),
                    ("VisibilityAnalyzer", self.visibility_analyzer),
                    ("HighlightSelector", self.highlight_selector)
                ]:
                    if processor:
                        methods = [method for method in dir(processor) 
                                 if not method.startswith('_') and callable(getattr(processor, method))]
                        self.log_message(f"  {name} methods: {methods[:5]}...")  # Show first 5 methods
                
                self.log_message("All processing modules initialized successfully")
            else:
                print("All processing modules initialized successfully")
            
        except ImportError as e:
            # Only log if the UI is ready
            if hasattr(self, 'log_text'):
                self.log_message(f"Warning: Some processing modules not available: {e}")
            else:
                print(f"Warning: Some processing modules not available: {e}")
            # Initialize with None values so we can check later
            self.metrics = None
            self.altitude_map = None
            self.footprint_map = None
            self.visibility_analyzer = None
            self.highlight_selector = None

    def create_ui(self):
        """Create the user interface"""
        # Create main frames
        self.create_frames()
        
        # Create batch processing toggle at the very top
        self.create_batch_toggle_section()
        
        # Create UI sections
        self.create_input_section()
        self.create_functions_section()
        self.create_controls_section()
        self.create_log_section()

    def create_batch_toggle_section(self):
        """Create the batch processing toggle button at the top"""
        # Batch toggle frame at the very top
        self.batch_toggle_frame = ttk.Frame(self.left_frame, padding="10")
        self.batch_toggle_frame.pack(fill=tk.X, pady=(5, 15), padx=10)
        
        # Large, prominent batch toggle button
        style = ttk.Style()
        style.configure("BatchToggle.TButton", font=('', 11, 'bold'))
        
        self.batch_toggle_button = ttk.Button(
            self.batch_toggle_frame,
            text="Enable Batch Processing",
            command=self.toggle_batch_mode,
            style="BatchToggle.TButton"
        )
        self.batch_toggle_button.pack(fill=tk.X)
        
        # Batch CSV selection frame (initially hidden)
        self.batch_csv_frame = ttk.LabelFrame(self.left_frame, text="Batch Processing", padding="10")
        
        # CSV file selection
        csv_selection_frame = ttk.Frame(self.batch_csv_frame)
        csv_selection_frame.pack(fill=tk.X, pady=(0, 10))
        csv_selection_frame.columnconfigure(1, weight=1)
        
        ttk.Label(csv_selection_frame, text="Batch CSV File:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.batch_csv_entry = ttk.Entry(csv_selection_frame, textvariable=self.batch_csv_path, width=50)
        self.batch_csv_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.batch_csv_button = ttk.Button(csv_selection_frame, text="Browse...", command=self.select_batch_csv)
        self.batch_csv_button.grid(row=0, column=2)
        
        # Template and help buttons
        buttons_frame = ttk.Frame(self.batch_csv_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.template_button = ttk.Button(
            buttons_frame,
            text="Create CSV Template",
            command=self.create_batch_csv_template
        )
        self.template_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Help text
        help_text = ttk.Label(
            self.batch_csv_frame,
            text="CSV columns: input_folder*, output_folder*, nav_file, nav_plot_file, lls_folder, phins_nav_file (*required)",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        help_text.pack(anchor='w')
        
        # Initially hide the batch CSV frame
        # (Don't pack it yet)

    def create_input_section(self):
        """Create the input/output file selection widgets"""
        self.input_frame = ttk.LabelFrame(self.left_frame, text="Input/Output Configuration", padding="10")
        self.input_frame.pack(fill=tk.X, pady=(0, 10), padx=10)
        self.input_frame.columnconfigure(1, weight=1)
        
        # Store reference to all input widgets for batch mode greying out
        self.input_widgets = []
        
        # Navigation Processing Input Section (for plotting)
        nav_frame = ttk.LabelFrame(self.input_frame, text="Navigation Data Plotting", padding="5")
        nav_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        nav_frame.columnconfigure(1, weight=1)
        
        # NAV_STATE file selection
        ttk.Label(nav_frame, text="NAV_STATE Text File:").grid(row=0, column=0, sticky='w')
        self.nav_state_entry = ttk.Entry(nav_frame, textvariable=self.nav_state_file_path, width=40)
        self.nav_state_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.nav_state_button = ttk.Button(nav_frame, text="Browse...", command=self.select_nav_state_file)
        self.nav_state_button.grid(row=0, column=2)
        
        # PHINS file selection for heave data
        ttk.Label(nav_frame, text="PHINS INS Text File:").grid(row=1, column=0, sticky='w')
        self.nav_plot_entry = ttk.Entry(nav_frame, textvariable=self.nav_plot_file_path, width=40)
        self.nav_plot_entry.grid(row=1, column=1, padx=5, sticky='ew')
        self.nav_plot_button = ttk.Button(nav_frame, text="Browse...", command=self.select_nav_plot_file)
        self.nav_plot_button.grid(row=1, column=2)
        
        # Add nav plot widgets to list
        self.input_widgets.extend([self.nav_state_entry, self.nav_state_button, self.nav_plot_entry, self.nav_plot_button])
        
        # LLS Input Section
        lls_frame = ttk.LabelFrame(self.input_frame, text="Laser Data (LLS) Inputs", padding="5")
        lls_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        lls_frame.columnconfigure(1, weight=1)
        
        # LLS folder selection
        ttk.Label(lls_frame, text="LLS Folder:").grid(row=0, column=0, sticky='w')
        self.lls_entry = ttk.Entry(lls_frame, textvariable=self.lls_path, width=40)
        self.lls_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.lls_button = ttk.Button(lls_frame, text="Browse...", command=self.select_lls_folder)
        self.lls_button.grid(row=0, column=2)
        
        # Phins Nav file selection
        ttk.Label(lls_frame, text="PhinsData Bin File:").grid(row=1, column=0, sticky='w')
        self.phins_nav_entry = ttk.Entry(lls_frame, textvariable=self.phins_nav_path, width=40)
        self.phins_nav_entry.grid(row=1, column=1, padx=5, sticky='ew')
        self.phins_nav_button = ttk.Button(lls_frame, text="Browse...", command=self.select_phins_nav_file)
        self.phins_nav_button.grid(row=1, column=2)
        
        # Add LLS widgets to list
        self.input_widgets.extend([self.lls_entry, self.lls_button, self.phins_nav_entry, self.phins_nav_button])
        
        # Imagery Input Section
        imagery_frame = ttk.LabelFrame(self.input_frame, text="Imagery Inputs", padding="5")
        imagery_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        imagery_frame.columnconfigure(1, weight=1)
        
        # Input folder selection
        ttk.Label(imagery_frame, text="Input Folder:").grid(row=0, column=0, sticky='w')
        self.input_entry = ttk.Entry(imagery_frame, textvariable=self.input_path, width=40)
        self.input_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.input_button = ttk.Button(imagery_frame, text="Browse...", command=self.select_input_folder)
        self.input_button.grid(row=0, column=2)
        
        # Vehicle Nav file selection
        ttk.Label(imagery_frame, text="Dive Nav File:").grid(row=1, column=0, sticky='w')
        self.nav_entry = ttk.Entry(imagery_frame, textvariable=self.nav_path, width=40)
        self.nav_entry.grid(row=1, column=1, padx=5, sticky='ew')
        self.nav_button = ttk.Button(imagery_frame, text="Browse...", command=self.select_nav_file)
        self.nav_button.grid(row=1, column=2)
        
        # Add imagery widgets to list
        self.input_widgets.extend([self.input_entry, self.input_button, self.nav_entry, self.nav_button])
        
        # Output folder selection
        output_frame = ttk.Frame(self.input_frame)
        output_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky='w')
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=40)
        self.output_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.output_button = ttk.Button(output_frame, text="Browse...", command=self.select_output_folder)
        self.output_button.grid(row=0, column=2)
        
        # Add output widgets to list
        self.input_widgets.extend([self.output_entry, self.output_button])

    def create_functions_section(self):
        """Create the functions selection checkbox widgets"""
        self.functions_frame = ttk.LabelFrame(self.left_frame, text="Processing Functions", padding="10")
        self.functions_frame.pack(fill=tk.X, pady=(0, 10), padx=10)
        self.functions_frame.columnconfigure(0, weight=1)
        
        # Navigation Processing Section
        nav_section = ttk.LabelFrame(self.functions_frame, text="Navigation Data Plotting", padding="5")
        nav_section.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        self.nav_processing_checkbox = ttk.Checkbutton(
            nav_section, 
            text="Process Navigation Data for Plotting", 
            variable=self.nav_processing_var,
            command=self.update_all_checkbox
        )
        self.nav_processing_checkbox.grid(row=0, column=0, sticky='w')
        
        # LLS Processing Section
        lls_section = ttk.LabelFrame(self.functions_frame, text="Laser Data Processing", padding="5")
        lls_section.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        self.lls_processing_checkbox = ttk.Checkbutton(
            lls_section, 
            text="Process LLS Data", 
            variable=self.lls_processing_var,
            command=self.update_all_checkbox
        )
        self.lls_processing_checkbox.grid(row=0, column=0, sticky='w')
        
        # Imagery Processing Section
        imagery_section = ttk.LabelFrame(self.functions_frame, text="Imagery Processing", padding="5")
        imagery_section.grid(row=2, column=0, columnspan=3, sticky='ew')
        
        # "All" checkbox for imagery functions
        self.all_checkbox = ttk.Checkbutton(
            imagery_section, text="All Imagery Functions", 
            variable=self.all_var, 
            command=self.toggle_all_functions
        )
        self.all_checkbox.grid(row=0, column=0, sticky='w')
        
        # Individual function checkboxes
        checkboxes = [
            ("Summary Metrics", self.basic_metrics_var),
            ("Location Map", self.location_map_var),
            ("Altitude Histogram", self.histogram_var),
            ("Footprint Map", self.footprint_map_var),
            ("Visibility Analysis", self.visibility_analyzer_var),
            ("Highlight Selection", self.highlight_selector_var)
        ]
        
        current_row = 1
        for text, var in checkboxes:
            checkbox = ttk.Checkbutton(
                imagery_section, text=text, 
                variable=var, 
                command=self.update_all_checkbox
            )
            checkbox.grid(row=current_row, column=0, sticky='w')
            
            # Special handling for visibility analyzer
            if text == "Visibility Analysis":
                self.create_visibility_options(imagery_section, current_row)
                current_row += 2  # Skip the next row for visibility options
            else:
                current_row += 1

    def create_visibility_options(self, parent, visibility_row):
        """Create visibility analyzer options"""
        self.visibility_model_frame = ttk.Frame(parent)
        self.visibility_model_row = visibility_row + 1  # Place it right after the visibility checkbox
        
        # Model type selection
        model_type_frame = ttk.Frame(self.visibility_model_frame)
        model_type_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        model_type_frame.columnconfigure(1, weight=1)
        
        ttk.Radiobutton(
            model_type_frame, text="Use Pre-trained Model:", 
            variable=self.model_type_var, value="model"
        ).grid(row=0, column=0, sticky='w')
        
        self.model_entry = ttk.Entry(model_type_frame, textvariable=self.model_path, width=30)
        self.model_entry.grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Button(
            model_type_frame, text="Browse...", 
            command=lambda: self.select_visibility_file("model")
        ).grid(row=0, column=2)
        
        # Training data option
        ttk.Radiobutton(
            model_type_frame, text="Train New Model:", 
            variable=self.model_type_var, value="training"
        ).grid(row=1, column=0, sticky='w')
        
        self.training_entry = ttk.Entry(model_type_frame, textvariable=self.training_path, width=30)
        self.training_entry.grid(row=1, column=1, padx=5, sticky='ew')
        
        ttk.Button(
            model_type_frame, text="Browse...", 
            command=lambda: self.select_visibility_file("training")
        ).grid(row=1, column=2)
        
        # Initially hide the visibility options
        self.toggle_visibility_options()

    def create_controls_section(self):
        """Create the controls section with threshold and process button"""
        controls_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10), padx=10)
        
        # Threshold control
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"Altitude Threshold: {self.altitude_threshold:.1f}m")
        self.threshold_label.pack(side=tk.LEFT)
        
        threshold_entry_frame = ttk.Frame(threshold_frame)
        threshold_entry_frame.pack(side=tk.RIGHT)
        
        ttk.Label(threshold_entry_frame, text="Threshold:").pack(side=tk.LEFT)
        threshold_entry = ttk.Entry(threshold_entry_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.pack(side=tk.LEFT, padx=5)
        threshold_entry.bind('<Return>', self.threshold_changed)
        threshold_entry.bind('<FocusOut>', self.threshold_changed)
        
        # Process button
        style = ttk.Style()
        style.configure("AccentButton.TButton", font=('', 10, 'bold'))
        
        self.process_button = ttk.Button(
            controls_frame, text="Process Images", 
            command=self.process_images,  # This comes from ProcessingController
            style="AccentButton.TButton"
        )
        self.process_button.pack(pady=10)

    def create_log_section(self):
        """Create the log output section"""
        log_frame = ttk.LabelFrame(self.right_frame, text="Processing Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create log text with scrollbar
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=60, height=20, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        progress_frame = ttk.Frame(log_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, 
            maximum=100, length=400
        )
        self.progress_bar.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(pady=(5, 0))

    # Canvas scroll event handlers
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Update the width of the window to fit the canvas"""
        canvas_width = event.width
        self.left_canvas.itemconfig(self.left_canvas_window, width=canvas_width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # File selection methods
    def select_lls_folder(self):
        """Select LLS input folder"""
        folder_path = filedialog.askdirectory(title="Select LLS Folder")
        if folder_path:
            self.lls_path.set(folder_path)
            self.log_message(f"LLS folder set to: {folder_path}")

    def select_phins_nav_file(self):
        """Select Phins navigation file"""
        file_path = filedialog.askopenfilename(
            title="Select Phins Navigation File",
            filetypes=[
                ("Binary Files", "*.bin"),
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.phins_nav_path.set(file_path)
            self.log_message(f"Phins navigation file set to: {file_path}")

    def select_input_folder(self):
        """Select input folder for imagery"""
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.input_path.set(folder_path)
            self.log_message(f"Input folder set to: {folder_path}")

    def select_output_folder(self):
        """Select output folder"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_path.set(folder_path)
            self.log_message(f"Output folder set to: {folder_path}")

    def select_nav_file(self):
        """Select vehicle navigation file"""
        file_path = filedialog.askopenfilename(
            title="Select Navigation File",
            filetypes=[
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("Binary Files", "*.bin"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.nav_path.set(file_path)
            self.log_message(f"Navigation file set to: {file_path}")

    def select_nav_plot_file(self):
        """Select PHINS INS file for plotting (optional heave data)"""
        file_path = filedialog.askopenfilename(
            title="Select PHINS INS File (Optional - for heave data)",
            filetypes=[
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.nav_plot_file_path.set(file_path)
            self.log_message(f"PHINS INS file set to: {file_path}")

    def select_nav_state_file(self):
        """Select NAV_STATE file for navigation plotting"""
        file_path = filedialog.askopenfilename(
            title="Select NAV_STATE Text File",
            filetypes=[
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.nav_state_file_path.set(file_path)
            self.log_message(f"NAV_STATE file set to: {file_path}")

    def select_visibility_file(self, file_type):
        """Select model file or training data directory for visibility analyzer"""
        if file_type == "model":
            file_path = filedialog.askopenfilename(
                title="Select Pre-trained Model File",
                filetypes=[("H5 Files", "*.h5"), ("All Files", "*.*")]
            )
            if file_path:
                self.model_path.set(file_path)
                self.model_type_var.set("model")
        elif file_type == "training":
            folder_path = filedialog.askdirectory(title="Select Training Data Folder")
            if folder_path:
                self.training_path.set(folder_path)
                self.model_type_var.set("training")

    def select_batch_csv(self):
        """Select batch processing CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Batch Processing CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.batch_csv_path.set(file_path)
            self.log_message(f"Batch CSV file set to: {file_path}")
            
            # Validate the CSV file
            self.validate_batch_csv(file_path)

    # UI event handlers
    def toggle_all_functions(self):
        """Toggle all function checkboxes"""
        all_selected = self.all_var.get()
        
        # Only toggle imagery functions, not LLS
        self.basic_metrics_var.set(all_selected)
        self.location_map_var.set(all_selected)
        self.histogram_var.set(all_selected)
        self.footprint_map_var.set(all_selected)
        self.visibility_analyzer_var.set(all_selected)
        self.highlight_selector_var.set(all_selected)
        
        self.toggle_visibility_options()

    def update_all_checkbox(self):
        """Update the 'All' checkbox based on individual selections"""
        imagery_functions = [
            self.basic_metrics_var.get(),
            self.location_map_var.get(),
            self.histogram_var.get(),
            self.footprint_map_var.get(),
            self.visibility_analyzer_var.get(),
            self.highlight_selector_var.get()
        ]
        
        if all(imagery_functions):
            self.all_var.set(True)
        else:
            self.all_var.set(False)
        
        self.toggle_visibility_options()

    def toggle_visibility_options(self):
        """Show or hide visibility model options based on checkbox state"""
        if hasattr(self, 'visibility_model_frame'):
            if self.visibility_analyzer_var.get():
                self.visibility_model_frame.grid(
                    row=self.visibility_model_row, column=0, columnspan=3, 
                    sticky='ew', pady=(5, 0)
                )
            else:
                self.visibility_model_frame.grid_remove()

    def toggle_batch_mode(self):
        """Toggle between batch and single processing modes"""
        self.batch_mode = not self.batch_mode
        self.batch_var.set(self.batch_mode)  # Keep both in sync
        
        if self.batch_mode:
            # Enable batch mode
            self.batch_toggle_button.config(text="Disable Batch Processing")
            self.batch_csv_frame.pack(fill=tk.X, pady=(0, 10), padx=10, after=self.batch_toggle_frame)
            
            # Disable/grey out single mode input widgets
            for widget in self.input_widgets:
                widget.config(state='disabled')
                
            self.log_message("Batch processing mode enabled")
            
        else:
            # Disable batch mode
            self.batch_toggle_button.config(text="Enable Batch Processing")
            self.batch_csv_frame.pack_forget()
            
            # Re-enable single mode input widgets
            for widget in self.input_widgets:
                widget.config(state='normal')
                
            self.log_message("Single processing mode enabled")

    def threshold_changed(self, *args):
        """Update the altitude threshold when changed"""
        try:
            new_threshold = float(self.threshold_var.get())
            self.altitude_threshold = new_threshold
            self.threshold_label.config(text=f"Altitude Threshold: {new_threshold:.1f}m")
            self.log_message(f"Altitude threshold set to: {new_threshold:.1f}m")
            
            # Update all processors if they exist
            if hasattr(self, 'metrics') and self.metrics:
                self.metrics.altitude_threshold = new_threshold
            if hasattr(self, 'altitude_map') and self.altitude_map:
                self.altitude_map.set_altitude_thresholds(new_threshold, self.low_altitude_threshold)
            if hasattr(self, 'footprint_map') and self.footprint_map:
                self.footprint_map.altitude_threshold = new_threshold
            if hasattr(self, 'visibility_analyzer') and self.visibility_analyzer:
                self.visibility_analyzer.altitude_threshold = new_threshold
                
        except ValueError:
            self.threshold_var.set(str(self.altitude_threshold))
            self.log_message("Invalid threshold value entered")

    def validate_batch_csv(self, csv_path):
        """Validate the batch CSV file format"""
        try:
            df = pd.read_csv(csv_path)
            
            required_cols = ['input_folder', 'output_folder']
            optional_cols = ['nav_file', 'lls_folder', 'phins_nav_file']
            
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                messagebox.showerror(
                    "Invalid CSV Format", 
                    f"Missing required columns: {', '.join(missing_required)}\n\n"
                    f"Required columns: {', '.join(required_cols)}\n"
                    f"Optional columns: {', '.join(optional_cols)}"
                )
                return False
            
            self.log_message(f"Batch CSV validated successfully: {len(df)} entries found")
            
            # Show summary of what will be processed
            lls_count = df['lls_folder'].notna().sum() if 'lls_folder' in df.columns else 0
            imagery_count = df['input_folder'].notna().sum()
            
            self.log_message(f"  - {imagery_count} imagery processing jobs")
            if lls_count > 0:
                self.log_message(f"  - {lls_count} LLS processing jobs")
            
            return True
            
        except Exception as e:
            messagebox.showerror("CSV Validation Error", f"Error reading CSV file: {str(e)}")
            return False

    def create_batch_csv_template(self):
        """Create a template CSV file for batch processing"""
        file_path = filedialog.asksaveasfilename(
            title="Save Batch CSV Template",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Create template data with examples
                template_data = {
                    'input_folder': [
                        'C:/path/to/imagery/folder1',
                        'C:/path/to/imagery/folder2',
                        'C:/path/to/imagery/folder3'
                    ],
                    'output_folder': [
                        'C:/path/to/output/folder1',
                        'C:/path/to/output/folder2', 
                        'C:/path/to/output/folder3'
                    ],
                    'nav_file': [
                        'C:/path/to/nav/file1.txt',
                        '',  # Optional - can be empty
                        'C:/path/to/nav/file3.txt'
                    ],
                    'nav_plot_file': [
                        'C:/path/to/nav_plot/file1.txt',
                        'C:/path/to/nav_plot/file2.txt',
                        ''  # Optional - can be empty if no navigation plotting needed
                    ],
                    'lls_folder': [
                        'C:/path/to/lls/folder1',
                        'C:/path/to/lls/folder2',
                        ''  # Optional - can be empty if no LLS processing needed
                    ],
                    'phins_nav_file': [
                        'C:/path/to/phins/nav1.bin',
                        'C:/path/to/phins/nav2.bin',
                        ''  # Optional - can be empty if no LLS processing needed
                    ]
                }
                
                df = pd.DataFrame(template_data)
                df.to_csv(file_path, index=False)
                
                self.log_message(f"Batch CSV template created: {file_path}")
                messagebox.showinfo(
                    "Template Created", 
                    f"Batch processing CSV template created at:\n{file_path}\n\n"
                    "Edit this file with your actual folder paths, then load it for batch processing.\n\n"
                    "Required columns: input_folder, output_folder\n"
                    "Optional columns: nav_file, nav_plot_file, lls_folder, phins_nav_file"
                )
                
            except Exception as e:
                messagebox.showerror("Template Creation Error", f"Error creating template: {str(e)}")

    def play_completion_sound(self):
        """Play completion sound"""
        try:
            import winsound
            sound_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "utils", "sounds", "beer_open.wav"
            )
            if os.path.exists(sound_path):
                winsound.PlaySound(sound_path, winsound.SND_FILENAME)
        except Exception as e:
            print(f"Could not play sound: {str(e)}")

# Note: The following methods are inherited from ProcessingController:
    # - setup_ui_update_thread()
    # - log_message()
    # - _update_log_text()
    # - update_progress()
    # - _update_progress_ui()
    # - process_images()
    # - _process_images_thread()
    # - analyze_images()
    # - process_lls_data()
    # - process_batch()
    # - validate_inputs()
    # And all the batch processing logic

if __name__ == "__main__":
    root = tk.Tk()
    app = AppWindow(root)
    root.mainloop()