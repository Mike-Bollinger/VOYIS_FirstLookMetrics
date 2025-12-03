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
        
        # Initialize ProcessingController first
        ProcessingController.__init__(self)
        
        # Initialize variables
        self.setup_variables()
        
        
        # Create the user interface
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
        self.nav_processing_var = tk.BooleanVar(value=True)
        self.nav_plot_file_path = tk.StringVar()  # Primary NAV_STATE file (backward compatibility)
        self.phins_ins_path = tk.StringVar()  # PHINS INS file for navigation processing
        
        # Directory-based navigation system (only mode now)
        self.nav_directory_path = tk.StringVar()  # Directory containing navigation files
        
        # Set navigation mode to directory only
        self.nav_merge_mode = tk.StringVar(value='directory')
        
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
        self.single_mode_frames = []  # ADCP file
        self.nav_veh_data_path = tk.StringVar()  # *_Veh_Data file
        self.nav_other_files = []  # List for additional navigation files

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
            text="Required: Output_folder. Optional: nav_directory (for navigation plots), Image_Input (for imagery), LLS_Input + PhinsData_Bin_file (for LLS).",
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
        
        # User guidance for navigation files
        guidance_frame = ttk.Frame(nav_frame)
        guidance_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        guidance_text = (
            "Smart Navigation Directory Selection: Select a directory containing your navigation files.\n"
            "The system will automatically identify and prioritize: PHINS INS, NAV_STATE, STATE, ADCP, and *_Veh_Data files." 
        )
        guidance_label = ttk.Label(guidance_frame, text=guidance_text, 
                                 wraplength=600, justify='left', 
                                 font=('', 9), foreground='blue')
        guidance_label.pack(anchor='w')
        
        # Directory selection frame (no mode selection - directory only)
        self.directory_nav_frame = ttk.Frame(nav_frame)
        self.directory_nav_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        self.directory_nav_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.directory_nav_frame, text="Navigation Files Directory:").grid(row=0, column=0, sticky='w')
        self.nav_directory_entry = ttk.Entry(self.directory_nav_frame, textvariable=self.nav_directory_path, width=50)
        self.nav_directory_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.nav_directory_button = ttk.Button(self.directory_nav_frame, text="Browse...", command=self.select_nav_directory)
        self.nav_directory_button.grid(row=0, column=2)
        
        # Preview/scan button for directory mode
        scan_frame = ttk.Frame(self.directory_nav_frame)
        scan_frame.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky='ew')
        
        self.scan_nav_button = ttk.Button(
            scan_frame, 
            text="Preview Navigation Files", 
            command=self.scan_navigation_directory,
            style="AccentButton.TButton"
        )
        self.scan_nav_button.pack(side=tk.LEFT)
        
        ttk.Label(
            scan_frame, 
            text="Preview which navigation files will be used from the selected directory",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Add nav plot widgets to list
        self.nav_widgets = [
            self.nav_directory_entry, self.nav_directory_button, self.scan_nav_button
        ]
        self.input_widgets.extend(self.nav_widgets)
        
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
        
        # Process and Stop buttons frame
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(pady=10, fill=tk.X)
        
        # Process button styling
        style = ttk.Style()
        style.configure("AccentButton.TButton", font=('', 10, 'bold'))
        style.configure("StopButton.TButton", font=('', 10, 'bold'))
        
        # Process button
        self.process_button = ttk.Button(
            buttons_frame, text="Process Images", 
            command=self.process_images,  # This comes from ProcessingController
            style="AccentButton.TButton"
        )
        self.process_button.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # Stop button
        self.stop_button = ttk.Button(
            buttons_frame, text="Stop Processing", 
            command=self.stop_processing,  # This comes from ProcessingController
            style="StopButton.TButton",
            state=tk.DISABLED  # Initially disabled
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True)

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

    def select_phins_ins_file(self):
        """Select PHINS INS file for navigation plotting (optional heave data)"""
        file_path = filedialog.askopenfilename(
            title="Select PHINS INS File (Optional - for heave data)",
            filetypes=[
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.phins_ins_path.set(file_path)
            self.log_message(f"PHINS INS file set to: {file_path}")

    def select_nav_directory(self):
        """Select directory containing navigation files"""
        directory = filedialog.askdirectory(
            title="Select Navigation Files Directory",
            initialdir=self.nav_directory_path.get() or os.getcwd()
        )
        if directory:
            self.nav_directory_path.set(directory)
            self.log_message(f"Selected navigation directory: {directory}")

    def scan_navigation_directory(self):
        """Scan the selected directory for navigation files and show preview"""
        directory = self.nav_directory_path.get()
        if not directory:
            messagebox.showwarning("No Directory", "Please select a navigation directory first.")
            return
        
        try:
            from src.models.nav_merger import scan_navigation_directory
            
            # Scan the directory
            nav_files = scan_navigation_directory(directory, self.log_message)
            
            if nav_files:
                # Show results in a popup
                result_text = "Found navigation files in priority order:\n\n"
                for file_type, file_path in nav_files.items():
                    filename = os.path.basename(file_path)
                    result_text += f"• {file_type}: {filename}\n"
                
                messagebox.showinfo("Navigation Files Found", result_text)
            else:
                messagebox.showwarning("No Files Found", 
                                     "No valid navigation files found in the selected directory.\n\n"
                                     "Expected file types: PHINS INS, NAV_STATE, STATE, ADCP, *_Veh_Data")
        
        except Exception as e:
            messagebox.showerror("Scan Error", f"Error scanning directory: {str(e)}")

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
            
            # Required and optional column names
            required_cols = ['Output_folder']
            optional_cols = ['nav_directory', 'LLS_Input', 'PhinsData_Bin_file', 'Image_Input']
            
            # Check if we have the required column
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                messagebox.showerror(
                    "Invalid CSV Format", 
                    f"Missing required columns: {', '.join(missing_required)}\n\n"
                    f"Required: {', '.join(required_cols)}\n"
                    f"Optional: {', '.join(optional_cols)}\n\n"
                    f"Found columns: {', '.join(df.columns)}"
                )
                return False
            
            self.log_message(f"Batch CSV validated successfully: {len(df)} entries found")
            
            # Show summary of what will be processed
            lls_count = df['LLS_Input'].notna().sum() if 'LLS_Input' in df.columns else 0
            imagery_count = df['Image_Input'].notna().sum() if 'Image_Input' in df.columns else 0
            nav_count = df['nav_directory'].notna().sum() if 'nav_directory' in df.columns else 0
            
            self.log_message(f"  - {imagery_count} image analysis jobs")
            if lls_count > 0:
                self.log_message(f"  - {lls_count} LLS processing jobs")
            if nav_count > 0:
                self.log_message(f"  - {nav_count} navigation processing jobs (using directory auto-detection)")
            
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
                # Create template data with standardized column names
                template_data = {
                    'nav_directory': [
                        'D:/AUV/VOYIS/PC-24-03/DIVE003/Vehicle_Data',
                        'D:/AUV/VOYIS/PC-24-04/DIVE004/Vehicle_Data'
                    ],
                    'dive_nav_file': [
                        'D:/AUV/VOYIS/PC-24-03/DIVE003/Vehicle_Data/DIVE003_Veh_Nav.txt',
                        'D:/AUV/VOYIS/PC-24-04/DIVE004/Vehicle_Data/DIVE004_Veh_Nav.txt'
                    ],
                    'LLS_Input': [
                        'D:/AUV/VOYIS/PC-24-03/DIVE003/LLS',
                        'D:/AUV/VOYIS/PC-24-04/DIVE004/LLS'
                    ],
                    'PhinsData_Bin_file': [
                        'D:/AUV/VOYIS/PC-24-03/DIVE003/Vehicle_Data/phinsdata_20240627_0535.bin',
                        'D:/AUV/VOYIS/PC-24-04/DIVE004/Vehicle_Data/phinsdata_20240628_0630.bin'
                    ],
                    'Image_Input': [
                        'D:/AUV/VOYIS/PC-24-03/DIVE003/DIVE003_raw_jpg_advanced',
                        'D:/AUV/VOYIS/PC-24-04/DIVE004/DIVE004_raw_jpg_advanced'
                    ],
                    'Output_folder': [
                        'D:/AUV/VOYIS/PC-24-03/DIVE003/Report_Plots',
                        'D:/AUV/VOYIS/PC-24-04/DIVE004/Report_Plots'
                    ]
                }
                
                df = pd.DataFrame(template_data)
                df.to_csv(file_path, index=False)
                
                self.log_message(f"Batch CSV template created: {file_path}")
                messagebox.showinfo(
                    "Template Created", 
                    f"Batch processing CSV template created at:\n{file_path}\n\n"
                    "Edit this file with your actual folder paths, then load it for batch processing.\n\n"
                    "Required: Output_folder (always required)\n"
                    "Navigation Module: nav_directory (auto-detects all nav files)\n"
                    "Image Analysis Module: Image_Input, dive_nav_file (optional individual nav file for legacy imagery processing)\n"
                    "LLS Analysis Module: LLS_Input, PhinsData_Bin_file\n"
                    "Each module can be run independently."
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

    def validate_navigation_files(self):
        """Scan selected navigation files and report missing required columns"""
        from src.models.nav_merger import NavigationDataMerger
        
        # Get list of selected navigation files
        nav_files = []
        file_info = []
        
        if self.nav_phins_ins_path.get():
            nav_files.append(self.nav_phins_ins_path.get())
            file_info.append(("PHINS INS (Priority 1)", self.nav_phins_ins_path.get()))
            
        if self.nav_nav_state_path.get():
            nav_files.append(self.nav_nav_state_path.get())
            file_info.append(("NAV_STATE (Priority 2)", self.nav_nav_state_path.get()))
            
        if self.nav_state_only_path.get():
            nav_files.append(self.nav_state_only_path.get())
            file_info.append(("STATE (Priority 3)", self.nav_state_only_path.get()))
            
        if self.nav_adcp_path.get():
            nav_files.append(self.nav_adcp_path.get())
            file_info.append(("ADCP (Priority 4)", self.nav_adcp_path.get()))
            
        if self.nav_veh_data_path.get():
            nav_files.append(self.nav_veh_data_path.get())
            file_info.append(("Veh_Data (Priority 5)", self.nav_veh_data_path.get()))
        
        if not nav_files:
            self.log_message("No navigation files selected to validate.")
            return
        
        self.log_message(f"\n=== Navigation File Validation ===")
        self.log_message(f"Scanning {len(nav_files)} navigation files...")
        
        # Initialize navigation merger for validation
        nav_merger = NavigationDataMerger(self.log_message)
        
        required_columns = ['time', 'latitude', 'longitude', 'depth']
        optional_columns = ['heading', 'pitch', 'roll', 'heave', 'altitude']
        all_available_cols = set()
        file_reports = []
        
        for (file_desc, file_path) in file_info:
            self.log_message(f"\nAnalyzing: {file_desc}")
            self.log_message(f"  File: {os.path.basename(file_path)}")
            
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    self.log_message(f"  ❌ ERROR: File not found")
                    continue
                
                # Identify file type
                file_type = nav_merger.identify_file_type(file_path)
                self.log_message(f"  📋 Detected type: {file_type}")
                
                # Try to load and get column info
                try:
                    df = nav_merger.load_and_standardize_file(file_path, file_type)
                    if df is not None and not df.empty:
                        available_cols = list(df.columns)
                        standardized_cols = [col for col in available_cols if col in (required_columns + optional_columns)]
                        all_available_cols.update(standardized_cols)
                        
                        # Check required columns
                        missing_required = [col for col in required_columns if col not in available_cols]
                        available_required = [col for col in required_columns if col in available_cols]
                        
                        # Check optional columns  
                        available_optional = [col for col in optional_columns if col in available_cols]
                        missing_optional = [col for col in optional_columns if col not in available_cols]
                        
                        self.log_message(f"  📊 Data rows: {len(df)}")
                        self.log_message(f"  ✅ Required columns found: {available_required}")
                        if missing_required:
                            self.log_message(f"  ❌ Missing required: {missing_required}")
                        else:
                            self.log_message(f"  ✅ All required columns present!")
                            
                        if available_optional:
                            self.log_message(f"  🔵 Optional columns found: {available_optional}")
                        if missing_optional:
                            self.log_message(f"  ⚪ Optional columns missing: {missing_optional}")
                        
                        file_reports.append({
                            'file': file_desc,
                            'path': file_path,
                            'type': file_type,
                            'rows': len(df),
                            'required_available': available_required,
                            'required_missing': missing_required,
                            'optional_available': available_optional,
                            'optional_missing': missing_optional,
                            'status': 'OK' if not missing_required else 'MISSING_REQUIRED'
                        })
                        
                    else:
                        self.log_message(f"  ❌ ERROR: Could not load data from file")
                        
                except Exception as load_error:
                    self.log_message(f"  ❌ ERROR loading file: {str(load_error)}")
                    
            except Exception as e:
                self.log_message(f"  ❌ ERROR analyzing file: {str(e)}")
        
        # Summary report
        self.log_message(f"\n=== Validation Summary ===")
        
        files_with_required = [r for r in file_reports if r['status'] == 'OK']
        files_missing_required = [r for r in file_reports if r['status'] == 'MISSING_REQUIRED']
        
        if files_with_required:
            self.log_message(f"✅ Files with all required columns ({len(files_with_required)}):")
            for report in files_with_required:
                self.log_message(f"   • {report['file']} ({report['rows']} rows)")
        
        if files_missing_required:
            self.log_message(f"❌ Files missing required columns ({len(files_missing_required)}):")
            for report in files_missing_required:
                self.log_message(f"   • {report['file']}: missing {report['required_missing']}")
        
        # Overall coverage assessment
        missing_overall = [col for col in required_columns if col not in all_available_cols]
        available_overall = [col for col in required_columns if col in all_available_cols]
        
        self.log_message(f"\n📈 Overall Coverage Across All Files:")
        self.log_message(f"✅ Available required data: {available_overall}")
        if missing_overall:
            self.log_message(f"❌ Still missing after combining files: {missing_overall}")
            self.log_message(f"⚠️  WARNING: These attributes need to be added from additional sources")
        else:
            self.log_message(f"🎉 SUCCESS: All required navigation data will be available after merging!")
        
        # Optional coverage
        available_optional_overall = [col for col in optional_columns if col in all_available_cols] 
        missing_optional_overall = [col for col in optional_columns if col not in all_available_cols]
        
        if available_optional_overall:
            self.log_message(f"🔵 Available optional data: {available_optional_overall}")
        if missing_optional_overall:
            self.log_message(f"⚪ Missing optional data: {missing_optional_overall}")
            
        self.log_message(f"=== End Validation ===\n")
        
        return file_reports

    def quick_validate_nav_file(self, file_path, file_desc):
        """Quickly validate a single navigation file and provide immediate feedback"""
        try:
            from src.models.nav_merger import NavigationDataMerger
            
            if not os.path.exists(file_path):
                self.log_message(f"⚠️  {file_desc} file not found: {os.path.basename(file_path)}")
                return
            
            nav_merger = NavigationDataMerger(self.log_message)
            file_type = nav_merger.identify_file_type(file_path)
            
            # Quick check - just load headers and first few rows
            try:
                df = nav_merger.load_and_standardize_file(file_path, file_type)
                if df is not None and not df.empty:
                    required_columns = ['time', 'latitude', 'longitude', 'depth']
                    available_required = [col for col in required_columns if col in df.columns]
                    missing_required = [col for col in required_columns if col not in df.columns]
                    
                    if not missing_required:
                        self.log_message(f"✅ {file_desc}: All required columns found ({len(df)} rows)")
                    else:
                        self.log_message(f"⚠️  {file_desc}: Missing required columns: {missing_required}")
                        self.log_message(f"   Available: {available_required}")
                else:
                    self.log_message(f"❌ {file_desc}: Could not load data from file")
                    
            except Exception as e:
                self.log_message(f"❌ {file_desc}: Error loading file - {str(e)}")
                
        except Exception as e:
            self.log_message(f"❌ Error validating {file_desc}: {str(e)}")

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