import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

class UIComponents:
    """Handles creation of all UI components"""
    
    def create_frames(self):
        """Create the main frames for the application with scrolling support"""
        LEFT_FRAME_WIDTH = 550
        
        # Create main container frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas container with scrolling
        canvas_container = ttk.Frame(main_frame)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Set up scrollable canvas
        self.left_canvas = tk.Canvas(canvas_container, width=LEFT_FRAME_WIDTH, borderwidth=0)
        left_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.left_canvas.yview)
        
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Create the main frame inside the canvas
        self.left_frame = ttk.Frame(self.left_canvas, width=LEFT_FRAME_WIDTH)
        self.left_canvas_window = self.left_canvas.create_window(
            (0, 0), window=self.left_frame, anchor="nw", width=LEFT_FRAME_WIDTH
        )
        
        # Configure bindings
        self.left_frame.bind("<Configure>", self._on_frame_configure)
        self.left_canvas.bind("<Configure>", self._on_canvas_configure)
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Right frame for log output
        self.right_frame = ttk.Frame(main_frame, padding="5")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_input_section(self):
        """Create the input/output file selection widgets"""
        self.input_frame = ttk.LabelFrame(self.left_frame, text="Input/Output Configuration", padding="10")
        self.input_frame.pack(fill=tk.X, pady=(5, 10), padx=10)
        self.input_frame.columnconfigure(1, weight=1)
        
        # Batch Processing Section - Make this prominent at the top
        self.create_batch_processing_section()
        
        # Single Processing Mode Section
        self.create_single_processing_section()

    def create_batch_processing_section(self):
        """Create the batch processing section"""
        batch_frame = ttk.LabelFrame(self.input_frame, text="Batch Processing Mode", padding="10")
        batch_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 15))
        batch_frame.columnconfigure(1, weight=1)
        
        # Large batch mode toggle button
        self.batch_mode_button = ttk.Button(
            batch_frame, 
            text="Enable Batch Processing Mode", 
            command=self.toggle_batch_mode_ui,
            style="AccentButton.TButton"
        )
        self.batch_mode_button.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky='ew')
        
        # Batch CSV file selection (initially hidden)
        self.batch_csv_frame = ttk.Frame(batch_frame)
        self.batch_csv_frame.grid(row=1, column=0, columnspan=3, sticky='ew')
        self.batch_csv_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.batch_csv_frame, text="Batch CSV File:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.batch_csv_entry = ttk.Entry(self.batch_csv_frame, textvariable=self.batch_csv_path, width=50)
        self.batch_csv_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.batch_csv_button = ttk.Button(self.batch_csv_frame, text="Browse...", command=self.select_batch_csv)
        self.batch_csv_button.grid(row=0, column=2)
        
        # Help text for batch CSV format
        help_text = ttk.Label(
            self.batch_csv_frame, 
            text="CSV must contain columns: input_folder, output_folder, nav_file (optional), lls_folder (optional), phins_nav_file (optional)",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        help_text.grid(row=1, column=0, columnspan=3, sticky='w', pady=(2, 0))
        
        # Create CSV template button
        template_button = ttk.Button(
            self.batch_csv_frame, 
            text="Create CSV Template", 
            command=self.create_batch_csv_template
        )
        template_button.grid(row=2, column=0, pady=(5, 0), sticky='w')
        
        # Initially hide the batch CSV selection
        self.batch_csv_frame.grid_remove()

    def create_single_processing_section(self):
        """Create the single processing section"""
        single_frame = ttk.LabelFrame(self.input_frame, text="Single Processing Mode", padding="5")
        single_frame.grid(row=1, column=0, columnspan=3, sticky='ew')
        single_frame.columnconfigure(1, weight=1)
        
        # LLS Input Section
        self._create_lls_input_section_in_frame(single_frame, 0)
        
        # Imagery Input Section  
        self._create_imagery_input_section_in_frame(single_frame, 1)
        
        # Output Section
        self._create_output_section_in_frame(single_frame, 2)

    def _create_lls_input_section_in_frame(self, parent, row):
        """Create LLS input section in specified frame"""
        lls_frame = ttk.LabelFrame(parent, text="Laser Data (LLS) Inputs", padding="5")
        lls_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        lls_frame.columnconfigure(1, weight=1)
        
        # LLS folder selection
        ttk.Label(lls_frame, text="LLS Folder:").grid(row=0, column=0, sticky='w')
        self.lls_entry = ttk.Entry(lls_frame, textvariable=self.lls_path, width=40)
        self.lls_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.lls_button = ttk.Button(lls_frame, text="Browse...", command=self.select_lls_folder)
        self.lls_button.grid(row=0, column=2)
        
        # Phins Nav file selection
        ttk.Label(lls_frame, text="Phins Nav File:").grid(row=1, column=0, sticky='w')
        self.phins_nav_entry = ttk.Entry(lls_frame, textvariable=self.phins_nav_path, width=40)
        self.phins_nav_entry.grid(row=1, column=1, padx=5, sticky='ew')
        self.phins_nav_button = ttk.Button(lls_frame, text="Browse...", command=self.select_phins_nav_file)
        self.phins_nav_button.grid(row=1, column=2)
        
        self.single_mode_frames = getattr(self, 'single_mode_frames', [])
        self.single_mode_frames.append(lls_frame)

    def _create_imagery_input_section_in_frame(self, parent, row):
        """Create imagery input section in specified frame"""
        imagery_frame = ttk.LabelFrame(parent, text="Imagery Inputs", padding="5")
        imagery_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        imagery_frame.columnconfigure(1, weight=1)
        
        # Input folder selection
        ttk.Label(imagery_frame, text="Input Folder:").grid(row=0, column=0, sticky='w')
        self.input_entry = ttk.Entry(imagery_frame, textvariable=self.input_path, width=40)
        self.input_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.input_button = ttk.Button(imagery_frame, text="Browse...", command=self.select_input_folder)
        self.input_button.grid(row=0, column=2)
        
        # Vehicle Nav file selection
        ttk.Label(imagery_frame, text="Vehicle Nav File:").grid(row=1, column=0, sticky='w')
        self.nav_entry = ttk.Entry(imagery_frame, textvariable=self.nav_path, width=40)
        self.nav_entry.grid(row=1, column=1, padx=5, sticky='ew')
        self.nav_button = ttk.Button(imagery_frame, text="Browse...", command=self.select_nav_file)
        self.nav_button.grid(row=1, column=2)
        
        self.single_mode_frames.append(imagery_frame)

    def _create_output_section_in_frame(self, parent, row):
        """Create output section in specified frame"""
        output_frame = ttk.LabelFrame(parent, text="Output", padding="5")
        output_frame.grid(row=row, column=0, columnspan=3, sticky='ew')
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky='w')
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=40)
        self.output_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.output_button = ttk.Button(output_frame, text="Browse...", command=self.select_output_folder)
        self.output_button.grid(row=0, column=2)

        self.single_mode_frames.append(output_frame)

    def create_functions_section(self):
        """Create the functions selection checkbox widgets"""
        self.functions_frame = ttk.LabelFrame(self.left_frame, text="Processing Functions", padding="10")
        self.functions_frame.pack(fill=tk.X, pady=(0, 10), padx=10)
        self.functions_frame.columnconfigure(0, weight=1)
        
        # Create subsections
        self._create_lls_processing_section()
        self._create_imagery_processing_section()

    def _create_lls_processing_section(self):
        """Create LLS processing section"""
        lls_section = ttk.LabelFrame(self.functions_frame, text="Laser Data Processing", padding="5")
        lls_section.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        self.lls_processing_checkbox = ttk.Checkbutton(
            lls_section, 
            text="Process LLS Data", 
            variable=self.lls_processing_var,
            command=self.update_all_checkbox
        )
        self.lls_processing_checkbox.grid(row=0, column=0, sticky='w')

    def _create_imagery_processing_section(self):
        """Create imagery processing section"""
        imagery_section = ttk.LabelFrame(self.functions_frame, text="Imagery Processing", padding="5")
        imagery_section.grid(row=1, column=0, columnspan=3, sticky='ew')
        
        # "All" checkbox
        self.all_checkbox = ttk.Checkbutton(
            imagery_section, text="All", 
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
        
        for i, (text, var) in enumerate(checkboxes, 1):
            checkbox = ttk.Checkbutton(
                imagery_section, text=text, 
                variable=var, 
                command=self.update_all_checkbox
            )
            checkbox.grid(row=i, column=0, sticky='w')
            
            # Special handling for visibility analyzer
            if text == "Visibility Analysis":
                self._create_visibility_options(imagery_section, i)

    def _create_visibility_options(self, parent, row):
        """Create visibility analyzer options"""
        self.visibility_model_frame = ttk.Frame(parent)
        self.visibility_model_row = row + 1
        
        # Model type selection
        model_type_frame = ttk.Frame(self.visibility_model_frame)
        model_type_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        
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

    def create_controls_section(self):
        """Create the controls section with threshold and process button"""
        controls_frame = ttk.LabelFrame(self.right_frame, text="Processing Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10), padx=10)
        
        # Altitude threshold
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Altitude Threshold (m):").pack(side=tk.LEFT)
        
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.pack(side=tk.RIGHT)
        self.threshold_var.trace('w', self.threshold_changed)
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"Current: {self.altitude_threshold:.1f}m")
        self.threshold_label.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Process button (larger and more prominent)
        style = ttk.Style()
        style.configure("AccentButton.TButton", font=('', 12, 'bold'))
        
        self.process_button = ttk.Button(
            controls_frame, text="Process Images", 
            command=self.process_images,
            style="AccentButton.TButton"
        )
        self.process_button.pack(pady=20, fill=tk.X)

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
        
        self.progress_var = tk.DoubleVar()
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
        x_screen, y_screen = self.root.winfo_pointerxy()
        widget_under_pointer = self.root.winfo_containing(x_screen, y_screen)
        
        if widget_under_pointer and (widget_under_pointer == self.left_canvas or 
                                    self.left_canvas.winfo_ismapped() and
                                    widget_under_pointer.winfo_toplevel() == self.root):
            if event.delta > 0:
                self.left_canvas.yview_scroll(-1, "units")
            else:
                self.left_canvas.yview_scroll(1, "units")