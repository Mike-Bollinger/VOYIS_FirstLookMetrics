import tkinter as tk
from tkinter import filedialog, messagebox
import os
import glob

class InputHandlers:
    """Handles all input file/folder selection and validation"""
    
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
            filetypes=[("Text Files", "*.txt"), ("Binary Files", "*.bin"), 
                      ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.phins_nav_path.set(file_path)
            self.log_message(f"Phins navigation file set to: {file_path}")

    def select_input_folder(self):
        """Select imagery input folder"""
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
            title="Select Vehicle Navigation File",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.nav_path.set(file_path)
            self.log_message(f"Vehicle navigation file set to: {file_path}")

    def select_visibility_file(self, file_type):
        """Select model file or training data directory for visibility analyzer"""
        if file_type == "model":
            default_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "v_a_pre-trained_models"
            )
            os.makedirs(default_dir, exist_ok=True)
            
            file_path = filedialog.askopenfilename(
                title="Select Pre-trained Visibility Model",
                initialdir=default_dir,
                filetypes=[("Model files", "*.h5"), ("All files", "*.*")]
            )
            if file_path:
                self.model_path.set(file_path)
                self.training_path.set("")
                self.model_type_var.set("model")
                self.log_message(f"Visibility model file set to: {file_path}")
        else:  # training data
            folder_path = filedialog.askdirectory(
                title="Select Visibility Training Data Directory"
            )
            if folder_path:
                self.training_path.set(folder_path)
                self.model_path.set("")
                self.model_type_var.set("training")
                self.log_message(f"Visibility training data directory set to: {folder_path}")

    def validate_inputs(self):
        """Validate all inputs before processing"""
        lls_selected = self.lls_processing_var.get()
        imagery_selected = any([
            self.basic_metrics_var.get(), self.location_map_var.get(),
            self.histogram_var.get(), self.footprint_map_var.get(),
            self.visibility_analyzer_var.get(), self.highlight_selector_var.get()
        ])
        
        if not lls_selected and not imagery_selected:
            messagebox.showwarning("No Functions Selected", 
                                 "Please select at least one processing function.")
            return False
        
        # Validate LLS inputs
        if lls_selected:
            if not self.lls_path.get():
                messagebox.showwarning("Missing LLS Input", "Please select an LLS folder.")
                return False
            if not self.phins_nav_path.get():
                messagebox.showwarning("Missing Phins Nav Input", 
                                     "Please select a Phins navigation file.")
                return False
        
        # Validate imagery inputs
        if imagery_selected:
            if not self.input_path.get():
                messagebox.showwarning("Missing Input", 
                                     "Please select an input folder for imagery processing.")
                return False
        
        # Output folder is required
        if not self.output_path.get():
            messagebox.showwarning("Missing Output", "Please select an output folder.")
            return False
        
        return True

    def toggle_all_functions(self):
        """Toggle all function checkboxes based on 'All Functions' checkbox"""
        state = self.all_var.get()
        self.basic_metrics_var.set(state)
        self.location_map_var.set(state)
        self.histogram_var.set(state)
        self.footprint_map_var.set(state)
        self.visibility_analyzer_var.set(state)
        self.highlight_selector_var.set(state)
        
        if state:
            self.toggle_visibility_options()
        else:
            self.visibility_model_frame.grid_remove()

    def update_all_checkbox(self):
        """Update 'All Functions' checkbox based on individual checkboxes"""
        if all([self.basic_metrics_var.get(), self.location_map_var.get(),
                self.histogram_var.get(), self.footprint_map_var.get(),
                self.visibility_analyzer_var.get(), self.highlight_selector_var.get()]):
            self.all_var.set(True)
        else:
            self.all_var.set(False)

    def toggle_visibility_options(self):
        """Show or hide visibility model options based on checkbox state"""
        if self.visibility_analyzer_var.get():
            self.visibility_model_frame.grid(
                row=self.visibility_model_row, column=0,
                columnspan=3, sticky='ew', padx=45, pady=(0, 5)
            )
        else:
            self.visibility_model_frame.grid_remove()
        
        self.update_all_checkbox()

    def toggle_batch_mode(self):
        """Toggle batch processing mode"""
        self.batch_mode = self.batch_var.get()
        if self.batch_mode:
            self.log_message("Batch processing mode enabled")
        else:
            self.log_message("Single processing mode enabled")

    def threshold_changed(self, *args):
        """Update the altitude threshold when changed"""
        try:
            new_threshold = float(self.threshold_var.get())
            self.altitude_threshold = new_threshold
            
            # Update all components with the new threshold
            if hasattr(self, 'metrics'):
                self.metrics.altitude_threshold = new_threshold
            if hasattr(self, 'altitude_map'):
                self.altitude_map.set_altitude_thresholds(new_threshold, self.low_altitude_threshold)
            if hasattr(self, 'footprint_map'):
                self.footprint_map.altitude_threshold = new_threshold
            if hasattr(self, 'visibility_analyzer'):
                self.visibility_analyzer.altitude_threshold = new_threshold
            
            self.threshold_label.config(text=f"Altitude Threshold: {new_threshold:.1f}m")
        except ValueError:
            self.log_message(f"Invalid threshold value. Using default: {self.altitude_threshold}")

    def validate_lls_inputs(self):
        """Validate LLS processing inputs"""
        if not self.lls_path.get():
            messagebox.showwarning("Missing LLS Input", "Please select an LLS folder.")
            return False
        
        if not self.phins_nav_path.get():
            messagebox.showwarning("Missing Phins Nav Input", "Please select a Phins navigation file.")
            return False
        
        if not os.path.exists(self.lls_path.get()):
            messagebox.showwarning("Invalid LLS Path", "LLS folder does not exist.")
            return False
        
        if not os.path.exists(self.phins_nav_path.get()):
            messagebox.showwarning("Invalid Phins Nav Path", "Phins navigation file does not exist.")
            return False
        
        # Check for LLS files in the folder
        lls_files = glob.glob(os.path.join(self.lls_path.get(), 'LLS_*.xyz'))
        if not lls_files:
            messagebox.showwarning("No LLS Files", "No LLS_*.xyz files found in the selected folder.")
            return False
        
        return True
    
    def validate_imagery_inputs(self):
        """Validate imagery processing inputs"""
        if not self.input_path.get():
            messagebox.showwarning("Missing Input", "Please select an input folder for imagery processing.")
            return False
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showwarning("Invalid Input Path", "Input folder does not exist.")
            return False
        
        # Check for image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_path.get(), f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(self.input_path.get(), f'*{ext.upper()}')))
        
        if not image_files:
            messagebox.showwarning("No Image Files", "No image files found in the selected folder.")
            return False
        
        return True
    
    def validate_output_folder(self):
        """Validate output folder"""
        if not self.output_path.get():
            messagebox.showwarning("Missing Output", "Please select an output folder.")
            return False
        
        # Try to create output folder if it doesn't exist
        try:
            os.makedirs(self.output_path.get(), exist_ok=True)
        except Exception as e:
            messagebox.showerror("Invalid Output Path", f"Cannot create output folder: {str(e)}")
            return False
        
        return True
    
    def check_nav_file_compatibility(self):
        """Check if navigation file is compatible with processing requirements"""
        nav_path = self.nav_path.get()
        if not nav_path or not os.path.exists(nav_path):
            return True  # Optional file
        
        # Basic file size check
        try:
            file_size = os.path.getsize(nav_path)
            if file_size == 0:
                messagebox.showwarning("Empty Nav File", "Navigation file appears to be empty.")
                return False
        except Exception as e:
            messagebox.showwarning("Nav File Error", f"Error reading navigation file: {str(e)}")
            return False
        
        return True
    
    def check_phins_nav_compatibility(self):
        """Check if Phins navigation file is compatible"""
        phins_path = self.phins_nav_path.get()
        if not phins_path or not os.path.exists(phins_path):
            return True  # Will be caught by main validation
        
        # Basic file checks
        try:
            file_size = os.path.getsize(phins_path)
            if file_size == 0:
                messagebox.showwarning("Empty Phins Nav File", "Phins navigation file appears to be empty.")
                return False
            
            # Check file extension compatibility
            _, ext = os.path.splitext(phins_path)
            if ext.lower() not in ['.bin', '.txt', '.csv']:
                result = messagebox.askyesno("Unknown File Type", 
                    f"Phins navigation file has extension '{ext}'. Continue anyway?")
                return result
                
        except Exception as e:
            messagebox.showwarning("Phins Nav File Error", f"Error reading Phins navigation file: {str(e)}")
            return False
        
        return True