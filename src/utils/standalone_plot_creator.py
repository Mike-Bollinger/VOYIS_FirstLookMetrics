"""
Standalone Plot Creator for VOYIS First Look Metrics
=====================================================
This script allows users to regenerate plots from existing Image_Metrics.csv files
without re-running the entire analysis pipeline.

Features:
- GUI for selecting parent folder
- Recursive search for DIVE###_Image_Metrics.csv files
- Radio buttons to select which plots to regenerate
- Currently supports: Visibility Distribution Analysis
- Designed to be extensible for other plot types

Author: Mike Bollinger, Github Copilot
Date: January 2026
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import traceback
from pathlib import Path


class StandalonePlotCreator:
    """GUI application for recreating plots from Image_Metrics CSV files"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("VOYIS-FLM Standalone Plot Creator")
        self.root.geometry("900x700")
        
        # Lazy import cv2 to avoid import errors on startup
        self.cv2 = None
        
        # Data containers
        self.csv_files: List[str] = []
        self.selected_csvs: List[bool] = []
        
        # Plot types available
        self.plot_types = {
            'visibility': tk.BooleanVar(value=True),
            # Add more plot types here in the future:
            # 'altitude': tk.BooleanVar(value=False),
            # 'footprint': tk.BooleanVar(value=False),
            # 'overlap': tk.BooleanVar(value=False),
        }
        
        # Color scheme for visibility categories
        self.color_map = {
            'zero': '#8B0000',        # Dark red
            'poor': '#FF4500',        # Orange red
            'fair': '#FFD700',        # Gold
            'good': '#32CD32',        # Lime green
            'excellent': '#006400'    # Dark green
        }
        
        # Category order
        self.category_order = ['zero', 'poor', 'fair', 'good', 'excellent']
        
        # Build the GUI
        self.create_ui()
        
        # Set window icon
        self.root.after(100, self.set_window_icon)
    
    def set_window_icon(self):
        """Set the window icon using the NOAA logo"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(current_dir, "NOAA_VOYIS_Logo.ico")
            
            if os.path.exists(logo_path):
                self.root.iconbitmap(logo_path)
        except Exception as e:
            print(f"Note: Could not set window icon: {str(e)}")
    
    def create_ui(self):
        """Create the user interface"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="VOYIS-FLM Standalone Plot Creator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(main_frame, text="Select Parent Folder", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.folder_path = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_path, width=60)
        folder_entry.grid(row=0, column=0, padx=5)
        
        browse_btn = ttk.Button(folder_frame, text="Browse...", command=self.browse_folder)
        browse_btn.grid(row=0, column=1, padx=5)
        
        search_btn = ttk.Button(folder_frame, text="Search for CSV Files", 
                               command=self.search_csv_files)
        search_btn.grid(row=0, column=2, padx=5)
        
        # CSV files list section
        csv_frame = ttk.LabelFrame(main_frame, text="Found CSV Files", padding="10")
        csv_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        csv_frame.columnconfigure(0, weight=1)
        csv_frame.rowconfigure(0, weight=1)
        
        # Create treeview for CSV files with checkboxes
        self.csv_tree = ttk.Treeview(csv_frame, columns=('path', 'images'), 
                                     show='tree headings', height=10)
        self.csv_tree.heading('#0', text='✓')
        self.csv_tree.heading('path', text='CSV File')
        self.csv_tree.heading('images', text='# Images')
        
        self.csv_tree.column('#0', width=30, stretch=False)
        self.csv_tree.column('path', width=500)
        self.csv_tree.column('images', width=100)
        
        self.csv_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for treeview
        csv_scrollbar_y = ttk.Scrollbar(csv_frame, orient=tk.VERTICAL, 
                                       command=self.csv_tree.yview)
        csv_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.csv_tree.configure(yscrollcommand=csv_scrollbar_y.set)
        
        csv_scrollbar_x = ttk.Scrollbar(csv_frame, orient=tk.HORIZONTAL, 
                                       command=self.csv_tree.xview)
        csv_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.csv_tree.configure(xscrollcommand=csv_scrollbar_x.set)
        
        # Bind click event to toggle checkboxes
        self.csv_tree.bind('<Button-1>', self.toggle_checkbox)
        
        # Selection buttons
        select_frame = ttk.Frame(csv_frame)
        select_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(select_frame, text="Select All", 
                  command=self.select_all_csvs).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="Deselect All", 
                  command=self.deselect_all_csvs).pack(side=tk.LEFT, padx=5)
        
        # Plot selection section
        plot_frame = ttk.LabelFrame(main_frame, text="Select Plots to Generate", padding="10")
        plot_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(plot_frame, text="Visibility Distribution Analysis", 
                       variable=self.plot_types['visibility'], 
                       value=True).grid(row=0, column=0, sticky=tk.W, padx=20)
        
        # Placeholder for future plot types
        ttk.Label(plot_frame, text="(More plot types coming soon...)", 
                 foreground='gray').grid(row=1, column=0, sticky=tk.W, padx=20)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.generate_btn = ttk.Button(action_frame, text="Generate Plots", 
                                      command=self.generate_plots, 
                                      state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(action_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.LEFT, padx=10)
        
        # Progress and log section
        log_frame = ttk.LabelFrame(main_frame, text="Progress & Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        # Progress bar
        self.progress = ttk.Progressbar(log_frame, mode='determinate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=2)
        main_frame.rowconfigure(5, weight=1)
    
    def log(self, message: str):
        """Add a message to the log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def browse_folder(self):
        """Open folder browser dialog"""
        folder = filedialog.askdirectory(title="Select Parent Folder")
        if folder:
            self.folder_path.set(folder)
            self.log(f"Selected folder: {folder}")
    
    def search_csv_files(self):
        """Recursively search for DIVE###_Image_Metrics.csv files"""
        folder = self.folder_path.get()
        
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder")
            return
        
        self.log("Searching for CSV files...")
        self.csv_files = []
        
        # Clear existing tree
        for item in self.csv_tree.get_children():
            self.csv_tree.delete(item)
        
        # Search for CSV files matching pattern
        pattern = os.path.join(folder, '**', '*Image_Metrics.csv')
        found_files = glob.glob(pattern, recursive=True)
        
        # Filter for DIVE pattern specifically (DIVE###_Image_Metrics.csv)
        import re
        dive_pattern = re.compile(r'DIVE\d+_Image_Metrics\.csv$', re.IGNORECASE)
        for csv_file in found_files:
            filename = os.path.basename(csv_file)
            if dive_pattern.search(filename):
                self.csv_files.append(csv_file)
                
                # Try to read CSV to get image count
                try:
                    df = pd.read_csv(csv_file)
                    image_count = len(df)
                except Exception as e:
                    image_count = "Error"
                    self.log(f"Warning: Could not read {filename}: {e}")
                
                # Add to tree with checkbox
                self.csv_tree.insert('', tk.END, text='☐', 
                                    values=(csv_file, image_count),
                                    tags=('unchecked',))
        
        self.log(f"Found {len(self.csv_files)} CSV file(s)")
        
        # Enable generate button if files found
        if self.csv_files:
            self.generate_btn.configure(state=tk.NORMAL)
        else:
            messagebox.showinfo("No Files Found", 
                              "No Image_Metrics.csv files found in the selected folder")
    
    def toggle_checkbox(self, event):
        """Toggle checkbox on tree item click"""
        region = self.csv_tree.identify_region(event.x, event.y)
        if region == "tree":
            item = self.csv_tree.identify_row(event.y)
            if item:
                # Toggle the checkbox
                current_tags = self.csv_tree.item(item, 'tags')
                if 'checked' in current_tags:
                    self.csv_tree.item(item, text='☐', tags=('unchecked',))
                else:
                    self.csv_tree.item(item, text='☑', tags=('checked',))
    
    def select_all_csvs(self):
        """Select all CSV files in the list"""
        for item in self.csv_tree.get_children():
            self.csv_tree.item(item, text='☑', tags=('checked',))
    
    def deselect_all_csvs(self):
        """Deselect all CSV files in the list"""
        for item in self.csv_tree.get_children():
            self.csv_tree.item(item, text='☐', tags=('unchecked',))
    
    def get_selected_csvs(self) -> List[str]:
        """Get list of selected CSV files"""
        selected = []
        for item in self.csv_tree.get_children():
            if 'checked' in self.csv_tree.item(item, 'tags'):
                csv_path = self.csv_tree.item(item, 'values')[0]
                selected.append(csv_path)
        return selected
    
    def generate_plots(self):
        """Generate selected plots for selected CSV files"""
        selected_csvs = self.get_selected_csvs()
        
        if not selected_csvs:
            messagebox.showwarning("No Selection", "Please select at least one CSV file")
            return
        
        # Check which plots are selected
        plots_to_generate = []
        if self.plot_types['visibility'].get():
            plots_to_generate.append('visibility')
        
        if not plots_to_generate:
            messagebox.showwarning("No Plots Selected", "Please select at least one plot type")
            return
        
        self.log("\n" + "="*60)
        self.log(f"Starting plot generation for {len(selected_csvs)} CSV file(s)")
        self.log(f"Plot types: {', '.join(plots_to_generate)}")
        self.log("="*60 + "\n")
        
        # Setup progress bar
        total_operations = len(selected_csvs) * len(plots_to_generate)
        self.progress['maximum'] = total_operations
        self.progress['value'] = 0
        
        success_count = 0
        error_count = 0
        
        # Generate plots for each CSV
        for idx, csv_path in enumerate(selected_csvs, 1):
            csv_filename = os.path.basename(csv_path)
            self.log(f"\n[{idx}/{len(selected_csvs)}] Processing: {csv_filename}")
            
            for plot_type in plots_to_generate:
                try:
                    if plot_type == 'visibility':
                        result = self.generate_visibility_plot(csv_path)
                        if result:
                            self.log(f"  ✓ Visibility plot created: {result}")
                            success_count += 1
                        else:
                            self.log(f"  ✗ Failed to create visibility plot")
                            error_count += 1
                    
                    # Add more plot types here in the future
                    
                except Exception as e:
                    self.log(f"  ✗ Error generating {plot_type} plot: {e}")
                    self.log(traceback.format_exc())
                    error_count += 1
                
                # Update progress
                self.progress['value'] += 1
                self.root.update_idletasks()
        
        # Summary
        self.log("\n" + "="*60)
        self.log(f"Plot generation complete!")
        self.log(f"Success: {success_count} | Errors: {error_count}")
        self.log("="*60)
        
        # Play completion sound
        self.play_completion_sound()
        
        messagebox.showinfo("Complete", 
                          f"Plot generation complete!\n\n"
                          f"Successful: {success_count}\n"
                          f"Errors: {error_count}")
    
    def generate_visibility_plot(self, csv_path: str) -> Optional[str]:
        """
        Generate visibility distribution plot from CSV file
        
        Args:
            csv_path: Path to the Image_Metrics CSV file
            
        Returns:
            Path to the generated plot, or None if failed
        """
        try:
            # Import cv2 only when needed
            if self.cv2 is None:
                try:
                    import cv2
                    self.cv2 = cv2
                except ImportError:
                    self.log(f"    Error: opencv-python (cv2) is not installed")
                    self.log(f"    Install it with: pip install opencv-python")
                    return None
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Verify required columns exist
            if 'visibility' not in df.columns:
                self.log(f"    Warning: 'visibility' column not found in CSV")
                return None
            
            # Determine output path - save in same directory as CSV
            csv_dir = os.path.dirname(csv_path)
            csv_basename = os.path.basename(csv_path).replace('_Image_Metrics.csv', '')
            output_path = os.path.join(csv_dir, f"{csv_basename}_Image_Visibility_Analysis.png")
            
            # Get visibility counts
            visibility_counts = df['visibility'].value_counts()
            
            # Create figure with improved layout - chart on top, images below
            fig = plt.figure(figsize=(16, 10))
            
            # Bar chart at the top (spanning full width, 2/3 of plot)
            ax_bar = plt.subplot2grid((3, 5), (0, 0), colspan=5, rowspan=2)
            
            # Prepare data for bar chart
            categories_present = [cat for cat in self.category_order 
                                if cat in visibility_counts.index]
            counts = [visibility_counts.get(cat, 0) for cat in categories_present]
            colors = [self.color_map.get(cat, '#808080') for cat in categories_present]
            
            bars = ax_bar.bar(range(len(categories_present)), counts, color=colors, 
                            edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(count)}',
                          ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax_bar.set_title('Visibility Distribution Analysis', fontsize=18, 
                           fontweight='bold', pad=20)
            ax_bar.set_xlabel('Visibility Category', fontsize=14, fontweight='bold')
            ax_bar.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
            ax_bar.set_xticks(range(len(categories_present)))
            ax_bar.set_xticklabels([cat.title() for cat in categories_present], 
                                  fontsize=12)
            ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
            ax_bar.set_axisbelow(True)
            
            # Example thumbnails below the chart (one row, 5 columns, bottom 1/3)
            thumbnail_axes = []
            for i in range(5):
                ax_thumb = plt.subplot2grid((3, 5), (2, i), rowspan=1)
                thumbnail_axes.append(ax_thumb)
            
            # Find and display example images for each category
            for idx, category in enumerate(self.category_order):
                if idx >= len(thumbnail_axes):
                    break
                
                ax = thumbnail_axes[idx]
                
                # Find images from this category, sorted by confidence (highest first)
                category_images = df[df['visibility'] == category]
                
                if len(category_images) > 0:
                    # Get highest confidence image
                    category_images_sorted = category_images.sort_values(
                        'visibility_confidence', ascending=False)
                    best_image = category_images_sorted.iloc[0]
                    
                    # Try to locate the image
                    image_path = self.find_image_from_csv(best_image, csv_path)
                    
                    # Load and display the thumbnail
                    if image_path and os.path.exists(image_path):
                        try:
                            img = self.cv2.imread(image_path)
                            if img is not None:
                                img_rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                                ax.imshow(img_rgb)
                                confidence_pct = best_image['visibility_confidence'] * 100
                                ax.set_title(f"{category.title()}\nConfidence: {confidence_pct:.1f}%", 
                                           fontsize=11, fontweight='bold',
                                           color=self.color_map.get(category, '#000000'),
                                           pad=8)
                                self.log(f"    ✓ Loaded {category} thumbnail")
                            else:
                                ax.text(0.5, 0.5, f'{category.title()}\nLoad Failed', 
                                      ha='center', va='center', fontsize=10)
                        except Exception as e:
                            self.log(f"    Warning: Error loading {category} thumbnail: {e}")
                            ax.text(0.5, 0.5, f'{category.title()}\nImage Error', 
                                  ha='center', va='center', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, f'{category.title()}\nNot Found', 
                              ha='center', va='center', fontsize=10)
                else:
                    ax.text(0.5, 0.5, f'{category.title()}\nNo Data', 
                          ha='center', va='center', fontsize=10, color='gray')
                
                ax.axis('off')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.log(f"    Error creating visibility plot: {e}")
            self.log(traceback.format_exc())
            return None
    
    def find_image_from_csv(self, row: pd.Series, csv_path: str) -> Optional[str]:
        """
        Find the image file path from CSV row data
        
        Args:
            row: Pandas series containing image data
            csv_path: Path to the CSV file (used to determine search locations)
            
        Returns:
            Path to the image file, or None if not found
        """
        # Try to get path from CSV first
        if 'file_path' in row and pd.notna(row['file_path']):
            if os.path.exists(row['file_path']):
                return row['file_path']
        
        # Get image filename
        if 'filename' in row:
            image_filename = row['filename']
        elif 'image' in row:
            image_filename = row['image']
        else:
            return None
        
        # Search in common locations relative to CSV
        csv_dir = os.path.dirname(csv_path)
        parent_dir = os.path.dirname(csv_dir)
        
        # Common directory names where images might be
        search_dirs = [
            csv_dir,
            parent_dir,
            os.path.join(parent_dir, 'image_raw'),
            os.path.join(parent_dir, 'images'),
            os.path.join(parent_dir, 'Image_raw'),
            os.path.join(parent_dir, 'test_images_raw'),
            os.path.join(parent_dir, 'test_images_auv_proc'),
            os.path.join(parent_dir, 'test_images_viewls_proc'),
        ]
        
        # Search for the image
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                test_path = os.path.join(search_dir, image_filename)
                if os.path.exists(test_path):
                    return test_path
        
        # Try recursive search as last resort (slower)
        try:
            for root, dirs, files in os.walk(parent_dir):
                if image_filename in files:
                    return os.path.join(root, image_filename)
        except Exception:
            pass
        
        return None
    
    def play_completion_sound(self):
        """Play a sound when plot generation completes"""
        try:
            import winsound
            sound_file = os.path.join(os.path.dirname(__file__), 'sounds', 'beer_open.wav')
            if os.path.exists(sound_file):
                # Play the custom WAV file asynchronously
                winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Use system sound as fallback if custom sound not found
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception as e:
            # Silently fail if sound can't play
            pass


def main():
    """Main entry point for standalone plot creator"""
    root = tk.Tk()
    app = StandalonePlotCreator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
