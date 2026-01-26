"""
Legacy First Look CSV Merger
Merges visibility_results.csv, image_locations.csv, and footprints.csv into a single Image_Metrics file.
Designed to process legacy dive data with a simple GUI interface.
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import re
from datetime import datetime


class CSVMergerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Legacy CSV Merger - VOYIS First Look")
        self.root.geometry("700x400")
        
        # Variables
        self.parent_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI elements"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Legacy First Look CSV Merger", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=20)
        
        # Parent folder selection
        parent_frame = tk.Frame(self.root)
        parent_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(parent_frame, text="Parent Folder (e.g., G:\\PC2403_Voyis\\):", 
                 font=("Arial", 10)).pack(anchor='w')
        
        parent_entry_frame = tk.Frame(parent_frame)
        parent_entry_frame.pack(fill='x', pady=5)
        
        tk.Entry(parent_entry_frame, textvariable=self.parent_folder, 
                 width=60).pack(side='left', fill='x', expand=True)
        tk.Button(parent_entry_frame, text="Browse...", 
                  command=self.browse_parent).pack(side='left', padx=5)
        
        # Output folder selection
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(output_frame, text="Additional Output Folder (optional):", 
                 font=("Arial", 10)).pack(anchor='w')
        
        output_entry_frame = tk.Frame(output_frame)
        output_entry_frame.pack(fill='x', pady=5)
        
        tk.Entry(output_entry_frame, textvariable=self.output_folder, 
                 width=60).pack(side='left', fill='x', expand=True)
        tk.Button(output_entry_frame, text="Browse...", 
                  command=self.browse_output).pack(side='left', padx=5)
        
        # Process button
        process_btn = tk.Button(
            self.root, 
            text="Process Dive Folders", 
            command=self.process_folders,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        process_btn.pack(pady=30)
        
        # Status text
        self.status_text = tk.Text(self.root, height=8, width=80, state='disabled')
        self.status_text.pack(pady=10, padx=20)
    
    def browse_parent(self):
        """Browse for parent folder"""
        folder = filedialog.askdirectory(title="Select Parent Folder (containing Dive folders)")
        if folder:
            self.parent_folder.set(folder)
    
    def browse_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Additional Output Folder")
        if folder:
            self.output_folder.set(folder)
    
    def log_status(self, message):
        """Add message to status text"""
        # Log to both GUI and terminal
        print(message)  # Terminal logging for debugging
        self.status_text.config(state='normal')
        self.status_text.insert('end', message + '\n')
        self.status_text.see('end')
        self.status_text.config(state='disabled')
        self.root.update()
    
    def process_folders(self):
        """Process all dive folders"""
        parent = self.parent_folder.get()
        output = self.output_folder.get()
        
        if not parent:
            messagebox.showerror("Error", "Please select a parent folder")
            return
        
        if not os.path.exists(parent):
            messagebox.showerror("Error", "Parent folder does not exist")
            return
        
        # Clear status
        self.status_text.config(state='normal')
        self.status_text.delete(1.0, 'end')
        self.status_text.config(state='disabled')
        
        self.log_status(f"Scanning parent folder: {parent}")
        
        # Find all dive folders
        dive_folders = self.find_dive_folders(parent)
        
        if not dive_folders:
            self.log_status("No dive folders found!")
            messagebox.showwarning("Warning", "No dive folders found in the parent directory")
            return
        
        self.log_status(f"Found {len(dive_folders)} dive folder(s)\n")
        
        # Process each dive folder
        success_count = 0
        for dive_folder in dive_folders:
            try:
                self.process_single_dive(dive_folder, output)
                success_count += 1
            except Exception as e:
                self.log_status(f"  ERROR: {str(e)}\n")
        
        self.log_status(f"\n{'='*60}")
        self.log_status(f"Processing complete!")
        self.log_status(f"Successfully processed: {success_count}/{len(dive_folders)} dives")
        
        messagebox.showinfo("Complete", 
                           f"Processing complete!\n{success_count}/{len(dive_folders)} dives processed successfully")
    
    def find_dive_folders(self, parent_path):
        """Find all folders that look like dive folders (case-insensitive: Dive###, DIVE###, dive###)"""
        dive_folders = []
        parent = Path(parent_path)
        
        # Look for folders named like "Dive###", "DIVE###", or "dive###" (case-insensitive)
        for item in parent.iterdir():
            if item.is_dir():
                # Check if folder name matches dive pattern (case-insensitive)
                if re.match(r'dive\d+', item.name, re.IGNORECASE):
                    # Check if it has either Image_Outputs or Imagery_Outputs subfolder
                    image_outputs = item / "Image_Outputs"
                    imagery_outputs = item / "Imagery_Outputs"
                    
                    if image_outputs.exists():
                        print(f"  Found dive folder: {item.name} (Image_Outputs)")
                        dive_folders.append(item)
                    elif imagery_outputs.exists():
                        print(f"  Found dive folder: {item.name} (Imagery_Outputs)")
                        dive_folders.append(item)
                    else:
                        print(f"  Skipping {item.name} (no Image_Outputs or Imagery_Outputs folder)")
        
        return sorted(dive_folders)
    
    def extract_dive_number(self, dive_path):
        """Extract dive number from path"""
        dive_name = Path(dive_path).name
        match = re.search(r'dive(\d+)', dive_name, re.IGNORECASE)
        if match:
            return match.group(1).zfill(3)  # Pad with zeros to 3 digits
        return "000"
    
    def process_single_dive(self, dive_path, additional_output):
        """Process a single dive folder"""
        dive_path = Path(dive_path)
        dive_number = self.extract_dive_number(dive_path)
        
        print(f"\n{'='*60}")
        print(f"Processing {dive_path.name}...")
        self.log_status(f"Processing {dive_path.name}...")
        
        # Locate output folder (try both Image_Outputs and Imagery_Outputs)
        image_outputs = dive_path / "Image_Outputs"
        if not image_outputs.exists():
            image_outputs = dive_path / "Imagery_Outputs"
        
        if not image_outputs.exists():
            raise Exception(f"Neither Image_Outputs nor Imagery_Outputs folder found in {dive_path.name}")
        
        print(f"  Using output folder: {image_outputs.name}")
        
        # Find the three CSV files (try both naming conventions)
        visibility_csv = image_outputs / "visibility_results.csv"
        locations_csv = image_outputs / "image_locations.csv"
        
        # Try both footprint naming conventions
        footprints_csv = image_outputs / "footprints.csv"
        if not footprints_csv.exists():
            footprints_csv = image_outputs / "image_footprints.csv"
        
        # Check which files exist
        files_found = []
        if visibility_csv.exists():
            files_found.append("visibility_results")
        if locations_csv.exists():
            files_found.append("image_locations")
        if footprints_csv.exists():
            files_found.append(footprints_csv.name)
        
        if not files_found:
            raise Exception(f"No CSV files found in {image_outputs}")
        
        print(f"  Found files: {', '.join(files_found)}")
        self.log_status(f"  Found: {', '.join(files_found)}")
        
        # Read and merge CSVs
        print(f"  Starting merge...")
        merged_df = self.merge_csvs(visibility_csv, locations_csv, footprints_csv)
        print(f"  Merge complete!")
        
        # Create output filename
        output_filename = f"DIVE{dive_number}_Image_Metrics.csv"
        
        # Save to Image_Outputs folder
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saving to Image_Outputs...")
        primary_output = image_outputs / output_filename
        
        # Remove existing file if it exists to ensure we can overwrite
        if primary_output.exists():
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Removing existing file...")
            primary_output.unlink()
        
        merged_df.to_csv(primary_output, index=False)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: {primary_output}")
        self.log_status(f"  Saved to: {primary_output}")
        
        # Save to additional output folder if specified
        if additional_output and os.path.exists(additional_output):
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saving to additional output folder...")
            secondary_output = Path(additional_output) / output_filename
            
            # Remove existing file if it exists
            if secondary_output.exists():
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Removing existing file...")
                secondary_output.unlink()
            
            merged_df.to_csv(secondary_output, index=False)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: {secondary_output}")
            self.log_status(f"  Saved to: {secondary_output}")
        
        print(f"  ✓ Complete: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        self.log_status(f"  Result: {len(merged_df)} rows, {len(merged_df.columns)} columns\n")
    
    def merge_csvs(self, visibility_path, locations_path, footprints_path):
        """Merge the three CSV files, avoiding duplicate columns"""
        # ALWAYS start with image_locations as the base (if it exists)
        # This ensures all images are preserved even if some don't have visibility or footprint data
        
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Starting merge process...")
        
        if not locations_path.exists():
            raise Exception("image_locations.csv is required as the base file")
        
        # Start with image locations as the base
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Reading image_locations.csv...")
        merged = pd.read_csv(locations_path)
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Loaded image_locations: {len(merged)} rows, {len(merged.columns)} columns")
        self.log_status(f"  Base: image_locations ({len(merged)} rows)")
        
        # Merge footprints (LEFT JOIN to preserve all images) - MERGE ON 'filename'
        if footprints_path.exists():
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] Reading footprints.csv...")
            foot_df = pd.read_csv(footprints_path)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] Loaded footprints: {len(foot_df)} rows, {len(foot_df.columns)} columns")
            
            # Remove 'index' column if it exists
            if 'index' in foot_df.columns:
                foot_df = foot_df.drop('index', axis=1)
            
            # EXPLICITLY merge on 'filename' column
            if 'filename' in merged.columns and 'filename' in foot_df.columns:
                # Get columns to exclude from footprints (duplicates of non-key columns)
                exclude_cols = [col for col in foot_df.columns if col in merged.columns and col != 'filename']
                unique_cols = [col for col in foot_df.columns if col not in exclude_cols]
                
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] Merging footprints on 'filename' (adding {len(unique_cols)-1} columns)...")
                
                # LEFT JOIN to keep all images
                merged = pd.merge(
                    merged, 
                    foot_df[unique_cols], 
                    on='filename', 
                    how='left',
                    suffixes=('', '_dup')
                )
                
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] Footprints merged. Result: {len(merged)} rows, {len(merged.columns)} columns")
                self.log_status(f"  + footprints ({len(foot_df)} rows, {len(unique_cols)-1} new columns)")
                
                # Remove any duplicate columns
                dup_cols = [col for col in merged.columns if col.endswith('_dup')]
                if dup_cols:
                    merged = merged.drop(dup_cols, axis=1)
            else:
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] WARNING: 'filename' column not found, skipping footprints merge")
        
        # Merge visibility results (LEFT JOIN to preserve all images) - MERGE ON 'image' column
        if visibility_path.exists():
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] Reading visibility_results.csv...")
            vis_df = pd.read_csv(visibility_path)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] Loaded visibility: {len(vis_df)} rows, {len(vis_df.columns)} columns")
            
            # Visibility uses 'image' column, need to match with 'filename' from merged
            if 'image' in vis_df.columns and 'filename' in merged.columns:
                # Rename 'image' to 'filename' for merge
                vis_df = vis_df.rename(columns={'image': 'filename'})
                
                # Get columns to exclude from visibility (duplicates of non-key columns)
                exclude_cols = [col for col in vis_df.columns if col in merged.columns and col != 'filename']
                unique_cols = [col for col in vis_df.columns if col not in exclude_cols]
                
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] Merging visibility on 'filename' (adding {len(unique_cols)-1} columns)...")
                
                # LEFT JOIN to keep all images
                merged = pd.merge(
                    merged, 
                    vis_df[unique_cols], 
                    on='filename', 
                    how='left',
                    suffixes=('', '_dup')
                )
                
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] Visibility merged. Result: {len(merged)} rows, {len(merged.columns)} columns")
                self.log_status(f"  + visibility ({len(vis_df)} rows, {len(unique_cols)-1} new columns)")
                
                # Remove any duplicate columns
                dup_cols = [col for col in merged.columns if col.endswith('_dup')]
                if dup_cols:
                    merged = merged.drop(dup_cols, axis=1)
            else:
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] WARNING: 'image' or 'filename' column not found, skipping visibility merge")
        
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Merge complete. Final: {len(merged)} rows, {len(merged.columns)} columns")
        return merged


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CSVMergerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
