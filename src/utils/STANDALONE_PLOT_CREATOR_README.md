# Standalone Plot Creator

## Overview
The Standalone Plot Creator is a utility tool that allows you to regenerate plots from existing `Image_Metrics.csv` files without re-running the entire VOYIS First Look Metrics analysis pipeline.

## Features
- **GUI-based workflow** - Easy-to-use graphical interface
- **Recursive CSV search** - Automatically finds all `*Image_Metrics.csv` files in a directory tree
- **Batch processing** - Process multiple CSV files at once
- **Selective plot generation** - Choose which plots to regenerate
- **Currently supported plots:**
  - Visibility Distribution Analysis

## Usage

### Running the Script
There are two ways to run the standalone plot creator:

1. **From command line:**
   ```bash
   python -m src.utils.standalone_plot_creator
   ```

2. **As a standalone script:**
   ```bash
   cd src/utils
   python standalone_plot_creator.py
   ```

### Workflow

1. **Select Parent Folder**
   - Click "Browse..." to select the parent directory containing your dive data
   - This should be a folder that contains one or more dive folders (e.g., `DIVE001`, `DIVE029`, etc.)

2. **Search for CSV Files**
   - Click "Search for CSV Files" to recursively find all `*Image_Metrics.csv` files
   - The tool will display all found files with their paths and image counts

3. **Select CSV Files**
   - Check the boxes next to the CSV files you want to process
   - Use "Select All" or "Deselect All" buttons for convenience

4. **Select Plot Types**
   - Currently only "Visibility Distribution Analysis" is available
   - More plot types will be added in future updates

5. **Generate Plots**
   - Click "Generate Plots" to start the process
   - Progress and log messages will appear in the bottom panel
   - Plots are saved in the same directory as the source CSV files

## Output

### Visibility Distribution Analysis
The visibility analysis plot includes:
- **Bar chart** showing distribution of images across visibility categories (Zero, Poor, Fair, Good, Excellent)
- **Example thumbnails** showing the highest confidence image from each category
- **Layout**: Bar chart spans the full width at the top, thumbnails displayed in a row below

Output filename format: `{DIVE_NAME}_Visibility_Analysis.png`

## Technical Details

### Image Location
The script attempts to locate images using:
1. `file_path` column in the CSV (if available)
2. Common directory names relative to the CSV:
   - Same directory as CSV
   - `image_raw/`
   - `images/`
   - `Image_raw/`
   - `test_images_raw/`
   - `test_images_auv_proc/`
   - `test_images_viewls_proc/`
3. Recursive search in parent directory (as fallback)

### Required Columns in CSV
For visibility analysis, the CSV must contain:
- `visibility` - Visibility category (zero, poor, fair, good, excellent)
- `visibility_confidence` - Confidence score (0-1)
- `filename` or `image` - Image filename
- `file_path` (optional) - Full path to image file

## Future Enhancements
Planned additions include:
- Altitude distribution plots
- Footprint coverage maps
- Overlap analysis charts
- Custom plot configurations
- Export options (PDF, SVG)

## Troubleshooting

### "No CSV files found"
- Ensure you selected the correct parent folder
- CSV files must have "Image_Metrics.csv" in their filename

### "Warning: 'visibility' column not found"
- The CSV may not have visibility analysis data
- Re-run the main application with visibility analysis enabled

### "Image not found" or "Load Failed"
- Images may have been moved or deleted
- Check that image files are in expected locations relative to CSV
- Verify the `file_path` column in the CSV is accurate

### GUI doesn't appear
- Ensure you have tkinter installed (usually included with Python)
- On Linux, you may need: `sudo apt-get install python3-tk`

## Requirements
- Python 3.7+
- pandas
- opencv-python (cv2)
- matplotlib
- tkinter (usually included with Python)

## Author
VOYIS First Look Metrics Team
January 2026
