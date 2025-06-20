# VOYIS First Look Metrics

**Version:** 1.4.0  
**Last Updated:** April 2025  
**Author:** Mike Bollinger  

## Overview
The VOYIS First Look Metrics application is designed to calculate summary statistics about a folder of images collected by an AUV. It analyzes processed stills and raw images, providing comprehensive metrics about image coverage, overlap, and quality. The application features a user-friendly graphical interface for selecting input and output folders and displays real-time processing status.

## Features
- **Image Counting and Classification**
  - Count and categorize processed stills and raw images
  - Calculate total storage size by image type
  
- **Geographic Analysis**
  - Extract and visualize GPS data from image EXIF
  - Generate image footprint maps showing coverage area
  - Calculate altitude statistics with configurable threshold warnings
  
- **Overlap Analysis**
  - Vertical overlap between sequential images in a survey line
  - Horizontal overlap between adjacent survey lines
  - Overall overlap count showing total overlapping images for each footprint
  - Color-coded visualization maps for all overlap types (vertical, horizontal, overall)
  
- **Visibility Analysis**
  - Machine learning-based classification of underwater image visibility
  - Categorization into zero, low, good, and great visibility classes
  - Random sampling of images below specified altitude threshold
  - CSV output with visibility metrics and geolocation data
  - Shapefile export with visibility information for GIS integration
  - Visualization of randomly selected sample locations
  
- **Highlight Selection**
  - Automatic selection of highlight images based on quality metrics 
  - Identification of representative imagery across survey area
  - HTML gallery of selected highlights with metadata
  
- **Highlight Panel Creator**
  - Creation of multi-image highlight panels for reports
  - Support for 2, 4, or 6 images in a publication-ready format
  - Automatic extraction of GPS metadata from images
  - Configurable 2-column layout optimized for portrait style reports
  - Drag-and-drop interface for easy image management
  
- **Export Capabilities**
  - Comprehensive summary report in text format
  - CSV export of image metadata with overlap statistics
  - Shapefile export for GIS integration including overlap metrics
  - Visualization plots showing coverage and overlap
  - High resolution image panels for inclusion in reports

- **User Interface**
  - Intuitive GUI for input/output folder selection
  - Real-time processing status updates
  - Error handling with detailed diagnostics
  - Configurable altitude threshold for filtering images

## Project Structure
```
VOYIS_FirstLookMetrics/ 
├── src/ 
│ ├── main.py # Entry point of the application 
│ ├── gui/ 
│ │ ├── init.py 
│ │ └── app_window.py # Defines the GUI window 
│ ├── models/ 
│ │ ├── init.py 
│ │ ├── metrics.py # Summary statistics for processed images 
│ │ ├── altitude_map.py # Creates location maps and altitude visualizations 
│ │ ├── footprint_map.py # Calculates and visualizes image footprints and overlap 
│ │ ├── visibility_analyzer.py # ML-based visibility classification 
│ │ └── highlight_selector.py # Analysis for selecting highlight images 
│ └── utils/ 
│ ├── init.py 
│ ├── path_utils.py # Utilities for path and file management 
│ └── highlight_panel_creator.py # Creates highlight image panels 
├── requirements.txt # Project dependencies 
└── README.md # Project documentation
```

## System Requirements
- Python 3.8 or higher
- Required libraries: tkinter, PIL/Pillow, numpy, matplotlib, geopandas
- Optional: 
  - rtree for improved spatial query performance
  - TensorFlow for visibility analysis
  - shapely for advanced spatial operations
  - tkinterdnd2 for drag-and-drop functionality in highlight panel creator

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd VOYIS_FirstLookMetrics
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Optional: Install TensorFlow for visibility analysis:
   ```
   pip install tensorflow
   ```
5. Optional: Install tkinterdnd2 for drag-and-drop support:
   ```
   pip install tkinterdnd2
   ```

## Usage
1. Run the application:
   ```
   python src/main.py
   ```
2. Use the GUI to select:
   - Input folder containing VOYIS images
   - Output folder for the results
   - Configure analysis options and select processing steps
   - Set altitude threshold for filtering (default: 8.0m)
   - Select navigation file for improved footprint orientation (optional)
   - Choose pre-trained visibility model or training data directory
3. Click "Process Images" and monitor progress in the status area
4. Examine the generated reports and visualizations in the output folder
5. Use the highlight panel creator for creating report-ready image panels:
   - Launch from the Tools menu
   - Select 2, 4, or 6 images layout
   - Add images via the file dialog or drag-and-drop
   - Save the panel as PNG, JPEG, or PDF

## Output Files
The application generates the following outputs:
- `image_metrics.txt`: Text file with comprehensive statistics
- `image_locations.csv`: CSV file with image metadata and GPS coordinates
- `image_locations_map.png`: Map showing image locations colored by altitude
- `altitude_histogram.png`: Histogram showing the distribution of image altitudes
- `image_footprints.csv`: CSV file with detailed image metadata and overlap statistics
- `image_footprints.shp`: Shapefile for GIS applications with overlap metrics as attributes
- `image_footprints_map.png`: Map showing image coverage with footprints
- `vertical_overlap_map.png`: Map showing overlap between sequential images
- `horizontal_overlap_map.png`: Map showing overlap between adjacent survey lines
- `overall_overlap_map.png`: Map showing total number of overlapping images per footprint
- `visibility_results.csv`: CSV file with visibility classifications and confidence scores
- `selected_images_map.png`: Map showing selected vs. non-selected images for visibility analysis
- `shapefiles/selected_images.shp`: Shapefile of randomly selected images with visibility data
- `highlight_images/`: Directory containing selected highlight images
- `highlight_images/highlights.html`: Interactive HTML gallery of highlight images
- `highlight_images/top_highlights_panel.png`: Auto-generated panel with top highlight images

## Version History
- **1.4.0** - Added highlight panel creator, and highlight image selection tools
- **1.3.0** - Added ML-based visibility analysis, improved workflow organization, and added shapefile exports
- **1.2.0** - Added overall overlap analysis and optimized spatial processing
- **1.1.0** - Added horizontal overlap analysis and GIS exports
- **1.0.0** - Initial release with basic image metrics and vertical overlap

## Known Issues
- Processing large datasets (>10,000 images) may be slow, especially for overlap calculations
- Requires geopandas for advanced mapping features
- TensorFlow needed for visibility analysis
- The visibility analysis may need additional training data for optimal performance in varied environments
- Highlight panel creator requires tkinterdnd2 for drag-and-drop functionality

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.