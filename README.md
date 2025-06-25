# VOYIS First Look Metrics

<div align="center">
  <img src="src/utils/NOAA_Voyis_Logo.png" alt="NOAA VOYIS Logo" width="200"/>
</div>

**Version:** 1.5.0  
**Last Updated:** June 2025  
**Author:** Mike Bollinger (Imagery), Jeff Coogan (LLS)

## Overview
The VOYIS First Look Metrics application is designed to calculate summary statistics about a folder of images collected by an AUV. It analyzes processed stills and raw images, providing comprehensive metrics about image coverage, overlap, and quality. The application features a user-friendly graphical interface for selecting input and output folders and displays real-time processing status.

## Features

### Core Image Analysis
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

### Laser Line Scan Data Analysis
- **LLS (Laser Line Scan) Data Processing**
  - Integrated processing of laser line scan data with navigation
  - Support for Phins navigation data integration
  - Automated LLS data quality assessment and filtering
  - Batch processing capabilities for multiple datasets
  
- **Visibility Analysis**
  - Machine learning-based classification of underwater image visibility
  - Categorization into zero, low, good, and great visibility classes
  - Random sampling of images below specified altitude threshold
  - CSV output with visibility metrics and geolocation data
  - Shapefile export with visibility information for GIS integration
  - Histogram visualization with example images from each category
  
- **Highlight Selection**
  - Automatic selection of highlight images based on quality metrics
  - Integration with visibility analysis results
  - Identification of representative imagery across survey area
  - HTML gallery of selected highlights with metadata
  
- **Batch Processing**
  - Process multiple dive datasets simultaneously
  - Configurable processing options for each dataset
  - Progress tracking and detailed logging
  - Error handling and recovery for robust processing

### User Interface
- **Intuitive GUI**
  - Easy folder and file selection
  - Real-time progress tracking
  - Detailed processing logs
  - Configurable processing options
  
- **Highlight Panel Creator**
  - Creation of multi-image highlight panels for reports
  - Support for 2, 4, or 6 images in publication-ready format
  - Automatic extraction of GPS metadata from images
  - Configurable 2-column layout optimized for portrait style reports
  - Drag-and-drop interface for easy image management

## Recent Updates (v1.5.0)
- **Enhanced LLS Processing**: Improved integration with navigation data and batch processing
- **Corrected Visibility Categories**: Updated to use proper model categories (zero_visibility, low_visibility, good_visibility, great_visibility)
- **Improved Batch Processing**: Better error handling and progress tracking
- **Enhanced Highlight Selection**: Better image quality metrics and visibility integration
- **UI Improvements**: LLS processing now enabled by default, better progress feedback

## System Requirements
- Python 3.8 or higher
- Required libraries: tkinter, PIL/Pillow, numpy, matplotlib, pandas
- Optional: 
  - geopandas for advanced mapping and GIS features
  - rtree for improved spatial query performance
  - TensorFlow for visibility analysis
  - shapely for advanced spatial operations
  - tkinterdnd2 for drag-and-drop functionality in highlight panel creator
  - scipy for enhanced image metrics

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Mike-Bollinger/VOYIS_FirstLookMetrics.git
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
5. Optional: Install additional packages for enhanced functionality:
   ```
   pip install geopandas rtree shapely tkinterdnd2 scipy
   ```

## Usage
1. Run the application:
   ```
   python src/main.py
   ```
2. Select input folders for imagery and/or LLS data
3. Choose output folder for results
4. Select desired processing options:
   - **LLS Processing**: Process laser line scan data
   - **Basic Metrics**: Calculate image statistics and metadata
   - **Location Maps**: Generate GPS-based visualizations
   - **Altitude Histograms**: Analyze altitude distributions
   - **Footprint Maps**: Calculate image coverage and overlaps
   - **Visibility Analysis**: ML-based visibility classification
   - **Highlight Selection**: Automatic selection of best images
5. Click "Process" to start analysis
6. Monitor progress and review results in the output folder

## Output Files
The application generates comprehensive outputs including:

### Core Outputs
- `image_metrics.txt`: Comprehensive statistics summary
- `image_locations.csv`: Image metadata with GPS coordinates
- `image_locations_map.png`: Location map colored by altitude
- `altitude_histogram.png`: Altitude distribution visualization

### Advanced Outputs
- `image_footprints.csv`: Detailed metadata with overlap statistics
- `image_footprints.shp`: GIS shapefile with overlap metrics
- `image_footprints_map.png`: Coverage map with footprints
- `vertical_overlap_map.png`: Sequential image overlap visualization
- `horizontal_overlap_map.png`: Adjacent survey line overlap
- `overall_overlap_map.png`: Total overlap count per footprint

### Visibility Analysis
- `visibility_results.csv`: Visibility classifications and confidence scores
- `visibility_analysis.png`: Distribution histogram with example images
- `selected_images_map.png`: Visualization of analyzed images

### Highlight Selection
- `highlight_images/`: Directory with selected highlight images
- `highlight_images/highlights.html`: Interactive HTML gallery
- `highlight_images/top_highlights_panel.png`: Publication-ready image panel

### LLS Processing
- `Voyis_QuickLook_Summary.txt`: LLS processing summary and statistics
- Various LLS data products and quality assessment files

## Version History
- **1.5.0** - Enhanced LLS integration, corrected visibility categories, improved batch processing
- **1.4.0** - Added highlight panel creator and highlight image selection tools
- **1.3.0** - Added ML-based visibility analysis, improved workflow organization, and shapefile exports
- **1.2.0** - Added overall overlap analysis and optimized spatial processing
- **1.1.0** - Added horizontal overlap analysis and GIS exports
- **1.0.0** - Initial release with basic image metrics and vertical overlap

## Known Issues
- Processing large datasets (>10,000 images) may be slow, especially for overlap calculations
- Visibility analysis requires pre-trained model (not included in repository due to size)
- Some advanced features require optional dependencies
- Batch processing of very large datasets may require significant memory

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.