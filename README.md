# VOYIS First Look Metrics

<div align="center">
  <img src="src/utils/NOAA_Voyis_Logo.png" alt="NOAA VOYIS Logo" width="200"/>
</div>

**Version:** 1.6.0  
**Last Updated:** July 2025  
**Author:** Mike Bollinger (Imagery, GUI, Navigation), Jeff Coogan (LLS), Noah Hunt (Navigation-R Version)

## Overview
The VOYIS First Look Metrics application is designed to calculate summary statistics about data collected by an AUV, including imagery analysis and navigation data processing. It analyzes processed stills and raw images, processes navigation data with advanced plotting capabilities, and provides comprehensive metrics about data coverage, overlap, and quality. The application features a user-friendly graphical interface for selecting input and output folders and displays real-time processing status.

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

### Laser Line Scan Data Analysis
- **LLS (Laser Line Scan) Data Processing**
  - Integrated processing of laser line scan data with navigation
  - Support for Phins navigation data integration
  - Automated LLS data quality assessment and filtering
  - Batch processing capabilities for multiple datasets

### Navigation Data Analysis
- **Navigation Data Processing**
  - Support for NAV_STATE.txt and PHINS INS.txt files
  - Comprehensive navigation plotting and quality assessment
  - Real-time heave data merging and time synchronization
  - Advanced crab index analysis for vehicle stability assessment
  
- **Navigation Visualizations**
  - Multi-panel navigation overview plots with mission timeline
  - Bathymetry maps with enhanced depth visualization using exponential scaling
  - Individual motion analysis (roll, pitch, heave) with time series plots
  - Crab index mapping showing starboard/port drift patterns
  - GPS track visualization with depth-colored trajectories
  - Motion statistics and quality metrics analysis
  
- **Advanced Navigation Features**
  - Automatic column mapping for flexible navigation file formats
  - Mission time synchronization between navigation and PHINS data
  - Statistical heave analysis when direct time merging fails
  - Professional coordinate axis formatting (no scientific notation)
  - High-resolution plot generation with customizable color schemes
  
### User Interface
- **Intuitive GUI**
  - Easy folder and file selection
  - Real-time progress tracking
  - Detailed processing logs
  - Configurable processing options

- **Batch Processing**
  - Process multiple dive datasets simultaneously
  - Configurable processing options for each dataset
  - Progress tracking and detailed logging
  - Error handling and recovery for robust processing  

- **Highlight Panel Creator**
  - Creation of multi-image highlight panels for reports
  - Support for 2, 4, or 6 images in publication-ready format
  - Automatic extraction of GPS metadata from images
  - Configurable 2-column layout optimized for portrait style reports
  - Drag-and-drop interface for easy image management

## Recent Updates (v1.6.0)
- **Comprehensive Navigation Processing**: Complete integration of advanced navigation data analysis with support for NAV_STATE.txt and PHINS INS.txt files
- **Enhanced Navigation Plotting**: Multi-panel overview plots, bathymetry maps with exponential depth scaling, and individual motion analysis
- **Crab Index Analysis**: Advanced vehicle stability assessment with starboard/port drift visualization
- **Mission Time Synchronization**: Robust time-based merging of navigation and heave data with fallback strategies
- **Professional Visualization**: High-quality plots with proper coordinate formatting, viridis colormap, and customizable point sizes
- **Improved Column Mapping**: Flexible parsing of navigation files with automatic column detection and robust error handling
- **Quality Assessment Integration**: Navigation quality metrics and statistics reporting for mission assessment

## System Requirements
- Python 3.8 or higher
- Required libraries: tkinter, PIL/Pillow, numpy, matplotlib, pandas, seaborn
- Optional: 
  - geopandas for advanced mapping and GIS features
  - rtree for improved spatial query performance
  - TensorFlow for visibility analysis
  - shapely for advanced spatial operations
  - tkinterdnd2 for drag-and-drop functionality in highlight panel creator
  - scipy for enhanced image metrics and statistical analysis

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
3. Choose navigation files (NAV_STATE.txt and optional PHINS INS.txt)
4. Choose output folder for results
5. Select desired processing options:
   - **Navigation Processing**: Analyze navigation data with comprehensive plotting
   - **LLS Processing**: Process laser line scan data
   - **Basic Metrics**: Calculate image statistics and metadata
   - **Location Maps**: Generate GPS-based visualizations
   - **Altitude Histograms**: Analyze altitude distributions
   - **Footprint Maps**: Calculate image coverage and overlaps
   - **Visibility Analysis**: ML-based visibility classification
   - **Highlight Selection**: Automatic selection of best images
6. Click "Process" to start analysis
7. Monitor progress and review results in the output folder

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

### Navigation Analysis
- `Nav_Multi_Panel_Overview.png`: Comprehensive navigation overview with 6-panel layout
- `Nav_Bathymetry_Map.png`: Depth-colored bathymetry map with exponential scaling
- `Nav_Roll_Map.png`, `Nav_Pitch_Map.png`, `Nav_Heave_Map.png`: Individual motion maps
- `Nav_Roll_Timeseries.png`, `Nav_Pitch_Timeseries.png`, `Nav_Heave_Timeseries.png`: Motion time series
- `Nav_Crab_Index_Map.png`: Crab index visualization showing vehicle drift patterns
- `Nav_Crab_Index_Timeseries.png`: Crab index time series analysis
- `Nav_Crab_Index_Histogram.png`: Crab index distribution histogram
- `Nav_Crab_Index_Analysis.png`: Enhanced crab index analysis with statistics
- `Nav_Depth_Profile.png`: Depth profile time series with mission timeline
- Navigation quality metrics and statistics (logged to console and processing log)

## Version History
- **1.6.0** - Comprehensive navigation data processing and advanced plotting capabilities, crab index analysis, mission time synchronization
- **1.5.0** - Enhanced LLS integration, corrected visibility categories, improved batch processing
- **1.4.0** - Added highlight panel creator and highlight image selection tools
- **1.3.0** - Added ML-based visibility analysis, improved workflow organization, and shapefile exports
- **1.2.0** - Added overall overlap analysis and optimized spatial processing
- **1.1.0** - Added horizontal overlap analysis and GIS exports
- **1.0.0** - Initial release with basic image metrics and vertical overlap

## Known Issues
- Processing large datasets (>10,000 images) may be slow, especially for overlap calculations
- Navigation processing requires specific file formats (NAV_STATE.txt, PHINS INS.txt)
- Visibility analysis requires pre-trained model (not included in repository due to size)
- Some advanced features require optional dependencies
- Batch processing of very large datasets may require significant memory
- PHINS heave data merging relies on mission time synchronization and may fall back to statistical methods if timestamps don't align

## Disclaimer

This repository is a scientific product and is not official communication of the National
Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA
GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its
use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from
the use of this GitHub project will be governed by all applicable Federal law. Any reference to
specific commercial products, processes, or services by service mark, trademark, manufacturer, or
otherwise, does not constitute or imply their endorsement, recommendation or favoring by the
Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC
bureau, shall not be used in any manner to imply endorsement of any commercial product or activity
by DOC or the United States Government.

## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. §105). The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.