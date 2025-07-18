# VOYIS First Look Metrics

<div align="center">
  <img src="src/utils/NOAA_Voyis_Logo.png" alt="NOAA VOYIS Logo" width="200"/>
</div>

**Version:** 1.6.0  
**Last Updated:** July 2025  
**Author:** Mike Bollinger (Imagery, GUI, Navigation), Jeff Coogan (LLS), Noah Hunt (Original Nav R Scripts)

## Overview
The VOYIS First Look Metrics application is a comprehensive post-mission analysis tool designed to process and analyze data from the VOYIS imaging and laser line scanning payload on an autonomous underwater vehicle (AUV). The application integrates three distinct processing modules: **Navigation Analysis**, **Laser Line Scanning (LLS)**, and **Imagery Analysis**. It provides detailed quality metrics, comprehensive visualizations, and statistical analysis of AUV navigation performance, imaging coverage, and LLS data quality. The application features a user-friendly graphical interface supporting both single-dive and batch processing modes.

## Features

### Navigation Analysis
- **Comprehensive Navigation Plotting**
  - Post-mission analysis of AUV navigation data from NAV_STATE and PHINS INS files
  - Support for both text-based navigation files and binary PHINS data formats
  - Automatic parsing and synchronization of navigation timestamps
  
- **Motion Analysis**
  - Heave, pitch, and roll motion visualization and statistics
  - Time series plots with depth-colored overlays for enhanced analysis
  - Motion distribution histograms with statistical summaries
  - Position-based motion maps showing spatial patterns
  
- **Crab Index Analysis**
  - Automatic calculation of vehicle crab index (heading vs. course over ground)
  - Crab index severity classification (good ≤5°, moderate 5-10°, severe >10°)
  - Spatial and temporal visualization of crabbing behavior
  - Statistical analysis of navigation quality and vehicle performance
  
- **Bathymetry and Depth Analysis**
  - Enhanced depth profile visualization with exponential scaling
  - Bathymetry maps with depth-colored spatial visualization
  - Depth statistics and quality metrics integration
  
- **Navigation Quality Metrics**
  - Position accuracy assessment and uncertainty analysis
  - Motion statistics and extrema identification
  - Comprehensive data quality reporting with automated threshold analysis

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
- **Integrated LLS Processing**
  - Enhanced integration with navigation data for improved spatial accuracy
  - Support for both PHINS INS and NAV_STATE navigation file formats
  - Automated temporal synchronization between LLS and navigation data
  - Robust error handling and quality assessment workflows
  
- **Navigation-LLS Synchronization**
  - Automatic detection and handling of different navigation file formats
  - Flexible path configuration for LLS data and navigation files
  - Improved batch processing with consistent file handling
  - Enhanced temp directory management for processing stability
  
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

### User Interface
- **Intuitive GUI with Navigation Integration**
  - Unified interface for navigation, LLS, and imagery processing
  - Separate file selection for different navigation data types
  - Clear distinction between NAV_STATE files and PHINS INS files
  - Real-time progress tracking with module-specific feedback
  - Enhanced processing logs with detailed status reporting
  
- **Flexible Processing Configuration**
  - Independent enable/disable controls for each processing module
  - Support for partial processing workflows (e.g., navigation-only analysis)
  - Configurable processing options with intelligent defaults
  - Improved batch processing interface with save/restore capabilities

- **Batch Processing**
  - Process multiple dive datasets simultaneously with improved reliability
  - Unified navigation file path handling for both LLS and navigation modules
  - Enhanced save/restore functionality for batch processing configurations
  - Improved error handling and recovery for robust large-scale processing
  - Consistent processing workflows across single and batch modes  

- **Highlight Panel Creator**
  - Creation of multi-image highlight panels for reports
  - Support for 2, 4, or 6 images in publication-ready format
  - Automatic extraction of GPS metadata from images
  - Configurable 2-column layout optimized for portrait style reports
  - Drag-and-drop interface for easy image management

## Recent Updates (v1.6.0)
- **Full Navigation Processing Integration**: Added comprehensive navigation analysis module with support for NAV_STATE and PHINS INS files
- **Crab Index Analysis**: Automatic calculation and visualization of vehicle crab index with severity classification
- **Enhanced Motion Analysis**: Detailed heave, pitch, and roll analysis with spatial and temporal visualization
- **Improved LLS-Navigation Integration**: Better synchronization and file handling between LLS processing and navigation data
- **Batch Processing Enhancements**: Unified navigation file path handling and improved save/restore functionality
- **Enhanced UI**: Separate navigation file selection controls and improved processing workflow organization
- **Robust Error Handling**: Improved error recovery and logging throughout all processing modules
- **Comprehensive Visualization**: New navigation plots including motion maps, crab index analysis, and bathymetry visualization

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
2. Select input folders and files:
   - **Image Input Folder**: Folder containing processed/raw images (for imagery analysis)
   - **LLS Input Folder**: Folder containing .xyz laser line scan files (for LLS processing)
   - **Navigation Files**: 
     - NAV_STATE file (for navigation plotting and analysis)
     - PHINS INS file (for navigation plotting with enhanced motion data)
     - PHINS Navigation file (for LLS processing integration)
     - Dive Navigation file (from VIP Common Data Export) (for image processing)
   - **Output Folder**: Directory for all processing results
3. Choose output folder for results
4. Select desired processing options:
   - **Navigation Plotting**: Comprehensive AUV navigation analysis with motion and crab index metrics
   - **LLS Processing**: Laser line scan data processing with navigation integration
   - **Basic Metrics**: Image statistics and metadata analysis
   - **Location Maps**: GPS-based imagery visualizations
   - **Altitude Histograms**: Altitude distribution analysis
   - **Footprint Maps**: Image coverage and overlap analysis
   - **Visibility Analysis**: ML-based underwater visibility classification
   - **Highlight Selection**: Automatic selection of best quality images
5. Click "Process" to start analysis
6. Monitor progress and review results in the output folder

### Processing Modes
- **Single Mode**: Process individual dive datasets with full manual control over all parameters
- **Batch Mode**: Process multiple dive datasets using CSV configuration files for automated workflows
  - Supports mixed processing configurations (e.g., some dives with navigation, others with LLS only)
  - Save and restore batch configurations for repeated processing workflows  
  - Unified navigation file path handling across all modules
  - Enhanced error recovery and detailed logging for each dive dataset
  - See `BATCH_CSV_FORMAT.md` for detailed CSV format specifications

## Integrated Workflow
The three processing modules work together to provide comprehensive dive analysis:

1. **Navigation Module** processes NAV_STATE or PHINS INS files to analyze vehicle motion, calculate crab index, and generate navigation quality metrics
2. **LLS Module** processes laser line scan data with navigation integration for spatial accuracy and temporal synchronization
3. **Imagery Module** analyzes image coverage, overlap, visibility, and quality metrics using GPS data from images or navigation files

All modules contribute to the master `Image_Metrics.csv` file, which serves as the central repository for all dive metrics and enables cross-module analysis and reporting.

## Output Files
The application generates comprehensive outputs organized by processing module:

### Navigation Analysis Module Outputs (New in v1.6.0)
- `Nav_Motion_Analysis.png`: Comprehensive 5x6 grid plot with motion analysis, crab index, and bathymetry
- `Nav_Depth_Profile.png`: Depth profile time series with inverted y-axis
- `Nav_Bathymetry_Map.png`: Spatial depth visualization with exponential scaling
- `Nav_Crab_Index_Map.png`: Spatial crab index distribution (red=starboard, blue=port)
- `Nav_Crab_Index_Timeseries.png`: Crab index time series with threshold lines
- `Nav_Crab_Index_Histogram.png`: Crab index distribution with severity classification
- `Nav_Heading_vs_COG.png`: Heading vs course over ground comparison
- `Nav_Heave_Map.png`: Spatial heave motion visualization
- `Nav_Heave_Timeseries.png`: Heave motion time series with statistics
- `Nav_Pitch_Map.png`: Spatial pitch attitude visualization
- `Nav_Pitch_Timeseries.png`: Pitch attitude time series with statistics
- `Nav_Roll_Map.png`: Spatial roll attitude visualization
- `Nav_Roll_Timeseries.png`: Roll attitude time series with statistics

### LLS Processing Module Outputs
- `LLS_Voyis_QuickLook_Summary.txt`: LLS processing summary and statistics
- `LLS_Output/`: Directory containing processed LLS data products
- `Vehicle_Output/`: Directory containing navigation-synchronized LLS results
- Various quality assessment files and filtered LLS datasets

### Imagery Analysis Module Outputs
- `Image_Metrics.csv`: **Master CSV file** containing integrated metrics from all of the Imaging Modules
- `Image_Locations_Map.png`: Location map colored by altitude
- `Image_Altitude_Histogram.png`: Altitude distribution visualization
- `Image_Footprints_Map.png`: Coverage map with image footprints
- `Image_Vertical_Overlap_Map.png`: Sequential image overlap visualization
- `Image_Horizontal_Overlap_Map.png`: Adjacent survey line overlap visualization
- `Image_Overall_Overlap_Map.png`: Total overlap count per footprint
- `Image_Footprints.csv`: Detailed metadata with overlap statistics
- `Image_Footprints.shp`: GIS shapefile with overlap metrics
- `Image_Locations.shp`: GIS shapefile with GPS coordinates

### Visibility Analysis Outputs
- `Image_Visibility_Results.csv`: Visibility classifications and confidence scores
- `Image_Visibility_Analysis.png`: Distribution histogram with example images
- `selected_images_map.png`: Visualization of analyzed images

### Highlight Selection Outputs
- `highlight_images/`: Directory with selected highlight images
- `highlight_images/highlights.html`: Interactive HTML gallery
- `highlight_images/top_highlights_panel.png`: Publication-ready image panel

## Version History
- **1.6.0** - Full navigation processing integration with comprehensive motion and crab index analysis
- **1.5.0** - Enhanced LLS integration, corrected visibility categories, improved batch processing
- **1.4.0** - Added highlight panel creator and highlight image selection tools
- **1.3.0** - Added ML-based visibility analysis, improved workflow organization, and shapefile exports
- **1.2.0** - Added overall overlap analysis and optimized spatial processing
- **1.1.0** - Added horizontal overlap analysis and GIS exports
- **1.0.0** - Initial release with basic image metrics and vertical overlap

## Known Issues
- **Navigation Processing**: Large navigation datasets may require significant processing time for crab index calculations
- **Memory Requirements**: Processing large datasets (>10,000 images) may be slow, especially for overlap calculations
- **Model Dependencies**: Visibility analysis requires pre-trained model (not included in repository due to size)
- **Optional Dependencies**: Some advanced features require optional dependencies (geopandas, tensorflow, etc.)
- **Batch Processing**: Very large batch processing workflows may require significant memory and disk space
- **Navigation File Paths**: Ensure consistent navigation file paths when switching between LLS and navigation processing modes

## Troubleshooting
- **Navigation plots appear empty**: Check that NAV_STATE or PHINS INS files contain the expected column names (latitude, longitude, heading, depth, etc.)
- **Crab index not calculated**: Ensure navigation file contains latitude, longitude, and heading columns with sufficient data points
- **LLS processing fails**: Verify that PHINSdata navigation file path is correctly set and file format is supported
- **Memory errors during batch processing**: Process datasets in smaller batches or increase system memory allocation
- **Missing output files**: Check log messages for processing errors and ensure all required input files are present

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