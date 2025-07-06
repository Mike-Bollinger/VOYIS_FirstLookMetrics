# Batch Processing CSV Format

This document describes the standardized CSV format for batch processing in the VOYIS First Look Metrics application.

## Standardized Column Names

The batch processing CSV file must use the following standardized column names in the specified order:

### Column Order and Requirements

| Column Name | Description | Module | Required for Module |
|-------------|-------------|---------|-------------------|
| `NAV_STATE_file` | Path to the navigation state/plot file | Navigation | ✓ (with PHINS_INS_file) |
| `PHINS_INS_file` | Path to additional PHINS INS file | Navigation | ✓ (with NAV_STATE_file) |
| `LLS_Input` | Path to the LLS (laser) data folder | LLS Analysis | ✓ (with PhinsData_Bin_file) |
| `PhinsData_Bin_file` | Path to the PHINS navigation binary file | LLS Analysis | ✓ (with LLS_Input) |
| `Image_Input` | Path to the input folder containing imagery | Image Analysis | ✓ |
| `Dive_Nav_file` | Path to the vehicle navigation file | Image Analysis | Optional |
| `Output_folder` | Path to the output folder for processed results | All | ✓ (Always Required) |

## Processing Modules

The application supports three independent processing modules:

### 1. Navigation Module
- **Required**: `NAV_STATE_file` and/or `PHINS_INS_file`
- **Purpose**: Process navigation data and create navigation plots

### 2. Image Analysis Module  
- **Required**: `Image_Input`
- **Optional**: `Dive_Nav_file` (enhances analysis with navigation data)
- **Purpose**: Analyze imagery, extract metrics, create location maps, footprint maps, visibility analysis

### 3. LLS Analysis Module
- **Required**: `LLS_Input` AND `PhinsData_Bin_file` (both required together)
- **Purpose**: Process laser scanner data and create LLS-based outputs

### Universal Requirement
- **Required**: `Output_folder` (always required for all modules)

## Example CSV Content

```csv
NAV_STATE_file,PHINS_INS_file,LLS_Input,PhinsData_Bin_file,Image_Input,Dive_Nav_file,Output_folder
C:/path/to/nav_plot/file1.txt,,C:/path/to/lls/folder1,C:/path/to/phins/nav1.bin,C:/path/to/imagery/folder1,C:/path/to/nav/file1.txt,C:/path/to/output/folder1
C:/path/to/nav_plot/file2.txt,,C:/path/to/lls/folder2,C:/path/to/phins/nav2.bin,C:/path/to/imagery/folder2,,C:/path/to/output/folder2
,,,,C:/path/to/imagery/folder3,C:/path/to/nav/file3.txt,C:/path/to/output/folder3
```

## Creating Template

You can create a template CSV file using the "Create CSV Template" button in the application's batch processing section. This will generate a properly formatted CSV file with example paths and all standardized column names in the correct order.

## Processing Rules

1. **Output always required**: Every job must specify `Output_folder`
2. **Module independence**: Each processing module can run independently with its required inputs
3. **Module requirements**: 
   - Navigation: `NAV_STATE_file` and/or `PHINS_INS_file`
   - Image Analysis: `Image_Input` (required), `Dive_Nav_file` (optional)
   - LLS Analysis: `LLS_Input` AND `PhinsData_Bin_file` (both required together)
4. **Empty optional fields**: Can be left blank or omitted entirely
5. **Path validation**: The application will validate that specified paths exist before processing
6. **Processing functions**: The enabled processing functions in the GUI apply to all batch jobs

## Notes

- Use forward slashes (/) or double backslashes (\\\\) in Windows paths
- Empty optional fields can be left blank or omitted entirely
- The application will create output directories if they don't exist
- Processing will continue with remaining jobs if individual jobs fail
- Each job must have inputs for at least one processing module to be valid
