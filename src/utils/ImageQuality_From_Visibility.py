"""
Visibility Summary Utility

This script recursively searches for Image_Metrics.csv files in dive folders,
extracts visibility data, and calculates the percentage of acceptable visibility
(Excellent + Good + Fair) for each dive.

Author: VOYIS First Look Metrics Team
Date: February 2026
"""

import os
import re
import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from typing import List, Dict, Tuple


def select_parent_directory() -> str:
    """
    Open a dialog box to select the parent directory containing dive folders.
    
    Returns:
        str: Selected directory path, or empty string if cancelled
    """
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    directory = filedialog.askdirectory(
        title="Select Parent Directory (e.g., D:\\EN2501\\Image_LLS)",
        mustexist=True
    )
    
    root.destroy()
    return directory


def find_image_metrics_files(parent_dir: str) -> List[Path]:
    """
    Recursively search for *_Image_Metrics.csv files in the parent directory.
    
    Args:
        parent_dir: Parent directory path to search
        
    Returns:
        List of Path objects for found CSV files
    """
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {parent_dir}")
    
    # Search recursively for files matching the pattern
    pattern = "*_Image_Metrics.csv"
    csv_files = list(parent_path.rglob(pattern))
    
    return csv_files


def extract_dive_number(file_path: Path) -> str:
    """
    Extract dive number from file path.
    Handles formats: DIVE###, Dive###, dive###, DIVE##, Dive##, dive##
    
    Args:
        file_path: Path object of the CSV file
        
    Returns:
        Formatted dive number (e.g., "DIVE013") or "UNKNOWN" if not found
    """
    # Get the filename
    filename = file_path.name
    
    # Try to match DIVE### or Dive### or dive## patterns
    pattern = r'(DIVE|Dive|dive)(\d{2,3})'
    match = re.search(pattern, filename)
    
    if match:
        # Return in uppercase format with padded zeros
        dive_prefix = "DIVE"
        dive_num = match.group(2).zfill(3)  # Pad to 3 digits
        return f"{dive_prefix}{dive_num}"
    
    # If not in filename, check parent directory names
    for parent in file_path.parents:
        match = re.search(pattern, parent.name)
        if match:
            dive_prefix = "DIVE"
            dive_num = match.group(2).zfill(3)
            return f"{dive_prefix}{dive_num}"
    
    return "UNKNOWN"


def calculate_visibility_percentage(csv_file: Path) -> Tuple[float, int, int, Dict[str, int]]:
    """
    Calculate the percentage of acceptable visibility observations.
    Acceptable = Excellent + Good + Fair
    
    Args:
        csv_file: Path to the Image_Metrics.csv file
        
    Returns:
        Tuple of (percentage, acceptable_count, total_count, category_counts)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if visibility column exists
        if 'visibility' not in df.columns:
            raise ValueError(f"'visibility' column not found in {csv_file}")
        
        # Get total observations (excluding NaN)
        total_count = df['visibility'].notna().sum()
        
        if total_count == 0:
            return 0.0, 0, 0, {}
        
        # Convert visibility values to lowercase for case-insensitive comparison
        visibility_lower = df['visibility'].str.lower()
        
        # Count observations by category
        excellent_count = (visibility_lower == 'excellent').sum()
        good_count = (visibility_lower == 'good').sum()
        fair_count = (visibility_lower == 'fair').sum()
        poor_count = (visibility_lower == 'poor').sum()
        zero_count = (visibility_lower == 'zero').sum()
        
        # Calculate acceptable count (Excellent + Good + Fair)
        acceptable_count = excellent_count + good_count + fair_count
        
        # Calculate percentage
        percentage = (acceptable_count / total_count) * 100
        
        # Category breakdown
        category_counts = {
            'Excellent': excellent_count,
            'Good': good_count,
            'Fair': fair_count,
            'Poor': poor_count,
            'Zero': zero_count
        }
        
        return percentage, acceptable_count, total_count, category_counts
        
    except Exception as e:
        raise Exception(f"Error processing {csv_file}: {str(e)}")


def generate_summary_report(results: List[Dict]) -> str:
    """
    Generate a formatted summary report from the results.
    
    Args:
        results: List of dictionaries containing dive analysis results
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VISIBILITY SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if not results:
        report_lines.append("No Image_Metrics.csv files found.")
        return "\n".join(report_lines)
    
    # Sort results by dive number
    results_sorted = sorted(results, key=lambda x: x['dive_number'])
    
    # Summary table
    report_lines.append(f"{'Dive':<12} {'Acceptable %':<15} {'Acceptable/Total':<20} {'File Path':<50}")
    report_lines.append("-" * 80)
    
    for result in results_sorted:
        dive = result['dive_number']
        percentage = result['percentage']
        acceptable = result['acceptable_count']
        total = result['total_count']
        file_path = result['file_path']
        
        report_lines.append(
            f"{dive:<12} {percentage:>6.2f}%{'':<8} {acceptable:>4}/{total:<4}{'':<10} {file_path}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DETAILED BREAKDOWN BY DIVE")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for result in results_sorted:
        report_lines.append(f"Dive: {result['dive_number']}")
        report_lines.append(f"  File: {result['file_path']}")
        report_lines.append(f"  Total Observations: {result['total_count']}")
        report_lines.append(f"  Acceptable (Excellent+Good+Fair): {result['acceptable_count']} ({result['percentage']:.2f}%)")
        report_lines.append(f"  Category Breakdown:")
        
        for category, count in result['category_counts'].items():
            pct = (count / result['total_count'] * 100) if result['total_count'] > 0 else 0
            report_lines.append(f"    - {category}: {count} ({pct:.2f}%)")
        
        report_lines.append("")
    
    # Overall statistics
    if len(results_sorted) > 1:
        report_lines.append("=" * 80)
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("=" * 80)
        
        total_obs = sum(r['total_count'] for r in results_sorted)
        total_acceptable = sum(r['acceptable_count'] for r in results_sorted)
        avg_percentage = (total_acceptable / total_obs * 100) if total_obs > 0 else 0
        
        report_lines.append(f"Total Dives Analyzed: {len(results_sorted)}")
        report_lines.append(f"Total Observations: {total_obs}")
        report_lines.append(f"Total Acceptable: {total_acceptable}")
        report_lines.append(f"Overall Acceptable Percentage: {avg_percentage:.2f}%")
        report_lines.append("")
    
    return "\n".join(report_lines)


def save_report(report: str, parent_dir: str):
    """
    Save the report to a text file in the parent directory.
    
    Args:
        report: Report content string
        parent_dir: Parent directory path
    """
    output_file = Path(parent_dir) / "Visibility_Summary_Report.txt"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")


def save_csv_summary(results: List[Dict], parent_dir: str):
    """
    Save the summary data to a CSV file.
    
    Args:
        results: List of dictionaries containing dive analysis results
        parent_dir: Parent directory path
    """
    output_file = Path(parent_dir) / "Visibility_Summary.csv"
    
    # Sort results by dive number
    results_sorted = sorted(results, key=lambda x: x['dive_number'])
    
    # Prepare data for CSV
    csv_data = []
    for result in results_sorted:
        csv_data.append({
            'Dive': result['dive_number'],
            'Acceptable_Percent': result['percentage'],
            'Acceptable_Count': result['acceptable_count'],
            'Total_Count': result['total_count'],
            'File_Path': result['file_path']
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    print(f"CSV summary saved to: {output_file}")


def main():
    """
    Main execution function.
    """
    print("VOYIS First Look Metrics - Visibility Summary Utility")
    print("=" * 80)
    print("")
    
    # Select parent directory
    parent_dir = select_parent_directory()
    
    if not parent_dir:
        print("No directory selected. Exiting.")
        return
    
    print(f"Selected directory: {parent_dir}")
    print("")
    
    try:
        # Find all Image_Metrics.csv files
        print("Searching for Image_Metrics.csv files...")
        csv_files = find_image_metrics_files(parent_dir)
        
        if not csv_files:
            print("No Image_Metrics.csv files found.")
            messagebox.showwarning(
                "No Files Found",
                f"No *_Image_Metrics.csv files found in:\n{parent_dir}"
            )
            return
        
        print(f"Found {len(csv_files)} file(s)")
        print("")
        
        # Process each file
        results = []
        
        for csv_file in csv_files:
            print(f"Processing: {csv_file}")
            
            dive_number = extract_dive_number(csv_file)
            percentage, acceptable, total, categories = calculate_visibility_percentage(csv_file)
            
            results.append({
                'dive_number': dive_number,
                'percentage': percentage,
                'acceptable_count': acceptable,
                'total_count': total,
                'category_counts': categories,
                'file_path': str(csv_file)
            })
            
            print(f"  Dive: {dive_number}")
            print(f"  Acceptable Visibility: {percentage:.2f}% ({acceptable}/{total})")
            print("")
        
        # Generate and display report
        report = generate_summary_report(results)
        print(report)
        
        # Save report and CSV
        save_report(report, parent_dir)
        save_csv_summary(results, parent_dir)
        
        # Show completion message
        messagebox.showinfo(
            "Processing Complete",
            f"Processed {len(results)} dive(s).\n\n"
            f"Text report saved to:\n{parent_dir}\\Visibility_Summary_Report.txt\n\n"
            f"CSV summary saved to:\n{parent_dir}\\Visibility_Summary.csv"
        )
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)


if __name__ == "__main__":
    main()
