"""
Merge Turbidity Data with Visibility Results

Merges turbidity measurements with visibility categories from images.
Uses visibility/images as the base dataset and adds matching turbidity data.
Only keeps records where there is a photo/visibility measurement.

Author: VOYIS First Look Metrics
Created: 2026-06-12
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots


def extract_timestamp_from_filename(filename: str) -> pd.Timestamp:
    """
    Extract timestamp from image filename.
    
    Expected format: ESC_stills_processed_PPS_2026-06-11T021831.017500_1.jpg
    Timestamp format: YYYY-MM-DDTHH:MM:SS.ffffff
    
    Args:
        filename: Image filename containing timestamp
        
    Returns:
        pd.Timestamp object in UTC timezone
    """
    # Pattern to match ISO timestamp in filename
    pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}\d{2}\d{2}\.\d+)'
    
    match = re.search(pattern, filename)
    if not match:
        return None
    
    timestamp_str = match.group(1)
    
    # Insert colons in time portion: HHMMSS -> HH:MM:SS
    # Format: 2026-06-11T021831.017500 -> 2026-06-11T02:18:31.017500
    parts = timestamp_str.split('T')
    if len(parts) != 2:
        return None
    
    date_part = parts[0]
    time_part = parts[1]
    
    # Extract HHMMSS and fractional seconds
    time_match = re.match(r'(\d{2})(\d{2})(\d{2})\.(\d+)', time_part)
    if not time_match:
        return None
    
    hh, mm, ss, fraction = time_match.groups()
    
    # Reconstruct ISO format timestamp
    iso_timestamp = f"{date_part}T{hh}:{mm}:{ss}.{fraction}"
    
    try:
        # Parse to timestamp with UTC timezone
        ts = pd.Timestamp(iso_timestamp, tz='UTC')
        return ts
    except Exception as e:
        print(f"Error parsing timestamp from {filename}: {e}")
        return None


def load_visibility_results(vis_csv_path: str) -> pd.DataFrame:
    """
    Load visibility results and extract timestamps from filenames.
    
    Args:
        vis_csv_path: Path to Image_Visibility_Results.csv
        
    Returns:
        DataFrame with columns: image, image_path, timestamp_ns, visibility, visibility_confidence
    """
    print(f"Loading visibility results from: {vis_csv_path}")
    
    vis_df = pd.read_csv(vis_csv_path)
    print(f"  Loaded {len(vis_df):,} visibility records")
    
    # Extract timestamps from image filenames
    print("  Extracting timestamps from image filenames...")
    vis_df['timestamp'] = vis_df['image'].apply(extract_timestamp_from_filename)
    
    # Remove rows where timestamp extraction failed
    before_count = len(vis_df)
    vis_df = vis_df.dropna(subset=['timestamp'])
    after_count = len(vis_df)
    
    if before_count != after_count:
        print(f"  Warning: Could not extract timestamps from {before_count - after_count} filenames")
    
    # Convert timestamps to nanoseconds (matching turbidity format)
    vis_df['timestamp_ns'] = vis_df['timestamp'].apply(lambda x: int(x.value))
    
    # Keep image info, timestamp, visibility, and confidence
    vis_df = vis_df[['image', 'image_path', 'timestamp_ns', 'visibility', 'confidence']].copy()
    vis_df = vis_df.rename(columns={'confidence': 'visibility_confidence'})
    
    # Sort by timestamp
    vis_df = vis_df.sort_values('timestamp_ns').reset_index(drop=True)
    
    print(f"  ✓ Extracted {len(vis_df):,} valid timestamps")
    
    # Print timestamp range for debugging
    if len(vis_df) > 0:
        min_ts = pd.to_datetime(vis_df['timestamp_ns'].min(), unit='ns')
        max_ts = pd.to_datetime(vis_df['timestamp_ns'].max(), unit='ns')
        print(f"  Timestamp range: {min_ts} to {max_ts}")
        print(f"  First image: {vis_df.iloc[0]['image']}")
        print(f"  Last image: {vis_df.iloc[-1]['image']}")
    
    return vis_df


def load_turbidity_compiled(turb_csv_path: str, dive_date: str = "2026-06-11") -> pd.DataFrame:
    """
    Load turbidity compiled CSV and fix datetime_utc timestamps.
    
    The turbidity datetime_utc column has time-of-day but wrong date (1970-01-01).
    We extract the time and combine with the correct dive date.
    
    Args:
        turb_csv_path: Path to turbidity_compiled.csv
        dive_date: Correct date for the dive (YYYY-MM-DD format)
        
    Returns:
        DataFrame with turbidity data and corrected timestamp_ns
    """
    print(f"\nLoading turbidity data from: {turb_csv_path}")
    
    turb_df = pd.read_csv(turb_csv_path)
    print(f"  Loaded {len(turb_df):,} turbidity records")
    
    # Show columns
    print(f"  Columns: {', '.join(turb_df.columns.tolist())}")
    
    # Show sample of datetime_utc values for debugging
    if 'datetime_utc' in turb_df.columns:
        print(f"\n  Sample datetime_utc values (before fix):")
        for i in [0, 1, 2]:
            if i < len(turb_df):
                print(f"    [{i}]: {turb_df.iloc[i]['datetime_utc']}")
    
    # Fix datetime_utc timestamps
    if 'datetime_utc' in turb_df.columns:
        print(f"\n  Fixing datetime_utc timestamps (wrong date, correct time)...")
        print(f"  Using dive date: {dive_date}")
        
        # Parse the datetime_utc column (which has time but wrong date)
        turb_df['datetime_parsed'] = pd.to_datetime(turb_df['datetime_utc'], utc=True)
        
        # Extract time components
        turb_df['time_only'] = turb_df['datetime_parsed'].dt.time
        
        # Combine correct date with extracted time
        turb_df['datetime_corrected'] = turb_df['time_only'].apply(
            lambda t: pd.Timestamp.combine(pd.Timestamp(dive_date).date(), t).tz_localize('UTC')
        )
        
        # Show corrected values
        print(f"  Sample corrected datetimes:")
        for i in [0, 1, 2]:
            if i < len(turb_df):
                print(f"    [{i}]: {turb_df.iloc[i]['datetime_corrected']}")
        
        # Create proper timestamp_ns from corrected datetime
        turb_df['timestamp_ns'] = turb_df['datetime_corrected'].apply(lambda x: int(x.value))
        
        # Update datetime_utc to the corrected value
        turb_df['datetime_utc'] = turb_df['datetime_corrected'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        print(f"  ✓ Fixed datetime_utc with correct date")
    else:
        print(f"  Warning: No datetime_utc column found, using existing timestamp_ns")
        # Ensure timestamp_ns is integer
        turb_df['timestamp_ns'] = turb_df['timestamp_ns'].astype(np.int64)
    
    # Sort by timestamp
    turb_df = turb_df.sort_values('timestamp_ns').reset_index(drop=True)
    
    # Print timestamp range for debugging
    if len(turb_df) > 0:
        min_ts = pd.to_datetime(turb_df['timestamp_ns'].min(), unit='ns')
        max_ts = pd.to_datetime(turb_df['timestamp_ns'].max(), unit='ns')
        print(f"  Timestamp range (corrected): {min_ts} to {max_ts}")
        
        # Show sample of turbidity values
        if 'turbidity_ntu' in turb_df.columns:
            valid_turb = turb_df['turbidity_ntu'].notna().sum()
            print(f"  Valid turbidity values: {valid_turb:,}/{len(turb_df):,}")
            if valid_turb > 0:
                print(f"  Turbidity range: {turb_df['turbidity_ntu'].min():.3f} - {turb_df['turbidity_ntu'].max():.3f} NTU")
    
    return turb_df


def merge_turbidity_to_visibility(
    vis_df: pd.DataFrame, 
    turb_df: pd.DataFrame,
    tolerance_seconds: float = 3.0
) -> pd.DataFrame:
    """
    Merge turbidity data into visibility data by nearest timestamp.
    Only keeps records where there is a photo/visibility measurement.
    
    Args:
        vis_df: Visibility DataFrame with timestamp_ns column (from images)
        turb_df: Turbidity DataFrame with timestamp_ns column
        tolerance_seconds: Maximum time difference for matching (default: 3.0 seconds)
        
    Returns:
        Merged DataFrame with turbidity columns added, only for images with visibility
    """
    print(f"\nMerging turbidity into visibility records (tolerance: {tolerance_seconds}s)...")
    print(f"  Base dataset: {len(vis_df):,} images with visibility measurements")
    
    # Check for timestamp overlap
    if len(vis_df) > 0 and len(turb_df) > 0:
        vis_min = vis_df['timestamp_ns'].min()
        vis_max = vis_df['timestamp_ns'].max()
        turb_min = turb_df['timestamp_ns'].min()
        turb_max = turb_df['timestamp_ns'].max()
        
        print(f"\n  Timestamp overlap check:")
        print(f"    Visibility range: {pd.to_datetime(vis_min, unit='ns')} to {pd.to_datetime(vis_max, unit='ns')}")
        print(f"    Turbidity range:  {pd.to_datetime(turb_min, unit='ns')} to {pd.to_datetime(turb_max, unit='ns')}")
        
        # Check if there's any overlap
        if vis_max < turb_min or vis_min > turb_max:
            print(f"  ✗ WARNING: NO TIMESTAMP OVERLAP! Merge will produce empty results.")
        else:
            overlap_start = max(vis_min, turb_min)
            overlap_end = min(vis_max, turb_max)
            print(f"    Overlap: {pd.to_datetime(overlap_start, unit='ns')} to {pd.to_datetime(overlap_end, unit='ns')}")
    
    # Convert tolerance to nanoseconds
    tolerance_ns = int(tolerance_seconds * 1e9)
    print(f"  Tolerance: {tolerance_ns:,} nanoseconds ({tolerance_seconds}s)")
    
    # Perform merge_asof (nearest neighbor merge)
    # Start with visibility (images) as base, add turbidity data
    print(f"\n  Performing merge_asof...")
    merged_df = pd.merge_asof(
        vis_df,
        turb_df,
        on='timestamp_ns',
        direction='nearest',
        tolerance=tolerance_ns
    )
    
    print(f"  Merge complete: {len(merged_df):,} records")
    
    # Count successful matches (where turbidity was found)
    matched_count = merged_df['turbidity_ntu'].notna().sum()
    match_rate = (matched_count / len(merged_df) * 100) if len(merged_df) > 0 else 0
    
    print(f"  ✓ Found turbidity for {matched_count:,}/{len(merged_df):,} images ({match_rate:.1f}%)")
    
    if matched_count < len(merged_df):
        unmatched = len(merged_df) - matched_count
        print(f"  ⚠ {unmatched:,} images without matching turbidity data")
        print(f"  → Keeping only records with both visibility and turbidity")
        
        # Keep only records where we have turbidity data
        merged_df = merged_df[merged_df['turbidity_ntu'].notna()].copy()
        print(f"  ✓ Final dataset: {len(merged_df):,} records with both measurements")
    
    return merged_df


def save_merged_data(merged_df: pd.DataFrame, output_path: str):
    """
    Save merged data to CSV with optimized column order.
    
    Args:
        merged_df: Merged DataFrame
        output_path: Output CSV path
    """
    print(f"\nSaving merged data to: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Reorder columns to put key fields first
    key_cols = ['image', 'timestamp_ns', 'datetime_utc', 'visibility', 'visibility_confidence', 
                'turbidity_ntu', 'latitude', 'longitude', 'depth_m', 'altitude_m']
    
    # Add any remaining columns
    other_cols = [col for col in merged_df.columns if col not in key_cols]
    ordered_cols = [col for col in key_cols if col in merged_df.columns] + other_cols
    
    merged_df = merged_df[ordered_cols]
    
    # Save to CSV
    merged_df.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved {len(merged_df):,} records")
    
    # Print column summary
    print("\nColumn summary:")
    for col in ordered_cols[:15]:  # Show first 15 columns
        if col in merged_df.columns:
            non_null = merged_df[col].notna().sum()
            print(f"  {col}: {non_null:,}/{len(merged_df):,} non-null")
    
    if len(ordered_cols) > 15:
        print(f"  ... and {len(ordered_cols) - 15} more columns")


def print_merge_statistics(merged_df: pd.DataFrame):
    """Print statistics about the merged dataset."""
    print("\n" + "=" * 70)
    print("MERGE STATISTICS")
    print("=" * 70)
    
    total_records = len(merged_df)
    print(f"\nTotal records (images with both visibility and turbidity): {total_records:,}")
    
    # Visibility category breakdown
    if 'visibility' in merged_df.columns:
        vis_counts = merged_df['visibility'].value_counts().sort_index()
        print("\nVisibility category distribution:")
        for category, count in vis_counts.items():
            pct = (count / total_records * 100)
            print(f"  {category}: {count:,} ({pct:.1f}%)")
    
    # Turbidity statistics
    if 'turbidity_ntu' in merged_df.columns:
        turb_valid = merged_df['turbidity_ntu'].notna()
        if turb_valid.sum() > 0:
            turb_stats = merged_df.loc[turb_valid, 'turbidity_ntu'].describe()
            print(f"\nTurbidity (NTU) statistics:")
            print(f"  Count: {turb_stats['count']:.0f}")
            print(f"  Mean: {turb_stats['mean']:.3f}")
            print(f"  Std: {turb_stats['std']:.3f}")
            print(f"  Min: {turb_stats['min']:.3f}")
            print(f"  25%: {turb_stats['25%']:.3f}")
            print(f"  50%: {turb_stats['50%']:.3f}")
            print(f"  75%: {turb_stats['75%']:.3f}")
            print(f"  Max: {turb_stats['max']:.3f}")
    
    # Navigation data availability
    nav_cols = ['latitude', 'longitude', 'depth_m', 'altitude_m']
    nav_available = [col for col in nav_cols if col in merged_df.columns]
    if nav_available:
        print(f"\nNavigation data availability:")
        for col in nav_available:
            count = merged_df[col].notna().sum()
            pct = (count / total_records * 100)
            print(f"  {col}: {count:,}/{total_records:,} ({pct:.1f}%)")
    
    print("\n" + "=" * 70 + "\n")


def analyze_turbidity_by_visibility(merged_df: pd.DataFrame, altitude_threshold: float = 10.0, output_dir: str = None):
    """
    Analyze turbidity by visibility category for low-altitude data.
    
    Args:
        merged_df: Merged DataFrame with turbidity and visibility
        altitude_threshold: Maximum altitude to include (meters)
        output_dir: Directory to save analysis outputs
    """
    print("\n" + "=" * 70)
    print("TURBIDITY-VISIBILITY ANALYSIS")
    print("=" * 70)
    
    # Filter by altitude
    if 'altitude_m' not in merged_df.columns:
        print("\n✗ Error: altitude_m column not found. Cannot filter by altitude.")
        return
    
    print(f"\nFiltering data: altitude < {altitude_threshold}m")
    print(f"  Original records: {len(merged_df):,}")
    
    subset = merged_df[merged_df['altitude_m'] < altitude_threshold].copy()
    print(f"  Filtered records: {len(subset):,}")
    
    if len(subset) == 0:
        print("  ✗ No records found with altitude < {altitude_threshold}m")
        return
    
    # Calculate statistics by visibility category
    print(f"\nCalculating mean turbidity and standard error by visibility category...")
    
    # Define visibility category order (worst to best)
    category_order = ['zero', 'poor', 'fair', 'good', 'excellent']
    
    # Group by visibility and calculate stats
    stats_list = []
    for category in category_order:
        cat_data = subset[subset['visibility'] == category]
        if len(cat_data) > 0:
            turb_values = cat_data['turbidity_ntu'].dropna()
            if len(turb_values) > 0:
                mean_turb = turb_values.mean()
                std_turb = turb_values.std()
                se_turb = std_turb / np.sqrt(len(turb_values)) if len(turb_values) > 1 else 0
                
                stats_list.append({
                    'visibility': category,
                    'n': len(turb_values),
                    'mean_turbidity': mean_turb,
                    'std': std_turb,
                    'se': se_turb,
                    'min': turb_values.min(),
                    'max': turb_values.max()
                })
    
    if len(stats_list) == 0:
        print("  ✗ No valid data to analyze")
        return
    
    stats_df = pd.DataFrame(stats_list)
    
    # Print summary table
    print("\nTurbidity by Visibility Category (altitude < {:.1f}m):".format(altitude_threshold))
    print("-" * 70)
    print(f"{'Category':<12} {'N':<8} {'Mean NTU':<12} {'Std':<12} {'SE':<12}")
    print("-" * 70)
    for _, row in stats_df.iterrows():
        print(f"{row['visibility']:<12} {row['n']:<8} {row['mean_turbidity']:<12.3f} {row['std']:<12.3f} {row['se']:<12.3f}")
    print("-" * 70)
    
    # Create plot
    print("\nCreating turbidity vs visibility plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot with error bars
    x_pos = np.arange(len(stats_df))
    bars = ax.bar(x_pos, stats_df['mean_turbidity'], 
                   color=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars (standard error)
    ax.errorbar(x_pos, stats_df['mean_turbidity'], 
                yerr=stats_df['se'],
                fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)
    
    # Customize plot
    ax.set_xlabel('Visibility Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Turbidity (NTU)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean Turbidity by Visibility Category\n(Altitude < {altitude_threshold}m, N={len(subset):,})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['visibility'], fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size labels on bars
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        ax.text(i, row['mean_turbidity'] + row['se'] + 0.1, 
                f"n={row['n']}", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plot_path = Path(output_dir) / f"turbidity_by_visibility_altitude{int(altitude_threshold)}m.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Plot saved: {plot_path}")
    
    plt.close()
    
    # Save statistics to CSV
    if output_dir:
        stats_path = Path(output_dir) / f"turbidity_visibility_stats_altitude{int(altitude_threshold)}m.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"  ✓ Statistics saved: {stats_path}")
    
    print("\n" + "=" * 70 + "\n")
    
    return stats_df


def main():
    """Main execution function."""
    
    # Configuration
    dive_name = "DIVE006"
    base_dir = Path(r"I:\Image_LLS_PRC") / dive_name
    
    # Input paths
    vis_csv_path = base_dir / "report_plots" / f"{dive_name}_Image_Visibility_Results.csv"
    turb_csv_path = base_dir / "report_plots" / f"{dive_name}_Nav_turbidity_compiled.csv"
    
    # Output path
    output_path = base_dir / "report_plots" / f"{dive_name}_turbidity_with_visibility.csv"
    
    # Tolerance for timestamp matching (seconds)
    tolerance = 3.0
    
    print("=" * 70)
    print("VISIBILITY-TURBIDITY MERGE TOOL")
    print("=" * 70)
    print(f"\nDive: {dive_name}")
    print(f"Tolerance: ±{tolerance} seconds")
    print(f"Strategy: Merge turbidity into visibility (keep only images with visibility)")
    
    # Check input files exist
    if not vis_csv_path.exists():
        print(f"\n✗ Error: Visibility file not found: {vis_csv_path}")
        return
    
    if not turb_csv_path.exists():
        print(f"\n✗ Error: Turbidity file not found: {turb_csv_path}")
        return
    
    try:
        # Load visibility data first to extract dive date
        vis_df = load_visibility_results(str(vis_csv_path))
        
        # Extract dive date from first image timestamp
        dive_date = None
        if len(vis_df) > 0:
            first_ts = pd.to_datetime(vis_df['timestamp_ns'].iloc[0], unit='ns')
            dive_date = first_ts.strftime('%Y-%m-%d')
            print(f"\nDetected dive date from images: {dive_date}")
        
        # Load turbidity data with correct date
        turb_df = load_turbidity_compiled(str(turb_csv_path), dive_date=dive_date)
        
        # Merge - start with visibility, add turbidity, keep only matched records
        merged_df = merge_turbidity_to_visibility(vis_df, turb_df, tolerance)
        
        # Print statistics
        print_merge_statistics(merged_df)
        
        # Save
        save_merged_data(merged_df, str(output_path))
        
        print("\n✓ Merge complete!")
        print(f"\nOutput file: {output_path}")
        
        # Perform analysis: turbidity by visibility for low altitude
        analyze_turbidity_by_visibility(
            merged_df, 
            altitude_threshold=10.0,
            output_dir=str(base_dir / "report_plots")
        )
        
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
