"""
Navigation Track to Shapefile Converter
Adapted from Remus_NavTXTToSHP.py (arcpy version) to use geopandas/shapely.

Reads vehicle navigation data from:
  1. *_Navdata.txt  (primary — Remus full-export, Lon/Lat in DDMM format)
  2. Fallback: NAV_STATE.txt / PHINS INS.txt / STATE.txt parsed via
     NavigationDataMerger (already handles those formats)

Output: <prefix>dissolved_navtrack.shp  (single dissolved LineString + WGS-84)
"""

import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ── DDMM coordinate parser ────────────────────────────────────────────────────

def _parse_ddmm_coord(raw: str) -> tuple[float | None, float | None]:
    """
    Parse a Navdata Lon/Lat field like '29N26.42398  87W41.44419'.

    Returns (latitude_dd, longitude_dd) as signed decimal degrees,
    or (None, None) on failure.
    """
    raw = raw.strip()
    # Pattern: <deg><hemi><min.mmmmm>  <deg><hemi><min.mmmmm>
    # Hemisphere letters: N/S for lat, E/W for lon
    pat = re.compile(
        r'(\d+)([NS])([\d.]+)\s+'
        r'(\d+)([EW])([\d.]+)',
        re.IGNORECASE
    )
    m = pat.search(raw)
    if not m:
        return None, None

    lat_deg, lat_hemi, lat_min = int(m.group(1)), m.group(2).upper(), float(m.group(3))
    lon_deg, lon_hemi, lon_min = int(m.group(4)), m.group(5).upper(), float(m.group(6))

    lat = lat_deg + lat_min / 60.0
    lon = lon_deg + lon_min / 60.0

    if lat_hemi == 'S':
        lat = -lat
    if lon_hemi == 'W':
        lon = -lon

    return lat, lon


# ── Navdata.txt reader ────────────────────────────────────────────────────────

def _load_navdata_txt(file_path: str, log_fn=None) -> pd.DataFrame | None:
    """
    Load a Remus *_Navdata.txt file (row 0 = headers, row 1 = units, row 2+ = data).
    Returns DataFrame with columns: datetime, latitude, longitude, depth_m
    (all others are dropped for efficiency).
    """
    def log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    try:
        # Row 0 = headers, row 1 = units — skip the units row
        df = pd.read_csv(file_path, header=0, skiprows=[1],
                         skipinitialspace=True, low_memory=False)

        # Clean column names
        df.columns = [c.strip().strip(',') for c in df.columns]

        # Find Time, Date, Lon/Lat columns (fuzzy match)
        def _find_col(candidates, columns):
            for cand in candidates:
                for col in columns:
                    if cand.lower() in col.lower():
                        return col
            return None

        time_col   = _find_col(['time'],         df.columns)
        date_col   = _find_col(['date'],         df.columns)
        lonlat_col = _find_col(['lon/lat', 'lonlat', 'lon_lat'], df.columns)
        depth_col  = _find_col(['depth of vehicle', 'depth'], df.columns)

        if not time_col or not date_col or not lonlat_col:
            log(f"  ⚠ Could not find required columns in {os.path.basename(file_path)}")
            log(f"    Columns found: {list(df.columns[:10])}")
            return None

        log(f"  Navdata columns: time='{time_col}', date='{date_col}', "
            f"lonlat='{lonlat_col}', depth='{depth_col}'")

        # Parse datetime
        try:
            df['datetime'] = pd.to_datetime(
                df[date_col].astype(str).str.strip() + ' ' +
                df[time_col].astype(str).str.strip(),
                format='%m/%d/%Y %H:%M:%S.%f',
                errors='coerce'
            )
        except Exception:
            df['datetime'] = pd.NaT

        # Parse Lon/Lat
        parsed = df[lonlat_col].astype(str).apply(_parse_ddmm_coord)
        df['latitude']  = parsed.apply(lambda x: x[0])
        df['longitude'] = parsed.apply(lambda x: x[1])

        # Parse depth
        if depth_col:
            df['depth_m'] = pd.to_numeric(df[depth_col], errors='coerce').abs()
        else:
            df['depth_m'] = float('nan')

        # Drop rows without valid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)]

        log(f"  Parsed {len(df):,} valid coordinate rows from Navdata file")
        return df[['datetime', 'latitude', 'longitude', 'depth_m']].reset_index(drop=True)

    except Exception as e:
        log(f"  ⚠ Error loading Navdata file: {e}")
        log(traceback.format_exc())
        return None


# ── Fallback: nav_merger reader ───────────────────────────────────────────────

def _load_via_nav_merger(directory: str, log_fn=None) -> pd.DataFrame | None:
    """
    Load navigation data using the existing NavigationDataMerger
    (handles NAV_STATE, STATE, PHINS INS, ADCP, etc.).
    Returns DataFrame with at least latitude and longitude columns.
    """
    def log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    try:
        from src.models.nav_merger import NavigationDataMerger
        merger = NavigationDataMerger(log_callback=log_fn)
        df = merger.merge_navigation_directory(directory, log_fn)

        if df is None or df.empty:
            log("  ⚠ nav_merger returned no data")
            return None

        # Ensure standard column names
        lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)

        if not lat_col or not lon_col:
            log(f"  ⚠ No lat/lon columns found in merged nav data. Columns: {list(df.columns)}")
            return None

        df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)]

        log(f"  Loaded {len(df):,} coordinate rows via nav_merger fallback")
        return df

    except Exception as e:
        log(f"  ⚠ nav_merger fallback failed: {e}")
        log(traceback.format_exc())
        return None


# ── Shapefile writer ──────────────────────────────────────────────────────────

def _write_shapefile(df: pd.DataFrame, out_path: str, log_fn=None) -> bool:
    """
    Write a dissolved LineString shapefile from lat/lon points.
    CRS: WGS-84 (EPSG:4326).
    """
    def log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    try:
        import geopandas as gpd
        from shapely.geometry import LineString

        coords = list(zip(df['longitude'], df['latitude']))
        if len(coords) < 2:
            log("  ⚠ Not enough points to create a LineString (need ≥ 2)")
            return False

        line = LineString(coords)
        gdf = gpd.GeoDataFrame(
            {'geometry': [line], 'pt_count': [len(coords)]},
            crs='EPSG:4326'
        )

        # Add summary attribute columns if available
        for col, label in [('depth_m', 'max_depth'), ('datetime', 'start_time')]:
            if col in df.columns:
                try:
                    if col == 'depth_m':
                        gdf[label] = df[col].dropna().max()
                    elif col == 'datetime':
                        dt_valid = df[col].dropna()
                        if not dt_valid.empty:
                            gdf[label] = str(dt_valid.iloc[0])
                except Exception:
                    pass

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdf.to_file(out_path)
        log(f"  ✓ Shapefile written: {os.path.basename(out_path)}")
        return True

    except ImportError:
        log("  ✗ geopandas is required for shapefile export (pip install geopandas)")
        return False
    except Exception as e:
        log(f"  ✗ Error writing shapefile: {e}")
        log(traceback.format_exc())
        return False


# ── Public entry point ────────────────────────────────────────────────────────

def nav_to_shapefile(
    nav_directory: str,
    output_dir: str,
    file_prefix: str = "",
    log_fn=None,
) -> bool:
    """
    Convert navigation track data to a dissolved LineString shapefile.

    Parameters
    ----------
    nav_directory : str
        Directory containing navigation files (*_Navdata.txt and/or
        NAV_STATE.txt, PHINS INS.txt, STATE.txt, etc.).
    output_dir : str
        Output directory for the shapefile.
    file_prefix : str
        Prefix for the output filename, e.g. "DIVE020_".
    log_fn : callable, optional
        Logging callback.

    Returns
    -------
    bool : True on success.
    """
    def log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    log("=" * 60)
    log("NAV TRACK → SHAPEFILE")
    log("=" * 60)

    directory = Path(nav_directory)
    df = None

    # ── Priority 1: *_Navdata.txt ─────────────────────────────────────────────
    navdata_files = sorted(directory.glob("*_Navdata.txt"))
    if not navdata_files:
        # Also search one level up (vehicle_raw might be a sibling of bags/)
        navdata_files = sorted(directory.parent.glob("*_Navdata.txt"))

    if navdata_files:
        navdata_path = navdata_files[0]
        log(f"  Primary source: {navdata_path.name}")
        df = _load_navdata_txt(str(navdata_path), log_fn)
        if df is None or df.empty:
            log("  ⚠ Primary source failed, trying fallback...")
            df = None

    # ── Priority 2: nav_merger fallback ──────────────────────────────────────
    if df is None:
        log("  Fallback source: nav_merger (NAV_STATE / PHINS INS / STATE / ADCP)")
        df = _load_via_nav_merger(str(directory), log_fn)

    if df is None or df.empty:
        log("  ✗ No usable navigation data found — shapefile not created")
        return False

    # ── Thin the track (max 50k points is plenty for a line) ─────────────────
    if len(df) > 50_000:
        step = len(df) // 50_000 + 1
        df = df.iloc[::step].reset_index(drop=True)
        log(f"  Thinned to {len(df):,} points for shapefile efficiency")

    # ── Build output filename ─────────────────────────────────────────────────
    stem = f"{file_prefix}dissolved_navtrack"
    out_shp = os.path.join(output_dir, stem + ".shp")

    return _write_shapefile(df, out_shp, log_fn)
