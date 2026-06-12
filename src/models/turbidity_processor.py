"""
Turbidity Data Processor
Reads turbidity data from ROS2 MCAP bag files and/or exported TURBIDITY.txt
files, compiles to CSV, and generates analysis plots.

Bag parsing approach (CDR struct offsets confirmed empirically):
  /r620/ros_remus/ds_msgs/turbidity   → NTU at byte offset 44 (float64 LE)
  /r620/ros_remus/status              → lat offset 28, lon offset 36,
                                        depth offset 52 (all float64 LE)
"""

import os
import csv
import struct
import datetime
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# ── CDR byte offsets ─────────────────────────────────────────────────────────
TURB_TOPIC  = "/r620/ros_remus/ds_msgs/turbidity"
STAT_TOPIC  = "/r620/ros_remus/status"
BATHY_TOPIC = "/r620/ros_remus/bathymetry"

NTU_OFFSET   = 44   # float64 NTU in turbidity payload
STAT_LAT_OFF = 28   # float64 latitude  in status payload
STAT_LON_OFF = 36   # float64 longitude in status payload
STAT_DEP_OFF = 52   # float64 depth (m, negative below surface) in status payload
BATHY_ALT_OFF = 44  # float64 altitude (m) in bathymetry payload
# ─────────────────────────────────────────────────────────────────────────────


def _ns_to_utc(ns: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ns / 1e9, tz=datetime.timezone.utc)


def _read_float64(data: bytes, offset: int) -> float | None:
    """Safe float64 read; returns None if out-of-bounds."""
    if len(data) < offset + 8:
        return None
    return struct.unpack_from('<d', data, offset)[0]


class TurbidityProcessor:
    """
    Processes turbidity data from ROS2 MCAP bag files and exported
    TURBIDITY text files.

    Workflow
    --------
     1. Recursively scan a directory for .mcap files and TURBIDITY.txt files.
     2. Parse turbidity (NTU + timestamp) and nav state (lat/lon/depth)
         from discovered sources.
     3. Time-merge the two streams by nearest timestamp.
     4. Export a compiled CSV.
     5. Generate plots: Turbidity vs Time, Turbidity vs Depth,
       Turbidity Map (lat/lon coloured by NTU using viridis).
    """

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor']   = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str):
        print(msg)
        if self.log_callback:
            self.log_callback(msg)

    # ── Bag discovery ─────────────────────────────────────────────────────────

    def find_mcap_files(self, root_dir: str) -> list[Path]:
        """Recursively find all .mcap files under root_dir."""
        root = Path(root_dir)
        mcap_files = sorted(root.rglob("*.mcap"))
        self.log(f"  Found {len(mcap_files)} .mcap file(s) under {root_dir}")
        return mcap_files

    def find_turbidity_text_files(self, root_dir: str) -> list[Path]:
        """Recursively find exported TURBIDITY.txt files under root_dir."""
        root = Path(root_dir)
        txt_files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.name.lower() == "turbidity.txt"
        )
        self.log(f"  Found {len(txt_files)} TURBIDITY.txt file(s) under {root_dir}")
        return txt_files
    
    def find_altitude_source_files(self, root_dir: str) -> list[Path]:
        """Recursively find ADCP.txt and BATHY.txt files for altitude data."""
        root = Path(root_dir)
        alt_files = []
        
        # Find ADCP files (preferred source for altitude)
        adcp_files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.name.upper() == "ADCP.TXT"
        )
        alt_files.extend(adcp_files)
        
        # Find BATHY files
        bathy_files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.name.upper() == "BATHY.TXT"
        )
        alt_files.extend(bathy_files)
        
        self.log(f"  Found {len(alt_files)} altitude source file(s) under {root_dir}")
        return alt_files

    # ── Bag parsing ───────────────────────────────────────────────────────────
    def _parse_altitude_from_nav_file(self, nav_path: Path) -> list[dict]:
        """
        Parse altitude data from ADCP.txt or BATHY.txt files.
        
        ADCP.txt expected columns: mission_msecs, altitude, ...
        BATHY.txt expected columns: latitude, longitude, depth, altitude
        
        Returns
        -------
        alt_rows : list of {timestamp_ns, altitude_m}
        """
        alt_rows: list[dict] = []
        
        try:
            with open(nav_path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                header_cols = set(reader.fieldnames or [])
                
                # Check if required columns exist
                if "altitude" not in header_cols:
                    return alt_rows
                
                has_mission_msecs = "mission_msecs" in header_cols
                
                if not has_mission_msecs:
                    # BATHY.txt doesn't have timestamps - skip it
                    return alt_rows
                
                bad_rows = 0
                for row in reader:
                    try:
                        mission_msecs = float(row["mission_msecs"])
                        ts = int(round(mission_msecs * 1_000_000.0))
                        
                        altitude = float(row["altitude"])
                        
                        # Validate altitude is realistic
                        if 0.0 <= altitude <= 10000.0:
                            alt_rows.append({
                                "timestamp_ns": ts,
                                "altitude_m": altitude,
                            })
                    except (TypeError, ValueError, KeyError):
                        bad_rows += 1
                
                if bad_rows > 0:
                    self.log(f"  [WARN] {nav_path.name}: skipped {bad_rows} malformed altitude row(s)")
        
        except Exception as e:
            self.log(f"  [ERROR] Failed to read altitude from {nav_path.name}: {e}")
        
        return alt_rows

    def _parse_all_altitude_source_files(self, alt_files: list[Path]) -> pd.DataFrame:
        """Parse all altitude source files; return alt_df."""
        all_alt: list[dict] = []
        
        for path in alt_files:
            self.log(f"  Parsing altitude from: {path.name}")
            a_rows = self._parse_altitude_from_nav_file(path)
            self.log(f"    → {len(a_rows):,} altitude rows")
            all_alt.extend(a_rows)
        
        if not all_alt:
            return pd.DataFrame()
        
        alt_df = pd.DataFrame(all_alt).sort_values("timestamp_ns").reset_index(drop=True)
        # Remove duplicates, keeping the first occurrence (ADCP preferred over BATHY)
        alt_df = alt_df.drop_duplicates(subset=["timestamp_ns"], keep="first")
        
        return alt_df

    # ── Bag parsing ───────────────────────────────────────────────────────────────
    def _parse_single_bag(self, mcap_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Parse one .mcap file.

        Returns
        -------
        turb_rows : list of {timestamp_ns, datetime_utc, turbidity_ntu}
        stat_rows : list of {timestamp_ns, latitude, longitude, depth_m}
        alt_rows  : list of {timestamp_ns, altitude_m}
        """
        try:
            from mcap.reader import make_reader
        except ImportError:
            raise ImportError(
                "The 'mcap' package is required for bag parsing. "
                "Install it with:  pip install mcap"
            )

        turb_rows: list[dict] = []
        stat_rows: list[dict] = []
        alt_rows: list[dict] = []

        try:
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages(
                    topics=[TURB_TOPIC, STAT_TOPIC, BATHY_TOPIC]
                ):
                    raw = message.data
                    ts  = message.log_time  # nanoseconds

                    if channel.topic == TURB_TOPIC:
                        ntu = _read_float64(raw, NTU_OFFSET)
                        if ntu is not None and 0.0 <= ntu < 10000.0:
                            turb_rows.append({
                                "timestamp_ns":  ts,
                                "datetime_utc":  _ns_to_utc(ts).strftime("%Y-%m-%d %H:%M:%S.%f"),
                                "turbidity_ntu": ntu,
                            })

                    elif channel.topic == STAT_TOPIC:
                        lat   = _read_float64(raw, STAT_LAT_OFF)
                        lon   = _read_float64(raw, STAT_LON_OFF)
                        depth = _read_float64(raw, STAT_DEP_OFF)
                        if (lat is not None and lon is not None and depth is not None
                                and -90 <= lat <= 90 and -180 <= lon <= 180):
                            # Depth convention: status payload stores a signed
                            # value where negative = below surface.
                            # Store as positive depth (metres below surface).
                            # Validate depth is realistic (0–10000 m); reject if out of range.
                            abs_depth = abs(depth)
                            if 0 <= abs_depth <= 10000:
                                stat_rows.append({
                                    "timestamp_ns": ts,
                                    "latitude":     lat,
                                    "longitude":    lon,
                                    "depth_m":      abs_depth,
                                })

                    elif channel.topic == BATHY_TOPIC:
                        altitude = _read_float64(raw, BATHY_ALT_OFF)
                        if altitude is not None and 0.0 <= altitude <= 10000.0:
                            alt_rows.append({
                                "timestamp_ns": ts,
                                "altitude_m":   altitude,
                            })

        except Exception as e:
            self.log(f"  [ERROR] Failed to read {mcap_path.name}: {e}")

        return turb_rows, stat_rows, alt_rows

    def _parse_all_bags(self, mcap_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Parse all bags; return (turb_df, stat_df, alt_df)."""
        all_turb: list[dict] = []
        all_stat: list[dict] = []
        all_alt: list[dict] = []

        for path in mcap_files:
            self.log(f"  Parsing: {path.name}")
            t_rows, s_rows, a_rows = self._parse_single_bag(path)
            self.log(f"    → {len(t_rows):,} turbidity, {len(s_rows):,} nav/status, {len(a_rows):,} altitude messages")
            all_turb.extend(t_rows)
            all_stat.extend(s_rows)
            all_alt.extend(a_rows)

        if not all_turb:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        turb_df = pd.DataFrame(all_turb).sort_values("timestamp_ns").reset_index(drop=True)
        stat_df = pd.DataFrame(all_stat).sort_values("timestamp_ns").reset_index(drop=True) if all_stat else pd.DataFrame()
        alt_df = pd.DataFrame(all_alt).sort_values("timestamp_ns").reset_index(drop=True) if all_alt else pd.DataFrame()

        return turb_df, stat_df, alt_df

    def _parse_single_turbidity_text(self, txt_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Parse one exported TURBIDITY.txt file.

        Expected columns include at least:
            mission_msecs, latitude, longitude, depth, data1
        Optional: altitude (if present)

        Returns
        -------
        turb_rows : list of {timestamp_ns, datetime_utc, turbidity_ntu}
        stat_rows : list of {timestamp_ns, latitude, longitude, depth_m}
        alt_rows  : list of {timestamp_ns, altitude_m}
        """
        turb_rows: list[dict] = []
        stat_rows: list[dict] = []
        alt_rows: list[dict] = []

        try:
            with open(txt_path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                required_cols = {"mission_msecs", "latitude", "longitude", "depth", "data1"}
                header_cols = set(reader.fieldnames or [])
                if not required_cols.issubset(header_cols):
                    self.log(
                        f"  [WARN] {txt_path.name} missing required columns: "
                        f"{sorted(required_cols - header_cols)}"
                    )
                    return turb_rows, stat_rows, alt_rows
                
                # Check if altitude is present
                has_altitude = "altitude" in header_cols

                bad_rows = 0
                for row in reader:
                    try:
                        mission_msecs = float(row["mission_msecs"])
                        ts = int(round(mission_msecs * 1_000_000.0))

                        ntu = float(row["data1"])
                        lat = float(row["latitude"])
                        lon = float(row["longitude"])
                        depth = float(row["depth"])

                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                            continue
                        if not (0.0 <= ntu < 10000.0):
                            continue

                        turb_rows.append({
                            "timestamp_ns": ts,
                            "datetime_utc": _ns_to_utc(ts).strftime("%Y-%m-%d %H:%M:%S.%f"),
                            "turbidity_ntu": ntu,
                        })
                        stat_rows.append({
                            "timestamp_ns": ts,
                            "latitude": lat,
                            "longitude": lon,
                            "depth_m": abs(depth),
                        })
                        
                        # Parse altitude if present
                        if has_altitude:
                            try:
                                altitude = float(row["altitude"])
                                if 0.0 <= altitude <= 10000.0:
                                    alt_rows.append({
                                        "timestamp_ns": ts,
                                        "altitude_m": altitude,
                                    })
                            except (ValueError, KeyError):
                                pass
                                
                    except (TypeError, ValueError, KeyError):
                        bad_rows += 1

                if bad_rows:
                    self.log(f"  [WARN] {txt_path.name}: skipped {bad_rows} malformed row(s)")

        except Exception as e:
            self.log(f"  [ERROR] Failed to read {txt_path.name}: {e}")

        return turb_rows, stat_rows, alt_rows

    def _parse_all_turbidity_text_files(
        self, txt_files: list[Path]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Parse all TURBIDITY.txt files; return (turb_df, stat_df, alt_df)."""
        all_turb: list[dict] = []
        all_stat: list[dict] = []
        all_alt: list[dict] = []

        for path in txt_files:
            self.log(f"  Parsing: {path.name}")
            t_rows, s_rows, a_rows = self._parse_single_turbidity_text(path)
            self.log(f"    → {len(t_rows):,} turbidity, {len(s_rows):,} nav/status, {len(a_rows):,} altitude rows")
            all_turb.extend(t_rows)
            all_stat.extend(s_rows)
            all_alt.extend(a_rows)

        if not all_turb:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        turb_df = pd.DataFrame(all_turb).sort_values("timestamp_ns").reset_index(drop=True)
        stat_df = pd.DataFrame(all_stat).sort_values("timestamp_ns").reset_index(drop=True) if all_stat else pd.DataFrame()
        alt_df = pd.DataFrame(all_alt).sort_values("timestamp_ns").reset_index(drop=True) if all_alt else pd.DataFrame()
        return turb_df, stat_df, alt_df

    def load_turbidity_and_status(self, nav_directory: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load turbidity, nav/status, and altitude rows from available sources.
        
        Returns
        -------
        turb_df : DataFrame with turbidity data
        stat_df : DataFrame with lat/lon/depth data
        alt_df  : DataFrame with altitude data
        """
        txt_files = self.find_turbidity_text_files(nav_directory)

        # Prefer text exports first. They are faster to parse and are often the
        # primary source for dives where bags are absent.
        if txt_files:
            txt_turb_df, txt_stat_df, txt_alt_df = self._parse_all_turbidity_text_files(txt_files)
            if not txt_turb_df.empty:
                self.log(
                    f"  Loaded {len(txt_turb_df):,} total turbidity rows from "
                    f"{len(txt_files)} text file(s)"
                )
                
                # If TURBIDITY.txt didn't have altitude, load from separate sources
                if txt_alt_df.empty:
                    self.log("  TURBIDITY.txt lacks altitude data - searching for ADCP/BATHY files")
                    alt_files = self.find_altitude_source_files(nav_directory)
                    if alt_files:
                        txt_alt_df = self._parse_all_altitude_source_files(alt_files)
                        if not txt_alt_df.empty:
                            self.log(f"  Loaded {len(txt_alt_df):,} altitude rows from separate nav files")
                    else:
                        self.log("  No ADCP/BATHY files found for altitude data")
                
                return txt_turb_df, txt_stat_df, txt_alt_df

            self.log("  TURBIDITY.txt file(s) found but no valid rows parsed; falling back to MCAP")

        mcap_files = self.find_mcap_files(nav_directory)
        if mcap_files:
            bag_turb_df, bag_stat_df, bag_alt_df = self._parse_all_bags(mcap_files)
        else:
            bag_turb_df, bag_stat_df, bag_alt_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if not bag_turb_df.empty:
            self.log(
                f"  Loaded {len(bag_turb_df):,} total turbidity rows "
                f"from {len(mcap_files)} MCAP file(s)"
            )
            
            # If bags didn't have altitude, try to load from separate files
            if bag_alt_df.empty:
                self.log("  MCAP bags lack altitude data - searching for ADCP/BATHY files")
                alt_files = self.find_altitude_source_files(nav_directory)
                if alt_files:
                    bag_alt_df = self._parse_all_altitude_source_files(alt_files)
                    if not bag_alt_df.empty:
                        self.log(f"  Loaded {len(bag_alt_df):,} altitude rows from separate nav files")

        return bag_turb_df, bag_stat_df, bag_alt_df

    # ── Time merge ────────────────────────────────────────────────────────────

    def _merge_with_nav(
        self, turb_df: pd.DataFrame, stat_df: pd.DataFrame, alt_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge turbidity with nav data and altitude by nearest timestamp (10-second tolerance).
        
        Parameters
        ----------
        turb_df : DataFrame with turbidity data
        stat_df : DataFrame with lat/lon/depth data
        alt_df  : DataFrame with altitude data
        
        Returns
        -------
        merged_df : DataFrame with all columns merged by timestamp
        """
        # First merge with nav/status data
        if stat_df.empty:
            self.log("  ⚠ No nav/status data found – lat/lon/depth columns will be NaN")
            turb_df["latitude"]  = float("nan")
            turb_df["longitude"] = float("nan")
            turb_df["depth_m"]   = float("nan")
            merged = turb_df.copy()
        else:
            merged = pd.merge_asof(
                turb_df.copy(),
                stat_df[["timestamp_ns", "latitude", "longitude", "depth_m"]].copy(),
                on="timestamp_ns",
                direction="nearest",
                tolerance=10_000_000_000,   # 10 seconds in nanoseconds
            )
            matched = merged["latitude"].notna().sum()
            self.log(f"  Time-matched {matched:,}/{len(merged):,} turbidity rows with nav data")
        
        # Then merge with altitude data
        if alt_df.empty:
            self.log("  ⚠ No altitude data found – altitude column will be NaN")
            merged["altitude_m"] = float("nan")
        else:
            merged = pd.merge_asof(
                merged,
                alt_df[["timestamp_ns", "altitude_m"]].copy(),
                on="timestamp_ns",
                direction="nearest",
                tolerance=10_000_000_000,   # 10 seconds in nanoseconds
            )
            matched_alt = merged["altitude_m"].notna().sum()
            self.log(f"  Time-matched {matched_alt:,}/{len(merged):,} turbidity rows with altitude data")
        
        return merged

    # ── CSV export ────────────────────────────────────────────────────────────

    def export_csv(
        self, df: pd.DataFrame, output_dir: str, file_prefix: str = ""
    ) -> str:
        """Write compiled turbidity CSV; returns the file path."""
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{file_prefix}turbidity_compiled.csv"
        out_path = os.path.join(output_dir, fname)

        columns = ["timestamp_ns", "datetime_utc", "turbidity_ntu",
                   "latitude", "longitude", "depth_m", "altitude_m"]
        # Only include columns that exist
        export_cols = [c for c in columns if c in df.columns]
        df[export_cols].to_csv(out_path, index=False)

        self.log(f"  ✓ Turbidity CSV saved: {fname}  ({len(df):,} rows)")
        return out_path

    # ── Plots ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_coord_axis(ax, which="both"):
        """Avoid scientific notation on coordinate axes."""
        fmt = FuncFormatter(lambda x, _: f"{x:.4f}")
        if which in ("x", "both"):
            ax.xaxis.set_major_formatter(fmt)
        if which in ("y", "both"):
            ax.yaxis.set_major_formatter(fmt)
    
    @staticmethod
    def _get_turbidity_colorbar_limits(data: pd.Series, percentile_range: tuple = (5, 95)) -> tuple:
        """
        Compute robust colorbar limits using percentile clipping.
        
        Handles datasets with low variation or outliers by clipping to 5th–95th percentile.
        This approach is well-documented in oceanographic data visualization (e.g., Balch et al.,
        JTECH 2005; Bailey & Wilis, RS 2010) for robust visualization of sparse or variable data.
        
        Parameters
        ----------
        data : pd.Series
            Turbidity values in NTU.
        percentile_range : tuple
            (lower_percentile, upper_percentile) for clipping. Default (5, 95).
            
        Returns
        -------
        (vmin, vmax) : tuple of float
            Robust color scale limits.
        """
        valid_data = data.dropna()
        if len(valid_data) == 0:
            return 0.0, 50.0  # Fallback
        
        vmin = np.percentile(valid_data, percentile_range[0])
        vmax = np.percentile(valid_data, percentile_range[1])
        
        # Ensure at least minimal range for visualization
        if vmax <= vmin:
            vmax = vmin + 1.0
        
        return vmin, vmax

    @staticmethod
    def _format_turbidity_time_axis(ax, time_data: pd.Series, log_callback=None):
        """Format turbidity time axis with midnight crossover handling."""
        if time_data.empty:
            return

        mission_start = time_data.min()
        mission_end = time_data.max()
        time_span = (mission_end - mission_start).total_seconds()

        start_seconds = mission_start.hour * 3600 + mission_start.minute * 60 + mission_start.second
        end_seconds = mission_end.hour * 3600 + mission_end.minute * 60 + mission_end.second
        unique_dates = time_data.dt.date.nunique()

        crosses_midnight = (
            (time_span > 6 * 3600 and end_seconds < start_seconds)
            or (time_span > 18 * 3600)
            or (unique_dates > 1)
        )

        if time_span > 2 * 86_400:
            # Defensive fallback for anomalous ranges to avoid MAXTICKS errors.
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            return

        if crosses_midnight:
            if log_callback:
                log_callback("  Detected midnight crossing in turbidity time series")

            mission_start_num = mdates.date2num(mission_start.to_pydatetime())

            def time_from_start_formatter(x, _pos):
                elapsed = (x - mission_start_num) * 24.0 * 3600.0
                if elapsed < 0:
                    elapsed = 0
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                return f"T+{hours:02d}:{minutes:02d}"

            if time_span > 4 * 3600:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
            else:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))

            ax.xaxis.set_major_formatter(FuncFormatter(time_from_start_formatter))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            if time_span > 7200:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
            elif time_span > 1800:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))

    def _plot_turbidity_vs_time(
        self, df: pd.DataFrame, output_dir: str, file_prefix: str
    ) -> str | None:
        """Turbidity (NTU) vs UTC time with midnight crossing detection."""
        try:
            df2 = df.dropna(subset=["turbidity_ntu"]).copy()
            if df2.empty:
                self.log("  ⚠ No turbidity data for time plot")
                return None

            df2["dt"] = pd.to_datetime(df2["datetime_utc"], utc=True, errors="coerce")

            # If source timestamps are inconsistent across files/sensors, datetime_utc can span
            # implausibly large ranges and cause matplotlib locator explosions.
            valid_dt = df2["dt"].dropna()
            if not valid_dt.empty:
                dt_span_seconds = (valid_dt.max() - valid_dt.min()).total_seconds()
            else:
                dt_span_seconds = float("inf")

            if dt_span_seconds > 2 * 86_400 and "timestamp_ns" in df2.columns:
                ts = pd.to_numeric(df2["timestamp_ns"], errors="coerce")
                valid_ts = ts.dropna()
                if not valid_ts.empty:
                    base_ts = valid_ts.min()
                    rel_ns = (ts - base_ts).clip(lower=0)
                    base_date = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
                    df2["dt"] = base_date + pd.to_timedelta(rel_ns, unit="ns", errors="coerce")
                    self.log("  Normalized plotting timeline to mission-relative timestamps")

            df2 = df2.dropna(subset=["dt", "turbidity_ntu"])
            if df2.empty:
                self.log("  ⚠ No valid datetime data for turbidity time plot")
                return None

            fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")

            # Use percentile-based limits to handle outliers robustly (5th–95th percentile).
            # This approach is standard in oceanographic remote sensing for low-variation datasets.
            vmin, vmax = self._get_turbidity_colorbar_limits(df2["turbidity_ntu"])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(
                df2["dt"], df2["turbidity_ntu"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                norm=norm, s=6, alpha=0.7, linewidths=0,
            )
            
            # Use native Matplotlib colorbar tick scaling/labeling.
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("Turbidity (NTU)", fontsize=10)

            self._format_turbidity_time_axis(ax, df2["dt"], self.log)
            
            fig.autofmt_xdate()

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Turbidity (NTU)")
            ax.set_title("Turbidity vs Time")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"{file_prefix}Turbidity_vs_Time.png"
            path  = os.path.join(output_dir, fname)
            plt.savefig(path, facecolor="white", bbox_inches="tight", dpi=200)
            plt.close(fig)
            self.log(f"  ✓ Plot saved: {fname}")
            return path
        except Exception as e:
            self.log(f"  ✗ turbidity_vs_time plot failed: {e}")
            self.log(traceback.format_exc())
            return None

    def _plot_turbidity_vs_depth(
        self, df: pd.DataFrame, output_dir: str, file_prefix: str
    ) -> str | None:
        """Turbidity (NTU) vs Depth (m)."""
        try:
            df2 = df.dropna(subset=["turbidity_ntu", "depth_m"]).copy()
            if df2.empty:
                self.log("  ⚠ No depth data for turbidity vs depth plot – skipping")
                return None

            fig, ax = plt.subplots(figsize=(7, 9), facecolor="white")

            # Use percentile-based limits to handle outliers robustly (5th–95th percentile).
            vmin, vmax = self._get_turbidity_colorbar_limits(df2["turbidity_ntu"])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(
                df2["turbidity_ntu"], df2["depth_m"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                norm=norm, s=8, alpha=0.6, linewidths=0,
            )
            
            # Use native Matplotlib colorbar tick scaling/labeling.
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("Turbidity (NTU)", fontsize=10)

            ax.invert_yaxis()   # depth increases downward
            ax.set_xlabel("Turbidity (NTU)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Turbidity vs Depth")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"{file_prefix}Turbidity_vs_Depth.png"
            path  = os.path.join(output_dir, fname)
            plt.savefig(path, facecolor="white", bbox_inches="tight", dpi=200)
            plt.close(fig)
            self.log(f"  ✓ Plot saved: {fname}")
            return path
        except Exception as e:
            self.log(f"  ✗ Turbidity_vs_Depth plot failed: {e}")
            self.log(traceback.format_exc())
            return None

    def _plot_depth_vs_time(self, df: pd.DataFrame, output_dir: str, file_prefix: str) -> str | None:
        """Depth (m) vs UTC time, colorized by turbidity (NTU)."""
        try:
            df2 = df.dropna(subset=["depth_m", "turbidity_ntu"]).copy()
            if df2.empty:
                self.log("  ⚠ No depth/time data for depth vs time plot – skipping")
                return None

            df2["dt"] = pd.to_datetime(df2["datetime_utc"], utc=True, errors="coerce")

            # Keep the time axis consistent with the main turbidity time plot.
            valid_dt = df2["dt"].dropna()
            if not valid_dt.empty:
                dt_span_seconds = (valid_dt.max() - valid_dt.min()).total_seconds()
            else:
                dt_span_seconds = float("inf")

            if dt_span_seconds > 2 * 86_400 and "timestamp_ns" in df2.columns:
                ts = pd.to_numeric(df2["timestamp_ns"], errors="coerce")
                valid_ts = ts.dropna()
                if not valid_ts.empty:
                    base_ts = valid_ts.min()
                    rel_ns = (ts - base_ts).clip(lower=0)
                    base_date = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
                    df2["dt"] = base_date + pd.to_timedelta(rel_ns, unit="ns", errors="coerce")
                    self.log("  Normalized depth-vs-time timeline to mission-relative timestamps")

            df2 = df2.dropna(subset=["dt", "depth_m", "turbidity_ntu"])
            if df2.empty:
                self.log("  ⚠ No valid datetime data for depth vs time plot")
                return None

            fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")

            vmin, vmax = self._get_turbidity_colorbar_limits(df2["turbidity_ntu"])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(
                df2["dt"], df2["depth_m"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                norm=norm, s=6, alpha=0.7, linewidths=0,
            )

            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("Turbidity (NTU)", fontsize=10)

            self._format_turbidity_time_axis(ax, df2["dt"], self.log)
            fig.autofmt_xdate()

            ax.invert_yaxis()
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Depth vs Time")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"{file_prefix}Depth_vs_Time.png"
            path = os.path.join(output_dir, fname)
            plt.savefig(path, facecolor="white", bbox_inches="tight", dpi=200)
            plt.close(fig)
            self.log(f"  ✓ Plot saved: {fname}")
            return path
        except Exception as e:
            self.log(f"  ✗ Depth_vs_Time plot failed: {e}")
            self.log(traceback.format_exc())
            return None

    def _plot_turbidity_map(
        self, df: pd.DataFrame, output_dir: str, file_prefix: str
    ) -> str | None:
        """
        Turbidity map: scatter plot of lat/lon coloured by NTU.
        viridis_r  →  yellow = low turbidity, purple = high turbidity.
        """
        try:
            df2 = df.dropna(subset=["turbidity_ntu", "latitude", "longitude"]).copy()
            if df2.empty:
                self.log("  ⚠ No lat/lon data for turbidity map – skipping")
                return None

            fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")

            # Use percentile-based limits to handle outliers robustly (5th–95th percentile).
            vmin, vmax = self._get_turbidity_colorbar_limits(df2["turbidity_ntu"])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(
                df2["longitude"], df2["latitude"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                norm=norm, s=10, alpha=0.8, linewidths=0,
            )
            
            # Use native Matplotlib colorbar tick scaling/labeling.
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("Turbidity (NTU)", fontsize=10)

            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title("Turbidity Map (Lat/Lon)")
            ax.grid(True, alpha=0.3)
            self._format_coord_axis(ax, "both")
            plt.xticks(rotation=30)

            plt.tight_layout()
            fname = f"{file_prefix}Turbidity_Map.png"
            path  = os.path.join(output_dir, fname)
            plt.savefig(path, facecolor="white", bbox_inches="tight", dpi=200)
            plt.close(fig)
            self.log(f"  ✓ Plot saved: {fname}")
            return path
        except Exception as e:
            self.log(f"  ✗ Turbidity_Map plot failed: {e}")
            self.log(traceback.format_exc())
            return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(
        self,
        nav_directory: str,
        output_dir: str,
        file_prefix: str = "",
    ) -> bool:
        """
        Full turbidity processing pipeline.

        Parameters
        ----------
        nav_directory : str
            Root directory to search recursively for .mcap bags and
            TURBIDITY.txt files.
        output_dir : str
            Directory where the CSV and plots will be saved.
        file_prefix : str
            Optional prefix for output filenames (e.g. "DIVE015_").

        Returns
        -------
        bool : True on success, False if no turbidity data was found.
        """
        self.log("=" * 60)
        self.log("TURBIDITY PROCESSING")
        self.log("=" * 60)
        self.log(f"Scanning for turbidity sources in: {nav_directory}")

        # 1. Load supported turbidity sources
        turb_df, stat_df, alt_df = self.load_turbidity_and_status(nav_directory)

        if turb_df.empty:
            self.log("  ✗ No turbidity data found in MCAP or TURBIDITY.txt sources – skipping")
            return False

        self.log(f"  Total turbidity messages: {len(turb_df):,}")

        # 2. Merge with nav and altitude
        merged_df = self._merge_with_nav(turb_df, stat_df, alt_df)

        # 3. Export CSV
        self.export_csv(merged_df, output_dir, file_prefix)

        # 4. Generate plots
        self.log("  Generating turbidity plots...")
        self._plot_turbidity_vs_time(merged_df, output_dir, file_prefix)
        self._plot_depth_vs_time(merged_df, output_dir, file_prefix)
        self._plot_turbidity_vs_depth(merged_df, output_dir, file_prefix)
        self._plot_turbidity_map(merged_df, output_dir, file_prefix)

        self.log("✓ Turbidity processing complete")
        return True
