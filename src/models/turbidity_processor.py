"""
Turbidity Data Processor
Reads turbidity data from ROS2 MCAP bag files, compiles to CSV,
and generates analysis plots.

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
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# ── CDR byte offsets ─────────────────────────────────────────────────────────
TURB_TOPIC  = "/r620/ros_remus/ds_msgs/turbidity"
STAT_TOPIC  = "/r620/ros_remus/status"

NTU_OFFSET   = 44   # float64 NTU in turbidity payload
STAT_LAT_OFF = 28   # float64 latitude  in status payload
STAT_LON_OFF = 36   # float64 longitude in status payload
STAT_DEP_OFF = 52   # float64 depth (m, negative below surface) in status payload
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
    Processes turbidity data from ROS2 MCAP bag files.

    Workflow
    --------
    1. Recursively scan a directory for .mcap files.
    2. Parse turbidity (NTU + timestamp) from the turbidity topic.
    3. Parse navigation state (lat, lon, depth + timestamp) from the
       status topic in the same bags.
    4. Time-merge the two streams by nearest timestamp.
    5. Export a compiled CSV.
    6. Generate plots: Turbidity vs Time, Turbidity vs Depth,
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

    # ── Bag parsing ───────────────────────────────────────────────────────────

    def _parse_single_bag(self, mcap_path: Path) -> tuple[list[dict], list[dict]]:
        """
        Parse one .mcap file.

        Returns
        -------
        turb_rows : list of {timestamp_ns, datetime_utc, turbidity_ntu}
        stat_rows : list of {timestamp_ns, latitude, longitude, depth_m}
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

        try:
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages(
                    topics=[TURB_TOPIC, STAT_TOPIC]
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
                            stat_rows.append({
                                "timestamp_ns": ts,
                                "latitude":     lat,
                                "longitude":    lon,
                                "depth_m":      abs(depth),
                            })

        except Exception as e:
            self.log(f"  [ERROR] Failed to read {mcap_path.name}: {e}")

        return turb_rows, stat_rows

    def _parse_all_bags(self, mcap_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parse all bags; return (turb_df, stat_df)."""
        all_turb: list[dict] = []
        all_stat: list[dict] = []

        for path in mcap_files:
            self.log(f"  Parsing: {path.name}")
            t_rows, s_rows = self._parse_single_bag(path)
            self.log(f"    → {len(t_rows):,} turbidity, {len(s_rows):,} nav/status messages")
            all_turb.extend(t_rows)
            all_stat.extend(s_rows)

        if not all_turb:
            return pd.DataFrame(), pd.DataFrame()

        turb_df = pd.DataFrame(all_turb).sort_values("timestamp_ns").reset_index(drop=True)
        stat_df = pd.DataFrame(all_stat).sort_values("timestamp_ns").reset_index(drop=True) if all_stat else pd.DataFrame()

        return turb_df, stat_df

    # ── Time merge ────────────────────────────────────────────────────────────

    def _merge_with_nav(
        self, turb_df: pd.DataFrame, stat_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge turbidity with nav data by nearest timestamp (10-second tolerance)."""
        if stat_df.empty:
            self.log("  ⚠ No nav/status data found – lat/lon/depth columns will be NaN")
            turb_df["latitude"]  = float("nan")
            turb_df["longitude"] = float("nan")
            turb_df["depth_m"]   = float("nan")
            return turb_df

        merged = pd.merge_asof(
            turb_df.copy(),
            stat_df[["timestamp_ns", "latitude", "longitude", "depth_m"]].copy(),
            on="timestamp_ns",
            direction="nearest",
            tolerance=10_000_000_000,   # 10 seconds in nanoseconds
        )
        matched = merged["latitude"].notna().sum()
        self.log(f"  Time-matched {matched:,}/{len(merged):,} turbidity rows with nav data")
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
                   "latitude", "longitude", "depth_m"]
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

    def _plot_turbidity_vs_time(
        self, df: pd.DataFrame, output_dir: str, file_prefix: str
    ) -> str | None:
        """Turbidity (NTU) vs UTC time."""
        try:
            df2 = df.dropna(subset=["turbidity_ntu"]).copy()
            if df2.empty:
                self.log("  ⚠ No turbidity data for time plot")
                return None

            df2["dt"] = pd.to_datetime(df2["datetime_utc"], utc=True)

            fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")

            sc = ax.scatter(
                df2["dt"], df2["turbidity_ntu"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                s=6, alpha=0.7, linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("Turbidity (NTU)", fontsize=10)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Turbidity (NTU)")
            ax.set_title("Turbidity vs Time")
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

            sc = ax.scatter(
                df2["turbidity_ntu"], df2["depth_m"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                s=8, alpha=0.6, linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("Turbidity (NTU)", fontsize=10)

            ax.invert_yaxis()   # depth increases downward
            ax.set_xlabel("Turbidity (NTU)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Turbidity vs Depth")
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

            sc = ax.scatter(
                df2["longitude"], df2["latitude"],
                c=df2["turbidity_ntu"], cmap="viridis_r",
                s=10, alpha=0.8, linewidths=0,
            )
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
            Root directory to search recursively for .mcap bag files.
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
        self.log(f"Scanning for .mcap files in: {nav_directory}")

        # 1. Find bags
        mcap_files = self.find_mcap_files(nav_directory)
        if not mcap_files:
            self.log("  ✗ No .mcap files found – turbidity processing skipped")
            return False

        # 2. Parse bags
        turb_df, stat_df = self._parse_all_bags(mcap_files)

        if turb_df.empty:
            self.log("  ✗ No turbidity messages found in any bag – skipping")
            return False

        self.log(f"  Total turbidity messages: {len(turb_df):,}")

        # 3. Merge with nav
        merged_df = self._merge_with_nav(turb_df, stat_df)

        # 4. Export CSV
        self.export_csv(merged_df, output_dir, file_prefix)

        # 5. Generate plots
        self.log("  Generating turbidity plots...")
        self._plot_turbidity_vs_time(merged_df, output_dir, file_prefix)
        self._plot_turbidity_vs_depth(merged_df, output_dir, file_prefix)
        self._plot_turbidity_map(merged_df, output_dir, file_prefix)

        self.log("✓ Turbidity processing complete")
        return True
