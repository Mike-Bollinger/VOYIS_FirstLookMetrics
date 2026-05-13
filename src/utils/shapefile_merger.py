# -*- coding: utf-8 -*-
"""
Shapefile Merger for VOYIS First Look Metrics
==============================================
Combines multiple per-dive shapefiles in a single directory into one merged
shapefile.  Two new attributes are appended to every feature:

  mission_cd  — A user-supplied mission code (e.g. "EN2501")
  dive_num    — The dive number extracted from the filename (DIVE001 → 1)

Only files whose names match the pattern  DIVE###_*.shp  (case-insensitive)
are eligible for inclusion.  A verification panel lets the user review and
deselect individual files before merging.

Dependencies:  geopandas, shapely (both listed in requirements.txt)

Author: Mike Bollinger, Github Copilot
Date:   May 2026
"""

import math
import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Lazy-import geopandas so the script can at least open and show an error
# message if it is missing, rather than crashing on import.
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pattern that a shapefile name must match to be considered a valid dive file
# ---------------------------------------------------------------------------
DIVE_PATTERN = re.compile(r"^(DIVE(\d+))_.*\.shp$", re.IGNORECASE)


def scan_directory(folder: str) -> List[dict]:
    """
    Scan *folder* for .shp files and classify each one.

    Returns a list of dicts with keys:
        filename   – bare filename (no directory)
        filepath   – full absolute path
        dive_label – e.g. "DIVE001"  (or "" if pattern not matched)
        dive_num   – integer dive number  (or None)
        valid      – True if the file matches the expected pattern
    """
    results = []
    try:
        entries = sorted(Path(folder).iterdir())
    except OSError:
        return results

    for entry in entries:
        if entry.suffix.lower() != ".shp":
            continue
        match = DIVE_PATTERN.match(entry.name)
        if match:
            results.append({
                "filename":   entry.name,
                "filepath":   str(entry.resolve()),
                "dive_label": match.group(1).upper(),
                "dive_num":   int(match.group(2)),
                "valid":      True,
            })
        else:
            results.append({
                "filename":   entry.name,
                "filepath":   str(entry.resolve()),
                "dive_label": "",
                "dive_num":   None,
                "valid":      False,
            })
    return results


# ===========================================================================
# Spatial helpers
# ===========================================================================

# Visibility category ordering (lowest → highest quality)
VIS_ORDER: Dict[str, int] = {
    "zero":      0,
    "poor":      1,
    "fair":      2,
    "good":      3,
    "excellent": 4,
}

# Available bin sizes in metres
BIN_SIZES_M = [10, 25, 50, 100]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _find_col(gdf, candidates: List[str]) -> Optional[str]:
    """Return the first column in *candidates* that exists in *gdf* (case-insensitive)."""
    lower_map = {c.lower(): c for c in gdf.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _bin_gdf_by_distance(gdf, bin_size_m: int):
    """
    Sort a per-dive GeoDataFrame along its track, compute cumulative distance,
    and assign each row to a distance bin.

    Returns (annotated_df, lat_col, lon_col) or (None, None, None) if the GDF
    does not have usable coordinate columns.
    """
    lat_col = _find_col(gdf, ["latitude", "lat"])
    lon_col = _find_col(gdf, ["longitude", "lon", "long"])
    if lat_col is None or lon_col is None:
        return None, None, None

    df = gdf.copy()
    df = df.dropna(subset=[lat_col, lon_col])
    if len(df) < 2:
        return None, None, None

    # Sort by datetime if available, otherwise preserve file order
    dt_col = _find_col(df, ["datetime_original", "datetime", "timestamp", "time"])
    if dt_col:
        try:
            df = df.sort_values(dt_col).reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Cumulative along-track distance
    lats = df[lat_col].to_numpy(dtype=float)
    lons = df[lon_col].to_numpy(dtype=float)
    cum_dist = [0.0]
    for i in range(1, len(df)):
        cum_dist.append(cum_dist[-1] + haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i]))

    df["_cum_dist_m"] = cum_dist
    df["_bin"]        = (df["_cum_dist_m"] // bin_size_m).astype(int)
    return df, lat_col, lon_col


# ===========================================================================
# Export processor functions
# Each receives (ext_rec, gdf, bin_size_m) and returns a GeoDataFrame or None.
#   ext_rec – scan result dict extended with "mission_cd"
#   gdf     – per-dive GeoDataFrame (mission_cd + dive_num already inserted)
# ===========================================================================

def _proc_altitude_polyline(ext_rec: dict, gdf, bin_size_m: int):
    """Average altitude per distance bin → polyline GeoDataFrame."""
    df, lat_col, lon_col = _bin_gdf_by_distance(gdf, bin_size_m)
    if df is None:
        return None

    alt_col = _find_col(df, ["altitude", "alt"])
    rows = []
    for bin_id, group in df.groupby("_bin", sort=True):
        coords = list(zip(group[lon_col].astype(float), group[lat_col].astype(float)))
        if len(coords) < 2:
            continue

        avg_alt = None
        if alt_col:
            vals = pd.to_numeric(group[alt_col], errors="coerce").dropna()
            if not vals.empty:
                avg_alt = round(float(vals.mean()), 4)

        rows.append({
            "mission_cd":  ext_rec["mission_cd"],
            "dive_num":    ext_rec["dive_num"],
            "bin_id":      int(bin_id),
            "bin_strt_m":  round(float(group["_cum_dist_m"].min()), 1),
            "bin_end_m":   round(float(group["_cum_dist_m"].max()), 1),
            "n_pts":       int(len(group)),
            "avg_alt_m":   avg_alt,
            "geometry":    LineString(coords),
        })
    return gpd.GeoDataFrame(rows, crs=gdf.crs) if rows else None


def _proc_visibility_polyline(ext_rec: dict, gdf, bin_size_m: int):
    """Modal visibility category per distance bin → polyline GeoDataFrame."""
    df, lat_col, lon_col = _bin_gdf_by_distance(gdf, bin_size_m)
    if df is None:
        return None

    vis_col = _find_col(df, ["visibility", "vis"])
    rows = []
    for bin_id, group in df.groupby("_bin", sort=True):
        coords = list(zip(group[lon_col].astype(float), group[lat_col].astype(float)))
        if len(coords) < 2:
            continue

        modal_vis = None
        if vis_col:
            cats  = group[vis_col].dropna().astype(str).str.lower()
            known = cats[cats.isin(VIS_ORDER)]
            if not known.empty:
                counts    = known.value_counts()
                max_count = counts.max()
                tied      = counts[counts == max_count].index.tolist()
                # On tie, select the highest-ranked category
                modal_vis = max(tied, key=lambda v: VIS_ORDER.get(v, -1))

        rows.append({
            "mission_cd":  ext_rec["mission_cd"],
            "dive_num":    ext_rec["dive_num"],
            "bin_id":      int(bin_id),
            "bin_strt_m":  round(float(group["_cum_dist_m"].min()), 1),
            "bin_end_m":   round(float(group["_cum_dist_m"].max()), 1),
            "n_pts":       int(len(group)),
            "vis_mode":    modal_vis,
            "geometry":    LineString(coords),
        })
    return gpd.GeoDataFrame(rows, crs=gdf.crs) if rows else None


# ===========================================================================
# Export registry
# To add a new summary export: define a processor function above, then append
# a dict to EXPORT_REGISTRY below — no other code changes required.
# ===========================================================================
# Required keys:
#   key        – unique str identifier (used as BooleanVar key)
#   label      – display text shown next to the checkbox
#   needs_bins – True if this export uses distance binning
#   suffix     – output filename: <safe_mission>_<suffix>.shp
#   processor  – callable(ext_rec, gdf, bin_size_m) → GeoDataFrame | None
#                Use None for the built-in merged-points export.
# ===========================================================================
EXPORT_REGISTRY: List[dict] = [
    {
        "key":        "merged_points",
        "label":      "Merged Point Shapefile  (all features concatenated)",
        "needs_bins": False,
        "suffix":     "Merged_Points",
        "processor":  None,                    # handled by _export_merged_points
    },
    {
        "key":        "altitude_polyline",
        "label":      "Altitude Summary Polyline  (avg altitude per distance bin)",
        "needs_bins": True,
        "suffix":     "Altitude_Summary",
        "processor":  _proc_altitude_polyline,
    },
    {
        "key":        "visibility_polyline",
        "label":      "Visibility Summary Polyline  (modal visibility per distance bin)",
        "needs_bins": True,
        "suffix":     "Visibility_Summary",
        "processor":  _proc_visibility_polyline,
    },
    # ── Add new summary exports below ──────────────────────────────────────
    # {
    #     "key":        "depth_polyline",
    #     "label":      "Depth Summary Polyline  (avg depth per distance bin)",
    #     "needs_bins": True,
    #     "suffix":     "Depth_Summary",
    #     "processor":  _proc_depth_polyline,
    # },
]


# ===========================================================================
# GUI
# ===========================================================================

class ShapefileMergerGUI:
    """Main application window for the Shapefile Merger utility."""

    # Column widths for the treeview
    _COL_CFG = [
        ("Include",     60,  "center"),
        ("Filename",   310,  "w"),
        ("Dive",        70,  "center"),
        ("Status",      80,  "center"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Shapefile Merger — VOYIS First Look Metrics")
        self.root.geometry("800x780")
        self.root.resizable(True, True)

        # State
        self.input_folder   = tk.StringVar()
        self.output_folder  = tk.StringVar()
        self.mission_code   = tk.StringVar()
        self.scan_results:  List[dict] = []
        self.include_vars:  List[tk.BooleanVar] = []

        # Export type checkboxes and bin size (initialised before _build_ui)
        self.export_vars:  Dict[str, tk.BooleanVar] = {
            e["key"]: tk.BooleanVar(value=True) for e in EXPORT_REGISTRY
        }
        self.bin_size_var: tk.IntVar = tk.IntVar(value=50)
        # Radio button widgets stored so we can enable/disable them as a group
        self._bin_radios:  List[tk.Radiobutton] = []

        self._build_ui()
        self.root.after(100, self._set_icon)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        # ── Title ──────────────────────────────────────────────────────
        tk.Label(
            self.root,
            text="Shapefile Merger",
            font=("Arial", 16, "bold"),
        ).pack(**pad)

        # ── Input directory ────────────────────────────────────────────
        frm_in = tk.LabelFrame(self.root, text="Input Directory", font=("Arial", 10))
        frm_in.pack(fill="x", padx=12, pady=4)

        tk.Entry(frm_in, textvariable=self.input_folder, width=68).pack(
            side="left", padx=6, pady=6, fill="x", expand=True
        )
        tk.Button(frm_in, text="Browse…", command=self._browse_input).pack(
            side="left", padx=4, pady=6
        )

        # ── Mission code ───────────────────────────────────────────────
        frm_mc = tk.LabelFrame(self.root, text="Mission Code", font=("Arial", 10))
        frm_mc.pack(fill="x", padx=12, pady=4)

        tk.Label(frm_mc, text="Mission Code:", font=("Arial", 10)).pack(
            side="left", padx=6, pady=6
        )
        tk.Entry(frm_mc, textvariable=self.mission_code, width=20).pack(
            side="left", padx=4, pady=6
        )
        tk.Label(
            frm_mc,
            text="(e.g. EN2501)  Will be stored as the 'mission_cd' attribute.",
            font=("Arial", 9),
            fg="#555555",
        ).pack(side="left", padx=8)

        # ── File verification panel ────────────────────────────────────
        frm_files = tk.LabelFrame(
            self.root, text="Detected Shapefiles — Review & Select", font=("Arial", 10)
        )
        frm_files.pack(fill="both", expand=True, padx=12, pady=4)

        # Toolbar above tree
        toolbar = tk.Frame(frm_files)
        toolbar.pack(fill="x", padx=4, pady=(4, 0))
        tk.Button(toolbar, text="Scan / Refresh", command=self._scan).pack(side="left", padx=2)
        tk.Button(toolbar, text="Select All",     command=self._select_all).pack(side="left", padx=2)
        tk.Button(toolbar, text="Select Valid",   command=self._select_valid).pack(side="left", padx=2)
        tk.Button(toolbar, text="Deselect All",   command=self._deselect_all).pack(side="left", padx=2)

        self._valid_count_label = tk.Label(toolbar, text="", font=("Arial", 9), fg="#333")
        self._valid_count_label.pack(side="right", padx=8)

        # Treeview with scrollbar
        tree_frame = tk.Frame(frm_files)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=4)

        cols = [c[0] for c in self._COL_CFG]
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=10)

        for col_name, width, anchor in self._COL_CFG:
            self.tree.heading(col_name, text=col_name)
            self.tree.column(col_name, width=width, anchor=anchor, stretch=(col_name == "Filename"))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Tag colours
        self.tree.tag_configure("valid",   background="#E8F5E9")
        self.tree.tag_configure("invalid", background="#FFEBEE")
        self.tree.tag_configure("checked", foreground="#000000")

        # Toggle include on click
        self.tree.bind("<ButtonRelease-1>", self._on_tree_click)

        # ── Output directory ───────────────────────────────────────────
        frm_out = tk.LabelFrame(self.root, text="Output Directory", font=("Arial", 10))
        frm_out.pack(fill="x", padx=12, pady=4)

        tk.Entry(frm_out, textvariable=self.output_folder, width=68).pack(
            side="left", padx=6, pady=6, fill="x", expand=True
        )
        tk.Button(frm_out, text="Browse…", command=self._browse_output).pack(
            side="left", padx=4, pady=6
        )

        # ── Export options ─────────────────────────────────────────────
        self._build_export_options()

        # ── Action buttons ─────────────────────────────────────────────
        frm_btns = tk.Frame(self.root)
        frm_btns.pack(pady=6)

        self._merge_btn = tk.Button(
            frm_btns,
            text="Merge Shapefiles",
            command=self._merge,
            font=("Arial", 11, "bold"),
            bg="#1565C0",
            fg="white",
            padx=16,
            pady=6,
        )
        self._merge_btn.pack(side="left", padx=8)

        tk.Button(
            frm_btns,
            text="Close",
            command=self.root.destroy,
            font=("Arial", 11),
            padx=16,
            pady=6,
        ).pack(side="left", padx=8)

        # ── Status log ─────────────────────────────────────────────────
        frm_log = tk.LabelFrame(self.root, text="Log", font=("Arial", 10))
        frm_log.pack(fill="x", padx=12, pady=(0, 8))

        self.log = scrolledtext.ScrolledText(frm_log, height=5, state="disabled", wrap="word")
        self.log.pack(fill="x", padx=4, pady=4)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select directory containing shapefiles")
        if folder:
            self.input_folder.set(folder)
            # Auto-set output to same folder if not already set
            if not self.output_folder.get():
                self.output_folder.set(folder)
            self._scan()

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            self.output_folder.set(folder)

    # ------------------------------------------------------------------
    # Export options panel
    # ------------------------------------------------------------------

    def _build_export_options(self):
        """Build the Export Options LabelFrame (checkboxes + bin size radios)."""
        frm = tk.LabelFrame(self.root, text="Export Options", font=("Arial", 10))
        frm.pack(fill="x", padx=12, pady=4)

        # ── Export type checkboxes ─────────────────────────────────────
        chk_frame = tk.Frame(frm)
        chk_frame.pack(fill="x", padx=6, pady=(6, 2))

        for entry in EXPORT_REGISTRY:
            tk.Checkbutton(
                chk_frame,
                text=entry["label"],
                variable=self.export_vars[entry["key"]],
                font=("Arial", 10),
                anchor="w",
                command=self._update_bin_state,
            ).pack(fill="x", pady=1)

        # ── Bin size radio buttons ─────────────────────────────────────
        bin_frame = tk.Frame(frm)
        bin_frame.pack(fill="x", padx=6, pady=(4, 6))

        tk.Label(bin_frame, text="Bin Size:", font=("Arial", 10)).pack(
            side="left", padx=(0, 8)
        )
        self._bin_radios = []
        for size in BIN_SIZES_M:
            rb = tk.Radiobutton(
                bin_frame,
                text=f"{size} m",
                variable=self.bin_size_var,
                value=size,
                font=("Arial", 10),
            )
            rb.pack(side="left", padx=6)
            self._bin_radios.append(rb)

        self._update_bin_state()

    def _update_bin_state(self):
        """Enable bin-size radios only when at least one polyline export is checked."""
        any_bins = any(
            e["needs_bins"] and self.export_vars[e["key"]].get()
            for e in EXPORT_REGISTRY
        )
        state = "normal" if any_bins else "disabled"
        for rb in self._bin_radios:
            rb.config(state=state)

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _scan(self):
        folder = self.input_folder.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Please select an input directory first.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Invalid Folder", f"Directory not found:\n{folder}")
            return

        self.scan_results = scan_directory(folder)
        self.include_vars = [tk.BooleanVar(value=r["valid"]) for r in self.scan_results]
        self._populate_tree()

        total   = len(self.scan_results)
        valid   = sum(1 for r in self.scan_results if r["valid"])
        invalid = total - valid
        self._valid_count_label.config(
            text=f"{total} .shp file(s) found  |  {valid} valid  |  {invalid} invalid"
        )
        self._log(f"Scanned '{folder}': {total} shapefile(s) found ({valid} valid, {invalid} invalid).")

        if invalid:
            self._log(
                "WARNING: Files highlighted in red do not match the DIVEXXX_*.shp naming "
                "convention and are excluded by default."
            )

    def _populate_tree(self):
        """Refresh the treeview from self.scan_results / self.include_vars."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        for i, (rec, var) in enumerate(zip(self.scan_results, self.include_vars)):
            include_sym = "[x]" if var.get() else "[ ]"
            tag = "valid" if rec["valid"] else "invalid"
            dive_text = rec["dive_label"] if rec["valid"] else "—"
            status    = "Valid" if rec["valid"] else "No Match"
            self.tree.insert(
                "",
                "end",
                iid=str(i),
                values=(include_sym, rec["filename"], dive_text, status),
                tags=(tag,),
            )

    # ------------------------------------------------------------------
    # Tree interaction
    # ------------------------------------------------------------------

    def _on_tree_click(self, event):
        """Toggle the Include checkbox when the Include column is clicked."""
        col = self.tree.identify_column(event.x)
        row = self.tree.identify_row(event.y)
        if not row:
            return
        # Column #1 is "Include"
        if col == "#1":
            idx = int(row)
            self.include_vars[idx].set(not self.include_vars[idx].get())
            self._populate_tree()  # Refresh checkmarks

    def _select_all(self):
        for v in self.include_vars:
            v.set(True)
        self._populate_tree()

    def _select_valid(self):
        for v, r in zip(self.include_vars, self.scan_results):
            v.set(r["valid"])
        self._populate_tree()

    def _deselect_all(self):
        for v in self.include_vars:
            v.set(False)
        self._populate_tree()

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(self):
        if not GEOPANDAS_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "geopandas is not installed.\n\nInstall it with:\n  pip install geopandas",
            )
            return

        # ── Validate inputs ────────────────────────────────────────────
        mission = self.mission_code.get().strip()
        if not mission:
            messagebox.showwarning("Mission Code Required", "Please enter a mission code before merging.")
            return

        output_dir = self.output_folder.get().strip()
        if not output_dir:
            messagebox.showwarning("Output Directory Required", "Please select an output directory.")
            return
        if not os.path.isdir(output_dir):
            messagebox.showerror("Invalid Output Directory", f"Directory not found:\n{output_dir}")
            return

        selected = [
            rec for rec, var in zip(self.scan_results, self.include_vars) if var.get()
        ]
        if not selected:
            messagebox.showwarning("Nothing Selected", "No shapefiles are selected for merging.")
            return

        # Require at least one export type to be checked
        any_export = any(self.export_vars[e["key"]].get() for e in EXPORT_REGISTRY)
        if not any_export:
            messagebox.showwarning("No Exports Selected", "Please select at least one export type.")
            return

        bin_size_m = self.bin_size_var.get()

        # ── Confirm ────────────────────────────────────────────────────
        needs_bins = any(
            e["needs_bins"] and self.export_vars[e["key"]].get() for e in EXPORT_REGISTRY
        )
        bin_note = f"Bin Size     : {bin_size_m} m\n" if needs_bins else ""
        msg = (
            f"Merge {len(selected)} shapefile(s)?\n\n"
            f"Mission Code : {mission}\n"
            f"Output Dir   : {output_dir}\n"
            f"{bin_note}"
            "\nContinue?"
        )
        if not messagebox.askyesno("Confirm Merge", msg):
            return

        # ── Disable button while running ───────────────────────────────
        self._merge_btn.config(state="disabled", text="Merging…")
        self.root.update_idletasks()

        try:
            self._run_merge(selected, mission, output_dir, bin_size_m)
        finally:
            self._merge_btn.config(state="normal", text="Merge Shapefiles")

    def _run_merge(self, selected: List[dict], mission: str, output_dir: str, bin_size_m: int):
        """
        Orchestrate all selected exports.
          Step 1 — read every selected shapefile once and annotate with mission/dive.
          Step 2 — dispatch to per-export-type handlers.
        """
        # ── Step 1: Read all selected shapefiles ──────────────────────
        self._log("─── Reading shapefiles ───────────────────────────────────")
        gdfs_by_dive: List[tuple] = []   # (ext_rec, annotated_gdf)
        crs_ref = None

        for rec in selected:
            self._log(f"  Reading {rec['filename']} …")
            self.root.update_idletasks()
            try:
                gdf = gpd.read_file(rec["filepath"])
            except Exception as exc:
                self._log(f"  ERROR reading {rec['filename']}: {exc}")
                continue

            if crs_ref is None:
                crs_ref = gdf.crs
            elif gdf.crs != crs_ref:
                self._log(
                    f"  WARNING: CRS mismatch in {rec['filename']} "
                    f"({gdf.crs} vs {crs_ref}) — reprojecting."
                )
                try:
                    gdf = gdf.to_crs(crs_ref)
                except Exception as exc:
                    self._log(f"  ERROR reprojecting {rec['filename']}: {exc}  Skipping.")
                    continue

            gdf.insert(0, "dive_num",   rec["dive_num"])
            gdf.insert(0, "mission_cd", mission)
            ext_rec = {**rec, "mission_cd": mission}

            gdfs_by_dive.append((ext_rec, gdf))
            self._log(f"    → {len(gdf)} feature(s), dive {rec['dive_label']}")

        if not gdfs_by_dive:
            messagebox.showerror("Read Failed", "No shapefiles could be read successfully.")
            return

        safe_mission  = re.sub(r"[^\w\-]", "_", mission)
        outputs_written: List[str] = []

        # ── Step 2: Dispatch to each selected export type ─────────────
        for entry in EXPORT_REGISTRY:
            if not self.export_vars[entry["key"]].get():
                continue
            if entry["processor"] is None:
                # Built-in merged-points export
                path = self._export_merged_points(gdfs_by_dive, safe_mission, output_dir, crs_ref)
            else:
                path = self._export_polyline(entry, gdfs_by_dive, safe_mission, output_dir, bin_size_m)
            if path:
                outputs_written.append(path)

        if outputs_written:
            summary = "\n".join(f"  \u2022 {p}" for p in outputs_written)
            self._log(f"\n\u2713 All done!  {len(outputs_written)} file(s) written:\n{summary}")
            messagebox.showinfo(
                "Export Complete",
                f"{len(outputs_written)} file(s) written:\n\n" + "\n".join(outputs_written),
            )
        else:
            messagebox.showwarning("Nothing Written", "No output files were produced.")

    def _export_merged_points(
        self,
        gdfs_by_dive: List[tuple],
        safe_mission: str,
        output_dir: str,
        crs_ref,
    ) -> Optional[str]:
        """Concatenate all per-dive GeoDataFrames into one point shapefile."""
        self._log("─── Merged Points export ─────────────────────────────────")
        all_gdfs = [gdf for _, gdf in gdfs_by_dive]
        merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=crs_ref)
        self._log(f"  {len(merged)} total features from {len(all_gdfs)} dive(s).")

        out_name = f"{safe_mission}_Merged_Points.shp"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            if not messagebox.askyesno("File Exists", f"Overwrite?\n{out_path}"):
                self._log("  Skipped (file exists).")
                return None

        try:
            merged.to_file(out_path)
            self._log(f"  Saved: {out_name}")
            return out_path
        except Exception as exc:
            self._log(f"  ERROR writing {out_name}: {exc}")
            messagebox.showerror("Write Error", str(exc))
            return None

    def _export_polyline(
        self,
        entry: dict,
        gdfs_by_dive: List[tuple],
        safe_mission: str,
        output_dir: str,
        bin_size_m: int,
    ) -> Optional[str]:
        """Run a registered polyline processor over all dives and write one shapefile."""
        label = entry["label"].split("(")[0].strip()
        self._log(f"─── {label} export ({bin_size_m} m bins) ──────────────────")
        bin_gdfs = []

        for ext_rec, gdf in gdfs_by_dive:
            self.root.update_idletasks()
            try:
                result = entry["processor"](ext_rec, gdf, bin_size_m)
            except Exception as exc:
                self._log(f"  ERROR processing {ext_rec['dive_label']}: {exc}")
                continue
            if result is not None and not result.empty:
                bin_gdfs.append(result)
                self._log(f"  {ext_rec['dive_label']}: {len(result)} bin(s)")
            else:
                self._log(f"  {ext_rec['dive_label']}: skipped (insufficient data)")

        if not bin_gdfs:
            self._log(f"  No data produced — skipping {entry['suffix']}.")
            return None

        combined = gpd.GeoDataFrame(
            pd.concat(bin_gdfs, ignore_index=True),
            crs=gdfs_by_dive[0][1].crs,
        )
        out_name = f"{safe_mission}_{entry['suffix']}.shp"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            if not messagebox.askyesno("File Exists", f"Overwrite?\n{out_path}"):
                self._log("  Skipped (file exists).")
                return None

        try:
            combined.to_file(out_path)
            self._log(f"  Saved: {out_name}  ({len(combined)} features)")
            return out_path
        except Exception as exc:
            self._log(f"  ERROR writing {out_name}: {exc}")
            messagebox.showerror("Write Error", str(exc))
            return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str):
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.config(state="disabled")
        print(message)

    # ------------------------------------------------------------------
    # Icon
    # ------------------------------------------------------------------

    def _set_icon(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(current_dir, "NOAA_VOYIS_Logo.ico")
            if os.path.exists(logo_path):
                self.root.iconbitmap(logo_path)
        except Exception:
            pass


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    if not GEOPANDAS_AVAILABLE:
        # Show a minimal Tk error and exit rather than crashing silently
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Dependency",
            "geopandas is required but is not installed.\n\n"
            "Install it with:\n  pip install geopandas",
        )
        root.destroy()
        sys.exit(1)

    root = tk.Tk()
    app = ShapefileMergerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
