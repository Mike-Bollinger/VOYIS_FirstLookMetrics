# -*- coding: utf-8 -*-
"""
CSV to Shapefile Converter for VOYIS Image Metrics
===================================================
Standalone utility that converts one or more DIVE###_Image_Metrics.csv files
into point shapefiles (WGS-84 / EPSG:4326).  Each row in the CSV becomes a
Point feature; the output shapefile is written to the same directory as the
source CSV, with the same stem (e.g. DIVE016_Image_Metrics.shp).

Column names are truncated to 10 characters to satisfy the DBF field-name
limit imposed by the shapefile format.

Dependencies: geopandas, pandas, shapely  (all in requirements.txt)

Author: Mike Bollinger, Github Copilot
Date:   May 2026
"""

import os
import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

try:
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    DEPS_OK = True
except ImportError as _dep_err:
    DEPS_OK = False
    _MISSING = str(_dep_err)


# ---------------------------------------------------------------------------
# Helper: build a rename map so every column is ≤10 chars, with no duplicates
# ---------------------------------------------------------------------------
def _make_dbf_rename(columns) -> dict:
    """
    Return a {original: truncated} dict for any column whose name exceeds
    10 characters.  Duplicates after truncation are resolved by appending
    a numeric suffix (_1, _2 …).
    """
    rename: dict[str, str] = {}
    seen: set[str] = set()

    # Reserve the names of short columns first so suffixes don't collide.
    for col in columns:
        if len(col) <= 10:
            seen.add(col)

    for col in columns:
        if len(col) <= 10:
            continue
        base = col[:10]
        candidate = base
        counter = 1
        while candidate in seen:
            suffix = f"_{counter}"
            candidate = base[: 10 - len(suffix)] + suffix
            counter += 1
        seen.add(candidate)
        rename[col] = candidate

    return rename


# ===========================================================================
# Core conversion logic
# ===========================================================================

def csv_to_shp(csv_path: str, log_fn=None) -> tuple[bool, str]:
    """
    Convert a single Image_Metrics CSV to a point shapefile.

    Parameters
    ----------
    csv_path : str
        Absolute path to the CSV file.
    log_fn : callable, optional
        Function accepting a string message for progress reporting.

    Returns
    -------
    (success: bool, message: str)
    """
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False, f"File not found: {csv_path}"

    # ---- Read CSV ----------------------------------------------------------
    log(f"  Reading {csv_path.name} …")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return False, f"Could not read CSV: {exc}"

    # ---- Validate required columns ----------------------------------------
    lat_col = next(
        (c for c in df.columns if c.strip().lower() in ("latitude", "lat")), None
    )
    lon_col = next(
        (c for c in df.columns if c.strip().lower() in ("longitude", "lon", "long")), None
    )
    if lat_col is None or lon_col is None:
        return False, (
            f"CSV must contain latitude and longitude columns. "
            f"Found: {list(df.columns)}"
        )

    # ---- Drop rows with missing coordinates --------------------------------
    before = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    dropped = before - len(df)
    if dropped:
        log(f"  Dropped {dropped} rows with missing lat/lon.")
    if df.empty:
        return False, "No valid coordinate rows remain after dropping NaN lat/lon."

    # ---- Coerce coordinate columns to numeric -----------------------------
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty:
        return False, "No numeric lat/lon values found."

    # ---- Rename columns to ≤10-char DBF names (generic) -------------------
    rename_map = _make_dbf_rename(df.columns)
    if rename_map:
        df = df.rename(columns=rename_map)
        log(f"  Renamed {len(rename_map)} column(s) to fit DBF 10-char limit.")

    # Resolve the (possibly renamed) lat/lon column names
    lat_final = rename_map.get(lat_col, lat_col)
    lon_final = rename_map.get(lon_col, lon_col)

    # ---- Build geometry ---------------------------------------------------
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df[lon_final], df[lat_final])
    ]

    # ---- Create GeoDataFrame and write shapefile --------------------------
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    out_path = csv_path.with_suffix(".shp")
    log(f"  Writing {out_path.name} …")
    try:
        gdf.to_file(str(out_path), driver="ESRI Shapefile")
    except Exception as exc:
        return False, f"Failed to write shapefile: {exc}"

    log(f"  Done — {len(gdf):,} features → {out_path}")
    return True, str(out_path)


# ===========================================================================
# GUI
# ===========================================================================

class CsvToShapefileApp:
    """Simple tkinter front-end for the CSV → shapefile converter."""

    WIN_W, WIN_H = 780, 560

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VOYIS — CSV to Shapefile Converter")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.resizable(True, True)

        self._csv_paths: list[str] = []
        self._build_ui()
        self.root.after(100, self._set_icon)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Top: instructions -----------------------------------------
        info = ttk.Label(
            self.root,
            text=(
                "Add one or more Image_Metrics CSV files below, then click Convert.\n"
                "Each shapefile is saved to the same folder as its source CSV."
            ),
            wraplength=740,
            justify="left",
        )
        info.pack(padx=12, pady=(10, 4), anchor="w")

        # ---- Middle: file list + buttons -------------------------------
        list_frame = ttk.LabelFrame(self.root, text="Selected CSV Files")
        list_frame.pack(fill="both", expand=True, padx=12, pady=4)

        # Scrollable listbox
        sb = ttk.Scrollbar(list_frame, orient="vertical")
        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=sb.set,
            selectmode="extended",
            activestyle="dotbox",
            font=("Consolas", 9),
        )
        sb.config(command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Buttons alongside the list
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=12, pady=(0, 4))

        ttk.Button(btn_frame, text="Add CSVs…",       command=self._add_csvs).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Clear All",       command=self._clear_all).pack(side="left")

        self.convert_btn = ttk.Button(
            btn_frame, text="Convert to Shapefiles", command=self._run_conversion
        )
        self.convert_btn.pack(side="right")

        # ---- Bottom: log output ----------------------------------------
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=False, padx=12, pady=(0, 10))

        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=9, state="disabled",
            font=("Consolas", 9), wrap="word",
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _add_csvs(self):
        paths = filedialog.askopenfilenames(
            title="Select Image Metrics CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        added = 0
        for p in paths:
            if p not in self._csv_paths:
                self._csv_paths.append(p)
                self.listbox.insert("end", p)
                added += 1
        if added:
            self._log(f"Added {added} file(s).")

    def _remove_selected(self):
        selected = list(self.listbox.curselection())
        for idx in reversed(selected):
            self._csv_paths.pop(idx)
            self.listbox.delete(idx)
        if selected:
            self._log(f"Removed {len(selected)} file(s).")

    def _clear_all(self):
        self._csv_paths.clear()
        self.listbox.delete(0, "end")
        self._log("Cleared file list.")

    def _run_conversion(self):
        if not DEPS_OK:
            messagebox.showerror(
                "Missing dependency",
                f"Required package not found:\n{_MISSING}\n\n"
                "Install with:  pip install geopandas pandas shapely",
            )
            return

        if not self._csv_paths:
            messagebox.showwarning("No files", "Please add at least one CSV file.")
            return

        self.convert_btn.config(state="disabled")
        self._log("=" * 60)
        self._log(f"Starting conversion of {len(self._csv_paths)} file(s)…")

        ok_count = 0
        fail_count = 0
        for csv_path in self._csv_paths:
            self._log(f"\n[{Path(csv_path).name}]")
            try:
                success, msg = csv_to_shp(csv_path, log_fn=self._log)
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    self._log(f"  ERROR: {msg}")
            except Exception:
                fail_count += 1
                self._log(f"  UNEXPECTED ERROR:\n{traceback.format_exc()}")

        self._log(
            f"\nFinished — {ok_count} succeeded, {fail_count} failed."
        )
        self._log("=" * 60)
        self.convert_btn.config(state="normal")

        if fail_count == 0:
            messagebox.showinfo(
                "Done",
                f"All {ok_count} shapefile(s) created successfully.",
            )
        else:
            messagebox.showwarning(
                "Completed with errors",
                f"{ok_count} succeeded, {fail_count} failed.\nSee the log for details.",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")
        self.root.update_idletasks()

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
    if not DEPS_OK:
        # Try to show a basic error without the full GUI
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Missing dependency",
                f"Required package not found:\n{_MISSING}\n\n"
                "Install with:  pip install geopandas pandas shapely",
            )
            root.destroy()
        except Exception:
            print(f"ERROR: Missing dependency — {_MISSING}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    app = CsvToShapefileApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
