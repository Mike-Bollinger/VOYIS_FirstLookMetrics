"""
Parser for PHINS Standard v2 NMEA-style navigation log files (phins_std.log).

Real-world PHINS Standard v2 output from the iXBlue/Exail C7 INS is a sequence
of NMEA-0183-style ASCII sentences (e.g. $HEHDT, $PIXSE,ATITUD, $PIXSE,POSITI)
with embedded null-byte padding between records, rather than a simple
delimited text export. Adapted from Remus_PHINS_StdV2_ToSHP.py.

This is the "last resort" navigation source: it is slower to parse than the
vehicle's exported PHINS_INS.txt/ADCP.txt and the resulting per-sentence rate
is much higher, so the output is downsampled. Only use this when no
PHINS_INS/ADCP/STATE text export can be found in the navigation directory.
"""

import csv
import datetime as dt
import os
import re

import numpy as np
import pandas as pd


def _verify_nmea_checksum(row_str):
    """Validate the trailing *HH checksum on an NMEA-style sentence."""
    if not row_str.startswith("$") or "*" not in row_str:
        return False
    data, checksum_str = row_str[1:].split("*", 1)
    calc = 0
    for ch in data:
        calc ^= ord(ch)
    try:
        expected = int(checksum_str[:2], 16)
    except ValueError:
        return False
    return calc == expected


def _parse_log_date(filename, fallback_date=None):
    """Extract the YYYYMMDD collection date encoded in the filename, falling back to a provided date."""
    base = os.path.basename(filename)
    match = re.search(r"phinsdata_(\d{8})", base)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", base)
    if match:
        return match.group(1).replace("-", "")
    match = re.search(r"(\d{8})", base)
    if match:
        return match.group(1)
    if fallback_date is not None:
        return fallback_date.strftime("%Y%m%d")
    raise ValueError(f"Could not determine collection date for PHINS Std v2 log: {filename}")


def _datetime_to_epoch_seconds(series):
    """Convert a datetime-like Series to float Unix epoch seconds.

    Robust to pandas' inferred datetime64 resolution (ns/us/ms/s), which
    varies depending on the pandas version and the precision of the parsed
    strings - naively assuming nanoseconds silently produces wrong scale.
    """
    dt_series = pd.to_datetime(series, utc=True)
    unit = getattr(dt_series.dt, "unit", "ns")
    divisor = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(unit, 1e9)
    return dt_series.astype("int64").to_numpy(dtype=np.float64) / divisor


def parse_phins_std_log(log_path, fallback_date=None, max_points=50_000, log_fn=None):
    """
    Parse a PHINS Standard v2 (NMEA-style) navigation log into a DataFrame
    indexed by absolute Unix epoch time, with heading/position/depth/altitude.

    Args:
        log_path: Path to the phins_std log file.
        fallback_date: datetime.date used when the collection date cannot be
            parsed from the filename (e.g. plain "phins_std.log").
        max_points: Downsample the result to roughly this many rows.
        log_fn: Optional logging callback.

    Returns:
        DataFrame with columns [phins_time, phins_heading, phins_lat,
        phins_lon, phins_depth, phins_pitch, phins_roll, altitude], or None
        if the file could not be parsed or contains no usable data.
    """
    def log(msg):
        print(msg)
        if log_fn:
            log_fn(msg)

    try:
        date_str = _parse_log_date(log_path, fallback_date)
    except ValueError as e:
        log(f"  \u2717 {e}")
        return None

    log(f"  Parsing PHINS Std v2 log (last resort, this can take a while): {os.path.basename(log_path)}")

    try:
        with open(log_path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        log(f"  \u2717 Could not read PHINS Std v2 log: {e}")
        return None

    log(f"  File size: {len(raw_bytes):,} bytes")

    frames = {
        "DEPIN_": {"Depth": [], "Time_REF": []},
        "ATITUD": {"Roll": [], "Pitch": [], "Time_REF": []},
        "HEHDT": {"Heading": [], "Time_REF": []},
        "POSITI": {"Latitude": [], "Longitude": [], "Altitude": [], "Time_REF": []},
        "LOGDVL": {"DVL_Distance_2btm": [], "Time_REF": []},
    }

    current_time = ""
    time_last = 0.0
    date = date_str
    decoded = raw_bytes.replace(b"\x00", b"").decode("latin-1")

    message_count = 0
    for row in csv.reader(decoded.splitlines()):
        row_str = ",".join(row)
        if not _verify_nmea_checksum(row_str):
            continue
        row = [col.split("*")[0] for col in row]
        if len(row) < 2:
            continue

        identifier = row[0]
        if identifier == "$PIXSE":
            message = row[1]
            if message == "TIME__" and len(row) >= 3:
                try:
                    t = float(row[2])
                    if t == 0:
                        continue
                    if t < time_last and time_last > 235900:
                        day = dt.datetime.strptime(date, "%Y%m%d") + dt.timedelta(days=1)
                        date = day.strftime("%Y%m%d")
                    time_last = t
                    current_time = date + " " + row[2]
                except ValueError:
                    pass
            elif message == "DEPIN_" and len(row) >= 4:
                try:
                    frames["DEPIN_"]["Depth"].append(float(row[2]))
                    frames["DEPIN_"]["Time_REF"].append(current_time)
                except ValueError:
                    pass
            elif message == "ATITUD" and len(row) >= 4:
                try:
                    frames["ATITUD"]["Roll"].append(float(row[2]))
                    frames["ATITUD"]["Pitch"].append(float(row[3]))
                    frames["ATITUD"]["Time_REF"].append(current_time)
                except ValueError:
                    pass
            elif message == "POSITI" and len(row) >= 5:
                try:
                    frames["POSITI"]["Latitude"].append(float(row[2]))
                    frames["POSITI"]["Longitude"].append(float(row[3]))
                    frames["POSITI"]["Altitude"].append(float(row[4]))
                    frames["POSITI"]["Time_REF"].append(current_time)
                except ValueError:
                    pass
            elif message == "LOGDVL" and len(row) >= 5:
                try:
                    sound_velocity = float(row[2])
                    if sound_velocity > 0:
                        frames["LOGDVL"]["DVL_Distance_2btm"].append(float(row[4]))
                        frames["LOGDVL"]["Time_REF"].append(current_time)
                except ValueError:
                    pass
        elif identifier == "$HEHDT" and len(row) >= 2:
            try:
                frames["HEHDT"]["Heading"].append(float(row[1]))
                frames["HEHDT"]["Time_REF"].append(current_time)
            except ValueError:
                pass

        message_count += 1
        if message_count % 200_000 == 0:
            log(f"    Processed {message_count:,} sentences...")

    log(f"  Total valid sentences processed: {message_count:,}")

    for key, data in frames.items():
        df = pd.DataFrame(data)
        if "Time_REF" in df.columns and not df.empty:
            df["Date_Time"] = pd.to_datetime(df["Time_REF"], format="%Y%m%d %H%M%S.%f", errors="coerce", utc=True)
            df = df.drop(columns=["Time_REF"])
        frames[key] = df

    positi = frames["POSITI"].dropna(subset=["Latitude", "Longitude", "Date_Time"]).copy()
    if positi.empty:
        log("  \u2717 No $PIXSE,POSITI sentences with valid position/time found")
        return None

    # PHINS reports west longitudes as 0-360 azimuth; normalize back to -180..180
    positi["Longitude"] = positi["Longitude"].apply(lambda lon: lon - 360 if lon > 180 else lon)
    positi = positi.sort_values("Date_Time").drop_duplicates(subset=["Date_Time"]).reset_index(drop=True)

    target_times = pd.to_datetime(positi["Date_Time"], utc=True)
    target_seconds = _datetime_to_epoch_seconds(target_times)

    def _interp(frame_key, value_col):
        src = frames[frame_key]
        if src.empty or value_col not in src.columns or "Date_Time" not in src.columns:
            return np.full(len(target_seconds), np.nan, dtype=np.float64)
        clean = src[["Date_Time", value_col]].dropna()
        if len(clean) < 2:
            return np.full(len(target_seconds), np.nan, dtype=np.float64)
        src_seconds = _datetime_to_epoch_seconds(clean["Date_Time"])
        src_values = pd.to_numeric(clean[value_col], errors="coerce").to_numpy(dtype=np.float64)
        order = np.argsort(src_seconds)
        return np.interp(target_seconds, src_seconds[order], src_values[order], left=np.nan, right=np.nan)

    out = pd.DataFrame({
        "phins_time": target_seconds,
        "phins_lat": positi["Latitude"].to_numpy(),
        "phins_lon": positi["Longitude"].to_numpy(),
        "phins_depth": _interp("DEPIN_", "Depth"),
        "phins_heading": _interp("HEHDT", "Heading"),
        "phins_pitch": _interp("ATITUD", "Pitch"),
        "phins_roll": _interp("ATITUD", "Roll"),
        "altitude": _interp("LOGDVL", "DVL_Distance_2btm"),
    })

    out = out.dropna(subset=["phins_time", "phins_heading"])
    if out.empty:
        log("  \u2717 No usable heading data parsed from PHINS Std v2 log")
        return None

    if len(out) > max_points:
        step = len(out) // max_points + 1
        out = out.iloc[::step].reset_index(drop=True)
        log(f"  Downsampled to {len(out):,} points (every {step} sample(s))")

    log(f"  \u2713 Parsed {len(out):,} navigation points from PHINS Std v2 log")
    return out
