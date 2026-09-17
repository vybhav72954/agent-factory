"""
research/pronostia/features.py

Per-snapshot condition indicators for all 17 PRONOSTIA run-to-failure bearings.

Each accelerometer snapshot is 0.1 s (2560 samples at 25.6 kHz) recorded every
10 s, with horizontal and vertical channels. Temperature (PT100, 10 Hz) is
recorded once a minute for 9 of the 17 bearings, all in Full_Test_Set.

Features per snapshot:
    h_rms, v_rms   RMS acceleration (g), horizontal / vertical
    hf_db          mean spectral power 5.0–12.8 kHz over both axes, in dB
                   (high-frequency vibration energy; the band acoustic sensors capture)
    temp_c         bearing temperature (°C), forward-filled from the latest
                   temperature file; NaN for bearings without temperature data
    speed_rpm, load_n   operating condition (constant per bearing)
    life_frac      snapshot index / (n_snapshots - 1); 1.0 at failure
    rul_pct        100 * (1 - life_frac): percent of life remaining

Operating conditions (PHM 2012 challenge definition):
    Bearing1_x: 1800 rpm, 4000 N    Bearing2_x: 1650 rpm, 4200 N    Bearing3_x: 1500 rpm, 5000 N

The 11 test bearings are read from Full_Test_Set (complete run-to-failure),
not the truncated Test_set.

Usage:
    python -m research.pronostia.features

Output:
    research/pronostia/data/features.csv.gz
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "research" / "pronostia" / "data" / "raw"
FEATURES_PATH = PROJECT_ROOT / "research" / "pronostia" / "data" / "features.csv.gz"

SAMPLE_RATE_HZ = 25_600
HF_BAND_HZ = (5_000.0, 12_800.0)
CONDITIONS = {"1": (1800.0, 4000.0), "2": (1650.0, 4200.0), "3": (1500.0, 5000.0)}


def bearing_dirs() -> list[Path]:
    """All 17 complete run-to-failure bearing directories."""
    dirs = sorted((RAW_DIR / "Learning_set").glob("Bearing*")) + sorted((RAW_DIR / "Full_Test_Set").glob("Bearing*"))
    if len(dirs) != 17:
        raise FileNotFoundError(f"expected 17 bearing folders under {RAW_DIR}, found {len(dirs)}")
    return dirs


def _read_rows(path: Path) -> np.ndarray:
    """Read a PRONOSTIA CSV. Some files use ';' instead of ','."""
    with path.open("r") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep, header=None, engine="c").to_numpy(dtype=np.float64)


def _elapsed_seconds(hms: np.ndarray) -> np.ndarray:
    """Convert (hour, minute, second) rows to monotonic seconds, handling midnight rollover."""
    secs = hms[:, 0] * 3600 + hms[:, 1] * 60 + hms[:, 2]
    rollovers = np.concatenate([[0], np.cumsum(np.diff(secs) < -43_200)])
    return secs + 86_400 * rollovers


def _snapshot_features(rows: np.ndarray) -> tuple[float, float, float]:
    """h_rms, v_rms, hf_db for one accelerometer snapshot."""
    h, v = rows[:, 4], rows[:, 5]
    h_rms = float(np.sqrt(np.mean(h ** 2)))
    v_rms = float(np.sqrt(np.mean(v ** 2)))
    freqs = np.fft.rfftfreq(len(h), d=1.0 / SAMPLE_RATE_HZ)
    band = (freqs >= HF_BAND_HZ[0]) & (freqs <= HF_BAND_HZ[1])
    power = (np.abs(np.fft.rfft(h - h.mean())) ** 2 + np.abs(np.fft.rfft(v - v.mean())) ** 2)[band]
    hf_db = float(10.0 * np.log10(power.mean() + 1e-12))
    return h_rms, v_rms, hf_db


def extract_bearing(bdir: Path) -> pd.DataFrame:
    """Feature rows for one bearing, one row per accelerometer snapshot."""
    name = bdir.name                                   # e.g. Bearing1_4
    speed, load = CONDITIONS[name[len("Bearing")]]
    acc_files = sorted(bdir.glob("acc_*.csv"))
    temp_files = sorted(bdir.glob("temp_*.csv"))

    acc_hms, feats = [], []
    for f in acc_files:
        rows = _read_rows(f)
        acc_hms.append(rows[0, :3])
        feats.append(_snapshot_features(rows))
    acc_t = _elapsed_seconds(np.array(acc_hms))

    temp = np.full(len(acc_files), np.nan)
    if temp_files:
        t_hms, t_val = [], []
        for f in temp_files:
            rows = _read_rows(f)
            t_hms.append(rows[0, :3])
            t_val.append(float(rows[:, 4].mean()))
        temp_t = _elapsed_seconds(np.array(t_hms))
        # Latest temperature reading at or before each snapshot; back-fill the start.
        idx = np.searchsorted(temp_t, acc_t, side="right") - 1
        temp = np.array(t_val)[np.clip(idx, 0, len(t_val) - 1)]

    n = len(acc_files)
    life_frac = np.arange(n) / (n - 1)
    df = pd.DataFrame(feats, columns=["h_rms", "v_rms", "hf_db"])
    df.insert(0, "snapshot", np.arange(n))
    df.insert(0, "bearing", name)
    df["elapsed_s"] = acc_t - acc_t[0]
    df["temp_c"] = temp
    df["speed_rpm"] = speed
    df["load_n"] = load
    df["life_frac"] = life_frac
    df["rul_pct"] = 100.0 * (1.0 - life_frac)
    df["has_temp"] = bool(temp_files)
    return df


def main() -> None:
    dirs = bearing_dirs()
    workers = max(1, min(len(dirs), (os.cpu_count() or 2) - 1))
    print(f"[features] {len(dirs)} bearings, {workers} worker processes")
    t0 = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for df in pool.map(extract_bearing, dirs):
            frames.append(df)
            print(f"  {df['bearing'].iloc[0]:11s} {len(df):5d} snapshots  "
                  f"life {df['elapsed_s'].iloc[-1] / 3600:4.1f} h  "
                  f"h_rms {df['h_rms'].iloc[:50].median():.2f} -> {df['h_rms'].iloc[-1]:.2f} g  "
                  f"temp {'yes' if df['has_temp'].iloc[0] else 'no '}  ({time.time() - t0:.0f}s)")
    out = pd.concat(frames, ignore_index=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEATURES_PATH, index=False, compression="gzip")
    print(f"[features] wrote {FEATURES_PATH.relative_to(PROJECT_ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
