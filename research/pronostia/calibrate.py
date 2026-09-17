"""
research/pronostia/calibrate.py

Calibrate the factory simulator's bearing degradation against PRONOSTIA.

The simulator (research/simulator/factory_simulator.py) makes every sensor a
LINEAR function of component wear, and wear grows linearly in time, so bearing
vibration and temperature rise in a straight line from new to failure. Real
rolling-element bearings do not: vibration stays near its healthy level for
most of the life and rises sharply near the end.

This script:
  1. Builds normalised health-indicator curves over life fraction for the 17
     PRONOSTIA bearings (horizontal vibration RMS) and the 9 with temperature.
  2. Builds the same curves for the original simulator, aging ONLY the bearing
     (PRONOSTIA isolates one bearing; in the full simulator, belt and spindle
     wear also raise vibration linearly and would mask the bearing's shape).
  3. Fits a calibrated bearing law:
        sensor = baseline + slope * g_k(wear),  g_k(w) = (exp(k w) - 1) / (exp(k) - 1)
     with k fitted to the PRONOSTIA median curve (separately for vibration and
     temperature; k > 0 is late onset, k < 0 is early rise), vibration slope set from the real failure/healthy RMS ratio,
     temperature slope from the real temperature rise, and bearing lifetime
     spread (lognormal wear-rate factor) matched to the real lifetime CV.
  4. Reports fidelity metrics for PRONOSTIA, the original simulator and the
     calibrated simulator.

Health-indicator normalisation (per trajectory, after an 11-point rolling median):
    HI_n(f) = (HI(f) - median(first 10% of life)) / (median(last 1% of life) - median(first 10%))

Fidelity metrics:
    onset_frac   first life fraction where HI_n >= 0.2
    auc          area under HI_n over life fraction (linear rise = 0.5; late onset -> small)
    ratio        failure / healthy vibration RMS
    delta_temp   failure - healthy temperature (°C)
    curve_rmse   RMSE of the median HI_n curve against the PRONOSTIA median curve

Units differ (PRONOSTIA vibration is acceleration in g, the simulator's is
velocity in mm/s), so the calibration matches dimensionless shape and the
failure/healthy ratio, not absolute levels.

Usage:
    python -m research.pronostia.calibrate [--n-lifecycles 40]

Output:
    research/pronostia/calibrated_bearing_law.json
    research/results/pronostia/calibration_summary.md
    research/results/pronostia/calibration_curves.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.pronostia.features import FEATURES_PATH
from research.simulator.factory_simulator import (
    BASE_WEAR_RATES,
    COMPONENT_SENSOR_EFFECTS,
    COMPONENTS,
    SENSOR_BASELINES,
    SENSOR_NAMES,
    FactorySimulator,
)

OUT_DIR = PROJECT_ROOT / "research" / "results" / "pronostia"
LAW_PATH = PROJECT_ROOT / "research" / "pronostia" / "calibrated_bearing_law.json"

GRID = np.linspace(0.0, 1.0, 101)
ROLL = 11
ONSET_LEVEL = 0.2
SIM_NOISE_SCALE = 0.5            # same noise as research/simulator/generate_training_data.py
CALIBRATED_SENSORS = {"Vibration_X": "vib", "Vibration_Y": "vib", "Acoustic_dB": "vib", "Bearing_Temp": "temp"}


# ─────────────────────────────────────────────────────────────────────────────
# Health-indicator curves
# ─────────────────────────────────────────────────────────────────────────────

def normalised_curve(values: np.ndarray) -> np.ndarray:
    """Smoothed, normalised health indicator resampled onto GRID (life fraction 0..1)."""
    s = pd.Series(values).rolling(ROLL, center=True, min_periods=1).median().to_numpy()
    n = len(s)
    healthy = np.median(s[: max(3, n // 10)])
    failed = np.median(s[-max(3, n // 100):])
    span = failed - healthy
    hi = (s - healthy) / span if abs(span) > 1e-12 else np.zeros_like(s)
    return np.interp(GRID, np.linspace(0.0, 1.0, n), hi)


def curve_metrics(curve: np.ndarray) -> dict:
    above = np.nonzero(curve >= ONSET_LEVEL)[0]
    return {
        "onset_frac": float(GRID[above[0]]) if len(above) else 1.0,
        "auc": float(np.trapezoid(np.clip(curve, 0, None), GRID)),
    }


def healthy_failed(values: np.ndarray) -> tuple[float, float]:
    s = pd.Series(values).rolling(ROLL, center=True, min_periods=1).median().to_numpy()
    n = len(s)
    return float(np.median(s[: max(3, n // 10)])), float(np.median(s[-max(3, n // 100):]))


def pronostia_curves(features: pd.DataFrame) -> dict:
    vib_curves, temp_curves, rows = [], [], []
    for bearing, g in features.groupby("bearing", sort=True):
        g = g.sort_values("snapshot")
        vib = g["h_rms"].to_numpy()
        curve = normalised_curve(vib)
        vib_curves.append(curve)
        healthy, failed = healthy_failed(vib)
        row = {"trajectory": bearing, "life_h": float(g["elapsed_s"].iloc[-1] / 3600),
               "condition": bearing[len("Bearing")], "ratio": failed / healthy, **curve_metrics(curve)}
        if g["has_temp"].iloc[0]:
            temp = g["temp_c"].to_numpy()
            tcurve = normalised_curve(temp)
            temp_curves.append(tcurve)
            t_h, t_f = healthy_failed(temp)
            row.update({"delta_temp": t_f - t_h, "temp_auc": curve_metrics(tcurve)["auc"]})
        rows.append(row)
    return {"vib": np.array(vib_curves), "temp": np.array(temp_curves), "table": pd.DataFrame(rows)}


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated simulator
# ─────────────────────────────────────────────────────────────────────────────

def g_k(w: np.ndarray | float, k: float):
    """Late-onset shape: 0 at w=0, 1 at w=1; k -> 0 recovers the linear law."""
    if abs(k) < 1e-6:
        return w
    return (np.exp(k * w) - 1.0) / (np.exp(k) - 1.0)


class CalibratedFactorySimulator(FactorySimulator):
    """FactorySimulator with the PRONOSTIA-calibrated bearing law.

    Only the bearing component's effect on the calibrated sensors changes;
    every other component and sensor keeps the original linear law. The
    bearing base wear rate is scaled by a per-simulator lognormal factor to
    reproduce real lifetime spread.
    """

    def __init__(self, law: dict, n_machines: int = 5, seed: int | None = None, noise_scale: float = 0.05,
                 bearing_only: bool = False):
        super().__init__(n_machines=n_machines, seed=seed, noise_scale=noise_scale)
        self.law = law
        self.bearing_only = bearing_only
        sigma = law["lifetime_lognormal_sigma"]
        self.bearing_rate_factor = float(self.rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma))

    def _compute_sensors(self, machine) -> np.ndarray:
        readings = {name: SENSOR_BASELINES[name] for name in SENSOR_NAMES}
        for comp, wear in machine.wears.items():
            for sensor_name, slope in COMPONENT_SENSOR_EFFECTS[comp]:
                if comp == "bearing" and sensor_name in CALIBRATED_SENSORS:
                    kind = CALIBRATED_SENSORS[sensor_name]
                    readings[sensor_name] += self.law["slopes"][sensor_name] * g_k(wear, self.law[f"k_{kind}"])
                else:
                    readings[sensor_name] += slope * wear
        noise_amp = self.noise_scale * (1.0 + machine.max_wear())
        arr = np.array([readings[n] for n in SENSOR_NAMES], dtype=np.float32)
        noise = self.rng.normal(0, 1, size=18).astype(np.float32) * noise_amp * \
            np.array([abs(SENSOR_BASELINES[n]) * 0.02 + 0.01 for n in SENSOR_NAMES], dtype=np.float32)
        return arr + noise

    def step(self):
        for m in self.machines:
            for comp in COMPONENTS:
                if comp == "bearing":
                    base = BASE_WEAR_RATES[comp] * self.bearing_rate_factor
                else:
                    base = 0.0 if self.bearing_only else BASE_WEAR_RATES[comp]
                m.wears[comp] = min(1.0, m.wears[comp] + base + m.fault_pressures[comp])
                if m.fault_durations[comp] > 0:
                    m.fault_durations[comp] -= 1
                    if m.fault_durations[comp] == 0:
                        m.fault_pressures[comp] = 0.0
            m.history.append(self._compute_sensors(m))
            if len(m.history) > 200:
                m.history = m.history[-200:]
        self.t += 1


def simulate_bearing_lives(make_sim, n: int, seed0: int) -> dict:
    """Natural aging to bearing failure; returns normalised curves + per-life metrics."""
    ix_vib = SENSOR_NAMES.index("Vibration_X")
    ix_temp = SENSOR_NAMES.index("Bearing_Temp")
    vib_curves, temp_curves, rows = [], [], []
    for i in range(n):
        sim = make_sim(seed0 + i)
        vib, temp = [], []
        while sim.machines[0].wears["bearing"] < 1.0 and sim.t < 20_000:
            sim.step()
            reading = sim.machines[0].history[-1]
            vib.append(reading[ix_vib])
            temp.append(reading[ix_temp])
        vib, temp = np.array(vib), np.array(temp)
        vc, tc = normalised_curve(vib), normalised_curve(temp)
        vib_curves.append(vc)
        temp_curves.append(tc)
        v_h, v_f = healthy_failed(vib)
        t_h, t_f = healthy_failed(temp)
        rows.append({"trajectory": f"sim_{i}", "life_ticks": sim.t, "ratio": v_f / v_h,
                     "delta_temp": t_f - t_h, "temp_auc": curve_metrics(tc)["auc"], **curve_metrics(vc)})
    return {"vib": np.array(vib_curves), "temp": np.array(temp_curves), "table": pd.DataFrame(rows)}


def fit_k(target_curve: np.ndarray) -> float:
    """Least-squares k for g_k(life fraction) against a median normalised curve."""
    ks = np.linspace(-60.0, 120.0, 18001)
    errors = [np.mean((g_k(GRID, k) - target_curve) ** 2) for k in ks]
    return float(ks[int(np.argmin(errors))])


def lifetime_sigma(table: pd.DataFrame) -> float:
    """Pooled within-condition lognormal sigma of bearing lifetimes."""
    logs = []
    for _, g in table.groupby("condition"):
        if len(g) > 1:
            lg = np.log(g["life_h"].to_numpy())
            logs.append(lg - lg.mean())
    resid = np.concatenate(logs)
    return float(np.sqrt(np.sum(resid ** 2) / (len(resid) - len(logs))))


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _summ(table: pd.DataFrame, col: str) -> str:
    v = table[col].dropna()
    if v.empty:
        return "–"
    return f"{v.median():.2f} [{v.quantile(0.25):.2f}, {v.quantile(0.75):.2f}]"


def write_summary(real: dict, orig: dict, cal: dict, law: dict, n_life: int) -> Path:
    real_med_v, real_med_t = np.median(real["vib"], axis=0), np.median(real["temp"], axis=0)

    def rmse(curves, target):
        return float(np.sqrt(np.mean((np.median(curves, axis=0) - target) ** 2)))

    md = ["# Simulator calibration against PRONOSTIA\n"]
    md.append(f"**Generated by:** `python -m research.pronostia.calibrate --n-lifecycles {n_life}`\n")
    md.append(f"PRONOSTIA: {len(real['table'])} run-to-failure bearings "
              f"({len(real['temp'])} with temperature). Simulator: {n_life} lives to bearing failure with only the "
              f"bearing aging (noise_scale {SIM_NOISE_SCALE}), matching the single-bearing PRONOSTIA rig.\n")
    md.append("Values are median [IQR] across trajectories. Curves are normalised health indicators over life "
              "fraction (0 = new, 1 = failure).\n")
    md.append("| Metric | PRONOSTIA (real) | Original simulator | Calibrated simulator |")
    md.append("|---|---|---|---|")
    md.append(f"| Vibration onset (life fraction where HI ≥ {ONSET_LEVEL}) | {_summ(real['table'], 'onset_frac')} | "
              f"{_summ(orig['table'], 'onset_frac')} | {_summ(cal['table'], 'onset_frac')} |")
    md.append(f"| Vibration curve area (linear = 0.50) | {_summ(real['table'], 'auc')} | "
              f"{_summ(orig['table'], 'auc')} | {_summ(cal['table'], 'auc')} |")
    md.append(f"| Vibration failure/healthy ratio | {_summ(real['table'], 'ratio')} | "
              f"{_summ(orig['table'], 'ratio')} | {_summ(cal['table'], 'ratio')} |")
    md.append(f"| Vibration curve RMSE vs PRONOSTIA median | 0 | {rmse(orig['vib'], real_med_v):.3f} | "
              f"{rmse(cal['vib'], real_med_v):.3f} |")
    md.append(f"| Temperature rise to failure (°C) | {_summ(real['table'], 'delta_temp')} | "
              f"{_summ(orig['table'], 'delta_temp')} | {_summ(cal['table'], 'delta_temp')} |")
    md.append(f"| Temperature curve area (linear = 0.50) | {_summ(real['table'], 'temp_auc')} | "
              f"{_summ(orig['table'], 'temp_auc')} | {_summ(cal['table'], 'temp_auc')} |")
    md.append(f"| Temperature curve RMSE vs PRONOSTIA median | 0 | {rmse(orig['temp'], real_med_t):.3f} | "
              f"{rmse(cal['temp'], real_med_t):.3f} |")
    cv = lambda s: float(np.std(s) / np.mean(s))
    md.append(f"| Lifetime coefficient of variation | {cv(real['table']['life_h']):.2f} (all conditions pooled) | "
              f"{cv(orig['table']['life_ticks']):.2f} | {cv(cal['table']['life_ticks']):.2f} |")

    md.append("\n## Fitted bearing law\n")
    md.append("`sensor = baseline + slope * g_k(bearing_wear)`, `g_k(w) = (exp(k w) - 1) / (exp(k) - 1)`\n")
    md.append(f"- Vibration shape k = **{law['k_vib']:.2f}** (applied to Vibration_X, Vibration_Y, Acoustic_dB)")
    md.append(f"- Temperature shape k = **{law['k_temp']:.2f}** (Bearing_Temp)")
    md.append("- Slopes: " + ", ".join(f"{s} {v:+.2f}" for s, v in law["slopes"].items()))
    md.append(f"- Bearing lifetime lognormal sigma = {law['lifetime_lognormal_sigma']:.3f} "
              "(pooled within-condition PRONOSTIA lifetimes)")
    md.append("\n## Per-bearing PRONOSTIA metrics\n")
    md.append(real["table"].round(3).to_markdown(index=False))
    md.append("\n## Caveats\n")
    md.append("- Units differ between PRONOSTIA (acceleration, g) and the simulator (velocity, mm/s); "
              "only dimensionless shape and failure/healthy ratio are calibrated.")
    md.append("- PRONOSTIA is an accelerated-degradation rig with a single bearing; the simulator's other five "
              "components (motor, oil seal, coolant seal, belt, spindle) have no counterpart and stay uncalibrated.")
    md.append("- PRONOSTIA temperature rises early in each test and then levels off (curve area > 0.5). Much of "
              "that early rise is likely the rig warming up rather than bearing wear, so read the fitted "
              "temperature shape with that in mind.")
    md.append("- A fitted k at the edge of the search range [-60, 120] means the real curve is steeper than g_k "
              "can express; the metrics table shows how close the calibrated curve still gets.")
    md.append("- The CNN-LSTM simulator checkpoint was trained on the ORIGINAL simulator. This script does not "
              "retrain it; retraining on the calibrated simulator is a separate step.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "calibration_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")

    curves = pd.DataFrame({
        "life_frac": GRID,
        "pronostia_vib_median": real_med_v, "pronostia_temp_median": real_med_t,
        "original_vib_median": np.median(orig["vib"], axis=0), "original_temp_median": np.median(orig["temp"], axis=0),
        "calibrated_vib_median": np.median(cal["vib"], axis=0), "calibrated_temp_median": np.median(cal["temp"], axis=0),
    })
    curves.to_csv(OUT_DIR / "calibration_curves.csv", index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-lifecycles", type=int, default=40)
    args = parser.parse_args()

    features = pd.read_csv(FEATURES_PATH)
    real = pronostia_curves(features)
    print(f"[calibrate] PRONOSTIA: {len(real['vib'])} vibration curves, {len(real['temp'])} temperature curves")

    # The original simulator's bearing law as an identity law: linear, original slopes, no lifetime spread.
    original_law = {"k_vib": 0.0, "k_temp": 0.0, "lifetime_lognormal_sigma": 0.0,
                    "slopes": {s: v for s, v in COMPONENT_SENSOR_EFFECTS["bearing"] if s in CALIBRATED_SENSORS}}
    orig = simulate_bearing_lives(
        lambda s: CalibratedFactorySimulator(original_law, n_machines=1, seed=s, noise_scale=SIM_NOISE_SCALE,
                                             bearing_only=True), args.n_lifecycles, 10_000)
    print("[calibrate] original simulator lives done")

    base_vib = {s: SENSOR_BASELINES[s] for s in ("Vibration_X", "Vibration_Y")}
    ratio = float(real["table"]["ratio"].median())
    orig_slopes = dict(COMPONENT_SENSOR_EFFECTS["bearing"])
    law = {
        "k_vib": fit_k(np.median(real["vib"], axis=0)),
        "k_temp": fit_k(np.median(real["temp"], axis=0)),
        "slopes": {
            "Vibration_X": base_vib["Vibration_X"] * (ratio - 1.0),
            "Vibration_Y": base_vib["Vibration_Y"] * (ratio - 1.0),
            "Acoustic_dB": orig_slopes["Acoustic_dB"],
            "Bearing_Temp": float(real["table"]["delta_temp"].median()),
        },
        "lifetime_lognormal_sigma": lifetime_sigma(real["table"]),
        "source": "PRONOSTIA / FEMTO-ST IEEE PHM 2012, research/pronostia/calibrate.py",
    }
    print(f"[calibrate] fitted k_vib={law['k_vib']:.2f} k_temp={law['k_temp']:.2f} "
          f"vib ratio={ratio:.2f} delta_temp={law['slopes']['Bearing_Temp']:.1f} sigma={law['lifetime_lognormal_sigma']:.3f}")

    cal = simulate_bearing_lives(
        lambda s: CalibratedFactorySimulator(law, n_machines=1, seed=s, noise_scale=SIM_NOISE_SCALE,
                                             bearing_only=True),
        args.n_lifecycles, 20_000)
    print("[calibrate] calibrated simulator lives done")

    LAW_PATH.write_text(json.dumps(law, indent=2), encoding="utf-8")
    path = write_summary(real, orig, cal, law, args.n_lifecycles)
    print(f"[calibrate] wrote {LAW_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[calibrate] wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
