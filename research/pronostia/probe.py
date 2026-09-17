"""
research/pronostia/probe.py

Response-surface geometry of the two new checkpoints, compared with the
turbofan and simulator checkpoints. Run BEFORE `replay.py`: the geometry
verdict and the expectation it implies are written first, then the replay
tests it.

Checkpoints (`--checkpoint`):
    pronostia             real-data model (train_model.py). Probe grid over the three
                          live bearing channels: Xs0 horizontal vibration, Xs2 bearing
                          temperature, Xs12 high-frequency vibration power. Other channels
                          stay at the median early-life reading of the training bearings.
                          Also reports degradation-phase validation and the Xs2-only curve
                          (Xs3/Xs4 are unmeasured constants on this rig).
    simulator_calibrated  simulator model retrained on the PRONOSTIA-calibrated simulator
                          (calibrated_checkpoint.py). Probe grid over Xs2/Xs3/Xs4, exactly as
                          research/probe_cliff_3d.py does for the other two checkpoints, on the
                          polarity-aware healthy baseline.

Grid: steps^3 scaled positions in [0.05, 0.95], flat-filled across the 50-row
window (the production injection pattern for critical sensors).

Geometry metrics (same definitions as research/probe_cliff_3d.py, plus one standard statistic):
    online/degraded/offline share   fraction of grid points per status bucket
    iqr_pct_of_range                (Q75 - Q25) / (max - min) of predicted RUL
    near_q25_q75                    fraction of points within ±1 RUL of Q25 or Q75
    likely_bimodal                  iqr_pct_of_range > 0.5 and near_q25_q75 > 0.5 (probe_cliff_3d rule)
    bimodality_coefficient          Sarle's b = (skew^2 + 1) / (excess_kurtosis + 3(n-1)^2/((n-2)(n-3))); > 0.555 suggests bimodality

Usage:
    python -m research.pronostia.probe --checkpoint pronostia [--steps 15]
    python -m research.pronostia.probe --checkpoint simulator_calibrated

Output:
    research/results/pronostia/probe_<checkpoint>.csv
    research/results/pronostia/probe_summary_<checkpoint>.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.replay_utils import PRONOSTIA_META, predict_batch, select_checkpoint, status_from_rul

OUT_DIR = PROJECT_ROOT / "research" / "results" / "pronostia"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
TRIOS = {
    "pronostia": {"Xs0": 4, "Xs2": 6, "Xs12": 16},
    "simulator_calibrated": {"Xs2": 6, "Xs3": 7, "Xs4": 8},
}


def geometry(rul: np.ndarray) -> dict:
    """Surface-shape metrics for an array of predicted RULs over a probe grid."""
    status = np.array([status_from_rul(r) for r in rul])
    q25, q75 = np.quantile(rul, [0.25, 0.75])
    rng = float(rul.max() - rul.min())
    n = len(rul)
    near = float((np.sum(np.abs(rul - q25) <= 1.0) + np.sum(np.abs(rul - q75) <= 1.0)) / n)
    iqr_pct = float((q75 - q25) / rng) if rng > 0 else 0.0
    skew, kurt = float(stats.skew(rul)), float(stats.kurtosis(rul))
    b = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return {
        "n": n,
        "rul_min": float(rul.min()), "rul_q25": float(q25), "rul_median": float(np.median(rul)),
        "rul_q75": float(q75), "rul_max": float(rul.max()),
        "online_share": float(np.mean(status == "ONLINE")),
        "degraded_share": float(np.mean(status == "DEGRADED")),
        "offline_share": float(np.mean(status == "OFFLINE")),
        "iqr_pct_of_range": iqr_pct,
        "near_q25_q75": near,
        "likely_bimodal": bool(iqr_pct > 0.5 and near > 0.5),
        "bimodality_coefficient": float(b),
    }


def healthy_reading(checkpoint: str) -> np.ndarray:
    """(18,) raw healthy reading used as the probe background."""
    import dl_engine.inference as inf

    if checkpoint == "pronostia":
        return np.asarray(json.loads(PRONOSTIA_META.read_text(encoding="utf-8"))["healthy_raw"], dtype=np.float32)
    return inf.get_healthy_baseline(noise_std_frac=0.0)[0]


def degradation_phase_metrics(meta: dict) -> dict:
    """RMSE and status accuracy on PRONOSTIA validation windows at or after each bearing's FPT."""
    from research.pronostia.train_model import bearing_windows, load_features

    feats = load_features(meta.get("bearing_set", "temp"))
    out = {}
    for b in meta["val_bearings"]:
        X, y, fpt = bearing_windows(feats[feats["bearing"] == b])
        X, y = X[fpt:], y[fpt:]
        pred = predict_batch(X)
        out[b] = {
            "n": int(len(y)),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "status_accuracy": float(np.mean([status_from_rul(p) == status_from_rul(t) for p, t in zip(pred, y)])),
        }
    return out


def probe_grid(healthy: np.ndarray, cols: dict[str, int], steps: int, lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    import dl_engine.inference as inf

    ranges = inf.get_scaler_ranges()
    grid = np.linspace(lo, hi, steps)
    names = list(cols)
    points = list(itertools.product(grid, repeat=len(names)))
    windows = np.tile(healthy.astype(np.float32), (len(points), 50, 1))
    for i, pos in enumerate(points):
        for name, s in zip(names, pos):
            col = cols[name]
            windows[i, :, col] = ranges["min"][col] + s * ranges["range"][col]
    df = pd.DataFrame(points, columns=[f"{n}_scaled" for n in names])
    df["rul"] = predict_batch(windows)
    return df


def _fmt_table(table: pd.DataFrame) -> str:
    fmt = table.copy()
    for c in ("online_share", "degraded_share", "offline_share", "iqr_pct_of_range", "near_q25_q75"):
        fmt[c] = fmt[c].map(lambda v: f"{v:.1%}")
    for c in ("rul_min", "rul_q25", "rul_median", "rul_q75", "rul_max", "bimodality_coefficient"):
        fmt[c] = fmt[c].map(lambda v: f"{v:.2f}")
    return fmt.to_markdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", choices=list(TRIOS), default="pronostia")
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()
    ck = args.checkpoint

    select_checkpoint(ck)
    import dl_engine.inference as inf

    healthy = healthy_reading(ck)
    trio = TRIOS[ck]
    t0 = time.time()
    grid = probe_grid(healthy, trio, args.steps)
    print(f"[probe] {ck}: {len(grid)} evaluations in {time.time() - t0:.1f}s")

    ranges = inf.get_scaler_ranges()
    healthy_scaled = {n: float((healthy[c] - ranges["min"][c]) / ranges["range"][c]) if ranges["range"][c] > 0 else 0.0
                      for n, c in trio.items()}
    healthy_rul = float(predict_batch(np.tile(healthy.astype(np.float32), (1, 50, 1)))[0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_DIR / f"probe_{ck}.csv", index=False)

    label = f"{ck} ({'/'.join(trio)})"
    rows = {label: geometry(grid["rul"].to_numpy())}
    for variant in ("turbofan", "simulator"):
        path = RESULTS_DIR / f"probe_cliff_3d_{variant}.csv"
        if path.exists():
            rows[f"{variant} (Xs2/Xs3/Xs4, probe_cliff_3d)"] = geometry(pd.read_csv(path)["rul"].to_numpy())
    other = "simulator_calibrated" if ck == "pronostia" else "pronostia"
    other_path = OUT_DIR / f"probe_{other}.csv"
    if other_path.exists():
        rows[f"{other} ({'/'.join(TRIOS[other])})"] = geometry(pd.read_csv(other_path)["rul"].to_numpy())
    table = pd.DataFrame(rows).T
    g = rows[label]

    titles = {"pronostia": "PRONOSTIA real-data checkpoint",
              "simulator_calibrated": "PRONOSTIA-calibrated simulator checkpoint"}
    md = [f"# {titles[ck]} — response-surface geometry\n"]
    md.append(f"**Generated by:** `python -m research.pronostia.probe --checkpoint {ck} --steps {args.steps}` "
              f"on {time.strftime('%Y-%m-%d %H:%M')}, before any strategy replay on this checkpoint.\n")

    if ck == "pronostia":
        meta = json.loads(PRONOSTIA_META.read_text(encoding="utf-8"))
        md.append(f"Checkpoint ({meta.get('bearing_set', 'temp')} bearings): trained on "
                  f"{', '.join(meta['train_bearings'])}; validated on {', '.join(meta['val_bearings'])} "
                  f"(all-window RMSE {meta['val_rmse']:.2f}).\n")
        md.append("Degradation start (FPT) as life fraction: "
                  + ", ".join(f"{b} {f:.2f}" for b, f in meta["fpt_life_fraction"].items()) + ".\n")
        md.append("\n**Degradation-phase validation** (windows at or after FPT; all-window numbers are dominated "
                  "by healthy windows labelled 100):\n")
        md.append("| Bearing | Windows after FPT | RMSE | Status accuracy |")
        md.append("|---|---|---|---|")
        for b, m in degradation_phase_metrics(meta).items():
            md.append(f"| {b} | {m['n']} | {m['rmse']:.2f} | {m['status_accuracy']:.1%} |")
    else:
        md.append("Checkpoint: CNN-LSTM retrained on 1000 lifecycles of the PRONOSTIA-calibrated simulator "
                  "(`research/pronostia/calibrated_checkpoint.py`).\n")

    md.append(f"\nHealthy reference reading: predicted RUL {healthy_rul:.1f}; scaled positions "
              + ", ".join(f"{k} {v:.2f}" for k, v in healthy_scaled.items()) + ".\n")
    md.append("\n## Geometry comparison\n")
    md.append(_fmt_table(table))

    if ck == "pronostia":
        xs2_curve = probe_grid(healthy, {"Xs2": 6}, 50)
        md.append("\n## Xs2 (bearing temperature) alone\n")
        md.append("On this checkpoint Xs3 (motor temp) and Xs4 (oil pressure) are unmeasured constants, so the "
                  "paper's Xs2/Xs3/Xs4 probe reduces to this curve.\n")
        md.append("| Xs2 scaled | RUL |")
        md.append("|---|---|")
        for r in xs2_curve.iloc[:: 5].itertuples():
            md.append(f"| {r.Xs2_scaled:.2f} | {r.rul:.1f} |")

    md.append("\n## Expectation recorded before replay\n")
    if g["degraded_share"] < 0.10:
        md.append(f"The DEGRADED share is {g['degraded_share']:.1%} (< 10%): the surface is cliff-like. From the "
                  "turbofan and simulator sweeps, the ranking should hinge on the few prompts where strategies "
                  "differ in severity or sensor routing, and should be sensitive to the HIGH multiplier.")
    else:
        md.append(f"The DEGRADED share is {g['degraded_share']:.1%} (≥ 10%): the surface is graded. From the "
                  "simulator sweep, the regex's fixed band-centre spike values should match or beat the LLMs "
                  "across most multiplier settings.")
    if ck == "pronostia":
        md.append("\nOnly Xs0, Xs1, Xs2 and Xs12 (plus W0/W2 between operating conditions) carry signal on this "
                  "checkpoint, so prompts routed to motor, oil, coolant or electrical sensors can act only through "
                  "their correlated Xs2 component.")

    path = OUT_DIR / f"probe_summary_{ck}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table[["degraded_share", "offline_share", "near_q25_q75", "bimodality_coefficient"]].round(3))
    print(f"[probe] wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
