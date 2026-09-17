"""
research/pronostia/replay.py

Replay every recorded strategy output on a new checkpoint, across the same
severity-multiplier grid as research/multiplier_sweep.py.

LLM outputs do not depend on the checkpoint, so the recorded runs from both
`baselines_comparison_turbofan.csv` and `baselines_comparison_simulator.csv`
are pooled (6 recorded repeats per prompt per strategy). Each
(source, prompt, repeat) gets its own seeded healthy baseline.

Correlation maps (`--correlations`):
    production       agents/diagnostic_agent.py SENSOR_CORRELATIONS, unchanged
    bearing_physics  sensitivity analysis for the PRONOSTIA checkpoint: bearing-related
                     primaries (Xs2 bearing temp, Xs0/Xs1 vibration, Xs12 acoustic) also
                     drive the vibration channels. On a real failing rolling-element bearing,
                     vibration and high-frequency energy rise first and temperature matters
                     little (calibrate.py: vibration onset at 96% of life; the PRONOSTIA model
                     barely responds to temperature). Intensities mirror the production map's
                     critical-trio strength. All other primaries keep the production map.

Run `python -m research.pronostia.probe --checkpoint <checkpoint>` first.

Usage:
    python -m research.pronostia.replay --checkpoint pronostia
    python -m research.pronostia.replay --checkpoint pronostia --correlations bearing_physics
    python -m research.pronostia.replay --checkpoint simulator_calibrated

Output:
    research/results/pronostia/multiplier_sweep_<checkpoint>[_bearing_physics].csv
    research/results/pronostia/multiplier_sweep_rows_<checkpoint>[_bearing_physics].csv.gz
    research/results/pronostia/multiplier_sweep_summary_<checkpoint>[_bearing_physics].md
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.multiplier_sweep import _parse_grid, load_recorded_outputs, run_sweep, write_summary

OUT_DIR = PROJECT_ROOT / "research" / "results" / "pronostia"

BEARING_PHYSICS_CORRELATIONS: dict[str, list[tuple[str, float]]] = {
    "Xs2":  [("Xs0", 0.90), ("Xs1", 0.90), ("Xs12", 0.80), ("Xs3", 0.95), ("Xs4", 0.90)],
    "Xs0":  [("Xs1", 0.90), ("Xs12", 0.80), ("Xs2", 0.60), ("Xs3", 0.52), ("Xs4", 0.45)],
    "Xs1":  [("Xs0", 0.90), ("Xs12", 0.80), ("Xs2", 0.60), ("Xs3", 0.52), ("Xs4", 0.45)],
    "Xs12": [("Xs0", 0.80), ("Xs1", 0.80), ("Xs2", 0.55), ("Xs3", 0.48), ("Xs4", 0.42)],
}
TITLES = {"pronostia": "PRONOSTIA real-data checkpoint",
          "simulator_calibrated": "PRONOSTIA-calibrated simulator checkpoint"}


@contextmanager
def correlation_map(name: str):
    """Temporarily swap entries of agents.diagnostic_agent.SENSOR_CORRELATIONS."""
    import agents.diagnostic_agent as da

    if name == "production":
        yield
        return
    saved = {k: da.SENSOR_CORRELATIONS[k] for k in BEARING_PHYSICS_CORRELATIONS}
    da.SENSOR_CORRELATIONS.update(BEARING_PHYSICS_CORRELATIONS)
    try:
        yield
    finally:
        da.SENSOR_CORRELATIONS.update(saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", choices=list(TITLES), default="pronostia")
    parser.add_argument("--correlations", choices=["production", "bearing_physics"], default="production")
    parser.add_argument("--low", default="0.15")
    parser.add_argument("--medium", default="0.10:0.60:0.05")
    parser.add_argument("--high", default="0.30:1.00:0.05")
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    ck = args.checkpoint

    if not (OUT_DIR / f"probe_summary_{ck}.md").exists():
        raise SystemExit(f"[replay] run `python -m research.pronostia.probe --checkpoint {ck}` first "
                         "(geometry is recorded before replay)")

    df = pd.concat([load_recorded_outputs("turbofan"), load_recorded_outputs("simulator")], ignore_index=True)
    with correlation_map(args.correlations):
        sweep, detail = run_sweep(ck, df, _parse_grid(args.low), _parse_grid(args.medium),
                                  _parse_grid(args.high), args.seed)

    suffix = "" if args.correlations == "production" else f"_{args.correlations}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"multiplier_sweep_{ck}{suffix}.csv"
    rows_path = OUT_DIR / f"multiplier_sweep_rows_{ck}{suffix}.csv.gz"
    sweep.to_csv(csv_path, index=False)
    detail.to_csv(rows_path, index=False, compression="gzip")
    note = ("Replays the recorded outputs pooled from `baselines_comparison_turbofan.csv` and "
            "`baselines_comparison_simulator.csv` (6 recorded repeats per prompt per strategy). "
            f"Correlation map: **{args.correlations}**.")
    md_path = write_summary(
        f"{TITLES[ck]} ({args.correlations} correlations)", sweep, detail, args.seed,
        f"python -m research.pronostia.replay --checkpoint {ck} --correlations {args.correlations}",
        note, OUT_DIR / f"multiplier_sweep_summary_{ck}{suffix}.md",
    )
    for p in (csv_path, rows_path, md_path):
        print(f"[replay] wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
