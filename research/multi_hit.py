"""
research/multi_hit.py

Multi-hit (cumulative damage) evaluation: the same fault description entered
repeatedly on one machine, as an operator would when a fault persists.

Replicates the dashboard's cumulative-damage mechanism (terminal/app.py and
terminal/factory_state.py):
  - hit k builds its base window from the machine's sensor history, padding
    sparse history with the latest reading (`_build_window`); a fresh machine
    starts from a seeded healthy baseline;
  - the strategy's recorded spike is injected with the production `_inject_spike`
    at the published severity multipliers and scored by the CNN-LSTM;
  - the last 2 rows of the injected window are appended to the history.
The dashboard also pushes one simulated display-only "sparkline" reading per
fault (random values on keyword-picked sensors). It is omitted here so the
history carries only the strategy's own injections.

Recorded outputs come from `baselines_comparison_{turbofan,simulator}.csv`,
pooled (6 recorded repeats per prompt per strategy); each recorded output is
repeated for every hit. No API calls.

Per-prompt targets (defined by the severity group of the TEST_PROMPTS anchor):
    HIGH    OFFLINE at hit 1
    MEDIUM  not OFFLINE at hit 1, and OFFLINE by hit 5 (a persisting real fault escalates)
    LOW     not OFFLINE by hit 3 (an early-warning sign does not trigger shutdown quickly)
The sequence target rate is the fraction of (prompt, repeat) sequences that meet their target.

Usage:
    python -m research.multi_hit [--hits 5] [--checkpoints turbofan simulator simulator_calibrated pronostia]

Output:
    research/results/multi_hit/multi_hit_rows.csv.gz
    research/results/multi_hit/multi_hit_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.multiplier_sweep import (DETERMINISTIC_STRATEGIES, LLM_STRATEGIES, LOCAL_LLM_STRATEGIES,
                                      load_recorded_outputs)
from research.replay_utils import (
    CALIBRATED_WEIGHTS,
    PRONOSTIA_WEIGHTS,
    PUBLISHED_MULTIPLIERS,
    bootstrap_ci,
    healthy_baselines,
    predict_batch,
    select_checkpoint,
    spikes_from_frame,
    status_from_rul,
)

OUT_DIR = PROJECT_ROOT / "research" / "results" / "multi_hit"
TAIL_ROWS = 2
WINDOW = 50
SEED = 20260918

# Severity group of each TEST_PROMPTS anchor (the evaluation_rubric.py group comments).
SEVERITY_GROUP = {
    "minor bearing wobble on Machine 1": "LOW", "slight motor temp drift on Machine 2": "LOW",
    "subtle pressure creep on Machine 3": "LOW", "intermittent vibration on Machine 4": "LOW",
    "early sign of coolant flow drop on Machine 5": "LOW",
    "bearing temperature spike on Machine 1": "MEDIUM", "motor temp anomaly on Machine 2": "MEDIUM",
    "oil pressure surge on Machine 3": "MEDIUM", "vibration above normal on Machine 4": "MEDIUM",
    "coolant flow disruption on Machine 5": "MEDIUM", "bearing overheat on metal press": "MEDIUM",
    "oil pressure surge on QC line": "MEDIUM",
    "catastrophic bearing failure on Machine 1": "HIGH", "complete motor breakdown on Machine 2": "HIGH",
    "oil pressure rupture on Machine 3": "HIGH", "explosion in Machine 4": "HIGH",
    "shaft lock on Machine 5": "HIGH", "catastrophic failure on paint and coat": "HIGH",
}


def window_from_history(history: list[np.ndarray]) -> np.ndarray:
    """Same rule as FactoryState._build_window for a history with at least one row."""
    rows = history[-WINDOW:]
    pad = [rows[-1]] * (WINDOW - len(rows))
    return np.stack(pad + rows).astype(np.float32)


def run_checkpoint(checkpoint: str, df: pd.DataFrame, hits: int) -> pd.DataFrame:
    from agents.diagnostic_agent import _inject_spike
    from agents.schemas import FaultSeverity

    select_checkpoint(checkpoint)
    keys = list(zip(df["source"], df["prompt"], df["repeat"].astype(int)))
    baselines = healthy_baselines(keys, checkpoint, SEED)
    spikes = spikes_from_frame(df)
    table = dict(zip((FaultSeverity.LOW, FaultSeverity.MEDIUM, FaultSeverity.HIGH), PUBLISHED_MULTIPLIERS))
    histories: list[list[np.ndarray]] = [[] for _ in range(len(df))]

    out = []
    for hit in range(1, hits + 1):
        bases = [baselines[k] if not h else window_from_history(h) for k, h in zip(keys, histories)]
        injected = [_inject_spike(b, s, multiplier_override=table[s.fault_severity]) for b, s in zip(bases, spikes)]
        ruls = predict_batch(np.stack(injected))
        for i, inj in enumerate(injected):
            histories[i].extend(list(inj[-TAIL_ROWS:]))
        res = df[["source", "strategy", "prompt", "repeat", "group"]].copy()
        res["checkpoint"] = checkpoint
        res["hit"] = hit
        res["rul"] = ruls
        res["status"] = [status_from_rul(r) for r in ruls]
        out.append(res)
        print(f"  {checkpoint}: hit {hit}/{hits}", flush=True)
    return pd.concat(out, ignore_index=True)


def sequence_outcomes(rows: pd.DataFrame, hits: int) -> pd.DataFrame:
    """One row per (checkpoint, strategy, source, prompt, repeat) with target flags."""
    key = ["checkpoint", "strategy", "source", "prompt", "repeat", "group"]
    wide = rows.pivot_table(index=key, columns="hit", values="status", aggfunc="first").reset_index()
    offline = wide[list(range(1, hits + 1))] == "OFFLINE"
    first_off = offline.idxmax(axis=1).where(offline.any(axis=1))
    wide["first_offline_hit"] = first_off
    wide["offline_hit1"] = offline[1]
    wide["offline_by3"] = offline[[1, 2, 3]].any(axis=1)
    wide["offline_by_last"] = offline.any(axis=1)
    target = np.select(
        [wide["group"] == "HIGH", wide["group"] == "MEDIUM", wide["group"] == "LOW"],
        [wide["offline_hit1"], (~wide["offline_hit1"]) & wide["offline_by_last"], ~wide["offline_by3"]],
        default=False,
    )
    wide["meets_target"] = target.astype(float)
    return wide


def write_summary(seq: pd.DataFrame, hits: int, checkpoints: list[str]) -> Path:
    order = [s for s in LLM_STRATEGIES + LOCAL_LLM_STRATEGIES + DETERMINISTIC_STRATEGIES if s in set(seq["strategy"])]
    md = ["# Multi-hit (cumulative damage) evaluation\n"]
    md.append(f"**Generated by:** `python -m research.multi_hit --hits {hits}`\n")
    md.append("Same recorded fault description entered on the same machine for each hit; the last 2 injected rows "
              "carry forward (dashboard mechanism, without the display-only sparkline row). Recorded outputs pooled "
              "from the turbofan and simulator baseline runs. Published multipliers 0.15 / 0.35 / 0.85.\n")
    md.append("Targets: HIGH → OFFLINE at hit 1; MEDIUM → not OFFLINE at hit 1 and OFFLINE by hit "
              f"{hits}; LOW → not OFFLINE by hit 3.\n")

    md.append("\n## 1. Sequence target rate\n")
    md.append("Fraction of sequences meeting their target, 95% bootstrap CI resampling prompts.\n")
    md.append("| Strategy | " + " | ".join(checkpoints) + " |")
    md.append("|---|" + "---|" * len(checkpoints))
    for s in order:
        cells = []
        for ck in checkpoints:
            g = seq[(seq["strategy"] == s) & (seq["checkpoint"] == ck)]
            lo, hi = bootstrap_ci(g["meets_target"], g["prompt"])
            cells.append(f"{g['meets_target'].mean():.0%} [{lo:.0%}, {hi:.0%}]")
        md.append(f"| {s} | " + " | ".join(cells) + " |")

    md.append("\n## 2. By severity group\n")
    for ck in checkpoints:
        md.append(f"\n**{ck}**\n")
        md.append(f"| Strategy | HIGH: OFFLINE at hit 1 | HIGH: OFFLINE by hit {hits} | MEDIUM: ONLINE/DEGRADED at hit 1 | "
                  f"MEDIUM: OFFLINE by hit {hits} | MEDIUM: mean hits to OFFLINE | LOW: not OFFLINE by hit 3 |")
        md.append("|---|---|---|---|---|---|---|")
        for s in order:
            g = seq[(seq["strategy"] == s) & (seq["checkpoint"] == ck)]
            hi_g, me_g, lo_g = (g[g["group"] == x] for x in ("HIGH", "MEDIUM", "LOW"))
            mean_hits = me_g["first_offline_hit"].dropna()
            md.append(f"| {s} | {hi_g['offline_hit1'].mean():.0%} | {hi_g['offline_by_last'].mean():.0%} | "
                      f"{(~me_g['offline_hit1']).mean():.0%} | {me_g['offline_by_last'].mean():.0%} | "
                      f"{('–' if mean_hits.empty else f'{mean_hits.mean():.1f}')} | {(~lo_g['offline_by3']).mean():.0%} |")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "multi_hit_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hits", type=int, default=5)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    args = parser.parse_args()
    checkpoints = args.checkpoints or (
        ["turbofan", "simulator"]
        + (["simulator_calibrated"] if CALIBRATED_WEIGHTS.exists() else [])
        + (["pronostia"] if PRONOSTIA_WEIGHTS.exists() else []))

    df = pd.concat([load_recorded_outputs("turbofan"), load_recorded_outputs("simulator")], ignore_index=True)
    df["group"] = df["prompt"].map(SEVERITY_GROUP)
    if df["group"].isna().any():
        raise ValueError(f"prompts without a severity group: {df.loc[df['group'].isna(), 'prompt'].unique()}")

    rows = pd.concat([run_checkpoint(ck, df, args.hits) for ck in checkpoints], ignore_index=True)
    seq = sequence_outcomes(rows, args.hits)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "multi_hit_rows.csv.gz", index=False, compression="gzip")
    seq.to_csv(OUT_DIR / "multi_hit_sequences.csv", index=False)
    path = write_summary(seq, args.hits, checkpoints)
    print(f"[multi_hit] wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
