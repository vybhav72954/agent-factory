"""
research/multiplier_sweep.py

Severity-multiplier sweep: how do the strategy rankings depend on the
SEVERITY_MULTIPLIERS table?

The turbofan ranking flipped (regex ahead -> LLMs ahead) when the HIGH
multiplier moved from 0.35 to 0.85. This script maps that effect instead of
reporting a single tuned constant.

No API calls. Injection depends only on (sensor_id, fault_severity,
spike_value) from each strategy's output, and those are already recorded per
(strategy, prompt, repeat) in `baselines_comparison_<variant>.csv`. The script
replays every recorded output through the production `_inject_spike` with a
`multiplier_override` taken from the swept table, then runs the real CNN-LSTM
(helpers in research/replay_utils.py).

Each (source, prompt, repeat) gets one seeded healthy baseline shared by every
strategy and every multiplier setting, so comparisons are paired.

`agentic_continuous` is excluded: its LLM-emitted continuous multiplier was
not recorded in the 2026-05 CSVs, so it cannot be replayed.

Usage:
    python -m research.multiplier_sweep                       # both checkpoints
    python -m research.multiplier_sweep --variant turbofan
    python -m research.multiplier_sweep --low 0.15 --medium 0.10:0.60:0.05 --high 0.30:1.00:0.05

Output (per variant):
    research/results/multiplier_sweep_<variant>.csv          one row per (low, medium, high, strategy)
    research/results/multiplier_sweep_rows_<variant>.csv.gz  one row per (setting, strategy, prompt, repeat)
    research/results/multiplier_sweep_summary_<variant>.md   replay validation, sweep tables, per-prompt drivers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.replay_utils import (
    PUBLISHED_MULTIPLIERS,
    check_batch_matches_single,
    healthy_baselines,
    replay,
    select_checkpoint,
    spikes_from_frame,
)

RESULTS_DIR = PROJECT_ROOT / "research" / "results"

LLM_STRATEGIES = ["agentic", "gemini_3_5_flash", "groq_gpt_oss_120b", "groq_qwen3_8",
                  "groq_llama3", "groq_llama4"]          # Llama rows: May 2026 runs, models since retired on Groq
DETERMINISTIC_STRATEGIES = ["keyword_regex_severity", "keyword_regex_extended", "embedding_severity",
                            "keyword_only", "fixed_midrange"]
DETERMINISTIC_STRATEGIES += ["tfidf_logreg", "minilm_logreg", "bge_m3_logreg", "minilm_finetuned"]  # trained, R1
# Small open models run locally (added 2026-09-17). Reported per strategy but kept out of the
# best-LLM-vs-regex win counts, which are defined on the hosted LLMs.
LOCAL_LLM_STRATEGIES = ["ollama_llama3_2_3b", "ollama_gemma3_4b", "ollama_qwen3_4b"]
REPLAYABLE_STRATEGIES = LLM_STRATEGIES + LOCAL_LLM_STRATEGIES + DETERMINISTIC_STRATEGIES


def _parse_grid(spec: str) -> list[float]:
    """Parse 'start:stop:step' (inclusive) or a single float into a list of floats."""
    if ":" not in spec:
        return [round(float(spec), 4)]
    start, stop, step = (float(x) for x in spec.split(":"))
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(n)]


def load_recorded_outputs(source: str) -> pd.DataFrame:
    """Recorded, replayable strategy outputs from `baselines_comparison_<source>.csv`."""
    from research.evaluation_rubric import TEST_PROMPTS

    df = pd.read_csv(RESULTS_DIR / f"baselines_comparison_{source}.csv")
    df = df[(~df["rejected"].astype(bool)) & (df["strategy"].isin(REPLAYABLE_STRATEGIES))].copy()
    expected = {p: s for p, _mid, s in TEST_PROMPTS}
    df["expected_status"] = df["prompt"].map(expected)
    if df["expected_status"].isna().any():
        missing = df.loc[df["expected_status"].isna(), "prompt"].unique()
        raise ValueError(f"prompts in CSV not found in TEST_PROMPTS: {missing}")
    df["published_match"] = df["status"] == df["expected_status"]
    df["source"] = source
    return df.reset_index(drop=True)


def run_sweep(
    checkpoint: str, df: pd.DataFrame, lows: list[float], mediums: list[float], highs: list[float], seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay recorded outputs under every multiplier setting on one checkpoint.

    Args:
        checkpoint: "turbofan", "simulator" or "pronostia".
        df:         rows from `load_recorded_outputs` (one or more sources concatenated).

    Returns:
        (aggregated rows per setting and strategy, per-output rows per setting)
    """
    select_checkpoint(checkpoint)
    keys = list(zip(df["source"], df["prompt"], df["repeat"].astype(int)))
    baselines = healthy_baselines(keys, checkpoint, seed)
    check_batch_matches_single(list(baselines.values()))
    spikes = spikes_from_frame(df)
    base_windows = [baselines[k] for k in keys]

    settings = [(lo, me, hi) for lo in lows for me in mediums for hi in highs if lo <= me <= hi]
    if PUBLISHED_MULTIPLIERS not in settings:
        settings.append(PUBLISHED_MULTIPLIERS)
    print(f"[sweep] {checkpoint}: {len(df)} recorded outputs x {len(settings)} settings "
          f"= {len(df) * len(settings)} predictions")

    rows, detail = [], []
    t0 = time.time()
    for k, (lo, me, hi) in enumerate(settings, start=1):
        ruls, statuses = replay(spikes, base_windows, (lo, me, hi))
        res = df[["source", "strategy", "prompt", "repeat", "sensor_id", "severity", "expected_status"]].copy()
        res["rul"] = ruls
        res["status"] = statuses
        res["match"] = res["status"] == res["expected_status"]
        detail.append(res.assign(low=lo, medium=me, high=hi))

        for strategy, sub in res.groupby("strategy"):
            row = {"checkpoint": checkpoint, "low": lo, "medium": me, "high": hi, "strategy": strategy,
                   "n": len(sub), "match_rate": sub["match"].mean(), "mean_rul": sub["rul"].mean()}
            for exp in ("ONLINE", "OFFLINE"):
                part = sub[sub["expected_status"] == exp]
                row[f"match_rate_expected_{exp.lower()}"] = part["match"].mean() if len(part) else np.nan
                row[f"n_expected_{exp.lower()}"] = len(part)
            rows.append(row)

        if k % 20 == 0 or k == len(settings):
            print(f"  {k}/{len(settings)} settings  {time.time() - t0:.0f}s")

    out = pd.DataFrame(rows)
    if "published_match" in df and set(df["source"]) == {checkpoint}:
        published = df.groupby("strategy")["published_match"].mean().rename("published_match_rate")
        out = out.merge(published, left_on="strategy", right_index=True, how="left")
    return out, pd.concat(detail, ignore_index=True)


def _present(strategies: list[str], available) -> list[str]:
    return [s for s in strategies if s in set(available)]


def _prompt_drivers(detail: pd.DataFrame, lo: float, me: float, hi: float) -> list[str]:
    """Markdown lines listing prompts where the LLMs and the regex disagree at one setting."""
    compared = _present(LLM_STRATEGIES, detail["strategy"]) + ["keyword_regex_severity"]
    sub = detail[(detail["low"] == lo) & (detail["medium"] == me) & (detail["high"] == hi)
                 & detail["strategy"].isin(compared)]
    g = (sub.groupby(["prompt", "strategy"])
            .agg(match=("match", "sum"), n=("match", "size"),
                 severity=("severity", lambda s: "/".join(sorted(set(s)))),
                 sensor=("sensor_id", lambda s: "/".join(sorted(set(s)))),
                 rul=("rul", "mean"))
            .reset_index())
    piv = g.pivot(index="prompt", columns="strategy", values="match")
    differing = piv[piv.nunique(axis=1) > 1].index
    lines = [f"\n**LOW {lo}, MEDIUM {me}, HIGH {hi}:** "
             f"{len(differing)} prompt(s) where the LLMs and the regex do not all score the same.\n"]
    if len(differing) == 0:
        return lines
    lines.append("| Prompt | Expected | Strategy | Matches | Severity | Sensor | Mean RUL |")
    lines.append("|---|---|---|---|---|---|---|")
    for prompt in differing:
        expected = sub.loc[sub["prompt"] == prompt, "expected_status"].iloc[0]
        for strategy in compared:
            r = g[(g["prompt"] == prompt) & (g["strategy"] == strategy)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| {prompt} | {expected} | {strategy} | {int(r['match'])}/{int(r['n'])} | "
                         f"{r['severity']} | {r['sensor']} | {r['rul']:.1f} |")
    return lines


def best_llm_gap(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per setting: best LLM match rate minus regex match rate (positive = LLM ahead)."""
    piv = sweep.pivot_table(index=["low", "medium", "high"], columns="strategy", values="match_rate")
    llms = _present(LLM_STRATEGIES, piv.columns)
    gap = pd.DataFrame(index=piv.index)
    gap["best_llm"] = piv[llms].max(axis=1)
    gap["best_llm_name"] = piv[llms].idxmax(axis=1)
    gap["regex"] = piv["keyword_regex_severity"]
    gap["gap_best_llm_minus_regex"] = gap["best_llm"] - gap["regex"]
    return gap.reset_index()


def write_summary(label: str, sweep: pd.DataFrame, detail: pd.DataFrame, seed: int, command: str,
                  sources_note: str, path: Path) -> Path:
    """Human-readable markdown: replay validation, 1-D sweeps, ranking-flip map, per-prompt drivers."""
    lo_p, me_p, hi_p = PUBLISHED_MULTIPLIERS
    md = [f"# Severity-Multiplier Sweep — {label}\n"]
    md.append(f"**Generated by:** `{command}` (seed {seed})\n")
    md.append(f"{sources_note} No API calls. `agentic_continuous` is excluded because its continuous "
              "multiplier was not recorded. `groq_llama3`/`groq_llama4` are the May 2026 runs; Groq has "
              "since retired both models.\n")

    if "published_match_rate" in sweep:
        md.append(f"\n## 1. Replay validation at the published multipliers (LOW {lo_p}, MEDIUM {me_p}, HIGH {hi_p})\n")
        md.append("If the replay is faithful, these match the recorded status-match rates.\n")
        md.append("| Strategy | Recorded | Replayed | Δ |")
        md.append("|---|---|---|---|")
        val = sweep[(sweep["low"] == lo_p) & (sweep["medium"] == me_p) & (sweep["high"] == hi_p)]
        for r in val.sort_values("strategy").itertuples():
            md.append(f"| {r.strategy} | {r.published_match_rate:.1%} | {r.match_rate:.1%} | "
                      f"{(r.match_rate - r.published_match_rate) * 100:+.1f} pts |")
    else:
        md.append("\n## 1. Replay validation\n")
        md.append("Not applicable: no recorded run exists on this checkpoint.\n")

    order = _present(LLM_STRATEGIES + LOCAL_LLM_STRATEGIES + DETERMINISTIC_STRATEGIES, sweep["strategy"])

    def _one_d_table(fixed_col: str, fixed_val: float, swept_col: str) -> None:
        sub = sweep[(sweep["low"] == lo_p) & (sweep[fixed_col] == fixed_val)]
        piv = sub.pivot_table(index="strategy", columns=swept_col, values="match_rate").reindex(order)
        cols = list(piv.columns)
        md.append("| Strategy | " + " | ".join(f"{c:.2f}" for c in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for strategy, vals in piv.iterrows():
            md.append(f"| {strategy} | " + " | ".join("–" if pd.isna(v) else f"{v:.0%}" for v in vals) + " |")

    md.append(f"\n## 2. HIGH multiplier sweep (LOW {lo_p}, MEDIUM {me_p})\n")
    md.append("Status-match rate per strategy as the HIGH multiplier changes.\n")
    _one_d_table("medium", me_p, "high")

    md.append(f"\n## 3. MEDIUM multiplier sweep (LOW {lo_p}, HIGH {hi_p})\n")
    _one_d_table("high", hi_p, "medium")

    gap = best_llm_gap(sweep[sweep["low"] == lo_p])
    n = len(gap)
    llm_ahead = int((gap["gap_best_llm_minus_regex"] > 0).sum())
    regex_ahead = int((gap["gap_best_llm_minus_regex"] < 0).sum())
    md.append(f"\n## 4. Who wins across the MEDIUM × HIGH grid (LOW {lo_p})\n")
    md.append(f"Across {n} settings: best LLM ahead of regex in **{llm_ahead}**, regex ahead in "
              f"**{regex_ahead}**, tied in **{n - llm_ahead - regex_ahead}**.\n")
    md.append("Cell value = best LLM match rate minus regex match rate, in percentage points "
              "(positive = an LLM is ahead).\n")
    piv = gap.pivot_table(index="medium", columns="high", values="gap_best_llm_minus_regex")
    cols = list(piv.columns)
    md.append("| MEDIUM \\ HIGH | " + " | ".join(f"{c:.2f}" for c in cols) + " |")
    md.append("|---|" + "---|" * len(cols))
    for medium, vals in piv.iterrows():
        md.append(f"| {medium:.2f} | " + " | ".join("" if pd.isna(v) else f"{v * 100:+.0f}" for v in vals) + " |")

    md.append("\n## 5. Which prompts create the LLM-vs-regex gap\n")
    md.append(f"At the published setting, and at the HIGH value (MEDIUM {me_p}) where the regex lead over "
              "the best LLM is largest.\n")
    md += _prompt_drivers(detail, lo_p, me_p, hi_p)
    row = gap[gap["medium"] == me_p].sort_values(["gap_best_llm_minus_regex", "high"]).iloc[0]
    if float(row["high"]) != hi_p:
        md += _prompt_drivers(detail, lo_p, me_p, float(row["high"]))

    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["turbofan", "simulator", "both"], default="both")
    parser.add_argument("--low", default="0.15", help="LOW multiplier value or start:stop:step (default 0.15)")
    parser.add_argument("--medium", default="0.10:0.60:0.05", help="MEDIUM grid (default 0.10:0.60:0.05)")
    parser.add_argument("--high", default="0.30:1.00:0.05", help="HIGH grid (default 0.30:1.00:0.05)")
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    lows, mediums, highs = _parse_grid(args.low), _parse_grid(args.medium), _parse_grid(args.high)
    variants = ["turbofan", "simulator"] if args.variant == "both" else [args.variant]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        sweep, detail = run_sweep(variant, load_recorded_outputs(variant), lows, mediums, highs, args.seed)
        csv_path = RESULTS_DIR / f"multiplier_sweep_{variant}.csv"
        rows_path = RESULTS_DIR / f"multiplier_sweep_rows_{variant}.csv.gz"
        sweep.to_csv(csv_path, index=False)
        detail.to_csv(rows_path, index=False, compression="gzip")
        md_path = write_summary(
            f"{variant} checkpoint", sweep, detail, args.seed,
            f"python -m research.multiplier_sweep --variant {variant}",
            f"Replays the recorded outputs in `baselines_comparison_{variant}.csv`.",
            RESULTS_DIR / f"multiplier_sweep_summary_{variant}.md",
        )
        for p in (csv_path, rows_path, md_path):
            print(f"[sweep] wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
