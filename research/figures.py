"""
research/figures.py

Publication-quality figure generator for the ForgeMind paper. Generates 6
figures from the snapshotted CSVs in `research/results/`:

  fig1 — Strategy comparison: 7 strategies x 2 checkpoints, grouped bars
  fig2 — Probe heatmap: turbofan vs simulator side-by-side
  fig3 — Sequence walkthrough: 3-hit MEDIUM trajectory on simulator (V0 vs V3)
  fig4 — Latency vs accuracy Pareto plot (simulator checkpoint)
  fig5 — Per-prompt match matrix: strategy x OFFLINE-prompt heatmap
  fig6 — Sensor selection: which sensor each strategy picks per OFFLINE prompt

Style targets the joint defaults of EAAI / Computers in Industry / IEEE TII:
- dpi=300 (publication)
- serif font, 11pt labels / 9pt ticks
- ASCII-safe labels (no Unicode arrows or check marks — cp1252 safety)
- Error bars on bar charts (from stability_runs std where applicable)
- Distinct categorical colour palette across strategies for consistency

Usage:
    python -m research.figures              # all six figures
    python -m research.figures --fig 1      # only fig1 (debug)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.evaluation_rubric import TEST_PROMPTS


RESULTS_DIR = PROJECT_ROOT / "research" / "results"
FIG_DIR = PROJECT_ROOT / "research" / "paper_assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        100,   # display; savefig uses 300
    "savefig.dpi":       700,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.25,
})

# Consistent strategy ordering across all figures (matches paper §5 tables)
STRATEGY_ORDER = [
    "keyword_only",
    "fixed_midrange",
    "keyword_regex_severity",
    "agentic",
    "gemini_3_5_flash",
    "groq_llama3",
    "groq_llama4",
    "agentic_continuous",   # Extension #6 — may be missing from older CSVs
]

# Pretty labels for each strategy (paper figures use these)
STRATEGY_LABEL = {
    "keyword_only":            "keyword-only",
    "fixed_midrange":          "fixed-midrange",
    "keyword_regex_severity":  "keyword+regex (strong)",
    "agentic":                 "Gemini 2.5 Flash",
    "gemini_3_5_flash":        "Gemini 3.5 Flash",
    "groq_llama3":             "Llama 3.3 70B",
    "groq_llama4":             "Llama 4 Scout",
    "agentic_continuous":      "Gemini 2.5 (cont.)",
}

# Categorical palette — distinct for each strategy class
STRATEGY_COLOR = {
    "keyword_only":            "#888888",   # grey, weak baseline
    "fixed_midrange":          "#bbbbbb",   # light grey, trivial
    "keyword_regex_severity":  "#2ca02c",   # green, strong baseline (winner)
    "agentic":                 "#1f77b4",   # blue, Gemini family
    "gemini_3_5_flash":        "#5fa9d6",   # light blue
    "groq_llama3":             "#d62728",   # red, Llama family
    "groq_llama4":             "#ff7f7f",   # light red
    "agentic_continuous":      "#9467bd",   # purple, continuous variant
}

EXPECTED_BY_PROMPT = {p: s for p, _mid, s in TEST_PROMPTS}


def _load_baselines(csv_name: str) -> pd.DataFrame | None:
    """Load a baselines comparison CSV; return None if missing."""
    path = RESULTS_DIR / csv_name
    if not path.exists():
        print(f"[fig] missing: {path.name} — skipping")
        return None
    df = pd.read_csv(path)
    df = df[~df["rejected"]].copy()
    df["expected"] = df["prompt"].map(EXPECTED_BY_PROMPT)
    df["match"] = df["status"] == df["expected"]
    return df


def _per_strategy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per strategy with mean status-match + bootstrap CI."""
    rows = []
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        n = len(sub)
        match_rate = float(sub["match"].mean())
        # Wilson-style CI via bootstrap (1000 resamples) — N=54-ish so this is fine
        boots = np.random.default_rng(42).choice(
            sub["match"].astype(int).values, size=(1000, n), replace=True
        ).mean(axis=1)
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        on  = sub[sub["expected"] == "ONLINE"]
        off = sub[sub["expected"] == "OFFLINE"]
        rows.append({
            "strategy": strat,
            "n": n,
            "match_rate": match_rate,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "online_match": float(on["match"].mean()) if len(on) else float("nan"),
            "offline_match": float(off["match"].mean()) if len(off) else float("nan"),
            "p95_latency": float(sub["latency_ms"].quantile(0.95)),
            "mean_latency": float(sub["latency_ms"].mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Strategy comparison (grouped bars: turbofan vs simulator)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_strategy_comparison():
    df_t = _load_baselines("baselines_comparison_turbofan.csv")
    df_s = _load_baselines("baselines_comparison_simulator.csv")
    if df_t is None or df_s is None:
        return
    stats_t = _per_strategy_stats(df_t).set_index("strategy")
    stats_s = _per_strategy_stats(df_s).set_index("strategy")

    # Use the UNION so simulator-only strategies (e.g. agentic_continuous) appear,
    # with NaN bars on the missing checkpoint (matplotlib draws nothing for NaN).
    present = [s for s in STRATEGY_ORDER if s in stats_t.index or s in stats_s.index]

    def _lookup(stats, s, col):
        return stats.loc[s, col] if s in stats.index else float("nan")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(present))
    bar_w = 0.38

    t_vals = [_lookup(stats_t, s, "match_rate") * 100 for s in present]
    s_vals = [_lookup(stats_s, s, "match_rate") * 100 for s in present]

    def _err(stats, s):
        if s not in stats.index:
            return 0.0, 0.0
        mr = stats.loc[s, "match_rate"]
        return (mr - stats.loc[s, "ci_lo"]) * 100, (stats.loc[s, "ci_hi"] - mr) * 100

    t_err = list(zip(*[_err(stats_t, s) for s in present]))
    s_err = list(zip(*[_err(stats_s, s) for s in present]))
    t_err = [list(t_err[0]), list(t_err[1])]
    s_err = [list(s_err[0]), list(s_err[1])]

    b1 = ax.bar(x - bar_w/2, t_vals, bar_w, yerr=t_err, label="Turbofan checkpoint",
                color="#8c6d31", edgecolor="black", linewidth=0.7, capsize=3, error_kw={"linewidth": 0.7})
    b2 = ax.bar(x + bar_w/2, s_vals, bar_w, yerr=s_err, label="Simulator checkpoint",
                color="#bcbd22", edgecolor="black", linewidth=0.7, capsize=3, error_kw={"linewidth": 0.7})

    # Annotate the regex baseline win line
    ax.axhline(94.4, color="#2ca02c", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(len(present) - 0.5, 95.2, "regex baseline (94.4%)",
            color="#2ca02c", fontsize=8, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in present], rotation=20, ha="right")
    ax.set_ylabel("Status-match rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Strategy comparison across both predictor checkpoints (N=54 per cell, 95% bootstrap CI)")
    ax.legend(loc="lower left")

    out = FIG_DIR / "fig1_strategy_comparison.tif"
    fig.savefig(out, dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[fig] wrote {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Probe heatmap: turbofan vs simulator
# ─────────────────────────────────────────────────────────────────────────────

def fig2_probe_heatmap_compare():
    t_path = RESULTS_DIR / "probe_cliff_3d_turbofan.csv"
    s_path = RESULTS_DIR / "probe_cliff_3d_simulator.csv"
    if not (t_path.exists() and s_path.exists()):
        print(f"[fig] fig2: missing probe CSVs")
        return
    df_t = pd.read_csv(t_path)
    df_s = pd.read_csv(s_path)

    # Two Xs4 slices each — low and high — to show the cliff structure
    def _pick_slice_vals(df):
        xs4_uniques = sorted(df["xs4_scaled"].unique())
        return [xs4_uniques[1], xs4_uniques[-2]]   # second-lowest, second-highest

    slices_t = _pick_slice_vals(df_t)
    slices_s = _pick_slice_vals(df_s)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)

    for col, (df, slices, label) in enumerate([
        (df_t, slices_t, "Turbofan checkpoint"),
        (df_s, slices_s, "Simulator checkpoint"),
    ]):
        for row, xs4_val in enumerate(slices):
            sl = df[np.isclose(df["xs4_scaled"], xs4_val)]
            pivot = sl.pivot(index="xs3_scaled", columns="xs2_scaled", values="rul")
            ax = axes[row, col]
            im = ax.imshow(
                pivot.values, origin="lower",
                extent=[pivot.columns.min(), pivot.columns.max(),
                        pivot.index.min(), pivot.index.max()],
                aspect="auto", cmap="RdYlGn", vmin=0, vmax=80,
            )
            # Contour the DEGRADED and OFFLINE boundaries
            ax.contour(pivot.columns, pivot.index, pivot.values,
                       levels=[15, 30], colors=["red", "orange"],
                       linewidths=1.2)
            ax.set_title(f"{label}, Xs4={xs4_val:.2f}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Xs2 (Bearing Temp, scaled)")
            if col == 0:
                ax.set_ylabel("Xs3 (Motor Temp, scaled)")

    cbar = fig.colorbar(im, ax=axes, label="Predicted RUL", shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        "RUL response surface: turbofan (bimodal, ~86% at modes) vs simulator (continuous)",
        fontsize=12, y=0.995
    )
    out = FIG_DIR / "fig2_probe_heatmap_compare.tif"
    fig.savefig(out, dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[fig] wrote {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Sequence walkthrough (3-hit MEDIUM)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_sequence_walkthrough():
    abl_path = RESULTS_DIR / "ablation_results_simulator.csv"
    if not abl_path.exists():
        print(f"[fig] fig3: missing {abl_path.name}")
        return
    df = pd.read_csv(abl_path)
    seq = df[df["variant"].isin(["V0_full_sequence", "V3_no_cumulative"])]
    if len(seq) == 0:
        print("[fig] fig3: no sequence rows found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Coloured RUL-band shading
    ax.axhspan(30, 80, color="#2ca02c", alpha=0.10, label="ONLINE band")
    ax.axhspan(15, 30, color="#ff7f0e", alpha=0.15, label="DEGRADED band")
    ax.axhspan(0, 15,  color="#d62728", alpha=0.10, label="OFFLINE band")

    for variant, color, marker, label in [
        ("V0_full_sequence", "#1f77b4", "o", "Cumulative damage ON (production)"),
        ("V3_no_cumulative", "#888888", "s", "Cumulative damage OFF (V3 ablation)"),
    ]:
        sub = seq[seq["variant"] == variant].sort_values("hit")
        if len(sub) == 0: continue
        ax.plot(sub["hit"], sub["rul"], marker=marker, markersize=10,
                linewidth=2, color=color, label=label)
        # Annotate each point with status
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['status']}\nRUL={row['rul']:.1f}",
                xy=(row["hit"], row["rul"]),
                xytext=(8, -10 if variant == "V0_full_sequence" else 10),
                textcoords="offset points",
                fontsize=8, color=color,
            )

    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Hit number (same MEDIUM prompt repeated)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-3, 60)
    ax.set_title("3-hit MEDIUM sequence on Machine 3 (simulator checkpoint)")
    ax.legend(loc="upper right", fontsize=8)

    out = FIG_DIR / "fig3_sequence_walkthrough.tif"
    fig.savefig(out, dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[fig] wrote {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Latency vs accuracy Pareto plot
# ─────────────────────────────────────────────────────────────────────────────

def fig4_latency_vs_accuracy():
    df = _load_baselines("baselines_comparison_simulator.csv")
    if df is None: return
    stats = _per_strategy_stats(df).set_index("strategy")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for strat, row in stats.iterrows():
        color = STRATEGY_COLOR.get(strat, "#444444")
        label = STRATEGY_LABEL.get(strat, strat)
        ax.scatter(row["p95_latency"], row["match_rate"] * 100,
                   s=140, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        # Label each point
        # Manual offsets so labels don't overlap
        offset_x = 1.4 if strat != "keyword_regex_severity" else 1.4
        offset_y = 0 if strat != "fixed_midrange" else -4
        ax.annotate(label, xy=(row["p95_latency"], row["match_rate"] * 100),
                    xytext=(offset_x * row["p95_latency"], row["match_rate"] * 100 + offset_y),
                    fontsize=9, va="center", color=color)

    # Pareto-frontier outline (manual: regex_severity dominates all others)
    pareto = stats.sort_values("p95_latency")
    pareto_x = []
    pareto_y = []
    best = 0.0
    for _, row in pareto.iterrows():
        if row["match_rate"] >= best:
            pareto_x.append(row["p95_latency"])
            pareto_y.append(row["match_rate"] * 100)
            best = row["match_rate"]
    ax.plot(pareto_x, pareto_y, color="black", linestyle="--", alpha=0.4,
            linewidth=1, label="Pareto front", zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(3, 1.5e4)
    ax.set_ylim(20, 105)
    ax.set_xlabel("P95 latency per call (ms, log scale)")
    ax.set_ylabel("Status-match rate (%)")
    ax.set_title("Accuracy vs latency on simulator checkpoint — regex baseline dominates")
    ax.legend(loc="lower left")

    out = FIG_DIR / "fig4_latency_vs_accuracy.tif"
    fig.savefig(out, dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[fig] wrote {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-prompt match matrix (strategy x prompt)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_per_prompt_match_matrix():
    df = _load_baselines("baselines_comparison_simulator.csv")
    if df is None: return
    off = df[df["expected"] == "OFFLINE"].copy()
    if len(off) == 0: return

    # Pivot: rows = strategy, cols = prompt, values = mean match rate
    pivot = off.groupby(["strategy", "prompt"])["match"].mean().unstack("prompt")
    # Order rows per STRATEGY_ORDER
    present_strats = [s for s in STRATEGY_ORDER if s in pivot.index]
    pivot = pivot.reindex(present_strats)
    # Shorten prompt labels
    short_cols = [p[:32] + ("..." if len(p) > 32 else "") for p in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    # Cell annotations
    for i, strat in enumerate(pivot.index):
        for j, _ in enumerate(pivot.columns):
            val = pivot.values[i, j]
            txt = f"{val:.0%}" if not np.isnan(val) else "-"
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(short_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([STRATEGY_LABEL.get(s, s) for s in pivot.index])
    ax.set_title("Per-prompt OFFLINE-match rate (simulator checkpoint)")
    cbar = fig.colorbar(im, ax=ax, label="Match rate", shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    out = FIG_DIR / "fig5_per_prompt_match_matrix.tif"
    fig.savefig(out, dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[fig] wrote {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Sensor selection convergence
# ─────────────────────────────────────────────────────────────────────────────

def fig6_sensor_convergence():
    df = _load_baselines("baselines_comparison_simulator.csv")
    if df is None: return
    off = df[df["expected"] == "OFFLINE"].copy()
    if len(off) == 0 or "sensor_id" not in off.columns:
        return

    # Per (strategy, prompt), modal sensor
    modal = (
        off.groupby(["strategy", "prompt"])["sensor_id"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "NA")
        .unstack("prompt")
    )
    present_strats = [s for s in STRATEGY_ORDER if s in modal.index]
    modal = modal.reindex(present_strats)
    short_cols = [p[:32] + ("..." if len(p) > 32 else "") for p in modal.columns]

    # Build a discrete colour map: one colour per sensor that appears
    all_sensors = sorted(set(modal.values.ravel()) - {"NA"})
    cmap = plt.get_cmap("tab20")
    sensor_color = {s: cmap(i / max(1, len(all_sensors) - 1)) for i, s in enumerate(all_sensors)}

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, strat in enumerate(modal.index):
        for j, prompt in enumerate(modal.columns):
            sensor = modal.values[i, j]
            color = sensor_color.get(sensor, "#dddddd")
            ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                        facecolor=color, edgecolor="black", linewidth=0.6))
            ax.text(j, i, sensor, ha="center", va="center", fontsize=9, color="black")
    ax.set_xlim(-0.5, len(modal.columns) - 0.5)
    ax.set_ylim(-0.5, len(modal.index) - 0.5)
    ax.set_xticks(range(len(modal.columns)))
    ax.set_xticklabels(short_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(modal.index)))
    ax.set_yticklabels([STRATEGY_LABEL.get(s, s) for s in modal.index])
    ax.invert_yaxis()
    ax.set_title("Sensor selection per strategy x OFFLINE-prompt — note LLM convergence on W0 for motor faults")
    # Build a legend
    legend_patches = [Patch(facecolor=sensor_color[s], edgecolor="black", label=s) for s in all_sensors]
    ax.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="Sensor (modal)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(False)

    out = FIG_DIR / "fig6_sensor_convergence.tif"
    fig.savefig(out, dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[fig] wrote {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

FIG_FUNCS = {
    1: fig1_strategy_comparison,
    2: fig2_probe_heatmap_compare,
    3: fig3_sequence_walkthrough,
    4: fig4_latency_vs_accuracy,
    5: fig5_per_prompt_match_matrix,
    6: fig6_sensor_convergence,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Generate only this single figure (debug)")
    args = parser.parse_args()

    if args.fig:
        FIG_FUNCS[args.fig]()
    else:
        for i in sorted(FIG_FUNCS):
            FIG_FUNCS[i]()
    print(f"[fig] done. Output in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
