"""
research/aei_figures.py

Figures for the AEI manuscript (research/AEI/manuscript.md), built from the current result files.

Every figure is written three times: a 300 dpi PNG for drafting, a 700 dpi LZW-compressed TIFF, and
a vector PDF, which is the form the journal prefers for charts. Styling follows Elsevier's artwork guidance: serif type, no figure titles (captions live
in the manuscript), ASCII-only labels, and the Okabe-Ito colour-blind-safe palette.

    Fig 1  pipeline and evaluation protocol schematic                         (drawn here)
    Fig 2  severity accuracy, original wording vs rewrites, paired per arm    anchored_original_vs_rewrites.csv
    Fig 3  accuracy against latency and cost, Pareto view                     anchored_classification.csv + fig 2 source
    Fig 4  common-interface decomposition, own vs shared routing              anchored_interface.csv
    Fig 5  multiplier sweep, best hosted LLM minus regex, three predictors    multiplier_sweep_*.csv
    Fig 6  calibration curves against the PRONOSTIA bearings                  pronostia/calibration_curves.csv
    Fig 7  response-surface geometry of the four predictors                   probe_*.csv
    Fig 8  external work orders: zero-shot, few-shot and in-house training    fmucd/agreement.csv + oracle.csv

Usage:
    python -m research.aei_figures            # all figures
    python -m research.aei_figures --only 2 4 # a subset, by number

Output:
    research/paper_assets/figures_aei/figNN_<name>.{png,tif} and MANIFEST.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = PROJECT_ROOT / "research" / "results"
OUT_DIR = PROJECT_ROOT / "research" / "paper_assets" / "figures_aei"

# Okabe-Ito, colour-blind safe
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#666666")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed TrueType, as the journal asks for vector artwork
})

HOSTED = ["gemini_2_5_flash__production", "gemini_2_5_flash_nothink__production",
          "gemini_3_5_flash__production", "gpt_oss_120b__production", "qwen3_8_27b__production"]
LOCAL = ["llama3_2_3b__production", "gemma3_4b__production", "qwen3_4b__production"]
TRAINED_R1 = ["tfidf_logreg", "minilm_logreg", "bge_m3_logreg", "minilm_finetuned"]
TRAINED_R2 = [a + "_indomain" for a in TRAINED_R1]
RULES = ["keyword_regex_severity", "keyword_regex_extended", "embedding_severity"]

LABEL = {
    "gemini_2_5_flash__production": "Gemini 2.5 Flash",
    "gemini_2_5_flash_nothink__production": "Gemini 2.5 Flash (no thinking)",
    "gemini_3_5_flash__production": "Gemini 3.5 Flash",
    "gpt_oss_120b__production": "gpt-oss-120b",
    "qwen3_8_27b__production": "Qwen 3.8 27B",
    "llama3_2_3b__production": "Llama 3.2 3B (local)",
    "gemma3_4b__production": "Gemma 3 4B (local)",
    "qwen3_4b__production": "Qwen 3 4B (local)",
    "tfidf_logreg": "TF-IDF", "minilm_logreg": "MiniLM", "bge_m3_logreg": "bge-m3",
    "minilm_finetuned": "MiniLM fine-tuned",
    "tfidf_logreg_indomain": "TF-IDF + reports", "minilm_logreg_indomain": "MiniLM + reports",
    "bge_m3_logreg_indomain": "bge-m3 + reports", "minilm_finetuned_indomain": "MiniLM f.t. + reports",
    "keyword_regex_severity": "Keyword regex", "keyword_regex_extended": "WordNet regex",
    "embedding_severity": "Prototype embedding",
    "bge_m3_logreg_external": "bge-m3 + our reports", "tfidf_logreg_external": "TF-IDF + our reports",
}
FAMILY_COLOUR = {"hosted": BLUE, "local": VERMILLION, "trained_r1": GREEN, "trained_r2": PURPLE, "rule": GREY}


def family(arm: str) -> str:
    if arm in HOSTED:
        return "hosted"
    if arm in LOCAL:
        return "local"
    if arm in TRAINED_R2:
        return "trained_r2"
    if arm in TRAINED_R1:
        return "trained_r1"
    return "rule"


def spread(values: list[float], min_gap: float, lo: float, hi: float) -> list[float]:
    """Push overlapping label positions apart while keeping their order and staying inside [lo, hi]."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for pos, i in enumerate(order):                      # upward pass
        if pos and out[i] - out[order[pos - 1]] < min_gap:
            out[i] = out[order[pos - 1]] + min_gap
    for pos in range(len(order) - 1, -1, -1):            # pull back inside the axis
        i = order[pos]
        if out[i] > hi:
            out[i] = hi - (len(order) - 1 - pos) * min_gap
    for pos, i in enumerate(order):
        if out[i] < lo:
            out[i] = lo + pos * min_gap
    return out


def place_labels(ax, xs, ys, texts, colours, fontsize=5.8):
    """Annotate points, choosing for each label the first offset that does not overlap an earlier one.

    Collision is checked in axes-fraction space with a rough text box, which is enough to keep a
    scatter of about twenty labels readable without a layout library.
    """
    boxes: list[tuple[float, float, float, float]] = []
    height = 0.045
    candidates = [(0.012, 0.010), (0.012, -0.050), (-0.012, 0.010), (-0.012, -0.050),
                  (0.012, 0.055), (0.012, -0.095), (-0.012, 0.055), (-0.012, -0.095)]
    for x, y, text, colour in zip(xs, ys, texts, colours):
        fx, fy = ax.transLimits.transform(ax.transScale.transform((x, y)))
        width = 0.0095 * len(text)
        chosen = None
        for dx, dy in candidates:
            left = fx + dx if dx > 0 else fx + dx - width
            rect = (left, fy + dy, left + width, fy + dy + height)
            if rect[0] < -0.02 or rect[2] > 1.02:
                continue
            if any(not (rect[2] < b[0] or rect[0] > b[2] or rect[3] < b[1] or rect[1] > b[3]) for b in boxes):
                continue
            chosen = (rect, dx > 0)
            break
        if chosen is None:
            rect, right = (fx + 0.012, fy + 0.010, fx + 0.012 + width, fy + 0.055), True
        else:
            rect, right = chosen
        boxes.append(rect)
        ax.annotate(text, (rect[0] if right else rect[2], rect[1] + height / 2), xycoords="axes fraction",
                    ha="left" if right else "right", va="center", fontsize=fontsize, color=colour)
        ax.plot([fx, rect[0] if right else rect[2]], [fy, rect[1] + height / 2], transform=ax.transAxes,
                linewidth=0.4, color=colour, alpha=0.5, zorder=2)


MANIFEST: list[dict] = []


def save(fig, number: int, name: str, sources: list[str], caption: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"fig{number}_{name}"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tif"), dpi=700, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".pdf"))     # vector, the journal's preferred form for charts
    plt.close(fig)
    MANIFEST.append({"number": number, "name": name, "sources": sources, "caption": caption})
    print(f"[figures] fig{number} {name}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    stages = [("Operator\nreport", GREY), ("Input\nguard", GREY), ("Severity\ninterface", BLUE),
              ("Multi-sensor\ninjection", GREY), ("RUL\npredictor", GREY), ("Capacity and\ndispatch", GREY)]
    x = 2.0
    for i, (text, colour) in enumerate(stages):
        box = FancyBboxPatch((x, 22), 13, 12, boxstyle="round,pad=0.4", linewidth=1.0,
                             edgecolor=colour, facecolor="white" if colour == GREY else "#E8F1F8")
        ax.add_patch(box)
        ax.text(x + 6.5, 28, text, ha="center", va="center", fontsize=7.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 13, 28), (x + 16, 28), arrowstyle="-|>", mutation_scale=8,
                                         linewidth=0.8, color="black"))
        x += 16
    ax.text(34.5, 17.0, "the component under test", ha="center", fontsize=7, color=BLUE)
    ax.add_patch(FancyArrowPatch((34.5, 19.0), (34.5, 21.6), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.8, color=BLUE))
    arms = ("Interfaces compared: 5 hosted LLM configurations | 3 local LLMs (4 GB GPU) | "
            "keyword and WordNet regex | prototype embedding |\ntrained classifiers on prompt vocabulary, "
            "on labelled reports, and on another site's reports")
    ax.text(50, 12, arms, ha="center", va="center", fontsize=7)
    tests = ("Test sets: 18 original prompts | 72 anchored rewrites | 42 design prompts | 105 machine-generated typos | "
             "300 real work orders\nPredictors: turbofan | simulator | calibrated simulator | PRONOSTIA bearings")
    ax.text(50, 4, tests, ha="center", va="center", fontsize=7)
    save(fig, 1, "pipeline_protocol", ["drawn in research/aei_figures.py"],
         "The five-stage pipeline. Only the severity interface is varied: every arm emits the same three fields "
         "(sensor, severity band, magnitude) and is replayed through identical downstream code.")


def fig2_original_vs_rewrites():
    df = pd.read_csv(R / "paraphrase" / "anchored_original_vs_rewrites.csv")
    df = df[df["metric"] == "severity accuracy"].set_index("arm")
    arms = HOSTED + LOCAL + TRAINED_R1 + TRAINED_R2 + RULES
    arms = [a for a in arms if a in df.index]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ends = [df.loc[a, "rewrites"] * 100 for a in arms]
    label_y = spread(ends, 3.0, 31, 104)
    for arm, y_end, y_lab in zip(arms, ends, label_y):
        r = df.loc[arm]
        colour = FAMILY_COLOUR[family(arm)]
        ax.plot([0, 1], [r["original"] * 100, y_end], marker="o", markersize=3.2,
                linewidth=1.1, color=colour, alpha=0.9)
        ax.plot([1.0, 1.08], [y_end, y_lab], linewidth=0.5, color=colour, alpha=0.6)
        ax.text(1.10, y_lab, LABEL.get(arm, arm), fontsize=6.2, va="center", color=colour)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original wording", "Rewrites"])
    ax.set_xlim(-0.12, 1.75)
    ax.set_ylim(30, 106)
    ax.set_ylabel("Severity accuracy (per cent)")
    handles = [plt.Line2D([], [], color=c, marker="o", markersize=3.2, linewidth=1.1, label=l)
               for l, c in (("Hosted LLM", BLUE), ("Local LLM", VERMILLION), ("Trained, prompt vocabulary", GREEN),
                            ("Trained, labelled reports", PURPLE), ("Rule based", GREY))]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    save(fig, 2, "original_vs_rewrites", ["research/results/paraphrase/anchored_original_vs_rewrites.csv"],
         "Severity accuracy on the 18 original prompts and on their 72 rewrites, paired by arm. Hosted language "
         "models hold their accuracy; rule-based and vocabulary-trained arms fall; labelled reports recover most "
         "of the loss.")


def fig3_pareto():
    cls = pd.read_csv(R / "paraphrase" / "anchored_classification.csv").set_index("arm")
    rew = pd.read_csv(R / "paraphrase" / "anchored_original_vs_rewrites.csv")
    rew = rew[rew["metric"] == "severity accuracy"].set_index("arm")["rewrites"]
    arms = [a for a in HOSTED + LOCAL + TRAINED_R1 + TRAINED_R2 + RULES if a in cls.index and a in rew.index]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = [cls.loc[a, "p50_latency_ms"] for a in arms]
    ys = [rew[a] * 100 for a in arms]
    for arm, x, y in zip(arms, xs, ys):
        usd = cls.loc[arm, "usd_per_1000"]
        size = 18 + 42 * (0 if pd.isna(usd) else min(usd, 10) / 10)
        ax.scatter(x, y, s=size, color=FAMILY_COLOUR[family(arm)], alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.45, 6e4)
    ax.set_ylim(33, 106)
    place_labels(ax, xs, ys, [LABEL.get(a, a) for a in arms], [FAMILY_COLOUR[family(a)] for a in arms])
    ax.set_xlabel("Median latency per report (ms, log scale)")
    ax.text(0.015, 0.99, "Marker size: interface cost per 1,000 reports (0 to 10 US dollars)",
            transform=ax.transAxes, va="top", fontsize=6.2, color="black")
    ax.set_ylabel("Severity accuracy on rewrites (per cent)")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    save(fig, 3, "accuracy_latency_cost",
         ["research/results/paraphrase/anchored_classification.csv",
          "research/results/paraphrase/anchored_original_vs_rewrites.csv"],
         "Accuracy on reworded reports against median latency, with marker size showing interface cost. The "
         "classifier given labelled reports sits within a few points of the hosted models at a fraction of the "
         "latency and no interface cost.")


def fig4_interface():
    df = pd.read_csv(R / "paraphrase" / "anchored_interface.csv")
    df = df[df["scope"] == "unfamiliar"]
    checkpoints = ["turbofan", "simulator", "simulator_calibrated"]
    arms = [a for a in HOSTED + LOCAL if a in set(df["arm"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    for ax, ck in zip(axes, checkpoints):
        sub = df[df["checkpoint"] == ck].set_index("arm")
        for i, arm in enumerate(arms):
            r = sub.loc[arm]
            colour = FAMILY_COLOUR[family(arm)]
            ax.plot([0, 1], [r["own"] * 100, r["common"] * 100], marker="o", markersize=3,
                    linewidth=1.0, color=colour, alpha=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Own\nrouting", "Shared\nrouting"])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title({"turbofan": "Turbofan", "simulator": "Simulator",
                      "simulator_calibrated": "Calibrated simulator"}[ck])
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("Status agreement on rewrites (per cent)")
    axes[0].set_ylim(35, 104)
    handles = [plt.Line2D([], [], color=c, marker="o", markersize=3, linewidth=1.0, label=l)
               for l, c in (("Hosted LLM", BLUE), ("Local LLM", VERMILLION))]
    axes[2].legend(handles=handles, loc="lower right", frameon=False)
    save(fig, 4, "interface_decomposition", ["research/results/paraphrase/anchored_interface.csv"],
         "Decision-level agreement when each arm uses its own sensor routing and when every arm shares the same "
         "routing. The hosted models' deficit on the simulator is a routing artefact, not a language failure.")


def fig5_sweep():
    files = [("Turbofan", R / "multiplier_sweep_turbofan.csv"),
             ("Simulator", R / "multiplier_sweep_simulator.csv"),
             ("Calibrated simulator", R / "pronostia" / "multiplier_sweep_simulator_calibrated.csv")]
    hosted_sweep = ["agentic", "gemini_3_5_flash", "groq_gpt_oss_120b", "groq_qwen3_8"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    mesh = None
    for ax, (title, path) in zip(axes, files):
        sw = pd.read_csv(path)
        sw = sw[sw["low"] == 0.15]
        piv = sw.pivot_table(index=["medium", "high"], columns="strategy", values="match_rate")
        gap = (piv[[c for c in hosted_sweep if c in piv.columns]].max(axis=1)
               - piv["keyword_regex_severity"]).unstack("high") * 100
        mesh = ax.pcolormesh(gap.columns.values, gap.index.values, gap.values, cmap="RdBu", vmin=-20, vmax=20,
                             shading="nearest")
        ax.scatter([0.85], [0.35], marker="*", s=40, color="black", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("HIGH multiplier")
    axes[0].set_ylabel("MEDIUM multiplier")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Best hosted LLM minus regex (points)")
    save(fig, 5, "multiplier_sweep",
         ["research/results/multiplier_sweep_turbofan.csv", "research/results/multiplier_sweep_simulator.csv",
          "research/results/pronostia/multiplier_sweep_simulator_calibrated.csv"],
         "Status agreement of the best hosted language model minus the keyword regular expression across the "
         "injection-constant grid, on the 18 original prompts. The star marks the published setting. The sign of "
         "the difference is a function of the constants.")


def fig6_calibration():
    cur = pd.read_csv(R / "pronostia" / "calibration_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharex=True)
    for ax, (chan, name) in zip(axes, (("vib", "Vibration (normalised)"), ("temp", "Temperature (normalised)"))):
        ax.plot(cur["life_frac"], cur[f"pronostia_{chan}_median"], color="black", linewidth=1.4, label="PRONOSTIA bearings")
        ax.plot(cur["life_frac"], cur[f"original_{chan}_median"], color=VERMILLION, linewidth=1.1,
                linestyle="--", label="Simulator, before")
        ax.plot(cur["life_frac"], cur[f"calibrated_{chan}_median"], color=GREEN, linewidth=1.1,
                linestyle="-.", label="Simulator, calibrated")
        ax.set_xlabel("Fraction of life")
        ax.set_ylabel(name)
        ax.grid(linewidth=0.3, alpha=0.5)
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, 6, "calibration_curves", ["research/results/pronostia/calibration_curves.csv"],
         "Median degradation curves of the real bearings and of the simulator before and after fitting the "
         "late-onset law. Vibration curve error falls from 0.54 to 0.03 and temperature error from 0.32 to 0.10.")


def fig7_geometry():
    sources = [("Turbofan", R / "probe_cliff_3d_turbofan.csv"), ("Simulator", R / "probe_cliff_3d_simulator.csv"),
               ("Calibrated simulator", R / "pronostia" / "probe_simulator_calibrated.csv"),
               ("PRONOSTIA", R / "pronostia" / "probe_pronostia.csv")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)
    for ax, (title, path) in zip(axes, sources):
        rul = pd.read_csv(path)["rul"].to_numpy()
        ax.hist(rul, bins=40, color=BLUE, alpha=0.85)
        ax.axvspan(0, 15, color=VERMILLION, alpha=0.18, linewidth=0)
        ax.axvspan(15, 30, color=YELLOW, alpha=0.3, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Predicted RUL (cycles)")
    axes[0].set_ylabel("Probe points")
    axes[0].text(0.5, 0.92, "OFFLINE", transform=axes[0].transAxes, fontsize=6, color=VERMILLION, ha="left")
    save(fig, 7, "predictor_geometry",
         ["research/results/probe_cliff_3d_turbofan.csv", "research/results/probe_cliff_3d_simulator.csv",
          "research/results/pronostia/probe_simulator_calibrated.csv",
          "research/results/pronostia/probe_pronostia.csv"],
         "Distribution of predicted remaining useful life over the probe grid for each predictor, with the "
         "DEGRADED band shaded yellow and the OFFLINE band red. The real-bearing predictor never enters the "
         "OFFLINE band, so no interface can trigger a shutdown on it.")


def fig8_external():
    agree = pd.read_csv(R / "fmucd" / "agreement.csv").set_index("arm")
    oracle = pd.read_csv(R / "fmucd" / "oracle.csv").set_index("model")
    rows = []
    for arm in HOSTED + LOCAL:
        if arm in agree.index:
            rows.append((LABEL.get(arm, arm), agree.loc[arm, "agreement"] * 100,
                         agree.loc[arm + "+fewshot", "agreement"] * 100 if arm + "+fewshot" in agree.index else np.nan,
                         family(arm)))
    for arm in ["keyword_regex_severity", "bge_m3_logreg", "bge_m3_logreg_external"]:
        if arm in agree.index:
            rows.append((LABEL.get(arm, arm), agree.loc[arm, "agreement"] * 100, np.nan,
                         "rule" if "regex" in arm else "trained_r1"))
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [r[1] for r in rows], width=0.36, color=[FAMILY_COLOUR[r[3]] for r in rows],
           label="zero-shot")
    ax.bar(x + 0.19, [r[2] for r in rows], width=0.36, color=[FAMILY_COLOUR[r[3]] for r in rows], alpha=0.45,
           hatch="//", label="with 20 examples from the site")
    ax.axhline(100 / 3, color="black", linewidth=0.8, linestyle=":", label="chance")
    ax.axhline(oracle["agreement"].max() * 100, color=GREEN, linewidth=1.1, linestyle="-",
               label="trained on 6,000 of the site's own work orders")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right")
    ax.set_ylabel("Agreement with the site's band (per cent)")
    ax.set_ylim(0, 88)
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=6.5)
    save(fig, 8, "external_work_orders",
         ["research/results/fmucd/agreement.csv", "research/results/fmucd/oracle.csv"],
         "Agreement with each organisation's own priority band on 300 real work orders. Every strategy trained "
         "elsewhere sits near chance, in-context examples recover part of the convention, and a classifier "
         "trained on the site's own records recovers most of it.")


def graphical_abstract():
    """The journal asks for one image, at least 531 x 1328 px, readable at 5 x 13 cm.

    It is not numbered with the figures and is uploaded as its own submission item, so it is
    written outside the fig* naming convention and carries a heading, which the figures do not.
    """
    df = pd.read_csv(R / "paraphrase" / "anchored_original_vs_rewrites.csv")
    df = df[df["metric"] == "severity accuracy"].set_index("arm")
    pct = lambda arm, col: df.loc[arm, col] * 100
    hosted = lambda col: float(np.mean([pct(a, col) for a in HOSTED]))

    panels = [
        ("The pipeline's own vocabulary",
         [("Hosted LLM", hosted("original"), BLUE),
          ("Trained\nclassifier", pct("tfidf_logreg", "original"), GREEN),
          ("Keyword\nregex", pct("keyword_regex_severity", "original"), GREY)]),
        ("Operators reword the same fault",
         [("Hosted LLM", hosted("rewrites"), BLUE),
          ("Trained on\nvocabulary", pct("bge_m3_logreg", "rewrites"), GREEN),
          ("Keyword\nregex", pct("keyword_regex_severity", "rewrites"), GREY)]),
        ("Plus about 130 labelled reports",
         [("Hosted LLM", hosted("rewrites"), BLUE),
          ("Trained on\nreports", pct("bge_m3_logreg_indomain", "rewrites"), PURPLE),
          ("Keyword\nregex", pct("keyword_regex_severity", "rewrites"), GREY)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.23, 2.09), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.72, bottom=0.28, wspace=0.16)
    for ax, (heading, bars) in zip(axes, panels):
        for i, (name, value, colour) in enumerate(bars):
            ax.bar(i, value, width=0.62, color=colour, edgecolor="none")
            ax.text(i, value + 2.5, f"{value:.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=5.6)
        ax.set_title(heading, fontsize=6.6, pad=4)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=6)
    axes[0].set_ylabel("Severity accuracy (per cent)", fontsize=6.4)
    fig.text(0.5, 0.955, "When does a language model earn its place in a maintenance pipeline?",
             ha="center", fontsize=8.2)
    fig.text(0.5, 0.045, "Only when operators write freely and no labelled history exists. "
                         "A CPU classifier does the rest.",
             ha="center", fontsize=6.4, color="#333333")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "graphical_abstract"
    with plt.rc_context({"savefig.bbox": None}):      # exact canvas, so the aspect ratio is kept
        for suffix, dpi, kwargs in ((".png", 300, {}), (".tif", 700, {"compression": "tiff_lzw"})):
            fig.savefig(stem.with_suffix(suffix), dpi=dpi, pil_kwargs=kwargs)
        fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[figures] graphical abstract: " + ", ".join(
        f"{n} {v:.1f}" for _, bars in panels for n, v, _ in bars), flush=True)


FIGURES = {1: fig1_pipeline, 2: fig2_original_vs_rewrites, 3: fig3_pareto, 4: fig4_interface,
           5: fig5_sweep, 6: fig6_calibration, 7: fig7_geometry, 8: fig8_external}


def write_manifest() -> Path:
    lines = ["# AEI figures\n", f"**Generated by:** `python -m research.aei_figures` on "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
             "300 dpi PNG for drafting, 700 dpi LZW TIFF for submission. Serif type, no in-figure titles, "
             "Okabe-Ito colour-blind-safe palette.\n"]
    for m in sorted(MANIFEST, key=lambda x: x["number"]):
        lines.append(f"\n## Figure {m['number']}: fig{m['number']}_{m['name']}\n")
        lines.append(f"**Caption.** {m['caption']}\n")
        lines.append("**Sources:** " + ", ".join(f"`{s}`" for s in m["sources"]))
    path = OUT_DIR / "MANIFEST.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", type=int, choices=sorted(FIGURES), default=None)
    parser.add_argument("--graphical-abstract", action="store_true",
                        help="rebuild only the graphical abstract, which is not a numbered figure")
    args = parser.parse_args()
    if args.graphical_abstract:
        graphical_abstract()
        return
    for number in args.only or sorted(FIGURES):
        FIGURES[number]()
    if not args.only:
        graphical_abstract()
        print(f"[figures] wrote {write_manifest().relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
