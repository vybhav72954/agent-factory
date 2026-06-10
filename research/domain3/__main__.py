"""
Run the full Bayesian-anomaly evaluation across all strategies.

Usage:
  python -m research.domain3
  python -m research.domain3 --stability-runs 3
  python -m research.domain3 --only-strategy agentic
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .bayesian_anomaly import (
    FAILURE_MODES,
    OBS_WINDOW_LEN,
    PriorUpdate,
    score,
    synthesize_observations,
)
from .evaluation import TEST_PROMPTS, labels_summary
from .strategies import STRATEGIES, run_one


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "domain3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(strategy_name: str, n_repeats: int, base_seed: int = 0) -> List[dict]:
    rows: List[dict] = []
    for repeat in range(n_repeats):
        for prompt_idx, (text, true_mode) in enumerate(TEST_PROMPTS):
            obs_seed = base_seed + repeat * 1000 + prompt_idx
            observations = synthesize_observations(true_mode, n_samples=OBS_WINDOW_LEN, seed=obs_seed)
            try:
                update, prior_used, posterior_dict, latency_ms = run_one(
                    strategy_name, text, observations
                )
                brier, nll, top1 = score(posterior_dict, true_mode)
                rows.append({
                    "strategy":        strategy_name,
                    "repeat":          repeat,
                    "prompt_idx":      prompt_idx,
                    "prompt":          text,
                    "true_mode":       true_mode,
                    "predicted_mode":  max(posterior_dict, key=lambda m: posterior_dict[m]),
                    "p_true":          posterior_dict[true_mode],
                    "brier":           brier,
                    "nll":             nll,
                    "top1_correct":    int(top1),
                    "update_mode":     update.failure_mode,
                    "update_delta":    update.prior_weight_delta,
                    "update_conf":     update.confidence,
                    "latency_ms":      latency_ms,
                    "error":           "",
                })
            except Exception as e:
                rows.append({
                    "strategy":        strategy_name,
                    "repeat":          repeat,
                    "prompt_idx":      prompt_idx,
                    "prompt":          text,
                    "true_mode":       true_mode,
                    "predicted_mode":  "",
                    "p_true":          0.0,
                    "brier":           2.0,
                    "nll":             20.0,
                    "top1_correct":    0,
                    "update_mode":     "",
                    "update_delta":    0.0,
                    "update_conf":     0.0,
                    "latency_ms":      0.0,
                    "error":           f"{type(e).__name__}: {e}",
                })
    return rows


def aggregate(rows: List[dict]) -> Dict[str, dict]:
    """Per-strategy averages."""
    by_strat: Dict[str, List[dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r)
    out: Dict[str, dict] = {}
    for s, lst in by_strat.items():
        n = len(lst)
        if n == 0:
            continue
        briers = np.array([r["brier"] for r in lst])
        nlls   = np.array([r["nll"] for r in lst])
        top1s  = np.array([r["top1_correct"] for r in lst])
        lats   = np.array([r["latency_ms"] for r in lst])
        out[s] = {
            "n":             n,
            "brier_mean":    float(briers.mean()),
            "brier_std":     float(briers.std()),
            "nll_mean":      float(nlls.mean()),
            "nll_std":       float(nlls.std()),
            "top1_rate":     float(top1s.mean()),
            "latency_mean":  float(lats.mean()),
            "latency_p95":   float(np.percentile(lats, 95)),
            "n_errors":      int(sum(1 for r in lst if r["error"])),
        }
    return out


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_md(agg: Dict[str, dict], path: Path) -> None:
    order = ["uniform", "keyword_only", "keyword_regex", "agentic"]
    rows = [s for s in order if s in agg]

    lines: List[str] = []
    lines.append("# Domain 3 — Bayesian Anomaly Classification\n")
    lines.append("**Domain:** Industrial anomaly classifier with LLM-emitted prior updates.\n")
    lines.append("**Predictor:** Frozen Gaussian Naive Bayes trained on synthetic per-mode multivariate Gaussian observations.\n")
    lines.append("**LLM role:** emits a probability shift over five failure-mode priors from operator natural-language descriptions. The shift is combined with the classifier's likelihood via standard Bayes update.\n")
    lines.append("**Test set:** 20 operator descriptions balanced across 5 modes (4 per mode).\n")
    lines.append(f"**Per-mode label balance:** {labels_summary()}\n")
    lines.append("**Metrics:** Brier score (lower=better, perfect=0, worst=2), NLL (lower=better), top-1 accuracy (higher=better).\n\n")

    lines.append("## Headline results\n")
    lines.append("| Strategy | Top-1 acc | Brier (mean) | NLL (mean) | Latency P95 | N | Errors |")
    lines.append("|----------|-----------|--------------|------------|-------------|---|--------|")
    for s in rows:
        a = agg[s]
        lines.append(
            f"| `{s}` | {a['top1_rate']:.1%} | {a['brier_mean']:.3f} ± {a['brier_std']:.3f} | "
            f"{a['nll_mean']:.3f} ± {a['nll_std']:.3f} | {a['latency_p95']:.1f} ms | {a['n']} | {a['n_errors']} |"
        )

    lines.append("\n## Interpretation\n")
    if "uniform" in agg and "agentic" in agg and "keyword_regex" in agg and "keyword_only" in agg:
        u  = agg["uniform"]
        k0 = agg["keyword_only"]
        k1 = agg["keyword_regex"]
        a  = agg["agentic"]
        lines.append(
            f"The uniform-prior baseline (likelihood-only, no operator shift) lands at "
            f"Brier {u['brier_mean']:.3f}, top-1 {u['top1_rate']:.1%}. The classifier alone is not enough — "
            f"operator information is load-bearing.\n"
        )
        lines.append(
            f"The deterministic strategies recover most of the gap: "
            f"`keyword_only` reaches top-1 {k0['top1_rate']:.1%} (Brier {k0['brier_mean']:.3f}), "
            f"`keyword_regex` reaches top-1 {k1['top1_rate']:.1%} (Brier {k1['brier_mean']:.3f}). "
            f"The agentic Gemini call lands at top-1 {a['top1_rate']:.1%} (Brier {a['brier_mean']:.3f}) "
            f"at a {a['latency_p95']:.0f} ms P95 cost.\n"
        )

        gap_brier_vs_k0 = a['brier_mean'] - k0['brier_mean']
        gap_top1_vs_k0  = a['top1_rate']  - k0['top1_rate']
        gap_brier_vs_k1 = a['brier_mean'] - k1['brier_mean']
        gap_top1_vs_k1  = a['top1_rate']  - k1['top1_rate']

        if gap_top1_vs_k0 >= 0.05 and gap_brier_vs_k0 <= -0.02:
            verdict = ("the LLM materially beats both deterministic baselines on this domain — "
                       "a notable break from the main-pipeline finding.")
        elif (abs(gap_top1_vs_k0) < 0.05) and (abs(gap_brier_vs_k0) < 0.05):
            verdict = ("the LLM ties the deterministic baselines on both metrics. The main-pipeline "
                       "finding (the regex baseline is competitive with the LLM) generalises to this "
                       "qualitatively different domain — output type, evaluation metric, and predictor "
                       "are all different here, yet the bottom-line ordering is the same.")
        else:
            verdict = ("the LLM is competitive with the deterministic baselines on one metric but "
                       "not the other. The main-pipeline finding (LLM does not clearly dominate the "
                       "regex baseline) generalises with some nuance.")

        lines.append(f"\n**Verdict:** {verdict}\n")
        lines.append(
            f"\n**Numerical detail:** agentic vs keyword_only — top-1 Δ = {gap_top1_vs_k0:+.1%}, "
            f"Brier Δ = {gap_brier_vs_k0:+.3f}. agentic vs keyword_regex — top-1 Δ = {gap_top1_vs_k1:+.1%}, "
            f"Brier Δ = {gap_brier_vs_k1:+.3f}. The LLM's latency cost vs deterministic strategies is "
            f"{a['latency_p95']:.0f}ms P95 vs {max(k0['latency_p95'], k1['latency_p95']):.1f}ms P95 — "
            f"a {a['latency_p95'] / max(0.001, max(k0['latency_p95'], k1['latency_p95'])):.0f}x slowdown.\n"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability-runs", type=int, default=3, help="repeats per prompt")
    ap.add_argument("--only-strategy", type=str, default=None, help="evaluate just this one")
    args = ap.parse_args()

    print(f"[domain3] test set: {len(TEST_PROMPTS)} prompts, {labels_summary()}")
    if args.only_strategy:
        if args.only_strategy not in STRATEGIES:
            print(f"unknown strategy {args.only_strategy!r}; choose from {list(STRATEGIES)}")
            return 2
        strategies_to_run = [args.only_strategy]
    else:
        strategies_to_run = list(STRATEGIES)

    all_rows: List[dict] = []
    total_runs = sum(len(TEST_PROMPTS) * args.stability_runs for _ in strategies_to_run)
    print(f"[domain3] {total_runs} pipeline calls across {len(strategies_to_run)} strategies")

    for s in strategies_to_run:
        print(f"[domain3] strategy: {s}")
        t0 = time.perf_counter()
        rows = evaluate(s, args.stability_runs)
        all_rows.extend(rows)
        elapsed = time.perf_counter() - t0
        avg_brier = float(np.mean([r["brier"] for r in rows]))
        top1 = float(np.mean([r["top1_correct"] for r in rows]))
        print(f"           done in {elapsed:.1f}s   brier={avg_brier:.3f}   top1={top1:.1%}")

    csv_path = RESULTS_DIR / "bayesian_results.csv"
    write_csv(all_rows, csv_path)
    print(f"[domain3] wrote {csv_path} ({len(all_rows)} rows)")

    agg = aggregate(all_rows)
    summary_path = RESULTS_DIR / "bayesian_summary.md"
    write_summary_md(agg, summary_path)
    print(f"[domain3] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
