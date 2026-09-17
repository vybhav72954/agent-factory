"""
research/paraphrase/score.py

Score a paraphrase prompt bank.

Banks (`--bank`):
    design    prompt_bank.py: 42 prompts, ground truth = design_severity
    anchored  anchored_bank.py: 18 TEST_PROMPTS anchors + 4 rewrites each; ground truth
              inherited from the anchor's TEST_PROMPTS group and expected status
    typo      typo_bank.py: the 18 anchors with seeded keyboard typos; labels inherited

Expected status follows TEST_PROMPTS: HIGH -> OFFLINE, LOW or MEDIUM -> ONLINE.

Arms:
  - non-LLM (computed here, no API): keyword_regex_severity, keyword_regex_extended
    (WordNet-expanded), embedding_severity (local MiniLM prototypes), keyword_only, fixed_midrange,
    and the trained classifiers of research/trained_baselines.py: regime R1 (prompt vocabulary)
    on every bank, regime R2 (`*_indomain`, cross-validated) on the design and anchored banks
  - LLM: every arm collected by collect.py (hosted models x production/semantic prompt, Gemini 2.5
    Flash thinking off, and three local models via Ollama)

Metrics:
  - severity accuracy (3-class), HIGH recall, over-escalation, fallback rate
  - status-match after replay at the published multipliers on every available checkpoint,
    with 95% bootstrap CIs resampling prompts (design) or anchors (anchored, typo)
  - hypothesis families fixed in research/AEI/analysis_plan_additions.md, section E:
    H1 hosted production-prompt LLMs vs regex and prototype embedding (design, anchored);
    H2 all production-prompt LLMs vs bge_m3_logreg R1 and R2 (design, anchored);
    H3 all production-prompt LLMs vs regex and bge_m3_logreg R1 (typo)
  - paired comparisons of every LLM arm against the regex and the embedding classifier:
    mean difference, 95% cluster-bootstrap CI, sign-flip permutation p, Holm-adjusted p
  - anchored bank only: original wording vs rewrites for every arm, paired by anchor
  - common-interface decomposition: status match when every arm's severity goes through the
    shared keyword routing and band-centre spike values, vs the arm's own routing and value
  - latency and, where token usage was recorded, cost per 1,000 diagnoses
  - breakdown by prompt style

Usage:
    python -m research.paraphrase.score --bank design
    python -m research.paraphrase.score --bank anchored
    python -m research.paraphrase.score --bank typo

Output:
    research/results/paraphrase/{paraphrase,anchored,typo}_summary.md
    research/results/paraphrase/{paraphrase,anchored,typo}_scores.csv
    research/results/paraphrase/{paraphrase,anchored,typo}_rows.csv.gz
    research/results/paraphrase/{prefix}_{primary,h2,h3,interface,...}.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.paraphrase.collect import ARMS, HOSTED_ARMS, LOCAL_ARMS, load_bank, load_outputs
from research.trained_baselines import ARMS as TRAINED_ARMS, IN_DOMAIN_SUFFIX, REFERENCE_ARM, indomain_path
from research.replay_utils import (
    CALIBRATED_WEIGHTS,
    PRONOSTIA_WEIGHTS,
    PUBLISHED_MULTIPLIERS,
    bootstrap_ci,
    healthy_baselines,
    holm_adjust,
    paired_difference,
    replay,
    select_checkpoint,
    spikes_from_frame,
)

OUT_DIR = PROJECT_ROOT / "research" / "results" / "paraphrase"
OUT_PREFIX = {"design": "paraphrase", "anchored": "anchored", "typo": "typo"}
RULE_BASED = ["keyword_regex_severity", "keyword_regex_extended", "embedding_severity", "keyword_only", "fixed_midrange"]
INDOMAIN_ARMS = [a + IN_DOMAIN_SUFFIX for a in TRAINED_ARMS]
NON_LLM = RULE_BASED + TRAINED_ARMS + INDOMAIN_ARMS          # computed here; R2 arms only where predictions exist
REFERENCES = ["keyword_regex_severity", "embedding_severity"]
# Hypothesis families (analysis_plan_additions.md section E): name -> (banks, arms, references)
PRODUCTION_ARMS = [a for a in ARMS if a.endswith("__production")]
FAMILIES = {
    "H1": (("design", "anchored"), [a for a in HOSTED_ARMS if a.endswith("__production")], REFERENCES),
    "H2": (("design", "anchored"), PRODUCTION_ARMS, [REFERENCE_ARM, REFERENCE_ARM + IN_DOMAIN_SUFFIX]),
    "H3": (("typo",), PRODUCTION_ARMS, ["keyword_regex_severity", REFERENCE_ARM]),
}
FAMILY_FILES = {"H1": "primary", "H2": "h2", "H3": "h3"}
REPEATS = 3
SEED = 20260917

# USD per 1M tokens, standard paid tier, retrieved 2026-09-17 from
# https://ai.google.dev/gemini-api/docs/pricing (page updated 2026-09-16) and
# https://console.groq.com/docs/models. Output includes thinking/reasoning tokens.
PRICES_PER_M = {
    "gemini_2_5_flash": (0.30, 2.50),
    "gemini_2_5_flash_nothink": (0.30, 2.50),
    "gemini_3_5_flash": (1.50, 9.00),
    "gpt_oss_120b":     (0.15, 0.60),
    "qwen3_8_27b":      (0.80, 4.00),
    # Local models: no API charge (quantised weights on a GTX 1650 4 GB; hardware cost not included)
    "llama3_2_3b":      (0.0, 0.0),
    "gemma3_4b":        (0.0, 0.0),
    "qwen3_4b":         (0.0, 0.0),
}
LOCAL_LATENCY_CSV = OUT_DIR / "local_latency_benchmark.csv"   # local-model latencies measured with no other job running


# ─────────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────────

def non_llm_outputs(prompts: list[dict], bank: str) -> pd.DataFrame:
    """Spike choices and latency of the non-LLM strategies for every prompt (repeated to match LLM N).

    Rule-based and R1 trained arms run here through baselines.STRATEGIES. R2 (`*_indomain`) arms are
    read from trained_baselines' cross-validated predictions for the bank, when present, and routed
    through the same keyword routing and band-centre spike values."""
    from agents.schemas import FaultSeverity
    from dl_engine.inference import get_healthy_baseline
    from research.baselines import STRATEGIES, _keyword_spike_with_severity

    base = get_healthy_baseline(noise_std_frac=0.0)
    for arm in ("embedding_severity", "keyword_regex_extended", *TRAINED_ARMS):
        STRATEGIES[arm](prompts[0]["text"], base)          # load models / build patterns before timing
    rows = []

    def add(arm, p, spike, latency):
        for r in range(REPEATS):
            rows.append({"arm": arm, "bank_id": p["id"], "prompt": p["text"], "repeat": r,
                         "sensor_id": spike["sensor_id"], "severity": spike["fault_severity"],
                         "spike_value": spike["spike_value"], "fallback": False, "latency_ms": latency,
                         "input_tokens": np.nan, "output_tokens": np.nan})

    for arm in RULE_BASED + TRAINED_ARMS:
        for p in prompts:
            t0 = time.perf_counter()
            _injected, spike = STRATEGIES[arm](p["text"], base)
            add(arm, p, spike, (time.perf_counter() - t0) * 1000)

    if bank in ("design", "anchored") and indomain_path(bank).exists():
        preds = pd.read_csv(indomain_path(bank)).set_index(["arm", "bank_id"])
        for arm in [a for a in INDOMAIN_ARMS if a in set(preds.index.get_level_values("arm"))]:
            for p in prompts:
                pred = preds.loc[(arm, p["id"])]
                spike = _keyword_spike_with_severity(p["text"], FaultSeverity(pred["severity"]), arm.upper())
                add(arm, p, spike.model_dump(mode="json"), float(pred["latency_ms"]))
    return pd.DataFrame(rows)


def assemble(bank: str) -> tuple[pd.DataFrame, list[str], list[dict]]:
    prompts, problems = load_bank(bank)
    if problems:
        raise SystemExit(f"[score] {bank} bank has problems: {problems}")
    llm = load_outputs(bank)
    llm = llm[llm["arm"].isin(ARMS)]
    expected = PRODUCTION_ARMS if bank == "typo" else ARMS          # the typo set was collected with production prompts only
    missing = [a for a in expected if a not in set(llm["arm"])]
    incomplete = [a for a, g in llm.groupby("arm") if len(g) < len(prompts) * REPEATS]
    for col in ("input_tokens", "output_tokens"):
        if col not in llm:
            llm[col] = np.nan
    df = pd.concat([non_llm_outputs(prompts, bank), llm], ignore_index=True)

    meta = pd.DataFrame(prompts).rename(columns={"id": "bank_id", "design_severity": "label"})
    if "anchor" not in meta:
        meta["anchor"] = meta["bank_id"]
    df = df.merge(meta[["bank_id", "label", "style", "anchor"]], on="bank_id", how="inner")
    df["expected_status"] = np.where(df["label"] == "HIGH", "OFFLINE", "ONLINE")
    df["severity_correct"] = (df["severity"] == df["label"]).astype(float)
    return df, missing + [f"{a} (incomplete)" for a in incomplete], prompts


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def classification_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm classification metrics. Local-model latency comes from local_latency_benchmark.csv when it
    exists (the non-replicated design- and typo-bank calls, made with no other job running), because the
    anchored-bank collection shared the machine with other jobs."""
    bench = pd.read_csv(LOCAL_LATENCY_CSV) if LOCAL_LATENCY_CSV.exists() else None
    rows = []
    for arm, g in df.groupby("arm"):
        high = g["label"] == "HIGH"
        latency = g["latency_ms"]
        if bench is not None and arm in set(bench["arm"]):
            latency = bench.loc[bench["arm"] == arm, "latency_ms"]
        row = {
            "arm": arm, "n": len(g),
            "severity_accuracy": float(g["severity_correct"].mean()),
            "high_recall": float((g.loc[high, "severity"] == "HIGH").mean()) if high.any() else np.nan,
            "over_escalation": float((g.loc[~high, "severity"] == "HIGH").mean()) if (~high).any() else np.nan,
            "fallback_rate": float(g["fallback"].astype(bool).mean()),
            "p50_latency_ms": float(latency.median()),
            "p95_latency_ms": float(latency.quantile(0.95)),
            "mean_input_tokens": float(g["input_tokens"].mean()),
            "mean_output_tokens": float(g["output_tokens"].mean()),
        }
        model = arm.split("__")[0]
        if model in PRICES_PER_M and g["input_tokens"].notna().any():
            pin, pout = PRICES_PER_M[model]
            row["usd_per_1000"] = 1000 * (row["mean_input_tokens"] * pin + row["mean_output_tokens"] * pout) / 1e6
        else:
            row["usd_per_1000"] = 0.0 if arm in NON_LLM else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def status_scores(df: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    select_checkpoint(checkpoint)
    keys = list(zip(df["bank_id"], df["repeat"].astype(int)))
    baselines = healthy_baselines(keys, checkpoint, SEED)
    ruls, statuses = replay(spikes_from_frame(df), [baselines[k] for k in keys], PUBLISHED_MULTIPLIERS)
    out = df[["arm", "bank_id", "repeat", "anchor", "style", "label", "expected_status"]].copy()
    out["checkpoint"] = checkpoint
    out["rul"] = ruls
    out["status"] = statuses
    out["match"] = (out["status"] == out["expected_status"]).astype(float)
    return out


def paired_table(df: pd.DataFrame, status_rows: pd.DataFrame, checkpoints: list[str], cluster: str) -> pd.DataFrame:
    """Every LLM arm vs each reference: severity accuracy and status match per checkpoint, Holm-adjusted."""
    metrics = [("severity accuracy", df.rename(columns={"severity_correct": "value"}))]
    for ck in checkpoints:
        metrics.append((f"status {ck}", status_rows[status_rows["checkpoint"] == ck].rename(columns={"match": "value"})))
    rows = []
    llm_arms = sorted(a for a in df["arm"].unique() if a not in NON_LLM)
    for ref in REFERENCES:
        for arm in llm_arms:
            for name, frame in metrics:
                cols = list(dict.fromkeys(["bank_id", "repeat", cluster, "value"]))
                m = frame[frame["arm"] == arm][cols].merge(
                    frame[frame["arm"] == ref][["bank_id", "repeat", "value"]],
                    on=["bank_id", "repeat"], suffixes=("_arm", "_ref"))
                res = paired_difference(m["value_arm"], m["value_ref"], m[cluster], seed=SEED)
                rows.append({"reference": ref, "arm": arm, "metric": name, **res})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p"].tolist())
    return table


def family_table(df: pd.DataFrame, cluster: str, arms: list[str], references: list[str]) -> pd.DataFrame:
    """One hypothesis family: severity accuracy of each LLM arm minus each reference, on unfamiliar wording
    only (all prompts in the design and typo banks; rewrites only in the anchored bank). Holm across the
    family. H1 is the original primary hypothesis (an LLM earns its cost on wording the regex was not built
    for); H2 and H3 are defined in research/AEI/analysis_plan_additions.md."""
    unfamiliar = df[df["style"] != "original"]
    rows = []
    for ref in references:
        for arm in sorted(arms):
            cols = list(dict.fromkeys(["bank_id", "repeat", cluster, "severity_correct"]))
            m = unfamiliar[unfamiliar["arm"] == arm][cols].merge(
                unfamiliar[unfamiliar["arm"] == ref][["bank_id", "repeat", "severity_correct"]],
                on=["bank_id", "repeat"], suffixes=("_arm", "_ref"))
            res = paired_difference(m["severity_correct_arm"], m["severity_correct_ref"], m[cluster], seed=SEED)
            rows.append({"reference": ref, "arm": arm, "arm_accuracy": float(m["severity_correct_arm"].mean()),
                         "reference_accuracy": float(m["severity_correct_ref"].mean()), **res})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p"].tolist())
    return table


def common_interface_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the outputs with every arm's sensor and spike value replaced by the shared keyword routing
    and band-centre value for its severity, so arms differ only in severity."""
    from agents.schemas import FaultSeverity
    from research.baselines import _keyword_spike_with_severity

    choice = {}
    for text, severity in set(zip(df["prompt"], df["severity"])):
        spike = _keyword_spike_with_severity(text, FaultSeverity(severity), "COMMON")
        choice[(text, severity)] = (spike.sensor_id, spike.spike_value)
    out = df.copy()
    picked = [choice[k] for k in zip(df["prompt"], df["severity"])]
    out["sensor_id"] = [c[0] for c in picked]
    out["spike_value"] = [c[1] for c in picked]
    return out


def interface_table(status_own: pd.DataFrame, status_common: pd.DataFrame, cluster: str) -> pd.DataFrame:
    """Status match with each arm's own routing vs the common interface, per arm, checkpoint and scope."""
    keys = ["arm", "checkpoint", "bank_id", "repeat"]
    m = status_own[list(dict.fromkeys(keys + [cluster, "style", "match"]))].merge(
        status_common[keys + ["match"]], on=keys, suffixes=("_own", "_common"))
    scopes = {"all": m, "unfamiliar": m[m["style"] != "original"]}
    rows = []
    for scope, frame in scopes.items():
        if frame.empty:
            continue
        for (arm, ck), g in frame.groupby(["arm", "checkpoint"]):
            res = paired_difference(g["match_common"], g["match_own"], g[cluster], seed=SEED)
            rows.append({"arm": arm, "checkpoint": ck, "scope": scope, "own": float(g["match_own"].mean()),
                         "common": float(g["match_common"].mean()), **res})
    return pd.DataFrame(rows)


def original_vs_rewrites(df: pd.DataFrame, status_rows: pd.DataFrame, checkpoints: list[str]) -> pd.DataFrame:
    """Per arm: score on the original anchor wording vs the mean over its rewrites, paired by anchor."""
    frames = [("severity accuracy", df.rename(columns={"severity_correct": "value"}))]
    for ck in checkpoints:
        frames.append((f"status {ck}", status_rows[status_rows["checkpoint"] == ck].rename(columns={"match": "value"})))
    rows = []
    for arm in sorted(df["arm"].unique()):
        for name, frame in frames:
            g = frame[frame["arm"] == arm]
            per_anchor = g.assign(kind=np.where(g["style"] == "original", "original", "rewrite")) \
                          .pivot_table(index="anchor", columns="kind", values="value", aggfunc="mean")
            res = paired_difference(per_anchor["rewrite"], per_anchor["original"],
                                    pd.Series(per_anchor.index), seed=SEED)
            rows.append({"arm": arm, "metric": name, "original": float(per_anchor["original"].mean()),
                         "rewrites": float(per_anchor["rewrite"].mean()), **res})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.1%}"


def _diff(r) -> str:
    return f"{r['diff'] * 100:+.1f} [{r['ci_lo'] * 100:+.0f}, {r['ci_hi'] * 100:+.0f}] p={r['p_holm']:.3f}"


FAMILY_TEXT = {
    "H1": "Original primary hypothesis (manuscript §8.1): an LLM earns its cost on wording the regex was not built "
          "for. Hosted production-prompt LLMs minus the regex and the prototype embedding classifier.",
    "H2": "Added 2026-09-17 (analysis_plan_additions.md): every production-prompt LLM, hosted and local, minus the "
          "pre-designated trained classifier `bge_m3_logreg` trained on production-prompt vocabulary (R1) and with "
          "in-domain labelled examples, cross-validated (R2).",
    "H3": "Added 2026-09-17 (analysis_plan_additions.md): every production-prompt LLM, hosted and local, minus the "
          "regex and `bge_m3_logreg` (R1) on programmatic typos.",
}


def write_summary(bank: str, cls: pd.DataFrame, status_rows: pd.DataFrame, df: pd.DataFrame,
                  paired: pd.DataFrame, families: dict[str, pd.DataFrame], orig: pd.DataFrame | None,
                  interface: pd.DataFrame, problems: list[str], cluster: str) -> Path:
    order = cls.sort_values("severity_accuracy", ascending=False).index
    checkpoints = list(dict.fromkeys(status_rows["checkpoint"]))
    md = [f"# Paraphrase experiment — {bank} bank\n"]
    md.append(f"**Generated by:** `python -m research.paraphrase.score --bank {bank}`\n")
    if problems:
        md.append("**Missing or incomplete arms:** " + ", ".join(problems) + "\n")

    md.append("\n## 1. Prompt set\n")
    if bank == "anchored":
        md.append("The 18 non-rejected TEST_PROMPTS (style `original`) plus four rewrites of each "
                  "(shopfloor, consequence, hinglish, sms). Each rewrite inherits its anchor's published severity "
                  "and expected status. CIs and paired tests resample anchors.\n")
    elif bank == "typo":
        from research.paraphrase.typo_bank import DROPPED_BY_GUARD
        md.append("The 18 TEST_PROMPTS anchors with seeded keyboard typos (`typo_bank.py`): at level p, "
                  "max(1, round(p·n)) of the n eligible words get one adjacent-key substitution, deletion, insertion "
                  "or transposition; p = 0.25 and 0.50, three variants each. No hand-written text. "
                  f"{len(DROPPED_BY_GUARD)} variants rejected by the input guard were dropped. Labels are inherited "
                  "from the anchor. CIs and paired tests resample anchors.\n")
    else:
        md.append("Ground truth is each prompt's design severity (`prompt_bank.py`), the same author-labelled "
                  "convention as TEST_PROMPTS. CIs and paired tests resample prompts.\n")
    prompts = df.drop_duplicates("bank_id")
    md.append("Prompts by severity: " + ", ".join(f"{k} {v}" for k, v in prompts["label"].value_counts().items())
              + "; by style: " + ", ".join(f"{k} {v}" for k, v in prompts["style"].value_counts().items()) + ".\n")

    md.append("\n## 2. Severity classification, latency and cost\n")
    md.append("| Arm | Severity accuracy | HIGH recall | Over-escalation | Fallback | P50 latency | P95 latency | USD per 1,000 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for arm in order:
        r = cls.loc[arm]
        usd = "–" if pd.isna(r["usd_per_1000"]) else f"{r['usd_per_1000']:.3f}"
        md.append(f"| {arm} | {_pct(r['severity_accuracy'])} | {_pct(r['high_recall'])} | {_pct(r['over_escalation'])} | "
                  f"{_pct(r['fallback_rate'])} | {r['p50_latency_ms']:.0f} ms | {r['p95_latency_ms']:.0f} ms | {usd} |")
    bench_note = (" from the design- and typo-bank calls, made with no other job on the machine "
                  "(`local_latency_benchmark.csv`, 147 calls per model)" if LOCAL_LATENCY_CSV.exists()
                  else " during collection, alongside other jobs")
    md.append("\nNon-LLM latency is measured locally on CPU (including injection); hosted LLM latency is the API round "
              f"trip; local LLM latency is measured on a GTX 1650 (4 GB){bench_note}. "
              "`*_indomain` latency is one prediction by the fold model. Cost uses recorded token usage (output "
              "includes thinking/reasoning tokens) and list prices retrieved 2026-09-17; local models have no API "
              "charge.\n")

    md.append("\n## 3. Status match after replay (published multipliers 0.15 / 0.35 / 0.85)\n")
    md.append(f"Mean with 95% bootstrap CI resampling {cluster}s.\n")
    md.append("| Arm | " + " | ".join(checkpoints) + " |")
    md.append("|---|" + "---|" * len(checkpoints))
    for arm in order:
        cells = []
        for ck in checkpoints:
            g = status_rows[(status_rows["arm"] == arm) & (status_rows["checkpoint"] == ck)]
            lo, hi = bootstrap_ci(g["match"], g[cluster])
            cells.append(f"{g['match'].mean():.1%} [{lo:.0%}, {hi:.0%}]")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")

    md.append("\n## 4. Hypothesis families: severity accuracy on unfamiliar wording\n")
    scope = {"anchored": "rewrites only", "design": "all prompts", "typo": "all prompts"}[bank]
    md.append(f"Scope: {scope}. Percentage points, 95% cluster-bootstrap CI ({cluster}s), sign-flip p (exact when "
              "≤ 20 non-zero clusters, otherwise 200,000 random flips), Holm within each family.\n")
    for name, table in families.items():
        md.append(f"\n### {name} ({len(table)} comparisons)\n")
        md.append(FAMILY_TEXT[name] + "\n")
        md.append("| Reference | Arm | Arm accuracy | Reference accuracy | Difference [95% CI] | p (Holm) |")
        md.append("|---|---|---|---|---|---|")
        for r in table.itertuples():
            md.append(f"| {r.reference} | {r.arm} | {r.arm_accuracy:.1%} | {r.reference_accuracy:.1%} | "
                      f"{r.diff * 100:+.1f} [{r.ci_lo * 100:+.0f}, {r.ci_hi * 100:+.0f}] | {r.p_holm:.4f} |")

    md.append("\n## 4b. Secondary paired comparisons: every LLM arm minus a non-LLM reference\n")
    md.append(f"Percentage-point difference, 95% cluster-bootstrap CI ({cluster}s), Holm-adjusted sign-flip p "
              "(exact or 200,000 random flips) "
              f"across all {len(paired)} comparisons (all prompts, including originals in the anchored bank). "
              "With few clusters and this many comparisons the correction is very conservative; read the CIs.\n")
    metrics = list(dict.fromkeys(paired["metric"]))
    for ref in REFERENCES:
        md.append(f"\n**Reference: `{ref}`**\n")
        md.append("| Arm | " + " | ".join(metrics) + " |")
        md.append("|---|" + "---|" * len(metrics))
        sub = paired[paired["reference"] == ref]
        for arm in [a for a in order if a in set(sub["arm"])]:
            cells = [_diff(sub[(sub["arm"] == arm) & (sub["metric"] == m)].iloc[0]) for m in metrics]
            md.append(f"| {arm} | " + " | ".join(cells) + " |")

    section = 5
    if orig is not None:
        md.append(f"\n## {section}. Original wording vs rewrites (paired by anchor)\n")
        md.append("Score on the published anchor prompt vs the mean over its four rewrites; difference = rewrites − "
                  "original, 95% CI over anchors, unadjusted sign-flip p.\n")
        metrics = list(dict.fromkeys(orig["metric"]))
        md.append("| Arm | " + " | ".join(metrics) + " |")
        md.append("|---|" + "---|" * len(metrics))
        for arm in order:
            cells = []
            for m in metrics:
                r = orig[(orig["arm"] == arm) & (orig["metric"] == m)].iloc[0]
                cells.append(f"{r['original']:.0%} → {r['rewrites']:.0%} ({r['diff'] * 100:+.0f}, p={r['p']:.3f})")
            md.append(f"| {arm} | " + " | ".join(cells) + " |")
        section += 1

    md.append(f"\n## {section}. Common-interface decomposition (status match)\n")
    md.append("Each arm's severity is re-injected through the shared keyword routing and band-centre spike values "
              "(`common`) and compared with the arm's own sensor and spike value (`own`). The regex, embedding and "
              "trained classifiers already use the common interface, so their difference is 0 by construction; "
              "`keyword_only` and `fixed_midrange` use preset spikes and can differ. "
              f"Scope: {scope}. Cells: own → common "
              f"(common − own in percentage points, 95% cluster-bootstrap CI over {cluster}s, unadjusted sign-flip p).\n")
    sub = interface[interface["scope"] == ("unfamiliar" if bank == "anchored" else "all")]
    cks = list(dict.fromkeys(sub["checkpoint"]))
    md.append("| Arm | " + " | ".join(cks) + " |")
    md.append("|---|" + "---|" * len(cks))
    for arm in order:
        cells = []
        for ck in cks:
            r = sub[(sub["arm"] == arm) & (sub["checkpoint"] == ck)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            cells.append(f"{r['own']:.0%} → {r['common']:.0%} ({r['diff'] * 100:+.0f} "
                         f"[{r['ci_lo'] * 100:+.0f}, {r['ci_hi'] * 100:+.0f}], p={r['p']:.3f})")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    section += 1

    md.append(f"\n## {section}. Severity accuracy by prompt style\n")
    by_style = df.pivot_table(index="arm", columns="style", values="severity_correct", aggfunc="mean")
    styles = list(by_style.columns)
    md.append("| Arm | " + " | ".join(styles) + " |")
    md.append("|---|" + "---|" * len(styles))
    for arm in order:
        md.append(f"| {arm} | " + " | ".join(f"{by_style.loc[arm, s]:.0%}" for s in styles) + " |")

    md.append("\n## Notes\n")
    md.append("- Expected status: HIGH → OFFLINE; LOW or MEDIUM → ONLINE (same convention as TEST_PROMPTS).")
    md.append("- Non-LLM arms are deterministic, computed once per prompt and repeated 3× so N matches the LLM arms.")
    md.append("- All non-LLM severity classifiers share the regex's keyword sensor routing and band-centre spike "
              "values, so they differ only in how severity is decided.")
    md.append("- Trained classifiers (`research/trained_baselines.py`): `tfidf_logreg`, `minilm_logreg`, "
              "`bge_m3_logreg`, `minilm_finetuned` trained on sentences built from the production prompt's vocabulary "
              "(R1); `*_indomain` adds labelled bank prompts outside the test prompt's cluster (R2).")
    md.append("- Hosted LLM arms: Gemini 2.5 Flash (default thinking and thinking off), Gemini 3.5 Flash, gpt-oss-120b "
              "and Qwen 3.8 27B; local arms: Llama 3.2 3B, Gemma 3 4B and Qwen 3 4B via Ollama (collect.py).")

    path = OUT_DIR / f"{OUT_PREFIX[bank]}_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", choices=list(OUT_PREFIX), default="design")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="default: turbofan simulator, plus simulator_calibrated and pronostia if their weights exist")
    args = parser.parse_args()

    checkpoints = args.checkpoints or (
        ["turbofan", "simulator"]
        + (["simulator_calibrated"] if CALIBRATED_WEIGHTS.exists() else [])
        + (["pronostia"] if PRONOSTIA_WEIGHTS.exists() else []))
    select_checkpoint(checkpoints[0])        # non-LLM strategies inject, so a model must be loaded

    df, problems, _prompts = assemble(args.bank)
    cluster = "bank_id" if args.bank == "design" else "anchor"
    cls = classification_metrics(df)
    status_rows = pd.concat([status_scores(df, ck) for ck in checkpoints], ignore_index=True)
    compared = [ck for ck in checkpoints if ck != "pronostia"]      # every arm ties on pronostia
    paired = paired_table(df, status_rows, compared, cluster)
    families = {}
    present = set(df["arm"])
    for name, (banks, arms, refs) in FAMILIES.items():
        if args.bank not in banks:
            continue
        missing = [a for a in arms + refs if a not in present]
        if missing:
            problems.append(f"{name} not computed (missing {', '.join(missing)})")
            continue
        families[name] = family_table(df, cluster, arms, refs)
    orig = original_vs_rewrites(df, status_rows, compared) if args.bank == "anchored" else None
    common = common_interface_frame(df)
    status_common = pd.concat([status_scores(common, ck) for ck in compared], ignore_index=True)
    interface = interface_table(status_rows[status_rows["checkpoint"].isin(compared)], status_common, cluster)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_PREFIX[args.bank]
    scores = (status_rows.groupby(["arm", "checkpoint", "style"])["match"].agg(["mean", "size"])
              .reset_index().rename(columns={"mean": "status_match", "size": "n"}))
    scores.to_csv(OUT_DIR / f"{prefix}_scores.csv", index=False)
    status_rows.to_csv(OUT_DIR / f"{prefix}_rows.csv.gz", index=False, compression="gzip")
    status_common.to_csv(OUT_DIR / f"{prefix}_rows_common_interface.csv.gz", index=False, compression="gzip")
    paired.to_csv(OUT_DIR / f"{prefix}_paired.csv", index=False)
    for name, table in families.items():
        table.to_csv(OUT_DIR / f"{prefix}_{FAMILY_FILES[name]}.csv", index=False)
    interface.to_csv(OUT_DIR / f"{prefix}_interface.csv", index=False)
    cls.to_csv(OUT_DIR / f"{prefix}_classification.csv")
    if orig is not None:
        orig.to_csv(OUT_DIR / f"{prefix}_original_vs_rewrites.csv", index=False)
    path = write_summary(args.bank, cls, status_rows, df, paired, families, orig, interface, problems, cluster)
    print(f"[score] wrote {path.relative_to(PROJECT_ROOT)} and {prefix}_*.csv")
    if problems:
        print("[score] problems:", "; ".join(problems))


if __name__ == "__main__":
    main()
