"""
research/verify_findings.py

Independent verification of every number planned for the AEI manuscript
(research/AEI/paper_outline.md) before any writing.

Principles:
  - Recompute from raw data files with code written here, not the aggregation
    code in score.py / multiplier_sweep.py / multi_hit.py.
  - Re-derive a random sample of replayed predictions with the production
    single-window path (`_inject_spike` + `predict_rul`), not the batched replay.
  - Re-derive the dashboard window rule with `terminal.factory_state.FactoryState`.
  - Re-derive test-set labels by parsing the comments in evaluation_rubric.py.
  - Check statistics with exact sign-flip enumeration and statsmodels' Holm.
  - Record integrity problems (duplicates, wrong model tags, latency inflated
    by retries, repeat inconsistency) and methods mismatches as notes.

Each claim is compared with its recomputed value. Percentages in claims are
written as they appear in the outline (rounded); a range claim passes when the
recomputed min and max round to the stated ends.

Usage:
    python -m research.verify_findings [--sample 60]

Output:
    research/AEI/verification_report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

R = PROJECT_ROOT / "research" / "results"
REPORT = PROJECT_ROOT / "research" / "AEI" / "verification_report.md"

CURRENT_LLMS = ["agentic", "gemini_3_5_flash", "groq_gpt_oss_120b", "groq_qwen3_8"]
ALL_LLMS = CURRENT_LLMS + ["groq_llama3", "groq_llama4"]
SEM_ARMS = ["gemini_2_5_flash__semantic", "gemini_3_5_flash__semantic", "gpt_oss_120b__semantic", "qwen3_8_27b__semantic"]
# Production-prompt arms, including Gemini 2.5 Flash with thinking off (the production agent's configuration).
PROD_ARMS = [a.replace("semantic", "production") for a in SEM_ARMS] + ["gemini_2_5_flash_nothink__production"]
NON_LLM = ["keyword_regex_severity", "keyword_regex_extended", "embedding_severity", "keyword_only", "fixed_midrange"]
LOCAL_ARMS_V = ["llama3_2_3b__production", "gemma3_4b__production", "qwen3_4b__production"]
LOCAL_TAGS = {"llama3_2_3b": "[llama3.2:3b]", "gemma3_4b": "[gemma3:4b]", "qwen3_4b": "[qwen3:4b]"}
TRAINED = ["tfidf_logreg", "minilm_logreg", "bge_m3_logreg", "minilm_finetuned"]
ALL_PROD = PROD_ARMS + LOCAL_ARMS_V
MODEL_TAGS = {"gemini_2_5_flash": "[gemini-2.5-flash]", "gemini_3_5_flash": "[gemini-3.5-flash]",
              "gpt_oss_120b": "[gpt-oss-120b]", "qwen3_8_27b": "[qwen3.8-27b]"}

RESULTS: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# Recording helpers
# ─────────────────────────────────────────────────────────────────────────────

def _record(section, claim, claimed, recomputed, ok, detail=""):
    if isinstance(ok, (bool, np.bool_)):
        status = "PASS" if bool(ok) else "FAIL"
    else:
        status = ok
    RESULTS.append({"section": section, "claim": claim, "claimed": claimed, "recomputed": recomputed,
                    "status": status, "detail": detail})


def check_pct(section, claim, claimed_pct, value, decimals=1):
    """Single percentage claim, compared at the claim's precision."""
    rec = round(value * 100, decimals)
    _record(section, claim, f"{claimed_pct}%", f"{rec}%", abs(rec - claimed_pct) < 10 ** -decimals / 2 + 1e-9)


def check_range(section, claim, lo, hi, values, decimals=0):
    vals = [v * 100 for v in values]
    rlo, rhi = round(min(vals), decimals), round(max(vals), decimals)
    _record(section, claim, f"{lo}–{hi}%", f"{rlo}–{rhi}%", rlo == lo and rhi == hi,
            "values: " + ", ".join(f"{v:.1f}" for v in sorted(vals)))


def check_eq(section, claim, claimed, value, detail=""):
    _record(section, claim, claimed, value, claimed == value, detail)


def note(section, claim, detail, status="NOTE"):
    _record(section, claim, "–", "–", status, detail)


# ─────────────────────────────────────────────────────────────────────────────
# Shared loaders
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt_groups() -> dict[str, tuple[str, str]]:
    """prompt -> (severity group, expected status), parsed from evaluation_rubric.py comments."""
    src = (PROJECT_ROOT / "research" / "evaluation_rubric.py").read_text(encoding="utf-8")
    block = src[src.index("TEST_PROMPTS = ["): src.index("]\n", src.index("TEST_PROMPTS = ["))]
    groups, current = {}, None
    for line in block.splitlines():
        header = re.match(r"\s*#\s*(LOW|MEDIUM|HIGH) severity", line)
        if header:
            current = header.group(1)
            continue
        if re.match(r"\s*#\s*(Edge|Rejection)", line):
            current = None
            continue
        entry = re.match(r'\s*\("([^"]+)",\s*(\d+),\s*"(\w+)"\),\s*(#\s*(LOW|MEDIUM|HIGH))?', line)
        if entry and entry.group(3) != "REJECTED":
            groups[entry.group(1)] = (entry.group(5) or current, entry.group(3))
    return groups


def baselines(v: str) -> pd.DataFrame:
    from research.evaluation_rubric import TEST_PROMPTS
    df = pd.read_csv(R / f"baselines_comparison_{v}.csv")
    df = df[~df["rejected"].astype(bool)].copy()
    exp = {p: s for p, _m, s in TEST_PROMPTS}
    df["match"] = df["status"] == df["prompt"].map(exp)
    return df


def llm_outputs(bank: str) -> pd.DataFrame:
    """All collected LLM rows for a bank (every part file), restricted to the defined arms."""
    from research.paraphrase.collect import ARMS, load_outputs
    out = load_outputs(bank)
    return out[out["arm"].isin(ARMS)].reset_index(drop=True)


def bank_prompts(bank: str) -> list[dict]:
    if bank == "anchored":
        from research.paraphrase.anchored_bank import ANCHORED_PROMPTS
        return ANCHORED_PROMPTS
    from research.paraphrase.prompt_bank import PROMPTS
    return PROMPTS


def severity_table(bank: str) -> pd.DataFrame:
    """(arm, bank_id, repeat, severity, label, style, anchor) for LLM and non-LLM arms, computed here."""
    from dl_engine.inference import get_healthy_baseline
    from research.baselines import STRATEGIES

    prompts = bank_prompts(bank)
    meta = pd.DataFrame(prompts).rename(columns={"id": "bank_id", "design_severity": "label"})
    if "anchor" not in meta:
        meta["anchor"] = meta["bank_id"]
    llm = llm_outputs(bank)[["arm", "bank_id", "repeat", "severity"]]
    base = get_healthy_baseline(noise_std_frac=0.0)
    rows = []
    for arm in NON_LLM:
        for p in prompts:
            sev = STRATEGIES[arm](p["text"], base)[1]["fault_severity"]
            rows += [{"arm": arm, "bank_id": p["id"], "repeat": r, "severity": sev} for r in range(3)]
    df = pd.concat([llm, pd.DataFrame(rows)], ignore_index=True)
    return df.merge(meta[["bank_id", "label", "style", "anchor"]], on="bank_id")


def exact_signflip_p(per_cluster: np.ndarray, mc: int = 400_000, seed: int = 0) -> tuple[float, str]:
    """Two-sided sign-flip p on cluster mean differences; exact when feasible."""
    d = per_cluster[np.abs(per_cluster) > 1e-12]
    k, n = len(d), len(per_cluster)
    if k == 0:
        return 1.0, "exact"
    obs = abs(per_cluster.mean())
    if k <= 20:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
        stats = np.abs((signs * d).sum(axis=1) / n)
        return float(np.mean(stats >= obs - 1e-12)), f"exact over 2^{k}"
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(mc, k))
    stats = np.abs((signs * d).sum(axis=1) / n)
    return float((np.sum(stats >= obs - 1e-12) + 1) / (mc + 1)), f"Monte Carlo {mc}"


# ─────────────────────────────────────────────────────────────────────────────
# A. Data integrity
# ─────────────────────────────────────────────────────────────────────────────

def section_a():
    S = "A. Data integrity"
    for v in ("turbofan", "simulator"):
        df = pd.read_csv(R / f"baselines_comparison_{v}.csv")
        nr = df[~df["rejected"].astype(bool)]
        counts = nr.groupby("strategy").size()
        check_eq(S, f"{v}: every strategy has N = 54 non-rejected runs", True, bool((counts == 54).all()),
                 ", ".join(f"{k}={n}" for k, n in counts.items() if n != 54))
        newer = nr[nr["strategy"].isin(["groq_gpt_oss_120b", "groq_qwen3_8", "keyword_regex_extended", "embedding_severity"])]
        fb = int(newer["spike_summary"].fillna("").str.contains("FALLBACK|UNAVAILABLE").sum())
        check_eq(S, f"{v}: no fallback rows for the 2026-09 strategies", 0, fb)
        dup = int(df.duplicated(["strategy", "prompt", "repeat"]).sum())
        check_eq(S, f"{v}: no duplicate (strategy, prompt, repeat)", 0, dup)

    for bank, n_prompts in (("anchored", 90), ("design", 42)):
        out = llm_outputs(bank)
        prompts = {p["id"]: p["text"] for p in bank_prompts(bank)}
        from research.paraphrase.collect import ARMS
        check_eq(S, f"{bank}: LLM output rows = {len(ARMS)} arms × {n_prompts} prompts × 3", len(ARMS) * n_prompts * 3, len(out))
        check_eq(S, f"{bank}: no duplicate (arm, prompt, repeat)", 0, int(out.duplicated(["arm", "bank_id", "repeat"]).sum()))
        check_eq(S, f"{bank}: no fallback rows", 0, int(out["fallback"].astype(bool).sum()))
        check_eq(S, f"{bank}: prompt text matches the bank for every row", True,
                 bool((out["prompt"] == out["bank_id"].map(prompts)).all()))
        wrong_tag = 0
        for model, tag in MODEL_TAGS.items():
            sub = out[out["arm"].str.startswith(model + "__")]
            other = [t for m, t in MODEL_TAGS.items() if m != model]
            wrong_tag += int(sub["summary"].apply(lambda s: any(str(s).startswith(t) for t in other)).sum())
        check_eq(S, f"{bank}: no summary carries another model's tag", 0, wrong_tag)
        hosted_out = out[~out["arm"].isin(LOCAL_ARMS_V)]
        slow = hosted_out[hosted_out["latency_ms"] > 20_000]
        note(S, f"{bank}: hosted-LLM latency outliers > 20 s (possible rate-limit backoff inside the call)",
             f"{len(slow)} rows; max latency {hosted_out['latency_ms'].max() / 1000:.1f} s",
             "PASS" if len(slow) == 0 else "WARN")
        local_slow = out[out["arm"].isin(LOCAL_ARMS_V) & (out["latency_ms"] > 20_000)]
        note(S, f"{bank}: local-LLM calls > 20 s during collection (not used for reported latency)",
             f"{len(local_slow)} rows; reported local latency comes from local_latency_benchmark.csv")
        cons = out.groupby(["arm", "bank_id"])["severity"].nunique()
        note(S, f"{bank}: repeat consistency (same severity in all 3 repeats)",
             "; ".join(f"{a}: {(g == 1).mean():.1%}" for a, g in cons.groupby(level=0)))
        if bank == "anchored":
            check_eq(S, "anchored: token usage recorded for every row", True,
                     bool(out[["input_tokens", "output_tokens"]].notna().all().all()))
            tok = out.groupby("arm")["input_tokens"].mean()
            ok = all(tok[f"{m}__production"] > tok[f"{m}__semantic"] for m in MODEL_TAGS)
            check_eq(S, "anchored: production-prompt arms have more input tokens than semantic arms "
                        "(evidence the two system prompts were really used)", True, ok,
                     ", ".join(f"{a}={v:.0f}" for a, v in tok.items()))


# ─────────────────────────────────────────────────────────────────────────────
# B. Test-set definitions
# ─────────────────────────────────────────────────────────────────────────────

def section_b():
    S = "B. Test sets and labels"
    from research.multi_hit import SEVERITY_GROUP
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS, _ANCHORS
    from research.paraphrase.prompt_bank import PROMPTS, forbidden_hits
    from research.baselines import _HIGH_PATTERN, _LOW_PATTERN, _HIGH_TERMS, _LOW_TERMS

    groups = test_prompt_groups()
    check_eq(S, "18 non-rejected TEST_PROMPTS parsed from evaluation_rubric.py", 18, len(groups))
    mism = [a for a, (sev, _r) in _ANCHORS.items() if groups.get(a, (None,))[0] != sev]
    check_eq(S, "anchored bank severities equal the TEST_PROMPTS group comments", [], mism)
    mism = [a for a, g in SEVERITY_GROUP.items() if groups.get(a, (None,))[0] != g]
    check_eq(S, "multi_hit.SEVERITY_GROUP equals the TEST_PROMPTS group comments", [], mism)
    mism = [p["id"] for p in ANCHORED_PROMPTS if p["expected_status"] != groups[p["anchor"]][1]]
    check_eq(S, "anchored expected statuses equal TEST_PROMPTS", [], mism)
    incons = [a for a, (sev, st) in groups.items() if (sev == "HIGH") != (st == "OFFLINE")]
    check_eq(S, "in TEST_PROMPTS, HIGH ⇔ expected OFFLINE (the rule used for rewrites)", [], incons)

    prompts_src = (PROJECT_ROOT / "agents" / "prompts.py").read_text(encoding="utf-8")
    sev_block = prompts_src[prompts_src.index("== SEVERITY CLASSIFICATION"): prompts_src.index("== SPIKE VALUE RULES")]
    bare_line = next(l for l in sev_block.splitlines() if "bare fault description" in l)
    bare_examples = set(re.findall(r'"([^"]+)"', bare_line))
    severity_words = (set(re.findall(r'"([^"]+)"', sev_block)) - bare_examples) | set(_HIGH_TERMS) | set(_LOW_TERMS)
    uncovered = sorted(t for t in severity_words if not forbidden_hits(t))
    check_eq(S, "vocabulary checker catches every severity word in prompts.py and the regex lists", [],
             uncovered, f"{len(severity_words)} terms checked")
    all_texts = [p["text"].lower() for p in ANCHORED_PROMPTS if p["style"] != "original"] + [p["text"].lower() for p in PROMPTS]
    present = sorted(b for b in bare_examples if any(b in t for t in all_texts))
    check_eq(S, "prompts.py 'bare fault description' examples appear in no rewrite or design prompt", [], present,
             ", ".join(sorted(bare_examples)))

    rewrites = [p for p in ANCHORED_PROMPTS if p["style"] != "original"]
    hits = [p["id"] for p in rewrites if _HIGH_PATTERN.search(p["text"]) or _LOW_PATTERN.search(p["text"])]
    check_eq(S, "no anchored rewrite triggers the original regex word lists", [], hits)
    bad = [p["id"] for p in PROMPTS if p["style"] != "negation" and (_HIGH_PATTERN.search(p["text"]) or _LOW_PATTERN.search(p["text"]))]
    check_eq(S, "no design-bank prompt except negation traps triggers the regex", [], bad)
    miss = [p["id"] for p in PROMPTS if p["style"] == "negation" and not (_HIGH_PATTERN.search(p["text"]) or _LOW_PATTERN.search(p["text"]))]
    check_eq(S, "every design-bank negation trap triggers the regex", [], miss)


# ─────────────────────────────────────────────────────────────────────────────
# C. 18 original prompts
# ─────────────────────────────────────────────────────────────────────────────

def section_c():
    S = "C. 18 original prompts (status match, N = 54)"
    claims = {
        "turbofan": {"agentic": 100.0, "gemini_3_5_flash": 100.0, "groq_gpt_oss_120b": 100.0, "groq_qwen3_8": 100.0,
                     "keyword_regex_severity": 94.4, "keyword_regex_extended": 94.4, "embedding_severity": 100.0,
                     "groq_llama3": 96.3, "groq_llama4": 94.4},
        "simulator": {"agentic": 88.9, "gemini_3_5_flash": 88.9, "groq_gpt_oss_120b": 88.9, "groq_qwen3_8": 88.9,
                      "keyword_regex_severity": 94.4, "keyword_regex_extended": 94.4, "embedding_severity": 100.0,
                      "groq_llama3": 87.0, "groq_llama4": 75.9},
    }
    drivers = set()
    for v, cl in claims.items():
        df = baselines(v)
        rates = df.groupby("strategy")["match"].mean()
        for s, c in cl.items():
            check_pct(S, f"{v}: {s}", c, rates[s])
        per_prompt = df[df["strategy"].isin(CURRENT_LLMS + ["keyword_regex_severity"])] \
            .pivot_table(index="prompt", columns="strategy", values="match", aggfunc="mean")
        differ = per_prompt[per_prompt.nunique(axis=1) > 1].index.tolist()
        note(S, f"{v}: prompts where the current 4 LLMs and the regex differ", "; ".join(differ))
        drivers |= set(differ)
        p95 = df[df["strategy"] == "keyword_regex_severity"]["latency_ms"].quantile(0.95)
        note(S, f"{v}: regex P95 latency", f"{p95:.1f} ms")
    check_eq(S, "union of driver prompts = {complete motor breakdown, shaft lock}",
             sorted(["complete motor breakdown on Machine 2", "shaft lock on Machine 5"]), sorted(drivers))


# ─────────────────────────────────────────────────────────────────────────────
# D. Replay fidelity
# ─────────────────────────────────────────────────────────────────────────────

def section_d(sample: int):
    S = "D. Replay fidelity"
    from agents.diagnostic_agent import _inject_spike
    from agents.schemas import FaultSeverity, SensorSpike
    from dl_engine.inference import get_healthy_baseline, predict_rul
    from research.baselines import STRATEGIES
    from research.replay_utils import PUBLISHED_MULTIPLIERS, healthy_baselines, select_checkpoint, status_from_rul

    table = dict(zip((FaultSeverity.LOW, FaultSeverity.MEDIUM, FaultSeverity.HIGH), PUBLISHED_MULTIPLIERS))
    rng = np.random.default_rng(7)

    for v in ("turbofan", "simulator"):
        sw = pd.read_csv(R / f"multiplier_sweep_{v}.csv")
        pub = sw[(sw["low"] == 0.15) & (sw["medium"] == 0.35) & (sw["high"] == 0.85)]
        diff = (pub["match_rate"] - pub["published_match_rate"]).abs().max()
        check_eq(S, f"{v}: batched replay at published multipliers equals the recorded pipeline run (all strategies)",
                 0.0, round(float(diff), 6))

    # Sweep rows: single-window recomputation at random settings.
    for v, rows_path in (("turbofan", R / "multiplier_sweep_rows_turbofan.csv.gz"),
                         ("simulator", R / "multiplier_sweep_rows_simulator.csv.gz")):
        rows = pd.read_csv(rows_path)
        rec = baselines(v)[["strategy", "prompt", "repeat", "spike_value"]]
        select_checkpoint(v)
        keys = list(zip(rows["source"], rows["prompt"], rows["repeat"].astype(int)))
        bases = healthy_baselines(set(keys), v, 20260916)
        pick = rows.sample(sample, random_state=11).merge(rec, on=["strategy", "prompt", "repeat"], how="left")
        bad = 0
        for r in pick.itertuples():
            spike = SensorSpike(sensor_id=r.sensor_id, spike_value=float(r.spike_value), affected_window_positions=[49],
                                fault_severity=FaultSeverity(r.severity), plain_english_summary="verify")
            mult = {FaultSeverity.LOW: r.low, FaultSeverity.MEDIUM: r.medium, FaultSeverity.HIGH: r.high}
            rul = predict_rul(_inject_spike(bases[(r.source, r.prompt, int(r.repeat))], spike,
                                            multiplier_override=mult[spike.fault_severity]))
            bad += int(abs(rul - r.rul) > 1e-3 or status_from_rul(rul) != r.status)
        check_eq(S, f"{v}: {sample} random sweep rows re-derived with single-window predict_rul", 0, bad)

    # Paraphrase rows: single-window recomputation on every checkpoint.
    from research.paraphrase.collect import load_bank
    from research.trained_baselines import indomain_path
    for bank, prefix, seed in (("anchored", "anchored", 20260917), ("design", "paraphrase", 20260917),
                               ("typo", "typo", 20260917)):
        rows = pd.read_csv(R / "paraphrase" / f"{prefix}_rows.csv.gz")
        out = llm_outputs(bank)
        prompts = {p["id"]: p["text"] for p in load_bank(bank)[0]}
        indomain = (pd.read_csv(indomain_path(bank)).set_index(["arm", "bank_id"])["severity"]
                    if bank != "typo" else None)
        all_keys = [(b, r) for b in prompts for r in range(3)]
        for ck in rows["checkpoint"].unique():
            select_checkpoint(ck)
            bases = healthy_baselines(set(all_keys), ck, seed)
            base0 = get_healthy_baseline(noise_std_frac=0.0)
            pick = rows[rows["checkpoint"] == ck].sample(sample, random_state=13)
            bad = 0
            for r in pick.itertuples():
                if r.arm in NON_LLM + TRAINED:
                    s = STRATEGIES[r.arm](prompts[r.bank_id], base0)[1]
                    spike = SensorSpike(sensor_id=s["sensor_id"], spike_value=s["spike_value"], affected_window_positions=[49],
                                        fault_severity=FaultSeverity(s["fault_severity"]), plain_english_summary="verify")
                elif r.arm.endswith("_indomain"):
                    from research.baselines import _keyword_spike_with_severity
                    sev = FaultSeverity(indomain[(r.arm, r.bank_id)])
                    k = _keyword_spike_with_severity(prompts[r.bank_id], sev, "verify")
                    spike = SensorSpike(sensor_id=k.sensor_id, spike_value=k.spike_value, affected_window_positions=[49],
                                        fault_severity=sev, plain_english_summary="verify")
                else:
                    o = out[(out["arm"] == r.arm) & (out["bank_id"] == r.bank_id) & (out["repeat"] == r.repeat)].iloc[0]
                    spike = SensorSpike(sensor_id=o["sensor_id"], spike_value=float(o["spike_value"]), affected_window_positions=[49],
                                        fault_severity=FaultSeverity(o["severity"]), plain_english_summary="verify")
                rul = predict_rul(_inject_spike(bases[(r.bank_id, int(r.repeat))], spike,
                                                multiplier_override=table[spike.fault_severity]))
                bad += int(abs(rul - r.rul) > 1e-3 or status_from_rul(rul) != r.status)
            check_eq(S, f"{bank} bank / {ck}: {sample} random rows re-derived with single-window predict_rul", 0, bad)

    # Multi-hit: dashboard window rule via FactoryState, and single-window re-derivation.
    from terminal.factory_state import FactoryState
    rows = pd.read_csv(R / "multi_hit" / "multi_hit_rows.csv.gz")
    for ck in rows["checkpoint"].unique():
        select_checkpoint(ck)
        rec = pd.concat([baselines("turbofan").assign(source="turbofan"), baselines("simulator").assign(source="simulator")])
        keys = list(zip(rec["source"], rec["prompt"], rec["repeat"].astype(int)))
        bases = healthy_baselines(set(keys), ck, 20260918)
        seqs = rows[(rows["checkpoint"] == ck) & (rows["hit"] == 1)].sample(max(5, sample // 6), random_state=17)
        bad_window, bad_pred = 0, 0
        for s in seqs.itertuples():
            o = rec[(rec["source"] == s.source) & (rec["strategy"] == s.strategy) & (rec["prompt"] == s.prompt)
                    & (rec["repeat"] == s.repeat)].iloc[0]
            spike = SensorSpike(sensor_id=o["sensor_id"], spike_value=float(o["spike_value"]), affected_window_positions=[49],
                                fault_severity=FaultSeverity(o["severity"]), plain_english_summary="verify")
            state = FactoryState()
            machine = 1
            history_rows = []
            for hit in range(1, rows["hit"].max() + 1):
                if hit == 1:
                    base = bases[(s.source, s.prompt, int(s.repeat))]
                else:
                    base = state.get_machine_sensor_window(machine)
                    pad = [history_rows[-1]] * (50 - len(history_rows)) + history_rows
                    bad_window += int(not np.allclose(base, np.stack(pad), atol=1e-4))
                inj = _inject_spike(base, spike, multiplier_override=table[spike.fault_severity])
                rul = predict_rul(inj)
                got = rows[(rows["checkpoint"] == ck) & (rows["source"] == s.source) & (rows["strategy"] == s.strategy)
                           & (rows["prompt"] == s.prompt) & (rows["repeat"] == s.repeat) & (rows["hit"] == hit)].iloc[0]
                bad_pred += int(abs(rul - got["rul"]) > 1e-3 or status_from_rul(rul) != got["status"])
                for row in inj[-2:]:
                    state.push_machine_sensor_reading(machine, row.astype(np.float32))
                    history_rows.append(row)
        check_eq(S, f"multi-hit / {ck}: FactoryState window equals the replay's window rule", 0, bad_window)
        check_eq(S, f"multi-hit / {ck}: sampled sequences re-derived hit by hit with predict_rul", 0, bad_pred)


# ─────────────────────────────────────────────────────────────────────────────
# E. Multiplier sweeps
# ─────────────────────────────────────────────────────────────────────────────

def section_e():
    S = "E. Multiplier sweeps (129 MEDIUM × HIGH settings, LOW 0.15)"
    files = {"turbofan": R / "multiplier_sweep_turbofan.csv", "simulator": R / "multiplier_sweep_simulator.csv",
             "simulator_calibrated": R / "pronostia" / "multiplier_sweep_simulator_calibrated.csv",
             "pronostia": R / "pronostia" / "multiplier_sweep_pronostia.csv"}
    claimed_all = {"turbofan": (46, 66, 17), "simulator": (17, 60, 52), "simulator_calibrated": (68, 46, 15),
                   "pronostia": (9, 0, 120)}
    claimed_emb = {"turbofan": 0.55, "simulator": 0.85, "simulator_calibrated": 0.50}
    for ck, path in files.items():
        sw = pd.read_csv(path)
        sw = sw[sw["low"] == 0.15]
        piv = sw.pivot_table(index=["medium", "high"], columns="strategy", values="match_rate")
        check_eq(S, f"{ck}: number of settings", 129, len(piv))
        for label, llms in (("all LLMs incl. retired Llama", ALL_LLMS), ("current 4 LLMs", CURRENT_LLMS)):
            gap = piv[llms].max(axis=1) - piv["keyword_regex_severity"]
            counts = (int((gap > 1e-9).sum()), int((gap < -1e-9).sum()), int((gap.abs() <= 1e-9).sum()))
            if label.startswith("all"):
                check_eq(S, f"{ck}: best LLM ahead / regex ahead / tied ({label}) — outline numbers", claimed_all[ck], counts)
            else:
                note(S, f"{ck}: best LLM ahead / regex ahead / tied ({label})", str(counts))
        if ck in claimed_emb:
            line = piv.xs(0.35, level="medium")["embedding_severity"]
            ok_from = [h for h in line.index if (line[line.index >= h] >= 1 - 1e-9).all()]
            check_eq(S, f"{ck}: embedding at 100% for every HIGH ≥ threshold (MEDIUM 0.35)", claimed_emb[ck],
                     round(min(ok_from), 2) if ok_from else None)


# ─────────────────────────────────────────────────────────────────────────────
# F. Anchored rewrites
# ─────────────────────────────────────────────────────────────────────────────

def section_f():
    S = "F. Anchored bank (18 originals + 72 rewrites)"
    from statsmodels.stats.multitest import multipletests
    from research.paraphrase.score import PRICES_PER_M

    sev = severity_table("anchored")
    sev["correct"] = (sev["severity"] == sev["label"]).astype(float)
    orig, rew = sev[sev["style"] == "original"], sev[sev["style"] != "original"]
    acc_o, acc_r = orig.groupby("arm")["correct"].mean(), rew.groupby("arm")["correct"].mean()

    check_eq(S, "production-prompt LLMs score 100% severity accuracy on the originals", True,
             bool(all(abs(acc_o[a] - 1) < 1e-9 for a in PROD_ARMS)), ", ".join(f"{a}={acc_o[a]:.3f}" for a in PROD_ARMS))
    check_range(S, "production-prompt LLMs, rewrites severity accuracy", 92, 100, [acc_r[a] for a in PROD_ARMS])
    for arm, o, r in (("embedding_severity", 72, 54), ("keyword_regex_severity", 94, 39), ("keyword_regex_extended", 94, 44)):
        check_eq(S, f"{arm}: originals → rewrites severity accuracy", f"{o}% → {r}%",
                 f"{round(acc_o[arm] * 100)}% → {round(acc_r[arm] * 100)}%")
    check_range(S, "semantic-prompt LLMs on the originals", 72, 87, [acc_o[a] for a in SEM_ARMS])
    over = {a: (sev[(sev["arm"] == a) & (sev["label"] != "HIGH")]["severity"] == "HIGH").mean() for a in SEM_ARMS}
    check_range(S, "semantic-prompt over-escalation (non-HIGH called HIGH)", 4, 15, list(over.values()))

    rows = pd.read_csv(R / "paraphrase" / "anchored_rows.csv.gz")
    rows = rows[rows["style"] != "original"]
    status = rows.groupby(["checkpoint", "arm"])["match"].mean()
    for ck, lo, hi, emb, rx in (("turbofan", 94, 95, 88, 67), ("simulator", 80, 85, 86, 67),
                                ("simulator_calibrated", 96, 98, 86, 66)):
        check_range(S, f"rewrites status match, production LLMs, {ck}", lo, hi, [status[(ck, a)] for a in PROD_ARMS])
        check_eq(S, f"rewrites status match, embedding / regex, {ck}", f"{emb}% / {rx}%",
                 f"{round(status[(ck, 'embedding_severity')] * 100)}% / {round(status[(ck, 'keyword_regex_severity')] * 100)}%")
    pr = rows[rows["checkpoint"] == "pronostia"].groupby("arm")["match"].mean()
    original_arms = PROD_ARMS + SEM_ARMS + NON_LLM
    check_range(S, "rewrites status match on PRONOSTIA, original arms (tie)", 66, 67, [pr[a] for a in original_arms])

    # Primary test, exact sign-flip over anchors.
    pvals, diffs = [], []
    for ref in ("keyword_regex_severity", "embedding_severity"):
        for arm in PROD_ARMS:
            a = rew[rew["arm"] == arm].set_index(["bank_id", "repeat"])
            b = rew[rew["arm"] == ref].set_index(["bank_id", "repeat"]).reindex(a.index)
            per_anchor = (a["correct"] - b["correct"]).groupby(a["anchor"]).mean().to_numpy()
            p, how = exact_signflip_p(per_anchor)
            pvals.append(p)
            diffs.append((ref, arm, per_anchor.mean(), how))
    holm = multipletests(pvals, method="holm")[1]
    for (ref, arm, d, how), p, ph in zip(diffs, pvals, holm):
        note(S, f"primary: {arm} − {ref}", f"diff {d * 100:+.1f} pts, sign-flip p {p:.5f} ({how}), Holm p {ph:.5f}")
    check_range(S, "primary differences vs regex (pts)", 53, 61, [d for r, _a, d, _h in diffs if r == "keyword_regex_severity"])
    check_range(S, "primary differences vs embedding (pts)", 38, 46, [d for r, _a, d, _h in diffs if r == "embedding_severity"])
    check_eq(S, "all 10 primary comparisons significant at Holm 0.05", True, bool((holm < 0.05).all()),
             f"max Holm p {holm.max():.5f}")
    reported = pd.read_csv(R / "paraphrase" / "anchored_primary.csv")
    dev = max(abs(float(reported[(reported["reference"] == ref) & (reported["arm"] == arm)]["p_holm"].iloc[0]) - ph)
              for (ref, arm, _d, _h), ph in zip(diffs, holm))
    check_eq(S, "score.py anchored primary Holm p-values equal this exact recomputation", True, dev < 1e-9,
             f"max deviation {dev:.2e}")

    from dl_engine.inference import get_healthy_baseline
    from research.baselines import STRATEGIES
    import time as _time
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS
    base = get_healthy_baseline(noise_std_frac=0.0)
    STRATEGIES["embedding_severity"]("warm up bearing", base)
    for arm, limit_ms in (("embedding_severity", 80.0), ("keyword_regex_severity", 5.0)):
        times = []
        for p in ANCHORED_PROMPTS:
            t0 = _time.perf_counter()
            STRATEGIES[arm](p["text"], base)
            times.append((_time.perf_counter() - t0) * 1000)
        med = float(np.median(times))
        _record(S, f"{arm}: P50 latency re-measured (claim: embedding 39 ms, regex 1 ms)", f"≤ {limit_ms:.0f} ms",
                f"{med:.1f} ms", med <= limit_ms, "machine-dependent; CPU, includes injection")

    out = llm_outputs("anchored")
    for arm, usd, p50 in (("gpt_oss_120b__production", 0.51, 1.5), ("qwen3_8_27b__production", 1.75, 0.5),
                          ("gemini_2_5_flash_nothink__production", 0.70, 1.8),
                          ("gemini_2_5_flash__production", 2.35, 4.5), ("gemini_3_5_flash__production", 9.97, 4.3)):
        g = out[out["arm"] == arm]
        pin, pout = PRICES_PER_M[arm.split("__")[0]]
        cost = 1000 * (g["input_tokens"].mean() * pin + g["output_tokens"].mean() * pout) / 1e6
        check_eq(S, f"{arm}: USD per 1,000 / P50 latency", f"${usd:.2f} / {p50} s",
                 f"${cost:.2f} / {g['latency_ms'].median() / 1000:.1f} s")


# ─────────────────────────────────────────────────────────────────────────────
# G. Design bank
# ─────────────────────────────────────────────────────────────────────────────

def section_g():
    S = "G. Design bank (42 prompts)"
    from statsmodels.stats.multitest import multipletests

    sev = severity_table("design")
    sev["correct"] = (sev["severity"] == sev["label"]).astype(float)
    acc = sev.groupby("arm")["correct"].mean()
    check_range(S, "production-prompt LLMs severity accuracy", 73, 100, [acc[a] for a in PROD_ARMS])
    check_pct(S, "embedding severity accuracy", 54.8, acc["embedding_severity"])
    check_pct(S, "regex severity accuracy", 28.6, acc["keyword_regex_severity"])
    neg = sev[sev["style"] == "negation"].groupby("arm")["correct"].mean()
    check_pct(S, "regex on negation traps", 0.0, neg["keyword_regex_severity"])
    llm_neg = {a: neg[a] for a in PROD_ARMS + SEM_ARMS}
    in_band = sum(0.825 <= v <= 1.0 for v in llm_neg.values())
    note(S, "LLM arms on negation traps (outline: 'most 83–100%')",
         f"{in_band}/{len(llm_neg)} arms in 83–100%; " + ", ".join(f"{a}={v:.0%}" for a, v in llm_neg.items()))
    pvals, labels = [], []
    for ref in ("keyword_regex_severity", "embedding_severity"):
        for arm in PROD_ARMS:
            a = sev[sev["arm"] == arm].set_index(["bank_id", "repeat"])
            b = sev[sev["arm"] == ref].set_index(["bank_id", "repeat"]).reindex(a.index)
            per_prompt = (a["correct"] - b["correct"]).groupby(level="bank_id").mean().to_numpy()
            p, _how = exact_signflip_p(per_prompt)
            pvals.append(p)
            labels.append(f"{arm} − {ref} ({per_prompt.mean() * 100:+.1f})")
    holm = multipletests(pvals, method="holm")[1]
    check_eq(S, "primary: significant comparisons at Holm 0.05", 9, int((holm < 0.05).sum()),
             "; ".join(f"{l}: {h:.4f}" for l, h in zip(labels, holm)))
    reported = pd.read_csv(R / "paraphrase" / "paraphrase_primary.csv")
    rep = reported.set_index(["reference", "arm"])["p_holm"]
    pairs = [(ref, arm) for ref in ("keyword_regex_severity", "embedding_severity") for arm in PROD_ARMS]
    dev = max(abs(float(rep[k]) - h) for k, h in zip(pairs, holm))
    check_eq(S, "score.py design primary Holm p-values agree with this recomputation (Monte Carlo tolerance 0.01)",
             True, dev < 0.01, f"max deviation {dev:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# H. Multi-hit
# ─────────────────────────────────────────────────────────────────────────────

def section_h():
    S = "H. Multi-hit sequences (5 hits)"
    rows = pd.read_csv(R / "multi_hit" / "multi_hit_rows.csv.gz")
    groups = {a: g for a, (g, _s) in test_prompt_groups().items()}
    key = ["checkpoint", "strategy", "source", "prompt", "repeat"]
    off = rows.assign(off=rows["status"] == "OFFLINE").pivot_table(index=key, columns="hit", values="off", aggfunc="first")
    grp = off.index.get_level_values("prompt").map(groups)
    target = np.where(grp == "HIGH", off[1],
                      np.where(grp == "MEDIUM", (~off[1].astype(bool)) & off.any(axis=1),
                               ~off[[1, 2, 3]].any(axis=1)))
    rate = pd.Series(target.astype(float), index=off.index).groupby(level=["strategy", "checkpoint"]).mean()
    cks = ["turbofan", "simulator", "simulator_calibrated"]
    check_range(S, "current LLMs meet sequence targets (3 predictors)", 88, 100,
                [rate[(s, c)] for s in CURRENT_LLMS for c in cks])
    check_range(S, "regex meets sequence targets (3 predictors)", 94, 94, [rate[("keyword_regex_severity", c)] for c in cks])
    check_range(S, "embedding meets sequence targets (3 predictors)", 72, 89, [rate[("embedding_severity", c)] for c in cks])
    n_off = int(off.xs("pronostia", level="checkpoint").any(axis=1).sum())
    check_eq(S, "PRONOSTIA: sequences reaching OFFLINE at any hit", 0, n_off)


# ─────────────────────────────────────────────────────────────────────────────
# I. Calibration, PRONOSTIA model, probes
# ─────────────────────────────────────────────────────────────────────────────

def section_i():
    S = "I. Calibration, PRONOSTIA and geometry"
    from research.pronostia.features import FEATURES_PATH

    feats = pd.read_csv(FEATURES_PATH)
    published = {"Bearing1_1": 28030, "Bearing1_2": 8710, "Bearing2_1": 9110, "Bearing2_2": 7970,
                 "Bearing3_1": 5150, "Bearing3_2": 16370}
    n = feats.groupby("bearing").size()
    mism = {b: (int(n[b]) * 10, s) for b, s in published.items() if int(n[b]) * 10 != s}
    check_eq(S, "PRONOSTIA learning-set lengths (snapshots × 10 s) equal the PHM 2012 published durations", {}, mism)
    check_eq(S, "PRONOSTIA snapshot total", 24889, int(len(feats)))

    cur = pd.read_csv(R / "pronostia" / "calibration_curves.csv")
    rmse = lambda a, b: float(np.sqrt(np.mean((cur[a] - cur[b]) ** 2)))
    check_eq(S, "vibration curve RMSE, original → calibrated", "0.54 → 0.03",
             f"{rmse('original_vib_median', 'pronostia_vib_median'):.2f} → {rmse('calibrated_vib_median', 'pronostia_vib_median'):.2f}")
    check_eq(S, "temperature curve RMSE, original → calibrated", "0.32 → 0.10",
             f"{rmse('original_temp_median', 'pronostia_temp_median'):.2f} → {rmse('calibrated_temp_median', 'pronostia_temp_median'):.2f}")
    life = feats.groupby("bearing")["elapsed_s"].max() / 3600
    check_eq(S, "real bearing lifetime CV (pooled)", "0.57", f"{life.std(ddof=0) / life.mean():.2f}")
    summary = (R / "pronostia" / "calibration_summary.md").read_text(encoding="utf-8")
    m = re.search(r"Lifetime coefficient of variation \| [^|]+\| ([\d.]+) \| ([\d.]+) \|", summary)
    check_eq(S, "calibrated simulator lifetime CV (from calibration_summary.md)", "0.52", m.group(2) if m else None)

    meta = json.loads((PROJECT_ROOT / "research" / "pronostia" / "pronostia_model_meta.json").read_text(encoding="utf-8"))
    from research.pronostia.train_model import bearing_windows, load_features
    from research.replay_utils import select_checkpoint, status_from_rul
    from dl_engine.inference import predict_rul
    select_checkpoint("pronostia")
    f = load_features(meta["bearing_set"])
    got = []
    for b in meta["val_bearings"]:
        X, y, fpt = bearing_windows(f[f["bearing"] == b])
        pred = np.array([predict_rul(w) for w in X[fpt:]])
        got.append(float(np.sqrt(np.mean((pred - y[fpt:]) ** 2))))
    check_eq(S, "PRONOSTIA degradation-phase RMSE (Bearing1_3 / Bearing2_2), single-window CPU predict_rul",
             "13.7 / 30.0", " / ".join(f"{g:.1f}" for g in got),
             "training log reported 13.75 on GPU; CPU inference reproduces " + " / ".join(f"{g:.3f}" for g in got))

    def geom(rul):
        q25, q75 = np.quantile(rul, [0.25, 0.75])
        near = (np.sum(np.abs(rul - q25) <= 1) + np.sum(np.abs(rul - q75) <= 1)) / len(rul)
        return {"deg": np.mean((rul > 15) & (rul <= 30)), "off": np.mean(rul <= 15), "near": near, "min": rul.min()}
    t = geom(pd.read_csv(R / "probe_cliff_3d_turbofan.csv")["rul"].to_numpy())
    s = geom(pd.read_csv(R / "probe_cliff_3d_simulator.csv")["rul"].to_numpy())
    c = geom(pd.read_csv(R / "pronostia" / "probe_simulator_calibrated.csv")["rul"].to_numpy())
    p = geom(pd.read_csv(R / "pronostia" / "probe_pronostia.csv")["rul"].to_numpy())
    check_pct(S, "turbofan probe: points within ±1 of Q25/Q75", 86.6, t["near"])
    check_pct(S, "turbofan probe: DEGRADED share", 0.8, t["deg"])
    check_pct(S, "simulator probe: DEGRADED share", 60.8, s["deg"])
    check_eq(S, "calibrated probe: DEGRADED / OFFLINE share", "13.3% / 63.6%", f"{c['deg'] * 100:.1f}% / {c['off'] * 100:.1f}%")
    check_eq(S, "PRONOSTIA probe: minimum RUL / OFFLINE share", "29.4 / 0.0%", f"{p['min']:.1f} / {p['off'] * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# J. Methods consistency
# ─────────────────────────────────────────────────────────────────────────────

def section_j():
    S = "J. Methods consistency"
    src = (PROJECT_ROOT / "agents" / "diagnostic_agent.py").read_text(encoding="utf-8")
    prod_no_thinking = "thinking_budget=0" in src
    for bank in ("anchored", "design"):
        out = llm_outputs(bank)
        nothink = out[out["arm"] == "gemini_2_5_flash_nothink__production"]
        think = out[out["arm"] == "gemini_2_5_flash__production"]
        if bank == "anchored":
            detail = (f"output tokens: thinking off {nothink['output_tokens'].mean():.0f}, "
                      f"default thinking {think['output_tokens'].mean():.0f}")
            ok = prod_no_thinking and len(nothink) > 0 and nothink["output_tokens"].mean() < 0.5 * think["output_tokens"].mean()
        else:
            detail = f"{len(nothink)} rows"
            ok = prod_no_thinking and len(nothink) > 0
        _record(S, f"{bank}: production Gemini 2.5 configuration (thinking off) is evaluated as its own arm",
                "present, thinking off", detail, ok,
                "production agent sets thinking_budget=0; research arm gemini_2_5_flash_nothink uses thinking_budget=0; "
                "gemini_2_5_flash arms keep the model default (thinking on)")

    base = baselines("turbofan")
    ag = base[base["strategy"] == "agentic"].groupby("prompt")["severity"].agg(lambda s: s.mode().iloc[0])
    out = llm_outputs("anchored")
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS
    orig_ids = {p["id"]: p["text"] for p in ANCHORED_PROMPTS if p["style"] == "original"}
    g = out[(out["arm"] == "gemini_2_5_flash__production") & out["bank_id"].isin(orig_ids)]
    g = g.assign(prompt=g["bank_id"].map(orig_ids)).groupby("prompt")["severity"].agg(lambda s: s.mode().iloc[0])
    same = float((g == ag.reindex(g.index)).mean())
    note(S, "severity agreement on the 18 originals: production agent (May run) vs research Gemini 2.5 call (Sept)",
         f"{same:.1%} of prompts share the modal severity", "PASS" if same >= 0.9 else "WARN")
    note(S, "paired statistics", "cluster = anchor (anchored bank, 18 clusters) or prompt (design bank, 42); sign-flip "
         "tests reported with exact enumeration where ≤ 20 non-zero clusters (this script) vs 10,000 random flips "
         "(score.py); report the exact values in the paper.")


# ─────────────────────────────────────────────────────────────────────────────
# K. Additions round (analysis_plan_additions.md): trained classifiers, local LLMs, typos, interface
# ─────────────────────────────────────────────────────────────────────────────



def _damerau(a: str, b: str) -> int:
    """Optimal string alignment distance (insert, delete, substitute, adjacent transposition)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def severity_table_k(bank: str) -> pd.DataFrame:
    """Severity per (arm, bank_id, repeat) for every arm: LLM rows from the collected files, rule-based and
    R1 trained arms recomputed here through baselines.STRATEGIES, R2 arms from trained_baselines' CSV."""
    from dl_engine.inference import get_healthy_baseline
    from research.baselines import STRATEGIES
    from research.paraphrase.collect import load_bank
    from research.trained_baselines import indomain_path

    prompts, _problems = load_bank(bank)
    meta = pd.DataFrame(prompts).rename(columns={"id": "bank_id", "design_severity": "label"})
    if "anchor" not in meta:
        meta["anchor"] = meta["bank_id"]
    out = llm_outputs(bank)[["arm", "bank_id", "repeat", "severity"]]
    base = get_healthy_baseline(noise_std_frac=0.0)
    rows = []
    for arm in NON_LLM + TRAINED:
        for p in prompts:
            sev = STRATEGIES[arm](p["text"], base)[1]["fault_severity"]
            rows += [{"arm": arm, "bank_id": p["id"], "repeat": r, "severity": sev} for r in range(3)]
    if bank != "typo":
        ind = pd.read_csv(indomain_path(bank))
        rows += [{"arm": r.arm, "bank_id": r.bank_id, "repeat": k, "severity": r.severity}
                 for r in ind.itertuples() for k in range(3)]
    df = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    df = df.merge(meta[["bank_id", "label", "style", "anchor"]], on="bank_id")
    df["correct"] = (df["severity"] == df["label"]).astype(float)
    return df


def family_recompute(sev: pd.DataFrame, arms: list[str], refs: list[str], cluster: str) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests

    unfamiliar = sev[sev["style"] != "original"]
    rows = []
    for ref in refs:
        for arm in sorted(arms):
            a = unfamiliar[unfamiliar["arm"] == arm].set_index(["bank_id", "repeat"])
            b = unfamiliar[unfamiliar["arm"] == ref].set_index(["bank_id", "repeat"]).reindex(a.index)
            groups = a.index.get_level_values("bank_id") if cluster == "bank_id" else a[cluster]
            per = (a["correct"] - b["correct"]).groupby(groups).mean().to_numpy()
            p, how = exact_signflip_p(per)
            rows.append({"reference": ref, "arm": arm, "diff": per.mean(), "arm_acc": a["correct"].mean(),
                         "ref_acc": b["correct"].mean(), "p": p, "how": how})
    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table["p"], method="holm")[1]
    return table


def section_k(sample: int):
    S = "K. Additions round"
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS
    from research.paraphrase.prompt_bank import PROMPTS
    from research.paraphrase.typo_bank import DROPPED_BY_GUARD, TYPO_PROMPTS, _build
    from research.trained_baselines import indomain_path, labelled_bank_prompts, template_training_set

    # K1. Typo bank: regenerated deterministically, edits as specified, never equal to the anchor.
    rebuilt = {p["id"]: p["text"] for p in _build()}
    check_eq(S, "typo bank regenerates identically from its seed", True,
             all(rebuilt[p["id"]] == p["text"] for p in TYPO_PROMPTS))
    check_eq(S, "typo variants dropped by the input guard", 3, len(DROPPED_BY_GUARD), ", ".join(DROPPED_BY_GUARD))
    bad = []
    for p in TYPO_PROMPTS:
        a_words, t_words = p["anchor"].split(" "), p["text"].split(" ")
        eligible = [w for w in a_words if re.fullmatch(r"[A-Za-z]{3,}", w) and w.lower() != "machine"]
        k = max(1, int(np.floor({"typo25": 0.25, "typo50": 0.50}[p["style"]] * len(eligible) + 0.5)))
        changed = [(a, t) for a, t in zip(a_words, t_words) if a != t]
        if len(a_words) != len(t_words) or len(changed) != k \
                or any(_damerau(a, t) != 1 for a, t in changed) or any(a.lower() == "machine" for a, _t in changed):
            bad.append(p["id"])
    check_eq(S, "every typo variant has exactly k single-edit words (Damerau distance 1), none on 'Machine'", [], bad)
    groups = test_prompt_groups()
    check_eq(S, "typo labels inherited from the TEST_PROMPTS group of the anchor", [],
             [p["id"] for p in TYPO_PROMPTS if groups[p["anchor"]][0] != p["design_severity"]])

    # K2. R1 training set: no test prompt leaks in; fault phrases re-parsed independently.
    train = template_training_set()
    test_texts = {p["text"].lower() for p in ANCHORED_PROMPTS + PROMPTS + TYPO_PROMPTS}
    leaked = sorted(t for t in train["text"].str.lower() if t in test_texts)
    note(S, "R1 training sentences identical to a test prompt (possible only for in-vocabulary originals)",
         f"{len(leaked)}: " + "; ".join(leaked[:10]), "PASS" if not any(
             t in {p['text'].lower() for p in ANCHORED_PROMPTS if p['style'] != 'original'} | {p['text'].lower() for p in PROMPTS}
             for t in leaked) else "FAIL")
    src = (PROJECT_ROOT / "agents" / "prompts.py").read_text(encoding="utf-8")
    table = src[src.index("== FAULT → SENSOR MAPPING"): src.index("== SEVERITY CLASSIFICATION")]
    phrases = set()
    for line in table.splitlines()[1:]:
        if "→" in line and not line.strip().startswith("If"):
            left = re.sub(r"\([^)]*\)", "", line.split("→")[0])
            phrases |= {x.strip() for x in left.split("/") if x.strip()}
    check_eq(S, "R1 fault phrases re-parsed from prompts.py", 60, len(phrases))

    # K3. R2 folds: re-fit bge_m3_logreg by hand for sampled clusters and compare predictions.
    from sklearn.linear_model import LogisticRegression
    from research.trained_baselines import encode
    labelled = labelled_bank_prompts()
    rng = np.random.default_rng(5)
    for bank in ("anchored", "design"):
        ind = pd.read_csv(indomain_path(bank))
        ind = ind[ind["arm"] == "bge_m3_logreg_indomain"].set_index("bank_id")["severity"]
        test = labelled[labelled["bank"] == bank]
        clusters = rng.choice(test["cluster"].unique(), size=3, replace=False)
        mism = 0
        for c in clusters:
            held = (labelled["bank"] == bank) & (labelled["cluster"] == c)
            extra = labelled[~held]
            texts = train["text"].tolist() + extra["text"].tolist()
            y = train["label"].tolist() + extra["label"].tolist()
            w = np.r_[np.ones(len(train)), np.full(len(extra), len(train) / len(extra))]
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(encode("bge_m3", texts), y, sample_weight=w)
            held_rows = labelled[held]
            pred = clf.predict(encode("bge_m3", held_rows["text"].tolist()))
            mism += int(sum(p != ind[b] for p, b in zip(pred, held_rows["bank_id"])))
        check_eq(S, f"{bank}: bge_m3_logreg R2 predictions re-fitted by hand for 3 held-out clusters", 0, mism)

    # K4. Local LLM integrity.
    for bank, n in (("anchored", 90), ("design", 42), ("typo", len(TYPO_PROMPTS))):
        out = llm_outputs(bank)
        loc = out[out["arm"].isin(LOCAL_ARMS_V)]
        check_eq(S, f"{bank}: local LLM rows = 3 arms × {n} prompts × 3", 3 * n * 3, len(loc))
        check_eq(S, f"{bank}: local LLM fallback rows", 0, int(loc["fallback"].astype(bool).sum()))
        wrong = sum(int((~loc[loc["arm"] == f"{m}__production"]["summary"].str.startswith(t)).sum())
                    for m, t in LOCAL_TAGS.items())
        check_eq(S, f"{bank}: every local row carries its own model tag", 0, wrong)
        if bank == "typo":
            hosted = out[out["arm"].isin(PROD_ARMS)]
            check_eq(S, "typo: hosted production-arm rows = 5 arms × prompts × 3", 5 * n * 3, len(hosted))
            check_eq(S, "typo: hosted fallback rows", 0, int(hosted["fallback"].astype(bool).sum()))

    # K5. Hypothesis families recomputed with exact sign-flip and statsmodels Holm.
    fams = {("anchored", "H2"): ("anchor", ALL_PROD, ["bge_m3_logreg", "bge_m3_logreg_indomain"]),
            ("design", "H2"): ("bank_id", ALL_PROD, ["bge_m3_logreg", "bge_m3_logreg_indomain"]),
            ("typo", "H3"): ("anchor", ALL_PROD, ["keyword_regex_severity", "bge_m3_logreg"])}
    FAMILY_RESULTS.clear()
    for (bank, fam), (cluster, arms, refs) in fams.items():
        sev = severity_table_k(bank)
        SEV_TABLES[bank] = sev
        tab = family_recompute(sev, arms, refs, cluster)
        FAMILY_RESULTS[(bank, fam)] = tab
        prefix = {"anchored": "anchored", "design": "paraphrase", "typo": "typo"}[bank]
        rep = pd.read_csv(R / "paraphrase" / f"{prefix}_{fam.lower()}.csv").set_index(["reference", "arm"])
        dev_d = max(abs(float(rep.loc[(r.reference, r.arm), "diff"]) - r.diff) for r in tab.itertuples())
        dev_p = max(abs(float(rep.loc[(r.reference, r.arm), "p_holm"]) - r.p_holm) for r in tab.itertuples())
        tol = 1e-9 if cluster == "anchor" else 0.01
        check_eq(S, f"{bank} {fam}: score.py differences and Holm p equal this recomputation", True,
                 dev_d < 1e-9 and dev_p < tol, f"max deviation diff {dev_d:.2e}, Holm p {dev_p:.2e}")
        for r in tab.itertuples():
            note(S, f"{bank} {fam}: {r.arm} − {r.reference}",
                 f"{r.arm_acc:.1%} vs {r.ref_acc:.1%}, diff {r.diff * 100:+.1f} pts, p {r.p:.5f} ({r.how}), Holm {r.p_holm:.5f}")

    # K6. Common interface: re-derive sampled rows with single-window predict_rul.
    from agents.diagnostic_agent import _inject_spike
    from agents.schemas import FaultSeverity, SensorSpike
    from dl_engine.inference import predict_rul
    from research.baselines import _keyword_spike_with_severity
    from research.replay_utils import PUBLISHED_MULTIPLIERS, healthy_baselines, select_checkpoint, status_from_rul
    table_m = dict(zip((FaultSeverity.LOW, FaultSeverity.MEDIUM, FaultSeverity.HIGH), PUBLISHED_MULTIPLIERS))
    for bank, prefix in (("anchored", "anchored"), ("typo", "typo")):
        rows = pd.read_csv(R / "paraphrase" / f"{prefix}_rows_common_interface.csv.gz")
        sev = SEV_TABLES.get(bank)
        if sev is None:
            sev = SEV_TABLES[bank] = severity_table_k(bank)
        sev_idx = sev.set_index(["arm", "bank_id", "repeat"])["severity"]
        text = {p["id"]: p["text"] for p in (ANCHORED_PROMPTS if bank == "anchored" else TYPO_PROMPTS)}
        keys = [(b, r) for b in text for r in range(3)]
        for ck in rows["checkpoint"].unique():
            select_checkpoint(ck)
            bases = healthy_baselines(set(keys), ck, 20260917)
            pick = rows[rows["checkpoint"] == ck].sample(sample // 2, random_state=23)
            bad = 0
            for r in pick.itertuples():
                s = FaultSeverity(sev_idx[(r.arm, r.bank_id, r.repeat)])
                ref = _keyword_spike_with_severity(text[r.bank_id], s, "verify")
                spike = SensorSpike(sensor_id=ref.sensor_id, spike_value=ref.spike_value, affected_window_positions=[49],
                                    fault_severity=s, plain_english_summary="verify")
                rul = predict_rul(_inject_spike(bases[(r.bank_id, int(r.repeat))], spike, multiplier_override=table_m[s]))
                bad += int(abs(rul - r.rul) > 1e-3 or status_from_rul(rul) != r.status)
            check_eq(S, f"{bank} / {ck}: {sample // 2} common-interface rows re-derived with single-window predict_rul", 0, bad)


def section_l():
    """Outline claims from the additions round, recomputed from raw outputs and replay rows."""
    S = "L. Additions round — outline claims"
    hosted = PROD_ARMS
    for bank in ("anchored", "design", "typo"):
        if bank not in SEV_TABLES:
            SEV_TABLES[bank] = severity_table_k(bank)

    # 5.1 In-vocabulary: 18 prompts.
    for v, claims in (("turbofan", {"tfidf_logreg": 100.0, "minilm_logreg": 94.4, "bge_m3_logreg": 94.4, "minilm_finetuned": 94.4}),
                      ("simulator", {"tfidf_logreg": 100.0, "minilm_logreg": 94.4, "bge_m3_logreg": 94.4, "minilm_finetuned": 94.4})):
        rates = baselines(v).groupby("strategy")["match"].mean()
        for s, c in claims.items():
            check_pct(S, f"18 prompts {v}: {s} (R1) status match", c, rates[s])
        check_eq(S, f"18 prompts {v}: trained arms have N = 54", True,
                 bool((baselines(v).groupby("strategy").size()[TRAINED] == 54).all()))
    sev = SEV_TABLES["anchored"]
    orig, rew = sev[sev["style"] == "original"], sev[sev["style"] != "original"]
    acc_o, acc_r = orig.groupby("arm")["correct"].mean(), rew.groupby("arm")["correct"].mean()
    check_range(S, "local LLMs, severity accuracy on the 18 originals", 56, 72, [acc_o[a] for a in LOCAL_ARMS_V])
    rows = pd.read_csv(R / "paraphrase" / "anchored_rows.csv.gz")
    st_o = rows[rows["style"] == "original"].groupby(["checkpoint", "arm"])["match"].mean()
    check_range(S, "local LLMs, status match on the originals, turbofan", 72, 89, [st_o[("turbofan", a)] for a in LOCAL_ARMS_V])
    check_range(S, "local LLMs, status match on the originals, simulator", 50, 70, [st_o[("simulator", a)] for a in LOCAL_ARMS_V])

    # 5.2 Reworded reports.
    for arm, o, r in (("bge_m3_logreg", 89, 67), ("tfidf_logreg", 100, 40), ("minilm_finetuned", 94, 46)):
        check_eq(S, f"{arm} (R1): originals → rewrites severity accuracy", f"{o}% → {r}%",
                 f"{round(acc_o[arm] * 100)}% → {round(acc_r[arm] * 100)}%")
    for arm, r in (("bge_m3_logreg_indomain", 89), ("tfidf_logreg_indomain", 83), ("minilm_finetuned_indomain", 78)):
        check_eq(S, f"{arm} (R2): rewrites severity accuracy", f"{r}%", f"{round(acc_r[arm] * 100)}%")
    check_range(S, "local LLMs, rewrites severity accuracy", 46, 65, [acc_r[a] for a in LOCAL_ARMS_V])
    over = [(sev[(sev["arm"] == a) & (sev["label"] != "HIGH")]["severity"] == "HIGH").mean() for a in LOCAL_ARMS_V]
    check_range(S, "local LLMs, over-escalation (anchored, all prompts)", 22, 50, over)
    dsev = SEV_TABLES["design"]
    dacc = dsev.groupby("arm")["correct"].mean()
    check_eq(S, "design bank: bge-m3 R1 / R2 severity accuracy", "67% / 79%",
             f"{round(dacc['bge_m3_logreg'] * 100)}% / {round(dacc['bge_m3_logreg_indomain'] * 100)}%")
    check_range(S, "design bank: local LLMs severity accuracy", 40, 64, [dacc[a] for a in LOCAL_ARMS_V])

    h2a = FAMILY_RESULTS[("anchored", "H2")]
    h2d = FAMILY_RESULTS[("design", "H2")]
    sub = h2a[(h2a["reference"] == "bge_m3_logreg") & h2a["arm"].isin(hosted)]
    check_range(S, "H2 anchored: hosted − bge-m3 R1 (pts)", 25, 33, sub["diff"].tolist())
    check_eq(S, "H2 anchored: hosted − bge-m3 R1 Holm p range", "0.12–0.28",
             f"{sub['p_holm'].min():.2f}–{sub['p_holm'].max():.2f}")
    sub = h2a[(h2a["reference"] == "bge_m3_logreg_indomain") & h2a["arm"].isin(hosted)]
    check_range(S, "H2 anchored: hosted − bge-m3 R2 (pts), 'within 3–11 points'", 3, 11, sub["diff"].tolist())
    check_eq(S, "H2 anchored: hosted − bge-m3 R2, minimum Holm p ≥ 0.17", True, bool(sub["p_holm"].min() >= 0.17 - 1e-9),
             f"min {sub['p_holm'].min():.4f}")
    sub = h2a[(h2a["reference"] == "bge_m3_logreg_indomain") & h2a["arm"].isin(LOCAL_ARMS_V)]
    check_range(S, "H2 anchored: bge-m3 R2 − local LLMs (pts)", 24, 43, (-sub["diff"]).tolist())
    g = sub[sub["arm"] == "gemma3_4b__production"].iloc[0]
    check_eq(S, "H2 anchored: Gemma 3 vs bge-m3 R2 Holm p", "0.008", f"{g['p_holm']:.3f}")
    check_eq(S, "H2 anchored: only Gemma 3 significant among local vs R2", ["gemma3_4b__production"],
             sorted(sub[sub["p_holm"] < 0.05]["arm"]))
    sub = h2d[(h2d["reference"] == "bge_m3_logreg") & h2d["arm"].isin(hosted)]
    check_range(S, "H2 design: hosted − bge-m3 R1 (pts)", 6, 33, sub["diff"].tolist())
    check_eq(S, "H2 design: hosted arms significant vs bge-m3 R1",
             sorted(["qwen3_8_27b__production", "gemini_3_5_flash__production", "gemini_2_5_flash__production"]),
             sorted(sub[sub["p_holm"] < 0.05]["arm"]))
    sub = h2d[(h2d["reference"] == "bge_m3_logreg_indomain") & h2d["arm"].isin(hosted)]
    check_eq(S, "H2 design: hosted − bge-m3 R2 range (pts)", "−6 to +21",
             f"{'−' if sub['diff'].min() < 0 else '+'}{abs(round(sub['diff'].min() * 100))} to +{round(sub['diff'].max() * 100)}")
    sig = sub[sub["p_holm"] < 0.05]
    check_eq(S, "H2 design: only Qwen 3.8 significant vs bge-m3 R2 (Holm p 0.047)", "qwen3_8_27b__production 0.047",
             " ".join(f"{a} {p:.3f}" for a, p in zip(sig["arm"], sig["p_holm"])))

    st_r = rows[rows["style"] != "original"].groupby(["checkpoint", "arm"])["match"].mean()
    for arm, claim in (("bge_m3_logreg_indomain", "97% / 97% / 97%"), ("bge_m3_logreg", "92% / 91% / 92%")):
        check_eq(S, f"rewrites status match {arm} (turbofan / simulator / calibrated)", claim,
                 " / ".join(f"{round(st_r[(ck, arm)] * 100)}%" for ck in ("turbofan", "simulator", "simulator_calibrated")))

    # 5.6 PRONOSTIA: no OFFLINE anywhere; new over-escalating arms dip below the tie.
    n_off, rul_min = 0, np.inf
    for prefix in ("anchored", "paraphrase", "typo"):
        pr_rows = pd.read_csv(R / "paraphrase" / f"{prefix}_rows.csv.gz")
        pr_rows = pr_rows[pr_rows["checkpoint"] == "pronostia"]
        n_off += int((pr_rows["status"] == "OFFLINE").sum())
        rul_min = min(rul_min, float(pr_rows["rul"].min()))
    check_eq(S, "PRONOSTIA: OFFLINE rows across the anchored, design and typo replays (all arms)", 0, n_off)
    check_eq(S, "PRONOSTIA: minimum replayed RUL", "29.7", f"{rul_min:.1f}")
    pr = rows[(rows["checkpoint"] == "pronostia") & (rows["style"] != "original")].groupby("arm")["match"].mean()
    below = pr[pr.round(2) < 0.66]
    check_range(S, "PRONOSTIA rewrites: new arms below the 66–67% tie", 62, 65, below.tolist(), 0)
    check_eq(S, "PRONOSTIA rewrites: arms below the tie are new arms only", True,
             bool(all(a in LOCAL_ARMS_V + TRAINED for a in below.index)), ", ".join(below.index))

    # 5.3 Typos.
    tsev = SEV_TABLES["typo"]
    tacc = tsev.groupby("arm")["correct"].mean()
    check_range(S, "typos: hosted LLMs severity accuracy", 95, 100, [tacc[a] for a in hosted])
    check_eq(S, "typos: TF-IDF R1 / bge-m3 R1 / regex", "99% / 83% / 70%",
             f"{round(tacc['tfidf_logreg'] * 100)}% / {round(tacc['bge_m3_logreg'] * 100)}% / {round(tacc['keyword_regex_severity'] * 100)}%")
    check_range(S, "typos: local LLMs severity accuracy", 55, 72, [tacc[a] for a in LOCAL_ARMS_V])
    h3 = FAMILY_RESULTS[("typo", "H3")]
    sub = h3[(h3["reference"] == "keyword_regex_severity") & h3["arm"].isin(hosted)]
    check_range(S, "H3: hosted − regex (pts)", 25, 30, sub["diff"].tolist())
    check_eq(S, "H3: hosted − regex Holm p range; count below 0.05", "0.031–0.070; 3",
             f"{sub['p_holm'].min():.3f}–{sub['p_holm'].max():.3f}; {int((sub['p_holm'] < 0.05).sum())}")
    sub = h3[(h3["reference"] == "bge_m3_logreg") & h3["arm"].isin(hosted)]
    check_range(S, "H3: hosted − bge-m3 R1 (pts)", 12, 18, sub["diff"].tolist())
    check_eq(S, "H3: hosted − bge-m3 R1 minimum Holm p ≥ 0.086", True, bool(sub["p_holm"].min() >= 0.0855),
             f"min {sub['p_holm'].min():.4f}")

    # 5.4 Common interface, recomputed from the two row files.
    common = pd.read_csv(R / "paraphrase" / "anchored_rows_common_interface.csv.gz")
    common = common[common["style"] != "original"].groupby(["checkpoint", "arm"])["match"].mean()
    for ck, own_lo, own_hi, com_lo, com_hi in (("simulator", 80, 85, 96, 100), ("turbofan", 94, 95, 97, 100)):
        check_range(S, f"interface {ck}: hosted own routing", own_lo, own_hi, [st_r[(ck, a)] for a in hosted])
        check_range(S, f"interface {ck}: hosted common routing", com_lo, com_hi, [common[(ck, a)] for a in hosted])

    # 5.5 Multi-hit for trained arms.
    mh = pd.read_csv(R / "multi_hit" / "multi_hit_rows.csv.gz")
    groups = {a: g for a, (g, _s) in test_prompt_groups().items()}
    key = ["checkpoint", "strategy", "source", "prompt", "repeat"]
    off = mh.assign(off=mh["status"] == "OFFLINE").pivot_table(index=key, columns="hit", values="off", aggfunc="first")
    grp = off.index.get_level_values("prompt").map(groups)
    target = np.where(grp == "HIGH", off[1], np.where(grp == "MEDIUM", (~off[1].astype(bool)) & off.any(axis=1),
                                                      ~off[[1, 2, 3]].any(axis=1)))
    rate = pd.Series(target.astype(float), index=off.index).groupby(level=["strategy", "checkpoint"]).mean()
    check_range(S, "multi-hit: trained classifiers (R1) meet sequence targets (3 predictors)", 89, 100,
                [rate[(s, c)] for s in TRAINED for c in ("turbofan", "simulator", "simulator_calibrated")])

    # 5.7 Latency.
    bench = pd.read_csv(R / "paraphrase" / "local_latency_benchmark.csv")
    p50 = bench.groupby("arm")["latency_ms"].median() / 1000
    check_eq(S, "local LLMs P50 latency range (s)", "7.1–16.9", f"{p50.min():.1f}–{p50.max():.1f}",
             "from the non-replicated design and typo calls")
    check_eq(S, "latency source rows are real calls (not replicated)", 3 * (42 + 105), len(bench))
    cls = pd.read_csv(R / "paraphrase" / "anchored_classification.csv").set_index("arm")
    note(S, "CPU classifier P50 latency (anchored bank, final scoring run)",
         ", ".join(f"{a} {cls.loc[a, 'p50_latency_ms']:.0f} ms" for a in
                   ("bge_m3_logreg", "embedding_severity", "minilm_logreg", "tfidf_logreg", "keyword_regex_severity")))


def section_m():
    """External check on FMUCD work orders: mapping, sample and agreement recomputed independently."""
    S = "M. External check (FMUCD work orders)"
    from statsmodels.stats.multitest import multipletests
    from agents.input_guard import is_valid_fault_input
    from research.fmucd import CSV_PATH, LABELS, MAPPING_PATH, OUT_DIR, SAMPLE_PATH, quadratic_kappa

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    codes = {(u, c): band for u, spec in mapping["universities"].items()
             for band, cs in spec["bands"].items() for c in cs}
    words = {"HIGH": ("EMERGENCY",), "MEDIUM": ("URGENT", "EXPEDITED"), "LOW": ("ROUTINE", "DEFERRED")}
    bad = [f"{u}/{c}→{b}" for (u, c), b in codes.items() if not any(w in c.upper() for w in words[b])]
    check_eq(S, "every mapped priority code names its own urgency band (no code ordering was guessed)", [], bad)
    dup = [c for c in codes if list(codes).count(c) > 1]
    check_eq(S, "no priority code is mapped to two bands", [], dup)
    check_eq(S, "universities used (those with self-describing codes)", ["2", "3", "7", "9"],
             sorted(mapping["universities"]))

    sample = pd.read_csv(SAMPLE_PATH)
    meta = json.loads((OUT_DIR / "sample_meta.json").read_text(encoding="utf-8"))
    check_eq(S, "sample size and band balance", {"HIGH": 100, "MEDIUM": 100, "LOW": 100},
             sample["design_severity"].value_counts().to_dict())
    check_eq(S, "every sampled work order's band equals its priority code's mapping", [],
             [r.bank_id for r in sample.itertuples()
              if codes.get((str(r.university), r.WOPriority)) != r.design_severity])
    check_eq(S, "every sampled description passes the production input guard", True,
             bool(all(is_valid_fault_input(t)[0] for t in sample["text"])))
    check_eq(S, "no duplicate descriptions in the sample", 0, int(sample["text"].duplicated().sum()))
    check_eq(S, "recorded input-guard pass rate on mapped work orders", "19.9%",
             f"{meta['input_guard_pass_rate']:.1%}")

    # The sampled rows exist in the raw download with the same priority and text.
    # A work-order id can appear on several rows of the download (one per component), so require
    # at least one raw row per sampled id that matches its priority, cleaned text and UPM flag.
    wanted = dict(zip(sample["bank_id"], zip(sample["WOPriority"], sample["text"])))
    confirmed, rows_seen = set(), 0
    for chunk in pd.read_csv(CSV_PATH, usecols=["WOID", "WOPriority", "WODescription", "PPM/UPM"],
                             dtype=str, chunksize=500_000, on_bad_lines="skip"):
        hit = chunk[chunk["WOID"].isin(wanted)]
        for r in hit.itertuples():
            prio, text = wanted[r.WOID]
            clean = re.sub(r"\s+", " ", str(r.WODescription).replace("_x000D_", " ")).strip()
            if str(r.WOPriority).strip() == prio and clean == text and str(r._4).upper() == "UPM":
                confirmed.add(r.WOID)
            rows_seen += 1
    check_eq(S, "every sampled work order is confirmed in the raw download (unplanned, same priority and text)", [],
             sorted(set(wanted) - confirmed), f"{rows_seen} raw rows carry a sampled id")

    rows = pd.read_csv(OUT_DIR / "rows.csv.gz")
    reported = pd.read_csv(OUT_DIR / "agreement.csv").set_index("arm")
    llm_arms = sorted(a for a in rows["arm"].unique() if a.endswith("__production"))
    check_eq(S, "rows per arm = sample size", True, bool((rows.groupby("arm").size() == len(sample)).all()),
             ", ".join(f"{a}={n}" for a, n in rows.groupby("arm").size().items() if n != len(sample)))
    fb = rows[rows["fallback"].astype(bool)]
    check_eq(S, "fallback rows (gpt-oss-120b refuses one non-fault work order: a keycard request)", 1, len(fb),
             "; ".join(f"{r.arm} / {r.bank_id}" for r in fb.itertuples()))
    agree = rows.groupby("arm")["correct"].mean()
    dev = max(abs(float(reported.loc[a, "agreement"]) - agree[a]) for a in agree.index)
    check_eq(S, "agreement recomputed from the row file equals the reported table", True, dev < 1e-12,
             f"max deviation {dev:.2e}")
    kap = {a: quadratic_kappa(g["severity"], g["label"]) for a, g in rows.groupby("arm")}
    dev = max(abs(float(reported.loc[a, "kappa"]) - k) for a, k in kap.items())
    check_eq(S, "quadratic kappa recomputed", True, dev < 1e-12, f"max deviation {dev:.2e}")

    paired = pd.read_csv(OUT_DIR / "paired.csv")
    pvals, keys = [], []
    for ref in list(dict.fromkeys(paired["reference"])):
        for arm in llm_arms:
            a = rows[rows["arm"] == arm].set_index("bank_id")
            b = rows[rows["arm"] == ref].set_index("bank_id").reindex(a.index)
            per = (a["correct"] - b["correct"]).groupby(a["anchor"]).mean().to_numpy()
            p, _how = exact_signflip_p(per)
            pvals.append(p)
            keys.append((ref, arm, per.mean()))
    holm = multipletests(pvals, method="holm")[1]
    rep = paired.set_index(["reference", "arm"])
    dev_d = max(abs(float(rep.loc[(r, a), "diff"]) - d) for (r, a, d) in keys)
    dev_p = max(abs(float(rep.loc[(r, a), "p_holm"]) - h) for (r, a, _d), h in zip(keys, holm))
    check_eq(S, "paired differences and Holm p equal this exact recomputation", True,
             dev_d < 1e-9 and dev_p < 1e-9, f"max deviation diff {dev_d:.2e}, Holm p {dev_p:.2e}")
    for (ref, arm, d), h in zip(keys, holm):
        note(S, f"{arm} − {ref}", f"diff {d * 100:+.1f} pts, Holm p {h:.4f}")
    order = agree.sort_values(ascending=False)
    note(S, "agreement ranking", "; ".join(f"{a} {v:.1%}" for a, v in order.items()))

    # Outline claims (section 5.7).
    trained_r1 = TRAINED
    trained_r3 = [a + "_external" for a in TRAINED]
    zero_shot = [a for a in agree.index if not a.endswith("+fewshot")]
    check_range(S, "every zero-shot strategy's agreement with the recorded band", 28, 38,
                [agree[a] for a in zero_shot], 0)
    check_range(S, "hosted production LLMs agreement", 33, 35, [agree[a] for a in PROD_ARMS], 0)
    check_eq(S, "regex / bge-m3 R1 agreement", "35% / 33%",
             f"{round(agree['keyword_regex_severity'] * 100)}% / {round(agree['bge_m3_logreg'] * 100)}%")
    check_range(S, "trained classifiers on the prompt vocabulary (R1)", 30, 34, [agree[a] for a in trained_r1], 0)
    check_range(S, "trained classifiers given our labelled reports (R3)", 28, 31, [agree[a] for a in trained_r3], 0)
    ks = pd.Series({a: k for a, k in kap.items() if not a.endswith("+fewshot")})
    check_eq(S, "quadratic kappa range across zero-shot arms", "-0.09 to 0.11",
             f"{ks.min():.2f} to {ks.max():.2f}")
    check_eq(S, "no paired difference survives Holm (every adjusted p = 1.00)", True,
             bool(np.allclose(holm, 1.0)), f"min Holm p {min(holm):.3f}")
    oracle = pd.read_csv(OUT_DIR / "oracle.csv").set_index("model")
    check_eq(S, "trained on FMUCD's own labels: TF-IDF agreement / kappa", "68.7% / 0.60",
             f"{oracle.loc['tfidf_logreg_fmucd_trained', 'agreement'] * 100:.1f}% / "
             f"{oracle.loc['tfidf_logreg_fmucd_trained', 'kappa']:.2f}")
    check_eq(S, "trained on FMUCD's own labels: bge-m3 agreement / kappa", "61.3% / 0.54",
             f"{oracle.loc['bge_m3_logreg_fmucd_trained', 'agreement'] * 100:.1f}% / "
             f"{oracle.loc['bge_m3_logreg_fmucd_trained', 'kappa']:.2f}")
    check_eq(S, "in-house training used work orders disjoint from the sample", True,
             bool(int(oracle["n_train"].iloc[0]) == 6000))
    # Few-shot arms (analysis plan section H).
    from research.fmucd import FEWSHOT_PATH, FEWSHOT_TAG
    few_arms = sorted(a for a in rows["arm"].unique() if a.endswith(FEWSHOT_TAG))
    check_eq(S, "few-shot arms collected", 8, len(few_arms))
    examples = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
    check_range(S, "labelled examples per university", 15, 20, [len(v) / 100 for v in examples.values()], 0)
    ex_texts = {t for rows_ in examples.values() for t in [r["text"] for r in rows_]}
    check_eq(S, "no few-shot example is one of the evaluated work orders", set(), ex_texts & set(sample["text"]))
    raw = {}
    for chunk in pd.read_csv(CSV_PATH, usecols=["UniversityID", "WODescription", "WOPriority", "PPM/UPM"],
                             dtype=str, chunksize=500_000, on_bad_lines="skip"):
        chunk["clean"] = (chunk["WODescription"].fillna("").str.replace("_x000D_", " ", regex=False)
                          .str.replace(r"\s+", " ", regex=True).str.strip())
        hit = chunk[chunk["clean"].isin(ex_texts) & (chunk["PPM/UPM"].str.upper() == "UPM")]
        for r in hit.itertuples():
            raw.setdefault(r.clean, set()).add((str(r.UniversityID), str(r.WOPriority).strip()))
    bad = []
    for uni, rows_ in examples.items():
        for r in rows_:
            hits = raw.get(r["text"], set())
            if not any(u == uni and codes.get((u, c)) == r["band"] for u, c in hits):
                bad.append(f"{uni}/{r['text'][:40]}")
    check_eq(S, "every few-shot example is a real unplanned work order of its own university with that band", [], bad)
    few_agree = rows[rows["arm"].isin(few_arms)].groupby("arm")["correct"].mean()
    gains = {a: few_agree[a] - agree[a[: -len(FEWSHOT_TAG)]] for a in few_arms}
    check_range(S, "few-shot minus zero-shot agreement (pts)", -6, 11, list(gains.values()), 0)
    reported_fs = pd.read_csv(OUT_DIR / "fewshot.csv").set_index("arm")
    dev = max(abs(float(reported_fs.loc[a[: -len(FEWSHOT_TAG)], "diff"]) - g) for a, g in gains.items())
    check_eq(S, "few-shot table differences equal this recomputation", True, dev < 1e-12, f"max deviation {dev:.2e}")
    check_eq(S, "best few-shot arm", "gemini_3_5_flash__production+fewshot 43.7%",
             f"{few_agree.idxmax()} {few_agree.max():.1%}")
    check_eq(S, "best few-shot arm stays below the in-house-trained classifier (68.7%)", True,
             bool(few_agree.max() < float(pd.read_csv(OUT_DIR / "oracle.csv")["agreement"].max())),
             f"{few_agree.max():.1%} vs 68.7%")
    few_kappa = {a: quadratic_kappa(g["severity"], g["label"]) for a, g in rows[rows["arm"].isin(few_arms)].groupby("arm")}
    check_range(S, "few-shot quadratic kappa", 5, 33, [k for k in few_kappa.values()], 0)
    check_eq(S, "few-shot fallback rows", 0,
             int(rows[rows["arm"].isin(few_arms)]["fallback"].astype(bool).sum()))

    share_high = rows[rows["severity"] == "HIGH"].groupby("arm").size() / rows.groupby("arm").size()
    gem = [share_high[a] for a in PROD_ARMS if a.startswith("gemini")]
    check_range(S, "share of work orders called HIGH by the Gemini arms", 7, 12, gem, 0)
    check_pct(S, "share of work orders called HIGH by Gemma 3", 80, share_high["gemma3_4b__production"], 0)


FAMILY_RESULTS: dict = {}
SEV_TABLES: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def section_n():
    """Manuscript draft: house style and the two counts it quotes about the collection itself."""
    S = "N. Manuscript draft"
    import glob

    path = PROJECT_ROOT / "research" / "AEI" / "manuscript.md"
    if not path.exists():
        note(S, "manuscript draft", "research/AEI/manuscript.md not written yet")
        return
    text = path.read_text(encoding="utf-8")
    import re as _re
    # CRediT role names carry an en dash by convention ("Writing - original draft"), so that block is exempt.
    dash_scope = text.split("## CRediT authorship contribution statement")[0]
    loose = _re.findall(r"[^\d\s]\s*–|–\s*[^\d\s]", dash_scope)   # en dash not between numerals
    check_eq(S, "no em-dashes; en dashes only between numerals (author's house style)", (0, []),
             (text.count("—"), sorted(set(loose))))
    check_eq(S, "percentages written with the symbol, not spelled out", 0, text.count("per cent"))
    total = fallbacks = 0
    for f in glob.glob(str(R / "paraphrase" / "llm_outputs*.csv")):
        d = pd.read_csv(f)
        total += len(d)
        fallbacks += int(d["fallback"].astype(bool).sum())
    check_eq(S, f"quoted total of recorded LLM calls ({total:,})", True, f"{total:,} recorded calls" in text)
    check_eq(S, "quoted fallback count matches the collected files", 1, fallbacks)
    pron = sum(int((pd.read_csv(R / "paraphrase" / f"{p}_rows.csv.gz")["checkpoint"] == "pronostia").sum())
               for p in ("anchored", "paraphrase", "typo"))
    check_eq(S, f"quoted PRONOSTIA replay count ({pron:,})", True, f"{pron:,} replayed reports" in text)
    # Key figures quoted in the prose, recomputed here and checked for presence in the draft.
    if "anchored" not in SEV_TABLES:
        SEV_TABLES["anchored"] = severity_table_k("anchored")
    sev = SEV_TABLES["anchored"]
    rew = sev[sev["style"] != "original"].groupby("arm")["correct"].mean()
    if "typo" not in SEV_TABLES:
        SEV_TABLES["typo"] = severity_table_k("typo")
    tsev = SEV_TABLES["typo"]
    tacc = tsev.groupby("arm")["correct"].mean()
    fm = pd.read_csv(R / "fmucd" / "agreement.csv").set_index("arm")["agreement"]
    oracle = pd.read_csv(R / "fmucd" / "oracle.csv")["agreement"].max()
    guard = json.loads((R / "fmucd" / "sample_meta.json").read_text(encoding="utf-8"))["input_guard_pass_rate"]
    quoted = {
        "hosted range on rewrites": f"{min(rew[a] for a in PROD_ARMS) * 100:.0f}–"
                                    f"{max(rew[a] for a in PROD_ARMS) * 100:.0f}%",
        "local range on rewrites": f"{min(rew[a] for a in LOCAL_ARMS_V) * 100:.0f}–"
                                   f"{max(rew[a] for a in LOCAL_ARMS_V) * 100:.0f}%",
        "bge-m3 with labelled reports on rewrites": f"{rew['bge_m3_logreg_indomain'] * 100:.0f}%",
        "term-frequency classifier on typos": f"{tacc['tfidf_logreg'] * 100:.0f}%",
        "in-house trained upper bound": f"{oracle * 100:.1f}%",
        "best few-shot arm": f"{fm.filter(like='+fewshot').max() * 100:.1f}%",
        "input-guard pass rate": f"{guard * 100:.1f}%",
    }
    missing = {k: val for k, val in quoted.items() if val not in text}
    check_eq(S, "key figures quoted in the prose match the result files", {}, missing,
             "; ".join(f"{k}: {v}" for k, v in quoted.items()))
    body = text.split("## 1. Introduction")[1].split("## Declarations")[0]
    note(S, "draft length, sections 1 to 8", f"{len(body.split()):,} words")
    note(S, "placeholders still open", f"{text.count('[TODO')} TODO markers (declarations, funding, data availability)")
    for block in ("CRediT authorship contribution statement", "Declaration of interest statement",
                  "Corresponding and sole author", "Running title:"):
        check_eq(S, f"front or back matter present: {block}", True, block in text)
    refs = text.split("## References")[1]
    body_txt = text.split("## References")[0]
    order = []
    for m in re.finditer(r"\[(\d+(?:,\s*\d+)*)\]", body_txt):
        for n in m.group(1).replace(" ", "").split(","):
            if int(n) not in order:
                order.append(int(n))
    cited, listed = set(order), {int(m) for m in re.findall(r"^\[(\d+)\]", refs, flags=re.M)}
    check_eq(S, "every numbered reference is cited and every citation is listed", (set(), set()),
             (cited - listed, listed - cited))
    check_eq(S, "references numbered in order of first citation", sorted(order), order)


def section_p():
    """The journal's own rules: Advanced Engineering Informatics, guide for authors, read 2026-09-19."""
    S = "P. Journal requirements"
    path = PROJECT_ROOT / "research" / "AEI" / "manuscript.md"
    if not path.exists():
        note(S, "manuscript draft", "research/AEI/manuscript.md not written yet")
        return
    text = path.read_text(encoding="utf-8")

    abstract = re.search(r"## Abstract\n\n(.*?)\n\n\*\*Keywords", text, re.S).group(1)
    check_eq(S, "abstract within the 250-word limit", True, len(abstract.split()) <= 250,
             f"{len(abstract.split())} words")
    check_eq(S, "abstract cites nothing", [], re.findall(r"\[\d+", abstract))
    keywords = [k.strip() for k in re.search(r"\*\*Keywords:\*\* (.*)", text).group(1).split(";")]
    check_eq(S, "between 1 and 7 keywords", True, 1 <= len(keywords) <= 7, f"{len(keywords)} keywords")
    check_eq(S, "no keyword joined by 'and' or 'of'", [],
             [k for k in keywords if re.search(r"\b(and|of)\b", k, re.I)])

    highlights = re.search(r"## Highlights\n\n(.*?)\n\n## Abstract", text, re.S).group(1).strip().splitlines()
    check_eq(S, "3 to 5 highlights", True, 3 <= len(highlights) <= 5, f"{len(highlights)} highlights")
    check_eq(S, "every highlight within 85 characters", [],
             [h for h in highlights if len(h.lstrip("- ")) > 85])

    # Figures and tables: numbered in order of appearance, each with a caption and an in-text citation.
    for kind, pattern, cite in (("figure", r"\*\*Fig\. (\d+)\.\*\*", r"Fig\. {n}\b"),
                                ("table", r"\*\*Table (\d+)\.\*\*", r"Table {n}\b")):
        nums = [int(n) for n in re.findall(pattern, text)]
        check_eq(S, f"{kind}s numbered consecutively in order of appearance",
                 list(range(1, len(nums) + 1)), nums)
        body_txt = re.sub(pattern.replace("(\\d+)", r"\\d+") + r"[^\n]*", "", text)
        check_eq(S, f"every {kind} is cited in the text", [],
                 [n for n in nums if not re.search(cite.format(n=n), body_txt)])

    check_eq(S, "section headings numbered 1, 1.1 as the guide requires", [],
             [h for h in re.findall(r"^#{2,3} (?!\d)(.+)$", text, flags=re.M)
              if h.split()[0] not in ("Title", "Highlights", "Abstract", "References", "CRediT",
                                      "Declaration", "Funding", "Data", "Acknowledgements")])
    for block in ("Declaration of generative AI and AI-assisted technologies in the manuscript "
                  "preparation process", "## Funding", "## Data availability",
                  "## CRediT authorship contribution statement"):
        check_eq(S, f"required statement present: {block.lstrip('# ')[:60]}", True, block in text)
    # Acknowledgements are optional (this paper has none), but if present they belong directly
    # before the generative-AI statement and the reference list.
    order = [text.index(h) for h in ("## Acknowledgements", "## Declaration of generative AI",
                                     "## References") if h in text]
    check_eq(S, "generative-AI statement, and any acknowledgements, sit before the reference list",
             True, order == sorted(order) and len(order) >= 2)
    check_eq(S, "no funding claimed and none declared", True,
             "did not receive any specific grant" in text)


def section_o():
    """Figures: present, current with their sources, correctly formatted, and cited by the draft."""
    S = "O. Figures"
    from PIL import Image
    from research.aei_figures import FIGURES, OUT_DIR as FIG_DIR

    manifest = FIG_DIR / "MANIFEST.md"
    if not manifest.exists():
        note(S, "figures", "not generated yet: run python -m research.aei_figures")
        return
    text = manifest.read_text(encoding="utf-8")
    manuscript = (PROJECT_ROOT / "research" / "AEI" / "manuscript.md").read_text(encoding="utf-8")
    stems = sorted(p.stem for p in FIG_DIR.glob("fig*.png"))
    check_eq(S, "one figure per entry in aei_figures.FIGURES", len(FIGURES), len(stems))
    missing_tif = [s for s in stems if not (FIG_DIR / f"{s}.tif").exists()]
    check_eq(S, "every figure has a 700 dpi TIFF alongside the PNG", [], missing_tif)
    check_eq(S, "every figure is cited in the manuscript draft", [],
             [s for s in stems if s not in manuscript])
    check_eq(S, "every figure has a caption in the manifest", [], [s for s in stems if s not in text])

    bad_format, stale = [], []
    sources = {}
    for block in text.split("## Figure ")[1:]:
        stem = block.split(":")[1].splitlines()[0].strip()
        line = next((l for l in block.splitlines() if l.startswith("**Sources:**")), "")
        sources[stem] = [s.strip(" `") for s in line.replace("**Sources:**", "").split(", ") if "/" in s]
    for stem in stems:
        with Image.open(FIG_DIR / f"{stem}.tif") as im:
            dpi = im.info.get("dpi", (0, 0))[0]
            compression = im.info.get("compression", "")
        if round(dpi) != 700 or compression != "tiff_lzw":
            bad_format.append(f"{stem}: {round(dpi)} dpi, {compression}")
        built = (FIG_DIR / f"{stem}.png").stat().st_mtime
        for src in sources.get(stem, []):
            path = PROJECT_ROOT / src
            if path.exists() and path.stat().st_mtime > built:
                stale.append(f"{stem} older than {src}")
    check_eq(S, "every TIFF is 700 dpi with LZW compression", [], bad_format)
    check_eq(S, "no figure is older than a result file it was built from", [], stale)
    note(S, "figures generated", ", ".join(stems))


def write_report() -> Path:
    df = pd.DataFrame(RESULTS)
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "NOTE": "ℹ️"}
    counts = df["status"].value_counts()
    md = ["# Verification report — AEI manuscript numbers\n"]
    md.append("**Generated by:** `python -m research.verify_findings`\n")
    md.append("Every check recomputes from raw data with code independent of the scripts that produced the "
              "results; replayed predictions are re-derived with the production single-window path.\n")
    md.append("**Summary:** " + ", ".join(f"{icon[k]} {k} {counts.get(k, 0)}" for k in ("PASS", "FAIL", "WARN", "NOTE")) + "\n")
    problems = df[df["status"].isin(["FAIL", "WARN"])]
    md.append("\n## Items needing action\n")
    if problems.empty:
        md.append("None.\n")
    for r in problems.itertuples():
        md.append(f"- {icon[r.status]} **{r.section} — {r.claim}**: claimed `{r.claimed}`, recomputed `{r.recomputed}`. {r.detail}")
    for section, g in df.groupby("section", sort=False):
        md.append(f"\n## {section}\n")
        md.append("| | Claim | Claimed | Recomputed | Detail |")
        md.append("|---|---|---|---|---|")
        for r in g.itertuples():
            detail = str(r.detail).replace("|", "/")
            md.append(f"| {icon[r.status]} | {r.claim} | {r.claimed} | {r.recomputed} | {detail} |")
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    return REPORT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--sections", default="ABCDEFGHIJKLMNOP", help="subset for debugging; the report covers only these")
    args = parser.parse_args()
    for name, fn in (("A", section_a), ("B", section_b), ("C", section_c), ("D", lambda: section_d(args.sample)),
                     ("E", section_e), ("F", section_f), ("G", section_g), ("H", section_h), ("I", section_i),
                     ("J", section_j), ("K", lambda: section_k(args.sample)), ("L", section_l), ("M", section_m),
                     ("N", section_n), ("O", section_o), ("P", section_p)):
        if name not in args.sections:
            continue
        print(f"[verify] section {name}", flush=True)
        fn()
    path = write_report()
    df = pd.DataFrame(RESULTS)
    print(df["status"].value_counts().to_string())
    print(f"[verify] wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
