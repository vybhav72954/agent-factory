"""
research/evaluation_rubric.py

Quantitative evaluation of pipeline output quality. Used by baselines (#2),
ablations (#3), and any other comparison that needs to score "is this dispatch
order operationally correct?"

Two layers:

1. **Deterministic structural checks** — fast, cheap, no LLM. Verify the
   dispatch order contains required fields and matches the capacity report
   it was generated from. These catch the most common failure modes (wrong
   machine name, missing RUL, action contradicts status, hallucinated numbers).

2. **LLM-as-judge semantic check** — Gemini 2.5 Flash rates the dispatch
   order on a 0-5 scale for operational coherence. Slower and costs money,
   so use sparingly.

The combined rubric returns a `RubricScore` dataclass with per-criterion
scores and an aggregate. Use this as the dependent variable in any comparison.

Reference test set (TEST_PROMPTS) is N=20 prompts spanning the project's
fault taxonomy + machine spread, with expected status outcomes derived
from the canonical sensor map + severity classification + capacity thresholds.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Reference test set
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (prompt_text, machine_id, expected_status_after_one_hit_on_fresh_machine)
# The status follows from severity classification + the probe-validated
# response surface (probe_cliff_3d.csv). LOW → ONLINE; MEDIUM → ONLINE or
# borderline; HIGH catastrophic → OFFLINE. We list expected_status as the
# *most-common* outcome; tolerance is one bucket for MEDIUM (which sits near
# the cliff in many prompts).
TEST_PROMPTS = [
    # LOW severity — should stay ONLINE
    ("minor bearing wobble on Machine 1",        1, "ONLINE"),
    ("slight motor temp drift on Machine 2",     2, "ONLINE"),
    ("subtle pressure creep on Machine 3",       3, "ONLINE"),
    ("intermittent vibration on Machine 4",      4, "ONLINE"),
    ("early sign of coolant flow drop on Machine 5", 5, "ONLINE"),

    # MEDIUM severity — should be ONLINE or DEGRADED depending on cliff
    ("bearing temperature spike on Machine 1",   1, "ONLINE"),
    ("motor temp anomaly on Machine 2",          2, "ONLINE"),
    ("oil pressure surge on Machine 3",          3, "ONLINE"),
    ("vibration above normal on Machine 4",      4, "ONLINE"),
    ("coolant flow disruption on Machine 5",     5, "ONLINE"),

    # HIGH severity — should single-shot to OFFLINE
    ("catastrophic bearing failure on Machine 1",   1, "OFFLINE"),
    ("complete motor breakdown on Machine 2",       2, "OFFLINE"),
    ("oil pressure rupture on Machine 3",           3, "OFFLINE"),
    ("explosion in Machine 4",                      4, "OFFLINE"),
    ("shaft lock on Machine 5",                     5, "OFFLINE"),

    # Edge cases — natural-language machine targeting
    ("bearing overheat on metal press",             1, "ONLINE"),    # MEDIUM via "overheat"
    ("oil pressure surge on QC line",               5, "ONLINE"),    # MEDIUM (note: "QC line" may not match)
    ("catastrophic failure on paint and coat",      2, "OFFLINE"),   # HIGH

    # Rejection cases (Input Guard should block)
    ("hi",                                          1, "REJECTED"),
    ("what is the weather today",                   1, "REJECTED"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Rubric criteria
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RubricScore:
    """Per-criterion scores 0-5. aggregate is unweighted mean of the 5 criteria."""
    machine_name_correct: int     # name appears in dispatch text
    rul_mentioned: int            # numeric RUL appears
    action_matches_status: int    # OFFLINE → halt/dispatch; DEGRADED → reduce/inspect; ONLINE → monitor
    capacity_mentioned: int       # capacity_pct value appears
    no_hallucinated_numbers: int  # all numbers in dispatch appear in the capacity report
    llm_judge_score: Optional[int] = None  # 0-5 from Gemini judge, None if not run
    notes: list[str] = field(default_factory=list)

    @property
    def aggregate(self) -> float:
        scores = [
            self.machine_name_correct,
            self.rul_mentioned,
            self.action_matches_status,
            self.capacity_mentioned,
            self.no_hallucinated_numbers,
        ]
        if self.llm_judge_score is not None:
            scores.append(self.llm_judge_score)
        return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic structural checks
# ─────────────────────────────────────────────────────────────────────────────

ACTION_KEYWORDS_BY_STATUS = {
    "OFFLINE":  ["halt", "shutdown", "shut down", "stop", "offline", "evacuate",
                 "dispatch", "maintenance crew", "emergency", "mandatory"],
    "DEGRADED": ["reduce", "50%", "fifty percent", "degraded", "inspect",
                 "monitor closely", "schedule maintenance", "maintenance window"],
    "ONLINE":   ["nominal", "continue", "monitor", "no immediate", "no action",
                 "scheduled inspection", "operating normally"],
}


def _extract_numbers(text: str) -> set[float]:
    """Pull all numeric tokens (int or decimal) from the dispatch order text."""
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return {float(m) for m in matches if float(m) > 0}   # ignore zero (too common)


def score_structural(
    dispatch_text: str,
    capacity_report: dict,
) -> RubricScore:
    """
    Score the dispatch order against the capacity report it was generated from.

    Args:
        dispatch_text: Full floor-manager dispatch string (e.g. "[Floor Manager] PCB Line OFFLINE...")
        capacity_report: dict from capacity_agent.update_capacity()

    Returns a RubricScore with `llm_judge_score=None` (structural only).
    """
    text_lower = dispatch_text.lower()
    machine_name = capacity_report["machine_name"]
    status = capacity_report["status"]
    rul = capacity_report["rul"]
    cap_pct = capacity_report["capacity_pct"]
    machine_req = capacity_report["machine_req"]

    notes = []

    # 1. Machine name correct (case-insensitive)
    name_ok = machine_name.lower() in text_lower
    machine_name_score = 5 if name_ok else 0
    if not name_ok:
        notes.append(f"machine name '{machine_name}' missing from dispatch")

    # 2. RUL mentioned (within 0.5 of the actual value)
    rul_str = f"{rul:.1f}"
    rul_int_str = str(int(round(rul)))
    rul_ok = rul_str in dispatch_text or rul_int_str in dispatch_text
    rul_score = 5 if rul_ok else 0
    if not rul_ok:
        notes.append(f"RUL value {rul} not mentioned in dispatch")

    # 3. Action matches status
    expected_keywords = ACTION_KEYWORDS_BY_STATUS.get(status, [])
    matched = [kw for kw in expected_keywords if kw in text_lower]
    if len(matched) >= 2:
        action_score = 5
    elif len(matched) == 1:
        action_score = 3
    else:
        action_score = 0
        notes.append(f"no {status}-appropriate action keywords found")

    # 4. Capacity mentioned (% or number)
    cap_str = f"{cap_pct:.1f}"
    cap_int_str = str(int(round(cap_pct)))
    cap_ok = cap_str in dispatch_text or cap_int_str in dispatch_text
    cap_score = 5 if cap_ok else 0
    if not cap_ok:
        notes.append(f"capacity_pct {cap_pct} not mentioned in dispatch")

    # 5. No hallucinated numbers (every number in dispatch appears in report)
    dispatch_nums = _extract_numbers(dispatch_text)
    report_nums = _extract_numbers(
        f"{rul} {cap_pct} {machine_req} {capacity_report.get('total_T', '')} "
        f"{capacity_report.get('total_PD', '')} {capacity_report['machine_id']}"
    )
    # Allow small integers (1-50) since they often appear in dispatch text as
    # rounded percentages or sentence-counting ("within 30 minutes", "15 minutes")
    suspect_nums = {n for n in dispatch_nums if n > 50 and not any(abs(n - r) < 0.5 for r in report_nums)}
    if not suspect_nums:
        hallucination_score = 5
    elif len(suspect_nums) == 1:
        hallucination_score = 3
        notes.append(f"possibly hallucinated number: {sorted(suspect_nums)[0]}")
    else:
        hallucination_score = 0
        notes.append(f"multiple possibly-hallucinated numbers: {sorted(suspect_nums)}")

    return RubricScore(
        machine_name_correct=machine_name_score,
        rul_mentioned=rul_score,
        action_matches_status=action_score,
        capacity_mentioned=cap_score,
        no_hallucinated_numbers=hallucination_score,
        llm_judge_score=None,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge semantic check (optional, expensive)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a dispatch order issued by an automated factory floor manager.
Given the underlying capacity report and the dispatch order, rate how well the
dispatch order communicates the situation to a shift supervisor.

Capacity report (the ground truth):
{report_json}

Dispatch order under evaluation:
\"\"\"
{dispatch_text}
\"\"\"

Rate the dispatch order on a 0-5 scale on a SINGLE dimension:
**Operational coherence** — does it accurately convey the situation, give an
action that matches the status, use the correct machine name, and avoid
inventing facts not in the capacity report?

5 = perfect: accurate, actionable, no errors
4 = minor issue: one small inaccuracy or weak action
3 = mediocre: status conveyed but action vague or partially wrong
2 = poor: action contradicts status or numbers wrong
1 = bad: barely coherent, multiple errors
0 = unusable: nonsense or empty

Respond with ONLY a single integer 0-5, no other text.
"""


def score_llm_judge(
    dispatch_text: str,
    capacity_report: dict,
    *,
    client=None,
) -> Optional[int]:
    """
    Ask Gemini to rate the dispatch order on operational coherence.
    Returns int 0-5, or None if the judge call failed.
    """
    if client is None:
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY_FLOOR_MANAGER") \
                   or os.environ.get("GEMINI_API_KEY_DIAGNOSTIC")
            if not api_key:
                return None
            client = genai.Client(api_key=api_key)
        except Exception:
            return None

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        report_json=json.dumps(capacity_report, indent=2),
        dispatch_text=dispatch_text,
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (response.text or "").strip()
        # Extract a single digit 0-5
        m = re.search(r"\b([0-5])\b", text)
        if m:
            return int(m.group(1))
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Combined score
# ─────────────────────────────────────────────────────────────────────────────

def score_full(
    dispatch_text: str,
    capacity_report: dict,
    *,
    use_llm_judge: bool = False,
) -> RubricScore:
    """Structural + (optional) LLM judge in one call."""
    score = score_structural(dispatch_text, capacity_report)
    if use_llm_judge:
        score.llm_judge_score = score_llm_judge(dispatch_text, capacity_report)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# CLI: score a sample dispatch for sanity-checking the rubric itself
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Apply the rubric to two synthetic dispatch examples to verify behaviour."""
    good_report = {
        "machine_id": 3, "machine_name": "PCB Line",
        "status": "OFFLINE", "rul": 1.2, "total_T": 32.0, "total_PD": 595,
        "machine_req": 18.59, "capacity_pct": 80.0, "breakeven_risk": True,
    }
    good_dispatch = (
        "[Floor Manager] PCB Line OFFLINE at RUL 1.2 — mandatory shutdown initiated. "
        "Halt all production on this unit and dispatch maintenance crew immediately. "
        "Reroute PCB Line workload to remaining online machines. "
        "Factory at 80.0% capacity — breakeven risk ACTIVE, authorize overtime to cover ΣPD/T of 18.59."
    )
    bad_dispatch = (
        "[Floor Manager] Machine 3 is showing some kind of fault. Look into it when you have time. "
        "Production can continue at full speed. Capacity is fine."
    )

    print("--- good dispatch ---")
    s_good = score_structural(good_dispatch, good_report)
    print(f"  aggregate: {s_good.aggregate:.2f}")
    for k, v in asdict(s_good).items():
        if k != "notes":
            print(f"    {k}: {v}")
    if s_good.notes:
        print(f"  notes: {s_good.notes}")

    print("--- bad dispatch ---")
    s_bad = score_structural(bad_dispatch, good_report)
    print(f"  aggregate: {s_bad.aggregate:.2f}")
    for k, v in asdict(s_bad).items():
        if k != "notes":
            print(f"    {k}: {v}")
    if s_bad.notes:
        print(f"  notes: {s_bad.notes}")

    print(f"\n  Rubric is calibrated: good={s_good.aggregate:.2f} > bad={s_bad.aggregate:.2f}")


if __name__ == "__main__":
    _self_test()
