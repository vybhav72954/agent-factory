"""
research/paraphrase/prompt_bank.py

Held-out prompt bank for the out-of-distribution severity experiment.

Every prompt is written to avoid the severity vocabulary that the regex
baseline (`research/baselines.py::_HIGH_PATTERN/_LOW_PATTERN`) and the
production LLM prompt (`agents/prompts.py` SEVERITY CLASSIFICATION) were built
around, EXCEPT the "negation" style, which deliberately contains a listed word
used in a way that should not trigger it (e.g. "no rupture ... just a slight
dip").

`design_severity` is the severity each prompt was written to express and is the
ground truth for scoring, the same convention as the author-labelled
TEST_PROMPTS in research/evaluation_rubric.py. Each prompt states its severity
plainly through consequence, magnitude or explicit reassurance; only the
vocabulary is unfamiliar, which is exactly what the experiment tests.

Styles:
    understated  hedged or mild phrasing
    jargon       shop-floor slang
    consequence  describes what happened rather than naming a severity
    quantitative numbers instead of adjectives
    codemixed    Hindi-English code-mixed operator speech
    terse        SMS-style shorthand
    negation     a listed severity word appears but is negated or incidental

Usage:
    python -m research.paraphrase.prompt_bank     # run the vocabulary and input-guard checks
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


PROMPTS: list[dict] = [
    # ── Design severity LOW ───────────────────────────────────────────────
    {"id": "L01", "machine_id": 1, "style": "understated",  "design_severity": "LOW",
     "text": "Machine 1 bearing is running a couple of degrees warmer than it did last week"},
    {"id": "L02", "machine_id": 2, "style": "understated",  "design_severity": "LOW",
     "text": "motor on Machine 2 feels a touch warmer than normal, nothing urgent"},
    {"id": "L03", "machine_id": 3, "style": "jargon",       "design_severity": "LOW",
     "text": "Machine 3 oil pressure gauge is sitting a hair under its usual mark"},
    {"id": "L04", "machine_id": 4, "style": "consequence",  "design_severity": "LOW",
     "text": "operators on Machine 4 can just about feel a faint buzz through the frame, parts still in tolerance"},
    {"id": "L05", "machine_id": 5, "style": "quantitative", "design_severity": "LOW",
     "text": "coolant flow on Machine 5 is down about 3 percent from baseline"},
    {"id": "L06", "machine_id": 1, "style": "codemixed",    "design_severity": "LOW",
     "text": "Machine 1 ka bearing thoda sa garam lag raha hai, abhi chinta ki baat nahi"},
    {"id": "L07", "machine_id": 2, "style": "terse",        "design_severity": "LOW",
     "text": "m2 motor temp up 2C vs yesterday, keep an eye on it"},
    {"id": "L08", "machine_id": 3, "style": "negation",     "design_severity": "LOW",
     "text": "no rupture or leak on Machine 3, just a slight dip in oil pressure"},
    {"id": "L09", "machine_id": 4, "style": "understated",  "design_severity": "LOW",
     "text": "Machine 4 spindle sounds a little louder than usual toward the end of each shift"},
    {"id": "L10", "machine_id": 5, "style": "quantitative", "design_severity": "LOW",
     "text": "Machine 5 current draw is 4 percent above its typical value"},
    {"id": "L11", "machine_id": 1, "style": "jargon",       "design_severity": "LOW",
     "text": "Machine 1 bearing grease looked a bit dark at the last check, temperature still normal"},
    {"id": "L12", "machine_id": 2, "style": "negation",     "design_severity": "LOW",
     "text": "Machine 2 is nowhere near a shutdown, the motor is only a degree or two warm"},
    {"id": "L13", "machine_id": 3, "style": "consequence",  "design_severity": "LOW",
     "text": "Machine 3 needed one extra second per cycle today, everything else looks fine"},
    {"id": "L14", "machine_id": 4, "style": "codemixed",    "design_severity": "LOW",
     "text": "Machine 4 mein halka sa vibration hai, production normal chal raha hai"},

    # ── Design severity MEDIUM ────────────────────────────────────────────
    {"id": "M01", "machine_id": 1, "style": "jargon",       "design_severity": "MEDIUM",
     "text": "Machine 1 bearing is running hot, needs looking at this shift"},
    {"id": "M02", "machine_id": 2, "style": "consequence",  "design_severity": "MEDIUM",
     "text": "Machine 2 motor temperature keeps climbing and the operator had to slow the line"},
    {"id": "M03", "machine_id": 3, "style": "quantitative", "design_severity": "MEDIUM",
     "text": "oil pressure on Machine 3 dropped 25 percent over the last hour"},
    {"id": "M04", "machine_id": 4, "style": "jargon",       "design_severity": "MEDIUM",
     "text": "Machine 4 is shaking noticeably and the parts are coming out rough"},
    {"id": "M05", "machine_id": 5, "style": "consequence",  "design_severity": "MEDIUM",
     "text": "coolant is pooling under Machine 5 and flow to the cutter is reduced"},
    {"id": "M06", "machine_id": 1, "style": "understated",  "design_severity": "MEDIUM",
     "text": "Machine 1 bearing temperature is well above where it normally sits"},
    {"id": "M07", "machine_id": 2, "style": "codemixed",    "design_severity": "MEDIUM",
     "text": "Machine 2 ka motor kaafi garam ho raha hai, check karna padega"},
    {"id": "M08", "machine_id": 3, "style": "terse",        "design_severity": "MEDIUM",
     "text": "M3 oil press low again, topping up didnt fix it"},
    {"id": "M09", "machine_id": 4, "style": "quantitative", "design_severity": "MEDIUM",
     "text": "vibration on Machine 4 has doubled since the morning inspection"},
    {"id": "M10", "machine_id": 5, "style": "negation",     "design_severity": "MEDIUM",
     "text": "nothing catastrophic yet, but Machine 5 motor current is running well above normal all shift"},
    {"id": "M11", "machine_id": 1, "style": "jargon",       "design_severity": "MEDIUM",
     "text": "Machine 1 spindle keeps loading up and tripping the overload about once an hour"},
    {"id": "M12", "machine_id": 2, "style": "consequence",  "design_severity": "MEDIUM",
     "text": "Machine 2 is throwing out more rejects since the torque readings started climbing"},
    {"id": "M13", "machine_id": 3, "style": "codemixed",    "design_severity": "MEDIUM",
     "text": "Machine 3 ka oil pressure baar baar gir raha hai"},
    {"id": "M14", "machine_id": 4, "style": "negation",     "design_severity": "MEDIUM",
     "text": "Machine 4 bearing is hot but it is not an emergency, we can finish the batch first"},

    # ── Design severity HIGH ──────────────────────────────────────────────
    {"id": "H01", "machine_id": 1, "style": "jargon",       "design_severity": "HIGH",
     "text": "Machine 1 bearing is cooked, there is metal dust all over the housing"},
    {"id": "H02", "machine_id": 2, "style": "consequence",  "design_severity": "HIGH",
     "text": "Machine 2 motor tripped and will not restart, the windings smell scorched"},
    {"id": "H03", "machine_id": 3, "style": "consequence",  "design_severity": "HIGH",
     "text": "oil is spraying out of a split hose on Machine 3 and pressure has gone to zero"},
    {"id": "H04", "machine_id": 4, "style": "jargon",       "design_severity": "HIGH",
     "text": "Machine 4 threw a bearing and the spindle is locked solid"},
    {"id": "H05", "machine_id": 5, "style": "quantitative", "design_severity": "HIGH",
     "text": "Machine 5 coolant flow is reading zero and the cutting head is glowing red"},
    {"id": "H06", "machine_id": 1, "style": "codemixed",    "design_severity": "HIGH",
     "text": "Machine 1 ka motor poori tarah jal gaya hai, bilkul nahi chal raha"},
    {"id": "H07", "machine_id": 2, "style": "terse",        "design_severity": "HIGH",
     "text": "M2 bearing gone. grinding noise then a bang. line down"},
    {"id": "H08", "machine_id": 3, "style": "consequence",  "design_severity": "HIGH",
     "text": "we had to pull the plug on Machine 3, the pump housing cracked and oil is everywhere"},
    {"id": "H09", "machine_id": 4, "style": "jargon",       "design_severity": "HIGH",
     "text": "Machine 4 gearbox has let go, the teeth are sheared off"},
    {"id": "H10", "machine_id": 5, "style": "quantitative", "design_severity": "HIGH",
     "text": "Machine 5 motor temperature went past 180 C in under a minute and the breaker kicked out"},
    {"id": "H11", "machine_id": 1, "style": "understated",  "design_severity": "HIGH",
     "text": "Machine 1 bearing is in a pretty bad way, I would not run it for another minute"},
    {"id": "H12", "machine_id": 2, "style": "negation",     "design_severity": "HIGH",
     "text": "the small vibration on Machine 2 turned into the rotor tearing loose from its mount"},
    {"id": "H13", "machine_id": 3, "style": "codemixed",    "design_severity": "HIGH",
     "text": "Machine 3 ka shaft toot gaya, machine band karni padi"},
    {"id": "H14", "machine_id": 4, "style": "negation",     "design_severity": "HIGH",
     "text": "early this morning the Machine 4 hydraulic line split and the press dropped its ram"},
]


# Word stems covering the regex word lists and the prompts.py SEVERITY
# CLASSIFICATION lists (HIGH, LOW and the MEDIUM examples), including
# inflections, so "smoking" or "drifting" also count. Multi-word list entries
# ("complete failure", "shaft lock", "first sign of", ...) are checked as phrases.
FORBIDDEN_STEMS = [
    # HIGH list
    "catastroph", "destroy", "ruptur", "burst", "explo", "fire", "burn", "smok",
    "shutdown", "shut down", "seiz", "halt", "critical", "emergenc", "meltdown",
    "outage", "broke",
    # LOW list
    "minor", "slight", "small", "subtl", "early", "wobbl", "drift", "creep",
    "intermittent", "occasional",
    # MEDIUM examples
    "spike", "surg", "anomal", "abnormal", "elevat", "high", "fluctuat",
    "instabil", "exceed", "warn", "alert",
]
FORBIDDEN_PHRASES = [
    r"complete\s+failure", r"total\s+failure", r"shaft\s+lock", r"stopped\s+completely",
    r"first\s+sign\s+of", r"trending\s+up", r"rising\s+slowly",
]


def forbidden_hits(text: str) -> list[str]:
    """Return every forbidden stem or phrase found in `text` (case-insensitive)."""
    lower = text.lower()
    tokens = re.findall(r"[a-z]+", lower)
    hits = []
    for stem in FORBIDDEN_STEMS:
        if " " in stem:
            if stem in lower:
                hits.append(stem)
        elif any(tok.startswith(stem) for tok in tokens):
            hits.append(stem)
    hits += [p for p in FORBIDDEN_PHRASES if re.search(p, lower)]
    return hits


def check_prompt_bank(prompts: list[dict] | None = None) -> list[str]:
    """Validate a prompt list (default: this bank). Returns a list of problems (empty when clean)."""
    from agents.input_guard import is_valid_fault_input
    from research.baselines import _HIGH_PATTERN, _LOW_PATTERN

    prompts = PROMPTS if prompts is None else prompts
    problems = []
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        problems.append("duplicate prompt ids")

    for p in prompts:
        ok, reason = is_valid_fault_input(p["text"])
        if not ok:
            problems.append(f"{p['id']}: rejected by input guard ({reason})")

        hits = forbidden_hits(p["text"])
        regex_triggered = bool(_HIGH_PATTERN.search(p["text"]) or _LOW_PATTERN.search(p["text"]))
        if p["style"] == "negation":
            if not regex_triggered:
                problems.append(f"{p['id']}: negation prompt does not trigger the regex, so it is not a trap")
        elif hits:
            problems.append(f"{p['id']}: contains listed severity vocabulary {hits}")

        if p["design_severity"] not in {"LOW", "MEDIUM", "HIGH"}:
            problems.append(f"{p['id']}: bad design_severity {p['design_severity']!r}")
    return problems


def main() -> None:
    from collections import Counter
    from research.baselines import _classify_severity_regex

    problems = check_prompt_bank()
    print(f"[prompt_bank] {len(PROMPTS)} prompts")
    print("  by design severity:", dict(Counter(p["design_severity"] for p in PROMPTS)))
    print("  by style:            ", dict(Counter(p["style"] for p in PROMPTS)))

    agree = sum(_classify_severity_regex(p["text"]).value == p["design_severity"] for p in PROMPTS)
    print(f"  regex agrees with the design severity on {agree}/{len(PROMPTS)}")

    if problems:
        print("[prompt_bank] PROBLEMS:")
        for msg in problems:
            print("  -", msg)
        sys.exit(1)
    print("[prompt_bank] all checks passed")


if __name__ == "__main__":
    main()
