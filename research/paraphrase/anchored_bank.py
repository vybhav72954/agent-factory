"""
research/paraphrase/anchored_bank.py

Anchored paraphrase set: every non-rejected prompt in TEST_PROMPTS
(research/evaluation_rubric.py) plus four rewrites of it.

Each rewrite keeps the anchor's machine, fault type and severity, so its label
is inherited from the published test set rather than assigned afresh:
    LOW / MEDIUM / HIGH  = the TEST_PROMPTS group the anchor belongs to
    expected status      = the anchor's TEST_PROMPTS expected status
Rewrites avoid the regex and production-prompt severity vocabulary
(prompt_bank.forbidden_hits); the anchors themselves are included unchanged as
style "original", so every strategy gets a paired original-vs-reworded
comparison on the same fault.

Rewrite styles:
    shopfloor    shop-floor slang
    consequence  what was observed and what it did, no severity adjectives
    hinglish     Hindi-English code-mixed operator speech
    sms          terse shorthand

Usage:
    python -m research.paraphrase.anchored_bank     # coverage, inheritance, vocabulary and input-guard checks
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.evaluation_rubric import TEST_PROMPTS

REWRITE_STYLES = ("shopfloor", "consequence", "hinglish", "sms")

# anchor text -> (severity group in TEST_PROMPTS, {style: rewrite})
_ANCHORS: dict[str, tuple[str, dict[str, str]]] = {
    # ── LOW group ──────────────────────────────────────────────────────────
    "minor bearing wobble on Machine 1": ("LOW", {
        "shopfloor":   "Machine 1 bearing has a tiny bit of play in it, nothing to stop for",
        "consequence": "you can just feel a faint shake at the Machine 1 bearing housing, output and finish are fine",
        "hinglish":    "Machine 1 ke bearing mein thoda sa khel hai, abhi chalne do",
        "sms":         "M1 bearing tiny play, ok for now, check next PM",
    }),
    "slight motor temp drift on Machine 2": ("LOW", {
        "shopfloor":   "Machine 2 motor is running a degree or so warmer than its usual, nothing to worry about",
        "consequence": "Machine 2 motor temperature has moved up by about one degree over the week, everything else normal",
        "hinglish":    "Machine 2 ka motor pichhle hafte se thoda sa zyada garam chal raha hai, bas nazar rakho",
        "sms":         "M2 motor temp +1C this week, just watch it",
    }),
    "subtle pressure creep on Machine 3": ("LOW", {
        "shopfloor":   "Machine 3 oil pressure needle has moved a touch off its usual spot, still in the green",
        "consequence": "Machine 3 pressure reading is a fraction off baseline and nothing else has changed",
        "hinglish":    "Machine 3 ka pressure halka sa idhar udhar hai, abhi koi dikkat nahi",
        "sms":         "M3 pressure a hair off baseline, in range, monitor",
    }),
    "intermittent vibration on Machine 4": ("LOW", {
        "shopfloor":   "Machine 4 gets a faint shudder now and then, parts still coming out clean",
        "consequence": "every so often you can feel a light buzz on Machine 4, it goes away by itself and quality is fine",
        "hinglish":    "Machine 4 mein kabhi kabhi halki vibration aati hai, production theek hai",
        "sms":         "M4 light vibration on and off, parts ok",
    }),
    "early sign of coolant flow drop on Machine 5": ("LOW", {
        "shopfloor":   "Machine 5 coolant flow looks a bit lazier than usual, nozzle still wetting the cut fine",
        "consequence": "coolant flow gauge on Machine 5 is reading just under its normal band and the cut is still cool",
        "hinglish":    "Machine 5 ka coolant flow thoda kam lag raha hai, abhi kaam chal raha hai",
        "sms":         "M5 coolant flow just under normal, cut ok, keep an eye",
    }),
    # ── MEDIUM group ───────────────────────────────────────────────────────
    "bearing temperature spike on Machine 1": ("MEDIUM", {
        "shopfloor":   "Machine 1 bearing got hot quickly, get someone to look at it this shift",
        "consequence": "Machine 1 bearing temperature jumped well past its usual level in the last hour, still running",
        "hinglish":    "Machine 1 ka bearing achanak bahut garam ho gaya, aaj check karwao",
        "sms":         "M1 bearing temp jumped, still running, need check today",
    }),
    "motor temp anomaly on Machine 2": ("MEDIUM", {
        "shopfloor":   "Machine 2 motor temperature is doing something odd, not where it should be, needs a look today",
        "consequence": "Machine 2 motor temperature keeps wandering well away from its usual reading and we had to ease off the feed",
        "hinglish":    "Machine 2 ke motor ka temperature ajeeb behave kar raha hai, aaj dekhna padega",
        "sms":         "M2 motor temp acting odd, off normal, check today",
    }),
    "oil pressure surge on Machine 3": ("MEDIUM", {
        "shopfloor":   "Machine 3 oil pressure kicked up hard a couple of times, get maintenance to look at it",
        "consequence": "Machine 3 oil pressure jumped well above its normal band twice this morning, the line is still running",
        "hinglish":    "Machine 3 ka oil pressure achanak bahut badh gaya, maintenance ko bulao",
        "sms":         "M3 oil pressure jumped twice, still running, maint pls check",
    }),
    "vibration above normal on Machine 4": ("MEDIUM", {
        "shopfloor":   "Machine 4 is shaking more than it should, finish on the parts is getting worse",
        "consequence": "vibration on Machine 4 is clearly past its normal level and the surface finish has started to suffer",
        "hinglish":    "Machine 4 zyada vibrate kar rahi hai, parts ki finish kharab ho rahi hai",
        "sms":         "M4 vibration well over normal, finish getting worse",
    }),
    "coolant flow disruption on Machine 5": ("MEDIUM", {
        "shopfloor":   "Machine 5 coolant keeps cutting in and out, the tool is running warm, needs fixing today",
        "consequence": "coolant flow on Machine 5 stops and starts and the cutter is coming out warmer than normal",
        "hinglish":    "Machine 5 ka coolant flow ruk ruk ke aa raha hai, aaj theek karna padega",
        "sms":         "M5 coolant flow stop-start, tool warm, fix today",
    }),
    # ── HIGH group ─────────────────────────────────────────────────────────
    "catastrophic bearing failure on Machine 1": ("HIGH", {
        "shopfloor":   "Machine 1 bearing is totally gone, it is grinding metal on metal, pull it off the line now",
        "consequence": "Machine 1 bearing came apart, there are rollers on the floor and the shaft is flopping around in the housing",
        "hinglish":    "Machine 1 ka bearing poora khatam ho gaya, machine turant band karo",
        "sms":         "M1 bearing gone. metal on metal. stop it now",
    }),
    "complete motor breakdown on Machine 2": ("HIGH", {
        "shopfloor":   "Machine 2 motor is dead, it will not turn over at all",
        "consequence": "Machine 2 motor quit with a loud bang and now it will not start at all",
        "hinglish":    "Machine 2 ki motor bilkul kharab ho gayi, chal hi nahi rahi",
        "sms":         "M2 motor dead, wont start",
    }),
    "oil pressure rupture on Machine 3": ("HIGH", {
        "shopfloor":   "Machine 3 oil line has let go, oil pressure is gone and oil is all over the floor",
        "consequence": "a hose on Machine 3 split open, oil pressure dropped to nothing and the pump is running dry",
        "hinglish":    "Machine 3 ki oil pipe phat gayi, pressure zero ho gaya, turant roko",
        "sms":         "M3 oil line split, pressure zero, oil everywhere",
    }),
    "explosion in Machine 4": ("HIGH", {
        "shopfloor":   "something in Machine 4 went bang and blew the guard off, everyone is clear of it",
        "consequence": "Machine 4 let out a huge bang, the housing is torn open and there are parts on the floor",
        "hinglish":    "Machine 4 mein zor ka dhamaka hua, cover udd gaya, sab log door raho",
        "sms":         "M4 machine big bang, housing torn open, clear the area",
    }),
    "shaft lock on Machine 5": ("HIGH", {
        "shopfloor":   "Machine 5 shaft is jammed solid and the motor is straining against it",
        "consequence": "Machine 5 shaft will not turn at all, the drive stalled and tripped out",
        "hinglish":    "Machine 5 ka shaft bilkul jam ho gaya hai, ghoom hi nahi raha",
        "sms":         "M5 shaft stuck solid, drive tripped",
    }),
    # ── Edge cases (TEST_PROMPTS comments: MEDIUM, MEDIUM, HIGH) ────────────
    "bearing overheat on metal press": ("MEDIUM", {
        "shopfloor":   "the metal press bearing is running too hot, get it looked at before end of shift",
        "consequence": "the bearing on the metal press is well over its normal temperature but the press is still stamping",
        "hinglish":    "metal press ka bearing bahut garam chal raha hai, shift khatam hone se pehle check karo",
        "sms":         "metal press bearing too hot, still running, check this shift",
    }),
    "oil pressure surge on QC line": ("MEDIUM", {
        "shopfloor":   "QC line oil pressure kicked up hard, needs a look today",
        "consequence": "oil pressure on the QC line jumped well past normal a few times this morning, the line is still moving",
        "hinglish":    "QC line ka oil pressure achanak badh gaya, aaj dekh lo",
        "sms":         "QC line oil pressure jumped, still moving, check today",
    }),
    "catastrophic failure on paint and coat": ("HIGH", {
        "shopfloor":   "paint and coat unit is completely wrecked, nothing on the machine works, take it off the line",
        "consequence": "the paint and coat machine went dead with a loud crack and there is paint and oil everywhere",
        "hinglish":    "paint and coat machine poori tarah kharab ho gayi, turant band karo",
        "sms":         "paint n coat machine totally dead, stop line",
    }),
}

_EXPECTED = {p: (mid, s) for p, mid, s in TEST_PROMPTS if s != "REJECTED"}


def _build() -> list[dict]:
    prompts = []
    for i, (anchor, (severity, rewrites)) in enumerate(_ANCHORS.items(), start=1):
        machine_id, expected_status = _EXPECTED[anchor]
        base = {"anchor": anchor, "machine_id": machine_id, "design_severity": severity,
                "expected_status": expected_status}
        prompts.append({"id": f"A{i:02d}-original", "style": "original", "text": anchor, **base})
        for style in REWRITE_STYLES:
            prompts.append({"id": f"A{i:02d}-{style}", "style": style, "text": rewrites[style], **base})
    return prompts


ANCHORED_PROMPTS: list[dict] = _build()


def check_anchored_bank() -> list[str]:
    """Coverage, label inheritance, vocabulary and input-guard checks. Returns a list of problems."""
    from research.paraphrase.prompt_bank import check_prompt_bank

    problems = []
    missing = set(_EXPECTED) - set(_ANCHORS)
    if missing:
        problems.append(f"TEST_PROMPTS anchors without rewrites: {sorted(missing)}")
    for anchor, (severity, rewrites) in _ANCHORS.items():
        if set(rewrites) != set(REWRITE_STYLES):
            problems.append(f"{anchor!r}: styles {sorted(rewrites)} != {sorted(REWRITE_STYLES)}")
        expected = _EXPECTED[anchor][1]
        if (severity == "HIGH") != (expected == "OFFLINE"):
            problems.append(f"{anchor!r}: severity {severity} inconsistent with expected status {expected}")
    rewrites_only = [p for p in ANCHORED_PROMPTS if p["style"] != "original"]
    problems += check_prompt_bank(rewrites_only)
    return problems


def main() -> None:
    from collections import Counter
    from research.baselines import _classify_severity_regex

    problems = check_anchored_bank()
    print(f"[anchored_bank] {len(ANCHORED_PROMPTS)} prompts ({len(_ANCHORS)} anchors x (original + {len(REWRITE_STYLES)} rewrites))")
    print("  by severity:", dict(Counter(p["design_severity"] for p in ANCHORED_PROMPTS)))
    for style in ("original",) + REWRITE_STYLES:
        group = [p for p in ANCHORED_PROMPTS if p["style"] == style]
        agree = sum(_classify_severity_regex(p["text"]).value == p["design_severity"] for p in group)
        print(f"  regex agrees with severity on {style:11s}: {agree}/{len(group)}")
    if problems:
        print("[anchored_bank] PROBLEMS:")
        for msg in problems:
            print("  -", msg)
        sys.exit(1)
    print("[anchored_bank] all checks passed")


if __name__ == "__main__":
    main()
