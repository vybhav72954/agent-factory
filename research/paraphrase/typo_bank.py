"""
research/paraphrase/typo_bank.py

Programmatic typo set: the 18 TEST_PROMPTS anchors with seeded keyboard typos.
Design fixed in research/AEI/analysis_plan_additions.md (section D). Unlike the
design and anchored banks, no text here is written or edited by hand, so this
set tests unfamiliar surface forms without author choice of wording.

Noise model, per prompt:
    eligible words  alphabetic tokens of 3 or more letters, except "Machine"
    k               max(1, round_half_up(p * n_eligible)) words, chosen uniformly without replacement
    edit per word   one of, uniformly: adjacent-key substitution, deletion,
                    adjacent-key insertion, transposition of neighbouring letters
Levels p = 0.25 and 0.50, three variants each. A variant equal to the original or
to an earlier variant of the same anchor and level is redrawn with the next seed.
Labels and expected status are inherited from the anchor.

Usage:
    python -m research.paraphrase.typo_bank     # print the set and run the checks
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TYPO_SEED = 20260917
LEVELS = {"typo25": 0.25, "typo50": 0.50}
VARIANTS = 3
MAX_ATTEMPTS = 50
EDIT_OPS = ("substitute", "delete", "insert", "transpose")

_KEY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _neighbours() -> dict[str, str]:
    pos = {ch: (r, c) for r, row in enumerate(_KEY_ROWS) for c, ch in enumerate(row)}
    out = {}
    for ch, (r, c) in pos.items():
        cands = [(r, c - 1), (r, c + 1), (r - 1, c), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c)]
        out[ch] = "".join(_KEY_ROWS[rr][cc] for rr, cc in cands
                          if 0 <= rr < len(_KEY_ROWS) and 0 <= cc < len(_KEY_ROWS[rr]))
    return out


NEIGHBOURS = _neighbours()


def _match_case(src: str, ch: str) -> str:
    return ch.upper() if src.isupper() else ch


def edit_word(word: str, rng: np.random.Generator) -> tuple[str, str]:
    """One random keyboard edit that changes the word. Returns (new word, operation)."""
    for _ in range(MAX_ATTEMPTS):
        op = EDIT_OPS[int(rng.integers(len(EDIT_OPS)))]
        i = int(rng.integers(len(word)))
        ch = word[i]
        near = NEIGHBOURS.get(ch.lower(), "")
        if op == "substitute" and near:
            new = word[:i] + _match_case(ch, near[int(rng.integers(len(near)))]) + word[i + 1:]
        elif op == "delete":
            new = word[:i] + word[i + 1:]
        elif op == "insert" and near:
            new = word[:i + 1] + _match_case(ch, near[int(rng.integers(len(near)))]) + word[i + 1:]
        elif op == "transpose":
            j = i + 1 if i + 1 < len(word) else i - 1
            a, b = sorted((i, j))
            new = word[:a] + word[b] + word[a] + word[b + 1:]
        else:
            continue
        if new != word:
            return new, op
    raise RuntimeError(f"could not edit {word!r}")


def perturb(text: str, p: float, rng: np.random.Generator) -> tuple[str, list[str]]:
    spans = [m for m in re.finditer(r"[A-Za-z]+", text) if len(m.group()) >= 3 and m.group().lower() != "machine"]
    k = max(1, int(np.floor(p * len(spans) + 0.5)))
    chosen = sorted(rng.choice(len(spans), size=min(k, len(spans)), replace=False).tolist())
    out, edits = text, []
    for idx in reversed(chosen):
        m = spans[idx]
        new, op = edit_word(m.group(), rng)
        out = out[:m.start()] + new + out[m.end():]
        edits.append(f"{m.group()}->{new} ({op})")
    return out, list(reversed(edits))


def _build() -> list[dict]:
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS

    anchors = [p for p in ANCHORED_PROMPTS if p["style"] == "original"]
    prompts = []
    for a_idx, anchor in enumerate(anchors, start=1):
        for l_idx, (style, p) in enumerate(LEVELS.items()):
            seen = {anchor["text"]}
            attempt = 0
            for v in range(1, VARIANTS + 1):
                while True:
                    rng = np.random.default_rng([TYPO_SEED, a_idx, l_idx, attempt])
                    attempt += 1
                    text, edits = perturb(anchor["text"], p, rng)
                    if text not in seen:
                        break
                    if attempt > MAX_ATTEMPTS:
                        raise RuntimeError(f"no distinct variant for {anchor['text']!r} at {style}")
                seen.add(text)
                prompts.append({
                    "id": f"T{a_idx:02d}-{style}-v{v}", "style": style, "text": text, "edits": edits,
                    "anchor": anchor["anchor"], "machine_id": anchor["machine_id"],
                    "design_severity": anchor["design_severity"], "expected_status": anchor["expected_status"],
                })
    return prompts


def _guarded(prompts: list[dict]) -> tuple[list[dict], list[str]]:
    from agents.input_guard import is_valid_fault_input

    kept, dropped = [], []
    for p in prompts:
        (kept if is_valid_fault_input(p["text"])[0] else dropped).append(p)
    return kept, [p["id"] for p in dropped]


TYPO_PROMPTS, DROPPED_BY_GUARD = _guarded(_build())


def check_typo_bank() -> list[str]:
    problems = []
    ids = [p["id"] for p in TYPO_PROMPTS]
    if len(ids) != len(set(ids)):
        problems.append("duplicate prompt ids")
    for p in TYPO_PROMPTS:
        if p["design_severity"] not in {"LOW", "MEDIUM", "HIGH"}:
            problems.append(f"{p['id']}: bad label")
        if not p["edits"]:
            problems.append(f"{p['id']}: no edit applied")
    return problems


def main() -> None:
    from collections import Counter
    from research.baselines import _classify_severity_regex

    print(f"[typo_bank] {len(TYPO_PROMPTS)} prompts kept, {len(DROPPED_BY_GUARD)} dropped by the input guard "
          f"{DROPPED_BY_GUARD}")
    print("  by style:", dict(Counter(p["style"] for p in TYPO_PROMPTS)))
    agree = sum(_classify_severity_regex(p["text"]).value == p["design_severity"] for p in TYPO_PROMPTS)
    print(f"  regex agrees with the inherited label on {agree}/{len(TYPO_PROMPTS)}")
    for p in TYPO_PROMPTS:
        print(f"  {p['id']:<16} {p['text']:<60} {'; '.join(p['edits'])}")
    problems = check_typo_bank()
    if problems:
        print("[typo_bank] PROBLEMS:", problems)
        sys.exit(1)
    print("[typo_bank] all checks passed")


if __name__ == "__main__":
    main()
