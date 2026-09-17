"""
research/paraphrase/prompts_semantic.py

Meaning-based variant of the production diagnostic prompt.

The production DIAGNOSTIC_SYSTEM_PROMPT (agents/prompts.py) tells the LLM to
classify severity "based on the WORDING ... not your own judgment" and lists
trigger words. That makes the LLM behave like the regex by instruction, so a
paraphrase test run only with that prompt cannot say whether the LLM *could*
generalise. This module swaps only the SEVERITY CLASSIFICATION block for a
consequence-based definition and leaves every other section (sensor map, fault
routing, spike-value rules, window rules, constraints, example) unchanged, so
the two arms differ in exactly one block.

Research-only: production prompts are not modified.
"""

from __future__ import annotations

from agents.prompts import DIAGNOSTIC_SYSTEM_PROMPT

_BLOCK_START = "== SEVERITY CLASSIFICATION"
_BLOCK_END = "== SPIKE VALUE RULES =="

SEMANTIC_SEVERITY_BLOCK = """\
== SEVERITY CLASSIFICATION (JUDGE BY MEANING) ==
Decide severity from what the description says is actually happening to the
machine, using engineering judgment. Descriptions may be informal, indirect,
understated, use shop-floor slang, or mix languages. Classify the underlying
situation, not particular words.

HIGH — the component has failed or failure is imminent; the machine cannot
       safely keep running and must stop now (single hit should drive RUL to OFFLINE).
MEDIUM — a genuine fault that needs attention soon; performance or output
         quality may already be affected, but the machine can keep running for now.
LOW — an early or mild sign; the machine runs normally and a routine check
      is enough (RUL should barely move).

If the situation is genuinely unclear, choose MEDIUM.

"""


def build_semantic_prompt(base_prompt: str = DIAGNOSTIC_SYSTEM_PROMPT) -> str:
    """Return `base_prompt` with its SEVERITY CLASSIFICATION block replaced.

    Raises:
        ValueError: if the block markers are missing or out of order, which
            means agents/prompts.py changed and this variant must be revisited.
    """
    start = base_prompt.find(_BLOCK_START)
    end = base_prompt.find(_BLOCK_END)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("SEVERITY CLASSIFICATION block not found in the production prompt")
    return base_prompt[:start] + SEMANTIC_SEVERITY_BLOCK + base_prompt[end:]


DIAGNOSTIC_SEMANTIC_SYSTEM_PROMPT = build_semantic_prompt()
