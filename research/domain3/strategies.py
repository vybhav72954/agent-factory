"""
Strategies that produce a PriorUpdate from operator text.

Mirrors the strategy ladder used in the main pipeline: uniform baseline, keyword
routing, keyword + severity regex, and a full agentic call (Gemini structured-
output). The downstream classifier and scoring code are shared.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Dict, Tuple

from .bayesian_anomaly import (
    FAILURE_MODES,
    PriorUpdate,
    UNIFORM_PRIOR,
    apply_prior_update,
    likelihood_per_mode,
    posterior,
)
from .prompts import PRIOR_UPDATE_SYSTEM_PROMPT


KEYWORD_MAP: Dict[str, Tuple[str, ...]] = {
    "bearing":    ("bearing", "grinding", "low rumble", "wobble"),
    "motor":      ("motor", "rpm", "torque", "stall", "winding"),
    "oil_seal":   ("oil", "lubrication", "seal", "leak", "pressure drop"),
    "coolant":    ("coolant", "overheat", "overheating", "cooling", "flow drop"),
    "electrical": ("voltage", "current", "amps", "fuse", "arc", "electrical", "spark"),
}

SEVERITY_KEYWORDS = {
    "high":   ("catastrophic", "severe", "critical", "major", "burning", "smoke", "fire", "stopped"),
    "medium": ("noticeable", "unusual", "elevated", "dipping", "louder", "warm"),
    "low":    ("minor", "slight", "small", "occasional", "intermittent"),
}

SEVERITY_DELTA = {"high": 0.45, "medium": 0.30, "low": 0.15}


def _route_keyword(text: str) -> str | None:
    """Return the failure mode whose keywords appear most often, or None."""
    t = text.lower()
    counts: Dict[str, int] = {m: 0 for m in FAILURE_MODES}
    for mode, kws in KEYWORD_MAP.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                counts[mode] += 1
    best = max(counts, key=lambda m: counts[m])
    return best if counts[best] > 0 else None


def _classify_severity(text: str) -> str:
    t = text.lower()
    for level in ("high", "medium", "low"):
        for kw in SEVERITY_KEYWORDS[level]:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                return level
    return "medium"


def strategy_uniform(_text: str) -> PriorUpdate:
    """Trivial baseline: no prior shift."""
    return PriorUpdate(
        failure_mode=FAILURE_MODES[0],
        prior_weight_delta=0.0,
        confidence=0.0,
        summary="Uniform-prior baseline; no operator-driven shift.",
    )


def strategy_keyword_only(text: str) -> PriorUpdate:
    """Keyword routing with a fixed positive delta."""
    mode = _route_keyword(text) or FAILURE_MODES[0]
    return PriorUpdate(
        failure_mode=mode,
        prior_weight_delta=0.30,
        confidence=0.5,
        summary=f"Keyword routing matched mode {mode!r} with fixed delta 0.30.",
    )


def strategy_keyword_regex(text: str) -> PriorUpdate:
    """Keyword routing + severity-driven delta. The 'strong deterministic baseline'."""
    mode = _route_keyword(text) or FAILURE_MODES[0]
    sev = _classify_severity(text)
    delta = SEVERITY_DELTA[sev]
    return PriorUpdate(
        failure_mode=mode,
        prior_weight_delta=delta,
        confidence=0.5 + 0.15 * {"low": 0, "medium": 1, "high": 2}[sev],
        summary=f"Regex routed mode={mode!r}, severity={sev!r}, delta={delta:.2f}.",
    )


def _gemini_diagnose(operator_text: str) -> PriorUpdate:
    """Call Gemini 2.5 Flash via google-genai SDK with PriorUpdate response_schema."""
    from google import genai
    from google.genai import types as gtypes

    api_key = (
        os.getenv("GEMINI_API_KEY_DIAGNOSTIC")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("no GEMINI_API_KEY_DIAGNOSTIC / GEMINI_API_KEY in env")
    client = genai.Client(api_key=api_key)
    prompt = f"{PRIOR_UPDATE_SYSTEM_PROMPT}\n\nOperator description: {operator_text}"
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PriorUpdate,
            temperature=0.2,
        ),
    )
    return PriorUpdate.model_validate_json(resp.text)


def strategy_agentic(text: str) -> PriorUpdate:
    """Full Gemini structured-output call."""
    try:
        return _gemini_diagnose(text)
    except Exception:
        return strategy_keyword_regex(text)


STRATEGIES: Dict[str, Callable[[str], PriorUpdate]] = {
    "uniform":               strategy_uniform,
    "keyword_only":          strategy_keyword_only,
    "keyword_regex":         strategy_keyword_regex,
    "agentic":               strategy_agentic,
}


def run_one(strategy_name: str, operator_text: str, observations) -> Tuple[PriorUpdate, Dict[str, float], Dict[str, float], float]:
    """Run a single end-to-end pipeline call. Returns (update, prior_used, posterior_dict, latency_ms)."""
    from .bayesian_anomaly import train_classifier  # cached below via module-level singleton

    clf = _get_classifier()
    fn = STRATEGIES[strategy_name]
    t0 = time.perf_counter()
    update = fn(operator_text)
    t1 = time.perf_counter()
    prior_used = apply_prior_update(UNIFORM_PRIOR, update)
    lik = likelihood_per_mode(clf, observations)
    post = posterior(prior_used, lik)
    return update, prior_used, post, (t1 - t0) * 1000.0


_CLF_SINGLETON = None


def _get_classifier():
    global _CLF_SINGLETON
    if _CLF_SINGLETON is None:
        from .bayesian_anomaly import train_classifier
        _CLF_SINGLETON = train_classifier(seed=0)
    return _CLF_SINGLETON
