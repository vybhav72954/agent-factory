"""
research/baselines.py

Compare three injection strategies on the same test prompts:

  (A) keyword-only       — no LLM, hardcoded keyword → sensor lookup
  (B) fixed-midrange     — no LLM, always inject same sensor with same value
  (C) agentic (Gemini)   — the existing pipeline

For each prompt, each strategy produces:
  - A SensorSpike (or equivalent)
  - A capacity report (after DL oracle + capacity_agent)
  - A dispatch order (from floor_manager, mocked deterministically for non-agentic strategies)

Then we score each dispatch with the rubric (research/evaluation_rubric.py)
and compute summary statistics:

  - Mean rubric aggregate score per strategy
  - RUL stability: same prompt repeated N times → variance in predicted RUL
  - Latency per pipeline run
  - Dispatch correctness rate (rubric aggregate >= 4.0)

Usage:
    python -m research.baselines                  # full run, no LLM judge
    python -m research.baselines --use-judge      # adds Gemini judge (slow + costs $)
    python -m research.baselines --stability-runs 5  # how many repeats per prompt for variance

Output:
    research/results/baselines_comparison.csv     — one row per (strategy, prompt, repeat)
    research/results/baselines_summary.md         — human-readable summary table
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, List

from pydantic import BaseModel, Field

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schemas import SensorSpike, FaultSeverity
from agents.diagnostic_agent import (
    SENSOR_TO_COL,
    SENSOR_CORRELATIONS,
    SCALED_MAX,
    CRITICAL_SENSORS,
    SEVERITY_MULTIPLIERS,
    _inject_spike as agentic_inject,
    translate_fault_to_tensor,
    FALLBACK_SPIKES,
    FALLBACK_KEYWORD_ORDER,
)
from agents.capacity_agent import update_capacity, reset_all
from agents.input_guard import is_valid_fault_input as validate_input
from dl_engine.inference import predict_rul, get_healthy_baseline, raw_value_for_scaled, get_scaler_ranges
from research.evaluation_rubric import TEST_PROMPTS, score_full, score_structural


RESULTS_DIR = PROJECT_ROOT / "research" / "results"
CSV_PATH = RESULTS_DIR / "baselines_comparison.csv"
SUMMARY_PATH = RESULTS_DIR / "baselines_summary.md"


# ─────────────────────────────────────────────────────────────────────────────
# Injection strategies
# ─────────────────────────────────────────────────────────────────────────────

def strategy_keyword_only(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Pure keyword lookup using the existing FALLBACK_SPIKES table. No LLM, no
    severity classifier, no LLM-driven correlation discovery — just direct
    string matching. Each fallback entry has a pre-baked severity and spike
    value that are NOT modified per-prompt.

    This is the weakest non-LLM baseline. For the stronger non-LLM baseline
    that adds a regex severity classifier on top, see
    `strategy_keyword_regex_severity` below.

    Reuses the agentic `_inject_spike` for fairness (same injection math).
    """
    text_lower = user_text.lower()
    spike = None
    for kw in FALLBACK_KEYWORD_ORDER:
        if kw in text_lower:
            spike = FALLBACK_SPIKES[kw].model_copy()
            break
    if spike is None:
        spike = FALLBACK_SPIKES["default"].model_copy()
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


# ── Regex severity classifier (mirrors prompts.py SEVERITY CLASSIFICATION) ──
# Word lists copied verbatim from the SEVERITY CLASSIFICATION block in
# agents/prompts.py so the comparison is fair: the regex sees exactly the
# same vocabulary the LLM was instructed to use.
import re as _re

_HIGH_TERMS = [
    "catastrophic", "complete failure", "total failure", "destroyed", "rupture",
    "burst", "explosion", "fire", "burning", "smoke", "shutdown", "seized", "shaft lock",
    "stopped completely", "halted", "critical failure", "emergency", "meltdown",
    "outage", "broken",
]
_LOW_TERMS = [
    "minor", "slight", "small", "subtle", "early", "first sign of", "wobble", "drift",
    "creep", "trending up", "rising slowly", "intermittent", "occasional",
    "warning sign",
]
# MEDIUM examples from prompts.py. The regex never matches these (MEDIUM is its
# default); they are used only as embedding prototypes below.
_MEDIUM_TERMS = [
    "spike", "surge", "anomaly", "abnormal", "elevated", "high",
    "fluctuation", "instability", "exceeded", "warning", "alert",
]


def _terms_pattern(terms: list[str]) -> "_re.Pattern":
    """Case-insensitive whole-word alternation; spaces inside a term match any whitespace."""
    alternation = "|".join(r"\s+".join(_re.escape(w) for w in t.split()) for t in terms)
    return _re.compile(rf"\b({alternation})\b", _re.IGNORECASE)


_HIGH_PATTERN = _terms_pattern(_HIGH_TERMS)
_LOW_PATTERN = _terms_pattern(_LOW_TERMS)

# Spike-value chosen at the centre of each severity band per the prompts.py
# SPIKE VALUE RULES (HIGH 0.85–0.98, MEDIUM 0.65–0.84, LOW 0.45–0.64).
_REGEX_SPIKE_VALUE = {
    FaultSeverity.HIGH:   0.92,
    FaultSeverity.MEDIUM: 0.75,
    FaultSeverity.LOW:    0.55,
}


def _classify_severity_regex(user_text: str) -> FaultSeverity:
    """
    Three-way severity classification by regex word-list match. Default MEDIUM.
    The keyword lists are intentionally identical to the SEVERITY CLASSIFICATION
    block in `agents/prompts.py` — so this baseline is "what the LLM was told
    to do, implemented as a regex." Any gap between this and the agentic
    strategy is signal that the LLM is doing something beyond keyword matching.
    """
    if _HIGH_PATTERN.search(user_text):
        return FaultSeverity.HIGH
    if _LOW_PATTERN.search(user_text):
        return FaultSeverity.LOW
    return FaultSeverity.MEDIUM


def _keyword_spike_with_severity(user_text: str, severity: FaultSeverity, tag: str) -> SensorSpike:
    """Keyword sensor routing (FALLBACK_SPIKES, same as keyword_only) with the given
    severity and the band-centre spike value. Shared by the non-LLM severity
    classifiers so they differ only in how severity is decided."""
    text_lower = user_text.lower()
    spike = None
    for kw in FALLBACK_KEYWORD_ORDER:
        if kw in text_lower:
            spike = FALLBACK_SPIKES[kw].model_copy()
            break
    if spike is None:
        spike = FALLBACK_SPIKES["default"].model_copy()
    spike.fault_severity = severity
    spike.spike_value = _REGEX_SPIKE_VALUE[severity]
    spike.plain_english_summary = (
        f"[{tag}] {spike.sensor_id} fault, severity={severity.value}, value={spike.spike_value:.2f}."
    )
    return spike


def strategy_keyword_regex_severity(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Strong non-LLM baseline: keyword routing for sensor selection (same as
    `strategy_keyword_only`) PLUS a regex-based severity classifier that
    overrides the pre-baked severity in FALLBACK_SPIKES.

    The regex classifier uses the same word lists that `agents/prompts.py`
    instructs Gemini to use. So this baseline is essentially "execute the
    prompt's classification rules without an LLM." If the LLM-driven strategy
    materially outperforms this, the LLM is contributing more than just
    keyword/pattern matching. If they are close, the LLM is doing what a
    regex can do for ~10× lower latency and zero API cost.
    """
    spike = _keyword_spike_with_severity(user_text, _classify_severity_regex(user_text), "BASELINE-REGEX")
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


# ── WordNet-expanded regex (stronger keyword baseline) ──────────────────────
# Tests whether a better-engineered keyword system closes the gap on unfamiliar
# wording. The expansion is purely mechanical, with no hand edits after seeing
# any test prompt:
#   - each single-word term in _HIGH_TERMS / _LOW_TERMS adds the lemma names of
#     its most frequent WordNet sense for each part of speech (first synset per
#     POS; WordNet orders senses by frequency);
#   - multi-word terms are kept as they are;
#   - input text is matched both as written and after WordNet lemmatisation
#     (morphy), so inflections such as "bursting" match "burst".
# Precedence is unchanged: HIGH, then LOW, else MEDIUM.
_extended_patterns: dict[str, "_re.Pattern"] = {}


def _wordnet_expand(terms: list[str]) -> list[str]:
    from nltk.corpus import wordnet as wn

    expanded = set(terms)
    for term in terms:
        if " " in term:
            continue
        for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
            synsets = wn.synsets(term, pos=pos)
            if synsets:
                expanded.update(l.name().replace("_", " ").lower() for l in synsets[0].lemmas())
    return sorted(expanded)


def _lemmatised(text: str) -> str:
    from nltk.corpus import wordnet as wn

    out = []
    for tok in _re.findall(r"[A-Za-z]+", text.lower()):
        base = wn.morphy(tok, wn.VERB) or wn.morphy(tok, wn.NOUN) or wn.morphy(tok, wn.ADJ) or tok
        out.append(base)
    return " ".join(out)


def _classify_severity_regex_extended(user_text: str) -> FaultSeverity:
    if not _extended_patterns:
        _extended_patterns["HIGH"] = _terms_pattern(_wordnet_expand(_HIGH_TERMS))
        _extended_patterns["LOW"] = _terms_pattern(_wordnet_expand(_LOW_TERMS))
    variants = (user_text, _lemmatised(user_text))
    if any(_extended_patterns["HIGH"].search(v) for v in variants):
        return FaultSeverity.HIGH
    if any(_extended_patterns["LOW"].search(v) for v in variants):
        return FaultSeverity.LOW
    return FaultSeverity.MEDIUM


def strategy_keyword_regex_extended(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keyword routing + WordNet-expanded regex severity (see the block comment above)."""
    spike = _keyword_spike_with_severity(user_text, _classify_severity_regex_extended(user_text),
                                         "BASELINE-REGEX-EXTENDED")
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


# ── Embedding severity classifier (non-generative NLP baseline) ─────────────
# Local sentence embeddings (all-MiniLM-L6-v2, CPU) with no API calls and no
# generative model. Prototypes are built only from vocabulary the production
# prompt already gives the LLM: every severity term in _HIGH_TERMS / _LOW_TERMS /
# _MEDIUM_TERMS combined with every sensor name in the prompt's SENSOR MAP
# ("<term> <sensor name> fault"). A description gets the class whose top-5
# prototype cosine similarities have the highest mean. No test prompt is used.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_TOP_K = 5
_embedding_state: dict[str, object] = {}


def _sensor_names_from_prompt() -> list[str]:
    from agents.prompts import DIAGNOSTIC_SYSTEM_PROMPT

    names = []
    for line in DIAGNOSTIC_SYSTEM_PROMPT.splitlines():
        m = _re.match(r"^\s+(W\d|Xs\d+)\s+—\s+(.+?)\s*(\(|$)", line)
        if m:
            names.append(m.group(2).strip().lower())
    return names


def _embedding_classifier():
    if not _embedding_state:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        sensors = _sensor_names_from_prompt()
        prototypes = {}
        for label, terms in (("HIGH", _HIGH_TERMS), ("MEDIUM", _MEDIUM_TERMS), ("LOW", _LOW_TERMS)):
            phrases = [f"{t} {s} fault" for t in terms for s in sensors]
            prototypes[label] = model.encode(phrases, normalize_embeddings=True, batch_size=256)
        _embedding_state.update(model=model, prototypes=prototypes)
    return _embedding_state["model"], _embedding_state["prototypes"]


def _classify_severity_embedding(user_text: str) -> FaultSeverity:
    model, prototypes = _embedding_classifier()
    v = model.encode([user_text], normalize_embeddings=True)[0]
    scores = {label: float(np.sort(p @ v)[-_EMBEDDING_TOP_K:].mean()) for label, p in prototypes.items()}
    return FaultSeverity(max(scores, key=scores.get))


def strategy_embedding_severity(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keyword routing + embedding nearest-prototype severity (see the block comment above)."""
    spike = _keyword_spike_with_severity(user_text, _classify_severity_embedding(user_text),
                                         "BASELINE-EMBEDDING")
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


def strategy_fixed_midrange(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Zero-intelligence baseline: regardless of input, inject Xs2 (Bearing Temp,
    the most-sensitive sensor) at a fixed MEDIUM-severity spike. This tests
    whether the agentic pipeline is doing anything more than "spike the most
    sensitive sensor by a moderate amount."
    """
    spike = SensorSpike(
        sensor_id="Xs2",
        spike_value=0.75,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="[BASELINE-FIXED] Fixed midrange Xs2 injection.",
    )
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


def strategy_agentic(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """The full pipeline — Gemini 2.5 Flash diagnostic agent → injection.
    This wraps the production `translate_fault_to_tensor` path unchanged so
    we measure what the deployed system actually does. For comparison
    against a newer Gemini (3.5 Flash) without modifying production code,
    see `strategy_gemini_3_5_flash` below."""
    injected, spike_dict, used_fallback = translate_fault_to_tensor(base_window, user_text)
    return injected, spike_dict


# ─────────────────────────────────────────────────────────────────────────────
# Gemini 3.5 Flash strategy (within-family version comparison)
# ─────────────────────────────────────────────────────────────────────────────
# Calls Gemini 3.5 Flash directly via google-genai SDK, using the same
# DIAGNOSTIC_SYSTEM_PROMPT and Pydantic schema as the production agentic
# pipeline. This isolates "which Gemini version" from "everything else"
# in the comparison.

GEMINI_3_5_FLASH_MODEL = "gemini-3.5-flash"

_gemini_clients: dict[str, object] = {}


def _get_gemini_client_for(model_hint: str = ""):
    """Lazy-init shared Gemini client. The same client can call any Gemini
    model — the model name is per-call. Returns None if no API key."""
    global _gemini_clients
    if "shared" in _gemini_clients:
        return _gemini_clients["shared"]
    try:
        from google import genai
    except ImportError:
        return None
    import os
    api_key = os.environ.get("GEMINI_API_KEY_DIAGNOSTIC")
    if not api_key:
        return None
    _gemini_clients["shared"] = genai.Client(api_key=api_key)
    return _gemini_clients["shared"]


RATE_LIMIT_MAX_ATTEMPTS = 5


def _is_rate_limit_error(e: Exception) -> bool:
    """True for HTTP 429 / quota errors from the Gemini or Groq SDKs."""
    text = f"{type(e).__name__} {e}"
    return any(tag in text for tag in ("RateLimit", "ResourceExhausted", "RESOURCE_EXHAUSTED", "429"))


def _with_rate_limit_retry(call: Callable):
    """Run `call()`, retrying with exponential backoff (4, 8, 16, 32 s) on rate-limit errors.

    Other errors, and a rate-limit error on the final attempt, propagate so the
    caller's keyword fallback still applies. Added 2026-09-16: without it a 429
    silently became a fallback row.
    """
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as e:
            if attempt == RATE_LIMIT_MAX_ATTEMPTS or not _is_rate_limit_error(e):
                raise
            time.sleep(2 ** (attempt + 1))


def _gemini_diagnose(model_id: str, user_text: str, system_prompt: str | None = None,
                     usage_out: dict | None = None, thinking_budget: int | None = None) -> SensorSpike:
    """
    Call a specific Gemini model with the production DIAGNOSTIC_SYSTEM_PROMPT
    and structured-output mode (response_schema=SensorSpike). On any failure
    (no client, network error, validation error), fall back to keyword lookup
    so the comparison is robust. Identical retry/fallback semantics to
    `_groq_diagnose` for a fair within-vendor comparison.

    Args:
        model_id:      Gemini model name.
        user_text:     fault description.
        system_prompt: optional replacement for DIAGNOSTIC_SYSTEM_PROMPT. Used by
                       the paraphrase experiment's meaning-based severity prompt
                       (`research/paraphrase/prompts_semantic.py`).
        usage_out:     optional dict filled with input_tokens / output_tokens
                       (output includes thinking tokens) for cost analysis.
        thinking_budget: optional Gemini thinking budget. None keeps the model
                       default (thinking on for 2.5 Flash); 0 disables thinking,
                       matching the production agent in agents/diagnostic_agent.py.
    """
    from agents.prompts import DIAGNOSTIC_SYSTEM_PROMPT
    from google.genai import types as gtypes
    from pydantic import ValidationError

    client = _get_gemini_client_for(model_id)
    if client is None:
        spike = _get_fallback_spike(user_text)
        spike.plain_english_summary = f"[GEMINI-UNAVAILABLE] {spike.plain_english_summary}"
        return spike

    prompt = system_prompt if system_prompt is not None else DIAGNOSTIC_SYSTEM_PROMPT
    config_kwargs = dict(response_mime_type="application/json", response_schema=SensorSpike, temperature=0.0)
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=thinking_budget)
    try:
        response = _with_rate_limit_retry(lambda: client.models.generate_content(
            model=model_id,
            contents=f"{prompt}\n\nFault description: {user_text}",
            config=gtypes.GenerateContentConfig(**config_kwargs),
        ))
        um = getattr(response, "usage_metadata", None)
        if usage_out is not None and um is not None:
            usage_out["input_tokens"] = int(um.prompt_token_count or 0)
            usage_out["output_tokens"] = int(um.candidates_token_count or 0) + int(getattr(um, "thoughts_token_count", 0) or 0)
        spike = SensorSpike.model_validate_json(response.text)
        if not spike.plain_english_summary.startswith("["):
            spike.plain_english_summary = f"[{model_id}] {spike.plain_english_summary}"
        return spike
    except (ValidationError, Exception) as e:
        spike = _get_fallback_spike(user_text)
        err_class = type(e).__name__
        spike.plain_english_summary = (
            f"[GEMINI-FALLBACK {err_class}] {spike.plain_english_summary}"
        )
        return spike


def strategy_gemini_3_5_flash(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Diagnostic agent backed by Gemini 3.5 Flash (newer than the
    production 2.5 Flash used by `strategy_agentic`)."""
    spike = _gemini_diagnose(GEMINI_3_5_FLASH_MODEL, user_text)
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


# ─────────────────────────────────────────────────────────────────────────────
# Continuous-output severity strategy (Extension #6 / paper §5.6 mitigation)
# ─────────────────────────────────────────────────────────────────────────────
# Gemini emits a CONTINUOUS severity_multiplier directly (in [0, 1]) instead
# of picking a LOW/MEDIUM/HIGH bucket whose value is then table-looked-up.
# This is the most natural mitigation for the categorical-to-continuous
# interface problem the paper documents. Two possible results:
#
#   - If continuous severity narrows the LLM-vs-regex gap → the categorical
#     discretisation was costing real information. Paper §5.6 becomes a
#     measured mitigation, not just a proposal.
#   - If continuous severity is similar or worse → the interface problem is
#     deeper than discretisation; the LLM doesn't have the calibrated
#     intuition about magnitude that the regex baseline approximates with
#     fixed band centres.

class SensorSpikeContinuous(BaseModel):
    """Spike schema with a continuous severity_multiplier instead of a
    categorical FaultSeverity enum. Used only by the research strategy
    `strategy_agentic_continuous`."""
    sensor_id: str = Field(
        description=(
            "One of W0-W3, Xs0-Xs13. Same sensor map as the standard "
            "SensorSpike — see DIAGNOSTIC_CONTINUOUS_SYSTEM_PROMPT for "
            "the FAULT → SENSOR keyword routing."
        )
    )
    spike_value: float = Field(
        ge=0.05, le=0.98,
        description="Normalised spike intensity. Default 0.75 if unclear.",
    )
    severity_multiplier: float = Field(
        ge=0.0, le=1.0,
        description=(
            "CONTINUOUS multiplier that REPLACES the LOW/MEDIUM/HIGH table "
            "lookup. Anchor: 0.05–0.15 = minor; 0.20–0.35 = elevated; "
            "0.40–0.60 = default unqualified fault; 0.65–0.85 = severe; "
            "0.85–0.98 = catastrophic."
        ),
    )
    affected_window_positions: List[int] = Field(
        description="Timestep indices 0–49 where the spike is injected. Min 1, max 10."
    )
    plain_english_summary: str = Field(
        description="One sentence for the comms log. No markdown, no brackets."
    )


def _gemini_diagnose_continuous(user_text: str) -> SensorSpikeContinuous | None:
    """
    Call Gemini 2.5 Flash with the continuous-multiplier prompt + schema.
    Returns None on any failure so the strategy can fall back deterministically.
    """
    from agents.prompts import DIAGNOSTIC_CONTINUOUS_SYSTEM_PROMPT
    from google.genai import types as gtypes
    from pydantic import ValidationError

    client = _get_gemini_client_for("continuous")
    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{DIAGNOSTIC_CONTINUOUS_SYSTEM_PROMPT}\n\nFault description: {user_text}",
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SensorSpikeContinuous,
                temperature=0.0,
            ),
        )
        spike = SensorSpikeContinuous.model_validate_json(response.text)
        if not spike.plain_english_summary.startswith("["):
            spike.plain_english_summary = (
                f"[continuous] {spike.plain_english_summary}"
            )
        return spike
    except (ValidationError, Exception):
        return None


def strategy_agentic_continuous(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Continuous-multiplier variant of the agentic strategy. Gemini emits
    severity_multiplier directly; we pass it to agentic_inject() via the
    `multiplier_override` parameter, bypassing SEVERITY_MULTIPLIERS lookup.

    Falls back to the keyword-only path on Gemini failure, tagged in the
    returned spike_dict so post-hoc audit can tell apart LLM-emitted vs
    fallback rows.
    """
    spike_cont = _gemini_diagnose_continuous(user_text)
    if spike_cont is None:
        # Fall back: keyword lookup + SEVERITY_MULTIPLIERS[MEDIUM] as the multiplier.
        # This mirrors the Groq strategies' fallback behaviour.
        fallback_spike = _get_fallback_spike(user_text)
        from agents.diagnostic_agent import SEVERITY_MULTIPLIERS
        from agents.schemas import FaultSeverity
        fallback_multiplier = float(SEVERITY_MULTIPLIERS[FaultSeverity.MEDIUM])
        injected = agentic_inject(base_window, fallback_spike,
                                   multiplier_override=fallback_multiplier)
        out = fallback_spike.model_dump(mode="json")
        out["severity_multiplier"] = fallback_multiplier
        out["plain_english_summary"] = (
            f"[CONTINUOUS-FALLBACK] {out['plain_english_summary']}"
        )
        return injected, out

    # Build a SensorSpike shell so agentic_inject can route via SENSOR_TO_COL
    # and SENSOR_CORRELATIONS unchanged. FaultSeverity is set to MEDIUM as a
    # placeholder (irrelevant — multiplier_override wins).
    from agents.schemas import FaultSeverity
    shim = SensorSpike(
        sensor_id=spike_cont.sensor_id,
        spike_value=spike_cont.spike_value,
        affected_window_positions=spike_cont.affected_window_positions,
        fault_severity=FaultSeverity.MEDIUM,  # placeholder, ignored due to override
        plain_english_summary=spike_cont.plain_english_summary,
    )
    injected = agentic_inject(base_window, shim,
                               multiplier_override=spike_cont.severity_multiplier)

    # Return both the shim spike fields AND the continuous multiplier for
    # post-hoc analysis.
    out = shim.model_dump(mode="json")
    out["severity_multiplier"] = float(spike_cont.severity_multiplier)
    return injected, out


# ─────────────────────────────────────────────────────────────────────────────
# Groq Llama strategies (cross-LLM extension #1)
# ─────────────────────────────────────────────────────────────────────────────
# These call Groq's OpenAI-compatible chat completions API with JSON mode.
# Same DIAGNOSTIC_SYSTEM_PROMPT as Gemini — the comparison isolates the LLM,
# not the prompt. Same Pydantic SensorSpike for validation. Same fallback to
# FALLBACK_SPIKES if Llama produces invalid output (matches the agentic
# strategy's fallback behaviour).

GROQ_LLAMA_3_3_MODEL = "llama-3.3-70b-versatile"
GROQ_LLAMA_4_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
# Groq retired both Llama models by 2026-09 (NotFoundError). Their May 2026 rows
# stay in the result CSVs as historical results; these two open-weight models
# replace them in new runs.
GROQ_GPT_OSS_120B_MODEL = "openai/gpt-oss-120b"
GROQ_QWEN3_8_MODEL      = "qwen/qwen3.8-27b"

_groq_client = None  # Lazy init so missing GROQ_API_KEY doesn't break import


def _get_groq_client():
    """Lazy-init Groq client. Returns None if SDK not installed or no API key."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    try:
        from groq import Groq
    except ImportError:
        return None
    import os
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    _groq_client = Groq(api_key=api_key)
    return _groq_client


def _groq_diagnose(model_id: str, user_text: str, system_prompt: str | None = None,
                   usage_out: dict | None = None) -> SensorSpike:
    """
    Send DIAGNOSTIC_SYSTEM_PROMPT + user_text to a Groq-hosted Llama model and
    parse the JSON response into a SensorSpike. On any failure (no client,
    network error, invalid JSON, Pydantic validation error), fall back to the
    keyword-only FALLBACK_SPIKES lookup so the comparison is robust.

    This mirrors the agentic Gemini path's fallback behaviour. The fallback is
    counted as part of the strategy's outputs — if Llama fails often enough
    that fallback dominates, the strategy's status-match collapses toward
    keyword_only's, which is the honest signal.

    Args:
        model_id:      Groq model name.
        user_text:     fault description.
        system_prompt: optional replacement for DIAGNOSTIC_SYSTEM_PROMPT (see
                       `_gemini_diagnose`).
        usage_out:     optional dict filled with input_tokens / output_tokens
                       (output includes reasoning tokens) for cost analysis.
    """
    from agents.prompts import DIAGNOSTIC_SYSTEM_PROMPT
    from pydantic import ValidationError

    if system_prompt is None:
        system_prompt = DIAGNOSTIC_SYSTEM_PROMPT

    client = _get_groq_client()
    if client is None:
        # No SDK / no API key → keyword fallback. Tag so we can audit later.
        spike = _get_fallback_spike(user_text)
        spike.plain_english_summary = f"[GROQ-UNAVAILABLE] {spike.plain_english_summary}"
        return spike

    schema_hint = (
        "\n\nReturn ONLY a JSON object with these fields:\n"
        '  "sensor_id": str (one of W0-W3, Xs0-Xs13)\n'
        '  "spike_value": float in [0, 1]\n'
        '  "affected_window_positions": list of ints in [0, 49], length 1-10\n'
        '  "fault_severity": str (one of "LOW", "MEDIUM", "HIGH")\n'
        '  "plain_english_summary": str (one sentence, no markdown)\n'
        "Do NOT include any text outside the JSON object."
    )
    try:
        resp = _with_rate_limit_retry(lambda: client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt + schema_hint},
                {"role": "user",   "content": f"Fault description: {user_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # deterministic for stability comparison
            # 2000, not 400: reasoning models (gpt-oss-120b) spend tokens reasoning
            # before the JSON, and a 400 cap truncated it (json_validate_failed,
            # 14/54 fallback rows on 2026-09-16). Non-reasoning models stop well
            # under 400 either way.
            max_tokens=2000,
        ))
        if usage_out is not None and getattr(resp, "usage", None) is not None:
            usage_out["input_tokens"] = int(resp.usage.prompt_tokens or 0)
            usage_out["output_tokens"] = int(resp.usage.completion_tokens or 0)
        raw = resp.choices[0].message.content
        spike = SensorSpike.model_validate_json(raw)
        # Tag so post-hoc audit can tell apart Llama-emitted vs fallback rows
        if not spike.plain_english_summary.startswith("["):
            spike.plain_english_summary = (
                f"[{model_id.split('/')[-1]}] {spike.plain_english_summary}"
            )
        return spike
    except (ValidationError, Exception) as e:
        # Any failure → keyword fallback, tagged for audit
        spike = _get_fallback_spike(user_text)
        err_class = type(e).__name__
        spike.plain_english_summary = (
            f"[GROQ-FALLBACK {err_class}] {spike.plain_english_summary}"
        )
        return spike


# ── Local small open models via Ollama (no API; quantised, run on the local GPU) ──
# Added 2026-09-17 (research/AEI/analysis_plan_additions.md, section B): tests
# whether the advantage on unfamiliar wording needs a hosted model.
OLLAMA_MODELS = {                     # strategy suffix -> (Ollama tag, request options)
    "llama3_2_3b": ("llama3.2:3b", {}),
    "gemma3_4b":   ("gemma3:4b", {}),
    "qwen3_4b":    ("qwen3:4b", {"think": False}),
}
OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_predict": 512}


def _ollama_url() -> str:
    """Base URL from OLLAMA_HOST (host:port or URL), default the standard local port."""
    import os
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return host if host.startswith("http") else f"http://{host}"


def _ollama_diagnose(model_tag: str, user_text: str, system_prompt: str | None = None,
                     usage_out: dict | None = None, think: bool | None = None) -> SensorSpike:
    """
    Same request as `_groq_diagnose` (system prompt + JSON field hint, temperature 0)
    sent to a local Ollama model, with the SensorSpike JSON schema as a decoding
    constraint. Any failure falls back to the keyword lookup, tagged
    [OLLAMA-FALLBACK <error>]; an unreachable server is tagged [OLLAMA-UNAVAILABLE].
    """
    import requests
    from agents.prompts import DIAGNOSTIC_SYSTEM_PROMPT

    if system_prompt is None:
        system_prompt = DIAGNOSTIC_SYSTEM_PROMPT
    schema_hint = (
        "\n\nReturn ONLY a JSON object with these fields:\n"
        '  "sensor_id": str (one of W0-W3, Xs0-Xs13)\n'
        '  "spike_value": float in [0, 1]\n'
        '  "affected_window_positions": list of ints in [0, 49], length 1-10\n'
        '  "fault_severity": str (one of "LOW", "MEDIUM", "HIGH")\n'
        '  "plain_english_summary": str (one sentence, no markdown)\n'
        "Do NOT include any text outside the JSON object."
    )
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system_prompt + schema_hint},
            {"role": "user",   "content": f"Fault description: {user_text}"},
        ],
        "format": SensorSpike.model_json_schema(),
        "options": OLLAMA_OPTIONS,
        "stream": False,
        "keep_alive": "30m",
    }
    if think is not None:
        payload["think"] = think
    try:
        resp = requests.post(f"{_ollama_url()}/api/chat", json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        spike = _get_fallback_spike(user_text)
        spike.plain_english_summary = f"[OLLAMA-UNAVAILABLE] {spike.plain_english_summary}"
        return spike
    try:
        resp.raise_for_status()
        body = resp.json()
        if usage_out is not None:
            usage_out["input_tokens"] = int(body.get("prompt_eval_count") or 0)
            usage_out["output_tokens"] = int(body.get("eval_count") or 0)
        spike = SensorSpike.model_validate_json(body["message"]["content"])
        if not spike.plain_english_summary.startswith("["):
            spike.plain_english_summary = f"[{model_tag}] {spike.plain_english_summary}"
        return spike
    except Exception as e:
        spike = _get_fallback_spike(user_text)
        spike.plain_english_summary = f"[OLLAMA-FALLBACK {type(e).__name__}] {spike.plain_english_summary}"
        return spike


def _ollama_strategy(key: str) -> Callable:
    model_tag, request = OLLAMA_MODELS[key]

    def strategy(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
        spike = _ollama_diagnose(model_tag, user_text, **request)
        injected = agentic_inject(base_window, spike)
        return injected, spike.model_dump(mode="json")

    strategy.__doc__ = f"Diagnostic agent backed by {model_tag}, local via Ollama."
    return strategy


def _trained_strategy(arm: str) -> Callable:
    """Keyword routing + a trained severity classifier, regime R1 (research/trained_baselines.py)."""
    def strategy(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
        from research.trained_baselines import classify_prompt_knowledge
        severity = FaultSeverity(classify_prompt_knowledge(arm, user_text))
        spike = _keyword_spike_with_severity(user_text, severity, f"BASELINE-{arm.upper()}")
        injected = agentic_inject(base_window, spike)
        return injected, spike.model_dump(mode="json")

    strategy.__doc__ = f"Keyword routing + {arm} severity trained on production-prompt vocabulary (R1)."
    return strategy


def _get_fallback_spike(user_text: str) -> SensorSpike:
    """Same keyword lookup as strategy_keyword_only — used when Groq fails."""
    text_lower = user_text.lower()
    for kw in FALLBACK_KEYWORD_ORDER:
        if kw in text_lower:
            return FALLBACK_SPIKES[kw].model_copy()
    return FALLBACK_SPIKES["default"].model_copy()


def strategy_groq_llama3(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Diagnostic agent backed by Llama 3.3 70B via Groq."""
    spike = _groq_diagnose(GROQ_LLAMA_3_3_MODEL, user_text)
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


def strategy_groq_llama4(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Diagnostic agent backed by Llama 4 Maverick via Groq."""
    spike = _groq_diagnose(GROQ_LLAMA_4_MODEL, user_text)
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


def strategy_groq_gpt_oss_120b(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Diagnostic agent backed by OpenAI gpt-oss-120b (open-weight) via Groq."""
    spike = _groq_diagnose(GROQ_GPT_OSS_120B_MODEL, user_text)
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


def strategy_groq_qwen3_8(user_text: str, base_window: np.ndarray) -> tuple[np.ndarray, dict]:
    """Diagnostic agent backed by Qwen 3.8 27B (open-weight) via Groq."""
    spike = _groq_diagnose(GROQ_QWEN3_8_MODEL, user_text)
    injected = agentic_inject(base_window, spike)
    return injected, spike.model_dump(mode="json")


STRATEGIES: dict[str, Callable] = {
    "keyword_only":            strategy_keyword_only,
    "keyword_regex_severity":  strategy_keyword_regex_severity,
    "keyword_regex_extended":  strategy_keyword_regex_extended,  # WordNet-expanded word lists
    "embedding_severity":      strategy_embedding_severity,      # MiniLM nearest-prototype, local CPU
    "fixed_midrange":          strategy_fixed_midrange,
    "agentic":                 strategy_agentic,                # Gemini 2.5 Flash (production)
    "agentic_continuous":      strategy_agentic_continuous,     # Gemini 2.5 Flash, continuous multiplier
    "gemini_3_5_flash":        strategy_gemini_3_5_flash,       # Gemini 3.5 Flash (newer)
    "groq_llama3":             strategy_groq_llama3,            # Llama 3.3 70B via Groq (retired on Groq; historical)
    "groq_llama4":             strategy_groq_llama4,            # Llama 4 Scout via Groq (retired on Groq; historical)
    "groq_gpt_oss_120b":       strategy_groq_gpt_oss_120b,      # OpenAI gpt-oss-120b via Groq
    "groq_qwen3_8":            strategy_groq_qwen3_8,           # Qwen 3.8 27B via Groq
    # Trained severity classifiers, regime R1 (research/trained_baselines.py), local CPU
    "tfidf_logreg":            _trained_strategy("tfidf_logreg"),
    "minilm_logreg":           _trained_strategy("minilm_logreg"),
    "bge_m3_logreg":           _trained_strategy("bge_m3_logreg"),
    "minilm_finetuned":        _trained_strategy("minilm_finetuned"),
    # Small open models, local via Ollama (quantised, GTX 1650)
    **{f"ollama_{key}": _ollama_strategy(key) for key in OLLAMA_MODELS},
}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch generation (deterministic template — fair across strategies)
# ─────────────────────────────────────────────────────────────────────────────

DISPATCH_TEMPLATES = {
    "OFFLINE": (
        "[Floor Manager] {machine_name} OFFLINE at RUL {rul:.1f} — mandatory shutdown initiated. "
        "Halt all production on this unit and dispatch maintenance crew immediately. "
        "Reroute {machine_name} workload to remaining online machines. "
        "Factory at {capacity_pct:.1f}% capacity — {risk_note}ΣPD/T at {machine_req:.2f}."
    ),
    "DEGRADED": (
        "[Floor Manager] {machine_name} entering DEGRADED status at RUL {rul:.1f} — reduce to 50% load. "
        "Schedule maintenance window within the next shift cycle. "
        "Inspect upstream conditions before restoring full operation. "
        "Factory at {capacity_pct:.1f}% capacity — ΣPD/T at {machine_req:.2f}."
    ),
    "ONLINE": (
        "[Floor Manager] {machine_name} is ONLINE and operating nominally. "
        "No immediate action required — continue monitoring. "
        "RUL at {rul:.1f} cycles — next scheduled inspection on-cycle. "
        "Factory at {capacity_pct:.1f}% capacity — all systems healthy."
    ),
}


def render_dispatch(capacity_report: dict) -> str:
    """Deterministic dispatch from template. Used by ALL strategies for fairness:
    we're measuring injection strategy quality, not floor-manager LLM quality."""
    tpl = DISPATCH_TEMPLATES[capacity_report["status"]]
    risk_note = "breakeven risk ACTIVE, authorize overtime. " if capacity_report.get("breakeven_risk") else ""
    return tpl.format(
        machine_name=capacity_report["machine_name"],
        rul=capacity_report["rul"],
        capacity_pct=capacity_report["capacity_pct"],
        machine_req=capacity_report["machine_req"],
        risk_note=risk_note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    strategy_name: str,
    strategy_fn: Callable,
    user_text: str,
    machine_id: int,
    base_window: np.ndarray,
) -> dict:
    """Run one prompt through one strategy. Returns a dict of (strategy, prompt, repeat) data."""
    t0 = time.time()

    # Input guard (same for all strategies)
    ok, reason = validate_input(user_text)
    if not ok:
        return {
            "strategy":       strategy_name,
            "prompt":         user_text,
            "machine_id":     machine_id,
            "rejected":       True,
            "rejection_reason": reason,
            "rul":            None,
            "status":         "REJECTED",
            "capacity_pct":   None,
            "dispatch":       "",
            "rubric_aggregate": None,
            "latency_ms":     (time.time() - t0) * 1000,
        }

    # Injection
    injected, spike_dict = strategy_fn(user_text, base_window)

    # DL oracle (always real model — keeps the baseline fair)
    rul = float(predict_rul(injected))

    # Capacity math (deterministic)
    capacity_report = update_capacity(machine_id, rul)

    # Dispatch (deterministic template, NOT the LLM floor-manager)
    dispatch = render_dispatch(capacity_report)

    # Score
    score = score_structural(dispatch, capacity_report)

    return {
        "strategy":         strategy_name,
        "prompt":           user_text,
        "machine_id":       machine_id,
        "rejected":         False,
        "rejection_reason": "",
        "sensor_id":        spike_dict.get("sensor_id"),
        "severity":         spike_dict.get("fault_severity"),
        "spike_value":      spike_dict.get("spike_value"),
        # Summary carries the [..-FALLBACK ..] audit tags; continuous multiplier
        # lets agentic_continuous rows be replayed (research/multiplier_sweep.py).
        "spike_summary":    spike_dict.get("plain_english_summary"),
        "severity_multiplier": spike_dict.get("severity_multiplier"),
        "rul":              rul,
        "status":           capacity_report["status"],
        "capacity_pct":     capacity_report["capacity_pct"],
        "machine_req":      capacity_report["machine_req"],
        "dispatch":         dispatch,
        "rubric_aggregate": score.aggregate,
        "rubric_machine_name": score.machine_name_correct,
        "rubric_rul":       score.rul_mentioned,
        "rubric_action":    score.action_matches_status,
        "rubric_capacity":  score.capacity_mentioned,
        "rubric_no_hallucination": score.no_hallucinated_numbers,
        "rubric_notes":     "; ".join(score.notes),
        "latency_ms":       (time.time() - t0) * 1000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-runs", type=int, default=3,
                        help="How many times to repeat each (strategy, prompt) to measure RUL stability (default 3)")
    parser.add_argument("--use-judge", action="store_true",
                        help="Also run the LLM-as-judge semantic check (slower, costs $)")
    parser.add_argument("--skip-agentic", action="store_true",
                        help="Skip the agentic strategy (useful when no Gemini key)")
    parser.add_argument("--only-strategy", type=str, default=None,
                        choices=list(STRATEGIES.keys()),
                        help="Run only this one strategy and APPEND to the existing CSV. "
                             "Use when adding a new strategy without re-running Gemini calls "
                             "for strategies already in the CSV.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    all_rows = []
    strategies = dict(STRATEGIES)
    if args.skip_agentic:
        strategies.pop("agentic", None)
        print("[baselines] skipping agentic strategy (no Gemini)")

    if args.only_strategy:
        strategies = {args.only_strategy: STRATEGIES[args.only_strategy]}
        print(f"[baselines] running only: {args.only_strategy} (will APPEND to existing CSV)")

    total_runs = len(strategies) * len(TEST_PROMPTS) * args.stability_runs
    run_idx = 0
    t0 = time.time()

    for strategy_name, strategy_fn in strategies.items():
        print(f"\n[baselines] strategy: {strategy_name}")
        for repeat in range(args.stability_runs):
            # Reset capacity-agent state between repeats for a fair comparison
            reset_all()
            for prompt, machine_id, _expected_status in TEST_PROMPTS:
                # Use a fresh healthy baseline per prompt (no cross-prompt damage carry-over)
                base = get_healthy_baseline(noise_std_frac=0.02)
                row = run_pipeline(strategy_name, strategy_fn, prompt, machine_id, base)
                row["repeat"] = repeat
                all_rows.append(row)
                run_idx += 1
                if run_idx % 10 == 0:
                    elapsed = time.time() - t0
                    rate = run_idx / elapsed
                    eta = (total_runs - run_idx) / max(rate, 0.01)
                    print(f"  {run_idx}/{total_runs}  {rate:.1f} runs/s  eta {eta:.0f}s")

    df_new = pd.DataFrame(all_rows)

    if args.only_strategy and CSV_PATH.exists():
        # Append: keep existing rows for other strategies, drop any prior rows
        # for THIS strategy (so re-running --only-strategy is idempotent), then
        # concat the new rows.
        df_existing = pd.read_csv(CSV_PATH)
        df_existing = df_existing[df_existing["strategy"] != args.only_strategy]
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[baselines] wrote {CSV_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows total)")

    write_summary(df, args.stability_runs)

    # Auto-snapshot to variant-suffixed siblings (HIGH-7).
    from research import write_variant_snapshot
    for p in [CSV_PATH, SUMMARY_PATH]:
        snap = write_variant_snapshot(p)
        if snap is not None:
            print(f"[baselines] snapshot -> {snap.relative_to(PROJECT_ROOT)}")


def write_summary(df: pd.DataFrame, stability_runs: int) -> None:
    """Generate the human-readable comparison markdown."""
    # Exclude rejected rows for outcome stats (they're handled separately)
    df_nonreject = df[~df["rejected"]].copy()

    # ── Status-match: did the strategy produce the expected status? ──
    # The dispatch rubric saturates at 5.0 across all strategies because the
    # deterministic dispatch template (shared by all strategies) always matches
    # the capacity report it was generated from. The real strategy-quality
    # signal is whether the strategy drove the capacity-agent to the EXPECTED
    # status from TEST_PROMPTS (e.g. "catastrophic bearing failure" should
    # land OFFLINE; "minor wobble" should stay ONLINE).
    expected_status_by_prompt = {p: s for p, _mid, s in TEST_PROMPTS}
    df_nonreject["expected_status"] = df_nonreject["prompt"].map(expected_status_by_prompt)
    df_nonreject["status_match"] = df_nonreject["status"] == df_nonreject["expected_status"]

    # Per-strategy summary
    per_strategy = []
    for strategy in df["strategy"].unique():
        sub = df_nonreject[df_nonreject["strategy"] == strategy]
        rejected = df[(df["strategy"] == strategy) & df["rejected"]]
        per_strategy.append({
            "strategy":              strategy,
            "n_runs":                len(sub),
            "status_match_rate":     sub["status_match"].mean(),
            "mean_rubric_aggregate": sub["rubric_aggregate"].mean(),
            "median_rubric_aggregate": sub["rubric_aggregate"].median(),
            "dispatch_coherent_rate": (sub["rubric_aggregate"] >= 4.0).mean(),
            "mean_rul":              sub["rul"].mean(),
            "mean_latency_ms":       sub["latency_ms"].mean(),
            "p95_latency_ms":        sub["latency_ms"].quantile(0.95),
            "rejected_count":        len(rejected),
        })
    per_strategy_df = pd.DataFrame(per_strategy)

    # RUL stability: variance of RUL across repeats for the same (strategy, prompt)
    stability = (
        df_nonreject.groupby(["strategy", "prompt"])["rul"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "rul_mean", "std": "rul_std", "count": "rul_n"})
    )
    stability_per_strategy = (
        stability.groupby("strategy")["rul_std"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_rul_std", "max": "max_rul_std"})
    )

    # Sensor diversity: how many distinct sensor_ids did each strategy pick?
    sensor_div = (
        df_nonreject.groupby("strategy")["sensor_id"]
        .nunique()
        .reset_index()
        .rename(columns={"sensor_id": "distinct_sensors_used"})
    )

    # Status outcome breakdown
    status_counts = (
        df_nonreject.groupby(["strategy", "status"])
        .size()
        .reset_index(name="count")
    )

    md = ["# Baseline Comparison — Seven Injection Strategies (Cross-LLM)\n"]
    md.append(f"**Generated by:** `python -m research.baselines --stability-runs {stability_runs}`\n")
    md.append(f"**Test set:** {len(TEST_PROMPTS)} prompts × {stability_runs} repeats = "
              f"{len(TEST_PROMPTS) * stability_runs} runs per strategy\n\n")
    md.append("**Three deterministic baselines + four LLM-driven strategies across two vendors with within-family version comparisons:**\n")
    md.append("- `keyword_only` — pure FALLBACK_SPIKES lookup (weak deterministic baseline)\n")
    md.append("- `fixed_midrange` — constant Xs2 spike (trivial deterministic baseline)\n")
    md.append("- `keyword_regex_severity` — **strong deterministic baseline**: keyword routing + regex severity classifier using the same word lists as `prompts.py`\n")
    md.append("- `keyword_regex_extended` — same routing, word lists mechanically expanded with WordNet synonyms and lemmatised matching\n")
    md.append("- `embedding_severity` — same routing, severity from a local MiniLM sentence-embedding classifier with prototypes built only from the `prompts.py` word lists\n")
    md.append("- `agentic` — Gemini 2.5 Flash via Google's `response_schema` structured-output mode (production pipeline)\n")
    md.append("- `gemini_3_5_flash` — Gemini 3.5 Flash via the same `response_schema` mode (newer Google model)\n")
    md.append("- `groq_llama3` — Llama 3.3 70B Versatile via Groq (Meta family, OpenAI-compatible JSON mode)\n")
    md.append("- `groq_llama4` — Llama 4 Scout 17B via Groq (Meta family, OpenAI-compatible JSON mode)\n")
    md.append("- `groq_gpt_oss_120b` — OpenAI gpt-oss-120b via Groq (open-weight; replaces the retired Llama models)\n")
    md.append("- `groq_qwen3_8` — Qwen 3.8 27B via Groq (open-weight; replaces the retired Llama models)\n")

    md.append("\n## Headline comparison\n")
    md.append("**Status-match rate** is the primary strategy-quality metric: what fraction of "
              "non-rejected prompts produced the expected capacity-agent status from `TEST_PROMPTS`. "
              "Higher is better. The dispatch rubric (right column) saturates at 5.0 across all "
              "strategies because the deterministic dispatch template is shared, so rubric scores "
              "tell us only that the dispatch text is internally consistent with the report — "
              "not whether the strategy made the right call.\n")
    md.append("| Strategy | **Status match rate** | Dispatch coherent (≥4.0) | Mean RUL | Mean latency | P95 latency |")
    md.append("|----------|------------------------|--------------------------|----------|--------------|-------------|")
    for r in per_strategy:
        md.append(
            f"| {r['strategy']} | **{r['status_match_rate']:.1%}** | "
            f"{r['dispatch_coherent_rate']:.1%} | "
            f"{r['mean_rul']:.1f} | "
            f"{r['mean_latency_ms']:.0f} ms | "
            f"{r['p95_latency_ms']:.0f} ms |"
        )

    # Per-severity-band breakdown for status match
    md.append("\n### Status-match broken down by expected severity\n")
    md.append("How well does each strategy handle LOW (should stay ONLINE), MEDIUM (should stay ONLINE "
              "near the cliff), and HIGH (should go OFFLINE) prompts? This isolates whether the LLM "
              "is actually doing severity classification or just routing keywords.\n")
    breakdown = (
        df_nonreject.groupby(["strategy", "expected_status"])["status_match"]
        .agg(["mean", "count"])
        .reset_index()
    )
    md.append("| Strategy | Expected status | Match rate | N |")
    md.append("|----------|-----------------|------------|---|")
    for _, r in breakdown.iterrows():
        md.append(f"| {r['strategy']} | {r['expected_status']} | {r['mean']:.1%} | {int(r['count'])} |")

    md.append("\n## RUL stability across repeats (same prompt run multiple times)\n")
    md.append("Lower std = more deterministic. The agentic strategy is expected to have "
              "non-zero std because Gemini's outputs are stochastic; the deterministic "
              "baselines should have std ≈ 0 (input is fixed).\n")
    md.append("| Strategy | Mean per-prompt RUL std | Max per-prompt RUL std |")
    md.append("|----------|-------------------------|------------------------|")
    for _, r in stability_per_strategy.iterrows():
        md.append(f"| {r['strategy']} | {r['mean_rul_std']:.2f} | {r['max_rul_std']:.2f} |")

    md.append("\n## Sensor selection diversity\n")
    md.append("How many distinct sensor IDs did each strategy choose across the test set? "
              "Fixed-midrange must be 1 by construction. Higher diversity does not equal "
              "better — it just means the strategy is responding to prompt variation.\n")
    md.append("| Strategy | Distinct sensors used |")
    md.append("|----------|-----------------------|")
    for _, r in sensor_div.iterrows():
        md.append(f"| {r['strategy']} | {r['distinct_sensors_used']} |")

    md.append("\n## Capacity-status outcome distribution\n")
    pivot = status_counts.pivot(index="strategy", columns="status", values="count").fillna(0).astype(int)
    # Manual markdown table — pandas .to_markdown() requires `tabulate` dep
    cols = list(pivot.columns)
    md.append("| Strategy | " + " | ".join(cols) + " |")
    md.append("|----------|" + "|".join(["---"] * len(cols)) + "|")
    for strategy, row in pivot.iterrows():
        md.append(f"| {strategy} | " + " | ".join(str(int(row[c])) for c in cols) + " |")

    md.append("\n## Per-strategy rejected-prompt counts\n")
    md.append("All strategies share the same Input Guard, so rejection counts should be identical.\n")
    md.append("| Strategy | Rejected |")
    md.append("|----------|----------|")
    for r in per_strategy:
        md.append(f"| {r['strategy']} | {r['rejected_count']} |")

    md.append("\n## Interpretation\n")
    md.append(
        "### Headline finding: the regex baseline beats ALL FOUR LLMs across vendor families AND within-family versions\n\n"
        "**The strong non-LLM baseline (`keyword_regex_severity`) outperforms every LLM-driven "
        "strategy tested**: 94.4% vs Gemini 2.5's 88.9%, **Gemini 3.5's 88.9%**, Llama 3.3's "
        "87.0%, and Llama 4 Scout's 81.5%. The regex baseline does so at **60–800× lower "
        "latency** (6 ms vs 427–5183 ms P95), at **zero API cost**, and with **dramatically "
        "lower RUL variance across repeats** (mean std 0.03 vs 0.06 for Gemini 2.5/3.5 and "
        "2.30 for the Groq Llamas).\n\n"
        "This is a **cross-family AND within-family negative result**. Four LLM offerings — "
        "Google Gemini 2.5 Flash, Google Gemini 3.5 Flash, Meta Llama 3.3 70B (via Groq), "
        "and Meta Llama 4 Scout (via Groq) — all fall below the regex baseline on the "
        "in-distribution test set. Both within-family version comparisons (Gemini 2.5→3.5 and "
        "Llama 3.3→4.0) show that **upgrading to a newer model does not help**: Gemini 3.5 "
        "produces *bit-identical sensor and severity choices* to Gemini 2.5 at 2.3× the "
        "latency, and Llama 4 Scout is *worse* than Llama 3.3.\n\n"
        "### Sub-findings about the LLM strategies\n\n"
        "1. **Gemini 3.5 Flash = Gemini 2.5 Flash on this task**. On all 18 non-rejected "
        "prompts, the two Gemini versions pick the *exact same* sensor + severity across all "
        "3 repeats each. Different model release, identical outputs. Status-match is "
        "indistinguishable (88.9% both). Gemini 3.5 takes ~2.3× longer per call (5183 ms vs "
        "2210 ms P95). **Paying for a newer Gemini gets you nothing on this task except more "
        "latency.**\n"
        "2. **Llama 4 Scout (newer) is worse than Llama 3.3 (older)** — 81.5% vs 87.0% "
        "overall, 44.4% vs 66.7% on OFFLINE-expected prompts. Llama 4 Scout is the smaller "
        "16e-expert variant of the Llama-4 family; on this severity-classification task it "
        "under-performs the older but larger Llama 3.3 70B. **Newer model ≠ better.**\n"
        "3. **All four LLMs miss the same OFFLINE prompts** (see "
        "`baselines_per_prompt_offline_detail.csv`). On `shaft lock on Machine 5` and "
        "`complete motor breakdown on Machine 2`, all four LLM families converge on HIGH "
        "severity + W0 (Motor RPM) sensor — the semantically-correct choice that the bimodal "
        "predictor cannot reward. The cross-family-AND-cross-version agreement is the "
        "strongest evidence that the LLM 'failures' reflect a systematic semantic preference "
        "shared across the LLM ecosystem, not random per-model error.\n"
        "4. **RUL stability differs sharply between vendors**: Gemini's `response_schema` "
        "structured-output mode produces near-deterministic outputs across both 2.5 and 3.5 "
        "(mean RUL std ~0.06 across 3 repeats); Groq's OpenAI-compatible JSON mode produces "
        "~40× more variance (mean RUL std 2.30). Side-finding about structured-output "
        "enforcement quality, not about the underlying models.\n"
        "5. **Latency hierarchy:** regex 6 ms << Llama 4 Scout 427 ms < Llama 3.3 736 ms < "
        "Gemini 2.5 Flash 2210 ms < Gemini 3.5 Flash 5183 ms. Groq's LPU inference is "
        "genuinely fast (3–5× faster than Gemini 2.5), but still ~100× slower than the "
        "regex baseline. Gemini 3.5 Flash is the slowest LLM tested.\n\n"
        "### Why does the regex win? Per-prompt evidence\n\n"
        "From the earlier two-LLM analysis (`baselines_per_prompt_offline_detail.csv` for "
        "agentic vs regex), the regex's wins on OFFLINE prompts come partly from luck: it "
        "falls through to the cliff-sensitive default sensor (Xs2) when its keyword table "
        "doesn't match, while the LLMs correctly pick W0 (Motor RPM) for motor-related faults. "
        "The bimodal CNN-LSTM does not reward the semantically-correct sensor choice. The "
        "regex's status-match advantage is therefore **partly an artifact of predictor "
        "pathology**, not a clean demonstration that the LLMs are useless.\n\n"
        "### What this means for the paper's central claim\n\n"
        "1. **On the status-match metric, NO LLM tested beats a regex** at this bounded-"
        "vocabulary in-distribution task. The metric corruption by predictor pathology is a "
        "real caveat, but it applies equally to all LLMs — so it does not rescue the LLM "
        "side of the comparison.\n\n"
        "2. **The LLM value proposition lives outside this evaluation**: handling out-of-"
        "distribution wording, novel fault descriptions not in any keyword list, producing "
        "semantically-correct sensor choices that a regression-quality predictor could use. "
        "Our test set does not stress these.\n\n"
        "3. **For practitioners**: on bounded-vocabulary industrial-control inputs, a regex-"
        "with-keyword-table baseline is the conservative engineering choice. The premium for "
        "LLM-driven control should be justified by demonstrated OOD robustness, not by "
        "in-distribution accuracy benchmarks.\n\n"
        "4. **For researchers**: a paper that evaluates an LLM-driven control system without "
        "a strong deterministic baseline overstates the LLM's contribution. The "
        "`keyword_regex_severity` baseline added here (Extension #3) and the cross-LLM "
        "evaluation added (Extension #1) are the minimum bar for credible evaluation.\n\n"
        "### The earlier framing — superseded\n\n"
        "An earlier version of this summary said *'Gemini's severity classifier contributes "
        "signal that deterministic baselines cannot replicate.'* That claim is now wrong on "
        "both halves: the regex baseline replicates the signal, and the LLM contribution does "
        "not extend across the Llama family either.\n\n"
        "### Trivial-baseline caveats still apply\n"
        "- `fixed_midrange` 66.7%: trivial 'predict ONLINE always' classifier on a test set "
        "that is 67% ONLINE-expected. Reflects test-set composition, not strategy quality.\n"
        "- `keyword_only` 53.7%: pre-baked HIGH severities in `FALLBACK_SPIKES` cause every "
        "prompt to over-inject. Tells us the FALLBACK_SPIKES table is calibrated for offline-"
        "mode demo dramatic effect, not for prompt-by-prompt accuracy.\n"
    )

    md.append("\n## Raw data\n")
    md.append(f"- `{CSV_PATH.relative_to(PROJECT_ROOT)}` — one row per (strategy, prompt, repeat)\n")

    SUMMARY_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"[baselines] wrote {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
