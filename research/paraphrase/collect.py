"""
research/paraphrase/collect.py

Collect LLM diagnostic outputs for a paraphrase prompt bank.

Banks (`--bank`):
    design    prompt_bank.py: 42 prompts with design severities
    anchored  anchored_bank.py: the 18 TEST_PROMPTS anchors + 4 rewrites each (90 prompts)
    typo      typo_bank.py: the 18 anchors with seeded keyboard typos (105 prompts, no hand-written text)

Arms = 4 models x 2 system prompts:
    models:  gemini-2.5-flash, gemini-3.5-flash (Google), gpt-oss-120b, Qwen 3.8 27B (Groq)
    prompts: production (agents/prompts.py) and semantic (prompts_semantic.py)

The paper's Llama 3.3 70B and Llama 4 Scout were retired on Groq by 2026-09,
so the two open-weight slots use gpt-oss-120b and Qwen 3.8 27B.

Local arms (production prompt only, added 2026-09-17, analysis_plan_additions.md
section B): Llama 3.2 3B, Gemma 3 4B and Qwen 3 4B (thinking off) served by
Ollama on the local GPU (`--provider ollama`; set OLLAMA_HOST if the server is
not on the default port). Each local model gets one untimed warm-up call first
so model loading is not counted as latency.

All Gemini arms use `research.baselines._gemini_diagnose` and all Groq arms use
`_groq_diagnose`, at temperature 0, so the arms differ only in model and
system prompt. (The published `agentic` strategy went through
`translate_fault_to_tensor`; that path uses the same model and prompt but adds
production retry logic, so it is not repeated here.)

Outputs are predictor-independent: only sensor_id, fault_severity and
spike_value drive injection, so `score.py` replays them on any checkpoint
without new API calls. Token usage (input, and output including
thinking/reasoning tokens) is recorded per call for the cost analysis.

Failed calls are retried with backoff. If every attempt falls back, the row is
kept with `fallback=True` (the same honest-signal policy as baselines.py) and
reported by score.py.

`--part NAME` writes to a separate file so providers can run in parallel
processes without overwriting each other; score.py reads every part.

Usage:
    python -m research.paraphrase.collect                                   # design bank, resumable
    python -m research.paraphrase.collect --bank anchored --provider gemini --part gemini
    python -m research.paraphrase.collect --bank anchored --provider groq --part groq
    python -m research.paraphrase.collect --bank typo --provider ollama --part local
    python -m research.paraphrase.collect --limit 2                         # smoke test: first 2 prompts, 1 repeat
    python -m research.paraphrase.collect --dry-run

Output:
    research/results/paraphrase/llm_outputs[_anchored][_<part>].csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.baselines import (
    GEMINI_3_5_FLASH_MODEL,
    GROQ_GPT_OSS_120B_MODEL,
    GROQ_QWEN3_8_MODEL,
    OLLAMA_MODELS,
    _gemini_diagnose,
    _groq_diagnose,
    _ollama_diagnose,
)
from research.paraphrase.prompts_semantic import DIAGNOSTIC_SEMANTIC_SYSTEM_PROMPT

OUT_DIR = PROJECT_ROOT / "research" / "results" / "paraphrase"
OUT_STEMS = {"design": "llm_outputs", "anchored": "llm_outputs_anchored", "typo": "llm_outputs_typo",
             "fmucd": "llm_outputs_fmucd", "fmucd_fewshot": "llm_outputs_fmucd_fewshot"}
OUT_CSV = OUT_DIR / "llm_outputs.csv"          # design bank, single-file default

MODELS: dict[str, tuple[str, str]] = {
    "gemini_2_5_flash": ("gemini", "gemini-2.5-flash"),
    "gemini_3_5_flash": ("gemini", GEMINI_3_5_FLASH_MODEL),
    "gpt_oss_120b":     ("groq", GROQ_GPT_OSS_120B_MODEL),
    "qwen3_8_27b":      ("groq", GROQ_QWEN3_8_MODEL),
}
PROMPT_VARIANTS: dict[str, str | None] = {
    "production": None,                                  # None -> DIAGNOSTIC_SYSTEM_PROMPT
    "semantic":   DIAGNOSTIC_SEMANTIC_SYSTEM_PROMPT,
}
ARMS = [f"{m}__{v}" for m in MODELS for v in PROMPT_VARIANTS]

# Gemini 2.5 Flash as the production agent calls it: thinking disabled
# (agents/diagnostic_agent.py sets thinking_budget=0). Production prompt only.
# Added 2026-09-17 after verification found the default-thinking arms differ from production.
MODELS["gemini_2_5_flash_nothink"] = ("gemini", "gemini-2.5-flash")
MODEL_KWARGS: dict[str, dict] = {"gemini_2_5_flash_nothink": {"thinking_budget": 0}}
ARMS.append("gemini_2_5_flash_nothink__production")
HOSTED_ARMS = list(ARMS)

# Small open models run locally through Ollama (production prompt only).
for _key, (_tag, _request) in OLLAMA_MODELS.items():
    MODELS[_key] = ("ollama", _tag)
    MODEL_KWARGS[_key] = _request
LOCAL_ARMS = [f"{key}__production" for key in OLLAMA_MODELS]
ARMS.extend(LOCAL_ARMS)

# Minimum seconds between calls per provider (Groq free tier is ~30 requests/min).
MIN_INTERVAL_S = {"gemini": 0.5, "groq": 2.2, "ollama": 0.0}
MAX_ATTEMPTS = 4
CHECKPOINT_EVERY = 50
FALLBACK_TAGS = ("-FALLBACK", "-UNAVAILABLE")
# Errors that retrying cannot fix (retired model id, bad key): abort instead of backing off.
FATAL_ERRORS = ("NotFoundError", "AuthenticationError", "PermissionDeniedError")


def load_bank(bank: str) -> tuple[list[dict], list[str]]:
    """(prompts, problems) for a bank name."""
    if bank == "design":
        from research.paraphrase.prompt_bank import PROMPTS, check_prompt_bank
        return PROMPTS, check_prompt_bank()
    if bank == "typo":
        from research.paraphrase.typo_bank import TYPO_PROMPTS, check_typo_bank
        return TYPO_PROMPTS, check_typo_bank()
    if bank.startswith("fmucd"):
        from research.fmucd import check_sample, load_sample          # real work orders, external check
        few = bank.endswith("fewshot")                                # few-shot: per-site examples in the system prompt
        problems = check_sample(fewshot=few)
        return ([] if problems else load_sample(fewshot=few)), problems
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS, check_anchored_bank
    return ANCHORED_PROMPTS, check_anchored_bank()


def output_path(bank: str, part: str | None) -> Path:
    return OUT_DIR / (f"{OUT_STEMS[bank]}_{part}.csv" if part else f"{OUT_STEMS[bank]}.csv")


def load_outputs(bank: str) -> pd.DataFrame:
    """All collected rows for a bank, across parts."""
    stem = OUT_STEMS[bank]
    files = sorted(OUT_DIR.glob(f"{stem}.csv")) + sorted(OUT_DIR.glob(f"{stem}_*.csv"))
    # A longer stem's files match this stem's glob (llm_outputs -> llm_outputs_anchored...,
    # llm_outputs_fmucd -> llm_outputs_fmucd_fewshot...), so drop every other bank's files.
    longer = [v for k, v in OUT_STEMS.items() if v != stem and v.startswith(stem)]
    files = [f for f in files
             if not any(f.name == o + ".csv" or f.name.startswith(o + "_") for o in longer)]
    if not files:
        raise FileNotFoundError(f"no collected outputs for bank {bank!r} in {OUT_DIR}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _call(arm: str, text: str, system_prompt_override: str | None = None):
    """One diagnostic call for an arm. Returns (SensorSpike, latency_ms, usage dict).

    `system_prompt_override` lets a bank attach a per-prompt system prompt (the FMUCD few-shot bank
    appends each site's own labelled examples to the production prompt).
    """
    model_key, variant = arm.split("__")
    provider, model_id = MODELS[model_key]
    system_prompt = system_prompt_override if system_prompt_override is not None else PROMPT_VARIANTS[variant]
    usage: dict = {}
    t0 = time.time()
    if provider == "gemini":
        spike = _gemini_diagnose(model_id, text, system_prompt=system_prompt, usage_out=usage,
                                 **MODEL_KWARGS.get(model_key, {}))
    elif provider == "ollama":
        spike = _ollama_diagnose(model_id, text, system_prompt=system_prompt, usage_out=usage,
                                 **MODEL_KWARGS.get(model_key, {}))
    else:
        spike = _groq_diagnose(model_id, text, system_prompt=system_prompt, usage_out=usage)
    return spike, (time.time() - t0) * 1000, usage


def _write(rows: list[dict], existing: pd.DataFrame, path: Path) -> int:
    new = pd.DataFrame(rows)
    if not existing.empty:
        keys = set(zip(new["arm"], new["bank_id"], new["repeat"]))
        keep = [k not in keys for k in zip(existing["arm"], existing["bank_id"], existing["repeat"])]
        new = pd.concat([existing[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", choices=list(OUT_STEMS), default="design")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arms", nargs="*", default=None, choices=ARMS)
    parser.add_argument("--provider", choices=["gemini", "groq", "ollama"], default=None,
                        help="Only arms served by this provider")
    parser.add_argument("--part", default=None, help="Write to a separate part file (for parallel runs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only the first N prompts with 1 repeat")
    parser.add_argument("--replicate-to", type=int, default=None,
                        help="With --repeats 1: store each output as this many repeat rows, flagged replicated=True "
                             "(used for the local models on the design and typo banks after their anchored-bank "
                             "repeats proved deterministic)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = args.arms or [a for a in ARMS if args.provider in (None, MODELS[a.split("__")[0]][0])]
    prompts, problems = load_bank(args.bank)
    if problems:
        print(f"[collect] {args.bank} bank has problems: {problems}")
        sys.exit(1)
    prompts = prompts[: args.limit] if args.limit else prompts
    repeats = 1 if args.limit else args.repeats

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path(args.bank, args.part)
    done: set[tuple[str, str, int]] = set()
    existing = pd.DataFrame()
    if path.exists():
        existing = pd.read_csv(path)
        ok = existing[~existing["fallback"].astype(bool)]
        done = set(zip(ok["arm"], ok["bank_id"], ok["repeat"].astype(int)))

    todo = [(arm, p, r) for arm in arms for p in prompts for r in range(repeats)
            if (arm, p["id"], r) not in done]
    print(f"[collect] {args.bank} bank -> {path.name}: {len(todo)} calls to make ({len(done)} already collected)")
    if args.dry_run or not todo:
        return

    last_call = {"gemini": 0.0, "groq": 0.0, "ollama": 0.0}
    warmed: set[str] = set()
    rows: list[dict] = []
    t_start = time.time()
    try:
        for i, (arm, p, r) in enumerate(todo, start=1):
            provider = MODELS[arm.split("__")[0]][0]
            if provider == "ollama" and arm not in warmed:
                _call(arm, p["text"], p.get("system_prompt"))   # load the model into memory; not timed or recorded
                warmed.add(arm)
            for attempt in range(1, MAX_ATTEMPTS + 1):
                wait = MIN_INTERVAL_S[provider] - (time.time() - last_call[provider])
                if wait > 0:
                    time.sleep(wait)
                spike, latency, usage = _call(arm, p["text"], p.get("system_prompt"))
                last_call[provider] = time.time()
                summary = spike.plain_english_summary
                fallback = any(tag in summary for tag in FALLBACK_TAGS)
                if "-UNAVAILABLE" in summary:
                    raise SystemExit(f"[collect] {provider} client unavailable (missing API key or SDK); aborting")
                fatal = next((e for e in FATAL_ERRORS if e in summary), None)
                if fatal:
                    raise SystemExit(f"[collect] {arm}: {fatal} from {provider}; check the model id and key. Aborting")
                if not fallback:
                    break
                time.sleep(2 ** attempt)
            row = {
                "arm": arm, "bank_id": p["id"], "prompt": p["text"], "repeat": r,
                "sensor_id": spike.sensor_id, "severity": spike.fault_severity.value,
                "spike_value": spike.spike_value, "summary": summary,
                "fallback": fallback, "attempts": attempt, "latency_ms": latency,
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "replicated": False,
            }
            rows.append(row)
            if args.replicate_to and repeats == 1:
                rows += [{**row, "repeat": k, "replicated": True} for k in range(1, args.replicate_to)]
            if i % CHECKPOINT_EVERY == 0 or i == len(todo):
                n = _write(rows, existing, path)
                print(f"  {i}/{len(todo)}  {time.time() - t_start:.0f}s  (saved {n} rows)", flush=True)
    finally:
        if rows:
            n = _write(rows, existing, path)
            print(f"[collect] wrote {path.relative_to(PROJECT_ROOT)} ({n} rows)")


if __name__ == "__main__":
    main()
