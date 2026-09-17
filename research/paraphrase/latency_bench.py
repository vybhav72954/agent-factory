"""
research/paraphrase/latency_bench.py

Idle-machine latency benchmark for the local Ollama arms.

Not used for the 2026-09-17 results: local_latency_benchmark.csv was instead built from the
design- and typo-bank calls, which ran with no other job on the machine (see
research/AEI/analysis_plan_additions.md, section B). Kept for re-measurement on other hardware.

Local models share the GPU and CPU with everything else on the machine, and they
were collected while other jobs ran, so their collection latencies are not clean.
This script times each local arm on the 90 anchored-bank prompts, once each,
after one untimed warm-up call, with nothing else running. score.py uses these
latencies for the local arms when the output file exists.

Usage:
    python -m research.paraphrase.latency_bench        # set OLLAMA_HOST if the server is not on the default port

Output:
    research/results/paraphrase/local_latency_benchmark.csv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.paraphrase.anchored_bank import ANCHORED_PROMPTS
from research.paraphrase.collect import LOCAL_ARMS, _call

OUT_CSV = PROJECT_ROOT / "research" / "results" / "paraphrase" / "local_latency_benchmark.csv"


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    rows = []
    gpu = gpu_name()
    for arm in LOCAL_ARMS:
        _call(arm, ANCHORED_PROMPTS[0]["text"])              # warm-up: load the model, not timed
        for p in ANCHORED_PROMPTS:
            spike, latency, usage = _call(arm, p["text"])
            rows.append({"arm": arm, "bank_id": p["id"], "latency_ms": latency,
                         "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                         "fallback": "-FALLBACK" in spike.plain_english_summary, "gpu": gpu})
        done = pd.DataFrame(rows)
        lat = done[done["arm"] == arm]["latency_ms"]
        print(f"[latency_bench] {arm}: P50 {lat.median():.0f} ms, P95 {lat.quantile(0.95):.0f} ms", flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[latency_bench] wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
