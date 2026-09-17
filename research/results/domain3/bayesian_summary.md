# Domain 3 — Bayesian Anomaly Classification

**Domain:** Industrial anomaly classifier with LLM-emitted prior updates.

**Predictor:** Frozen Gaussian Naive Bayes trained on synthetic per-mode multivariate Gaussian observations.

**LLM role:** emits a probability shift over five failure-mode priors from operator natural-language descriptions. The shift is combined with the classifier's likelihood via standard Bayes update.

**Test set:** 20 operator descriptions balanced across 5 modes (4 per mode).

**Per-mode label balance:** {'bearing': 4, 'motor': 4, 'oil_seal': 4, 'coolant': 4, 'electrical': 4}

**Metrics:** Brier score (lower=better, perfect=0, worst=2), NLL (lower=better), top-1 accuracy (higher=better).


## Headline results

| Strategy | Top-1 acc | Brier (mean) | NLL (mean) | Latency P95 | N | Errors |
|----------|-----------|--------------|------------|-------------|---|--------|
| `uniform` | 38.3% | 0.685 ± 0.308 | 1.323 ± 0.679 | 0.1 ms | 60 | 0 |
| `keyword_only` | 91.7% | 0.267 ± 0.251 | 0.576 ± 0.436 | 1.0 ms | 60 | 0 |
| `keyword_regex` | 88.3% | 0.292 ± 0.270 | 0.623 ± 0.497 | 0.7 ms | 60 | 0 |
| `agentic` | 88.3% | 0.279 ± 0.268 | 0.600 ± 0.496 | 3939.9 ms | 60 | 0 |

## Interpretation

The uniform-prior baseline (likelihood-only, no operator shift) lands at Brier 0.685, top-1 38.3%. The classifier alone is not enough — operator information is load-bearing.

The deterministic strategies recover most of the gap: `keyword_only` reaches top-1 91.7% (Brier 0.267), `keyword_regex` reaches top-1 88.3% (Brier 0.292). The agentic Gemini call lands at top-1 88.3% (Brier 0.279) at a 3940 ms P95 cost.


**Verdict:** the LLM ties the deterministic baselines on both metrics. The main-pipeline finding (the regex baseline is competitive with the LLM) generalises to this qualitatively different domain — output type, evaluation metric, and predictor are all different here, yet the bottom-line ordering is the same.


**Numerical detail:** agentic vs keyword_only — top-1 Δ = -3.3%, Brier Δ = +0.011. agentic vs keyword_regex — top-1 Δ = +0.0%, Brier Δ = -0.014. The LLM's latency cost vs deterministic strategies is 3940ms P95 vs 1.0ms P95 — a 3767x slowdown.
