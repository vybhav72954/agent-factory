# ForgeMind Research Artifacts

This directory contains the research artifacts associated with the ForgeMind project. All scripts here are read-only consumers of the runtime pipeline (`agents/`, `dl_engine/`, `terminal/`); none of them modify it.

## Honest scoping (read this first)

An earlier version of this README claimed the work amounted to a finding about "categorical-to-continuous interface mismatch in LLM-driven control." An adversarial review found that claim does not survive scrutiny:

1. The borrowed CNN-LSTM checkpoint is **bimodal** — it outputs RUL values clustered near 1.2 or 70 across 95% of probe points, not a smooth continuous range. The "narrow DEGRADED basin" finding is a consequence of this bimodality, not evidence about LLM-driven control in general.
2. The non-LLM baselines (keyword-only, fixed-midrange) are weak by design and don't include the strongest reasonable non-LLM baseline (keyword + regex severity rules).
3. The HVAC second-domain demo uses a simulator deliberately constructed with the same severity-multiplier structure as the system under study, so any "failure-mode reproduction" in that demo is a property of the simulator's construction.
4. The original `related_work.md` was written from prior knowledge without verifying citation accuracy.

What the current artifacts **do** support (updated after simulator pivot, N=54 per strategy):

> ForgeMind is an end-to-end agentic-pipeline architecture for industrial fault simulation. We pivoted from a borrowed N-CMAPSS turbofan-trained CNN-LSTM (which exhibited bimodal output collapse on our inputs) to a **physics-informed factory simulator** that the model was retrained against; the simulator's 18 sensors actually correspond to the named factory-machine concepts (Bearing Temp, Motor RPM, Oil Pressure, etc.).
>
> We characterise the pipeline through probe sweeps, **seven-strategy baseline comparison** spanning four LLMs across two vendors with within-family version comparisons (Google Gemini 2.5 Flash, Google Gemini 3.5 Flash, Meta Llama 3.3 70B via Groq, Meta Llama 4 Scout via Groq) and three deterministic baselines including a **strong deterministic baseline** using keyword routing + regex severity classification.
>
> **Headline finding (cross-model, cross-LLM, N=54 per strategy):** the strong deterministic baseline (`keyword_regex_severity`) achieves **94.4% status-match** on both the turbofan checkpoint AND the simulator checkpoint — outperforming every LLM tested. On the **simulator-trained checkpoint**: agentic Gemini 2.5 88.9%, Gemini 3.5 88.9%, Llama 3.3 85.2%, Llama 4 Scout 79.6%. On the **turbofan checkpoint**: agentic Gemini 2.5 88.9%, Gemini 3.5 88.9%, Llama 3.3 87.0%, Llama 4 Scout 81.5%. **The negative result reproduces across two distinct predictor regimes** (bimodal turbofan + continuous simulator), ruling out the OOD-model critique. Per-prompt inspection shows all four LLM families converge on Motor RPM (W0) for motor-fault prompts — semantically correct — but the predictor does not reward W0-only injections without correlated Xs2/Xs3 activation.

This is a **mid-IF-journal-tier systems / engineering contribution** with a **cross-family, cross-model negative empirical result as the central finding**. The result is honestly scoped (specific to bounded-vocabulary industrial-fault prompts) but it is real, reproducible across LLM vendors AND across model variants, and it directly contradicts the "LLM-driven control adds operational value" assumption that motivates much current agentic-AI deployment in industry.

## Reading order

If you're reviewing or writing up this work, read the artifacts in this order. All findings are scoped to the current ForgeMind pipeline + the specific N-CMAPSS-trained CNN-LSTM checkpoint in `dl_engine/weights/`.

1. **`results/probe_cliff_3d_summary.md`** — the active probe summary (currently for the simulator checkpoint, continuous-output narrative). For the turbofan bimodality finding, see `probe_cliff_3d_summary_turbofan.md`. The summary script is variant-aware (HIGH-3 / CRIT-3 fix) and emits appropriate language depending on the loaded model.
2. **`evaluation_rubric.py`** — measurement infrastructure used by baselines and ablations.
3. **`results/baselines_summary.md`** — three-strategy comparison. Note the explicit caveat about the trivial-baseline issue.
4. **`results/ablation_summary.md`** — component contributions. V1 is explicitly marked as a no-op ablation.
5. **`domain2/hvac_summary.md`** — second-domain demo. The "failure-mode reproduction" claim is **retracted** in the current version of this file; the demo now supports only architectural-portability.
6. **`related_work.md`** — literature framing. **Citations are unverified** (flagged in the file itself) and should not be cited from in any submission without independent verification.
7. **`paper_draft.md`** — EAAI-style abstract + intro draft (current scope).
8. **`extensions_roadmap.md`** — seven extensions, prioritised, with estimated effort and where each result would appear in the paper.

## Reading `research/results/` — variant convention

After the simulator pivot (2026-05-25), every research artifact exists in several versions:

| Filename pattern | Contents | When updated |
|---|---|---|
| `<artifact>_turbofan.<ext>` | Snapshot from the **N-CMAPSS-trained** CNN-LSTM (original model) | Frozen — only updated if turbofan probe / baselines are re-run |
| `<artifact>_simulator.<ext>` | Snapshot from the **active** simulator-trained CNN-LSTM (v2 — the canonical paper-cited results) | Updated whenever simulator runs are re-executed under the v2 weights |
| `<artifact>_simulator_v1.<ext>` | Snapshot from the **v1** simulator model (severity p=[0.4, 0.4, 0.2] — pre-rebalance) | Frozen historical reference |
| `<artifact>_simulator_v3.<ext>` | Snapshot from the **v3** simulator model (fault_probability=0.75 — tested then rolled back; see BUG_REPORT MED-10) | Frozen for paper appendix; weights themselves were lost in the rollback |

For the paper, cite the explicitly-suffixed files (`_turbofan` / `_simulator`) so the variant is unambiguous. The `_simulator_v2.*` snapshots were removed in the 2026-05-26 cleanup because they were superseded by the reconciled `_simulator.*` files after the v2 baselines re-run; if you need the pre-reconciliation v2 state, recover from git history.

The production runtime (`python -m terminal.app`) still loads the **turbofan** weights by default (no `FORGEMIND_USE_SIMULATOR_MODEL` env var). To run the TUI against the simulator-trained model: set the env var first.

## File map

```
research/
├── README.md                          # this file
├── probe_cliff_3d.py                  # 3D RUL surface sweep
├── evaluation_rubric.py               # structural + LLM-judge rubric
├── baselines.py                       # strategy comparison
├── ablation.py                        # component ablations
├── related_work.md                    # literature review (UNVERIFIED CITATIONS)
├── paper_draft.md                     # EAAI abstract + intro draft
├── extensions_roadmap.md              # 7 extensions to strengthen the paper
├── domain2/
│   ├── hvac_demo.py                   # second-domain instantiation
│   └── hvac_summary.md                # findings writeup (claims retracted from v1)
└── results/                                       # generated artifacts (variant-suffixed)
    ├── probe_cliff_3d_{turbofan,simulator}.csv    # 3375 rows each (15³ sweep)
    ├── probe_cliff_3d_summary_{turbofan,simulator}.md
    ├── probe_cliff_3d_xs4_slices_{turbofan,simulator}.png
    ├── probe_cliff_3d_xs2_curves_{turbofan,simulator}.png
    ├── baselines_comparison_{turbofan,simulator}.csv      # 60 rows per strategy
    ├── baselines_summary_{turbofan,simulator}.md
    ├── baselines_per_prompt_offline_detail_{turbofan,simulator}.csv
    ├── ablation_results_{turbofan,simulator}.csv
    ├── ablation_summary_{turbofan,simulator}.md
    └── domain3/                                    # Bayesian anomaly second domain
        ├── bayesian_results.csv
        └── bayesian_summary.md
research/paper_assets/                              # curated, paper-ready
├── MANIFEST.md                                     # paper-asset index + cross-refs
├── figures/                                        # 6 publication-quality figures
│   ├── fig1_strategy_comparison.png
│   ├── fig2_probe_heatmap_compare.png
│   ├── fig3_sequence_walkthrough.png
│   ├── fig4_latency_vs_accuracy.png
│   ├── fig5_per_prompt_match_matrix.png
│   └── fig6_sensor_convergence.png
└── tables/                                         # 3 paper tables as standalone MD
    ├── table1_turbofan_results.md
    ├── table2_simulator_results.md
    └── table3_bayesian_domain.md
```

## How to reproduce all artifacts from scratch

From the project root. **All three main experiment scripts auto-snapshot their
outputs with a model-variant suffix** (e.g. `probe_cliff_3d.py` writes
`probe_cliff_3d_simulator.csv` when the simulator model is loaded and
`probe_cliff_3d_turbofan.csv` otherwise). The variant convention is enforced
by `research/__init__.py:write_variant_snapshot`. Default-named (no-suffix)
files were removed in the 2026-05-26 cleanup since they only duplicated the
active-variant snapshot and were a source of confusion.

### On the turbofan model (default — no env var)
```powershell
python -m research.probe_cliff_3d --steps 15        # ~30s, no LLM
python -m research.baselines --stability-runs 3     # ~10 min, uses LLMs
python -m research.ablation                          # ~3 min, uses Gemini
```
→ produces `*_turbofan.*` snapshots.

### On the simulator model
```powershell
$env:FORGEMIND_USE_SIMULATOR_MODEL = "1"
python -m research.probe_cliff_3d --steps 15
python -m research.baselines --stability-runs 3
python -m research.ablation
Remove-Item Env:\FORGEMIND_USE_SIMULATOR_MODEL    # reset so production sees turbofan
```
→ produces `*_simulator.*` snapshots.

### Auxiliary
```powershell
python -m research.evaluation_rubric                # instant, rubric self-test
python -m research.domain2.hvac_demo                # ~30s, uses Gemini
```

Without a Gemini API key the probe, rubric self-test, and the `keyword_only` /
`fixed_midrange` / `keyword_regex_severity` strategies still run.

### Retraining the simulator model
```powershell
python -m research.simulator.generate_training_data --n-lifecycles 1000
python -m research.simulator.train_simulator_model --epochs 25 --device cuda
```
Output: `dl_engine/weights/best_model_simulator.pt` + `scaler_simulator.pkl`.
Existing `_v1.pt.bak` files preserve the previous training run for comparison.

## Paper-section mapping (current scope — EAAI tier)

| Paper section | Source artifact |
|---|---|
| Abstract / contribution | `paper_draft.md` §Abstract + §1.2 |
| Introduction / motivation | `paper_draft.md` §1.1–§1.4 |
| Related work | `related_work.md` **after citation verification** |
| System architecture | `../CLAUDE.md` §3–4 + `../audit_and_demo_guide.md` |
| Evaluation methodology | `evaluation_rubric.py` docstring + `TEST_PROMPTS` |
| **§5 Findings — Before/After Methodological Validation** | **`paper_draft.md` §5 (Phase 5 deliverable)** — the central empirical section, paper-ready |
| §5.1 Initial turbofan-checkpoint result | `results/baselines_summary_turbofan.md` + Table 1 in paper_draft.md §5.1 |
| §5.2 Methodological critique + probe | `results/probe_cliff_3d_summary_turbofan.md` |
| §5.3 Simulator pivot + retrain | `simulator/` directory + `factory_simulator.py` module docstring |
| §5.4 Re-validated simulator result | `results/baselines_summary_simulator.md` + Table 2 in paper_draft.md §5.4 |
| §5.5 What changed / what didn't | `results/baselines_per_prompt_offline_detail_*.csv` (per-prompt comparison) |
| §5.7 Methodological lessons | derived from `BUG_REPORT_2026-05-25.md` MED-10 finding |
| Architectural portability | `domain2/hvac_summary.md` (architecture-only claim) |
| Limitations + future work | `extensions_roadmap.md` |

## What's NOT in here (and shouldn't be)

- Runtime code (`agents/`, `dl_engine/`, `terminal/`) — system under study
- Textual TUI — irrelevant to the paper, useful for the demo
- Test suite (`tests/`) — verifies runtime correctness, not measured by the paper
- `audit_and_demo_guide.md` — demo preparation, not research output

## Limitations of the current research artifacts (named explicitly so reviewers don't have to)

- **N=18 non-rejected prompts** is small. Any aggregate metric carries wide uncertainty intervals that we do not currently compute (Extension #4).
- ~~**One LLM (Gemini 2.5 Flash)**. No evidence that findings transfer to other LLM families (Extension #1).~~ **RESOLVED 2026-05-24**: cross-family evaluation completed with Llama 3.3 70B and Llama 4 Scout via Groq. All three LLMs lose to the regex baseline.
- **One predictor checkpoint** (the borrowed CNN-LSTM), and it is bimodal. Findings may not transfer to a properly-calibrated regression model (Extension #6).
- **No human evaluation**. The rubric is structural; "operational coherence" is not measured by human raters (Extension #4).
- ~~**No mitigation evaluated**. We propose mitigations but do not implement and measure any (Extension #2).~~ **RESOLVED 2026-05-26**: continuous-output mitigation evaluated as `agentic_continuous` strategy. Gemini emits `severity_multiplier ∈ [0,1]` directly via `response_schema`. Recovers nothing — the categorical interface is not the bottleneck (paper §5.6.1).
- ~~**Second-domain simulator is constructed to mirror the first**. Architectural-portability claim only (Extension #5).~~ **RESOLVED 2026-05-26**: built `research/domain3/` — Bayesian anomaly classifier with LLM-emitted `PriorUpdate` over 5 failure modes. Qualitatively different on all four axes (output type, evaluation metric, LLM role, predictor class). The headline finding generalises (paper §5.6.3).
- **Related-work citations are unverified** (Extension #7).
- **Simulator not validated against real PdM data** (Extension #8). The physics couplings in `research/simulator/factory_simulator.py` are physically defensible (cited in the module docstring) but not empirically calibrated. The work is therefore a synthetic study by construction. Validation against PRONOSTIA bearings or similar real-equipment data is listed as Extension #8.
