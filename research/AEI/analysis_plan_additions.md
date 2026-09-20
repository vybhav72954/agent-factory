# Analysis plan — additions round (fixed 2026-09-17, before any new result was produced)

Written before collecting or scoring anything below, so the paper can state that designs, references,
hypothesis families and expectations were set in advance. No human raters are used anywhere; every
label is inherited from TEST_PROMPTS or the design bank, as before.

## Why

Pre-submission review (2026-09-17) named four likely reviewer objections: (1) the fault-injection setting
looks artificial; (2) the embedding baseline is untrained; (3) the rewrites are author-written;
(4) "LLMs understand paraphrase better" is unsurprising. The additions below target (2), (3) and part of (1).

## A. Trained non-LLM severity classifiers (`research/trained_baselines.py`)

All share the regex's keyword sensor routing and band-centre spike values, so only severity differs.

**Training regimes**

- **R1, prompt knowledge.** Synthetic sentences generated only from vocabulary the production prompt gives
  the LLM: the severity term lists in `baselines.py` (HIGH / LOW / MEDIUM examples), the left-hand fault phrases of
  the prompt's FAULT → SENSOR table, and machine numbers 1–5. One sentence per (severity term, fault phrase)
  pair with a seeded choice of template and machine, plus bare fault phrases (label MEDIUM unless the table
  annotates the line "(HIGH severity)"). No test prompt is used. Usable on every test set.
- **R2, in-domain labelled examples.** R1 plus labelled prompts from both paraphrase banks, excluding every
  prompt in the test prompt's cluster: leave-one-anchor-out on the anchored bank (training adds the design bank
  and the other 17 anchors), leave-one-prompt-out on the design bank (training adds the anchored bank and the other
  41 design prompts). In-domain examples are weighted so both sources carry equal total weight. This asks how much a
  small set of labelled operator reports substitutes for an LLM.

**Models (hyperparameters fixed now, no tuning on test data)**

| Arm | Features | Classifier |
|---|---|---|
| `tfidf_logreg` | TF-IDF word 1–2-grams + character 2–5-grams | logistic regression, C = 1, balanced classes |
| `minilm_logreg` | all-MiniLM-L6-v2 sentence embeddings (normalised) | same |
| `bge_m3_logreg` | BAAI/bge-m3 dense embeddings (multilingual, normalised) | same |
| `minilm_finetuned` | all-MiniLM-L6-v2 fine-tuned end to end, mean pooling + linear head | class-balanced cross-entropy, 4 epochs, AdamW lr 3e-5 (encoder) / 1e-3 (head), batch 32, seed 0; R1 trained on CPU, the 60 R2 folds on the GPU (CPU was too slow alongside collection); prediction on CPU |

**Pre-designated reference:** `bge_m3_logreg` (the multilingual model, chosen because two rewrite styles are
code-mixed or shorthand). The others are reported with CIs but are not in the hypothesis family.
R1 models are also added to the 18-prompt baseline runs, multiplier sweeps and multi-hit runs (deterministic, no calls).

## B. Local small open LLMs (Ollama, GTX 1650 4 GB, quantised)

`llama3.2:3b`, `gemma3:4b`, `qwen3:4b` (thinking off). (`qwen3.5:4b` was the first choice but needs a newer Ollama than the installed 0.14.2; substituted before any call.) Production system
prompt, JSON-schema constrained output, temperature 0, 3 repeats, same fallback policy as the API arms.
**Change made during collection (2026-09-17, before any local result was scored):** on the anchored bank the local models proved deterministic (severity identical in all 3 repeats for every prompt; all fields identical for 100% of Gemma 3 and 96.7% of Llama 3.2 prompts at the time of the decision; the completed anchored run confirmed severity identical for all 90 prompts of all three models, all fields for 100% / 96.7% / 96.7% of Gemma 3 / Llama 3.2 / Qwen 3), and Gemma 3 does not fit in 4 GB (about 36 s per call with CPU offload). So the anchored bank keeps 3 real repeats, and on the design and typo banks each local model is called once per prompt and the output is stored as 3 repeat rows flagged `replicated=True`. Cluster means, and therefore all paired tests, are unaffected by the replication.
Latency is measured on this GPU; API cost is zero (hardware reported instead). Question: does the
advantage require a hosted frontier-scale model?
Scope: design, anchored and typo banks. Their in-vocabulary result comes from the anchored bank's 18 originals on all
four checkpoints; they are not added to the multiplier sweeps or multi-hit runs, which stay defined on the hosted LLMs
(compute: about 8 s per local call on this GPU). Latency: planned as a separate idle-machine benchmark (`latency_bench.py`). Replaced before scoring by the non-replicated calls of the design and typo bank collection (147 calls per model, 20:02–21:25 on 2026-09-17), which ran with no other job on the machine; anchored-bank collection latencies are not used because other jobs shared the CPU and GPU.

## C. Common-interface decomposition (replay only, no new calls)

Every arm's severity is re-injected through the shared keyword routing and band-centre spike values, and status
match is compared with the arm's own routing and spike value. The difference isolates what sensor routing and
spike-value choice contribute on each checkpoint.

**Expectation recorded now:** on the simulator checkpoint the LLMs' status-match deficit against the embedding
classifier on rewrites (80–85% vs 86%) disappears or reverses under the common interface, because the deficit comes
from W0 routing rather than severity understanding. On turbofan and the calibrated simulator the LLM advantage stays.

## D. Programmatic typo set (not author-written)

Each of the 18 anchors' original wording is perturbed by a seeded keyboard-typo model: of the n alphabetic words
with 3 or more letters (excluding "Machine"), k = max(1, round(p·n)) are chosen uniformly without replacement and each
gets one edit chosen uniformly from adjacent-key substitution, deletion, adjacent-key insertion and transposition.
Two noise levels (p = 0.25, 0.50) × 3 variants = 108 prompts, labels inherited from the anchor; a variant identical to
an earlier one for the same anchor and level is redrawn with the next seed. Variants rejected by the input guard are
dropped (count reported). No text is hand-edited. (Revised before generation: a per-word probability left many short
prompts unchanged.) Arms: the five hosted production-prompt configurations, the three local models, and all non-LLM
classifiers (R1 only).

## E. Hypothesis families (sign-flip test over anchors, exact when ≤ 20 non-zero clusters; Holm within family)

- **H1 (unchanged, already reported):** production-prompt hosted LLMs vs regex and vs prototype embedding,
  severity accuracy on anchored rewrites (10 comparisons).
- **H2 (new):** every production-prompt LLM (5 hosted + 3 local = 8) vs `bge_m3_logreg` R1 and vs
  `bge_m3_logreg` R2, severity accuracy on anchored rewrites (16 comparisons). Replicated on the design bank.
- **H3 (new):** the same 8 LLM configurations vs regex and vs `bge_m3_logreg` R1, severity accuracy on the typo set
  (16 comparisons).

Everything else (other classifiers, status match, style breakdowns, decomposition) is reported with 95% cluster
bootstrap CIs and no family claim.

## F. Framing (writing stage)

Severity triage of free-text operator reports is the core task. The prognostic pipeline is the setting in which its
consequences are measured, presented as scenario ("what-if") testing on a digital twin: an operator report seeds a
fault scenario, and the question is whether the maintenance decision that follows is right.

## G. External check on real maintenance work orders (FMUCD), added 2026-09-18

**Source.** Facility Management Unified Classification Database, Mendeley Data doi:10.17632/cb8d2nsjss.1 — 3,693,867 work orders from 12 North American universities, with a free-text `WODescription` and a `WOPriority` code. Licence to confirm at download (the Mendeley page says CC BY 4.0, the Data in Brief article says CC BY-NC); the data is not redistributed, only the sampled work-order ids and our own outputs.

**Why.** It is the only public corpus found (2026-09-18 search) that pairs real maintenance free text with a priority assigned by the organisation itself. It tests whether the ranking measured on the author-written banks survives on text nobody here wrote. MaintIE and MaintNet have no severity label; MSHA narratives and FAA Service Difficulty Reports carry severity-like fields but for injuries and aircraft defects.

**Stage 1 — mapping (labels only, before any strategy is run).** Inspect `WOPriority` per university together with work-order type and description length, using label metadata only, never model outputs. Write a mapping from each university's codes to LOW / MEDIUM / HIGH into `research/external/fmucd_mapping.json` with a one-line rationale per university, and drop universities whose codes cannot be ordered. The mapping is frozen before stage 2.

**Stage 2 — sample and score.** Unplanned work orders only; description non-empty, 3–200 characters, passing the production input guard (the pipeline accepts machine-fault text only — report the pass rate); duplicate descriptions removed; then a seeded stratified sample of about 300, balanced across the three mapped bands and spread across universities.

**Arms.** The five hosted production-prompt configurations, the three local models, the rule-based baselines and the R1 trained classifiers, plus **R3**: the trained classifiers refitted on the R1 sentences plus all 132 labelled bank prompts (no fold needed — FMUCD is a different source). R3 asks whether labelled reports from one site transfer to another organisation's work orders. One call per prompt (hosted arms agreed across repeats on 96–100% of prompts on the earlier banks; the local models were deterministic).

**Metrics.** Three-class agreement with the mapped band, quadratic-weighted kappa, confusion matrix, bootstrap CIs and paired sign-flip differences clustered by university (12 clusters) and by work order. No status replay: the sensor routing has no meaning for building-maintenance text.

**Expectation recorded now.** Agreement will be lower than on our banks for every arm, because priority encodes scheduling policy as much as severity. The ranking hosted LLMs > trained classifiers > regex should survive, and the R3 classifiers should lose part of the advantage that in-domain labels gave them, because their labelled examples come from a different site.

**Reporting.** Supplementary section with the label noise stated openly; no claim that recorded priority equals fault severity.

**Stage 3, added after stage 2 (2026-09-18).** Every arm landed within 28–38% on a balanced three-class sample, so the question became whether the recorded band is predictable from the text at all. Classifiers were trained on FMUCD's own labels (6,000 mapped work orders disjoint from the sample, same universities and filters) and tested on the same 300: TF-IDF 68.7% (quadratic kappa 0.60), bge-m3 61.3% (0.54). This step was added in response to the near-chance result, and is reported as such; it uses no model output from stage 2.

## H. Few-shot in-context learning on the work orders (added 2026-09-19, before collection)

**Question.** The in-house-trained classifier reaches 68.7% on a site's own priority band while every zero-shot strategy sits near chance. Does an LLM close that gap when it is given some of the same site's labels in the prompt, instead of training on thousands of them?

**Design.** For each of the four universities, 20 labelled work orders are drawn (seeded, balanced across that site's mapped bands) from the mapped pool, excluding the 300 evaluated ones. They are appended to the production system prompt as a labelled example block, so each test work order sees examples from its own site only. Everything else is unchanged: same 300 work orders, same arms (five hosted production configurations and three local models), temperature 0, one call per work order, same fallback policy. The examples may overlap the 6,000 used by the in-house classifier; both stand for "labels this organisation already has".

**Metrics.** Agreement with the recorded band, quadratic kappa, and the paired few-shot minus zero-shot difference per arm (same work orders, clustered by university), plus the distance to the 68.7% in-house-trained classifier.

**Expectation recorded now.** Few-shot lifts every arm above its zero-shot agreement, because the examples reveal the site's conventions, but stays below the classifier trained on 6,000 in-house work orders. If instead a 20-example prompt matches 68.7%, the practical guidance changes: a handful of labelled reports would replace a training set.
