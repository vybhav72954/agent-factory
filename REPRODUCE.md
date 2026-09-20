# Reproduction Guide

Step-by-step commands to reproduce every result on a fresh machine. Generated 2026-05-25 after the audit-round-2 cleanup.

Tested on Windows 11 / PowerShell with `uv` and Python 3.13. Linux/macOS equivalents in collapsed sections at the end.

---

## 0. Audit-round-2 bug fixes (caught during this session)

| # | Issue | Fix |
|---|---|---|
| AR2-1 | Stray `=0.11.0` pip-log file in repo root | Deleted |
| AR2-2 | Paper §1.2 / §5.2 / §5.5 claimed "95% of probe points at modes" | Actual is **86.6%** — fixed in paper |
| AR2-3 | Paper §5.5 said simulator DEGRADED was 61.5% | Actual is **60.8%** — fixed |
| AR2-4 | Paper §5.3 said "66,398 training pairs" | Changed to "approximately 60,000" so future regenerations stay accurate |
| AR2-5 | Paper §5.3 sequence said "RUL 32 → 21 → 7" | Updated to actual "34.7 → 21.1 → 6.5" |
| AR2-6 | `training_data.npz` was the v3 file (fault_prob=0.75) while active weights are v2 | Regenerated training data with current code; future trainings now produce v2-equivalent models |
| AR2-7 | `README.md` headline numbers were from N=18 stability=1 run (pre-CRIT-2) | Updated to N=54 stability=3 numbers |
| AR2-8 | `README.md` variant convention table didn't mention v1/v2/v3 simulator snapshots | Expanded to 6-row table |
| AR2-9 | `extensions_roadmap.md` completion-tracking table missed Extension #8 and showed #6 as un-started | Updated to reflect actual state |

**Tests: 435 passing, 9 skipped, 0 failures.**

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.13 (any 3.11+ works) |
| `uv` | latest (preferred) OR `pip` |
| NVIDIA GPU + CUDA driver | Optional but recommended; CPU works too (training will take ~75 min instead of ~3 min) |
| API keys | Google Gemini (mandatory for `agentic` strategy); Groq (mandatory for `groq_llama3`/`groq_llama4`) |

### 1.1 Clone and set up the environment

```powershell
git clone <repo-url> ForgeMind
cd ForgeMind

# Install dependencies (preferred — uv is much faster than pip for large packages)
uv sync

# Alternative: pip
pip install -r requirements.txt
```

If your `uv`-managed venv doesn't have pip, install it with `python -m ensurepip` before any `pip install` fallback works.

### 1.2 Install CUDA-enabled PyTorch (skip if CPU is fine)

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# Confirm:
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 1.3 Set API keys in `.env`

Create `.env` in the project root (gitignored):

```
GEMINI_API_KEY_DIAGNOSTIC=<your-gemini-key>
GEMINI_API_KEY_FLOOR_MANAGER=<your-gemini-key>
GROQ_API_KEY=<your-groq-key>
# Optional: switch model variant globally (default: turbofan model)
# FORGEMIND_USE_SIMULATOR_MODEL=1
```

If you don't have an API key, deterministic strategies (`keyword_only`, `keyword_regex_severity`, `fixed_midrange`) still work; LLM strategies will fall back to a keyword lookup tagged `[GROQ-UNAVAILABLE]` or `[GEMINI-UNAVAILABLE]`.

### 1.4 Verify the install

```powershell
python -m pytest tests/ -q --tb=line --no-header
```

Expected: **435 passed, 9 skipped** in ~3 minutes. The 9 skipped are live-Gemini tests gated by `FORGEMIND_RUN_LIVE=1`.

---

## 2. Run the demo (Textual TUI)

```powershell
# Default: loads turbofan-trained model
python -m terminal.app

# Or with simulator-trained model (recommended for the new physics-coherent demo)
$env:FORGEMIND_USE_SIMULATOR_MODEL = "1"
python -m terminal.app
```

In a second terminal:
```powershell
Get-Content forgemind.log -Wait
```
for live pipeline logs.

Keyboard: `Ctrl+R` resets all machines, `Ctrl+Q` quits.

Try prompts from `audit_and_demo_guide.md` (e.g. "bearing temperature spike on Machine 3", "catastrophic motor failure on Machine 5").

---

## 3. Reproduce every research result

All three experiment scripts auto-snapshot to `_turbofan.*` or `_simulator.*` based on the loaded variant. No manual renaming needed.

### 3.1 On the turbofan checkpoint (default — no env var)

```powershell
# Make sure the env var is unset
Remove-Item Env:\FORGEMIND_USE_SIMULATOR_MODEL -ErrorAction SilentlyContinue

# Probe sweep — ~30 seconds, no LLM calls
python -m research.probe_cliff_3d --steps 15

# Baselines — ~10 minutes, uses Gemini + Groq APIs
python -m research.baselines --stability-runs 3

# Ablations — ~3 minutes, uses Gemini
python -m research.ablation
```

Outputs in `research/results/`:
- `probe_cliff_3d.csv` + `_turbofan` copy
- `baselines_comparison.csv` + `_turbofan` copy
- `ablation_results.csv` + `_turbofan` copy
- Plus summary `.md` and probe `.png` files for each

### 3.2 On the simulator checkpoint

```powershell
$env:FORGEMIND_USE_SIMULATOR_MODEL = "1"

python -m research.probe_cliff_3d --steps 15
python -m research.baselines --stability-runs 3
python -m research.ablation

# Always reset when done so production stays on turbofan
Remove-Item Env:\FORGEMIND_USE_SIMULATOR_MODEL
```

Outputs: same as 3.1 plus `_simulator.*` snapshots.

### 3.3 Regenerate per-prompt OFFLINE detail (optional)

```powershell
python -c @"
import pandas as pd
from research.evaluation_rubric import TEST_PROMPTS
df = pd.read_csv('research/results/baselines_comparison_simulator.csv')
df = df[~df['rejected']].copy()
exp = {p: s for p, _mid, s in TEST_PROMPTS}
df['expected'] = df['prompt'].map(exp)
df['match'] = df['status'] == df['expected']
off = df[df['expected']=='OFFLINE']
rows = []
for prompt in sorted(off['prompt'].unique()):
    sub = off[off['prompt']==prompt]
    row = {'prompt': prompt, 'expected': 'OFFLINE'}
    for s in sorted(sub['strategy'].unique()):
        sg = sub[sub['strategy']==s]
        row[f'{s}_sensor'] = sg['sensor_id'].mode().iloc[0] if len(sg['sensor_id'].mode()) else 'NA'
        row[f'{s}_sev'] = sg['severity'].mode().iloc[0] if len(sg['severity'].mode()) else 'NA'
        row[f'{s}_rul'] = round(sg['rul'].mean(), 1)
        row[f'{s}_match'] = f'{int(sg[\"match\"].sum())}/{len(sg)}'
    rows.append(row)
pd.DataFrame(rows).to_csv('research/results/baselines_per_prompt_offline_detail.csv', index=False)
print('Wrote per-prompt detail')
"@
```

---

## 4. Retrain the simulator model (optional)

Takes ~3 min on GPU, ~75 min on CPU.

```powershell
# Step 1: regenerate training data (~30 sec)
python -m research.simulator.generate_training_data --n-lifecycles 1000

# Step 2: train the model
python -m research.simulator.train_simulator_model --epochs 25 --device cuda
# (or --device cpu if no GPU)
```

Output:
- `dl_engine/weights/best_model_simulator.pt` (replaces current)
- `dl_engine/weights/scaler_simulator.pkl`

Verify the new checkpoint:
```powershell
python -c @"
import os; os.environ['FORGEMIND_USE_SIMULATOR_MODEL']='1'
import dl_engine.inference as inf
inf.reset_loaded_model()
import numpy as np
b = inf.get_healthy_baseline(noise_std_frac=0.0)
print(f'Healthy RUL: {inf.predict_rul(b):.1f}  (expect ~80-90 for a properly-trained simulator model)')
"@
```

To preserve the previous weights before retraining:
```powershell
Copy-Item dl_engine/weights/best_model_simulator.pt dl_engine/weights/best_model_simulator_v2.pt.bak
Copy-Item dl_engine/weights/scaler_simulator.pkl dl_engine/weights/scaler_simulator_v2.pkl.bak
```

---

## 5. Headline result reproduction commands (one-liners)

```powershell
# Sanity-check active model variant
python -c "import dl_engine.inference as inf; inf.load_model(); print(inf.get_loaded_variant())"

# Headline table from baselines comparison CSV (works on either checkpoint's data)
python -c @"
import pandas as pd
from research.evaluation_rubric import TEST_PROMPTS
df = pd.read_csv('research/results/baselines_comparison_simulator.csv')
df = df[~df['rejected']].copy()
exp = {p: s for p, _mid, s in TEST_PROMPTS}
df['expected'] = df['prompt'].map(exp)
df['match'] = df['status'] == df['expected']
r = []
for s in sorted(df['strategy'].unique()):
    sub = df[df['strategy']==s]
    on  = sub[sub['expected']=='ONLINE']
    off = sub[sub['expected']=='OFFLINE']
    r.append((s, sub['match'].mean(), on['match'].mean(), off['match'].mean(), sub['latency_ms'].quantile(0.95)))
r.sort(key=lambda x: x[1])
print(f'{\"Strategy\":<25} {\"Match\":>7} {\"ONLINE\":>7} {\"OFFLINE\":>8} {\"P95ms\":>8}')
for s,m,on,off,lat in r:
    print(f'{s:<25} {m:>6.1%} {on:>6.1%} {off:>7.1%} {lat:>8.0f}')
"@
```

---

## 6. Paper-table reproduction

To regenerate **paper Table 1 (turbofan)** and **paper Table 2 (simulator v2)** from the frozen snapshots:

```powershell
python -c @"
import pandas as pd
from research.evaluation_rubric import TEST_PROMPTS
exp = {p: s for p, _mid, s in TEST_PROMPTS}
def headline(csv, label):
    df = pd.read_csv(csv)
    df = df[~df['rejected']].copy()
    df['expected'] = df['prompt'].map(exp)
    df['match'] = df['status'] == df['expected']
    print(f'=== {label} ===')
    r = []
    for s in sorted(df['strategy'].unique()):
        sub = df[df['strategy']==s]
        on  = sub[sub['expected']=='ONLINE']
        off = sub[sub['expected']=='OFFLINE']
        r.append((s, sub['match'].mean(), on['match'].mean(), off['match'].mean(), sub['latency_ms'].quantile(0.95)))
    r.sort(key=lambda x: x[1])
    for s,m,on,off,lat in r:
        print(f'  {s:<25} match={m:>5.1%} online={on:>5.1%} offline={off:>5.1%} P95={lat:>5.0f}ms')
    print()
headline('research/results/baselines_comparison_turbofan.csv', 'Paper Table 1 (turbofan)')
headline('research/results/baselines_comparison_simulator.csv', 'Paper Table 2 (simulator v2)')
"@
```

Expected output:
```
=== Paper Table 1 (turbofan) ===
  keyword_only              match=53.7% online=30.6% offline=100.0% P95=    6ms
  fixed_midrange            match=66.7% online=100.0% offline= 0.0% P95=    5ms
  groq_llama4               match=81.5% online=100.0% offline=44.4% P95=  427ms
  groq_llama3               match=87.0% online=97.2% offline=66.7% P95=  736ms
  agentic                   match=88.9% online=100.0% offline=66.7% P95= 2210ms
  gemini_3_5_flash          match=88.9% online=100.0% offline=66.7% P95= 5183ms
  keyword_regex_severity    match=94.4% online=100.0% offline=83.3% P95=    6ms

=== Paper Table 2 (simulator v2) ===
  keyword_only              match=38.9% online= 8.3% offline=100.0% P95=   91ms
  fixed_midrange            match=66.7% online=100.0% offline= 0.0% P95=    5ms
  groq_llama4               match=79.6% online=100.0% offline=38.9% P95=  610ms
  groq_llama3               match=85.2% online=94.4% offline=66.7% P95=  629ms
  agentic                   match=88.9% online=100.0% offline=66.7% P95= 1662ms
  gemini_3_5_flash          match=88.9% online=100.0% offline=66.7% P95= 5525ms
  keyword_regex_severity    match=94.4% online=100.0% offline=83.3% P95=    7ms
```

---

## 7. Switching model variants safely

```python
import os
import dl_engine.inference as inf

# Turbofan → simulator
os.environ["FORGEMIND_USE_SIMULATOR_MODEL"] = "1"
inf.reset_loaded_model()              # required — clears the per-session cache
rul = inf.predict_rul(window)         # triggers lazy load with new variant

# Simulator → turbofan
del os.environ["FORGEMIND_USE_SIMULATOR_MODEL"]
inf.reset_loaded_model()
rul = inf.predict_rul(window)

# Confirm which is loaded
print(inf.get_loaded_variant())       # 'turbofan' | 'simulator' | None
```

---

## 8. Linux / macOS notes

Replace PowerShell-isms with bash equivalents:

| PowerShell | bash |
|---|---|
| `$env:VAR = "1"` | `export VAR=1` |
| `Remove-Item Env:\VAR` | `unset VAR` |
| `Copy-Item a b` | `cp a b` |
| `Get-Content file -Wait` | `tail -f file` |
| `.venv\Scripts\python.exe` | `.venv/bin/python` |
| `@"..."@` here-string | `<<'EOF' ... EOF` heredoc |

Test suite, scripts, and the TUI itself all work identically.

---

## 9. Known gotchas

1. **Windows console can't render Unicode box-drawing or check marks.** All scripts use ASCII alternatives now. If you see a `UnicodeEncodeError`, it's a regression — file an issue.
2. **The `_loaded_variant` cache is per-Python-session.** If you set the env var inside a running Python session, call `inf.reset_loaded_model()` before the next `predict_rul()`.
3. **CUDA install can be slow.** The wheel is ~2.6 GB. If `pip install torch --index-url https://download.pytorch.org/whl/cu128` hangs, that's just the download — give it 5-15 minutes.
4. **Active simulator weights (v2) were trained on slightly different data than what `research/simulator/data/training_data.npz` currently holds**. The v2 weights are from the original 66,398-pair training set before the MED-12 dedup fix. The current npz (regenerated 2026-05-25) is what future re-trainings will use, producing a v2-equivalent model with ~66,384 pairs.
5. **v3 simulator weights are lost** (rolled back during MED-10 experiment). Result snapshots `*_simulator_v3.*` are preserved for the paper appendix. To recreate v3 weights: edit `research/simulator/generate_training_data.py` to set `fault_probability=0.75`, regenerate, retrain. ~5 minutes total.

---

## 10. Sanity check: end-to-end smoke test (~2 minutes)

The fastest way to verify everything works after a clone:

```powershell
# 1. Install
uv sync

# 2. Run tests (covers all 435 tests including simulator + threading)
python -m pytest tests/ -q --tb=line --no-header
# expect: 435 passed, 9 skipped

# 3. Simulator self-test (writes nothing — safe on a populated repo)
python -m research.simulator.factory_simulator
# expect: prints sensor readings + bearing-fault → RUL=0 demo

# 4. Rubric self-test (writes nothing)
python -m research.evaluation_rubric
# expect: prints 'good dispatch' score 5.00 and 'bad dispatch' score 1.00

# 5. (OPTIONAL — only on a fresh clone with no result files) Run a tiny probe
#    WARNING: a 5-step probe WILL overwrite the high-quality 15-step probe
#    files via the auto-snapshot mechanism. Only run this on a fresh clone
#    where probe_cliff_3d_turbofan.csv doesn't already exist, OR use it
#    knowing you'll need to re-run `python -m research.probe_cliff_3d --steps 15`
#    afterwards to restore the canonical 3375-point probe.
python -m research.probe_cliff_3d --steps 5 --no-plots
```

If steps 1-4 succeed, the repo is in a reproducible state. Step 5 is destructive — skip it unless you actually want to test the probe pipeline.
