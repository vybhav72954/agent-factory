# ForgeMind

> **End-to-End AI System for Industrial Predictive Maintenance**
> Combining Deep Learning, LLM-based Agents, and Real-Time Operations Analytics

---

## Overview

ForgeMind is a **full-stack predictive maintenance system** that forecasts machine failures using deep learning and translates those predictions into **actionable operational decisions** through a modular, multi-agent pipeline.

Unlike traditional ML projects that stop at prediction, ForgeMind closes the loop:

> **Fault Input → Sensor Interpretation → RUL Prediction → Capacity Impact → Operational Decisions → Real-Time Dashboard**

---

## Key Highlights

* **CNN + LSTM Model** for Remaining Useful Life (RUL) prediction on multivariate time-series data
* **LLM-Assisted Agent Pipeline** for fault interpretation and decision orchestration (Gemini 2.5 Flash + Groq)
* **Dual model variants** — turbofan (N-CMAPSS) and physics-informed factory simulator
* **Factory State Engine** for centralized, real-time system memory
* **Operations Analytics Layer** converting predictions into business insights
* **Interactive Terminal Dashboard** (Textual UI) for live monitoring and simulation
* **Comprehensive Testing Suite** including integration and failure scenarios (474 tests)

---

## System Architecture

```text
User Input (Fault Description)
        ↓
Input Guard (validation)
        ↓
Diagnostic Agent (LLM → structured sensor spike)
        ↓
DL Engine (CNN-LSTM → RUL prediction)
        ↓
Capacity Agent (system impact computation)
        ↓
Ops Analytics (alerts, scheduling, health metrics)
        ↓
FactoryState (central memory)
        ↓
Floor Manager (decision communication)
        ↓
Terminal Dashboard (real-time visualization)
```

---

## Deep Learning Engine

### Problem

Predict **Remaining Useful Life (RUL)** of industrial machines from multivariate sensor time-series.

### Pipeline

* Sliding window time-series construction (50 × 18 features)
* Unit-wise data separation (prevents leakage)
* MinMax scaling (train-only fitting)
* CNN + LSTM hybrid architecture

### Model Design

* **CNN Layers** — capture local degradation patterns
* **LSTM Layers** — model long-term temporal dependencies
* **MLP Head** — regression output (RUL in shift-cycles)

### Model Variants

| Variant | Weights | Output | When to use |
|---|---|---|---|
| **Turbofan** (default) | `best_model.pt` + `scaler.pkl` | Bimodal | N-CMAPSS DS02 baseline |
| **Simulator** | `best_model_simulator.pt` + `scaler_simulator.pkl` | Continuous | Physics-informed factory data |

Set `FORGEMIND_USE_SIMULATOR_MODEL=1` in `.env` to use the simulator variant.

---

## Agent Pipeline

ForgeMind uses a **modular multi-agent design**, where each component has a clearly defined responsibility:

| Agent | Role |
|---|---|
| Input Guard | Filters invalid/noise inputs |
| Diagnostic Agent | Converts fault text → structured sensor anomaly |
| DL Oracle | Predicts RUL |
| Capacity Agent | Converts RUL → system capacity impact |
| Floor Manager | Generates human-readable dispatch decisions |

### Design Principle

> LLMs are used for **interpretation and communication**, not core logic. Core routing is deterministic.

Severity-driven injection: Gemini classifies fault severity (LOW / MEDIUM / HIGH), which directly controls injection magnitude via `SEVERITY_MULTIPLIERS`. A single "catastrophic bearing failure" takes a machine offline in one shot; repeated moderate faults degrade it progressively.

---

## Factory State

A centralized state object maintains:

* Machine health and RUL per machine
* Per-machine sensor ring buffers (60 readings × 18 sensors)
* Capacity metrics and breakeven risk flags
* Maintenance schedule and shift health
* Cumulative damage across fault cycles

> Ensures synchronization across ML, agents, UI, and analytics layers.

---

## Operations Analytics

Transforms predictions into actionable insights:

* **RUL Cliff Detection** — flags sudden ≥40% RUL drops
* **Sensor Saturation Detection** — warns on 5+ consecutive saturated readings
* **Predictive Maintenance Scheduling** — ranked queue: OFFLINE → TODAY → THIS WEEK
* **Shift Health Monitoring** — CRITICAL / AT RISK / CAUTION / NOMINAL
* **Degradation Leaderboard** — slope-ranked: FAST / SLOW / STABLE / IMPROVING

---

## Terminal Dashboard

A real-time interactive 4-pane system:

1. **Sensor Feed + RUL + Reliability**
2. **Capacity Dashboard + Maintenance Queue**
3. **Agent Communication Log**
4. **Chaos Engine** — fault injection interface

Keyboard shortcuts: `Ctrl+R` reset all machines · `Ctrl+Q` quit

---

## Installation

```bash
git clone https://github.com/vybhav72954/agent-factory.git
cd agent-factory

# Preferred
uv sync

# Alternative
pip install -r requirements.txt
```

### Required weights

Place in `dl_engine/weights/`:

```text
dl_engine/weights/
  ├── best_model.pt              # turbofan variant
  ├── scaler.pkl
  ├── best_model_simulator.pt   # simulator variant
  └── scaler_simulator.pkl
```

### Environment variables

Create a `.env` file at the project root:

```env
GEMINI_API_KEY_DIAGNOSTIC=<key>
GEMINI_API_KEY_FLOOR_MANAGER=<key>
GROQ_API_KEY=<key>
FORGEMIND_USE_SIMULATOR_MODEL=1   # omit or set to 0 for turbofan
```

Missing Gemini keys → deterministic fallback (no crash).
Missing Groq key → keyword fallback tagged `[GROQ-UNAVAILABLE]`.

---

## Usage

Run the dashboard:

```bash
python -m terminal.app
```

Example fault descriptions:

```text
bearing overheating on Machine 3
catastrophic pressure rupture on Machine 1
minor vibration anomaly on Machine 5
```

---

## Dataset

Two training data sources are supported:

| Source | Description |
|---|---|
| **N-CMAPSS DS02** | NASA aircraft engine turbofan dataset — multivariate sensor degradation |
| **Physics-informed simulator** | Custom factory simulator (`research/simulator/`) — 6 components, 18 sensors, polarity-aware degradation |

N-CMAPSS source: [Kaggle — N-CMAPSS DS02](https://www.kaggle.com/datasets/chaturvedivybhav/aircraft-ds02-006)

To regenerate simulator training data:

```bash
python -m research.simulator.generate_training_data
python -m research.simulator.train_simulator_model --epochs 25
```

---

## Testing

```bash
# Full suite (474 tests, ~9 skipped)
python -m pytest tests/ -q

# By category
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/terminal/
```

No API keys required — all LLM calls are mocked in tests.

---

## Research

ForgeMind is the artifact for the paper:

> **"The Geometry of Apparent LLM Advantage in Automated Prognostic Health Management"**

The `research/` directory contains the full evaluation pipeline:

* `research/baselines.py` — 7-strategy comparison (keyword, regex, Groq Llama, Gemini, agentic)
* `research/probe_cliff_3d.py` — RUL surface sweep across sensor space
* `research/ablation.py` — component ablation studies
* `research/domain3/` — Bayesian second-domain generalization demo
* `research/figures.py` — publication-grade figure generation

Reproduce all results:

```bash
python -m research.baselines --stability-runs 3
python -m research.probe_cliff_3d --steps 15
python -m research.ablation
```

See `REPRODUCE.md` for the full step-by-step.

---

## Tech Stack

* Python 3.10+
* PyTorch
* NumPy / scikit-learn
* Textual (terminal UI)
* Gemini 2.5 Flash (Google AI)
* Groq (Llama 3 / Llama 4)
* Pydantic
* Pytest

---

## License

MIT License
