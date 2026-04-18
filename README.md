# ForgeMind

> **End-to-End AI System for Industrial Decision Intelligence**
> Combining Deep Learning, LLM-based Agents, and Real-Time Operations Analytics

---

## 🚀 Overview

ForgeMind is a **full-stack predictive maintenance system** that forecasts machine failures using deep learning and translates those predictions into **actionable operational decisions** through a modular, multi-agent pipeline.

Unlike traditional ML projects that stop at prediction, ForgeMind closes the loop:

> **Fault Input → Sensor Interpretation → RUL Prediction → Capacity Impact → Operational Decisions → Real-Time Dashboard**

---

## 🧠 Key Highlights

* 🔬 **CNN + LSTM Model** for Remaining Useful Life (RUL) prediction on multivariate time-series data
* 🤖 **LLM-Assisted Agent Pipeline** for fault interpretation and decision orchestration
* 🏭 **Factory State Engine** for centralized, real-time system memory
* 📊 **Operations Analytics Layer** converting predictions into business insights
* 🖥️ **Interactive Terminal Dashboard** (Textual UI) for live monitoring and simulation

---

## 🏗️ System Architecture

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

## 🔬 Deep Learning Engine

### 📌 Problem

Predict **Remaining Useful Life (RUL)** of industrial machines from sensor data.

### ⚙️ Pipeline

* Sliding window time-series construction (50 × 18 features)
* Unit-wise data separation (prevents leakage)
* MinMax scaling (train-only fitting)
* CNN + LSTM hybrid architecture

### 🧠 Model Design

* **CNN Layers** → capture local degradation patterns
* **LSTM Layers** → model long-term temporal dependencies
* **MLP Head** → regression output (RUL)

### 📊 Metrics

* **RMSE** → statistical accuracy
* **NASA Score** → asymmetric cost-sensitive evaluation

> NASA scoring penalizes late failure predictions more heavily, aligning with real-world risk.

---

## 🤖 Agent Pipeline

ForgeMind uses a **modular multi-agent design**, where each component has a clearly defined responsibility:

| Agent            | Role                                            |
| ---------------- | ----------------------------------------------- |
| Input Guard      | Filters invalid/noise inputs                    |
| Diagnostic Agent | Converts fault text → structured sensor anomaly |
| DL Oracle        | Predicts RUL                                    |
| Capacity Agent   | Converts RUL → system capacity impact           |
| Floor Manager    | Generates human-readable decisions              |

### 🔥 Design Principle

> LLMs are used for **interpretation and communication**, not core logic.

---

## 🧠 Factory State (Core System Layer)

A centralized state object maintains:

* Machine health & RUL
* Sensor histories
* Capacity metrics
* Maintenance schedules
* Logs & analytics

> Ensures synchronization across ML, agents, UI, and analytics layers.

---

## 📊 Operations Analytics

Transforms predictions into actionable insights:

* 🚨 **RUL Cliff Detection** — sudden degradation alerts
* ⚠️ **Sensor Saturation Detection** — data reliability warnings
* 📅 **Predictive Maintenance Scheduling**
* 🏭 **Shift Health Monitoring**
* 📉 **Degradation Leaderboard**

> Bridges the gap between ML output and business decision-making.

---

## 🖥️ Terminal Dashboard (Textual UI)

A real-time interactive system with 4 panes:

1. **Sensor Feed + RUL + Reliability**
2. **Capacity Dashboard + Maintenance Queue**
3. **Agent Communication Log**
4. **Chaos Engine (fault injection interface)**

Run:

```bash
python -m terminal.app
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Ensure model weights exist:

```text
dl_engine/weights/
  ├── best_model.pt
  └── scaler.pkl
```

---

## ▶️ Usage

Run the dashboard:

```bash
python -m terminal.app
```

Then enter fault descriptions like:

```text
bearing overheating on Machine 3
pressure surge in hydraulic line
```

---

## 💡 Design Strengths

* ✅ End-to-end system (not just ML model)
* ✅ Strong separation of concerns
* ✅ Hybrid AI (DL + Rules + LLMs)
* ✅ Robustness (fallbacks, validation, retries)

---

## 📦 Tech Stack

* Python
* PyTorch
* NumPy
* Textual (terminal UI)
* LLM APIs (Gemini / structured prompting)
---

## 🎯 Project Positioning

This project is best described as:

> 🔥 **An End-to-End Predictive Maintenance Decision System**

Not just:

* ❌ “a deep learning model”
* ❌ “an agent system”

But a **complete AI-powered operations pipeline**.

---

## 📜 License

MIT License

---
