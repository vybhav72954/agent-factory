# TEAM AGENT — Technical Runbook

## Agentic Operations Pipeline: Diagnostic → Oracle → Capacity → Floor Manager

**Owners:** 2 people · **Environment:** Local machine + Gemini 2.5 Flash API (paid)

**Your deliverable:** A function `run_agent_loop(user_text, machine_id, base_window) -> dict` that takes a professor's chaos input and returns a complete result package (spike data, RUL, capacity report, dispatch orders). You own `/agents/*`.

---

## 0. What You're Building — The 30-Second Version

```
Professor types: "bearing temperature surge on Machine 4"
                              │
                              ▼
              ┌─ Input Guard ──────────────┐
              │ Is this a valid fault?     │──── NO → "[REJECTED] Try: ..."
              │ Keyword check, no LLM      │
              └────────────┬───────────────┘
                           │ YES
                           ▼
              ┌─ Agent 1: Diagnostic ──────┐
              │ Gemini 2.5 Flash           │
              │ Pydantic structured output │
              │ → SensorSpike object       │
              │ + validate + retry         │
              │ + deterministic fallback   │
              └────────────┬───────────────┘
                           │ Modified (50, 18) tensor
                           ▼
              ┌─ DL Oracle ────────────────┐
              │ predict_rul(tensor) → 12.0 │
              │ (You call this, don't own) │
              └────────────┬───────────────┘
                           │ RUL float
                           ▼
              ┌─ Agent 2: Capacity ────────┐
              │ Pure Python math. NO LLM.  │
              │ RUL=12 → OFFLINE           │
              │ Capacity drops to 80%      │
              │ ΣPD/T = 1.12 → CRITICAL    │
              └────────────┬───────────────┘
                           │ Capacity report dict
                           ▼
              ┌─ Agent 3: Floor Manager ───┐
              │ Gemini 2.5 Flash           │
              │ Plain text dispatch orders │
              │ + offline fallback cache   │
              └────────────┬───────────────┘
                           │ Dispatch string
                           ▼
              Return everything to Terminal UI
```

You own every box above **except** the DL Oracle. That's Team DL's `predict_rul()` — you just call it.

---

## 1. Dependencies — Complete Package List

```bash
pip install google-genai pydantic numpy
```

| Package | Version | PyPI name | What it does in this project |
|---|---|---|---|
| **google-genai** | ≥1.0 | `google-genai` | Google's official SDK for Gemini API calls. Used by Agent 1 (Diagnostic) and Agent 3 (Floor Manager). Supports structured JSON output via Pydantic schemas. |
| **pydantic** | ≥2.0 | `pydantic` | Data validation and schema definition. Defines the `SensorSpike` model that Gemini is forced to conform to. Eliminates manual JSON parsing and most validation code. |
| **numpy** | ≥1.24 | `numpy` | Array manipulation. Used to build, modify, and pass the `(50, 18)` sensor tensor between agents and the DL Oracle. |

**Not used (and why):**

| Package | Why we skipped it |
|---|---|
| LangChain | Our pipeline is linear with exactly 2 LLM calls. LangChain adds a massive dependency tree and abstraction layers for features we don't use (memory, retrieval, complex chains). |
| LangGraph | Would be the right choice if we needed conditional branching, parallel agent execution, or human-in-the-loop approval. Our pipeline is strictly sequential — overkill for now. Worth considering as a v2 refactor if the project grows. |
| CrewAI | Role-based multi-agent framework. Our agents don't negotiate or collaborate — they chain. No benefit. |
| LlamaIndex | RAG framework. We don't do document retrieval. |

---

## 2. File Structure

```
/agents/
├── schemas.py              # Pydantic models — SensorSpike schema for structured output
├── input_guard.py          # Pre-LLM keyword filter — rejects nonsensical input
├── diagnostic_agent.py     # Agent 1: NL → SensorSpike via Gemini structured output
├── capacity_agent.py       # Agent 2: deterministic MIOM capacity math (NO LLM)
├── floor_manager.py        # Agent 3: capacity report → dispatch orders (Gemini)
├── fallback_cache.py       # 5 pre-scripted scenario responses for offline mode
├── agent_loop.py           # Orchestrator chaining all agents in sequence
└── prompts.py              # All system prompts versioned and centralized
```

---

## 3. Gemini Client Setup

```python
# Shared setup — used by diagnostic_agent.py and floor_manager.py

import os
from google import genai

# API key from environment variable — never hardcode in committed files
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Day 1 verification — run this to confirm your key works:
if __name__ == "__main__":
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in exactly 5 words."
    )
    print(response.text)
    # Expected: some 5-word greeting
```

**API key management:**

```bash
# Add to your shell profile (.bashrc / .zshrc):
export GEMINI_API_KEY="your-key-here"

# Or create a .env file (add .env to .gitignore!):
echo "GEMINI_API_KEY=your-key-here" > .env

# Load in Python:
from dotenv import load_dotenv
load_dotenv()
```

If you use `.env`, add `python-dotenv` to your dependencies:

```bash
pip install python-dotenv
```

---

## 4. Pydantic Schemas (`schemas.py`)

This is the foundation of the structured output approach. Instead of asking Gemini to "return only JSON" (which it violates 10-20% of the time with markdown fences, extra prose, or missing fields), you define a Pydantic model and Gemini **guarantees** the response conforms to it.

```python
# agents/schemas.py

from pydantic import BaseModel, Field
from enum import Enum


class FaultSeverity(str, Enum):
    """Severity levels for fault injection."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SensorSpike(BaseModel):
    """
    Schema for Agent 1 (Diagnostic Agent) output.

    When passed to Gemini as response_schema, the API guarantees
    the response will be valid JSON matching this schema.
    No markdown fences. No extra prose. No missing fields.

    You still need to validate VALUE RANGES (e.g., sensor_id must
    be one of 18 valid names) — Pydantic handles structure,
    your code handles domain logic.
    """
    sensor_id: str = Field(
        description=(
            "Which sensor to spike. Must be exactly one of: "
            "W0, W1, W2, W3, Xs0, Xs1, Xs2, Xs3, Xs4, Xs5, "
            "Xs6, Xs7, Xs8, Xs9, Xs10, Xs11, Xs12, Xs13"
        )
    )
    spike_value: float = Field(
        ge=0.0, le=1.0,
        description="Injected sensor value in normalized [0.0, 1.0] range."
    )
    affected_window_positions: list[int] = Field(
        description=(
            "Which timestep positions (0-49) to inject the spike into. "
            "Maximum 10 positions."
        )
    )
    fault_severity: FaultSeverity = Field(
        description="Severity classification of the fault."
    )
    plain_english_summary: str = Field(
        description="One-sentence human-readable description of the fault."
    )
```

### 4.1 How Structured Output Works

```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Your prompt here",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SensorSpike,
    ),
)

# response.text is GUARANTEED to be valid JSON matching SensorSpike
# No markdown fences. No extra text. Just JSON.
spike = SensorSpike.model_validate_json(response.text)
print(spike.sensor_id)       # "Xs4"
print(spike.spike_value)     # 0.95
print(spike.fault_severity)  # FaultSeverity.HIGH
```

**What this eliminates vs the old approach:**

| Old approach (raw prompting) | New approach (structured output) |
|---|---|
| Prompt says "return only JSON" — Gemini sometimes ignores this | Gemini server-side enforces JSON schema |
| Need `_strip_markdown_fences()` to remove \`\`\`json wrapping | Never happens — response is always clean JSON |
| `json.loads()` can crash on malformed output | `model_validate_json()` gives clear Pydantic errors |
| Must manually check every field exists and has right type | Pydantic validates types automatically |
| ~20% failure rate on complex schemas | ~2-3% failure rate (mostly value range issues) |

**What you still need to validate manually:**

- `sensor_id` is one of the 18 valid sensor names (Gemini might hallucinate `Xs17`)
- `affected_window_positions` values are all 0–49 (Pydantic validates they're ints, not their range)
- List length ≤ 10

---

## 5. Input Guard (`input_guard.py`)

**Purpose:** Reject garbage inputs before they waste a Gemini API call. Fast, deterministic, local.

```python
# agents/input_guard.py

VALID_KEYWORDS = {
    # Machine / component nouns
    "machine", "cnc", "press", "lathe", "mill", "compressor", "motor",
    "pump", "valve", "filter", "shaft", "rotor", "stator", "gearbox",
    "bearing", "turbine", "actuator", "conveyor", "spindle", "piston",
    # Sensor / measurement nouns
    "temperature", "temp", "pressure", "vibration", "rpm", "speed",
    "sensor", "coolant", "lubrication", "friction", "flow", "torque",
    "voltage", "current", "frequency", "amplitude", "noise", "heat",
    # Fault / event verbs and nouns
    "surge", "spike", "leak", "overheat", "overload", "failure", "fault",
    "wear", "crack", "misalignment", "imbalance", "corrosion",
    "degradation", "malfunction", "shutdown", "stall", "jam", "rupture",
    "erosion", "fatigue", "wobble", "oscillation", "fluctuation",
}


def is_valid_fault_input(user_text: str) -> tuple[bool, str]:
    """
    Fast keyword check. No LLM call. No network.

    Returns:
        (True, "")          — input looks like a valid fault description
        (False, reason_str) — input should be rejected, reason shown to user
    """
    text = user_text.strip()

    if len(text) < 5:
        return False, (
            "Input too short. Describe a machine fault "
            "(e.g., 'bearing temperature spike on Machine 4')."
        )

    if len(text) > 500:
        return False, "Input too long. Keep fault descriptions under 500 characters."

    words = set(text.lower().split())
    if not words.intersection(VALID_KEYWORDS):
        return False, (
            "Unrecognized fault type. Try something like: "
            "'high pressure in compressor', 'bearing overheat on Machine 3', "
            "or 'vibration spike on CNC-Alpha'."
        )

    return True, ""
```

---

## 6. Agent 1 — Diagnostic Agent (`diagnostic_agent.py`)

**Purpose:** Translate natural-language fault → structured `SensorSpike` → injected sensor tensor.

**LLM:** Gemini 2.5 Flash with Pydantic structured output.

### 6.1 Sensor ID ↔ Column Index Mapping

The `(50, 18)` tensor has this column layout:

| Column | Sensor ID | Factory fiction name |
|---|---|---|
| 0 | W0 | Load setting (Mach) |
| 1 | W1 | Ambient pressure (altitude) |
| 2 | W2 | Throttle resolver angle (TRA) |
| 3 | W3 | Inlet temperature (T2) |
| 4 | Xs0 | Physical sensor 1 |
| 5 | Xs1 | Physical sensor 2 |
| 6 | Xs2 | Physical sensor 3 — pressure |
| 7 | Xs3 | Physical sensor 4 |
| 8 | Xs4 | Physical sensor 5 — bearing temperature |
| 9 | Xs5 | Physical sensor 6 |
| 10 | Xs6 | Physical sensor 7 |
| 11 | Xs7 | Physical sensor 8 — vibration |
| 12 | Xs8 | Physical sensor 9 |
| 13 | Xs9 | Physical sensor 10 |
| 14 | Xs10 | Physical sensor 11 — RPM |
| 15 | Xs11 | Physical sensor 12 |
| 16 | Xs12 | Physical sensor 13 |
| 17 | Xs13 | Physical sensor 14 |

### 6.2 System Prompt

```python
# agents/prompts.py

DIAGNOSTIC_SYSTEM_PROMPT = """You are a sensor diagnostic translator for a factory floor.
The factory has 18 sensors: 4 operating condition sensors (W0-W3)
and 14 physical sensors (Xs0-Xs13).
Normal operating range for all sensors is [0.0, 1.0] after normalization.

Given a fault description, determine:
1. Which sensor would be affected
2. What spike value represents this fault (0.0-1.0 scale)
3. Which timestep positions in a 50-step window show the fault
4. The severity level (LOW/MEDIUM/HIGH)
5. A one-sentence plain English summary

sensor_id MUST be exactly one of: W0, W1, W2, W3, Xs0, Xs1, Xs2, Xs3, Xs4, Xs5, Xs6, Xs7, Xs8, Xs9, Xs10, Xs11, Xs12, Xs13.
affected_window_positions MUST contain only integers between 0 and 49.
Maximum 10 positions."""
```

### 6.3 Complete Implementation

```python
# agents/diagnostic_agent.py

import json
import numpy as np
import os
from google import genai
from google.genai import types
from pydantic import ValidationError

from .schemas import SensorSpike
from .prompts import DIAGNOSTIC_SYSTEM_PROMPT

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ── Valid sensor IDs for domain validation ──
VALID_SENSORS = {f"W{i}" for i in range(4)} | {f"Xs{i}" for i in range(14)}

MAX_RETRIES = 2

# ── Deterministic fallbacks ──
# Used when Gemini fails all retries OR returns invalid sensor IDs
FALLBACK_SPIKES = {
    "temperature": SensorSpike(
        sensor_id="Xs4", spike_value=0.95,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity="HIGH",
        plain_english_summary="Temperature sensor critical spike detected."
    ),
    "pressure": SensorSpike(
        sensor_id="Xs2", spike_value=0.92,
        affected_window_positions=[40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        fault_severity="HIGH",
        plain_english_summary="Pressure sensor abnormal reading."
    ),
    "vibration": SensorSpike(
        sensor_id="Xs7", spike_value=0.88,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity="MEDIUM",
        plain_english_summary="Vibration levels elevated beyond safe threshold."
    ),
    "speed": SensorSpike(
        sensor_id="Xs10", spike_value=0.90,
        affected_window_positions=[46, 47, 48, 49],
        fault_severity="MEDIUM",
        plain_english_summary="Rotational speed anomaly detected."
    ),
    "default": SensorSpike(
        sensor_id="Xs0", spike_value=0.93,
        affected_window_positions=[47, 48, 49],
        fault_severity="HIGH",
        plain_english_summary="General sensor anomaly detected."
    ),
}


def _validate_domain(spike: SensorSpike) -> tuple[bool, str]:
    """
    Pydantic validates structure. This validates domain-specific rules
    that Pydantic alone can't enforce.

    Returns (is_valid, error_message).
    """
    if spike.sensor_id not in VALID_SENSORS:
        return False, f"Hallucinated sensor_id '{spike.sensor_id}'"

    if not all(0 <= p <= 49 for p in spike.affected_window_positions):
        return False, f"Position out of range: {spike.affected_window_positions}"

    if len(spike.affected_window_positions) > 10:
        return False, f"Too many positions: {len(spike.affected_window_positions)}"

    if len(spike.affected_window_positions) == 0:
        return False, "Empty positions list"

    return True, ""


def _get_fallback(user_text: str) -> SensorSpike:
    """Keyword-match to best deterministic fallback."""
    text_lower = user_text.lower()
    for keyword, spike in FALLBACK_SPIKES.items():
        if keyword != "default" and keyword in text_lower:
            # Return a copy with [FALLBACK] tag
            return spike.model_copy(update={
                "plain_english_summary": spike.plain_english_summary + " [FALLBACK]"
            })
    fallback = FALLBACK_SPIKES["default"]
    return fallback.model_copy(update={
        "plain_english_summary": fallback.plain_english_summary + " [FALLBACK]"
    })


def _sensor_id_to_column(sensor_id: str) -> int:
    """
    Convert sensor_id string → column index in the (50, 18) tensor.
    Feature order: [W0, W1, W2, W3, Xs0, Xs1, ... Xs13]
    """
    if sensor_id.startswith("Xs"):
        return int(sensor_id.replace("Xs", "")) + 4   # Xs0→4, Xs13→17
    elif sensor_id.startswith("W"):
        return int(sensor_id.replace("W", ""))          # W0→0, W3→3
    else:
        raise ValueError(f"Unknown sensor_id: {sensor_id}")


def translate_fault_to_tensor(
    base_window: np.ndarray,
    user_text: str
) -> tuple[np.ndarray, dict, bool]:
    """
    Main entry point for Agent 1.

    Args:
        base_window: (50, 18) numpy array — baseline sensor readings
        user_text:   professor's fault description string

    Returns:
        injected_window: (50, 18) numpy array with spike applied
        spike_dict:      dict version of the SensorSpike (for JSON serialization)
        used_fallback:   True if Gemini failed and deterministic fallback was used
    """
    spike = None
    used_fallback = False

    # ── Try Gemini with structured output (with retries) ──
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{DIAGNOSTIC_SYSTEM_PROMPT}\n\nFault description: {user_text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SensorSpike,
                ),
            )

            # Pydantic parse — structure is guaranteed by Gemini, but validate anyway
            candidate = SensorSpike.model_validate_json(response.text)

            # Domain validation — check sensor_id is real, positions in range
            is_valid, error = _validate_domain(candidate)
            if is_valid:
                spike = candidate
                break
            else:
                print(f"  [Diagnostic] Domain validation failed (attempt {attempt+1}): {error}")

        except ValidationError as e:
            print(f"  [Diagnostic] Pydantic validation error (attempt {attempt+1}): {e}")
        except json.JSONDecodeError as e:
            print(f"  [Diagnostic] JSON decode error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"  [Diagnostic] API error (attempt {attempt+1}): {e}")

    # ── Fallback if all retries failed ──
    if spike is None:
        spike = _get_fallback(user_text)
        used_fallback = True
        print(f"  [Diagnostic] Using deterministic fallback: {spike.sensor_id}")

    # ── Inject spike into tensor ──
    injected = base_window.copy()
    col = _sensor_id_to_column(spike.sensor_id)
    for pos in spike.affected_window_positions:
        injected[pos, col] = spike.spike_value

    return injected, spike.model_dump(), used_fallback
```

### 6.4 What Changed vs Raw JSON Prompting

| Aspect | Old (raw prompt) | New (structured output) |
|---|---|---|
| Prompt | "Return ONLY valid JSON..." | Schema enforced server-side by Gemini |
| Parsing | `json.loads()` + manual field checks | `SensorSpike.model_validate_json()` |
| Markdown fences | Need `_strip_markdown_fences()` | Never happens |
| Type validation | Manual isinstance checks | Pydantic handles automatically |
| Failure rate | ~15-20% need retry | ~2-3% (mostly domain range issues) |
| Fallback | Same — deterministic keyword match | Same — deterministic keyword match |

---

## 7. Agent 2 — Capacity Agent (`capacity_agent.py`)

**Purpose:** Deterministic MIOM capacity math. **ZERO LLM calls. ZERO packages beyond Python stdlib.**

### 7.1 The Operations Concepts

| Concept | Formula | What it means |
|---|---|---|
| **Available Time (T)** | `Σ machine.available_time` | Total production hours across all machines |
| **Product Demand (ΣPD)** | `Σ machine.product_demand` | Total units that need to be produced |
| **Machine Requirement** | `ΣPD / T` | Units per hour needed — if > 1.0, demand exceeds capacity |
| **Capacity %** | `(T / max_T) × 100` | How much of the factory's potential is being used |
| **Break-even Risk** | `ΣPD / T > 1.0` | Factory cannot meet demand with current capacity |

### 7.2 Three-State Machine Status

| RUL Range | Status | Available Time | Rationale |
|---|---|---|---|
| RUL > 30 | **ONLINE** | 100% of base time (8.0 hrs) | Healthy — full production |
| 15 < RUL ≤ 30 | **DEGRADED** | 50% of base time (4.0 hrs) | Reduced speed, maintenance window opening |
| RUL ≤ 15 | **OFFLINE** | 0% (0.0 hrs) | Mandatory shutdown |

### 7.3 Complete Implementation

```python
# agents/capacity_agent.py
# ════════════════════════════════════════════════════════════
# NO LLM CALLS IN THIS FILE. EVER. PURE PYTHON MATH.
# No google-genai. No pydantic. No numpy. Just arithmetic.
# ════════════════════════════════════════════════════════════

from typing import Dict

MACHINES: Dict[int, dict] = {
    1: {"name": "CNC-Alpha",    "base_time": 8.0, "product_demand": 120, "available_time": 8.0, "rul": 999.0, "status": "ONLINE"},
    2: {"name": "CNC-Beta",     "base_time": 8.0, "product_demand": 95,  "available_time": 8.0, "rul": 999.0, "status": "ONLINE"},
    3: {"name": "Press-Gamma",  "base_time": 8.0, "product_demand": 140, "available_time": 8.0, "rul": 999.0, "status": "ONLINE"},
    4: {"name": "Lathe-Delta",  "base_time": 8.0, "product_demand": 110, "available_time": 8.0, "rul": 999.0, "status": "ONLINE"},
    5: {"name": "Mill-Epsilon", "base_time": 8.0, "product_demand": 130, "available_time": 8.0, "rul": 999.0, "status": "ONLINE"},
}

RUL_OFFLINE_THRESHOLD  = 15
RUL_DEGRADED_THRESHOLD = 30
DEGRADED_FACTOR        = 0.5


def update_capacity(machine_id: int, new_rul: float) -> dict:
    """
    Update a machine's RUL and recompute factory-wide capacity.

    Args:
        machine_id: 1–5
        new_rul:    float from predict_rul()

    Returns:
        dict with all capacity metrics
    """
    machine = MACHINES[machine_id]
    machine['rul'] = new_rul

    # ── Determine status ──
    if new_rul <= RUL_OFFLINE_THRESHOLD:
        machine['available_time'] = 0.0
        machine['status'] = "OFFLINE"
    elif new_rul <= RUL_DEGRADED_THRESHOLD:
        machine['available_time'] = machine['base_time'] * DEGRADED_FACTOR
        machine['status'] = "DEGRADED"
    else:
        machine['available_time'] = machine['base_time']
        machine['status'] = "ONLINE"

    # ── Factory-wide metrics ──
    total_T  = sum(m['available_time'] for m in MACHINES.values())
    total_PD = sum(m['product_demand'] for m in MACHINES.values())
    max_T    = len(MACHINES) * 8.0

    machine_req = total_PD / total_T if total_T > 0 else float('inf')
    capacity_pct = (total_T / max_T) * 100

    return {
        "machine_id":      machine_id,
        "machine_name":    machine['name'],
        "status":          machine['status'],
        "rul":             round(new_rul, 1),
        "total_T":         round(total_T, 2),
        "total_PD":        total_PD,
        "machine_req":     round(machine_req, 3),
        "capacity_pct":    round(capacity_pct, 1),
        "breakeven_risk":  machine_req > 1.0,
    }


def get_all_machine_statuses() -> list[dict]:
    """Current status of all machines — used by UI dashboard."""
    return [
        {
            "id": mid,
            "name": m['name'],
            "status": m['status'],
            "rul": m['rul'],
            "available_time": m['available_time'],
            "base_time": m['base_time'],
        }
        for mid, m in MACHINES.items()
    ]


def reset_all():
    """Reset all machines to ONLINE / full capacity. For demo restarts."""
    for m in MACHINES.values():
        m['available_time'] = m['base_time']
        m['rul'] = 999.0
        m['status'] = "ONLINE"
```

### 7.4 Hand-Verification Table — Test These

| Scenario | total_T | ΣPD | ΣPD/T | capacity% |
|---|---|---|---|---|
| All ONLINE | 40.0 | 595 | 14.875 | 100% |
| Machine 4 DEGRADED (RUL=22) | 36.0 | 595 | 16.528 | 90% |
| Machine 4 OFFLINE (RUL=12) | 32.0 | 595 | 18.594 | 80% |
| Machines 3+4 OFFLINE | 24.0 | 595 | 24.792 | 60% |
| All OFFLINE (RUL=0 everywhere) | 0.0 | 595 | ∞ | 0% |

Write unit tests that assert these exact numbers.

---

## 8. Agent 3 — Floor Manager (`floor_manager.py`)

**Purpose:** Turn capacity report into actionable dispatch orders.

**LLM:** Gemini 2.5 Flash — plain text output (NOT structured, because the output is prose, not data).

```python
# agents/floor_manager.py

import os
from google import genai
from .prompts import FLOOR_MANAGER_SYSTEM_PROMPT

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def issue_dispatch_orders(capacity_report: dict) -> str:
    """
    Agent 3: Capacity report → actionable dispatch text.

    Args:
        capacity_report: dict from capacity_agent.update_capacity()

    Returns:
        str: Floor manager dispatch text (max 4 sentences)
    """
    context = f"""Machine {capacity_report['machine_id']} ({capacity_report['machine_name']}) is now {capacity_report['status']}.
RUL = {capacity_report['rul']:.1f} cycles.
Factory capacity: {capacity_report['capacity_pct']}%.
Total available time (T): {capacity_report['total_T']} hours.
Sum of Product Demand (ΣPD): {capacity_report['total_PD']} units.
Machine Requirement ratio (ΣPD/T): {capacity_report['machine_req']}.
Break-even risk: {'YES — CRITICAL' if capacity_report['breakeven_risk'] else 'No'}."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{FLOOR_MANAGER_SYSTEM_PROMPT}\n\nCapacity Report:\n{context}"
        )
        return response.text.strip()
    except Exception as e:
        print(f"  [Floor Manager] API error: {e}")
        return _template_fallback(capacity_report)


def _template_fallback(report: dict) -> str:
    """Hardcoded template when Gemini is unavailable."""
    status = report['status']
    mid = report['machine_id']
    name = report['machine_name']
    cap = report['capacity_pct']

    if status == "OFFLINE":
        return (
            f"[Floor Manager] Machine {mid} ({name}) is OFFLINE. "
            f"Halting production on this unit and dispatching maintenance crew. "
            f"Rerouting workload to remaining machines. "
            f"Factory at {cap}% capacity — monitor ΣPD/T closely."
        )
    elif status == "DEGRADED":
        return (
            f"[Floor Manager] Machine {mid} ({name}) entering DEGRADED mode at 50% load. "
            f"Schedule preventive maintenance within next 2 shift-cycles. "
            f"Do not increase load on this unit. "
            f"Factory at {cap}% capacity."
        )
    else:
        return (
            f"[Floor Manager] Machine {mid} ({name}) is ONLINE and operating normally. "
            f"Factory at {cap}% capacity. No action required."
        )
```

### 8.1 System Prompt

```python
# In agents/prompts.py

FLOOR_MANAGER_SYSTEM_PROMPT = """You are a pragmatic factory floor manager issuing dispatch orders.
You receive a capacity report from a deterministic Python engine.

Your rules:
1. NEVER invent or modify any number — use only the figures given to you.
2. Issue specific, actionable orders (reroute production, authorize overtime, schedule maintenance).
3. Maximum 4 sentences. Terminal-style. No bullet points. No markdown.
4. Begin with [Floor Manager].
5. For DEGRADED machines: recommend reduced load and a maintenance window.
6. For OFFLINE machines: reroute production to remaining machines and escalate.
7. For ONLINE machines with low RUL: suggest monitoring."""
```

### 8.2 Why No Structured Output for Agent 3?

Agent 1 returns **data** (sensor ID, spike value, positions) — structure matters, types matter, ranges matter. Structured output is the right tool.

Agent 3 returns **prose** (dispatch orders) — there's no schema to enforce. Forcing structured output would mean something like `{"sentence_1": "...", "sentence_2": "..."}` which is worse than just getting free-form text. Plain text Gemini call is correct here.

---

## 9. Offline Fallback Cache (`fallback_cache.py`)

**Purpose:** When the internet dies during the demo, the full pipeline still works using pre-scripted responses with live numbers injected.

```python
# agents/fallback_cache.py

"""
Pre-scripted responses for offline demo mode.

WHEN ACTIVATED:
- First Gemini API exception → sets OFFLINE_MODE = True in agent_loop.py
- All subsequent calls skip Gemini and use these scenarios
- No retry thrashing during a live presentation

DESIGN:
- floor_manager_response uses {format_placeholders}
- Placeholders get filled with REAL capacity numbers from Agent 2
- So even offline, the numbers in dispatch orders are live and correct
"""

from .schemas import SensorSpike

CACHED_SCENARIOS = {
    "bearing_overheat": {
        "trigger_keywords": ["bearing", "overheat", "temperature", "temp", "thermal", "hot"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs4", spike_value=0.95,
            affected_window_positions=[45, 46, 47, 48, 49],
            fault_severity="HIGH",
            plain_english_summary="Bearing temperature sensor critical — exceeding thermal limits. [OFFLINE MODE]"
        ),
        "floor_manager_response": (
            "[Floor Manager] Machine {machine_id} ({machine_name}) bearing temp critical. "
            "HALT production immediately and dispatch maintenance crew. "
            "Reroute workload to machines with capacity headroom. "
            "Factory at {capacity_pct}% — authorize overtime if ΣPD/T exceeds 1.0."
        )
    },
    "pressure_surge": {
        "trigger_keywords": ["pressure", "surge", "psi", "hydraulic", "pneumatic", "compressed"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs2", spike_value=0.92,
            affected_window_positions=[40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            fault_severity="HIGH",
            plain_english_summary="Pressure sensor surge detected — possible seal failure. [OFFLINE MODE]"
        ),
        "floor_manager_response": (
            "[Floor Manager] Pressure anomaly on Machine {machine_id} ({machine_name}). "
            "Reduce load to 50% pending inspection. "
            "Check upstream valve integrity before restoring full operation. "
            "Current capacity: {capacity_pct}%."
        )
    },
    "vibration_anomaly": {
        "trigger_keywords": ["vibration", "vibrating", "shaking", "oscillation", "imbalance", "wobble"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs7", spike_value=0.88,
            affected_window_positions=[44, 45, 46, 47, 48, 49],
            fault_severity="MEDIUM",
            plain_english_summary="Vibration levels above normal — potential rotor imbalance. [OFFLINE MODE]"
        ),
        "floor_manager_response": (
            "[Floor Manager] Vibration alert on Machine {machine_id} ({machine_name}). "
            "Schedule balancing during next shift changeover. "
            "Monitor closely — if RUL drops below 15, pull offline. "
            "Factory holding at {capacity_pct}%."
        )
    },
    "rpm_fluctuation": {
        "trigger_keywords": ["rpm", "speed", "rotation", "spin", "motor", "drive"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs10", spike_value=0.90,
            affected_window_positions=[46, 47, 48, 49],
            fault_severity="MEDIUM",
            plain_english_summary="RPM fluctuation detected — possible drive belt issue. [OFFLINE MODE]"
        ),
        "floor_manager_response": (
            "[Floor Manager] RPM instability on Machine {machine_id} ({machine_name}). "
            "Reduce to DEGRADED mode at 50% load. "
            "Inspect drive assembly before next full-speed cycle. "
            "ΣPD/T now at {machine_req}."
        )
    },
    "general_fault": {
        "trigger_keywords": [],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs0", spike_value=0.93,
            affected_window_positions=[47, 48, 49],
            fault_severity="HIGH",
            plain_english_summary="General sensor anomaly — unclassified fault pattern. [OFFLINE MODE]"
        ),
        "floor_manager_response": (
            "[Floor Manager] Anomaly detected on Machine {machine_id} ({machine_name}). "
            "Initiating precautionary slowdown. "
            "Maintenance team: inspect and report within 30 minutes. "
            "Factory capacity at {capacity_pct}%."
        )
    }
}


def match_scenario(user_text: str) -> dict:
    """Find best matching cached scenario by keyword overlap."""
    text_lower = user_text.lower()
    best_match = None
    best_score = 0

    for name, scenario in CACHED_SCENARIOS.items():
        keywords = scenario["trigger_keywords"]
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_match = scenario

    return best_match if best_match else CACHED_SCENARIOS["general_fault"]
```

---

## 10. Orchestrator (`agent_loop.py`)

This is the single function that Team UI calls. It chains everything.

```python
# agents/agent_loop.py

import numpy as np
from .input_guard import is_valid_fault_input
from .diagnostic_agent import translate_fault_to_tensor, _sensor_id_to_column
from .capacity_agent import update_capacity, get_all_machine_statuses
from .floor_manager import issue_dispatch_orders
from .fallback_cache import match_scenario

# ── Offline mode toggle ──
# Set to True on first Gemini failure. Stays True for rest of session.
OFFLINE_MODE = False


def run_agent_loop(
    user_text: str,
    machine_id: int,
    base_window: np.ndarray,
    predict_rul_fn
) -> dict:
    """
    Full agent pipeline: Input → Diagnostic → Oracle → Capacity → Floor Manager.

    This is THE function Team UI calls.

    Args:
        user_text:      Professor's fault description string
        machine_id:     Which machine to target (1-5)
        base_window:    (50, 18) numpy array — current sensor baseline
        predict_rul_fn: The predict_rul function (dummy_oracle or dl_engine.inference)

    Returns:
        dict with all results — see schema below
    """
    global OFFLINE_MODE

    # ════════════════════════════════════════
    # Step 0: Input Guard (no LLM, no network)
    # ════════════════════════════════════════
    valid, reason = is_valid_fault_input(user_text)
    if not valid:
        return {
            "valid": False,
            "rejection_reason": reason,
            "spike": None,
            "rul": None,
            "capacity_report": None,
            "dispatch_orders": None,
            "machine_statuses": get_all_machine_statuses(),
            "used_fallback": False,
        }

    # ════════════════════════════════════════
    # Step 1: Diagnostic Agent (Gemini or fallback)
    # ════════════════════════════════════════
    if OFFLINE_MODE:
        scenario = match_scenario(user_text)
        spike_obj = scenario["diagnostic_spike"]
        spike_dict = spike_obj.model_dump()
        injected = _inject_spike(base_window, spike_dict)
        used_fallback = True
    else:
        injected, spike_dict, used_fallback = translate_fault_to_tensor(base_window, user_text)
        if used_fallback:
            OFFLINE_MODE = True

    # ════════════════════════════════════════
    # Step 2: DL Oracle (Team DL's function)
    # ════════════════════════════════════════
    rul = predict_rul_fn(injected)

    # ════════════════════════════════════════
    # Step 3: Capacity Agent (pure Python)
    # ════════════════════════════════════════
    capacity_report = update_capacity(machine_id, rul)

    # ════════════════════════════════════════
    # Step 4: Floor Manager (Gemini or fallback)
    # ════════════════════════════════════════
    if OFFLINE_MODE:
        scenario = match_scenario(user_text)
        dispatch = scenario["floor_manager_response"].format(**capacity_report)
    else:
        try:
            dispatch = issue_dispatch_orders(capacity_report)
        except Exception:
            OFFLINE_MODE = True
            scenario = match_scenario(user_text)
            dispatch = scenario["floor_manager_response"].format(**capacity_report)

    # ════════════════════════════════════════
    # Return everything
    # ════════════════════════════════════════
    return {
        "valid": True,
        "rejection_reason": "",
        "spike": spike_dict,
        "rul": rul,
        "capacity_report": capacity_report,
        "dispatch_orders": dispatch,
        "machine_statuses": get_all_machine_statuses(),
        "used_fallback": OFFLINE_MODE,
    }


def _inject_spike(base_window: np.ndarray, spike: dict) -> np.ndarray:
    """Helper for offline-mode spike injection."""
    injected = base_window.copy()
    col = _sensor_id_to_column(spike['sensor_id'])
    for pos in spike['affected_window_positions']:
        injected[pos, col] = spike['spike_value']
    return injected


def reset_offline_mode():
    """Call this to re-enable Gemini after a disconnect is resolved."""
    global OFFLINE_MODE
    OFFLINE_MODE = False
```

### 10.1 Return Schema — What Team UI Receives

```python
{
    "valid": True,                    # Did input pass the guard?
    "rejection_reason": "",           # Why rejected (empty if valid)
    "spike": {                        # Agent 1 output (SensorSpike as dict)
        "sensor_id": "Xs4",
        "spike_value": 0.95,
        "affected_window_positions": [45, 46, 47, 48, 49],
        "fault_severity": "HIGH",
        "plain_english_summary": "Bearing temp critical."
    },
    "rul": 12.0,                     # DL Oracle output
    "capacity_report": {              # Agent 2 output
        "machine_id": 4,
        "machine_name": "Lathe-Delta",
        "status": "OFFLINE",
        "rul": 12.0,
        "total_T": 32.0,
        "total_PD": 595,
        "machine_req": 18.594,
        "capacity_pct": 80.0,
        "breakeven_risk": True,
    },
    "dispatch_orders": "[Floor Manager] Machine 4 OFFLINE...",   # Agent 3 output
    "machine_statuses": [             # All 5 machines for dashboard
        {"id": 1, "name": "CNC-Alpha", "status": "ONLINE", "rul": 999.0, ...},
        ...
    ],
    "used_fallback": False,           # True = Gemini was unavailable
}
```

---

## 11. Your Day-by-Day Checklist

| Day | Task | Done? |
|---|---|---|
| 1 | Install deps: `pip install google-genai pydantic numpy`. Set up `GEMINI_API_KEY`. Verify API works with test call. Write `prompts.py` with both system prompts. | ☐ |
| 2 | Write `schemas.py` with `SensorSpike` model. Build `capacity_agent.py` with 3-state logic. Unit-test against the hand-verification table (Section 7.4). | ☐ |
| 3 | Build `diagnostic_agent.py` with structured output + domain validation + retry + fallback. Test with 5+ fault descriptions. Verify `SensorSpike` objects parse correctly. | ☐ |
| 4 | Build `floor_manager.py` with template fallback. Test full chain: fault → spike → dummy RUL → capacity → dispatch. | ☐ |
| 5 | Build `input_guard.py`. Wire everything into `agent_loop.py`. **Phase 1 integration with Team UI** — `run_agent_loop()` callable from the terminal app. | ☐ |
| 6 | Stress-test: 10 valid chaos inputs + 5 nonsensical inputs + 3 edge cases (empty, very long, unicode). Fix any failures. | ☐ |
| 7 | Build `fallback_cache.py` with 5 scenarios. Test offline mode: disconnect wifi, verify full pipeline works with cached responses and live capacity numbers. | ☐ |
| 8 | Draft 8–10 demo scenarios the professor might type. Run all of them. Tune prompts if Gemini returns weak sensor mappings. | ☐ |
| 9 | Final `agent_loop.py` with offline mode toggle. Performance testing — measure end-to-end latency per chaos input. | ☐ |
| **10** | **INTEGRATION: Team DL swaps dummy oracle → real model. Test full pipeline end-to-end.** | ☐ |
| 11 | Integration bug fixes. Offline fallback retest. | ☐ |
| 12–13 | Full rehearsals with all teams. | ☐ |
| 14 | Final bug fixes only. | ☐ |

---

## 12. Testing Checklist

Run every one of these before declaring your code done:

```
INPUT GUARD:
  [ ] "bearing temperature spike on Machine 3"   → PASS (valid)
  [ ] "hello world"                               → REJECTED
  [ ] "what's for lunch"                          → REJECTED
  [ ] "x"                                         → REJECTED (too short)
  [ ] "a" * 600                                   → REJECTED (too long)

DIAGNOSTIC AGENT (STRUCTURED OUTPUT):
  [ ] "bearing overheat"         → SensorSpike with sensor_id in VALID_SENSORS
  [ ] "pressure surge"           → SensorSpike, spike_value in [0, 1]
  [ ] "vibration anomaly"        → SensorSpike, positions all in [0, 49]
  [ ] "motor speed fluctuation"  → SensorSpike, severity in LOW/MEDIUM/HIGH
  [ ] "coolant leak near pump"   → SensorSpike (Gemini picks appropriate sensor)
  [ ] Verify: returned object is a valid dict (spike_dict), not raw Pydantic

CAPACITY AGENT:
  [ ] All ONLINE                        → capacity=100%, total_T=40.0
  [ ] Machine 4 DEGRADED (RUL=22)       → capacity=90%,  total_T=36.0
  [ ] Machine 4 OFFLINE  (RUL=12)       → capacity=80%,  total_T=32.0
  [ ] Machines 3+4 OFFLINE              → capacity=60%,  total_T=24.0
  [ ] reset_all() restores everything   → capacity=100%

FLOOR MANAGER:
  [ ] Output starts with "[Floor Manager]"
  [ ] Output is ≤ 4 sentences
  [ ] Output does NOT invent numbers (only uses report values)
  [ ] OFFLINE scenario mentions rerouting
  [ ] DEGRADED scenario mentions reduced load

OFFLINE FALLBACK:
  [ ] Disconnect wifi → "bearing overheat" → falls back, pipeline completes
  [ ] Disconnect wifi → "pressure surge"   → falls back, different scenario matched
  [ ] Verify dispatch orders contain REAL capacity numbers (not placeholders)
  [ ] Verify [OFFLINE MODE] tag appears in spike summary

END-TO-END (agent_loop.py):
  [ ] Valid input → all fields populated in return dict
  [ ] Invalid input → valid=False, rejection_reason filled
  [ ] Return dict matches schema in Section 10.1 exactly
  [ ] Latency: < 5s per chaos input (with Gemini), < 100ms (offline fallback)
```

---

## 13. Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'types' from 'google.genai'` | Old version of google-genai | `pip install --upgrade google-genai` — need ≥1.0 |
| `ValidationError: field required` from Pydantic | Gemini returned JSON missing a field despite schema | Shouldn't happen with structured output. If it does, it's a Gemini bug — retry handles it. |
| Gemini returns `sensor_id: "Xs17"` | Hallucinated sensor ID not in valid set | Domain validation catches this → retry → fallback if persistent |
| `GEMINI_API_KEY` not found | Environment variable not set | `export GEMINI_API_KEY="your-key"` in shell |
| Rate limit (429) from Gemini | Too many rapid calls during stress testing | Add 1s sleep between test calls. In production, the fallback cache absorbs this. |
| Capacity agent math is wrong | Floating point or logic error | Cross-check against the hand-verification table in Section 7.4 |
| Floor Manager invents numbers | System prompt not strong enough | Strengthen the "NEVER invent numbers" rule. Add few-shot examples to the prompt. |
| Offline mode never deactivates | `OFFLINE_MODE` stays True after wifi reconnects | Call `reset_offline_mode()` or restart the app. By design, we don't auto-retry mid-demo. |

---

## 14. Package Summary

Complete `requirements.txt` entries for Team Agent:

```
google-genai>=1.0.0
pydantic>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0    # optional — for .env file loading
```

**Total dependency footprint:** 4 packages + their transitive deps. Compare this to LangChain which pulls in 50+ packages. For a 14-day sprint, minimal dependencies = fewer things that can break.

---

*You own `/agents/*`. Your deliverable is `run_agent_loop()` — a single function that takes chaos and returns order. Pydantic handles the structure, domain validation handles the ranges, fallback handles the failures. Ship it reliable.*