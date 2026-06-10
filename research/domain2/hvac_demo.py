"""
research/domain2/hvac_demo.py

Building HVAC monitoring as a second-domain instantiation of the ForgeMind
agentic-pipeline architecture. No CNN-LSTM, no N-CMAPSS data — just a 5-zone
physics-based simulator and Gemini structured output.

The point is to show that the architecture (Input Guard → Diagnostic Agent →
Predictor → Capacity Math → Dispatch) is not tied to predictive maintenance.

What this demo proves:
  - The structured-output diagnostic agent works for any sensor-named domain
  - The Pydantic schema + severity classification pattern transfers
  - The capacity-math pattern (aggregate health over multiple units) transfers
  - The dispatch-template pattern transfers
  - The SAME categorical-to-quantitative impedance mismatch shows up here too
    (Gemini's severity label vs simulator's continuous comfort response)

What this demo does NOT prove:
  - That the architecture handles fundamentally different control problems
    (e.g. continuous setpoint adjustment vs categorical fault injection)
  - That the architecture scales to 100s of units / 1000s of signals

Usage:
    python -m research.domain2.hvac_demo                # run the bundled scenarios
    python -m research.domain2.hvac_demo --interactive  # type your own faults
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from pydantic import BaseModel, Field
from enum import Enum

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Zone state + simulator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoneState:
    """One HVAC zone. All sensors in their natural physical units."""
    zone_id: int
    zone_name: str
    temp_celsius: float = 22.0       # comfort: 20-24
    humidity_pct: float = 45.0       # comfort: 30-60
    airflow_cfm: float = 350.0       # comfort: 300-400
    co2_ppm: float = 600.0           # comfort: <800
    occupancy: int = 10              # affects load
    comfort_score: float = 100.0     # 0-100, recomputed by simulator
    status: str = "COMFORTABLE"      # COMFORTABLE / DEGRADED / UNCOMFORTABLE


ZONES_DEFAULT = [
    ZoneState(1, "Lobby"),
    ZoneState(2, "Office Floor A"),
    ZoneState(3, "Office Floor B"),
    ZoneState(4, "Conference Wing"),
    ZoneState(5, "Server Room"),
]


def compute_comfort(zone: ZoneState) -> float:
    """
    Deterministic comfort score. Returns 0-100.

    100 = ideal (temp 22, humidity 45, airflow 350, co2 <600).
    Linear penalty for each axis outside its comfort band, then 0-clamped.

    This is the "predictor" — analogous to the CNN-LSTM's RUL output, but
    transparent and differentiable. It still has a cliff-shaped response
    (because penalties compound multiplicatively for severe deviations)
    so the same categorical-to-quantitative interface problem applies.
    """
    score = 100.0
    # Temperature penalty: 5 points per degree outside 20-24
    if zone.temp_celsius < 20:
        score -= (20 - zone.temp_celsius) * 5
    elif zone.temp_celsius > 24:
        score -= (zone.temp_celsius - 24) * 5

    # Humidity penalty: 1.5 points per % outside 30-60
    if zone.humidity_pct < 30:
        score -= (30 - zone.humidity_pct) * 1.5
    elif zone.humidity_pct > 60:
        score -= (zone.humidity_pct - 60) * 1.5

    # Airflow penalty: 0.3 points per CFM below 250 (overprovisioned is fine)
    if zone.airflow_cfm < 250:
        score -= (250 - zone.airflow_cfm) * 0.3

    # CO2 penalty: 0.1 points per ppm above 800
    if zone.co2_ppm > 800:
        score -= (zone.co2_ppm - 800) * 0.1

    # Compounding cliff: if multiple axes are bad simultaneously, multiply penalty
    deviations = [
        abs(zone.temp_celsius - 22) / 4,
        abs(zone.humidity_pct - 45) / 30,
        max(0, 350 - zone.airflow_cfm) / 100,
        max(0, zone.co2_ppm - 800) / 400,
    ]
    n_severe = sum(1 for d in deviations if d > 1.0)
    if n_severe >= 2:
        score -= 20 * (n_severe - 1)  # cliff: each additional severe axis costs 20

    return max(0.0, min(100.0, score))


def classify_zone(comfort: float) -> str:
    if comfort >= 70:
        return "COMFORTABLE"
    if comfort >= 40:
        return "DEGRADED"
    return "UNCOMFORTABLE"


def update_zone(zones: list[ZoneState], spike: "HvacSpike") -> ZoneState:
    """Apply a spike to a zone and return its updated state."""
    zone = next(z for z in zones if z.zone_id == spike.zone_id)
    # ── Severity multiplier — DOMAIN-SPECIFIC, do not generalise ──────────
    # These multipliers are tuned to the HVAC simulator's comfort-score
    # response curve (defined in compute_comfort() below), NOT to the main
    # ForgeMind pipeline. The main pipeline uses
    # {LOW: 0.15, MEDIUM: 0.35, HIGH: 0.85} for the CNN-LSTM's RUL response.
    # The HVAC simulator's response is roughly 2× less sensitive per unit
    # delta because comfort penalties are compounding-but-bounded rather
    # than cliff-shaped, so HVAC needs larger multipliers to reach
    # UNCOMFORTABLE on a fresh zone. Keeping these values separate is
    # intentional — re-using the CNN-LSTM-tuned values here would produce
    # demos that under-react to faults.
    # See research/BUG_REPORT_2026-05-25.md MED-11.
    multiplier = {"LOW": 0.30, "MEDIUM": 0.85, "HIGH": 1.50}[spike.severity.value]

    if spike.signal == "temperature":
        zone.temp_celsius += spike.delta * multiplier
    elif spike.signal == "humidity":
        zone.humidity_pct += spike.delta * multiplier
    elif spike.signal == "airflow":
        zone.airflow_cfm += spike.delta * multiplier   # negative delta = airflow drop
    elif spike.signal == "co2":
        zone.co2_ppm += spike.delta * multiplier
    else:
        raise ValueError(f"unknown signal: {spike.signal}")

    zone.comfort_score = compute_comfort(zone)
    zone.status = classify_zone(zone.comfort_score)
    return zone


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic agent schema (Pydantic — same pattern as ForgeMind SensorSpike)
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class HvacSpike(BaseModel):
    """Structured-output schema for the HVAC diagnostic agent."""
    zone_id: int = Field(ge=1, le=5,
                          description="Zone ID 1-5. Lobby(1), Office A(2), Office B(3), Conference(4), Server Room(5).")
    signal: str = Field(
        description="One of: 'temperature' (degrees C delta), 'humidity' (% delta), "
                    "'airflow' (CFM delta, negative for drop), 'co2' (ppm delta)."
    )
    delta: float = Field(
        description="Signed magnitude in the signal's native unit. "
                    "Temperature: ±5-15. Humidity: ±10-30. Airflow: -50 to -200 (drop). CO2: +100 to +1500."
    )
    severity: Severity = Field(description="LOW / MEDIUM / HIGH — multiplier on delta")
    summary: str = Field(description="One-sentence plain-English description")


HVAC_PROMPT = """\
You are a building-systems diagnostic translator. The user describes a comfort
or HVAC complaint in natural language, and you output a structured intervention
on one of 5 zones.

Zones:
  1. Lobby — entry traffic, variable load
  2. Office Floor A — fixed occupancy
  3. Office Floor B — fixed occupancy
  4. Conference Wing — burst occupancy during meetings
  5. Server Room — high heat load, tight tolerance

Signal map:
  - temperature → degrees Celsius offset from current
  - humidity    → percentage points offset from current
  - airflow     → CFM offset (use NEGATIVE for airflow drops/loss)
  - co2         → ppm offset

Severity rules (read carefully — default is MEDIUM):
  - HIGH only for: "freezing", "boiling", "outage", "failure", "broken",
                   "stuck", "shut down", "no airflow"
  - LOW only for: "slight", "minor", "barely", "tiny", "drift"
  - MEDIUM (default) for: "warm", "cold", "stuffy", "humid", "dry", "low flow",
                          "high co2", and any unqualified fault

Pick a SINGLE signal per spike. Choose the most relevant signal from the
user's wording.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Input guard (mirrors agents/input_guard.py)
# ─────────────────────────────────────────────────────────────────────────────

HVAC_KEYWORDS = {
    "hot", "cold", "warm", "cool", "freezing", "boiling", "stuffy", "humid",
    "dry", "airflow", "flow", "co2", "carbon", "air", "vent", "hvac",
    "temperature", "temp", "humidity", "zone", "office", "lobby", "conference",
    "server", "room", "outage", "failure", "broken", "stuck",
}


def validate_input(user_text: str) -> tuple[bool, str]:
    text = user_text.strip().lower()
    if len(text) < 5:
        return False, "Too short. Describe an HVAC complaint."
    if len(text) > 500:
        return False, "Too long. Keep complaints under 500 chars."
    if not any(kw in text for kw in HVAC_KEYWORDS):
        return False, "Unrecognized — try 'too hot in lobby' or 'high CO2 in office'."
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic agent (Gemini structured output OR deterministic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_with_gemini(user_text: str) -> Optional[HvacSpike]:
    """Try Gemini. Return None if unavailable so caller can fall back."""
    try:
        from google import genai
        from google.genai import types as gtypes
        api_key = os.environ.get("GEMINI_API_KEY_DIAGNOSTIC") \
               or os.environ.get("GEMINI_API_KEY_FLOOR_MANAGER")
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=HVAC_PROMPT + f"\n\nUser complaint: {user_text}\n\nOutput JSON only.",
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HvacSpike,
            ),
        )
        return HvacSpike.model_validate_json(response.text)
    except Exception as e:
        print(f"[hvac] Gemini call failed: {type(e).__name__}: {e}")
        return None


def diagnose_fallback(user_text: str) -> HvacSpike:
    """Keyword-only fallback for offline mode."""
    text = user_text.lower()
    zone_id = 2  # default Office Floor A
    if "lobby" in text: zone_id = 1
    elif "office b" in text or "floor b" in text: zone_id = 3
    elif "conference" in text: zone_id = 4
    elif "server" in text: zone_id = 5

    if "freezing" in text or "boiling" in text or "outage" in text:
        sev = Severity.HIGH
    elif "minor" in text or "slight" in text:
        sev = Severity.LOW
    else:
        sev = Severity.MEDIUM

    if any(kw in text for kw in ["hot", "warm", "boiling", "temp"]):
        signal, delta = "temperature", 6.0
    elif any(kw in text for kw in ["cold", "cool", "freezing"]):
        signal, delta = "temperature", -6.0
    elif any(kw in text for kw in ["humid", "humidity", "stuffy"]):
        signal, delta = "humidity", 20.0
    elif any(kw in text for kw in ["dry"]):
        signal, delta = "humidity", -20.0
    elif any(kw in text for kw in ["airflow", "flow", "vent"]):
        signal, delta = "airflow", -150.0
    elif any(kw in text for kw in ["co2", "carbon", "stale"]):
        signal, delta = "co2", 600.0
    else:
        signal, delta = "temperature", 5.0

    return HvacSpike(
        zone_id=zone_id, signal=signal, delta=delta, severity=sev,
        summary=f"[FALLBACK] {signal} fault in zone {zone_id}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Capacity math (aggregate building health)
# ─────────────────────────────────────────────────────────────────────────────

def building_report(zones: list[ZoneState]) -> dict:
    """Aggregate building comfort report — analogous to capacity_agent."""
    n_comfortable = sum(1 for z in zones if z.status == "COMFORTABLE")
    n_degraded    = sum(1 for z in zones if z.status == "DEGRADED")
    n_uncomf      = sum(1 for z in zones if z.status == "UNCOMFORTABLE")
    avg_comfort = sum(z.comfort_score for z in zones) / len(zones)
    building_pct = (n_comfortable / len(zones)) * 100
    return {
        "n_zones":         len(zones),
        "n_comfortable":   n_comfortable,
        "n_degraded":      n_degraded,
        "n_uncomfortable": n_uncomf,
        "avg_comfort":     round(avg_comfort, 1),
        "building_pct":    round(building_pct, 1),
        "breakeven_risk":  n_uncomf >= 2 or n_degraded + n_uncomf >= 3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch (deterministic template — analogous to floor_manager fallback)
# ─────────────────────────────────────────────────────────────────────────────

DISPATCH_TEMPLATES = {
    "UNCOMFORTABLE": (
        "[Facilities] {zone_name} is UNCOMFORTABLE (score {comfort_score:.0f}) — "
        "dispatch HVAC technician immediately. "
        "Building at {building_pct:.0f}% comfortable. "
        "{risk_note}"
    ),
    "DEGRADED": (
        "[Facilities] {zone_name} is DEGRADED (score {comfort_score:.0f}) — "
        "schedule HVAC inspection within the next 4 hours. "
        "Building at {building_pct:.0f}% comfortable. "
        "{risk_note}"
    ),
    "COMFORTABLE": (
        "[Facilities] {zone_name} is COMFORTABLE (score {comfort_score:.0f}) — "
        "no action required, continue monitoring. "
        "Building at {building_pct:.0f}% comfortable."
    ),
}


def render_dispatch(zone: ZoneState, building: dict) -> str:
    risk_note = "Comfort risk ACTIVE — escalate to facilities manager." if building["breakeven_risk"] else ""
    return DISPATCH_TEMPLATES[zone.status].format(
        zone_name=zone.zone_name,
        comfort_score=zone.comfort_score,
        building_pct=building["building_pct"],
        risk_note=risk_note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_hvac_pipeline(user_text: str, zones: list[ZoneState]) -> dict:
    ok, reason = validate_input(user_text)
    if not ok:
        return {"rejected": True, "rejection_reason": reason, "dispatch": ""}

    spike = diagnose_with_gemini(user_text) or diagnose_fallback(user_text)
    zone = update_zone(zones, spike)
    building = building_report(zones)
    dispatch = render_dispatch(zone, building)

    return {
        "rejected":      False,
        "spike":         spike.model_dump(mode="json"),
        "zone_state":    asdict(zone),
        "building":      building,
        "dispatch":      dispatch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo scenarios
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    # MEDIUM-severity defaults
    "lobby feels too warm",
    "office floor A getting humid",
    # HIGH-severity (catastrophic language)
    "server room chiller outage — temperature stuck climbing fast",
    "complete airflow failure in conference wing",
    # LOW-severity (minor language)
    "slight cold drift in office floor B",
]


def run_demo() -> None:
    zones = [ZoneState(z.zone_id, z.zone_name) for z in ZONES_DEFAULT]  # fresh copies

    print(f"\n=== HVAC Demo — second-domain instantiation ===\n")
    print(f"5 zones, all starting COMFORTABLE (score=100).\n")

    for i, complaint in enumerate(SCENARIOS, 1):
        print(f"--- Scenario {i}: \"{complaint}\" ---")
        result = run_hvac_pipeline(complaint, zones)
        if result["rejected"]:
            print(f"  REJECTED: {result['rejection_reason']}\n")
            continue
        z = result["zone_state"]
        b = result["building"]
        print(f"  Spike:    zone {result['spike']['zone_id']} · {result['spike']['signal']} "
              f"· delta {result['spike']['delta']:.1f} · severity {result['spike']['severity']}")
        print(f"  Zone:     {z['zone_name']:20s} comfort={z['comfort_score']:5.1f} status={z['status']}")
        print(f"  Building: {b['n_comfortable']}/{b['n_zones']} comfortable, "
              f"{b['n_degraded']} degraded, {b['n_uncomfortable']} uncomfortable")
        print(f"  Dispatch: {result['dispatch']}\n")


def run_interactive() -> None:
    zones = [ZoneState(z.zone_id, z.zone_name) for z in ZONES_DEFAULT]
    print(f"\n=== HVAC Interactive Demo ===\n")
    print("Type complaints. Ctrl+C to exit. Type 'reset' to restore comfort.\n")
    while True:
        try:
            text = input("complaint> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "reset":
            zones = [ZoneState(z.zone_id, z.zone_name) for z in ZONES_DEFAULT]
            print("  All zones reset.\n")
            continue
        result = run_hvac_pipeline(text, zones)
        print(json.dumps(result, indent=2, default=str))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_demo()


if __name__ == "__main__":
    main()
