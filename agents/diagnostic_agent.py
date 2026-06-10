# agents/diagnostic_agent.py

import os
import time
from pathlib import Path


import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from .schemas import SensorSpike, FaultSeverity
from .prompts import DIAGNOSTIC_SYSTEM_PROMPT
from .log_config import get_logger


# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Get API key for diagnostic agent
api_key = os.getenv("GEMINI_API_KEY_DIAGNOSTIC")

if not api_key:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY_DIAGNOSTIC not found in environment. "
        "Diagnostic agent will use deterministic fallback spikes.",
        stacklevel=2,
    )
    client = None
else:
    # print(f"Loaded API key for DIAGNOSTIC agent: {bool(api_key)}")  # commented — noisy before TUI
    client = genai.Client(api_key=api_key)

# ── Structured logger ──────────────────────────────────────────────────────────
log = get_logger("diagnostic")


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_RETRIES   = 2
VALID_SENSORS = {f"W{i}" for i in range(4)} | {f"Xs{i}" for i in range(14)}

# ── Sensor ID → tensor column index ───────────────────────────────────────────
# Feature order in the (50, 18) tensor: [W0, W1, W2, W3, Xs0, Xs1, ... Xs13]
SENSOR_TO_COL: dict[str, int] = {
    **{f"W{i}":  i     for i in range(4)},   # W0→0, W1→1, W2→2, W3→3
    **{f"Xs{i}": i + 4 for i in range(14)},  # Xs0→4, Xs1→5, ..., Xs13→17
}

# ── Deterministic fallback spikes ─────────────────────────────────────────────
# Keyed by the dominant keyword in the fault description.
# Used when all Gemini retries fail or all return invalid sensor IDs.
# Values here are tuned for the CNN-LSTM to produce a meaningful RUL drop.
FALLBACK_SPIKES: dict[str, SensorSpike] = {
    "temperature": SensorSpike(
        sensor_id="Xs2", spike_value=0.95,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Bearing temperature sensor Xs2 — critical thermal spike. [FALLBACK]"
    ),
    "bearing": SensorSpike(
        sensor_id="Xs2", spike_value=0.93,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Bearing temperature sensor Xs2 — overheat detected. [FALLBACK]"
    ),
    "pressure": SensorSpike(
        sensor_id="Xs4", spike_value=0.92,
        affected_window_positions=[40, 41, 43, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Oil pressure sensor Xs4 — abnormal surge reading. [FALLBACK]"
    ),
    "vibration": SensorSpike(
        sensor_id="Xs0", spike_value=0.88,
        affected_window_positions=[43, 44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="Vibration sensor Xs0 — oscillation above safe threshold. [FALLBACK]"
    ),
    "rpm": SensorSpike(
        sensor_id="W0", spike_value=0.89,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="Motor RPM sensor W0 — rotational speed anomaly. [FALLBACK]"
    ),
    "speed": SensorSpike(
        sensor_id="W0", spike_value=0.87,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="Motor RPM sensor W0 — drive fluctuation detected. [FALLBACK]"
    ),
    "coolant": SensorSpike(
        sensor_id="W3", spike_value=0.91,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Coolant flow sensor W3 — flow disruption detected. [FALLBACK]"
    ),
    "leak": SensorSpike(
        sensor_id="W3", spike_value=0.90,
        affected_window_positions=[43, 44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Coolant flow sensor W3 — possible seal failure. [FALLBACK]"
    ),
    "overload": SensorSpike(
        sensor_id="Xs6", spike_value=0.94,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Spindle load sensor Xs6 — machine overload condition. [FALLBACK]"
    ),
    "default": SensorSpike(
        sensor_id="Xs2", spike_value=0.93,
        affected_window_positions=[47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Sensor anomaly detected — defaulting to bearing thermal fault. [FALLBACK]"
    ),
}

# Keyword priority order — first match wins
FALLBACK_KEYWORD_ORDER = [
    "bearing", "temperature", "pressure", "vibration",
    "coolant", "leak", "rpm", "speed", "overload",
]


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_domain(spike: SensorSpike) -> tuple[bool, str]:
    """
    Validates domain-specific constraints that Pydantic cannot enforce.
    Called after every Gemini response, before accepting the spike.

    Pydantic handles: field presence, types, spike_value in [0,1].
    This handles: sensor_id must be a real sensor, positions in valid range.

    Returns:
        (True, "")              → spike is valid, use it
        (False, error_message)  → spike is invalid, log error and retry
    """
    if spike.sensor_id not in VALID_SENSORS:
        return False, (
            f"sensor_id '{spike.sensor_id}' is not a valid sensor. "
            f"Valid sensors are: W0–W3, Xs0–Xs13."
        )

    bad_positions = [p for p in spike.affected_window_positions if not (0 <= p <= 49)]
    if bad_positions:
        return False, (
            f"affected_window_positions contains out-of-range values: {bad_positions}. "
            f"All positions must be integers 0–49."
        )

    if len(spike.affected_window_positions) == 0:
        return False, "affected_window_positions is empty — must contain at least 1 position."

    if len(spike.affected_window_positions) > 10:
        return False, (
            f"affected_window_positions has {len(spike.affected_window_positions)} items — "
            f"maximum is 10."
        )

    # Warn about positions in first half of window (not a failure, just suspicious)
    early_positions = [p for p in spike.affected_window_positions if p < 25]
    if early_positions and len(early_positions) == len(spike.affected_window_positions):
        # All positions are in the first half — likely Gemini misunderstood the window
        # Still accept it (not a hard failure), but log a warning
        log.warning(
            "All spike positions are early in window (%s). "
            "Fault may not affect recent readings strongly.",
            early_positions,
        )

    return True, ""


# ── Fallback selection ─────────────────────────────────────────────────────────

def _get_fallback(user_text: str) -> SensorSpike:
    """
    Keyword-match the user's input to the best deterministic fallback.
    Checks keywords in priority order (FALLBACK_KEYWORD_ORDER).
    Returns a copy of the matching SensorSpike (not the original).
    """
    text_lower = user_text.lower()

    for keyword in FALLBACK_KEYWORD_ORDER:
        if keyword in text_lower:
            spike = FALLBACK_SPIKES[keyword]
            log.info("Fallback matched keyword: '%s' → %s", keyword, spike.sensor_id)
            return spike

    # No keyword matched — use default
    log.info("No keyword matched. Using default fallback.")
    return FALLBACK_SPIKES["default"]


# ── Sensor correlation map ─────────────────────────────────────────────────────
# When the primary sensor spikes, correlated sensors also degrade.
# Model probing showed single-sensor spikes barely move RUL, but 3+ sensor
# ramps produce dramatic drops (3-sensor ramp: RUL 71 → 8).
#
# CRITICAL: the CNN-LSTM is primarily sensitive to Xs2 (col 6) and Xs3 (col 7).
# These are the key degradation indicators in the N-CMAPSS turbofan data.
# Every fault type MUST include Xs2/Xs3 at some intensity — physically all
# machine faults eventually stress these thermal/pressure channels.
#
# Intensity controls the strength of the correlation, NOT the demo flow:
#   0.65–0.70 = direct thermal/pressure fault (strongest RUL impact)
#   0.50–0.60 = mechanically coupled fault (moderate impact)
#   0.40–0.50 = indirect/operating-condition fault (mild impact)

# ── Severity-driven escalation ────────────────────────────────────────────────
# Earlier versions used per-position caps to force a fixed 3-hit walkthrough.
# That layer ignored Gemini's severity classification — a "critical failure"
# prompt on a fresh machine produced the same outcome as "minor wobble".
# This layer is gone. The injection magnitude now flows directly from the
# upstream agents' decisions:
#
#   target_scaled = current + spike_value × correlation_intensity × multiplier
#
# Where the multiplier is selected from the Gemini-classified FaultSeverity:
#
#   LOW    → 0.15   (early warning — small RUL nudge, stays ONLINE)
#   MEDIUM → 0.26   (real fault — 3 hits walk through ONLINE→DEGRADED→OFFLINE)
#   HIGH   → 0.65   (catastrophic — single shot reaches OFFLINE on a fresh machine)
#
# Tuned together with the critical-trio cross-correlations (Xs2↔Xs3↔Xs4
# at 0.85–0.95 — see SENSOR_CORRELATIONS below) to hit the CNN-LSTM's
# probe-validated cliff points:
#
#   Hit 1 (MEDIUM, fresh):  Xs2≈0.31, Xs3≈0.33, Xs4≈0.28 → RUL ~65 (ONLINE)
#   Hit 2 (MEDIUM):          Xs2≈0.51, Xs3≈0.53, Xs4≈0.46 → RUL ~30 (DEGRADED)
#   Hit 3 (MEDIUM):          Xs2≈0.71, Xs3≈0.73, Xs4≈0.64 → RUL ~5  (OFFLINE)
#   Single-shot HIGH:        Xs2≈0.71, Xs3≈0.71, Xs4≈0.64 → RUL ~5  (OFFLINE)
#
# Earlier tunings (HIGH=1.00 then 0.75, MEDIUM=0.70 then 0.35) skipped
# DEGRADED on hit 2: the per-hit Xs2 delta was too large AND Xs4 didn't
# drag enough to suppress the cliff. The combined retune (smaller deltas,
# stronger Xs4 drag) lands hit 2 right in the DEGRADED window.
#
# Gemini decides severity from explicit wording in prompts.py:SEVERITY
# CLASSIFICATION (default MEDIUM; HIGH only for catastrophic words like
# "rupture"/"complete failure"; LOW for "minor"/"slight"/"wobble").
#
# Critical sensors (Xs2/Xs3/Xs4) still use FLAT FILL (all 50 rows at target)
# because the CNN-LSTM averages across the 50-row window — a linspace ramp
# averages out to a lower effective value and the model under-responds.
SEVERITY_MULTIPLIERS: dict[FaultSeverity, float] = {
    FaultSeverity.LOW:    0.15,
    FaultSeverity.MEDIUM: 0.35,   # Increased from 0.26 to land in DEGRADED reliably
    FaultSeverity.HIGH:   0.85,   # Increased from 0.65 to reach OFFLINE reliably
}

# Hard ceiling/floor — purely to keep scaled values within the
# scaler's [0, 1] domain. Not a demo-flow knob.
SCALED_MAX: float = 0.98
SCALED_MIN: float = 0.02

CRITICAL_SENSORS: set[str] = {"Xs2", "Xs3", "Xs4"}

# ─────────────────────────────────────────────────────────────────────────────
# Sensor polarity — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
# Sensors that decrease with wear in the physics-informed simulator.
# For these, injection SUBTRACTS from current_scaled instead of adding.
# N-CMAPSS (turbofan) sensors all increase with degradation, so this set is
# only meaningful when get_loaded_variant() == "simulator".
#
# Derived COL indices below are exported for dl_engine/inference.py to use
# in get_healthy_baseline() — avoids duplicating the [0, 3, 8, 12] literal.
SIMULATOR_DROPPING_SENSORS: set[str] = {"W0", "W3", "Xs4", "Xs8"}
SIMULATOR_DROPPING_COLS: list[int] = sorted(SENSOR_TO_COL[s] for s in SIMULATOR_DROPPING_SENSORS)


def _is_dropping(sensor_id: str) -> bool:
    """Return True if this sensor's polarity is negative (drops with wear).

    Self-loads the model if it hasn't been loaded yet — otherwise
    get_loaded_variant() returns None and dropping sensors would be silently
    treated as additive (turbofan-style). See BUG_REPORT_2026-05-25.md
    CRIT-4 for the failure mode.

    NOTE: the `from dl_engine.inference import ...` below is intentionally
    inside the function body, NOT module-level. Reason: dl_engine.inference
    lazy-imports SIMULATOR_DROPPING_COLS from THIS module (see
    inference.py::get_healthy_baseline). A module-level import here would
    close the cycle at import time. Python's import cache makes the per-call
    cost trivial (just sys.modules dict lookups). See BUG_REPORT LOW-14 for
    why the "move to module level" suggestion was rejected.
    """
    from dl_engine.inference import get_loaded_variant, load_model
    if get_loaded_variant() is None:
        load_model()
    return get_loaded_variant() == "simulator" and sensor_id in SIMULATOR_DROPPING_SENSORS


# Format: primary_sensor → [(correlated_sensor, intensity_fraction), ...]
# Canonical sensor map (matches terminal/layout.py SENSOR_DISPLAY_NAMES):
#   Xs2 = Bearing Temp,  Xs3 = Motor Temp,  Xs4 = Oil Pressure  (the critical trio)
#   Xs0/Xs1 = Vibration X/Y, Xs5 = Oil Temp, Xs6 = Spindle Load, Xs7 = Torque,
#   Xs8 = Hydraulic PSI, Xs9 = Coolant Temp, Xs10 = Ambient Temp,
#   Xs11 = Current Amps, Xs12 = Acoustic dB, Xs13 = Cycle Time
#   W0 = Motor RPM, W1 = Feed Rate, W2 = Power kW, W3 = Coolant Flow
#
# Every primary drags Xs2/Xs3/Xs4 along so the CNN-LSTM cliff is reachable —
# the model's RUL sensitivity comes from Xs2/Xs3 (cliff drivers) with Xs4
# co-elevation shifting the cliff position. This is a physics property of
# the trained network, not the sensor names.
SENSOR_CORRELATIONS: dict[str, list[tuple[str, float]]] = {
    # ── Critical thermal trio cross-correlate STRONGLY ────────────────────
    # Tightened to 0.85–0.95 so that when any of Xs2/Xs3/Xs4 is the primary,
    # the other two flat-fill nearly as hard. This is what keeps the
    # CNN-LSTM's cliff suppressed and gives a clean DEGRADED zone on hit 2
    # of a MEDIUM walkthrough (see SEVERITY_MULTIPLIERS docstring above).
    "Xs2":  [("Xs3", 0.95), ("Xs4", 0.90)],   # Bearing Temp → Motor Temp + Oil Pressure
    "Xs3":  [("Xs2", 0.95), ("Xs4", 0.90)],   # Motor Temp  → Bearing Temp + Oil Pressure
    "Xs4":  [("Xs2", 0.85), ("Xs3", 0.85)],   # Oil Pressure → Bearing Temp + Motor Temp
    # ── Oil Temp → lubricant cascade ──────────────────────────────────────
    "Xs5":  [("Xs2", 0.62), ("Xs3", 0.55), ("Xs4", 0.55)],
    # ── Vibration → friction heat → thermal cascade ───────────────────────
    "Xs0":  [("Xs2", 0.60), ("Xs3", 0.52), ("Xs4", 0.45), ("Xs1", 0.50)],
    "Xs1":  [("Xs2", 0.60), ("Xs3", 0.52), ("Xs4", 0.45), ("Xs0", 0.50)],
    # ── Mechanical load → friction → thermal ──────────────────────────────
    "Xs6":  [("Xs2", 0.58), ("Xs3", 0.52), ("Xs4", 0.48)],   # Spindle Load
    "Xs7":  [("Xs2", 0.62), ("Xs3", 0.55), ("Xs4", 0.48)],   # Torque
    # ── Hydraulic PSI → pressure system → thermal stress ──────────────────
    "Xs8":  [("Xs4", 0.65), ("Xs2", 0.55), ("Xs3", 0.48)],   # leads with Xs4 (oil pressure)
    # ── Coolant Temp/Flow → cooling loss → thermal cascade ────────────────
    "Xs9":  [("Xs2", 0.62), ("Xs3", 0.55), ("Xs4", 0.45)],   # Coolant Temp rising
    "W3":   [("Xs9", 0.50), ("Xs2", 0.58), ("Xs3", 0.50), ("Xs4", 0.42)],   # Coolant Flow drop
    # ── Ambient → background environmental thermal stress ─────────────────
    "Xs10": [("Xs2", 0.52), ("Xs3", 0.45), ("Xs4", 0.40)],
    # ── Electrical → motor stress → thermal ───────────────────────────────
    "Xs11": [("Xs2", 0.55), ("Xs3", 0.50), ("Xs4", 0.45)],   # Current Amps
    "Xs13": [("Xs2", 0.50), ("Xs3", 0.45), ("Xs4", 0.40)],   # Cycle Time
    # ── Acoustic anomaly → mechanical → friction heat ─────────────────────
    "Xs12": [("Xs2", 0.55), ("Xs3", 0.48), ("Xs4", 0.42)],
    # ── Operating conditions → off-design thermal equilibrium ─────────────
    "W0":   [("Xs2", 0.52), ("Xs3", 0.48), ("Xs4", 0.42)],   # Motor RPM
    "W1":   [("Xs2", 0.48), ("Xs3", 0.42), ("Xs4", 0.38)],   # Feed Rate
    "W2":   [("Xs2", 0.55), ("Xs3", 0.48), ("Xs4", 0.42)],   # Power kW
}


def _severity_multiplier(severity: FaultSeverity | str) -> float:
    """Map a FaultSeverity (or its str value) to the injection multiplier."""
    if isinstance(severity, str):
        try:
            severity = FaultSeverity(severity)
        except ValueError:
            return SEVERITY_MULTIPLIERS[FaultSeverity.MEDIUM]
    return SEVERITY_MULTIPLIERS.get(severity, SEVERITY_MULTIPLIERS[FaultSeverity.MEDIUM])


def _inject_spike(
    base_window: np.ndarray,
    spike: SensorSpike,
    multiplier_override: float | None = None,
) -> np.ndarray:
    """
    Inject a fault into a COPY of base_window using ADDITIVE multi-sensor
    injection. The injection magnitude is driven by Gemini's severity
    classification — there are no hand-coded per-hit caps.

    Each sensor's new scaled position is:
        target = current + spike_value × intensity × severity_multiplier
    bounded only by SCALED_MAX (0.98) to stay inside the scaler domain.

    Critical sensors (Xs2/Xs3/Xs4) use FLAT FILL (all 50 rows at target)
    because the CNN-LSTM averages across the 50-row window — a linspace ramp
    averages to a lower effective value and the model under-responds.

    Non-critical sensors use a gradual RAMP (linspace) for visual realism.

    The base_window carries accumulated damage from previous faults via
    factory_state._build_window() padding with h[-1] (latest reading), so
    severity-driven injection is inherently cumulative across multiple hits.

    Args:
        base_window:         (50, 18) float32 array — sensor readings in raw units
        spike:               validated SensorSpike object
        multiplier_override: optional float in [0, 1] that REPLACES the
                             severity-table lookup. Used by the continuous-output
                             research strategy (`strategy_agentic_continuous` in
                             `research/baselines.py`) to bypass the categorical
                             LOW/MEDIUM/HIGH → multiplier table and use a
                             directly-LLM-emitted continuous multiplier.
                             Production callers leave this None.

    Returns:
        (50, 18) float32 array — copy with severity-driven correlated injection
    """
    from dl_engine.inference import raw_value_for_scaled, get_scaler_ranges

    injected = base_window.copy()
    primary_col = SENSOR_TO_COL[spike.sensor_id]
    if multiplier_override is not None:
        # Clamp into the same domain SEVERITY_MULTIPLIERS uses (defensive).
        multiplier = max(0.0, min(1.0, float(multiplier_override)))
    else:
        multiplier = _severity_multiplier(spike.fault_severity)

    ranges = get_scaler_ranges()

    def _current_scaled(col: int, raw_val: float) -> float:
        """Convert raw sensor value to [0,1] scaled position."""
        lo  = float(ranges["min"][col])
        rng = float(ranges["range"][col])
        return (raw_val - lo) / rng if rng > 0 else 0.0

    # ── Primary sensor: severity-driven additive injection ────────────────
    raw_start      = float(injected[0, primary_col])
    current_scaled = _current_scaled(primary_col, raw_start)
    delta          = spike.spike_value * multiplier

    if _is_dropping(spike.sensor_id):
        target_scaled = max(SCALED_MIN, current_scaled - delta)
    else:
        target_scaled = min(SCALED_MAX, current_scaled + delta)

    raw_end = raw_value_for_scaled(primary_col, target_scaled)

    if spike.sensor_id in CRITICAL_SENSORS:
        injected[:, primary_col] = raw_end
    else:
        ramp = np.linspace(raw_start, raw_end, 50).astype(np.float32)
        injected[:, primary_col] = ramp

    log.debug(
        "Spike inject: %s (col %d) %s %.1f → %.1f "
        "(scaled %.2f → %.2f, severity=%s ×%.2f)",
        spike.sensor_id, primary_col,
        "FLAT" if spike.sensor_id in CRITICAL_SENSORS else "RAMP",
        raw_start, raw_end, current_scaled, target_scaled,
        spike.fault_severity.value, multiplier,
    )

    # ── Correlated sensors: same severity multiplier, scaled by intensity ─
    correlations = SENSOR_CORRELATIONS.get(spike.sensor_id, [])
    for corr_sensor_id, intensity in correlations:
        corr_col     = SENSOR_TO_COL[corr_sensor_id]
        corr_start   = float(injected[0, corr_col])
        corr_current = _current_scaled(corr_col, corr_start)
        corr_delta   = spike.spike_value * intensity * multiplier

        if _is_dropping(corr_sensor_id):
            corr_target = max(SCALED_MIN, corr_current - corr_delta)
        else:
            corr_target = min(SCALED_MAX, corr_current + corr_delta)

        corr_end = raw_value_for_scaled(corr_col, corr_target)

        if corr_sensor_id in CRITICAL_SENSORS:
            injected[:, corr_col] = corr_end
        else:
            corr_ramp = np.linspace(corr_start, corr_end, 50).astype(np.float32)
            injected[:, corr_col] = corr_ramp

        log.debug(
            "  + correlated %s (col %d) %s → scaled %.2f→%.2f "
            "(intensity %.0f%%, ×%.2f)",
            corr_sensor_id, corr_col,
            "FLAT" if corr_sensor_id in CRITICAL_SENSORS else "RAMP",
            corr_current, corr_target, intensity * 100, multiplier,
        )

    return injected


# ── Main entry point ───────────────────────────────────────────────────────────

def translate_fault_to_tensor(
    base_window: np.ndarray,
    user_text: str,
) -> tuple[np.ndarray, dict, bool]:
    """
    Agent 1 public interface. Called by agent_loop.py.

    Converts a plain-English fault description into a modified sensor tensor
    by asking Gemini to identify the affected sensor and spike parameters,
    then injecting those values into the base window.

    Args:
        base_window: (50, 18) float32 numpy array — current sensor baseline
        user_text:   professor's fault description

    Returns:
        injected_window: (50, 18) float32 numpy array with spike applied
        spike_dict:      SensorSpike fields as plain dict (for logging/UI)
        used_fallback:   True if Gemini failed and hardcoded fallback was used
    """
    # If client is None (no API key), skip Gemini entirely
    if client is None:
        log.warning("No API key — skipping Gemini, using deterministic fallback.")
        spike = _get_fallback(user_text)
        injected = _inject_spike(base_window, spike)
        return injected, spike.model_dump(mode="json"), True

    spike: SensorSpike | None = None
    last_error: str = ""

    for attempt in range(MAX_RETRIES + 1):

        # ── Build the prompt ──────────────────────────────────────────────────
        # On retries: inject the specific validation error so Gemini learns
        # what went wrong and corrects it, rather than repeating the same mistake.
        if attempt == 0:
            prompt_contents = (
                f"{DIAGNOSTIC_SYSTEM_PROMPT}\n\n"
                f"Fault description: {user_text}"
            )
        else:
            prompt_contents = (
                f"{DIAGNOSTIC_SYSTEM_PROMPT}\n\n"
                f"Fault description: {user_text}\n\n"
                f"CORRECTION REQUIRED (attempt {attempt + 1} of {MAX_RETRIES + 1}):\n"
                f"Your previous response was rejected for this reason: {last_error}\n"
                f"Please fix this specific issue and return a corrected response."
            )

        # ── Call Gemini ───────────────────────────────────────────────────────
        try:
            t_call = time.time()
            log.info("Gemini call attempt %d/%d  model=gemini-2.5-flash", attempt + 1, MAX_RETRIES + 1)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SensorSpike,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            api_ms = round((time.time() - t_call) * 1000, 1)
            log.info("Gemini responded in %.0fms", api_ms)

            candidate = SensorSpike.model_validate_json(response.text)

            is_valid, error = _validate_domain(candidate)
            if is_valid:
                spike = candidate
                log.info(
                    "✓ Attempt %d ACCEPTED: sensor=%s  value=%.2f  severity=%s  positions=%s",
                    attempt + 1, spike.sensor_id, spike.spike_value,
                    spike.fault_severity.value, spike.affected_window_positions,
                )
                break
            else:
                last_error = error
                log.warning("✗ Attempt %d domain validation fail: %s", attempt + 1, error)

        except ValidationError as e:
            last_error = f"Pydantic validation error: {e}"
            log.warning("✗ Attempt %d Pydantic fail: %s", attempt + 1, e)

        except Exception as e:
            last_error = f"API error: {e}"
            log.error("✗ Attempt %d API fail: %s", attempt + 1, e)

    # ── Fallback if all attempts failed ───────────────────────────────────────
    used_fallback = False
    if spike is None:
        log.warning("All %d Gemini attempts failed. Using deterministic fallback.", MAX_RETRIES + 1)
        spike = _get_fallback(user_text)
        used_fallback = True

    # ── Inject into tensor ────────────────────────────────────────────────────
    injected = _inject_spike(base_window, spike)

    # mode="json" serialises the FaultSeverity enum as its str value ("HIGH"),
    # so downstream consumers (log lines, UI comms pane) see "HIGH" not
    # "FaultSeverity.HIGH". The fields are otherwise JSON-friendly already.
    return injected, spike.model_dump(mode="json"), used_fallback
