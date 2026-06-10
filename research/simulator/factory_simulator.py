"""
research/simulator/factory_simulator.py

Physics-informed factory-floor simulator. Each machine has 6 internal
components whose wear states evolve over time; 18 sensors are physically
informed functions of those wear states with cross-coupling and noise.

The 18-sensor layout MATCHES the canonical UI map in
`terminal/layout.py::SENSOR_DISPLAY_NAMES`. That means:

    Xs2 (Bearing Temp)   driven by bearing_wear (primary), oil_seal_wear (suppresses)
    Xs3 (Motor Temp)     driven by motor_wear, bearing_wear (thermal coupling)
    Xs4 (Oil Pressure)   driven by oil_seal_wear (DROPS with wear, not rises)
    W0  (Motor RPM)      driven by motor_wear, belt_wear (instability)
    W3  (Coolant Flow)   driven by coolant_seal_wear (DROPS with wear)
    Xs0/Xs1 (Vibration)  driven by bearing_wear, belt_wear, spindle_wear
    Xs7 (Torque)         driven by spindle_wear, belt_wear
    Xs11 (Current Amps)  driven by motor_wear, spindle_load
    Xs12 (Acoustic dB)   driven by sum of all wear
    ...

So when the LLM diagnostic agent says "bearing temperature spike → Xs2", the
simulator's bearing-wear pressure is what gets injected — and the downstream
ML model trained on this simulator's output will respond appropriately.

Design philosophy:
- Deterministic-with-noise. Same seed → same trajectory.
- Component wears in [0, 1]. 0 = new, 1 = failed.
- Sensor readings in raw physical units (not normalized — that happens at scaler boundary).
- RUL is the ground-truth label: RUL = 100 * (1 - max(wear))^2  (failure when max wear == 1).
- Faults are injected via inject_fault(machine_id, sensor_id, severity) which routes
  to the component(s) most physically associated with that sensor.

== CALIBRATION & VALIDATION ==

**This simulator is not calibrated against real predictive-maintenance data.**
The physics couplings (bearing wear → temp +45°C, vibration +2.5 mm/s, motor
wear → current +10A, oil-seal wear → oil pressure -250 kPa, etc.) are chosen
to be *physically defensible* — bearing wear *does* cause rising temperature
and vibration; oil-seal failure *does* drop hydraulic pressure — but the
specific magnitudes and cross-coupling strengths are engineering judgments,
not empirically calibrated against a labelled bearing/motor dataset.

The simulator's role in this project is **methodological**: provide a
domain-coherent training distribution so the LLM-driven diagnostic agent can
inject faults that are physically meaningful (e.g. "bearing temperature spike"
actually raises bearing temp), unlike the borrowed N-CMAPSS turbofan model
where the LLM's named sensors had no relationship to the model's training
features. The simulator does NOT claim to predict real factory-machine
behaviour; it claims to be **internally consistent** with industrial-PdM
physics intuitions.

Defensibility of the major physics choices (cited from PdM survey literature):
- Bearing wear → thermal + vibration cascade: standard in REB (rolling-element
  bearing) degradation models. See e.g. PRONOSTIA dataset (FEMTO-ST).
- Oil-seal failure → pressure DROP + temperature RISE: hydraulic-system
  failure-mode literature, e.g. Mobley "Root Cause Failure Analysis" (1999).
- Motor degradation → current draw RISE + RPM instability: motor-current
  signature analysis (MCSA) — Thomson & Fenger (2001).
- Coolant-seal failure → flow DROP + temp RISE downstream: HVAC and machine-
  tool cooling-system fault models.
- Cross-coupling between bearing + motor thermal channels: justified by
  shared thermal mass of motor housing + bearing seat.

For full validation, the simulator's output would need to be compared
quantitatively against a labelled real-equipment degradation dataset
(PRONOSTIA bearings, MIMII industrial audio, or proprietary plant data).
This is listed as Extension #8 in `research/extensions_roadmap.md` —
without it, this is a "synthetic study" by construction. See
`research/BUG_REPORT_2026-05-25.md` INFO-19.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants — sensor layout, component->sensor mapping, baseline physical values
# ─────────────────────────────────────────────────────────────────────────────

# Sensor column order matches the rest of the project (see SENSOR_TO_COL in
# agents/diagnostic_agent.py).
SENSOR_NAMES = [
    "Motor_RPM",      # 0  W0
    "Feed_Rate",      # 1  W1
    "Power_kW",       # 2  W2
    "Coolant_Flow",   # 3  W3
    "Vibration_X",    # 4  Xs0
    "Vibration_Y",    # 5  Xs1
    "Bearing_Temp",   # 6  Xs2  ★ KEY
    "Motor_Temp",     # 7  Xs3  ★ KEY
    "Oil_Pressure",   # 8  Xs4  ★ KEY
    "Oil_Temp",       # 9  Xs5
    "Spindle_Load",   # 10 Xs6
    "Torque",         # 11 Xs7
    "Hydraulic_PSI",  # 12 Xs8
    "Coolant_Temp",   # 13 Xs9
    "Ambient_Temp",   # 14 Xs10
    "Current_Amps",   # 15 Xs11
    "Acoustic_dB",    # 16 Xs12
    "Cycle_Time",     # 17 Xs13
]

# Component names and what they physically represent
COMPONENTS = ["bearing", "motor", "oil_seal", "coolant_seal", "belt", "spindle"]

# Baseline sensor values in "healthy" condition (raw physical units chosen for
# realism — not load-bearing for the model, but readable in logs).
SENSOR_BASELINES = {
    "Motor_RPM":     1800.0,   # rpm
    "Feed_Rate":      150.0,   # mm/min
    "Power_kW":         8.5,   # kW
    "Coolant_Flow":    35.0,   # L/min
    "Vibration_X":      0.3,   # mm/s RMS
    "Vibration_Y":      0.3,   # mm/s RMS
    "Bearing_Temp":    55.0,   # °C
    "Motor_Temp":      65.0,   # °C
    "Oil_Pressure":   420.0,   # kPa
    "Oil_Temp":        50.0,   # °C
    "Spindle_Load":    45.0,   # %
    "Torque":          22.0,   # Nm
    "Hydraulic_PSI":  900.0,   # PSI
    "Coolant_Temp":    25.0,   # °C
    "Ambient_Temp":    22.0,   # °C
    "Current_Amps":    18.0,   # A
    "Acoustic_dB":     65.0,   # dB
    "Cycle_Time":      12.0,   # s
}

# How each component's wear (0→1) modifies each sensor's baseline value.
# Each entry is (sensor_name, slope) — sensor = baseline + slope * wear.
# Slopes calibrated so that at wear=1, sensors are in clearly-faulted territory.
# Slopes are SIGNED — Oil_Pressure drops, others rise, etc.
COMPONENT_SENSOR_EFFECTS: dict[str, list[tuple[str, float]]] = {
    "bearing": [
        ("Bearing_Temp",   45.0),   # 55 → 100 °C at full wear
        ("Vibration_X",     2.5),   # 0.3 → 2.8 mm/s RMS
        ("Vibration_Y",     2.3),
        ("Motor_Temp",     12.0),   # thermal coupling
        ("Acoustic_dB",    15.0),   # bearing whine
        ("Cycle_Time",      2.0),   # slight slowdown
    ],
    "motor": [
        ("Motor_Temp",     35.0),   # 65 → 100 °C
        ("Current_Amps",   10.0),   # 18 → 28 A
        ("Power_kW",        3.0),
        ("Motor_RPM",     -80.0),   # RPM instability/drop
        ("Acoustic_dB",     8.0),
    ],
    "oil_seal": [
        ("Oil_Pressure", -250.0),   # 420 → 170 kPa (DROPS with seal failure)
        ("Hydraulic_PSI", -400.0),
        ("Oil_Temp",       20.0),
        ("Bearing_Temp",   18.0),   # less lubrication → bearing heats
    ],
    "coolant_seal": [
        ("Coolant_Flow",  -20.0),   # 35 → 15 L/min
        ("Coolant_Temp",   25.0),   # 25 → 50 °C
        ("Motor_Temp",     15.0),   # less cooling → motor heats
        ("Ambient_Temp",    3.0),
    ],
    "belt": [
        ("Motor_RPM",     -60.0),   # belt slip → lower RPM
        ("Torque",         -8.0),   # transmission loss
        ("Vibration_X",     1.5),
        ("Vibration_Y",     1.4),
        ("Acoustic_dB",    10.0),
        ("Cycle_Time",      3.0),   # slowdown
    ],
    "spindle": [
        ("Spindle_Load",   35.0),   # 45 → 80%
        ("Torque",         15.0),
        ("Current_Amps",    6.0),
        ("Power_kW",        2.0),
        ("Cycle_Time",      4.0),
        ("Vibration_X",     0.8),
    ],
}

# Reverse map: which component(s) does each sensor name (from prompts.py /
# canonical UI map) primarily report on? Used by inject_fault() to translate
# "Xs2 (BearingTemp) spike" into "bearing component takes fault pressure".
SENSOR_TO_PRIMARY_COMPONENT = {
    "Xs2": "bearing",         # Bearing_Temp → bearing
    "Xs3": "motor",           # Motor_Temp → motor
    "Xs4": "oil_seal",        # Oil_Pressure → oil_seal
    "Xs5": "oil_seal",        # Oil_Temp → oil_seal
    "Xs8": "oil_seal",        # Hydraulic_PSI → oil_seal
    "Xs0": "bearing",         # Vibration_X → bearing (primary), belt (secondary)
    "Xs1": "bearing",         # Vibration_Y → bearing
    "Xs7": "spindle",         # Torque → spindle
    "Xs6": "spindle",         # Spindle_Load → spindle
    "Xs9": "coolant_seal",    # Coolant_Temp → coolant_seal
    "Xs10": "coolant_seal",   # Ambient_Temp → coolant_seal (proxy)
    "Xs11": "motor",          # Current_Amps → motor
    "Xs12": "bearing",        # Acoustic_dB → bearing (everything contributes but bearings dominate)
    "Xs13": "spindle",        # Cycle_Time → spindle/belt
    "W0":  "motor",           # Motor_RPM → motor (primary), belt (secondary)
    "W1":  "spindle",         # Feed_Rate → spindle
    "W2":  "motor",           # Power_kW → motor
    "W3":  "coolant_seal",    # Coolant_Flow → coolant_seal
}

SEVERITY_FAULT_PRESSURE = {
    "LOW":    0.002,   # adds 0.002 wear/timestep — slow degradation
    "MEDIUM": 0.010,   # adds 0.010 wear/timestep — noticeable degradation
    "HIGH":   0.045,   # adds 0.045 wear/timestep — catastrophic, fails fast
}

# Base wear rates per timestep (per component) — natural aging without faults
BASE_WEAR_RATES = {
    "bearing":      0.0008,
    "motor":        0.0006,
    "oil_seal":     0.0005,
    "coolant_seal": 0.0005,
    "belt":         0.0007,
    "spindle":      0.0006,
}


# ─────────────────────────────────────────────────────────────────────────────
# Machine + simulator dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MachineState:
    """One machine's component wear state + recent sensor history.

    `history` is a list of (18,) numpy arrays — one entry per `sim.step()`
    call. **It is bounded to the last 200 readings** by `step()` (see the
    `if len(m.history) > 200` trim). For lifecycles that exceed 200 ticks
    (the default `max_steps=800`), the early history is silently discarded.

    This is safe for the project's only consumer — `get_window(window_size)`
    where `window_size <= 200` — but be aware if you add a consumer that
    needs the full lifecycle history (e.g. for plotting). The 200-tick
    ceiling exists to keep per-machine memory bounded (200 * 18 * 4 bytes
    ≈ 14KB / machine). See BUG_REPORT LOW-16.
    """
    machine_id: int
    wears: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in COMPONENTS})
    fault_pressures: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in COMPONENTS})
    fault_durations: dict[str, int] = field(default_factory=lambda: {c: 0 for c in COMPONENTS})
    history: list[np.ndarray] = field(default_factory=list)   # bounded to last 200 entries (see docstring)

    def max_wear(self) -> float:
        return max(self.wears.values())

    def true_rul(self) -> float:
        """Ground-truth RUL. RUL = 100 * (1 - max_wear)^2 → smooth degradation."""
        return 100.0 * (1.0 - self.max_wear()) ** 2

    def is_failed(self) -> bool:
        return self.max_wear() >= 1.0


class FactorySimulator:
    """
    Physics-informed factory simulator.

    Usage:
        sim = FactorySimulator(n_machines=5, seed=42)
        for _ in range(200):
            sim.step()
        sim.inject_fault(machine_id=3, sensor_id="Xs2", severity="HIGH")
        for _ in range(50):
            sim.step()
        window = sim.get_window(machine_id=3)   # (50, 18) most recent readings
        rul    = sim.true_rul(machine_id=3)
    """

    def __init__(self, n_machines: int = 5, seed: Optional[int] = None, noise_scale: float = 0.05):
        self.n_machines = n_machines
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale
        self.machines = [MachineState(machine_id=i + 1) for i in range(n_machines)]
        self.t = 0   # global timestep counter

    # ── Sensor computation ─────────────────────────────────────────────────
    def _compute_sensors(self, machine: MachineState) -> np.ndarray:
        """Compute the 18 sensor readings for this machine, given component wears."""
        readings = {name: SENSOR_BASELINES[name] for name in SENSOR_NAMES}

        # Add each component's contribution
        for comp, wear in machine.wears.items():
            for sensor_name, slope in COMPONENT_SENSOR_EFFECTS[comp]:
                readings[sensor_name] += slope * wear

        # Noise scales mildly with overall wear (worse machines are noisier)
        noise_amp = self.noise_scale * (1.0 + machine.max_wear())
        arr = np.array([readings[n] for n in SENSOR_NAMES], dtype=np.float32)
        # Per-sensor noise as a fraction of baseline magnitude
        noise = self.rng.normal(0, 1, size=18).astype(np.float32) * noise_amp * \
                np.array([abs(SENSOR_BASELINES[n]) * 0.02 + 0.01 for n in SENSOR_NAMES], dtype=np.float32)
        return arr + noise

    # ── Time step ──────────────────────────────────────────────────────────
    def step(self):
        """Advance time by one tick. Each machine ages."""
        for m in self.machines:
            # Apply base wear + active fault pressure
            for comp in COMPONENTS:
                rate = BASE_WEAR_RATES[comp] + m.fault_pressures[comp]
                m.wears[comp] = min(1.0, m.wears[comp] + rate)
                # Decay fault pressure over time (fault "burns out" after ~50 ticks)
                if m.fault_durations[comp] > 0:
                    m.fault_durations[comp] -= 1
                    if m.fault_durations[comp] == 0:
                        m.fault_pressures[comp] = 0.0

            # Compute and append sensor reading
            reading = self._compute_sensors(m)
            m.history.append(reading)
            # Keep history bounded (twice window size for slicing safety)
            if len(m.history) > 200:
                m.history = m.history[-200:]

        self.t += 1

    # ── Fault injection (called by the agentic pipeline) ───────────────────
    def inject_fault(self, machine_id: int, sensor_id: str, severity: str,
                     duration: int = 30) -> None:
        """
        Translate "Xs2 spike severity=HIGH" into "bearing component takes
        HIGH fault pressure for `duration` ticks."

        Args:
            machine_id: 1-indexed machine ID
            sensor_id:  one of W0-W3, Xs0-Xs13 (the LLM's output)
            severity:   LOW / MEDIUM / HIGH
            duration:   timesteps the fault persists (default 30)
        """
        m = self.machines[machine_id - 1]
        component = SENSOR_TO_PRIMARY_COMPONENT.get(sensor_id, "bearing")
        pressure = SEVERITY_FAULT_PRESSURE.get(severity, SEVERITY_FAULT_PRESSURE["MEDIUM"])
        m.fault_pressures[component] = pressure
        m.fault_durations[component] = duration

    # ── Query interfaces ───────────────────────────────────────────────────
    def get_window(self, machine_id: int, window_size: int = 50) -> np.ndarray:
        """Return the most recent `window_size` sensor readings as (window_size, 18)."""
        m = self.machines[machine_id - 1]
        if len(m.history) < window_size:
            # Pad with the current reading (or baseline if no history)
            if len(m.history) == 0:
                pad = self._compute_sensors(m)
            else:
                pad = m.history[0]
            history = [pad] * (window_size - len(m.history)) + m.history
        else:
            history = m.history[-window_size:]
        return np.stack(history, axis=0).astype(np.float32)

    def true_rul(self, machine_id: int) -> float:
        return self.machines[machine_id - 1].true_rul()

    def reset_machine(self, machine_id: int) -> None:
        """Reset a single machine to brand-new state."""
        self.machines[machine_id - 1] = MachineState(machine_id=machine_id)

    def reset_all(self) -> None:
        """Reset every machine."""
        for i in range(self.n_machines):
            self.reset_machine(i + 1)
        self.t = 0


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run as `python -m research.simulator.factory_simulator`)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    """Quick demo: simulate a machine to failure with a bearing fault."""
    sim = FactorySimulator(n_machines=1, seed=42, noise_scale=0.0)

    print("=== Factory simulator self-test ===")
    print(f"Healthy baseline (t=0):")
    sensors = sim.get_window(1, window_size=1)[0]
    for i, name in enumerate(SENSOR_NAMES):
        print(f"  {name:14s} = {sensors[i]:7.2f}")
    print(f"  true_rul = {sim.true_rul(1):.1f}")

    print("\n--- Aging for 100 ticks (no faults) ---")
    for _ in range(100):
        sim.step()
    print(f"After natural aging (t=100):")
    print(f"  max_wear = {sim.machines[0].max_wear():.3f}")
    print(f"  true_rul = {sim.true_rul(1):.1f}")
    sensors = sim.get_window(1, window_size=1)[0]
    print(f"  Bearing_Temp = {sensors[6]:.2f} (baseline 55, expected ~+small)")

    print("\n--- Inject HIGH bearing fault, run for 30 ticks ---")
    sim.inject_fault(machine_id=1, sensor_id="Xs2", severity="HIGH", duration=30)
    for _ in range(30):
        sim.step()
    print(f"After HIGH bearing fault:")
    print(f"  bearing_wear = {sim.machines[0].wears['bearing']:.3f}")
    print(f"  max_wear     = {sim.machines[0].max_wear():.3f}")
    print(f"  true_rul     = {sim.true_rul(1):.1f}")
    sensors = sim.get_window(1, window_size=1)[0]
    print(f"  Bearing_Temp = {sensors[6]:.2f} (baseline 55, faulted should be much higher)")
    print(f"  Vibration_X  = {sensors[4]:.2f} (baseline 0.3, should be higher)")
    print(f"  Motor_Temp   = {sensors[7]:.2f} (baseline 65, coupled rise)")


if __name__ == "__main__":
    _self_test()
