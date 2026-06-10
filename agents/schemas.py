# agents/schemas.py

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List


class FaultSeverity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class SensorSpike(BaseModel):
    """
    Structured output schema for Agent 1.
    Gemini is forced to return JSON conforming to this shape.
    Structural validation (field existence, types) is guaranteed by Gemini.
    Domain validation (valid sensor names, position ranges) is done manually after.
    """

    sensor_id: str = Field(
        description=(
            "The sensor column to spike. MUST be exactly one of these 18 values: "
            "W0 (Motor RPM), W1 (Feed Rate), W2 (Power kW), W3 (Coolant Flow), "
            "Xs0 (Vibration X), Xs1 (Vibration Y), Xs2 (Bearing Temp), Xs3 (Motor Temp), "
            "Xs4 (Oil Pressure), Xs5 (Oil Temp), Xs6 (Spindle Load), Xs7 (Torque), "
            "Xs8 (Hydraulic PSI), Xs9 (Coolant Temp), Xs10 (Ambient Temp), "
            "Xs11 (Current Amps), Xs12 (Acoustic dB), Xs13 (Cycle Time). "
            "No other values are valid. Do not invent sensor names."
        )
    )

    spike_value: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Injected sensor reading in normalized [0.0, 1.0] range. "
            "Normal operation is 0.3–0.7. "
            "HIGH severity faults: 0.85–0.98. "
            "MEDIUM severity faults: 0.65–0.84. "
            "LOW severity faults: 0.45–0.64. "
            "Never use exactly 0.0 or 1.0."
        )
    )

    affected_window_positions: List[int] = Field(
        description=(
            "Timestep indices (0–49) where the spike is injected. "
            "Index 0 = oldest reading, index 49 = most recent. "
            "Sudden faults: inject only at the last 3–5 positions (45–49). "
            "Progressive/gradual faults: inject at 5–10 positions spread across 35–49. "
            "Maximum 10 positions total. Minimum 1 position. "
            "All values must be integers between 0 and 49 inclusive."
        )
    )

    fault_severity: FaultSeverity = Field(
        description=(
            "Severity of the fault. "
            "HIGH = immediate danger, machine should go OFFLINE (RUL likely ≤ 15). "
            "MEDIUM = degraded performance, machine should slow down (RUL likely 15–30). "
            "LOW = early warning, monitor closely (RUL likely > 30)."
        )
    )

    plain_english_summary: str = Field(
        description=(
            "One sentence describing the fault for the terminal log. "
            "Be specific: include the sensor name and physical meaning. "
            "Example: 'Bearing temperature sensor Xs2 critical — thermal threshold exceeded.' "
            "Do NOT include brackets, markdown, or special characters."
        )
    )
