"""
Test set + end-to-end evaluation runner for the Bayesian anomaly domain.

20 hand-crafted operator descriptions covering the 5 failure modes plus ambiguous
cases. Each prompt is paired with a true mode label used to score posterior
Brier / NLL / top-1.
"""

from __future__ import annotations

from typing import List, Tuple

# (operator_text, true_failure_mode)
TEST_PROMPTS: List[Tuple[str, str]] = [
    # ---- bearing ----
    ("the bearing is making a weird grinding noise",                            "bearing"),
    ("low rumble from spindle housing, sounds like a worn bearing",             "bearing"),
    ("Machine 2 has a wobble I can feel through the panel",                     "bearing"),
    ("intermittent grinding when the press cycles, otherwise quiet",            "bearing"),

    # ---- motor ----
    ("motor is drawing way more current than usual",                            "motor"),
    ("Machine 1 stalls under load, torque feels off",                           "motor"),
    ("RPM is dropping intermittently on the conveyor motor",                    "motor"),
    ("noticeable hum from the winding, motor running hot",                      "motor"),

    # ---- oil_seal ----
    ("oil pressure gauge keeps dipping during the cycle",                       "oil_seal"),
    ("there's a small oil leak under the gearbox seal",                         "oil_seal"),
    ("lubrication line pressure is unstable, pressure drop alarms",             "oil_seal"),
    ("seal looks weeping, oil pooling under the housing",                       "oil_seal"),

    # ---- coolant ----
    ("the chiller is barely keeping up, coolant flow looks low",                "coolant"),
    ("Machine 4 is running hot, cooling loop temp is climbing",                 "coolant"),
    ("noticeable overheating on the spindle, coolant flow drop suspected",     "coolant"),
    ("coolant pump output is half what it normally is",                         "coolant"),

    # ---- electrical ----
    ("blown fuse on the contactor, intermittent arcing reported",               "electrical"),
    ("voltage spike took out the VFD, sparks at the panel",                     "electrical"),
    ("current draw on the main feed jumped, electrical issue suspected",        "electrical"),
    ("noticeable amps spike, breaker tripped twice this shift",                 "electrical"),
]


def labels_summary() -> dict:
    """Count per-mode prompts for sanity checking."""
    from collections import Counter
    return dict(Counter(mode for _, mode in TEST_PROMPTS))
