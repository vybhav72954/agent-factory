"""
Stub RUL predictor for development and testing.

Returns a fixed RUL value so the full UI pipeline can be built and tested
before the DL model is ready.

On Day 10, replace:
    from terminal.dummy_oracle import predict_rul
with:
    from dl_engine.inference import predict_rul
"""

import numpy as np


def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    Stub RUL prediction function.

    Args:
        sensor_tensor: numpy array of shape (50, 18)
                      - 50 timesteps of sensor data
                      - 18 sensors (W0-W3, Xs0-Xs13)

    Returns:
        float: RUL prediction (cycles remaining)

    Test values:
        15.0 → OFFLINE (red, critical)
        25.0 → DEGRADED (yellow, warning)
        50.0 → ONLINE (green, healthy)
    """
    return 15.0
