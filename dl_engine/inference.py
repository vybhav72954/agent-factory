# dl_engine/inference.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT FILE — this is the ONLY file other teams import.
# Public API: predict_rul(sensor_tensor: np.ndarray) -> float
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import threading
import torch
import numpy as np
import joblib
from pathlib import Path

# ── Model definition — single source of truth in model.py ─────────────────────
from .model import CNNLSTM_RUL


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
# These are protected by `_load_lock` for thread safety. The Textual TUI
# dispatches the pipeline through @work(thread=True), and the research scripts
# never call from multiple threads — but defensive locking keeps the contract
# safe for any future multi-threaded consumer (e.g. a server wrapper around
# predict_rul). The lock is held only during load/clear, not during prediction
# itself, so concurrent predict_rul() calls after the first load are unblocked.
# See BUG_REPORT_2026-05-25.md INFO-20.
_model  = None
_scaler = None
_loaded_variant = None   # "turbofan" or "simulator" — for diagnostics
_load_lock = threading.Lock()


# Environment-variable toggle for the research extension that retrained the
# CNN-LSTM on a physics-informed factory simulator. Production pipeline default
# remains the original N-CMAPSS-trained checkpoint so existing tests still pass.
#
# Set FORGEMIND_USE_SIMULATOR_MODEL=1 to load the simulator-trained checkpoint
# from `dl_engine/weights/best_model_simulator.pt` + `scaler_simulator.pkl`.
def _resolve_paths() -> tuple[str, str, str]:
    """Return (weights_path, scaler_path, variant_label) based on env var."""
    if os.environ.get("FORGEMIND_USE_SIMULATOR_MODEL") == "1":
        return (
            "dl_engine/weights/best_model_simulator.pt",
            "dl_engine/weights/scaler_simulator.pkl",
            "simulator",
        )
    return (
        "dl_engine/weights/best_model.pt",
        "dl_engine/weights/scaler.pkl",
        "turbofan",
    )


def load_model(
    weights_path: str | None = None,
    scaler_path : str | None = None,
    variant     : str | None = None,
):
    """Load model weights and scaler. Called automatically on first predict_rul().

    With no arguments, honours FORGEMIND_USE_SIMULATOR_MODEL env var.

    With explicit paths the variant label defaults to "custom". Pass
    variant="simulator" for a checkpoint trained on simulator data (e.g. the
    PRONOSTIA-calibrated simulator) so the polarity-aware helpers
    (get_healthy_baseline, diagnostic_agent._is_dropping) treat W0/W3/Xs4/Xs8
    as dropping sensors.

    **Thread safety:** holds `_load_lock` during the entire load so concurrent
    callers can't observe a half-loaded state. The lock is also re-checked
    after acquisition (double-checked locking) so that if a peer thread
    completed the load while we were waiting on the lock, we skip the redundant
    re-load. Once loaded, `predict_rul()` itself runs lock-free because PyTorch
    inference is read-only on the model state.
    """
    global _model, _scaler, _loaded_variant

    if weights_path is None or scaler_path is None:
        w, s, variant = _resolve_paths()
        weights_path = weights_path or w
        scaler_path  = scaler_path  or s
    else:
        variant = variant or "custom"

    with _load_lock:
        # Double-check: another thread may have loaded the SAME variant
        # while we were blocked acquiring the lock. Avoid the redundant
        # torch.load() + state_dict assignment in that case.
        if _model is not None and _loaded_variant == variant:
            return

        checkpoint = torch.load(weights_path, map_location="cpu")  # CPU-safe

        # Two on-disk formats are supported:
        #   (a) wrapped: dict with "model_state_dict" + optional "config" (original)
        #   (b) raw:    bare state_dict OrderedDict (simulator-trained, simpler)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            cfg = checkpoint.get("config", {})
        else:
            state_dict = checkpoint   # raw state_dict (what torch.save(model.state_dict()) produces)
            cfg = {}

        new_model = CNNLSTM_RUL(
            n_features  = cfg.get("n_features",  18),
            window      = cfg.get("window",      50),
            cnn_filters = cfg.get("cnn_filters", 64),
            lstm_hidden = cfg.get("lstm_hidden", 128),
            lstm_layers = cfg.get("lstm_layers", 2),
            dropout     = cfg.get("dropout",     0.3),
        )
        new_model.load_state_dict(state_dict)
        new_model.eval()

        # Scaler — try joblib first (production format), fall back to pickle
        # (research/simulator/ saves with pickle for portability).
        try:
            new_scaler = joblib.load(scaler_path)
        except Exception:
            with open(scaler_path, "rb") as f:
                new_scaler = pickle.load(f)

        # Atomic-from-readers' perspective: assign the three globals after
        # both the model and scaler have been fully constructed. A reader
        # that races us either sees the OLD model+scaler (still consistent
        # internally) or the NEW model+scaler — never a partial mixture.
        _model = new_model
        _scaler = new_scaler
        _loaded_variant = variant


def reset_loaded_model():
    """Clear the cached model + scaler so the next consumer call reloads from disk.

    The reload itself is **deferred** — this function does not call
    `load_model()` itself. Any subsequent call to `predict_rul()`,
    `get_healthy_baseline()`, `raw_value_for_scaled()`, or `get_scaler_ranges()`
    triggers `load_model()` lazily and picks up whatever
    `FORGEMIND_USE_SIMULATOR_MODEL` currently says.

    Typical usage when switching variants mid-session::

        import os, dl_engine.inference as inf

        os.environ["FORGEMIND_USE_SIMULATOR_MODEL"] = "1"
        inf.reset_loaded_model()         # clear stale turbofan cache
        rul = inf.predict_rul(window)    # triggers load → variant="simulator"

    If you don't call this after changing the env var, the old (cached)
    variant stays active for the rest of the session. See BUG_REPORT LOW-18.
    """
    global _model, _scaler, _loaded_variant
    with _load_lock:
        _model = None
        _scaler = None
        _loaded_variant = None


def get_loaded_variant() -> str | None:
    """Returns 'turbofan', 'simulator', 'custom', or None if not yet loaded."""
    return _loaded_variant



def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    Predict Remaining Useful Life for a single window of sensor data.

    Parameters
    ----------
    sensor_tensor : np.ndarray, shape (50, 18)
        One sliding window — 50 time-steps × 18 features
        (4 operating conditions + 14 physical sensors), RAW (unscaled).
        The scaler is applied internally.

    Returns
    -------
    float
        Predicted RUL in production shift-cycles. Always >= 0.
    """
    if _model is None:
        load_model()

    assert sensor_tensor.shape == (50, 18), (
        f"predict_rul expects shape (50, 18), got {sensor_tensor.shape}"
    )

    scaled = _scaler.transform(sensor_tensor.astype(np.float32))  # (50, 18)
    scaled = np.clip(scaled, 0.0, 1.0)
    t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)    # (1, 50, 18)

    with torch.no_grad():
        rul = _model(t).item()

    return max(0.0, rul)


# ── Scaler-range utilities ────────────────────────────────────────────────────
# These expose the scaler's learned min/max so other modules can produce
# tensors in the correct raw-unit domain (the model fails silently on
# synthetic [0,1] data because the scaler collapses it to near-zero).

def get_healthy_baseline(noise_std_frac: float = 0.02) -> np.ndarray:
    """
    Build a (50, 18) tensor representing nominal operating conditions
    in RAW physical units (pre-scaling).

    Args:
        noise_std_frac: noise magnitude as a fraction of each feature's range.
                        0.02 means ±2% jitter.  Set to 0.0 for deterministic output.

    Returns:
        np.ndarray, shape (50, 18), dtype float32
    """
    if _scaler is None:
        load_model()

    # Define healthy scaled positions [0, 1].
    # For the turbofan model (N-CMAPSS), all sensors rise with degradation,
    # so healthy is low (0.10).
    # For the simulator model, some sensors drop (W0, W3, Xs4, Xs8), so
    # healthy for those is high (0.90).
    healthy_scaled = np.full(18, 0.10)

    if _loaded_variant == "simulator":
        # Source of truth: agents.diagnostic_agent.SIMULATOR_DROPPING_COLS
        # Lazy import to avoid circular dependency at module load.
        from agents.diagnostic_agent import SIMULATOR_DROPPING_COLS
        for i in SIMULATOR_DROPPING_COLS:
            healthy_scaled[i] = 0.90

    lo  = _scaler.data_min_                     # (18,)
    rng = _scaler.data_range_                   # (18,)
    healthy_raw = lo + healthy_scaled * rng      # raw-unit "healthy" vector

    baseline = np.tile(healthy_raw, (50, 1)).astype(np.float32)

    if noise_std_frac > 0:
        noise = np.random.normal(0, noise_std_frac * rng, size=(50, 18))
        baseline += noise.astype(np.float32)

    return baseline


def raw_value_for_scaled(sensor_col: int, scaled_target: float) -> float:
    """
    Inverse-map a desired [0, 1] scaled value back to raw physical units.

    Example: if the scaler learned  min=400, max=550 for column 8 (Xs4),
    then raw_value_for_scaled(8, 0.95) → 400 + 0.95 * 150 = 542.5

    The diagnostic agent stores spike_value in [0, 1]; this converts it
    to the raw value that predict_rul()'s internal scaler.transform()
    will map back to 0.95.

    Args:
        sensor_col:    column index 0–17 in the (50, 18) tensor
        scaled_target: desired position in [0, 1] after scaling

    Returns:
        float — raw-unit value
    """
    if _scaler is None:
        load_model()

    lo  = _scaler.data_min_[sensor_col]
    rng = _scaler.data_range_[sensor_col]
    return float(lo + scaled_target * rng)


def get_scaler_ranges() -> dict:
    """
    Return the scaler's learned min/max per feature for debugging.

    Returns:
        dict with keys 'min', 'max', 'range' — each a (18,) numpy array
    """
    if _scaler is None:
        load_model()

    return {
        "min":   _scaler.data_min_.copy(),
        "max":   _scaler.data_min_ + _scaler.data_range_,
        "range": _scaler.data_range_.copy(),
    }
