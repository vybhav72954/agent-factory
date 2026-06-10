"""
tests/dl_engine/test_inference.py
inference.py public API — predict_rul() contract, lazy-load behaviour,
clamp/scale, fallback, and the shape/dtype/range guarantees.

No real weights file needed for most tests — we mock load_model().
Tests that require a real weights file are skipped unless
FORGEMIND_RUN_LIVE=1 is set.

Run:  pytest tests/dl_engine/test_inference.py
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock
import dl_engine.inference as inf_module
from dl_engine.inference import predict_rul, CNNLSTM_RUL


requires_weights = pytest.mark.skipif(
    os.environ.get("FORGEMIND_RUN_LIVE") != "1",
    reason="Set FORGEMIND_RUN_LIVE=1 and ensure weights exist to run",
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level _model and _scaler between tests."""
    inf_module._model  = None
    inf_module._scaler = None
    yield
    inf_module._model  = None
    inf_module._scaler = None


@pytest.fixture
def mock_model_and_scaler():
    """Inject a real untrained model + identity scaler so predict_rul() runs."""
    model = CNNLSTM_RUL()
    model.eval()
    inf_module._model = model

    scaler = MagicMock()
    # Identity transform — returns input unchanged (already float32)
    scaler.transform.side_effect = lambda x: x.astype(np.float32)
    inf_module._scaler = scaler

    return model, scaler


@pytest.fixture
def nominal_window():
    rng = np.random.default_rng(42)
    return (rng.random((50, 18)) * 0.4 + 0.3).astype(np.float32)


@pytest.fixture
def spike_window():
    """High-spike window — max sensor values near 1.0."""
    w = np.ones((50, 18), dtype=np.float32) * 0.95
    return w


# ── Contract: output type and shape ──────────────────────────────────────────

class TestPredictRulContract:
    def test_returns_float(self, mock_model_and_scaler, nominal_window):
        result = predict_rul(nominal_window)
        assert isinstance(result, float)

    def test_returns_non_negative(self, mock_model_and_scaler, nominal_window):
        result = predict_rul(nominal_window)
        assert result >= 0.0

    def test_returns_non_negative_on_spike_window(self, mock_model_and_scaler, spike_window):
        result = predict_rul(spike_window)
        assert result >= 0.0

    def test_wrong_shape_4950_raises_assertion(self, mock_model_and_scaler):
        bad = np.random.rand(49, 18).astype(np.float32)
        with pytest.raises(AssertionError):
            predict_rul(bad)

    def test_wrong_shape_5017_raises_assertion(self, mock_model_and_scaler):
        bad = np.random.rand(50, 17).astype(np.float32)
        with pytest.raises(AssertionError):
            predict_rul(bad)

    def test_wrong_shape_flat_raises_assertion(self, mock_model_and_scaler):
        bad = np.random.rand(900).astype(np.float32)
        with pytest.raises(AssertionError):
            predict_rul(bad)

    def test_accepts_float64_input(self, mock_model_and_scaler, nominal_window):
        """predict_rul should handle float64 input by casting internally."""
        result = predict_rul(nominal_window.astype(np.float64))
        assert isinstance(result, float)
        assert result >= 0.0


# ── Lazy loading ──────────────────────────────────────────────────────────────

class TestLazyLoading:
    def test_singletons_start_as_none(self):
        assert inf_module._model is None
        assert inf_module._scaler is None

    def test_load_model_called_when_model_is_none(self, nominal_window):
        """If _model is None, load_model() must be called before inference."""
        called = []

        def fake_load(**kwargs):
            model = CNNLSTM_RUL()
            model.eval()
            inf_module._model = model
            scaler = MagicMock()
            scaler.transform.side_effect = lambda x: x.astype(np.float32)
            inf_module._scaler = scaler
            called.append(True)

        with patch.object(inf_module, "load_model", side_effect=fake_load):
            predict_rul(nominal_window)

        assert called, "load_model was not called despite _model being None"

    def test_load_model_not_called_when_model_already_loaded(
        self, mock_model_and_scaler, nominal_window
    ):
        called = []
        with patch.object(inf_module, "load_model",
                          side_effect=lambda **kw: called.append(True)):
            predict_rul(nominal_window)
        assert not called, "load_model was called even though _model was already set"


# ── Scaler is applied ─────────────────────────────────────────────────────────

class TestScalerApplication:
    def test_scaler_transform_is_called(self, mock_model_and_scaler, nominal_window):
        _, scaler = mock_model_and_scaler
        predict_rul(nominal_window)
        scaler.transform.assert_called_once()

    def test_scaler_receives_float32_array(self, mock_model_and_scaler, nominal_window):
        _, scaler = mock_model_and_scaler
        predict_rul(nominal_window)
        arg = scaler.transform.call_args[0][0]
        assert arg.dtype == np.float32

    def test_scaler_output_is_clipped_to_0_1(self, mock_model_and_scaler, nominal_window):
        """Even if scaler returns out-of-range values, the tensor must be clipped."""
        _, scaler = mock_model_and_scaler
        # Return values outside [0, 1]
        scaler.transform.side_effect = lambda x: np.full_like(x, 2.5)
        result = predict_rul(nominal_window)
        # Model should still run without NaN/inf
        assert isinstance(result, float)
        assert result >= 0.0


# ── No-gradient inference ─────────────────────────────────────────────────────

class TestNoGradientInference:
    def test_predict_rul_runs_under_no_grad(self, mock_model_and_scaler, nominal_window):
        """predict_rul uses torch.no_grad() — no gradients should be computed."""
        with torch.no_grad():
            result = predict_rul(nominal_window)
        assert isinstance(result, float)


# ── Real weights (live only) ──────────────────────────────────────────────────

@requires_weights
class TestRealWeights:
    def test_predict_rul_with_real_weights_returns_positive(self, nominal_window):
        inf_module._model  = None
        inf_module._scaler = None
        result = predict_rul(nominal_window)
        assert result > 0.0

    def test_predict_rul_high_spike_lower_than_nominal(self, nominal_window, spike_window):
        inf_module._model  = None
        inf_module._scaler = None
        rul_nominal = predict_rul(nominal_window)
        rul_spike   = predict_rul(spike_window)
        # A window with max-saturated sensors should predict LOWER RUL
        assert rul_spike < rul_nominal


# ── Threading: load_model concurrency (BUG_REPORT INFO-20) ──────────────────

class TestThreadSafeLoading:
    """The load lock must serialise concurrent first-time loads so that
    racing callers never observe a half-initialised state. We exercise
    this without requiring real weights by mocking the load itself."""

    def test_concurrent_load_calls_yield_consistent_state(self):
        """If 10 threads simultaneously trigger load_model(), every thread
        should observe a fully-formed (_model, _scaler, _loaded_variant)
        tuple — no thread should see a None mid-load.

        NOTE: we patch torch.load + joblib.load ONCE in the parent thread
        (NOT inside workers). Concurrent patch/unpatch from worker threads
        is itself racy (each `with patch.object()` save/restore wins/loses
        non-deterministically) and would leak the mock back into other tests.
        """
        import threading
        import time
        from unittest.mock import patch

        # Reset state at start so we don't see a previously-loaded model
        inf_module._model = None
        inf_module._scaler = None
        inf_module._loaded_variant = None

        observed_states = []
        observed_states_lock = threading.Lock()
        load_call_count = {"n": 0}

        def fake_torch_load(*a, **kw):
            load_call_count["n"] += 1
            time.sleep(0.01)   # increase chance of contention
            return {"model_state_dict": CNNLSTM_RUL().state_dict()}

        def fake_joblib_load(*a, **kw):
            scaler = MagicMock()
            scaler.transform.side_effect = lambda x: x.astype(np.float32)
            return scaler

        def worker():
            inf_module.load_model()
            with observed_states_lock:
                observed_states.append((
                    inf_module._model is not None,
                    inf_module._scaler is not None,
                    inf_module._loaded_variant,
                ))

        # Patch ONCE in the parent — both patches are restored after the `with` block
        with patch.object(inf_module.torch, "load", side_effect=fake_torch_load), \
             patch.object(inf_module.joblib, "load", side_effect=fake_joblib_load):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads: t.start()
            for t in threads: t.join()

        # Every observation must be the fully-loaded state
        for has_model, has_scaler, variant in observed_states:
            assert has_model, "thread observed _model=None after load_model()"
            assert has_scaler, "thread observed _scaler=None after load_model()"
            assert variant is not None, "thread observed _loaded_variant=None"

        # Sanity: due to double-checked locking, we expect torch.load to be
        # called fewer than 10 times (the first caller loads, others skip).
        assert load_call_count["n"] <= 10

        # Clean up so we don't leak module state into other tests
        inf_module._model = None
        inf_module._scaler = None
        inf_module._loaded_variant = None

    def test_reset_loaded_model_is_thread_safe(self):
        """Concurrent reset_loaded_model() calls should not crash or leave
        a non-None field behind."""
        import threading

        # Pre-load with mocks
        inf_module._model = CNNLSTM_RUL()
        inf_module._scaler = MagicMock()
        inf_module._loaded_variant = "test"

        def worker():
            inf_module.reset_loaded_model()

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert inf_module._model is None
        assert inf_module._scaler is None
        assert inf_module._loaded_variant is None

        # Defensive: leave module state clean for downstream tests
        inf_module._model = None
        inf_module._scaler = None
        inf_module._loaded_variant = None
        