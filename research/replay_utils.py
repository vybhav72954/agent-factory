"""
research/replay_utils.py

Shared helpers for replaying recorded diagnostic outputs through the
production injection code and a CNN-LSTM checkpoint, without API calls.

Injection depends only on (sensor_id, fault_severity, spike_value) plus the
base window, so any recorded strategy output can be re-scored on any
checkpoint and under any SEVERITY_MULTIPLIERS table. Used by
research/multiplier_sweep.py, research/pronostia/replay.py and
research/paraphrase/score.py.

Checkpoints:
    turbofan              dl_engine/weights/best_model.pt                        (FORGEMIND_USE_SIMULATOR_MODEL=0)
    simulator             dl_engine/weights/best_model_simulator.pt              (FORGEMIND_USE_SIMULATOR_MODEL=1)
    simulator_calibrated  dl_engine/weights/best_model_simulator_calibrated.pt   (research/pronostia/calibrated_checkpoint.py)
    pronostia             dl_engine/weights/best_model_pronostia.pt              (research/pronostia/train_model.py)
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PUBLISHED_MULTIPLIERS = (0.15, 0.35, 0.85)   # LOW, MEDIUM, HIGH at the time of the 2026-05-26 run
CHECKPOINTS = ("turbofan", "simulator", "simulator_calibrated", "pronostia")
BATCH_SIZE = 1024

PRONOSTIA_WEIGHTS = PROJECT_ROOT / "dl_engine" / "weights" / "best_model_pronostia.pt"
PRONOSTIA_SCALER = PROJECT_ROOT / "dl_engine" / "weights" / "scaler_pronostia.pkl"
CALIBRATED_WEIGHTS = PROJECT_ROOT / "dl_engine" / "weights" / "best_model_simulator_calibrated.pt"
CALIBRATED_SCALER = PROJECT_ROOT / "dl_engine" / "weights" / "scaler_simulator_calibrated.pkl"
PRONOSTIA_META = PROJECT_ROOT / "research" / "pronostia" / "pronostia_model_meta.json"


def select_checkpoint(name: str) -> None:
    """Load a checkpoint into dl_engine.inference, clearing any cached model.

    For turbofan/simulator the env var is set explicitly, which beats `.env`
    because `load_dotenv()` in diagnostic_agent does not override existing
    values.
    """
    import dl_engine.inference as inf

    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {name!r}; expected one of {CHECKPOINTS}")
    inf.reset_loaded_model()
    if name == "pronostia":
        inf.load_model(str(PRONOSTIA_WEIGHTS), str(PRONOSTIA_SCALER))
        return
    if name == "simulator_calibrated":
        # Same sensor polarity as the simulator, so load with the simulator variant label.
        inf.load_model(str(CALIBRATED_WEIGHTS), str(CALIBRATED_SCALER), variant="simulator")
        return
    os.environ["FORGEMIND_USE_SIMULATOR_MODEL"] = "1" if name == "simulator" else "0"
    inf.load_model()
    if inf.get_loaded_variant() != name:
        raise RuntimeError(f"requested checkpoint {name!r} but loaded {inf.get_loaded_variant()!r}")


def status_from_rul(rul: float) -> str:
    """Same thresholds as capacity_agent.update_capacity (RUL <= 15 OFFLINE, <= 30 DEGRADED)."""
    from agents.capacity_agent import RUL_DEGRADED_THRESHOLD, RUL_OFFLINE_THRESHOLD
    if not np.isfinite(rul) or rul <= RUL_OFFLINE_THRESHOLD:
        return "OFFLINE"
    if rul <= RUL_DEGRADED_THRESHOLD:
        return "DEGRADED"
    return "ONLINE"


def predict_batch(windows: np.ndarray) -> np.ndarray:
    """Batched equivalent of dl_engine.inference.predict_rul for an (N, 50, 18) raw array."""
    import torch
    import dl_engine.inference as inf

    n = windows.shape[0]
    scaled = inf._scaler.transform(windows.reshape(-1, 18).astype(np.float32)).reshape(n, 50, 18)
    scaled = np.clip(scaled, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            t = torch.tensor(scaled[i:i + BATCH_SIZE], dtype=torch.float32)
            out[i:i + BATCH_SIZE] = inf._model(t).reshape(-1).cpu().numpy()
    return np.maximum(out, 0.0)


def check_batch_matches_single(windows: list[np.ndarray]) -> None:
    """Guard: batched inference must equal predict_rul on the same windows."""
    from dl_engine.inference import predict_rul

    sample = windows[:4]
    batched = predict_batch(np.stack(sample))
    single = np.array([predict_rul(w) for w in sample])
    if not np.allclose(batched, single, atol=1e-3):
        raise RuntimeError(f"batched inference {batched} != predict_rul {single}")


def healthy_baselines(keys: Iterable, checkpoint: str, seed: int, noise_std_frac: float = 0.02) -> dict:
    """One seeded healthy (50, 18) raw baseline per key, shared across strategies and settings.

    turbofan/simulator use dl_engine.inference.get_healthy_baseline (scaled 0.10,
    polarity-aware). pronostia uses the median early-life reading of its
    training bearings (pronostia_model_meta.json), with the same noise model.
    """
    import dl_engine.inference as inf

    keys = sorted(set(keys))
    if checkpoint == "pronostia":
        meta = json.loads(PRONOSTIA_META.read_text(encoding="utf-8"))
        healthy = np.asarray(meta["healthy_raw"], dtype=np.float32)
        rng_range = inf.get_scaler_ranges()["range"]
    baselines = {}
    for i, key in enumerate(keys):
        np.random.seed(seed + i)
        if checkpoint == "pronostia":
            base = np.tile(healthy, (50, 1)).astype(np.float32)
            if noise_std_frac > 0:
                base += np.random.normal(0, noise_std_frac * rng_range, size=(50, 18)).astype(np.float32)
            baselines[key] = base
        else:
            baselines[key] = inf.get_healthy_baseline(noise_std_frac=noise_std_frac)
    return baselines


def spikes_from_frame(df: pd.DataFrame) -> list:
    """SensorSpike objects from rows with sensor_id / severity / spike_value columns."""
    from agents.schemas import FaultSeverity, SensorSpike

    return [
        SensorSpike(
            sensor_id=r.sensor_id,
            spike_value=float(r.spike_value),
            affected_window_positions=[49],
            fault_severity=FaultSeverity(r.severity),
            plain_english_summary="replay",
        )
        for r in df.itertuples()
    ]


def replay(spikes: list, base_windows: list[np.ndarray], multipliers: tuple[float, float, float]) -> tuple[np.ndarray, list[str]]:
    """Inject each spike into its base window with the given (LOW, MEDIUM, HIGH) table; return RULs and statuses."""
    from agents.diagnostic_agent import _inject_spike
    from agents.schemas import FaultSeverity

    table = dict(zip((FaultSeverity.LOW, FaultSeverity.MEDIUM, FaultSeverity.HIGH), multipliers))
    windows = np.stack([
        _inject_spike(base, spike, multiplier_override=table[spike.fault_severity])
        for spike, base in zip(spikes, base_windows)
    ])
    ruls = predict_batch(windows)
    return ruls, [status_from_rul(r) for r in ruls]


EXACT_SIGNFLIP_MAX = 20    # enumerate all 2^k sign patterns up to this many non-zero clusters
MC_SIGNFLIP_DRAWS = 200_000


def paired_difference(a: pd.Series, b: pd.Series, groups: pd.Series, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of two aligned 0/1 (or real) score series from the same items.

    Differences are averaged within each group (e.g. prompt) first, so repeats of
    the same prompt are not treated as independent. Returns the mean difference
    (a - b), a 95% bootstrap CI resampling groups, and a two-sided sign-flip
    p-value on the per-group differences: exact (all 2^k sign patterns of the
    k non-zero groups) when k <= EXACT_SIGNFLIP_MAX, otherwise Monte Carlo.
    """
    diffs = (a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    per_group = pd.Series(diffs).groupby(groups.to_numpy()).mean().to_numpy()
    k = len(per_group)
    observed = float(per_group.mean())
    rng = np.random.default_rng(seed)
    boots = per_group[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    nonzero = per_group[np.abs(per_group) > 1e-12]
    if len(nonzero) == 0:
        p, method = 1.0, "exact"
    elif len(nonzero) <= EXACT_SIGNFLIP_MAX:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(nonzero))))
        stats = np.abs((signs * nonzero).sum(axis=1) / k)
        p, method = float(np.mean(stats >= abs(observed) - 1e-12)), "exact"
    else:
        signs = rng.choice((-1.0, 1.0), size=(MC_SIGNFLIP_DRAWS, len(nonzero)))
        stats = np.abs((signs * nonzero).sum(axis=1) / k)
        p = float((np.sum(stats >= abs(observed) - 1e-12) + 1) / (MC_SIGNFLIP_DRAWS + 1))
        method = f"monte carlo {MC_SIGNFLIP_DRAWS}"
    return {"diff": observed, "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "p": p, "p_method": method, "n_groups": k}


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_ci(values: pd.Series, groups: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of `values`, resampling whole groups (e.g. prompts) with replacement."""
    rng = np.random.default_rng(seed)
    by_group = values.groupby(groups).agg(["sum", "count"])
    sums, counts = by_group["sum"].to_numpy(), by_group["count"].to_numpy()
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
