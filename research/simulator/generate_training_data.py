"""
research/simulator/generate_training_data.py

Generate training data for the CNN-LSTM from the physics-informed factory
simulator. Produces (X, y) pairs where X is a (50, 18) sensor window and
y is the simulator's ground-truth RUL at that moment.

Strategy: simulate N lifecycles. Each lifecycle starts a fresh machine and
runs until failure (max_wear ≥ 1.0) OR a max-step cap. Throughout each
lifecycle, take periodic snapshots — each snapshot gives one (X, y) pair.

Some lifecycles age naturally; others have random faults injected at random
times to expose the model to the kind of input patterns the agentic pipeline
will produce at inference time. This is the *crucial* fix to the OOD problem
that the N-CMAPSS-trained model had.

Usage:
    python -m research.simulator.generate_training_data
    python -m research.simulator.generate_training_data --n-lifecycles 2000

Output:
    research/simulator/data/training_data.npz   — arrays X, y
    research/simulator/data/scaler.pkl          — fitted MinMaxScaler for inference
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.simulator.factory_simulator import (
    FactorySimulator, SENSOR_NAMES, SEVERITY_FAULT_PRESSURE,
)


DATA_DIR = PROJECT_ROOT / "research" / "simulator" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH    = DATA_DIR / "training_data.npz"
SCALER_PATH  = DATA_DIR / "scaler.pkl"


# Sensor IDs the LLM diagnostic agent might inject. Drawn from prompts.py
# FAULT → SENSOR MAPPING.
INJECTABLE_SENSORS = ["Xs2", "Xs3", "Xs4", "Xs0", "Xs1", "Xs7", "Xs6",
                      "W0", "W3", "Xs9", "Xs11", "Xs12"]


def simulate_one_lifecycle(
    seed: int,
    window_size: int = 50,
    max_steps: int = 800,
    snapshot_every: int = 10,
    fault_probability: float = 0.4,   # MED-10: TESTED 0.75 → regression, see below
) -> list[tuple[np.ndarray, float]]:
    """
    Simulate one machine's life. Optionally inject 1-3 random faults at
    random times. Return list of (sensor_window, true_rul) pairs.

    snapshot_every controls how many ticks between training-data snapshots.
    With snapshot_every=10 and max_steps=800 we get up to 80 snapshots per
    lifecycle (fewer if the machine fails early).

    fault_probability=0.4 (the production value): 40% of lifecycles get
    1-3 random fault injections; the other 60% age naturally without
    interventions. This mix was preserved after BUG_REPORT_2026-05-25.md
    MED-10 *tested* a bump to 0.75 and found it produced a regression on
    all baseline strategies. Compare:

        v2 (fault_prob=0.4):  keyword_regex 94.4%, agentic 88.9%,
                              llama3 85.2%, llama4 72.2%
        v3 (fault_prob=0.75): keyword_regex 85.2%, agentic 77.8%,
                              llama3 59.3%, llama4 63.0%

    The v3 model over-reacts to mild MEDIUM-severity injections (flagging
    them as OFFLINE) because its training distribution is dominated by
    fault-driven trajectories with limited "mild fault stays ONLINE"
    examples. The v3 weights and result snapshots are preserved under
    `dl_engine/weights/*_simulator_v3.pt.bak` and
    `research/results/*_simulator_v3.*` for the paper's appendix.
    """
    rng = np.random.default_rng(seed)
    sim = FactorySimulator(n_machines=1, seed=seed, noise_scale=0.5)

    # Warm-up: run window_size ticks so we have a full window for the first snapshot
    for _ in range(window_size):
        sim.step()

    # Maybe schedule 1-3 faults at random ticks in the future
    fault_schedule: list[tuple[int, str, str]] = []   # (tick, sensor_id, severity)
    if rng.random() < fault_probability:
        n_faults = rng.integers(1, 4)   # 1 to 3 faults
        for _ in range(n_faults):
            fault_tick = int(rng.uniform(window_size + 20, max_steps - 50))
            sensor_id  = rng.choice(INJECTABLE_SENSORS)
            # Distribution chosen to match inference-time severity frequency.
            # The LLM diagnostic agent defaults to MEDIUM (per prompts.py
            # SEVERITY CLASSIFICATION), so MEDIUM dominates at inference.
            # An earlier p=[0.4, 0.4, 0.2] under-represented MEDIUM and
            # may have biased the model away from mid-severity faults
            # (see BUG_REPORT_2026-05-25.md HIGH-8).
            severity   = rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.2, 0.5, 0.3])
            fault_schedule.append((fault_tick, str(sensor_id), str(severity)))

    pairs: list[tuple[np.ndarray, float]] = []
    last_snapshot_tick = -1   # MED-12: avoid duplicate snapshot when failure aligns with snapshot tick

    for t in range(window_size, max_steps):
        # Trigger any faults scheduled for this tick
        for ft, sid, sev in fault_schedule:
            if ft == t:
                sim.inject_fault(machine_id=1, sensor_id=sid, severity=sev, duration=40)

        sim.step()

        # Snapshot a training pair
        if t % snapshot_every == 0:
            window = sim.get_window(1, window_size=window_size)
            rul = sim.true_rul(1)
            pairs.append((window, rul))
            last_snapshot_tick = t

        # Stop early if the machine has failed
        if sim.machines[0].is_failed():
            # One last snapshot at failure — but only if we didn't already
            # snapshot at this exact tick (MED-12).
            if t != last_snapshot_tick:
                window = sim.get_window(1, window_size=window_size)
                pairs.append((window, 0.0))
            break

    return pairs


def generate(n_lifecycles: int = 1000, window_size: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full training dataset."""
    print(f"[gen] simulating {n_lifecycles} lifecycles...")
    t0 = time.time()
    all_pairs: list[tuple[np.ndarray, float]] = []
    for i in range(n_lifecycles):
        pairs = simulate_one_lifecycle(seed=i, window_size=window_size)
        all_pairs.extend(pairs)
        if (i + 1) % max(1, n_lifecycles // 20) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_lifecycles - i - 1) / rate
            print(f"[gen]   {i+1}/{n_lifecycles} lifecycles  "
                  f"({len(all_pairs)} pairs)  rate={rate:.0f} lc/s  eta={eta:.0f}s")

    X = np.stack([p[0] for p in all_pairs], axis=0).astype(np.float32)   # (N, 50, 18)
    y = np.array([p[1] for p in all_pairs], dtype=np.float32)             # (N,)

    print(f"[gen] done: X={X.shape}  y={y.shape}  RUL min={y.min():.1f}  max={y.max():.1f}")
    return X, y


def fit_scaler(X: np.ndarray) -> MinMaxScaler:
    """Fit a MinMaxScaler on the full sensor distribution (flattened across all snapshots)."""
    flat = X.reshape(-1, X.shape[-1])   # (N*50, 18)
    scaler = MinMaxScaler()
    scaler.fit(flat)
    return scaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-lifecycles", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=50)
    args = parser.parse_args()

    X, y = generate(n_lifecycles=args.n_lifecycles, window_size=args.window_size)

    print("[gen] fitting MinMaxScaler...")
    scaler = fit_scaler(X)

    np.savez_compressed(DATA_PATH, X=X, y=y)
    print(f"[gen] wrote {DATA_PATH.relative_to(PROJECT_ROOT)} ({DATA_PATH.stat().st_size / 1e6:.1f} MB)")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[gen] wrote {SCALER_PATH.relative_to(PROJECT_ROOT)}")

    # Quick stats
    print(f"\n=== Training-data quality check ===")
    print(f"  N pairs:       {len(X)}")
    print(f"  RUL quantiles: min={y.min():.1f}  25%={np.percentile(y, 25):.1f}  "
          f"50%={np.percentile(y, 50):.1f}  75%={np.percentile(y, 75):.1f}  max={y.max():.1f}")
    print(f"  RUL mean:      {y.mean():.1f}")
    print(f"  RUL std:       {y.std():.1f}")
    # Class-balance proxy: how many in each bucket
    n_online   = int((y > 30).sum())
    n_degraded = int(((y > 15) & (y <= 30)).sum())
    n_offline  = int((y <= 15).sum())
    print(f"  ONLINE > 30:   {n_online} ({n_online/len(y):.1%})")
    print(f"  DEGRADED 15-30:{n_degraded} ({n_degraded/len(y):.1%})")
    print(f"  OFFLINE <= 15:  {n_offline} ({n_offline/len(y):.1%})")
    print(f"\nDEGRADED bucket fraction is the key training-coverage metric.")
    print(f"For comparison, the OOD turbofan model had 0.8% of probe points DEGRADED.")
    print(f"If this number is >15%, the model should learn a smooth RUL response.")


if __name__ == "__main__":
    main()
