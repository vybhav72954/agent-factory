"""
research/pronostia/calibrated_checkpoint.py

Retrain the simulator CNN-LSTM on the PRONOSTIA-calibrated simulator.

Same generation protocol as research/simulator/generate_training_data.py
(1000 lifecycles, 40% with 1-3 random faults, snapshots every 10 ticks, all six
components aging) and the same trainer as
research/simulator/train_simulator_model.py. The only change is the simulator:
`CalibratedFactorySimulator` with the bearing law fitted in calibrate.py
(late-onset vibration, early-rise temperature, lognormal bearing lifetime spread).

Sensor polarity is unchanged from the simulator, so the checkpoint is loaded
with variant="simulator" (research/replay_utils.py, checkpoint name
"simulator_calibrated").

Usage:
    python -m research.pronostia.calibrated_checkpoint [--n-lifecycles 1000] [--epochs 25]

Output:
    research/pronostia/data/calibrated_training_data.npz   (gitignored)
    research/pronostia/data/calibrated_scaler.pkl
    dl_engine/weights/best_model_simulator_calibrated.pt
    dl_engine/weights/scaler_simulator_calibrated.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.pronostia.calibrate import LAW_PATH, CalibratedFactorySimulator
from research.simulator.generate_training_data import fit_scaler, simulate_one_lifecycle
from research.simulator.train_simulator_model import train

DATA_DIR = PROJECT_ROOT / "research" / "pronostia" / "data"
DATA_PATH = DATA_DIR / "calibrated_training_data.npz"
SCALER_SRC = DATA_DIR / "calibrated_scaler.pkl"
WEIGHTS_PATH = PROJECT_ROOT / "dl_engine" / "weights" / "best_model_simulator_calibrated.pt"
SCALER_PATH = PROJECT_ROOT / "dl_engine" / "weights" / "scaler_simulator_calibrated.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-lifecycles", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    law = json.loads(LAW_PATH.read_text(encoding="utf-8"))
    make_sim = lambda seed: CalibratedFactorySimulator(law, n_machines=1, seed=seed, noise_scale=0.5)

    print(f"[calibrated] simulating {args.n_lifecycles} lifecycles with the calibrated bearing law...")
    t0 = time.time()
    pairs = []
    for i in range(args.n_lifecycles):
        pairs.extend(simulate_one_lifecycle(seed=i, make_sim=make_sim))
        if (i + 1) % max(1, args.n_lifecycles // 10) == 0:
            print(f"  {i + 1}/{args.n_lifecycles}  pairs={len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    X = np.stack([p[0] for p in pairs]).astype(np.float32)
    y = np.array([p[1] for p in pairs], dtype=np.float32)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PATH, X=X, y=y)
    with open(SCALER_SRC, "wb") as f:
        pickle.dump(fit_scaler(X), f)
    print(f"[calibrated] X={X.shape}  ONLINE {np.mean(y > 30):.1%}  DEGRADED {np.mean((y > 15) & (y <= 30)):.1%}  "
          f"OFFLINE {np.mean(y <= 15):.1%}")

    train(epochs=args.epochs, device=args.device, data_path=DATA_PATH, scaler_src_path=SCALER_SRC,
          model_dst_path=WEIGHTS_PATH, scaler_dst_path=SCALER_PATH)


if __name__ == "__main__":
    main()
