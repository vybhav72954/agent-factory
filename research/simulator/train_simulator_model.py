"""
research/simulator/train_simulator_model.py

Train CNNLSTM_RUL on the simulator-generated data. Reuses the existing
architecture from `dl_engine/model.py` unchanged — only the training data
and scaler are new.

Output:
    dl_engine/weights/best_model_simulator.pt
    dl_engine/weights/scaler_simulator.pkl

Usage:
    python -m research.simulator.train_simulator_model
    python -m research.simulator.train_simulator_model --epochs 30 --batch-size 128
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dl_engine.model import CNNLSTM_RUL


DATA_PATH        = PROJECT_ROOT / "research" / "simulator" / "data" / "training_data.npz"
SCALER_SRC_PATH  = PROJECT_ROOT / "research" / "simulator" / "data" / "scaler.pkl"
WEIGHTS_DIR      = PROJECT_ROOT / "dl_engine" / "weights"
MODEL_DST_PATH   = WEIGHTS_DIR / "best_model_simulator.pt"
SCALER_DST_PATH  = WEIGHTS_DIR / "scaler_simulator.pkl"


def load_and_scale() -> tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Load NPZ, apply scaler, return tensors + scaler for downstream save."""
    print(f"[train] loading {DATA_PATH.relative_to(PROJECT_ROOT)}...")
    data = np.load(DATA_PATH)
    X = data["X"]   # (N, 50, 18) raw
    y = data["y"]   # (N,)

    with open(SCALER_SRC_PATH, "rb") as f:
        scaler: MinMaxScaler = pickle.load(f)

    # Apply scaler (reshape, transform, reshape back)
    N, W, F = X.shape
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, W, F)
    X_scaled = np.clip(X_scaled, 0.0, 1.0).astype(np.float32)

    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(y.astype(np.float32))

    print(f"[train] X={tuple(X_t.shape)}  y={tuple(y_t.shape)}")
    return X_t, y_t, scaler


def train(
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    patience: int = 5,
    device: str = "cpu",
) -> None:
    X, y, scaler = load_and_scale()

    # Train/val split — both the split AND the DataLoader shuffle order
    # are seeded for bit-reproducibility (BUG_REPORT LOW-15). Without the
    # DataLoader generator, shuffle=True would use its own RNG and produce
    # different batch orderings on repeated runs even with random_split seeded.
    split_gen   = torch.Generator().manual_seed(42)
    loader_gen  = torch.Generator().manual_seed(42)

    ds = TensorDataset(X, y)
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=split_gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=loader_gen)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = CNNLSTM_RUL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = patience

    print(f"[train] device={device}  epochs={epochs}  batch={batch_size}  "
          f"n_train={n_train}  n_val={n_val}", flush=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_losses.append(criterion(pred, yb).item())
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        train_mse = float(np.mean(train_losses))
        val_mse   = float(np.mean(val_losses))
        scheduler.step(val_mse)
        val_preds_arr = np.concatenate(val_preds)
        val_trues_arr = np.concatenate(val_trues)
        val_mae = float(np.mean(np.abs(val_preds_arr - val_trues_arr)))

        elapsed = time.time() - t0
        is_best = val_mse < best_val_loss
        marker = "*BEST*" if is_best else "      "
        print(f"  epoch {epoch:2d}/{epochs}  train_mse={train_mse:7.2f}  "
              f"val_mse={val_mse:7.2f}  val_mae={val_mae:5.2f}  "
              f"elapsed={elapsed:.0f}s  {marker}", flush=True)

        if is_best:
            best_val_loss = val_mse
            best_epoch = epoch
            patience_left = patience
            # Save weights
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DST_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"[train] early stopping at epoch {epoch} (best was {best_epoch})")
                break

    # Save scaler alongside
    with open(SCALER_DST_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[train] done. best epoch={best_epoch}  best_val_mse={best_val_loss:.2f}")
    print(f"[train] wrote {MODEL_DST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[train] wrote {SCALER_DST_PATH.relative_to(PROJECT_ROOT)}")

    # ── Quick check: is the trained model BIMODAL or properly continuous? ──
    model.load_state_dict(torch.load(MODEL_DST_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        sample = X[:5000].to(device)
        sample_pred = model(sample).cpu().numpy()
    print(f"\n=== Bimodality check on 5000 validation samples ===")
    print(f"  pred quantiles: min={sample_pred.min():.2f}  "
          f"25%={np.percentile(sample_pred, 25):.2f}  "
          f"50%={np.percentile(sample_pred, 50):.2f}  "
          f"75%={np.percentile(sample_pred, 75):.2f}  "
          f"max={sample_pred.max():.2f}")
    print(f"  true quantiles: min={y[:5000].min():.2f}  "
          f"25%={np.percentile(y[:5000].numpy(), 25):.2f}  "
          f"50%={np.percentile(y[:5000].numpy(), 50):.2f}  "
          f"75%={np.percentile(y[:5000].numpy(), 75):.2f}  "
          f"max={y[:5000].max():.2f}")
    # Coverage check
    mode_lo = float(np.percentile(sample_pred, 25))
    mode_hi = float(np.percentile(sample_pred, 75))
    near_lo = int((np.abs(sample_pred - mode_lo) <= 1.0).sum())
    near_hi = int((np.abs(sample_pred - mode_hi) <= 1.0).sum())
    print(f"  near 25%-mode (+/-1): {near_lo}/{len(sample_pred)} ({near_lo/len(sample_pred):.1%})")
    print(f"  near 75%-mode (+/-1): {near_hi}/{len(sample_pred)} ({near_hi/len(sample_pred):.1%})")
    bimodal = (near_lo + near_hi) / len(sample_pred)
    print(f"  total at modes:     {bimodal:.1%}")
    if bimodal > 0.85:
        print("  [!] STILL BIMODAL — the simulator data didn't fix the collapse. "
              "Investigate (architecture, regularization, data balance).")
    elif bimodal > 0.50:
        print("  [!] Partially bimodal. Better than turbofan model but not ideal.")
    else:
        print("  [OK] Output is properly continuous. Ready for downstream evaluation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto' (use CUDA if available)")
    args = parser.parse_args()

    # Auto-detect or honour explicit choice
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available, falling back to CPU", flush=True)
        device = "cpu"
    if device == "cuda":
        print(f"[train] using GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)


if __name__ == "__main__":
    main()
