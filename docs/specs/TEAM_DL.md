# TEAM DL — Technical Runbook

## CNN-LSTM Remaining Useful Life Prediction Engine

**Owners:** 2 people · **Environment:** Kaggle H100 (training) → GTX 1650 (inference)

**Your single deliverable:** A function called `predict_rul(sensor_tensor) -> float` that other teams call. Nothing else. You own `/dl_engine/*`.

---

## 0. CRITICAL: Dataset Key Fix

The original project spec assumed HDF5 keys like `W`, `X_s`, `Y`, `A`. **This is wrong.** The actual dataset from the Kaggle source (`kaggle.com/datasets/shreyaravi0/aircraft`) uses `_dev` and `_test` suffixed keys. Here is what `list(f.keys())` actually returns:

```
<<<<<<< Updated upstream
['A_dev', 'A_test', 'A_var',
 'T_dev', 'T_test', 'T_var',
 'W_dev', 'W_test', 'W_var',
 'X_s_dev', 'X_s_test', 'X_s_var',
 'X_v_dev', 'X_v_test', 'X_v_var',
 'Y_dev', 'Y_test']
=======
Sensor window (50 timesteps × 18 features)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  CNN Block (Conv1d × 2)       │
    │  Local temporal patterns      │
    │  (B, 50, 18) → (B, 50, 128)   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  LSTM Block (2-layer)         │
    │  Degradation trajectory       │
    │  (B, 50, 128) → (B, 128)      │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Regression Head              │
    │  Linear(128→64→1)             │
    │  (B, 128) → (B,)              │
    └───────────────┬───────────────┘
                    │
                    ▼
            RUL prediction (float)
            "This machine has ~22 shift-cycles left"
>>>>>>> Stashed changes
```

**The `_dev` suffix = training data. The `_test` suffix = test data.** The dataset is **pre-split** for you. You do NOT manually split by unit IDs — it's already done inside the `.h5` file. The `_var` keys contain variable name strings for reference.

**If you use `f['W']` you will get a `KeyError`. Use `f['W_dev']` and `f['W_test']`.**

---

## 1. Dataset: N-CMAPSS DS02

**File:** `N-CMAPSS_DS02-006.h5` (≈2.45 GB) from `kaggle.com/datasets/shreyaravi0/aircraft`

This is the N-CMAPSS (New CMAPSS) dataset — turbofan engine degradation under realistic flight conditions. DS02 is specifically for data-driven prognostics (not DS01 which is for model-based diagnostics).

### 1.1 Corrected HDF5 Schema

| HDF5 Key | Split | Contents | Shape |
|---|---|---|---|
| `W_dev` | Train | Operating conditions (Mach, altitude, TRA, T2) | `(N_train, 4)` |
| `W_test` | Test | Operating conditions | `(N_test, 4)` |
| `X_s_dev` | Train | Physical sensor readings (14 sensors) | `(N_train, 14)` |
| `X_s_test` | Test | Physical sensor readings | `(N_test, 14)` |
| `X_v_dev` | Train | Virtual sensor readings — **DO NOT USE** | `(N_train, 14)` |
| `X_v_test` | Test | Virtual sensor readings — **DO NOT USE** | `(N_test, 14)` |
| `T_dev` | Train | Health state auxiliary variables | `(N_train, 4)` |
| `T_test` | Test | Health state auxiliary variables | `(N_test, 4)` |
| `Y_dev` | Train | RUL ground truth labels | `(N_train, 1)` |
| `Y_test` | Test | RUL ground truth labels | `(N_test, 1)` |
| `A_dev` | Train | Unit/cycle metadata | `(N_train, 4)` |
| `A_test` | Test | Unit/cycle metadata | `(N_test, 4)` |
| `W_var` | — | Variable names for W columns | string array |
| `X_s_var` | — | Variable names for X_s columns | string array |
| `X_v_var` | — | Variable names for X_v columns | string array |
| `A_var` | — | Variable names for A columns | string array |
| `T_var` | — | Variable names for T columns | string array |

### 1.2 What You Actually Use

**Model input:** `X_s` (14 physical sensors) + `W` (4 operating conditions) = **18 features per timestep.**

You concatenate `[W, X_s]` along axis=1 to get your 18-column feature matrix.

Do **NOT** use `X_v` (virtual sensors) — they're model-derived and add complexity without added DL challenge. Do **NOT** use `T` (health state) — those leak future information.

### 1.3 Train / Test Split

The split is baked into the file. Based on the published protocol:

| Set | HDF5 suffix | Engine Units |
|---|---|---|
| Development (train) | `_dev` | 2, 5, 10, 16, 18, 20 (6 units) |
| Test | `_test` | 11, 14, 15 (3 units) |

Verify this on Day 1 by printing unique unit IDs from `A_dev[:, 0]` and `A_test[:, 0]`.

---

## 2. Data Pipeline — Complete, Corrected Code

This is the code you should actually run. Copy-paste safe.

```python
# dl_engine/dataset.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# ════════════════════════════════════════════════════════════════════
# STEP 1: Load the .h5 file with CORRECT keys
# ════════════════════════════════════════════════════════════════════

def load_ncmapss(h5_path, sampling=10):
    """
    Load N-CMAPSS DS02 from the Kaggle dataset.

    Args:
        h5_path:  Path to N-CMAPSS_DS02-006.h5 (or whatever the file is named)
        sampling: Take every Nth sample (10 = 0.1Hz, 1 = full 1Hz).
                  Use 10 to start. Try 1 only if memory allows on H100.

    Returns:
        W_train, Xs_train, Y_train, A_train,
        W_test,  Xs_test,  Y_test,  A_test
    """
    with h5py.File(h5_path, 'r') as f:
        # Print keys to verify — do this on your FIRST run
        print("HDF5 keys:", list(f.keys()))

        # ── Training (development) set ──
        W_train  = np.array(f['W_dev'],   dtype=np.float32)[::sampling]
        Xs_train = np.array(f['X_s_dev'], dtype=np.float32)[::sampling]
        Y_train  = np.array(f['Y_dev'],   dtype=np.float32)[::sampling]
        A_train  = np.array(f['A_dev'],   dtype=np.float32)[::sampling]

        # ── Test set ──
        W_test   = np.array(f['W_test'],   dtype=np.float32)[::sampling]
        Xs_test  = np.array(f['X_s_test'], dtype=np.float32)[::sampling]
        Y_test   = np.array(f['Y_test'],   dtype=np.float32)[::sampling]
        A_test   = np.array(f['A_test'],   dtype=np.float32)[::sampling]

        # ── Variable names (for EDA, optional) ──
        W_var  = [s.decode() for s in np.array(f['W_var'])]
        Xs_var = [s.decode() for s in np.array(f['X_s_var'])]
        print(f"W  columns: {W_var}")
        print(f"Xs columns: {Xs_var}")

    print(f"\nTrain: W={W_train.shape}, Xs={Xs_train.shape}, Y={Y_train.shape}, A={A_train.shape}")
    print(f"Test:  W={W_test.shape},  Xs={Xs_test.shape},  Y={Y_test.shape},  A={A_test.shape}")

    # Verify unit IDs
    train_units = np.unique(A_train[:, 0].astype(int))
    test_units  = np.unique(A_test[:, 0].astype(int))
    print(f"\nTrain units: {train_units}")
    print(f"Test units:  {test_units}")

    return W_train, Xs_train, Y_train, A_train, W_test, Xs_test, Y_test, A_test


# ════════════════════════════════════════════════════════════════════
# STEP 2: Concatenate features and normalize
# ════════════════════════════════════════════════════════════════════

def prepare_features(W_train, Xs_train, Y_train, W_test, Xs_test, Y_test):
    """
    Concatenate W + Xs → 18 features, fit MinMaxScaler on train only.

    Returns:
        X_train_scaled, Y_train_flat,
        X_test_scaled,  Y_test_flat,
        scaler   (SAVE THIS — critical for inference)
    """
    # Concatenate: (N, 4) + (N, 14) → (N, 18)
    X_train = np.concatenate([W_train, Xs_train], axis=1)
    X_test  = np.concatenate([W_test,  Xs_test],  axis=1)

    # Flatten Y from (N, 1) → (N,)
    Y_train_flat = Y_train.squeeze(-1)
    Y_test_flat  = Y_test.squeeze(-1)

    # Min-Max normalize: fit on TRAIN, transform BOTH
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\nScaled X_train: {X_train_scaled.shape}, range [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    print(f"Scaled X_test:  {X_test_scaled.shape},  range [{X_test_scaled.min():.3f}, {X_test_scaled.max():.3f}]")
    print(f"Y_train range:  [{Y_train_flat.min():.1f}, {Y_train_flat.max():.1f}]")

    return X_train_scaled, Y_train_flat, X_test_scaled, Y_test_flat, scaler


# ════════════════════════════════════════════════════════════════════
# STEP 3: Sliding window dataset
# ════════════════════════════════════════════════════════════════════

WINDOW = 50
STRIDE = 1

class NCMAPSSDataset(Dataset):
    """
    Converts a flat (N, 18) feature matrix + (N,) RUL vector
    into sliding windows of shape (num_windows, 50, 18).

    IMPORTANT: Windows must NOT cross unit boundaries.
    If you naively slide across the whole array, windows near
    unit transitions will contain data from two different engines.
    This is wrong and will silently corrupt your training.
    """
    def __init__(self, X, Y, A, window=50, stride=1):
        """
        Args:
            X: (N, 18) scaled feature matrix
            Y: (N,) RUL labels
            A: (N, 4+) metadata — column 0 is unit_id
            window: sliding window length
            stride: step between windows
        """
        self.samples = []
        self.labels  = []

        unit_ids = A[:, 0].astype(int)
        unique_units = np.unique(unit_ids)

        for uid in unique_units:
            mask = unit_ids == uid
            X_unit = X[mask]
            Y_unit = Y[mask]

            # Slide within this unit only
            for i in range(0, len(X_unit) - window, stride):
                self.samples.append(X_unit[i : i + window])
                self.labels.append(Y_unit[i + window - 1])  # RUL at end of window

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels  = np.array(self.labels,  dtype=np.float32)
        print(f"  Created {len(self.samples)} windows (window={window}, stride={stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx]),  # (50, 18)
            torch.tensor(self.labels[idx])    # scalar
        )


# ════════════════════════════════════════════════════════════════════
# STEP 4: Full pipeline — call this from your notebook
# ════════════════════════════════════════════════════════════════════

def build_dataloaders(h5_path, batch_size=512, sampling=10, window=50, stride=1):
    """
    End-to-end: h5 file → DataLoaders ready for training.
    """
    # Load
    W_tr, Xs_tr, Y_tr, A_tr, W_te, Xs_te, Y_te, A_te = load_ncmapss(h5_path, sampling)

    # Prepare features + scaler
    X_tr, Y_tr_flat, X_te, Y_te_flat, scaler = prepare_features(
        W_tr, Xs_tr, Y_tr, W_te, Xs_te, Y_te
    )

    # Save scaler NOW — you will need this for inference
    joblib.dump(scaler, '/kaggle/working/scaler.pkl')
    print("\n✅ Scaler saved to /kaggle/working/scaler.pkl")

    # Build datasets (respecting unit boundaries)
    print("\nBuilding train windows...")
    train_ds = NCMAPSSDataset(X_tr, Y_tr_flat, A_tr, window, stride)
    print("Building test windows...")
    test_ds  = NCMAPSSDataset(X_te, Y_te_flat, A_te, window, stride)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, test_loader, scaler
```

### 2.1 Why Windows Must Respect Unit Boundaries

This is subtle and easy to miss. The flat arrays contain data from multiple engine units concatenated together. If you do a naive sliding window across the entire array, some windows will straddle two different engines — e.g., the last 20 timesteps of Unit 2 followed by the first 30 timesteps of Unit 5. This is physically meaningless and will silently corrupt training. The `NCMAPSSDataset` class above handles this by windowing **per unit**.

### 2.2 Subsampling Strategy

| `sampling` | Effective rate | Memory | Use when |
|---|---|---|---|
| `10` | 0.1 Hz | ~1/10th | First pipeline validation, debugging, hyperparameter search |
| `5` | 0.2 Hz | ~1/5th | Intermediate — good accuracy/speed tradeoff |
| `1` | 1.0 Hz (full) | Full (~2.5 GB raw) | Final training run if H100 memory allows |

**Start with `sampling=10`. Always.** Validate the entire pipeline works before scaling up.

---

## 3. Model Architecture

```python
# dl_engine/model.py

import torch
import torch.nn as nn

class CNNLSTM_RUL(nn.Module):
    """
    CNN-LSTM for Remaining Useful Life regression.

    Architecture:
        Input (B, 50, 18)
          → Conv1d block (local temporal pattern extraction)
          → LSTM block (degradation trajectory modeling)
          → Regression head (single RUL float)

    Why CNN before LSTM?
        Conv1d captures short-range sensor correlations (vibration spikes,
        thermal transients) within a few timesteps. LSTM then models the
        longer-range degradation trend across the full 50-step window.

    Why BatchNorm1d?
        N-CMAPSS has massive intra-unit variance across flight phases
        (takeoff vs cruise vs descent). BatchNorm stabilizes gradients.
    """
    def __init__(self, n_features=18, window=50, cnn_filters=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # ── CNN Block ──
        # Conv1d expects (Batch, Channels, Length)
        # We permute (B, 50, 18) → (B, 18, 50) before this
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── LSTM Block ──
        # After CNN + permute back: (B, 50, 128) → LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,  # 128
            hidden_size=lstm_hidden,     # 128
            num_layers=lstm_layers,      # 2
            batch_first=True,
            dropout=dropout
        )

        # ── Regression Head ──
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, 50, 18) — Batch × Window × Features
        x = x.permute(0, 2, 1)           # → (B, 18, 50)  for Conv1d
        x = self.cnn(x)                   # → (B, 128, 50)
        x = x.permute(0, 2, 1)           # → (B, 50, 128) for LSTM
        _, (h_n, _) = self.lstm(x)        # h_n: (2, B, 128)
        x = h_n[-1]                       # → (B, 128) last layer hidden
        return self.regressor(x).squeeze(-1)  # → (B,) RUL predictions
```

### 3.1 Tensor Shape Trace — Print This Out

```
Input:              (B, 50, 18)     ← DataLoader output
After permute:      (B, 18, 50)     ← Conv1d needs (B, Channels, Length)
After Conv1d #1:    (B, 64, 50)     ← 64 filters, kernel=3, padding=1 preserves length
After Conv1d #2:    (B, 128, 50)    ← 128 filters
After permute back: (B, 50, 128)    ← LSTM needs (B, Length, Features)
LSTM h_n:           (2, B, 128)     ← 2 layers, last timestep hidden state
h_n[-1]:            (B, 128)        ← take last layer only
After Linear(128→64): (B, 64)
After Linear(64→1):   (B, 1)
After squeeze:         (B,)         ← final RUL prediction per sample
```

### 3.2 Parameter Count

```python
model = CNNLSTM_RUL()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,}")
print(f"Trainable:    {trainable:,}")
# Expected: ~400K–500K parameters — lightweight, trains fast
```

---

## 4. Training Loop — Complete

```python
# dl_engine/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time

def train_model(model, train_loader, test_loader, config, device='cuda'):
    """
    Full training loop with checkpointing, early stopping, and logging.
    Run this inside your Kaggle notebook.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

    for epoch in range(config['epochs']):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)   # (B, 50, 18)
            Y_batch = Y_batch.to(device)   # (B,)

            optimizer.zero_grad()
            pred = model(X_batch)           # (B,)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_losses.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_true.append(Y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_rmse   = np.sqrt(val_loss)

        # ── Scheduler step ──
        scheduler.step(val_loss)

        # ── Logging ──
        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_rmse:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s")

        # ── Checkpoint ── (EVERY improvement, not just final)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'config': config
            }, '/kaggle/working/best_model.pt')
            print(f"  ✅ Checkpoint saved (val_rmse={val_rmse:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop']:
                print(f"\n⛔ Early stopping at epoch {epoch+1} (no improvement for {config['early_stop']} epochs)")
                break

    return history


# ── Config ──
CONFIG = {
    "window":       50,
    "batch_size":   512,
    "lr":           1e-3,
    "epochs":       50,
    "early_stop":   10,
    "dropout":      0.3,
    "lstm_hidden":  128,
    "cnn_filters":  64,
}
```

### 4.1 Asymmetric Scoring Function (Secondary Metric)

The PHM community uses an asymmetric scoring function where **late predictions are penalized exponentially more** than early predictions. This makes physical sense — predicting a machine will last longer than it actually does is far more dangerous than being cautious.

```python
def phm_score(y_true, y_pred):
    """
    Asymmetric scoring function for RUL prediction.
    Late predictions (d > 0, i.e., predicted RUL > actual) are penalized harder.
    """
    d = y_pred - y_true  # positive = late (dangerous), negative = early (safe)
    scores = np.where(d < 0,
                      np.exp(-d / 13.0) - 1,   # early: gentle penalty
                      np.exp( d / 10.0) - 1)   # late:  harsh penalty
    return np.sum(scores)
```

---

## 5. Inference API — THE CONTRACT

This is the file that Team Agent and Team UI will import. **Do not change the function signature.**

```python
# dl_engine/inference.py

import torch
import joblib
import numpy as np
from .model import CNNLSTM_RUL

_model  = None
_scaler = None

def load_model(weights_path='dl_engine/weights/best_model.pt',
               scaler_path='dl_engine/weights/scaler.pkl'):
    """Load model and scaler. Called automatically on first predict_rul() call."""
    global _model, _scaler
    checkpoint = torch.load(weights_path, map_location='cpu')  # CPU-safe for GTX 1650
    _model = CNNLSTM_RUL()
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    _scaler = joblib.load(scaler_path)
    print(f"✅ Model loaded (trained to epoch {checkpoint.get('epoch', '?')}, "
          f"val_rmse={checkpoint.get('val_rmse', '?')})")


def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    THE CONTRACT FUNCTION — do not change this signature.

    Args:
        sensor_tensor: numpy array of shape (50, 18)
                       Raw (pre-normalization) sensor values.
                       Columns 0-3 = W (operating conditions)
                       Columns 4-17 = Xs (physical sensors)

    Returns:
        float: RUL prediction in production shift-cycles, clamped to >= 0.0

    Notes:
        - Scaling is handled internally via the saved scaler.
        - Model runs on CPU. No GPU required.
        - First call triggers lazy model loading (~2s).
        - Subsequent calls are <10ms.
    """
    if _model is None:
        load_model()

    # Validate input shape
    assert sensor_tensor.shape == (50, 18), \
        f"Expected shape (50, 18), got {sensor_tensor.shape}"

    # Scale using the SAME scaler that was fit during training
    scaled = _scaler.transform(sensor_tensor)  # (50, 18)

    # Convert to tensor and add batch dimension
    t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 50, 18)

    # Predict
    with torch.no_grad():
        rul = _model(t).item()

    return max(0.0, rul)  # RUL cannot be negative
```

### 5.1 What Other Teams Send You

Team Agent's Diagnostic Agent produces a numpy array of shape `(50, 18)` containing **raw sensor values** (not pre-scaled). Your `predict_rul()` function handles the scaling internally. This is by design — it means other teams never need to touch your scaler.

### 5.2 What You Return

A single `float`. Represents RUL in shift-cycles. Typical range depends on training data — could be 0 to ~125+ cycles. The Capacity Agent uses thresholds on this number:

| Your output | What happens downstream |
|---|---|
| RUL > 30 | Machine stays ONLINE (full capacity) |
| 15 < RUL ≤ 30 | Machine goes DEGRADED (50% capacity) |
| RUL ≤ 15 | Machine goes OFFLINE (0% capacity, maintenance triggered) |

You don't need to know or care about these thresholds. Just return an accurate number.

---

## 6. GTX 1650 Smoke Test — Day 5

Do NOT wait until integration day. Run this on Day 5 with random weights:

```python
# Run this on the LOCAL machine, not Kaggle
import torch
import numpy as np
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 1. Import your model class
from dl_engine.model import CNNLSTM_RUL

# 2. Create + save random model
model = CNNLSTM_RUL()
torch.save({'model_state_dict': model.state_dict(), 'epoch': 0}, 'test_weights.pt')

# 3. Reload on CPU (exactly like inference.py does)
checkpoint = torch.load('test_weights.pt', map_location='cpu')
model2 = CNNLSTM_RUL()
model2.load_state_dict(checkpoint['model_state_dict'])
model2.eval()

# 4. Forward pass with correct shape
dummy = torch.randn(1, 50, 18)
with torch.no_grad():
    out = model2(dummy)

print(f"\nOutput shape: {out.shape}")     # Expected: torch.Size([1])
print(f"Output value: {out.item():.4f}")  # Expected: some random float
print(f"\n✅ Smoke test passed!")

# 5. Clean up
import os
os.remove('test_weights.pt')
```

**Also check:** Print `torch.__version__` on both Kaggle and your local machine. If they differ (e.g., Kaggle has 2.1 and local has 2.0), `torch.load()` can fail silently or throw cryptic errors. Pin the same version.

---

## 7. Checkpoint Artifacts — What to Commit

After training is complete, download these two files from Kaggle output and commit to the repo:

| File | Repo path | Size (approx) | What it contains |
|---|---|---|---|
| `best_model.pt` | `dl_engine/weights/best_model.pt` | ~2–5 MB | Model state dict, optimizer state, epoch, val_rmse, config |
| `scaler.pkl` | `dl_engine/weights/scaler.pkl` | ~2 KB | Fitted MinMaxScaler (18 feature ranges from training data) |

**Both files must be committed before Day 10 (Integration Day).**

---

## 8. Your Day-by-Day Checklist

| Day | Task | Done? |
|---|---|---|
| 1 | Download dataset from Kaggle source, run `list(f.keys())`, verify `_dev`/`_test` keys, print shapes and unit IDs | ☐ |
| 2 | Implement `dataset.py` with corrected keys, validate `build_dataloaders()` runs end-to-end with `sampling=10` | ☐ |
| 3 | Implement `model.py`, run one training epoch on small subset, verify loss decreases | ☐ |
| 4 | Full training run #1 with `sampling=10`, checkpoint saves working | ☐ |
| 5 | Analyze run #1 results. **GTX 1650 smoke test with random weights.** Record PyTorch version on both machines. | ☐ |
| 6 | Training run #2 — tune hyperparameters based on run #1 (try `sampling=5` or `1` if memory allows) | ☐ |
| 7 | Validate on test units — compute RMSE and PHM score. Is RMSE acceptable? | ☐ |
| 8 | Export `best_model.pt` + `scaler.pkl` from Kaggle. Commit to repo. | ☐ |
| 9 | Buffer — retrain if RMSE too high. Local inference test on GTX 1650 with real weights. | ☐ |
| **10** | **INTEGRATION: swap dummy oracle → your inference.py.** Verify end-to-end with Team Agent + Team UI. | ☐ |
| 11–13 | Integration bugs, rehearsals. | ☐ |
| 14 | Final bug fixes only. | ☐ |

---

## 9. Common Failure Modes and Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: "object 'W' doesn't exist"` | Using old key names. Dataset has `W_dev` / `W_test` suffixed keys. | Use `f['W_dev']` not `f['W']`. |
| OOM on Kaggle H100 | Full 1Hz data + large batch size | Use `sampling=10`, reduce `batch_size` to 256 |
| Val loss never decreases | Learning rate too high, or data pipeline bug | Check that scaler is fit on train only. Try `lr=3e-4`. |
| RMSE looks good on train, terrible on test | Overfitting OR windows crossing unit boundaries | Ensure windowing respects unit boundaries (check `NCMAPSSDataset`). Add more dropout. |
| `torch.load()` fails on GTX 1650 | PyTorch version mismatch | Pin same version. Use `map_location='cpu'` always. |
| RUL predictions are all ~0 or all ~same number | Scaler not saved/loaded correctly, or model never converged | Verify `scaler.pkl` was saved from the same run. Check that `val_loss` actually decreased during training. |
| Inference returns negative numbers | Model regression head has no floor | Already handled — `predict_rul()` clamps with `max(0.0, rul)`. |

---

*You own `/dl_engine/*`. Your only deliverable to other teams is `predict_rul(sensor_tensor) -> float`. Make it work, make it accurate, make it fast.*
