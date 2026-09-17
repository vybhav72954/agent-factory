"""
research/pronostia/train_model.py

Train a third CNN-LSTM RUL checkpoint on real PRONOSTIA bearings.

Same architecture (dl_engine/model.py::CNNLSTM_RUL) and the same (50, 18)
window contract as the turbofan and simulator checkpoints, so the production
injection code (`agents.diagnostic_agent._inject_spike`) and
`dl_engine.inference.load_model(weights, scaler)` work unchanged.

Channel mapping (only physically measured quantities go into semantically
matching slots; everything the PRONOSTIA rig does not measure is held
constant, so injecting into it has no effect):

    col  sensor  ForgeMind name   PRONOSTIA source
    0    W0      Motor RPM        shaft speed (rpm, per operating condition)
    2    W2      Power kW / load  radial load (N, per operating condition)
    4    Xs0     Vibration X      log10 horizontal RMS acceleration (g)
    5    Xs1     Vibration Y      log10 vertical RMS acceleration (g)
    6    Xs2     Bearing Temp     PT100 bearing temperature (°C)
    16   Xs12    Acoustic dB      high-frequency (5–12.8 kHz) vibration power (dB)
    all other columns             constant 0

Bearings (`--bearings`):
    all   (default) all 17 run-to-failure bearings. The 8 without temperature data
          get a constant temperature: the median early-life temperature of the
          temperature-equipped bearings under the same operating condition.
          Validation: Bearing1_3, Bearing2_2 (long degradation phases).
    temp  only the 9 bearings with temperature. Validation: Bearing1_7, Bearing2_6.
          First version (2026-09-16); too little degradation data: degradation-phase
          validation RMSE 30–46 and a mostly flat response surface. Kept for the record
          (weights backed up as *_temp9.*.bak).

Label: piecewise-linear RUL anchored at the first predicting time (FPT), the
standard PRONOSTIA labelling. Real bearings show no measurable degradation for
most of their life (median vibration onset at 96% of life, see calibrate.py),
so a linear percent-of-life label is not learnable: a first run with that label
collapsed to predicting the mean (val RMSE 28.9) and then overfit
bearing-specific signatures (val RMSE 43). Here:
    RUL = 100                                   before FPT
    RUL = 100 * (n - 1 - i) / (n - 1 - FPT)     from FPT to failure
FPT is the first of 5 consecutive snapshots whose trailing 11-point median of
log10 horizontal RMS exceeds the mean + 3 SD of the first 10% of life. The
pipeline thresholds (RUL <= 30 DEGRADED, <= 15 OFFLINE) therefore mean the
last 30% / 15% of the degradation phase.

Healthy windows (label 100) outnumber degradation-phase windows about 3:1, so
training batches are drawn with a weighted sampler giving each group half the
probability mass, and the checkpoint is selected on degradation-phase
validation RMSE (windows at or after FPT) rather than all-window RMSE.

The first 49 snapshots are front-padded with the first reading (as the
simulator's get_window does) so early life is represented.

Usage:
    python -m research.pronostia.train_model [--bearings all|temp] [--epochs 60] [--device cuda]

Output:
    dl_engine/weights/best_model_pronostia.pt
    dl_engine/weights/scaler_pronostia.pkl
    research/pronostia/pronostia_model_meta.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.capacity_agent import RUL_DEGRADED_THRESHOLD, RUL_OFFLINE_THRESHOLD
from dl_engine.model import CNNLSTM_RUL
from research.pronostia.features import FEATURES_PATH

WEIGHTS_PATH = PROJECT_ROOT / "dl_engine" / "weights" / "best_model_pronostia.pt"
SCALER_PATH = PROJECT_ROOT / "dl_engine" / "weights" / "scaler_pronostia.pkl"
META_PATH = PROJECT_ROOT / "research" / "pronostia" / "pronostia_model_meta.json"

WINDOW = 50
N_FEATURES = 18
VAL_BEARINGS = {"all": ["Bearing1_3", "Bearing2_2"], "temp": ["Bearing1_7", "Bearing2_6"]}
CHANNELS = {                       # column index -> (sensor id, feature column, transform)
    0:  ("W0",   "speed_rpm", None),
    2:  ("W2",   "load_n",    None),
    4:  ("Xs0",  "h_rms",     "log10"),
    5:  ("Xs1",  "v_rms",     "log10"),
    6:  ("Xs2",  "temp_c",    None),
    16: ("Xs12", "hf_db",     None),
}
SEED = 20260916
FPT_SIGMA = 3.0
FPT_CONSECUTIVE = 5
FPT_SMOOTH = 11


def first_predicting_time(h_rms: np.ndarray) -> int:
    """Index of degradation onset: 5 consecutive smoothed log-RMS points above early-life mean + 3 SD."""
    s = pd.Series(np.log10(np.maximum(h_rms, 1e-6))).rolling(FPT_SMOOTH, min_periods=1).median().to_numpy()
    base = s[: max(20, len(s) // 10)]
    above = (s > base.mean() + FPT_SIGMA * base.std()).astype(int)
    runs = np.convolve(above, np.ones(FPT_CONSECUTIVE, dtype=int), mode="valid") == FPT_CONSECUTIVE
    return int(np.argmax(runs)) if runs.any() else len(s) - 1


def fpt_labels(n: int, fpt: int) -> np.ndarray:
    """Piecewise-linear RUL: 100 before FPT, linear to 0 at failure."""
    i = np.arange(n)
    span = max(1, n - 1 - fpt)
    return np.where(i < fpt, 100.0, 100.0 * (n - 1 - i) / span).astype(np.float32)


def fill_missing_temperature(feats: pd.DataFrame) -> pd.DataFrame:
    """Constant temperature for bearings without temperature data: the median
    early-life (first 10%) temperature of temperature-equipped bearings under the
    same operating condition."""
    feats = feats.copy()
    feats["condition"] = feats["bearing"].str[len("Bearing")]
    early = (feats[feats["has_temp"].astype(bool)]
             .sort_values(["bearing", "snapshot"])
             .groupby("bearing", group_keys=False)
             .apply(lambda g: g.head(max(5, len(g) // 10))))
    healthy_by_condition = early.groupby("condition")["temp_c"].median()
    missing = ~feats["has_temp"].astype(bool)
    feats.loc[missing, "temp_c"] = feats.loc[missing, "condition"].map(healthy_by_condition)
    return feats


def snapshot_matrix(g: pd.DataFrame) -> np.ndarray:
    """(n_snapshots, 18) raw matrix for one bearing using the channel mapping."""
    x = np.zeros((len(g), N_FEATURES), dtype=np.float32)
    for col, (_sid, feature, transform) in CHANNELS.items():
        v = g[feature].to_numpy(dtype=np.float64)
        x[:, col] = np.log10(np.maximum(v, 1e-6)) if transform == "log10" else v
    return x


def bearing_windows(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """All (50, 18) windows for one bearing (front-padded), their FPT-based RUL labels, and the FPT index."""
    g = g.sort_values("snapshot")
    x = snapshot_matrix(g)
    padded = np.concatenate([np.repeat(x[:1], WINDOW - 1, axis=0), x], axis=0)
    idx = np.arange(len(x))[:, None] + np.arange(WINDOW)[None, :]
    fpt = first_predicting_time(g["h_rms"].to_numpy())
    return padded[idx], fpt_labels(len(x), fpt), fpt


def load_features(bearings: str = "all") -> pd.DataFrame:
    """Feature table for the chosen bearing set, with temperature filled where needed."""
    feats = pd.read_csv(FEATURES_PATH)
    if bearings == "temp":
        return feats[feats["has_temp"].astype(bool)].copy()
    return fill_missing_temperature(feats)


def status(rul: np.ndarray) -> np.ndarray:
    return np.where(rul <= RUL_OFFLINE_THRESHOLD, "OFFLINE",
                    np.where(rul <= RUL_DEGRADED_THRESHOLD, "DEGRADED", "ONLINE"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bearings", choices=["all", "temp"], default="all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    feats = load_features(args.bearings)
    val_bearings = VAL_BEARINGS[args.bearings]
    bearings = sorted(feats["bearing"].unique())
    train_bearings = [b for b in bearings if b not in val_bearings]
    print(f"[train] {len(bearings)} bearings; train={train_bearings}  val={val_bearings}")

    windows = {b: bearing_windows(g) for b, g in feats.groupby("bearing")}
    fpt_frac = {b: round(w[2] / (len(w[1]) - 1), 3) for b, w in windows.items()}
    X_tr = np.concatenate([windows[b][0] for b in train_bearings])
    y_tr = np.concatenate([windows[b][1] for b in train_bearings])
    X_va = np.concatenate([windows[b][0] for b in val_bearings])
    y_va = np.concatenate([windows[b][1] for b in val_bearings])
    va_degr = np.concatenate([np.arange(len(windows[b][1])) >= windows[b][2] for b in val_bearings])

    # Scaler fitted on training snapshots only (not on windows, to avoid padding weight).
    train_rows = np.concatenate([snapshot_matrix(g.sort_values("snapshot"))
                                 for b, g in feats.groupby("bearing") if b in train_bearings])
    scaler = MinMaxScaler().fit(train_rows)

    def scale(X: np.ndarray) -> torch.Tensor:
        s = scaler.transform(X.reshape(-1, N_FEATURES)).reshape(X.shape)
        return torch.tensor(np.clip(s, 0.0, 1.0), dtype=torch.float32)

    healthy = y_tr >= 100.0
    weights = np.where(healthy, 0.5 / healthy.sum(), 0.5 / max(1, (~healthy).sum()))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(y_tr),
                                    replacement=True, generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(TensorDataset(scale(X_tr), torch.tensor(y_tr)), batch_size=args.batch_size,
                              sampler=sampler)
    X_va_t = scale(X_va)

    device = torch.device(args.device)
    model = CNNLSTM_RUL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.MSELoss()
    print(f"[train] device={device}  n_train={len(y_tr)} ({(~healthy).sum()} degradation-phase)  "
          f"n_val={len(y_va)} ({va_degr.sum()} degradation-phase)")

    def predict(X_t: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return np.maximum(torch.cat([model(X_t[i:i + 512].to(device)).cpu()
                                         for i in range(0, len(X_t), 512)]).numpy(), 0.0)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, patience_left = float("inf"), 0, args.patience
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        pred = predict(X_va_t)
        rmse_all = float(np.sqrt(np.mean((pred - y_va) ** 2)))
        rmse_degr = float(np.sqrt(np.mean((pred[va_degr] - y_va[va_degr]) ** 2)))
        scheduler.step(rmse_degr)
        improved = rmse_degr < best
        print(f"  epoch {epoch:2d}  train_mse={np.mean(losses):8.2f}  val_rmse_all={rmse_all:6.2f}  "
              f"val_rmse_degradation={rmse_degr:6.2f}  {'*' if improved else ' '}  {time.time() - t0:.0f}s",
              flush=True)
        if improved:
            best, best_epoch, patience_left = rmse_degr, epoch, args.patience
            torch.save(model.state_dict(), WEIGHTS_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate the best checkpoint on each validation bearing, all windows and degradation phase.
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    per_bearing = {}
    all_pred = []
    for b in val_bearings:
        Xb, yb, fpt = windows[b]
        pred = predict(scale(Xb))
        all_pred.append(pred)
        d = slice(fpt, None)
        per_bearing[b] = {
            "n": int(len(yb)), "n_degradation": int(len(yb) - fpt),
            "rmse": float(np.sqrt(np.mean((pred - yb) ** 2))),
            "status_accuracy": float(np.mean(status(pred) == status(yb))),
            "rmse_degradation": float(np.sqrt(np.mean((pred[d] - yb[d]) ** 2))),
            "status_accuracy_degradation": float(np.mean(status(pred[d]) == status(yb[d]))),
        }
    pred_all = np.concatenate(all_pred)

    # Healthy reference window: median of the first 10% of life of the training bearings.
    early = pd.concat([g.sort_values("snapshot").head(max(5, len(g) // 10))
                       for b, g in feats.groupby("bearing") if b in train_bearings])
    healthy_raw = np.median(snapshot_matrix(early), axis=0)

    meta = {
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "bearing_set": args.bearings,
        "architecture": "CNNLSTM_RUL default config (dl_engine/model.py)",
        "window": WINDOW,
        "label": "piecewise-linear RUL: 100 before first predicting time (FPT), linear to 0 at failure",
        "fpt_rule": f"{FPT_CONSECUTIVE} consecutive points of trailing {FPT_SMOOTH}-pt median log10 h_rms "
                    f"> early-life mean + {FPT_SIGMA} SD",
        "fpt_life_fraction": fpt_frac,
        "temperature_fill": ("constant per operating condition (median early-life temperature of "
                             "temperature-equipped bearings)") if args.bearings == "all" else "none",
        "sampling": "weighted: healthy and degradation-phase windows each 50% of draws",
        "model_selection": "degradation-phase validation RMSE",
        "channels": {str(c): {"sensor_id": s, "feature": f, "transform": t} for c, (s, f, t) in CHANNELS.items()},
        "constant_columns": [c for c in range(N_FEATURES) if c not in CHANNELS],
        "train_bearings": train_bearings,
        "val_bearings": val_bearings,
        "best_epoch": best_epoch,
        "val_rmse_degradation": float(best),
        "val_rmse": float(np.sqrt(np.mean((pred_all - y_va) ** 2))),
        "val_per_bearing": per_bearing,
        "val_prediction_quantiles": {q: float(np.quantile(pred_all, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "healthy_raw": healthy_raw.tolist(),
        "seed": SEED,
        "device": str(device),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] best epoch {best_epoch}  degradation-phase val RMSE {best:.2f}")
    for b, m in per_bearing.items():
        print(f"  {b}: all RMSE {m['rmse']:.2f} / status {m['status_accuracy']:.1%}   "
              f"degradation RMSE {m['rmse_degradation']:.2f} / status {m['status_accuracy_degradation']:.1%} "
              f"(n={m['n_degradation']})")
    for p in (WEIGHTS_PATH, SCALER_PATH, META_PATH):
        print(f"[train] wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
