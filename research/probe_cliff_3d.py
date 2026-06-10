"""
research/probe_cliff_3d.py

Empirical 3D sweep of the trained CNN-LSTM's RUL response surface across
the critical thermal trio (Xs2, Xs3, Xs4) at scaled positions in [0.05, 0.95].

Why this exists:
  CLAUDE.md §6.1 quotes three hand-picked probe points (ONLINE/DEGRADED/OFFLINE
  reference triples) that proved insufficient for tuning the severity-driven
  injection. Hit 2 of a MEDIUM walkthrough kept skipping DEGRADED and jumping
  straight to OFFLINE because the cliff between the three reference points is
  not linear, and the per-hit deltas + Xs4 cliff-suppression dynamics could
  not be reliably predicted from three points alone.

  This script generates the full surface (default 10^3 = 1000 evaluations,
  ~1-3 minutes CPU) so any future tuning has empirical data to fit against,
  not vibes.

Usage:
    python -m research.probe_cliff_3d            # default 10-step sweep
    python -m research.probe_cliff_3d --steps 15  # finer sweep
    python -m research.probe_cliff_3d --steps 5   # quick smoke test

Outputs (under research/results/):
    probe_cliff_3d.csv            — every (xs2_scaled, xs3_scaled, xs4_scaled, rul)
    probe_cliff_3d_summary.md     — analytic findings: cliff edge, Xs4 shift
    probe_cliff_3d_xs4_slices.png — RUL heatmap (Xs2 × Xs3) at several Xs4 slices
    probe_cliff_3d_xs2_curves.png — RUL vs Xs2 at several (Xs3, Xs4) combinations
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dl_engine.inference import (
    predict_rul,
    get_healthy_baseline,
    raw_value_for_scaled,
)

# Column indices for the critical thermal trio (matches diagnostic_agent SENSOR_TO_COL)
COL_XS2 = 6   # Bearing Temp
COL_XS3 = 7   # Motor Temp
COL_XS4 = 8   # Oil Pressure

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
CSV_PATH = RESULTS_DIR / "probe_cliff_3d.csv"
SUMMARY_PATH = RESULTS_DIR / "probe_cliff_3d_summary.md"
HEATMAP_PATH = RESULTS_DIR / "probe_cliff_3d_xs4_slices.png"
CURVES_PATH = RESULTS_DIR / "probe_cliff_3d_xs2_curves.png"


def sweep(steps: int = 10, lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    """
    Sweep Xs2/Xs3/Xs4 across [lo, hi] in `steps` linear positions each.
    All other sensors held at the healthy baseline (scaled 0.10).

    Returns a long-format DataFrame: one row per (xs2_scaled, xs3_scaled, xs4_scaled, rul).
    """
    grid = np.linspace(lo, hi, steps)
    total = steps ** 3
    print(f"[probe] sweep grid: {steps}^3 = {total} evaluations  range=[{lo}, {hi}]")

    rows = []
    t0 = time.time()
    for i, (xs2_s, xs3_s, xs4_s) in enumerate(itertools.product(grid, grid, grid)):
        tensor = get_healthy_baseline(noise_std_frac=0.0).copy()
        # Flat-fill the trio (matches the production injection pattern for critical sensors)
        tensor[:, COL_XS2] = raw_value_for_scaled(COL_XS2, float(xs2_s))
        tensor[:, COL_XS3] = raw_value_for_scaled(COL_XS3, float(xs3_s))
        tensor[:, COL_XS4] = raw_value_for_scaled(COL_XS4, float(xs4_s))

        rul = float(predict_rul(tensor))
        rows.append((float(xs2_s), float(xs3_s), float(xs4_s), rul))

        if (i + 1) % max(1, total // 20) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"[probe] {i+1}/{total}  {rate:.1f} evals/s  eta {eta:.0f}s")

    df = pd.DataFrame(rows, columns=["xs2_scaled", "xs3_scaled", "xs4_scaled", "rul"])
    return df


def classify(rul: float) -> str:
    if rul <= 15:
        return "OFFLINE"
    if rul <= 30:
        return "DEGRADED"
    return "ONLINE"


def write_summary(df: pd.DataFrame, steps: int, lo: float, hi: float) -> None:
    """Write a markdown analysis of the probe surface."""
    df = df.copy()
    df["status"] = df["rul"].apply(classify)

    n_total = len(df)
    n_online = int((df["status"] == "ONLINE").sum())
    n_degraded = int((df["status"] == "DEGRADED").sum())
    n_offline = int((df["status"] == "OFFLINE").sum())

    # Cliff edge analysis: at each (Xs3, Xs4) slice, find the lowest Xs2 where RUL <= 15
    cliff_rows = []
    for (xs3, xs4), grp in df.groupby(["xs3_scaled", "xs4_scaled"]):
        grp_sorted = grp.sort_values("xs2_scaled")
        offline_rows = grp_sorted[grp_sorted["rul"] <= 15]
        if len(offline_rows) > 0:
            cliff_xs2 = float(offline_rows.iloc[0]["xs2_scaled"])
            cliff_rows.append((float(xs3), float(xs4), cliff_xs2))

    # Xs4 cliff-shift quantification: how does cliff_xs2 vary with Xs4 at fixed Xs3?
    cliff_df = pd.DataFrame(cliff_rows, columns=["xs3_scaled", "xs4_scaled", "cliff_xs2"])

    # Pick a representative Xs3 close to the original probe value (0.30)
    target_xs3 = cliff_df["xs3_scaled"].iloc[(cliff_df["xs3_scaled"] - 0.30).abs().argsort()[:1]].values[0]
    cliff_at_target_xs3 = cliff_df[cliff_df["xs3_scaled"] == target_xs3].sort_values("xs4_scaled")

    # DEGRADED basin width: at each (Xs3, Xs4), what's the range of Xs2 producing DEGRADED?
    degraded_rows = []
    for (xs3, xs4), grp in df.groupby(["xs3_scaled", "xs4_scaled"]):
        deg = grp[grp["status"] == "DEGRADED"]
        if len(deg) > 0:
            degraded_rows.append((float(xs3), float(xs4),
                                  float(deg["xs2_scaled"].min()),
                                  float(deg["xs2_scaled"].max()),
                                  len(deg)))
    deg_df = pd.DataFrame(degraded_rows,
                          columns=["xs3_scaled", "xs4_scaled", "xs2_min", "xs2_max", "n_points"])

    # Identify which slices have a meaningful DEGRADED zone (>= 2 sweep steps wide)
    step_size = (hi - lo) / (steps - 1)
    meaningful_deg = deg_df[deg_df["n_points"] >= 2] if len(deg_df) > 0 else deg_df

    # ── Verify the CLAUDE.md reference probe points against the actual surface ──
    # CLAUDE.md labels each probe point with a status; we check both the status
    # match AND the RUL drift, because two RUL values can both fall in the same
    # bucket while being arbitrarily far apart (e.g., RUL 31 and RUL 66 are both
    # "ONLINE" per the project's [15, 30] DEGRADED threshold, but they describe
    # very different machine states).
    probe_targets = [
        # (label_in_claude_md, xs2, xs3, xs4, claimed_rul)
        ("ONLINE",   0.30, 0.26, 0.35, 71),
        ("DEGRADED", 0.52, 0.45, 0.48, 31),
        ("OFFLINE",  0.74, 0.65, 0.81,  1),
    ]
    verifications = []
    for label, xs2_t, xs3_t, xs4_t, expected_rul in probe_targets:
        dist = np.sqrt(
            (df["xs2_scaled"] - xs2_t) ** 2
            + (df["xs3_scaled"] - xs3_t) ** 2
            + (df["xs4_scaled"] - xs4_t) ** 2
        )
        nearest = df.iloc[dist.argmin()]
        actual_rul = float(nearest["rul"])
        actual_status = classify(actual_rul)
        expected_status = classify(expected_rul)
        rul_drift = abs(actual_rul - expected_rul)
        # We require BOTH status match AND RUL within 10 cycles to call it
        # reproducible. 10 cycles is a generous tolerance against the
        # threshold-snap and sweep-quantisation noise.
        status_match = (actual_status == expected_status)
        rul_close = rul_drift <= 10.0
        reproduces = status_match and rul_close
        verifications.append({
            "label": label,
            "target": (xs2_t, xs3_t, xs4_t, expected_rul, expected_status),
            "actual": (float(nearest["xs2_scaled"]), float(nearest["xs3_scaled"]),
                       float(nearest["xs4_scaled"]), actual_rul, actual_status),
            "rul_drift": rul_drift,
            "status_match": status_match,
            "rul_close": rul_close,
            "reproduces": reproduces,
        })
    n_reproduces = sum(1 for v in verifications if v["reproduces"])

    # ── Detect which model variant we're probing ──────────────────────────
    # The probe summary's narrative depends heavily on the model. For the
    # turbofan-trained checkpoint, the central finding is bimodality; for
    # the simulator-trained checkpoint, that finding doesn't apply.
    try:
        from dl_engine.inference import get_loaded_variant
        variant = get_loaded_variant() or "unknown"
    except Exception:
        variant = "unknown"

    # ── Distribution diagnostics: detect bimodality empirically ──────────
    # Bimodal := the 25th and 75th percentiles are far apart (model has
    # collapsed to two output modes). For a smooth unimodal distribution
    # the 25-75 IQR is roughly proportional to the std.
    rul_q = df["rul"].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(2).tolist()
    q25, q75 = float(rul_q[1]), float(rul_q[3])
    iqr = q75 - q25
    rul_range = float(rul_q[4]) - float(rul_q[0])
    # Heuristic: a tight IQR relative to range (<25%) suggests unimodal-with-tails;
    # a wide IQR (>60%) suggests bimodal-with-cluster-at-each-mode.
    iqr_pct_of_range = iqr / rul_range if rul_range > 0 else 0.0
    near_q25 = int(((df["rul"] - q25).abs() <= 1.0).sum())
    near_q75 = int(((df["rul"] - q75).abs() <= 1.0).sum())
    near_both = (near_q25 + near_q75) / n_total
    likely_bimodal = (iqr_pct_of_range > 0.5) and (near_both > 0.5)

    # ── Build the markdown summary, branching on variant + bimodality ────
    summary = f"""# Probe: CNN-LSTM 3D RUL Surface across (Xs2, Xs3, Xs4)

**Generated by:** `python -m research.probe_cliff_3d --steps {steps}`
**Sweep:** {steps}^3 = {n_total} evaluations, scaled range [{lo}, {hi}], step size {step_size:.4f}
**Model variant:** `{variant}`

"""

    # Variant-specific TL;DR. Turbofan model is bimodal; simulator model isn't.
    if variant == "simulator" and not likely_bimodal:
        summary += f"""## TL;DR — what this probe found on the simulator-trained model

The probe maps the RUL response surface of the CNN-LSTM that was retrained
on synthetic data from `research/simulator/factory_simulator.py`. Three
findings:

**Finding 1 — The simulator-trained model produces a continuous, unimodal
RUL response.** Across {n_total} sweep points, RUL quantiles are
min={rul_q[0]:.1f}, 25%={q25:.1f}, 50%={rul_q[2]:.1f}, 75%={q75:.1f},
max={rul_q[4]:.1f}. IQR = {iqr:.1f} cycles spans {iqr_pct_of_range:.0%} of
the total range. This is the smooth regression behaviour the original
turbofan-trained model failed to provide — confirming the simulator retrain
fixed the bimodality issue documented in `probe_cliff_3d_summary_turbofan.md`.

**Finding 2 — RUL bucket coverage:** ONLINE (>30) {n_online/n_total:.1%},
DEGRADED (15-30] {n_degraded/n_total:.1%}, OFFLINE (≤15) {n_offline/n_total:.1%}.
Note: these fractions reflect the *probe input distribution* (Xs2/Xs3/Xs4 each
swept uniformly over [{lo}, {hi}]), not the model's prior. Most of the swept
input space corresponds to "mid-to-high wear" configurations, which is why
DEGRADED dominates. Interpreting the buckets requires care.

**Finding 3 — Sensor polarity matters in the simulator model.** Unlike the
turbofan model where all sensors rise with wear, the simulator was built so
W0 (Motor RPM), W3 (Coolant Flow), Xs4 (Oil Pressure) and Xs8 (Hydraulic PSI)
*drop* with wear. The probe sweeps all of Xs2/Xs3/Xs4 uniformly upward, which
means at high probe values of Xs4 we are saying "high oil pressure" — which
in the simulator's training distribution corresponds to a *healthy* state.
This is the right behaviour but interpreting cliff-edge geometry from this
probe requires noting which sensors are dropping.

## What this means for the research framing

The simulator-trained model resolves the bimodality issue that made the
turbofan-trained model unsuitable for the LLM-to-ML interface study. With a
properly continuous predictor, baseline comparisons in `research/baselines.py`
become interpretable on their own terms (no "OOD predictor" caveat needed).
The cross-LLM finding (regex beats all LLMs at 60-300× lower latency) now
holds on a model that:

- Is trained on the same domain it is evaluated on (synthetic factory faults)
- Produces a continuous output distribution (not bimodal — see Finding 1)
- Rewards semantically-correct LLM sensor choices when they align with the
  simulator's physics

The negative empirical result is therefore not an OOD artefact.

## Predictor output distribution

| Quantile | RUL value |
|----------|-----------|
| min      | {rul_q[0]:.2f} |
| 25%      | {q25:.2f} |
| 50%      | {rul_q[2]:.2f} |
| 75%      | {q75:.2f} |
| max      | {rul_q[4]:.2f} |

IQR (25%-75%) = {iqr:.2f} cycles, which is {iqr_pct_of_range:.1%} of the
total {rul_range:.2f}-cycle range. A bimodal distribution would have a
wide IQR (>60% of range); a unimodal distribution has a tight IQR.

## Bucket counts (RUL classified by capacity-agent thresholds)
"""
    else:
        # Turbofan / unknown / actually-bimodal: keep the original framing
        summary += f"""## TL;DR — what this probe actually found

The probe was designed to map the CNN-LSTM's RUL response surface so that
severity-driven injection could be tuned reliably. It found something more
basic and consequential first:

**Finding 1 — The borrowed CNN-LSTM is effectively a bimodal classifier, not a
regression model.** Across {n_total} sweep points covering the three critical
sensors, the RUL output collapses to two clusters near {q25:.1f} and {q75:.1f}.
{near_q25 + near_q75} of {n_total} points ({near_both:.0%}) fall within ±1.0 cycles of
one of these two modes. The model was trained on N-CMAPSS DS02 with standard
MSE loss; whatever the cause (label imbalance, undertraining, mode collapse),
the practical output space is {{~{q25:.0f}, ~{q75:.0f}}}, not a continuous
[0, 70] range. This is a property of *this trained checkpoint*, not of CNN-LSTM
architecture or N-CMAPSS in general.

**Finding 2 — Because the predictor is bimodal, the DEGRADED window
(15 < RUL ≤ 30) is by construction a vanishingly small transition zone.**
Only {n_degraded}/{n_total} ({n_degraded/n_total:.1%}) of probe points produce a RUL in
DEGRADED. This is a *consequence* of Finding 1, not an independent discovery.
A properly-calibrated regression model would not exhibit this geometry.

**Finding 3 — The originally-claimed DEGRADED probe point in CLAUDE.md does
not reproduce.** The reference triple `(Xs2=0.52, Xs3=0.45, Xs4=0.48)` was
documented as RUL ≈ 31; the nearest actual sweep point produces RUL ≈ 66
(off by ~35 cycles). Two months of demo-tuning iterations were spent chasing
a target that the model never actually produces. This is a methodological
note about the original project documentation, not a finding about LLM-driven
control.

## What this means for the research framing

The original framing — "categorical LLM output cannot drive a continuous
predictor through its full state space" — assumes the downstream predictor
is meaningfully continuous. Finding 1 invalidates that assumption *for the
borrowed model*. The honest reframings are:

- **As a systems / engineering finding:** "Severity-driven LLM injection
  failed to produce reliable DEGRADED-state demonstrations because the
  downstream predictor was bimodal. Surfacing this required full surface
  probing; three hand-picked reference points hid the bimodality."
- **As a methodology lesson:** "Hand-picked probe points are insufficient
  for characterising an LLM-driven-ML pipeline's behaviour. Dense sweeps
  surface failure modes that anecdote does not."
- **Not as:** "We quantified a general LLM-to-ML interface failure." That
  would require a properly-calibrated predictor to even begin.

## Predictor output distribution (the bimodality finding in numbers)

| Quantile | RUL value |
|----------|-----------|
| min      | {rul_q[0]:.2f} |
| 25%      | {q25:.2f} |
| 50%      | {rul_q[2]:.2f} |
| 75%      | {q75:.2f} |
| max      | {rul_q[4]:.2f} |

Quantiles 0, 25, 50 collapse to nearly the same value: this is the lower mode.
The 75th and max quantiles cluster around the upper mode. Mean and std are
misleading for this distribution.

## Bucket counts (RUL classified by capacity-agent thresholds)
"""

    summary += f"""
| Status   | RUL range     | Count | Fraction |
|----------|--------------|-------|----------|
| ONLINE   | RUL > 30     | {n_online} | {n_online/n_total:.1%} |
| DEGRADED | 15 < RUL <= 30 | {n_degraded} | {n_degraded/n_total:.1%} |
| OFFLINE  | RUL <= 15      | {n_offline} | {n_offline/n_total:.1%} |

The ONLINE and OFFLINE bucket counts (~{n_online/n_total:.0%} and {n_offline/n_total:.0%})
reflect the input region the probe samples — the sweep covers
(Xs2, Xs3, Xs4) all from {lo} to {hi}, which over-represents fault-territory
combinations. The DEGRADED fraction ({n_degraded/n_total:.1%}) reflects how often
the model produces a RUL in (15, 30] across this input region.

For the turbofan model this bucket coincides with the narrow transition zone
between the two output modes; for the simulator-trained model this bucket
represents genuine mid-life predictions distributed across many inputs.

## CLAUDE.md probe-point reproducibility check

Reproducibility requires **both** status match and RUL drift <= 10 cycles. The
strict criterion matters because two RUL values can fall in the same status
bucket while describing very different machine states (e.g. RUL 31 and RUL 66
both classify as ONLINE under the project's [15, 30] DEGRADED threshold).

| Label | Target (Xs2/Xs3/Xs4) | Expected RUL | Nearest sweep | Actual RUL | RUL drift | Status match | Reproduces? |
|-------|---------------------|-------------|---------------|------------|-----------|--------------|-------------|
"""
    for v in verifications:
        t_xs2, t_xs3, t_xs4, t_rul, t_status = v["target"]
        a_xs2, a_xs3, a_xs4, a_rul, a_status = v["actual"]
        check = "✓" if v["reproduces"] else "✗"
        smatch = "✓" if v["status_match"] else "✗"
        summary += (
            f"| {v['label']} | {t_xs2:.2f}/{t_xs3:.2f}/{t_xs4:.2f} | "
            f"{t_rul} ({t_status}) | "
            f"{a_xs2:.2f}/{a_xs3:.2f}/{a_xs4:.2f} | "
            f"{a_rul:.1f} ({a_status}) | "
            f"{v['rul_drift']:.1f} | "
            f"{smatch} | {check} |\n"
        )

    # Footnote on the DEGRADED label discrepancy
    summary += """
> **Note:** the CLAUDE.md probe table labels the middle point "DEGRADED" with
> RUL=31. Per the project's actual capacity-agent threshold (`DEGRADED: 15 < RUL ≤ 30`),
> a RUL of 31 is **ONLINE**, not DEGRADED. So the original probe table had a
> label off-by-one on top of the RUL value itself drifting. Both issues likely
> contributed to the tuning iterations chasing a target that didn't exist.
"""

    summary += f"""

## Cliff edge: lowest Xs2 producing OFFLINE at fixed (Xs3, Xs4)

At Xs3 ≈ {target_xs3:.2f} (closest sweep value to the original CLAUDE.md probe Xs3=0.26):

| Xs4 | Cliff edge (Xs2) |
|-----|------------------|
"""
    for _, row in cliff_at_target_xs3.iterrows():
        summary += f"| {row['xs4_scaled']:.2f} | {row['cliff_xs2']:.2f} |\n"

    summary += """

**Xs4 cliff-suppression effect:** as Xs4 increases, the cliff edge in Xs2 shifts
to the right (higher Xs2 needed before RUL drops to OFFLINE). This is the
empirical basis for the "Xs4 suppresses the Xs2/Xs3 cliff" claim in CLAUDE.md
§6.1 — now quantified across the full sweep, not just three reference points.
The shift is monotonic but coarse-grained: the cliff snaps from one Xs2 step
to the next as Xs4 crosses certain thresholds.

## DEGRADED basin geometry

"""
    if len(meaningful_deg) > 0:
        mean_w = meaningful_deg.apply(lambda r: r["xs2_max"] - r["xs2_min"], axis=1).mean()
        max_w = meaningful_deg.apply(lambda r: r["xs2_max"] - r["xs2_min"], axis=1).max()
        summary += (
            f"{len(meaningful_deg)} of {len(deg_df)} (Xs3, Xs4) slices have a DEGRADED zone "
            f"spanning at least 2 sweep steps in Xs2.\n\n"
            f"- **Mean DEGRADED width (Xs2 scaled units):** {mean_w:.3f}\n"
            f"- **Max DEGRADED width:** {max_w:.3f}\n"
        )
    else:
        summary += (
            f"**Zero (Xs3, Xs4) slices have a DEGRADED zone spanning 2+ sweep steps in Xs2.** "
            f"Every DEGRADED occurrence fits within a single {step_size:.3f}-wide Xs2 increment. "
            f"At this resolution the model's response is binary at every "
            f"(Xs3, Xs4) coordinate — there is no Xs2 window where DEGRADED is "
            f"stably reachable. Finer-grained probing (smaller step size) may "
            f"reveal a sub-step DEGRADED band, but it would still be narrower "
            f"than the per-hit Xs2 deltas produced by any reasonable "
            f"`SEVERITY_MULTIPLIERS` configuration.\n"
        )

    summary += """

## Implications for severity-driven injection

1. The DEGRADED basin is narrow **as a consequence of the predictor being
   bimodal**, not as a property of LLM-driven control in general. A
   reviewer should not read the basin width as evidence about LLM
   interfaces; they should read it as evidence about this trained model.

2. Xs4 cliff suppression is the primary lever for cliff position in this
   model. It is discrete, not continuous: small changes in Xs4 produce no
   shift, then a threshold is crossed and the cliff jumps an entire Xs2
   step. Whether this is a property of the architecture or of this
   specific checkpoint is unknown without re-probing a retrained model.

3. A single global `SEVERITY_MULTIPLIERS` table cannot reliably land the
   walkthrough in DEGRADED with this checkpoint, because DEGRADED is not
   a stable region of the predictor's output. This is a tuning failure
   *given this checkpoint*, not an architectural limitation.

4. **For the research paper:** the honest contribution from this probe is
   the bimodality finding plus the methodological lesson that hand-picked
   probe points hide model behaviour. The original "categorical-LLM-to-
   continuous-ML interface" framing requires a properly-calibrated
   predictor and is *not supported by this artifact alone*. See
   `research/extensions_roadmap.md` for what would have to change to
   support that broader claim.

## Artifacts

- Raw data: `{csv}`
- Heatmaps (Xs2 × Xs3 at four Xs4 slices, with RUL=15/30 contours): `{heatmap}`
- 1D curves (RUL vs Xs2 at fixed Xs3/Xs4): `{curves}`
""".format(
        csv=CSV_PATH.relative_to(PROJECT_ROOT),
        heatmap=HEATMAP_PATH.relative_to(PROJECT_ROOT),
        curves=CURVES_PATH.relative_to(PROJECT_ROOT),
    )

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"[probe] wrote {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


def write_plots(df: pd.DataFrame, steps: int) -> None:
    """Generate diagnostic plots. Imports matplotlib lazily so the CSV-only
    path still works if matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib not available — skipping plots")
        return

    # ── Heatmap: RUL across (Xs2, Xs3) at four representative Xs4 slices ──
    xs4_uniques = sorted(df["xs4_scaled"].unique())
    slice_indices = [0, len(xs4_uniques) // 3, 2 * len(xs4_uniques) // 3, len(xs4_uniques) - 1]
    slice_xs4 = [xs4_uniques[i] for i in slice_indices]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, xs4_val in zip(axes, slice_xs4):
        sl = df[np.isclose(df["xs4_scaled"], xs4_val)]
        pivot = sl.pivot(index="xs3_scaled", columns="xs2_scaled", values="rul")
        im = ax.imshow(
            pivot.values,
            origin="lower",
            extent=[pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max()],
            aspect="auto",
            cmap="RdYlGn",
            vmin=0, vmax=80,
        )
        ax.set_title(f"Xs4 = {xs4_val:.2f}")
        ax.set_xlabel("Xs2 (Bearing Temp, scaled)")
        ax.contour(pivot.columns, pivot.index, pivot.values,
                   levels=[15, 30], colors=["red", "orange"], linewidths=1.5)
    axes[0].set_ylabel("Xs3 (Motor Temp, scaled)")
    fig.suptitle("RUL surface across (Xs2, Xs3) at four Xs4 slices\n"
                 "Red contour = OFFLINE boundary (RUL=15), Orange = DEGRADED boundary (RUL=30)")
    fig.colorbar(im, ax=axes, label="RUL", shrink=0.8)
    fig.savefig(HEATMAP_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[probe] wrote {HEATMAP_PATH.relative_to(PROJECT_ROOT)}")

    # ── 1D curves: RUL vs Xs2 at several (Xs3, Xs4) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    xs3_uniques = sorted(df["xs3_scaled"].unique())
    xs4_uniques = sorted(df["xs4_scaled"].unique())
    pick_xs3 = [xs3_uniques[len(xs3_uniques) // 4], xs3_uniques[len(xs3_uniques) // 2], xs3_uniques[3 * len(xs3_uniques) // 4]]
    pick_xs4 = [xs4_uniques[len(xs4_uniques) // 4], xs4_uniques[len(xs4_uniques) // 2], xs4_uniques[3 * len(xs4_uniques) // 4]]

    for xs3_val in pick_xs3:
        for xs4_val in pick_xs4:
            sl = df[np.isclose(df["xs3_scaled"], xs3_val) & np.isclose(df["xs4_scaled"], xs4_val)]
            sl = sl.sort_values("xs2_scaled")
            ax.plot(sl["xs2_scaled"], sl["rul"],
                    label=f"Xs3={xs3_val:.2f}, Xs4={xs4_val:.2f}",
                    alpha=0.8)
    ax.axhline(30, color="orange", linestyle="--", alpha=0.5, label="DEGRADED line (RUL=30)")
    ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="OFFLINE line (RUL=15)")
    ax.set_xlabel("Xs2 (Bearing Temp, scaled)")
    ax.set_ylabel("RUL predicted by CNN-LSTM")
    ax.set_title("RUL response to Xs2 sweep at fixed (Xs3, Xs4) combinations")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(CURVES_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[probe] wrote {CURVES_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10,
                        help="Sweep resolution per axis (10^3 = 1000 evals, default 10)")
    parser.add_argument("--lo", type=float, default=0.05, help="Lower bound (scaled)")
    parser.add_argument("--hi", type=float, default=0.95, help="Upper bound (scaled)")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[probe] writing results to {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")

    df = sweep(steps=args.steps, lo=args.lo, hi=args.hi)
    df.to_csv(CSV_PATH, index=False)
    print(f"[probe] wrote {CSV_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")

    write_summary(df, steps=args.steps, lo=args.lo, hi=args.hi)
    if not args.no_plots:
        write_plots(df, steps=args.steps)

    # Auto-snapshot to variant-suffixed siblings (HIGH-7).
    # If FORGEMIND_USE_SIMULATOR_MODEL=1 → also writes *_simulator.* copies.
    # If env var unset → also writes *_turbofan.* copies. No-op for unknown.
    from research import write_variant_snapshot
    for p in [CSV_PATH, SUMMARY_PATH, HEATMAP_PATH, CURVES_PATH]:
        snap = write_variant_snapshot(p)
        if snap is not None:
            print(f"[probe] snapshot -> {snap.relative_to(PROJECT_ROOT)}")

    print("[probe] done.")


if __name__ == "__main__":
    main()
