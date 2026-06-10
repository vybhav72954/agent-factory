"""
Bayesian anomaly classifier with LLM-emitted prior updates.

Self-contained second-domain demo. Differs from the main pipeline along all four
axes flagged in the HVAC retraction:

  axis              main pipeline                this domain
  -----             -------------                -----------
  output            categorical status bucket    posterior over 5 failure modes
  metric            status-match rate            Brier score + NLL + top-1
  LLM role          sensor + severity choice     shift over class priors
  predictor         CNN-LSTM (cliff geometry)    Gaussian Naive Bayes (probabilistic)

The classifier is trained once on synthetic per-mode multivariate Gaussian
observations and frozen. The LLM only affects the prior, not the likelihood. This
isolates the LLM's marginal value cleanly: any improvement over the uniform-prior
baseline is attributable to the prior shift alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sklearn.naive_bayes import GaussianNB


FAILURE_MODES: List[str] = ["bearing", "motor", "oil_seal", "coolant", "electrical"]
N_MODES = len(FAILURE_MODES)
N_FEATURES = 6
UNIFORM_PRIOR: Dict[str, float] = {m: 1.0 / N_MODES for m in FAILURE_MODES}


_MODE_SIGNATURES: Dict[str, Tuple[np.ndarray, float]] = {
    "bearing":    (np.array([0.55, 0.40, 0.35, 0.45, 0.40, 0.40]), 0.32),
    "motor":      (np.array([0.40, 0.55, 0.40, 0.45, 0.50, 0.45]), 0.32),
    "oil_seal":   (np.array([0.40, 0.40, 0.55, 0.45, 0.40, 0.40]), 0.32),
    "coolant":    (np.array([0.40, 0.40, 0.45, 0.55, 0.40, 0.40]), 0.32),
    "electrical": (np.array([0.40, 0.45, 0.35, 0.40, 0.55, 0.55]), 0.32),
}

OBS_WINDOW_LEN = 3


class PriorUpdate(BaseModel):
    """LLM-emitted shift over failure-mode priors."""

    failure_mode: str = Field(..., description="One of FAILURE_MODES.")
    prior_weight_delta: float = Field(..., ge=-0.5, le=0.5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=1, max_length=300)

    @field_validator("failure_mode")
    @classmethod
    def _mode_in_set(cls, v: str) -> str:
        if v not in FAILURE_MODES:
            raise ValueError(f"failure_mode must be one of {FAILURE_MODES}, got {v!r}")
        return v


def synthesize_observations(true_mode: str, n_samples: int = 50, seed: int | None = None) -> np.ndarray:
    """Sample n_samples (rows) x N_FEATURES observation matrix from the true mode's
    multivariate Gaussian. Used for both training and inference time."""
    if true_mode not in _MODE_SIGNATURES:
        raise ValueError(f"unknown true_mode {true_mode!r}")
    mean, std = _MODE_SIGNATURES[true_mode]
    rng = np.random.default_rng(seed)
    obs = rng.normal(loc=mean, scale=std, size=(n_samples, N_FEATURES))
    return np.clip(obs, 0.0, 1.0)


def build_training_set(samples_per_mode: int = 400, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a balanced labeled training set across all five modes."""
    rng = np.random.default_rng(seed)
    X_blocks, y_blocks = [], []
    for idx, mode in enumerate(FAILURE_MODES):
        mode_seed = int(rng.integers(0, 2**31 - 1))
        X_blocks.append(synthesize_observations(mode, n_samples=samples_per_mode, seed=mode_seed))
        y_blocks.append(np.full(samples_per_mode, idx, dtype=int))
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def train_classifier(seed: int = 0) -> GaussianNB:
    """Train and freeze a Gaussian NB on synthetic data. Called once per session."""
    X, y = build_training_set(seed=seed)
    clf = GaussianNB()
    clf.fit(X, y)
    return clf


def likelihood_per_mode(clf: GaussianNB, observations: np.ndarray) -> np.ndarray:
    """Compute per-mode log-likelihood of an observation window, summed over rows.

    Returns shape (N_MODES,) of log-likelihoods, normalised to a probability vector
    via the log-sum-exp trick so downstream caller can multiply by a prior directly.
    """
    logp_per_row = clf.predict_log_proba(observations) - np.log(clf.class_prior_ + 1e-12)
    summed = logp_per_row.sum(axis=0)
    summed = summed - summed.max()
    lik = np.exp(summed)
    return lik / lik.sum()


def apply_prior_update(prior: Dict[str, float], update: PriorUpdate) -> Dict[str, float]:
    """Apply a PriorUpdate to a prior dict, returning a normalised new prior.

    Implementation: add the delta to the target mode's mass, redistribute the
    inverse delta evenly across the other modes (or clip if necessary to keep all
    masses in [0, 1]), then renormalise.
    """
    new = {m: float(prior[m]) for m in FAILURE_MODES}
    target = update.failure_mode
    delta = float(update.prior_weight_delta)

    new[target] = max(0.0, new[target] + delta)
    if delta > 0.0:
        other_total = sum(new[m] for m in FAILURE_MODES if m != target)
        if other_total > 0.0:
            scale = max(0.0, (1.0 - new[target])) / other_total
            for m in FAILURE_MODES:
                if m != target:
                    new[m] = new[m] * scale
    elif delta < 0.0:
        spread = (-delta) / (N_MODES - 1)
        for m in FAILURE_MODES:
            if m != target:
                new[m] = new[m] + spread

    total = sum(new.values())
    if total <= 0.0:
        return dict(UNIFORM_PRIOR)
    return {m: new[m] / total for m in FAILURE_MODES}


def posterior(prior: Dict[str, float], likelihood: np.ndarray) -> Dict[str, float]:
    """Combine prior dict + likelihood vector into a posterior dict."""
    prior_vec = np.array([prior[m] for m in FAILURE_MODES])
    unnorm = prior_vec * likelihood
    total = unnorm.sum()
    if total <= 0.0:
        return dict(UNIFORM_PRIOR)
    return {m: float(unnorm[i] / total) for i, m in enumerate(FAILURE_MODES)}


@dataclass
class PipelineResult:
    true_mode: str
    predicted_mode: str
    posterior: Dict[str, float]
    prior_used: Dict[str, float]
    brier: float
    nll: float
    top1_correct: bool
    strategy_label: str
    prior_update: PriorUpdate | None
    latency_ms: float


def score(posterior_dict: Dict[str, float], true_mode: str) -> Tuple[float, float, bool]:
    """Brier score, NLL, top-1 correctness."""
    truth = np.array([1.0 if m == true_mode else 0.0 for m in FAILURE_MODES])
    probs = np.array([posterior_dict[m] for m in FAILURE_MODES])
    brier = float(np.sum((probs - truth) ** 2))
    p_true = max(float(posterior_dict[true_mode]), 1e-12)
    nll = float(-np.log(p_true))
    top1 = max(posterior_dict, key=lambda m: posterior_dict[m]) == true_mode
    return brier, nll, top1
