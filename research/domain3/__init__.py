"""
research/domain3/ — Bayesian anomaly classification (genuinely different second domain).

Re-instantiates the ForgeMind agentic pipeline structure on a qualitatively
different control problem:

    Operator text -> Input Guard -> Prior-Update Agent (LLM) -> Bayesian Classifier
    -> Posterior Vector over failure modes -> Operator Recommendation

Differs from the main PdM domain in four respects (the criteria HVAC failed):

  1. Output type: continuous probability vector over 5 failure modes,
     not a categorical ONLINE/DEGRADED/OFFLINE bucket.
  2. Evaluation metric: Brier score + negative log-likelihood + top-1 accuracy,
     not status-match rate.
  3. LLM's role: emits a prior-shift on failure modes (a probability-mass
     redistribution), not a sensor + severity choice.
  4. Predictor type: Gaussian Naive Bayes (probabilistic) trained once on
     synthetic data, not a CNN-LSTM with cliff geometry.

The headline question this domain answers for the paper: "Does the regex
baseline still beat the LLM when the output type and evaluation metric are
qualitatively different from the main pipeline?"

Run: `python -m research.domain3.bayesian_anomaly`
"""
