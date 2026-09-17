"""PRONOSTIA (FEMTO-ST, IEEE PHM 2012) real-data validation.

Two parts, both answering the reviewer point that ForgeMind has no real
equipment data:

1. Simulator calibration (`calibrate.py`): compare the factory simulator's
   bearing degradation curves with real run-to-failure bearings, fit a
   calibrated bearing law, and report fidelity before and after.
2. Real-data predictor (`train_model.py`, `probe.py`, `replay.py`): train a
   third CNN-LSTM on PRONOSTIA features mapped into the 18-channel window,
   characterise its response surface BEFORE replaying strategies (so the
   geometry-based prediction of which strategy should win is made first),
   then replay every recorded strategy output against it.

Data: `research/pronostia/data/raw/` is a clone of
https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset (gitignored).
Cite: Nectoux et al., "PRONOSTIA: An experimental platform for bearings
accelerated degradation tests", IEEE PHM 2012.

Workflow (run from the project root):

    python -m research.pronostia.features      # per-snapshot features for all 17 bearings
    python -m research.pronostia.calibrate     # simulator vs PRONOSTIA fidelity + calibrated law
    python -m research.pronostia.train_model   # third checkpoint (GPU if available)
    python -m research.pronostia.probe         # response-surface geometry
    python -m research.pronostia.replay        # recorded strategy outputs on the PRONOSTIA model
"""
