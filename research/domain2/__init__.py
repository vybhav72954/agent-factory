"""
research/domain2/ — second-domain generalization demo.

Demonstrates that the ForgeMind agentic-pipeline architecture is not tied to
predictive maintenance or N-CMAPSS data. We re-instantiate the pattern on
a building HVAC monitoring scenario:

  Input Guard → Diagnostic Agent → Physics Simulator → Capacity Math → Floor Manager → Output

The substitutions:
  - DL Oracle (CNN-LSTM)            → simple physics simulator (no ML)
  - Sensor tensor (50,18)           → zone state vector (n_zones, n_signals)
  - RUL prediction                  → comfort-score prediction
  - Capacity report                 → zone-occupancy / comfort report
  - Machines (5)                    → HVAC zones (5)

The same Gemini structured-output approach drives the diagnostic agent. The
same kind of categorical-to-quantitative interface problem applies — Gemini
emits severity LOW/MEDIUM/HIGH and a target zone, the simulator produces a
continuous comfort score, the cliff between "comfortable" and "uncomfortable"
shows the same sharp-cliff geometry.
"""
