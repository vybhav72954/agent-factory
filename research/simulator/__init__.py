"""
research/simulator/ — physics-informed factory simulator.

Replaces the borrowed N-CMAPSS turbofan training data with a domain-coherent
simulator whose 18 sensors actually correspond to the factory-machine concepts
they are named after (Bearing Temp, Motor RPM, Oil Pressure, etc.).

Purpose: give the CNN-LSTM a training distribution where LLM-driven fault
descriptions (e.g. "bearing temperature spike") map to physically meaningful
interventions, so the downstream model can respond sensibly. Without this,
the LLM-driven agentic pipeline is operating on a model that has never seen
anything resembling a factory-machine fault.

Modules:
    factory_simulator.py  — the simulator itself
    generate_training_data.py  — produces training CSVs / NPZ for the model

Run as scripts:
    python -m research.simulator.generate_training_data --n-lifecycles 1000
"""
