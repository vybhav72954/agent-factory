"""ForgeMind research artifacts — probes, baselines, ablations, evaluation.

These modules are deliberately separated from the runtime pipeline (agents/,
dl_engine/, terminal/) so that experimental code never accidentally affects
demo behaviour. Run as scripts from the project root:

    python -m research.probe_cliff_3d
    python -m research.baselines
    python -m research.ablation
    python -m research.evaluation_rubric
    python -m research.domain2.hvac_demo
"""

from __future__ import annotations

import shutil
from pathlib import Path


def write_variant_snapshot(default_path: Path) -> Path | None:
    """
    After a research script writes its output to `default_path`, copy it to a
    variant-suffixed sibling (e.g., `foo.csv` -> `foo_simulator.csv`) so
    re-running with a different model loaded preserves the previous variant's
    snapshot. Returns the variant path, or None if no variant is loaded or
    the source path doesn't exist.

    Convention: default_path always holds the most recent run (current
    behaviour). The variant-suffixed copy is the per-model archival snapshot.

    Used by probe_cliff_3d.py, baselines.py, ablation.py.
    """
    if not default_path.exists():
        return None
    try:
        from dl_engine.inference import get_loaded_variant
        variant = get_loaded_variant()
    except Exception:
        variant = None
    if variant is None or variant == "unknown":
        return None
    # Suffix the variant before the extension(s): foo.csv -> foo_simulator.csv
    # Handle compound suffixes like .tar.gz only via the final suffix.
    stem = default_path.stem
    suffix = default_path.suffix
    suffixed = default_path.with_name(f"{stem}_{variant}{suffix}")
    shutil.copy(str(default_path), str(suffixed))
    return suffixed
