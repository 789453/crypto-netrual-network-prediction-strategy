"""Run the prespecified low-capacity probability calibration diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from crypto_timing.redesign_calibration import calibration_diagnostic


if __name__ == "__main__":
    root = Path("outputs/redesign")
    selection_path = root / "evaluation/selection.json"
    leader = json.loads(selection_path.read_text(encoding="utf-8"))["direction_leader"]
    result = calibration_diagnostic(
        root / "cache_v3", root / "baselines", root / "models/magnitude_seed20261004",
        root / "models" / leader, selection_path,
        root / "evaluation/calibration_diagnostic.json")
    print({"direction_shrinkage": result["direction_shrinkage"],
           "magnitude_shrinkage": result["magnitude_shrinkage"]}, flush=True)
