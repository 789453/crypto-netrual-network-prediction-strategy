"""Copy compact, auditable research results; leave raw data and weights local."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    """Replace undefined diagnostic correlations with JSON null in public copies."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--report", type=Path, default=Path("reports/2026-10-03"))
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    runs = {}
    for run in sorted((args.outputs / "models").iterdir()):
        if not (run / "summary.json").is_file():
            continue
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        summary["best_checkpoint_sha256"] = _sha256(run / "best.pt")
        summary["history"] = [json.loads(line) for line in
                              (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        runs[run.name] = summary
    (args.report / "training_runs.json").write_text(json.dumps(runs, indent=2) + "\n",
                                                      encoding="utf-8")
    paths = (
        (args.outputs / "cache_v1" / "manifest.json", "cache_v1_manifest.json"),
        (args.outputs / "cache_v2" / "manifest.json", "cache_v2_manifest.json"),
        (args.outputs / "baselines" / "summary.json", "baselines_summary.json"),
        (args.outputs / "evaluation" / "selection.json", "selection.json"),
        (args.outputs / "evaluation" / "final_test.json", "final_test.json"),
        (args.outputs / "evaluation" / "post_test_diagnostics.json", "post_test_diagnostics.json"),
    )
    sources = {}
    for source, name in paths:
        sources[name] = _sha256(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        (args.report / name).write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (args.report / "source_hashes.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8"
    )
    print(f"published {len(runs)} neural runs and compact reports to {args.report}")


if __name__ == "__main__":
    main()
