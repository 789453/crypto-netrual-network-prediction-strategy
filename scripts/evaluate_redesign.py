"""Select v3 models on 2025 validation, then assess seen 2026 history."""

from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.redesign_evaluation import historical_diagnostic, select_redesign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/redesign/cache_v3"))
    parser.add_argument("--baselines", type=Path, default=Path("outputs/redesign/baselines"))
    parser.add_argument("--magnitude", type=Path,
                        default=Path("outputs/redesign/models/magnitude_seed20261004"))
    parser.add_argument("--models", type=Path, default=Path("outputs/redesign/models"))
    parser.add_argument("--source", type=Path,
                        default=Path("D:/Trading/practical_crypto_strategy/data/parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/redesign/evaluation"))
    parser.add_argument("--phase", choices=("select", "historical", "all"), default="all")
    args = parser.parse_args()
    roots = sorted(path for path in args.models.glob("direction_seed*")
                   if (path / "summary.json").exists())
    if not roots:
        raise RuntimeError("no completed direction run")
    if args.phase in ("select", "all"):
        select_redesign(args.cache, args.baselines, args.magnitude, roots,
                        args.source, args.out)
    if args.phase in ("historical", "all"):
        historical_diagnostic(args.cache, args.baselines, args.magnitude, roots,
                              args.source, args.out)


if __name__ == "__main__":
    main()
