from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.baselines import fit_baselines, predict_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args()
    if args.split:
        predict_baseline(args.cache, args.output, args.split)
    else:
        fit_baselines(args.cache, args.output)


if __name__ == "__main__":
    main()
