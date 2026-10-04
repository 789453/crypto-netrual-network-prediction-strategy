"""Fit simple train-only probability and conditional-mean references."""

from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.redesign_baselines import fit_redesign_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/redesign/cache_v3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/redesign/baselines"))
    args = parser.parse_args()
    fit_redesign_baselines(args.cache, args.output)


if __name__ == "__main__":
    main()
