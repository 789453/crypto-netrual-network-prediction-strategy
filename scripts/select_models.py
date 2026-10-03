from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.evaluation import select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache_v1"))
    parser.add_argument("--runs", type=Path, default=Path("outputs/models"))
    parser.add_argument("--baselines", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--source", type=Path, default=Path("D:/Trading/practical_crypto_strategy/data/parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    select(args.cache, args.runs, args.baselines, args.source, args.output)


if __name__ == "__main__":
    main()
