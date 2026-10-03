from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("asym4h", "logratio4h", "asym1h", "return4h", "joint4h"), default="joint4h")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--train-stride-hours", type=int, default=2)
    parser.add_argument("--fast-blocks", type=int, default=3)
    parser.add_argument("--no-slow", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--no-market", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(mode=args.mode, seed=args.seed, batch_size=args.batch_size,
                         max_epochs=args.max_epochs, patience=args.patience,
                         train_stride_hours=args.train_stride_hours,
                         fast_blocks=args.fast_blocks,
                         use_slow=not args.no_slow, use_stats=not args.no_stats,
                         use_market=not args.no_market)
    train(args.cache, args.output, config)


if __name__ == "__main__":
    main()
