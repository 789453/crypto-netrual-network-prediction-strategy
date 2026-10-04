"""Train one isolated direction or magnitude model with resumable checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.redesign_training import RedesignTrainConfig, train_redesign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/redesign/cache_v3"))
    parser.add_argument("--baselines", type=Path, default=Path("outputs/redesign/baselines"))
    parser.add_argument("--kind", choices=("direction", "magnitude"), required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()
    output = args.output or Path("outputs/redesign/models") / f"{args.kind}_seed{args.seed}"
    train_redesign(args.cache, output,
                   RedesignTrainConfig(kind=args.kind, seed=args.seed, max_epochs=args.max_epochs),
                   args.baselines)


if __name__ == "__main__":
    main()
