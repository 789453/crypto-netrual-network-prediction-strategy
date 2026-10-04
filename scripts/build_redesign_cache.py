"""Build isolated v3 features and executable-return labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.redesign_cache import build_redesign_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"D:\Trading\practical_crypto_strategy\data\parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/redesign/cache_v3"))
    args = parser.parse_args()
    manifest = build_redesign_cache(args.source, args.output)
    print(f"v3 cache ready: {manifest['hour_count']} hours × {len(manifest['symbols'])} symbols")


if __name__ == "__main__":
    main()
