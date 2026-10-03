from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.cache import build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("D:/Trading/practical_crypto_strategy/data/parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cache_v1"))
    args = parser.parse_args()
    manifest = build_cache(args.source, args.output)
    print(f"complete: {manifest['hour_count']} hours, {len(manifest['symbols'])} symbols", flush=True)


if __name__ == "__main__":
    main()
