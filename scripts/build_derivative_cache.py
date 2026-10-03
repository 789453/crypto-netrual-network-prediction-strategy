from __future__ import annotations

import argparse
from pathlib import Path

from crypto_timing.derivative_cache import build_derivative_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("outputs/cache_v1"))
    parser.add_argument("--source", type=Path, default=Path("D:/Trading/practical_crypto_strategy/data/parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cache_v2"))
    args = parser.parse_args()
    manifest = build_derivative_cache(args.base, args.source, args.output)
    print(f"complete: market channels={len(manifest['market_names'])}", flush=True)


if __name__ == "__main__":
    main()
