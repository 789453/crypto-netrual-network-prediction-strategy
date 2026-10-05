from pathlib import Path
from crypto_timing.mechanism_data import build_mechanism_cache

if __name__ == "__main__":
    build_mechanism_cache(Path("D:/Trading/practical_crypto_strategy/data/parquet"),
                          Path("outputs/mechanism/cache_v4"))
