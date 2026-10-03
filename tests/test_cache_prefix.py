from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_timing.cache import build_cache, load_cache


SYMBOLS = (
    "ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT",
)


def _write_source(root, bars: int) -> None:
    timestamp = pd.date_range("2023-01-01", periods=bars, freq="5min", tz="UTC")
    open_ms = timestamp.to_numpy(dtype="datetime64[ms]").astype("int64")
    number = np.arange(bars)
    for si, symbol in enumerate(SYMBOLS):
        op = 100 + si * 10 + number * .001
        cl = op * (1 + .0002 * np.sin(number / 17 + si))
        hi = np.maximum(op, cl) * 1.001
        lo = np.minimum(op, cl) * .999
        volume = np.full(bars, 1000.0)
        quote = volume * (op + cl) / 2
        frame = pd.DataFrame({
            "date": timestamp,
            "open_time": open_ms,
            "close_time": open_ms + 299_999,
            "open": op, "high": hi, "low": lo, "close": cl,
            "volume": volume, "quote_volume": quote, "trade_count": np.full(bars, 100),
            "taker_buy_quote_volume": quote * (0.5 + .1 * np.sin(number / 31)),
        })
        target = root / symbol
        target.mkdir(parents=True)
        frame.to_parquet(target / "5m.parquet", index=False)


def test_feature_prefix_and_label_horizon_are_invariant_to_future_bars(tmp_path) -> None:
    full_source = tmp_path / "full_source"
    cut_source = tmp_path / "cut_source"
    _write_source(full_source, 2508)
    _write_source(cut_source, 2400)
    full_root = tmp_path / "full_cache"
    cut_root = tmp_path / "cut_cache"
    build_cache(full_source, full_root)
    build_cache(cut_source, cut_root)
    full = load_cache(full_root)
    cut = load_cache(cut_root)
    for name in ("fast", "slow", "stats", "market"):
        np.testing.assert_allclose(full[name][: len(cut[name])], cut[name],
                                   equal_nan=True, rtol=1e-6, atol=1e-6)
    # Last 4h of the shorter file has no mature 4h label.
    np.testing.assert_allclose(full["targets"][: len(cut["targets"]) - 5], cut["targets"][: -5],
                               equal_nan=True, rtol=1e-6, atol=1e-6)
