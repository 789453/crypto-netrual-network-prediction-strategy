"""Build an hourly decision store from completed five-minute futures bars.

The store retains NaNs until train-only scaling. ``date`` in the source denotes
bar *open*; decision timestamps denote the close of the preceding 5m bar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = (
    "date", "open_time", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "trade_count", "taker_buy_quote_volume",
)
FAST_NAMES = (
    "r5", "body", "range", "upper_wick", "lower_wick", "close_location",
    "log_quote", "log_trades", "taker_imbalance", "signed_quote_log",
    "vwap_gap", "r1h", "r4h", "rv1h", "rv4h", "asym1h", "asym4h",
    "quote_surprise", "trade_surprise", "flow1h", "flow4h", "illiquidity",
    "btc_r5", "btc_r1h", "eth_r5", "breadth", "dispersion",
    "hour_sin", "hour_cos", "week_sin", "week_cos",
)
SLOW_NAMES = (
    "r1h", "r4h", "rv1h", "rv4h", "asym1h", "asym4h",
    "quote_surprise", "flow1h", "flow4h", "illiquidity",
    "r24h", "rv24h", "asym24h", "r7d", "rv7d", "flow24h",
    "btc_r1h", "btc_r24h", "hour_sin", "week_sin",
)
STAT_CHANNELS = (0, 1, 2, 6, 7, 8, 10, 11, 13, 15, 17, 19, 21, 22, 25, 26)
MARKET_NAMES = (
    "btc_r1h", "btc_r4h", "btc_r24h", "btc_rv24h", "eth_r1h",
    "breadth1h", "dispersion1h", "own_beta_btc", "own_residual1h",
    "own_corr_btc",
)
TARGET_NAMES = (
    "asym4h", "logratio4h", "relative_var4h", "risk_return4h",
    "raw_return4h", "execution_return4h", "asym1h", "past_sigma5m",
)
HORIZON_BARS = 48
WARMUP_BARS = 2016


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum with full-window requirement; no future observations."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    clean = np.where(finite, x, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(clean)))
    count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    result = np.full(len(x), np.nan, dtype=np.float64)
    result[window - 1 :] = cumulative[window:] - cumulative[:-window]
    result[window - 1 :][count[window:] - count[:-window] != window] = np.nan
    return result


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_sum(values, window) / window


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=np.float64),
        where=np.isfinite(denominator) & (denominator != 0),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_bars(frame: pd.DataFrame, symbol: str, expected_dates: np.ndarray | None) -> np.ndarray:
    date = frame["date"].to_numpy(dtype="datetime64[ns]")
    if len(date) == 0 or np.any(np.diff(date) != np.timedelta64(5, "m")):
        raise ValueError(f"{symbol}: 5m time grid has a gap, duplicate or nonmonotonic key")
    if expected_dates is not None and not np.array_equal(date, expected_dates):
        raise ValueError(f"{symbol}: time grid differs from first symbol")
    open_time = frame["open_time"].to_numpy(np.int64)
    close_time = frame["close_time"].to_numpy(np.int64)
    if np.any(open_time != date.astype("datetime64[ms]").astype(np.int64)):
        raise ValueError(f"{symbol}: date is not the millisecond bar-open time")
    if np.any(close_time != open_time + 300_000 - 1):
        raise ValueError(f"{symbol}: close_time does not identify a complete 5m bar")
    op, hi, lo, cl, vol, q, trades, buy = (
        frame[name].to_numpy(np.float64) for name in (
            "open", "high", "low", "close", "volume", "quote_volume",
            "trade_count", "taker_buy_quote_volume",
        )
    )
    valid = (
        np.isfinite(op) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl)
        & (op > 0) & (cl > 0) & (lo > 0)
        & (hi >= np.maximum(op, cl)) & (lo <= np.minimum(op, cl))
        & np.isfinite(vol) & (vol >= 0) & np.isfinite(q) & (q >= 0)
        & np.isfinite(trades) & (trades >= 0) & np.isfinite(buy)
        & (buy >= -1e-5) & (buy <= q + 1e-3)
    )
    if not valid.all():
        raise ValueError(f"{symbol}: {int((~valid).sum())} invalid OHLC/turnover rows")
    return date


def _segment_sum(cumulative: np.ndarray, first: np.ndarray, last_exclusive: np.ndarray) -> np.ndarray:
    return cumulative[last_exclusive] - cumulative[first]


def _targets_for_symbol(
    op: np.ndarray, cl: np.ndarray, r: np.ndarray, end_index: np.ndarray,
) -> np.ndarray:
    n = len(op)
    hcount = len(end_index)
    output = np.full((hcount, len(TARGET_NAMES)), np.nan, np.float32)
    past_var = rolling_mean(np.square(r), WARMUP_BARS)[end_index]
    sigma = np.sqrt(past_var)
    entry = end_index + 2  # 01:00 decision -> 01:05 execution open
    safe = (entry + HORIZON_BARS < n) & np.isfinite(sigma) & (sigma > 0)
    if not safe.any():
        return output
    good = np.flatnonzero(safe)
    e = entry[good]
    first_return = np.log(cl[e] / op[e])
    positive_sq = np.square(np.maximum(r, 0.0))
    negative_sq = np.square(np.minimum(r, 0.0))
    # r[e+1] ... r[e+H-1] plus the execution-bar open->close return.
    pos_cs = np.concatenate(([0.0], np.cumsum(positive_sq)))
    neg_cs = np.concatenate(([0.0], np.cumsum(negative_sq)))

    def semivariance(bars: int) -> tuple[np.ndarray, np.ndarray]:
        u = np.square(np.maximum(first_return, 0.0)) + _segment_sum(pos_cs, e + 1, e + bars)
        d = np.square(np.minimum(first_return, 0.0)) + _segment_sum(neg_cs, e + 1, e + bars)
        return u, d

    u4, d4 = semivariance(HORIZON_BARS)
    u1, d1 = semivariance(12)
    eps4 = np.maximum(1e-10, 0.01 * HORIZON_BARS * past_var[good])
    eps1 = np.maximum(1e-10, 0.01 * 12 * past_var[good])
    a4 = (u4 - d4) / (u4 + d4 + 2 * eps4)
    a1 = (u1 - d1) / (u1 + d1 + 2 * eps1)
    logratio = np.log((u4 + eps4) / (d4 + eps4))
    relative_var = np.log((u4 + d4 + 2 * eps4) / (HORIZON_BARS * past_var[good] + 2 * eps4))
    raw_return = np.log(cl[e + HORIZON_BARS - 1] / op[e])
    execution_return = np.log(op[e + HORIZON_BARS] / op[e])
    risk_return = np.clip(raw_return / (sigma[good] * np.sqrt(HORIZON_BARS) + 1e-8), -5.0, 5.0)
    output[good] = np.column_stack((
        a4, logratio, relative_var, risk_return, raw_return,
        execution_return, a1, sigma[good],
    )).astype(np.float32)
    return output


def build_cache(source_root: Path, output_root: Path) -> dict[str, object]:
    symbols = tuple(sorted(p.name for p in source_root.iterdir() if (p / "5m.parquet").is_file()))
    if len(symbols) != 12 or "BTCUSDT" not in symbols or "ETHUSDT" not in symbols:
        raise ValueError(f"expected the audited 12-contract universe, got {symbols}")
    output_root.mkdir(parents=True, exist_ok=True)
    files = [source_root / symbol / "5m.parquet" for symbol in symbols]
    raw: dict[str, list[np.ndarray]] = {name: [] for name in SOURCE_COLUMNS if name not in {"date", "open_time", "close_time"}}
    date = None
    for symbol, path in zip(symbols, files, strict=True):
        frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS))
        observed_date = _validate_bars(frame, symbol, date)
        if date is None:
            date = observed_date
        for name in raw:
            raw[name].append(frame[name].to_numpy(np.float64))
        print(f"loaded {symbol}: {len(frame)} bars", flush=True)
    assert date is not None
    n = len(date)
    if n % 12:
        raise ValueError("5m source does not contain complete hourly groups")
    arrays = {key: np.column_stack(values) for key, values in raw.items()}
    op, hi, lo, cl = (arrays[name] for name in ("open", "high", "low", "close"))
    vol, quote, trades, buy = (
        arrays[name] for name in ("volume", "quote_volume", "trade_count", "taker_buy_quote_volume")
    )
    r = np.zeros_like(cl)
    r[1:] = np.log(cl[1:] / cl[:-1])
    n_hours, n_symbols = n // 12, len(symbols)
    end_index = np.arange(n_hours) * 12 + 11
    decision_time = date[end_index] + np.timedelta64(5, "m")
    btc = symbols.index("BTCUSDT")
    eth = symbols.index("ETHUSDT")
    market_breadth = (r > 0).mean(axis=1)
    market_dispersion = r.std(axis=1)
    btc_r1h = rolling_sum(r[:, btc], 12)
    eth_r1h = rolling_sum(r[:, eth], 12)
    fast = np.full((n, n_symbols, len(FAST_NAMES)), np.nan, np.float32)
    slow = np.full((n_hours, n_symbols, len(SLOW_NAMES)), np.nan, np.float32)
    stats = np.full((n_hours, n_symbols, len(STAT_CHANNELS) * 5), np.nan, np.float32)
    market = np.full((n_hours, n_symbols, len(MARKET_NAMES)), np.nan, np.float32)
    targets = np.full((n_hours, n_symbols, len(TARGET_NAMES)), np.nan, np.float32)
    # End-of-bar UTC clock, never the source's bar-open hour.
    clock = pd.DatetimeIndex(date + np.timedelta64(5, "m"))
    hour_angle = 2 * np.pi * (clock.hour.to_numpy() + clock.minute.to_numpy() / 60) / 24
    week_angle = 2 * np.pi * (clock.dayofweek.to_numpy() * 24 + clock.hour.to_numpy()) / 168

    for si, symbol in enumerate(symbols):
        v = vol[:, si]
        q = quote[:, si]
        tr = trades[:, si]
        b = buy[:, si]
        rr = r[:, si]
        flow = np.where(q > 0, 2 * b / np.maximum(q, 1e-12) - 1, np.nan)
        logq = np.log1p(q)
        logtr = np.log1p(tr)
        rv1h = np.sqrt(rolling_sum(np.square(rr), 12))
        rv4h = np.sqrt(rolling_sum(np.square(rr), 48))
        u1 = rolling_sum(np.square(np.maximum(rr, 0)), 12)
        d1 = rolling_sum(np.square(np.minimum(rr, 0)), 12)
        u4 = rolling_sum(np.square(np.maximum(rr, 0)), 48)
        d4 = rolling_sum(np.square(np.minimum(rr, 0)), 48)
        asym1 = _safe_div(u1 - d1, u1 + d1 + 1e-12)
        asym4 = _safe_div(u4 - d4, u4 + d4 + 1e-12)
        qbaseline = np.roll(rolling_mean(logq, 288), 1)
        tbaseline = np.roll(rolling_mean(logtr, 288), 1)
        qbaseline[0] = np.nan
        tbaseline[0] = np.nan
        vwap = _safe_div(q, v)
        close_location = np.nan_to_num(
            2 * _safe_div(cl[:, si] - lo[:, si], hi[:, si] - lo[:, si]) - 1,
            nan=0.0,
        )
        block = (
            rr, np.log(cl[:, si] / op[:, si]), np.log(hi[:, si] / lo[:, si]),
            np.log(hi[:, si] / np.maximum(op[:, si], cl[:, si])),
            np.log(np.minimum(op[:, si], cl[:, si]) / lo[:, si]),
            close_location, logq, logtr, flow, flow * logq,
            np.log(_safe_div(vwap, cl[:, si])), rolling_sum(rr, 12), rolling_sum(rr, 48),
            rv1h, rv4h, asym1, asym4, logq - qbaseline, logtr - tbaseline,
            rolling_mean(flow, 12), rolling_mean(flow, 48),
            np.log1p(1e6 * np.abs(rr) / (q + 1)), r[:, btc], btc_r1h,
            r[:, eth], market_breadth, market_dispersion,
            np.sin(hour_angle), np.cos(hour_angle), np.sin(week_angle), np.cos(week_angle),
        )
        fast[:, si, :] = np.column_stack(block).astype(np.float32)
        # A zero-turnover bar has no observed flow or VWAP, not a true zero reading.
        for channel in (8, 9, 10, 19, 20):
            fast[q <= 0, si, channel] = np.nan
        r24h = rolling_sum(rr, 288)
        rv24h = np.sqrt(rolling_sum(np.square(rr), 288))
        u24 = rolling_sum(np.square(np.maximum(rr, 0)), 288)
        d24 = rolling_sum(np.square(np.minimum(rr, 0)), 288)
        r7d = rolling_sum(rr, WARMUP_BARS)
        rv7d = np.sqrt(rolling_sum(np.square(rr), WARMUP_BARS))
        flow24h = rolling_mean(flow, 288)
        slow_block = (
            fast[end_index, si, 11], fast[end_index, si, 12], fast[end_index, si, 13],
            fast[end_index, si, 14], fast[end_index, si, 15], fast[end_index, si, 16],
            fast[end_index, si, 17], fast[end_index, si, 19], fast[end_index, si, 20],
            fast[end_index, si, 21], r24h[end_index], rv24h[end_index],
            _safe_div(u24 - d24, u24 + d24 + 1e-12)[end_index],
            r7d[end_index], rv7d[end_index], flow24h[end_index],
            btc_r1h[end_index], rolling_sum(r[:, btc], 288)[end_index],
            np.sin(hour_angle[end_index]), np.sin(week_angle[end_index]),
        )
        slow[:, si, :] = np.column_stack(slow_block).astype(np.float32)
        selected = fast[:, si, STAT_CHANNELS].reshape(n_hours, 12, len(STAT_CHANNELS))
        finite = np.isfinite(selected)
        hourly_mean = _safe_div(
            np.where(finite, selected, 0).sum(axis=1), finite.sum(axis=1)
        )
        last = fast[end_index, si][:, STAT_CHANNELS]
        mean4 = np.column_stack([rolling_mean(hourly_mean[:, j], 4) for j in range(len(STAT_CHANNELS))])
        mean24 = np.column_stack([rolling_mean(hourly_mean[:, j], 24) for j in range(len(STAT_CHANNELS))])
        sq24 = np.column_stack([rolling_mean(np.square(hourly_mean[:, j]), 24) for j in range(len(STAT_CHANNELS))])
        std24 = np.sqrt(np.maximum(sq24 - np.square(mean24), 0))
        stats[:, si, :] = np.column_stack((last, hourly_mean, mean4, mean24, std24)).astype(np.float32)
        own_hour = slow[:, si, 0].astype(np.float64)
        btc_hour = btc_r1h[end_index]
        own_mean = rolling_mean(own_hour, 168)
        btc_mean = rolling_mean(btc_hour, 168)
        covariance = rolling_mean(own_hour * btc_hour, 168) - own_mean * btc_mean
        btc_variance = rolling_mean(np.square(btc_hour), 168) - np.square(btc_mean)
        own_variance = rolling_mean(np.square(own_hour), 168) - np.square(own_mean)
        beta = _safe_div(covariance, btc_variance + 1e-12)
        corr = _safe_div(covariance, np.sqrt(np.maximum(own_variance * btc_variance, 0)) + 1e-12)
        market_block = (
            btc_hour, rolling_sum(r[:, btc], 48)[end_index],
            rolling_sum(r[:, btc], 288)[end_index],
            np.sqrt(rolling_sum(np.square(r[:, btc]), 288))[end_index],
            eth_r1h[end_index],
            rolling_mean(market_breadth, 12)[end_index],
            rolling_mean(market_dispersion, 12)[end_index],
            beta, own_hour - beta * btc_hour, corr,
        )
        market[:, si, :] = np.column_stack(market_block).astype(np.float32)
        targets[:, si, :] = _targets_for_symbol(op[:, si], cl[:, si], rr, end_index)
        print(f"featured {symbol}", flush=True)

    for name, array in (("fast", fast), ("slow", slow), ("stats", stats),
                        ("market", market), ("targets", targets), ("decision_time", decision_time)):
        np.save(output_root / f"{name}.npy", array)
    manifest = {
        "version": 1,
        "symbols": list(symbols),
        "bar_count_per_symbol": n,
        "hour_count": n_hours,
        "first_decision_utc": str(decision_time[0]),
        "last_decision_utc": str(decision_time[-1]),
        "fast_names": list(FAST_NAMES), "slow_names": list(SLOW_NAMES),
        "stat_channels": list(STAT_CHANNELS), "market_names": list(MARKET_NAMES),
        "target_names": list(TARGET_NAMES),
        "execution_delay_bars": 1,
        "horizon_bars": HORIZON_BARS,
        "source_files_sha256": {str(path.relative_to(source_root)): _sha256(path) for path in files},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_cache(root: Path, *, mmap_mode: str | None = "r") -> dict[str, np.ndarray]:
    result = {name: np.load(root / f"{name}.npy", mmap_mode=mmap_mode) for name in (
        "fast", "slow", "stats", "market", "targets", "decision_time",
    )}
    result["manifest"] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return result
