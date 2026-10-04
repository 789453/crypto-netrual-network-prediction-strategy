"""Version-three point-in-time features and executable-return distribution labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import SOURCE_COLUMNS, _sha256, _validate_bars, rolling_mean, rolling_sum

FAST_NAMES = (
    "r5", "body", "range", "upper_wick", "lower_wick", "close_location",
    "quote_surprise", "trades_surprise", "taker_pressure", "vwap_gap",
    "illiquidity", "btc_r5", "eth_r5", "breadth5", "pressure_absorption",
    "btc_residual5",
)
SLOW_NAMES = (
    "r1h", "rv1h", "weighted_pressure1h", "quote_surprise1h",
    "trades_surprise1h", "range1h", "btc_r1h", "btc_residual1h",
    "pressure_absorption1h", "breadth1h",
)
STATE_NAMES = (
    "r1h", "r4h", "r24h", "r7d", "r30d", "rv1h", "rv4h", "rv24h",
    "rv7d", "rv30d", "pressure1h", "pressure4h", "quote_surprise1h",
    "quote_surprise24h", "beta_btc7d", "btc_residual4h", "btc_r4h",
    "btc_r24h", "log_causal_scale", "rv24h_over_rv7d", "hour_sin",
    "hour_cos", "week_sin", "week_cos",
)
MAG_NAMES = (
    "log_rv1h", "log_rv24h", "log_rv7d", "log_rv30d",
    "rv24h_over_rv7d", "range1h", "range24h", "quote_surprise1h",
    "log_quote1h", "breadth1h",
)
TARGET_NAMES = ("simple_return4h", "scaled_return4h", "magnitude_bin", "up_label", "causal_scale")
TRAIN_END = np.datetime64("2025-07-01T00:00:00")
CAL_END = np.datetime64("2025-11-01T00:00:00")
VALID_END = np.datetime64("2026-02-01T00:00:00")
HORIZON_BARS = 48
WARMUP_HOURS = 720


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan),
                     where=np.isfinite(b) & (b != 0))


def _lag(values: np.ndarray, count: int = 1) -> np.ndarray:
    result = np.roll(values, count)
    result[:count] = np.nan
    return result


def _rv(hour_square: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.maximum(rolling_sum(hour_square, window), 0))


def build_redesign_cache(source_root: Path, output_root: Path) -> dict:
    symbols = tuple(sorted(p.name for p in source_root.iterdir() if (p / "5m.parquet").is_file()))
    if len(symbols) != 12 or "BTCUSDT" not in symbols or "ETHUSDT" not in symbols:
        raise ValueError(f"expected audited 12 symbols, got {symbols}")
    files = [source_root / symbol / "5m.parquet" for symbol in symbols]
    raw = {name: [] for name in SOURCE_COLUMNS if name not in {"date", "open_time", "close_time"}}
    date = None
    for symbol, path in zip(symbols, files, strict=True):
        frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS))
        observed = _validate_bars(frame, symbol, date)
        if date is None:
            date = observed
        for name in raw:
            raw[name].append(frame[name].to_numpy(np.float64))
        print(f"loaded {symbol}: {len(frame)} 5m bars", flush=True)
    assert date is not None
    arrays = {key: np.column_stack(value) for key, value in raw.items()}
    op, hi, lo, cl = (arrays[key] for key in ("open", "high", "low", "close"))
    quote, trades, buy = (arrays[key] for key in ("quote_volume", "trade_count", "taker_buy_quote_volume"))
    n_bars, n_symbols = op.shape
    if n_bars % 12:
        raise ValueError("incomplete hourly group")
    n_hours = n_bars // 12
    end = np.arange(n_hours) * 12 + 11
    decision_time = date[end] + np.timedelta64(5, "m")
    btc, eth = symbols.index("BTCUSDT"), symbols.index("ETHUSDT")
    r = np.zeros_like(cl)
    r[1:] = np.log(cl[1:] / cl[:-1])
    hour_r = r.reshape(n_hours, 12, n_symbols).sum(axis=1)
    hour_sq = np.square(r).reshape(n_hours, 12, n_symbols).sum(axis=1)
    hour_quote = quote.reshape(n_hours, 12, n_symbols).sum(axis=1)
    hour_buy = buy.reshape(n_hours, 12, n_symbols).sum(axis=1)
    hour_trades = trades.reshape(n_hours, 12, n_symbols).sum(axis=1)
    hour_hi = hi.reshape(n_hours, 12, n_symbols).max(axis=1)
    hour_lo = lo.reshape(n_hours, 12, n_symbols).min(axis=1)
    hour_breadth = (r > 0).reshape(n_hours, 12, n_symbols).mean(axis=(1, 2))
    fast = np.full((n_bars, n_symbols, len(FAST_NAMES)), np.nan, np.float32)
    slow = np.full((n_hours, n_symbols, len(SLOW_NAMES)), np.nan, np.float32)
    state = np.full((n_hours, n_symbols, len(STATE_NAMES)), np.nan, np.float32)
    magnitude = np.full((n_hours, n_symbols, len(MAG_NAMES)), np.nan, np.float32)
    targets = np.full((n_hours, n_symbols, len(TARGET_NAMES)), np.nan, np.float32)
    scale_floors: dict[str, float] = {}
    btc_r = r[:, btc]
    btc_hour = hour_r[:, btc]
    hour_angle = 2 * np.pi * (decision_time.astype("datetime64[h]").astype(np.int64) % 24) / 24
    week_angle = 2 * np.pi * (decision_time.astype("datetime64[h]").astype(np.int64) % 168) / 168
    train_hours = decision_time < TRAIN_END

    for si, symbol in enumerate(symbols):
        q, tr, b, rr = quote[:, si], trades[:, si], buy[:, si], r[:, si]
        hq, ht, hb, hr, hs = (hour_quote[:, si], hour_trades[:, si], hour_buy[:, si],
                               hour_r[:, si], hour_sq[:, si])
        f = _ratio(2 * b - q, q)
        hf = _ratio(2 * hb - hq, hq)
        logq, logtr = np.log1p(q), np.log1p(tr)
        hlogq, hlogtr = np.log1p(hq), np.log1p(ht)
        q_surp = logq - _lag(rolling_mean(logq, 288))
        tr_surp = logtr - _lag(rolling_mean(logtr, 288))
        hq_surp = hlogq - _lag(rolling_mean(hlogq, 24))
        ht_surp = hlogtr - _lag(rolling_mean(hlogtr, 24))
        beta_mean = rolling_mean(hr, 168)
        btc_mean = rolling_mean(btc_hour, 168)
        beta_cov = rolling_mean(hr * btc_hour, 168) - beta_mean * btc_mean
        beta_var = rolling_mean(np.square(btc_hour), 168) - np.square(btc_mean)
        beta = _ratio(beta_cov, beta_var + 1e-12)
        beta_bar = np.repeat(_lag(beta), 12)
        kappa_num = _lag(rolling_sum(np.where(np.isfinite(f), rr * f, 0), 288))
        kappa_den = _lag(rolling_sum(np.where(np.isfinite(f), f * f, 0), 288))
        kappa = np.clip(_ratio(kappa_num, kappa_den + 1e-10), -.1, .1)
        hk_num = _lag(rolling_sum(np.where(np.isfinite(hf), hr * hf, 0), 24))
        hk_den = _lag(rolling_sum(np.where(np.isfinite(hf), hf * hf, 0), 24))
        hk = np.clip(_ratio(hk_num, hk_den + 1e-10), -.1, .1)
        location = np.where(hi[:, si] > lo[:, si],
                            2 * (cl[:, si] - lo[:, si]) / np.maximum(hi[:, si] - lo[:, si], 1e-12) - 1,
                            0)
        vwap = _ratio(q, arrays["volume"][:, si])
        fast[:, si, :] = np.column_stack((
            rr, np.log(cl[:, si] / op[:, si]), np.log(hi[:, si] / lo[:, si]),
            np.log(hi[:, si] / np.maximum(op[:, si], cl[:, si])),
            np.log(np.minimum(op[:, si], cl[:, si]) / lo[:, si]), location,
            q_surp, tr_surp, f, np.log(_ratio(vwap, cl[:, si])),
            np.log1p(1e6 * np.abs(rr) / (q + 1)), btc_r, r[:, eth],
            (r > 0).mean(axis=1), rr - kappa * f, rr - beta_bar * btc_r,
        )).astype(np.float32)
        for channel in (8, 9, 14):
            fast[q <= 0, si, channel] = np.nan
        hourly_range = np.log(hour_hi[:, si] / hour_lo[:, si])
        slow[:, si, :] = np.column_stack((
            hr, np.sqrt(hs), hf, hq_surp, ht_surp, hourly_range,
            btc_hour, hr - beta * btc_hour, hr - hk * hf, hour_breadth,
        )).astype(np.float32)

        returns = [rolling_sum(hr, w) for w in (1, 4, 24, 168, 720)]
        volatilities = [_rv(hs, w) for w in (1, 4, 24, 168, 720)]
        flow4 = _ratio(2 * rolling_sum(hb, 4) - rolling_sum(hq, 4), rolling_sum(hq, 4))
        q24_surp = hlogq - _lag(rolling_mean(hlogq, 168))
        btc_r4 = rolling_sum(btc_hour, 4)
        btc_r24 = rolling_sum(btc_hour, 24)
        ratio24_7 = _ratio(volatilities[2], volatilities[3] / np.sqrt(7) + 1e-10)
        s24 = rolling_mean(np.square(rr), 288)[end]
        s7 = rolling_mean(np.square(rr), 2016)[end]
        raw_scale = np.sqrt(48 * (0.5 * s24 + 0.5 * s7))
        valid_floor = raw_scale[train_hours & np.isfinite(raw_scale) & (np.arange(n_hours) >= WARMUP_HOURS)]
        if len(valid_floor) < 100:
            raise ValueError(f"not enough train scale history: {symbol}")
        floor = float(np.quantile(valid_floor, .05))
        scale_floors[symbol] = floor
        causal_scale = np.maximum(raw_scale, floor)
        state[:, si, :] = np.column_stack((
            *returns, *volatilities, hf, flow4, hq_surp, q24_surp,
            beta, returns[1] - beta * btc_r4, btc_r4, btc_r24,
            np.log(causal_scale), ratio24_7,
            np.sin(hour_angle), np.cos(hour_angle), np.sin(week_angle), np.cos(week_angle),
        )).astype(np.float32)
        range24 = np.log(_ratio(
            rolling_maximum(hour_hi[:, si], 24), rolling_minimum(hour_lo[:, si], 24)
        ))
        magnitude[:, si, :] = np.column_stack((
            np.log(volatilities[0] + 1e-8), np.log(volatilities[2] + 1e-8),
            np.log(volatilities[3] + 1e-8), np.log(volatilities[4] + 1e-8),
            ratio24_7, hourly_range, range24, hq_surp, hlogq, hour_breadth,
        )).astype(np.float32)
        entry = end + 2
        valid_target = (entry + HORIZON_BARS < n_bars) & np.isfinite(causal_scale)
        at = np.flatnonzero(valid_target)
        gross = op[entry[at] + HORIZON_BARS, si] / op[entry[at], si] - 1
        y = gross / causal_scale[at]
        band = np.searchsorted((.5, 1, 2), np.abs(y), side="right")
        sign = np.where(y == 0, .5, (y > 0).astype(float))
        targets[at, si, :] = np.column_stack((gross, y, band, sign, causal_scale[at])).astype(np.float32)
        print(f"featured {symbol}: train scale floor {floor:.6f}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    for name, value in (("fast", fast), ("slow", slow), ("state", state),
                        ("magnitude", magnitude), ("targets", targets),
                        ("decision_time", decision_time)):
        np.save(output_root / f"{name}.npy", value)
    manifest = {
        "version": 3, "symbols": list(symbols), "bar_count_per_symbol": n_bars,
        "hour_count": n_hours, "first_decision_utc": str(decision_time[0]),
        "last_decision_utc": str(decision_time[-1]),
        "fast_names": FAST_NAMES, "slow_names": SLOW_NAMES,
        "state_names": STATE_NAMES, "magnitude_names": MAG_NAMES,
        "target_names": TARGET_NAMES, "scale_floors": scale_floors,
        "fast_window_bars": 144, "slow_window_hours": 72,
        "warmup_hours": WARMUP_HOURS, "horizon_bars": HORIZON_BARS,
        "entry_delay_minutes": 5,
        "source_files_sha256": {str(p.relative_to(source_root)): _sha256(p) for p in files},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def rolling_maximum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).max().to_numpy()


def rolling_minimum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).min().to_numpy()


def load_redesign_cache(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 3:
        raise ValueError("expected v3 cache")
    return {"manifest": manifest, **{name: np.load(root / f"{name}.npy", mmap_mode="r")
                                     for name in ("fast", "slow", "state", "magnitude", "targets", "decision_time")}}
