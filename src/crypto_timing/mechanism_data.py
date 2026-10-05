"""Minute-source, causal economic representations. Fitted quantities live in folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import SOURCE_COLUMNS, _sha256, rolling_mean, rolling_sum

FAST_NAMES = ("return", "body", "range", "upper_wick", "lower_wick", "close_location",
              "quote_surprise", "trades_surprise", "pressure", "vwap_gap", "illiquidity",
              "btc_return", "eth_return", "breadth", "absorption", "btc_residual",
              "quote_level", "trade_size", "no_trade", "return_copy")
SELECTED = (0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18)
EXTRA_NAMES = ("pressure_leads_price", "price_leads_pressure", "quote_concentration",
               "sign_continuity", "peak_retrace", "path_efficiency")
SLOW_NAMES = ("return", "log_risk", "pressure", "quote_surprise", "trades_surprise",
              "range", "btc_return", "btc_residual", "absorption", "breadth")
STATE_NAMES = ("return1h", "return4h", "return24h", "return7d", "return30d",
               "log_risk1h", "log_risk24h", "log_risk7d", "log_risk30d", "risk_short_long",
               "pressure1h", "pressure4h", "quote_surprise1h", "log_quote1h", "trade_size1h",
               "beta_btc7d", "btc_return4h", "eth_return4h", "breadth1h", "dispersion1h",
               "hour_sin", "hour_cos", "week_sin", "week_cos")
MINUTE_NAMES = ("return_shape", "body_shape", "range_shape", "close_location", "pressure",
                "quote_surprise", "trades_surprise", "vwap_gap_shape", "log_sigma",
                "illiquidity", "no_trade", "pressure_lag_response")
RAW_TARGET_NAMES = ("G1", "G4", "G8", "U5", "D5", "U1", "D1", "raw_scale4", "future_no_trade")
SHAPE_FAST = (0, 1, 2, 3, 4, 9, 11, 12, 14, 15, 19)
BOUNDED = {"fast": (5, 8, 13, 18), "slow": (2, 9), "state": (10, 11, 18, 20, 21, 22, 23),
           "minute": (3, 4, 10), "extra": (2, 3, 5)}


def lag(x: np.ndarray, n: int = 1) -> np.ndarray:
    out = np.roll(x, n, axis=0)
    out[:n] = np.nan
    return out


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan),
                     where=np.isfinite(b) & (b > 0))


def ewm_past(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, min_periods=span, adjust=False).mean().shift(1).to_numpy()


def aggregate(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if len(frame) % minutes:
        raise ValueError("incomplete aggregation block")
    out = {}
    for name in SOURCE_COLUMNS:
        x = frame[name].to_numpy().reshape(-1, minutes)
        if name in {"date", "open_time", "open"}:
            out[name] = x[:, 0]
        elif name in {"close_time", "close"}:
            out[name] = x[:, -1]
        elif name == "high":
            out[name] = x.max(1)
        elif name == "low":
            out[name] = x.min(1)
        else:
            out[name] = x.sum(1)
    return pd.DataFrame(out)


def validate(frame: pd.DataFrame, minutes: int = 1) -> dict:
    date = frame.date.to_numpy(dtype="datetime64[ns]")
    otime = frame.open_time.to_numpy()
    if len(date) == 0 or np.any(np.diff(date) != np.timedelta64(minutes, "m")):
        raise ValueError("gap, duplicate, or unordered time key")
    if np.any(otime != date.astype("datetime64[ms]").astype(np.int64)):
        raise ValueError("open time mismatch")
    if np.any(frame.close_time.to_numpy() != otime + minutes * 60000 - 1):
        raise ValueError("incomplete bar")
    op, hi, lo, cl, vol, q, tr, b = [frame[k].to_numpy(np.float64) for k in
                                    ("open", "high", "low", "close", "volume", "quote_volume",
                                     "trade_count", "taker_buy_quote_volume")]
    valid = (np.isfinite(np.column_stack((op, hi, lo, cl, vol, q, tr, b))).all(1)
             & (lo > 0) & (hi >= np.maximum(op, cl)) & (lo <= np.minimum(op, cl))
             & (vol >= 0) & (q >= 0) & (tr >= 0) & (b >= -1e-6) & (b <= q + 1e-3))
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} invalid OHLC/flow rows")
    return {"rows": len(frame), "first_open_utc": str(date[0]), "last_open_utc": str(date[-1]),
            "zero_quote": int((q == 0).sum()), "missing_rows": 0, "invalid_rows": 0}


def base_features(frame: pd.DataFrame, per_hour: int) -> tuple[np.ndarray, np.ndarray, dict]:
    op, hi, lo, cl, q, tr, buy, volume = [frame[k].to_numpy(np.float64) for k in
                                        ("open", "high", "low", "close", "quote_volume",
                                         "trade_count", "taker_buy_quote_volume", "volume")]
    r = np.log(cl / lag(cl))
    sigma = np.sqrt(ewm_past(r * r, 24 * per_hour))
    sigma = np.maximum(sigma, 1e-7)
    qb = ewm_past(q, 24 * per_hour)
    tb = ewm_past(tr, 24 * per_hour)
    pressure = ratio(2 * buy - q, q)
    # Historical impact coefficient; no contemporary numerator in its denominator.
    k = ratio(lag(rolling_sum(np.nan_to_num(r * pressure), 24 * per_hour)),
              lag(rolling_sum(np.nan_to_num(pressure ** 2), 24 * per_hour)) + 1e-9)
    close_location = np.where(hi > lo, 2 * (cl - lo) / np.maximum(hi - lo, 1e-12) - 1, 0)
    vgap = np.log(ratio(ratio(np.where(q > 0, q, np.nan), volume), cl))
    ilq = np.log1p(ratio(np.abs(r), np.maximum(q, .01 * qb)) * 1e6)
    x = np.column_stack((r, np.log(cl / op), np.log(hi / lo),
                         np.log(hi / np.maximum(op, cl)), np.log(np.minimum(op, cl) / lo),
                         close_location, np.log(ratio(q + 1e-9, qb + 1e-9)),
                         np.log(ratio(tr + 1e-9, tb + 1e-9)), pressure, vgap, ilq,
                         r, r, np.zeros(len(r)), r - k * pressure, r,
                         np.log(qb + 1e-9), np.log(ratio(q + 1e-9, tr + 1e-9)),
                         (q == 0).astype(float), r))
    return x, sigma, {"r": r, "q": q, "tr": tr, "b": buy, "pressure": pressure}


def field_transform(raw: np.ndarray) -> np.ndarray:
    x = raw.copy()
    # The economic zero stays zero for signed fields. Bounded quantities bypass centering.
    x[:, SHAPE_FAST] = np.arcsinh(x[:, SHAPE_FAST] / .001)
    return x


def dynamic_transform(raw: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = raw.copy()
    x[:, SHAPE_FAST] = np.arcsinh(x[:, SHAPE_FAST] / sigma[:, None])
    return x


def path_extras(frame: pd.DataFrame, sigma5: np.ndarray) -> np.ndarray:
    cl = frame.close.to_numpy()
    q = frame.quote_volume.to_numpy()
    pressure = ratio(2 * frame.taker_buy_quote_volume.to_numpy() - q, q)
    r = np.log(cl / lag(cl)).reshape(-1, 5)
    f = pressure.reshape(-1, 5)
    qc = q.reshape(-1, 5)
    prices = cl.reshape(-1, 5)
    lead = np.nansum(f[:, :-1] * r[:, 1:], axis=1) / sigma5
    chase = np.nansum(r[:, :-1] * f[:, 1:], axis=1) / sigma5
    concentration = ratio(qc.max(1), qc.sum(1))
    continuity = (np.sign(r[:, 1:]) == np.sign(r[:, :-1])).mean(1)
    retrace = np.log(prices.max(1) / prices[:, -1]) / sigma5
    efficiency = ratio(np.abs(r.sum(1)), np.abs(r).sum(1) + 1e-9)
    return np.column_stack((np.arcsinh(lead), np.arcsinh(chase), concentration,
                            continuity, np.arcsinh(retrace), efficiency)).astype(np.float32)


def raw_targets(frame: pd.DataFrame, scale4: np.ndarray) -> np.ndarray:
    op = frame.open.to_numpy(np.float64)
    nh = len(op) // 60
    entry = np.arange(nh) * 60 + 65  # decision at next hour; five-minute delay
    out = np.full((nh, len(RAW_TARGET_NAMES)), np.nan)
    good = np.flatnonzero(entry + 480 < len(op))
    e = entry[good]
    for col, h in enumerate((60, 240, 480)):
        out[good, col] = op[e + h] / op[e] - 1
    for sampling, col in ((5, 3), (1, 5)):
        # Entry and expiry are exactly shared with terminal returns.
        increments = np.log(op[sampling:] / op[:-sampling])
        u = np.maximum(increments, 0) ** 2
        d = np.minimum(increments, 0) ** 2
        if sampling == 5:
            u, d = u[::5], d[::5]
            indices, count = e // 5, 48
        else:
            indices, count = e, 240
        for offset, x in enumerate((u, d)):
            cs = np.r_[0., np.cumsum(x)]
            out[good, col + offset] = cs[indices + count] - cs[indices]
    out[:, 7] = scale4
    no_trade = (frame.quote_volume.to_numpy() == 0).astype(float)
    cs = np.r_[0, np.cumsum(no_trade)]
    out[good, 8] = cs[e + 480] - cs[e]
    return out.astype(np.float32)


def build_mechanism_cache(source: Path, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    symbols = sorted(p.name for p in source.iterdir() if (p / "1m.parquet").exists())
    if len(symbols) != 12:
        raise ValueError("expected twelve minute-source contracts")
    first = pd.read_parquet(source / symbols[0] / "1m.parquet", columns=["date"])
    n = len(first)
    dates = first.date.to_numpy(dtype="datetime64[ns]")
    if dates[0].astype("datetime64[m]").astype(int) % 60 or n % 60:
        raise ValueError("source must consist of complete UTC hours")
    nh, ns = n // 60, len(symbols)
    arrays = {}
    for name, shape in {"minute": (n, ns, 12), "extra": (n // 5, ns, 6),
                        "slow_raw": (nh, ns, 10), "slow_field": (nh, ns, 10),
                        "slow_dynamic": (nh, ns, 10), "state_raw": (nh, ns, 24),
                        "state_field": (nh, ns, 24), "state_dynamic": (nh, ns, 24),
                        "targets": (nh, ns, 9),
                        **{f"fast_{p}": (n // 5, ns, 20) for p in ("raw", "field", "dynamic")}}.items():
        arrays[name] = np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
    audit = {}
    returns5 = np.empty((n // 5, ns))
    returns1h = np.empty((nh, ns))
    sigmas5 = np.empty_like(returns5)
    sigmas1h = np.empty_like(returns1h)
    for si, symbol in enumerate(symbols):
        path = source / symbol / "1m.parquet"
        frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS) + ["log_return", "vwap", "taker_buy_ratio", "taker_buy_volume"])
        stats = validate(frame)
        if not np.array_equal(frame.date.to_numpy(dtype="datetime64[ns]"), dates):
            raise ValueError(f"{symbol}: inconsistent time grid")
        five = aggregate(frame, 5)
        old = pd.read_parquet(source / symbol / "5m.parquet", columns=list(SOURCE_COLUMNS))
        validate(old, 5)
        if not np.array_equal(five.date.to_numpy(dtype="datetime64[ns]"), old.date.to_numpy(dtype="datetime64[ns]")):
            raise ValueError("1m/5m dates disagree")
        stats["aggregation_max_relative_error"] = {}
        stats["aggregation_disagreement_rows"] = {}
        for col in SOURCE_COLUMNS[3:]:
            a, b = five[col].to_numpy(float), old[col].to_numpy(float)
            err = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
            stats["aggregation_max_relative_error"][col] = err
            stats["aggregation_disagreement_rows"][col] = int((~np.isclose(a, b, rtol=2e-6, atol=1e-8)).sum())
        stats["canonical_source"] = "updated 1m; all coarser bars rebuilt, no old 5m values mixed"
        hourly = aggregate(frame, 60)
        fx, fs, f = base_features(five, 12)
        mx, ms, m = base_features(frame, 60)
        hx, hs, h = base_features(hourly, 1)
        stats["derived_audit"] = {}
        for col, calc in (("log_return", m["r"]), ("vwap", ratio(m["q"], frame.volume.to_numpy())),
                          ("taker_buy_ratio", ratio(frame.taker_buy_volume.to_numpy(), frame.volume.to_numpy()))):
            observed = frame[col].to_numpy(float) if col in frame else pd.read_parquet(path, columns=[col])[col].to_numpy(float)
            valid = np.isfinite(calc) & np.isfinite(observed)
            stats["derived_audit"][col] = {"compared": int(valid.sum()),
                                          "max_absolute_error": float(np.abs(calc[valid] - observed[valid]).max()),
                                          "used_as_input": False}
        arrays["minute"][:, si] = np.column_stack((np.arcsinh(mx[:, 0] / ms), np.arcsinh(mx[:, 1] / ms),
            np.arcsinh(mx[:, 2] / ms), mx[:, 5], mx[:, 8], mx[:, 6], mx[:, 7],
            np.arcsinh(mx[:, 9] / ms), np.log(ms), mx[:, 10], mx[:, 18],
            np.arcsinh(lag(m["pressure"]) * m["r"] / ms)))
        arrays["extra"][:, si] = path_extras(frame, fs)
        for prep, x in (("raw", fx), ("field", field_transform(fx)), ("dynamic", dynamic_transform(fx, fs))):
            arrays[f"fast_{prep}"][:, si] = x
        returns5[:, si], sigmas5[:, si] = f["r"], fs
        returns1h[:, si], sigmas1h[:, si] = h["r"], hs
        r, sq = h["r"], f["r"].reshape(nh, 12)
        hour_square = (sq * sq).sum(1)
        risk = [np.sqrt(rolling_sum(hour_square, w)) for w in (1, 24, 168, 720)]
        ret = [rolling_sum(r, w) for w in (1, 4, 24, 168, 720)]
        scale4 = np.sqrt(48 * (.5 * rolling_mean(f["r"] ** 2, 288)[11::12]
                              + .5 * rolling_mean(f["r"] ** 2, 2016)[11::12]))
        s = np.column_stack((r, np.log(risk[0] + 1e-9), h["pressure"], hx[:, 6], hx[:, 7],
                             hx[:, 2], r, r, hx[:, 14], np.zeros(nh)))
        for prep in ("raw", "field", "dynamic"):
            values = s.copy()
            if prep != "raw":
                for j in (0, 5, 6, 7, 8):
                    values[:, j] = np.arcsinh(values[:, j] / (hs if prep == "dynamic" else .01))
            arrays[f"slow_{prep}"][:, si] = values
        pressure4 = ratio(2 * rolling_sum(h["b"], 4) - rolling_sum(h["q"], 4), rolling_sum(h["q"], 4))
        time = dates[59::60] + np.timedelta64(1, "m")
        hours = time.astype("datetime64[h]").astype(np.int64)
        state = np.column_stack((*ret, *[np.log(v + 1e-9) for v in risk],
            np.log(ratio(risk[1], risk[2] / np.sqrt(7))), h["pressure"], pressure4,
            hx[:, 6], hx[:, 16], hx[:, 17], np.zeros(nh), r, r, np.zeros(nh), np.zeros(nh),
            np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
            np.sin(2 * np.pi * hours / 168), np.cos(2 * np.pi * hours / 168)))
        for prep in ("raw", "field", "dynamic"):
            values = state.copy()
            if prep != "raw":
                for j, window in enumerate((1, 4, 24, 168, 720)):
                    values[:, j] = np.arcsinh(state[:, j] / (hs * np.sqrt(window) if prep == "dynamic" else .01))
            arrays[f"state_{prep}"][:, si] = values
        arrays["targets"][:, si] = raw_targets(frame, scale4)
        stats["source_sha256"] = _sha256(path)
        stats["five_source_sha256"] = _sha256(source / symbol / "5m.parquet")
        audit[symbol] = stats
        print(json.dumps({"event": "minute_audited", "symbol": symbol, **stats}, ensure_ascii=False), flush=True)
        for value in arrays.values():
            value.flush()
    btc, eth = symbols.index("BTCUSDT"), symbols.index("ETHUSDT")
    for si in range(ns):
        hr, br = returns1h[:, si], returns1h[:, btc]
        beta = ratio(lag(rolling_mean(hr * br, 168) - rolling_mean(hr, 168) * rolling_mean(br, 168)),
                     lag(rolling_mean(br * br, 168) - rolling_mean(br, 168) ** 2) + 1e-10)
        beta5 = np.repeat(beta, 12)
        for prep in ("raw", "field", "dynamic"):
            div5 = sigmas5[:, si] if prep == "dynamic" else .001
            divh = sigmas1h[:, si] if prep == "dynamic" else .01
            transform = (lambda a, d: a) if prep == "raw" else (lambda a, d: np.arcsinh(a / d))
            fast = arrays[f"fast_{prep}"]
            fast[:, si, 11] = transform(returns5[:, btc], div5)
            fast[:, si, 12] = transform(returns5[:, eth], div5)
            fast[:, si, 13] = 2 * (returns5 > 0).mean(1) - 1
            fast[:, si, 15] = transform(returns5[:, si] - beta5 * returns5[:, btc], div5)
            slow = arrays[f"slow_{prep}"]
            slow[:, si, 6] = transform(br, divh)
            slow[:, si, 7] = transform(hr - beta * br, divh)
            slow[:, si, 9] = 2 * (returns1h > 0).mean(1) - 1
            state = arrays[f"state_{prep}"]
            state[:, si, 15] = beta
            state[:, si, 16] = transform(rolling_sum(br, 4), divh * 2)
            state[:, si, 17] = transform(rolling_sum(returns1h[:, eth], 4), divh * 2)
            state[:, si, 18] = 2 * (returns1h > 0).mean(1) - 1
            state[:, si, 19] = np.log(returns1h.std(1) + 1e-9)
    np.save(root / "decision_time.npy", dates[59::60] + np.timedelta64(1, "m"))
    for value in arrays.values():
        value.flush()
    manifest = {"version": 4, "symbols": symbols, "minute_rows_per_symbol": n,
                "hours": nh, "fast_names": FAST_NAMES, "selected_fast": SELECTED,
                "slow_names": SLOW_NAMES, "state_names": STATE_NAMES, "minute_names": MINUTE_NAMES,
                "extra_names": EXTRA_NAMES, "raw_target_names": RAW_TARGET_NAMES,
                "entry_delay_minutes": 5, "warmup_hours": 720, "audit": audit}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
