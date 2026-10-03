"""Point-in-time mark/index and settled funding challenge features."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import _sha256, load_cache

DERIVATIVE_NAMES = (
    "mark_index_log_basis", "basis_change_4h", "mark_return_1h",
    "index_return_1h", "settled_funding_rate", "funding_age_hours",
)


def _asof_derivative_features(path: Path, decision_time: np.ndarray) -> np.ndarray:
    source = pd.read_parquet(path, columns=[
        "date", "mark_close", "index_close", "mark_close_time", "index_close_time",
        "funding_rate",
    ])
    decisions = pd.DataFrame({
        "decision": pd.to_datetime(decision_time, utc=True).astype("datetime64[ns, UTC]")
    })
    price = source.dropna(subset=["mark_close", "index_close", "mark_close_time",
                                  "index_close_time"]).copy()
    if price.duplicated("date").any():
        raise ValueError(f"duplicate mark/index bar in {path}")
    expected_end = price["date"] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    if not (price["mark_close_time"].eq(expected_end)
            & price["index_close_time"].eq(expected_end)).all():
        raise ValueError(f"mark/index close times do not match source hour: {path}")
    price = price.sort_values("date")
    price["known"] = price["mark_close_time"] + pd.Timedelta(milliseconds=1) + pd.Timedelta(minutes=5)
    price["known"] = price["known"].astype("datetime64[ns, UTC]")
    price["basis"] = np.log(price["mark_close"] / price["index_close"])
    price["basis_change_4h"] = price["basis"] - price["basis"].shift(4)
    price.loc[price["date"].sub(price["date"].shift(4)).ne(pd.Timedelta(hours=4)),
              "basis_change_4h"] = np.nan
    price["mark_return_1h"] = np.log(price["mark_close"] / price["mark_close"].shift(1))
    price["index_return_1h"] = np.log(price["index_close"] / price["index_close"].shift(1))
    gap = price["date"].sub(price["date"].shift(1)).ne(pd.Timedelta(hours=1))
    price.loc[gap, ["mark_return_1h", "index_return_1h"]] = np.nan
    price_join = pd.merge_asof(
        decisions, price[["known", "basis", "basis_change_4h", "mark_return_1h",
                          "index_return_1h"]].sort_values("known"),
        left_on="decision", right_on="known", direction="backward",
        tolerance=pd.Timedelta(hours=2),
    )
    # Existing settlement history only; the upcoming settlement rate is unknown.
    funding = source.dropna(subset=["funding_rate"])[["date", "funding_rate"]].sort_values("date").copy()
    if funding.duplicated("date").any():
        raise ValueError(f"duplicate funding event in {path}")
    funding["known"] = funding["date"] + pd.Timedelta(minutes=5)
    funding["known"] = funding["known"].astype("datetime64[ns, UTC]")
    funding_join = pd.merge_asof(
        decisions, funding[["known", "date", "funding_rate"]].sort_values("known"),
        left_on="decision", right_on="known", direction="backward",
        tolerance=pd.Timedelta(hours=16),
    )
    age = (decisions["decision"] - funding_join["date"]).dt.total_seconds() / 3600
    output = np.column_stack((
        price_join["basis"].to_numpy(), price_join["basis_change_4h"].to_numpy(),
        price_join["mark_return_1h"].to_numpy(), price_join["index_return_1h"].to_numpy(),
        funding_join["funding_rate"].to_numpy(), age.to_numpy(),
    )).astype(np.float32)
    return output


def build_derivative_cache(base_root: Path, source_root: Path, output_root: Path) -> dict:
    base = load_cache(base_root)
    output_root.mkdir(parents=True, exist_ok=True)
    market = np.asarray(base["market"])
    enriched = np.full((*market.shape[:2], market.shape[2] + len(DERIVATIVE_NAMES)),
                       np.nan, np.float32)
    enriched[:, :, :market.shape[2]] = market
    symbols = base["manifest"]["symbols"]
    derivative_paths = []
    for si, symbol in enumerate(symbols):
        path = source_root / "derivatives" / f"{symbol}.parquet"
        extra = _asof_derivative_features(path, base["decision_time"])
        enriched[:, si, market.shape[2]:] = extra
        derivative_paths.append(path)
        print(f"enriched {symbol}: basis={np.isfinite(extra[:, 0]).mean():.3f} "
              f"funding={np.isfinite(extra[:, 4]).mean():.3f}", flush=True)
    for name in ("fast", "slow", "stats", "targets", "decision_time"):
        source = base_root / f"{name}.npy"
        destination = output_root / f"{name}.npy"
        if not destination.exists():
            os.link(source, destination)
    np.save(output_root / "market.npy", enriched)
    manifest = dict(base["manifest"])
    manifest["version"] = 2
    manifest["market_names"] = manifest["market_names"] + list(DERIVATIVE_NAMES)
    manifest["derivative_availability"] = "mark/index close+5m; funding settlement+5m"
    manifest["derivative_files_sha256"] = {
        str(path.relative_to(source_root)): _sha256(path) for path in derivative_paths
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest
