"""Verify the frozen v3 time, source and target contract against raw bars."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crypto_timing.cache import _sha256
from crypto_timing.redesign_cache import load_redesign_cache
from crypto_timing.redesign_training import LABEL_MATURITY, RedesignStore


def main() -> None:
    source = Path("D:/Trading/practical_crypto_strategy/data/parquet")
    cache_root = Path("outputs/redesign/cache_v3")
    output = Path("outputs/redesign/evaluation/audit.json")
    cache = load_redesign_cache(cache_root)
    store = RedesignStore(cache_root)
    manifest = cache["manifest"]
    rng = np.random.default_rng(20261004)
    checks = 0
    for si, symbol in enumerate(manifest["symbols"]):
        path = source / symbol / "5m.parquet"
        if _sha256(path) != manifest["source_files_sha256"][str(path.relative_to(source))]:
            raise ValueError(f"source file changed: {symbol}")
        bars = pd.read_parquet(path, columns=["date", "open", "close"])
        dates = bars["date"].to_numpy(dtype="datetime64[ns]")
        op = bars["open"].to_numpy(np.float64)
        cl = bars["close"].to_numpy(np.float64)
        r = np.zeros(len(op))
        r[1:] = np.log(cl[1:] / cl[:-1])
        for split, keys in store.splits.items():
            own = keys[keys % store.n_symbols == si]
            selected = rng.choice(own, min(12, len(own)), replace=False)
            for key in selected:
                hour = int(key // store.n_symbols)
                bar_end = hour * 12 + 11
                entry = bar_end + 2
                actual = np.asarray(store.targets[hour, si], np.float64)
                expected_date = dates[bar_end] + np.timedelta64(5, "m")
                if store.dates[hour] != expected_date:
                    raise AssertionError("decision time differs from completed bar end")
                if dates[entry] != expected_date + np.timedelta64(5, "m"):
                    raise AssertionError("execution open is not decision + 5m")
                if dates[entry + 48] != dates[entry] + np.timedelta64(4, "h"):
                    raise AssertionError("exit open is not entry + 4h")
                gross = op[entry + 48] / op[entry] - 1
                s24 = np.square(r[bar_end - 287:bar_end + 1]).mean()
                s7 = np.square(r[bar_end - 2015:bar_end + 1]).mean()
                v = max(manifest["scale_floors"][symbol], np.sqrt(48 * (.5 * s24 + .5 * s7)))
                y = gross / v
                expected = [gross, y, np.searchsorted((.5, 1, 2), abs(y), side="right"),
                            .5 if y == 0 else float(y > 0), v]
                np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-5)
                checks += 1
    for split, end in (("train", np.datetime64("2025-07-01")),
                       ("calibration", np.datetime64("2025-11-01")),
                       ("selection", np.datetime64("2026-02-01"))):
        maturity = store.dates[store.splits[split] // store.n_symbols] + LABEL_MATURITY
        if maturity.max() > end:
            raise AssertionError(f"unmatured {split} label")
    checkpoints = list(Path("outputs/redesign/models").glob("*/best.pt"))
    if len(checkpoints) != 4:
        raise AssertionError(f"expected 3 direction and 1 magnitude checkpoint, got {len(checkpoints)}")
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["cache_manifest"] != manifest:
            raise AssertionError(f"checkpoint cache mismatch: {checkpoint}")
    result = {"source_sha_verified": len(manifest["symbols"]),
              "sampled_raw_target_checks": checks,
              "splits": {name: len(keys) for name, keys in store.splits.items()},
              "checkpoint_manifests_verified": len(checkpoints),
              "feature_finite_rate": {name: float(np.isfinite(cache[name]).mean())
                                      for name in ("fast", "slow", "state", "magnitude")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
