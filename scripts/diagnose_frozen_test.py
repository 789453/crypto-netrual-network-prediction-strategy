"""Post-test diagnostic correlations; never changes model or policy selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _block_interval(x: np.ndarray, y: np.ndarray, block: np.ndarray,
                    draws: np.ndarray) -> dict:
    count = int(block.max()) + 1
    aggregates = np.column_stack([
        np.bincount(block, weights=z, minlength=count)
        for z in (np.ones(len(x)), x, y, x * x, y * y, x * y)
    ])
    sampled = aggregates[draws].sum(axis=1)
    n, sx, sy, sxx, syy, sxy = sampled.T
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    correlations = covariance / np.sqrt(variance_x * variance_y)
    return {
        "correlation": _corr(x, y),
        "block_bootstrap_95pct_interval": np.quantile(correlations, [.025, .975]).tolist(),
        "positive_resample_fraction": float((correlations > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache_v1"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    selection = json.loads((args.evaluation / "selection.json").read_text(encoding="utf-8"))
    frozen_test = json.loads((args.evaluation / "final_test.json").read_text(encoding="utf-8"))
    names = (selection["trading_winner"], selection["predictive_leader"])
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    n_symbols = len(manifest["symbols"])
    target = np.load(args.cache / "targets.npy", mmap_mode="r")
    decision_time = np.load(args.cache / "decision_time.npy", mmap_mode="r")
    rng = np.random.default_rng(20261003)
    diagnostics = {
        "status": "post-test exploratory diagnostic; no model, threshold or policy change",
        "block_days": 7,
        "bootstrap_draws": 2000,
        "seed": 20261003,
        "models": {},
    }
    for name in dict.fromkeys(names):
        saved = np.load(args.evaluation / f"{name}_test_predictions.npz")
        keys = saved["keys"]
        hour, symbol = keys // n_symbols, keys % n_symbols
        dates = decision_time[hour].astype("datetime64[D]").astype(np.int64)
        block = ((dates - dates.min()) // 7).astype(np.int32)
        draws = rng.integers(0, int(block.max()) + 1, size=(2000, int(block.max()) + 1))
        observed_asym = np.asarray(target[hour, symbol, 0], dtype=np.float64)
        observed_execution = np.asarray(target[hour, symbol, 5], dtype=np.float64)
        fitted = selection["candidates"][name]["calibration"]
        mode = selection["candidates"][name]["mode"]
        prediction = saved["prediction"].astype(np.float64)
        predicted_asym = np.tanh(prediction[:, 0])
        if mode in {"asym4h", "logratio4h", "asym1h"}:
            features = predicted_asym[:, None]
        elif mode == "return4h":
            features = prediction[:, 2:3]
        else:
            features = np.column_stack((predicted_asym, prediction[:, 2], prediction[:, 1]))
        score = fitted["intercept"] + ((features - fitted["mean"]) / fitted["std"]) @ fitted["coef"]
        candidate = frozen_test["candidates"][name]
        result = {
            "predicted_asym_vs_observed_asym": _block_interval(predicted_asym, observed_asym, block, draws),
            "calibrated_score_vs_executable_return": _block_interval(score, observed_execution, block, draws),
            "observed_future_asym_vs_executable_return": _corr(observed_asym, observed_execution),
        }
        if abs(result["calibrated_score_vs_executable_return"]["correlation"] -
               candidate["calibrated_execution_correlation"]) > 1e-8:
            raise ValueError(f"frozen test correlation mismatch: {name}")
        diagnostics["models"][name] = result
    path = args.evaluation / "post_test_diagnostics.json"
    path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
