"""One-shot frozen test scoring. Selection file is read, never rewritten."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .baselines import predict_baseline
from .cache import load_cache
from .evaluation import (_apply_calibrator, _funding_per_hour, _pearson,
                         _phase_metrics, _prediction_features, SIDE_COST)
from .model import CryptoTimingNetwork, ModelConfig
from .training import PreparedStore, predict, split_indices


def _network_prediction(cache_root: Path, run: Path, keys: np.ndarray) -> np.ndarray:
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    store = PreparedStore(cache_root, scalers=saved["scalers"])
    config = ModelConfig(**saved["model_config"])
    model = CryptoTimingNetwork(config)
    model.load_state_dict(saved["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return predict(model, store, keys, device, batch_size=256)


def run_final_test(cache_root: Path, run_root: Path, baseline_root: Path,
                   source_root: Path, selection_path: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cache = load_cache(cache_root)
    dates = np.asarray(cache["decision_time"])
    symbols = cache["manifest"]["symbols"]
    keys = split_indices(dates, cache["targets"])["test"]
    n_symbols = len(symbols)
    hour, symbol = keys // n_symbols, keys % n_symbols
    target = cache["targets"][hour, symbol]
    funding = _funding_per_hour(cache, source_root)
    report = {"selection_file": str(selection_path),
              "test_first_decision": str(dates[hour[0]]),
              "test_last_decision": str(dates[hour[-1]]),
              "test_rows": len(keys), "assumed_side_cost": SIDE_COST,
              "trading_winner_frozen": selection["trading_winner"],
              "predictive_leader_frozen": selection["predictive_leader"],
              "candidates": {}}
    predict_baseline(cache_root, baseline_root, "test")
    for name, frozen in selection["candidates"].items():
        if name in {"ridge", "lightgbm"}:
            saved = np.load(baseline_root / f"{name}_test_predictions.npz")
            if not np.array_equal(saved["keys"], keys):
                raise ValueError(f"test keys differ: {name}")
            pred = saved["prediction"]
        elif name == "momentum4h" or name == "historical_asym4h":
            col = 1 if name == "momentum4h" else 5
            pred = np.zeros((len(keys), 3), np.float32)
            pred[:, 0] = cache["slow"][hour, symbol, col]
        else:
            pred = _network_prediction(cache_root, run_root / name, keys)
            np.savez_compressed(output_root / f"{name}_test_predictions.npz",
                                keys=keys, prediction=pred)
        features = _prediction_features(pred, frozen["mode"])
        score = _apply_calibrator(features, frozen["calibration"])
        mult = frozen["selected_multiplier"]
        threshold = None if mult is None else mult * 2 * SIDE_COST
        metrics = _phase_metrics(keys, score, target[:, 5], funding, dates, n_symbols, threshold)
        costs = {}
        for cost in (0.0, 0.0001, 0.0003, 0.0006, 0.0010):
            costs[f"{cost:.4f}"] = _phase_metrics(
                keys, score, target[:, 5], funding, dates, n_symbols, threshold,
                side_cost=cost)
        report["candidates"][name] = {
            "mode": frozen["mode"], "selected_multiplier": mult,
            "asym4h_correlation": _pearson(np.tanh(pred[:, 0]), target[:, 0]),
            "asym1h_correlation": _pearson(np.tanh(pred[:, 0]), target[:, 6]),
            "risk_return_correlation": _pearson(pred[:, 2], target[:, 3]),
            "calibrated_execution_correlation": _pearson(score, target[:, 5]),
            "costed_policy": metrics, "cost_sensitivity": costs,
        }
        print(f"test {name}: corr={report['candidates'][name]['calibrated_execution_correlation']:.5f} "
              f"net_sharpe={metrics['mean_sharpe']:.3f}", flush=True)
    report["fixed_long"] = _phase_metrics(
        keys, np.zeros(len(keys)), target[:, 5], funding, dates, n_symbols, 0, fixed_long=True)
    report["no_trade"] = _phase_metrics(
        keys, np.zeros(len(keys)), target[:, 5], funding, dates, n_symbols, None)
    (output_root / "final_test.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n",
                                                   encoding="utf-8")
    return report
