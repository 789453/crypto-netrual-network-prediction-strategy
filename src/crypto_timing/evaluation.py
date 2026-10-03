"""Frozen validation selection and four-phase non-overlapping 4h backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .cache import load_cache
from .training import CALIBRATION_END, split_indices

SIDE_COST = 0.0006  # Scenario assumption: 4bp fee + 2bp slippage per side.
THRESHOLD_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 1.5)


def _prediction_features(prediction: np.ndarray, mode: str) -> np.ndarray:
    asym = np.tanh(prediction[:, 0])
    if mode in {"asym4h", "logratio4h", "asym1h"}:
        return asym[:, None]
    if mode == "return4h":
        return prediction[:, 2:3]
    if mode == "joint4h":
        return np.column_stack((asym, prediction[:, 2], prediction[:, 1]))
    if mode == "tabular":
        return np.column_stack((asym, prediction[:, 2]))
    if mode in {"momentum", "historical_asym"}:
        return prediction[:, :1]
    raise ValueError(mode)


def _fit_calibrator(x: np.ndarray, y: np.ndarray) -> dict:
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-5)
    model = Ridge(alpha=100.0)
    model.fit((x - mean) / std, y)
    return {"mean": mean.tolist(), "std": std.tolist(),
            "coef": model.coef_.tolist(), "intercept": float(model.intercept_)}


def _apply_calibrator(x: np.ndarray, fit: dict) -> np.ndarray:
    return np.asarray(fit["intercept"] + ((x - fit["mean"]) / fit["std"]) @ fit["coef"],
                      dtype=np.float64)


def _funding_per_hour(cache: dict, source_root: Path) -> np.ndarray:
    dates = np.asarray(cache["decision_time"])
    symbols = cache["manifest"]["symbols"]
    result = np.zeros((len(dates), len(symbols)), np.float32)
    entry = dates + np.timedelta64(5, "m")
    exit_time = entry + np.timedelta64(4, "h")
    for si, symbol in enumerate(symbols):
        frame = pd.read_parquet(source_root / "derivatives" / f"{symbol}.parquet",
                                columns=["date", "funding_rate"])
        frame = frame.loc[frame["funding_rate"].notna()].sort_values("date")
        event = frame["date"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        rate = frame["funding_rate"].to_numpy(np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(rate)))
        lower = np.searchsorted(event, entry, side="right")
        upper = np.searchsorted(event, exit_time, side="right")
        result[:, si] = (cumulative[upper] - cumulative[lower]).astype(np.float32)
    return result


def _candidate_predictions(cache: dict, run_root: Path, baseline_root: Path,
                           validation_keys: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    result: dict[str, tuple[np.ndarray, str]] = {}
    for run in sorted(run_root.glob("*/validation_predictions.npz")):
        data = np.load(run)
        if not np.array_equal(data["keys"], validation_keys):
            raise ValueError(f"validation keys differ: {run}")
        summary = json.loads((run.parent / "summary.json").read_text(encoding="utf-8"))
        result[run.parent.name] = (data["prediction"], summary["mode"])
    for name in ("ridge", "lightgbm"):
        data = np.load(baseline_root / f"{name}_validation_predictions.npz")
        if not np.array_equal(data["keys"], validation_keys):
            raise ValueError(f"validation keys differ: {name}")
        result[name] = (data["prediction"], "tabular")
    # Transparent rule baselines use the same historical hourly snapshot.
    hour = validation_keys // len(cache["manifest"]["symbols"])
    symbol = validation_keys % len(cache["manifest"]["symbols"])
    for name, col, mode in (("momentum4h", 1, "momentum"),
                            ("historical_asym4h", 5, "historical_asym")):
        value = cache["slow"][hour, symbol, col]
        prediction = np.zeros((len(value), 3), np.float32)
        prediction[:, 0] = value
        result[name] = (prediction, mode)
    return result


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _phase_metrics(keys: np.ndarray, score: np.ndarray, execution_return: np.ndarray,
                   funding: np.ndarray, dates: np.ndarray, n_symbols: int,
                   threshold: float | None, *, fixed_long: bool = False,
                   side_cost: float = SIDE_COST) -> dict:
    hour = keys // n_symbols
    phases = []
    for phase in range(4):
        phase_mask = dates[hour].astype("datetime64[h]").astype(np.int64) % 4 == phase
        phase_keys = keys[phase_mask]
        if not len(phase_keys):
            continue
        phase_hour = phase_keys // n_symbols
        phase_symbol = phase_keys % n_symbols
        if len(phase_keys) % n_symbols or not np.array_equal(
            phase_symbol.reshape(-1, n_symbols), np.broadcast_to(np.arange(n_symbols),
                                                                 (len(phase_keys) // n_symbols, n_symbols))
        ):
            raise ValueError("phase has incomplete symbol panel")
        phase_score = score[phase_mask].reshape(-1, n_symbols)
        if fixed_long:
            position = np.ones_like(phase_score)
        elif threshold is None:
            position = np.zeros_like(phase_score)
        else:
            position = np.where(phase_score > threshold, 1.0,
                                np.where(phase_score < -threshold, -1.0, 0.0))
        previous = np.vstack((np.zeros((1, n_symbols)), position[:-1]))
        turnover = np.abs(position - previous)
        gross = position * execution_return[phase_mask].reshape(-1, n_symbols)
        funding_pnl = -position * funding[phase_hour, phase_symbol].reshape(-1, n_symbols)
        net = (gross + funding_pnl - side_cost * turnover).mean(axis=1)
        net[-1] -= side_cost * np.abs(position[-1]).mean()
        equity = np.cumprod(1 + net)
        peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
        annual_factor = 365 * 6
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(annual_factor)) if net.std() > 0 else 0.0
        phases.append({
            "phase": phase, "periods": len(net), "sharpe": sharpe,
            "annualized_log_return": float(np.log1p(net).mean() * annual_factor),
            "cumulative_return": float(equity[-1] - 1),
            "max_drawdown": float(np.min(equity / peak - 1)),
            "active_fraction": float((position != 0).mean()),
            "mean_turnover": float(turnover.mean()),
            "funding_log_contribution": float(funding_pnl.mean() * annual_factor),
            "cost_log_contribution": float(-side_cost * turnover.mean() * annual_factor),
        })
    return {"phases": phases, "mean_sharpe": float(np.mean([p["sharpe"] for p in phases])),
            "min_sharpe": float(np.min([p["sharpe"] for p in phases])),
            "mean_annualized_log_return": float(np.mean([p["annualized_log_return"] for p in phases])),
            "mean_active_fraction": float(np.mean([p["active_fraction"] for p in phases]))}


def _score_candidate(cache: dict, keys: np.ndarray, prediction: np.ndarray, mode: str,
                     funding: np.ndarray) -> tuple[dict, np.ndarray]:
    n_symbols = len(cache["manifest"]["symbols"])
    hour = keys // n_symbols
    symbol = keys % n_symbols
    dates = np.asarray(cache["decision_time"])
    target = cache["targets"][hour, symbol]
    calibration = dates[hour] < CALIBRATION_END
    selection = dates[hour] >= CALIBRATION_END
    features = _prediction_features(prediction, mode)
    fit = _fit_calibrator(features[calibration], target[calibration, 5])
    expected_return = _apply_calibrator(features, fit)
    predictive = {
        "asym_correlation_selection": _pearson(np.tanh(prediction[selection, 0]), target[selection, 0]),
        "asym1h_correlation_selection": _pearson(np.tanh(prediction[selection, 0]), target[selection, 6]),
        "risk_return_correlation_selection": _pearson(prediction[selection, 2], target[selection, 3]),
        "calibrated_execution_correlation_selection": _pearson(expected_return[selection], target[selection, 5]),
    }
    options = {}
    for mult in THRESHOLD_MULTIPLIERS:
        threshold = mult * SIDE_COST * 2
        metrics = _phase_metrics(keys[selection], expected_return[selection], target[selection, 5],
                                 funding, dates, n_symbols, threshold)
        options[str(mult)] = metrics
    # Validation selection requires some actual participation in all phase accounts.
    eligible = [(mult, result) for mult, result in options.items()
                if result["mean_active_fraction"] >= 0.05]
    best = max(eligible, key=lambda item: (item[1]["mean_sharpe"],
                                         item[1]["mean_annualized_log_return"])) if eligible else None
    selected_mult = float(best[0]) if best is not None else None
    record = {"mode": mode, "calibration": fit, "predictive": predictive,
              "threshold_options": options, "selected_multiplier": selected_mult,
              "selected_metrics": best[1] if best else None}
    return record, expected_return


def select(cache_root: Path, run_root: Path, baseline_root: Path, source_root: Path,
           output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_root)
    splits = split_indices(cache["decision_time"], cache["targets"])
    keys = splits["validation"]
    funding = _funding_per_hour(cache, source_root)
    candidates = _candidate_predictions(cache, run_root, baseline_root, keys)
    if not candidates:
        raise RuntimeError("no completed validation candidates")
    report = {"assumed_side_cost": SIDE_COST, "calibration_period": "2025-07-01 to 2025-10-31",
              "selection_period": "2025-11-01 to 2026-01-31", "candidates": {}}
    for name, (prediction, mode) in candidates.items():
        record, _ = _score_candidate(cache, keys, prediction, mode, funding)
        report["candidates"][name] = record
        print(f"{name}: score_corr={record['predictive']['calibrated_execution_correlation_selection']:.5f} "
              f"threshold={record['selected_multiplier']} "
              f"sharpe={record['selected_metrics']['mean_sharpe'] if record['selected_metrics'] else None}",
              flush=True)
    selection_time = np.asarray(cache["decision_time"])[keys // len(cache["manifest"]["symbols"])] >= CALIBRATION_END
    selected_keys = keys[selection_time]
    selected_target = cache["targets"][selected_keys // len(cache["manifest"]["symbols"]),
                                      selected_keys % len(cache["manifest"]["symbols"]), 5]
    report["fixed_long_selection"] = _phase_metrics(
        selected_keys, np.zeros(len(selected_keys)), selected_target, funding,
        cache["decision_time"], len(cache["manifest"]["symbols"]), 0, fixed_long=True)
    # Prespecified winner: most robust phase-average net Sharpe among active policies;
    # if none beats doing nothing, deploy no-trade and still retain the predictive leader.
    ranked = [(name, rec) for name, rec in report["candidates"].items()
              if rec["selected_metrics"] is not None]
    ranked.sort(key=lambda item: (item[1]["selected_metrics"]["mean_sharpe"],
                                  item[1]["selected_metrics"]["min_sharpe"]), reverse=True)
    predictive_leader = max(report["candidates"], key=lambda name: report["candidates"][name]["predictive"]
                            ["calibrated_execution_correlation_selection"])
    report["predictive_leader"] = predictive_leader
    report["trading_winner"] = ranked[0][0] if ranked and ranked[0][1]["selected_metrics"]["mean_sharpe"] > 0 else None
    (output_root / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n",
                                                   encoding="utf-8")
    return report
