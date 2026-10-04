"""One-parameter probability shrinkage fitted on the early-stop period only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .redesign_evaluation import (_direction_metrics, _labels, _load_prediction,
                                  _log_softmax, _magnitude_metrics, _pearson, _readout)
from .redesign_training import RedesignStore


def _fit_shrinkage(logits: np.ndarray, labels: np.ndarray,
                   prior_logit: np.ndarray, kind: str) -> tuple[float, float, float]:
    row = np.arange(len(labels))
    band = labels[:, 2].astype(int)
    up = labels[:, 3]
    options = np.linspace(0, 1.5, 301)
    losses = np.empty(len(options))
    for i, value in enumerate(options):
        adjusted = prior_logit[None, :] + value * (logits - prior_logit[None, :])
        if kind == "direction":
            selected = adjusted[row, band]
            losses[i] = (np.logaddexp(0, selected) - up * selected).mean()
        else:
            losses[i] = -_log_softmax(adjusted)[row, band].mean()
    best = int(np.argmin(losses))
    raw_index = int(np.argmin(np.abs(options - 1)))
    return float(options[best]), float(losses[best]), float(losses[raw_index])


def calibration_diagnostic(cache_root: Path, baseline_root: Path, magnitude_root: Path,
                           direction_root: Path, selection_path: Path, output_path: Path) -> dict:
    store = RedesignStore(cache_root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["cache_manifest"] != store.manifest:
        raise ValueError("calibration belongs to a different data cache")
    prior = np.asarray(selection["direction_prior"])
    band_prior = np.asarray(selection["representatives"]["band_prior"])
    direction_prior = np.log(prior / (1 - prior))
    magnitude_prior = np.log(band_prior)
    cal_keys = store.splits["calibration"]
    cal_label = _labels(store, cal_keys)
    d_cal = _load_prediction(direction_root / "calibration_predictions.npz", cal_keys)["logits"]
    q_cal = _load_prediction(magnitude_root / "calibration_predictions.npz", cal_keys)["logits"]
    direction_lambda, fitted_dir_ce, raw_dir_ce = _fit_shrinkage(
        d_cal, cal_label, direction_prior, "direction")
    magnitude_lambda, fitted_mag_ce, raw_mag_ce = _fit_shrinkage(
        q_cal, cal_label, magnitude_prior, "magnitude")
    result = {"protocol": "diagnostic only; two shared shrinkage coefficients fitted on Jul-Oct 2025",
              "direction_shrinkage": direction_lambda, "magnitude_shrinkage": magnitude_lambda,
              "calibration_fitted_direction_ce": fitted_dir_ce,
              "calibration_raw_direction_ce": raw_dir_ce,
              "calibration_fitted_magnitude_ce": fitted_mag_ce,
              "calibration_raw_magnitude_ce": raw_mag_ce,
              "periods": {}}
    reps = np.asarray(selection["representatives"]["values"])
    for split in ("calibration", "selection", "historical"):
        keys = store.splits[split]
        label = _labels(store, keys)
        baseline = _load_prediction(baseline_root / f"{split}_predictions.npz", keys)
        d = _load_prediction(direction_root / f"{split}_predictions.npz", keys)["logits"]
        q = _load_prediction(magnitude_root / f"{split}_predictions.npz", keys)["logits"]
        d_adjusted = direction_prior[None, :] + direction_lambda * (d - direction_prior[None, :])
        q_adjusted = magnitude_prior[None, :] + magnitude_lambda * (q - magnitude_prior[None, :])
        raw_score = _readout(d, q, label[:, 4], reps)
        adjusted_score = _readout(d_adjusted, q_adjusted, label[:, 4], reps)
        result["periods"][split] = {
            "raw_direction": _direction_metrics(d, keys, store, prior,
                                                 baseline["direction_logits"])["overall"],
            "adjusted_direction": _direction_metrics(d_adjusted, keys, store, prior,
                                                      baseline["direction_logits"])["overall"],
            "raw_magnitude": _magnitude_metrics(q, keys, store,
                                                baseline["magnitude_logits"], band_prior)["overall"],
            "adjusted_magnitude": _magnitude_metrics(q_adjusted, keys, store,
                                                     baseline["magnitude_logits"], band_prior)["overall"],
            "raw_score_return_correlation": _pearson(raw_score, label[:, 0]),
            "adjusted_score_return_correlation": _pearson(adjusted_score, label[:, 0]),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
