"""Point-in-time tabular baselines on the same hourly decisions and labels."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

from .training import PreparedStore


def snapshot_features(store: PreparedStore, keys: np.ndarray) -> np.ndarray:
    hour = keys // store.n_symbols
    symbol = keys % store.n_symbols
    end = hour * 12 + 11
    return np.concatenate((
        store.features["fast"][end, symbol],
        store.features["slow"][hour, symbol],
        store.features["stats"][hour, symbol],
        store.features["market"][hour, symbol],
        store.masks["fast"][end, symbol],
        store.masks["slow"][hour, symbol],
    ), axis=1).astype(np.float32)


def _target(store: PreparedStore, keys: np.ndarray, column: int) -> np.ndarray:
    hour = keys // store.n_symbols
    symbol = keys % store.n_symbols
    return store.targets[hour, symbol, column].astype(np.float32)


def fit_baselines(cache_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    store = PreparedStore(cache_root)
    train_keys = store.splits["train"]
    calibration_keys = store.splits["calibration"]
    validation_keys = store.splits["validation"]
    x_train = snapshot_features(store, train_keys)
    x_cal = snapshot_features(store, calibration_keys)
    x_val = snapshot_features(store, validation_keys)
    y_asym = _target(store, train_keys, 0)
    y_return = _target(store, train_keys, 3)
    y_cal_asym = _target(store, calibration_keys, 0)
    y_cal_return = _target(store, calibration_keys, 3)
    summary = {"train_rows": len(train_keys), "calibration_rows": len(calibration_keys),
               "validation_rows": len(validation_keys), "feature_count": x_train.shape[1]}
    ridge_asym = Ridge(alpha=1000).fit(x_train, y_asym)
    ridge_return = Ridge(alpha=1000).fit(x_train, y_return)
    ridge_pred = np.column_stack((
        np.arctanh(np.clip(ridge_asym.predict(x_val), -.98, .98)),
        np.zeros(len(x_val)), ridge_return.predict(x_val),
    )).astype(np.float32)
    np.savez_compressed(output_root / "ridge_validation_predictions.npz",
                        keys=validation_keys, prediction=ridge_pred)
    np.savez_compressed(output_root / "ridge_weights.npz",
                        asym_coef=ridge_asym.coef_, asym_intercept=ridge_asym.intercept_,
                        return_coef=ridge_return.coef_, return_intercept=ridge_return.intercept_)
    summary["ridge"] = {
        "calibration_asym_mse": float(np.mean(np.square(ridge_asym.predict(x_cal) - y_cal_asym))),
        "calibration_return_mse": float(np.mean(np.square(ridge_return.predict(x_cal) - y_cal_return))),
    }
    lgb_predictions = []
    for name, y_train, y_cal in (("asym", y_asym, y_cal_asym),
                                 ("return", y_return, y_cal_return)):
        train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
        cal_data = lgb.Dataset(x_cal, label=y_cal, reference=train_data, free_raw_data=False)
        model = lgb.train(
            {"objective": "regression", "metric": "l2", "learning_rate": 0.04,
             "num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.85,
             "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 10,
             "num_threads": 12, "verbosity": -1, "seed": 20261003},
            train_data, num_boost_round=700, valid_sets=[cal_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        model.save_model(str(output_root / f"lightgbm_{name}.txt"))
        lgb_predictions.append(model.predict(x_val, num_iteration=model.best_iteration))
        summary[f"lightgbm_{name}"] = {"best_iteration": model.best_iteration,
                                          "calibration_mse": float(model.best_score["valid_0"]["l2"])}
        print(f"LightGBM {name}: {summary[f'lightgbm_{name}']}", flush=True)
    lgb_pred = np.column_stack((
        np.arctanh(np.clip(lgb_predictions[0], -.98, .98)),
        np.zeros(len(x_val)), lgb_predictions[1],
    )).astype(np.float32)
    np.savez_compressed(output_root / "lightgbm_validation_predictions.npz",
                        keys=validation_keys, prediction=lgb_pred)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def predict_baseline(cache_root: Path, output_root: Path, split: str = "test") -> None:
    store = PreparedStore(cache_root)
    keys = store.splits[split]
    x = snapshot_features(store, keys)
    weights = np.load(output_root / "ridge_weights.npz")
    asym = x @ weights["asym_coef"] + weights["asym_intercept"]
    ret = x @ weights["return_coef"] + weights["return_intercept"]
    ridge_pred = np.column_stack((np.arctanh(np.clip(asym, -.98, .98)),
                                  np.zeros(len(x)), ret)).astype(np.float32)
    np.savez_compressed(output_root / f"ridge_{split}_predictions.npz", keys=keys, prediction=ridge_pred)
    lgb_asym = lgb.Booster(model_file=str(output_root / "lightgbm_asym.txt"))
    lgb_return = lgb.Booster(model_file=str(output_root / "lightgbm_return.txt"))
    asym = lgb_asym.predict(x, num_iteration=lgb_asym.best_iteration)
    ret = lgb_return.predict(x, num_iteration=lgb_return.best_iteration)
    lgb_pred = np.column_stack((np.arctanh(np.clip(asym, -.98, .98)),
                                np.zeros(len(x)), ret)).astype(np.float32)
    np.savez_compressed(output_root / f"lightgbm_{split}_predictions.npz", keys=keys, prediction=lgb_pred)
