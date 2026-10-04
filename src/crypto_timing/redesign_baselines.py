"""Train-only conditional-sign, magnitude and direct-return reference models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from .redesign_training import RedesignStore, _constant_direction_prior, month_direction_scores


def _snapshot(store: RedesignStore, keys: np.ndarray, name: str) -> np.ndarray:
    hour, symbol = keys // store.n_symbols, keys % store.n_symbols
    return np.column_stack((store.values[name][hour, symbol], store.masks[name][hour, symbol]))


def fit_redesign_baselines(cache_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    store = RedesignStore(cache_root)
    train = store.splits["train"]
    labels = np.asarray(store.targets[train // store.n_symbols, train % store.n_symbols])
    band, up, scaled = labels[:, 2].astype(int), labels[:, 3], labels[:, 1]
    x_state = _snapshot(store, train, "state")
    x_mag = _snapshot(store, train, "magnitude")
    priors = _constant_direction_prior(store)
    direction = []
    for j in range(4):
        keep = (band == j) & (up != .5)
        clf = LogisticRegression(C=.1, max_iter=200, solver="lbfgs")
        clf.fit(x_state[keep], up[keep].astype(int))
        direction.append(clf)
        print(f"baseline direction band {j}: {int(keep.sum())} rows", flush=True)
    magnitude = LogisticRegression(C=.1, max_iter=200, solver="lbfgs")
    magnitude.fit(x_mag, band)
    direct = Ridge(alpha=1000.0).fit(x_state, scaled)
    joblib.dump({"direction": direction, "magnitude": magnitude, "direct": direct,
                 "manifest": store.manifest, "scalers": store.scalers}, output_root / "baselines.joblib")
    monthly = {}
    for split in ("calibration", "selection", "historical"):
        keys = store.splits[split]
        xs = _snapshot(store, keys, "state")
        xm = _snapshot(store, keys, "magnitude")
        prob = np.column_stack([model.predict_proba(xs)[:, 1] for model in direction])
        logits = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1 - prob, 1e-6, 1))
        mag_prob = magnitude.predict_proba(xm)
        mag_logits = np.log(np.clip(mag_prob, 1e-8, 1))
        direct_scaled = direct.predict(xs)
        np.savez_compressed(output_root / f"{split}_predictions.npz", keys=keys,
                            direction_logits=logits.astype(np.float32),
                            magnitude_logits=mag_logits.astype(np.float32),
                            direct_scaled=direct_scaled.astype(np.float32))
        monthly[split] = month_direction_scores(logits, keys, store, priors)
    summary = {"train_rows": len(train), "feature_state": x_state.shape[1],
               "feature_magnitude": x_mag.shape[1], "direction_prior": priors.tolist(),
               "monthly": monthly, "cache_manifest": store.manifest}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("redesign baselines complete", flush=True)
    return summary
