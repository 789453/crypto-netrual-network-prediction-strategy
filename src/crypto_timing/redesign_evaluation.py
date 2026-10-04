"""Frozen selection and explicitly exploratory history for the v3 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import SIDE_COST, _funding_per_hour, _phase_metrics
from .redesign_cache import load_redesign_cache
from .redesign_training import RedesignStore, _constant_direction_prior

THRESHOLDS = (0.0, 0.0006, 0.0012, 0.0018)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def _bce(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, logits) - target * logits


def _load_prediction(path: Path, keys: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        if not np.array_equal(saved["keys"], keys):
            raise ValueError(f"prediction keys differ: {path}")
        return {name: np.asarray(saved[name]) for name in saved.files if name != "keys"}


def _labels(store: RedesignStore, keys: np.ndarray) -> np.ndarray:
    return np.asarray(store.targets[keys // store.n_symbols, keys % store.n_symbols])


def _direction_metrics(logits: np.ndarray, keys: np.ndarray, store: RedesignStore,
                       prior: np.ndarray, linear: np.ndarray) -> dict:
    label = _labels(store, keys)
    band, up = label[:, 2].astype(int), label[:, 3]
    idx = np.arange(len(keys))
    loss = _bce(logits[idx, band], up)
    linear_loss = _bce(linear[idx, band], up)
    prior_logit = np.log(prior[band] / (1 - prior[band]))
    constant_loss = _bce(prior_logit, up)
    month = store.dates[keys // store.n_symbols].astype("datetime64[M]")

    def summarize(mask: np.ndarray) -> dict:
        return {"rows": int(mask.sum()), "ce": float(loss[mask].mean()),
                "linear_ce": float(linear_loss[mask].mean()),
                "constant_ce": float(constant_loss[mask].mean()),
                "gain_vs_linear": float((linear_loss[mask] - loss[mask]).mean()),
                "gain_vs_constant": float((constant_loss[mask] - loss[mask]).mean())}

    return {"overall": summarize(np.ones(len(keys), bool)),
            "months": {str(m): summarize(month == m) for m in np.unique(month)},
            "bands": {str(j): summarize(band == j) for j in range(4)},
            "symbols": {symbol: summarize(keys % store.n_symbols == si)
                        for si, symbol in enumerate(store.manifest["symbols"])}}


def _magnitude_metrics(logits: np.ndarray, keys: np.ndarray, store: RedesignStore,
                       linear: np.ndarray, prior: np.ndarray) -> dict:
    band = _labels(store, keys)[:, 2].astype(int)
    idx = np.arange(len(keys))
    loss = -_log_softmax(logits)[idx, band]
    linear_loss = -_log_softmax(linear)[idx, band]
    constant_loss = -np.log(prior[band])
    month = store.dates[keys // store.n_symbols].astype("datetime64[M]")

    def summarize(mask: np.ndarray) -> dict:
        return {"rows": int(mask.sum()), "ce": float(loss[mask].mean()),
                "linear_ce": float(linear_loss[mask].mean()),
                "constant_ce": float(constant_loss[mask].mean())}

    return {"overall": summarize(np.ones(len(keys), bool)),
            "months": {str(m): summarize(month == m) for m in np.unique(month)}}


def _representatives(store: RedesignStore) -> dict:
    label = _labels(store, store.splits["train"])
    band, up, abs_y = label[:, 2].astype(int), label[:, 3], np.abs(label[:, 1])
    rep = np.empty((4, 2), np.float64)
    support = np.zeros((4, 2), int)
    tail_quantiles = {}
    for j in range(4):
        keep = band == j
        pooled = float(abs_y[keep].mean())
        if j == 3:
            tail_quantiles = {"p10": float(np.quantile(abs_y[keep], .1)),
                              "p90": float(np.quantile(abs_y[keep], .9))}
        for side, sign in enumerate((0, 1)):
            selected = keep & (up == sign)
            n = int(selected.sum())
            support[j, side] = n
            weight = 500 if j == 3 else 100
            rep[j, side] = (abs_y[selected].sum() + weight * pooled) / (n + weight)
    count = np.bincount(band, minlength=4)
    return {"values": rep.tolist(), "support": support.tolist(),
            "band_counts": count.tolist(), "band_prior": (count / count.sum()).tolist(),
            "tail_quantiles": tail_quantiles,
            "shrinkage_pseudo_rows": [100, 100, 100, 500]}


def _readout(direction_logits: np.ndarray, magnitude_logits: np.ndarray,
             scale: np.ndarray, representatives: np.ndarray,
             tail_multiplier: float = 1.0) -> np.ndarray:
    pi = 1 / (1 + np.exp(-np.clip(direction_logits.astype(np.float64), -30, 30)))
    q = np.exp(_log_softmax(magnitude_logits))
    rep = representatives.copy()
    rep[3] *= tail_multiplier
    return scale * np.sum(q * (pi * rep[:, 1] - (1 - pi) * rep[:, 0]), axis=1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")


def _score_metrics(keys: np.ndarray, score: np.ndarray, store: RedesignStore,
                   funding: np.ndarray, threshold: float | None,
                   *, fixed_long: bool = False, side_cost: float = SIDE_COST) -> dict:
    label = _labels(store, keys)
    target = label[:, 0].astype(np.float64)
    metrics = _phase_metrics(keys, score, np.log1p(target), funding, store.dates,
                             store.n_symbols, threshold, fixed_long=fixed_long,
                             side_cost=side_cost)
    metrics["score_return_correlation"] = _pearson(score, target)
    metrics["score_mean"] = float(score.mean())
    metrics["realized_return_mean"] = float(target.mean())
    metrics["score_std"] = float(score.std())
    month = store.dates[keys // store.n_symbols].astype("datetime64[M]")
    metrics["monthly_correlation"] = {
        str(m): _pearson(score[month == m], target[month == m]) for m in np.unique(month)
    }
    by_symbol = {}
    for si, symbol in enumerate(store.manifest["symbols"]):
        keep = keys % store.n_symbols == si
        by_symbol[symbol] = _pearson(score[keep], target[keep])
    metrics["per_symbol_correlation"] = by_symbol
    if len(keys) % store.n_symbols == 0 and np.array_equal(
        keys.reshape(-1, store.n_symbols) % store.n_symbols,
        np.broadcast_to(np.arange(store.n_symbols), (len(keys) // store.n_symbols, store.n_symbols))
    ):
        sx = score.reshape(-1, store.n_symbols)
        sy = target.reshape(-1, store.n_symbols)
        metrics["market_time_correlation"] = _pearson(sx.mean(axis=1), sy.mean(axis=1))
        metrics["within_hour_correlation"] = _pearson(
            (sx - sx.mean(axis=1, keepdims=True)).ravel(),
            (sy - sy.mean(axis=1, keepdims=True)).ravel())
    rank = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort")
    decile = np.minimum(9, rank * 10 // len(rank))
    metrics["deciles"] = [{"n": int((decile == j).sum()),
                           "score": float(score[decile == j].mean()),
                           "realized_return": float(target[decile == j].mean())}
                          for j in range(10)]
    return metrics


def _inputs(store: RedesignStore, baseline_root: Path, magnitude_root: Path,
            direction_roots: list[Path], split: str) -> tuple[np.ndarray, dict, dict, dict]:
    keys = store.splits[split]
    base = _load_prediction(baseline_root / f"{split}_predictions.npz", keys)
    mag = _load_prediction(magnitude_root / f"{split}_predictions.npz", keys)
    directions = {root.name: _load_prediction(root / f"{split}_predictions.npz", keys)
                  for root in direction_roots}
    return keys, base, mag, directions


def select_redesign(cache_root: Path, baseline_root: Path, magnitude_root: Path,
                    direction_roots: list[Path], source_root: Path, output_root: Path) -> dict:
    """Use only July 2025 through January 2026 and freeze the choice."""
    output_root.mkdir(parents=True, exist_ok=True)
    store = RedesignStore(cache_root)
    cache = load_redesign_cache(cache_root)
    funding = _funding_per_hour(cache, source_root)
    prior = _constant_direction_prior(store)
    reps = _representatives(store)
    rep = np.asarray(reps["values"])
    data = {}
    for split in ("calibration", "selection"):
        keys, base, mag, directions = _inputs(store, baseline_root, magnitude_root,
                                               direction_roots, split)
        label = _labels(store, keys)
        q_metrics = _magnitude_metrics(mag["logits"], keys, store,
                                       base["magnitude_logits"], np.asarray(reps["band_prior"]))
        data[split] = {"keys": keys, "base": base, "mag": mag,
                       "directions": directions, "labels": label,
                       "magnitude_metrics": q_metrics}
    cal = data["calibration"]
    sel = data["selection"]
    candidates = {}
    for name in sorted(cal["directions"]):
        cal_metrics = _direction_metrics(cal["directions"][name]["logits"], cal["keys"],
                                         store, prior, cal["base"]["direction_logits"])
        sel_metrics = _direction_metrics(sel["directions"][name]["logits"], sel["keys"],
                                         store, prior, sel["base"]["direction_logits"])
        candidates[name] = {"calibration": cal_metrics, "selection": sel_metrics,
                            "best_epoch": json.loads((next(root for root in direction_roots if root.name == name)
                                                       / "summary.json").read_text(encoding="utf-8"))["best_epoch"]}
    leader = max(candidates, key=lambda name: candidates[name]["calibration"]["overall"]["gain_vs_linear"])
    leader_gain = candidates[leader]["calibration"]["overall"]["gain_vs_linear"]
    leader_months = candidates[leader]["calibration"]["months"]
    qualifies = leader_gain > 0 and sum(m["gain_vs_linear"] > 0 for m in leader_months.values()) >= 2
    without_best_month = float(np.mean(sorted(m["gain_vs_linear"] for m in leader_months.values())[:-1]))
    score_candidates = {}
    for split, bundle in data.items():
        label = bundle["labels"]
        v = label[:, 4].astype(np.float64)
        base = bundle["base"]
        q = bundle["mag"]["logits"]
        p_nn = bundle["directions"][leader]["logits"]
        score_candidates[split] = {
            "network": _readout(p_nn, q, v, rep),
            "linear_direction": _readout(base["direction_logits"], q, v, rep),
            "linear_both": _readout(base["direction_logits"], base["magnitude_logits"], v, rep),
            "direct_ridge": v * base["direct_scaled"],
        }
    options = {}
    for name in score_candidates["selection"]:
        options[name] = {str(thr): _score_metrics(sel["keys"], score_candidates["selection"][name],
                                                   store, funding, thr)
                         for thr in THRESHOLDS}
    eligible = []
    for name, record in options.items():
        for thr in THRESHOLDS:
            outcome = record[str(thr)]
            if outcome["mean_active_fraction"] >= .05:
                eligible.append((name, thr, outcome))
    economic_best = max(eligible, key=lambda row: (row[2]["mean_sharpe"],
                                                   row[2]["min_sharpe"])) if eligible else None
    winner = (economic_best[0] if economic_best and economic_best[2]["mean_sharpe"] > 0
              else None)
    threshold = economic_best[1] if winner else None
    calibration_scores = {name: {"return_correlation": _pearson(score, cal["labels"][:, 0])}
                          for name, score in score_candidates["calibration"].items()}
    result = {"protocol": "selection only; 2026-02 onward was seen in earlier project runs and is exploratory",
              "assumed_side_cost": SIDE_COST, "thresholds": THRESHOLDS,
              "cache_manifest": store.manifest, "direction_prior": prior.tolist(),
              "representatives": reps, "magnitude_calibration": cal["magnitude_metrics"],
              "magnitude_selection": sel["magnitude_metrics"],
              "direction_candidates": candidates, "direction_leader": leader,
              "direction_qualifies": bool(qualifies),
              "direction_gain_without_best_month": without_best_month,
              "calibration_scores": calibration_scores,
              "selection_options": options,
              "economic_winner": winner, "economic_threshold": threshold,
              "selection_fixed_long": _score_metrics(sel["keys"], np.zeros(len(sel["keys"])),
                                                     store, funding, 0, fixed_long=True),
              "research_decision": ("no_deploy" if not (qualifies and winner == "network")
                                    else "research_candidate_only")}
    (output_root / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"direction_leader": leader, "direction_gain": leader_gain,
                      "direction_qualifies": qualifies, "economic_winner": winner,
                      "economic_threshold": threshold, "research_decision": result["research_decision"]}),
          flush=True)
    return result


def historical_diagnostic(cache_root: Path, baseline_root: Path, magnitude_root: Path,
                          direction_roots: list[Path], source_root: Path, output_root: Path) -> dict:
    """Read the frozen selection without using 2026 data to choose any setting."""
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    store = RedesignStore(cache_root)
    cache = load_redesign_cache(cache_root)
    if selection["cache_manifest"] != store.manifest:
        raise ValueError("selection belongs to a different cache")
    funding = _funding_per_hour(cache, source_root)
    keys, base, mag, directions = _inputs(store, baseline_root, magnitude_root,
                                           direction_roots, "historical")
    reps = np.asarray(selection["representatives"]["values"])
    v = _labels(store, keys)[:, 4].astype(np.float64)
    leader = selection["direction_leader"]
    scores = {"network": _readout(directions[leader]["logits"], mag["logits"], v, reps),
              "linear_direction": _readout(base["direction_logits"], mag["logits"], v, reps),
              "linear_both": _readout(base["direction_logits"], base["magnitude_logits"], v, reps),
              "direct_ridge": v * base["direct_scaled"]}
    result = {"status": "exploratory historical diagnostic; period previously inspected",
              "direction": {name: _direction_metrics(pred["logits"], keys, store,
                                                     np.asarray(selection["direction_prior"]),
                                                     base["direction_logits"])
                            for name, pred in directions.items()},
              "magnitude": _magnitude_metrics(mag["logits"], keys, store,
                                                base["magnitude_logits"],
                                                np.asarray(selection["representatives"]["band_prior"])),
              "scores": {}, "frozen_winner": selection["economic_winner"],
              "frozen_threshold": selection["economic_threshold"]}
    for name, score in scores.items():
        threshold = selection["economic_threshold"] if name == selection["economic_winner"] else .0012
        result["scores"][name] = _score_metrics(keys, score, store, funding, threshold)
    result["fixed_long"] = _score_metrics(keys, np.zeros(len(keys)), store, funding, 0,
                                          fixed_long=True)
    if selection["economic_winner"]:
        winner = selection["economic_winner"]
        result["tail_sensitivity"] = {}
        for mult in (.8, 1.0, 1.2):
            if winner == "network":
                altered = _readout(directions[leader]["logits"], mag["logits"], v, reps, mult)
            elif winner == "linear_direction":
                altered = _readout(base["direction_logits"], mag["logits"], v, reps, mult)
            elif winner == "linear_both":
                altered = _readout(base["direction_logits"], base["magnitude_logits"], v, reps, mult)
            else:
                continue
            result["tail_sensitivity"][str(mult)] = _score_metrics(keys, altered, store, funding,
                                                                     selection["economic_threshold"])
    result["zero_cost_diagnostic"] = {}
    for split, bundle in (("calibration", None), ("selection", None), ("historical", None)):
        if split == "historical":
            pred_scores = scores
            pred_keys = keys
        else:
            pred_keys, b, m, d = _inputs(store, baseline_root, magnitude_root, direction_roots, split)
            scale = _labels(store, pred_keys)[:, 4].astype(np.float64)
            pred_scores = {"network": _readout(d[leader]["logits"], m["logits"], scale, reps),
                           "linear_direction": _readout(b["direction_logits"], m["logits"], scale, reps),
                           "linear_both": _readout(b["direction_logits"], b["magnitude_logits"], scale, reps),
                           "direct_ridge": scale * b["direct_scaled"]}
        np.savez_compressed(output_root / f"{split}_scores.npz", keys=pred_keys, **pred_scores)
        if split in {"selection", "historical"}:
            result["zero_cost_diagnostic"][split] = {
                name: _score_metrics(pred_keys, pred_scores[name], store, funding,
                                     .0012 if name == "network" else selection["economic_threshold"],
                                     side_cost=0.0)
                for name in {"network", selection["economic_winner"]} if name is not None
            }
    (output_root / "historical_diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n",
                                                               encoding="utf-8")
    print(json.dumps({"historical_network_mean_sharpe": result["scores"]["network"]["mean_sharpe"],
                      "frozen_winner": result["frozen_winner"]}), flush=True)
    return result
