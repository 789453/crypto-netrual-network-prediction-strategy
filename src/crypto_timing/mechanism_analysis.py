"""Training-only redundancy, time-block uncertainty and level-signal diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from .mechanism_data import FAST_NAMES, SELECTED, BOUNDED
from .mechanism_model import CausalLevelFilter, MechanismConfig, MechanismNetwork
from .mechanism_training import FOLDS, MechanismStore, correlation, fit_scaler, infer, metrics, normalize, scaled_labels


def feature_audit(root: Path, fold: str) -> dict:
    dates = np.load(root / "decision_time.npy")
    stop = int(np.searchsorted(dates, np.datetime64(FOLDS[fold][0]))) - 9
    arr = np.load(root / "fast_dynamic.npy", mmap_mode="r")
    center, scale = fit_scaler(arr, stop * 12, BOUNDED["fast"])
    idx = np.arange(720 * 12, stop * 12 - 1, 13)  # all twelve 5m phases, deterministic sample
    x, m = normalize(arr[idx], center, scale)
    z, _ = normalize(arr[idx + 1], center, scale)
    valid = m.all(-1)
    x = x.astype(np.float64)[valid]
    z = z.astype(np.float64)[valid]
    active = x.std(0) > 1e-6
    corr = np.zeros((20,20))
    spearman = np.zeros_like(corr)
    lag_corr = np.zeros_like(corr)
    corr[np.ix_(active, active)] = np.corrcoef(x[:, active], rowvar=False)
    ranks = np.column_stack([rankdata(x[:, j]) for j in np.flatnonzero(active)])
    spearman[np.ix_(active, active)] = np.corrcoef(ranks, rowvar=False)
    zn = np.nan_to_num((z - z.mean(0)) / np.maximum(z.std(0), 1e-8))
    xn = (x - x.mean(0)) / np.maximum(x.std(0), 1e-8)
    lag_corr = xn.T @ zn / len(x)
    pairs = []
    for a in range(20):
        for b in range(a+1,20):
            if abs(corr[a,b]) > .65:
                pairs.append({"a": FAST_NAMES[a], "b": FAST_NAMES[b], "pearson": float(corr[a,b]),
                              "spearman": float(spearman[a,b]), "lag5m": float(lag_corr[a,b])})
    pairs.sort(key=lambda row: -abs(row["pearson"]))
    state = np.load(root / "state_dynamic.npy", mmap_mode="r")
    risk = np.asarray(state[idx // 12, :, 6])[valid]
    high = risk > np.nanmedian(risk)
    state_pairs = {}
    for row in pairs[:12]:
        a,b = FAST_NAMES.index(row["a"]), FAST_NAMES.index(row["b"])
        state_pairs[f'{row["a"]}/{row["b"]}'] = {"low_risk": correlation(x[~high,a],x[~high,b]),
                                                "high_risk": correlation(x[high,a],x[high,b])}
    raw = np.load(root / "targets.npy")
    floors = np.nanquantile(raw[720:stop, :, 7], .05, axis=0)
    y5, _ = scaled_labels(raw, floors, 5)
    y1, _ = scaled_labels(raw, floors, 1)
    eligible = np.isfinite(y5[:stop]).all(-1)
    eligible[:720] = False
    labels = {}
    for name, y in (("5m",y5[:stop]),("1m",y1[:stop])):
        rows = y[eligible]
        labels[name] = {"Q_return_correlation": correlation(rows[:,4],rows[:,1]),
                        "A_return_correlation": correlation(rows[:,5],rows[:,1]),
                        "logT_return_correlation": correlation(rows[:,3],rows[:,1]),
                        "Q_std": float(rows[:,4].std()), "A_std": float(rows[:,5].std()),
                        "return_tail_fraction_abs_gt2": float((np.abs(rows[:,1])>2).mean())}
    return {"fold": fold, "names": FAST_NAMES, "selected": [FAST_NAMES[j] for j in SELECTED],
            "removed": [FAST_NAMES[j] for j in range(20) if j not in SELECTED], "sample_rows": len(x),
            "pearson": corr.tolist(), "spearman": spearman.tolist(), "lag5m": lag_corr.tolist(),
            "near_pairs": pairs, "state_pairs": state_pairs, "labels": labels,
            "selection_policy": "return_copy is diagnostic only and excluded from all networks; mechanism template removes body, two wicks, duplicate ETH trend and activity level already in state; correlations diagnose, never validation fit"}


def paired_skill_interval(a: np.ndarray, b: np.ndarray, labels: np.ndarray, block: int = 72) -> dict:
    y = labels[..., 1]
    difference = (((a-y)**2 - (b-y)**2).mean(1))
    denominator = float((y*y).mean())
    rng = np.random.default_rng(20261004)
    n = len(difference)
    estimates = []
    for _ in range(500):
        starts = rng.integers(0, n, size=int(np.ceil(n/block)))
        idx = (starts[:,None]+np.arange(block)[None,:]) % n
        estimates.append(float(difference[idx.reshape(-1)[:n]].mean()/denominator))
    return {"skill_increment": float(difference.mean()/denominator),
            "ci95": np.quantile(estimates,[.025,.975]).tolist(), "block_hours": block,
            "bootstrap_replicates": 500, "interpretation": "paired correlated-clock moving blocks; exploratory, no multiple-search correction"}


def ema_levels(signal: np.ndarray, dates: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(signal)
    result[0] = signal[0]
    for i in range(1,len(signal)):
        delta = max(float((dates[i]-dates[i-1])/np.timedelta64(1,"h")),1)
        effective = 1 - (1-alpha)**delta
        result[i] = (1-effective)*result[i-1] + effective*signal[i]
    return result


def hysteresis_states(signal: np.ndarray) -> np.ndarray:
    state = np.zeros(signal.shape[1])
    states = np.zeros_like(signal)
    # Fixed alpha-score thresholds; diagnostic direction state, not position or order.
    for i,row in enumerate(signal):
        state = np.where(row > .05,1,np.where(row < -.05,-1,
                    np.where(np.abs(row)<.02,0,state)))
        states[i] = state
    return states


def signal_summary(signal: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> dict:
    out = metrics(signal, labels, dates)
    out["adjacent_flip_rate"] = float((np.sign(signal[1:])!=np.sign(signal[:-1])).mean())
    out["rms_revision"] = float(np.sqrt(np.mean(np.diff(signal,axis=0)**2)))
    out["autocorrelation_1h"] = correlation(signal[:-1],signal[1:])
    out["hysteresis_change_rate"] = float((np.diff(hysteresis_states(signal),axis=0)!=0).mean())
    out["response_lag_hours"] = {str(lag): correlation(signal[:-lag] if lag else signal,
                                  labels[lag:,:,1] if lag else labels[:,:,1]) for lag in (0,1,2,4,8)}
    out["horizon_correlations"] = {str(h): correlation(signal, labels[:,:,i]) for i,h in enumerate((1,4,8))}
    return out


def signal_study(root: Path, chosen: dict, fold: str, folder: Path) -> dict:
    assert torch.cuda.is_available()
    store = MechanismStore(root / "cache_v4", fold, torch.device("cuda"))
    cfg = MechanismConfig(**chosen)
    model = MechanismNetwork(cfg).cuda()
    ckpt = torch.load(folder / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    hours = store.splits["train"]
    pred, _, _ = infer(model, store, hours, cfg)
    labels = store.labels[hours]
    # Adjacent real clock blocks only; never penalize adjacency of shuffled minibatch rows.
    consecutive = np.flatnonzero(np.r_[False,np.diff(hours)==1])
    pairs = consecutive[(consecutive % 24) == 0]
    starts = [int(i-23) for i in pairs if i>=23 and np.all(np.diff(hours[i-23:i+1])==1)]
    blocks = np.stack([pred[s:s+24] for s in starts])
    target_blocks = np.stack([labels[s:s+24,:,1] for s in starts])
    filt = CausalLevelFilter()
    optimizer = torch.optim.Adam(filt.parameters(),lr=.05)
    x = torch.from_numpy(blocks.transpose(1,0,2).reshape(24,-1))
    y = torch.from_numpy(target_blocks.transpose(1,0,2).reshape(24,-1))
    history = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = (filt(x)[4:]-y[4:]).square().mean()  # fixed four-hour filter warmup, not padded future
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    alpha = float(filt.alpha().detach())
    study = {"fold":fold,"learned_alpha":alpha,"learned_half_life_hours":float(np.log(.5)/np.log(1-alpha)),
             "training_blocks":len(starts),"training_method":"one bounded filter parameter, frozen neural encoder, contiguous train-only 24h blocks",
             "training_loss_start":history[0],"training_loss_end":history[-1],"splits":{}}
    for split in ("validation","replay"):
        data = np.load(folder / f"{split}.npz")
        s, d, lab = data["signal"],data["dates"],data["labels"]
        variants = {"raw":s,"ema_1h":ema_levels(s,d,.5),"ema_2h":ema_levels(s,d,1-2**(-.5)),
                    "ema_4h":ema_levels(s,d,1-2**(-.25)),"learned":ema_levels(s,d,alpha)}
        study["splits"][split] = {name:signal_summary(v,lab,d) for name,v in variants.items()}
        study["splits"][split]["example"] = {"dates":[str(v) for v in d[:168]],
            "raw":s[:168,4].tolist(),"ema_2h":variants["ema_2h"][:168,4].tolist(),
            "learned":variants["learned"][:168,4].tolist(),"future_return":lab[:168,4,1].tolist(),
            "hysteresis":hysteresis_states(variants["learned"])[:168,4].tolist(),"symbol":"BTCUSDT"}
    return study


def head_diagnostics(folder: Path, store: MechanismStore) -> dict:
    saved = torch.load(folder / "best.pt", map_location=store.device, weights_only=False)
    cfg = MechanismConfig(**saved["config"])
    model = MechanismNetwork(cfg).to(store.device).eval()
    model.load_state_dict(saved["model"])
    diagnostics = {}
    for split in ("validation","replay"):
        hours = store.splits[split]
        ysource = store.labels1 if cfg.path_frequency == 1 else store.labels
        probabilities, conditional, band_ce, dist_ce, predictions_path = [], [], [], [], []
        for start in range(0,len(hours),24):
            x,y = store.batch(hours[start:start+24],cfg)
            with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
                output = model(*x)
            p = output["probability"].float().cpu().numpy()
            probabilities.append(p)
            predictions_path.append(output["path"].float().cpu().numpy())
            yy = y.cpu().numpy()
            if cfg.label == "four_bin":
                logits = output["sign"].float().cpu().numpy()
                band = yy[:,7].astype(int)
                selected = logits[np.arange(len(logits)),band]
                sign_target=np.where(yy[:,1]==0,.5,(yy[:,1]>0).astype(float))
                conditional.append(np.logaddexp(0,selected)-sign_target*selected)
                logq = output["band"].float().log_softmax(-1).cpu().numpy()
                band_ce.append(-logq[np.arange(len(logq)),band])
            if cfg.label == "distribution":
                logp = output["distribution"].float().log_softmax(-1).cpu().numpy()
                dist_ce.append(-logp[np.arange(len(logp)),yy[:,6].astype(int)])
        target = ysource[hours].reshape(-1,8)
        path = np.concatenate(predictions_path)
        if cfg.label in ("joint","path"):
            prior = ysource[store.splits["train"],:,3:6].mean((0,1))
            err = ((path-target[:,3:6])**2).mean(0)
            null = ((target[:,3:6]-prior)**2).mean(0)
            diagnostics[split] = {"path_mse":err.tolist(),"path_skill_vs_train_mean":(1-err/null).tolist(),
                                   "path_names":["log_total_activity","signed_square_pressure_Q","relative_asymmetry_A"]}
        else:
            diagnostics[split] = {}
        if cfg.label in ("distribution","four_bin"):
            p = np.clip(np.concatenate(probabilities),1e-6,1-1e-6)
            up = np.where(target[:,1]==0,.5,(target[:,1]>0).astype(float)) if cfg.label=="four_bin" else (target[:,1]>0).astype(float)
            calibration=[]
            for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
                mask=(p>=lo)&(p<hi)
                if mask.any():
                    calibration.append({"count":int(mask.sum()),"probability":float(p[mask].mean()),"realized_up":float(up[mask].mean())})
            diagnostics[split].update({"unconditional_bce":float((-up*np.log(p)-(1-up)*np.log(1-p)).mean()),
                                        "brier":float(((p-up)**2).mean()),"calibration":calibration})
            if conditional:
                diagnostics[split]["realized_band_conditional_bce"] = float(np.concatenate(conditional).mean())
                diagnostics[split]["band_ce"] = float(np.concatenate(band_ce).mean())
            if dist_ce:
                diagnostics[split]["joint_distribution_ce"] = float(np.concatenate(dist_ce).mean())
    # Active parameter counts include only gradients of the actual objective, not unused heads.
    model.train()
    x,y = store.batch(store.splits["train"][:8],cfg)
    out = model(*x)
    from .mechanism_model import losses
    primary,auxiliary = losses(out,y,cfg)
    (primary+(.2*auxiliary if cfg.label=="joint" else 0)).backward()
    diagnostics["total_parameters"] = sum(p.numel() for p in model.parameters())
    diagnostics["active_objective_parameters"] = sum(p.numel() for p in model.parameters() if p.grad is not None)
    diagnostics["scale_floor_checkpoint_matches"] = bool(np.allclose(saved["scale_floors"],store.floors))
    for name in ("fast","slow","state"):
        key=f"{cfg.prep}_{name}"
        if saved["scalers"][key] != store.scalers[key]:
            raise ValueError(f"checkpoint scaler reconstruction mismatch: {folder}/{key}")
    diagnostics["scaler_checkpoint_matches"] = True
    return diagnostics
