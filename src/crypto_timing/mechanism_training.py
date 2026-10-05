"""Fold-local fitting, balanced-clock CUDA optimization and mechanism diagnostics."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .mechanism_data import BOUNDED, SELECTED
from .mechanism_model import MechanismConfig, MechanismNetwork, losses

FOLDS = {
    "f1": ("2025-07-01", "2025-11-01", "2026-02-01"),
    "f2": ("2026-02-01", "2026-06-01", "2026-09-25"),
}


def fit_scaler(array: np.ndarray, stop: int, bounded: tuple, raw: bool = False,
               stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    center = np.zeros(array.shape[1:], np.float32)
    scale = np.ones_like(center)
    for si in range(array.shape[1]):
        # Minute subsampling is co-prime with hour and 5m; all intraday phases covered.
        x = np.asarray(array[:stop:stride, si], np.float64)
        q = np.nanquantile(x, [.25, .5, .75], axis=0)
        center[si] = q[1]
        scale[si] = np.maximum((q[2] - q[0]) / 1.349, 1e-5)
    if not raw:
        center[:, bounded], scale[:, bounded] = 0, 1
    return center, scale


def normalize(array: np.ndarray, center: np.ndarray, scale: np.ndarray):
    mask = np.isfinite(array)
    z = (np.asarray(array) - center) / scale
    return np.nan_to_num(np.clip(z, -8, 8), nan=0, posinf=0, neginf=0).astype(np.float16), mask


def scaled_labels(raw: np.ndarray, floors: np.ndarray, frequency: int = 5) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(raw[..., 7], floors)
    y = raw[..., :3] / (scale[..., None] * np.sqrt(np.array([1, 4, 8]) / 4))
    u, d = (raw[..., 3], raw[..., 4]) if frequency == 5 else (raw[..., 5], raw[..., 6])
    t = (u + d) / (scale * scale + 1e-12)
    q = (u - d) / (scale * scale + 1e-12)
    a = (u - d) / (u + d + .02 * scale * scale + 1e-12)
    # Distribution bins are globally ordered: negative tail ... near zero ... positive tail.
    band = np.searchsorted([.5, 1, 2], np.abs(y[..., 1]), side="right")
    category = np.where(y[..., 1] > 0, 4 + band, 3 - band)
    out = np.concatenate((y, np.stack((np.log(t + .02), q, a, category, band), -1)), -1)
    return out.astype(np.float32), scale.astype(np.float32)


class MechanismStore:
    def __init__(self, root: Path, fold: str, device: torch.device):
        self.root, self.fold, self.device = root, fold, device
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.dates = np.load(root / "decision_time.npy")
        self.raw = np.load(root / "targets.npy")
        self.ns = len(self.manifest["symbols"])
        train_end, val_end, replay_end = [np.datetime64(v) for v in FOLDS[fold]]
        self.train_stop = int(np.searchsorted(self.dates, train_end))
        self.floors = np.nanquantile(self.raw[720:self.train_stop - 9, :, 7], .05, axis=0).astype(np.float32)
        self.labels, self.scale = scaled_labels(self.raw, self.floors)
        self.labels1, _ = scaled_labels(self.raw, self.floors, 1)
        state = np.load(root / "state_dynamic.npy", mmap_mode="r")
        good = np.isfinite(self.raw).all(-1) & np.isfinite(state).all(-1) & (self.raw[..., 8] == 0)
        good[:720] = False
        # Common maturity for 1/4/8h comparisons and quality control: latest label needs 8h+5m.
        maturity = self.dates + np.timedelta64(485, "m")
        periods = {"train": (self.dates < train_end) & (maturity <= train_end),
                   "validation": (self.dates >= train_end) & (maturity <= val_end),
                   "replay": (self.dates >= val_end) & (maturity <= replay_end)}
        self.splits = {k: np.flatnonzero(good.all(1) & period) for k, period in periods.items()}
        self.gpu, self.scalers, self.current_prep = {}, {}, None
        # Sign-specific conditional representatives are strictly fitted on this fold's training labels.
        y = self.labels[self.splits["train"], :, 1].reshape(-1)
        cat = self.labels[self.splits["train"], :, 6].reshape(-1).astype(int)
        self.representatives = np.array([y[cat == k].mean() if (cat == k).any() else 0 for k in range(8)], np.float32)
        # Four-bin mapping needs negatives ordered by magnitude, positives ordered by magnitude.
        self.four_representatives = np.r_[self.representatives[:4][::-1], self.representatives[4:]].astype(np.float32)
        self.random_selected = tuple(sorted(np.random.default_rng(20261004).choice(19, len(SELECTED), replace=False)))
        self.prep_audit = {}

    def load_prep(self, prep: str):
        if self.current_prep == prep:
            return
        for name, multiple in (("fast", 12), ("slow", 1), ("state", 1)):
            arr = np.load(self.root / f"{name}_{prep}.npy", mmap_mode="r")
            center, scale = fit_scaler(arr, (self.train_stop - 9) * multiple, BOUNDED[name], prep == "raw")
            value, mask = normalize(arr, center, scale)
            self.gpu[name] = (torch.from_numpy(value).to(self.device), torch.from_numpy(mask).to(self.device))
            self.scalers[f"{prep}_{name}"] = {"center": center.tolist(), "scale": scale.tolist()}
            # Distribution diagnostics at every phase in each month; no fixed-hour phase shortcut.
            times = np.repeat(self.dates, multiple).astype("datetime64[M]")
            monthly = {}
            for month in np.unique(times):
                sub = value[times == month].astype(np.float32)
                m = mask[times == month]
                monthly[str(month)] = {"center": float(sub[m].mean()), "rms": float(np.sqrt((sub[m] ** 2).mean())),
                                       "clipped": float(((np.abs(sub) >= 8) & m).sum() / max(m.sum(), 1))}
            self.prep_audit[f"{prep}_{name}"] = monthly
        self.current_prep = prep

    def load_optional(self, name: str):
        if name in self.gpu:
            return
        arr = np.load(self.root / f"{name}.npy", mmap_mode="r")
        multiple = 60 if name == "minute" else 12
        center, scale = fit_scaler(arr, (self.train_stop - 9) * multiple, BOUNDED[name], stride=17 if name == "minute" else 1)
        self.scalers[name] = {"center": center.tolist(), "scale": scale.tolist()}
        # Chunk by symbol to avoid a full-size float32 minute normalization copy.
        values = torch.empty(arr.shape, dtype=torch.float16, device=self.device)
        masks = torch.empty(arr.shape, dtype=torch.bool, device=self.device)
        for si in range(self.ns):
            value, mask = normalize(arr[:, si:si+1], center[si:si+1], scale[si:si+1])
            values[:, si:si+1] = torch.from_numpy(value).to(self.device)
            masks[:, si:si+1] = torch.from_numpy(mask).to(self.device)
        self.gpu[name] = (values, masks)

    def batch(self, hours: np.ndarray, cfg: MechanismConfig):
        self.load_prep(cfg.prep)
        h = torch.as_tensor(np.repeat(hours, self.ns), device=self.device, dtype=torch.long)
        s = torch.arange(self.ns, device=self.device).repeat(len(hours))
        inds = {"fast": (h * 12 + 11)[:, None] - torch.arange(143, -1, -1, device=self.device),
                "slow": h[:, None] - torch.arange(cfg.slow_hours - 1, -1, -1, device=self.device), "state": h}
        selected = (tuple(range(19)) if cfg.features == "all" else tuple(range(16)) if cfg.features == "original" else
                    self.random_selected if cfg.features == "random" else SELECTED)
        tensors = []
        for name in ("fast", "slow", "state"):
            sym = s if name == "state" else s[:, None]
            value, mask = [v[inds[name], sym] for v in self.gpu[name]]
            if name == "fast":
                value, mask = value[..., list(selected)], mask[..., list(selected)]
                if cfg.minute == "aggregate":
                    self.load_optional("extra")
                    ev, em = [v[inds[name], sym] for v in self.gpu["extra"]]
                    value, mask = torch.cat((value, ev), -1), torch.cat((mask, em), -1)
            tensors.extend((value.float(), mask.float()))
        if cfg.minute == "encoder":
            self.load_optional("minute")
            mi = (h * 60 + 59)[:, None] - torch.arange(179, -1, -1, device=self.device)
            tensors.extend(v[mi, s[:, None]].float() for v in self.gpu["minute"])
        else:
            tensors.extend((None, None))
        labels = self.labels1 if cfg.path_frequency == 1 else self.labels
        y = torch.from_numpy(labels[hours].reshape(-1, 8)).to(self.device)
        rep = self.four_representatives if cfg.label == "four_bin" else self.representatives
        tensors.append(torch.from_numpy(rep).to(self.device))
        return tuple(tensors), y


def correlation(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-10 and b.std() > 1e-10 else 0.0


def metrics(pred: np.ndarray, labels: np.ndarray, dates: np.ndarray, native: int = 4) -> dict:
    # Equal-clock loss; all twelve correlated contracts contribute only one clock's weight.
    y = labels[..., {1: 0, 4: 1, 8: 2}[native]]
    error = ((pred - y) ** 2).mean(1)
    null = (y ** 2).mean(1)
    result = {"mse": float(error.mean()), "zero_mse": float(null.mean()),
              "skill": float(1 - error.mean() / null.mean()), "correlation": correlation(pred, y),
              "sign_accuracy": float((np.sign(pred) == np.sign(y)).mean()),
              "signal_std": float(pred.std()), "mean_signal": float(pred.mean()), "hours": len(pred)}
    months = dates.astype("datetime64[M]")
    result["monthly"] = {}
    for month in np.unique(months):
        k = months == month
        result["monthly"][str(month)] = {"skill": float(1 - error[k].mean() / null[k].mean()),
                                         "correlation": correlation(pred[k], y[k]), "hours": int(k.sum())}
    result["positive_months"] = sum(v["skill"] > 0 for v in result["monthly"].values())
    return result


@torch.inference_mode()
def infer(model: MechanismNetwork, store: MechanismStore, hours: np.ndarray, cfg: MechanismConfig,
          representations: bool = False):
    model.eval()
    result, paths, reps = [], [], []
    for start in range(0, len(hours), 24):
        x, _ = store.batch(hours[start:start + 24], cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(*x)
        result.append(out["signal"].float().cpu().numpy().reshape(-1, store.ns))
        paths.append(out["path"].float().cpu().numpy().reshape(-1, store.ns, 3))
        if representations:
            reps.append(out["representation"].float().cpu().numpy())
    return np.concatenate(result), np.concatenate(paths), np.concatenate(reps) if representations else None


def gradient_diagnostic(primary, aux, parameters):
    gs = torch.autograd.grad(primary, parameters, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(aux, parameters, retain_graph=True, allow_unused=True)
    zero = primary.new_tensor(0., dtype=torch.float32)
    norm_s = torch.sqrt(sum(((g.float() ** 2).sum() for g in gs if g is not None), zero) + 1e-16)
    norm_a = torch.sqrt(sum(((g.float() ** 2).sum() for g in ga if g is not None), zero) + 1e-16)
    dot = sum(((s.float() * a.float()).sum() for s, a in zip(gs, ga) if s is not None and a is not None), zero)
    return float(norm_s), float(norm_a), float(dot / (norm_s * norm_a + 1e-12))


def train_trial(store: MechanismStore, cfg: MechanismConfig, folder: Path, *, epochs: int = 8,
                steps_per_epoch: int = 128, batch_hours: int = 24, full: bool = False) -> dict:
    contract = {"config": asdict(cfg), "fold": store.fold, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
                "batch_hours": batch_hours, "full": full, "cache_audit": store.manifest["audit"]}
    if (folder / "summary.json").exists():
        saved = json.loads((folder / "summary.json").read_text())
        if saved["contract"] != contract:
            raise ValueError(f"existing run contract changed: {folder}")
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(cfg.seed)
    store.load_prep(cfg.prep)
    model = MechanismNetwork(cfg).to(store.device)
    train_labels = store.labels1 if cfg.path_frequency == 1 else store.labels
    train_hours = store.splits["train"]
    val_hours = store.splits["validation"]
    with torch.no_grad():
        model.signal.bias.fill_(float(train_labels[train_hours, :, {1: 0, 4: 1, 8: 2}[cfg.horizon]].mean()))
        model.path.bias.copy_(torch.from_numpy(train_labels[train_hours, :, 3:6].mean((0, 1))).to(store.device))
        categories = train_labels[train_hours, :, 6].astype(int).reshape(-1)
        model.distribution.bias.copy_(torch.tensor(np.log(np.bincount(categories, minlength=8) / len(categories)), device=store.device))
        if cfg.label == "four_bin":
            band=train_labels[train_hours,:,7].astype(int).reshape(-1)
            target=train_labels[train_hours,:,1].reshape(-1)
            signs=np.where(target==0,.5,(target>0).astype(float))
            priors=np.array([(signs[band==j].sum()+1)/((band==j).sum()+2) for j in range(4)])
            model.sign.bias.copy_(torch.tensor(np.log(priors/(1-priors)),device=store.device))
            magnitude_prior=np.bincount(band,minlength=4)/len(band)
            model.legacy_magnitude[-1].bias.copy_(torch.tensor(np.log(magnitude_prior),device=store.device))
            torch.nn.init.normal_(model.legacy_magnitude[-1].weight,std=.002)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 or name.endswith("bias") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": .001}, {"params": no_decay, "weight_decay": 0}], lr=2e-4)
    steps = math.ceil(len(train_hours) / batch_hours) if full else steps_per_epoch
    best, best_epoch, history, grad_log, update_log = float("inf"), 0, [], [], []
    aux_weight, ema_ratio, total = .2, None, epochs * steps
    started = time.perf_counter()
    seen_hours = set()
    print(json.dumps({"event": "start", "folder": str(folder), "parameters": sum(p.numel() for p in model.parameters()),
                      "train_hours": len(train_hours), "steps": steps, "config": asdict(cfg)}), flush=True)
    for epoch in range(epochs):
        model.train()
        chosen = rng.permutation(train_hours)
        if not full:
            chosen = chosen[:steps * batch_hours]
        running, seen, epoch_grads = 0., 0, []
        for bi, start in enumerate(range(0, len(chosen), batch_hours)):
            hours = chosen[start:start + batch_hours]
            seen_hours.update(map(int, hours))
            x, y = store.batch(hours, cfg)
            step = epoch * steps + bi
            # Early decay reaches low LR within the evaluated budget, rather than after early stopping.
            progress = min(max((step - 24) / max(.65 * total - 24, 1), 0), 1)
            lr = (2e-5 + (2e-4 - 2e-5) * (step + 1) / 24 if step < 24 else
                  2e-5 + .5 * (2e-4 - 2e-5) * (1 + math.cos(math.pi * progress)))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(*x)
                primary, auxiliary = losses(out, y, cfg)
            if cfg.label == "joint" and bi % 16 == 0:
                ns, na, cosine = gradient_diagnostic(primary, auxiliary, model.shared_parameters())
                measured = min(max(ns / max(na, 1e-6), .01), 10)
                ema_ratio = measured if ema_ratio is None else .9 * ema_ratio + .1 * measured
                aux_weight = min(.5, max(.01, .3 * ema_ratio)) if cfg.sharing != "isolated" else .2
                epoch_grads.append({"main_norm": ns, "aux_norm": na, "cosine": cosine, "weight": aux_weight,
                                    "weighted_ratio": aux_weight * na / max(ns, 1e-6)})
            loss = primary + (aux_weight * auxiliary if cfg.label == "joint" else 0)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(gradient):
                raise RuntimeError("nonfinite gradient")
            monitored = {name: p.detach().clone() for name, p in model.named_parameters()
                         if bi == 0 and name in ("fast.input.0.weight", "memory.recurrent.weight_hh_l0",
                                                 "slow.input.0.weight", "fusion.0.weight", "signal.weight")}
            opt.step()
            if monitored:
                updated = {name: float((p.detach() - monitored[name]).norm() / monitored[name].norm().clamp_min(1e-8))
                           for name, p in model.named_parameters() if name in monitored}
                update_log.append({"epoch": epoch+1, "relative_updates": updated})
            running += float(primary.detach()) * len(hours)
            seen += len(hours)
        pred, predicted_path, _ = infer(model, store, val_hours, cfg)
        # Native target early stopping for horizon sensitivity, common 4h metric reported separately.
        score = metrics(pred, train_labels[val_hours], store.dates[val_hours], cfg.horizon)
        record = {"epoch": epoch + 1, "train_primary": running / seen, "validation": score,
                  "lr_end": lr, "clock_exposures": seen, "gradients": epoch_grads,
                  "elapsed_seconds": round(time.perf_counter() - started, 2)}
        history.append(record)
        grad_log.extend(epoch_grads)
        (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({"event": "epoch", "run": folder.name, "epoch": epoch+1, "train": running/seen,
                          "val_skill": score["skill"], "lr": lr, "elapsed_s": record["elapsed_seconds"]}), flush=True)
        stopping_loss = float(((predicted_path-train_labels[val_hours,:,3:6])**2).mean()) if cfg.label == "path" else score["mse"]
        record["stopping_loss"] = stopping_loss
        if stopping_loss < best:
            best, best_epoch = stopping_loss, epoch + 1
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "contract": contract,
                        "scalers": store.scalers, "scale_floors": store.floors.tolist(),
                        "representatives": store.representatives.tolist(), "epoch": best_epoch}, folder / "best.pt")
    saved = torch.load(folder / "best.pt", weights_only=False, map_location=store.device)
    model.load_state_dict(saved["model"])
    results = {}
    for split in ("validation", "replay"):
        hours = store.splits[split]
        pred, paths, _ = infer(model, store, hours, cfg)
        results[split] = metrics(pred, train_labels[hours], store.dates[hours], cfg.horizon)
        results[split]["common_4h"] = metrics(pred, train_labels[hours], store.dates[hours], 4)
        results[split]["path_mse"] = ((paths - train_labels[hours, :, 3:6]) ** 2).mean((0, 1)).tolist()
        np.savez_compressed(folder / f"{split}.npz", hours=hours, dates=store.dates[hours], signal=pred,
                            path=paths, labels=train_labels[hours], scale=store.scale[hours], raw=store.raw[hours])
    # Train-only linear probe of the learned representation, not a traditional model contest.
    probe_train = train_hours[::max(len(train_hours) // 1800, 1)]
    probe_val = val_hours[::2]
    _, _, tr = infer(model, store, probe_train, cfg, True)
    vp, _, vr = infer(model, store, probe_val, cfg, True)
    tr, vr = tr.astype(np.float64), vr.astype(np.float64)
    mu, sd = tr.mean(0), np.maximum(tr.std(0), 1e-3)
    xt = np.c_[(tr - mu) / sd, np.ones(len(tr))]
    xv = np.c_[(vr - mu) / sd, np.ones(len(vr))]
    yy = train_labels[probe_train, :, 1].reshape(-1)
    coef = np.linalg.solve(xt.T @ xt + 100 * np.eye(xt.shape[1]), xt.T @ yy)
    probe = metrics((xv @ coef).reshape(-1, store.ns), train_labels[probe_val], store.dates[probe_val])
    singular = np.linalg.svd(tr - tr.mean(0), compute_uv=False)
    spectrum = singular ** 2 / np.maximum((singular ** 2).sum(), 1e-12)
    rank = float(np.exp(-(spectrum * np.log(spectrum + 1e-12)).sum()))
    # Diagnostic interventions are labelled OOD sensitivity, never called causal contribution.
    x, _ = store.batch(val_hours[::max(len(val_hours) // 24, 1)][:24], cfg)
    model.eval()
    sensitivity = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = model(*x)["signal"].float()
        for name, indices in {"fast": (0, 1), "slow": (2, 3), "state": (4, 5),
                              **({"minute": (6, 7)} if cfg.minute == "encoder" else {})}.items():
            altered = list(x)
            for index in indices:
                altered[index] = torch.zeros_like(altered[index])
            change = model(*altered)["signal"].float() - base
            sensitivity[name] = float(torch.sqrt((change ** 2).mean()))
        shuffled = list(x)
        shuffled[0], shuffled[1] = x[0].flip(1), x[1].flip(1)
        sensitivity["reversed_fast_order"] = float(torch.sqrt(((model(*shuffled)["signal"].float() - base) ** 2).mean()))
        activation = model(*x)["local_activation"].float()
        sensitivity["activation_rms"] = float(torch.sqrt((activation ** 2).mean()))
    summary = {"contract": contract, "config": asdict(cfg), "fold": store.fold,
               "parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
               "total_steps": epochs * steps, "unique_train_hours": len(seen_hours), "train_hours": len(train_hours),
               "seconds": round(time.perf_counter() - started, 2), "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
               "results": results, "probe": probe, "effective_rank": rank, "ood_sensitivity": sensitivity,
               "gradient_diagnostics": grad_log, "relative_parameter_updates": update_log,
               "scale_floors": store.floors.tolist()}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
