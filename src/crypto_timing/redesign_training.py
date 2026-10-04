"""Full-clock, direction-first training for the isolated v3 experiment."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .redesign_cache import CAL_END, TRAIN_END, VALID_END, WARMUP_HOURS, load_redesign_cache
from .redesign_model import (DirectionConfig, DirectionNetwork, MagnitudeNetwork,
                             direction_loss, magnitude_loss)
from .training import _fit_per_symbol, _normalize

HIST_END = np.datetime64("2026-09-25T00:00:00")
LABEL_MATURITY = np.timedelta64(4 * 60 + 5, "m")


@dataclass(frozen=True)
class RedesignTrainConfig:
    kind: str = "direction"
    seed: int = 20261004
    batch_hours: int = 24
    max_epochs: int = 12
    peak_lr: float = 2e-4
    floor_lr: float = 2e-5
    weight_decay: float = 1e-3
    patience: int = 4
    min_epochs: int = 3


def split_keys(dates: np.ndarray, targets: np.ndarray, state: np.ndarray,
               magnitude: np.ndarray) -> dict[str, np.ndarray]:
    n_hours, n_symbols = targets.shape[:2]
    valid = np.isfinite(targets).all(axis=-1)
    valid &= np.isfinite(state).all(axis=-1)
    valid &= np.isfinite(magnitude).all(axis=-1)
    valid[:WARMUP_HOURS] = False
    maturity = dates + LABEL_MATURITY
    masks = {
        "train": (dates < TRAIN_END) & (maturity <= TRAIN_END),
        "calibration": (dates >= TRAIN_END) & (dates < CAL_END) & (maturity <= CAL_END),
        "selection": (dates >= CAL_END) & (dates < VALID_END) & (maturity <= VALID_END),
        "historical": (dates >= VALID_END) & (dates < HIST_END),
    }
    flat = np.arange(n_hours * n_symbols, dtype=np.int32).reshape(n_hours, n_symbols)
    return {name: flat[valid & period[:, None]] for name, period in masks.items()}


class RedesignStore:
    def __init__(self, root: Path, scalers: dict | None = None):
        cache = load_redesign_cache(root)
        self.manifest = cache["manifest"]
        self.dates = np.asarray(cache["decision_time"])
        self.targets = np.asarray(cache["targets"])
        self.n_symbols = len(self.manifest["symbols"])
        self.splits = split_keys(self.dates, self.targets, cache["state"], cache["magnitude"])
        train_hours = int(np.searchsorted(self.dates, TRAIN_END))
        self.scalers = {} if scalers is None else scalers
        self.values = {}
        self.masks = {}
        for name in ("fast", "slow", "state", "magnitude"):
            values = np.asarray(cache[name])
            if name not in self.scalers:
                stride = 12 if name == "fast" else 1
                limit = train_hours * 12 if name == "fast" else train_hours
                self.scalers[name] = _fit_per_symbol(values, limit, stride)
            self.values[name], self.masks[name] = _normalize(values, *self.scalers[name])

    def batch(self, keys: np.ndarray, kind: str, device: torch.device) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        hour, symbol = keys // self.n_symbols, keys % self.n_symbols
        labels = np.asarray(self.targets[hour, symbol])
        band = torch.from_numpy(np.ascontiguousarray(labels[:, 2])).long().to(device)
        up = torch.from_numpy(np.ascontiguousarray(labels[:, 3])).float().to(device)
        if kind == "magnitude":
            names = ("magnitude",)
            indices = (hour,)
        else:
            fast_idx = (hour * 12 + 11)[:, None] - np.arange(143, -1, -1)[None, :]
            slow_idx = hour[:, None] - np.arange(71, -1, -1)[None, :]
            names = ("fast", "slow", "state")
            indices = (fast_idx, slow_idx, hour)
        tensors = []
        for name, idx in zip(names, indices, strict=True):
            symbols = symbol[:, None] if name in {"fast", "slow"} else symbol
            for values in (self.values[name], self.masks[name]):
                array = np.ascontiguousarray(values[idx, symbols])
                tensors.append(torch.from_numpy(array).to(device))
        return tuple(tensors), band, up


def _optimizer(model: nn.Module, cfg: RedesignTrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim == 1 or name.endswith("bias") else decay).append(param)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.peak_lr)


def _set_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int,
            warmup_steps: int, cfg: RedesignTrainConfig) -> float:
    if step < warmup_steps:
        lr = cfg.floor_lr + (cfg.peak_lr - cfg.floor_lr) * (step + 1) / warmup_steps
    else:
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1)
        lr = cfg.floor_lr + .5 * (cfg.peak_lr - cfg.floor_lr) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _constant_direction_prior(store: RedesignStore) -> np.ndarray:
    keys = store.splits["train"]
    y = np.asarray(store.targets[keys // store.n_symbols, keys % store.n_symbols])
    band, up = y[:, 2].astype(int), y[:, 3]
    return np.array([(up[band == j].sum() + 1) / ((band == j).sum() + 2) for j in range(4)])


@torch.inference_mode()
def infer(model: nn.Module, store: RedesignStore, keys: np.ndarray,
          kind: str, device: torch.device, batch_size: int = 384) -> np.ndarray:
    model.eval()
    result = np.empty((len(keys), 4), np.float32)
    for start in range(0, len(keys), batch_size):
        x, _, _ = store.batch(keys[start:start + batch_size], kind, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(*x)
        result[start:start + batch_size] = pred.float().cpu().numpy()
    return result


def _bce_numpy(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, logits) - target * logits


def month_direction_scores(logits: np.ndarray, keys: np.ndarray,
                           store: RedesignStore, prior: np.ndarray,
                           linear_logits: np.ndarray | None = None) -> dict:
    n = store.n_symbols
    label = np.asarray(store.targets[keys // n, keys % n])
    band, up = label[:, 2].astype(int), label[:, 3]
    selected = logits[np.arange(len(keys)), band]
    base = np.log(prior[band] / (1 - prior[band]))
    loss = _bce_numpy(selected, up)
    null = _bce_numpy(base, up)
    linear = (_bce_numpy(linear_logits[np.arange(len(keys)), band], up)
              if linear_logits is not None else None)
    month = store.dates[keys // n].astype("datetime64[M]")
    result = {}
    for value in np.unique(month):
        keep = month == value
        result[str(value)] = {"network_ce": float(loss[keep].mean()),
                              "constant_ce": float(null[keep].mean()),
                              "gain_vs_constant": float((null[keep] - loss[keep]).mean()),
                              "rows": int(keep.sum())}
        if linear is not None:
            result[str(value)]["linear_ce"] = float(linear[keep].mean())
            result[str(value)]["gain_vs_linear"] = float((linear[keep] - loss[keep]).mean())
    return result


def _save_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_redesign(cache_root: Path, output_root: Path, cfg: RedesignTrainConfig,
                   baseline_root: Path | None = None) -> dict:
    if cfg.kind not in {"direction", "magnitude"}:
        raise ValueError(cfg.kind)
    output_root.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = RedesignStore(cache_root)
    prior = _constant_direction_prior(store)
    model_cfg = DirectionConfig(fast_dim=len(store.manifest["fast_names"]),
                                slow_dim=len(store.manifest["slow_names"]),
                                state_dim=len(store.manifest["state_names"]))
    model = (DirectionNetwork(model_cfg, float(np.log(prior.mean() / (1 - prior.mean()))))
             if cfg.kind == "direction" else MagnitudeNetwork(len(store.manifest["magnitude_names"]))).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = _optimizer(model, cfg)
    train_keys = store.splits["train"].reshape(-1, store.n_symbols)
    cal_keys = store.splits["calibration"]
    linear_logits = None
    if cfg.kind == "direction":
        if baseline_root is None:
            raise ValueError("direction training needs a frozen train-only linear baseline")
        baseline = np.load(baseline_root / "calibration_predictions.npz")
        if not np.array_equal(baseline["keys"], cal_keys):
            raise ValueError("baseline calibration keys differ")
        linear_logits = np.asarray(baseline["direction_logits"], np.float64)
    if train_keys.size == 0 or cal_keys.size == 0:
        raise ValueError("empty train or calibration split")
    batches_per_epoch = math.ceil(len(train_keys) / cfg.batch_hours)
    total_steps = batches_per_epoch * cfg.max_epochs
    rng = np.random.default_rng(cfg.seed)
    best_score = -float("inf")
    best_epoch = -1
    start_epoch = 0
    last_path = output_root / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(cfg) or saved["cache_manifest"] != store.manifest:
            raise ValueError("resume contract differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        best_score, best_epoch, start_epoch = saved["best_score"], saved["best_epoch"], saved["epoch"] + 1
        print(f"resuming {cfg.kind} from epoch {start_epoch + 1}", flush=True)
    print(json.dumps({"event": "start", "kind": cfg.kind, "device": str(device),
                      "parameters": params, "train_keys": int(train_keys.size),
                      "calibration_keys": len(cal_keys), "batches_per_epoch": batches_per_epoch}), flush=True)
    for epoch in range(start_epoch, cfg.max_epochs):
        started = time.perf_counter()
        model.train()
        permuted = train_keys[rng.permutation(len(train_keys))]
        running, seen, lr = 0.0, 0, 0.0
        for batch_index, start in enumerate(range(0, len(permuted), cfg.batch_hours)):
            keys = permuted[start:start + cfg.batch_hours].reshape(-1)
            x, band, up = store.batch(keys, cfg.kind, device)
            lr = _set_lr(optimizer, epoch * batches_per_epoch + batch_index,
                         total_steps, batches_per_epoch, cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(*x)
                loss = (direction_loss(logits, band, up, model.offset.weight)
                        if cfg.kind == "direction" else magnitude_loss(logits, band))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {cfg.kind} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(keys)
            seen += len(keys)
        predicted = infer(model, store, cal_keys, cfg.kind, device)
        if cfg.kind == "direction":
            monthly = month_direction_scores(predicted, cal_keys, store, prior, linear_logits)
            score = float(np.mean([item["gain_vs_linear"] for item in monthly.values()]))
            val_loss = float(np.mean([item["network_ce"] for item in monthly.values()]))
        else:
            label = np.asarray(store.targets[cal_keys // store.n_symbols,
                                             cal_keys % store.n_symbols, 2], dtype=int)
            logp = predicted - np.logaddexp.reduce(predicted, axis=1, keepdims=True)
            val_loss = float(-logp[np.arange(len(label)), label].mean())
            score = -val_loss
            monthly = {}
        record = {"epoch": epoch + 1, "train_loss_online": running / seen,
                  "validation_loss": val_loss, "selection_score": score,
                  "monthly": monthly, "lr_end": lr, "seen": seen,
                  "seconds": round(time.perf_counter() - started, 2),
                  "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                  if device.type == "cuda" else None}
        with (output_root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({key: value for key, value in record.items() if key != "monthly"}), flush=True)
        if score > best_score + 1e-5:
            best_score, best_epoch = score, epoch
            _save_checkpoint({"model": model.state_dict(), "model_config": asdict(model_cfg)
                              if cfg.kind == "direction" else {"input_dim": len(store.manifest["magnitude_names"])},
                              "config": asdict(cfg), "epoch": epoch, "score": score,
                              "scalers": store.scalers, "cache_manifest": store.manifest,
                              "direction_prior": prior.tolist()}, output_root / "best.pt")
        _save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "config": asdict(cfg), "epoch": epoch, "best_score": best_score,
                          "best_epoch": best_epoch, "numpy_rng": rng.bit_generator.state,
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                          "cache_manifest": store.manifest}, last_path)
        if epoch + 1 >= cfg.min_epochs and epoch - best_epoch >= cfg.patience:
            print(f"early stopping {cfg.kind} after epoch {epoch + 1}", flush=True)
            break
    best = torch.load(output_root / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    for split in ("calibration", "selection", "historical"):
        keys = store.splits[split]
        result = infer(model, store, keys, cfg.kind, device)
        np.savez_compressed(output_root / f"{split}_predictions.npz", keys=keys, logits=result)
    summary = {"kind": cfg.kind, "seed": cfg.seed, "parameters": params,
               "best_epoch": best_epoch + 1, "best_score": best_score,
               "train_rows": int(train_keys.size), "calibration_rows": len(cal_keys),
               "selection_rows": len(store.splits["selection"]),
               "historical_rows": len(store.splits["historical"])}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **summary}), flush=True)
    return summary
