"""Train-only normalization, chronological splits and GPU training."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .cache import load_cache
from .model import CryptoTimingNetwork, ModelConfig

TRAIN_END = np.datetime64("2025-07-01T00:00:00")
VALID_END = np.datetime64("2026-02-01T00:00:00")
CALIBRATION_END = np.datetime64("2025-11-01T00:00:00")
TEST_END = np.datetime64("2026-09-25T00:00:00")
LABEL_DELAY = np.timedelta64(4 * 60 + 5, "m")


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "joint4h"
    seed: int = 20261003
    batch_size: int = 256
    max_epochs: int = 6
    train_stride_hours: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 2
    fast_blocks: int = 3
    use_slow: bool = True
    use_stats: bool = True
    use_market: bool = True


def split_indices(decision_time: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    n_hours, n_symbols = targets.shape[:2]
    eligible = np.isfinite(targets[:, :, [0, 1, 2, 3, 4, 5, 6, 7]]).all(axis=-1)
    eligible[:168] = False
    label_end = decision_time + LABEL_DELAY
    groups = {
        "train": (decision_time < TRAIN_END) & (label_end <= TRAIN_END),
        "validation": (decision_time >= TRAIN_END) & (decision_time < VALID_END) & (label_end <= VALID_END),
        "calibration": (decision_time >= TRAIN_END) & (decision_time < CALIBRATION_END) & (label_end <= CALIBRATION_END),
        "selection": (decision_time >= CALIBRATION_END) & (decision_time < VALID_END) & (label_end <= VALID_END),
        "test": (decision_time >= VALID_END) & (decision_time < TEST_END),
    }
    flat = np.arange(n_hours * n_symbols).reshape(n_hours, n_symbols)
    return {name: flat[eligible & time_mask[:, None]] for name, time_mask in groups.items()}


def _fit_per_symbol(values: np.ndarray, train_hour_count: int, sample_stride: int) -> tuple[np.ndarray, np.ndarray]:
    n_symbols = values.shape[1]
    n_features = values.shape[2]
    center = np.zeros((n_symbols, n_features), np.float32)
    scale = np.ones((n_symbols, n_features), np.float32)
    sample = values[:train_hour_count:sample_stride]
    for si in range(n_symbols):
        for fi in range(n_features):
            column = np.asarray(sample[:, si, fi])
            column = column[np.isfinite(column)]
            if len(column) < 100:
                continue
            q25, q50, q75 = np.percentile(column, (25, 50, 75))
            center[si, fi] = q50
            scale[si, fi] = max((q75 - q25) / 1.349, 1e-5)
    return center, scale


def _normalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values)
    normalized = np.clip((values - center[None, :, :]) / scale[None, :, :], -8, 8)
    return np.where(mask, normalized, 0).astype(np.float32), mask.astype(np.float32)


class PreparedStore:
    def __init__(self, root: Path, scalers: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
        cache = load_cache(root)
        self.manifest = cache["manifest"]
        self.decision_time = np.asarray(cache["decision_time"])
        self.targets = np.asarray(cache["targets"])
        self.splits = split_indices(self.decision_time, self.targets)
        train_hours = int(np.searchsorted(self.decision_time, TRAIN_END))
        self.scalers = {} if scalers is None else scalers
        self.features: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        for name in ("fast", "slow", "stats", "market"):
            data = np.asarray(cache[name])
            if name not in self.scalers:
                stride = 12 if name == "fast" else 1
                limit = train_hours * 12 if name == "fast" else train_hours
                self.scalers[name] = _fit_per_symbol(data, limit, stride)
            self.features[name], self.masks[name] = _normalize(data, *self.scalers[name])
        self.n_symbols = self.targets.shape[1]

    def batch(self, keys: np.ndarray, device: torch.device) -> tuple[tuple[Tensor, ...], Tensor]:
        hour = keys // self.n_symbols
        symbol = keys % self.n_symbols
        fast_end = hour * 12 + 11
        fast_idx = fast_end[:, None] - np.arange(287, -1, -1)[None, :]
        slow_idx = hour[:, None] - np.arange(167, -1, -1)[None, :]
        arrays = (
            self.features["fast"][fast_idx, symbol[:, None]],
            self.masks["fast"][fast_idx, symbol[:, None]],
            self.features["slow"][slow_idx, symbol[:, None]],
            self.masks["slow"][slow_idx, symbol[:, None]],
            self.features["stats"][hour, symbol],
            self.masks["stats"][hour, symbol],
            self.features["market"][hour, symbol],
            self.masks["market"][hour, symbol],
        )
        tensors = tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays)
        target = torch.from_numpy(np.ascontiguousarray(self.targets[hour, symbol])).to(device)
        return tensors, target


def _loss(pred: Tensor, target: Tensor, mode: str) -> Tensor:
    asym = torch.tanh(pred[:, 0].float())
    logratio = 2 * pred[:, 0].float()
    volatility = pred[:, 1].float()
    risk_return = pred[:, 2].float()
    vol_loss = F.smooth_l1_loss(volatility, target[:, 2].float().clamp(-5, 5))
    if mode == "asym4h":
        return F.smooth_l1_loss(asym, target[:, 0].float()) + 0.15 * vol_loss
    if mode == "logratio4h":
        return 0.5 * F.smooth_l1_loss(logratio, target[:, 1].float().clamp(-6, 6)) + 0.15 * vol_loss
    if mode == "asym1h":
        return F.smooth_l1_loss(asym, target[:, 6].float())
    if mode == "return4h":
        return F.smooth_l1_loss(risk_return, target[:, 3].float()) + 0.15 * vol_loss
    if mode == "joint4h":
        return (
            0.5 * F.smooth_l1_loss(asym, target[:, 0].float())
            + 0.35 * F.smooth_l1_loss(risk_return, target[:, 3].float())
            + 0.15 * vol_loss
        )
    raise ValueError(f"unknown mode: {mode}")


def _save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def predict(model: CryptoTimingNetwork, store: PreparedStore, keys: np.ndarray,
            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = np.empty((len(keys), 3), np.float32)
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        inputs, _ = store.batch(batch_keys, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(*inputs)
        result[start : start + len(batch_keys)] = output.float().cpu().numpy()
    return result


@torch.inference_mode()
def evaluate_loss(model: CryptoTimingNetwork, store: PreparedStore, keys: np.ndarray,
                  device: torch.device, batch_size: int, mode: str) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(keys), batch_size):
        inputs, target = store.batch(keys[start : start + batch_size], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(*inputs)
        total += float(_loss(pred, target, mode)) * len(target)
    return total / len(keys)


def train(cache_root: Path, output_root: Path, cfg: TrainConfig) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = PreparedStore(cache_root)
    model_cfg = ModelConfig(fast_blocks=cfg.fast_blocks, use_slow=cfg.use_slow,
                            use_stats=cfg.use_stats, use_market=cfg.use_market,
                            market_dim=store.features["market"].shape[2])
    model = CryptoTimingNetwork(model_cfg).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    history_path = output_root / "history.jsonl"
    train_keys = store.splits["train"]
    early_keys = store.splits["calibration"]
    best_loss = float("inf")
    best_epoch = -1
    start_epoch = 0
    resume_rng_state = None
    resume_torch_rng_state = None
    resume_cuda_rng_state = None
    last_checkpoint = output_root / "last.pt"
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(cfg):
            raise ValueError("resume configuration differs from checkpoint")
        if "cache_manifest" in saved and saved["cache_manifest"] != store.manifest:
            raise ValueError("feature cache changed since last checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        best_loss, best_epoch = saved["best_loss"], saved["best_epoch"]
        resume_rng_state = saved.get("rng_state")
        resume_torch_rng_state = saved.get("torch_rng_state")
        resume_cuda_rng_state = saved.get("cuda_rng_state")
        print(f"resuming from epoch {start_epoch}", flush=True)
    print(json.dumps({"event": "start", "mode": cfg.mode, "seed": cfg.seed,
                      "parameters": parameters, "train_rows": len(train_keys),
                      "early_validation_rows": len(early_keys), "device": str(device)}), flush=True)
    rng = np.random.default_rng(cfg.seed)
    if resume_rng_state is not None:
        rng.bit_generator.state = resume_rng_state
    if resume_torch_rng_state is not None:
        torch.set_rng_state(resume_torch_rng_state)
    if resume_cuda_rng_state is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(resume_cuda_rng_state)
    for epoch in range(start_epoch, cfg.max_epochs):
        epoch_started = time.perf_counter()
        model.train()
        # Alternate clock phases across epochs so all hourly decisions are learned
        # while a single epoch uses less-overlapping labels.
        active = train_keys[(train_keys // store.n_symbols) % cfg.train_stride_hours == epoch % cfg.train_stride_hours]
        active = rng.permutation(active)
        running = 0.0
        seen = 0
        for start in range(0, len(active), cfg.batch_size):
            keys = active[start : start + cfg.batch_size]
            inputs, target = store.batch(keys, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(*inputs)
                loss = _loss(output, target, cfg.mode)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(keys)
            seen += len(keys)
        validation_loss = evaluate_loss(model, store, early_keys, device, cfg.batch_size, cfg.mode)
        record = {"epoch": epoch + 1, "train_loss": running / seen,
                  "early_validation_loss": validation_loss,
                  "seen": seen, "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                  if device.type == "cuda" else None,
                  "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            _save_atomic({"model": model.state_dict(), "model_config": asdict(model_cfg),
                          "train_config": asdict(cfg), "scalers": store.scalers,
                          "epoch": epoch, "validation_loss": validation_loss,
                          "parameters": parameters, "cache_manifest": store.manifest},
                         output_root / "best.pt")
        _save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": asdict(cfg), "epoch": epoch, "best_loss": best_loss,
                      "best_epoch": best_epoch, "rng_state": rng.bit_generator.state,
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                      "cache_manifest": store.manifest}, last_checkpoint)
        if epoch - best_epoch >= cfg.patience:
            print(f"early stopping after epoch {epoch + 1}", flush=True)
            break
    saved = torch.load(output_root / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    validation_keys = store.splits["validation"]
    prediction = predict(model, store, validation_keys, device, cfg.batch_size)
    np.savez_compressed(output_root / "validation_predictions.npz",
                        keys=validation_keys, prediction=prediction)
    summary = {"mode": cfg.mode, "seed": cfg.seed, "best_epoch": best_epoch + 1,
               "early_validation_loss": best_loss, "parameters": parameters,
               "validation_rows": len(validation_keys), "train_rows": len(train_keys)}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **summary}), flush=True)
    return summary
