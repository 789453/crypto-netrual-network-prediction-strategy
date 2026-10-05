"""Checkpoint-only inference: completed inputs and past risk, no future labels required."""
from __future__ import annotations

import numpy as np
import torch

from .mechanism_data import SELECTED
from .mechanism_model import MechanismConfig, MechanismNetwork


class MechanismPredictor:
    def __init__(self, checkpoint, device="cuda"):
        self.device=torch.device(device)
        if self.device.type=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        saved=torch.load(checkpoint,map_location=self.device,weights_only=False)
        self.cfg=MechanismConfig(**saved["config"])
        self.model=MechanismNetwork(self.cfg).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.scalers=saved["scalers"]
        self.floors=np.array(saved["scale_floors"],np.float32)
        self.representatives=np.array(saved["representatives"],np.float32)
        if self.cfg.label=="four_bin":
            self.representatives=np.r_[self.representatives[:4][::-1],self.representatives[4:]]

    def _prepare(self, name, values, symbols):
        key=name if name in ("minute","extra") else f"{self.cfg.prep}_{name}"
        center=np.asarray(self.scalers[key]["center"],np.float32)[symbols]
        scale=np.asarray(self.scalers[key]["scale"],np.float32)[symbols]
        if values.ndim==3:
            center,scale=center[:,None,:],scale[:,None,:]
        mask=np.isfinite(values)
        z=np.nan_to_num(np.clip((values-center)/scale,-8,8),nan=0,posinf=0,neginf=0)
        # Match the frozen research input precision before expanding to model float32.
        return z.astype(np.float16).astype(np.float32),mask.astype(np.float32)

    @torch.inference_mode()
    def predict(self, inputs: dict, symbols: np.ndarray, past_scale4: np.ndarray) -> dict:
        """Inputs use the saved prep's *pre-scaler* dictionary; fast includes all 20 cache fields.

        Each window must end at the same completed decision time. Callers own timestamps;
        future labels, train cutoffs and future trading availability are not arguments.
        """
        symbols=np.asarray(symbols,int)
        if (symbols<0).any() or (symbols>=len(self.floors)).any():
            raise ValueError("unknown contract id")
        if np.asarray(past_scale4).shape!=(len(symbols),) or not np.isfinite(past_scale4).all() or (np.asarray(past_scale4)<=0).any():
            raise ValueError("positive, finite causal risk required")
        dims={"fast":(144,20),"slow":(self.cfg.slow_hours,10),"state":(24,)}
        if self.cfg.minute=="encoder":
            dims["minute"]=(180,12)
        if self.cfg.minute=="aggregate":
            dims["extra"]=(144,6)
        for name,shape in dims.items():
            if name not in inputs or np.asarray(inputs[name]).shape!=(len(symbols),*shape):
                raise ValueError(f"{name}: expected completed window shape {(len(symbols),*shape)}")
        selected=(tuple(range(19)) if self.cfg.features=="all" else tuple(range(16)) if self.cfg.features=="original" else
                  tuple(sorted(np.random.default_rng(20261004).choice(19,len(SELECTED),replace=False))) if self.cfg.features=="random" else SELECTED)
        tensors=[]
        for name in ("fast","slow","state"):
            value,mask=self._prepare(name,np.asarray(inputs[name]),symbols)
            if name=="fast":
                value,mask=value[...,list(selected)],mask[...,list(selected)]
                if self.cfg.minute=="aggregate":
                    ev,em=self._prepare("extra",np.asarray(inputs["extra"]),symbols)
                    value,mask=np.concatenate((value,ev),-1),np.concatenate((mask,em),-1)
            tensors.extend(torch.from_numpy(np.ascontiguousarray(v)).to(self.device) for v in (value,mask))
        if self.cfg.minute=="encoder":
            tensors.extend(torch.from_numpy(np.ascontiguousarray(v)).to(self.device) for v in self._prepare("minute",np.asarray(inputs["minute"]),symbols))
        else:
            tensors.extend((None,None))
        tensors.append(torch.from_numpy(self.representatives).to(self.device))
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
            output=self.model(*tensors)
        signal=output["signal"].float().cpu().numpy()
        scale=np.maximum(past_scale4,self.floors[symbols])*np.sqrt(self.cfg.horizon/4)
        return {"signal_level":signal,"return_estimate":signal*scale if self.cfg.label!="path" else None,
                "causal_scale":scale,"path_auxiliary":output["path"].float().cpu().numpy(),
                "semantics":"signed_square_pressure_proxy" if self.cfg.label=="path" else f"risk_adjusted_return_{self.cfg.horizon}h",
                "path_auxiliary_trained":self.cfg.label in ("joint","path")}


class LevelSignalState:
    def __init__(self,half_life_hours=2.):
        if half_life_hours<=0:
            raise ValueError("positive half life required")
        self.half_life=half_life_hours
        self.value=None
        self.time=None

    def update(self,time,signal):
        time=np.datetime64(time,"ns")
        if np.isnat(time) or not np.isfinite(signal).all():
            raise ValueError("finite completed timestamp and signal required")
        signal=np.asarray(signal,np.float64)
        if self.time is not None:
            if time<=self.time or signal.shape!=self.value.shape:
                raise ValueError("strictly increasing time and stable symbol order required")
            delta=float((time-self.time)/np.timedelta64(1,"h"))
            alpha=1-2**(-delta/self.half_life)
            self.value=(1-alpha)*self.value+alpha*signal
        else:
            self.value=signal.copy()
        self.time=time
        return self.value.copy()
