"""Checkpoint inference without labels; caller supplies completed features and matured baseline."""
from pathlib import Path
import numpy as np
import torch
from .review_model import ReviewConfig,ReviewNetwork
from .mechanism_training import normalize

class ReviewPredictor:
    def __init__(self,path:Path,device='cuda'):
        self.device=torch.device(device);self.saved=torch.load(path,map_location=self.device,weights_only=False)
        self.cfg=ReviewConfig(**self.saved['config']);self.model=ReviewNetwork(self.cfg).to(self.device).eval()
        self.model.load_state_dict(self.saved['model'])
    @torch.inference_mode()
    def predict(self,features,asset_ids,past_scale,baseline=None):
        ids=np.asarray(asset_ids,int);past_scale=np.asarray(past_scale,float)
        n=len(ids);ns=len(self.saved['floors'])
        if ids.shape!=(n,) or np.any(ids<0) or np.any(ids>=ns) or past_scale.shape!=(n,) or np.any(~np.isfinite(past_scale)) or np.any(past_scale<=0):raise ValueError('invalid asset/known scale')
        tensors=[]
        for name,length in (('fast',144),('slow',self.cfg.slow_hours),('state',None),('minute',180)):
            x=np.asarray(features[name]);dim={'fast':20,'slow':10,'state':24,'minute':12}[name]
            if x.shape!=((n,dim) if length is None else (n,length,dim)):raise ValueError(f'{name} completed-window shape')
            scaler=self.saved['scalers']['minute' if name=='minute' else f'dynamic_{name}'];c=np.asarray(scaler['center'])[ids];s=np.asarray(scaler['scale'])[ids]
            if length is not None:c=c[:,None];s=s[:,None]
            value,mask=normalize(x,c,s)
            if name=='fast':
                keep=[j for j in range(16) if self.cfg.features not in ('no_body','no_wicks') or j not in ((1,) if self.cfg.features=='no_body' else (3,4))]
                value,mask=value[...,keep],mask[...,keep]
            if name=='state' and self.cfg.events:
                event=np.asarray(features['events']);sc=self.saved['event_scaler'];ev,em=normalize(event,np.asarray(sc['center'])[ids],np.asarray(sc['scale'])[ids]);value,mask=np.concatenate((value,ev),-1),np.concatenate((mask,em),-1)
            tensors.extend((torch.tensor(value,dtype=torch.float32,device=self.device),torch.tensor(mask,dtype=torch.float32,device=self.device)))
        if self.cfg.baseline and baseline is None:raise ValueError('matured causal baseline required; no future-label fallback')
        base=np.zeros(n,np.float32) if not self.cfg.baseline else np.asarray(baseline,np.float32)
        if base.shape!=(n,) or not np.isfinite(base).all():raise ValueError('invalid matured baseline')
        tensors.append(torch.tensor(base,device=self.device))
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):out=self.model(*tensors)
        signal=out['signal'].float().cpu().numpy();scale=np.maximum(past_scale,np.asarray(self.saved['floors'])[ids])
        return {'signal_level':signal,'return_estimate':signal*scale,'risk_scale':scale,'delta':out['delta'].float().cpu().numpy(),'target':'4h simple return at decision+5m open, standardized by known risk'}
