"""Label-free simultaneous twelve-coin horizon inference from completed representations."""
from pathlib import Path
import json
import numpy as np
import torch
from .alpha_models import HorizonReadout
from .alpha_data import HORIZONS
from .alpha_policy import apply_calibration,target_weights,Policy

class AlphaPredictor:
    def __init__(self,folder:Path,calibration,policy,device='cuda',kind='joint'):
        self.device=torch.device(device);self.models=[];self.calibration=calibration;self.policy=Policy(**policy)
        for seed in (20261004,20261005,20261006):
            ck=torch.load(folder/f'{kind}_{seed}.pt',map_location=self.device,weights_only=False);m=HorizonReadout(kind).to(self.device).eval();m.load_state_dict(ck['model']);self.models.append((m,ck))
    @torch.inference_mode()
    def predict_completed(self,representation_history,past_scale4,covariance,beta,known_funding_hour):
        x=np.asarray(representation_history,np.float32);v=np.asarray(past_scale4,float)
        if x.shape!=(4,12,128) or v.shape!=(12,) or not np.isfinite(x).all() or not np.isfinite(v).all() or (v<=0).any():raise ValueError('complete simultaneous history and known past risk required')
        preds=[];risk=None
        for model,ck in self.models:
            risk=np.maximum(v,np.asarray(ck['training_risk_floor']))[:,None]*np.sqrt(HORIZONS/4)
            z=np.clip((x-np.asarray(ck['center']))/np.asarray(ck['scale']),-8,8)
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                p=model(torch.as_tensor(z[None],device=self.device),torch.ones((1,4,12),dtype=torch.bool,device=self.device),torch.as_tensor(risk[None],dtype=torch.float32,device=self.device))
            preds.append(p.float().cpu().numpy())
        raw=np.mean(preds,0);cal=apply_calibration(raw,risk[None],self.calibration);j=list(HORIZONS).index(self.policy.horizon)
        weights=target_weights(cal[0,:,j],np.asarray(known_funding_hour),np.asarray(covariance),np.asarray(beta),self.policy)
        return {'raw_returns':raw[0].tolist(),'calibrated_returns':cal[0].tolist(),'target_weights_if_flat':weights.tolist(),'target_horizons_hours':HORIZONS.tolist(),'risk_scale':risk.tolist(),'policy_horizon_hours':self.policy.horizon,'requires_position_state':'existing holdings must use the same decide() hysteresis/expiry logic'}
