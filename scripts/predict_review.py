from pathlib import Path
import json,numpy as np,torch
from crypto_timing.review_inference import ReviewPredictor
from crypto_timing.review_analysis import apply_calibrator,smooth
from crypto_timing.mechanism_data import base_features,aggregate
import pandas as pd

def main():
    torch.set_num_threads(4);root=Path('outputs/review_v5');base=Path('outputs/mechanism/cache_v4');manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'));h=len(np.load(base/'decision_time.npy'))-1
    features={}
    for name,length,multiple in (('fast',144,12),('slow',168,1),('minute',180,60)):
        file='minute' if name=='minute' else f'{name}_dynamic';arr=np.load(base/f'{file}.npy',mmap_mode='r');end=(h+1)*multiple;features[name]=np.transpose(arr[end-length:end],(1,0,2))
    features['state']=np.load(base/'state_dynamic.npy',mmap_mode='r')[h];features['events']=np.load(root/'cache/events.npy',mmap_mode='r')[h]
    # Current risk is known; never obtain it from future-label eligibility.
    raw=np.load(base/'targets.npy',mmap_mode='r');scale=raw[h,:,7];assert np.isfinite(scale).all()
    predictions=[]
    for seed in (20261004,20261005,20261006):
        p=ReviewPredictor(root/'confirm'/f'f2_selected_{seed}'/'final.pt');assert not p.cfg.baseline,'supply a streaming matured baseline for a baseline-enabled model';predictions.append(p.predict(features,np.arange(12),scale))
    signal=np.mean([p['signal_level'] for p in predictions],0);estimate=np.mean([p['return_estimate'] for p in predictions],0)
    result={'decision_utc':str(np.load(base/'decision_time.npy')[h]),'symbols':manifest['symbols'],'signal_level':signal.tolist(),'return_estimate':estimate.tolist(),'future_labels_available':bool(np.isfinite(raw[h,:,1]).any()),'cuda':torch.cuda.get_device_name(0),'contract':'completed feature arrays + current past-risk only; raw-source causal construction responsibility remains explicit'}
    if (root/'second_stage_f2.json').exists():
        meta=json.loads((root/'second_stage_f2.json').read_text(encoding='utf-8'));past=np.arange(h-71,h+1);preds=[]
        for seed in (20261004,20261005,20261006):
            predictor=ReviewPredictor(root/'confirm'/f'f2_selected_{seed}'/'final.pt');rows=[]
            for start in range(0,len(past),12):
                hh=past[start:start+12];xx={}
                for name,length,multiple in (('fast',144,12),('slow',168,1),('minute',180,60)):
                    file='minute' if name=='minute' else f'{name}_dynamic';arr=np.load(base/f'{file}.npy',mmap_mode='r')
                    xx[name]=np.stack([np.transpose(arr[(j+1)*multiple-length:(j+1)*multiple],(1,0,2)) for j in hh]).reshape(-1,length,arr.shape[-1])
                xx['state']=np.load(base/'state_dynamic.npy',mmap_mode='r')[hh].reshape(-1,24)
                pp=predictor.predict(xx,np.tile(np.arange(12),len(hh)),np.asarray(raw[hh,:,7]).reshape(-1));rows.append(pp['signal_level'].reshape(-1,12))
            preds.append(np.concatenate(rows))
        advantage=np.mean(preds,0);state=np.load(base/'state_dynamic.npy',mmap_mode='r')[past];times=np.load(base/'decision_time.npy')[past]
        calibrated=apply_calibrator({'delta':advantage,'baseline':np.zeros_like(advantage)},state,meta['calibrator']);scale=np.maximum(raw[past,:,7],np.asarray(predictor.saved['floors']))
        values=calibrated if meta['filter']['unit']=='normalized' else calibrated*scale;filtered=smooth(values,times,meta['filter']['half_life']);filtered=filtered if meta['filter']['unit']=='normalized' else filtered/scale
        result.update(calibrated_level=calibrated[-1].tolist(),filtered_level=filtered[-1].tolist(),filtered_return_estimate=(filtered[-1]*scale[-1]).tolist(),filter=meta['filter'],filter_warm_start_hours=72,frozen_oof_calibration=True)
    assert not result['future_labels_available'] and np.isfinite(signal).all();(root/'latest_inference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
