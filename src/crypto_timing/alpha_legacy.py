"""Explicit adapters; legacy native contracts remain separately identified."""
from pathlib import Path
import json
import numpy as np
from .alpha_data import ROOT,BASE,save_json
from .review_data import identity
from .review_inference import ReviewPredictor

def anchor():
    out=ROOT/'v5_anchor_return4.npy'
    if out.exists():return np.load(out)
    dates=np.load(BASE/'decision_time.npy');ns=12;result=np.full((len(dates),ns),np.nan,np.float32);used=[]
    # Cache genuine v5 output wherever available; fill only missing clocks by checkpoint inference.
    stages=[('2025-01-01','2025-03-01','oof_f1_early'),('2025-03-01','2025-07-01','oof_f1_late'),('2025-07-01','2026-02-01','f1_selected'),('2026-02-01','2026-09-25','f2_selected')]
    arrays={name:np.load(BASE/f'{name}_dynamic.npy' if name!='minute' else BASE/'minute.npy',mmap_mode='r') for name in ('fast','slow','state','minute')};scale=np.load(BASE/'targets.npy',mmap_mode='r')[...,7]
    for begin,end,stage in stages:
        wanted=np.flatnonzero((dates>=np.datetime64(begin))&(dates<np.datetime64(end))&(np.arange(len(dates))<len(dates)-1));each=[]
        for seed in (20261004,20261005,20261006):
            folder=Path('outputs/review_v5/confirm')/(f'{stage}_{seed}' if 'selected' in stage else stage if seed==20261004 else f'{stage}_seed{seed}')
            values=np.full((len(dates),ns),np.nan,np.float32)
            for split in ('validation','replay'):
                f=np.load(folder/f'{split}.npz');values[f['hours']]=f['signal']*f['scale'];used.append(folder/f'{split}.npz')
            missing=wanted[~np.isfinite(values[wanted]).all(1)]
            if len(missing):
                predictor=ReviewPredictor(folder/'final.pt')
                for start in range(0,len(missing),24):
                    hh=missing[start:start+24];hi=np.repeat(hh,ns);si=np.tile(np.arange(ns),len(hh));features={'state':np.asarray(arrays['state'][hi,si])}
                    for name,multiple,length in (('fast',12,144),('slow',1,168),('minute',60,180)):
                        idx=(hi*multiple+multiple-1)[:,None]-np.arange(length-1,-1,-1);features[name]=np.asarray(arrays[name][idx,si[:,None]])
                    r=predictor.predict(features,si,np.asarray(scale[hi,si]));values[hh]=r['return_estimate'].reshape(-1,ns)
            each.append(values[wanted]);used.append(folder/'final.pt')
        result[wanted]=np.mean(each,0);print(f'v5 anchor {stage} {len(wanted)} clocks',flush=True)
    np.save(out,result);save_json(ROOT/'v5_anchor_manifest.json',{'inputs':identity(used),'output':identity([out]),'source':identity([Path(__file__)]),'target':'native delayed5m 4h expected simple return; OOF projection required for other horizons','switches':[{'begin':b,'end':e,'run':r} for b,e,r in stages]});return result

def legacy_scores():
    root=Path('outputs/redesign/evaluation');dates=np.load(Path('outputs/redesign/cache_v3/decision_time.npy'))
    assert json.loads(Path('outputs/redesign/cache_v3/manifest.json').read_text(encoding='utf-8'))['symbols']==json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))['symbols']
    common=np.load(BASE/'decision_time.npy');result=np.full((len(common),12),np.nan,np.float32);used=[]
    for split in ('selection','historical'):
        path=root/f'{split}_scores.npz';data=np.load(path);hh=data['keys']//12;si=data['keys']%12;indices=np.searchsorted(common,dates[hh]);assert np.array_equal(common[indices],dates[hh]);result[indices,si]=data['network'];used.append(path)
    save_json(ROOT/'legacy_adapter.json',{'inputs':identity(used),'meaning':'saved native v3 raw-return scores, aligned by UTC/asset; original inputs remain older5m, so bridge only, not fair model ranking','missing_hours':int((~np.isfinite(result).all(1)).sum())})
    return result

def native_curve():
    from .evaluation import _funding_per_hour
    root=Path('outputs/redesign/cache_v3');dates=np.load(root/'decision_time.npy');manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));f=np.load('outputs/redesign/evaluation/historical_scores.npz');keys=f['keys'];hour=keys//12;sym=keys%12
    target=np.load(root/'targets.npy',mmap_mode='r');funding=_funding_per_hour({'manifest':manifest,'decision_time':dates},Path('D:/Trading/practical_crypto_strategy/data/parquet'))
    selection=json.loads(Path('outputs/redesign/evaluation/selection.json').read_text(encoding='utf-8'));threshold=.0012;expected=json.loads(Path('outputs/redesign/evaluation/historical_diagnostic.json').read_text(encoding='utf-8'))['scores']['network']['phases'];checks=[];arrays={}
    for phase in range(4):
        keep=dates[hour].astype('datetime64[h]').astype(np.int64)%4==phase;hh=hour[keep];ss=sym[keep];score=f['network'][keep].reshape(-1,12);position=np.where(score>threshold,1.,np.where(score<-threshold,-1.,0.));prior=np.vstack((np.zeros((1,12)),position[:-1]));turn=np.abs(position-prior)
        returns=(position*target[hh,ss,0].reshape(-1,12)-position*funding[hh,ss].reshape(-1,12)-.0006*turn).mean(1);returns[-1]-=.0006*abs(position[-1]).mean();nav=np.cumprod(1+returns)
        arrays[f'dates_{phase}']=dates[hh.reshape(-1,12)[:,0]];arrays[f'nav_{phase}']=nav
        checks.append({'phase':phase,'recomputed_return':float(nav[-1]-1),'saved_return':expected[phase]['cumulative_return'],'difference':float(nav[-1]-1-expected[phase]['cumulative_return'])})
    np.savez_compressed(ROOT/'v3_native_curve.npz',**arrays);save_json(ROOT/'v3_native_reproduction.json',{'checks':checks,'cost':.0006,'threshold':threshold,'note':'native older5m inputs, four separate account phases, original6bp and event funding; not fair leaderboard'})
