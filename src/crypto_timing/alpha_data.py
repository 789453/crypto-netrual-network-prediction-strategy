"""Unified next-minute horizon/visibility contract built on the existing v5 cache."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .review_data import identity
from .review_inference import ReviewPredictor

BASE=Path('outputs/mechanism/cache_v4')
ROOT=Path('outputs/alpha_strategy')
SOURCE=Path('D:/Trading/practical_crypto_strategy/data/parquet')
HORIZONS=np.array([2,4,8,12])
ENCODER=Path('outputs/review_v5/confirm/oof_f1_early/final.pt')

def save_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def build_market():
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    symbols=manifest['symbols'];dates=np.load(BASE/'decision_time.npy');n=len(dates)
    files=[SOURCE/s/'1m.parquet' for s in symbols]+[SOURCE/'derivatives'/f'{s}.parquet' for s in symbols]
    contract={'source':identity(files),'base':identity(list(BASE.glob('*.npy'))+[BASE/'manifest.json']),'encoder':identity([ENCODER]),'builder':identity([Path(__file__)])}
    if (ROOT/'market_manifest.json').exists():
        old=json.loads((ROOT/'market_manifest.json').read_text(encoding='utf-8'))
        assert old['contract']==contract,'inputs changed; use a new output identity'
        return old
    price=np.zeros((n,len(symbols)),np.float64);minute=np.zeros((n-1,60,len(symbols)),np.float32)
    fcash=np.zeros_like(price);frate=np.zeros_like(price);expected=np.zeros_like(price);audit={};fm=np.zeros((n-1,61,len(symbols)),np.float32)
    execution=dates+np.timedelta64(1,'m')
    for j,s in enumerate(symbols):
        f=pd.read_parquet(SOURCE/s/'1m.parquet',columns=['date','open'])
        times=f.date.to_numpy(dtype='datetime64[ns]');op=f.open.to_numpy(float)
        ix=np.searchsorted(times,execution[:-1]);assert np.array_equal(times[ix],execution[:-1])
        price[:-1,j]=op[ix];price[-1,j]=op[-1];execution[-1]=times[-1]
        grid=ix[:,None]+np.arange(60)[None];minute[:,:,j]=op[np.minimum(grid,len(op)-1)]
        der=pd.read_parquet(SOURCE/'derivatives'/f'{s}.parquet')
        events=der.loc[der.funding_rate.notna()].sort_values('date')
        et=events.date.dt.tz_convert(None).to_numpy(dtype='datetime64[ns]');rate=events.funding_rate.to_numpy(float);mark=events.funding_mark_price.to_numpy(float,copy=True)
        direct=np.isfinite(mark)&(mark>0)
        bars=der.loc[der.mark_open.notna()].copy();bt=bars.date.dt.tz_convert(None).to_numpy(dtype='datetime64[ns]');bi=np.searchsorted(bt,et.astype('datetime64[h]').astype('datetime64[ns]'))
        match=(bi<len(bt));match[match]&=bt[bi[match]]==et[match].astype('datetime64[h]').astype('datetime64[ns]')
        replace_mark=(~direct)&match;mark[replace_mark]=bars.mark_open.to_numpy()[bi[replace_mark]]
        proxy=~(np.isfinite(mark)&(mark>0));mark[proxy]=op[np.minimum(np.searchsorted(times,et[proxy],side='right')-1,len(op)-1)]
        k=np.searchsorted(execution,et,side='right')-1;valid=(k>=0)&(k<n-1)&(et<execution[-1])
        np.add.at(fcash[:,j],k[valid],rate[valid]*mark[valid]);np.add.at(frate[:,j],k[valid],rate[valid])
        elapsed=np.ceil((et[valid]-execution[k[valid]])/np.timedelta64(1,'m')).astype(int);np.add.at(fm[:,:,j],(k[valid],np.minimum(elapsed,60)),rate[valid]*mark[valid])
        # Only settled events visible by the decision; never use next settled rate.
        prev=np.searchsorted(et,dates,side='right')-1;known=prev>=1
        interval=(et[1:]-et[:-1])/np.timedelta64(1,'h')
        expected[known,j]=rate[prev[known]]/np.maximum(interval[prev[known]-1],1)
        rounded_interval=np.round(interval,3)
        audit[s]={'events':int(valid.sum()),'observed_interval_hours':sorted(map(float,np.unique(rounded_interval))),'direct_funding_mark':int((direct&valid).sum()),'hour_mark_open_approximation':int((replace_mark&valid).sum()),'minute_trade_price_proxy':int((proxy&valid).sum()),'last_price_minute':str(times[-1])}
    target=np.full((n,len(symbols),4),np.nan,np.float32)
    for a,h in enumerate(HORIZONS):
        rows=np.flatnonzero((np.arange(n)+h<n)&(dates+np.timedelta64(int(h),'h')+np.timedelta64(1,'m')<execution[-1]))
        target[rows,:,a]=price[rows+h]/price[rows]-1
    state=np.load(BASE/'state_dynamic.npy',mmap_mode='r');valid=np.isfinite(state).all(-1);valid[:723]=False;valid[-1]=False
    # Tradable flow ends on price visibility, not the 12h maturity of research labels.
    for name,value in {'price':price,'minute_open':minute,'funding_cash_minute':fm,'funding_cash_unit':fcash,'funding_rate':frate,'expected_funding_hour':expected,'target':target,'valid':valid,'execution':execution}.items():np.save(ROOT/f'{name}.npy',value)
    out={'contract':contract,'symbols':symbols,'hours':n,'entry':'decision+1m open; last endpoint is final observed open','funding':audit,'limitations':'maker fee scenarios assume next-minute open fills; no queue/spread proof; no historical maintenance-margin tiers'}
    out['content']=identity(list(ROOT.glob('*.npy')));save_json(ROOT/'market_manifest.json',out);return out

def encode():
    out=ROOT/'representation.npy'
    if out.exists():return
    predictor=ReviewPredictor(ENCODER)
    arrays={k:np.load(BASE/f'{k}_dynamic.npy' if k!='minute' else BASE/'minute.npy',mmap_mode='r') for k in ('fast','slow','state','minute')}
    n,ns=arrays['state'].shape[:2];result=np.zeros((n,ns,128),np.float32);score=np.zeros((n,ns),np.float32)
    import torch
    for start in range(720,n,24):
        hours=np.arange(start,min(n,start+24));hi=np.repeat(hours,ns);si=np.tile(np.arange(ns),len(hours));features={}
        for name,multiple,length in (('fast',12,144),('slow',1,168),('minute',60,180)):
            idx=(hi*multiple+multiple-1)[:,None]-np.arange(length-1,-1,-1)
            features[name]=np.asarray(arrays[name][idx,si[:,None]])
        features['state']=np.asarray(arrays['state'][hi,si]);tensors=[]
        from .mechanism_training import normalize
        for name in ('fast','slow','state','minute'):
            sc=predictor.saved['scalers']['minute' if name=='minute' else 'dynamic_'+name];c=np.asarray(sc['center'])[si];s=np.asarray(sc['scale'])[si]
            if name!='state':c=c[:,None];s=s[:,None]
            v,m=normalize(features[name],c,s)
            if name=='fast':v,m=v[...,:16],m[...,:16]
            tensors.extend((torch.as_tensor(v,dtype=torch.float32,device='cuda'),torch.as_tensor(m,dtype=torch.float32,device='cuda')))
        tensors.append(torch.zeros(len(hi),device='cuda'))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):r=predictor.model(*tensors)
        result[hours]=r['representation'].float().cpu().numpy().reshape(-1,ns,128);score[hours]=r['signal'].float().cpu().numpy().reshape(-1,ns)
        if start%2400<24:print(f'encode {start}/{n}',flush=True)
    np.save(out,result);np.save(ROOT/'frozen_encoder_signal.npy',score)
    save_json(ROOT/'representation_manifest.json',{'encoder':identity([ENCODER]),'outputs':identity([out,ROOT/'frozen_encoder_signal.npy']),'source':identity([Path(__file__)])})

def summaries():
    out=ROOT/'summary_features.npy'
    if out.exists():return np.load(out,mmap_mode='r')
    state=np.load(BASE/'state_dynamic.npy',mmap_mode='r');fast=np.load(BASE/'fast_dynamic.npy',mmap_mode='r');slow=np.load(BASE/'slow_dynamic.npy',mmap_mode='r')
    n,ns=state.shape[:2];x=[np.nan_to_num(state)]
    for arr,multiple,lengths in ((fast,12,(12,48,144)),(slow,1,(4,24,168))):
        end=np.arange(n)*multiple+multiple-1
        for length in lengths:
            # Complete trailing sequence summaries; include direction, pressure and distribution.
            cs=np.concatenate((np.zeros((1,ns,arr.shape[-1])),np.cumsum(np.nan_to_num(arr),axis=0)))
            mean=(cs[end+1]-cs[np.maximum(end+1-length,0)])/length
            x.append(mean[...,:19] if multiple==12 else mean)
        x.append(np.nan_to_num(arr[end,...,:19] if multiple==12 else arr[end]))
    result=np.concatenate(x,-1).astype(np.float32);np.save(out,result);return result

def scale_at(cut):
    raw=np.load(BASE/'targets.npy',mmap_mode='r');floor=np.nanquantile(raw[720:max(721,cut-13),:,7],.05,axis=0)
    risk=np.maximum(raw[...,7],floor)
    return (risk[...,None]*np.sqrt(HORIZONS/4)).astype(np.float32)
