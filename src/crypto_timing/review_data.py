"""v5: causal event inputs, disjoint targets, observable eligibility and provenance."""
from __future__ import annotations
import hashlib,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .cache import _sha256
from .mechanism_data import base_features,aggregate,lag,validate
from .mechanism_training import MechanismStore,FOLDS,scaled_labels,fit_scaler,normalize

EVENT_NAMES=("shock_shape","shock_age","repair_shape","activity_at_shock","pressure_before","pressure_after","continuation","shock_present")
SEGMENT_NAMES=("log_return_0_1h","log_return_1_2h","log_return_2_3h","log_return_3_4h","log_return_4_8h")

def identity(paths):
    return {str(p):_sha256(Path(p)) for p in paths}

def training_identity():
    files=[Path('src/crypto_timing')/name for name in ('review_data.py','review_model.py','review_training.py','mechanism_training.py','mechanism_data.py','mechanism_model.py','redesign_model.py','cache.py')]
    return identity(files)

def event_features(op,cl,q,pressure,sigma,qbase):
    """Only the completed trailing 180 minutes; no retrospectively selected turn."""
    nh=len(op)//60
    out=np.full((nh,8),np.nan,np.float32)
    r=np.log(cl/lag(cl)); z=r/np.maximum(sigma,1e-7)
    for h in range(3,nh):
        end=h*60+60; begin=end-180
        rows=z[begin:end]
        if not np.isfinite(rows).all(): continue
        k=begin+int(np.argmax(np.abs(rows))); sign=np.sign(z[k])
        repair=-sign*np.log(cl[end-1]/cl[k])/max(sigma[k],1e-7)
        before=np.nanmean(pressure[max(begin,k-15):k+1]); after=np.nanmean(pressure[k:end])
        continuation=sign*np.log(cl[end-1]/cl[max(k,end-16)])/max(sigma[k],1e-7)
        out[h]=[np.arcsinh(z[k]),(end-1-k)/180,np.arcsinh(repair),np.log((q[k]+1e-9)/(qbase[k]+1e-9)),before,after,np.arcsinh(continuation),float(abs(z[k])>=3)]
    return out

def disjoint_targets(op):
    nh=len(op)//60; e=np.arange(nh)*60+65
    out=np.full((nh,5),np.nan); nxt=np.full(nh,np.nan)
    good=np.flatnonzero(e+480<len(op)); idx=e[good]
    for j,(a,b) in enumerate(((0,60),(60,120),(120,180),(180,240),(240,480))):
        out[good,j]=np.log(op[idx+b]/op[idx+a])
    one=np.flatnonzero(e+60<len(op)); nxt[one]=op[e[one]+60]/op[e[one]]-1
    return out.astype(np.float32),nxt.astype(np.float32)

def build_overlay(source:Path,base:Path,root:Path):
    root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    source_hash=identity([source/s/'1m.parquet' for s in manifest['symbols']])
    for s in manifest['symbols']:
        assert source_hash[str(source/s/'1m.parquet')]==manifest['audit'][s]['source_sha256'],'minute source changed; rebuild v4 first'
    base_files=list(base.glob('*.npy'))+[base/'manifest.json']
    contract={'source':source_hash,'base_cache':identity(base_files),'builder':identity([Path(__file__),'src/crypto_timing/mechanism_data.py','src/crypto_timing/cache.py'])}
    if (root/'manifest.json').exists():
        saved=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if saved['contract']!=contract: raise ValueError('v5 overlay source/cache contract changed; use a new directory')
        assert identity([root/p for p in saved['content_sha256']])=={str(root/p):v for p,v in saved['content_sha256'].items()}
        return saved
    nh,ns=manifest['hours'],len(manifest['symbols'])
    events=np.full((nh,ns,8),np.nan,np.float32); seg=np.full((nh,ns,5),np.nan,np.float32); nxt=np.full((nh,ns),np.nan,np.float32)
    checks={}; extremes=[]
    for si,s in enumerate(manifest['symbols']):
        frame=pd.read_parquet(source/s/'1m.parquet'); validate(frame)
        op,cl,q=map(lambda k:frame[k].to_numpy(float),('open','close','quote_volume'))
        mx,ms,m=base_features(frame,60)
        qb=pd.Series(q).ewm(span=1440,min_periods=1440,adjust=False).mean().shift(1).to_numpy()
        events[:,si]=event_features(op,cl,q,m['pressure'],ms,qb)
        seg[:,si],nxt[:,si]=disjoint_targets(op)
        # An actual raw-source cut reproduces local minute/5m fields at a decision.
        cut=60*13000
        short=frame.iloc[:cut]; sx,ss,sm=base_features(short,60)
        expected=np.column_stack((np.arcsinh(sx[:,0]/ss),np.arcsinh(sx[:,1]/ss),np.arcsinh(sx[:,2]/ss),sx[:,5],sx[:,8],sx[:,6],sx[:,7],np.arcsinh(sx[:,9]/ss),np.log(ss),sx[:,10],sx[:,18],np.arcsinh(lag(sm['pressure'])*sm['r']/ss)))
        cached=np.load(base/'minute.npy',mmap_mode='r')[cut-180:cut,si]
        np.testing.assert_allclose(expected[-180:],cached,rtol=2e-6,atol=2e-6,equal_nan=True)
        fx,fs,_=base_features(aggregate(short,5),12)
        full5=np.load(base/'fast_raw.npy',mmap_mode='r')[:cut//5,si,:11]
        np.testing.assert_allclose(fx[-144:,:11],full5[-144:],rtol=2e-6,atol=2e-6,equal_nan=True)
        prefix_event=event_features(op[:cut],cl[:cut],q[:cut],m['pressure'][:cut],ms[:cut],qb[:cut])[-1]
        np.testing.assert_array_equal(prefix_event,events[cut//60-1,si])
        raw=np.load(base/'targets.npy',mmap_mode='r')[:,si]
        finite=np.isfinite(seg[:,si,:4]).all(1)&np.isfinite(raw[:,1])
        np.testing.assert_allclose(np.expm1(seg[finite,si,:4].astype(float).sum(1)),raw[finite,1],rtol=3e-5,atol=2e-7)
        checks[s]={'raw_prefix_minutes':cut,'minute_180_and_fast_144_equal':True,'event_equal':True,'target_compounding_equal':True}
        date=frame.date.to_numpy(dtype='datetime64[ns]'); crash=(date>=np.datetime64('2025-10-10T17'))&(date<np.datetime64('2025-10-11T01'))
        ix=np.flatnonzero(crash); moves=np.log(op[ix+1]/op[ix]); k=ix[np.argmax(np.abs(moves))]
        extremes.append({'symbol':s,'minute_utc':str(date[k]),'open':float(op[k]),'next_open':float(op[k+1]),'simple_move':float(op[k+1]/op[k]-1),'quote_volume':float(q[k]),'trades':int(frame.trade_count.iloc[k]),'internal_ohlc_valid':True,'truth_status':'internally legal source record; no independent exchange-tick verification; retained'})
        print(json.dumps({'event':'review_overlay_asset','symbol':s,'prefix_verified':True}),flush=True)
    for name,value in (('events',events),('segments',seg),('next_return',nxt)): np.save(root/f'{name}.npy',value)
    result={'version':5,'contract':contract,'event_names':EVENT_NAMES,'segment_names':SEGMENT_NAMES,'prefix_checks':checks,'extreme_source_audit':extremes,'content_sha256':{p.name:_sha256(p) for p in root.glob('*.npy')}}
    (root/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def causal_baseline(labels,states,valid,delay=5,half_life=720,ridge=1000):
    """Per-asset continuous-state ridge; only matured labels enter sufficient stats."""
    n,ns=labels.shape
    z=np.stack((np.ones_like(labels),states[...,6]+3,states[...,9],states[...,16],states[...,12]),-1)
    z=np.nan_to_num(np.clip(z,-3,3)); z[...,0]=1
    gram=np.zeros((ns,5,5)); rhs=np.zeros((ns,5)); result=np.zeros_like(labels,dtype=np.float32)
    decay=2**(-1/half_life); eye=np.eye(5)*ridge
    for h in range(n):
        gram*=decay; rhs*=decay; matured=h-delay
        if matured>=0:
            good=valid[matured]&np.isfinite(labels[matured]); zz=z[matured]*good[:,None]
            gram+=zz[:,:,None]*zz[:,None,:]; rhs+=zz*np.nan_to_num(labels[matured])[:,None]
        coef=np.linalg.solve(gram+eye,rhs)
        result[h]=(coef*z[h]).sum(-1)
    return result

class ReviewStore(MechanismStore):
    def __init__(self,base,overlay,device,train_end,val_end,replay_end):
        super().__init__(base,'f1',device)
        self.fold=f'{train_end}/{val_end}/{replay_end}'
        te,ve,re=map(np.datetime64,(train_end,val_end,replay_end))
        self.train_stop=int(np.searchsorted(self.dates,te)); stop=self.train_stop-9
        self.floors=np.nanquantile(self.raw[720:stop,:,7],.05,axis=0).astype(np.float32)
        self.labels,self.scale=scaled_labels(self.raw,self.floors); self.labels1=self.labels
        state=np.load(base/'state_dynamic.npy',mmap_mode='r')
        self.valid=np.isfinite(state).all(-1)&np.isfinite(self.raw[...,:8]).all(-1); self.valid[:720]=False
        maturity=self.dates+np.timedelta64(485,'m')
        periods={'train':(self.dates<te)&(maturity<=te),'validation':(self.dates>=te)&(maturity<=ve),'replay':(self.dates>=ve)&(maturity<=re)}
        self.splits={k:np.flatnonzero(self.valid.any(1)&p) for k,p in periods.items()}
        self.quality_counts={k:{'observable_hours':len(h),'legacy_future_filtered_hours':int((self.raw[h,:,8]==0).all(1).sum()),'valid_asset_samples':int(self.valid[h].sum())} for k,h in self.splits.items()}
        self.baseline=causal_baseline(self.labels[...,1],state,self.valid)
        self.events=np.load(overlay/'events.npy',mmap_mode='r')
        ec,es=fit_scaler(self.events,stop,(1,4,5,7)); self.event_scaler={'center':ec.tolist(),'scale':es.tolist()}
        ev,em=normalize(self.events,ec,es); self.event_gpu=(torch.from_numpy(ev).to(device),torch.from_numpy(em).to(device))
        self.segments=np.load(overlay/'segments.npy')
        self.segment_scaled=self.segments/(self.scale[...,None]*np.sqrt(np.array([1,1,1,1,4])/4))
        h=self.splits['train']; good=self.valid[h]
        q=self.labels[h,:,4][good]; self.qscale=float(max(np.median(np.abs(q)),.02))
        path=self.labels[...,3:6].copy(); path[...,1]=np.arcsinh(path[...,1]/self.qscale)
        self.path_mean=path[h][good].mean(0); self.path_sd=np.maximum(path[h][good].std(0),.01)
        self.path_shaped=(path-self.path_mean)/self.path_sd
        segt=np.arcsinh(self.segment_scaled); self.segment_mean=segt[h][good].mean(0); self.segment_sd=np.maximum(segt[h][good].std(0),.01)
        self.segment_target=(segt-self.segment_mean)/self.segment_sd
        self.tail_threshold=float(np.quantile(np.abs(self.labels[h,:,1][good]),.99))
        self.cache_identity=json.loads((overlay/'manifest.json').read_text(encoding='utf-8'))
        self.gpu,self.scalers,self.current_prep={},{},None

    def batch(self,hours,cfg):
        base_cfg=replace(cfg,features='original') if cfg.features in ('no_body','no_wicks') else cfg
        x,y=super().batch(hours,base_cfg)
        tensors=list(x); si=np.tile(np.arange(self.ns),len(hours)); hi=np.repeat(hours,self.ns)
        if cfg.features in ('no_body','no_wicks'):
            keep=[j for j in range(16) if j not in ((1,) if cfg.features=='no_body' else (3,4))]
            tensors[0],tensors[1]=tensors[0][...,keep],tensors[1][...,keep]
        if cfg.events:
            ev,em=[v[torch.as_tensor(hi,device=self.device),torch.as_tensor(si,device=self.device)].float() for v in self.event_gpu]
            tensors[4]=torch.cat((tensors[4],ev),-1); tensors[5]=torch.cat((tensors[5],em),-1)
        baseline=self.baseline[hours] if cfg.baseline else np.zeros((len(hours),self.ns),np.float32)
        tensors[-1]=torch.as_tensor(baseline.reshape(-1),device=self.device)
        rawy=self.labels[hours].reshape(-1,8)
        target={'y':y[:,1],'raw_path':y[:,3:6],'path':torch.as_tensor(self.path_shaped[hours].reshape(-1,3),device=self.device),
                'segments':torch.as_tensor(self.segment_target[hours].reshape(-1,5),dtype=torch.float32,device=self.device),
                'tail':torch.as_tensor((np.abs(rawy[:,1])>=self.tail_threshold).astype(np.float32),device=self.device),
                'valid':torch.as_tensor(self.valid[hours].reshape(-1),device=self.device),'baseline':tensors[-1]}
        return tuple(tensors),target
