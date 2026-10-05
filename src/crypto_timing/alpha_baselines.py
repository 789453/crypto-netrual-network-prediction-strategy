"""Competitive tabular references including all twelve minute-branch fields."""
from pathlib import Path
import json
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from .alpha_data import ROOT,BASE,save_json,summaries
from .review_data import identity

def augmented_features():
    path=ROOT/'fair_summary_features.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    base=summaries();n,ns=base.shape[:2];minute=np.load(BASE/'minute.npy',mmap_mode='r');extra=np.zeros((n,ns,36),np.float32);end=np.arange(n)*60+59
    for j in range(ns):
        x=np.nan_to_num(np.asarray(minute[:,j],np.float32));cs=np.vstack((np.zeros((1,12)),np.cumsum(x,axis=0,dtype=np.float64)))
        for k,length in enumerate((60,180)):extra[:,j,k*12:(k+1)*12]=(cs[end+1]-cs[np.maximum(end+1-length,0)])/length
        extra[:,j,24:]=x[end]
    result=np.concatenate((base,extra),-1);np.save(path,result);return result

def fit_fair_baselines():
    features=augmented_features();dates=np.load(BASE/'decision_time.npy');target=np.load(ROOT/'target.npy');valid=np.load(ROOT/'valid.npy');n=len(dates)
    for month in np.arange(np.datetime64('2025-01','M'),np.datetime64('2026-10','M')):
        folder=ROOT/'monthly'/str(month);record=folder/'fair_baselines.json'
        contract={'source':identity([Path(__file__)]),'feature_file':identity([ROOT/'fair_summary_features.npy']),'old_fit_contract':json.loads((folder/'summary.json').read_text(encoding='utf-8'))['contract'],'feature_count':features.shape[-1]}
        if record.exists():
            assert json.loads(record.read_text(encoding='utf-8'))['contract']==contract;continue
        cutoff=np.datetime64(month,'ns');cut=np.searchsorted(dates,cutoff);begin=np.searchsorted(dates,cutoff-np.timedelta64(730,'D'))
        hh=np.flatnonzero((dates>=dates[max(723,begin)])&(dates+np.timedelta64(721,'m')<=cutoff)&valid.all(1)&np.isfinite(target).all((1,2)));core=hh[dates[hh]<cutoff-np.timedelta64(60,'D')-np.timedelta64(721,'m')];inner=hh[dates[hh]>=cutoff-np.timedelta64(60,'D')]
        from .alpha_data import scale_at
        risk=scale_at(cut);yp=target/risk
        def prep(h):
            c=np.median(features[h],axis=(0,1));s=np.maximum(np.std(features[h],axis=(0,1)),.05);return c,s,np.clip((features-c)/s,-8,8).astype(np.float32)
        c,s,x=prep(core);xc=x[core].reshape(-1,x.shape[-1]);xv=x[inner].reshape(-1,x.shape[-1]);yi=yp[core].reshape(-1,4);yv=yp[inner].reshape(-1,4)
        opts=[]
        for alpha in (1000.,10000.):
            m=Ridge(alpha=alpha).fit(xc,yi);opts.append((float(((m.predict(xv)-yv)**2).mean()),alpha))
        alpha=min(opts)[1];cc,ss,xf=prep(hh);xfit=xf[hh].reshape(-1,xf.shape[-1]);yf=yp[hh].reshape(-1,4);outer=np.load(folder/'predictions.npz')['hours'];xo=xf[outer].reshape(-1,xf.shape[-1]);m=Ridge(alpha=alpha).fit(xfit,yf);preds={'Ridge':(m.predict(xo).reshape(-1,12,4)*risk[outer]).astype(np.float32)}
        np.savez(folder/'fair_ridge.npz',coef=m.coef_,intercept=m.intercept_,center=cc,scale=ss);tree=[];raw=[]
        for j in range(4):
            params={'objective':'regression','learning_rate':.04,'num_leaves':15,'min_data_in_leaf':1000,'lambda_l2':100,'feature_fraction':.8,'num_threads':8,'verbosity':-1,'seed':20261004};im=lgb.train(params,lgb.Dataset(xc,label=yi[:,j]),num_boost_round=100);ops=[(float(((im.predict(xv,num_iteration=e)-yv[:,j])**2).mean()),e) for e in (50,100)];rounds=min(ops)[1]
            m=lgb.train(params,lgb.Dataset(xfit,label=yf[:,j]),num_boost_round=rounds);m.save_model(str(folder/f'fair_lightgbm_{j}.txt'));raw.append(m.predict(xo).reshape(-1,12)*risk[outer,:,j]);tree.append({'horizon':(2,4,8,12)[j],'rounds':rounds,'inner_candidates':ops})
        preds['LightGBM']=np.stack(raw,-1).astype(np.float32);np.savez_compressed(folder/'fair_baselines.npz',hours=outer,**preds);save_json(record,{'contract':contract,'ridge_inner_candidates':opts,'ridge_alpha':alpha,'LightGBM':tree,'same_clock_training_hours':len(hh),'output':identity([folder/'fair_baselines.npz'])});print(f'fair references complete {month}',flush=True)
