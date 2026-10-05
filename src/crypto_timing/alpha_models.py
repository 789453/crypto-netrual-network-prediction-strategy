"""One controlled market/relative temporal upgrade on frozen v5 representations."""
from pathlib import Path
import math,json
import numpy as np
import torch
from torch import nn
import lightgbm as lgb
from sklearn.linear_model import Ridge
from .alpha_data import ROOT,BASE,HORIZONS,scale_at,summaries,save_json
from .review_data import identity,training_identity

class HorizonReadout(nn.Module):
    def __init__(self,kind='joint'):
        super().__init__();self.kind=kind
        if kind=='direct':self.head=nn.Linear(128,4)
        else:
            self.project=nn.Sequential(nn.Linear(128,48),nn.GELU(),nn.LayerNorm(48))
            self.market=nn.GRU(48,24,batch_first=True);self.relative=nn.GRU(96,32,batch_first=True)
            self.market_head=nn.Linear(24,4);self.relative_head=nn.Linear(32,4)
        for name,p in self.named_parameters():
            if 'head' in name:nn.init.zeros_(p)
    def forward(self,x,valid,risk):
        if self.kind=='direct':return self.head(x[:,-1]).float()*risk
        z=self.project(x);mask=valid[...,None].float();pool=(z*mask).sum(2)/mask.sum(2).clamp_min(1)
        m=self.market(pool)[0][:,-1];b,l,n,d=z.shape
        rel=torch.cat((z-pool[:,:,None],pool[:,:,None].expand_as(z)),-1).permute(0,2,1,3).reshape(b*n,l,96)
        r=self.relative(rel)[0][:,-1].reshape(b,n,32)
        with torch.autocast('cuda',enabled=False):
            mr=self.market_head(m.float())*risk.mean(1)
            ar=self.relative_head(r.float())*risk
            vm=valid[:,-1,:,None].float();ar=ar-(ar*vm).sum(1,keepdim=True)/vm.sum(1,keepdim=True).clamp_min(1)
        return mr[:,None]+ar

def loss(pred,target,risk,mask,kind):
    weight=mask[...,None].float();den=(weight.sum()*4).clamp_min(1)
    main=(((pred-target)/risk).square()*weight).sum()/den
    if kind=='direct':return main
    count=weight.sum(1).clamp_min(1);m=(target*weight).sum(1)/count;pm=(pred*weight).sum(1)/count
    common=((pm-m)/risk.mean(1)).square().mean()
    return main+.1*common

def score(pred,y,risk,mask,mean):
    p=pred[mask]/risk[mask];t=y[mask]/risk[mask]
    if not len(t):return {'mse':0.,'zero_mse':0.,'mean_mse':0.,'ic':0.,'signal_std':0.,'signal_mean':0.}
    return {'mse':float(np.mean((p-t)**2)),'zero_mse':float(np.mean(t*t)),'mean_mse':float(np.mean((t-mean)**2)),'ic':float(np.corrcoef(p.reshape(-1),t.reshape(-1))[0,1]) if p.std()>1e-8 else 0.,'signal_std':float(p.std()),'signal_mean':float(p.mean())}

def train_month(month,rep,features,target,valid,dates):
    folder=ROOT/'monthly'/str(month);folder.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path('src/crypto_timing/alpha_data.py')]
    contract={'source':identity(files),'v5_dependencies':training_identity(),'data':json.loads((ROOT/'market_manifest.json').read_text(encoding='utf-8'))['contract'],'month':str(month),'encoder':json.loads((ROOT/'representation_manifest.json').read_text(encoding='utf-8')),'epochs_max':4,'seeds':[20261004,20261005,20261006],'window_months':24,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    if (folder/'summary.json').exists():
        old=json.loads((folder/'summary.json').read_text(encoding='utf-8'));assert old['contract']==contract,'stale monthly models';return
    cutoff=np.datetime64(month,'ns');cut=np.searchsorted(dates,cutoff);begin=np.searchsorted(dates,cutoff-np.timedelta64(730,'D'))
    hh=np.flatnonzero((dates>=dates[max(723,begin)])&(dates+np.timedelta64(721,'m')<=cutoff)&valid.all(1)&np.isfinite(target).all((1,2)))
    core=hh[dates[hh]<cutoff-np.timedelta64(60,'D')-np.timedelta64(721,'m')];inner=hh[dates[hh]>=cutoff-np.timedelta64(60,'D')]
    end=np.datetime64(month+1,'ns');outer=np.flatnonzero((dates>=cutoff)&(dates<end)&valid.all(1));risk=scale_at(cut)
    yg=np.nan_to_num(target);records={};preds={};risk_gpu=torch.as_tensor(risk,device='cuda');y_gpu=torch.as_tensor(yg,device='cuda');valid_gpu=torch.as_tensor(valid,device='cuda')
    def prep(h):
        c=rep[h].mean((0,1));s=np.maximum(rep[h].std((0,1)),.05)
        return c,s,torch.as_tensor(np.clip((rep-c)/s,-8,8),device='cuda')
    def infer(model,x,h):
        model.eval();out=[]
        with torch.inference_mode():
            for st in range(0,len(h),128):
                k=torch.as_tensor(h[st:st+128],device='cuda');idx=k[:,None]-torch.arange(3,-1,-1,device='cuda')[None]
                with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x[idx],valid_gpu[idx],risk_gpu[k])
                out.append(o.float().cpu().numpy())
        return np.concatenate(out)
    def fit(kind,seed,h,c,s,x,epochs,watch,phase):
        torch.manual_seed(seed);model=HorizonReadout(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        rng=np.random.default_rng(seed);history=[];fixed=h[::max(1,len(h)//512)];mean=(yg[h]/risk[h]).mean((0,1))
        for ep in range(epochs):
            model.train();order=rng.permutation(h);online=[]
            for st in range(0,len(order),128):
                k=torch.as_tensor(order[st:st+128],device='cuda');idx=k[:,None]-torch.arange(3,-1,-1,device='cuda')[None]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x[idx],valid_gpu[idx],risk_gpu[k]);obj=loss(p,y_gpu[k],risk_gpu[k],valid_gpu[k],kind)
                obj.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();online.append(float(obj.detach()))
            row={'epoch':ep+1,'phase':phase,'online_loss':float(np.mean(online)),'train_eval':score(infer(model,x,fixed),yg[fixed],risk[fixed],valid[fixed],mean)}
            for name,w in watch.items():
                usable=w[np.isfinite(target[w]).all((1,2))]
                if len(usable):row[name]=score(infer(model,x,usable),yg[usable],risk[usable],valid[usable],mean)
            history.append(row)
        return model,history
    for kind in ('direct','joint'):
        runs=[];raw=[]
        for seed in (20261004,20261005,20261006):
            c,s,x=prep(core);model,curves=fit(kind,seed,core,c,s,x,4,{'internal':inner,'external_observe_only':outer},'inner_selection')
            # Shared horizon mean MSE; external records never select the budget.
            chosen=min((2,4),key=lambda e:curves[e-1]['internal']['mse']);del model,x
            c,s,x=prep(hh);model,final_curves=fit(kind,seed,hh,c,s,x,chosen,{'external_observe_only':outer},'full_refit')
            pp=infer(model,x,outer);raw.append(pp)
            torch.save({'model':model.state_dict(),'kind':kind,'center':c,'scale':s,'contract':contract,'training_risk_floor':np.nanquantile(np.load(BASE/'targets.npy',mmap_mode='r')[720:cut-13,:,7],.05,axis=0),'epochs':chosen},folder/f'{kind}_{seed}.pt')
            runs.append({'seed':seed,'chosen_epochs':chosen,'curves':curves,'refit_curves':final_curves,'parameters':sum(p.numel() for p in model.parameters())});del model,x
        preds['NN_'+kind]=np.mean(raw,0).astype(np.float32);records['NN_'+kind]=runs
    # Tabular references share all visible summary fields and masks; fit preprocessing only on the fitting interval.
    def tab(h):
        c=np.median(features[h],axis=(0,1));s=np.maximum(np.std(features[h],axis=(0,1)),.05)
        return c,s,np.clip((features-c)/s,-8,8).astype(np.float32)
    c,s,x=tab(core);xc=x[core].reshape(-1,x.shape[-1]);yi=(yg[core]/risk[core]).reshape(-1,4);xv=x[inner].reshape(-1,x.shape[-1]);yv=(yg[inner]/risk[inner]).reshape(-1,4)
    candidates=[]
    for alpha in (1000.,10000.):
        m=Ridge(alpha=alpha).fit(xc,yi);candidates.append((float(((m.predict(xv)-yv)**2).mean()),alpha))
    alpha=min(candidates)[1];c,s,x=tab(hh);xf=x[hh].reshape(-1,x.shape[-1]);yf=(yg[hh]/risk[hh]).reshape(-1,4)
    m=Ridge(alpha=alpha).fit(xf,yf);preds['Ridge']=(m.predict(x[outer].reshape(-1,x.shape[-1])).reshape(-1,12,4)*risk[outer]).astype(np.float32);records['Ridge']={'alpha':alpha,'inner_candidates':candidates}
    np.savez(folder/'ridge.npz',coef=m.coef_,intercept=m.intercept_,center=c,scale=s)
    raw=[];tree_records=[]
    ci,si,xi=tab(core)
    for j in range(4):
        params={'objective':'regression','learning_rate':.04,'num_leaves':15,'min_data_in_leaf':1000,'lambda_l2':100,'feature_fraction':.8,'num_threads':8,'verbosity':-1,'seed':20261004}
        inner_model=lgb.train(params,lgb.Dataset(xi[core].reshape(-1,xi.shape[-1]),label=yi[:,j]),num_boost_round=100)
        opts=[(float(((inner_model.predict(xi[inner].reshape(-1,xi.shape[-1]),num_iteration=e)-yv[:,j])**2).mean()),e) for e in (50,100)]
        rounds=min(opts)[1];m=lgb.train(params,lgb.Dataset(xf,label=yf[:,j]),num_boost_round=rounds);m.save_model(str(folder/f'lightgbm_{j}.txt'))
        raw.append(m.predict(x[outer].reshape(-1,x.shape[-1])).reshape(-1,12)*risk[outer,:,j]);tree_records.append({'horizon':int(HORIZONS[j]),'rounds':rounds,'inner_candidates':opts})
    preds['LightGBM']=np.stack(raw,-1).astype(np.float32);records['LightGBM']=tree_records
    # Mechanical signals are independently calibrated later from OOF history.
    st=np.load(BASE/'state_dynamic.npy',mmap_mode='r');preds['Trend']=np.repeat(np.asarray(st[outer,:,1])[...,None],4,-1).astype(np.float32);preds['Reversal']=-np.repeat(np.asarray(st[outer,:,0])[...,None],4,-1).astype(np.float32)
    preds['Frozen_v5']=np.repeat(np.load(ROOT/'frozen_encoder_signal.npy')[outer,...,None],4,-1)*risk[outer]
    np.savez_compressed(folder/'predictions.npz',hours=outer,risk=risk[outer],**preds)
    save_json(folder/'summary.json',{'contract':contract,'fit_hours':len(hh),'inner_train_hours':len(core),'inner_val_hours':len(inner),'forecast_hours':len(outer),'records':records,'maturity':'decision+12h+1m <= monthly cutoff','feature_count':features.shape[-1]})
    print(f'month complete {month}: {len(hh)} train {len(outer)} predict',flush=True)
