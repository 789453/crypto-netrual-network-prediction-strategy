"""Matched v5 hypotheses, exact head reconstruction and code-bound artifacts."""
from __future__ import annotations
import json,math,time,platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .review_model import ReviewNetwork,review_losses,gradient_budget
from .review_data import training_identity
from .mechanism_training import correlation

def decomposition(pred,y,valid=None):
    if valid is None: valid=np.isfinite(y)&np.isfinite(pred)
    p=np.asarray(pred,dtype=float)[valid];t=np.asarray(y,dtype=float)[valid]
    den=float(np.mean(t*t)); cov=float(np.mean((p-p.mean())*(t-t.mean())))
    result={'mse':float(np.mean((p-t)**2)),'zero_mse':den,'skill':float(1-np.mean((p-t)**2)/den),'correlation':correlation(p,t),'mean_signal':float(p.mean()),'mean_target':float(t.mean()),'signal_std':float(p.std()),'target_std':float(t.std()),
            'covariance_gain':2*cov/den,'mean_alignment':float((2*p.mean()*t.mean()-p.mean()**2)/den),'variance_penalty':float(-p.var()/den),'samples':len(p)}
    assert abs(result['skill']-(result['covariance_gain']+result['mean_alignment']+result['variance_penalty']))<1e-9
    return result

def score_predictions(data):
    result=decomposition(data['signal'],data['y'],data['valid']); result['hours']=len(data['hours'])
    result['monthly']={}
    for month in np.unique(data['dates'].astype('datetime64[M]')):
        k=data['dates'].astype('datetime64[M]')==month
        result['monthly'][str(month)]=decomposition(data['signal'][k],data['y'][k],data['valid'][k])
    return result

@torch.inference_mode()
def predict(model,store,hours,cfg,representations=False):
    model.eval(); keys=('signal','delta','baseline','path','segments','tail'); arrays={k:[] for k in keys};reps=[]; sensitivity=[]
    for start in range(0,len(hours),24):
        x,t=store.batch(hours[start:start+24],cfg)
        with torch.autocast('cuda',dtype=torch.bfloat16): out=model(*x)
        for k in keys: arrays[k].append(out[k].float().cpu().numpy().reshape((len(hours[start:start+24]),store.ns)+tuple(out[k].shape[1:])))
        if representations: reps.append(out['representation'].float().cpu().numpy().reshape(-1,store.ns,128))
    result={k:np.concatenate(v) for k,v in arrays.items()}
    result.update(hours=hours,dates=store.dates[hours],y=store.labels[hours,:,1],raw_path=store.labels[hours,:,3:6],shaped_path=store.path_shaped[hours],segment_target=store.segment_target[hours],segment_raw=store.segments[hours],scale=store.scale[hours],raw_return=store.raw[hours,:,1],valid=store.valid[hours],future_no_trade=store.raw[hours,:,8])
    if representations: result['representation']=np.concatenate(reps)
    return result

def selected_probe(model,store,cfg):
    h=store.splits['train'][::4];cut=int(len(h)*.8); maturity=store.dates[h]+np.timedelta64(245,'m')
    inner_train=h[:cut][maturity[:cut]<=store.dates[h[cut]]]; inner_val=h[cut:]
    train=predict(model,store,inner_train,cfg,True); val=predict(model,store,inner_val,cfg,True)
    def flatten(data):
        mask=data['valid'].reshape(-1);return data['representation'].reshape(-1,128)[mask].astype(float),(data['y']-data['baseline']).reshape(-1)[mask]
    xt,y=flatten(train);xv,yv=flatten(val);mu=xt.mean(0);sd=np.maximum(xt.std(0),1e-3)
    xt=(xt-mu)/sd;xv=(xv-mu)/sd;ym=y.mean();xm=xt.mean(0)
    values=[];fits={}
    for strength in (100.,1000.,10000.,100000.):
        w=np.linalg.solve(xt.T@xt+strength*np.eye(128),xt.T@(y-ym))
        loss=float(((ym+xv@w-yv)**2).mean());values.append({'ridge':strength,'inner_mse':loss});fits[strength]=w
    chosen=min(values,key=lambda v:v['inner_mse'])['ridge']
    alltrain=predict(model,store,h,cfg,True);xx,yy=flatten(alltrain);mu=xx.mean(0);sd=np.maximum(xx.std(0),1e-3);z=(xx-mu)/sd;ym=yy.mean()
    w=np.linalg.solve(z.T@z+chosen*np.eye(128),z.T@(yy-ym))
    results={}; maxerr=0
    weight=model.signal.weight.detach().cpu().numpy()[0].astype(float);bias=float(model.signal.bias.detach().cpu())
    for split in ('validation','replay'):
        data=predict(model,store,store.splits[split],cfg,True);r=data['representation'].astype(float)
        reconstruction=r@weight+bias+data['baseline']; err=float(np.max(np.abs(reconstruction-data['signal'])))
        maxerr=max(maxerr,err);assert err<2e-6,'original output head reconstruction mismatch'
        pp=ym+((r-mu)/sd)@w+data['baseline'];results[split]=decomposition(pp,data['y'],data['valid'])
    s=np.linalg.svd(xx-xx.mean(0),compute_uv=False);sp=s*s/max((s*s).sum(),1e-12)
    return {'original_head_max_abs_error':maxerr,'ridge':chosen,'inner_time_selection':values,'intercept_penalized':False,'probe_train_hours':len(h),'results':results,'effective_rank':float(np.exp(-(sp*np.log(sp+1e-12)).sum()))}

def train_review(store,cfg,folder:Path,epochs=3,steps_per_epoch=None,probe=False):
    contract={'config':asdict(cfg),'fold':store.fold,'epochs':epochs,'steps_per_epoch':steps_per_epoch,'batch_hours':24,'source_sha256':training_identity(),'cache':store.cache_identity['contract'],'overlay':store.cache_identity['content_sha256'],
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'checkpoint_selection':'fixed final epoch; outer labels never choose weights'}
    if (folder/'summary.json').exists():
        saved=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if saved['contract']!=contract: raise ValueError(f'code/data/config changed; refusing stale run {folder}')
        return saved
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed);torch.cuda.reset_peak_memory_stats()
    model=ReviewNetwork(cfg).to(store.device);train=store.splits['train'];store.load_prep(cfg.prep);store.load_optional('minute')
    with torch.no_grad():
        residual=store.labels[train,:,1]-(store.baseline[train] if cfg.baseline else 0)
        model.signal.bias.fill_(float(residual[store.valid[train]].mean()))
        model.path.bias.copy_(torch.tensor(store.labels[train,:,3:6][store.valid[train]].mean(0) if cfg.auxiliary=='raw' else np.zeros(3),device=store.device))
        model.tail_head.bias.fill_(math.log(.01/.99))
    opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>1],'weight_decay':.001},{'params':[p for p in model.parameters() if p.ndim==1],'weight_decay':0}],lr=2e-4)
    rng=np.random.default_rng(cfg.seed);steps=math.ceil(len(train)/24) if steps_per_epoch is None else steps_per_epoch;total=epochs*steps
    history=[];gradients=[];contributions=[];seen=set();violations=0;maxratio=0.;soft=.2;started=time.perf_counter()
    print(json.dumps({'event':'review_start','run':str(folder),'hours':len(train),'epochs':epochs,'steps':steps,'config':asdict(cfg)}),flush=True)
    for epoch in range(epochs):
        model.train()
        order=rng.permutation(train)[:steps*24];running=0.;count=0;components_total={}
        for bi,offset in enumerate(range(0,len(order),24)):
            hh=order[offset:offset+24];seen.update(map(int,hh));x,target=store.batch(hh,cfg);step=epoch*steps+bi
            progress=min(max((step-24)/max(.65*total-24,1),0),1);lr=2e-5+.5*(2e-4-2e-5)*(1+math.cos(math.pi*progress))
            if step<24: lr=2e-4*(step+1)/24
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): out=model(*x);primary,aux,components=review_losses(out,target,cfg)
            if not torch.isfinite(primary+aux):raise RuntimeError('nonfinite review loss')
            if components:
                # Separate task diagnostics precede the aggregate budget; all common modules included.
                if bi%64==0:
                    common=[(n,p) for n,p in model.named_parameters() if n.startswith(('fast.','context.'))]
                    pc=torch.autograd.grad(primary,[p for _,p in common],retain_graph=True,allow_unused=True)
                    entry={'epoch':epoch+1,'batch':bi,'tasks':{}}
                    for name,loss in components.items():
                        gs=torch.autograd.grad(loss,[p for _,p in common],retain_graph=True,allow_unused=True)
                        group={}
                        for (pn,_),g,s in zip(common,gs,pc):
                            if g is None:continue
                            key=pn.split('.')[0];v=group.setdefault(key,{'aux2':0.,'main2':0.,'dot':0.})
                            v['aux2']+=float(g.float().square().sum())
                            if s is not None:v['main2']+=float(s.float().square().sum());v['dot']+=float((g.float()*s.float()).sum())
                        entry['tasks'][name]=group
                    contributions.append(entry)
                if cfg.budget=='soft' and bi%16==0:
                    fast=list(model.fast.parameters());gg=torch.autograd.grad(primary,fast,retain_graph=True,allow_unused=True);aa=torch.autograd.grad(aux,fast,retain_graph=True,allow_unused=True)
                    pn=sum(float(g.float().square().sum()) for g in gg if g is not None)**.5;an=sum(float(g.float().square().sum()) for g in aa if g is not None)**.5
                    soft=.9*soft+.1*min(.5,max(.01,.3*pn/max(an,1e-8)))
                log=gradient_budget(model,primary,aux,cfg.budget,soft)
                violations+=log['violations'];maxratio=max(maxratio,log['max_parameter_ratio'])
                if bi%16==0:gradients.append({'epoch':epoch+1,'batch':bi,**log})
            else:primary.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();n=int(target['valid'].sum());running+=float(primary.detach())*n;count+=n
            for name,value in components.items():components_total[name]=components_total.get(name,0)+float(value.detach())*n
        # OOF curves are diagnostic only: fixed final weights, no minimum-OOF selection.
        data=predict(model,store,store.splits['validation'],cfg);score=score_predictions(data)
        record={'epoch':epoch+1,'train_mse':running/count,'aux_components':{n:v/count for n,v in components_total.items()},'diagnostic':score,'lr':lr,'elapsed_seconds':round(time.perf_counter()-started,2)};history.append(record)
        (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({'event':'review_epoch','run':folder.name,'epoch':epoch+1,'mse':running/count,'diagnostic_skill':score['skill'],'elapsed_s':record['elapsed_seconds']}),flush=True)
    assert cfg.budget!='hard' or violations==0,'auxiliary hard budget violated'
    saved={'model':model.state_dict(),'config':asdict(cfg),'contract':contract,'scalers':store.scalers,'event_scaler':store.event_scaler,'floors':store.floors,'path_mean':store.path_mean,'path_sd':store.path_sd,'qscale':store.qscale,'segment_mean':store.segment_mean,'segment_sd':store.segment_sd,'tail_threshold':store.tail_threshold,'epoch':epochs}
    torch.save(saved,folder/'final.pt');results={}
    for split in ('validation','replay'):
        data=predict(model,store,store.splits[split],cfg,True);np.savez_compressed(folder/f'{split}.npz',**data);results[split]=score_predictions(data)
        reconstructed=data['representation'].astype(float)@model.signal.weight.detach().cpu().numpy()[0].astype(float)+float(model.signal.bias.detach().cpu())+data['baseline']
        assert np.max(np.abs(reconstructed-data['signal']))<2e-6
    # Critical fields' actual first-layer/input sensitivity, per asset and channel.
    model.eval();x,t=store.batch(store.splits['validation'][::max(len(store.splits['validation'])//12,1)][:12],cfg);xx=list(x);xx[0]=xx[0].detach().requires_grad_(True);xx[6]=xx[6].detach().requires_grad_(True)
    # cuDNN eval RNN has no backward reserve; native RNN keeps eval/dropout semantics.
    with torch.backends.cudnn.flags(enabled=False),torch.autocast('cuda',dtype=torch.bfloat16):
        oo=model(*xx)
        gf,gm=torch.autograd.grad(oo['delta'].sum(),(xx[0],xx[6]))
    sensitivity={'fast':gf.abs().mean(1).reshape(-1,store.ns,cfg.fast_dim).mean(0).cpu().tolist(),'minute':gm.abs().mean(1).reshape(-1,store.ns,12).mean(0).cpu().tolist(),'meaning':'local derivative in standardized input units; not causal feature attribution'}
    probe_result=selected_probe(model,store,cfg) if probe else None
    summary={'contract':contract,'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters()),'unique_train_hours':len(seen),'train_hours':len(train),'total_steps':epochs*steps,'seconds':round(time.perf_counter()-started,2),'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20,'results':results,'budget_violations_all_batches':violations,'max_shared_parameter_ratio_all_batches':maxratio,'gradient_diagnostics':gradients,'task_contributions':contributions,'quality':store.quality_counts,'probe':probe_result,'input_sensitivity':sensitivity,'target_transform':{'qscale':store.qscale,'path_mean':store.path_mean.tolist(),'path_sd':store.path_sd.tolist(),'tail_threshold':store.tail_threshold}}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
