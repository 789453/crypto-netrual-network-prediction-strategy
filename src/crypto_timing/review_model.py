"""Small, hypothesis-specific v5 network; every output has its own estimand."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .mechanism_model import MechanismConfig,MechanismNetwork,masked_mean

@dataclass(frozen=True)
class ReviewConfig(MechanismConfig):
    features:str='original'
    fast_dim:int=16
    minute:str='encoder'
    auxiliary:str='shaped_q'
    budget:str='hard'
    events:bool=False
    minute_order:str='quarter'
    segments:bool=False
    tail:bool=False
    baseline:bool=False

class PatchReadout(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind
        self.gru=nn.GRU(48,32,batch_first=True) if kind=='gru' else None
        self.project=nn.Sequential(nn.Linear(48*7+(32 if self.gru else 0),48),nn.GELU(),nn.LayerNorm(48))
    def forward(self,x,mask):
        # Both arms have exactly the same last, recent, whole and four age summaries.
        patches=x.reshape(len(x),36,5,48).mean(2)
        pm=mask.any(-1).reshape(len(x),36,5).any(-1)[...,None]
        summaries=[patches[:,-1],masked_mean(patches[:,-12:],pm[:,-12:]),masked_mean(patches,pm)]
        summaries += [masked_mean(v,m) for v,m in zip(patches.chunk(4,1),pm.chunk(4,1))]
        if self.gru: summaries.append(self.gru(patches)[0][:,-1])
        return self.project(torch.cat(summaries,-1))

class ReviewNetwork(MechanismNetwork):
    def __init__(self,cfg):
        super().__init__(cfg)
        if cfg.events: self.context=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.LayerNorm(32))
        if cfg.minute_order in ('common','gru'): self.minute_readout=PatchReadout(cfg.minute_order)
        self.segment_head=nn.Linear(128,5); self.tail_head=nn.Linear(128,1)
        for layer in (self.segment_head,self.tail_head): nn.init.normal_(layer.weight,std=.002);nn.init.zeros_(layer.bias)
        # No inactive legacy/distribution outputs are instantiated in v5.
        del self.distribution,self.band,self.sign,self.legacy_magnitude
    def forward(self,fast,fm,slow,sm,state,cm,minute=None,mm=None,baseline=None):
        context=self.context(torch.cat((state*cm,cm),-1)); local=self.fast(fast,fm)
        signal_local=local
        if self.film:
            gamma,beta=(.1*torch.tanh(self.film(context))).chunk(2,-1)
            signal_local=local*(1+gamma[:,None])+beta[:,None]
        memory=self.memory(signal_local,fm); sl=self.slow(slow,sm)
        back=self.slow_readout(torch.cat([masked_mean(v,m) for v,m in zip(sl.chunk(4,1),sm.chunk(4,1))],-1))
        pieces=[memory,back,context]
        if self.minute:
            ml=self.minute(minute,mm)
            if self.cfg.minute_order=='quarter':
                patches=ml.reshape(len(ml),36,5,48).mean(2)
                pieces.append(self.minute_readout(torch.cat([v.mean(1) for v in patches.chunk(4,1)],-1)))
            else: pieces.append(self.minute_readout(ml,mm))
        h=self.fusion(torch.cat(pieces,-1))
        risk=self.risk_readout(torch.cat((masked_mean(local,fm),local[:,-1],context),-1))
        with torch.autocast(device_type=fast.device.type,enabled=False):
            delta=F.linear(h.float(),self.signal.weight.float(),self.signal.bias.float()).squeeze(-1)
        base=torch.zeros_like(delta) if baseline is None else baseline
        return {'signal':delta+base,'delta':delta,'baseline':base,'path':self.path(risk),'segments':self.segment_head(h),'tail':self.tail_head(h).squeeze(-1),'representation':h,'local_activation':local}

def weighted_mean(x,valid):
    while valid.ndim<x.ndim: valid=valid.unsqueeze(-1)
    weight=valid.expand_as(x).float(); safe=torch.where(weight.bool(),x,torch.zeros_like(x))
    return safe.sum()/weight.sum().clamp_min(1)

def review_losses(out,target,cfg):
    valid=target['valid']; primary=weighted_mean((out['signal'].float()-torch.nan_to_num(target['y']))**2,valid)
    components={}
    if cfg.auxiliary=='raw':
        for j,name in enumerate(('logT','Q','A')): components[name]=weighted_mean((out['path'][:,j].float()-torch.nan_to_num(target['raw_path'][:,j]))**2,valid)
    elif cfg.auxiliary in ('ta','shaped_q'):
        for j,name in enumerate(('logT','asinhQ','A')):
            if j==1 and cfg.auxiliary=='ta': continue
            components[name]=weighted_mean((out['path'][:,j].float()-torch.nan_to_num(target['path'][:,j]))**2,valid)
    if cfg.segments:
        for j in range(5): components[f'segment_{j+1}']=weighted_mean((out['segments'][:,j].float()-torch.nan_to_num(target['segments'][:,j]))**2,valid)
    if cfg.tail: components['tail']=weighted_mean(F.binary_cross_entropy_with_logits(out['tail'].float(),target['tail'],reduction='none'),valid)
    aux=torch.stack(list(components.values())).mean() if components else primary*0
    return primary,aux,components

def gradient_budget(model,primary,auxiliary,mode='hard',soft_weight=.2,ratio=.3):
    """Bound actual shared gradient tensors, after independently clipping task gradients."""
    named=list(model.named_parameters()); params=[p for _,p in named]
    gp=torch.autograd.grad(primary,params,retain_graph=True,allow_unused=True)
    ga=torch.autograd.grad(auxiliary,params,retain_graph=True,allow_unused=True)
    zero=primary.new_zeros((),dtype=torch.float32)
    pn=torch.sqrt(sum((g.float().square().sum() for g in gp if g is not None),zero)+1e-24)
    an=torch.sqrt(sum((g.float().square().sum() for g in ga if g is not None),zero)+1e-24)
    pm=torch.clamp(1/pn,max=1); am=torch.clamp(1/an,max=1)
    groups={}; violations=0; max_ratio=0.
    for (name,p),g,a in zip(named,gp,ga):
        if g is None and a is None: continue
        main=None if g is None else g.float()*pm
        additional=None if a is None else a.float()*am
        if main is not None and additional is not None:
            gn=main.norm(); aa=additional.norm()
            factor=torch.clamp(ratio*gn/aa.clamp_min(1e-24),max=1) if mode=='hard' else additional.new_tensor(soft_weight)
            additional=additional*factor
            rr=float(additional.norm()/gn.clamp_min(1e-24)); max_ratio=max(max_ratio,rr)
            if rr>ratio+1e-5: violations+=1
            group=name.split('.')[0]; entry=groups.setdefault(group,{'main2':0.,'aux2':0.,'dot':0.,'parameters':0})
            entry['main2']+=float(main.square().sum());entry['aux2']+=float(additional.square().sum());entry['dot']+=float((main*additional).sum());entry['parameters']+=p.numel()
        p.grad=(additional if main is None else main if additional is None else main+additional).to(p.dtype)
    for entry in groups.values():
        entry['ratio']=(entry['aux2']/max(entry['main2'],1e-24))**.5
        entry['cosine']=entry['dot']/max((entry['main2']*entry['aux2'])**.5,1e-24)
    return {'groups':groups,'max_parameter_ratio':max_ratio,'violations':violations,'primary_preclip_norm':float(pn),'aux_preclip_norm':float(an),'soft_weight':soft_weight}
