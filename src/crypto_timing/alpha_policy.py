"""Causal OOF edge extraction and a common cost-aware holding policy."""
from dataclasses import dataclass,asdict
import numpy as np

def calibrate(raw,y,risk):
    """Separate predicted common and relative returns; only matured OOF rows enter."""
    result=[]
    for j in range(4):
        p=raw[:,:,j];t=y[:,:,j];v=risk[:,:,j];cv=v.mean(1)
        px=p.mean(1)/cv;ty=t.mean(1)/cv
        def fit(x,y):
            xm=x.mean();ym=y.mean();xc=x-xm;yc=y-ym
            a=float(np.clip(np.sum(xc*yc)/(np.sum(xc*xc)+1e-4*len(x)),0,3))
            # Intercept shrink is explicit; raw model levels are not silently de-meaned.
            b=float(np.clip((ym-a*xm)/5,-.02,.02));return a,b
        ac,bc=fit(px,ty);pr=(p-p.mean(1)[:,None])/v;tr=(t-t.mean(1)[:,None])/v
        ar,br=fit(pr.reshape(-1),tr.reshape(-1));ar=float(ar)
        fitted=(ac*px+bc)[:,None]*cv[:,None]+ar*pr*v
        scaled=fitted/v;edges=np.quantile(scaled.reshape(-1),np.arange(1,10)/10)
        bins=np.searchsorted(edges,scaled);means=[];counts=[];wins=[];tails=[]
        # Bin statistics are retained for information diagnostics, not force-trading.
        for k in range(10):
            yy=t[bins==k];counts.append(len(yy));means.append(float(yy.mean()) if len(yy) else 0.);wins.append(float((yy>0).mean()) if len(yy) else 0.);tails.append(float(np.quantile(yy,.01)) if len(yy) else 0.)
        result.append({'common_slope':ac,'common_intercept':bc,'relative_slope':ar,'quantile_edges':edges.tolist(),'raw_return_bin_mean':means,'bin_count':counts,'bin_up_fraction':wins,'bin_q01':tails})
    return result

def apply_calibration(raw,risk,fit):
    result=np.zeros_like(raw)
    for j,r in enumerate(fit):
        p=raw[:,:,j];v=risk[:,:,j];cv=v.mean(1);common=p.mean(1)/cv
        result[:,:,j]=(r['common_slope']*common+r['common_intercept'])[:,None]*cv[:,None]+r['relative_slope']*(p-p.mean(1)[:,None])
    return result

@dataclass(frozen=True)
class Policy:
    horizon:int=8
    multiplier:float=1.5
    sizing:str='invested'
    design_fee:float=.0004
    cap:float=2.
    net_fraction:float=.5
    coin_cap:float=.75
    annual_risk:float=.60
    maintain:float=.25
    name:str='cost_aware'

def project(w,cap,net_cap,coin_cap):
    w=np.clip(w,-coin_cap,coin_cap)
    if np.abs(w).sum()>cap:w*=cap/np.abs(w).sum()
    pos=np.maximum(w,0);neg=np.maximum(-w,0);l=pos.sum();s=neg.sum()
    if l-s>net_cap and l>0:pos*= (s+net_cap)/l
    if s-l>net_cap and s>0:neg*= (l+net_cap)/s
    return pos-neg

def covariance(prices,h):
    start=max(1,h-720);r=prices[start:h+1]/prices[start-1:h]-1
    if len(r)<48:return np.eye(prices.shape[1])*1e-5,np.ones(prices.shape[1])
    cov=np.cov(r,rowvar=False);cov=.7*cov+.3*np.diag(np.diag(cov))+np.eye(cov.shape[0])*1e-8
    market=r.mean(1);beta=np.mean((r-r.mean(0))*(market-market.mean())[:,None],0)/max(market.var(),1e-10)
    return cov,beta

def target_weights(mu,funding,cov,beta,policy):
    cost=2*policy.design_fee*policy.multiplier
    directional=mu-funding*policy.horizon
    eligible=(np.abs(directional)>cost)&(directional*mu>0)
    if not eligible.any():return np.zeros(len(mu))
    # Explicit common/relative allocation; do not force the full forecast to be neutral.
    v=np.sqrt(np.maximum(np.diag(cov),1e-9));common=float(directional.mean());relative=directional-common
    rel=relative/v;rel-=beta*float(rel@beta)/max(float(beta@beta),1e-9)
    market=np.full(len(mu),common)/v
    def normalized(x):return x/max(np.abs(x).sum(),1e-9)
    w=.5*normalized(market)+.5*normalized(rel)
    # Raw total expected edge still must pay its own transaction costs.
    w=np.where(eligible&(w*directional>0),w,0)
    if np.abs(w).sum()<1e-9:return np.zeros(len(mu))
    w*=policy.cap/np.abs(w).sum()
    if policy.sizing=='proportional':w*=min(1,float(np.median(np.abs(directional[eligible])-cost))/max(4*policy.design_fee,1e-5))
    risk=float(np.sqrt(max(w@cov@w,1e-12))*np.sqrt(24*365.25))
    w*=min(1,policy.annual_risk/max(risk,1e-10))
    return project(w,policy.cap,policy.cap*policy.net_fraction,policy.coin_cap)

def decide(mu,funding,cov,beta,current,age,policy):
    desired=target_weights(mu,funding,cov,beta,policy);change=np.zeros(len(mu),bool);reason=np.full(len(mu),'hold',object)
    edge=mu-funding*policy.horizon
    for j,old in enumerate(current):
        if abs(old)<1e-9:
            change[j]=abs(desired[j])>1e-9;reason[j]='entry' if change[j] else 'cash'
        elif age[j]>=policy.horizon:
            change[j]=True;desired[j]=0;reason[j]='expiry'
        elif old*edge[j]<-abs(old)*2*policy.design_fee*policy.multiplier:
            change[j]=True;reason[j]='reverse'
        elif np.sign(old)*edge[j]<policy.maintain*policy.design_fee:
            change[j]=True;desired[j]=0;reason[j]='edge_lost'
    proposed=np.where(change,desired,current)
    # Drift/risk guard is a constraint, not gratuitous hourly exact rebalancing.
    guarded=project(proposed,policy.cap,policy.cap*policy.net_fraction,policy.coin_cap)
    annual=np.sqrt(max(guarded@cov@guarded,1e-12))*np.sqrt(24*365.25)
    if annual>policy.annual_risk*1.15:guarded*=policy.annual_risk/annual
    extra=np.abs(guarded-proposed)>1e-8;change|=extra;reason[extra]='risk_limit'
    return guarded,change,reason

def policy_grid():
    return [Policy(horizon=h,multiplier=k,sizing=s) for h in (2,4,8,12) for k in (1.,1.5,2.) for s in ('invested','proportional')]
