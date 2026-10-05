"""Unchanged v5 forecasts -> shrunk directional support -> small-to-large positions.

No neural inference or classifier fitting. The ten-bin frequencies are already
saved in the matured, past-only monthly calibration artifact. Probabilities are
estimates of price direction, not probability of a profitable net transaction.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .alpha_policy import project,decide


def monotone_probability(record, horizon=12, prior_blocks=20.):
    """Conservatively pool 12 correlated coins and overlapping H-hour labels.

    count/(12*H) is a deliberately conservative design discount, NOT an estimate
    of independent sample size or a confidence interval. PAVA only regularizes
    the saved ten probabilities; it does not change any return forecast.
    """
    count=np.asarray(record['bin_count'],float)
    observed=np.asarray(record['bin_up_fraction'],float)
    if count.shape!=(10,) or observed.shape!=(10,):raise ValueError('ten bins required')
    if np.any(count<0) or np.any((observed<0)|(observed>1)):raise ValueError('invalid bin frequency')
    effective=count/(12*horizon)
    weight=effective+prior_blocks
    probability=(effective*observed+.5*prior_blocks)/weight
    blocks=[]
    for i,(p,w) in enumerate(zip(probability,weight)):
        blocks.append([i,i,float(p*w),float(w)])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            right=blocks.pop();left=blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    result=np.empty(10)
    for a,b,p,w in blocks:result[a:b+1]=p/w
    return {'probability_up':result.tolist(),'raw_probability_up':observed.tolist(),
            'effective_count_design':effective.tolist(),'prior_blocks':prior_blocks,
            'quantile_edges':record['quantile_edges']}


def probability_from_saved(mu,risk,table):
    mu=np.asarray(mu);risk=np.asarray(risk)
    if np.any(risk<=0) or not np.isfinite(mu).all() or not np.isfinite(risk).all():
        raise ValueError('finite forecasts and positive past risk required')
    bins=np.searchsorted(table['quantile_edges'],mu/risk)
    return np.asarray(table['probability_up'])[bins]


@dataclass(frozen=True)
class ProbabilityMapping:
    shape:str='continuous'
    confidence_full:float=.05
    cost_per_side:float=.0002
    minimum_hold:int=4
    review_hours:int=12
    reversal_confirm:int=2
    weak_confirm:int=3
    absolute_band:float=.015
    relative_band:float=.30
    minimum_entry:float=.002
    fixed_expiry:bool=False
    core_overlay:bool=False
    overlay_fraction:float=.1


def support_strength(mu,p_up,mapping):
    # Never flip the neural return direction based on an auxiliary probability.
    probability=np.where(mu>0,p_up,np.where(mu<0,1-p_up,.5))
    excess=np.maximum(probability-.5,0.)
    if mapping.shape=='discrete':
        strength=np.select([excess>=.05,excess>=.03,excess>=.015,excess>=.005],
                           [1.,.65,.35,.15],default=0.)
    elif mapping.shape=='continuous':strength=np.clip(excess/mapping.confidence_full,0,1)
    else:raise ValueError(mapping.shape)
    # Soft cost discount: e=2bp still gets 1/3 of its confidence-sized allocation.
    strength*=np.abs(mu)/(np.abs(mu)+2*mapping.cost_per_side)
    return probability,strength


def target_probability_weights(mu,p_up,cov,beta,policy,mapping):
    probability,strength=support_strength(mu,p_up,mapping)
    v=np.sqrt(np.maximum(np.diag(cov),1e-9))
    common=float(mu.mean());relative=(mu-common)/v
    relative-=beta*float(relative@beta)/max(float(beta@beta),1e-9)
    market=np.full(len(mu),common)/v
    def normalized(x):return x/max(float(np.abs(x).sum()),1e-9)
    base=.5*normalized(market)+.5*normalized(relative)
    w=policy.cap*base*strength
    w=np.where(w*mu>0,w,0.)
    # Crucially NO re-normalization after confidence sizing: weak signals stay small.
    annual=np.sqrt(max(float(w@cov@w),1e-12))*np.sqrt(24*365.25)
    w*=min(1,policy.annual_risk/max(annual,1e-9))
    return project(w,policy.cap,policy.cap*policy.net_fraction,policy.coin_cap)


class ProbabilityDecision:
    """Stateful hysteresis; one fresh instance per account, causal row-by-row."""
    def __init__(self,p_up,mapping):
        self.p_up=np.asarray(p_up);self.mapping=mapping
        self.weak=np.zeros(self.p_up.shape[1],int)
        self.opposite=np.zeros(self.p_up.shape[1],int)
        self.last_review=np.full(self.p_up.shape[1],-10000,int)
        self.desired_history=[]
        self.core_life=np.zeros(self.p_up.shape[1],bool)
        self.core_age=np.zeros(self.p_up.shape[1],int)
        self.role_history=[]

    def __call__(self,row,mu,cov,beta,current,age,policy):
        m=self.mapping
        desired=target_probability_weights(mu,self.p_up[row],cov,beta,policy,m)
        if m.core_overlay:
            return self._core_overlay(row,mu,cov,beta,current,age,policy,desired)
        self.desired_history.append(desired.copy())
        change=np.zeros(len(mu),bool);reason=np.full(len(mu),'hold',object)
        target=current.copy()
        for j,old in enumerate(current):
            weak=abs(desired[j])<m.minimum_entry
            opposite=old*desired[j]<0
            self.weak[j]=self.weak[j]+1 if weak else 0
            self.opposite[j]=self.opposite[j]+1 if opposite else 0
            if abs(old)<1e-9:
                if not weak:
                    target[j]=desired[j];change[j]=True;reason[j]='prob_entry';self.last_review[j]=row
            elif m.fixed_expiry and age[j]>=policy.horizon:
                target[j]=0.;change[j]=True;reason[j]='expiry';self.last_review[j]=row
            elif age[j]>=m.minimum_hold:
                if self.opposite[j]>=m.reversal_confirm:
                    target[j]=desired[j];change[j]=True;reason[j]='confirmed_reverse';self.last_review[j]=row
                elif self.weak[j]>=m.weak_confirm:
                    target[j]=0.;change[j]=True;reason[j]='support_lost';self.last_review[j]=row
                elif not weak and not opposite and row-self.last_review[j]>=m.review_hours:
                    band=max(m.absolute_band,m.relative_band*abs(old))
                    if abs(desired[j]-old)>band:
                        target[j]=desired[j];change[j]=True;reason[j]='confidence_resize'
                    self.last_review[j]=row
        guarded=project(target,policy.cap,policy.cap*policy.net_fraction,policy.coin_cap)
        annual=np.sqrt(max(float(guarded@cov@guarded),1e-12))*np.sqrt(24*365.25)
        if annual>policy.annual_risk*1.15:guarded*=policy.annual_risk/max(annual,1e-9)
        extra=np.abs(guarded-target)>1e-8;change|=extra;reason[extra]='risk_limit'
        return guarded,change,reason

    def _core_overlay(self,row,mu,cov,beta,current,age,policy,soft):
        """Preserve economic core; fund small weak forecasts from unused risk budget."""
        m=self.mapping
        # Core lifespan starts when a weak same-side holding is upgraded, not at
        # the physical position's earlier entry. Physical episode age is unchanged.
        self.core_age+=self.core_life
        core_current=np.where(self.core_life,current,0.)
        core,core_change,core_reason=decide(mu,np.zeros(len(mu)),cov,beta,core_current,self.core_age,policy)
        weak_target=np.where(np.abs(mu)<=2*policy.design_fee*policy.multiplier,soft*m.overlay_fraction,0.)
        self.desired_history.append(core+weak_target)
        target=current.copy();change=np.zeros(len(mu),bool);reason=np.full(len(mu),'hold',object)
        for j,old in enumerate(current):
            core_action=bool(core_change[j] and (self.core_life[j] or abs(core[j])>1e-9))
            if core_action:
                target[j]=core[j];change[j]=True;reason[j]='core_'+str(core_reason[j])
                if not self.core_life[j] or current[j]*core[j]<=0:self.core_age[j]=0
                self.core_life[j]=abs(core[j])>1e-9;self.last_review[j]=row
                self.weak[j]=self.opposite[j]=0
                continue
            if self.core_life[j]:continue
            d=weak_target[j];weak=abs(d)<m.minimum_entry;opposite=old*d<0
            self.weak[j]=self.weak[j]+1 if weak else 0
            self.opposite[j]=self.opposite[j]+1 if opposite else 0
            if abs(old)<1e-9 and not weak:
                target[j]=d;change[j]=True;reason[j]='weak_prob_entry';self.last_review[j]=row
            elif abs(old)>1e-9:
                if m.fixed_expiry and age[j]>=policy.horizon:
                    target[j]=0.;change[j]=True;reason[j]='weak_expiry';self.last_review[j]=row
                elif age[j]>=m.minimum_hold:
                    if self.opposite[j]>=m.reversal_confirm:
                        target[j]=d;change[j]=True;reason[j]='weak_confirmed_reverse';self.last_review[j]=row
                    elif self.weak[j]>=m.weak_confirm:
                        target[j]=0.;change[j]=True;reason[j]='weak_support_lost';self.last_review[j]=row
                    elif not weak and not opposite and row-self.last_review[j]>=m.review_hours:
                        if abs(d-old)>max(m.absolute_band,m.relative_band*abs(old)):
                            target[j]=d;change[j]=True;reason[j]='weak_confidence_resize'
                        self.last_review[j]=row
        guarded=project(target,policy.cap,policy.cap*policy.net_fraction,policy.coin_cap)
        annual=np.sqrt(max(float(guarded@cov@guarded),1e-12))*np.sqrt(24*365.25)
        if annual>policy.annual_risk*1.15:guarded*=policy.annual_risk/max(annual,1e-9)
        extra=np.abs(guarded-target)>1e-8;change|=extra;reason[extra]='risk_limit'
        self.core_life &=np.abs(guarded)>1e-9
        self.role_history.append(self.core_life.copy())
        return guarded,change,reason

    def specification(self):return asdict(self.mapping)
