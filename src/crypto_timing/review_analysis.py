"""Bias/dynamics, frozen OOF calibration, calendar-time diagnostics and research NAV."""
from __future__ import annotations
import numpy as np
from .review_training import decomposition
from .mechanism_training import correlation

def time_interval(a,b,y,dates,valid,block_hours=72,repetitions=400):
    # Natural-time windows with no circular wrapping across endpoint/calendar gaps.
    av=np.where(valid,(a-y)**2,np.nan);bv=np.where(valid,(b-y)**2,np.nan)
    diff=np.nanmean(av-bv,axis=1);den=np.nanmean(np.where(valid,y*y,np.nan),axis=1)
    stop=np.searchsorted(dates,dates+np.timedelta64(block_hours,'h'),side='left')
    starts=np.flatnonzero(dates+np.timedelta64(block_hours,'h')<=dates[-1]+np.timedelta64(1,'h'))
    assert len(starts)>0
    rng=np.random.default_rng(20261004);est=[];n=len(dates)
    for _ in range(repetitions):
        chosen=[];count=0
        while count<n:
            k=int(rng.choice(starts));rows=np.arange(k,stop[k]);chosen.append(rows);count+=len(rows)
        ix=np.concatenate(chosen)[:n];est.append(float(np.nanmean(diff[ix])/np.nanmean(den[ix])))
    return {'increment':float(np.nanmean(diff)/np.nanmean(den)),'ci95':np.quantile(est,[.025,.975]).tolist(),'natural_block_hours':block_hours,'replicates':repetitions,'no_circular_wrap':True,'scope':'conditional on fixed historical predictions; no selection adjustment'}

def smooth(values,dates,half_life):
    if half_life==0:return values.copy()
    out=values.copy()
    for j in range(1,len(values)):
        gap=float((dates[j]-dates[j-1])/np.timedelta64(1,'h'));alpha=1-2**(-gap/half_life)
        out[j]=(1-alpha)*out[j-1]+alpha*values[j]
    return out

def state_groups(state,thresholds):
    return 2*(state[...,6]>thresholds[0]).astype(int)+(state[...,12]>thresholds[1]).astype(int)

def fit_calibrator(data,state,penalty,statewise=False):
    valid=data['valid'];d=data['delta'][valid].astype(float);res=(data['y']-data['baseline'])[valid].astype(float)
    center=float(d.mean());target=float(res.mean());slope=float(np.sum((d-center)*(res-target))/(np.sum((d-center)**2)+penalty*len(d)));slope=float(np.clip(slope,0,1))
    intercept=float((target-slope*center)/(1+10))
    thresholds=np.nanmedian(state[...,6][valid]),np.nanmedian(state[...,12][valid]);groups=state_groups(state,thresholds);slopes=[slope]*4
    if statewise:
        for g in range(4):
            k=valid&(groups==g);dd=data['delta'][k].astype(float);rr=(data['y']-data['baseline'])[k].astype(float)-intercept
            # Pool state slopes strongly toward the global estimate, rather than search regimes.
            numerator=float((dd*rr).sum()+penalty*len(dd)*slope);den=float((dd*dd).sum()+penalty*len(dd))
            slopes[g]=float(np.clip(numerator/max(den,1e-12),0,1))
    return {'penalty':penalty,'statewise':statewise,'slope':slope,'state_slopes':slopes,'intercept':intercept,'thresholds':list(map(float,thresholds)),'baseline_coefficient':1.,'fit_samples':len(d),'intercept_shrinkage':10}

def apply_calibrator(data,state,config):
    slopes=np.asarray(config['state_slopes'])[state_groups(state,config['thresholds'])]
    return data['baseline']+config['intercept']+slopes*data['delta']

def concat_data(a,b):
    return {k:np.concatenate((a[k],b[k])) for k in a if isinstance(a[k],np.ndarray) and k in b}

def second_stage(train,val,train_state,val_state):
    candidates=[]
    for statewise in (False,True):
        for penalty in (.0001,.001,.01,.1):
            c=fit_calibrator(train,train_state,penalty,statewise);pred=apply_calibrator(val,val_state,c)
            candidates.append({'config':c,'validation':decomposition(pred,val['y'],val['valid'])})
    chosen=min(candidates,key=lambda r:r['validation']['mse'])['config']
    # Freeze filter selection on out-of-fold signals, never the encoder's own training predictions.
    combined=concat_data(train,val);states=np.concatenate((train_state,val_state));final=fit_calibrator(combined,states,chosen['penalty'],chosen['statewise'])
    vpred=apply_calibrator(val,val_state,chosen);filters=[]
    for unit in ('normalized','raw_return'):
        for tau in (0,1,2,4):
            values=vpred if unit=='normalized' else vpred*val['scale']; pp=smooth(values,val['dates'],tau)
            if unit=='raw_return': pp=pp/val['scale']
            filters.append({'unit':unit,'half_life':tau,'validation':decomposition(pp,val['y'],val['valid'])})
    selected_filter=min(filters,key=lambda v:v['validation']['mse'])
    return {'calibrator':final,'calibration_candidates':candidates,'filter_candidates':filters,'filter':{k:selected_filter[k] for k in ('unit','half_life')},'fitting_material':'time-out-of-fold fixed-epoch neural predictions; no outer labels','final_refit':'OOF fit+one-month OOF validation, before outer cutoff'}

def diagnostic_breakdown(data,state):
    p,y,v=data['signal'],data['y'],data['valid'];error=(p-y)**2
    flat=error[v]; order=np.sort(flat); q2=data['raw_path'][...,1][v]**2
    crash=(data['dates']>=np.datetime64('2025-10-10T17'))&(data['dates']<np.datetime64('2025-10-11T01'))
    common_p=np.nanmean(np.where(v,p,np.nan),1); common_y=np.nanmean(np.where(v,y,np.nan),1)
    per_asset={str(j):decomposition(p[:,j],y[:,j],v[:,j]) for j in range(y.shape[1])}
    residual=decomposition(p-common_p[:,None],y-common_y[:,None],v)
    risk_threshold=float(np.nanmedian(state[...,6][v]));activity_threshold=float(np.nanmedian(state[...,12][v]));groups=state_groups(state,(risk_threshold,activity_threshold))
    strata={f'risk{g//2}_activity{g%2}':decomposition(p,y,v&(groups==g)) for g in range(4)}
    responses={f'future_segment_{j+1}':correlation(p[v],data['segment_raw'][...,j][v]) for j in range(5)}
    by_clock=error.mean(1);topclock=np.argsort(by_clock)[-12:]
    return {'per_asset':per_asset,'market_common':decomposition(common_p,common_y),'within_clock_residual':residual,'states':strata,'segment_correlations':responses,'top1pct_error_share':float(order[-max(1,len(order)//100):].sum()/order.sum()),'largest_12_clock_error_share':float(by_clock[topclock].sum()/by_clock.sum()),'largest_12_clock_dates':[str(x) for x in data['dates'][topclock]],'crash_clock_error_share':float(by_clock[crash].sum()/by_clock.sum()),'without_crash_diagnostic':decomposition(p[~crash],y[~crash],v[~crash]),'Q_top1pct_square_share':float(np.sort(q2)[-max(1,len(q2)//100):].sum()/q2.sum()),'state_threshold_status':'descriptive evaluation strata, never used to fit/freeze deployed calibration','leave_one_asset_out_correlation':[correlation(np.delete(p,j,1),np.delete(y,j,1)) for j in range(y.shape[1])]}

def lifetime(signal,segment_raw,dates,valid):
    exact={};ac={};lookup={int(d.astype('datetime64[h]').astype(int)):i for i,d in enumerate(dates)}
    for lag in (1,2,4,8):
        a=[];b=[]
        for i,d in enumerate(dates):
            j=lookup.get(int(d.astype('datetime64[h]').astype(int))+lag)
            if j is not None:a.append(i);b.append(j)
        if not a: ac[str(lag)]=None;continue
        mask=valid[a]&valid[b];ac[str(lag)]=correlation(signal[a][mask],signal[b][mask])
    for j in range(5): exact[SEGMENT_LABELS[j]]=correlation(signal[valid],segment_raw[...,j][valid])
    return {'exact_hour_autocorrelation':ac,'disjoint_forward_correlations':exact,'interpretation':'observed signal persistence and nonoverlap response, not an optimal holding period'}

SEGMENT_LABELS=('0_1h','1_2h','2_3h','3_4h','4_8h')

def nav_backtest(signal,hours,dates,next_returns,fee_bps=4):
    """Four equal capital sleeves; mark once per exact one-hour open-to-open interval."""
    assert np.all(np.diff(hours)==1),'NAV needs complete hourly clocks; do not stitch gaps'
    ns=signal.shape[1];sleeves=np.zeros((4,ns));prev=np.zeros(ns);gross=[];net=[];turn=[];weights=[]
    for i,h in enumerate(hours):
        sleeves[h%4]=np.tanh(signal[i]/.10)/(4*ns)
        w=sleeves.sum(0);r=float(w@next_returns[h]);fee=fee_bps/10000;cost=0.
        # Aggregate target weights rebalance hourly. Include mark-to-market drift turnover.
        for _ in range(5):cost=fee*np.abs(w*(1-cost)-prev).sum()
        turnover=np.abs(w*(1-cost)-prev).sum()
        gross.append(r);net.append((1-cost)*(1+r)-1);turn.append(turnover);weights.append(w.copy())
        prev=w*(1+next_returns[h])/(1+r)
    # End liquidation is charged once, at the endpoint, with no fictitious extra return.
    liquidation=np.abs(prev).sum()*fee_bps/10000
    net[-1]=(1+net[-1])*(1-liquidation)-1;turn[-1]+=np.abs(prev).sum()
    gross=np.asarray(gross);net=np.asarray(net);nav=np.cumprod(1+net);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
    weight=np.asarray(weights)
    assert np.max(np.abs(weight).sum(1))<=1+1e-9
    return {'nav':nav,'gross_nav':np.cumprod(1+gross),'net_returns':net,'gross_returns':gross,'turnover':np.asarray(turn),'weights':weight,'dates':dates+np.timedelta64(65,'m'),
            'summary':{'total_return':float(nav[-1]-1),'gross_return':float(np.prod(1+gross)-1),'max_drawdown':float(np.min(nav/peak-1)),'annualized_sharpe':float(net.mean()/max(net.std(),1e-12)*np.sqrt(8760)),'mean_gross_exposure':float(np.abs(weight).sum(1).mean()),'mean_net_exposure':float(weight.sum(1).mean()),'total_turnover':float(np.sum(turn)),'fee_bps_one_way':fee_bps,'hours':len(hours),'funding_included':False,'price_fill_model':'updated 1m open at decision+5m; four-phase constant-notional research approximation; no slippage beyond fee sensitivity'}}
