"""Unified information, low-freedom strategy selection and frozen account attribution."""
from pathlib import Path
from dataclasses import replace,asdict
import json
import numpy as np
from scipy.stats import rankdata
from .alpha_data import ROOT,BASE,HORIZONS,save_json
from .alpha_policy import Policy,policy_grid,calibrate,apply_calibration
from .alpha_account import account
from .review_data import identity

MODELS=('NN_direct','NN_joint','Ridge','LightGBM','Trend','Reversal','Frozen_v5')

def corr(a,b):
    k=np.isfinite(a)&np.isfinite(b);x=np.asarray(a)[k];y=np.asarray(b)[k]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>3 and x.std()>1e-10 and y.std()>1e-10 else 0.

def rankcorr(a,b):
    k=np.isfinite(a)&np.isfinite(b);return corr(rankdata(a[k]),rankdata(b[k])) if k.sum()>3 else 0.

def cross_ic(p,y):
    p=p-p.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True)
    den=np.sqrt((p*p).sum(1)*(y*y).sum(1));return np.divide((p*y).sum(1),den,out=np.zeros(len(p)),where=den>1e-12)

def information(p,y,dates,symbols):
    result={};months=dates.astype('datetime64[M]')
    for j,h in enumerate(HORIZONS):
        good=np.isfinite(y[:,:,j]).all(1)&np.isfinite(p[:,:,j]).all(1);pp=p[good,:,j];yy=y[good,:,j];dd=dates[good];mm=months[good]
        bycoin={s:{'IC_G':corr(pp[:,i],yy[:,i]),'RankIC_G':rankcorr(pp[:,i],yy[:,i])} for i,s in enumerate(symbols)};monthly={}
        for m in np.unique(mm):
            k=mm==m;monthly[str(m)]={s:{'IC_G':corr(pp[k,i],yy[k,i]),'RankIC_G':rankcorr(pp[k,i],yy[k,i]),'samples':int(k.sum())} for i,s in enumerate(symbols)}
        for s in symbols:
            ics=np.array([v[s]['IC_G'] for v in monthly.values()]);bycoin[s]['monthly_TS_ICIR']=float(ics.mean()/max(ics.std(ddof=1),1e-10)) if len(ics)>1 else 0.
        cx=cross_ic(pp,yy);rx=cross_ic(rankdata(pp,axis=1),rankdata(yy,axis=1))
        common=corr(pp.mean(1),yy.mean(1));relative=corr((pp-pp.mean(1,keepdims=True)).reshape(-1),(yy-yy.mean(1,keepdims=True)).reshape(-1))
        # Non-circular natural-calendar 72h blocks, joint market clocks resampled.
        blocks=((dd-dd[0].astype('datetime64[D]'))/np.timedelta64(72,'h')).astype(int);groups=[np.flatnonzero(blocks==b) for b in np.unique(blocks)];rng=np.random.default_rng(20261004);boot=[]
        for _ in range(200):
            ids=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))]);x=pp[ids]-pp[ids].mean(0);z=yy[ids]-yy[ids].mean(0);den=np.sqrt((x*x).sum(0)*(z*z).sum(0));ic=np.divide((x*z).sum(0),den,out=np.zeros(len(symbols)),where=den>1e-12);boot.append(float(ic.mean()))
        result[str(h)]={'by_coin':bycoin,'monthly':monthly,'mean_TS_IC_G':float(np.mean([v['IC_G'] for v in bycoin.values()])),'mean_TS_RankIC_G':float(np.mean([v['RankIC_G'] for v in bycoin.values()])),'mean_hourly_CS_IC_G':float(cx.mean()),'mean_hourly_CS_RankIC_G':float(rx.mean()),'common_IC_G':common,'relative_IC_G':relative,'TS_IC_72h_block_95pct':np.quantile(boot,[.025,.975]).tolist(),'hours':len(dd)}
    return result

def attribute(summary,arrays,episodes,symbols,dates):
    gain=arrays['pnl']+arrays['funding']-arrays['fees'];h=arrays['hours'];net=gain.sum(1);bycoin={};monthly={};days=dates[h].astype('datetime64[D]');dayp=np.array([net[days==d].sum() for d in np.unique(days)])
    for j,s in enumerate(symbols):
        eps=[e for e in episodes if e['symbol_id']==j];r=np.array([e['price_pnl']+e['funding']-e['fee'] for e in eps]);wins=r[r>0];losses=r[r<0]
        w=arrays['weights'][:,j]
        bycoin[s]={'net_pnl_contribution':float(gain[:,j].sum()),'price_pnl':float(arrays['pnl'][:,j].sum()),'funding':float(arrays['funding'][:,j].sum()),'fees':float(arrays['fees'][:,j].sum()),'mean_abs_weight':float(np.abs(w).mean()),'mean_long_weight':float(np.maximum(w,0).mean()),'mean_short_weight':float(np.maximum(-w,0).mean()),'long_price_pnl':float(arrays['long_pnl'][:,j].sum()),'short_price_pnl':float(arrays['short_pnl'][:,j].sum()),'episodes':len(eps),'win_rate':float((r>0).mean()) if len(r) else 0.,'profit_loss_ratio':float(wins.mean()/max(-losses.mean(),1e-12)) if len(wins) and len(losses) else 0.,'holding_mean':float(np.mean([e['holding_hours'] for e in eps])) if eps else 0.,'holding_q90':float(np.quantile([e['holding_hours'] for e in eps],.9)) if eps else 0.}
    for m in np.unique(dates[h].astype('datetime64[M]')):
        k=dates[h].astype('datetime64[M]')==m;ids=np.flatnonzero(k);monthly[str(m)]={'account_return':float(arrays['nav'][ids[-1]+1]/arrays['nav'][ids[0]]-1),'net_pnl':float(net[k].sum()),'by_coin':{s:float(gain[k,j].sum()) for j,s in enumerate(symbols)},'average_gross':float(np.abs(arrays['weights'][k]).sum(1).mean()),'turnover':float(arrays['turnover'][k].sum())}
    state=np.load(BASE/'state_dynamic.npy',mmap_mode='r')[h];risklevel=state[:,:,6].mean(1)
    historical_risk=np.load(BASE/'state_dynamic.npy',mmap_mode='r')[:np.searchsorted(dates,np.datetime64('2025-07-01')), :,6].mean(1);historical_risk=historical_risk[np.isfinite(historical_risk)]
    assert len(historical_risk)>0 and np.isfinite(risklevel).all()
    threshold=np.median(historical_risk);pastmarket=state[:,:,2].mean(1)
    states={}
    for label,k in [('past_up',pastmarket>=0),('past_down',pastmarket<0),('high_risk',risklevel>=threshold),('low_risk',risklevel<threshold)]:states[label]={'net_pnl':float(net[k].sum()),'hours':int(k.sum()),'mean_gross':float(np.abs(arrays['weights'][k]).sum(1).mean()) if k.any() else 0.}
    return {'by_coin':bycoin,'monthly':monthly,'states':states,'risk_state_threshold_past_log_sigma24h':float(threshold),'top1_coin_pnl':float(gain.sum(0).max()),'net_pnl_without_best_coin':float(net.sum()-gain.sum(0).max()),'top5_days_pnl':float(np.sort(dayp)[-5:].sum()),'net_pnl_without_top5_days':float(net.sum()-np.sort(dayp)[-5:].sum()),'holding_hours': [e['holding_hours'] for e in episodes],'account_identity_error':float(abs(summary['return']-net.sum()))}

def refresh_saved_attributions():
    """Recalculate descriptive attribution without rerunning model/policy selection."""
    results=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));dates=np.load(BASE/'decision_time.npy');symbols=results['market']['symbols']
    for name,model in results['models'].items():model['attribution']=attribute(model['cost_scenarios']['4bp'],np.load(ROOT/f'account_{name}_4bp.npz'),json.loads((ROOT/f'episodes_{name}.json').read_text(encoding='utf-8')),symbols,dates)
    results['identity']=identity([Path(__file__),Path('src/crypto_timing/alpha_policy.py'),Path('src/crypto_timing/alpha_account.py')]);save_json(ROOT/'results.json',results)

def daily_attribution(strategy,market):
    y=np.asarray(strategy);x=np.asarray(market)[:len(y)];xx=np.column_stack((np.ones(len(y)),x));b=np.linalg.lstsq(xx,y,rcond=None)[0];e=y-xx@b
    meat=(xx*e[:,None]).T@(xx*e[:,None]);lags=min(7,len(y)-1)
    for lag in range(1,lags+1):
        cross=(xx[lag:]*e[lag:,None]).T@(xx[:-lag]*e[:-lag,None]);meat+=(1-lag/(lags+1))*(cross+cross.T)
    inv=np.linalg.pinv(xx.T@xx);se=np.sqrt(np.maximum(np.diag(inv@meat@inv),0))
    return {'daily_intercept':float(b[0]),'annualized_arithmetic_intercept':float(b[0]*365.25),'market_beta':float(b[1]),'intercept_HAC7_t':float(b[0]/max(se[0],1e-12)),'meaning':'descriptive ex-post attribution, not tradable hindsight hedge'}

def equity_interval(daily,other=None):
    groups=[np.arange(k,min(k+7,len(daily))) for k in range(0,len(daily),7)];rng=np.random.default_rng(20261004);values=[]
    for _ in range(300):
        ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);a=np.prod(1+daily[ix])-1
        if other is not None:a-=np.prod(1+other[ix])-1
        values.append(float(a))
    return {'calendar_7day_block_95pct':np.quantile(values,[.025,.975]).tolist(),'replicates':300,'meaning':'dependent historical uncertainty; not corrected for development selection'}

def analyze():
    dates=np.load(BASE/'decision_time.npy');meta=json.loads((ROOT/'market_manifest.json').read_text(encoding='utf-8'));symbols=meta['symbols'];n=len(dates);ns=len(symbols)
    raw={name:np.full((n,ns,4),np.nan,np.float32) for name in MODELS};risks=np.full((n,ns,4),np.nan,np.float32)
    months=np.arange(np.datetime64('2025-01','M'),np.datetime64('2026-10','M'))
    for month in months:
        f=np.load(ROOT/'monthly'/str(month)/'predictions.npz');hh=f['hours'];risks[hh]=f['risk']
        for name in MODELS:raw[name][hh]=f[name]
        fair=np.load(ROOT/'monthly'/str(month)/'fair_baselines.npz');assert np.array_equal(hh,fair['hours'])
        for name in ('Ridge','LightGBM'):raw[name][hh]=fair[name]
    from .alpha_legacy import anchor,legacy_scores,native_curve
    raw['Frozen_v5']=np.repeat(anchor()[...,None],4,-1)
    target=np.load(ROOT/'target.npy');price=np.load(ROOT/'price.npy');funding=np.load(ROOT/'funding_cash_unit.npy');expected=np.load(ROOT/'expected_funding_hour.npy');minute=np.load(ROOT/'minute_open.npy',mmap_mode='r');fm=np.load(ROOT/'funding_cash_minute.npy',mmap_mode='r')
    calibrated={name:np.full_like(raw[name],np.nan) for name in MODELS};calmeta={};quantiles={}
    for month in months:
        cutoff=np.datetime64(month,'ns');nextcut=np.datetime64(month+1,'ns');hh=np.flatnonzero((dates>=cutoff)&(dates<nextcut)&np.isfinite(risks).all((1,2)))
        history=np.flatnonzero((dates>=cutoff-np.timedelta64(90,'D'))&(dates+np.timedelta64(721,'m')<=cutoff)&np.isfinite(risks).all((1,2))&np.isfinite(target).all((1,2)))
        for name in MODELS:
            if len(history)<240:calibrated[name][hh]=raw[name][hh];continue
            fit=calibrate(raw[name][history],target[history],risks[history]);calibrated[name][hh]=apply_calibration(raw[name][hh],risks[hh],fit)
            calmeta[f'{month}/{name}']={'fit_hours':len(history),'latest_target_maturity':str(dates[history[-1]]+np.timedelta64(721,'m')),'cutoff':str(cutoff),'fit':fit}
    save_json(ROOT/'calibration.json',calmeta)
    dev=np.flatnonzero((dates>=np.datetime64('2025-03-01'))&(dates<np.datetime64('2025-07-01')))
    frozen=np.flatnonzero((dates>=np.datetime64('2025-07-01'))&np.isfinite(price).all(1)&(np.arange(n)<n-1))
    infos={};results={'design':'rule selection only March-June; all existing history exploratory','models':{},'information':{},'calibration_file':'calibration.json','market':meta}
    for name in MODELS:
        infos[name]={}
        for label,hh in [('development',dev),('frozen',frozen),('F1',frozen[dates[frozen]<np.datetime64('2026-02-01')]),('F2',frozen[dates[frozen]>=np.datetime64('2026-02-01')])]:
            infos[name][label]={'raw':information(raw[name][hh],target[hh],dates[hh],symbols),'calibrated':information(calibrated[name][hh],target[hh],dates[hh],symbols)}
        print(f'information complete {name}',flush=True)
    results['information']=infos;save_json(ROOT/'information.json',infos)
    grid=policy_grid();shared=Policy();selected={};development={}
    def run(hh,sig,pol,fee=.0004,mode='economic',retain=False,minute_check=False):
        return account(price,sig,funding,expected,dates,hh,pol,fee,minute if minute_check else None,fm if minute_check else None,mode,retain)
    for name in MODELS:
        records=[]
        for pol in grid:
            j=list(HORIZONS).index(pol.horizon);r,*_=run(dev,calibrated[name][dev,:,j],pol)
            utility=r['return']-.5*abs(r['max_drawdown']) if not r['bankrupt'] else -1e6
            records.append({'policy':asdict(pol),'result':r,'utility':utility})
        best=max(records,key=lambda r:r['utility']);selected[name]=Policy(**best['policy']);development[name]={'candidates':records,'chosen':best}
        print(f'policy {name} {best["policy"]} dev return {best["result"]["return"]:.4f}',flush=True)
    # A five-weight mixture is selected in development and then frozen.
    base=max(('Ridge','LightGBM','Trend','Reversal'),key=lambda name:development[name]['chosen']['utility']);mixtures=[]
    for weight in (0.,.25,.5,.75,1.):
        pol=selected['NN_joint'];j=list(HORIZONS).index(pol.horizon);sig=weight*calibrated['NN_joint'][dev,:,j]+(1-weight)*calibrated[base][dev,:,j];r,*_=run(dev,sig,pol);mixtures.append({'neural_weight':weight,'baseline':base,'utility':r['return']-.5*abs(r['max_drawdown']),'result':r})
    mixture=max(mixtures,key=lambda r:r['utility']);results['mixture_development']=mixtures;results['mixture_frozen']=mixture
    calibrated['Frozen_mix']=mixture['neural_weight']*calibrated['NN_joint']+(1-mixture['neural_weight'])*calibrated[base];selected['Frozen_mix']=selected['NN_joint']
    save_json(ROOT/'policies.json',{'selected':{name:asdict(p) for name,p in selected.items()},'shared':asdict(shared),'development':development,'mixture':mixture,'search_count':len(grid)*len(MODELS)+len(mixtures),'selection_cost_per_side':.0004})
    for name in (*MODELS,'Frozen_mix'):
        own=selected[name];j=list(HORIZONS).index(own.horizon);sj=list(HORIZONS).index(shared.horizon)
        equal,*_=run(frozen,calibrated[name][frozen,:,sj],shared)
        results['models'][name]={'shared_policy_4bp':equal,'own_policy':asdict(own),'cost_scenarios':{}}
        for fee,label in ((0.,'gross'),(.0002,'2bp'),(.0004,'4bp')):
            r,a,e,t=run(frozen,calibrated[name][frozen,:,j],own,fee,retain=label=='4bp',minute_check=True)
            np.savez_compressed(ROOT/f'account_{name}_{label}.npz',**a)
            results['models'][name]['cost_scenarios'][label]=r
            if label=='4bp':
                results['models'][name]['attribution']=attribute(r,a,e,symbols,dates);save_json(ROOT/f'trades_{name}.json',t);save_json(ROOT/f'episodes_{name}.json',e)
        print(f'frozen account {name} {results["models"][name]["cost_scenarios"]["4bp"]["return"]:.4f}',flush=True)
    results['leverage']={}
    for cap in (1.,2.,3.,5.):
        pol=replace(selected['NN_joint'],cap=cap);j=list(HORIZONS).index(pol.horizon);results['leverage'][str(cap)]={}
        for fee,label in ((0.,'gross'),(.0002,'2bp'),(.0004,'4bp')):
            r,a,e,t=run(frozen,calibrated['NN_joint'][frozen,:,j],pol,fee,minute_check=True);results['leverage'][str(cap)][label]=r;np.savez_compressed(ROOT/f'leverage_{cap}_{label}.npz',**a)
    # Requested near-full exposure is a predeclared stress/control, not forced evidence of alpha.
    pol=replace(selected['NN_joint'],cap=2.);j=list(HORIZONS).index(pol.horizon)
    r,a,e,t=run(frozen,calibrated['NN_joint'][frozen,:,j],pol,mode='full_rank',minute_check=True);results['full_rank_control']=r;np.savez_compressed(ROOT/'full_rank_control.npz',**a)
    # Market and cash references use exactly the same price/funding/fee account.
    r,a,e,t=run(frozen,np.zeros((len(frozen),ns)),Policy(horizon=12),mode='market',minute_check=True);results['market_reference']=r;np.savez_compressed(ROOT/'market_reference.npz',**a);market_daily=a['daily_returns'];results['cash_reference']={'return':0.,'max_drawdown':0.}
    for name in results['models']:
        a=np.load(ROOT/f'account_{name}_4bp.npz');results['models'][name]['market_attribution']=daily_attribution(a['daily_returns'],market_daily);results['models'][name]['net_return_interval']=equity_interval(a['daily_returns'])
    neural_daily=np.load(ROOT/'account_NN_joint_4bp.npz')['daily_returns'];base_daily=np.load(ROOT/f'account_{base}_4bp.npz')['daily_returns'];results['paired_neural_increment']=equity_interval(neural_daily,base_daily);results['paired_neural_increment']['baseline_chosen_in_development']=base
    # Predictor fixed, policy changed: phase-style versus economic versus original v5 contract.
    economic,*_=run(frozen,calibrated['Frozen_v5'][frozen,:,1],Policy(horizon=4))
    r,a,e,t=run(frozen,calibrated['Frozen_v5'][frozen,:,1],Policy(horizon=4),mode='phase');results['policy_bridge']={'same_predictor_unified_phase':r,'same_predictor_economic_4h':economic,'old_v5_native':json.loads(Path('outputs/review_v5/analysis_results.json').read_text(encoding='utf-8'))['folds']}
    np.savez_compressed(ROOT/'bridge_phase.npz',**a)
    legacy=legacy_scores();common=frozen[np.isfinite(legacy[frozen]).all(1)];segments=np.split(common,np.flatnonzero(np.diff(common)>1)+1);common=max(segments,key=len)
    # Exact common clocks only: no zeros imputed into missing legacy predictions.
    results['legacy_common_bridge']={'hours':len(common),'models':{}}
    for name,sig in [('v3_saved_score',legacy[common]),('v5_same_policy',calibrated['Frozen_v5'][common,:,1]),('new_joint_same_policy',calibrated['NN_joint'][common,:,1])]:
        r,a,e,t=run(common,sig,Policy(horizon=4));results['legacy_common_bridge']['models'][name]=r;np.savez_compressed(ROOT/f'bridge_{name}.npz',**a)
    native_curve();results['legacy_common_bridge']['date_start']=str(dates[common[0]]);results['legacy_common_bridge']['date_end']=str(dates[common[-1]]);results['legacy_common_bridge']['scope']='longest contiguous common forecast interval; absent forecasts never filled with zero'
    # Incremental information: development-only projection on reference scores.
    controls=np.stack([raw[k][:,:,1] for k in ('Ridge','LightGBM','Trend','Reversal')],-1);xx=controls[dev].reshape(-1,4);yy=raw['NN_joint'][dev,:,1].reshape(-1);c=xx.mean(0);sd=np.maximum(xx.std(0),1e-5);z=(xx-c)/sd;coef=np.linalg.solve(z.T@z+100*np.eye(4),z.T@(yy-yy.mean()))
    predicted=yy.mean()+((controls[frozen]-c)/sd)@coef;residual=raw['NN_joint'][frozen,:,1]-predicted
    results['incremental_information']={'neural_residual_IC_G4':corr(residual.reshape(-1),target[frozen,:,1].reshape(-1)),'neural_baseline_signal_correlations':{k:corr(raw['NN_joint'][frozen,:,1].reshape(-1),raw[k][frozen,:,1].reshape(-1)) for k in ('Ridge','LightGBM','Trend','Reversal')},'projection_fit':'development only; residualization descriptive and deployable in signal space','coefficients':coef.tolist()}
    cy=target[dev,:,1].reshape(-1);cymean=cy.mean();bc=np.linalg.solve(z.T@z+100*np.eye(4),z.T@(cy-cymean));yres=target[frozen,:,1]-(cymean+((controls[frozen]-c)/sd)@bc);results['incremental_information']['baseline_controlled_partial_IC_G4']=corr(residual.reshape(-1),yres.reshape(-1))
    results['leave_one_coin_out']={};pol=selected['NN_joint'];j=list(HORIZONS).index(pol.horizon)
    for i,s in enumerate(symbols):
        sig=calibrated['NN_joint'][frozen,:,j].copy();sig[:,i]=0;r,*_=run(frozen,sig,pol);results['leave_one_coin_out'][s]={'return_4bp':r['return'],'drawdown':r['max_drawdown'],'meaning':'disable that coin, rerun unchanged policy and constraints; no refit'}
    # OOF quantile boundaries fixed before each forecast month; costs applied as an illustrative completed round trip.
    cumulative=np.concatenate((np.zeros((1,ns)),np.cumsum(funding,axis=0)))
    for name in MODELS:
        quantiles[name]={}
        for j,horizon in enumerate(HORIZONS):
            buckets=[[] for _ in range(10)];netb=[[] for _ in range(10)]
            for month in months[6:]:
                fit=calmeta.get(f'{month}/{name}');
                if fit is None:continue
                hh=frozen[dates[frozen].astype('datetime64[M]')==month];hh=hh[np.isfinite(target[hh,:,j]).all(1)]
                bins=np.searchsorted(fit['fit'][j]['quantile_edges'],calibrated[name][hh,:,j]/risks[hh,:,j]);ff=(cumulative[hh+horizon]-cumulative[hh])/price[hh]
                for b in range(10):
                    k=bins==b;ret=target[hh,:,j][k];sgn=-1 if b<5 else 1;buckets[b].extend(ret.tolist());netb[b].extend((sgn*ret-sgn*ff[k]-.0008).tolist())
            quantiles[name][str(horizon)]=[{'bin':b,'side':'short' if b<5 else 'long','samples':len(x),'raw_mean':float(np.mean(x)) if x else 0.,'directional_4bp_roundtrip_mean':float(np.mean(netb[b])) if x else 0.,'win_fraction_directional':float(np.mean(np.array(x)*(-1 if b<5 else 1)>0)) if x else 0.,'q01':float(np.quantile(x,.01)) if x else 0.} for b,x in enumerate(buckets)]
    results['quantiles']=quantiles
    # Old native artifacts are deliberately not relabelled as same-data competitors.
    results['legacy_v3']={'selection':json.loads(Path('outputs/redesign/evaluation/selection.json').read_text(encoding='utf-8'),parse_constant=lambda _:None),'historical':json.loads(Path('outputs/redesign/evaluation/historical_diagnostic.json').read_text(encoding='utf-8'),parse_constant=lambda _:None),'status':'native legacy replay retained; old 5m inputs are not identically portable to the new 1m contract; omitted from fair leaderboard; undefined legacy correlations remain null'}
    results['identity']=identity([Path(__file__),Path('src/crypto_timing/alpha_policy.py'),Path('src/crypto_timing/alpha_account.py')]);save_json(ROOT/'results.json',results)
    # Unified prediction records are compact arrays plus an explicit business schema.
    np.savez_compressed(ROOT/'unified_predictions.npz',hours=frozen,decision=dates[frozen],visible_until=dates[frozen],earliest_fill=dates[frozen]+np.timedelta64(1,'m'),target_horizons=HORIZONS,risk=risks[frozen],**{name:raw[name][frozen] for name in MODELS},**{name+'_calibrated':calibrated[name][frozen] for name in (*MODELS,'Frozen_mix')})
    save_json(ROOT/'prediction_schema.json',{'symbols':symbols,'units':'raw simple return per horizon; risk past scale; decision/visibility UTC','model_cutoff':'month start for monthly readouts and baselines; v5 anchor retains original Nov24/Mar25/Jul25/Feb26 switches','encoder_cutoff':'2024-11-01','calibration_cutoff':'month start; latest 90d matured OOF','Frozen_v5_target':'native 4h+5m; next1m and other horizons are OOF projections','policy_selection_cutoff':'2025-07-01','status':'historical frozen-rule replay, not newly arrived forward evidence'})
    from .alpha_inference import AlphaPredictor
    from .alpha_policy import covariance
    latest=int(frozen[-1]);cov,beta=covariance(price,latest-1);predictor=AlphaPredictor(ROOT/'monthly/2026-09',calmeta['2026-09/NN_joint']['fit'],asdict(selected['NN_joint']))
    live=predictor.predict_completed(np.load(ROOT/'representation.npy',mmap_mode='r')[latest-3:latest+1],np.load(BASE/'targets.npy',mmap_mode='r')[latest,:,7],cov,beta,expected[latest])
    assert np.max(np.abs(np.asarray(live['raw_returns'])-raw['NN_joint'][latest]))<2e-6,'live/history head mismatch'
    live['decision_utc']=str(dates[latest]);live['earliest_fill_utc']=str(dates[latest]+np.timedelta64(1,'m'));live['observed_only']=True;save_json(ROOT/'latest_label_free_prediction.json',live)
    print('analysis complete',flush=True)
