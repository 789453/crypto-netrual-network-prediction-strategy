"""Fixed forecasts, three development-only mappings, then zero-funding accounts."""
from pathlib import Path
from dataclasses import asdict,replace
import hashlib,json,sys,argparse
import numpy as np
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crypto_timing.alpha_policy import Policy
from crypto_timing.alpha_account import account
from crypto_timing.alpha_probability_policy import (
    ProbabilityMapping,ProbabilityDecision,monotone_probability,
    probability_from_saved,support_strength)
from crypto_timing.alpha_analysis import corr,rankcorr,equity_interval

ROOT=Path('outputs/alpha_strategy');BASE=Path('outputs/mechanism/cache_v4')
OUT=Path('outputs/probability_position_2026-10-04')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_stream():
    dates=np.load(BASE/'decision_time.npy');raw=np.load(ROOT/'v5_anchor_return4.npy')
    risk=np.full_like(raw,np.nan);mu=np.full_like(raw,np.nan);p_up=np.full_like(raw,np.nan)
    fits=read(ROOT/'calibration.json');tables={};used=[BASE/'decision_time.npy',ROOT/'v5_anchor_return4.npy',ROOT/'calibration.json',ROOT/'price.npy',ROOT/'execution.npy',ROOT/'market_manifest.json',ROOT/'system_lock.json',ROOT/'unified_predictions.npz',ROOT/'minute_open.npy',ROOT/'target.npy']
    for month in np.arange(np.datetime64('2025-01','M'),np.datetime64('2026-10','M')):
        source=ROOT/'monthly'/str(month)/'predictions.npz';d=np.load(source);used.append(source)
        h=d['hours'];risk[h]=d['risk'][:,:,3];record=fits.get(f'{month}/Frozen_v5')
        if record is None:continue
        assert np.datetime64(record['latest_target_maturity'])<=np.datetime64(record['cutoff'])<=dates[h].min()
        r=record['fit'][3];p=raw[h];cv=risk[h].mean(1);common=p.mean(1)
        mu[h]=(r['common_slope']*common/cv+r['common_intercept'])[:,None]*cv[:,None]+r['relative_slope']*(p-common[:,None])
        table=monotone_probability(r);table.update({k:record[k] for k in ('fit_hours','cutoff','latest_target_maturity')})
        tables[str(month)]=table;p_up[h]=probability_from_saved(mu[h],risk[h],table)
    saved=np.load(ROOT/'unified_predictions.npz');h=saved['hours']
    np.testing.assert_allclose(mu[h],saved['Frozen_v5_calibrated'][:,:,3],rtol=1e-5,atol=3e-8)
    # Use the exact saved float32 frozen stream, not a numerically reconstructed version.
    mu[h]=saved['Frozen_v5_calibrated'][:,:,3]
    for month in np.unique(dates[h].astype('datetime64[M]')):
        k=h[dates[h].astype('datetime64[M]')==month];p_up[k]=probability_from_saved(mu[k],risk[k],tables[str(month)])
    return dates,raw,risk,mu,p_up,tables,used,h

def segment(a,dates,mask):
    ids=np.flatnonzero(mask);gain=a['pnl']-a['fees'];nav=a['nav'][ids[0]:ids[-1]+2]
    return {'return':float(nav[-1]/nav[0]-1),'hour_boundary_drawdown':float((nav/np.maximum.accumulate(nav)-1).min()),
            'net_pnl_initial_equity':float(gain[ids].sum()),'mean_gross':float(np.abs(a['weights'][ids]).sum(1).mean()),
            'participation':float((np.abs(a['weights'][ids]).sum(1)>1e-6).mean()),'fee_initial_equity':float(a['fees'][ids].sum())}

def diagnostics(mu,p_up,target,h,dates,symbols):
    valid=np.isfinite(target[h,:,3]).all(1);hh=h[valid];m=mu[hh];y=target[hh,:,3];p=p_up[hh];up=y>0
    brier=float(np.mean((p-up)**2));base=float(up.mean())
    bins=[]
    for left,right in zip([0,.45,.48,.50,.52,.55],[.45,.48,.50,.52,.55,1.00001]):
        k=(p>=left)&(p<right);n=int(k.sum())
        bins.append({'probability_range':[left,min(right,1.)],'samples_correlated':n,'predicted_up':float(p[k].mean()) if n else None,'realized_up':float(up[k].mean()) if n else None,'mean_return':float(y[k].mean()) if n else None})
    bycoin={s:{'IC':corr(m[:,j],y[:,j]),'RankIC':rankcorr(m[:,j],y[:,j]),'Brier':float(np.mean((p[:,j]-up[:,j])**2))} for j,s in enumerate(symbols)}
    nonoverlap=(hh-hh[0])%12==0
    return {'IC_mean_TS':float(np.mean([v['IC'] for v in bycoin.values()])),
            'RankIC_mean_TS':float(np.mean([v['RankIC'] for v in bycoin.values()])),
            'Brier':brier,'Brier_50pct':.25,'Brier_expost_constant_descriptive':base*(1-base),
            'nonoverlap_clock_Brier':float(np.mean((p[nonoverlap]-up[nonoverlap])**2)),
            'calibration_bins':bins,'by_coin':bycoin,'valid_hours':len(hh),'nonoverlap_clocks':int(nonoverlap.sum()),
            'meaning':'Directional probability from saved past bins; overlapping labels/cross-coins correlated. Ex-post constant is descriptive, not a deployable baseline.'}

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['empirical','implied','core_overlay'],default='empirical');args=parser.parse_args()
    if args.family=='implied':OUT=Path('outputs/probability_position_implied_2026-10-04')
    if args.family=='core_overlay':OUT=Path('outputs/probability_position_core_overlay_2026-10-04')
    OUT.mkdir(parents=True,exist_ok=True)
    dates,raw,risk,mu,p_up,tables,used,frozen=load_stream();source={str(p):digest(p) for p in used}
    empirical=p_up.copy()
    if args.family in ('implied','core_overlay'):p_up=ndtr(mu/risk).astype(np.float32)
    price=np.load(ROOT/'price.npy');execution=np.load(ROOT/'execution.npy');minute=np.load(ROOT/'minute_open.npy',mmap_mode='r')
    zeros=np.zeros_like(price);pol=Policy(**read(ROOT/'system_lock.json')['policy'])
    dev=np.flatnonzero((dates>=np.datetime64('2025-03-01'))&(dates<np.datetime64('2025-07-01')))
    assert np.isfinite(mu[dev]).all() and np.isfinite(p_up[dev]).all()
    candidates=[ProbabilityMapping(confidence_full=.05),ProbabilityMapping(confidence_full=.02),ProbabilityMapping(shape='discrete')]
    if args.family=='implied':candidates=[ProbabilityMapping(confidence_full=x) for x in (.01,.02,.05)]
    if args.family=='core_overlay':candidates=[ProbabilityMapping(confidence_full=.02,core_overlay=True,overlay_fraction=x,minimum_entry=.0002,absolute_band=.003) for x in (.05,.1,.2)]
    development=[]
    for spec in candidates:
        adapter=ProbabilityDecision(p_up[dev],spec)
        r,*_=account(price,mu[dev],zeros,zeros,dates,dev,pol,fee=.0002,decision_adapter=adapter)
        utility=r['return']-.5*abs(r['max_drawdown']) if not r['bankrupt'] else -1e6
        development.append({'mapping':asdict(spec),'result':r,'utility':utility})
        print('development',spec.shape,spec.confidence_full,'return',r['return'],'gross',r['average_gross'],flush=True)
    chosen=max(development,key=lambda x:x['utility']);mapping=ProbabilityMapping(**chosen['mapping'])
    lock={'unchanged_model':'Frozen_v5','probability_family':args.family,'unchanged_saved_forecast_sha256':source[str(ROOT/'unified_predictions.npz')],
          'development_window':'2025-03-01 <= t < 2025-07-01','objective':'return - .5*abs(hour boundary drawdown)',
          'candidates':development,'chosen_mapping':asdict(mapping),'cost_per_side':.0002,'funding':False,
          'historical_status':'previously inspected history; families added after inspecting earlier families; not new forward evidence',
          'probability_semantics':'saved shrunk directional bin frequency' if args.family=='empirical' else 'Normal symmetric noise assumption: Phi(unchanged return forecast / past horizon risk); uncalibrated implied probability',
          'policy':asdict(pol)}
    save(OUT/'mapping_lock.json',lock)
    outputs={};arrays={};episodes={};trades={}
    for name in ['probability','old_hard_gate','rank','fixed12_ablation']:
        adapter=None;mode='economic'
        if name in ('probability','fixed12_ablation'):
            spec=mapping if name=='probability' else replace(mapping,fixed_expiry=True)
            adapter=ProbabilityDecision(p_up[frozen],spec)
        elif name=='rank':mode='full_rank'
        r,a,e,t=account(price,mu[frozen],zeros,zeros,dates,frozen,pol,fee=.0002,minute=minute,mode=mode,retain=True,decision_adapter=adapter)
        assert not r['bankrupt'] and np.isclose(r['return'],(a['pnl']-a['fees']).sum(),atol=1e-10)
        assert np.isclose(r['fee'],.0002*a['traded_notional'].sum(),atol=1e-10)
        assert np.all(a['funding']==0)
        if adapter is not None:
            a['desired_weights']=np.asarray(adapter.desired_history)
            if adapter.mapping.core_overlay:a['core_role']=np.asarray(adapter.role_history)
        np.savez_compressed(OUT/f'account_{name}.npz',**a);save(OUT/f'episodes_{name}.json',e);save(OUT/f'trades_{name}.json',t)
        outputs[name]=r;arrays[name]=a;episodes[name]=e;trades[name]=t
        print('historical',name,'return',r['return'],'gross',r['average_gross'],'participation',r['participation'],flush=True)
    symbols=read(ROOT/'market_manifest.json')['symbols'];target=np.load(ROOT/'target.npy')
    info=diagnostics(mu,p_up,target,frozen,dates,symbols);partitions={};coins={}
    for name,a in arrays.items():
        d=dates[a['hours']];gain=a['pnl']-a['fees'];partitions[name]={}
        for period,k in [('F1',d<np.datetime64('2026-02-01')),('F2',d>=np.datetime64('2026-02-01'))]:partitions[name][period]=segment(a,dates,k)
        for month in np.unique(d.astype('datetime64[M]')):partitions[name][str(month)]=segment(a,dates,d.astype('datetime64[M]')==month)
        coins[name]={}
        for j,s in enumerate(symbols):
            eps=[x for x in episodes[name] if x['symbol_id']==j]
            coins[name][s]={'net_contribution_initial_equity':float(gain[:,j].sum()),'fee_initial_equity':float(a['fees'][:,j].sum()),
                           'mean_abs_weight':float(np.abs(a['weights'][:,j]).mean()),'holding_mean':float(np.mean([x['holding_hours'] for x in eps])) if eps else 0.,
                           'episodes':len(eps),'long_price_pnl':float(a['long_pnl'][:,j].sum()),'short_price_pnl':float(a['short_pnl'][:,j].sum())}
    # Attribute sub-8bp ENTRY episodes, not all their later hours to a contemporaneous signal.
    weak=[];other=[];oldgate=2*pol.design_fee*pol.multiplier
    for e in episodes['probability']:
        entry_abs=abs(float(mu[e['start_hour'],e['symbol_id']]))
        (weak if entry_abs<oldgate else other).append(e)
    def group(e):
        pn=np.array([x['price_pnl']-x['fee'] for x in e])
        return {'episodes':len(e),'net_pnl_initial_equity':float(pn.sum()),'price_pnl_initial_equity':float(sum(x['price_pnl'] for x in e)),
                'fees_initial_equity':float(sum(x['fee'] for x in e)),'episode_win_rate':float((pn>0).mean()) if len(e) else 0.,
                'holding_mean':float(np.mean([x['holding_hours'] for x in e])) if e else 0.}
    # Episode cost attribution must reconcile even when sizing within an episode changes.
    weak_result={'below_old_8bp_entry':group(weak),'at_or_above_old_8bp_entry':group(other),
                 'meaning':'Group whole episode by unchanged forecast amplitude at initial entry; resizing fees stay in that episode.'}
    assert abs(weak_result['below_old_8bp_entry']['net_pnl_initial_equity']+weak_result['at_or_above_old_8bp_entry']['net_pnl_initial_equity']-outputs['probability']['return'])<1e-9
    role_attribution=None
    if mapping.core_overlay:
        a=arrays['probability'];role=a['core_role'];fee_core=np.zeros_like(a['fees']);fee_weak=np.zeros_like(a['fees'])
        for t in trades['probability']:
            row=int(np.searchsorted(frozen,t['hour']));j=t['symbol_id'];old=t['quantity_before'];new=t['quantity_after'];price_t=t['price']
            if row==len(frozen):parts=[(len(frozen)-1,bool(role[-1,j]),t['fee'])]
            else:
                previous=bool(role[row-1,j]) if row else False;current=bool(role[row,j])
                if old*new<0:parts=[(row,previous,.0002*abs(old)*price_t),(row,current,.0002*abs(new)*price_t)]
                else:parts=[(row,previous if abs(new)<abs(old) else current,t['fee'])]
            for k,is_core,cost in parts:(fee_core if is_core else fee_weak)[k,j]+=cost
        np.testing.assert_allclose(fee_core+fee_weak,a['fees'],atol=1e-12)
        core_pnl=np.where(role,a['pnl'],0);weak_pnl=np.where(role,0,a['pnl'])
        role_attribution={'core_price_pnl':float(core_pnl.sum()),'core_fees':float(fee_core.sum()),'core_net_pnl':float((core_pnl-fee_core).sum()),
                          'weak_price_pnl':float(weak_pnl.sum()),'weak_fees':float(fee_weak.sum()),'weak_net_pnl':float((weak_pnl-fee_weak).sum()),
                          'weak_mean_gross':float(np.where(role,0,np.abs(a['weights'])).sum(1).mean()),
                          'meaning':'Actual hourly role PnL and fees; same-side upgrade reallocates existing quantity to core. Descriptive attribution, not isolated incremental alpha.'}
        assert abs(role_attribution['core_net_pnl']+role_attribution['weak_net_pnl']-outputs['probability']['return'])<1e-9
    pdaily=arrays['probability']['daily_returns'];ndays=np.unique(dates[frozen].astype('datetime64[D]'));gp=arrays['probability']['pnl']-arrays['probability']['fees'];dayp=np.array([gp[dates[frozen].astype('datetime64[D]')==d].sum() for d in ndays])
    concentration={'top5_day_pnl':float(np.sort(dayp)[-5:].sum()),'pnl_without_top5_days':float(gp.sum()-np.sort(dayp)[-5:].sum()),
                   'pnl_without_best_coin':float(gp.sum()-gp.sum(0).max())}
    results={'lock':lock,'accounts':outputs,'partitions':partitions,'coins':coins,'information_unchanged':info,'weak_entry_episodes':weak_result,'role_attribution':role_attribution,
             'concentration':concentration,'historical_intervals':{'probability':equity_interval(pdaily),'paired_vs_old':equity_interval(pdaily,arrays['old_hard_gate']['daily_returns'])},
             'source_sha256':source,'symbols':symbols,'earliest_fill':str(execution[frozen[0]]),'last_fill':str(execution[frozen[-1]+1]),
             'account_semantics':'Fresh zero-funding accounts on unchanged cached forecasts; NOT fixed-path funding subtraction.',
             'old_gate_design_roundtrip':oldgate,'mapping_trial_count_current_family':3,'ablation_count_current_family':1,
             'total_cross_family_development_trials':{'empirical':3,'implied':6,'core_overlay':9}[args.family]}
    _,strength=support_strength(mu[frozen],p_up[frozen],mapping)
    np.savez_compressed(OUT/'signal_mapping.npz',dates=dates,execution=execution,hours=frozen,raw_return4=raw,unchanged_return12=mu,risk12=risk,
                        probability_up=p_up,empirical_probability_up=empirical,selected_strength_frozen=strength,symbols=np.asarray(symbols))
    save(OUT/'monthly_probability_tables.json',tables);save(OUT/'results.json',results)
    assert all(digest(p)==v for p,v in source.items()),'saved source changed'
    save(OUT/'audit.json',{'unchanged_forecast_verified':True,'saved_source_files_unchanged':True,'causal_monthly_tables_verified':True,
                         'fresh_zero_funding_accounts':True,'fee_and_cash_reconciliation':True,'weak_episode_reconciliation':True,
                         'source_files':len(source),'cuda_neural_training_or_inference':False,
                         'code_sha256':{str(p):digest(p) for p in [Path(__file__),Path('src/crypto_timing/alpha_probability_policy.py'),Path('src/crypto_timing/alpha_account.py')]}})
    print('completed',OUT,flush=True)

if __name__=='__main__':main()
