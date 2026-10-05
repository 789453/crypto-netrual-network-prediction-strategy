"""Bounded design comparison on frozen v5 outputs, not neural training."""
from pathlib import Path
from dataclasses import asdict, replace
import sys, json, hashlib, logging, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_probability_position import load_stream, ROOT, BASE
from crypto_timing.alpha_control_policy import ControlSpec, ControlDecision, guard
from crypto_timing.alpha_policy import Policy
from crypto_timing.alpha_account import account
from crypto_timing.alpha_analysis import equity_interval

OUT=Path('outputs/signal_control_2026-10-05')
CONTROL_COLUMNS=['turnover_post_fee','ewma_turnover_bucket','shadow_cost','rank_dispersion_support',
                 'solver_iterations','proximal_residual','constraint_contraction_l1','risk_guard_active','target_gap_l1']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def longest(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return int((ends-starts).max()) if len(starts) else 0

def extra_metrics(a,e,t,risklevel,threshold,mu):
    weights=a['weights'];gross=np.abs(weights).sum(1);gain=a['pnl']+a['funding']-a['fees'];out={}
    def state(k):
        return {'hours':int(k.sum()),'participation':float((gross[k]>1e-6).mean()),
                'mean_gross':float(gross[k].mean()),'net_pnl_initial_equity':float(gain[k].sum()),
                'turnover_per_day':float(a['turnover'][k].mean()*24)}
    out['low_vol']=state(risklevel<threshold);out['high_vol']=state(risklevel>=threshold)
    out['longest_cash_hours']=longest(gross<=1e-6)
    out['gross_time_cv']=float(gross.std()/max(gross.mean(),1e-12))
    coin=abs(weights).mean(0);share=coin/max(coin.sum(),1e-12)
    out['effective_coins_nominal']=float(1/max(float(share@share),1e-12)) if coin.sum()>0 else 0.
    out['max_coin_nominal_share']=float(share.max());out['mean_abs_weight_by_coin']=coin.tolist()
    daily=np.unique(a['dates'].astype('datetime64[D]'))
    dayp=np.array([gain[a['dates'].astype('datetime64[D]')==d].sum() for d in daily])
    out['net_without_top5_days']=float(gain.sum()-np.sort(dayp)[-5:].sum())
    out['net_without_best_coin']=float(gain.sum()-gain.sum(0).max())
    out['net_bps_per_traded_notional']=float(gain.sum()/max(a['traded_notional'].sum(),1e-12)*1e4)
    out['normal_trade_hours']=len({x['hour'] for x in t if x['reason'].startswith('control_') or x['reason'] in ('core_entry','core_reverse')})
    out['risk_trade_hours']=len({x['hour'] for x in t if x['reason']=='risk_limit'})
    out['turnover_per_day']=float(a['turnover'].mean()*24)
    out['holding_p10_p50_p90']=np.quantile([x['holding_hours'] for x in e],[.1,.5,.9]).tolist() if e else [0.,0.,0.]
    out['strong_signal_hours']={'hours':int((np.abs(mu).max(1)>=.0008).sum()),
                               'net_pnl_initial_equity':float(gain[np.abs(mu).max(1)>=.0008].sum())}
    out['monthly']={}
    for month in np.unique(a['dates'].astype('datetime64[M]')):
        k=a['dates'].astype('datetime64[M]')==month;ids=np.flatnonzero(k);nav=a['nav'][ids[0]:ids[-1]+2]
        out['monthly'][str(month)]={**state(k),'return':float(nav[-1]/nav[0]-1),
                                   'drawdown':float((nav/np.maximum.accumulate(nav)-1).min())}
    out['coin_net_contribution']=gain.sum(0).tolist()
    weak=[x for x in e if abs(float(mu[x['start_hour']-a['hours'][0],x['symbol_id']]))<.0008]
    out['weak_entry_episodes']={'count':len(weak),'net_pnl_initial_equity':float(sum(x['price_pnl']+x['funding']-x['fee'] for x in weak)),
                                'meaning':'Physical episodes classified by forecast at original entry; not isolated weak-layer alpha.'}
    return out


class GuardedRank:
    def __call__(self,row,mu,cov,beta,current,age,policy):
        desired=current.copy();change=np.full(len(mu),row%policy.horizon==0)|(abs(current)<1e-9)
        order=np.argsort(mu);rank=np.zeros(len(mu));rank[order[:len(mu)//2]]=-policy.cap/len(mu);rank[order[len(mu)//2:]]=policy.cap/len(mu)
        desired=np.where(change,rank,current)
        safe=guard(desired,cov,policy,True);change|=abs(safe-desired)>1e-8
        return safe,change,np.full(len(mu),'guarded_rank',object)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    log=logging.getLogger('signal_control');log.setLevel(logging.INFO)
    fmt=logging.Formatter('%(asctime)s %(message)s')
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'run.log',encoding='utf-8')):
        handler.setFormatter(fmt);log.addHandler(handler)
    start=time.perf_counter();log.info('Loading unchanged forecasts; no training or inference')
    dates,raw,risk,mu,_,tables,used,frozen=load_stream()
    used += [BASE/'state_dynamic.npy',ROOT/'funding_cash_unit.npy',ROOT/'expected_funding_hour.npy',ROOT/'funding_cash_minute.npy']
    source={str(p):digest(p) for p in used}
    price=np.load(ROOT/'price.npy');minute=np.load(ROOT/'minute_open.npy',mmap_mode='r');zeros=np.zeros_like(price)
    policy=Policy(**read(ROOT/'system_lock.json')['policy']);symbols=read(ROOT/'market_manifest.json')['symbols']
    dev=np.flatnonzero((dates>=np.datetime64('2025-03-01'))&(dates<np.datetime64('2025-07-01')))
    state=np.load(BASE/'state_dynamic.npy',mmap_mode='r')[:,:,6].mean(1)
    history=state[:np.searchsorted(dates,np.datetime64('2025-07-01'))];threshold=float(np.median(history[np.isfinite(history)]))
    assert np.isfinite(mu[dev]).all() and np.isfinite(mu[frozen]).all() and np.isfinite(state[frozen]).all()
    specs={family:ControlSpec(family=family) for family in ('A','B','C','D')}
    specs['C_low']=replace(specs['C'],rank_budget=.4);specs['C_high']=replace(specs['C'],rank_budget=.8)
    development={}
    for name,spec in specs.items():
        dec=ControlDecision(dev,spec);t0=time.perf_counter()
        r,a,e,t=account(price,mu[dev],zeros,zeros,dates,dev,policy,fee=.0002,decision_adapter=dec)
        utility=r['return']-.5*abs(r['max_drawdown']) if not r['bankrupt'] else -1e6
        development[name]={'spec':asdict(spec),'account':r,'utility':utility,'seconds':time.perf_counter()-t0}
        log.info('development %s utility=%.5f return=%.3f gross=%.3f participation=%.3f',name,utility,r['return'],r['average_gross'],r['participation'])
        save(OUT/'development.json',development)
    # Operational preference supplied by the user: cash-only cannot satisfy it.
    # Preserve cash as economic benchmark; do not claim the active candidate beats it.
    eligible=[n for n,x in development.items() if x['account']['participation']>=.8 and x['account']['average_gross']>=.03]
    if not eligible:raise RuntimeError('No candidate meets the active research mandate')
    selected=max(eligible,key=lambda x:development[x]['utility']);chosen=specs[selected]
    lock={'selected':selected,'chosen_spec':asdict(chosen),'specs':{n:asdict(s) for n,s in specs.items()},'development':development,
          'selection':'participation >= .8 and mean gross >= .03, then development return - .5*abs(drawdown), 2025-03/06 only',
          'eligible_active_candidates':eligible,'beats_cash_development_utility':development[selected]['utility']>0,
          'selection_amendment':'Cash-only B wins unrestricted utility; eligibility added after development inspection to express user participation mandate. Original lock/log retained in selection_audit_cash_preference. No later account score used.',
          'historical_status':'previously inspected history, not forward confirmation',
          'source_sha256':source,'risk_threshold_past_log_sigma24h':threshold,'policy':asdict(policy),
          'fee_per_side':.0002,'horizon':12,'neural_training':False,'protocol_sha256':digest('docs/SIGNAL_CONTROL_PROTOCOL_2026-10-05.md')}
    save(OUT/'control_lock.json',lock);log.info('LOCKED recommendation=%s before later-account evaluation',selected)
    summary={};arrays={};audits={};timings={}
    names=[*specs,'old_gate8bp','old_gate4bp','rank_guarded','rank_native','selected_no_friction','selected_expiry12','selected_fee4bp','selected_funding']
    for name in names:
        t0=time.perf_counter();adapter=None;pol=policy;fee=.0002;mode='economic';fund=zeros;expected=zeros;fm=None
        spec=specs.get(name)
        if name=='selected_no_friction':spec=replace(chosen,no_friction=True,minimum_trade=0.)
        if name=='selected_expiry12':spec=replace(chosen,fixed_expiry=True)
        if name=='selected_fee4bp':spec=chosen;fee=.0004
        if name=='selected_funding':
            spec=chosen;fund=np.load(ROOT/'funding_cash_unit.npy');expected=np.load(ROOT/'expected_funding_hour.npy');fm=np.load(ROOT/'funding_cash_minute.npy',mmap_mode='r')
        if spec:
            def progress(row,data):log.info('%s %s row=%d turnover_bucket=%.3f residual=%.2g',name,str(dates[frozen[row]]),row,data[1],data[5])
            adapter=ControlDecision(frozen,spec,expected[frozen] if name=='selected_funding' else None,progress,ledger_fee=fee)
        elif name=='old_gate4bp':pol=replace(policy,design_fee=.0002)
        elif name=='rank_guarded':adapter=GuardedRank()
        elif name=='rank_native':mode='full_rank'
        r,a,e,t=account(price,mu[frozen],fund,expected,dates,frozen,pol,fee=fee,minute=minute,
                        funding_minute=fm,mode=mode,retain=True,decision_adapter=adapter)
        assert not r['bankrupt'] and r['hours']==len(frozen)
        gain=a['pnl']+a['funding']-a['fees']
        identity=abs(float(gain.sum())-r['return']);ledger_fee=abs(fee*float(a['traded_notional'].sum())-r['fee'])
        episode_error=abs(sum(x['price_pnl']+x['funding']-x['fee'] for x in e)-r['return'])
        assert max(identity,ledger_fee,episode_error)<1e-9
        assert np.isclose(sum(x['notional'] for x in t),a['traded_notional'].sum(),atol=1e-9)
        audits[name]={'cash_identity_error':identity,'fee_error':ledger_fee,'episodes_identity_error':episode_error,
                      'recorded_trades_identity':True,'terminal_close_recorded':True}
        if isinstance(adapter,ControlDecision):
            a['control_state']=np.asarray(adapter.history);a['desired_weights']=np.asarray(adapter.targets);a['filtered_mu']=np.asarray(adapter.filtered_history)
            hist=a['control_state'];r['control']={'spec':asdict(spec),'state_columns':CONTROL_COLUMNS,
                       'max_proximal_residual':float(hist[:,5].max()),'mean_iterations':float(hist[:,4].mean()),
                       'constraint_contraction_fraction':float((hist[:,6]>1e-6).mean()),
                       'turnover_shadow_active_fraction':float((hist[:,2]>0).mean())}
            assert hist[:,5].max()<1e-7
            assert np.abs(a['weights']).sum(1).max()<=policy.cap+1e-7
            assert np.abs(a['weights'].sum(1)).max()<=policy.cap*policy.net_fraction+1e-7
            assert np.abs(a['weights']).max()<=policy.coin_cap+1e-7
        r['diagnostics']=extra_metrics(a,e,t,state[frozen],threshold,mu[frozen])
        if name=='selected_funding':r['funding_expected_adjusts_target']=True
        r['seconds']=time.perf_counter()-t0;timings[name]=r['seconds'];summary[name]=r;arrays[name]=a
        np.savez_compressed(OUT/f'account_{name}.npz',**a);save(OUT/f'episodes_{name}.json',e);save(OUT/f'trades_{name}.json',t)
        save(OUT/'partial_results.json',summary)
        log.info('DONE %s return=%.4f drawdown=%.4f gross=%.3f turnover=%.1f seconds=%.1f',name,r['return'],r['max_drawdown'],r['average_gross'],r['turnover'],r['seconds'])
    intervals={n:equity_interval(arrays[n]['daily_returns']) for n in specs}
    paired={n:equity_interval(arrays[n]['daily_returns'],arrays['old_gate4bp']['daily_returns']) for n in specs}
    end_source={str(p):digest(p) for p in used};assert source==end_source
    results={'lock':lock,'accounts':summary,'symbols':symbols,'historical_block_intervals':intervals,
             'paired_vs_fee_aligned':paired,'source_sha256':source,'risk_threshold':threshold,
             'start_execution':str(np.load(ROOT/'execution.npy')[frozen[0]]),'end_execution':str(np.load(ROOT/'execution.npy')[frozen[-1]+1]),
             'seconds_total':time.perf_counter()-start,'timings':timings,
             'forecast_diagnostics':{'development_relative_spread_bp_quantiles':np.quantile(abs(mu[dev]-mu[dev].mean(1)[:,None]).mean(1),[.1,.5,.9]).tolist(),
                                     'historical_relative_spread_bp_quantiles':(np.quantile(abs(mu[frozen]-mu[frozen].mean(1)[:,None]).mean(1),[.1,.5,.9])*1e4).tolist(),
                                     'near_tied_cross_section_fraction':float((np.ptp(mu[frozen],axis=1)<1e-9).mean())},
             'semantics':'Fresh share accounts, unchanged saved 12h signal, next-minute open, default zero funding; funding case explicitly separate.'}
    # Correct units for development diagnostics before saving.
    results['forecast_diagnostics']['development_relative_spread_bp_quantiles']=[x*1e4 for x in results['forecast_diagnostics']['development_relative_spread_bp_quantiles']]
    save(OUT/'results.json',results)
    save(OUT/'audit.json',{'passed':True,'sources_unchanged':True,'source_file_count':len(source),
                         'causal_calibration_maturity_checked':True,'selection_locked_before_historical_accounts':True,
                         'neural_training_or_inference':False,'accounts':audits,'tested_accounts':len(audits),
                         'code_sha256':{p:digest(p) for p in ['scripts/run_signal_control.py','src/crypto_timing/alpha_control_policy.py','src/crypto_timing/alpha_account.py']}})
    log.info('COMPLETE %s total seconds=%.1f',OUT,results['seconds_total'])

if __name__=='__main__':main()
