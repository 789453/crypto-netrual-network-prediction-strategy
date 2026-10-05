"""Predeclared single E compatibility check after A-D; never reselect A-D."""
from pathlib import Path
import time,sys,logging,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_signal_control import OUT,ROOT,BASE,read,save,digest,extra_metrics
from run_probability_position import load_stream
from crypto_timing.alpha_control_policy import CoreParticipationDecision
from crypto_timing.alpha_policy import Policy
from crypto_timing.alpha_account import account
from crypto_timing.alpha_analysis import equity_interval

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'extension.log',encoding='utf-8')])
    log=logging.getLogger('core_participation')
    start=time.perf_counter();dates,raw,risk,mu,_,tables,used,frozen=load_stream()
    used += [BASE/'state_dynamic.npy',ROOT/'funding_cash_unit.npy',ROOT/'expected_funding_hour.npy',ROOT/'funding_cash_minute.npy']
    source={str(p):digest(p) for p in used};results=read(OUT/'results.json');assert source==results['source_sha256']
    if not (OUT/'results_before_E.json').exists():
        (OUT/'results_before_E.json').write_bytes((OUT/'results.json').read_bytes())
    pol=Policy(**read(ROOT/'system_lock.json')['policy']);price=np.load(ROOT/'price.npy');zeros=np.zeros_like(price)
    minute=np.load(ROOT/'minute_open.npy',mmap_mode='r');state=np.load(BASE/'state_dynamic.npy',mmap_mode='r')[:,:,6].mean(1)
    dev=np.flatnonzero((dates>=np.datetime64('2025-03-01'))&(dates<np.datetime64('2025-07-01')))
    spec=CoreParticipationDecision(frozen).specification()
    save(OUT/'extension_lock.json',{'spec':spec,'design_basis':'A-D reduce turnover but sacrifice original strong event return; one fixed 10% weak sleeve plus shared 4h ordinary clock.',
                                  'additional_parameter_search':False,'original_selected_unchanged':results['lock']['selected'],
                                  'development_trials_added':1,'historical_scores_previously_seen':True,
                                  'original_results_sha256':digest(OUT/'results_before_E.json'),'source_sha256':source})
    dec=CoreParticipationDecision(dev);r,_,_,_=account(price,mu[dev],zeros,zeros,dates,dev,pol,fee=.0002,decision_adapter=dec)
    save(OUT/'extension_development.json',{'spec':spec,'account':r,'utility':r['return']-.5*abs(r['max_drawdown'])})
    log.info('E development return=%.5f participation=%.5f',r['return'],r['participation'])
    def progress(row,data):log.info('E %s row=%d core_coins=%d weak_gross=%.4f',dates[frozen[row]],row,data[1],data[4])
    dec=CoreParticipationDecision(frozen,logger=progress);t0=time.perf_counter()
    r,a,e,t=account(price,mu[frozen],zeros,zeros,dates,frozen,pol,fee=.0002,minute=minute,retain=True,decision_adapter=dec)
    assert not r['bankrupt'] and r['hours']==len(frozen)
    gain=a['pnl']-a['fees'];identity=abs(float(gain.sum())-r['return']);episode_error=abs(sum(x['price_pnl']-x['fee'] for x in e)-r['return'])
    fee_error=abs(float(a['traded_notional'].sum())*.0002-r['fee']);assert max(identity,episode_error,fee_error)<1e-9
    a['core_role']=np.asarray(dec.role_history);a['control_state']=np.asarray(dec.history)
    a['desired_weights']=np.asarray(dec.targets);a['filtered_mu']=np.asarray(dec.filtered_history)
    r['control']={'spec':spec,'state_columns':['change_weight_l1','core_coin_count','normal_basket_active','risk_guard_active','weak_gross','max_core_age'],
                  'meaning':'Shared real net account; role attribution descriptive, not an independent sleeve account.'}
    r['diagnostics']=extra_metrics(a,e,t,state[frozen],results['risk_threshold'],mu[frozen]);r['seconds']=time.perf_counter()-t0
    np.savez_compressed(OUT/'account_E.npz',**a);save(OUT/'episodes_E.json',e);save(OUT/'trades_E.json',t)
    results['accounts']['E']=r;results['extension']={'spec':spec,'development':read(OUT/'extension_development.json'),'seconds':time.perf_counter()-start}
    results['historical_block_intervals']['E']=equity_interval(a['daily_returns'])
    results['timings']['E']=r['seconds'];results['seconds_total']+=results['extension']['seconds']
    assert source=={str(p):digest(p) for p in used}
    save(OUT/'results.json',results);audit=read(OUT/'audit.json')
    audit['accounts']['E']={'cash_identity_error':identity,'episodes_identity_error':episode_error,'fee_error':fee_error}
    audit['tested_accounts']=len(audit['accounts'])
    audit['code_sha256']['scripts/run_signal_control_extension.py']=digest(Path(__file__))
    audit['code_sha256']['src/crypto_timing/alpha_control_policy.py']=digest('src/crypto_timing/alpha_control_policy.py')
    save(OUT/'audit.json',audit);log.info('DONE E return=%.5f minute_drawdown=%.5f daily_turnover=%.5f',r['return'],r['minute_max_drawdown'],r['diagnostics']['turnover_per_day'])

if __name__=='__main__':main()
