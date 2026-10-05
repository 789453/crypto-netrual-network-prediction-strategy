"""Remove E weak participation without optimising any new parameter."""
import sys,time,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_signal_control import ROOT,BASE,OUT,read,save,digest,extra_metrics
from run_probability_position import load_stream
from crypto_timing.alpha_control_policy import CoreParticipationDecision
from crypto_timing.alpha_policy import Policy
from crypto_timing.alpha_account import account

def main():
    start=time.perf_counter();dates,_,_,mu,_,_,_,hours=load_stream();result=read(OUT/'results.json')
    pol=Policy(**result['lock']['policy']);price=np.load(ROOT/'price.npy');zeros=np.zeros_like(price)
    minute=np.load(ROOT/'minute_open.npy',mmap_mode='r');dec=CoreParticipationDecision(hours,weak_enabled=False)
    r,a,e,t=account(price,mu[hours],zeros,zeros,dates,hours,pol,fee=.0002,minute=minute,retain=True,decision_adapter=dec)
    a['core_role']=np.asarray(dec.role_history);a['control_state']=np.asarray(dec.history)
    a['desired_weights']=np.asarray(dec.targets);a['filtered_mu']=np.asarray(dec.filtered_history)
    state=np.load(BASE/'state_dynamic.npy',mmap_mode='r')[:,:,6].mean(1)
    r['diagnostics']=extra_metrics(a,e,t,state[hours],result['risk_threshold'],mu[hours]);r['control']={'spec':dec.specification()}
    r['seconds']=time.perf_counter()-start;assert not r['bankrupt']
    identity=abs(float((a['pnl']-a['fees']).sum())-r['return']);fee_error=abs(float(a['traded_notional'].sum())*.0002-r['fee'])
    episode_error=abs(sum(x['price_pnl']-x['fee'] for x in e)-r['return']);assert max(identity,fee_error,episode_error)<1e-9
    np.savez_compressed(OUT/'account_E_core_only.npz',**a);save(OUT/'episodes_E_core_only.json',e);save(OUT/'trades_E_core_only.json',t)
    result['accounts']['E_core_only']=r;result['seconds_total']+=r['seconds'];result['timings']['E_core_only']=r['seconds'];save(OUT/'results.json',result)
    audit=read(OUT/'audit.json');audit['accounts']['E_core_only']={'cash_identity_error':identity,'fee_error':fee_error,'episodes_identity_error':episode_error}
    audit['tested_accounts']=len(audit['accounts']);audit['code_sha256']['scripts/run_signal_core_ablation.py']=digest(Path(__file__))
    audit['code_sha256']['src/crypto_timing/alpha_control_policy.py']=digest('src/crypto_timing/alpha_control_policy.py');save(OUT/'audit.json',audit)
    print('E core-only ablation',r['return'],r['minute_max_drawdown'],r['participation'],flush=True)

if __name__=='__main__':main()
