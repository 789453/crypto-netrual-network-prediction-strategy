"""Descriptive realised response and minute risk audit; never retune controls."""
from pathlib import Path
import numpy as np
from run_signal_control import ROOT,OUT,BASE,read,save,digest
from run_probability_position import load_stream

def refresh_funding(result,dates,mu,frozen,price,minute):
    """Only rerun the changed funding case, keep all unaffected locked accounts."""
    if result['accounts']['selected_funding'].get('funding_expected_adjusts_target',False):return
    from crypto_timing.alpha_control_policy import ControlSpec,ControlDecision
    from crypto_timing.alpha_policy import Policy
    from crypto_timing.alpha_account import account
    from run_signal_control import extra_metrics
    import time
    t0=time.perf_counter();pol=Policy(**result['lock']['policy']);spec=ControlSpec(**result['lock']['chosen_spec'])
    expected=np.load(ROOT/'expected_funding_hour.npy');fund=np.load(ROOT/'funding_cash_unit.npy');fm=np.load(ROOT/'funding_cash_minute.npy',mmap_mode='r')
    dec=ControlDecision(frozen,spec,expected[frozen]);r,a,e,t=account(price,mu[frozen],fund,expected,dates,frozen,pol,
          fee=.0002,minute=minute,funding_minute=fm,retain=True,decision_adapter=dec)
    gain=a['pnl']+a['funding']-a['fees'];assert np.isclose(gain.sum(),r['return'],atol=1e-10)
    a['control_state']=np.asarray(dec.history);a['desired_weights']=np.asarray(dec.targets);a['filtered_mu']=np.asarray(dec.filtered_history)
    hist=a['control_state'];r['control']={**result['accounts']['selected_funding']['control'],
      'max_proximal_residual':float(hist[:,5].max()),'mean_iterations':float(hist[:,4].mean()),
      'constraint_contraction_fraction':float((hist[:,6]>1e-6).mean()),'turnover_shadow_active_fraction':float((hist[:,2]>0).mean())}
    state=np.load(BASE/'state_dynamic.npy',mmap_mode='r')[:,:,6].mean(1)
    r['diagnostics']=extra_metrics(a,e,t,state[frozen],result['risk_threshold'],mu[frozen]);r['seconds']=time.perf_counter()-t0
    r['funding_expected_adjusts_target']=True
    np.savez_compressed(OUT/'account_selected_funding.npz',**a);save(OUT/'episodes_selected_funding.json',e);save(OUT/'trades_selected_funding.json',t)
    result['accounts']['selected_funding']=r;result['seconds_total']+=r['seconds'];result['timings']['selected_funding']=r['seconds']
    print('Refreshed funding-aware D target',r['return'],flush=True)

def main():
    dates,_,_,mu,_,_,_,frozen=load_stream();target=np.load(ROOT/'target.npy');price=np.load(ROOT/'price.npy')
    result=read(OUT/'results.json');minute=np.load(ROOT/'minute_open.npy',mmap_mode='r');evidence={}
    refresh_funding(result,dates,mu,frozen,price,minute)
    dev=np.flatnonzero((dates>=np.datetime64('2025-03-01'))&(dates+np.timedelta64(721,'m')<np.datetime64('2025-07-01')))
    for period,hours in [('development_matured',dev),('inspected_history',frozen)]:
        valid=np.isfinite(target[hours]).all((1,2));hours=hours[valid];m=mu[hours];y=target[hours]
        record=[]
        for label,low,high in [('below4bp',0.,.0004),('4to8bp',.0004,.0008),('above8bp',.0008,float('inf'))]:
            mask=(abs(m)>=low)&(abs(m)<high)&(m!=0)
            signed=np.sign(m)[...,None]*y
            record.append({'bucket':label,'coin_hours_correlated':int(mask.sum()),'distinct_clocks':int(mask.any(1).sum()),
                           'mean_abs_prediction12_bp':float(abs(m)[mask].mean()*1e4),
                           'mean_signed_realised_bp_by_2_4_8_12h':(signed[mask].mean(0)*1e4).tolist(),
                           'net12_bp_if_forced_roundtrip4bp':float(signed[mask,3].mean()*1e4-4),
                           'meaning':'Descriptive conditional mean, overlapping horizons and correlated coins; forced one-roundtrip thought experiment, not realised strategy attribution.'})
        evidence[period]=record
    fits=read(ROOT/'calibration.json')
    evidence['relative_slopes_by_month']={key.split('/')[0]:r['fit'][3]['relative_slope'] for key,r in fits.items() if key.endswith('/Frozen_v5')}
    minute_audit={}
    for name in ['D','E','E_core_only','old_gate8bp','old_gate4bp','rank_guarded','rank_native']:
        a=np.load(OUT/f'account_{name}.npz');fee=result['accounts'][name]['fee_per_side'];h=a['hours'];w=a['weights'];totalfees=a['fees'].sum(1)
        post=a['nav'][:-1]-totalfees
        terminal_ratio=np.sum(abs(w[-1])*price[h[-1]+1]/price[h[-1]])
        post[-1]=(a['nav'][-2]-totalfees[-1])/(1-fee*terminal_ratio)
        qty=w*post[:,None]/price[h]
        np.testing.assert_allclose(qty*(price[h+1]-price[h]),a['pnl'],atol=1e-9,rtol=1e-8)
        peak=0.;soft=hard=0
        for row,hour in enumerate(h):
            path=np.concatenate((np.asarray(minute[hour],float),price[hour+1][None]),0)
            equity=post[row]+(path-price[hour])@qty[row]
            leverage=np.abs(path*qty[row]).sum(1)/equity
            maximum=float(leverage.max());peak=max(peak,maximum);soft+=maximum>2.1;hard+=maximum>3.
        assert soft==result['accounts'][name]['minute_gross_limit_breach_hours']
        minute_audit[name]={'maximum_minute_gross':peak,'hours_above_normal_cap_plus5pct':int(soft),'hours_above_hard_cap3':int(hard)}
    evidence['minute_risk_audit']=minute_audit;save(OUT/'signal_evidence.json',evidence)
    # Role attribution on the single real E account, including upgrades/reversals.
    ea=np.load(OUT/'account_E.npz');role=ea['core_role'];efee=np.zeros_like(ea['fees']);wfee=efee.copy()
    for trade in read(OUT/'trades_E.json'):
        row=int(np.searchsorted(ea['hours'],trade['hour']));j=trade['symbol_id'];old=trade['quantity_before'];new=trade['quantity_after']
        if row==len(role):parts=[(row-1,bool(role[-1,j]),trade['fee'])]
        else:
            previous=bool(role[row-1,j]) if row else False;now=bool(role[row,j])
            if old*new<0:parts=[(row,previous,.0002*abs(old)*trade['price']),(row,now,.0002*abs(new)*trade['price'])]
            else:parts=[(row,previous if abs(new)<abs(old) else now,trade['fee'])]
        for idx,is_core,cost in parts:(efee if is_core else wfee)[idx,j]+=cost
    # The legacy ledger omits trade log rows with |quantity change| <= 1e-12,
    # while still accounting their floating-point fee. Attribute that tiny residue.
    residue=ea['fees']-efee-wfee
    assert np.max(np.abs(residue))<1e-9
    efee+=np.where(role,residue,0.);wfee+=np.where(role,0.,residue)
    np.testing.assert_allclose(efee+wfee,ea['fees'],atol=1e-12)
    core_pnl=np.where(role,ea['pnl'],0.);weak_pnl=np.where(role,0.,ea['pnl'])
    attribution={'core_price_pnl':float(core_pnl.sum()),'core_fee':float(efee.sum()),'core_net':float((core_pnl-efee).sum()),
                 'weak_price_pnl':float(weak_pnl.sum()),'weak_fee':float(wfee.sum()),'weak_net':float((weak_pnl-wfee).sum()),
                 'weak_mean_gross':float(np.where(role,0.,abs(ea['weights'])).sum(1).mean()),
                 'fee_log_rounding_residue':float(residue.sum()),
                 'meaning':'Descriptive actual-role attribution; weak -> core upgrades transfer existing quantity without synthetic fees. Incremental weak effect also changes shared entry clock, evaluated separately by E_core_only.'}
    assert abs(attribution['core_net']+attribution['weak_net']-result['accounts']['E']['return'])<1e-9
    normal=np.flatnonzero(ea['control_state'][:,2]>0)
    assert len(normal)<2 or np.diff(normal).min()>=4
    attribution['minimum_normal_basket_spacing_hours']=int(np.diff(normal).min()) if len(normal)>1 else None
    result['E_role_attribution']=attribution
    for name,r in result['accounts'].items():
        weights=np.load(OUT/f'account_{name}.npz')['weights'];gross=abs(weights).sum(1)
        r['diagnostics']['participation_gross_at_least_1pct']=float((gross>=.01).mean())
        r['diagnostics']['participation_gross_at_least_3pct']=float((gross>=.03).mean())
        r['diagnostics']['participation_threshold_note']='Reported physical participation uses gross>1e-6; 1%/3% NAV gross are additional descriptive meaningful-scale thresholds, not retuned entry gates or actual exchange minimum orders.'
    result['signal_evidence']=evidence;save(OUT/'results.json',result)
    audit=read(OUT/'audit.json');audit['minute_hard_cap_audit']=minute_audit
    audit['code_sha256']['scripts/analyze_signal_control.py']=digest(Path(__file__))
    audit['code_sha256']['src/crypto_timing/alpha_control_policy.py']=digest('src/crypto_timing/alpha_control_policy.py')
    audit['code_sha256']['scripts/run_signal_control.py']=digest('scripts/run_signal_control.py')
    funded=np.load(OUT/'account_selected_funding.npz');fund_e=read(OUT/'episodes_selected_funding.json');fr=result['accounts']['selected_funding']
    audit['accounts']['selected_funding']={'cash_identity_error':abs(float((funded['pnl']+funded['funding']-funded['fees']).sum())-fr['return']),
        'fee_error':abs(float(funded['traded_notional'].sum())*.0002-fr['fee']),
        'episodes_identity_error':abs(sum(x['price_pnl']+x['funding']-x['fee'] for x in fund_e)-fr['return'])}
    assert max(audit['accounts']['selected_funding'].values())<1e-9
    audit['E_role_pnl_reconciles']=True;audit['E_normal_basket_spacing_verified']=True
    import xml.etree.ElementTree as ET
    suites=ET.parse(OUT/'tests.xml').getroot().findall('.//testsuite')
    assert sum(int(x.get('failures','0'))+int(x.get('errors','0')) for x in suites)==0
    audit['pytest_passed_tests']=sum(int(x.get('tests','0')) for x in suites)
    save(OUT/'audit.json',audit)
    print('Response/maturity and minute risk audit complete',minute_audit,flush=True)

if __name__=='__main__':main()
