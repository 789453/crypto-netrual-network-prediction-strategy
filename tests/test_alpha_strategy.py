import numpy as np
import torch
from crypto_timing.alpha_policy import Policy,project,decide,calibrate,apply_calibration
from crypto_timing.alpha_account import account
from crypto_timing.alpha_models import HorizonReadout
from crypto_timing.alpha_data import ROOT,BASE

def fixture():
    price=np.ones((100,12))*100;mu=np.ones((12,12))*.01;dates=np.datetime64('2025-01-01')+np.arange(100)*np.timedelta64(1,'h');hours=np.arange(50,62);fund=np.zeros_like(price)
    return price,mu,fund,fund.copy(),dates,hours,Policy(horizon=4,annual_risk=10)

def test_fee_is_per_side_and_ledger_reconciles():
    args=fixture();r,a,e,t=account(*args,retain=True)
    assert np.isclose(r['fee'],.0004*r['total_traded_notional_initial_equity_units'])
    assert np.isclose(r['return'],r['price_pnl']+r['funding']-r['fee'])
    assert r['fee']>0 and any(x['reason']=='expiry' for x in t)

def test_fixed_shares_no_hourly_cost_and_real_holding_life():
    r,a,e,t=account(*fixture(),retain=True)
    assert not any(x['hour'] in (51,52,53) for x in t)
    assert all(x['holding_hours']<=4 for x in e)

def test_account_causality_before_future_prices():
    args=list(fixture());r,a,*_=account(*args)
    args[0]=args[0].copy();args[0][64:]=10000
    r2,a2,*_=account(*args)
    np.testing.assert_allclose(a['nav'],a2['nav'])

def test_funding_is_held_quantity_times_mark_and_terminal_fee():
    args=list(fixture());args[2][51]=.001*100;r,a,*_=account(*args)
    qty=a['weights'][1]*a['nav'][1]/100
    assert np.isclose(a['funding'][1].sum(),-qty.sum()*.1)
    assert np.isclose(r['return'],(a['pnl']+a['funding']-a['fees']).sum())

def test_risk_budget_and_expiry_are_explicit():
    w=project(np.array([2.,2.,-1.,-1.]),2.,1.,.75)
    assert abs(w).sum()<=2+1e-8 and abs(w.sum())<=1+1e-8 and np.max(abs(w))<=.75
    desired,change,reason=decide(np.ones(12)*.1,np.zeros(12),np.eye(12)*1e-6,np.ones(12),np.ones(12)*.05,np.ones(12)*4,Policy(horizon=4,annual_risk=10))
    assert np.all(change) and np.all(desired==0) and np.all(reason=='expiry')

def test_joint_head_is_asset_permutation_equivariant():
    torch.manual_seed(19);m=HorizonReadout('joint').cuda().eval()
    with torch.no_grad():m.relative_head.weight.normal_(std=.1);m.market_head.weight.normal_(std=.1)
    x=torch.randn(2,4,12,128,device='cuda');v=torch.ones(2,4,12,dtype=torch.bool,device='cuda');risk=torch.rand(2,12,4,device='cuda')*.01+.01;perm=torch.randperm(12,device='cuda')
    with torch.no_grad():a=m(x,v,risk);b=m(x[:,:,perm],v[:,:,perm],risk[:,perm])
    torch.testing.assert_close(a[:,perm],b,atol=2e-7,rtol=2e-5)

def test_calibration_only_uses_given_history_and_preserves_shape():
    rng=np.random.default_rng(1);p=rng.normal(size=(300,12,4))*.003;y=p+rng.normal(size=p.shape)*.02;risk=np.ones_like(p)*.02
    fit=calibrate(p,y,risk);out=apply_calibration(p,risk,fit)
    assert out.shape==p.shape and np.isfinite(out).all()
    assert all(0<=r['common_slope']<=3 and 0<=r['relative_slope']<=3 for r in fit)

def test_actual_next_minute_horizons_and_funding_conservation():
    price=np.load(ROOT/'price.npy');target=np.load(ROOT/'target.npy');h=18000
    for j,H in enumerate((2,4,8,12)):np.testing.assert_allclose(target[h,:,j],price[h+H]/price[h]-1,atol=1e-8)
    np.testing.assert_allclose(np.load(ROOT/'funding_cash_minute.npy').sum(1),np.load(ROOT/'funding_cash_unit.npy')[:-1],rtol=2e-7,atol=1e-7)
    assert np.isfinite(price[-1]).all() and np.isnan(target[-1]).all()

def test_market_rebalance_has_finite_interval_and_terminal_trade_is_recorded():
    args=list(fixture());r,a,e,t=account(*args,mode='market',retain=True)
    assert not any(x['hour'] in (51,52,53) for x in t)
    assert any(x['reason']=='terminal' for x in t)
    assert np.isclose(sum(x['notional'] for x in t),r['total_traded_notional_initial_equity_units'])

def test_cash_does_not_become_an_unlabelled_funding_carry_strategy():
    args=list(fixture());args[1][:]=0;args[3][:]=-.01;r,a,e,t=account(*args,retain=True)
    assert r['return']==0 and r['average_gross']==0 and not t

def test_primary_selection_ignores_later_account_performance():
    from crypto_timing.alpha_system import select_primary
    development={name:{'chosen':{'utility':value},'frozen_return':later} for name,value,later in [('Frozen_v5',.3,-.9),('NN_direct',.2,10.),('NN_joint',.1,20.)]}
    assert select_primary(development)=='Frozen_v5'
    development['NN_joint']['frozen_return']=-100.;development['Frozen_v5']['frozen_return']=100.
    assert select_primary(development)=='Frozen_v5'

def test_risk_attribution_excludes_warmup_and_partitions_all_hours():
    import json
    from crypto_timing.alpha_analysis import attribute
    from crypto_timing.alpha_data import BASE
    results=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));name='Frozen_v5';m=results['models'][name];a=np.load(ROOT/f'account_{name}_4bp.npz');e=json.loads((ROOT/f'episodes_{name}.json').read_text(encoding='utf-8'))
    v=attribute(m['cost_scenarios']['4bp'],a,e,results['market']['symbols'],np.load(BASE/'decision_time.npy'));s=v['states']
    assert s['high_risk']['hours']+s['low_risk']['hours']==len(a['hours'])
    assert s['high_risk']['hours']>0 and s['low_risk']['hours']>0
    assert abs(s['high_risk']['net_pnl']+s['low_risk']['net_pnl']-m['cost_scenarios']['4bp']['return'])<1e-8
