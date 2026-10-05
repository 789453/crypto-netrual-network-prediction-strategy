import numpy as np
import pytest
from scipy.optimize import minimize
from crypto_timing.alpha_control_policy import (
    ControlSpec, ControlDecision, prox_two_costs, solve_control, guard, rank_target)
from crypto_timing.alpha_policy import Policy
from crypto_timing.alpha_account import account


def test_scalar_closed_form_and_complete_entry_exit_cost():
    A=np.array([[.01]]); old=np.array([.02]); a=np.array([.0008]); k=.03;c=.0002
    got,_,res=solve_control(A,a,old,k,c,0.,tol=1e-10)
    expected=old+np.sign(a-.01*old)*np.maximum(np.abs(a-.01*old)-c,0)/(.01+k)
    np.testing.assert_allclose(got,expected,atol=1e-8)
    cash,_,_=solve_control(A,np.array([.00035]),np.zeros(1),k,c,c)
    assert cash[0]==0 and res<1e-8


def test_joint_optimizer_matches_independent_convex_qp_oracle():
    rng=np.random.default_rng(18);M=rng.normal(size=(5,5))*.02
    A=M.T@M+np.eye(5)*.003;a=rng.normal(size=5)*.002;old=rng.normal(size=5)*.08
    k=.01;c=.0002;e=.0002
    w,_,res=solve_control(A,a,old,k,c,e,max_iter=600,tol=1e-10)
    # Epigraph QP: independent SLSQP validation, no nonsmooth numerical gradient.
    def fun(x):
        v=x[:5];return .5*v@A@v-a@v+.5*k*np.sum((v-old)**2)+c*x[5:10].sum()+e*x[10:].sum()
    def constraints(x):
        v=x[:5];u=x[5:10];z=x[10:]
        return np.r_[u-v+old,u+v-old,z-v,z+v]
    x0=np.r_[old,np.zeros(5),abs(old)]
    out=minimize(fun,x0,method='SLSQP',constraints={'type':'ineq','fun':constraints},
                 options={'ftol':1e-13,'maxiter':500})
    assert out.success and res<1e-8
    np.testing.assert_allclose(w,out.x[:5],atol=5e-6)


def test_prox_covers_negative_old_and_no_trade_interval():
    old=np.array([-.1,.1,0.]);v=old.copy()
    np.testing.assert_allclose(prox_two_costs(v,old,.02,0.,1.),old)
    np.testing.assert_allclose(prox_two_costs(-v,-old,.02,.01,1.),
                               -prox_two_costs(v,old,.02,.01,1.))


def test_guard_common_constraints_and_neutral_rank():
    pol=Policy(cap=2,coin_cap=.25,annual_risk=.3);cov=np.eye(12)*.001
    w=guard(np.linspace(-2,3,12),cov,pol,True)
    assert abs(w.sum())<1e-10 and abs(w).sum()<=2
    assert abs(w).max()<=.25
    assert np.sqrt(w@cov@w)*np.sqrt(24*365.25)<=.3+1e-10


def test_rank_ties_and_small_dispersion_do_not_become_full_budget():
    zero,_,g=rank_target(np.ones(12)*.001,np.zeros(12),.6,.0002)
    assert np.all(zero==0) and g==0
    z=np.linspace(-1,1,12)
    weak,_,_=rank_target(z*1e-6,z,.6,.0002)
    strong,_,_=rank_target(z*.001,z,.6,.0002)
    assert abs(weak).sum()<abs(strong).sum()/100


def test_repeated_prediction_rejected_and_future_prefix_is_invariant():
    rng=np.random.default_rng(14);mu=rng.normal(size=(30,12))*.001
    def replay(stream):
        dec=ControlDecision(np.arange(len(stream)),ControlSpec());current=np.zeros(12);out=[]
        for r,m in enumerate(stream):
            current,_,_=dec(r,m,np.eye(12)*1e-5,np.ones(12),current,np.ones(12)*r,Policy())
            out.append(current.copy())
        return dec,np.array(out)
    a,wa=replay(mu);other=mu.copy();other[20:]*=-100;b,wb=replay(other)
    np.testing.assert_array_equal(wa[:20],wb[:20])
    with pytest.raises(ValueError,match='duplicate'):
        a(29,mu[-1],np.eye(12)*1e-5,np.ones(12),wa[-1],np.ones(12),Policy())


def test_permutation_equivariance_of_control_and_safe_reduction_bypass():
    rng=np.random.default_rng(31);mu=rng.normal(size=12)*.003;M=rng.normal(size=(12,12))*.002
    cov=M@M.T+np.eye(12)*1e-5;perm=rng.permutation(12);pol=Policy()
    a=ControlDecision(np.arange(4));b=ControlDecision(np.arange(4))
    w,_,_=a(0,mu,cov,np.ones(12),np.zeros(12),np.zeros(12),pol)
    wp,_,_=b(0,mu[perm],cov[perm][:,perm],np.ones(12),np.zeros(12),np.zeros(12),pol)
    np.testing.assert_allclose(w[perm],wp,atol=1e-9)
    new,changed,why=a(1,mu,cov*100,np.ones(12),w,np.ones(12),pol)
    assert changed.any() and 'risk_limit' in why and abs(new).sum()<abs(w).sum()


def test_control_fixed_shares_ledger_and_no_mechanical_expiry():
    n=70;price=np.ones((n,12))*100;dates=np.datetime64('2025-01-01')+np.arange(n)*np.timedelta64(1,'h')
    hours=np.arange(25,65);mu=np.ones((len(hours),12))*.002;zero=np.zeros_like(price)
    dec=ControlDecision(hours,ControlSpec(family='A',inertia=0.))
    r,a,episodes,trades=account(price,mu,zero,zero,dates,hours,Policy(horizon=12),fee=.0002,retain=True,decision_adapter=dec)
    assert np.isclose(r['return'],(a['pnl']-a['fees']).sum())
    assert np.isclose(sum(t['notional'] for t in trades)*.0002,r['fee'])
    assert not any(t['reason']=='expiry' for t in trades)
    assert max(e['holding_hours'] for e in episodes)>12


def test_cash_signal_remains_cash():
    dec=ControlDecision(np.arange(10))
    for row in range(10):
        w,changed,_=dec(row,np.zeros(12),np.eye(12)*1e-5,np.ones(12),np.zeros(12),np.zeros(12),Policy())
        assert not changed.any() and np.all(w==0)


def test_post_fee_turnover_bucket_matches_actual_share_fills():
    rng=np.random.default_rng(7);price=100*np.cumprod(1+rng.normal(size=(80,12))*.001,axis=0)
    hours=np.arange(30,70);mu=np.tile(np.linspace(-.002,.004,12),(40,1))
    dates=np.datetime64('2025-01-01')+np.arange(80)*np.timedelta64(1,'h');zero=np.zeros_like(price)
    dec=ControlDecision(hours,ControlSpec(family='D'),ledger_fee=.0004)
    r,a,_,_=account(price,mu,zero,zero,dates,hours,Policy(),fee=.0004,decision_adapter=dec)
    np.testing.assert_allclose(np.asarray(dec.history)[:-1,0],a['turnover'][:-1],atol=2e-8)


def test_core_upgrade_starts_own_life_and_shared_increase_clock():
    from crypto_timing.alpha_control_policy import CoreParticipationDecision
    dec=CoreParticipationDecision(np.arange(25));pol=Policy(horizon=12,annual_risk=100.)
    cov=np.eye(12)*1e-5;weak=np.ones(12)*.0006;strong=np.ones(12)*.005
    current=np.zeros(12)
    for row in range(5):
        current,_,_=dec(row,weak,cov,np.ones(12),current,np.ones(12)*row,pol)
    assert abs(current).sum()>0 and not dec.core.any()
    for row in range(5,8):
        current,_,_=dec(row,strong,cov,np.ones(12),current,np.ones(12)*row,pol)
        assert not dec.core.any()
    current,_,why=dec(8,strong,cov,np.ones(12),current,np.ones(12)*8,pol)
    assert dec.core.any() and np.all(dec.core_age==0)
    for row in range(9,20):
        current,_,_=dec(row,strong,cov,np.ones(12),current,np.ones(12)*row,pol)
    assert dec.core.any() and np.all(dec.core_age==11)
    current,_,why=dec(20,strong,cov,np.ones(12),current,np.ones(12)*20,pol)
    assert not dec.core.any() and np.all(current==0) and 'core_expiry' in why


def test_tracking_target_respects_past_expected_funding_sign():
    hours=np.arange(4);mu=np.ones(12)*.0006;cov=np.eye(12)*1e-5;pol=Policy()
    positive=ControlDecision(hours,ControlSpec(family='D'),np.ones((4,12))*.00005)
    negative=ControlDecision(hours,ControlSpec(family='D'),np.ones((4,12))*(-.00005))
    wpos,_,_=positive(0,mu,cov,np.ones(12),np.zeros(12),np.zeros(12),pol)
    wneg,_,_=negative(0,mu,cov,np.ones(12),np.zeros(12),np.zeros(12),pol)
    assert wpos.sum()<wneg.sum()


def test_core_ablation_removes_weak_positions_without_changing_strong_gate():
    from crypto_timing.alpha_control_policy import CoreParticipationDecision
    dec=CoreParticipationDecision(np.arange(5),weak_enabled=False);pol=Policy(horizon=12)
    current=np.zeros(12);cov=np.eye(12)*1e-5
    for row in range(4):
        current,_,_=dec(row,np.ones(12)*.0006,cov,np.ones(12),current,np.zeros(12),pol)
        assert np.all(current==0)
    current,_,_=dec(4,np.ones(12)*.005,cov,np.ones(12),current,np.zeros(12),pol)
    assert dec.core.any() and abs(current).sum()>0
