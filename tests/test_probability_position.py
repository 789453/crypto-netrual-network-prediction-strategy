from dataclasses import replace
import numpy as np
import pytest
from crypto_timing.alpha_policy import Policy,decide
from crypto_timing.alpha_account import account
from crypto_timing.alpha_probability_policy import (
    monotone_probability,probability_from_saved,ProbabilityMapping,
    target_probability_weights,ProbabilityDecision,support_strength)


def record():
    return {'bin_count':[1440]*10,'bin_up_fraction':[.4,.45,.44,.48,.5,.49,.53,.51,.56,.6],
            'quantile_edges':list(np.linspace(-.1,.1,9))}


def test_saved_frequency_shrink_and_monotone_pooling():
    table=monotone_probability(record());p=np.array(table['probability_up'])
    assert np.all(np.diff(p)>=0) and np.all((p>=0)&(p<=1))
    assert p[0]>.4 and p[-1]<.6
    empty=record();empty['bin_count']=[0]*10
    np.testing.assert_equal(monotone_probability(empty)['probability_up'],[.5]*10)


def test_probability_lookup_uses_only_saved_table_and_past_scale():
    t=monotone_probability(record());mu=np.array([-.02,0,.02]);risk=np.ones(3)*.1
    p=probability_from_saved(mu,risk,t)
    assert np.all(np.diff(p)>=0)
    with pytest.raises(ValueError):probability_from_saved(mu,np.zeros(3),t)


def test_sub_cost_prediction_opens_small_position_without_normalizing_to_full():
    mu=np.full(12,.0002);cov=np.eye(12)*1e-6;beta=np.ones(12)
    policy=Policy(horizon=12,annual_risk=10)
    old,*_=decide(mu,np.zeros(12),cov,beta,np.zeros(12),np.zeros(12),policy)
    assert np.all(old==0)
    small=target_probability_weights(mu,np.full(12,.51),cov,beta,policy,ProbabilityMapping())
    large=target_probability_weights(mu,np.full(12,.55),cov,beta,policy,ProbabilityMapping())
    assert 0<np.abs(small).sum()<np.abs(large).sum()<policy.cap
    _,s=support_strength(mu,np.full(12,.55),ProbabilityMapping())
    np.testing.assert_allclose(s,1/3)


def test_no_support_or_no_neural_direction_does_not_force_trade():
    for mu,p in [(np.zeros(12),np.full(12,.6)),(np.ones(12)*.002,np.full(12,.5)),(-np.ones(12)*.002,np.full(12,.6))]:
        w=target_probability_weights(mu,p,np.eye(12)*1e-6,np.ones(12),Policy(),ProbabilityMapping())
        np.testing.assert_equal(w,np.zeros(12))


def fixture():
    price=np.ones((100,12))*100;mu=np.ones((36,12))*.002;zeros=np.zeros_like(price)
    dates=np.datetime64('2025-01-01')+np.arange(100)*np.timedelta64(1,'h')
    return price,mu,zeros,zeros.copy(),dates,np.arange(50,86),Policy(horizon=12,annual_risk=10)


def test_renewal_keeps_shares_and_episode_beyond_twelve_hours():
    args=fixture();prob=np.full((36,12),.55)
    r,a,e,t=account(*args,fee=.0002,retain=True,decision_adapter=ProbabilityDecision(prob,ProbabilityMapping()))
    assert all(x['holding_hours']==36 for x in e)
    assert set(x['reason'] for x in t)=={'prob_entry','terminal'}
    assert np.isclose(r['return'],(a['pnl']-a['fees']).sum())
    fixed=ProbabilityDecision(prob,replace(ProbabilityMapping(),fixed_expiry=True))
    _,_,ef,_=account(*args,fee=.0002,decision_adapter=fixed)
    assert max(x['holding_hours'] for x in ef)<=12


def test_confirmation_prevents_single_hour_probability_noise_exit():
    args=fixture();prob=np.full((36,12),.55);prob[8]=.5
    _,_,_,t=account(*args,fee=.0002,retain=True,decision_adapter=ProbabilityDecision(prob,ProbabilityMapping()))
    assert not any(x['hour']==58 for x in t)
    prob[8:11]=.5
    _,_,_,t=account(*args,fee=.0002,retain=True,decision_adapter=ProbabilityDecision(prob,ProbabilityMapping()))
    assert any(x['hour']==60 and x['reason']=='support_lost' for x in t)


def test_adapter_interface_preserves_original_cash_and_positions():
    args=fixture()
    def original(row,mu,cov,beta,current,age,policy):
        return decide(mu,np.zeros(12),cov,beta,current,age,policy)
    r,a,_,_=account(*args,fee=.0002)
    rr,aa,_,_=account(*args,fee=.0002,decision_adapter=original)
    np.testing.assert_equal(a['nav'],aa['nav']);np.testing.assert_equal(a['weights'],aa['weights'])
    assert r==rr


def test_later_price_and_probability_do_not_change_earlier_positions():
    args=list(fixture());p=np.full((36,12),.55)
    _,a,_,_=account(*args,fee=.0002,decision_adapter=ProbabilityDecision(p,ProbabilityMapping()))
    args[0]=args[0].copy();args[0][70:]*=2;p[20:]=.1
    _,aa,_,_=account(*args,fee=.0002,decision_adapter=ProbabilityDecision(p,ProbabilityMapping()))
    np.testing.assert_equal(a['weights'][:19],aa['weights'][:19])
    np.testing.assert_equal(a['nav'][:20],aa['nav'][:20])


def test_zero_overlay_exactly_recovers_original_core_account():
    args=fixture();p=np.full((36,12),.55)
    _,a,_,_=account(*args,fee=.0002)
    mapping=ProbabilityMapping(core_overlay=True,overlay_fraction=0.)
    _,aa,_,_=account(*args,fee=.0002,decision_adapter=ProbabilityDecision(p,mapping))
    np.testing.assert_allclose(a['nav'],aa['nav'],atol=1e-14)
    np.testing.assert_allclose(a['weights'],aa['weights'],atol=1e-14)


def test_overlay_adds_weak_positions_without_replacing_core_with_full_budget():
    args=list(fixture());args[1][:]=.0002;p=np.full((36,12),.505)
    mapping=ProbabilityMapping(core_overlay=True,overlay_fraction=.1,minimum_entry=.0002)
    r,a,_,t=account(*args,fee=.0002,retain=True,decision_adapter=ProbabilityDecision(p,mapping))
    assert 0<r['average_gross']<.2
    assert any(x['reason']=='weak_prob_entry' for x in t)


def test_weak_to_core_upgrade_starts_new_core_lifetime_without_fake_physical_close():
    args=list(fixture());args[1][:15]=.0002;p=np.full((36,12),.55)
    mapping=ProbabilityMapping(core_overlay=True,overlay_fraction=.1,minimum_entry=.0002)
    _,_,_,t=account(*args,fee=.0002,retain=True,decision_adapter=ProbabilityDecision(p,mapping))
    assert any(x['hour']==65 and x['reason']=='core_entry' for x in t)
    assert not any(65<x['hour']<77 and x['reason']=='core_expiry' for x in t)
    assert any(x['hour']==77 and x['reason']=='core_expiry' for x in t)
    assert all(x['quantity_before']*x['quantity_after']>0 for x in t if x['hour']==65)
