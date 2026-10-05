import numpy as np
import torch
import importlib.util
from pathlib import Path
import pytest
from torch import nn
from crypto_timing.review_model import ReviewConfig,ReviewNetwork,review_losses,gradient_budget,PatchReadout
from crypto_timing.review_data import causal_baseline,disjoint_targets,event_features
from crypto_timing.review_data import identity
from crypto_timing.review_inference import ReviewPredictor
from crypto_timing.review_training import decomposition
from crypto_timing.review_analysis import time_interval,lifetime,nav_backtest,fit_calibrator,apply_calibrator,smooth

def test_budget_covers_context_and_zero_main_parameters():
    model=nn.Module();model.fast=nn.Linear(3,3);model.context=nn.Linear(3,3);model.risk=nn.Linear(3,1)
    model.zero_main=nn.Parameter(torch.ones(3))
    x=torch.ones(4,3);h=model.fast(x)+model.context(x);primary=(h[:,0]**2).mean()+0*model.zero_main.sum();aux=1e7*((model.risk(h)**2).mean()+model.zero_main.square().sum())
    result=gradient_budget(model,primary,aux)
    assert {'fast','context'}.issubset(result['groups'])
    assert result['violations']==0 and result['max_parameter_ratio']<=.30001
    assert all(v['ratio']<=.30001 for v in result['groups'].values())
    assert torch.isfinite(model.risk.weight.grad).all()
    # A shared parameter tensor with exactly zero main gradient cannot get an auxiliary update.
    torch.testing.assert_close(model.zero_main.grad,torch.zeros_like(model.zero_main))

def test_primary_mean_target_not_aux_transform_and_missing_safe():
    signal=torch.tensor([0.,0.,0.],requires_grad=True)
    out={'signal':signal,'path':torch.zeros(3,3,requires_grad=True),'segments':torch.zeros(3,5),'tail':torch.zeros(3)}
    target={'y':torch.tensor([1.,100.,float('nan')]),'valid':torch.tensor([True,True,False]),'raw_path':torch.zeros(3,3),'path':torch.ones(3,3)}
    main,_,_=review_losses(out,target,ReviewConfig());main.backward()
    torch.testing.assert_close(signal.grad,torch.tensor([-1.,-100.,0.]))

def test_baseline_never_sees_unmatured_labels():
    rng=np.random.default_rng(3);y=rng.normal(size=(100,2));state=rng.normal(size=(100,2,24));valid=np.ones_like(y,dtype=bool)
    b=causal_baseline(y,state,valid);altered=y.copy();altered[60:]+=100
    changed=causal_baseline(altered,state,valid)
    np.testing.assert_array_equal(b[:65],changed[:65]);assert not np.allclose(b[65:],changed[65:])
    prefix=causal_baseline(y[:60],state[:60],valid[:60]);np.testing.assert_array_equal(prefix,b[:60])

def test_segment_boundaries_and_exact_compounding():
    op=np.exp(np.arange(1200)*.0001);segments,nxt=disjoint_targets(op)
    np.testing.assert_allclose(segments[0,:4],.006,rtol=1e-6)
    np.testing.assert_allclose(np.expm1(segments[0,:4].sum()),op[305]/op[65]-1,rtol=1e-6)
    assert np.isclose(nxt[0],op[125]/op[65]-1)

def test_event_is_prefix_causal():
    rng=np.random.default_rng(4);cl=np.exp(np.cumsum(rng.normal(0,.001,600)));op=cl.copy();q=np.ones(600);pressure=np.zeros(600);sigma=np.ones(600)*.001
    full=event_features(op,cl,q,pressure,sigma,q);short=event_features(op[:480],cl[:480],q[:480],pressure[:480],sigma[:480],q[:480])
    np.testing.assert_array_equal(full[:8],short)

def test_exact_signal_head_reconstruction():
    cfg=ReviewConfig(events=True,minute_order='gru',baseline=True);model=ReviewNetwork(cfg).eval()
    x=torch.randn(2,144,16);s=torch.randn(2,168,10);c=torch.randn(2,32);m=torch.randn(2,180,12);b=torch.tensor([.01,-.02])
    with torch.no_grad():out=model(x,torch.ones_like(x),s,torch.ones_like(s),c,torch.ones_like(c),m,torch.ones_like(m),b)
    reconstructed=out['representation']@model.signal.weight.T+model.signal.bias
    torch.testing.assert_close(reconstructed[:,0]+b,out['signal'],rtol=0,atol=0)

def test_minute_readouts_keep_identical_base_summaries():
    common=PatchReadout('common');ordered=PatchReadout('gru')
    assert common.project[0].in_features==48*7 and ordered.project[0].in_features==48*7+32

def test_skill_identity_and_time_gap_lags():
    p=np.array([[.1],[.2],[.4],[.3]]);y=np.array([[-1.],[.1],[.5],[.2]])
    d=decomposition(p,y);assert abs(d['skill']-sum(d[k] for k in ('covariance_gain','mean_alignment','variance_penalty')))<1e-12
    dates=np.array(['2025-01-01T00','2025-01-01T01','2025-01-01T04','2025-01-01T05'],dtype='datetime64[h]')
    raw=np.tile(y[:,:,None],(1,1,5));result=lifetime(p,raw,dates,np.ones_like(y,dtype=bool))
    assert result['exact_hour_autocorrelation']['1']==1.0

def test_calendar_blocks_and_constant_filter():
    dates=np.arange(np.datetime64('2025-01-01T00'),np.datetime64('2025-01-15T00'),np.timedelta64(1,'h'));dates=np.delete(dates,np.arange(30,39))
    y=np.ones((len(dates),2));p=y*.1;ci=time_interval(np.zeros_like(p),p,y,dates,np.ones_like(y,dtype=bool),repetitions=20)
    np.testing.assert_allclose(ci['ci95'],[.19,.19]);np.testing.assert_array_equal(smooth(p,dates,2),p)

def test_four_phase_nav_never_sums_overlapping_labels():
    n=20;signal=np.ones((n,2))*.1;hours=np.arange(n);dates=np.arange(np.datetime64('2025-01-01'),np.datetime64('2025-01-01')+np.timedelta64(n,'h'),np.timedelta64(1,'h'))
    r=np.ones((n,2))*.01;result=nav_backtest(signal,hours,dates,r,4)
    np.testing.assert_allclose(result['weights'][3:].sum(1),np.tanh(1))
    assert result['summary']['total_return']<result['summary']['gross_return'] and result['summary']['mean_gross_exposure']<=1
    assert np.all(result['turnover'][5:15]>0), 'mark-to-market weight drift must be charged too'

def test_calibrator_uses_only_supplied_oof_material_and_bounded_slopes():
    rng=np.random.default_rng(5);d=rng.normal(0,.1,(100,2));data={'delta':d,'baseline':np.zeros_like(d),'y':.3*d+rng.normal(0,.01,d.shape),'valid':np.ones_like(d,dtype=bool)};state=rng.normal(size=(100,2,24))
    c=fit_calibrator(data,state,.001,True);pred=apply_calibrator(data,state,c)
    assert all(0<=x<=1 for x in c['state_slopes']);assert pred.shape==d.shape

def test_actual_input_change_refuses_reuse(tmp_path):
    path=tmp_path/'input.txt';path.write_text('before');expected=identity([path])
    spec=importlib.util.spec_from_file_location('review_runner_test',Path('scripts/run_review_research.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_mapping(expected);path.write_text('after')
    with pytest.raises(ValueError,match='refusing stale'):module.verify_mapping(expected)

def test_v5_inference_no_labels_and_no_diagnostic_copy(tmp_path):
    cfg=ReviewConfig(width=64);net=ReviewNetwork(cfg)
    scalers={}
    for name,dim in (('fast',20),('slow',10),('state',24),('minute',12)):
        scalers['minute' if name=='minute' else f'dynamic_{name}']={'center':np.zeros((2,dim)).tolist(),'scale':np.ones((2,dim)).tolist()}
    path=tmp_path/'v5.pt';torch.save({'config':cfg.__dict__,'model':net.state_dict(),'floors':np.ones(2)*.01,'scalers':scalers},path)
    predictor=ReviewPredictor(path,'cpu');features={name:np.zeros((2,length,dim)) for name,length,dim in (('fast',144,20),('slow',168,10),('minute',180,12))};features['state']=np.zeros((2,24))
    out=predictor.predict(features,[0,1],[.02,.03]);features['fast'][...,-1]=1e7
    changed=predictor.predict(features,[0,1],[.02,.03]);np.testing.assert_array_equal(out['signal_level'],changed['signal_level']);np.testing.assert_allclose(out['return_estimate'],out['signal_level']*np.array([.02,.03]))
