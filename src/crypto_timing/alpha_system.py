"""Select the deployable neural family ONLY from development, then size its frozen account."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from .alpha_data import ROOT,BASE,HORIZONS,save_json
from .alpha_policy import Policy,covariance
from .alpha_account import account
from .review_data import identity

def select_primary(development):
    candidates=('Frozen_v5','NN_direct','NN_joint')
    return max(candidates,key=lambda name:development[name]['chosen']['utility'])

def freeze_system():
    policies=json.loads((ROOT/'policies.json').read_text(encoding='utf-8'));name=select_primary(policies['development']);policy=policies['selected'][name]
    lock={'model':name,'policy':policy,'selection_window':'2025-03/06 only','selection_field':'development chosen utility; no frozen score read','candidates':{n:policies['development'][n]['chosen']['utility'] for n in ('Frozen_v5','NN_direct','NN_joint')},'source':identity([Path(__file__)]),'additional_model_family_choices':3}
    save_json(ROOT/'system_lock.json',lock)
    return lock,policies

def complete_system():
    lock,policies=freeze_system();name=lock['model'];policy=lock['policy']
    # Only AFTER choosing and writing the lock is frozen evaluation loaded.
    data=np.load(ROOT/'unified_predictions.npz');hours=data['hours'];dates=np.load(BASE/'decision_time.npy');price=np.load(ROOT/'price.npy');fund=np.load(ROOT/'funding_cash_unit.npy');expected=np.load(ROOT/'expected_funding_hour.npy');minute=np.load(ROOT/'minute_open.npy',mmap_mode='r');fm=np.load(ROOT/'funding_cash_minute.npy',mmap_mode='r');pol=Policy(**policy);j=list(HORIZONS).index(pol.horizon);signal=data[name+'_calibrated'][:,:,j]
    from dataclasses import replace
    result={}
    for cap in (1.,2.,3.,5.):
        result[str(cap)]={}
        for fee,label in ((0.,'gross'),(.0002,'2bp'),(.0004,'4bp')):
            r,a,e,t=account(price,signal,fund,expected,dates,hours,replace(pol,cap=cap),fee,minute,fm);result[str(cap)][label]=r;np.savez_compressed(ROOT/f'primary_leverage_{cap}_{label}.npz',**a)
    from .alpha_analysis import equity_interval,attribute,daily_attribution
    symbols=json.loads((ROOT/'market_manifest.json').read_text(encoding='utf-8'))['symbols'];rank_results={};rank_attribution={}
    for cap in (1.,2.,3.,5.):
        rank_results[str(cap)]={}
        for fee,label in ((0.,'gross'),(.0002,'2bp'),(.0004,'4bp')):
            rr,aa,ee,tt=account(price,signal,fund,expected,dates,hours,replace(pol,cap=cap),fee,minute,fm,mode='full_rank',retain=(cap==2. and label=='4bp'))
            rank_results[str(cap)][label]=rr;np.savez_compressed(ROOT/f'primary_rank_{cap}_{label}.npz',**aa)
            if cap==2. and label=='4bp':
                rank_attribution=attribute(rr,aa,ee,symbols,dates);save_json(ROOT/'primary_rank_trades.json',tt);save_json(ROOT/'primary_rank_episodes.json',ee);np.savez_compressed(ROOT/'primary_full_rank.npz',**aa)
    r=rank_results['2.0']['4bp'];leave_out={}
    for i,symbol in enumerate(symbols):
        removed=signal.copy();removed[:,i]=0.;rr,*_=account(price,removed,fund,expected,dates,hours,pol,.0004,minute,fm)
        leave_out[symbol]={'return_4bp':rr['return'],'drawdown':rr['max_drawdown'],'meaning':'disable coin, unchanged policy and constraints, no refit'}
    main_results=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));base=policies['mixture']['baseline'];paired=equity_interval(np.load(ROOT/f'account_{name}_4bp.npz')['daily_returns'],np.load(ROOT/f'account_{base}_4bp.npz')['daily_returns'])
    rank_daily=np.load(ROOT/'primary_full_rank.npz')['daily_returns'];rank_intervals={str(cap):equity_interval(np.load(ROOT/f'primary_rank_{cap}_4bp.npz')['daily_returns']) for cap in (1.,2.,3.,5.)}
    save_json(ROOT/'primary_system_results.json',{'lock':lock,'leverage':result,'full_rank_control':r,'full_rank_leverage':rank_results,'full_rank_attribution':rank_attribution,'full_rank_net_intervals':rank_intervals,'full_rank_paired_increment':equity_interval(rank_daily,np.load(ROOT/f'account_{base}_4bp.npz')['daily_returns']),'full_rank_market_attribution':daily_attribution(rank_daily,np.load(ROOT/'market_reference.npz')['daily_returns']),'leave_one_coin_out':leave_out,'net_return_interval':main_results['models'][name]['net_return_interval'],'baseline_selected_in_development':base,'paired_increment':paired,'source':identity([Path(__file__),'src/crypto_timing/alpha_policy.py','src/crypto_timing/alpha_account.py'])})
    # Dedicated primary snapshot retains a label-free caller contract.
    raw=data[name][-1];risk=data['risk'][-1];cal=data[name+'_calibrated'][-1];latest={'model':name,'decision_utc':str(data['decision'][-1]),'earliest_fill_utc':str(data['earliest_fill'][-1]),'raw_returns':raw.tolist(),'calibrated_returns':cal.tolist(),'policy':policy,'forecast_source':'verified causal monthly prediction stream; no target read','head_live_validation':False}
    if name in ('NN_direct','NN_joint'):
        from .alpha_inference import AlphaPredictor
        cm=json.loads((ROOT/'calibration.json').read_text(encoding='utf-8'))['2026-09/'+name]['fit'];predictor=AlphaPredictor(ROOT/'monthly/2026-09',cm,policy,kind='direct' if name=='NN_direct' else 'joint');hh=hours[-1];cov,beta=covariance(price,hh-1);p=predictor.predict_completed(np.load(ROOT/'representation.npy',mmap_mode='r')[hh-3:hh+1],np.load(BASE/'targets.npy',mmap_mode='r')[hh,:,7],cov,beta,expected[hh]);assert np.max(abs(np.asarray(p['raw_returns'])-raw))<2e-6;latest.update(p);latest['head_live_validation']=True
    else:
        from .review_inference import ReviewPredictor
        from .alpha_policy import apply_calibration,target_weights
        hh=int(hours[-1]);ids=np.arange(12);features={}
        for key,multiple,length in (('fast',12,144),('slow',1,168),('minute',60,180)):
            arr=np.load(BASE/f'{key}_dynamic.npy' if key!='minute' else BASE/'minute.npy',mmap_mode='r');idx=hh*multiple+multiple-1-np.arange(length-1,-1,-1);features[key]=np.asarray(arr[idx]).transpose(1,0,2)
        features['state']=np.asarray(np.load(BASE/'state_dynamic.npy',mmap_mode='r')[hh]);past=np.load(BASE/'targets.npy',mmap_mode='r')[hh,:,7];returns=[]
        for seed in (20261004,20261005,20261006):returns.append(ReviewPredictor(Path(f'outputs/review_v5/confirm/f2_selected_{seed}/final.pt')).predict(features,ids,past)['return_estimate'])
        raw_live=np.repeat(np.mean(returns,0)[:,None],4,-1);assert np.max(abs(raw_live-raw))<2e-6
        cm=json.loads((ROOT/'calibration.json').read_text(encoding='utf-8'))['2026-09/Frozen_v5']['fit'];cal_live=apply_calibration(raw_live[None],risk[None],cm)[0];cov,beta=covariance(price,hh-1)
        latest['calibrated_returns']=cal_live.tolist();latest['target_weights_if_flat']=target_weights(cal_live[:,j],expected[hh],cov,beta,pol).tolist();latest['head_live_validation']=True;latest['target_identity']='native4h+5m, mature-OOF projection to next1m 2/4/8/12h'
    save_json(ROOT/'latest_primary_prediction.json',latest);print(f'primary frozen system {name}',flush=True)
