from __future__ import annotations
import json,sys,platform
from pathlib import Path
import numpy as np
import torch
from crypto_timing.review_model import ReviewConfig,ReviewNetwork
from crypto_timing.review_data import ReviewStore,identity
from crypto_timing.review_training import predict,score_predictions,decomposition
from crypto_timing.review_analysis import second_stage,apply_calibrator,concat_data,smooth,time_interval,diagnostic_breakdown,lifetime,nav_backtest
from crypto_timing.mechanism_training import normalize,scaled_labels
from crypto_timing.mechanism_data import FAST_NAMES,STATE_NAMES,MINUTE_NAMES

ROOT=Path('outputs/review_v5');BASE=Path('outputs/mechanism/cache_v4');OVER=ROOT/'cache'
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def save(name,data): (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def load(folder,split):
    with np.load(folder/f'{split}.npz') as f:return {k:f[k] for k in f.files}

def rebase(data,floors,raw):
    data={k:v.copy() for k,v in data.items()};newscale=np.maximum(raw[data['hours'],:,7],floors);factor=data['scale']/newscale
    for name in ('signal','delta','baseline'):data[name]*=factor
    data['y']=data['raw_return']/newscale;data['scale']=newscale
    return data

def oof_ensemble(name,split,floors,raw):
    folders=[name,f'{name}_seed20261005',f'{name}_seed20261006']
    ds=[rebase(load(ROOT/'confirm'/f,split),floors,raw) for f in folders]
    for dd in ds[1:]:assert np.array_equal(dd['hours'],ds[0]['hours'])
    data=ds[0]
    for key in ('signal','delta','path','segments','tail'):data[key]=np.mean([d[key] for d in ds],axis=0)
    return data

def field_audit(saved,store):
    result={};critical_fast=(0,1,3,4,6,8,11,15)
    for name,file,field_names,indices,multiple,scaler in [('fast','fast_dynamic',FAST_NAMES,critical_fast,12,saved['scalers']['dynamic_fast']),('state','state_dynamic',STATE_NAMES,(5,6,9,12,16),1,saved['scalers']['dynamic_state']),('minute','minute',MINUTE_NAMES,tuple(range(12)),60,saved['scalers']['minute'])]:
        arr=np.load(BASE/f'{file}.npy',mmap_mode='r');times=np.repeat(store.dates,multiple).astype('datetime64[M]');center=np.array(scaler['center']);scale=np.array(scaler['scale']);rows=[]
        for month in np.unique(times):
            ids=np.flatnonzero(times==month)
            for si,symbol in enumerate(store.manifest['symbols']):
                x=np.asarray(arr[ids,si],float)
                for j in indices:
                    raw=x[:,j];mask=np.isfinite(raw);raw=raw[mask]
                    if not len(raw):continue
                    z=np.clip((raw-center[si,j])/scale[si,j],-8,8)
                    rows.append({'month':str(month),'symbol':symbol,'field':field_names[j],'valid':len(raw),'raw_q01':float(np.quantile(raw,.01)),'raw_median':float(np.median(raw)),'raw_q99':float(np.quantile(raw,.99)),'normalized_mean':float(z.mean()),'normalized_rms':float(np.sqrt((z*z).mean())),'clip_fraction':float((np.abs(z)>=8).mean())})
        result[name]=rows
    return result

def tail_audit(data):
    result={}
    for j,name in enumerate(('logT','Q','A')):
        y=data['raw_path'][...,j][data['valid']].astype(float);err=(data['path'][...,j][data['valid']]-y)**2;v=np.sort(y*y);ee=np.sort(err)
        result[name]={'std':float(y.std()),'label_square_top12_share':float(v[-12:].sum()/max(v.sum(),1e-12)),'label_square_top1pct_share':float(v[-max(len(v)//100,1):].sum()/max(v.sum(),1e-12)), 'raw_numeric_error_not_native_if_transformed':float(err.mean()),'interpretation':'native target errors below depend on the explicitly saved transform'}
    return result

def active_parameter_count(saved):
    cfg=saved['config'];inactive=[]
    if cfg['auxiliary']=='none':inactive+=['path.','risk_readout.']
    if not cfg['segments']:inactive+=['segment_head.']
    if not cfg['tail']:inactive+=['tail_head.']
    total=sum(v.numel() for v in saved['model'].values())
    connected=sum(v.numel() for name,v in saved['model'].items() if not name.startswith(tuple(inactive)))
    return {'instantiated':total,'enabled_loss_connected_tensors':connected,'scope':'parameter tensors connected to enabled losses; zero/unused coordinates inside a tensor not separately counted'}

def main():
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda')
    screen,confirm,locked=read('screen_results.json'),read('confirmation_results.json'),read('locked_design.json');cfg=ReviewConfig(**locked['config']);state=np.load(BASE/'state_dynamic.npy',mmap_mode='r');raw=np.load(BASE/'targets.npy');nxt=np.load(OVER/'next_return.npy')
    analysis={'screen':{},'folds':{},'field_audit_files':{},'historical_status':'all history exploratory; fixed hypotheses and temporal provenance do not create unseen forward data'}
    reference=load(ROOT/'screen'/'R','validation')
    for name,r in screen.items():
        data=load(ROOT/'screen'/name,'validation');assert np.array_equal(data['hours'],reference['hours'])
        checkpoint=torch.load(ROOT/'screen'/name/'final.pt',weights_only=False,map_location='cpu')
        yy_all,_=scaled_labels(raw,np.asarray(checkpoint['floors']));stop=np.searchsorted(np.load(BASE/'decision_time.npy'),np.datetime64('2025-03-01'))-9
        train_valid=np.isfinite(state[720:stop]).all(-1)&np.isfinite(yy_all[720:stop]).all(-1)
        train_constant=yy_all[720:stop,:,3:6][train_valid].mean(0)
        aux={}
        native=data['raw_path'] if r['config']['auxiliary']=='raw' else data['shaped_path']
        for j,target in enumerate(('logT','Q' if r['config']['auxiliary']=='raw' else 'asinhQ','A')):
            if r['config']['auxiliary'] in ('none','ta') and (r['config']['auxiliary']=='none' or j==1):continue
            yy=native[...,j];pp=data['path'][...,j];valid=data['valid'];const=float(train_constant[j]) if r['config']['auxiliary']=='raw' else 0.
            aux[target]={'mse':float(np.mean((yy[valid]-pp[valid])**2)),'train_constant_mse':float(np.mean((yy[valid]-const)**2)),'error_top1pct_share':float(np.sort((yy[valid]-pp[valid])**2)[-max(int(valid.sum())//100,1):].sum()/max(np.sum((yy[valid]-pp[valid])**2),1e-12))}
        analysis['screen'][name]={'paired_vs_R':time_interval(reference['signal'],data['signal'],data['y'],data['dates'],data['valid']),'decomposition':score_predictions(data),'native_auxiliary':aux,'tail_audit':tail_audit(data),'checkpoint_status':'fixed last step, not best on OOF window'}
    for fold,dates,oof in [('f1',('2025-07-01','2025-11-01','2026-02-01'),'oof_f1_early'),('f2',('2026-02-01','2026-06-01','2026-09-25'),'oof_f2')]:
        s=ReviewStore(BASE,OVER,device,*dates);mainfolder=ROOT/'confirm'/f'{fold}_selected_20261004';saved=torch.load(mainfolder/'final.pt',weights_only=False,map_location=device)
        otr=oof_ensemble(oof,'validation',saved['floors'],raw);ov=oof_ensemble(oof,'replay',saved['floors'],raw)
        meta=second_stage(otr,ov,state[otr['hours']],state[ov['hours']]);save(f'second_stage_{fold}.json',meta)
        foldresult={'second_stage':meta,'splits':{},'stream':{},'full_target_geometry':{},'parameter_activity':active_parameter_count(saved)}
        for split in ('validation','replay'):
            contrast=load(ROOT/'confirm'/f'{fold}_matched_task_contrast',split);valid=contrast['valid'];geometry={}
            for j,label in ((0,'logT'),(1,'Q'),(2,'A')):
                rr=contrast['raw_path'][...,j][valid].astype(float);zz=contrast['shaped_path'][...,j][valid].astype(float);pp=contrast['path'][...,j][valid].astype(float)
                for target,values in (('raw',rr),('transformed_standardized',zz)):
                    square=np.sort(values*values);top=max(len(square)//100,1);geometry[f'{label}_{target}']={'std':float(values.std()),'largest12_square_share':float(square[-12:].sum()/max(square.sum(),1e-12)),'top1pct_square_share':float(square[-top:].sum()/max(square.sum(),1e-12))}
                errors=(pp-zz)**2;ordered=np.sort(errors);geometry[f'{label}_native_prediction']={'mse':float(errors.mean()),'train_constant_mse':float((zz*zz).mean()),'skill_vs_train_constant':float(1-errors.mean()/max((zz*zz).mean(),1e-12)),'top1pct_residual_loss_share':float(ordered[-max(len(ordered)//100,1):].sum()/max(ordered.sum(),1e-12))}
            foldresult['full_target_geometry'][split]=geometry
        if fold=='f1':
            inner={}
            for split in ('validation','replay'):
                dd=oof_ensemble('oof_f1_late',split,saved['floors'],raw);pp=apply_calibrator(dd,state[dd['hours']],meta['calibrator']);inner[split]={'raw':decomposition(dd['signal'],dd['y'],dd['valid']),'frozen_calibration':decomposition(pp,dd['y'],dd['valid'])}
            foldresult['internal_future_calibration_check']=inner
        for split in ('validation','replay'):
            ds=[load(ROOT/'confirm'/f'{fold}_selected_{seed}',split) for seed in (20261004,20261005,20261006)];data=ds[0]
            for dd in ds[1:]:assert np.array_equal(dd['hours'],data['hours'])
            for k in ('signal','delta','path','segments','tail'):data[k]=np.mean([d[k] for d in ds],axis=0)
            calibrated=apply_calibrator(data,state[data['hours']],meta['calibrator']);variants={'neural_raw':data['signal'],'baseline_only':s.baseline[data['hours']],'OOF_calibrated':calibrated}
            unit=meta['filter']['unit'];tau=meta['filter']['half_life'];values=calibrated if unit=='normalized' else calibrated*data['scale'];pp=smooth(values,data['dates'],tau);variants['OOF_filtered']=pp if unit=='normalized' else pp/data['scale']
            contrast=load(ROOT/'confirm'/f'{fold}_matched_task_contrast',split);variants['matched_task_contrast']=contrast['signal']
            result={}
            for name,pred in variants.items():
                current={**data,'signal':pred};legacy=(data['future_no_trade']==0).all(1)
                result[name]={'metrics':score_predictions(current),'legacy_future_quality_sensitivity':decomposition(pred[legacy],data['y'][legacy],data['valid'][legacy]),'interval_vs_zero':time_interval(np.zeros_like(pred),pred,data['y'],data['dates'],data['valid']), 'interval_vs_neural':time_interval(data['signal'],pred,data['y'],data['dates'],data['valid']),'diagnostics':diagnostic_breakdown(current,state[data['hours']]),'lifetime':lifetime(pred,data['segment_raw'],data['dates'],data['valid'])}
            base_error=result['baseline_only']['metrics']['mse'];original=result['neural_raw']['metrics']
            for name in result:
                d=result[name]['metrics'];result[name]['increment_over_causal_baseline']=1-d['mse']/base_error
                result[name]['shrinkage_diagnostics']={'dynamic_covariance_retention':d['covariance_gain']/original['covariance_gain'] if abs(original['covariance_gain'])>1e-12 else None,'signal_std_retention':d['signal_std']/max(original['signal_std'],1e-12),'warning':'Lower total MSE after near-zero shrinkage is not evidence of stronger dynamic information.'}
            foldresult['splits'][split]=result
            np.savez_compressed(ROOT/f'ensemble_{fold}_{split}.npz',**data,calibrated=calibrated,filtered=variants['OOF_filtered'])
        # Fill the purge gap with actual model inference: complete hourly NAV, never stitch prices across a gap.
        start=s.splits['validation'][0];stop=s.splits['replay'][-1]+1;hours=np.arange(start,stop);datasets=[]
        for seed in (20261004,20261005,20261006):
            ck=torch.load(ROOT/'confirm'/f'{fold}_selected_{seed}'/'final.pt',weights_only=False,map_location=device);model=ReviewNetwork(ReviewConfig(**ck['config'])).to(device);model.load_state_dict(ck['model']);datasets.append(predict(model,s,hours,cfg));del model
        stream=datasets[0]
        for key in ('signal','delta','path','segments','tail'):stream[key]=np.mean([d[key] for d in datasets],axis=0)
        calibrated=apply_calibrator(stream,state[hours],meta['calibrator']);unit=meta['filter']['unit'];values=calibrated if unit=='normalized' else calibrated*stream['scale'];filtered=smooth(values,stream['dates'],meta['filter']['half_life']);filtered=filtered if unit=='normalized' else filtered/stream['scale']
        for name,pred in {'neural_raw':stream['signal'],'OOF_calibrated':calibrated,'OOF_filtered':filtered,'baseline_only':s.baseline[hours]}.items():
            curves={}
            for fee in (2,4,8):
                nav=nav_backtest(pred,hours,stream['dates'],nxt,fee);curves[str(fee)]=nav['summary']
                if fee==4:np.savez_compressed(ROOT/f'nav_{fold}_{name}.npz',**{k:v for k,v in nav.items() if k!='summary'})
            foldresult['stream'][name]=curves
        np.savez_compressed(ROOT/f'stream_{fold}.npz',**stream,calibrated=calibrated,filtered=filtered)
        audit=field_audit(saved,s);save(f'field_audit_{fold}.json',audit);analysis['field_audit_files'][fold]=f'field_audit_{fold}.json'
        foldresult['quality']=s.quality_counts;foldresult['stream_hours']=len(hours);analysis['folds'][fold]=foldresult;save('analysis_results.json',analysis)
        del s;torch.cuda.empty_cache();print(json.dumps({'event':'review_analysis_fold','fold':fold,'calibration':meta['calibrator'],'filter':meta['filter'],'nav':foldresult['stream']},ensure_ascii=False),flush=True)
    runtime={'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'analysis_source_sha256':identity([Path(__file__),'src/crypto_timing/review_analysis.py']),'screen_runs':len(screen),'full_runs':len(confirm)}
    save('runtime_manifest.json',runtime);print('v5 analysis complete',flush=True)
if __name__=='__main__':main()
