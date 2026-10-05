"""Focused v5 experiment sequence; all historical results remain exploratory."""
from pathlib import Path
from dataclasses import replace
import argparse,json
import torch
from crypto_timing.review_data import ReviewStore,build_overlay,identity
from crypto_timing.review_model import ReviewConfig
from crypto_timing.review_training import train_review

ROOT=Path('outputs/review_v5');BASE=Path('outputs/mechanism/cache_v4');OVER=ROOT/'cache'
def save(name,data): (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def verify_mapping(expected):
    if identity(expected.keys())!=expected: raise ValueError('actual input/source contents changed; refusing stale cache or run reuse')
def verify_inputs():
    meta=json.loads((OVER/'manifest.json').read_text(encoding='utf-8'))
    for key in ('source','base_cache'): verify_mapping(meta['contract'][key])
    verify_mapping({str(OVER/name):sha for name,sha in meta['content_sha256'].items()})
    save('preflight_manifest.json',{'actual_content_matches_contract':True,'source_and_cache':meta['contract'],'overlay':meta['content_sha256'],'orchestrator_sha256':identity([Path(__file__)])})
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('cache','smoke','screen','confirm','all'),default='all');args=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage in ('cache','all'):
        build_overlay(Path('D:/Trading/practical_crypto_strategy/data/parquet'),BASE,OVER)
        if args.stage=='cache':return
    verify_inputs()
    if args.stage=='smoke':
        s=ReviewStore(BASE,OVER,device,'2025-03-01','2025-05-01','2025-07-01')
        train_review(s,ReviewConfig(events=True,minute_order='gru',segments=True,tail=True,baseline=True),ROOT/'smoke',epochs=1,steps_per_epoch=2);return
    if args.stage in ('screen','all'):
        s=ReviewStore(BASE,OVER,device,'2025-03-01','2025-05-01','2025-07-01');records={}
        base=ReviewConfig()
        configs={'R':replace(base,auxiliary='none'),'raw_soft':replace(base,auxiliary='raw',budget='soft'),'raw_hard':replace(base,auxiliary='raw'),
                 'TA_hard':replace(base,auxiliary='ta'),'shapedQ_hard':base,'shapedQ_tail':replace(base,tail=True)}
        for name,c in configs.items():records[name]=train_review(s,c,ROOT/'screen'/name,epochs=4,steps_per_epoch=128);save('screen_results.json',records)
        # Only first two OOF months select mechanisms; later two months do not change the choice.
        winner=min(('R','TA_hard','shapedQ_hard','shapedQ_tail'),key=lambda n:records[n]['results']['validation']['mse']);chosen=ReviewConfig(**records[winner]['config'])
        extras={'baseline':replace(chosen,baseline=True),'no_body':replace(chosen,features='no_body',fast_dim=15),'no_wicks':replace(chosen,features='no_wicks',fast_dim=14)}
        for name,c in extras.items():records[name]=train_review(s,c,ROOT/'screen'/name,epochs=4,steps_per_epoch=128);save('screen_results.json',records)
        chosen=replace(chosen,baseline=True)
        event_configs={'common':replace(chosen,minute_order='common'),'events_common':replace(chosen,events=True,minute_order='common'),
                       'events_gru':replace(chosen,events=True,minute_order='gru'),'events_gru_segments':replace(chosen,events=True,minute_order='gru',segments=True)}
        for name,c in event_configs.items():records[name]=train_review(s,c,ROOT/'screen'/name,epochs=4,steps_per_epoch=128);save('screen_results.json',records)
        finalists=[winner,'baseline',*event_configs];selected=min(finalists,key=lambda n:records[n]['results']['validation']['mse'])
        locked={'auxiliary_choice':winner,'selected_run':selected,'config':records[selected]['config'],'selection_window':'2025-03/04 internal OOF only','later_window_not_selection':'2025-05/06','event_hypothesis_remains_inspectable':records['events_gru_segments']['config'],'no_body_and_no_wicks':'separate mechanism diagnostics; not combined deletion'}
        save('locked_design.json',locked);del s;torch.cuda.empty_cache()
    if args.stage in ('confirm','all'):
        locked=json.loads((ROOT/'locked_design.json').read_text(encoding='utf-8'));cfg=ReviewConfig(**locked['config']);records={}
        # Truly time-out-of-fold predictions for the second stage; no in-sample fitted signals.
        for name,dates in [('oof_f1_early',('2024-11-01','2025-02-01','2025-03-01')),('oof_f1_late',('2025-03-01','2025-05-01','2025-07-01')),('oof_f2',('2025-11-01','2026-01-01','2026-02-01'))]:
            s=ReviewStore(BASE,OVER,device,*dates)
            for seed in (20261004,20261005,20261006):
                run=name if seed==20261004 else f'{name}_seed{seed}'
                records[run]=train_review(s,replace(cfg,seed=seed),ROOT/'confirm'/run,epochs=3,probe=name=='oof_f1_late' and seed==20261004);save('confirmation_results.json',records)
            del s;torch.cuda.empty_cache()
        for fold,dates in [('f1',('2025-07-01','2025-11-01','2026-02-01')),('f2',('2026-02-01','2026-06-01','2026-09-25'))]:
            s=ReviewStore(BASE,OVER,device,*dates)
            for seed in (20261004,20261005,20261006):
                name=f'{fold}_selected_{seed}';records[name]=train_review(s,replace(cfg,seed=seed),ROOT/'confirm'/name,epochs=3,probe=seed==20261004);save('confirmation_results.json',records)
            contrast=replace(cfg,auxiliary='shaped_q') if cfg.auxiliary=='none' and not cfg.segments and not cfg.tail else replace(cfg,auxiliary='none',segments=False,tail=False)
            name=f'{fold}_matched_task_contrast';records[name]=train_review(s,contrast,ROOT/'confirm'/name,epochs=3);save('confirmation_results.json',records)
            if fold=='f1':
                # Full proposal is a fixed mechanism experiment, not a replacement chosen on outer results.
                proposal=replace(ReviewConfig(**locked['event_hypothesis_remains_inspectable']),auxiliary='shaped_q',tail=True)
                name='f1_mechanism_proposal';records[name]=train_review(s,proposal,ROOT/'confirm'/name,epochs=3,probe=True);save('confirmation_results.json',records)
            del s;torch.cuda.empty_cache()
        print(json.dumps({'event':'review_research_complete','screen_runs':len(json.loads((ROOT/'screen_results.json').read_text())),'full_runs':len(records),'locked':locked}),flush=True)
if __name__=='__main__':main()
