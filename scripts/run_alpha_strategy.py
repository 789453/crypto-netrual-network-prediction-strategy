from pathlib import Path
import argparse,platform,sys,json
import numpy as np
import torch
from crypto_timing.alpha_data import ROOT,BASE,build_market,encode,summaries,save_json
from crypto_timing.alpha_models import train_month

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['data','models','baselines','analysis','report','all'],default='all');p.add_argument('--month');args=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');ROOT.mkdir(parents=True,exist_ok=True)
    save_json(ROOT/'runtime.json',{'python':sys.executable,'version':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)})
    if args.stage in ('data','all'):build_market();encode();summaries()
    if args.stage in ('models','all'):
        build_market();rep=np.load(ROOT/'representation.npy',mmap_mode='r');features=summaries();target=np.load(ROOT/'target.npy');valid=np.load(ROOT/'valid.npy');dates=np.load(BASE/'decision_time.npy')
        months=[np.datetime64(args.month,'M')] if args.month else np.arange(np.datetime64('2025-01','M'),np.datetime64('2026-10','M'))
        for month in months:train_month(month,rep,features,target,valid,dates)
    if args.stage in ('baselines','all'):
        from crypto_timing.alpha_baselines import fit_fair_baselines
        fit_fair_baselines()
    if args.stage in ('analysis','all'):
        from crypto_timing.alpha_analysis import analyze
        analyze()
        from crypto_timing.alpha_system import complete_system
        complete_system()
    if args.stage in ('report','all'):
        if args.stage=='all':
            import pytest
            assert pytest.main(['-q','--junitxml=outputs/alpha_strategy/tests.xml'])==0
        from build_alpha_report import main as report
        report()
if __name__=='__main__':main()
