"""Verify source-to-training-to-account-to-report identities after the research run."""
from pathlib import Path
import json,hashlib
import numpy as np
from crypto_timing.alpha_data import ROOT,BASE,save_json
from crypto_timing.review_data import identity,training_identity

def main():
    market=json.loads((ROOT/'market_manifest.json').read_text(encoding='utf-8'))
    for key in ('source','base','encoder','builder'):
        expected=market['contract'][key];assert identity(expected)==expected,key
    assert identity(market['content'])==market['content']
    representation=json.loads((ROOT/'representation_manifest.json').read_text(encoding='utf-8'))
    for key in ('encoder','outputs','source'):assert identity(representation[key])==representation[key]
    total=0;monthly=[];date=np.load(BASE/'decision_time.npy')
    for month in np.arange(np.datetime64('2025-01','M'),np.datetime64('2026-10','M')):
        folder=ROOT/'monthly'/str(month);s=json.loads((folder/'summary.json').read_text(encoding='utf-8'));contract=s['contract'];assert identity(contract['source'])==contract['source'];assert contract['v5_dependencies']==training_identity()
        fair=json.loads((folder/'fair_baselines.json').read_text(encoding='utf-8'));assert identity(fair['contract']['source'])==fair['contract']['source'];assert identity(fair['contract']['feature_file'])==fair['contract']['feature_file'];assert identity(fair['output'])==fair['output']
        f=np.load(folder/'predictions.npz');assert np.all(date[f['hours']]>=np.datetime64(month,'ns'))
        for kind in ('direct','joint'):
            for seed in (20261004,20261005,20261006):assert (folder/f'{kind}_{seed}.pt').exists();total+=1
        monthly.append(str(month))
    meta=json.loads((ROOT/'calibration.json').read_text(encoding='utf-8'));assert all(np.datetime64(v['latest_target_maturity'])<=np.datetime64(v['cutoff']) for v in meta.values())
    policy=json.loads((ROOT/'policies.json').read_text(encoding='utf-8'));assert policy['search_count']==173
    result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));assert identity(result['identity'])==result['identity'];errors={}
    for model,record in result['models'].items():
        a=np.load(ROOT/f'account_{model}_4bp.npz');net=(a['pnl']+a['funding']-a['fees']).sum();assert np.isfinite(a['nav']).all();error=abs(net-(a['nav'][-1]-1));assert error<1e-7,(model,error);assert np.all(np.diff(a['hours'])==1);assert np.isclose(a['fees'].sum(),.0004*a['traded_notional'].sum());errors[model]=float(error)
        trades=json.loads((ROOT/f'trades_{model}.json').read_text(encoding='utf-8'));assert abs(sum(x['notional'] for x in trades)-a['traded_notional'].sum())<1e-7
    from crypto_timing.alpha_system import select_primary
    primary=json.loads((ROOT/'primary_system_results.json').read_text(encoding='utf-8'));assert primary['lock']['model']==select_primary(policy['development']);assert identity(primary['source'])==primary['source'];assert identity(primary['lock']['source'])==primary['lock']['source']
    primary_errors={}
    for prefix in ('primary_leverage','primary_rank'):
        for cap in (1.,2.,3.,5.):
            for label,fee in (('gross',0.),('2bp',.0002),('4bp',.0004)):
                a=np.load(ROOT/f'{prefix}_{cap}_{label}.npz');error=abs((a['pnl']+a['funding']-a['fees']).sum()-(a['nav'][-1]-1));assert error<1e-7;assert np.isfinite(a['nav']).all();assert np.all(np.diff(a['hours'])==1);assert np.isclose(a['fees'].sum(),fee*a['traded_notional'].sum());primary_errors[f'{prefix}_{cap}_{label}']=float(error)
    latest=json.loads((ROOT/'latest_primary_prediction.json').read_text(encoding='utf-8'));assert latest['head_live_validation'] and latest['model']==primary['lock']['model']
    native=json.loads((ROOT/'v3_native_reproduction.json').read_text(encoding='utf-8'));assert max(abs(x['difference']) for x in native['checks'])<1e-10
    delivery=json.loads((ROOT/'delivery_manifest.json').read_text(encoding='utf-8'))['sha256'];assert identity(delivery)==delivery
    import xml.etree.ElementTree as ET
    tests=ET.parse(ROOT/'tests.xml').getroot();assert len(tests.findall('.//testcase'))==52 and not tests.findall('.//failure') and not tests.findall('.//error')
    out={'raw_cache_source_verified':True,'encoder_representation_verified':True,'monthly_cutoffs':monthly,'final_neural_readouts':total,'same_field_fair_baselines_verified':21,'calibration_maturity_verified':True,'continuous_accounts_reconcile':errors,'primary_and_full_rank_accounts_reconcile':primary_errors,'development_only_primary_selection_verified':True,'primary_live_label_free_inference_verified':True,'native_v3_curve_reproduced':True,'real_trades_fee_identity_verified':True,'delivery_files_verified':len(delivery),'tests_passed':52,'actual_outputs':identity([ROOT/'results.json',ROOT/'primary_system_results.json',ROOT/'unified_predictions.npz',ROOT/'latest_label_free_prediction.json',ROOT/'latest_primary_prediction.json',ROOT/'policies.json'])}
    save_json(ROOT/'final_audit.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
