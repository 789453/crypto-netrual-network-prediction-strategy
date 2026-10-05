"""Final artifact identity, immutable sources and independent ledger audit."""
from pathlib import Path
import json,sys,numpy as np
from run_signal_control import OUT,read,save,digest

REPORT=Path('reports/signal_control_2026-10-05')
def main():
    results=read(OUT/'results.json');identity=digest(OUT/'results.json');audit=read(OUT/'audit.json')
    assert audit['pytest_passed_tests']==36 and audit['tested_accounts']==16
    assert all(digest(path)==expected for path,expected in results['source_sha256'].items())
    for filename in ['index.html','coins.html']:
        assert identity in (REPORT/filename).read_text(encoding='utf-8')
    assert identity in Path('docs/SIGNAL_CONTROL_RESEARCH_2026-10-05.md').read_text(encoding='utf-8')
    assert (OUT/'results.json').read_bytes()==(REPORT/'results.json').read_bytes()
    html=(REPORT/'coins.html').read_text(encoding='utf-8')
    assert html.count('class="panel"')==12 and html.count('<option value=')==12
    assert len(list((REPORT/'assets').glob('*.svg')))==16
    ledger={}
    for name,r in results['accounts'].items():
        a=np.load(OUT/f'account_{name}.npz');gain=a['pnl']+a['funding']-a['fees']
        fee_error=abs(float(a['traded_notional'].sum())*r['fee_per_side']-r['fee'])
        nav_error=float(np.max(abs(a['nav']-np.r_[1.,1+np.cumsum(gain.sum(1))])))
        contribution_error=abs(float(np.sum(r['diagnostics']['coin_net_contribution']))-r['return'])
        episode_error=abs(sum(e['price_pnl']+e['funding']-e['fee'] for e in read(OUT/f'episodes_{name}.json'))-r['return'])
        log_error=abs(sum(t['fee'] for t in read(OUT/f'trades_{name}.json'))-r['fee'])
        assert max(fee_error,nav_error,contribution_error,episode_error,log_error)<1e-9
        ledger[name]={'fee_error':fee_error,'nav_error':nav_error,'coin_contribution_error':contribution_error,
                      'episode_error':episode_error,'trade_log_fee_error':log_error}
    code_paths=['scripts/run_signal_control.py','scripts/run_signal_control_extension.py','scripts/run_signal_core_ablation.py',
                'scripts/analyze_signal_control.py','scripts/build_signal_control_report.py','scripts/run_signal_control.ps1',
                'scripts/audit_signal_control.py','src/crypto_timing/alpha_control_policy.py','src/crypto_timing/alpha_account.py',
                'tests/test_control_position.py']
    final={'passed':True,'result_sha256':identity,'accounts':16,'pytest_passed_tests':36,
           'source_files_unchanged':len(results['source_sha256']),'ledgers':ledger,
           'shared_html_markdown_result_identity':True,'twelve_coin_panels':True,
           'code_sha256':{p:digest(p) for p in code_paths},
           'final_protocol_sha256':digest('docs/SIGNAL_CONTROL_PROTOCOL_2026-10-05.md'),
           'center_comparison_protocol_sha256':results['lock']['protocol_sha256'],
           'protocol_note':'Final contract appends the explicitly historical-result-aware E extension; center lock retains its earlier contract identity.',
           'browser_qa':'Overview rendered; BTC selection changes the visible panel and contribution table; no layout clipping observed.',
           'maker_fill_or_liquidation_verified':False,'neural_retraining_or_reinference':False}
    save(OUT/'final_audit.json',final);save(REPORT/'final_audit.json',final)
    (REPORT/'tests.xml').write_bytes((OUT/'tests.xml').read_bytes())
    print(json.dumps({k:final[k] for k in ['passed','result_sha256','accounts','pytest_passed_tests','source_files_unchanged']},ensure_ascii=False))

if __name__=='__main__':main()
