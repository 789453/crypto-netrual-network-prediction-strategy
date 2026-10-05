"""Plot saved paths only: 2bp per fill, no funding, no model or policy rerun."""
from pathlib import Path
import hashlib
import gzip
import base64
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path('outputs/alpha_strategy')
BASE=Path('outputs/mechanism/cache_v4')
OUT=Path('reports/2bp_curve_gallery_2026-10-04')
FEE=.0002
BLUE='#76baff';MINT='#6ee3c0';GOLD='#ffcc83';INK='#0c1828'
PERIODS=[('2025-03-01','开发期开始'),('2025-07-01','冻结 F1'),('2026-02-01','冻结 F2')]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pct(x):return f'{x*100:+.2f}%'
def drawdown(nav):return float(np.min(nav/np.maximum.accumulate(nav)-1))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reconstruct(a,price):
    """Keep executed quantities and fees exactly; remove funding from cash equity."""
    hours=a['hours'];w=a['weights'];fees=a['fees'];old=a['nav']
    total_fee=fees.sum(1);post=old[:-1]-total_fee
    # The last saved fees row includes both current entry and terminal liquidation.
    terminal_ratio=np.sum(np.abs(w[-1])*price[hours[-1]+1]/price[hours[-1]])
    post[-1]=(old[-2]-total_fee[-1])/(1-FEE*terminal_ratio)
    qty=w*post[:,None]/price[hours]
    np.testing.assert_allclose(qty*(price[hours+1]-price[hours]),a['pnl'],rtol=2e-9,atol=2e-10)
    terminal=FEE*np.abs(qty[-1])*price[hours[-1]+1]
    entry_fee=total_fee.copy();entry_fee[-1]-=terminal.sum()
    np.testing.assert_allclose(fees.sum(),FEE*a['traded_notional'].sum(),rtol=1e-10)
    gain=a['pnl']-fees
    nav=np.r_[1.,1+np.cumsum(gain.sum(1))]
    coin_nav=np.vstack((np.ones(12),1+12*np.cumsum(gain,axis=0)))
    np.testing.assert_allclose(coin_nav.mean(1),nav,atol=2e-14)
    new_post=nav[:-1]-entry_fee
    assert np.all(new_post>0),'fee-only fixed path equity exhausted'
    weight=qty*price[hours]/new_post[:,None]
    assert np.isfinite(nav).all() and abs(nav[-1]-1-gain.sum())<1e-12
    return {'hours':hours,'nav':nav,'coin_nav':coin_nav,'weight':weight,'fees':fees,
            'qty':qty,'funding_removed':float(a['funding'].sum()),
            'return':float(nav[-1]-1),'drawdown':drawdown(nav),
            'average_gross':float(np.abs(weight).sum(1).mean()),
            'fee_initial_equity':float(fees.sum())}

def buyhold(price):
    # One entry and one final close, initial cash exactly 1; no funding or rebalancing.
    nav=price/price[0]/(1+FEE);nav[-1]*=1-FEE
    return nav

def style():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,
        'figure.facecolor':INK,'axes.facecolor':INK,'savefig.facecolor':INK,
        'text.color':'#e8f3ff','axes.labelcolor':'#afc4db','xtick.color':'#afc4db',
        'ytick.color':'#afc4db','axes.edgecolor':'#3b526b','grid.color':'#30475d',
        'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'2bp-gallery'})

def savefig(name,png=False):
    p=OUT/'assets'/f'{name}.svg';plt.savefig(p,bbox_inches='tight',metadata={'Date':'2026-10-04'})
    if png:plt.savefig(p.with_suffix('.png'),dpi=160,bbox_inches='tight')
    plt.close();svg=p.read_text(encoding='utf-8');return svg[svg.index('<svg'):]

def plot_item(key,j,path,clock,raw,cal,benchmark,matched,execution,dates,symbols):
    overall=j is None;label='十二合约总账户' if overall else symbols[j]
    h=path['hours'];nav_time=np.r_[execution[h[0]],execution[h+1]]
    nav=path['nav'] if overall else path['coin_nav'][:,j]
    bh=benchmark.mean(1) if overall else benchmark[:,j]
    fair=matched.mean(1) if overall else matched[:,j]
    weights=path['weight'];pos_time=np.r_[execution[h],execution[h[-1]+1]]
    raw_line=raw.mean(axis=1) if overall else raw[:,j]
    cal_line=cal.mean(axis=1) if overall else cal[:,j]
    fig,axes=plt.subplots(3,1,figsize=(14.8,8.9),sharex=True,
                          gridspec_kw={'height_ratios':[1.35,1.,1.]})
    axes[0].plot(clock,bh,color=GOLD,lw=1.15,ls='--',alpha=.8,label='Buy & hold · 25-01起 · 1倍')
    axes[0].plot(execution[h[0]:],fair,color=MINT,lw=1.35,label='Buy & hold · 策略同期 · 1倍')
    axes[0].plot(nav_time,nav,color=BLUE,lw=1.65,label='2bp费用重述 · 总净值' if overall else '2bp费用重述 · 等份分币子账本')
    axes[0].axhline(1,color='#6a839c',lw=.65,alpha=.7)
    axes[0].set_ylabel('净值 / 初始资金');axes[0].legend(loc='upper left',fontsize=8,ncol=3)
    axes[0].set_title(label+'  |  '+('风险约束主方案' if key=='primary' else '近满仓排名对照 · 2倍'),loc='left',fontsize=14,pad=25)
    if overall:
        for yy,c,n in [(np.abs(weights).sum(1),BLUE,'实际毛敞口'),(weights.sum(1),MINT,'实际净敞口')]:
            axes[1].step(pos_time,np.r_[yy,0],where='post',color=c,lw=1.1,label=n)
        axes[1].set_ylabel('持仓名义 / 账户权益');axes[1].legend(loc='upper left',fontsize=8,ncol=2)
    else:
        ww=weights[:,j]*100
        axes[1].step(pos_time,np.r_[ww,0],where='post',color=BLUE,lw=.9,label='实际有符号权重：正多／负空')
        axes[1].set_ylabel('持仓 / 总账户权益 · %');axes[1].legend(loc='upper left',fontsize=8)
    axes[1].axhline(0,color='#7289a2',lw=.7)
    axes[2].plot(dates,raw_line*1e4,color=BLUE,lw=.8,alpha=.65,label='原v5原生4h预测 · bp'+(' · 十二币均值' if overall else ''))
    axes[2].plot(dates,cal_line*1e4,color=GOLD,lw=1.,label='成熟折外12h投影 · bp'+(' · 十二币均值' if overall else ''))
    axes[2].axhline(0,color='#7289a2',lw=.7);axes[2].set_ylabel('预期简单收益 · bp');axes[2].legend(loc='upper left',fontsize=8,ncol=2)
    ticks=np.r_[np.datetime64('2025-01-01'),np.arange(np.datetime64('2025-03','M'),np.datetime64('2026-10','M'),3).astype('datetime64[D]')]
    for ax in axes:
        ax.axvspan(np.datetime64('2025-01-01'),np.datetime64('2025-07-01'),color='#93a6bd',alpha=.07)
        for cut,period in PERIODS:ax.axvline(np.datetime64(cut),color='#9ab0c7',ls=(0,(4,4)),lw=.8,alpha=.8)
        ax.grid(alpha=.32);ax.set_xlim(np.datetime64('2025-01-01'),execution[-1]);ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'));ax.tick_params(axis='x',labelbottom=True)
    for cut,period in PERIODS:axes[0].text(np.datetime64(cut),1.015,period,transform=axes[0].get_xaxis_transform(),fontsize=8,color='#b4c9de',ha='center')
    axes[1].text(.018,.06,'25-01—06无已存持仓账本；空白不代表空仓',transform=axes[1].transAxes,fontsize=8,color='#afc4db')
    axes[2].set_xlabel('UTC 日期 · 季度 YY-MM · 虚线为区间切换')
    fig.tight_layout(h_pad=1.8)
    name=f'{key}_{"portfolio" if overall else symbols[j]}'
    svg=savefig(name,png=overall)
    stats={'return':float(nav[-1]-1),'drawdown':drawdown(nav),'benchmark_same_period_return':float(fair[-1]-1),'benchmark_jan_return':float(bh[-1]-1),
           'mean_abs_weight':path['average_gross'] if overall else float(np.abs(weights[:,j]).mean()),
           'fee_initial_equity':path['fee_initial_equity'] if overall else float(path['fees'][:,j].sum()*12)}
    return svg,stats

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True);style()
    meta=read(ROOT/'market_manifest.json');symbols=meta['symbols'];dates=np.load(BASE/'decision_time.npy');execution=np.load(ROOT/'execution.npy');price=np.load(ROOT/'price.npy')
    raw=np.load(ROOT/'v5_anchor_return4.npy');start=np.searchsorted(dates,np.datetime64('2025-01-01'));clock=execution[start:]
    cal=np.full_like(raw,np.nan);fits=read(ROOT/'calibration.json');used=[ROOT/'price.npy',ROOT/'execution.npy',BASE/'decision_time.npy',ROOT/'v5_anchor_return4.npy',ROOT/'calibration.json',ROOT/'market_manifest.json']
    for month in np.arange(np.datetime64('2025-01','M'),np.datetime64('2026-10','M')):
        fpath=ROOT/'monthly'/str(month)/'predictions.npz';f=np.load(fpath);hh=f['hours'];record=fits.get(f'{month}/Frozen_v5')
        if record is None:continue  # January has a raw stream, not a mature calibrated 12h forecast.
        risk=f['risk'][:,:,3];r=record['fit'][3];p=raw[hh];cv=risk.mean(1);common=p.mean(1)
        cal[hh]=(r['common_slope']*common/cv+r['common_intercept'])[:,None]*cv[:,None]+r['relative_slope']*(p-common[:,None]);used.append(fpath)
    saved=np.load(ROOT/'unified_predictions.npz');np.testing.assert_allclose(cal[saved['hours']],saved['Frozen_v5_calibrated'][:,:,3],rtol=1e-5,atol=3e-8);used.append(ROOT/'unified_predictions.npz')
    paths={}
    for key,file in [('primary','account_Frozen_v5_2bp.npz'),('rank','primary_rank_2.0_2bp.npz')]:
        path=ROOT/file;paths[key]=reconstruct(np.load(path),price);used.append(path)
    h=paths['primary']['hours'];assert np.array_equal(h,paths['rank']['hours']);benchmark=buyhold(price[start:].copy());matched=buyhold(price[h[0]:].copy())
    panels=[];stats={};packed={};source_before={str(p):digest(p) for p in used}
    for j in [None]+list(range(12)):
        asset='portfolio' if j is None else symbols[j];title='总账户' if j is None else symbols[j];parts=[];stats[asset]={}
        for key,path in paths.items():
            svg,s=plot_item(key,j,path,clock,raw[start:],cal[start:],benchmark,matched,execution,dates[start:],symbols);stats[asset][key]=s
            packed[f'{key}/{asset}']=base64.b64encode(gzip.compress(svg.encode('utf-8'),compresslevel=9,mtime=0)).decode('ascii')
        panels.append(f'<section id="{asset}" data-asset="{asset}"><div class="section-head"><div><span class="eyebrow">'+('PORTFOLIO' if j is None else f'CONTRACT {j+1:02d} / 12')+f'</span><h2>{title}</h2></div><div class="metric" data-metric="{asset}"></div></div><figure><div class="chart-frame"><img data-chart-asset="{asset}" alt="{title}：净值、持仓与信号三层图" width="2356" height="1400"></div><a class="download" data-download="{asset}" href="assets/primary_{asset}.svg" download>下载此图 SVG ↗</a></figure></section>')
        print('Drawn',asset,flush=True)
    np.savez_compressed(OUT/'derived_paths.npz',symbols=np.asarray(symbols),signal_dates=dates[start:],raw_return4=raw[start:],calibrated_return12=cal[start:],execution=execution,hours=h,**{f'{k}_{field}':v[field] for k,v in paths.items() for field in ('nav','coin_nav','weight','qty')})
    evidence={'cost_per_side':FEE,'funding_in_plotted_equity':False,'policy_rerun':False,'training_or_inference_rerun':False,'signals_start':'2025-01-01','calibrated_projection_start':'2025-02-01','saved_holdings_start':str(execution[h[0]]),'last_price':str(execution[-1]),'statistics':stats,'funding_removed_initial_equity':{k:v['funding_removed'] for k,v in paths.items()},'sources_sha256':source_before,'semantics':'Keep saved 2bp actual quantities and fees; price PnL minus trading fees only. Coin ledger normalized to initial 1/12 capital; its arithmetic mean exactly equals total. This is fixed-path cost restatement, not reselected/rebalanced no-funding strategy.'}
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    nav=''.join(f'<a href="#{s}">{"总账户" if s=="portfolio" else s.replace("USDT", "")}</a>' for s in ['portfolio']+symbols)
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth;color-scheme:dark}body{margin:0;background:#08121f;color:#ecf4ff;font:15px/1.85 "Microsoft YaHei","Segoe UI",sans-serif}a{color:#91c7ff;text-decoration:none}a:hover{text-decoration:underline}.shell{max-width:1690px;margin:auto;display:grid;grid-template-columns:210px minmax(0,1fr)}aside{position:sticky;top:0;height:100vh;padding:30px 22px;border-right:1px solid #2b4057}.brand{font-size:18px;letter-spacing:2px}.aside-note{font-size:12px;color:#b7c9df;margin:12px 0 24px}nav a{display:block;padding:7px 10px;border-radius:6px}nav a:hover{background:#19314c}main{min-width:0;padding:42px 42px 65px}.eyebrow{font-size:11px;letter-spacing:2px;color:#91c7ff}h1{font-size:39px;line-height:1.45;margin:10px 0 14px}h2{margin:5px 0 10px;font-size:26px}p{color:#b9cde3}.subtitle{font-size:17px;max-width:1020px}.note{border-left:3px solid #ffcc83;background:#24293b;padding:17px 21px;color:#d7e1ed;margin:22px 0;border-radius:0 9px 9px 0}.controls{display:flex;flex-wrap:wrap;gap:18px;background:#102033;border:1px solid #2b4057;border-radius:10px;padding:18px;margin:26px 0}.controls label{flex:1;min-width:220px;font-size:12px;color:#b9cde3}select{display:block;width:100%;padding:10px 12px;color:#e8f3ff;background:#18304b;border:1px solid #44607c;border-radius:7px;font:14px "Microsoft YaHei";margin-top:8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:#102033;border:1px solid #2b4057;padding:18px 20px;border-radius:10px}.value{font-size:27px}.label{font-size:11px;color:#b9cde3}.periods{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}.periods div{border-top:2px solid #587694;padding:10px 0;color:#aac2dc;font-size:12px}.periods strong{color:#e8f3ff;display:block;font-weight:500}section{margin-top:48px;scroll-margin-top:20px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}.metric{font-size:12px;color:#b9cde3;text-align:right;margin-bottom:12px}figure{margin:0;background:#0c1828;border:1px solid #2b4057;border-radius:12px;padding:12px 15px}figure svg{width:100%;height:auto;display:block}.download{font-size:12px;display:block;text-align:right;padding:8px 8px 0}[hidden]{display:none!important}footer{border-top:1px solid #2b4057;padding-top:25px;margin-top:55px;color:#afc4db;font-size:12px}.links{display:flex;gap:25px;flex-wrap:wrap}@media(max-width:1100px){main{padding:28px 24px}.cards{grid-template-columns:1fr 1fr}h1{font-size:33px}}@media(max-width:760px){.shell{display:block}aside{position:relative;height:auto;border-bottom:1px solid #2b4057;border-right:0;padding:18px}nav{display:flex;overflow:auto}nav a{white-space:nowrap}main{padding:25px 16px}.periods{grid-template-columns:1fr 1fr}.section-head{display:block}.metric{text-align:left}h1{font-size:27px}figure{padding:8px}.controls label{min-width:100%}}'''
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2bp 纯手续费 · 总账户与十二合约曲线</title><style>'''+css+'''</style></head><body><div class="shell"><aside><div class="brand">CRYPTO / CURVES</div><div class="aside-note">2025.01 → 2026.09<br>2bp PER FILL · NO FUNDING</div><nav>'''+nav+'''</nav></aside><main><span class="eyebrow">SAVED PATHS · NO DATA / MODEL RERUN</span><h1>总净值、真实持仓与预测信号。<br>十二个合约，逐一展开。</h1><p class="subtitle">单次成交单边2bp；图中净值只保留价格盈亏与手续费。统一季度日期标签，虚线划分开发期、冻结F1与F2。两套已存持仓路径可以直接切换。</p><div class="controls"><label>已存策略路径<select id="policy" aria-label="已存策略路径"><option value="primary">风险约束主方案 · v5＋12h投影</option><option value="rank">近满仓排名对照 · 2倍额度</option></select></label><label>图组筛选<select id="asset" aria-label="图组筛选"><option value="all">总账户＋全部12个合约</option><option value="portfolio">只看总账户</option>'''+''.join(f'<option value="{s}">{s}</option>' for s in symbols)+'''</select></label></div><div class="cards" id="cards"></div><div class="note"><strong>保存范围：</strong>信号自25-01开始，成熟12h投影自25-02开始；完整持仓与策略账本自25-07开始。25-01—06的策略净值和持仓留空，不填为1或0。<br><strong>费用口径：</strong>沿用已保存2bp账户的实际成交与份额，净值重计为「1＋累计价格PnL−累计手续费」，剔除资金费，未重新生成交易。资金费原先会影响持仓资金路径，因此这里是固定路径的费用重述，不能冒充从头重跑的无资金费策略。</div><div class="periods"><div><strong>25-01—02</strong>原始时间外信号／折外预热</div><div><strong>25-03—06</strong>开发信号；无已存持仓账本</div><div><strong>25-07—26-01</strong>连续冻结历史 F1</div><div><strong>26-02—09</strong>连续冻结历史 F2</div></div><p>总账户的buy-and-hold为十二币等份初始资金、1倍买入持有；逐币为相应合约1倍持有。金色虚线从25-01开始，绿色曲线从策略同期25-07开始，均只计开平单边2bp。分币策略曲线按初始资金1/12计账，十二条的算术平均严格等于总净值；它们不是重新优化的十二个独立账户。持仓线为实际份额在费用重述权益下的名义权重，正多负空；总信号线为十二币预测均值。</p>'''+''.join(panels)+'''<footer>来源：原v5时间外预测、月度已存风险与校准系数、已存2bp持仓账本及下一分钟价格缓存。没有读取或重新构建原始行情、训练网络、重新推理或重选交易。<br>4h原生信号与12h折外投影是不同预测对象；排名对照每12h复核，可持续同方向持有，不能把复核期限写成实际持仓寿命。基准与策略杠杆不同，净值对比不等同于风险匹配alpha检验。<div class="links"><a href="evidence.json">计算口径与源文件核验 ↗</a><a href="derived_paths.npz">费用重述与持仓数组 ↗</a><a href="../alpha_strategy_2026-10-04/index.html">原完整研究报告 ↗</a></div></footer></main></div><script>'''
    payload={'stats':stats,'names':{'primary':'风险约束主方案','rank':'近满仓排名对照'},'fundingRemoved':evidence['funding_removed_initial_equity'],'charts':packed}
    js='const D='+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+''';const P=document.getElementById('policy'),A=document.getElementById('asset');const pct=x=>(x>=0?'+':'')+(100*x).toFixed(2)+'%';const CACHE=new Map();async function render(img){const asset=img.dataset.chartAsset,k=P.value,id=k+'/'+asset;img.dataset.policy=k;try{if(!CACHE.has(id)){const bytes=Uint8Array.from(atob(D.charts[id]),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));const svg=await new Response(stream).text();CACHE.set(id,URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'})));}if(img.dataset.policy!==k)return;img.src=CACHE.get(id);const a=document.querySelector('[data-download="'+asset+'"]');a.href=img.src;a.download=k+'_'+asset+'.svg';}catch(e){img.src='assets/'+k+'_'+asset+'.svg';console.error('Chart decode:',e.message)}}function show(){const k=P.value,s=D.stats.portfolio[k];document.querySelectorAll('section[data-asset]').forEach(e=>e.hidden=A.value!=='all'&&e.dataset.asset!==A.value);document.getElementById('cards').innerHTML=[[pct(s.return),'2bp费用重述 · 累计收益'],[pct(s.drawdown),'小时边界最大回撤'],[s.mean_abs_weight.toFixed(3),'实际平均毛敞口'],[pct(s.benchmark_same_period_return),'同期1倍buy-and-hold收益']].map(v=>'<div class="card"><div class="value">'+v[0]+'</div><div class="label">'+v[1]+'</div></div>').join('');document.querySelectorAll('[data-metric]').forEach(e=>{const q=D.stats[e.dataset.metric][k];e.textContent='费用重述收益 '+pct(q.return)+'　|　同期 B&H '+pct(q.benchmark_same_period_return)+'　|　记账回撤 '+pct(q.drawdown)});document.querySelectorAll('img[data-chart-asset]').forEach(img=>{if(!img.closest('section').hidden&&(img.dataset.seen==='1'||A.value!=='all'))render(img)});}P.addEventListener('change',show);A.addEventListener('change',show);document.querySelectorAll('nav a').forEach(e=>e.addEventListener('click',()=>{A.value='all';show()}));const observer=new IntersectionObserver(items=>items.forEach(it=>{if(it.isIntersecting){it.target.dataset.seen='1';render(it.target)}}),{rootMargin:'600px'});document.querySelectorAll('img[data-chart-asset]').forEach(img=>observer.observe(img));show();'''
    page=page.replace('figure svg{width:100%;height:auto;display:block}','.chart-frame{aspect-ratio:2356/1400}.chart-frame img{width:100%;height:100%;display:block;object-fit:contain}')
    (OUT/'index.html').write_text(page+js+'</script></body></html>',encoding='utf-8')
    (OUT/'README.md').write_text('# 2bp纯手续费曲线图集\n\n[打开HTML](index.html)。复用保存结果绘图，无训练、推理或交易规则重跑。2025年1月起显示信号和基准，保存策略账本从7月起；总账户与十二合约各含净值、持仓、信号，支持主方案/近满仓对照切换。净值仅价格PnL减已存2bp手续费；具体语义与校验见[evidence.json](evidence.json)。\n',encoding='utf-8')
    assert all(digest(p)==v for p,v in source_before.items()),'source outputs modified'
    manifest={'source_files_unchanged':True,'account_reconstruction_verified':True,'coin_ledgers_average_reconciles':True,'saved_calibration_stream_verified':True,'charts':26,'panels_per_path':13,'cost_per_side':FEE,'sha256':{str(p):digest(p) for p in [Path(__file__),*[p for p in OUT.glob('*.*') if p.name!='manifest.json'],*(OUT/'assets').glob('*')]}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Completed',OUT/'index.html',flush=True)

if __name__=='__main__':main()
