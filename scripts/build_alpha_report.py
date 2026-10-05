"""Scientific figures, a complete Chinese report and a self-contained interactive page."""
from pathlib import Path
import json,html,hashlib,shutil
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from crypto_timing.alpha_data import ROOT,HORIZONS,save_json

OUT=Path('reports/alpha_strategy_2026-10-04');DOC=Path('docs/ALPHA_STRATEGY_RESEARCH_REPORT_2026-10-04.md')
NAMES={'NN_joint':'联合状态神经网络','NN_direct':'单时刻神经读出','Ridge':'Ridge','LightGBM':'LightGBM','Trend':'趋势','Reversal':'反转','Frozen_v5':'原v5锚点／期限投影','Frozen_mix':'开发冻结混合'}
COLORS=['#75b9ff','#6cdeba','#ffc97c','#bca8ff','#fb8ba0','#96c3cf','#ddd0ae','#a3d390']
def pct(x):return f'{100*x:+.2f}%'
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def table(headers,rows):return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def style():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'figure.facecolor':'#102034','axes.facecolor':'#102034','savefig.facecolor':'#102034','text.color':'#edf5ff','axes.labelcolor':'#b8cbe1','xtick.color':'#b8cbe1','ytick.color':'#b8cbe1','axes.edgecolor':'#425a74','grid.color':'#344b63','axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'crypto-alpha-strategy'})
def savefig(name):
    path=OUT/'assets'/f'{name}.svg';plt.savefig(path,bbox_inches='tight',metadata={'Date':'2026-10-04'})
    if name in ('primary','primary_rank','equity','bridge','leverage'):plt.savefig(OUT/'assets'/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close();text=path.read_text(encoding='utf-8');return text[text.index('<svg'):]
def plot_nav(ax,arr,label,color):
    dates=arr['dates'];ax.plot(dates,arr['nav'][1:],label=label,color=color,lw=1.35);ax.grid(alpha=.3);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.tick_params(axis='x',rotation=25)

def figures(r):
    style();svgs={};models=list(r['models']);symbols=r['market']['symbols']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for label,color in zip(('gross','2bp','4bp'),COLORS):plot_nav(axes[0],np.load(ROOT/f'account_NN_joint_{label}.npz'),label,color)
    for name,color in zip(('NN_joint','NN_direct','LightGBM','Ridge','Frozen_mix'),COLORS):plot_nav(axes[1],np.load(ROOT/f'account_{name}_4bp.npz'),NAMES[name],color)
    for ax,title in zip(axes,('联合状态：同一冻结政策，费用分别重算','相同数据与账本：各模型同预算开发政策')):ax.set_title(title);ax.set_ylabel('账户净值');ax.legend(fontsize=8);ax.axvline(np.datetime64('2026-02-01'),color='#9aacbf',ls=':',lw=.8)
    fig.tight_layout();svgs['equity']=savefig('equity')
    fig,axes=plt.subplots(1,2,figsize=(13,4.))
    for ax,names in zip(axes,(('NN_joint','NN_direct','LightGBM','Ridge'),('NN_joint','Frozen_mix','Trend','Reversal'))):
        for name,color in zip(names,COLORS):
            a=np.load(ROOT/f'account_{name}_4bp.npz');dd=a['nav']/np.maximum.accumulate(a['nav'])-1;ax.plot(a['dates'],dd[1:]*100,color=color,label=NAMES[name])
        ax.grid(alpha=.3);ax.set_ylabel('小时边界回撤 / %');ax.legend(fontsize=8);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();svgs['drawdown']=savefig('drawdown')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));yy=np.arange(len(models)-1)
    for ax,section in zip(axes,('development','frozen')):
        names=models[:-1]
        for j,h in enumerate(HORIZONS):ax.barh(yy+(j-1.5)*.18,[r['information'][name][section]['raw'][str(h)]['mean_TS_IC_G'] for name in names],height=.18,label=f'{h}h',color=COLORS[j])
        ax.set_yticks(yy,[NAMES[x] for x in names],fontsize=8);ax.axvline(0,color='#ddd',lw=.7);ax.set_title('开发期原收益IC' if section=='development' else '冻结历史原收益IC');ax.set_xlabel('十二币时间序列IC均值');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.3)
    fig.tight_layout();svgs['information']=savefig('information')
    fig,axes=plt.subplots(2,1,figsize=(13,7.2))
    for ax,name in zip(axes,('NN_joint','LightGBM')):
        monthly=r['information'][name]['frozen']['raw']['4']['monthly'];months=list(monthly);matrix=np.array([[monthly[m][s]['IC_G'] for m in months] for s in symbols]);im=ax.imshow(matrix,cmap='RdBu_r',vmin=-.15,vmax=.15,aspect='auto');ax.set_xticks(range(len(months)),months,rotation=30,fontsize=8);ax.set_yticks(range(12),symbols,fontsize=8);ax.set_title(NAMES[name]+' · 原收益4h TS IC（全部十二币）');fig.colorbar(im,ax=ax,label='IC')
    fig.tight_layout();svgs['matrix']=savefig('matrix')
    fig,axes=plt.subplots(2,2,figsize=(13,7.))
    for ax,h in zip(axes.flat,HORIZONS):
        bins=r['quantiles']['NN_joint'][str(h)];ax.bar(np.arange(10),[x['directional_4bp_roundtrip_mean']*1e4 for x in bins],color=[COLORS[3]]*5+[COLORS[1]]*5);ax.axhline(0,color='#aaa',lw=.8);ax.set_title(f'{h}h：因果折外分位，低五组空／高五组多');ax.set_xlabel('分位（边界在当月前冻结）');ax.set_ylabel('扣往返8bp与资金费的均值 / bp');ax.grid(axis='y',alpha=.3)
    fig.tight_layout();svgs['quantiles']=savefig('quantiles')
    a=np.load(ROOT/'account_NN_joint_4bp.npz');fig,axes=plt.subplots(2,1,figsize=(13,5.7));w=a['weights'];axes[0].plot(a['dates'],abs(w).sum(1),color=COLORS[0],label='毛敞口');axes[0].plot(a['dates'],w.sum(1),color=COLORS[2],label='净敞口');axes[0].axhline(2,color='#ccc',ls=':',lw=.8);axes[0].legend(fontsize=8);axes[0].set_ylabel('名义敞口 / 权益');im=axes[1].imshow(w.T,aspect='auto',cmap='RdBu_r',vmin=-.75,vmax=.75);axes[1].set_yticks(range(12),symbols,fontsize=8);ix=np.linspace(0,len(w)-1,9).astype(int);axes[1].set_xticks(ix,[str(a['dates'][k].astype('datetime64[D]')) for k in ix],rotation=25);fig.colorbar(im,ax=axes[1],label='逐币权重');axes[0].set_title('固定份额持有、到期退出及风险调整的实际仓位路径');fig.tight_layout();svgs['positions']=savefig('positions')
    fig,axes=plt.subplots(1,2,figsize=(13,4.4));eps=load('episodes_NN_joint.json');boxes=[[e['holding_hours'] for e in eps if e['symbol_id']==j] or [0] for j in range(12)];axes[0].boxplot(boxes,tick_labels=[s.replace('USDT','') for s in symbols],patch_artist=True,boxprops={'facecolor':'#75b9ff'},medianprops={'color':'#ffc97c'});axes[0].set_ylabel('实际持仓小时（无交易以0占位）');axes[0].tick_params(axis='x',rotation=35);axes[0].grid(axis='y',alpha=.3)
    coins=r['models']['NN_joint']['attribution']['by_coin'];x=np.arange(12);axes[1].bar(x,[coins[s]['price_pnl']*100 for s in symbols],label='价格PnL',color=COLORS[0]);axes[1].bar(x,[coins[s]['funding']*100 for s in symbols],label='资金费',color=COLORS[1]);axes[1].bar(x,[-coins[s]['fees']*100 for s in symbols],label='交易费',color=COLORS[2]);axes[1].set_xticks(x,[s.replace('USDT','') for s in symbols],rotation=35);axes[1].set_ylabel('初始权益贡献 / %');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.3);fig.tight_layout();svgs['holdings']=savefig('holdings')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for cap,color in zip((1.,2.,3.,5.),COLORS):plot_nav(axes[0],np.load(ROOT/f'leverage_{cap}_4bp.npz'),f'毛上限{cap:g}',color)
    caps=list(r['leverage']);x=np.arange(len(caps));axes[1].bar(x-.18,[r['leverage'][k]['4bp']['average_gross'] for k in caps],.36,label='实际平均毛敞口',color=COLORS[0]);axes[1].bar(x+.18,[r['leverage'][k]['4bp']['average_margin_fraction_at_5x'] for k in caps],.36,label='5倍合约保证金占比',color=COLORS[2]);axes[1].set_xticks(x,caps);axes[1].set_xlabel('账户毛杠杆上限');axes[1].legend(fontsize=8);axes[0].legend(fontsize=8);axes[0].set_title('同一政策重新走完整账户路径');axes[1].set_title('杠杆额度与实际使用量分开');fig.tight_layout();svgs['leverage']=savefig('leverage')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for name,color in zip(('v3_saved_score','v5_same_policy','new_joint_same_policy'),COLORS):plot_nav(axes[0],np.load(ROOT/f'bridge_{name}.npz'),name,color)
    a=np.load(ROOT/'v3_native_curve.npz')
    for phase,color in enumerate(COLORS[:4]):axes[1].plot(a[f'dates_{phase}'],a[f'nav_{phase}'],label=f'原v3相位{phase}',color=color)
    axes[0].set_title('共同连续日期：统一1m、4h政策、4bp账本');axes[1].set_title('原v3原生合同：旧5m、6bp、四独立相位');
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.3);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();svgs['bridge']=savefig('bridge')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,month in zip(axes,('2025-01','2025-07','2026-02')):
        records=load(f'monthly/{month}/summary.json')['records']['NN_joint'][0];curves=records['curves']
        for key,color,label in zip(('train_eval','internal','external_observe_only'),COLORS,('固定训练eval','内部时间验证','外部仅观察')):
            ax.plot([x['epoch'] for x in curves],[100*(1-x[key]['mse']/x[key]['zero_mse']) for x in curves],marker='o',color=color,label=label)
        ax.axvline(records['chosen_epochs'],color='#ffc97c',ls=':',lw=.8);ax.set_title(month+' 联合头／种子20261004');ax.set_xlabel('训练轮次');ax.set_ylabel('各自零预测MSE skill / %');ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();svgs['training']=savefig('training')
    fig,ax=plt.subplots(figsize=(12,4.))
    names=list(r['models']);x=np.arange(len(names))
    for j,(fee,color) in enumerate(zip(('gross','2bp','4bp'),COLORS)):ax.bar(x+(j-1)*.24,[r['models'][name]['cost_scenarios'][fee]['return']*100 for name in names],.24,label=fee,color=color)
    ax.set_xticks(x,[NAMES[n] for n in names],rotation=20,fontsize=9);ax.axhline(0,color='#ddd',lw=.8);ax.set_ylabel('冻结历史累计账户收益 / %');ax.legend();ax.grid(axis='y',alpha=.3);fig.tight_layout();svgs['ranking']=savefig('ranking')
    return svgs

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True);r=load('results.json');runtime=load('runtime.json');policies=load('policies.json');symbols=r['market']['symbols'];svgs=figures(r)
    tests=ET.parse(ROOT/'tests.xml').getroot();count=len(tests.findall('.//testcase'));assert not tests.findall('.//failure') and not tests.findall('.//error')
    system=load('primary_system_results.json');primary=system['lock']['model'];primary_model=r['models'][primary];ps=primary_model['cost_scenarios']['4bp'];pp=primary_model['own_policy'];pci=system['net_return_interval']['calendar_7day_block_95pct'];pinc=system['paired_increment']['calendar_7day_block_95pct']
    primary_conclusion='主神经策略已按开发期选择冻结；当前历史证据仍不能证明稳定超额，不能以杠杆或高利用率替代信息检验。'
    if ps['return']>0 and pci[0]>0 and pinc[0]>0:primary_conclusion='主神经策略在冻结历史中保留正净收益及基线增量区间；新增市场、真实maker成交与保证金可行性仍待验证。'
    primary_rows=[];primary_coins=[];primary_months=[];rank_cap_rows=[];primary_leave=[];rank_coins=[];primary_ic_rows=[]
    for window in ('development','F1','F2','frozen'):
        for score in ('raw','calibrated'):
            q=r['information'][primary][window][score][str(pp['horizon'])];primary_ic_rows.append([window,score,f"{q['mean_TS_IC_G']:+.4f}",f"{q['mean_TS_RankIC_G']:+.4f}",f"{q['mean_hourly_CS_IC_G']:+.4f}",f"{q['common_IC_G']:+.4f}",f"{q['relative_IC_G']:+.4f}"])
    for cap,v in system['leverage'].items():
        q=v['4bp'];primary_rows.append([cap,pct(v['gross']['return']),pct(v['2bp']['return']),pct(q['return']),pct(q['max_drawdown']),pct(q['minute_max_drawdown']),f"{q['average_gross']:.3f}",f"{q['gross_utilization']*100:.1f}%",f"{q['average_margin_fraction_at_5x']*100:.1f}%"])
    pic=r['information'][primary]['frozen']['raw'][str(pp['horizon'])]['by_coin']
    for symbol in symbols:
        a=pic[symbol];b=primary_model['attribution']['by_coin'][symbol];primary_coins.append([symbol,f"{a['IC_G']:+.4f}",f"{a['RankIC_G']:+.4f}",f"{a['monthly_TS_ICIR']:+.3f}",pct(b['net_pnl_contribution']),pct(b['funding']),f"{b['mean_abs_weight']:.3f}",b['episodes'],f"{b['holding_mean']:.2f}",f"{b['win_rate']*100:.1f}%",pct(b['long_price_pnl']),pct(b['short_price_pnl'])])
        q=system['leave_one_coin_out'][symbol];primary_leave.append([symbol,pct(q['return_4bp']),pct(q['drawdown'])])
        b=system['full_rank_attribution']['by_coin'][symbol];rank_coins.append([symbol,pct(b['net_pnl_contribution']),pct(b['funding']),pct(-b['fees']),f"{b['mean_abs_weight']:.3f}",b['episodes'],f"{b['holding_mean']:.2f}",f"{b['win_rate']*100:.1f}%"])
    for month,a in primary_model['attribution']['monthly'].items():primary_months.append([month,pct(a['account_return']),f"{a['average_gross']:.3f}",*[pct(a['by_coin'][s]) for s in symbols]])
    for cap,v in system['full_rank_leverage'].items():
        q=v['4bp'];rank_cap_rows.append([cap,pct(v['gross']['return']),pct(v['2bp']['return']),pct(q['return']),pct(q['minute_max_drawdown']),f"{q['average_gross']:.3f}",f"{q['average_margin_fraction_at_5x']*100:.1f}%",f"{q['holding_mean']:.2f}",q['minute_gross_limit_breach_hours']])
    style();fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for label,color in zip(('gross','2bp','4bp'),COLORS):plot_nav(axes[0],np.load(ROOT/f'account_{primary}_{label}.npz'),label,color)
    for cap,color in zip((1.,2.,3.,5.),COLORS):plot_nav(axes[1],np.load(ROOT/f'primary_leverage_{cap}_4bp.npz'),f'毛上限{cap:g}',color)
    axes[0].set_title(NAMES[primary]+' · 主方案费用情景');axes[1].set_title('主方案1/2/3/5毛上限 · 4bp完整路径')
    for ax in axes:ax.legend(fontsize=8);ax.set_ylabel('净值');ax.axvline(np.datetime64('2026-02-01'),color='#a5b5c7',ls=':',lw=.8)
    fig.tight_layout();svgs['primary']=savefig('primary')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for label,color in zip(('gross','2bp','4bp'),COLORS):plot_nav(axes[0],np.load(ROOT/f'primary_rank_2.0_{label}.npz'),label,color)
    for cap,color in zip((1.,2.,3.,5.),COLORS):plot_nav(axes[1],np.load(ROOT/f'primary_rank_{cap}_4bp.npz'),f'毛上限{cap:g}',color)
    for ax,title in zip(axes,('近满仓固定排名对照 · 2倍额度三费用','近满仓对照 · 四额度的4bp净值')):ax.set_title(title);ax.legend(fontsize=8);ax.set_ylabel('净值')
    fig.tight_layout();svgs['primary_rank']=savefig('primary_rank')
    fig,axes=plt.subplots(2,1,figsize=(13,7.))
    a=np.load(ROOT/f'account_{primary}_4bp.npz');axes[0].plot(a['dates'],np.abs(a['weights']).sum(1),label='主方案实际毛敞口',color=COLORS[0]);axes[0].plot(a['dates'],a['weights'].sum(1),label='净敞口',color=COLORS[2]);axes[0].legend(fontsize=8);axes[0].grid(alpha=.3);axes[0].set_title('主方案实际参与率与份额持仓路径')
    monthly=r['information'][primary]['frozen']['raw'][str(pp['horizon'])]['monthly'];mm=list(monthly);matrix=np.array([[monthly[m][s]['IC_G'] for m in mm] for s in symbols]);im=axes[1].imshow(matrix,cmap='RdBu_r',vmin=-.15,vmax=.15,aspect='auto');axes[1].set_xticks(range(len(mm)),mm,rotation=25,fontsize=8);axes[1].set_yticks(range(12),symbols,fontsize=8);axes[1].set_title(f'主方案{pp["horizon"]}h · 十二币各月TS IC');fig.colorbar(im,ax=axes[1],label='IC');fig.tight_layout();svgs['primary_information']=savefig('primary_information')
    own=r['models']['NN_joint'];s=own['cost_scenarios']['4bp'];p=own['own_policy'];models=list(r['models']);rankrows=[];sharedrows=[]
    for name in models:
        m=r['models'][name];ss=m['cost_scenarios']['4bp'];rankrows.append([NAMES[name],m['own_policy']['horizon'],m['own_policy']['multiplier'],m['own_policy']['sizing'],pct(m['cost_scenarios']['gross']['return']),pct(m['cost_scenarios']['2bp']['return']),pct(ss['return']),pct(ss['max_drawdown']),f"{ss['daily_sharpe']:.2f}",f"{ss['average_gross']:.3f}",f"{ss['holding_mean']:.2f}"])
        q=m['shared_policy_4bp'];sharedrows.append([NAMES[name],pct(q['return']),pct(q['max_drawdown']),f"{q['average_gross']:.3f}",f"{q['price_bps_per_turnover']:.2f}"])
    icrows=[];coinrows=[];lever=[];months=[]
    for name in models[:-1]:
        for h in HORIZONS:
            a=r['information'][name]['frozen']['raw'][str(h)];icrows.append([NAMES[name],h,f"{a['mean_TS_IC_G']:+.4f}",f"{a['mean_TS_RankIC_G']:+.4f}",f"{a['mean_hourly_CS_IC_G']:+.4f}",f"{a['common_IC_G']:+.4f}",f"{a['relative_IC_G']:+.4f}",'/'.join(f'{v:+.4f}' for v in a['TS_IC_72h_block_95pct'])])
    chosen_ic=r['information']['NN_joint']['frozen']['raw'][str(p['horizon'])]['by_coin'];coins=own['attribution']['by_coin']
    for symbol in symbols:
        a=chosen_ic[symbol];b=coins[symbol];coinrows.append([symbol,f"{a['IC_G']:+.4f}",f"{a['RankIC_G']:+.4f}",f"{a['monthly_TS_ICIR']:+.3f}",pct(b['net_pnl_contribution']),pct(b['funding']),f"{b['mean_abs_weight']:.3f}",b['episodes'],f"{b['holding_mean']:.2f}",f"{100*b['win_rate']:.1f}%",f"{b['profit_loss_ratio']:.2f}",pct(b['long_price_pnl']),pct(b['short_price_pnl'])])
    for cap,v in r['leverage'].items():
        a=v['4bp'];lever.append([cap,pct(v['gross']['return']),pct(v['2bp']['return']),pct(a['return']),pct(a['max_drawdown']),pct(a['minute_max_drawdown']),f"{a['average_gross']:.3f}",f"{a['gross_utilization']*100:.1f}%",f"{a['average_margin_fraction_at_5x']*100:.1f}%",a['minute_gross_limit_breach_hours']])
    for month,a in own['attribution']['monthly'].items():months.append([month,pct(a['account_return']),f"{a['average_gross']:.3f}",f"{a['turnover']:.2f}",*[pct(a['by_coin'][s]) for s in symbols]])
    at=own['market_attribution'];inc=r['incremental_information'];interval=own['net_return_interval']['calendar_7day_block_95pct'];best=r['paired_neural_increment']['baseline_chosen_in_development'];paired=r['paired_neural_increment']['calendar_7day_block_95pct']
    supported=s['return']>0 and interval[0]>0 and paired[0]>0
    conclusion=('冻结历史显示正净收益及相对开发期基线的增量区间；仍需新增市场前瞻验证。' if supported else '本轮已把预测器转为完整策略并完成公平比较；现有证据仍不能证明稳定的神经网络超额收益。')
    fundingrows=[[symbol,*[r['market']['funding'][symbol][key] for key in ('events','direct_funding_mark','hour_mark_open_approximation','minute_trade_price_proxy')]] for symbol in symbols]
    trace=[['共同竞技场','同一1m、下一分钟open、2/4/8/12h、连续份额账户；旧工件保留'],['信息与成本边界','全部十二币/月/期限、IC/RankIC/ICIR、共同/相对、因果分位、资金费'],['低自由度策略','24候选/模型，4bp开发，固定寿命/滞回/风险；同政策与自选政策两层'],['结构适配','冻结v5编码器；一个共同/相对联合状态GRU候选，三读出种子'],['冻结向前','月初读出/基线重训、90日成熟折外校准；连续持仓交接；仅历史回放']]
    active=load('monthly/2025-07/summary.json')['records'];native=load('v3_native_reproduction.json');latest=load('latest_label_free_prediction.json')
    md=f'''# 深度学习信息优势与统一持仓策略研究

日期：2026-10-04。主要依据：[完整分析与下一阶段设计](V5_INFORMATION_ALPHA_AND_STRATEGY_DESIGN_2026-10-04.md)，执行依据：[锁定实施合同](ALPHA_STRATEGY_IMPLEMENTATION_PROTOCOL_2026-10-04.md)。[交互可视化](../reports/alpha_strategy_2026-10-04/index.html)与[完整机器结果](../reports/alpha_strategy_2026-10-04/results.json)对应同一实际产物。

**{primary_conclusion}** 主策略为{NAMES[primary]}，冻结历史gross={pct(primary_model['cost_scenarios']['gross']['return'])}、单边2bp={pct(primary_model['cost_scenarios']['2bp']['return'])}、单边4bp={pct(ps['return'])}。主方案、近满仓对照和结构候选的完整结果分别展示。区间未修正开发选择偏差，历史已被查看，不能称前瞻确认。

## 1 对齐要求与研究取舍

{mdtable(['文档要求','实际完成'],trace)}

使用12币、23,552,640条一分钟源记录，复用既有v5数据、编码器、标准化、推理合同和成熟历史思想。21个月预测起点（2025-01—2026-09）；两种神经读出各三种子，每月先内部选择再全窗口重拟合。**冻结的是2024-11之前的单个v5编码器；本轮126个最终神经读出检查点不是126次完整编码器重训。** 四期限Ridge与受约束LightGBM每月同窗口拟合；公平基线含状态、5m/1h序列摘要及全部12个分钟支路字段的1h/3h均值和末态，共176个字段。初始140字段基线产物不进入最终排行榜。

传统模型CPU拟合、神经网络CUDA拟合/推理，全部来自universal环境：`{runtime['python']}`；GPU为{runtime['gpu']}，PyTorch {runtime['torch']}、CUDA {runtime['cuda']}。不重新复制原项目版本，也不部署资金或启动未来任务。

## 2 标签、共同信息与结构

新目标为下一分钟open入场、2/4/8/12h后open离场的简单收益G；每个目标除以当时可见4h风险×sqrt(H/4)，风险下限只由月初前成熟训练样本拟合。保留普通MSE均值解释、不截尾、不删极端行情。统一12h+1m成熟隔离仅约束拟合/校准，**不让实时预测或小时盈亏在最后12h停止**；最后研究成交可到2026-09-24 23:01，末端按已观测23:59 open清仓。

原v5锚点使用原三种子权重及原Nov24/Mar25/Jul25/Feb26切换，保留其5m入场4h目标身份；适配新入场和其他期限仅为成熟折外投影。原v5生成表示的单种子早期编码器是另一对象，两者没有混称。

新增联合读出读取最近4个整点的128维融合表示，单币投影48维；池化共同状态后GRU24，币种相对状态与共同状态拼接后GRU32。市场头输出共同原收益，币种头输出原收益增量，增量按当时有效币均值居中；总收益仍直接监督。共同监督权重0.1作为识别约束，不向推理传未来市场收益。新增可训练参数{active['NN_joint'][0]['parameters']:,}，单时刻线性四期限读出{active['NN_direct'][0]['parameters']:,}。不宣称新增读出已修复冻结编码器可能丢失的全部路径信息，也不把容量增大当成优势证明。

每月最多24个月成熟训练样本、最近60日内部验证，候选训练预算2/4轮；只按内部误差选择，再重拟合。LightGBM15叶、最小叶1000、L2=100，50/100轮内部选择；Ridge1000/10000内部选择。两方目标、截止、成本与策略搜索预算一致。21月全部训练曲线与零预测/训练均值基准保存。核心三条曲线是同一内部拟合模型的固定训练eval、内部时间验证、外部仅观察；外部成绩不选epoch。最终全窗口重拟合曲线另存，不能将其内部样本误写成时间外验证。旧v5历史只有在线训练平均，不能倒造早期checkpoint的固定eval曲线。

内部候选共用该月初已知的风险下限，表示与表格的中心/尺度只由内部训练段拟合。风险下限未再做内部嵌套拟合，因此内部MSE比较的独立性弱于全流程嵌套验证；冻结观察月没有参与归一化参数拟合。该边界不能用内部误差优势掩盖，最终信息与策略判断仍以之后月份为对象。

## 3 信息性与期限

{mdtable(['预测器','H小时','平均TS IC(G)','平均TS RankIC(G)','小时CS IC(G)','共同IC(G)','相对IC(G)','72h块区间'],icrows)}

IC使用未来原收益G；TS逐币跨时间，CS每小时十二币横截面；月度TS ICIR为同一币各月IC均值/标准差，不乘独立小时根号，也不是策略信息比率。共同/相对是预测/目标同刻均值与残差的诊断，不自动等于可交易beta对冲收益。原分数与校准后信息矩阵都保留。重叠标签仅用于信息测量；账户使用真实价格份额盈亏，不累加重叠标签。

校准只读最近90日已成熟的真实月度时间外预测，分开共同/相对斜率（0—3、固定岭1e-4），共同截距收缩5倍、上限标准化0.02。分位边界在当月前冻结；分位表展示原收益、方向胜率、样本数、1%尾部、往返8bp与实际资金费后的条件均值。其样本是重叠事件，不是独立完整交易数；强分位也不强制成交。

神经信号控制Ridge/LightGBM/趋势/反转后的4h偏相关={inc['baseline_controlled_partial_IC_G4']:+.4f}；预测残差与G4相关={inc['neural_residual_IC_G4']:+.4f}。投影只在开发期拟合。开发冻结混合的神经权重为{r['mixture_frozen']['neural_weight']:.2f}，基线为{r['mixture_frozen']['baseline']}；权重若为零，应解释为当前混合没有采用神经增量。

## 4 从信息到持仓

统一决策先把标准化收益还原为bps，再扣过去已结算资金费按最近观察间隔估计的持有期费用。新开仓价格方向必须与扣资金费后的优势同向，且超过往返成本×倍数。没有预测方向时不会悄悄变成资金费套利。共同与beta控制后的相对预算各半，经总价格优势资格、毛/净/单币上限与过去协方差风险约束后合成。

每模型24个开发候选：H=2/4/8/12、成本倍数1/1.5/2、充分投资/优势比例两种。开发目标为4bp累计收益−0.5×回撤幅度；未以低费率或后续盈利来回选。固定共同政策H8、倍数1.5、充分投资，先隔离模型效应；再按相同24项预算选择各自政策。总主要策略尝试{policies['search_count']}项，另有预先规定的风险/满仓对照，不把每次局部修改当成独立证据。

联合网络选定H={p['horizon']}、倍数={p['multiplier']}、仓位映射={p['sizing']}。已有仓位使用较低维持门槛，明显反向、剩余优势消失、固定寿命到期或风险超限才成交；到期清仓，下一决策可再入场。期间保持份额，价格漂移不自动收费。月度换模型保留账户和仓位状态，末端真实清仓。年化风险预算60%、过去30日小时协方差70%+对角30%；风险匹配只用过去，不事后缩放净值。

### 同政策比较

{mdtable(['预测器','固定H8政策4bp收益','回撤','平均毛敞口','价格PnL/实际成交名义bp'],sharedrows)}

### 同预算各自开发政策

{mdtable(['预测器','H','成本倍数','仓位','gross','2bp','4bp','回撤','日Sharpe','平均毛敞口','平均持仓h'],rankrows)}

gross定义为**零交易手续费、仍计资金费**；另有独立价格PnL归因。2/4bp按每次开仓或平仓的实际名义变化收取，反手平旧开新两侧，末端收费；同一冻结规则分别重算权益、仓位与风险路径。maker仅为用户指定的费率情景，下一分钟open是假设成交，不证明真实限价被动成交。

## 5 满仓、杠杆与账户

{mdtable(['毛上限','gross','2bp','4bp','小时回撤','分钟回撤','实际平均毛','额度利用率','5倍保证金占比','分钟超过毛上限5%的小时'],lever)}

主账户毛上限2、净上限1、单币0.75；1/2/3/5都重走现金与份额路径。5倍是用户追加压力情景，文档主预算仍以2倍及3倍对照为准。合约保证金5倍与账户毛5倍不是同一概念。上限是成交决策约束，价格漂移可在两次决策间越限，分钟统计明确报告；没有历史维护保证金档位、精确连续标记价或强平撮合模型，因此这些是受限名义杠杆模拟。

充分投资映射会在合格优势范围内使用预算，但平均毛敞口仅{s['average_gross']:.3f}、参与率{s['participation']*100:.1f}%。另加固定多空排名的近满仓对照：平均毛{r['full_rank_control']['average_gross']:.3f}、4bp收益{pct(r['full_rank_control']['return'])}、回撤{pct(r['full_rank_control']['max_drawdown'])}。它强制交易排名，不作为经济优势证明或后续选型。不能为接近满仓而宣称无成本余量的信号有alpha。

账户采用一个币一个净份额、现金加份额的自融资等价账本（负现金为线性合约融资等价项，非实际交易所可用余额），权益变化严格等于价格PnL+资金费−手续费；保证金另按真实总名义/5计算。既不对信号相位重复计保证金，也不双收两个版本的手续费。联合策略权益核验误差={own['attribution']['account_identity_error']:.3e}。

## 6 十二币、月份与持仓结果

下表收益贡献以初始权益为单位，可加总；不是十二个各自独立账户的复合收益。信息列对应选定{p['horizon']}h原预测，交易列对应冻结4bp账户。

{mdtable(['品种','IC','RankIC','月TS ICIR','净贡献','资金费','平均绝对权重','持仓段','平均h','胜率','盈亏比','多头价格PnL','空头价格PnL'],coinrows)}

{mdtable(['月份','账户月收益','平均毛','相对权益换手',*symbols],months)}

最佳单币贡献{pct(own['attribution']['top1_coin_pnl'])}，不含最佳单币的原位置PnL合计{pct(own['attribution']['net_pnl_without_best_coin'])}；最强5日贡献{pct(own['attribution']['top5_days_pnl'])}，去掉它们的原路径PnL合计{pct(own['attribution']['net_pnl_without_top5_days'])}。这两项是集中性算术诊断，不伪装成重优化账户。另对十二币逐个禁用交易、保持规则约束重跑，完整结果在机器文件。多空、过去上涨/下跌、高/低风险状态、真实寿命分布与退出原因均保留；分层不反过来选择后续币池。

共同市场参照为同账本、过去风险匹配的等权持有，4bp收益{pct(r['market_reference']['return'])}；现金0。联合策略对市场的描述性beta={at['market_beta']:.3f}，年化算术截距={pct(at['annualized_arithmetic_intercept'])}，HAC7截距t={at['intercept_HAC7_t']:.2f}。事后回归是归因，不是用未来beta构造的可交易对冲。

## 7 新旧净值衔接

保留原v3旧5m、6bp、四独立相位曲线，并用当前真实资金费事件复算；四相位末值与保存值的最大差{max(abs(x['difference']) for x in native['checks']):.3e}，差异若非零明确留在复现文件。原v5无完整资金费、每小时精确目标权重账本也原样保留。

共同连续区间为{r['legacy_common_bridge']['date_start']}—{r['legacy_common_bridge']['date_end']}，共{r['legacy_common_bridge']['hours']}个时刻；原v3保存分数、原v5和新联合网络均进同一1m/4h/4bp账户。旧v3输入仍为原5m来源，所以这是口径桥接，**不进公平模型排行榜**；缺失预测未填零。另固定原v5的校准4h预测流，比较统一相位风格与成本持仓政策，分别收益{pct(r['policy_bridge']['same_predictor_unified_phase']['return'])}和{pct(r['policy_bridge']['same_predictor_economic_4h']['return'])}，隔离政策变化。

正式连续账户贯穿2025-07到2026-09，切换月沿用仓位、计实际换仓成本，不拼接从1开始的胜者片段。原生曲线仍显示各自合同，不能把不同输入/费用/政策的高低当作模型进步。

## 8 数据、资金费与自查

{mdtable(['品种','事件','直接结算标记价','对应小时mark_open近似','分钟交易价代理'],fundingrows)}

结算事件直接来自数据，观察到的间隔按合约真实事件列报告，不强制全币8h；费率预测只读决策前已结算事件。现金费用用实际持仓份额×结算费率×结算标记价；缺失直接标记价时使用对应小时mark_open近似，仍缺失时用事件当时分钟价格代理。**近似覆盖不等于精确标记价覆盖。** 分钟open路径核验反映价格风险，也不是完整交易所维护保证金/强平验证。[交易所资金费说明](https://www.binance.com/en/support/faq/detail/360033525031)、[标记价与强平说明](https://www.binance.com/en-NZ/support/faq/detail/360033525271)支持这种区分。

{count}项检查通过，含手续费双边/反手、份额保持/到期、资金费守恒、资产排列等变性、期限对齐、未来价格不改过去结果以及旧合同核验。原始数据/缓存、v5来源、月度训练/公平基线、预测与政策/账本/报告分别记录身份；最终核验文件在`outputs/alpha_strategy/final_audit.json`。

无标签推理已在{latest['decision_utc']}实际执行，输出十二币四期限原收益/校准收益与空仓目标权重；保存为`latest_label_free_prediction.json`。接口只接收完成的4时刻表示、过去风险、过去协方差/beta及已知资金费，不接收未来标签。历史与实时读出误差已核验；已有仓位使用同一个decide滞回/到期状态逻辑。

开发选定主方案另在同一最新时刻实际运行原三种子CUDA预测并核验，输出`latest_primary_prediction.json`；它是主方案推理证据，与联合候选快照分开保留。源、账户、报告及主方案/近满仓的24条费用与杠杆路径在最终核验中逐项检查。

## 9 复现与结论边界

仓库根目录通过`./scripts/run_alpha_strategy.ps1`依次执行`-Stage data`、`models`、`baselines`、`analysis`、`test`、`report`，或`all`（报告前须已有测试结果）。固定环境为universal，CUDA不可用即失败。`outputs/alpha_strategy`保存月度权重、预测、校准、统一表、逐币仓位/实际成交、净值、资金费、归因与来源，不覆盖旧输出。

新增读出候选采用滚动月度更新；原v5主方案保留其原Nov24/Mar25/Jul25/Feb26模型切换，按月更新成熟折外校准。没有启动真实或影子资金交易，也没有新增前瞻记录。能够正确运行、有阶段性IC、某些毛收益或某条开发净值为正，都不足以证明稳定超额。主要判断需同时看费用余量、时间迁移、基线增量、集中性与风险使用。{primary_conclusion}
'''
    primary_md=f'''## 0 开发期锁定的主策略（先看此节）

**主策略是{NAMES[primary]}，H={pp['horizon']}h、成本倍数={pp['multiplier']}、仓位映射={pp['sizing']}。** 三种神经方案仅按开发期效用选型，没有按后续净值回选。下面联合状态网络属于结构候选的完整诊断，不能将它自动混称为主方案。

{primary_conclusion} 主方案gross={pct(primary_model['cost_scenarios']['gross']['return'])}、单边2bp={pct(primary_model['cost_scenarios']['2bp']['return'])}、单边4bp={pct(ps['return'])}；净收益区间[{pct(pci[0])}, {pct(pci[1])}]，相对开发期选定基线{system['baseline_selected_in_development']}的收益差区间[{pct(pinc[0])}, {pct(pinc[1])}]。原v5为主时，12h仍是4h信号的折外期限投影与持有规则，不称原模型重新训练了12h。

{mdtable(['毛上限','gross','2bp','4bp','小时回撤','分钟回撤','平均毛','额度利用率','5倍保证金占比'],primary_rows)}

主方案平均毛敞口{ps['average_gross']:.3f}、参与率{ps['participation']*100:.1f}%、实际平均持仓{ps['holding_mean']:.2f}h，尚未达到接近满仓。近满仓排名对照平均毛{system['full_rank_control']['average_gross']:.3f}，4bp收益{pct(system['full_rank_control']['return'])}、回撤{pct(system['full_rank_control']['max_drawdown'])}。它每12h复核排名，保持方向的持仓不人为切段，因此实际平均寿命{system['full_rank_control']['holding_mean']:.2f}h；不是12h必平仓。该对照不使用经济门槛或60%风险缩放，不能与风险约束主方案混称；实际排名仍来自神经预测。只按固定对照规则给出以下全部情景，没有按后续成绩替换主方案。三方案选择额外计3次系统比较，主要政策尝试仍为173次。

{mdtable(['排名毛上限','gross','2bp','4bp','分钟回撤','平均毛','5倍保证金占比','平均持仓h','分钟毛超限小时'],rank_cap_rows)}

近满仓2倍的4bp净收益7日时间块区间为[{', '.join(pct(x) for x in system['full_rank_net_intervals']['2.0']['calendar_7day_block_95pct'])}]，相对开发期基线差区间为[{', '.join(pct(x) for x in system['full_rank_paired_increment']['calendar_7day_block_95pct'])}]；同样未修正多次开发与历史已见偏差。每档区间全部保留，不能只展示5倍收益而隐藏其分钟回撤。按2倍排名对照的逐币持仓贡献如下：

{mdtable(['品种','排名净贡献','资金费','负手续费','平均权重','持仓段','平均h','胜率'],rank_coins)}

{mdtable(['品种','IC','RankIC','月ICIR','净贡献','资金费','平均权重','持仓段','平均h','胜率','多头PnL','空头PnL'],primary_coins)}

{mdtable(['时期','信号','平均TS IC','平均TS RankIC','小时CS IC','共同IC','相对IC'],primary_ic_rows)}

主方案选定12h的原收益Pearson IC={r['information'][primary]['frozen']['raw'][str(pp['horizon'])]['mean_TS_IC_G']:+.4f}，RankIC={r['information'][primary]['frozen']['raw'][str(pp['horizon'])]['mean_TS_RankIC_G']:+.4f}。Pearson为正而排名关联接近零、最强5日贡献超过全部净收益，支持“部分幅度/尾部时段存在可用判断”，尚不支持广泛稳定的排序优势。不能单凭正IC把整个时间流都放大。F1为2025-07—2026-01，F2为2026-02—09；分期校准与收益迁移均可独立复核。

{mdtable(['月份','账户月收益','平均毛',*symbols],primary_months)}

主方案最佳单币贡献{pct(primary_model['attribution']['top1_coin_pnl'])}，不含最佳单币的原路径贡献{pct(primary_model['attribution']['net_pnl_without_best_coin'])}；不含最强5日的原路径贡献{pct(primary_model['attribution']['net_pnl_without_top5_days'])}。逐个禁用币种并保持原规则的重跑如下，不能按这些后续成绩删币回选：

{mdtable(['禁用品种','4bp重跑收益','回撤'],primary_leave)}

对同账本过去风险匹配市场的描述性beta={primary_model['market_attribution']['market_beta']:.3f}，HAC7截距t={primary_model['market_attribution']['intercept_HAC7_t']:.2f}。全体币/月/高低风险与多空归因均在主策略机器结果中；结构候选的更详细机制诊断见后续各节。

'''
    md=md.replace('## 1 对齐要求与研究取舍',primary_md+'## 1 对齐要求与研究取舍');DOC.write_text(md,encoding='utf-8')
    def fig(name,caption):return '<figure>'+svgs[name]+'<figcaption>'+caption+'</figcaption></figure>'
    sections=[]
    sections.append('<section id="primary"><div class="kicker">00 / DEVELOPMENT-LOCKED PRIMARY SYSTEM</div><h2>'+NAMES[primary]+'：开发期选定的主方案</h2>'+f'<p>{primary_conclusion} H={pp["horizon"]}h、成本倍数{pp["multiplier"]}、{pp["sizing"]}。下方联合结构仅为候选诊断，未因更复杂而默认替换。</p>'+fig('primary','三种费用各自记账；四种杠杆额度分别重算。垂直虚线不是重置账户。')+table(['毛上限','gross','2bp','4bp','小时回撤','分钟回撤','平均毛','利用率','5倍保证金'],primary_rows)+table(['品种','IC','RankIC','月ICIR','净贡献','资金费','平均权重','持仓段','平均h','胜率','多头PnL','空头PnL'],primary_coins)+f'<div class="note">净收益区间[{pct(pci[0])}, {pct(pci[1])}]；相对开发期基线增量区间[{pct(pinc[0])}, {pct(pinc[1])}]。原v5若被选用，12h是折外期限投影与持有规则，不冒充原生12h训练。</div></section>')
    sections.append('<section id="primary-details"><div class="kicker">00B / PRIMARY ASSETS & CAPITAL USE</div><h2>主方案逐月信息、持仓与近满仓对照</h2>'+fig('primary_information','主方案实际仓位与所选12h期限的逐币/月信息，固定全体十二币。')+table(['月份','账户月收益','平均毛',*symbols],primary_months)+fig('primary_rank','固定神经分数排序、每12h复核；同方向不人为切段。排名对照不经过经济门槛或60%风险缩放，不能混称为主方案。')+table(['排名毛上限','gross','2bp','4bp','分钟回撤','平均毛','5倍保证金','平均持仓h','分钟毛超限小时'],rank_cap_rows)+table(['品种','排名净贡献','资金费','负手续费','平均权重','持仓段','平均h','胜率'],rank_coins)+table(['禁用品种','主方案4bp重跑收益','回撤'],primary_leave)+f'<div class="note">主方案平均毛{ps["average_gross"]:.3f}，实际平均持仓{ps["holding_mean"]:.2f}h。排名对照平均毛{system["full_rank_control"]["average_gross"]:.3f}，平均持仓{system["full_rank_control"]["holding_mean"]:.2f}h。2倍排名净收益区间：'+', '.join(pct(x) for x in system['full_rank_net_intervals']['2.0']['calendar_7day_block_95pct'])+'；排名对照不按后续成绩替换主方案。</div></section>')
    sections.append('<section id="contract"><div class="kicker">01 / COMMON RESEARCH CONTRACT</div><h2>把信息、策略与账户放进同一条件</h2>'+table(['文档要求','实际完成'],trace)+'<p>冻结早期v5编码器，月度更新多期限读出；一个联合市场/相对状态候选。基线读取包含分钟支路的176维可见摘要。开发期2025-03—06，之后仅冻结规则历史重放。</p></section>')
    sections.append('<section id="information"><div class="kicker">02 / INFORMATION BEFORE EXPOSURE</div><h2>在哪些币、哪些期限存在信息？</h2>'+fig('information','原收益G上的逐币时间序列IC均值；标准化目标、截面IC和策略收益分别计量。')+fig('matrix','保留全部十二币与月份；不按后续成绩挑选币池。')+table(['模型','H','TS IC','TS RankIC','CS IC','共同','相对','72h块区间'],icrows)+'<div class="note">ICIR使用月度TS IC均值/标准差；相邻目标重叠，区间按共同市场自然时间块抽样。</div><div class="controls"><label>预测器<select id="model">'+''.join(f'<option value="{n}">{NAMES[n]}</option>' for n in models[:-1])+'</select></label><label>信息期限<select id="horizon">'+''.join(f'<option value="{h}">{h}小时</option>' for h in HORIZONS)+'</select></label></div><div id="coin-evidence"></div></section>')
    sections.append('<section id="training"><div class="kicker">03 / FITTING & TRANSFER</div><h2>一次有机制理由的结构调整</h2><p>单币编码 → 共同池化与相对状态 → 最近4时刻联合记忆 → 共同市场头＋币种增量头。未来共同标签仅用于监督，不进入推理。</p>'+fig('training','同一内部拟合模型：固定训练eval、内部时间验证、外部仅观察。虚线预算由内部验证选择，外部不能回选。')+fig('quantiles','分位边界来自已成熟折外历史。均值已扣完整往返8bp与资金费，是重叠事件诊断，不是成交收益承诺。')+'</section>')
    sections.append('<section id="strategy"><div class="kicker">04 / PREDICTOR × HOLDING POLICY</div><h2>比较预测器，也比较每一笔成交的必要性</h2><p>同一个经济决策器，再给每个模型同样24项开发选择预算。实际份额持有、较低维持门槛、到期退出；无需每小时追踪精确目标权重。</p>'+table(['模型','固定H8政策4bp','回撤','平均毛','价格产出bp'],sharedrows)+fig('ranking','gross零手续费仍计资金费；2/4bp开平各收，同一冻结政策重算。')+table(['模型','H','成本倍数','仓位','gross','2bp','4bp','回撤','日Sharpe','平均毛','实际平均h'],rankrows)+'</section>')
    sections.append('<section id="equity"><div class="kicker">05 / CONTINUOUS ACCOUNT</div><h2>一条账户路径，完整呈现成本与回撤</h2><div class="note">下一分钟open为研究成交假设；maker为费率情景。缺少真实排队/盘口成交证据，不能写成已验证被动成交。gross仍计资金费。</div>'+fig('equity','月度模型更新沿用真实持仓；虚线为2026-02，账户没有重置。')+fig('drawdown','小时边界回撤与报告中的分钟路径回撤分开显示。')+fig('positions','仓位、参与率与风险利用来自实际份额账本。')+'</section>')
    sections.append('<section id="coins"><div class="kicker">06 / TWELVE ASSETS & HOLDING LIFE</div><h2>全体品种的信息、贡献与真实持有期</h2>'+fig('holdings','实际持仓寿命和价格/资金费/交易费贡献。初始权益单位可加总，不是独立账户收益。')+table(['品种','IC','RankIC','月ICIR','净贡献','资金费','平均权重','持仓段','平均h','胜率','盈亏比','多头PnL','空头PnL'],coinrows)+table(['月份','账户月收益','平均毛','换手',*symbols],months)+'</section>')
    sections.append('<section id="risk"><div class="kicker">07 / CAPITAL & LEVERAGE</div><h2>接近满仓的代价，不由杠杆掩盖</h2>'+fig('leverage','1/2/3/5毛上限分别重新记账；5倍合约保证金与账户毛敞口各自计量。')+table(['毛上限','gross','2bp','4bp','小时回撤','分钟回撤','平均毛','利用率','5倍保证金','分钟越限小时'],lever)+f'<div class="note">强制排名近满仓对照：平均毛{r["full_rank_control"]["average_gross"]:.3f}，4bp收益{pct(r["full_rank_control"]["return"])}。这是风险/利用率对照，未按后续成绩选型。缺少完整历史维护保证金，不宣称通过强平验证。</div></section>')
    sections.append('<section id="bridge"><div class="kicker">08 / LEGACY BRIDGE</div><h2>把模型变化与口径变化分开</h2>'+fig('bridge','左：最长共同连续区间、同一研究账本；右：旧v3原生6bp四独立相位。旧输入身份不同，桥接不进入公平排行榜。')+f'<p>固定同一v5校准4h预测流：统一相位风格收益{pct(r["policy_bridge"]["same_predictor_unified_phase"]["return"])}；成本持仓政策收益{pct(r["policy_bridge"]["same_predictor_economic_4h"]["return"])}。原v5原生曲线及来源保留。</p></section>')
    sections.append('<section id="evidence"><div class="kicker">09 / ATTRIBUTION & HANDOFF</div><h2>结论可以被复核，也可以被反驳</h2>'+f'<p>{conclusion}</p><p>联合网络净收益区间[{pct(interval[0])}, {pct(interval[1])}]；相对开发选定基线收益差区间[{pct(paired[0])}, {pct(paired[1])}]。未修正开发选择偏差。</p>'+table(['品种','资金费事件','直接标记价','小时mark_open近似','分钟价格代理'],fundingrows)+f'<p>{count}项检查通过 · {html.escape(runtime["gpu"])} · CUDA {runtime["cuda"]}<br>最新无标签预测：{latest["decision_utc"]}，十二币四期限。</p><div class="links"><a href="../../docs/ALPHA_STRATEGY_RESEARCH_REPORT_2026-10-04.md">完整Markdown研究报告 →</a><a href="results.json">完整机器证据 →</a></div></section>')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth;color-scheme:dark}body{margin:0;background:#091522;color:#edf5ff;font:15px/1.85 "Microsoft YaHei","Segoe UI",sans-serif}a{color:#80c2ff;text-decoration:none}a:hover{text-decoration:underline}.shell{max-width:1760px;margin:auto;display:grid;grid-template-columns:230px minmax(0,1fr)}aside{position:sticky;top:0;height:100vh;padding:30px 23px;border-right:1px solid #2b425c}.brand{font-size:20px;letter-spacing:3px}.sub{color:#ffc97c;font-size:12px;margin:14px 0 24px}nav a{display:block;padding:10px 8px;color:#b8cbe1;border-radius:7px}nav a:hover{background:#17334d}main{min-width:0;padding:48px 48px 70px}.kicker{color:#80c2ff;font-size:12px;letter-spacing:2px}h1{font-size:43px;line-height:1.35;margin:16px 0 23px}h2{font-size:27px;line-height:1.45;margin:8px 0 20px}p{color:#b9cee4}.lead{font-size:18px;max-width:1000px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}.card{background:#102034;border:1px solid #304963;padding:22px;border-radius:11px}.value{font-size:28px}.label{font-size:12px;color:#b9cee4}.note{background:#292b3c;border-left:3px solid #ffc97c;padding:18px 22px;margin:22px 0;border-radius:0 8px 8px 0;color:#dddfe9}section{margin-top:64px;scroll-margin-top:22px}figure{margin:24px 0;padding:18px;border:1px solid #304963;background:#102034;border-radius:12px}figure svg{display:block;width:100%;height:auto}figcaption{color:#adbed1;font-size:13px;margin-top:13px}.table{overflow:auto;border:1px solid #304963;border-radius:10px;margin:22px 0}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}th{background:#19344f;text-align:left;font-weight:500;padding:12px}td{padding:11px 12px;border-top:1px solid #304963}tbody tr:nth-child(even){background:#102034}tbody tr:hover{background:#16324a}.controls{display:flex;gap:18px;margin-top:25px}.controls label{flex:1;color:#b9cee4}select{display:block;width:100%;padding:11px;background:#19344f;border:1px solid #52728e;color:#edf5ff;border-radius:8px;font:inherit;margin-top:8px}.links{display:flex;gap:24px;flex-wrap:wrap}footer{border-top:1px solid #304963;margin-top:65px;padding-top:23px;font-size:12px;color:#a9bfd6}@media(max-width:1100px){main{padding:30px 25px}.cards{grid-template-columns:1fr 1fr}h1{font-size:35px}}@media(max-width:760px){.shell{display:block}aside{position:relative;height:auto;padding:20px;border-right:0;border-bottom:1px solid #304963}nav{display:flex;overflow:auto}nav a{white-space:nowrap}main{padding:28px 18px}h1{font-size:29px}.controls{flex-direction:column}}@media print{aside{display:none}.shell{display:block}figure{break-inside:avoid}main{padding:10px}}'''
    controls={name:{str(h):r['information'][name]['frozen']['raw'][str(h)]['by_coin'] for h in HORIZONS} for name in models[:-1]};payload={'information':controls,'names':NAMES,'symbols':symbols}
    js='const evidence='+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+';const m=document.getElementById("model"),h=document.getElementById("horizon");function show(){const r=evidence.information[m.value][h.value];document.getElementById("coin-evidence").innerHTML="<p>所选期限只更新信息表，不重新优化策略或净值。</p><div class=table><table><thead><tr><th>品种</th><th>TS IC(G)</th><th>TS RankIC(G)</th><th>月度TS ICIR</th></tr></thead><tbody>"+evidence.symbols.map(s=>"<tr><td>"+s+"</td><td>"+r[s].IC_G.toFixed(4)+"</td><td>"+r[s].RankIC_G.toFixed(4)+"</td><td>"+r[s].monthly_TS_ICIR.toFixed(3)+"</td></tr>").join("")+"</tbody></table></div>"}m.addEventListener("change",show);h.addEventListener("change",show);m.value="NN_joint";h.value="'+str(p['horizon'])+'";show();'
    js=js.replace('m.value="NN_joint";h.value="'+str(p['horizon'])+'"','m.value="'+primary+'";h.value="'+str(pp['horizon'])+'"')
    nav=[('primary','主策略与净值'),('primary-details','主方案品种与满仓'),('contract','统一研究条件'),('information','信息与期限'),('training','模型与训练'),('strategy','策略与基线'),('equity','候选净值与回撤'),('coins','候选品种与持仓'),('risk','候选满仓与杠杆'),('bridge','新旧衔接'),('evidence','归因与交付')]
    page='<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>深度学习信息优势与统一策略 · 2026-10-04</title><style>'+css+'</style></head><body><div class="shell"><aside><div class="brand">CRYPTO / ALPHA</div><div class="sub">INFORMATION → HOLDING → EQUITY</div><nav>'+''.join(f'<a href="#{x}">{y}</a>' for x,y in nav)+'</nav></aside><main><div class="kicker">2026.10.04 · 1 MINUTE · UNIVERSAL CUDA</div><h1>让预测信息，<br>接受成本与持仓的检验。</h1><p class="lead">'+conclusion+'</p><div class="cards">'+''.join(f'<div class="card"><div class="value">{a}</div><div class="label">{b}</div></div>' for a,b in [(pct(own['cost_scenarios']['gross']['return']),'联合网络 · gross含资金费'),(pct(own['cost_scenarios']['2bp']['return']),'单边2bp · 开平分别收费'),(pct(s['return']),'单边4bp · 冻结历史'),(f"{s['average_gross']:.3f}",'实际平均毛敞口／2倍上限')])+'</div><div class="note">规则在2025-07前开发冻结，之后仅按锁定月度更新规则回放。所有历史已被查看；没有真实新增前瞻证据。经济优势、信号信息与高仓位利用分别评价。</div>'+''.join(sections)+'<footer>依据《神经网络的信息优势与可交易策略设计》 · 无外部CDN · 所有图来自实际预测与账户<br>费用不证明maker成交；标记价近似及缺失维护保证金限制了实盘可行性判断。</footer></main></div><script>'+js+'</script></body></html>'
    # The main header describes the development-selected system; candidate evidence stays in its own sections.
    page=page.replace('<p class="lead">'+conclusion+'</p>','<p class="lead">'+primary_conclusion+'</p>')
    page=page.replace('一条账户路径，完整呈现成本与回撤','联合候选：净值、成本与回撤').replace('全体品种的信息、贡献与真实持有期','联合候选：十二币贡献与持有期').replace('接近满仓的代价，不由杠杆掩盖','联合候选：杠杆与近满仓结果')
    evidence_insert=table(['时期','信号','平均TS IC','平均TS RankIC','CS IC','共同','相对'],primary_ic_rows)+f'<p>主方案最强5日之外的原路径PnL合计{pct(primary_model["attribution"]["net_pnl_without_top5_days"])}；Pearson为正而排名关联接近零，收益集中，不能单凭正IC放大整个时间流。</p>'
    page=page.replace('主方案逐月信息、持仓与近满仓对照</h2>','主方案逐月信息、持仓与近满仓对照</h2>'+evidence_insert)
    page=page.replace('<select id="model">','<select id="model" aria-label="预测器">').replace('<select id="horizon">','<select id="horizon" aria-label="信息期限">')
    start=page.index('<div class="cards">');end=page.index('<div class="note">',start)
    cards='<div class="cards">'+''.join(f'<div class="card"><div class="value">{a}</div><div class="label">{b}</div></div>' for a,b in [(pct(primary_model['cost_scenarios']['gross']['return']),'主方案 · gross含资金费'),(pct(primary_model['cost_scenarios']['2bp']['return']),'主方案 · 单边2bp'),(pct(ps['return']),'主方案 · 单边4bp'),(f"{ps['average_gross']:.3f}",'主方案平均毛敞口／2倍上限')])+'</div>'
    page=page[:start]+cards+page[end:]
    (OUT/'index.html').write_text(page,encoding='utf-8');combined=dict(r);combined['primary_system']=system;save_json(OUT/'results.json',combined);(OUT/'README.md').write_text(f'# 信息优势与统一策略研究\n\n{primary_conclusion}\n\n[交互报告](index.html) · [完整研究报告](../../docs/ALPHA_STRATEGY_RESEARCH_REPORT_2026-10-04.md) · [机器证据](results.json)\n',encoding='utf-8')
    paths=list(Path('src/crypto_timing').glob('alpha_*.py'))+[Path('scripts/run_alpha_strategy.py'),Path('scripts/run_alpha_strategy.ps1'),Path('scripts/audit_alpha_strategy.py'),Path(__file__),Path('tests/test_alpha_strategy.py'),Path('README.md'),Path('docs/ALPHA_STRATEGY_IMPLEMENTATION_PROTOCOL_2026-10-04.md'),DOC,OUT/'index.html',OUT/'results.json',OUT/'README.md']+list((OUT/'assets').glob('*'));save_json(ROOT/'delivery_manifest.json',{'sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})
    print(f'Report {OUT}/index.html\nMarkdown {DOC}',flush=True)
if __name__=='__main__':main()
