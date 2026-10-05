"""Self-contained v5 visual evidence and the complete Chinese research handoff."""
from __future__ import annotations
import json,html,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path('outputs/review_v5');OUT=Path('reports/review_2026-10-04');DOC=Path('docs/V5_RESEARCH_REPORT_2026-10-04.md')
COLORS=['#77baff','#67dfbe','#f6c37b','#b3a0ff','#fa8e9a']
NAMES={'R':'直接收益','raw_soft':'原Q/软预算','raw_hard':'原Q/硬预算','TA_hard':'logT+A','shapedQ_hard':'压缩Q','shapedQ_tail':'压缩Q+尾部','baseline':'成熟标签基线','no_body':'只去body','no_wicks':'只去影线','common':'共同近期摘要','events_common':'冲击修复+摘要','events_gru':'冲击修复+patch GRU','events_gru_segments':'事件+非重叠期限'}
VARIANTS={'neural_raw':'原始神经信号','OOF_calibrated':'折外校准','OOF_filtered':'折外校准与滤波','baseline_only':'慢基线诊断','matched_task_contrast':'同架构任务对照'}
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def pct(x):return f'{x*100:+.4f}%'
def table(headers,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def architecture():
    boxes=[(25,65,185,70,'1m / 180根','局部TCN → 36patch读出'),(25,165,185,70,'5m / 144根','局部TCN96 → GRU64'),(25,265,185,70,'1h / 168根与状态','背景分段 / 轻FiLM'),(270,160,200,85,'融合表示 h128','线性增量 a · 原头可重建'),(280,305,180,65,'成熟历史基线 b','独立收缩 · 5h标签延迟'),(535,150,180,90,'总预测 b+a','直接Y4 MSE · 主均值不截尾'),(535,290,180,85,'独立辅助对象','活动 / 压力 / 事件 / 期限'),(775,150,175,90,'冻结二阶段处理','三种子真实折外材料'),(775,305,175,65,'研究用途诊断','非重叠小时账本 / 净值')]
    content=''
    for x,y,w,h,a,b in boxes:content+=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#152a42" stroke="#3f5874"/><text x="{x+w/2}" y="{y+29}" text-anchor="middle" fill="#edf4ff" font-size="15">{a}</text><text x="{x+w/2}" y="{y+52}" text-anchor="middle" fill="#b7c9df" font-size="11">{b}</text>'
    arrows='<path d="M210 100 H240 V182 H270 M210 200 H270 M210 300 H240 V220 H270 M470 202 H535 M460 337 H495 V222 H535 M715 195 H775 M860 240 V305" stroke="#77baff" stroke-width="2" fill="none" marker-end="url(#arrow)"/><path d="M210 220 H250 V280 H505 V327 H535 M470 245 V280 H505" stroke="#f6c37b" stroke-width="2" stroke-dasharray="5 4" fill="none" marker-end="url(#arrow)"/>'
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 975 420" role="img" aria-label="v5研究框架与可选机制"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#77baff"/></marker></defs><text x="25" y="30" fill="#f6c37b" font-size="13">研究框架：辅助、事件与慢基线均为受检验候选；实际主选型未默认启用</text>'+arrows+content+'<text x="25" y="405" fill="#a7bbd2" font-size="12">共享预算覆盖全部实际共同参数 · 每个输出对象独立说明 · 信号水平不当作独立脉冲累加</text></svg>'
def style():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'figure.facecolor':'#101d30','axes.facecolor':'#101d30','savefig.facecolor':'#101d30','text.color':'#edf4ff','axes.labelcolor':'#b7c9df','xtick.color':'#b7c9df','ytick.color':'#b7c9df','axes.edgecolor':'#405570','grid.color':'#30435d','axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'crypto-review-v5'})
def savefig(name):
    p=OUT/'assets'/f'{name}.svg';plt.savefig(p,bbox_inches='tight',metadata={'Date':'2026-10-04'})
    if name in ('nav','drawdown'):plt.savefig(OUT/'assets'/f'{name}.png',dpi=160,bbox_inches='tight')
    plt.close();s=p.read_text(encoding='utf-8');return s[s.index('<svg'):]

def figures(screen,confirm,analysis):
    style();svgs={}
    fig,axes=plt.subplots(1,2,figsize=(13,6));keys=list(screen);yy=np.arange(len(keys))
    for ax,split,label in zip(axes,('validation','replay'),('内部选择：2025-03/04','内部后续：2025-05/06')):
        ax.barh(yy,[screen[k]['results'][split]['skill']*100 for k in keys],color=[COLORS[1] if k=='R' else COLORS[0] for k in keys]);ax.set_yticks(yy,[NAMES[k] for k in keys]);ax.invert_yaxis();ax.axvline(0,color='#8296ae',lw=.8);ax.set_title(label);ax.set_xlabel('预测MSE skill / %');ax.grid(axis='x',alpha=.35)
    fig.tight_layout();svgs['hypotheses']=savefig('hypotheses')
    fig,axes=plt.subplots(1,2,figsize=(13,4.4))
    for ax,fold in zip(axes,('f1','f2')):
        rows=analysis['folds'][fold]['splits'];labels=[];cov=[];mean=[];var=[]
        for split in ('validation','replay'):
            for name in ('neural_raw','OOF_calibrated','OOF_filtered'):
                d=rows[split][name]['metrics'];labels.append(('观察' if split=='validation' else '后续')+' / '+VARIANTS[name]);cov.append(d['covariance_gain']*100);mean.append(d['mean_alignment']*100);var.append(d['variance_penalty']*100)
        y=np.arange(len(labels));ax.barh(y-.23,cov,.23,color=COLORS[0],label='动态协方差');ax.barh(y,mean,.23,color=COLORS[1],label='均值匹配');ax.barh(y+.23,var,.23,color=COLORS[2],label='幅度代价');ax.set_yticks(y,labels,fontsize=8);ax.axvline(0,color='#aaa',lw=.8);ax.invert_yaxis();ax.set_title(fold.upper());ax.set_xlabel('skill分解 / %');ax.grid(axis='x',alpha=.3)
    axes[0].legend(fontsize=8);fig.tight_layout();svgs['decomposition']=savefig('decomposition')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,fold in zip(axes,('f1','f2')):
        for name,color in zip(('neural_raw','OOF_calibrated','OOF_filtered','baseline_only'),COLORS):
            d=np.load(ROOT/f'nav_{fold}_{name}.npz');ax.plot(d['dates'],d['nav'],color=color,label=VARIANTS[name],lw=1.5)
        ax.axhline(1,color='#8b9cb0',lw=.7);ax.set_title(f'{fold.upper()} 连续小时账本 · 单边4bps');ax.set_ylabel('净值');ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.grid(alpha=.3);ax.tick_params(axis='x',rotation=25)
    axes[0].legend(fontsize=8);fig.tight_layout();svgs['nav']=savefig('nav')
    fig,axes=plt.subplots(1,2,figsize=(13,4.1))
    for ax,fold in zip(axes,('f1','f2')):
        for name,color in zip(('neural_raw','OOF_calibrated','OOF_filtered'),COLORS):
            d=np.load(ROOT/f'nav_{fold}_{name}.npz');peak=np.maximum.accumulate(np.r_[1.,d['nav']])[1:];ax.plot(d['dates'],(d['nav']/peak-1)*100,color=color,label=VARIANTS[name]);
        ax.set_title(f'{fold.upper()} 回撤');ax.set_ylabel('%');ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'));ax.grid(alpha=.3)
    axes[0].legend(fontsize=8);fig.tight_layout();svgs['drawdown']=savefig('drawdown')
    fig,axes=plt.subplots(1,2,figsize=(13,4.4))
    for ax,fold in zip(axes,('f1','f2')):
        d=analysis['folds'][fold]['splits']['replay']
        for name,color in zip(('neural_raw','OOF_calibrated','OOF_filtered'),COLORS):
            values=d[name]['lifetime']['disjoint_forward_correlations'];ax.plot(range(5),list(values.values()),marker='o',color=color,label=VARIANTS[name])
        ax.set_xticks(range(5),['0—1h','1—2h','2—3h','3—4h','4—8h']);ax.axhline(0,color='#8b9cb0',lw=.7);ax.set_title(f'{fold.upper()} 非重叠未来段');ax.set_ylabel('信号与该段对数收益相关');ax.grid(alpha=.3)
    axes[0].legend(fontsize=8);fig.tight_layout();svgs['lifetime']=savefig('lifetime')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,fold in zip(axes,('f1','f2')):
        for name,color in zip(('neural_raw','OOF_calibrated','OOF_filtered'),COLORS):
            months={}
            for split in ('validation','replay'):months.update(analysis['folds'][fold]['splits'][split][name]['metrics']['monthly'])
            ax.plot(range(len(months)),[v['skill']*100 for v in months.values()],marker='o',color=color,label=VARIANTS[name]);ax.set_xticks(range(len(months)),list(months),rotation=30,fontsize=8)
        ax.axhline(0,color='#aaa',lw=.7);ax.set_title(f'{fold.upper()} 逐月预测误差');ax.set_ylabel('skill / %');ax.grid(alpha=.3)
    axes[0].legend(fontsize=8);fig.tight_layout();svgs['monthly']=savefig('monthly')
    fig,axes=plt.subplots(1,2,figsize=(13,4.2))
    for name in ('raw_soft','raw_hard','shapedQ_hard','events_gru_segments'):
        record=screen[name];log=record['gradient_diagnostics'];axes[0].plot([g['max_parameter_ratio'] for g in log],lw=1,label=NAMES[name]);
    axes[0].axhline(.3,color=COLORS[2],ls='--',label='硬上限0.3');axes[0].set_yscale('symlog',linthresh=.3);axes[0].set_title('已记录共享参数张量最大辅/主梯度比');axes[0].set_xlabel('记录批次');axes[0].legend(fontsize=8)
    for name,color in zip(('R','shapedQ_hard','events_gru_segments'),COLORS):
        hist=read(f'screen/{name}/history.json');axes[1].plot([h['epoch'] for h in hist],[h['train_mse'] for h in hist],marker='o',color=color,label=NAMES[name])
    axes[1].set_title('相同曝光预算的直接收益训练误差');axes[1].set_xlabel('预算轮：128更新/轮');axes[1].legend(fontsize=8);axes[1].grid(alpha=.3);fig.tight_layout();svgs['optimization']=savefig('optimization')
    fig,ax=plt.subplots(figsize=(13,4.5));rows=read('field_audit_f2.json')['minute'];fields=list(dict.fromkeys(r['field'] for r in rows));months=list(dict.fromkeys(r['month'] for r in rows));matrix=np.zeros((len(fields),len(months)))
    for r in rows:matrix[fields.index(r['field']),months.index(r['month'])]=max(matrix[fields.index(r['field']),months.index(r['month'])],r['clip_fraction']*100)
    im=ax.imshow(matrix,aspect='auto',cmap='magma');ax.set_yticks(range(len(fields)),fields,fontsize=8);ix=np.arange(0,len(months),3);ax.set_xticks(ix,[months[j] for j in ix],rotation=35,fontsize=8);ax.set_title('逐字段×月份：十二币最大分钟截尾率（完整逐币证据另存JSON）');fig.colorbar(im,ax=ax,label='截尾率 / %');fig.tight_layout();svgs['field_drift']=savefig('field_drift')
    fig,axes=plt.subplots(1,2,figsize=(13,4.4));names=['logT','Q','A']
    for ax,run,title in zip(axes,('raw_hard','shapedQ_hard'),('原始路径监督','训练尺度压缩后的路径监督')):
        native=analysis['screen'][run]['native_auxiliary'];keys=list(native);ax.bar(range(len(keys)),[native[k]['mse'] for k in keys],color=COLORS[:len(keys)]);ax.set_xticks(range(len(keys)),keys);ax.set_yscale('log');ax.set_title(title+'：内部折外自身误差');ax.set_ylabel('各自目标MSE（量纲不同不可横向排名）');ax.grid(axis='y',alpha=.3)
    fig.tight_layout();svgs['supervision']=savefig('supervision')
    return svgs

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True)
    screen,confirm,locked,analysis,runtime=map(read,('screen_results.json','confirmation_results.json','locked_design.json','analysis_results.json','runtime_manifest.json'))
    suites=ET.parse(ROOT/'tests.xml').getroot().findall('testsuite');tests=sum(int(s.attrib['tests']) for s in suites);assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    svgs=figures(screen,confirm,analysis)
    rows=[]
    for name,r in screen.items():
        interval=analysis['screen'][name]['paired_vs_R'];rows.append([NAMES[name],r['config']['auxiliary'],r['config']['budget'],r['parameters'],pct(r['results']['validation']['skill']),pct(r['results']['replay']['skill']),f"{pct(interval['increment'])} [{pct(interval['ci95'][0])}, {pct(interval['ci95'][1])}]",r['budget_violations_all_batches']])
    score_rows=[];decomp_rows=[];nav_rows=[]
    for fold in ('f1','f2'):
        for split in ('validation','replay'):
            for name,result in analysis['folds'][fold]['splits'][split].items():
                d=result['metrics'];ci=result['interval_vs_zero']['ci95'];score_rows.append([fold,split,VARIANTS[name],pct(d['skill']),f'[{pct(ci[0])}, {pct(ci[1])}]',f"{d['correlation']:+.4f}"])
                decomp_rows.append([fold,split,VARIANTS[name],pct(d['covariance_gain']),pct(d['mean_alignment']),pct(d['variance_penalty']),pct(d['skill'])])
        for name,fees in analysis['folds'][fold]['stream'].items():
            d=fees['4'];nav_rows.append([fold,VARIANTS[name],pct(d['gross_return']),pct(d['total_return']),pct(d['max_drawdown']),f"{d['annualized_sharpe']:.2f}",f"{d['mean_gross_exposure']:.3f}",pct(fees['2']['total_return']),pct(fees['8']['total_return'])])
    full_rows=[];probe_rows=[]
    for name,r in confirm.items():
        full_rows.append([name,r['parameters'],f"{r['unique_train_hours']:,}/{r['train_hours']:,}",r['total_steps'],pct(r['results']['validation']['skill']),pct(r['results']['replay']['skill']),f"{r['max_shared_parameter_ratio_all_batches']:.6f}",r['budget_violations_all_batches']])
        if r['probe']:
            p=r['probe'];probe_rows.append([name,f"{p['original_head_max_abs_error']:.2e}",p['ridge'],f"{p['effective_rank']:.2f}",pct(p['results']['validation']['skill']),pct(p['results']['replay']['skill'])])
    cal_rows=[]
    geometry_rows=[]
    shrink_rows=[]
    for fold in ('f1','f2'):
        m=analysis['folds'][fold]['second_stage'];c=m['calibrator'];cal_rows.append([fold,c['penalty'],c['statewise'],f"{c['slope']:.4f}",'/'.join(f'{x:.3f}' for x in c['state_slopes']),f"{c['intercept']:+.5f}",m['filter']['unit'],m['filter']['half_life']])
        for split,g in analysis['folds'][fold]['full_target_geometry'].items():
            for key in ('Q_raw','Q_transformed_standardized'):
                v=g[key];geometry_rows.append([fold,split,key,f"{v['std']:.3f}",f"{v['largest12_square_share']*100:.2f}%",f"{v['top1pct_square_share']*100:.2f}%"])
        for split in ('validation','replay'):
            for name in ('OOF_calibrated','OOF_filtered'):
                r=analysis['folds'][fold]['splits'][split][name];d=r['shrinkage_diagnostics'];shrink_rows.append([fold,split,VARIANTS[name],f"{d['signal_std_retention']*100:.2f}%",f"{d['dynamic_covariance_retention']*100:.2f}%" if d['dynamic_covariance_retention'] is not None else 'NA',pct(r['increment_over_causal_baseline'])])
    proposed=confirm['f1_mechanism_proposal'];selected=locked['selected_run']
    positive=all(analysis['folds'][f]['splits']['replay']['OOF_filtered']['metrics']['skill']>0 for f in ('f1','f2'))
    conclusion='冻结折外处理在两折后续窗口均取得正预测skill，但条件区间与选择边界仍须保留。' if positive else '修正后的框架仍未在两折后续窗口一致保留预测改善；研究合同更清楚，市场优势尚未成立。'
    trace=[['辅助尺度','原始/TA/asinhQ/尾部；训练折变换；自身误差与尾部贡献','已实现、训练；主收益MSE不变'],['共享预算','全部实际共享参数，逐批逐参数张量≤0.3；任务独立裁剪','所有硬预算批次零违反'],['均值与动态','精确skill三项分解；市场/残差/逐币/状态/尾部','全部窗口实际计算'],['样本总体','在线可观察逐币资格，未来零成交仅敏感性','十二币原始源与主网格保留'],['表示诊断','原checkpoint头重建；不惩罚截距；训练内部选岭','机器精度重建与真实探针成绩'],['慢基线与校准','成熟5h、30d衰减、强岭；真实时间折外预测','未将负增量分支强塞入主候选'],['事件与分钟','因果冲击/修复；共同近期摘要；36patch GRU','控制信息差异；完整方案另作全覆盖'],['期限与信号','5个非重叠对数收益段；真实小时滞后；两种平滑量纲','不相加简单收益均值，不累积预测水平'],['净值诊断','固定四相位、延迟5m、非重叠小时计账、费用敏感性','真实连续小时预测填补purge间隔'],['工件与自查','训练依赖源码、实际源/缓存内容比对、前缀等价','旧v4身份不补造；新工件独立保存']]
    md=f'''# 加密货币深度学习 v5：监督几何、状态适配与事件寿命研究

设计依据：[独立评审](V4_INDEPENDENT_RESEARCH_REVIEW_2026-10-04.md)，以及[锁定研究合同](V5_RESEARCH_PROTOCOL_2026-10-04.md)。[可视化报告](../reports/review_2026-10-04/index.html)配有配置交互、净值、回撤、误差分解和机制图；[机器证据](../reports/review_2026-10-04/results.json)可复核数字。

**结论：{conclusion}** 本轮完成的是评审定位问题的修正、有限机制实验与冻结二阶段预测系统；没有把“公式更合理”当成“预测已经有效”。全部已有历史仍是探索/回放。

## 1 实际完成与研究取舍

共{len(screen)}个唯一固定预算机制实验、{len(confirm)}次完整时钟训练，其中含三个折外起点各三种子（9次）、两折各三种子（6次）、两次同架构任务对照，以及一次完整事件方案。每个完整运行三遍全部训练时刻，固定最终权重，不在外层窗口早停挑权重。{tests}项合同测试通过。

{mdtable(['评审问题','实现与证据','完成性质'],trace)}

保留v4所有历史工件，本轮另存`outputs/review_v5`。修正与新机制的比较分开；不做传统模型竞赛，不把撮合/杠杆工程作为研究成果。净值只是用途诊断。

## 2 标签与监督几何

主对象保持`Y4=G4/v`：整点决策，5分钟后open入场，4h后的open到期；G为简单收益，v为当时过去风险与训练折5%下限。普通MSE的条件均值含义保持，**不截尾、不删除极端行情、不把主目标换成Huber或符号类别**。

辅助分工：`log(T+0.02)`是活动；A是相对不对称；Q是平方压力。Q的变换为`asinh(Q/q0)`，q0是训练样本绝对Q中位数、下限0.02；三个变换对象分别用训练均值和标准差整理。TA方案只用活动与A。原始三项保留为失配对照。逆变换压缩Q预测不代表原Q条件均值；本轮没有如此声称。

尾部事件为训练期`|Y4|`的99%分位门槛、独立BCE头，初始事件先验1%。5个期限辅助对象是入场后0—1、1—2、2—3、3—4、4—8h的**非重叠对数收益**，按过去风险/期限缩放后asinh及训练尺度整理。前三类风险任务、尾部事件与期限辅助均是表示任务；最终主信号依然直接监督Y4。各段观测对数收益可精确求和再expm1得到累计简单收益，但各段预测条件均值不能据此未经分布处理精确复合。

{mdtable(['机制','辅助对象','预算','参数','内部选择skill','内部后续skill','相对R增量与95%自然时间块区间','硬预算违反张量次数'],rows)}

选择窗为2025-03/04，内部后续05/06未参与选择。每个筛选4×128×24时刻曝光，共512次更新，非四次全量epoch。路径、事件和基线没有因为金融直觉而被默认保留；选择结果为`{selected}`。R仍为神经网络时序模型。只删除body、只删除影线是两个独立诊断，不把五字段合并删除的差异解释为去冗余总效果。

原Q和压缩Q自身误差、训练常数误差、实际残差最大1%贡献已保存；它们衡量不同对象，不直接横向排名。图中对数轴用于显示数值几何。2025-10-10极端行情保留，源记录OHLC/成交内部合法性经过审计；没有独立交易所逐笔证据，不能声称已证明全部极端价格真实性。

{mdtable(['折','窗口','路径对象','标准差','最大12样本平方质量占比','最大1%占比'],geometry_rows)}

这些平方标签集中度用于比较监督几何，不偷换为具体模型残差损失；实际残差的尾部比例与逐任务自身skill另存在机器结果。

## 3 真正共享预算及其边界

分别求主/辅全参数梯度，独立裁剪任务梯度总范数至1。凡主/辅共同使用的参数张量，再限制合成辅梯度范数≤主梯度30%；主梯度为零的共享参数张量得到零辅助更新。这覆盖fast、context，期限任务存在时也覆盖GRU、慢背景、分钟、融合与调制。方向独有、辅助独有梯度分别处理，不再作全模型总裁剪导致跨任务重缩放。

“逐参数”指参数张量的范数，**不是每个标量坐标都按比例限制**。辅助允许提供不同方向，但其张量整体影响受限。记录全部批次违反总数与最大比；每16批记录模块范数/夹角，每64批记录各个目标在fast/context的独立梯度。Adam的自适应动量更新不等同于原始梯度比例，0.3不保证最终参数更新或长期表示影响也精确等于30%。

软预算对照是v5任务独立裁剪下的软权重对照，**不是v4全模型裁剪的逐位复现**。与v4跨版本分数的差异也同时包含样本资格、固定权重选择和优化合同修正，不能都归于某一个新模块。

## 4 数据资格、字段和事件输入

继续使用十二币23,552,640条1m记录，统一来源重建；实际源/缓存文件内容哈希与合同重新比对，改变时拒绝复用。逐币有效输入与成熟合法价格决定主样本，某币未来零成交不再使其他币同一时刻消失。最长8h+5m边界用于主要配对评分；交易连续流中补推原purge间隔，无缺口拼接。旧未来零成交口径的评分只作敏感性，比例与分数单独保存。

分钟、关键快字段和状态保存“币种×字段×月份”的原始1/50/99分位、标准化均值/RMS/截尾率，完整JSON在`field_audit_f1/f2.json`；分钟图显示每月各币最大截尾率。快关键字段为收益、body、上下影线、活动、压力、BTC共振与残差，状态覆盖风险/活动/市场。首层输入局部导数逐币逐字段保存，它是标准化坐标下的局部敏感性，不是因果贡献。clip±8的信息损失仍存在，形状/背景职责不等于完整可逆。

事件字段在最近180分钟内找到过去风险标准化最大已发生分钟冲击，记录压缩幅度、距今、相反方向修复、冲击时活动、冲击前后压力、近期延续和是否超过3倍风险；没有使用未来完整转折点或只保留未来反转样本。十二币实际切断原始数据至第780,000分钟的前缀重建，与缓存最近180分钟、144根5m局部字段及事件字段一致。市场共振仍从统一历史价序列计算，未强制中性化。

分钟共同读出保留末patch、最近12patch、全窗及四个年龄段；有无36patch GRU只改变额外递归状态，不移除共同近期信息。原四段平均保留为旧读出候选。完整事件方案有{proposed['parameters']:,}参数，并完成三次全覆盖；其外层分数只诊断机制，不反过来替换已锁定主候选。

## 5 慢基线、真实时间折外校准与收缩

慢基线用5维连续可见状态：截距、24h log风险、短长风险比、BTC4h形状、活动surprise；每币在线岭，岭先验1000，30天半衰期，输入截到±3。整点时最多更新5小时前起点的4h+5m已成熟标签。网络总输出`b+a`在原Y4上监督，b由预先固定的独立成熟历史统计产生，a不存在任意互相抵消的识别自由度。它只是一个有研究理由的候选，并非被假定能正确预测缓慢收益漂移。

校准材料来自固定epoch的过去模型：F1网络截止2024-11，11—01拟合、02选择；F2网络截止2025-11，11—12拟合、01选择。模型没有见过这些窗口标签，也不以窗口最优checkpoint生成“伪折外”预测。不同折风险单位先以各自scale恢复原收益估计，再在最终训练折尺度中重表达；不改变原折外网络输入或调用未来外层标签。

校准拟合增量斜率与强收缩截距，斜率限制[0,1]；风险/活动四个可解释状态的斜率向总体估计收缩，不搜索隐含regime。岭预算仅0.0001/0.001/0.01/0.1，截距额外10倍收缩；在最后一个训练内部折外月选择，随后用这些折外材料重新拟合并冻结。没有扣外层真实均值，没有用oracle斜率修饰正式成绩。慢基线系数固定为1的校准适用于实际配置包含基线时；不包含基线时b=0，独立基线仍单独诊断。

{mdtable(['折','岭预算','状态收缩','总体斜率','四状态斜率','截距','滤波量纲','半衰期h'],cal_rows)}

{mdtable(['折','窗口','二阶段机制','保留原信号标准差','保留原动态协方差','相对独立慢基线误差增量'],shrink_rows)}

若信号幅度与动态协方差几乎都被收缩掉，正确解释是原模型置信度不足，不能称为提取了更强方向信息。低MSE与接近零输出的稳定可以同时成立；研究目的仍是找到可迁移增量，而不是漂亮的零曲线。

滤波也只用折外材料，在0/1/2/4h低自由度范围选择；半衰期0是允许不滤波。平滑标准化优势与平滑原收益后除当前风险分别比较，冻结后不重挑。它是滚动预测降噪器，不是针对一个固定事件的最优贝叶斯更新。信号水平不累积成脉冲。

## 6 完整覆盖、原头与探针

{mdtable(['运行','参数','独特训练时刻/全体','更新步','外层观察skill','后续skill','全批次最大共享比','违反次数'],full_rows)}

F1/F2主模型分别截至2025-07/2026-02训练。三种子是初始化敏感性，不是三份独立市场。外层窗口不用于选结构、权重或二阶段参数。三个固定完整覆盖epoch与筛选固定步预算分别标明，不能互称等量epoch。

主候选总实例化参数{analysis['folds']['f1']['parameter_activity']['instantiated']:,}；启用损失连接的参数张量计数{analysis['folds']['f1']['parameter_activity']['enabled_loss_connected_tensors']:,}。未启用的风险/期限/尾部头没有被算作主目标的有效容量；同一个已连接参数张量内的零梯度坐标不另作永久失活判断。

{mdtable(['运行','原头最大重建误差','训练内部选岭','有效秩','新探针观察skill','新探针后续skill'],probe_rows)}

重建用同一保存h、原checkpoint权重/截距及已知b，验证预测与表示口径一致。新探针以训练内后20%时段选岭、成熟边界purge、截距不惩罚；编码器本身仍由整个训练集拟合，所以这是训练内部读出正则诊断，不是另一次完整折外编码器验证。新探针差不能证明表示不含原信号映射；有效秩也不是预测信息量目标。

## 7 预测结果与误差分解

{mdtable(['折','窗口','信号机制','skill','固定预测95%自然72h块区间','相关'],score_rows)}

区间按真实72小时区间抽样，不按保留行数、不圆周连接首尾；同一时刻币种先聚合。它仍是给定固定预测的条件区间，不涵盖探索/结构选择不确定性。

`skill = 2Cov(s,Y)/E[Y²] + (2μsμY−μs²)/E[Y²] − Var(s)/E[Y²]`逐窗口精确核验。

{mdtable(['折','窗口','信号机制','动态协方差','均值匹配','幅度代价','合计skill'],decomp_rows)}

正相关不足以抵消平均偏置和过大方差。所有正式成绩使用冻结处理；无极端事件、单币排除等结果仅作事后诊断，原完整主评分始终保留。机器结果还提供普通/极端样本误差贡献、最大的12个时刻、市场共同分量、截面残差、逐币、风险/活动状态和逐月结果。市场/残差分解用于认识信息来源，不强制输出市场中性。

## 8 事件次序与信息寿命

用原始观测的5个互不重叠未来收益段衡量信号关联；入场同为决策+5m。自相关使用真实UTC小时键连接1/2/4/8h，不把缺口后的下一行叫下一小时。未来段相关和信号持续性分开报告，不把相关峰值命名为最优持有期。

事件/分钟GRU/期限辅助的必要证据是相同条件下主目标及非重叠未来段增量能在时间迁移保留，并不由一次极端事件或一币支配。此次只对所列有限实现作判断；未获得增量时不能据此否定所有分钟信息或所有合理路径任务。

## 9 研究净值：用途诊断及边界

固定四相位4h信号袖套，每小时仅刷新一个等权信号槽，十二币等权、目标`tanh(s/0.10)/(4×12)`，整体目标毛敞口≤1。合成目标权重每小时再平衡，计入价格变动造成的权重漂移换手；它不是四小时固定份额完全不交易的账本。信号由整点已完成数据产生，5m后open生效；收益逐真实1h open-to-open计入，不累加重叠4h标签。费用先扣再计该小时价格收益，末端清仓换手也计费。

单边单位换手4bps为**研究假设**，2/8bps作敏感性，不是已核实的账户费率、不按最高净值选择。没有资金费、盘口滑点、成交限制与强平工程，图为恒定名义权重近似账本，不能称实盘可实现收益。gross/net和回撤同时展示，模型判断仍优先预测与机制证据。

{mdtable(['折','信号','无费用收益','4bps净收益','最大回撤','小时年化Sharpe','平均毛敞口','2bps净收益','8bps净收益'],nav_rows)}

两折曲线是各自训练起点后的连续日历流，F1为2025-07至2026-01，F2为2026-02至2026-09-24；分别从单位净值开始，不把两套模型与缺口未经解释拼成一条连续实盘策略。

## 10 复现与后续边界

实际解释器`{runtime['python']}`，PyTorch`{runtime['torch']}`，CUDA{runtime['cuda']}，{runtime['gpu']}。入口用指定conda PowerShell hook激活universal，CUDA不可用立即失败。

```powershell
./scripts/run_review_research.ps1 -Stage cache
./scripts/run_review_research.ps1 -Stage test
./scripts/run_review_research.ps1 -Stage screen
./scripts/run_review_research.ps1 -Stage confirm
./scripts/run_review_research.ps1 -Stage analysis
./scripts/run_review_research.ps1 -Stage predict
./scripts/run_review_research.ps1 -Stage report
```

训练工件绑定实际数据/配置/环境及包括`redesign_model.py`、cache与v4依赖在内的源码哈希；入口还重新读取实际源和缓存内容比对，改变就拒绝复用。训练时身份与分析/交付时身份分别保存，旧v4不补造训练时证明。完整分钟缺口拒绝入仓，字段mask不是严格跳过缺失token的RNN，扩展到稀疏/中断行情仍需另立缺失语义；当前完整市场的监督研究没有因此延迟。

最新2026-09-25 00:00 UTC决策没有未来标签，已通过CUDA三种子检查点推理与冻结折外校准/滤波调用。推理接收完整特征窗口和当时风险，不读取未来标签；调用方仍承担原始行情时间可见性和状态构建责任。启用慢基线的配置必须另提供成熟标签统计生成的b，不能回退到未来标签缓存。当前主候选未启用该分支。共同配对评分保留最长8h成熟边界，原始缓存末端未成熟的4h/8h标签不用于评分，不妨碍标签无关的当前预测。

本轮是有限明确修正和实验闭环，不是要求正收益才停止。下一研究机会应来自已定位的状态/期限增量与新到达数据，而不是再横向添加一批模型名称。所有预测、配置、变换、校准、梯度、净值与字段审计在`outputs/review_v5`，报告紧凑JSON与SVG便于复核。
'''
    DOC.write_text(md,encoding='utf-8')
    def fig(name,caption):return f'<div class="figure">{svgs[name]}<p class="caption">{caption}</p></div>'
    sections=[]
    sections.append(f'<section id="contract"><div class="eyebrow">01 / RESEARCH CONTRACT</div><h2>把研究仪器校准，再判断机制增量</h2><div class="figure">{architecture()}<p class="caption">框架拥有独立主目标、候选辅助、成熟历史基线和冻结二阶段处理；实际选型按内部证据决定。</p></div>{table(["评审问题","实现与证据","完成性质"],trace)}</section>')
    sections.append(f'<section id="supervision"><div class="eyebrow">02 / TARGET GEOMETRY</div><h2>保留主收益均值，重做路径辅助的几何</h2><div class="columns"><div class="card"><h3>主目标</h3><p>延迟5m入场的4h简单收益 / 过去风险；普通MSE、不截尾、不删极端日。</p></div><div class="card"><h3>辅助目标</h3><p>活动、相对不对称、训练尺度asinhQ、独立尾部事件与非重叠期限；每个头有不同含义。</p></div></div>{fig("supervision","各自目标数值尺度不同；压缩Q的逆变换不代表原Q条件均值。")}{fig("hypotheses","512更新的同预算内部折外研究；只在03/04选型，05/06保留。")}{table(["机制","辅助","预算","参数","选择skill","后续skill","相对R与95%区间","违反"],rows)}<label for="experiment-select">查看配置的研究证据</label><select id="experiment-select">'+''.join(f'<option value="{k}">{NAMES[k]}</option>' for k in screen)+'</select><div id="detail"></div></section>')
    sections[-1]=sections[-1].replace('<label for="experiment-select">',table(['折','窗口','Q对象','标准差','最大12平方占比','最大1%占比'],geometry_rows)+'<label for="experiment-select">')
    sections.append(f'<section id="budget"><div class="eyebrow">03 / SHARED GRADIENTS</div><h2>每一批约束全部实际共享参数</h2><p>任务梯度独立裁剪；共享参数张量的辅助范数≤主梯度30%，不以全模型裁剪重新耦合。期限头加入时，预算也覆盖GRU、慢背景、分钟与融合。</p>{fig("optimization","硬预算全批次零违反；0.3是梯度张量比例，不是Adam参数更新或长期表示影响的比例。")}{table(["运行","参数","全量时刻覆盖","更新","观察skill","后续skill","最大共享比","违反"],full_rows)}</section>')
    sections.append(f'<section id="calibration"><div class="eyebrow">04 / MATURED HISTORY & OOF</div><h2>慢基线、偏置和可靠度分开处理</h2><p>只使用至少5小时前已成熟标签。二阶段机制从真正按时间产生的折外预测学习，冻结外层评价。允许收缩至零，但不将牺牲动态信息包装为成功。</p>{table(["折","岭","状态收缩","总体斜率","四状态斜率","截距","平滑量纲","半衰期h"],cal_rows)}{fig("decomposition","skill精确拆成动态协方差、均值匹配与幅度代价。正相关与负skill可以同时出现。")}{table(["折","窗口","信号","skill","自然72h块95%区间","相关"],score_rows)}</section>')
    sections[-1]=sections[-1].replace('</section>', '<div class="note">第一折校准斜率仅0.0245，动态幅度被大幅收缩；接近零的误差和净值不能解释为强预测信号。</div>'+table(['折','窗口','信号','标准差保留','动态协方差保留','相对成熟基线增量'],shrink_rows)+'</section>')
    sections.append(f'<section id="representation"><div class="eyebrow">05 / EVENT ORDER & LIFETIME</div><h2>问具体时间关系，保留正确的信息</h2><p>因果冲击、距今、修复、活动与前后压力；共同近期摘要之上比较36patch GRU。非重叠未来段与自相关分别按真实UTC时钟核验。</p>{fig("lifetime","入场同为决策+5m；0—1、1—2、2—3、3—4、4—8h互不重叠，不是移位的累计4h标签。")}{table(["运行","原头重建误差","内部选岭","有效秩","探针观察skill","探针后续skill"],probe_rows)}<div class="note">原头能从保存表示重建；重新拟合探针差不能证明表示里没有原信号。输入导数与反序列敏感性也不等于因果贡献。</div>{fig("field_drift","逐币×通道×月份原分布、标准化中心/RMS和截尾全部另存。图示各月最大截尾率。")}{fig("monthly","外层窗口未选择结构、权重或校准参数。所有历史依然为探索与回放。")}</section>')
    sections.append(f'<section id="nav"><div class="eyebrow">06 / RESEARCH EQUITY</div><h2>净值是用途检验，不能替代预测证据</h2><div class="note">四相位4h、十二币等权、tanh(s/0.10)、毛敞口≤1；延迟5m open、真实非重叠1h价格计账。单边4bps是固定研究假设。未含资金费与盘口滑点。</div>{fig("nav","各折从1开始，连续时钟含purge间隔的实际补推，末端清仓换手计费。")}{fig("drawdown","费用、幅度和时间聚合都会影响净值；图不是实盘可实现收益承诺。")}{table(["折","信号","gross","4bps net","回撤","Sharpe","毛敞口","2bps net","8bps net"],nav_rows)}</section>')
    sections.append(f'<section id="delivery"><div class="eyebrow">07 / VERIFICATION & HANDOFF</div><h2>结论、源码与工件各自可追溯</h2><p>{tests}项合同测试通过；十二币实际原始前缀重建通过；训练依赖源码与真实输入哈希绑定，来源变化拒绝复用。</p><p>{html.escape(runtime["python"])}<br>{html.escape(runtime["gpu"])} · CUDA {runtime["cuda"]} · PyTorch {runtime["torch"]}</p><div class="columns"><div class="card"><h3>完整Markdown研究报告</h3><p>定义、研究取舍、全部数值与复现步骤。</p><a href="../../docs/V5_RESEARCH_REPORT_2026-10-04.md">打开研究文档 →</a></div><div class="card"><h3>机器证据与SVG</h3><p>实际训练、折外校准、预测分解、净值及九张可导出图。</p><a href="results.json">下载完整结果 →</a></div></div><p>{conclusion}已有历史不能被重新命名为方案锁定后新到达的独立确认。</p></section>')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth;color-scheme:dark}body{margin:0;background:#091422;color:#edf4ff;font:15px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}a{color:#77baff;text-decoration:none}a:hover{text-decoration:underline}.shell{display:grid;grid-template-columns:230px minmax(0,1fr);max-width:1700px;margin:auto}aside{position:sticky;top:0;height:100vh;padding:32px 24px;border-right:1px solid #293b52}.brand{letter-spacing:3px;font-size:20px}.version{color:#f6c37b;font-size:12px;margin:14px 0 25px}nav a{display:block;color:#b5c8df;padding:10px;border-radius:8px;margin-bottom:5px}nav a:hover{background:#19304a;text-decoration:none}main{min-width:0;padding:48px 48px 70px}.eyebrow{font-size:12px;color:#77baff;letter-spacing:2px}h1{font-size:43px;line-height:1.35;letter-spacing:1px;margin:16px 0 20px}h2{font-size:27px;line-height:1.45;margin:8px 0 18px}h3{font-size:18px;margin:0 0 12px}p{color:#b7c9df}.lead{font-size:18px;max-width:950px}.chips span{display:inline-block;font-size:12px;border:1px solid #3b536f;padding:5px 12px;border-radius:30px;margin:4px 7px 4px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}.card{background:#101d30;border:1px solid #2a415c;border-radius:12px;padding:22px}.value{font-size:29px;color:#edf4ff}.label{font-size:12px;color:#a9bdd5}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{margin-top:60px;scroll-margin-top:22px}.figure{background:#101d30;border:1px solid #2a415c;border-radius:13px;padding:18px;margin:25px 0}.figure svg{width:100%;height:auto;display:block}.caption{font-size:13px;margin:13px 0 0}.note{border-left:3px solid #f6c37b;background:#282639;padding:17px 22px;border-radius:0 9px 9px 0;margin:22px 0;color:#dedbec}.table-wrap{overflow:auto;border:1px solid #2a415c;border-radius:10px;margin:22px 0}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}th{text-align:left;background:#19304a;padding:13px;font-weight:500}td{padding:11px 13px;border-top:1px solid #2a415c}tbody tr:nth-child(even){background:#101d30}tbody tr:hover{background:#1a3048}label{display:block;margin:24px 0 10px}select{background:#19304a;color:#edf4ff;border:1px solid #55718e;border-radius:8px;padding:12px;width:100%;font:inherit}#detail{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:15px}#detail div{background:#12263c;padding:15px;border-radius:8px}#detail span{display:block;font-size:12px;color:#a9bdd5}footer{font-size:12px;color:#9ab0cb;margin-top:60px;border-top:1px solid #2a415c;padding-top:25px}@media(max-width:1100px){main{padding:32px 25px}h1{font-size:35px}.cards{grid-template-columns:1fr 1fr}.columns{grid-template-columns:1fr}}@media(max-width:760px){.shell{display:block}aside{position:relative;height:auto;border-right:0;border-bottom:1px solid #293b52;padding:20px}nav{display:flex;overflow:auto}nav a{white-space:nowrap}.version{margin-bottom:8px}main{padding:28px 18px}h1{font-size:29px}#detail{grid-template-columns:1fr 1fr}}@media print{aside{display:none}.shell{display:block}.figure{break-inside:avoid}main{padding:12px}}'''
    payload={k:{'name':NAMES[k],'skill':pct(v['results']['validation']['skill']),'later':pct(v['results']['replay']['skill']),'parameters':v['parameters'],'auxiliary':v['config']['auxiliary'],'budget':v['config']['budget'],'max_ratio':f"{v['max_shared_parameter_ratio_all_batches']:.6f}",'violations':v['budget_violations_all_batches'],'baseline':v['config']['baseline'],'events':v['config']['events']} for k,v in screen.items()}
    js='const evidence='+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+';const selector=document.getElementById("experiment-select");function show(){const r=evidence[selector.value];const fields=[["内部选择skill",r.skill],["内部后续skill",r.later],["参数",r.parameters],["辅助对象",r.auxiliary],["预算",r.budget],["全批次最大共享比",r.max_ratio],["预算违反",r.violations],["慢基线",r.baseline],["事件字段",r.events]];document.getElementById("detail").innerHTML=fields.map(([k,v])=>`<div><span>${k}</span>${v}</div>`).join("");}selector.addEventListener("change",show);show();'
    nav=[('contract','要求与证据'),('supervision','标签与消融'),('budget','共享预算'),('calibration','状态与校准'),('representation','事件与表示'),('nav','净值与回撤'),('delivery','复现与交付')]
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>加密货币深度学习独立评审改进 · v5</title><style>'+css+'</style></head><body><div class="shell"><aside><div class="brand">CRYPTO / DL</div><div class="version">REVIEW → RESEARCH · V5</div><nav>'+''.join(f'<a href="#{k}">{v}</a>' for k,v in nav)+'</nav></aside><main><div class="eyebrow">2026.10.04 · 1 MINUTE SOURCE · CUDA</div><h1>从监督几何，<br>到可校准的动态信号。</h1><p class="lead">'+conclusion+'</p><div class="chips"><span>直接收益条件均值</span><span>真实时间折外学习</span><span>事件次序与非重叠期限</span><span>净值作为用途诊断</span></div><div class="cards">'+''.join(f'<div class="card"><div class="value">{v}</div><div class="label">{label}</div></div>' for v,label in [(len(screen),'集中机制消融'),(len(confirm),'完整覆盖训练'),(f'{tests} / {tests}','合同检查通过'),('2折 × 3种子','冻结外层历史迁移')])+'</div><div class="note">模型设计的修正已经落实；是否形成稳定市场优势由实际误差、状态分解和迁移证据回答。已有历史始终是探索与回放。</div>'+''.join(sections)+'<footer>依据《加密货币深度学习v4独立研究评审》 · 无外部CDN · 所有图由实际保存产物生成<br>净值未计资金费与真实盘口滑点；源记录内部合法性不等于独立成交真实性证明。</footer></main></div><script>'+js+'</script></body></html>'
    (OUT/'index.html').write_text(doc,encoding='utf-8')
    compact={'runtime':runtime,'locked_design':locked,'screen':{k:{x:v[x] for x in ('config','parameters','results','budget_violations_all_batches','max_shared_parameter_ratio_all_batches','target_transform')} for k,v in screen.items()},'confirmation':{k:{x:v[x] for x in ('config','parameters','unique_train_hours','train_hours','total_steps','results','probe','quality','budget_violations_all_batches','max_shared_parameter_ratio_all_batches')} for k,v in confirm.items()},'analysis':analysis,'cache_audit':read('cache/manifest.json'),'preflight':read('preflight_manifest.json'),'tests':tests}
    compact['latest_label_free_cuda_inference']=read('latest_inference.json') if (ROOT/'latest_inference.json').exists() else None
    (OUT/'results.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
    files=list(Path('src/crypto_timing').glob('review_*.py'))+list(Path('scripts').glob('*review*.py'))+[Path('scripts/run_review_research.ps1'),Path('tests/test_review_contract.py'),Path('docs/V5_RESEARCH_PROTOCOL_2026-10-04.md'),Path('README.md'),DOC,OUT/'index.html',OUT/'results.json']+list((OUT/'assets').glob('*'))
    (ROOT/'delivery_manifest.json').write_text(json.dumps({'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(f'# v5 独立评审驱动研究\n\n{conclusion}\n\n- [可视化报告](index.html)\n- [完整研究文档](../../docs/V5_RESEARCH_REPORT_2026-10-04.md)\n- [机器结果](results.json)\n\n{len(screen)}个集中机制实验、{len(confirm)}个完整覆盖训练、{tests}项合同测试。净值是未含资金费/盘口滑点的固定规则研究诊断。\n',encoding='utf-8')
    print(f'Report: {OUT}/index.html\nMarkdown: {DOC}',flush=True)
if __name__=='__main__':main()
