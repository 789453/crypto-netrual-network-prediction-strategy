"""One result identity -> Chinese Markdown, overview HTML, twelve-coin HTML."""
from pathlib import Path
import sys, json, hashlib, html, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path('outputs/signal_control_2026-10-05');OUT=Path('reports/signal_control_2026-10-05')
DOC=Path('docs/SIGNAL_CONTROL_RESEARCH_2026-10-05.md')
COLORS=['#60a5fa','#c084fc','#34d399','#fbbf24','#f472b6','#94a3b8']
LABELS={'A':'A 成本连续控制','B':'B 持续排名','C':'C 联合配置','D':'D 成交预算跟踪',
        'E':'E 强核心＋弱期参与','E_core_only':'E 移除弱期参与',
        'C_low':'C 相对预算0.4','C_high':'C 相对预算0.8','old_gate8bp':'旧设计门槛8bp','old_gate4bp':'旧设计门槛对齐4bp',
        'rank_guarded':'排名＋共同风险边界','rank_native':'原生2倍排名',
        'selected_no_friction':'候选去摩擦与惯性','selected_expiry12':'候选12h必平','selected_fee4bp':'候选账本每侧4bp',
        'selected_funding':'候选真实资金费'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(v):return f'{v*100:+.2f}%'
def rate(v):return f'{v*100:.1f}%'
def svg(name,fig):
    path=OUT/'assets'/f'{name}.svg';fig.savefig(path,bbox_inches='tight',metadata={'Date':'2026-10-05'})
    if name=='overview':fig.savefig(path.with_suffix('.png'),dpi=135,bbox_inches='tight')
    plt.close(fig);s=path.read_text(encoding='utf-8');return s[s.index('<svg'):]
def configure():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'figure.facecolor':'#101c2d',
       'axes.facecolor':'#101c2d','savefig.facecolor':'#101c2d','text.color':'#e2e8f0',
       'axes.labelcolor':'#cbd5e1','xtick.color':'#94a3b8','ytick.color':'#94a3b8','axes.edgecolor':'#334155',
       'grid.color':'#334155','axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'signal-control-20261005'})
def table_header():return '| 版本 | 净收益 | 分钟回撤 | 平均毛 | 参与率 | 均持h | 每日换手 | 低波参与 | 有效币数 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|'
def mdrow(name,r):
    d=r['diagnostics'];return f"| {LABELS[name]} | {pct(r['return'])} | {pct(r['minute_max_drawdown'])} | {r['average_gross']:.3f} | {rate(r['participation'])} | {r['holding_mean']:.1f} | {d['turnover_per_day']:.3f} | {rate(d['low_vol']['participation'])} | {d['effective_coins_nominal']:.1f} |"
def html_table(accounts,names):
    head='<tr>'+''.join(f'<th>{x}</th>' for x in ['版本','净收益','分钟回撤','平均毛','参与率','均持h','每日换手','低波参与','有效币数'])+'</tr>'
    rows=[]
    for n in names:
        r=accounts[n];d=r['diagnostics'];cells=[LABELS[n],pct(r['return']),pct(r['minute_max_drawdown']),f"{r['average_gross']:.3f}",rate(r['participation']),f"{r['holding_mean']:.1f}",f"{d['turnover_per_day']:.3f}",rate(d['low_vol']['participation']),f"{d['effective_coins_nominal']:.1f}"]
        rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
    return '<div class="table"><table>'+head+''.join(rows)+'</table></div>'
CSS='''body{margin:0;background:#081321;color:#e2e8f0;font:16px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1260px;margin:auto;padding:38px 24px}h1{font-size:34px;line-height:1.3}h2{font-size:23px;margin-top:38px}p{max-width:1060px;color:#cbd5e1}a{color:#93c5fd}svg{width:100%;height:auto}section{padding:20px;background:#101c2d;border:1px solid #26384e;border-radius:14px;margin:22px 0}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid #26384e}td:first-child,th:first-child{text-align:left}th{color:#94a3b8}.identity{color:#94a3b8;font-size:13px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:#142237;padding:18px;border-radius:12px}.card b{display:block;color:#60a5fa;font-size:26px}select{font:inherit;background:#142237;color:#e2e8f0;border:1px solid #415675;padding:10px;border-radius:8px}.panel[hidden]{display:none}@media(max-width:740px){.cards{grid-template-columns:1fr 1fr}main{padding:24px 14px}}'''
def page(title,body,identity,links):
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body><main><h1>{title}</h1><p class="identity">2026-10-05 · 结果 SHA256 {identity}</p><nav>{links}</nav>{body}</main></body></html>'


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True);configure()
    result=read(ROOT/'results.json');accounts=result['accounts'];selected=result['lock']['selected'];r=accounts[selected];d=r['diagnostics'];symbols=result['symbols']
    identity=hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest()
    paths={n:np.load(ROOT/f'account_{n}.npz') for n in accounts}
    names=['A','C','D','E','old_gate8bp','rank_guarded']
    center_names=['A','B','C','D','E','old_gate8bp','old_gate4bp','rank_guarded']
    a=paths[selected];dates=a['dates'];clock=np.r_[dates[0],dates+np.timedelta64(1,'h')]+np.timedelta64(1,'m')
    fig,axes=plt.subplots(4,1,figsize=(14,13),sharex=True,gridspec_kw={'height_ratios':[1.5,1,1,1]})
    for n,c in zip(names,COLORS):
        p=paths[n];nav=p['nav'];axes[0].plot(clock,nav,label=LABELS[n],color=c,lw=1.4)
        axes[1].plot(clock,100*(nav/np.maximum.accumulate(nav)-1),color=c,lw=.8)
        # Daily averaging only for readable overview; coin panels preserve hourly weights.
        daily=np.arange(0,len(dates)-23,24);weights=p['weights']
        avg=np.array([abs(weights[k:k+24]).sum(1).mean() for k in daily]);axes[2].plot(dates[daily],avg,color=c,lw=.85)
        axes[3].plot(dates,np.cumsum(p['turnover']),color=c,lw=1.1)
    axes[0].set_title('固定预测下的持仓控制比较 · 每侧2bp · 零资金费',loc='left',pad=18)
    axes[0].legend(ncol=3,fontsize=9);axes[0].set_ylabel('总账户净值');axes[1].set_ylabel('小时回撤 %')
    axes[2].set_ylabel('日均毛敞口');axes[3].set_ylabel('累计实际换手')
    for ax in axes:ax.grid(alpha=.35)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));fig.tight_layout();overview=svg('overview',fig)
    months=list(d['monthly']);fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    for n,c in zip(['A','D','E','old_gate8bp'],COLORS):
        m=accounts[n]['diagnostics']['monthly'];axes[0].plot(months,[m[x]['return']*100 for x in months],marker='o',color=c,label=LABELS[n])
        axes[1].plot(months,[m[x]['mean_gross'] for x in months],marker='o',color=c)
    axes[0].legend(ncol=4);axes[0].set_ylabel('月净收益 %');axes[1].set_ylabel('月均毛敞口');axes[0].axhline(0,color='#94a3b8',lw=.7)
    for ax in axes:ax.grid(alpha=.35)
    axes[1].tick_params(axis='x',rotation=45);fig.tight_layout();monthly=svg('monthly',fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5));x=np.arange(len(symbols))
    for off,n,c in zip([-.25,0,.25],['C','D','E'],COLORS):
        dx=accounts[n]['diagnostics'];axes[0].bar(x+off,np.array(dx['mean_abs_weight_by_coin'])*100,.24,color=c,label=LABELS[n]);axes[1].bar(x+off,np.array(dx['coin_net_contribution'])*100,.24,color=c)
    for ax in axes:ax.set_xticks(x,symbols,rotation=65);ax.grid(axis='y',alpha=.3)
    axes[0].legend(fontsize=8);axes[0].set_ylabel('平均绝对账户权重 %');axes[1].set_ylabel('总账户净贡献 · 初始权益百分点');fig.tight_layout();coins=svg('coin_allocation',fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for n,c in zip(['A','C','D','E','old_gate8bp'],COLORS):
        rr=accounts[n];dd=rr['diagnostics'];axes[0].scatter(dd['turnover_per_day'],rr['return']*100,s=90,color=c)
        axes[0].annotate(n,(dd['turnover_per_day'],rr['return']*100),xytext=(8,4),textcoords='offset points')
        axes[1].scatter(rr['annual_vol']*100,rr['return']*100,s=90,color=c);axes[1].annotate(n,(rr['annual_vol']*100,rr['return']*100),xytext=(8,4),textcoords='offset points')
    axes[0].set_xlabel('每日实际换手');axes[1].set_xlabel('实现年化波动 %')
    for ax in axes:ax.set_ylabel('历史净收益 %');ax.grid(alpha=.35)
    fig.tight_layout();frontier=svg('frontier',fig)
    dev=result['lock']['development'];old=accounts['old_gate4bp'];original=accounts['old_gate8bp'];no=accounts['selected_no_friction'];expiry=accounts['selected_expiry12']
    sensitivity='\n'.join(mdrow(n,accounts[n]) for n in ['C_low','C','C_high'])
    md=f'''# 固定神经信号到持仓的研究与实现

2026-10-05。结果版本 `{identity}`。行情回放 {result['start_execution']}—{result['end_execution']}，原数据时钟为 UTC；北京时间为 UTC+8。不重训、不重推理，预测/成熟校准/行情来源哈希前后完全一致。本文、综合 HTML 与分币 HTML 引用同一机器结果。

## 先给结论

已实现四种兼容当前份额账户的持仓控制。按开发期活跃候选规则选择 **{LABELS[selected]}**，其后历史净收益 **{pct(r['return'])}**、分钟最大回撤 **{pct(r['minute_max_drawdown'])}**、平均毛 **{r['average_gross']:.3f}**、参与率 **{rate(r['participation'])}**、均持 **{r['holding_mean']:.1f}h**。它是可接入的研究候选，不能据此宣布稳定 alpha 或默认替换旧主方案。

用户关注的低波动参与为 **{rate(d['low_vol']['participation'])}**，低波动平均毛 **{d['low_vol']['mean_gross']:.3f}**；普通调仓篮子至少相隔 4h，高信号不会绕过此间隔。风险缩仓即时复核。名义配置有效币数 **{d['effective_coins_nominal']:.1f}/12**，最长连续现金 **{d['longest_cash_hours']}h**，每天实际换手 **{d['turnover_per_day']:.3f}**。这些是可直接核验的行为指标，不等同于交易盈利概率。

重要负面证据：开发期全部活跃版本的收益减回撤效用小于零。无约束选择会推荐不交易的 B；为表达用户的参与要求，开发结果出现后增补参与率至少 80%、平均毛至少 .03 的活跃资格，再按原效用选择。旧锁与日志保留，未读后续完整收益选择。本候选的价值须同时评价体验/风险控制与经济收益，不能把强制活跃包装成超过现金的证据。

## 为什么排名底仓在当前信号上不宜默认成立

保存的校准 12h 预测，后续 **{rate(result['forecast_diagnostics']['near_tied_cross_section_fraction'])}** 时段横截面近乎并列；横截面平均绝对偏差的 10/50/90 分位分别为 **{', '.join(f'{v:.4f}' for v in result['forecast_diagnostics']['historical_relative_spread_bp_quantiles'])}bp**。成熟月度校准可将相对斜率收缩为零，保留共同方向。这与网络原生输出缺少排序能力不是同一个结论；本轮比较的是已锁定校准流。

原生强制排名使用 argsort，即使预测相同也强制六多六空，因此可能承载由数组次序决定的固定资产多空差异。B 使用并列平均秩和原始价差折扣，避免把这种结构当成排名 alpha。B 的后续参与率 **{rate(accounts['B']['participation'])}**、净收益 **{pct(accounts['B']['return'])}**。零仓位说明这个明确经济/配置版本在当前流上无法支持持续排名，并不证明所有原生排名策略无效。

C 仅将相对配置作为 $Pw$ 的跟踪偏好，收益项仍只有一份 $\bar\mu^\top w$；不对共同方向追加第二份信息，也不事后拼接两条净值。C 相对预算 .4/.6/.8 的结果全部保留，没有按后续赢家回改参数。若希望研究原生 4h 的相对排序，必须明确改了信号对象和期限校准，另做因果比较；本轮未悄悄回退到原生排序制造底仓。

## 数学设计如何转成有效行为

收益预测、组合目标和实际成交分开。每个新的小时预测只更新一次有界 EMA/秩状态；同一预测重复调用直接拒绝。这是持续性状态，不是独立证据累计，也不称置信度。

基本凸目标是 $\tfrac12w^\top Aw-d^\top w+\tfrac\kappa2\|w-w^-\|^2+(c+\tau)\|w-w^-\|_1+c_{{out}}\|w\|_1$。12h 收益与 $12\Sigma_{{1h}}$ 在同一规划期限，价格风险和固定名义惩罚分开。$c=2bp$ 是单边成本，$c_{{out}}=2bp$ 是滚动终端准备金，只影响决策、不重复扣账。空仓完整入场需要覆盖这两个线性代价；已有头寸的边际条件不同。没有把 12h 预测当成每小时独立收益相加。

标量无终端准备金时，$w^{{new}}=w^-+S_c(a-qw^-)/(q+\kappa)$。当前仓位的边际优势没有越过成本区间时不交易；小超额从小仓位开始，不再过门槛就归一满额。风险的方差项和正的名义惩罚共同防止低波动被逆方差放大。

B 的排名先平滑再映射，规模乘真实价差支持 $MAD/(MAD+4bp)$，弱价差的折扣不会重新归一化掉。C 添加 $.02\|Pw-v^{{rank}}\|^2/2$；D 添加明确的目标跟踪偏好，它不声称精确收益幅度可信。这些偏好可能改善参与，也可能在无优势时增加损失，必须接受净成本检验。

普通控制采用随过去波动上升的惯性、24h 衰减的真实换手额度影子价格、4h 篮子间隔和最小目标偏差。强信号到来提高有限目标，并不触发每小时追满；弱化后不机械 12h 清仓。没有将 EMA、概率、累计积分和多个反转阈值全部叠加；随机开仓未作为主方案，因为在同等预期敞口下，独立随机化没有比例手续费优势，且可能提高风险。

严格范围：快速近端法只求解无约束凸目标；后续名义投影与风险收缩是保守可行近似，不等于完整耦合约束最优解。候选被约束改变的小时比例 **{rate(r['control']['constraint_contraction_fraction'])}**。均值误差矩阵 $\Omega$ 未估计，种子分歧与市场波动都没有被称作真实置信区间。协方差按小时方差线性扩到 12h，是近似而非重叠收益的独立性证明。

## 同一风险条件的中心比较

{table_header()}
{chr(10).join(mdrow(n,accounts[n]) for n in center_names)}

“共同风险边界”指统一过去协方差风险限制，不是事后实现波动相等。不同版本实际使用风险仍不同；不以总收益直接宣称映射超额。原生排名在下表单独保留身份。

## 费用、持有和跟踪的作用

{table_header()}
{chr(10).join(mdrow(n,accounts[n]) for n in ['old_gate8bp','old_gate4bp','rank_native','selected_no_friction','selected_expiry12','selected_fee4bp','selected_funding'])}

把旧门槛从往返8bp对齐为4bp后，重新回放净收益从 **{pct(original['return'])}** 变为 **{pct(old['return'])}**，每天换手从 **{original['diagnostics']['turnover_per_day']:.3f}** 变为 **{old['diagnostics']['turnover_per_day']:.3f}**。这项变化单列，不能归因于新控制器。

候选去掉费用准备金、换手影子价格与控制惯性、并允许小时跟踪，净收益 **{pct(no['return'])}**、每天换手 **{no['diagnostics']['turnover_per_day']:.3f}**。这个联合消融检验整个摩擦控制的价值，不分别识别每个参数贡献。恢复 12h 必平后净收益 **{pct(expiry['return'])}**、均持 **{expiry['holding_mean']:.1f}h**，是持仓续存的消融。

4bp 压力仅改变真实每侧账本费用，保留锁定 2bp 决策参数；真实资金费版本同时使用过去已结算费率作预期、按实际事件扣收费用，是另一个新决策账户。原生排名没有 .60 风险收缩，不能把其收益当公平胜出；分钟超限小时数 **{accounts['rank_native']['minute_gross_limit_breach_hours']}**，候选为 **{r['minute_gross_limit_breach_hours']}**。

## 参数附近、时间和币种均衡

{table_header()}
{sensitivity}

候选低波动/高波动的净贡献分别为 **{d['low_vol']['net_pnl_initial_equity']*100:+.2f}/{d['high_vol']['net_pnl_initial_equity']*100:+.2f}** 个初始权益百分点。毛敞口时间变异系数 **{d['gross_time_cv']:.3f}**，最大单币平均名义份额 **{rate(d['max_coin_nominal_share'])}**。均匀分配是名义层面的结果，不代表十二币独立，也不代表等风险贡献；同向持仓仍受组合协方差约束。

低于旧8bp门槛入场的物理持仓段共 **{d['weak_entry_episodes']['count']}** 段，整体净贡献 **{d['weak_entry_episodes']['net_pnl_initial_equity']*100:+.2f}** 个初始权益百分点。持仓段可能后来升级为强信号，因此它不是独立弱层收益。强预测小时的净贡献 **{d['strong_signal_hours']['net_pnl_initial_equity']*100:+.2f}** 个百分点；它是按当时可见信号归类的描述，不是事后事件择优拼接。

剔除最盈利5天的候选净贡献 **{d['net_without_top5_days']*100:+.2f}** 个百分点，剔除最盈利币的净贡献 **{d['net_without_best_coin']*100:+.2f}** 个百分点。七日自然块收益区间 **{result['historical_block_intervals'][selected]['calendar_7day_block_95pct']}**；未修正开发选型，也不是新前瞻证据。逐月曲线、12币真实持仓/收益信号/贡献在两份 HTML 中完整展示。

## 当前接入结论与优化边界

已提供 `ControlDecision` 适配现有 `alpha_account.account(..., decision_adapter=...)`。每个账户使用独立状态，份额、现金、实际成交费和终端清仓由原账本统一处理。旧主策略、原模型和预测文件未被修改，也未部署资金。完整运行入口为 `scripts/run_signal_control.ps1`，可复现实验与报告。

候选是“持续小仓位参与＋有限调整”的研究实现；是否取代强事件核心，需要它在相同风险资源下保留足够净收益，同时改善参与和成本。当前开发未超过现金的证据必须保留。若活跃偏好有稳定成本，应如实表达为配置偏好；不能继续以提高参与率为目标不断搜索收益最好的一条曲线。

当前额度较保守，没有为了用满2倍预算重新放大弱信号。下一阶段有价值的新增证据是：原生相对信息为何被校准抹掉、对收益幅度/寿命的因果校验、约束活跃时用完整联合解做小样本对照、挂单成交选择偏差。这些问题先于更多阈值搜索，也不需要重训网络。此处不安排自动未来任务。

## 性能、日志与核验

六次中心开发、一次 E 开发和十六个完整账户回放，含补充核验/重复运行累计耗时 **{result['seconds_total']:.1f}s**。D 完整 {r['hours']}h 账户耗时 **{r['seconds']:.1f}s**，平均近端迭代 **{r['control']['mean_iterations']:.2f}**，最大残差 **{r['control']['max_proximal_residual']:.3g}**。复用冻结预测、每日过去协方差快照和分钟 mmap；求解仅12维、向量化软阈值、从当前仓位热启动。未用深度模型前向计算。

运行日志每720个预测时点记录日期、成交桶和残差；E 另记录强核心币数和弱仓额度。保存比较锁、开发、十六个账户、真实逐笔/持仓段、控制状态、逐月/逐币与源哈希。36项相关测试通过；凸解独立 QP 对照、因果前缀、并列秩、重复预测拒绝、风险缩仓、币种置换、实际费用换手以及核心升级寿命均纳入验证。所有账户份额/现金、逐笔费用、终端关闭和持仓段净贡献核对见 `audit.json`；完整测试结果见 `tests.xml`。

分钟风险只能用保存路径检测，当前控制每小时执行，未实现分钟可成交的风险减仓；检测少超限不等于通过强平审计。maker每侧2bp与下一分钟open是研究假设，未证明真实被动成交、滑点、订单最小量或维护保证金档位。

方法参考 [Boyd 等原研究](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) 的单周期滚动成本控制框架。本文是基于冻结项目输出的工程与历史研究，不声称解出完整动态最优交易。

[综合 HTML](../reports/signal_control_2026-10-05/index.html) · [十二币 HTML](../reports/signal_control_2026-10-05/coins.html) · [结果 JSON](../reports/signal_control_2026-10-05/results.json) · [数学设计](SIGNAL_POSITION_MAPPING_DESIGN_2026-10-05.md) · [比较合同](SIGNAL_CONTROL_PROTOCOL_2026-10-05.md)
'''
    er=accounts['E'];ed=er['diagnostics'];coreonly=accounts['E_core_only'];role=result['E_role_attribution'];risk_audit=result['signal_evidence']['minute_risk_audit']
    response_rows=[]
    for part in ['development_matured','inspected_history']:
        for item in result['signal_evidence'][part]:
            vals=item['mean_signed_realised_bp_by_2_4_8_12h']
            response_rows.append('| '+('成熟开发' if part=='development_matured' else '已见后续')+' | '+item['bucket']+' | '+str(item['distinct_clocks'])+' | '+' | '.join(f'{v:+.2f}' for v in vals)+' |')
    supplement=f'''## 当前兼容方案 E：保留强核心，给弱期有限参与

**从用户需求的完整兼容性看，建议以 E 作为新增研究接入版本，D 作为低换手参照。** E 在 A—D 的完整历史已见后提出，只增加一套固定设计，没有搜索弱预算比例。D 的原开发锁保持不变，E 不冒充它在未见后续中的开发胜者。E 开发净收益 {pct(result['extension']['development']['account']['return'])}；后续净收益 **{pct(er['return'])}**、分钟回撤 **{pct(er['minute_max_drawdown'])}**。这组结果只能叫探索性历史证据。

经济职责分开：强核心保留旧 8bp 成本资格、风险预算、边际支持消失与 12h 核心复核；弱期采用 D 的跟踪结构，但把名义预算缩为常态额度的 10%，正常弱额度最多 .2、净 .1、单币 .075，且不要求用满。缩放同时作用于二次惩罚与目标，避免重新归一满额。核心和弱期只有一个净头寸、一个现金份额账本，先合成再执行共同风控和收费。

所有普通加仓/弱期调整共享 **4h 篮子间隔**，实际最小间隔已核对为 **{role['minimum_normal_basket_spacing_hours']}h**；核心失效、到期和安全缩仓可以立即发生，因此不是任意两筆成交都必须间隔4h。强信号升级时，核心12h寿命从真实升级时点开始，物理持仓寿命仍从最初小仓位入场开始，不伪造平仓或开仓。E 仍保留核心到期；没有把每个已有暴露自动改名为永久底仓。

{table_header()}
{chr(10).join(mdrow(n,accounts[n]) for n in ['old_gate8bp','D','E_core_only','E'])}

E 平均毛 **{er['average_gross']:.3f}**，低波动参与 **{rate(ed['low_vol']['participation'])}**、低波动平均毛 **{ed['low_vol']['mean_gross']:.3f}**，名义有效币数 **{ed['effective_coins_nominal']:.1f}/12**，最长现金 **{ed['longest_cash_hours']}h**。每天换手 **{ed['turnover_per_day']:.3f}**，比旧8bp主方案低 **{(1-ed['turnover_per_day']/original['diagnostics']['turnover_per_day'])*100:.1f}%**；这才是保留强事件能力时更切题的比较。E 并不达到 D 的极低换手，风险利用也明显不同。

**参与口径不能隐藏微小残仓**：总表“参与率”使用毛敞口大于百万分之一的物理非零标准，不能解释为每时刻都有充分风险投入。按毛敞口至少净资产1%，E覆盖率 **{rate(ed['participation_gross_at_least_1pct'])}**、D **{rate(d['participation_gross_at_least_1pct'])}**、旧主方案 **{rate(original['diagnostics']['participation_gross_at_least_1pct'])}**；按至少3%，E为 **{rate(ed['participation_gross_at_least_3pct'])}**。这些是描述阈值，不回改入场规则，也不假称交易所最小订单。E毛敞口时间变异系数 **{ed['gross_time_cv']:.3f}**，仍然有强事件集中；提高时间覆盖不等于每月额度均匀。

剥离弱期后 E 核心版本净收益 **{pct(coreonly['return'])}**、参与率 **{rate(coreonly['participation'])}**。这个消融改变了弱期持仓，也改变共享普通时钟的占用，E 与核心版的收益差不能全称为独立弱 alpha。按实际角色分账，核心净贡献 **{role['core_net']*100:+.2f}**、弱期净贡献 **{role['weak_net']*100:+.2f}** 个初始权益百分点，弱期均毛 **{role['weak_mean_gross']:.4f}**。角色收入/费用和总账户完全对齐；升级转移的旧份额不是独立策略再投资。

E 剔除最盈利5天的净贡献 **{ed['net_without_top5_days']*100:+.2f}** 个百分点，说明还须看尾部集中。分钟峰值毛 **{risk_audit['E']['maximum_minute_gross']:.3f}**，超过3倍硬额度小时 **{risk_audit['E']['hours_above_hard_cap3']}**；小时控制之外的分钟路径只是检测，并不等于分钟可执行减仓或强平审计。E 当前默认零资金费，没有为它补做完整真实资金费/滑点压力，D 的资金费情景不能代替 E。

## 持有期限的实现响应，而非预测自相关

下表以入场时预测方向乘未来2/4/8/12h实际收益，分组门槛完全依据入场时可见幅度。开发只保留在2025-07-01前已成熟的12h目标；后续只作诊断，不回改映射。均值单位bp，时点与币种高度相关，不把样本行数当作独立证据。

| 期间 | 入场预测幅度 | 不同预测时点 | 实现2h | 实现4h | 实现8h | 实现12h |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(response_rows)}

这是无风险等权方向响应，不是策略净收益或经过择优的胜率。若各期限响应不单调，不能仅凭预测持续推断收益寿命；若弱组均值没有覆盖完整4bp，其小仓位参与的依据就应是延续/配置偏好和整体成本结果，不能称为必然正期望。当前版本没有据此估计新衰减参数、重训练或动态回选期限。

'''
    md=md.replace('## 为什么排名底仓在当前信号上不宜默认成立',supplement+'## 为什么排名底仓在当前信号上不宜默认成立')
    DOC.write_text(md,encoding='utf-8')
    body=f'<p>冻结模型与12h成熟投影，只研究配置与成交。兼容研究实现：<b>E 强核心＋弱期参与</b>；原开发锁仍为<b>{LABELS[selected]}</b>。E 在A—D历史结果已见后提出，是探索性扩展。旧主方案没有被自动替换。</p>'
    body+='<div class="cards">'+''.join(f'<div class="card">{label}<b>{value}</b></div>' for label,value in [('E历史净收益',pct(er['return'])),('E低波动参与',rate(ed['low_vol']['participation'])),('E每日换手',f"{ed['turnover_per_day']:.3f}"),('E分钟回撤',pct(er['minute_max_drawdown']))])+'</div>'
    body+='<h2>中心版本与共同风险边界</h2>'+html_table(accounts,center_names)+'<section>'+overview+'</section>'
    body+='<h2>保留强核心与弱期参与的兼容扩展</h2>'+html_table(accounts,['old_gate8bp','D','E_core_only','E'])
    body+=f'<p>E以10%弱期预算填补空仓，普通加仓/调整共用4h时钟，安全减仓可立即发生。实际弱角色净贡献{role["weak_net"]*100:+.2f}个百分点；整体差额还包含核心时钟变化，不能全称为弱alpha。核心到期仍保留12h。</p>'
    body+=f'<p>物理非零参与率会计入微小残仓。总毛敞口至少净资产1%时，E覆盖{rate(ed["participation_gross_at_least_1pct"])}、D覆盖{rate(d["participation_gross_at_least_1pct"])}、旧主方案{rate(original["diagnostics"]["participation_gross_at_least_1pct"])}。E毛敞口时间变异系数{ed["gross_time_cv"]:.3f}，尚不能称每月额度均匀。</p>'
    body+='<h2>收益、风险与换手的比较</h2><p>过去协方差约束统一，实际风险利用率仍然不同。这张图是历史经济效率比较，不宣称各点都是最优前沿。</p><section>'+frontier+'</section>'
    body+='<h2>横截面信息与四种设计</h2><p>'+html.escape(f"后续{rate(result['forecast_diagnostics']['near_tied_cross_section_fraction'])}时段的校准预测近乎并列。排名必须保留真实间距；缺少价差时不应强制六多六空。A依据绝对收益，B依据排名配置偏好，C合并一份收益与相对偏好，D依据有限目标跟踪。")+'</p>'
    body+='<h2>费用与持有消融</h2>'+html_table(accounts,['old_gate8bp','old_gate4bp','rank_native','selected_no_friction','selected_expiry12','selected_fee4bp','selected_funding'])
    body+='<p>原生排名未使用共同风险缩放，身份单列。4bp压力仅变真实账本费率；资金费情景重新决策/结算。去摩擦是联合消融，不能逐项识别各参数。</p>'
    body+='<h2>逐月收益和名义使用</h2><section>'+monthly+'</section><h2>币种名义配置与真实净贡献</h2><section>'+coins+'</section>'
    body+='<h2>参数附近与研究局限</h2>'+html_table(accounts,['C_low','C','C_high'])
    body+=f'<p>候选约束收缩比例{rate(r["control"]["constraint_contraction_fraction"])}。凸解＋风险收缩是明确的工程近似，不是完整约束最优解；没有实测均值置信区间。剔除最好5天净贡献{d["net_without_top5_days"]*100:+.2f}个百分点。历史已见，maker成交能力、冲击与强平未验证。</p>'
    body+=f'<p class="identity">累计研究运行{result["seconds_total"]:.1f}s · 冻结来源 SHA256 前后一致 · 16个账本守恒核验 · 36项测试通过 · 无训练与推理</p>'
    (OUT/'index.html').write_text(page('连续信号到持仓：研究与比较',body,identity,'<a href="coins.html">十二币图集</a> · <a href="../../docs/SIGNAL_CONTROL_RESEARCH_2026-10-05.md">完整研究文档</a> · <a href="results.json">机器结果</a>'),encoding='utf-8')
    # Interactive controls switch fixed, exported research panels; no remote assets.
    coin_body='<p>每币只有一个实际净头寸。净值贡献为 1＋12×累计该币净损益，其均值严格等于总账户净值；它不是独立再投资子策略。持仓权重以总账户权益计，保留完整小时路径。所有日期为原始 UTC，成交为下一分钟。</p>'
    options=''.join(f'<option value="{j}">{s}</option>' for j,s in enumerate(symbols))
    coin_body+='<label for="coin">选择合约 </label><select id="coin">'+options+'</select>'
    panels=[]
    compare=['E','D','C','old_gate8bp','rank_guarded']
    frozen=np.load(Path('outputs/alpha_strategy')/'unified_predictions.npz')
    unchanged_mu=frozen['Frozen_v5_calibrated'][:,:,3]
    for j,symbol in enumerate(symbols):
        fig,axes=plt.subplots(4,1,figsize=(14,12),sharex=True,gridspec_kw={'height_ratios':[1.25,1,1,1]})
        for n,c in zip(compare,COLORS):
            p=paths[n];gain=p['pnl'][:,j]+p['funding'][:,j]-p['fees'][:,j]
            axes[0].plot(clock,1+len(symbols)*np.r_[0,np.cumsum(gain)],color=c,lw=1.1,label=LABELS[n])
            axes[1].step(dates,p['weights'][:,j]*100,where='post',color=c,lw=.75,alpha=.85)
        p=paths[selected];axes[2].plot(dates,p['filtered_mu'][:,j]*1e4,color='#60a5fa',lw=.7,label='持续层平滑12h收益 · D / E弱期')
        mu=unchanged_mu[:,j]
        axes[2].plot(dates,mu*1e4,color='#94a3b8',alpha=.5,lw=.6,label='未变原12h校准收益')
        axes[2].axhline(8,color='#fbbf24',ls='--',lw=.6);axes[2].axhline(-8,color='#fbbf24',ls='--',lw=.6)
        axes[3].plot(dates,np.cumsum(p['traded_notional'][:,j]),color='#34d399',label='该币累计实际成交名义 / 初始权益')
        axes[0].set_title(f'{symbol} · 总账户内净贡献、真实持仓与冻结收益信号',loc='left',pad=18)
        axes[0].legend(ncol=3,fontsize=8);axes[2].legend(fontsize=8);axes[3].legend(fontsize=8)
        for ax,label in zip(axes,['等份贡献净值','账户有符号权重 %','预测累计收益 bp','累计成交名义']):ax.set_ylabel(label);ax.grid(alpha=.35)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));fig.tight_layout();plot=svg(f'coin_{symbol}',fig)
        rows='<tr><th>版本</th><th>净贡献·百分点</th><th>平均绝对权重</th></tr>'
        for n in compare:
            dx=accounts[n]['diagnostics'];rows+=f'<tr><td>{LABELS[n]}</td><td>{dx["coin_net_contribution"][j]*100:+.3f}</td><td>{dx["mean_abs_weight_by_coin"][j]*100:.3f}%</td></tr>'
        panels.append(f'<section class="panel" data-coin="{j}" '+('' if j==0 else 'hidden')+'>'+plot+'<div class="table"><table>'+rows+'</table></div></section>')
    coin_body+=''.join(panels)+'<script>document.getElementById("coin").addEventListener("change",e=>{document.querySelectorAll(".panel").forEach(p=>p.hidden=p.dataset.coin!==e.target.value);});</script>'
    (OUT/'coins.html').write_text(page('十二币：真实持仓与收益贡献',coin_body,identity,'<a href="index.html">综合分析</a> · <a href="../../docs/SIGNAL_CONTROL_RESEARCH_2026-10-05.md">完整研究</a>'),encoding='utf-8')
    (OUT/'results.json').write_bytes((ROOT/'results.json').read_bytes())
    (OUT/'audit.json').write_bytes((ROOT/'audit.json').read_bytes())
    manifest={'result_sha256':identity,'markdown':str(DOC),'overview':str(OUT/'index.html'),'coins':str(OUT/'coins.html'),
              'assets':len(list((OUT/'assets').glob('*.svg'))),'selected':selected,'no_external_assets':True}
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
