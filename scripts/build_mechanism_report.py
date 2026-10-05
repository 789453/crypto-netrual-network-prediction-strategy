"""Evidence-driven Chinese Markdown and a self-contained visual research report."""
from __future__ import annotations

import html
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT=Path("outputs/mechanism")
OUT=Path("reports/mechanism_2026-10-04")
DOC=Path("docs/V4_MECHANISM_RESEARCH_REPORT_2026-10-04.md")
PALETTE=["#5ad9c2","#ab9aff","#f3b36c","#78adf6"]
STAGE_LABELS={"preprocessing":"字段与尺度", "redundancy":"特征去冗余", "label":"监督对象",
              "sharing":"任务共享", "memory":"事件记忆", "width":"局部宽度",
              "state_modulation":"背景调制", "slow_background":"慢背景覆盖", "minute_information":"分钟增量"}
DESCRIPTIONS={"raw":"统一静态稳健缩放","field":"字段适配与静态整理","dynamic":"因果形状与背景状态",
              "return":"连续收益 R","path":"纯路径 P（Q 代理）","joint":"路径辅助直接收益 P+R",
              "distribution":"统一八类分布","four_bin":"四档条件方向与独立状态幅度",
              "mean":"TCN 与局部摘要","ordered":"TCN 与有序分段","gru":"TCN 与单层 GRU","lstm":"TCN 与单层 LSTM",
              "partial":"底层共享／上层分开","isolated":"完全隔离","full":"完全共享",
              "none":"五分钟主体","aggregate":"分钟路径聚合","encoder":"180 分钟短编码器"}


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def pct(x):
    return f"{x*100:+.3f}%"


def display_name(name):
    concise={"prep_raw":"静态稳健","prep_field":"字段适配","prep_dynamic":"因果双路",
             "features_original":"原16字段","features_all":"扩展19字段","features_selected":"精选14字段","features_random":"随机14字段",
             "label_return":"连续收益R","label_path":"纯路径Q","label_joint":"路径辅助P+R","label_distribution":"统一分布","label_four_bin":"四档分解",
             "sharing_isolated":"参数隔离","sharing_partial":"部分共享","sharing_full":"完全共享",
             "memory_mean":"局部摘要","memory_ordered":"有序分段","memory_gru":"TCN+GRU","memory_lstm":"TCN+LSTM",
             "modulation_False":"无调制","modulation_True":"轻调制","minute_none":"5m主体","minute_aggregate":"分钟聚合","minute_encoder":"分钟编码"}
    if name in concise:
        return concise[name]
    key=name.split("_")[-1]
    return DESCRIPTIONS.get(key,name.replace("_"," "))


def style():
    plt.rcParams.update({"figure.facecolor":"#101b2c","axes.facecolor":"#101b2c","savefig.facecolor":"#101b2c",
                         "text.color":"#edf2fa","axes.labelcolor":"#b9c8dc","xtick.color":"#b9c8dc","ytick.color":"#b9c8dc",
                         "axes.edgecolor":"#3e5069","grid.color":"#32445c","axes.spines.top":False,"axes.spines.right":False,
                         "font.family":"Microsoft YaHei","font.size":10,"svg.hashsalt":"crypto-mechanisms-v4"})


def save(name):
    path=OUT/"assets"/f"{name}.svg"
    plt.savefig(path,bbox_inches="tight",metadata={"Date":"2026-10-04"})
    plt.close()
    source=path.read_text(encoding="utf-8")
    return source[source.index("<svg"):]


def figures(screen,stages,confirmed,ensemble,signals,feature):
    style()
    svgs={}
    fig,axes=plt.subplots(3,3,figsize=(15,11.2))
    for ax,stage in zip(axes.flat,stages):
        candidates=stage["candidates"]
        v=[100*screen[n]["results"]["validation"]["skill"] for n in candidates]
        bars=ax.barh(np.arange(len(v)),v,color=[PALETTE[0] if n==stage["selected"] else "#7185a5" for n in candidates],height=.52)
        ax.set_yticks(np.arange(len(v)),[display_name(n) for n in candidates],fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0,color="#8c9db8",lw=.8)
        ax.set_title(STAGE_LABELS[stage["stage"]],loc="left",pad=12)
        ax.set_xlabel("4h 收益 MSE skill（%）",fontsize=8)
        ax.grid(axis="x",alpha=.3)
        ax.margins(x=.2)
    fig.suptitle("逐项消融：每一组只改变对应机制，组间锚点按顺序更新",x=.03,ha="left",fontsize=15)
    fig.tight_layout(rect=[0,0,1,.96],w_pad=3,h_pad=3)
    svgs["ablations"]=save("ablations")
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,fold in zip(axes,("f1","f2")):
        for j,split in enumerate(("validation","replay")):
            value=ensemble[fold][split]["skill"]*100
            interval=np.array(ensemble[fold][split]["uncertainty_vs_zero"]["ci95"])*100
            ax.plot([j,j],interval,color=PALETTE[j],lw=3)
            ax.scatter([j],[value],color=PALETTE[j],s=65,zorder=3)
        ax.axhline(0,color="#8c9db8",lw=.8)
        ax.set_xticks([0,1],["四个月早停窗","后续历史诊断窗"])
        ax.set_ylabel("三种子均值信号 MSE skill（%）")
        ax.set_title(f"{fold.upper()} · 72 小时时间块区间",loc="left")
        ax.grid(axis="y",alpha=.3)
    fig.tight_layout()
    svgs["confirmation"]=save("confirmation")
    rows=[]
    months=[]
    for fold in ("f1","f2"):
        combined={**ensemble[fold]["validation"]["monthly"],**ensemble[fold]["replay"]["monthly"]}
        months+=list(combined)
        rows.append(combined)
    months=sorted(set(months))
    matrix=np.array([[100*r.get(m,{}).get("skill",np.nan) for m in months] for r in rows])
    cmap=LinearSegmentedColormap.from_list("skill",["#b085d8","#192a40","#54d4bd"])
    fig,ax=plt.subplots(figsize=(14,3.2))
    limit=max(np.nanmax(np.abs(matrix)),.2)
    im=ax.imshow(matrix,cmap=cmap,vmin=-limit,vmax=limit,aspect="auto")
    ax.set_xticks(range(len(months)),months,rotation=35,ha="right",fontsize=8)
    ax.set_yticks([0,1],["F1 冻结模型","F2 更新模型"])
    for i in range(2):
        for j in range(len(months)):
            if np.isfinite(matrix[i,j]):
                ax.text(j,i,f"{matrix[i,j]:+.2f}",ha="center",va="center",fontsize=8,color="#ffffff")
    fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label="月度 skill %")
    ax.set_title("跨月稳定性：同一折内的三种子均值，空白为未评价月份",loc="left",pad=15)
    fig.tight_layout()
    svgs["monthly"]=save("monthly")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for k,name in enumerate(("return","joint","proposal")):
        history=json.loads((ROOT/"confirm"/f"f1_{name}_seed20261004"/"history.json").read_text())
        epochs=[h["epoch"] for h in history]
        axes[0].plot(epochs,[h["train_primary"] for h in history],marker="o",color=PALETTE[k],label=name)
        axes[1].plot(epochs,[100*h["validation"]["skill"] for h in history],marker="o",color=PALETTE[k],label=name)
    axes[0].set_title("完整时钟训练 · 主收益 MSE",loc="left")
    axes[1].set_title("同一训练轨迹 · 时间外主信号",loc="left")
    for ax in axes:
        ax.set_xlabel("完整训练轮次")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
    axes[1].set_ylabel("MSE skill %")
    axes[1].axhline(0,color="#8c9db8",lw=.8)
    fig.tight_layout()
    svgs["training"]=save("training")
    fig,ax=plt.subplots(figsize=(10.5,9))
    names=list(feature["names"])
    im=ax.imshow(feature["pearson"],cmap=cmap,vmin=-1,vmax=1)
    ax.set_xticks(range(20),names,rotation=65,ha="right",fontsize=8)
    ax.set_yticks(range(20),names,fontsize=8)
    ax.set_title("训练期字段相关性 · return_copy 仅供诊断，从未进入候选网络",loc="left",pad=18)
    fig.colorbar(im,ax=ax,fraction=.03,pad=.03,label="Pearson")
    fig.tight_layout()
    svgs["redundancy"]=save("redundancy")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for fold,ax in zip(("f1","f2"),axes):
        results=signals[fold]["splits"]["replay"]
        names=["raw","ema_1h","ema_2h","ema_4h","learned"]
        x=[100*results[n]["adjacent_flip_rate"] for n in names]
        y=[100*results[n]["skill"] for n in names]
        for i,name in enumerate(names):
            ax.scatter(x[i],y[i],s=60,color=PALETTE[0] if name=="raw" else PALETTE[1])
            ax.annotate(name,(x[i],y[i]),xytext=(5,5),textcoords="offset points",fontsize=8)
        ax.axhline(0,color="#8c9db8",lw=.8)
        ax.set(xlabel="相邻信号符号翻转率 %",ylabel="同一 4h 对象 MSE skill %",title=f"{fold.upper()} · 平滑的预测与翻转取舍")
        ax.grid(alpha=.3)
    fig.tight_layout()
    svgs["filter_tradeoff"]=save("filter_tradeoff")
    fig,axes=plt.subplots(2,1,figsize=(13,6.2),sharex=True,gridspec_kw={"height_ratios":[3,1]})
    ex=signals["f2"]["splits"]["replay"]["example"]
    for name,c in (("raw",PALETTE[0]),("ema_2h",PALETTE[1]),("learned",PALETTE[2])):
        axes[0].plot(ex[name],label=name,color=c,lw=1.5)
    axes[0].axhline(0,color="#8c9db8",lw=.8)
    axes[0].legend(frameon=False,ncol=3)
    axes[0].set_ylabel("风险调整信号水平")
    axes[0].set_title(f"BTC 首周样例 · {ex['dates'][0]} UTC · 无水平脉冲累加",loc="left")
    axes[1].step(np.arange(len(ex["hysteresis"])),ex["hysteresis"],color=PALETTE[1],where="post")
    axes[1].set_ylabel("滞回方向状态")
    axes[1].set_xlabel("连续评价时刻索引（缺记录间隔在滤波中按实际时间处理）")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    svgs["signal_path"]=save("signal_path")
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for i,key in enumerate(("sharing_isolated","sharing_partial","sharing_full")):
        row=screen[key]
        grads=row["gradient_diagnostics"]
        if grads:
            axes[0].plot([g["cosine"] for g in grads],color=PALETTE[i],alpha=.8,label=key.split("_")[-1])
            axes[1].plot([g["weighted_ratio"] for g in grads],color=PALETTE[i],alpha=.8,label=key.split("_")[-1])
    axes[0].axhline(0,color="#8c9db8",lw=.8)
    axes[0].set_title("共享局部层梯度夹角",loc="left")
    axes[1].axhline(.3,color="#8c9db8",lw=.8,ls="--")
    axes[1].set_title("加权辅助梯度／主梯度",loc="left")
    for ax in axes:
        ax.set_xlabel("固定间隔诊断批次")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    svgs["gradients"]=save("gradients")
    return svgs


def architecture_svg(proposal):
    blocks=[(25,40,190,78,"180 × 1m · 12 字段","局部 TCN48 → 36 有序片段"),
            (25,145,190,78,f"144 × 5m · {proposal['fast_dim']} 字段","TCN96 → 轻调制 → GRU64"),
            (25,250,190,78,"168 × 1h · 10 字段","连续局部覆盖 → 有序分段"),
            (25,355,190,78,"当前状态 · 24 维","风险尺度／共同市场／流动性"),
            (310,150,205,110,"融合128 + 残差块","多路拼接 · 延后压缩"),
            (625,70,200,85,"直接有符号收益信号","主任务 MSE · 固定未来 4h"),
            (625,195,200,85,"总活动与不对称辅助","局部TCN + 状态 → logT/Q/A"),
            (625,330,200,85,"因果水平更新","EMA／约束学习滤波／滞回诊断")]
    rectangles="".join(f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#162a40" stroke="#38556e"/><text x="{x+w/2}" y="{y+30}" text-anchor="middle" fill="#eef4fc" font-size="15">{html.escape(a)}</text><text x="{x+w/2}" y="{y+55}" text-anchor="middle" fill="#b7c9de" font-size="12">{html.escape(b)}</text></g>' for x,y,w,h,a,b in blocks)
    paths='<path d="M215 79 H255 V184 H310 M215 184 H310 M215 289 H255 V220 H310 M215 394 H275 V250 H310 M515 185 H560 V112 H625 M725 155 H845 V372 H825" fill="none" stroke="#58d6c0" stroke-width="2" marker-end="url(#arrow)"/><path d="M215 204 H285 V305 H580 V237 H625 M215 394 H285 V305" fill="none" stroke="#ab9aff" stroke-width="2" marker-end="url(#arrow)"/>'
    return f'<svg viewBox="0 0 855 460" role="img" aria-label="一分钟、五分钟、小时与状态输入到直接信号和风险辅助的架构"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#58d6c0"/></marker></defs>{paths}{rectangles}</svg>'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"assets").mkdir(exist_ok=True)
    screen,stages,confirmed,ensemble=read("screen_results.json"),read("selection_log.json"),read("confirmation_results.json"),read("ensemble_results.json")
    runtime,locked,heads,uncertainty=read("runtime_manifest.json"),read("locked_design.json"),read("head_diagnostics.json"),read("paired_uncertainty.json")
    suites=ET.parse(ROOT/"tests.xml").getroot().findall("testsuite")
    passed=sum(int(s.attrib["tests"])-int(s.attrib.get("failures",0))-int(s.attrib.get("errors",0))-int(s.attrib.get("skipped",0)) for s in suites)
    assert passed == 27 and all(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0)) == 0 for s in suites)
    files=sorted(list(Path("src/crypto_timing").glob("mechanism_*.py"))+list(Path("scripts").glob("*mechanism*.py"))+[Path("scripts/run_mechanism_research.ps1"),Path("tests/test_mechanism_contract.py"),Path("docs/V4_IMPLEMENTATION_PROTOCOL_2026-10-04.md"),Path("docs/V4_FEATURE_DICTIONARY_2026-10-04.md"),Path("README.md")])
    delivery={"contract_tests_passed":passed,"source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},"note":"Final delivery sources; runtime_manifest retains the analysis-time source snapshot."}
    (ROOT/"delivery_manifest.json").write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding="utf-8")
    signals={f:read(f"signal_study_{f}.json") for f in ("f1","f2")}
    feature={f:read(f"feature_audit_{f}.json") for f in ("f1","f2")}
    manifest=read("cache_v4/manifest.json")
    svgs=figures(screen,stages,confirmed,ensemble,signals,feature["f1"])
    actual_unique=runtime["unique_screen_training_runs"]
    selected=locked["selected_config"]
    proposal=confirmed["f1_proposal"]
    architecture=architecture_svg(locked["proposal_config"])
    zero_minutes=sum(a["zero_quote"] for a in manifest["audit"].values())
    open_diffs=sum(a["aggregation_disagreement_rows"]["open"] for a in manifest["audit"].values())
    all_pred=[ensemble[f][s]["skill"] for f in ("f1","f2") for s in ("validation","replay")]
    conclusion=("两次历史迁移的信号改善均为正，但仍需新到达数据确认。" if all(v>0 for v in all_pred) else
                "信号改善尚未在两次历史迁移中一致保留；实现完成与可预测优势成立应分别判断。")
    topline=f"完成一分钟源、多尺度表示、收益与路径监督、共享和记忆消融及因果水平信号的研究闭环。{conclusion}"
    data_rows=[]
    for symbol,a in manifest["audit"].items():
        data_rows.append(f"| {symbol} | {a['rows']:,} | {a['zero_quote']} | {a['aggregation_disagreement_rows']['open']} | {a['aggregation_disagreement_rows']['quote_volume']} |")
    score_rows=[]
    for fold in ("f1","f2"):
        for split in ("validation","replay"):
            r=ensemble[fold][split]
            ci=r["uncertainty_vs_zero"]["ci95"]
            score_rows.append(f"| {fold} {split} | {r['hours']:,} | {r['mse']:.6f} | {pct(r['skill'])} | [{pct(ci[0])}, {pct(ci[1])}] | {r['correlation']:+.4f} | {r['positive_months']}/{len(r['monthly'])} |")
    ablation_rows=[]
    for stage in stages:
        first=stage["candidates"][0]
        for name in stage["candidates"]:
            r=screen[name]
            ci=uncertainty.get(f"{first} → {name}")
            interval="参照" if ci is None else f"{pct(ci['skill_increment'])} [{pct(ci['ci95'][0])}, {pct(ci['ci95'][1])}]"
            ablation_rows.append(f"| {STAGE_LABELS[stage['stage']]} | {name} | {r['parameters']:,} | {r['best_epoch']} | {pct(r['results']['validation']['skill'])} | {pct(r['results']['replay']['skill'])} | {interval} |")
    module_rows=[]
    for key in ("memory_mean","memory_ordered","memory_gru","memory_lstm","minute_none","minute_aggregate","minute_encoder","full_document_proposal"):
        r=screen[key]
        d=r["ood_sensitivity"]
        module_rows.append(f"| {key} | {r['effective_rank']:.2f}/128 | {pct(r['probe']['skill'])} | {d['reversed_fast_order']:.5f} | {d['fast']:.5f} | {d['state']:.5f} |")
    filter_rows=[]
    for fold in ("f1","f2"):
        for name in ("raw","ema_1h","ema_2h","ema_4h","learned"):
            r=signals[fold]["splits"]["replay"][name]
            filter_rows.append(f"| {fold} | {name} | {pct(r['skill'])} | {r['correlation']:+.4f} | {r['adjacent_flip_rate']*100:.2f}% | {r['rms_revision']:.5f} | {r['hysteresis_change_rate']*100:.2f}% |")
    confirm_rows=[]
    for key,r in confirmed.items():
        confirm_rows.append(f"| {key} | {r['parameters']:,} | {r['best_epoch']}/5 | {r['unique_train_hours']:,}/{r['train_hours']:,} | {pct(r['results']['validation']['skill'])} | {pct(r['results']['replay']['skill'])} | {r['gpu_peak_mb']:.0f} |")
    label_rows=[]
    for name in ("label_return","label_path","label_joint","label_distribution","label_four_bin"):
        d=heads.get(name,heads.get(screen[name]["artifact_run"],{}))
        val=d.get("validation",{})
        own=", ".join(f"{key}={value:.5f}" for key,value in val.items() if isinstance(value,(int,float)))
        if "path_skill_vs_train_mean" in val:
            own="logT/Q/A skill="+" / ".join(pct(v) for v in val["path_skill_vs_train_mean"])
        label_rows.append(f"| {name} | {own or '直接收益主头；详见共同信号 MSE'} | {pct(screen[name]['results']['validation']['skill'])} |")
    sensitivity_rows=[]
    for name in ("horizon_1","horizon_4","horizon_8","minute_none","path_sampling_1m"):
        r=screen[name]
        sensitivity_rows.append(f"| {name} | {r['config']['horizon']}h | {r['config']['path_frequency']}m | {r['config']['label']} | {pct(r['results']['validation']['skill'])} | {pct(r['results']['replay']['skill'])} | {r['results']['replay']['correlation']:+.4f} |")
    checklist=[("数据与已完成对齐","全量 1m 网格、OHLC、流量审计；5m/1h 因果重建；旧文件差异可见"),
               ("字段级适配","三套预处理；有界零点保留；过去风险与活动尺度；形状/背景双路"),
               ("冗余与选择","原16字段、扩展19字段、精选14字段、随机14字段；训练相关、秩、5m滞后、状态诊断"),
               ("收益与路径","R/P/P+R/统一分布/四档；同边界；5m主路径、1m敏感性；1/4/8h单独目标"),
               ("共享与干扰","隔离/部分/全共享；实际局部梯度夹角、范数与辅助预算"),
               ("时序表示","摘要/有序分段/GRU/LSTM，64/96/128宽度；有效秩与表示线性探针"),
               ("背景与分钟","72/168h、轻调制开关；分钟聚合与180分钟短编码器"),
               ("信号语义","水平预测直接监督；固定/学习因果滤波；实际连续块；翻转、响应和滞回状态"),
               ("优化与迁移","CUDA；早衰减到低学习率；五次完整时钟覆盖；三种子、两折、月度与时间块区间"),
               ("报告与复现","HTML/SVG/Markdown/JSON；缓存哈希、配置、检查点、预测、逐项自查" )]
    check_rows="\n".join(f"| {a} | 已完成 | {b} |" for a,b in checklist)
    markdown=f"""# 一分钟数据驱动的加密货币中低频深度学习机制研究 · v4

研究日期：2026-10-04。唯一设计依据：[标签、特征、表示学习与信号机制研究](DEEP_LEARNING_MODEL_MECHANISMS_REVIEW_2026-10-04.md)。[可视化报告](../reports/mechanism_2026-10-04/index.html)，[实施合同](V4_IMPLEMENTATION_PROTOCOL_2026-10-04.md)，[特征字典](V4_FEATURE_DICTIONARY_2026-10-04.md)。

**{topline}** 本轮没有把传统模型竞赛或交易执行当作目标。零信号只作为 MSE 的量纲归一参照；线性探针只用于读出网络表示。不能将完成框架解释为保证获利。

## 1 文档要求与完成证据

| 核心要求 | 状态 | 本轮实际落实 |
|---|---|---|
{check_rows}

共 {len(screen)} 个消融配置，其中 {actual_unique} 个唯一训练运行，重复配置复用相同预测，没有冒充新实验；另有 {len(confirmed)} 个完整时钟训练运行。被自查撤下的局部路径对照和含人工复制通道的早期运行单独归档，不进入最终证据。四档零收益方向恢复v3的0.5合同并重训；1/4/8h加入同预算4h对照。27 项合同测试通过。

## 2 数据：用一分钟重建同一个世界

源目录 `D:/Trading/practical_crypto_strategy/data/parquet`。全量读取十二币，各 1,962,720 行，共 {12*1962720:,} 行，从 2023-01-01 00:00 到 2026-09-24 23:59 UTC。逐行网格、开闭时间、OHLC、量额和主动买额界限通过。无缺行/重复键；无成交分钟共 {zero_minutes:,} 条，作为单独语义保留。

旧五分钟文件与一分钟聚合并不完全一致：全币共 {open_diffs} 个开盘价差异记录，部分量额差异也存在，尽管收盘价全部一致。因此本轮所有频率和标签统一从更新后一分钟重建，未混用旧五分钟 OHLCV。差异原因为上游数据版本/修正尚未独立证明，本轮没有擅自宣称其中某份是交易所最终真实版本。完整差异幅度、源哈希与派生字段核对存于 manifest。

| 合约 | 分钟行数 | 无成交分钟 | 与旧5m开盘价不同 | 与旧5m成交额不同 |
|---|---:|---:|---:|---:|
{chr(10).join(data_rows)}

预制 `log_return`、`vwap`、`taker_buy_ratio` 全量核对，仅作为审计，不直接继承。缺记录触发失败；无成交产生特定标志，压力/VWAP 缺失；预热不足保留 NaN/mask；无发布时间证据的陈旧衍生字段不进入本轮。这不是逐笔订单流或盘口。

## 3 输入、预处理与标签的明确含义

每小时决策，读入截至该时刻已完成的 180 根一分钟、144 根五分钟及 168 个小时背景。12 币共享网络，BTC/ETH/广度和本币残差保留，不强制市场中性。分钟输入并未把同一个 4h 标签复制六十次。

收益形状除以此前完成收益估计的 EWMA 风险，当前跳变不更新自己的分母；使用 asinh 平滑压缩有符号尾部，不去掉漂移。量额/笔数用当前量相对过去 EWMA 的 log 比率；自然有界比例和周期字段在适配方案中保留零点/区间。风险 log 水平、短长风险比、流动性与长期趋势由背景状态保留。无界字段再由每折训练期中位数/IQR 整理，clip ±8，缺失填零并给 mask。五分钟拟合覆盖全部相位；分钟统计按与60、5互素的17间隔抽样，覆盖全部相位。截尾率、月度中心/RMS 已记录，clip 仍会损失部分真实尾部，不把它称为无代价降噪。

`quote_surprise1h` 的定义现在明确为当前小时量额相对此前 EWMA；没有继续使用含糊的 `quote_surprise24h` 命名。量额单位固定 USDT，流动性代理的参考量为 10⁶ USDT；活动比值和压力具有单位一致性测试，未宣称所有含固定参考单位的字段任意换单位都自动不变。小时/周内 sin/cos 作为季节条件，本轮未额外拟合逐时段季节去量基准。

主标签是决策后五分钟开盘至到期开盘的简单收益 G，Y=G/v；v 使用当前已知五分钟过去24h与7d平方收益混合尺度，5%下限只按本折训练期拟合。1/4/8h尺度分别按平方根时间调整。Y 的 MSE 估计等风险条件优势，其原尺度误差权重为 1/v²；乘回 v 恢复量纲，并不取消训练权重。未裁剪收益标签，也未将 asinh 后均值误认为原收益均值。

路径标签在完全相同的入场和到期边界内，使用 open-to-open 对数增量计算 U/D。主采样5m，1m单独检验。T=(U+D)/v²，Q=(U−D)/v²，A=(U−D)/(U+D+0.02v²)，训练显式输出 log(T+0.02)、Q、A。不会先预测两个风险大数再机械相除作为唯一信号。纯P的Q输出只叫压力代理；P+R有直接收益锚。纯P用自身路径误差选择权重，经济锚定模型按直接信号误差选择。未来零成交路径作为标签质量控制剔除；因此评分对象是有完整活动记录的路径，未来可用性并不是输入特征，也不把质量筛选效果当作可实时知道的优势。

统一分布是八个有符号类别，尾部档内代表值只在训练期拟合；四档分解用独立状态幅度头和条件方向头，推理只用预测幅度概率组合，不用真实未来档位。它们是对标签分解机制的同输入复现，未声称逐参数复制旧v3整个网络。各自 CE/BCE/路径误差与共同收益信号评分分开报告。

## 4 多尺度网络与模块职责

完整文档方案实测 {proposal['parameters']:,} 总参数，实际目标活跃参数为 {heads['confirm_f1_proposal']['active_objective_parameters']:,}，在0.2—0.5M设计预算内。主干为 TCN96+单层GRU64，慢背景TCN48，分钟TCN48，融合128与一个残差块。TCN核5、扩张1/2形成连续局部覆盖；慢分支不会把最大跨度说成完整记忆。分钟核3、扩张1/2后以完整五分钟片段压缩，再保留有序分段，随后与中频记忆分别读出融合。

底层局部编码可共享，风险在局部均值/末态与背景中独立读出；它不直接把总风险梯度灌入收益GRU。完全共享/隔离也已对照。GRU在每个窗口重置，不跨打乱样本携带隐藏状态。轻调制用0.1·tanh约束，初始接近恒等。mask显式进入投影、全缺失token被屏蔽，但卷积不是逐通道的严格缺失归一化算子，循环门仍会在零token上演化，限制已保留。

最终历史筛选候选为 `{json.dumps(selected,ensure_ascii=False)}`。完整文档方案与简化候选都做了全量训练；没有因某个简化候选短期得分更好而删除GRU/分钟路线的实现或将神经网络研究停止。筛选是有顺序的局部研究，不是穷尽所有交互组合的全局最优证明。

## 5 消融证据：不能只看最高分

每个唯一筛选运行8轮×128步，24个分散时刻/批、每时刻12币，合计1024更新步与24,576时刻曝光；均匀抽样，不只挑高波动。并不叫8次完整历史遍历。月度损失和逐个有效时刻另行记录。AdamW峰值2e−4，24步warmup，到65%预算已降至2e−5；时序dropout .05、融合 .10、梯度裁剪1。未使用翻转收益或打乱时间作增强。

skill = 1−MSE(signal,Y4)/MSE(0,Y4)。越高越好，百分号表示该无量纲评分的百分点，不是交易收益。区间以同一时刻12币先平均后按72小时移动块重抽样500次；保留共同市场及重叠标签依赖的近似，不把12币当12份独立行情。小增量且区间跨零只算未获支持，不称为模块有效；区间没有对多次搜索校正。

| 问题 | 配置 | 总参数 | 最佳轮 | F1早停窗skill | 后续诊断窗skill | 相对组首增量及95%区间 |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(ablation_rows)}

收益、路径与分布各自的监督评分：

| 标签路线 | 自身任务或概率指标（F1早停窗） | 共同4h信号skill |
|---|---|---:|
{chr(10).join(label_rows)}

1h/8h是单独训练的收益目标，比较各自标准化收益误差和跨跨度相关，不把原始分数平均；它们对4h的MSE仅是代理诊断，不当4h收益预测能力排名。1m路径敏感性没有同时改变输入，归因限制清楚。

| 频率/跨度对照 | 主跨度 | 路径采样 | 标签 | 自身主信号验证skill | 自身后续skill | 后续相关 |
|---|---|---|---|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

## 6 表示、模块与梯度自查

训练相关性证实 return/body 的Pearson约 {feature['f1']['near_pairs'][1]['pearson']:.4f}，量额和笔数surprise也高度相关；显式 return_copy 的1.0相关只是数学诊断，该列从最终全部网络排除。原16组、扩展19组、精选14组和随机14组分别重新训练。精选不是按验证标签筛选；去掉近重复、相同市场趋势与已在状态中的活动水平，保留经济上不同的残差/吸收职责。表中结果不支持仅凭热力图就断言某个精选簇是唯一正确答案。

| 模块配置 | 表示有效秩 | 冻结表示线性探针skill | 反转快序列变化RMS | 快分支屏蔽变化RMS | 状态屏蔽变化RMS |
|---|---:|---:|---:|---:|---:|
{chr(10).join(module_rows)}

有效秩按训练表示协方差谱熵计算，探针训练只用训练表示、固定L2=100。反序列和分支置零是分布外敏感性，不能叫因果贡献；结论以重新训练的消融为主，敏感性为辅。首层激活RMS、层更新/权重比、完整梯度诊断已保存。辅助权重由主/辅共享局部梯度的EMA范数比按0.3预算调整、限制在0.01—0.5；这不是每批严格保证比例0.3。孤立风险的共享梯度为零；单次负夹角不自动判辅助无效。此轮未把PCGrad或GradNorm名称当作已经实施的算法。

## 7 完整覆盖、多种子与向前回放

F1训练从预热结束到2025-06末，早停2025-07—10，后续诊断2025-11—2026-01。F2重新拟合截至2026-01末约37个月历史，早停2026-02—05，后续诊断2026-06—09-24。每折尺度下限、scaler和分布代表值重新拟合，所有分割用最长8h+5m成熟时间purge，保证各标签比较样本完全一致。历史标签成熟前不进此前训练集。

完整覆盖候选各5轮；每轮覆盖全部有效训练时刻，早停保留最优轮而仍完成既定低学习率研究预算。R/P+R/完整文档方案先在F1比较；锁定结构后再做其他种子和F2，F2只重新拟合/选择权重，不重搜结构。多种子是初始化敏感性，不是三份独立市场。下表证明实际覆盖，不仅报告训练轮数。

| 运行 | 总参数 | 最佳轮/完成轮 | 独特时刻/全部时刻 | 早停窗skill | 后续窗skill | 峰值GPU MB |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(confirm_rows)}

三种子均值信号：

| 窗口 | 时刻数 | MSE | skill | 95%时间块区间 | 相关 | 正skill月份 |
|---|---:|---:|---:|---|---:|---:|
{chr(10).join(score_rows)}

**{conclusion}** 2026年2—9月此前已经被旧研究查看，本轮虽然严格按时间回放，仍属探索历史，不能重新命名为独立未见测试。筛选窗本身也参与权重/结构选择。新到达且方案锁定后未参与选择的数据才可提供前向确认；当前原始文件没有这段数据。本轮没有将这个限制当成停止神经网络实现的理由。

## 8 信号机制：尺度、有效期与可靠度

网络输出是最新未来区间的水平预测，不是独立脉冲。固定EMA用 α=1−2^(−Δ/τ)，保持常数尺度，缺少评价点时按实际时间间隔调整α。学习滤波仅学习一个有界α∈[0.1,0.95]，在冻结网络的训练期连续24小时块上直接监督，前4小时作暖启动。它是冻结信号之上的监督滤波实验，不是已经联合微调整个编码器。

F1学习α={signals['f1']['learned_alpha']:.4f}（半衰期{signals['f1']['learned_half_life_hours']:.3f}h），F2为{signals['f2']['learned_alpha']:.4f}（{signals['f2']['learned_half_life_hours']:.3f}h）。原始、1/2/4h固定滤波及学习滤波均评价同一个4h对象，降低翻转并不自动表示增加预测信息。未来滞后收益相关是响应描述，未声称是因果响应。

| 折 | 更新机制 | 后续窗skill | 相关 | 符号翻转 | 修正RMS | 滞回状态变动 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(filter_rows)}

滞回开阈值0.05、退出0.02仅用于方向状态诊断，未搜交易最高净值，不称为订单或持仓。信号可信度由跨月、多种子与时间块区间、风险辅助输出和信号幅度诊断表达；没有另造未经校准的confidence分类分数。此轮没有成本/杠杆/撮合工程，也没有用净值隐藏预测误差。

## 9 复现、交付与研究边界

实际解释器 `{runtime['python']}`，PyTorch `{runtime['torch']}`、CUDA {runtime['cuda']}，{runtime['gpu']}。CUDA不可用时立即失败，没有CPU训练回退。指定的`Scripts/activate`是POSIX脚本，Windows入口使用同一conda的PowerShell hook再`conda activate universal`。

```powershell
./scripts/run_mechanism_research.ps1 -Stage cache
./scripts/run_mechanism_research.ps1 -Stage test
./scripts/run_mechanism_research.ps1 -Stage research
$env:PYTHONPATH='src'
python scripts/audit_mechanism_features.py
python scripts/analyze_mechanism_research.py
./scripts/run_mechanism_research.ps1 -Stage report
```

核心源码：`mechanism_data.py`、`mechanism_model.py`、`mechanism_training.py`、`mechanism_analysis.py`、`mechanism_inference.py`。本地`outputs/mechanism`保存源哈希manifest、scaler、随机种子、最佳检查点、学习曲线、验证/回放预测及运行时源码指纹。完整训练可重跑，完成的相同合同直接复用；中断中的单次trial从头重跑，不伪称优化器级续训。源码改动后应换输出目录或撤下受影响trial，不能拿旧summary冒充新实验。检查点推理只需要已完成输入与过去风险，已在2026-09-25 00:00 UTC源数据末端无未来标签情况下通过CUDA调用。GPU峰值包含驻留缓存，不仅是模型参数。

HTML自包含、图为可导出的SVG；`reports/mechanism_2026-10-04/results.json`提供紧凑机器证据。主设计及所有核心消融已实现、训练、核验。注意力、asset embedding、风险预训练、RevIN逆变换、PCGrad、额外季节基准、近期适配及更多收益核在原文属于可研究的后续选项，本轮没有声称全部实施，也没有引入未经比较的默认行为。收益头只估计指定监督对象，后续发现机会尺度漂移仍应通过新的按计划训练和未见数据检验。
"""
    DOC.write_text(markdown,encoding="utf-8")
    # Compact artifacts retain evidence without embedding the repeated raw source-audit contracts.
    compact={"runtime":runtime,"stages":stages,"screen":{k:{x:v[x] for x in ("config","parameters","best_epoch","results","probe","effective_rank","artifact_run")} for k,v in screen.items()},
             "confirmation":{k:{x:v[x] for x in ("config","parameters","best_epoch","results","unique_train_hours","train_hours","gpu_peak_mb")} for k,v in confirmed.items()},
             "ensemble":ensemble,"uncertainty":uncertainty,"native_heads":heads,"signals":signals,"data_audit":manifest["audit"],
             "feature_audits":feature,"preprocessing_audits":{f:read(f"preprocessing_audit_{f}.json") for f in ("f1","f2")},
             "verification":{"contract_tests":passed,"delivery_manifest":delivery,"latest_label_free_cuda_inference":read("latest_inference.json")}}
    (OUT/"results.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding="utf-8")
    def table(rows,headers):
        return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{html.escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(c.strip())}</td>' for c in row.strip('|').split('|'))+'</tr>' for row in rows)+'</tbody></table></div>'
    experiment_html=table(ablation_rows,["问题","配置","参数","最佳轮","F1验证skill","后续skill","成对增量与95%区间"])
    confirmation_html=table(score_rows,["窗口","时刻数","MSE","skill","95%区间","相关","正月份"])
    data_html=table(data_rows,["合约","分钟行数","无成交","open差异","成交额差异"])
    sensitivity_html=table(sensitivity_rows,["频率/跨度对照","主跨度","路径采样","监督","自身验证skill","自身后续skill","后续相关"])
    checklist_html=''.join(f'<li><span class="check">✓</span><div><b>{html.escape(a)}</b><p>{html.escape(b)}</p></div></li>' for a,b in checklist)
    training_html=table(confirm_rows,["运行","参数","最佳/完成","覆盖时刻","验证skill","后续skill","GPU MB"])
    selected_text=' · '.join([DESCRIPTIONS.get(selected[k],str(selected[k])) for k in ("prep","label","memory","sharing","minute")])
    doc_link="../../docs/V4_MECHANISM_RESEARCH_REPORT_2026-10-04.md"
    experiment_json=json.dumps({k:{'config':v['config'],'skill':v['results']['validation']['skill'],'replay':v['results']['replay']['skill'],'rank':v['effective_rank'],'probe':v['probe']['skill'],'parameters':v['parameters'],'best_epoch':v['best_epoch']} for k,v in screen.items()},ensure_ascii=False).replace('</','<\\/')
    html_report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>加密货币深度学习机制研究 · v4</title>
<style>
:root{{color-scheme:dark;--bg:#091322;--panel:#101b2c;--line:#26394e;--fg:#edf3fc;--muted:#b1c2d8;--teal:#5ad9c2;--violet:#ab9aff}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.8 'Microsoft YaHei','Segoe UI',sans-serif}}a{{color:var(--teal);text-decoration:none}}a:hover{{text-decoration:underline}}.shell{{display:grid;grid-template-columns:225px minmax(0,1fr);max-width:1700px;margin:auto}}aside{{position:sticky;top:0;height:100vh;border-right:1px solid var(--line);padding:32px 22px}}aside .brand{{font-size:18px;letter-spacing:2px}}aside .version{{font-size:12px;color:var(--teal);margin:10px 0 28px}}nav a{{display:block;padding:9px 12px;color:var(--muted);border-radius:7px;margin-bottom:5px}}nav a:hover,nav a.active{{background:#1a2d43;color:var(--fg);text-decoration:none}}main{{min-width:0;padding:44px 48px 80px}}.eyebrow{{font-size:12px;letter-spacing:2px;color:var(--teal)}}h1{{font-size:42px;line-height:1.35;max-width:900px;margin:14px 0 20px;letter-spacing:1px}}h2{{font-size:26px;margin:0 0 14px;line-height:1.45}}h3{{font-size:19px;margin:26px 0 12px}}p{{color:var(--muted);margin:12px 0}}.lead{{font-size:17px;max-width:950px}}.chip{{display:inline-block;padding:5px 12px;border:1px solid var(--line);border-radius:20px;margin:4px 6px 4px 0;color:var(--muted);font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:30px 0}}.card{{background:var(--panel);padding:20px 22px;border:1px solid var(--line);border-radius:12px}}.card .value{{font-size:27px;color:var(--fg);font-variant-numeric:tabular-nums}}.card .label{{font-size:12px;color:var(--muted)}}section{{margin-top:54px;scroll-margin-top:24px}}.figure{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:22px 0}}.figure svg{{width:100%;height:auto;display:block}}.caption{{font-size:13px;color:var(--muted);margin-top:12px}}.note{{border-left:3px solid var(--teal);background:#112b35;padding:15px 20px;border-radius:0 8px 8px 0;margin:20px 0;color:#d5e9ec}}.note.warn{{border-left-color:var(--violet);background:#24243b}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.table-wrap{{overflow:auto;margin:20px 0;border:1px solid var(--line);border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}}th{{text-align:left;padding:13px 14px;background:#182a40;color:#dce9fa;font-weight:500;position:sticky;top:0}}td{{padding:11px 14px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}}tbody tr:nth-child(even){{background:#0f1e30}}tbody tr:hover{{background:#1b3046}}.checklist{{display:grid;grid-template-columns:1fr 1fr;list-style:none;padding:0;gap:16px}}.checklist li{{display:flex;gap:14px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}}.check{{color:var(--teal);font-size:20px}}.checklist p{{margin:4px 0 0;font-size:13px}}code{{font:13px 'Consolas',monospace;background:#1c2c42;padding:3px 6px;border-radius:4px}}footer{{margin-top:60px;padding-top:22px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}.detail{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px}}select{{background:#1a2c43;border:1px solid #4a627f;padding:10px;color:var(--fg);border-radius:6px;width:100%;font:inherit}}#detail-body{{margin-top:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}#detail-body div{{padding:12px;background:#15283d;border-radius:7px}}#detail-body span{{display:block;color:var(--muted);font-size:12px}}@media(max-width:1100px){{main{{padding:32px 26px}}h1{{font-size:34px}}.cards{{grid-template-columns:1fr 1fr}}.columns{{grid-template-columns:1fr}}}}@media(max-width:760px){{.shell{{display:block}}aside{{position:relative;height:auto;padding:20px;border-right:0;border-bottom:1px solid var(--line)}}nav{{display:flex;overflow:auto}}nav a{{white-space:nowrap}}aside .version{{margin-bottom:10px}}main{{padding:26px 18px}}h1{{font-size:29px}}.checklist{{grid-template-columns:1fr}}#detail-body{{grid-template-columns:1fr 1fr}}}}@media print{{aside{{display:none}}.shell{{display:block}}main{{padding:10px}}.figure{{break-inside:avoid}}section{{margin-top:25px}}}}
</style></head><body><div class="shell"><aside><div class="brand">CRYPTO / DL</div><div class="version">MECHANISM RESEARCH · V4</div><nav><a href="#overview">研究结论</a><a href="#contract">要求与证据</a><a href="#architecture">模型与标签</a><a href="#ablation">逐项消融</a><a href="#representation">表示与梯度</a><a href="#migration">时间迁移</a><a href="#signal">信号机制</a><a href="#data">数据审计</a><a href="#reproduce">复现与交付</a></nav></aside><main>
<header id="overview"><div class="eyebrow">2026.10.04 · 1 MINUTE SOURCE · CUDA</div><h1>一分钟数据驱动的<br>加密货币深度学习择时。</h1><p class="lead">{html.escape(topline)}</p><span class="chip">神经网络为研究主线</span><span class="chip">全部历史为探索与回放</span><span class="chip">信号水平 ≠ 交易动作</span><div class="cards"><div class="card"><div class="value">2,355万</div><div class="label">全量一分钟原始记录</div></div><div class="card"><div class="value">{actual_unique} + {len(confirmed)}</div><div class="label">唯一消融 + 完整覆盖训练</div></div><div class="card"><div class="value">{proposal['parameters']/1000:.1f}K</div><div class="label">完整文档方案总参数</div></div><div class="card"><div class="value">2折 × 3种子</div><div class="label">按时间迁移与初始化验证</div></div></div><div class="note warn">{html.escape(conclusion)} 分数是预测误差改善，不是回测收益；本轮没有以传统基线竞赛或撮合工程替代模型研究。</div></header>
<section id="contract"><div class="eyebrow">01 / TRACEABLE REQUIREMENTS</div><h2>从设计建议到可核验的能力</h2><p>唯一依据为指定机制研究文档。每一项都有代码、训练或审计证据，建议结构与实测结论分开。</p><ul class="checklist">{checklist_html}</ul></section>
<section id="architecture"><div class="eyebrow">02 / MULTISCALE REPRESENTATION</div><h2>形状与背景分工，直接监督最终信号</h2><div class="figure">{architecture}<p class="caption">完整文档方案已训练。局部编码共享，上层风险与收益记忆分开；分钟、五分钟分别有序读出后融合。</p></div><div class="columns"><div class="card"><h3>收益族 R</h3><p>入场开盘到4h到期开盘的简单收益，经当时过去风险标准化。主头直接学习有符号条件优势，MSE保持条件均值目标。</p></div><div class="card"><h3>路径族 P / P+R</h3><p>共同边界内分别学习总活动 log T、有符号平方压力 Q、相对不对称 A。P 的 Q 是压力代理；P+R 由直接收益头锚定。</p></div></div><p>主路径采样5m，1m仅作频率敏感性。1/4/8h分别监督，不未经尺度统一就混合。四档模型推理用预测路由，绝不使用真实未来档位。</p><div class="note">最终筛选候选：{html.escape(selected_text)}。完整 GRU + 分钟方案和候选都做了全量训练；模块建议没有因一次短期失利而被永久撤下。</div>{table(label_rows,["监督路线","自身任务/概率评分","共同4h信号skill"])}</section>
<section id="ablation"><div class="eyebrow">03 / CONTROLLED ABLATIONS</div><h2>先处理表达，再研究共享、记忆与分钟增量</h2><p>单个唯一筛选运行1024个更新步、24个分散时刻/批、8个预算轮次。每组只改变对应机制；组间锚点顺序更新，因此各组最高分不应直接作模块增量归因。</p><div class="figure">{svgs['ablations']}<p class="caption">颜色标识该组局部胜者。改善很小且成对时间块区间跨零时，只记为尚未获得支持。</p></div>{experiment_html}{sensitivity_html}<div class="detail"><label for="experiment-select">查看单个配置的证据</label><select id="experiment-select">{''.join(f'<option value="{html.escape(k)}">{html.escape(k)}</option>' for k in screen)}</select><div id="detail-body" aria-live="polite"></div></div><p>原v3字段组、扩展组、机制精选和等数量随机删除均重新训练。人工 return_copy 只在相关性诊断中出现，从最终训练网络排除。</p></section>
<section id="representation"><div class="eyebrow">04 / REPRESENTATION & TASK INTERFERENCE</div><h2>不以模块名称代替作用证据</h2><div class="figure">{svgs['redundancy']}<p class="caption">{feature['f1']['sample_rows']:,} 个训练期抽样字段观测；另外保存秩相关、真实5m滞后相关和高低风险状态关系。有界比例保留零点，标准化与去冗余分开。</p></div>{table(module_rows,["模块","有效秩","探针skill","反序列RMS","快分支RMS","状态RMS"])}<div class="figure">{svgs['gradients']}<p class="caption">测量共享局部层实际梯度。辅助预算是平滑和有界的近似控制；孤立风险没有共享梯度。负夹角不是删除辅助任务的充分条件。</p></div><p>分支置零和反序列属于分布外敏感性；重新训练的消融提供主要证据。探针仅从训练表示拟合，不是另一场传统模型竞赛。</p></section>
<section id="migration"><div class="eyebrow">05 / FORWARD REPLAY</div><h2>完整覆盖之后，再看月份与种子</h2><div class="figure">{svgs['training']}<p class="caption">每轮完整覆盖有效训练时刻。学习率在预算内早衰减，保留最优权重同时完成低学习率阶段研究。</p></div>{training_html}<div class="figure">{svgs['confirmation']}<p class="caption">三种子均值信号；72小时时间块95%区间，12币先按同一时刻聚合。区间未校正结构搜索。</p></div>{confirmation_html}<div class="figure">{svgs['monthly']}<p class="caption">F1早停2025-07—10、后续2025-11—2026-01；F2早停2026-02—05、后续2026-06—09。每折重新拟合训练统计。</p></div><div class="note warn">旧研究已查看2026年2—9月。严格时间回放保留信息合同，但不能将这段历史重新命名为独立未见测试。当前文件没有方案锁定后新到达的确认数据。</div></section>
<section id="signal"><div class="eyebrow">06 / LEVEL SIGNAL MECHANISM</div><h2>减少抖动要同时检查信息损失</h2><p>重叠未来区间的预测是更新后的水平。EMA保持常数幅度，禁止将它当成新脉冲反复叠加。学习滤波在冻结网络的训练期连续24h块上只学习一个有界参数。</p><div class="figure">{svgs['filter_tradeoff']}<p class="caption">更低翻转不自动意味着更高预测能力；所有机制评分同一个4h对象，未按交易净值挑最高参数。</p></div>{table(filter_rows,["折","机制","后续skill","相关","符号翻转","修正RMS","滞回变动"])}<div class="figure">{svgs['signal_path']}<p class="caption">固定阈值滞回仅是方向状态诊断，不是订单或持仓。学习α：F1={signals['f1']['learned_alpha']:.4f}，F2={signals['f2']['learned_alpha']:.4f}。</p></div></section>
<section id="data"><div class="eyebrow">07 / MINUTE-SOURCE AUDIT</div><h2>一致的时间网格不等于不同文件的值一致</h2><p>全量网格、OHLC和成交字段合法性检查通过。旧5m文件存在少量价格/量额差异；本轮全部重建自更新后1m，确保分钟增量比较使用同一个行情来源。</p>{data_html}<p>源派生列仅审计、不继承；无成交、预热不足、缺记录分别处理。最长8h+5m成熟时间purge保持各标签对照同一评价样本。未知发布时间的衍生字段没有混入长历史。</p></section>
<section id="reproduce"><div class="eyebrow">08 / REPRODUCIBILITY</div><h2>设计、检查点和预测可以逐项追溯</h2><p>解释器：<code>{html.escape(runtime['python'])}</code><br>设备：{html.escape(runtime['gpu'])} · CUDA {runtime['cuda']} · PyTorch {html.escape(runtime['torch'])}。</p><p>本地 outputs/mechanism 保存完整配置、源哈希、scaler、学习曲线、检查点和预测。相同完成合同复用；中断中的单次运行从头重跑。27项合同测试通过。</p><div class="columns"><div class="card"><h3>完整研究文档</h3><p>定义、消融、诊断、限制和复现步骤。</p><a href="{doc_link}">打开 Markdown →</a></div><div class="card"><h3>可复核数字与图</h3><p>紧凑JSON与八张可导出SVG；此HTML不依赖外部CDN。</p><a href="results.json">机器可读结果 →</a></div></div><p>注意力、风险预训练、asset embedding、PCGrad、额外季节基准等原文后续选项未被冒称已实现。当前完成的是核心框架及文档要求的主要消融闭环，独立前向优势仍需新数据证据。</p></section><footer>依据《加密货币深度学习预测的标签、特征、表示学习与信号机制研究》 · 2026-10-04<br>所有图和结论由本轮实际训练产物生成；参数建议、实现能力与市场优势分别陈述。</footer>
</main></div><script id="experiment-data" type="application/json">{experiment_json}</script><script>
const experiments=JSON.parse(document.getElementById('experiment-data').textContent);const selector=document.getElementById('experiment-select');function showExperiment(){{const e=experiments[selector.value];const values=[['验证skill',(e.skill*100).toFixed(3)+'%'],['后续skill',(e.replay*100).toFixed(3)+'%'],['有效秩',e.rank.toFixed(2)+'/128'],['表示探针skill',(e.probe*100).toFixed(3)+'%'],['总参数',e.parameters.toLocaleString()],['最佳轮',e.best_epoch],['字段',e.config.prep+' / '+e.config.features],['记忆与宽度',e.config.memory+' / '+e.config.width],['分钟机制',e.config.minute]];const body=document.getElementById('detail-body');body.replaceChildren();values.forEach(([label,value])=>{{const div=document.createElement('div');const span=document.createElement('span');span.textContent=label;const strong=document.createElement('strong');strong.textContent=value;div.append(span,strong);body.append(div);}});}}selector.addEventListener('change',showExperiment);showExperiment();const sections=document.querySelectorAll('header[id],section[id]');const observer=new IntersectionObserver(entries=>{{entries.forEach(entry=>{{if(entry.isIntersecting){{document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+entry.target.id));}}}});}},{{rootMargin:'-10% 0px -65% 0px'}});sections.forEach(s=>observer.observe(s));
</script></body></html>'''
    (OUT/"index.html").write_text(html_report,encoding="utf-8")
    (OUT/"README.md").write_text(f"# v4 机制研究\n\n{topline}\n\n- [可视化报告](index.html)\n- [完整研究文档](../../docs/V4_MECHANISM_RESEARCH_REPORT_2026-10-04.md)\n- [机器结果](results.json)\n\n{actual_unique} 个唯一消融训练 + {len(confirmed)} 个完整时钟训练；CUDA、多种子、两次历史迁移。\n",encoding="utf-8")
    print(f"report: {OUT/'index.html'}\nmarkdown: {DOC}",flush=True)


if __name__=="__main__":
    main()
