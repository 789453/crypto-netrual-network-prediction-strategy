"""Create a self-contained visual research report from frozen v3 artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crypto_timing.redesign_training import RedesignStore

ROOT = Path("outputs/redesign")
OUT = Path("reports/redesign_2026-10-04")
COLORS = {"network": "#8c65e8", "linear_direction": "#22b8a5",
          "linear_both": "#f5ab55", "direct_ridge": "#6380d9"}


def _asset(name: str) -> str:
    return f'<img src="assets/{name}.svg" alt="{html.escape(name)}">'


def _fig(name: str) -> None:
    path = OUT / "assets" / f"{name}.svg"
    plt.savefig(path, bbox_inches="tight", facecolor="#111525")
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                    + "\n", encoding="utf-8")
    plt.close()


def _style() -> None:
    plt.rcParams.update({"figure.facecolor": "#111525", "axes.facecolor": "#171d30",
                         "savefig.facecolor": "#111525", "axes.edgecolor": "#77819a",
                         "axes.labelcolor": "#edf0fa", "xtick.color": "#aeb8cf",
                         "ytick.color": "#aeb8cf", "text.color": "#edf0fa",
                         "grid.color": "#39435a", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})


def _training_plot() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for si, seed in enumerate((20261004, 20261005, 20261006)):
        folder = ROOT / "models" / f"direction_seed{seed}"
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_loss_online"] for row in history], alpha=.55,
                linestyle="--", color=["#8c65e8", "#d17ae3", "#6f9bf5"][si],
                label=f"seed {seed} train")
        ax.plot(epochs, [row["validation_loss"] for row in history],
                color=["#8c65e8", "#d17ae3", "#6f9bf5"][si],
                label=f"seed {seed} Jul–Oct")
        best = json.loads((folder / "summary.json").read_text())["best_epoch"]
        ax.scatter([best], [history[best - 1]["validation_loss"]], marker="o", s=50,
                   color=["#8c65e8", "#d17ae3", "#6f9bf5"][si], zorder=5)
    ax.set(xlabel="Full-clock epoch", ylabel="Conditional direction BCE",
           title="Optimization works; forward validation peaks early")
    ax.grid(alpha=.35)
    ax.legend(ncol=3, fontsize=7, loc="lower left")
    _fig("training_dynamics")


def _direction_plot(selection: dict, historical: dict) -> None:
    months = sorted({*selection["direction_candidates"][selection["direction_leader"]]
                     ["calibration"]["months"],
                     *selection["direction_candidates"][selection["direction_leader"]]
                     ["selection"]["months"],
                     *historical["direction"][selection["direction_leader"]]["months"]})
    fig, ax = plt.subplots(figsize=(12, 4.4))
    for si, seed in enumerate((20261004, 20261005, 20261006)):
        name = f"direction_seed{seed}"
        merged = {**selection["direction_candidates"][name]["calibration"]["months"],
                  **selection["direction_candidates"][name]["selection"]["months"],
                  **historical["direction"][name]["months"]}
        values = [merged[m]["gain_vs_linear"] for m in months]
        ax.plot(months, values, marker="o", markersize=3, linewidth=1.5,
                color=["#8c65e8", "#d17ae3", "#6f9bf5"][si], label=f"seed {seed}")
    ax.axhline(0, color="#e5e8f0", linewidth=.8)
    ax.axvline("2025-11", color="#f5ab55", linestyle="--", alpha=.8)
    ax.axvline("2026-02", color="#f5ab55", linestyle="--", alpha=.8)
    ax.text(2, ax.get_ylim()[1] * .88, "early-stop calibration", fontsize=9, color="#f5ab55")
    ax.set(xlabel="UTC month", ylabel="Linear BCE − network BCE (higher is better)",
           title="Direction advantage fails to carry through time")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=.3)
    ax.legend(ncol=3, fontsize=8)
    _fig("monthly_direction_gain")


def _band_plot(selection: dict, historical: dict) -> None:
    leader = selection["direction_leader"]
    sets = (selection["direction_candidates"][leader]["calibration"],
            selection["direction_candidates"][leader]["selection"],
            historical["direction"][leader])
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(4)
    for i, (name, metrics, color) in enumerate(zip(
        ("Jul–Oct 2025", "Nov–Jan", "seen 2026"), sets,
        ("#8c65e8", "#22b8a5", "#f5ab55"), strict=True
    )):
        ax.bar(x + (i - 1) * .25,
               [metrics["bands"][str(j)]["gain_vs_linear"] for j in range(4)],
               width=.23, color=color, label=name)
    ax.axhline(0, color="#e5e8f0", linewidth=.8)
    ax.set_xticks(x, ["|Y| < .5", ".5–1", "1–2", "≥ 2"])
    ax.set(ylabel="Linear BCE − network BCE", title="Conditional direction by future magnitude band")
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8)
    _fig("band_direction_gain")


def _risk_plot(selection: dict, historical: dict) -> None:
    stages = ("Jul–Oct", "Nov–Jan", "seen 2026")
    rows = (selection["magnitude_calibration"]["overall"],
            selection["magnitude_selection"]["overall"], historical["magnitude"]["overall"])
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(3)
    for k, label, color in (("constant_ce", "constant", "#77819a"),
                            ("linear_ce", "risk logistic", "#22b8a5"),
                            ("ce", "risk MLP", "#8c65e8")):
        offset = {"constant_ce": -.24, "linear_ce": 0, "ce": .24}[k]
        ax.bar(x + offset, [row[k] for row in rows], width=.23, label=label, color=color)
    ax.set_xticks(x, stages)
    ax.set(ylabel="Magnitude cross entropy", title="Magnitude is learnable, but MLP adds little")
    ax.set_ylim(1.115, 1.175)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    _fig("magnitude_ce")


def _decile_plot(selection: dict, historical: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, title, accessor in ((axes[0], "Nov–Jan 2025/26 selection",
                                lambda n: selection["selection_options"][n]["0.0012"]),
                               (axes[1], "Previously viewed 2026 history",
                                lambda n: historical["scores"][n])):
        for name in ("network", "linear_direction"):
            dec = accessor(name)["deciles"]
            ax.plot(range(1, 11), [row["realized_return"] * 1e4 for row in dec],
                    marker="o", linewidth=1.6, color=COLORS[name], label=name)
        ax.axhline(0, color="#e5e8f0", linewidth=.8)
        ax.set(xlabel="Predicted-return decile", title=title)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Mean realized 4h return (bp)")
    axes[1].legend(fontsize=8)
    _fig("score_deciles")


def _trade_plot(selection: dict, historical: dict) -> None:
    names = ("network", "linear_direction", "linear_both", "direct_ridge")
    fig, ax = plt.subplots(figsize=(9, 4.3))
    selected = [max(selection["selection_options"][name].items(),
                    key=lambda row: row[1]["mean_sharpe"])[1]["mean_sharpe"] for name in names]
    seen = [historical["scores"][name]["mean_sharpe"] for name in names]
    x = np.arange(len(names))
    ax.bar(x - .18, selected, width=.34, color="#22b8a5", label="best selection threshold")
    ax.bar(x + .18, seen, width=.34, color="#f5ab55",
           label="fixed/explanatory threshold on seen history")
    ax.axhline(0, color="#e5e8f0", linewidth=.8)
    ax.set_xticks(x, ("Network", "Linear π + NN q", "Linear π + q", "Direct ridge"))
    ax.set(ylabel="Mean 4-phase net Sharpe", title="Positive validation peak does not persist")
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8)
    _fig("strategy_sharpe")


def _signal_sources_plot(selection: dict, historical: dict) -> None:
    cases = (("Network\nselection", selection["selection_options"]["network"]["0.0012"]),
             ("Linear π\nselection", selection["selection_options"]["linear_direction"]["0.0018"]),
             ("Network\nseen history", historical["scores"]["network"]),
             ("Linear π\nseen history", historical["scores"]["linear_direction"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(cases))
    for key, label, color, offset in (
        ("score_return_correlation", "pooled", "#8c65e8", -.23),
        ("market_time_correlation", "market time", "#22b8a5", 0),
        ("within_hour_correlation", "within hour", "#f5ab55", .23),
    ):
        ax.bar(x + offset, [entry[key] for _, entry in cases], width=.22,
               label=label, color=color)
    ax.set_xticks(x, [name for name, _ in cases])
    ax.axhline(0, color="#e5e8f0", linewidth=.8)
    ax.set(ylabel="Predicted vs realized return correlation",
           title="Most selection-period association is common market timing")
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8)
    _fig("signal_sources")


def _moving_block_interval(store: RedesignStore, split: str, name: str) -> tuple[float, float]:
    keys = store.splits[split]
    with (np.load(ROOT / "models" / name / f"{split}_predictions.npz") as nn,
          np.load(ROOT / "baselines" / f"{split}_predictions.npz") as base):
        if not (np.array_equal(nn["keys"], keys) and np.array_equal(base["keys"], keys)):
            raise ValueError("report prediction keys differ")
        label = store.targets[keys // store.n_symbols, keys % store.n_symbols]
        band, up = label[:, 2].astype(int), label[:, 3]
        row = np.arange(len(keys))
        a = base["direction_logits"][row, band].astype(float)
        b = nn["logits"][row, band].astype(float)
        difference = np.logaddexp(0, a) - up * a - (np.logaddexp(0, b) - up * b)
    days = store.dates[keys // store.n_symbols].astype("datetime64[D]")
    unique, inverse = np.unique(days, return_inverse=True)
    daily = np.bincount(inverse, weights=difference) / np.bincount(inverse)
    # Circular 7-day moving blocks keep the 12-asset panel and within-week dependence.
    rng = np.random.default_rng(20261004)
    draws = np.empty(2000)
    block_count = int(np.ceil(len(unique) / 7))
    offsets = np.arange(7)
    for i in range(len(draws)):
        starts = rng.integers(0, len(unique), size=block_count)
        sample = ((starts[:, None] + offsets) % len(unique)).ravel()[:len(unique)]
        draws[i] = daily[sample].mean()
    return tuple(float(value) for value in np.quantile(draws, (.025, .975)))


def _row(label: str, values: list[str]) -> str:
    return "<tr><th>" + html.escape(label) + "</th>" + "".join(
        f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"


def main() -> None:
    OUT.joinpath("assets").mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "evaluation" / "selection.json").read_text())
    historical = json.loads((ROOT / "evaluation" / "historical_diagnostic.json").read_text())
    calibration = json.loads((ROOT / "evaluation" / "calibration_diagnostic.json").read_text())
    store = RedesignStore(ROOT / "cache_v3")
    leader = selection["direction_leader"]
    _style()
    _training_plot()
    _direction_plot(selection, historical)
    _band_plot(selection, historical)
    _risk_plot(selection, historical)
    _decile_plot(selection, historical)
    _trade_plot(selection, historical)
    _signal_sources_plot(selection, historical)
    cal = selection["direction_candidates"][leader]["calibration"]["overall"]
    sel = selection["direction_candidates"][leader]["selection"]["overall"]
    seen = historical["direction"][leader]["overall"]
    ci_cal = _moving_block_interval(store, "calibration", leader)
    ci_sel = _moving_block_interval(store, "selection", leader)
    ci_seen = _moving_block_interval(store, "historical", leader)
    winner = selection["economic_winner"]
    thr = selection["economic_threshold"]
    win_sel = selection["selection_options"][winner][str(thr)] if winner else None
    win_seen = historical["scores"][winner] if winner else None
    cost_sel = np.mean([phase["cost_log_contribution"] for phase in win_sel["phases"]])
    cost_seen = np.mean([phase["cost_log_contribution"] for phase in win_seen["phases"]])
    funding_sel = np.mean([phase["funding_log_contribution"] for phase in win_sel["phases"]])
    funding_seen = np.mean([phase["funding_log_contribution"] for phase in win_seen["phases"]])
    tail_rows = "".join(_row(f"尾档代表值 × {mult}",
                                  [f"{record['mean_sharpe']:+.2f}",
                                   f"{record['mean_active_fraction']:.1%}"])
                        for mult, record in historical.get("tail_sensitivity", {}).items())
    symbol_rows = "".join(_row(symbol, [
        f"{selection['direction_candidates'][leader]['calibration']['symbols'][symbol]['gain_vs_linear']:+.4f}",
        f"{selection['direction_candidates'][leader]['selection']['symbols'][symbol]['gain_vs_linear']:+.4f}",
        f"{historical['direction'][leader]['symbols'][symbol]['gain_vs_linear']:+.4f}",
    ]) for symbol in store.manifest["symbols"])
    audit = json.loads((ROOT / "evaluation" / "audit.json").read_text())
    zero = historical["zero_cost_diagnostic"]
    calibration_rows = "".join(_row(label, [
        f"{calibration['periods'][period]['raw_direction']['gain_vs_linear']:+.5f}",
        f"{calibration['periods'][period]['adjusted_direction']['gain_vs_linear']:+.5f}",
        f"{calibration['periods'][period]['raw_score_return_correlation']:+.4f}",
        f"{calibration['periods'][period]['adjusted_score_return_correlation']:+.4f}",
    ]) for period, label in (("calibration", "Jul–Oct 校准"),
                             ("selection", "Nov–Jan 选择"),
                             ("historical", "已查看 2026 历史")))
    period_rows = "".join(_row(title, [f"{row['ce']:.5f}", f"{row['linear_ce']:.5f}",
                                       f"{row['gain_vs_linear']:+.5f}",
                                       f"[{ci[0]:+.5f}, {ci[1]:+.5f}]"])
                          for title, row, ci in (("2025 Jul–Oct · early stop", cal, ci_cal),
                                                 ("2025 Nov–2026 Jan · selection", sel, ci_sel),
                                                 ("2026 Feb–Sep · seen history", seen, ci_seen)))
    risk = [selection["magnitude_calibration"]["overall"],
            selection["magnitude_selection"]["overall"], historical["magnitude"]["overall"]]
    risk_rows = "".join(_row(title, [f"{row['ce']:.5f}", f"{row['linear_ce']:.5f}",
                                     f"{row['constant_ce']:.5f}"])
                        for title, row in zip(("Calibration", "Selection", "Seen history"), risk, strict=True))
    seeds = "".join(_row(str(seed), [str(selection["direction_candidates"][f"direction_seed{seed}"]["best_epoch"]),
                                        f"{selection['direction_candidates'][f'direction_seed{seed}']['calibration']['overall']['gain_vs_linear']:+.5f}",
                                        f"{selection['direction_candidates'][f'direction_seed{seed}']['selection']['overall']['gain_vs_linear']:+.5f}"])
                    for seed in (20261004, 20261005, 20261006))
    html_doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Crypto timing · redesign v3</title>
<style>
:root{{--bg:#0b0e19;--panel:#161c2e;--line:#303a50;--ink:#edf0fa;--muted:#aeb8cf;--mint:#22b8a5;--violet:#a885fa;--amber:#f5ab55}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}}
.wrap{{max-width:1200px;margin:auto;padding:32px 26px 90px}}header{{padding:52px 0 36px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-size:12px;letter-spacing:.18em;color:var(--mint);font-weight:700;text-transform:uppercase}}
h1{{font-size:clamp(32px,5vw,58px);line-height:1.12;margin:14px 0;letter-spacing:-.03em}}h2{{font-size:25px;margin:58px 0 18px}}
h3{{font-size:18px;margin:25px 0 8px}}p{{max-width:930px;color:var(--muted)}}strong{{color:var(--ink)}}
.lead{{font-size:18px;max-width:900px}}.pill{{display:inline-block;border:1px solid #634b47;color:#ffcfb5;background:#31221e;border-radius:100px;padding:5px 13px;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}}.card,.figure,.note{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px}}
.card .num{{font-size:33px;font-weight:800;letter-spacing:-.03em}}.card small{{display:block;color:var(--muted)}}
.mint{{color:var(--mint)}}.violet{{color:var(--violet)}}.amber{{color:var(--amber)}}.figure{{margin:20px 0;padding:18px}}
.figure img{{width:100%;display:block}}.figure figcaption{{font-size:13px;color:var(--muted);padding:4px 10px 6px}}
.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-size:12px}}
.equation{{padding:20px;background:#111525;border-left:3px solid var(--violet);font-family:Georgia,serif;font-size:20px;overflow:auto}}
.flow{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.step{{flex:1;min-width:180px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}}
.step b{{color:var(--mint)}}.note{{border-left:3px solid var(--amber)}}.contents a,footer a{{color:var(--mint);text-decoration:none}}.contents{{display:flex;gap:18px;flex-wrap:wrap;margin-top:25px}}
@media(max-width:800px){{.grid,.twocol{{grid-template-columns:1fr}}.wrap{{padding:24px 16px}}table{{font-size:12px}}th,td{{padding:7px}}}}
</style></head><body><div class="wrap"><header><div class="eyebrow">Research report · 04 Oct 2026 · Version 3</div>
<h1>加密货币择时模型<br>预测目标与训练结构重设计</h1><p class="lead">把 4 小时可执行收益拆解为<span class="violet">条件方向</span>与<span class="mint">幅度概率</span>，以真实时间顺序验证神经网络是否提供简单模型没有的信号。</p>
<span class="pill">研究结论：暂不部署</span><nav class="contents"><a href="#contract">时间与数据</a><a href="#model">模型与训练</a><a href="#predictive">预测证据</a><a href="#trade">交易证据</a><a href="#conclusion">结论与复现</a></nav></header>
<div class="grid"><div class="card"><small>方向主网络</small><div class="num violet">101,573</div><small>参数；独立幅度网络 868 参数</small></div>
<div class="card"><small>校准期对线性方向 CE 增益</small><div class="num mint">+{cal['gain_vs_linear']:.4f}</div><small>四个月均为正；早停选择在此期进行</small></div>
<div class="card"><small>后续选择期对线性方向 CE 增益</small><div class="num amber">{sel['gain_vs_linear']:+.4f}</div><small>优势未延续；2026 历史同样不支持部署</small></div></div>
<section id="contract"><h2>01 · 时间、数据与研究合同</h2><p>12 个永续合约、统一 5 分钟 K 线。整点仅使用已完成 bar；5 分钟后按下一开盘价进入，4 小时后按开盘价退出。训练期截至 2025-06-30，2025-07–10 用于逐轮早停，2025-11–2026-01 用于交易规则选择。2026-02–09 曾在旧研究中查看，本文只作探索性历史诊断，<strong>不是新的独立测试集</strong>。</p>
<div class="flow"><div class="step"><b>Train</b><br>2023 → 2025-06<br>253,944 条小时×合约标签</div><div class="step"><b>Early stop</b><br>2025-07 → 10<br>35,376 条</div><div class="step"><b>Selection</b><br>2025-11 → 2026-01<br>26,448 条</div><div class="step"><b>Seen history</b><br>2026-02 → 09<br>67,920 条</div></div>
<p>每条训练标签须在本期截止前成熟；标准化、因果波动下限、方向先验和档位代表值只由训练期拟合。小时相位每轮完整覆盖，按分散小时组成 24×12 的批次。训练输入包含过去 12 小时的 5 分钟序列、72 小时的完整小时序列及少量 7/30 日状态。</p></section>
<section id="model"><h2>02 · 标签、结构与优化</h2><div class="equation">G = P<sub>exit</sub> / P<sub>entry</sub> − 1 &nbsp; · &nbsp; Y = G / v<sub>past</sub><br>q<sub>j</sub> = P(|Y| 在第 j 档 | X) &nbsp; · &nbsp; π<sub>j</sub> = P(Y &gt; 0 | J=j, X)</div>
<p>幅度档固定为 |Y| 的 [0,.5)、[.5,1)、[1,2)、[2,∞)。方向主干只接受真实幅度档对应的 BCE 梯度；独立小网络接受幅度 CE。快 96/慢 48 的两层因果卷积与 32 维状态投影直接拼接，融合到 96 维后经过一个残差块，输出共享基础方向 logit 和四个收缩档位偏移。没有波动辅助损失更新方向主干。</p>
<div class="flow"><div class="step"><b>快序列</b><br>144 × 16 + mask<br>TCN 96 × 2 → 末态/近1h/全窗</div><div class="step"><b>慢序列</b><br>72 × 10 + mask<br>TCN 48 × 2 → 末态/全窗</div><div class="step"><b>状态与融合</b><br>24 + mask → 32<br>直接拼接 → 96 → 1 残差块</div><div class="step"><b>独立双输出</b><br>方向 π₁…π₄<br>风险 q₁…q₄ → 收益读出</div></div>
<p>训练使用 AdamW、2e−4 峰值到 2e−5 余弦衰减、首轮 warmup、时序 dropout .05、融合 dropout .10、梯度裁剪 1.0、最多 12 个完整覆盖轮、耐心 4 轮。三个预设随机种子分别在第 4、3、4 轮达到最佳校准期权重。</p>
<figure class="figure">{_asset('training_dynamics')}<figcaption>虚线是训练在线损失，实线是校准期方向 BCE；彩色圆点是冻结的最佳权重。训练持续改善，前向期不随之改善。</figcaption></figure>
<table><thead><tr><th>随机种子</th><th>最佳 epoch</th><th>Jul–Oct 相对线性 CE 增益</th><th>Nov–Jan 相对线性 CE 增益</th></tr></thead><tbody>{seeds}</tbody></table></section>
<section id="predictive"><h2>03 · 预测增量的证据</h2><p>条件方向的交叉熵必须与各档训练期常数先验、同状态低容量 logistic 比较。下表正值表示神经网络优于线性模型。置信区间是按完整跨币面板做 7 日循环移动块、2,000 次重采样的描述性区间；校准期使用它选择 epoch，因此区间不修正模型选择偏差。</p>
<table><thead><tr><th>时期</th><th>网络 BCE</th><th>线性 BCE</th><th>增益</th><th>95% 块重采样区间</th></tr></thead><tbody>{period_rows}</tbody></table>
<figure class="figure">{_asset('monthly_direction_gain')}<figcaption>按月等权选权重，不把四个交易相位当成独立实验。竖线分别标出早停与选择期、选择期与已查看历史的边界。</figcaption></figure>
<figure class="figure">{_asset('band_direction_gain')}<figcaption>仅评价样本真实所属的幅度档。尾档约占训练标签 4.97%，不能靠低幅度档的改善掩盖尾档方向。</figcaption></figure>
<h3>逐合约方向增量</h3><p>下表是条件方向 BCE 相对同状态 logistic 的增益。三个时期使用同一冻结权重，任何单币改善都没有被重新筛选为新策略。</p>
<table><thead><tr><th>合约</th><th>Jul–Oct 校准</th><th>Nov–Jan 选择</th><th>已查看 2026 历史</th></tr></thead><tbody>{symbol_rows}</tbody></table>
<h3>幅度是可学的，复杂幅度模型没有稳定增量</h3><table><thead><tr><th>时期</th><th>小 MLP CE</th><th>线性风险 CE</th><th>常数先验 CE</th></tr></thead><tbody>{risk_rows}</tbody></table>
<figure class="figure">{_asset('magnitude_ce')}<figcaption>幅度网络对常数明显改善，但和线性风险模型相比几乎持平；不应把幅度收益算作方向主干的功劳。</figcaption></figure>
<p>收益读出使用仅由训练期估计、向正负合并档位均值收缩的正负代表幅度，最后逐样本乘回可见的 v。尾档代表值约为负侧 {selection['representatives']['values'][3][0]:.3f}、正侧 {selection['representatives']['values'][3][1]:.3f} 个尺度单位，不以任意整数或固定上限代替真实尾部。</p>
<h3>低自由度概率校准的诊断</h3><p>只在 Jul–Oct 校准期拟合共享的方向预测偏移收缩系数 {calibration['direction_shrinkage']:.3f} 和幅度概率系数 {calibration['magnitude_shrinkage']:.3f}，向训练期先验收缩，不使用逐币或逐档校准器。下表只做敏感性诊断，不改变此前冻结的交易规则；早停也使用 Jul–Oct，因此该期校准结果有双重使用风险。</p>
<table><thead><tr><th>时期</th><th>原始方向 CE 增益</th><th>收缩后 CE 增益</th><th>原始收益相关</th><th>收缩后收益相关</th></tr></thead><tbody>{calibration_rows}</tbody></table>
<p>收缩使已查看历史的方向交叉熵相对线性基线略为正，但没有改善收益相关，选择期对线性模型的方向增益仍为负。概率评分更好并不能自动转化成可交易收益。</p>
<figure class="figure">{_asset('score_deciles')}<figcaption>按预测简单收益从低到高做全样本十分位。曲线包含共同市场漂移，不能单独视为单币择时 alpha；交易效果还需扣费用、资金费和换手。</figcaption></figure></section>
<section id="trade"><h2>04 · 交易映射与经济验证</h2><p>12 币等权，−1/0/+1 仓位；四个互不重叠的 4 小时相位分别计算。单边费用和滑点假设为 6bp，并按真实仓位变化收费；资金费按事件区间结算。只在 2025-11–2026-01 从 0、6、12、18bp 预测收益阈值中选择，要求平均参与率至少 5%。四相位共享行情，Sharpe 是描述性比较，不是四份独立置信证据。</p>
<div class="twocol"><div class="card"><small>选择期最佳经济规则</small><div class="num mint">{html.escape(str(winner))}</div><small>阈值 {thr*1e4:.0f}bp；四相位平均净 Sharpe {win_sel['mean_sharpe']:+.2f}；参与率 {win_sel['mean_active_fraction']:.1%}</small></div>
<div class="card"><small>相同冻结规则在已查看历史</small><div class="num amber">{win_seen['mean_sharpe']:+.2f}</div><small>四相位平均净 Sharpe；参与率 {win_seen['mean_active_fraction']:.1%}，低于选择期门槛</small></div></div>
<p>方向神经网络在选择期各阈值的平均净 Sharpe 最高仍为 {max(v['mean_sharpe'] for v in selection['selection_options']['network'].values()):+.2f}。经济胜者来自线性方向 + 神经幅度的组合，并且后续已查看历史未保持正收益；因此网络主策略没有完成“方向增量 → 预期收益 → 净交易优势”的三段证据链。</p>
<figure class="figure">{_asset('strategy_sharpe')}<figcaption>左柱是各模型选择期阈值中的最好值；右柱是固定或展示用阈值在已查看历史中的表现。不同柱阈值并不总相同，图用于辨认时期衰减，不能用来重新选模。</figcaption></figure>
<figure class="figure">{_asset('signal_sources')}<figcaption>整体相关、同时刻 12 币平均分数与平均收益的时间相关、去掉同时刻市场均值后的相关。选择期的收益相关主要来自共同市场时点，不能解释为 12 币相对排序优势。</figcaption></figure>
<h3>零固定成本与 6bp/侧成本</h3><p>费用不是全部失败原因：选择期网络在零固定成本下为正、加入交易成本后为负；已查看的 2026 历史即使零固定成本也没有转正。资金费仍计入两列，零成本仅去掉固定费和滑点情景。</p>
<table><thead><tr><th>规则与时期</th><th>零固定成本净 Sharpe</th><th>6bp/侧净 Sharpe</th></tr></thead><tbody>
{_row('网络 · 选择期', [f"{zero['selection']['network']['mean_sharpe']:+.2f}", f"{selection['selection_options']['network']['0.0012']['mean_sharpe']:+.2f}"])}
{_row('网络 · 已查看历史', [f"{zero['historical']['network']['mean_sharpe']:+.2f}", f"{historical['scores']['network']['mean_sharpe']:+.2f}"])}
{_row('冻结线性方向 · 选择期', [f"{zero['selection'][winner]['mean_sharpe']:+.2f}", f"{win_sel['mean_sharpe']:+.2f}"])}
{_row('冻结线性方向 · 已查看历史', [f"{zero['historical'][winner]['mean_sharpe']:+.2f}", f"{win_seen['mean_sharpe']:+.2f}"])}
</tbody></table>
<table><thead><tr><th>冻结线性方向策略</th><th>选择期</th><th>已查看历史</th></tr></thead><tbody>
{_row('平均年化换手成本贡献', [f'{cost_sel:.1%}', f'{cost_seen:.1%}'])}
{_row('平均年化资金费贡献', [f'{funding_sel:.1%}', f'{funding_seen:.1%}'])}
{_row('平均参与率', [f"{win_sel['mean_active_fraction']:.1%}", f"{win_seen['mean_active_fraction']:.1%}"])}
</tbody></table>
<h3>开放尾档代表值敏感性</h3><p>只改变已冻结规则读出中的最高幅度档代表值，不重新选阈值。它是对均值近似的压力测试，不构成新的策略选择。</p><table><thead><tr><th>已查看历史</th><th>四相位平均净 Sharpe</th><th>参与率</th></tr></thead><tbody>{tail_rows}</tbody></table>
<div class="note"><strong>防止“测试复活”：</strong>2026 年 2–9 月历史不能因为本次换了标签和网络就重新成为独立测试。当前研究决策是 no_deploy。确认性检验需要在本报告与代码锁定后，等待全新、未参与设计的新数据；也可做多折向前验证，但每折所有训练拟合量都必须重拟合。</div></section>
<section id="conclusion"><h2>05 · 结论、限制与复现</h2><p><strong>训练系统已经正常优化。</strong>三次方向训练从约 0.690 的在线 BCE 下降到约 0.671–0.674，验证峰值却出现在第 3–4 轮，之后反弹。早停保护了权重，但并没有创造可迁移的方向映射。小幅信号在校准期跨四月出现，后续对线性基线转负；幅度网络的改善大体可由简洁风险变量解释。这更符合非平稳弱信号与模型灵活度所致的泛化问题，而非梯度无法更新、样本数量太少或标签尺度数值下溢。</p>
<p><strong>未解决的经济问题：</strong>分档概率只近似收益分布，档内幅度仍用训练期常数；开盘价成交是假设，6bp/侧是情景成本，尚无盘口容量约束；均值预测的绝对校准和阶段漂移仍弱。不能把选择期最佳 Sharpe 当真实未来收益保证，也不能据此扩大模型或微调旧测试区间。</p>
<p>下一轮研究应先做跨市场状态的因果机制检查、分数中共同市场成分与单币时间成分拆分、训练期拟合的轻量概率收缩和滚动折稳定性。保持当前模型与规则冻结，再用新数据决定是否值得部署；不对已查看历史重搜窗口、阈值或收益符号。</p>
<footer><p>复现：<code>scripts/build_redesign_cache.py</code> → <code>scripts/run_redesign_baselines.py</code> → <code>scripts/train_redesign.py</code>（三方向种子、一幅度种子）→ <code>scripts/audit_redesign.py</code> → <code>scripts/evaluate_redesign.py --phase select</code> → <code>scripts/calibrate_redesign.py</code> → <code>scripts/evaluate_redesign.py --phase historical</code> → <code>scripts/build_redesign_report.py</code>。审计核对了 {audit['source_sha_verified']} 个原始文件 SHA、{audit['sampled_raw_target_checks']} 个随机目标与执行时点、{audit['checkpoint_manifests_verified']} 份最佳权重的数据合同。训练/预测工件位于 <code>outputs/redesign/</code>；结果 JSON 位于其 <code>evaluation/</code>。原设计依据见 <a href="../../docs/TRAINING_REDESIGN_2026-10-04.md">重设计文档</a>，旧结果见 <a href="../2026-10-03/README.md">既有报告</a>。</p></footer></section>
</div></body></html>'''
    (OUT / "index.html").write_text(html_doc, encoding="utf-8")
    readme = f"""# 加密货币择时重设计 v3：训练与验证结果

完整视觉报告：[打开报告](index.html)。生成日期：2026-10-04。

**研究结论：暂不部署。** 方向网络参数 101,573，幅度网络参数 868。2025-07–10 的方向 BCE 相对线性模型增益 {cal['gain_vs_linear']:+.5f}，2025-11–2026-01 为 {sel['gain_vs_linear']:+.5f}；此前已被查看的 2026-02–09 历史为 {seen['gain_vs_linear']:+.5f}。三种随机种子最佳轮次为 4、3、4。

| 时间段 | 网络方向 BCE | 线性方向 BCE | 增益 | 7日移动块95%区间 |
|---|---:|---:|---:|---:|
| 2025-07–10，早停 | {cal['ce']:.5f} | {cal['linear_ce']:.5f} | {cal['gain_vs_linear']:+.5f} | [{ci_cal[0]:+.5f}, {ci_cal[1]:+.5f}] |
| 2025-11–2026-01，规则选择 | {sel['ce']:.5f} | {sel['linear_ce']:.5f} | {sel['gain_vs_linear']:+.5f} | [{ci_sel[0]:+.5f}, {ci_sel[1]:+.5f}] |
| 2026-02–09，已查看历史 | {seen['ce']:.5f} | {seen['linear_ce']:.5f} | {seen['gain_vs_linear']:+.5f} | [{ci_seen[0]:+.5f}, {ci_seen[1]:+.5f}] |

幅度网络 BCE：校准 {risk[0]['ce']:.5f}，选择 {risk[1]['ce']:.5f}，历史 {risk[2]['ce']:.5f}；幅度线性模型分别为 {risk[0]['linear_ce']:.5f}、{risk[1]['linear_ce']:.5f}、{risk[2]['linear_ce']:.5f}。幅度任务可学，但神经结构的增量不稳定。

选择期最高净 Sharpe 是 `{winner}`、阈值 {thr*1e4:.0f}bp、四相位平均 {win_sel['mean_sharpe']:+.2f}；同一冻结规则在已查看历史的平均净 Sharpe 为 {win_seen['mean_sharpe']:+.2f}。采用 6bp/侧、逐次换手成本与事件资金费。神经网络选择期所有预定阈值的最好净 Sharpe 仍为负。

训练在线损失下降，验证最优出现在第 3–4 轮。当前证据指向弱、非平稳方向信号的泛化失败，并不支持“梯度不更新”或“只需增加分钟/盘口数据”的解释。2026 历史曾被前次研究查看，不能称为新方案的独立测试；真正确认要等锁定方案后新到达的数据。

原始数据合同审计：12 个文件 SHA、576 个随机执行收益和标签、4 份模型检查点与缓存一致。零固定成本时，方向神经网络选择期平均 Sharpe {zero['selection']['network']['mean_sharpe']:+.2f}，已查看历史 {zero['historical']['network']['mean_sharpe']:+.2f}；6bp/侧分别为 {selection['selection_options']['network']['0.0012']['mean_sharpe']:+.2f} 和 {historical['scores']['network']['mean_sharpe']:+.2f}。

共享先验收缩校准在 Jul–Oct 拟合方向系数 {calibration['direction_shrinkage']:.3f}、幅度系数 {calibration['magnitude_shrinkage']:.3f}。后续选择期方向 CE 相对线性模型仍为 {calibration['periods']['selection']['adjusted_direction']['gain_vs_linear']:+.5f}；此前已查看的历史虽为 {calibration['periods']['historical']['adjusted_direction']['gain_vs_linear']:+.5f}，收益相关却仍为 {calibration['periods']['historical']['adjusted_score_return_correlation']:+.4f}。这是诊断，不重选策略。

流程：`scripts/build_redesign_cache.py` → `scripts/run_redesign_baselines.py` → `scripts/train_redesign.py`（方向三种子、幅度一种子）→ `scripts/audit_redesign.py` → `scripts/evaluate_redesign.py --phase select` → `scripts/calibrate_redesign.py` → `scripts/evaluate_redesign.py --phase historical` → `scripts/build_redesign_report.py`。源设计：[TRAINING_REDESIGN_2026-10-04.md](../../docs/TRAINING_REDESIGN_2026-10-04.md)。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"built {OUT / 'index.html'} and {OUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
