# 2bp纯手续费曲线图集

## GitHub 快照的文件范围

2026.10.5 快照保留本图集的 HTML、SVG/PNG、`evidence.json` 与 `manifest.json`。`derived_paths.npz` 是本地生成的二进制派生数组，按照本轮数据排除规则不上传；HTML 中该数组的下载入口仅在本地文件存在时可用。图表展示已经内嵌必要的可视化结果，不依赖下载该数组。需要数组时，用本地已有数据和 `scripts/build_2bp_curve_gallery.py` 重新生成。原 manifest 中的数组哈希保留作历史核验，文件省略不改变已有收益或成交路径。

[打开HTML](index.html)。复用保存结果绘图，无训练、推理或交易规则重跑。2025年1月起显示信号和基准，保存策略账本从7月起；总账户与十二合约各含净值、持仓、信号，支持主方案/近满仓对照切换。净值仅价格PnL减已存2bp手续费；具体语义与校验见[evidence.json](evidence.json)。

2026-10-05状态确认：本图集没有将主方案的设计门槛同步降低，也不模拟maker挂单成交。它保留原含资金费账户的实际成交份额，再剔除资金费重述净值，不能当作重新决策的零资金费账户。

[新仓位映射设计](../../docs/SIGNAL_POSITION_MAPPING_DESIGN_2026-10-05.md)仅提出下一阶段方案，未修改本图集或已有策略。[综合研究HTML](../alpha_strategy_2026-10-04/index.html)与[正式Markdown报告](../../docs/ALPHA_STRATEGY_RESEARCH_REPORT_2026-10-04.md)保留原口径。后续本类HTML主要承载逐币信号、目标/实际仓位、成交与净值，综合比较另放总报告，两者共享同一结果版本。
