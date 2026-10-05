# v4 特征与监督字典

所有输入由更新后 `1m.parquet` 的原始列重算。源时间为 bar open，字段在 close_time+1ms 完整可见。UTC 整小时为决策点，时间偏移不是“以日期列直接并表”。一分钟/五分钟/小时每频率独立计算风险与活动基准；此处描述变换前的经济字段，训练 scaler 另按折拟合。

记 r=log(C/C_prev)、σ=此前收益平方 EWMA 的平方根（span24h，shift1，数值下限1e−7）；Q为USDT成交额，N为成交笔数，B为主动买额；Q̄/N̄为此前24h EWMA。β用此前168小时协方差/方差，k用此前24h压力与收益的平方/乘积和估计。压力 F=(2B−Q)/Q，Q=0时缺失。

## 五分钟字段（原16、扩展19、精选14）

| 名称 | 公式/量纲 | 机制/绝对状态 | 缺失/零点 | 精选组 |
|---|---|---|---|---|
| return | log(C/C_prev) | 基础路径、漂移与冲击 | 0无收益；首行缺失 | 是 |
| body | log(C/O) | 实体，与return近复制 | 0无实体 | 否 |
| range | log(H/L) | 区间活动 | 0无振幅 | 是 |
| upper_wick | log(H/max(O,C)) | 上影；与区间职责重叠 | 0无上影 | 否 |
| lower_wick | log(min(O,C)/L) | 下影；与区间职责重叠 | 0无下影 | 否 |
| close_location | 2(C−L)/(H−L)−1 | 区间位置、有界 | H=L时0；自然[−1,1] | 是 |
| quote_surprise | log((Q+ε)/(Q̄+ε)) | 相对活动 | 0等于过去算术EWMA；无成交大负值另有标志 | 是 |
| trades_surprise | log((N+ε)/(N̄+ε)) | 相对成交频度 | 同上；高相关不等于经济完全同义 | 是 |
| pressure | (2B−Q)/Q | 买卖平衡 | 0平衡；[−1,1]；无成交缺失 | 是 |
| vwap_gap | log((Q/volume)/C) | 聚合成交均价相对收盘 | 0相等；Q/volume无效则缺失 | 是 |
| illiquidity | log(1+10⁶·abs(r)/max(Q,.01Q̄)) | USDT固定参考下的冲击/量额代理 | 不是真实价差；低成交分母由过去量额下限约束 | 是 |
| btc_return | BTC同频r | 共同市场 | 0无共同收益 | 是 |
| eth_return | ETH同频r | 第二共同市场入口，与状态ETH信息重复 | 0无共同收益 | 否 |
| breadth | 2·mean(r_i>0)−1 | 12币共同方向广度 | [−1,1]，0约半数上涨 | 是 |
| absorption | r−kF | 交互/压力未解释价格响应 | 压力缺失则缺失 | 是 |
| btc_residual | r−β·r_BTC | 相对于共同市场的偏离 | 不是输出强制中性 | 是 |
| quote_level | log(Q̄+ε) | 长期活动规模，状态已有入口 | ε=10⁻⁹ USDT | 否 |
| trade_size | log((Q+ε)/(N+ε)) | 单笔规模、与频度分工 | 单位USDT/笔，无成交由标志识别 | 是 |
| no_trade | 1[Q=0] | 基础数据可靠性 | 二元0/1，不当作卖出压力 | 是 |

`return_copy`只供冗余数学审计，所有最终训练输入均排除。原字段组使用上述前16项；扩展组为前19项；机制精选为14项；随机组用固定种子从19项取14项，保持删除数量一致。没有从验证标签拟合筛选阈值。

## 三套预处理

| 方案 | 价格/有符号字段 | 自然有界字段 | 后续数值整理 |
|---|---|---|---|
| raw 静态控制 | 原对数收益/价格比；已定义log活动字段 | 仍逐币居中缩放，复现统一整理假设 | 训练中位数、IQR/1.349、±8截尾 |
| field 字段适配 | 五分钟asinh(x/.001)，小时asinh(x/.01) | 保留零点和原界限 | 其他字段训练中位数/IQR |
| dynamic 形状+状态 | asinh(x/σ_past)，当前冲击不改自身分母 | 保留零点和原界限 | 同上；移除的风险/活动尺度在状态保留 |

上述价格字段为return/body/range/wicks/vwap_gap/BTC/ETH/absorption/residual。并非对所有字段强制正态化。clip仍存在尾部信息损失；mask代表数值有效，而无成交另有语义。单位变化实验验证压力、活动比值和相对价格表达；固定参考流动性代理必须同步转换USDT参考量，不能只改数据单位。

## 分钟输入与分钟聚合增量

一分钟短输入12字段：return_shape、body_shape、range_shape、close_location、pressure、quote_surprise、trades_surprise、vwap_gap_shape、log_sigma、illiquidity、no_trade、pressure_lag_response。最后一项为asinh(F_prev·r/σ_past)，只是聚合压力领先响应代理，不是逐笔订单流。

| 五分钟内增量 | 公式与职责 | 预处理 |
|---|---|---|
| pressure_leads_price | sum(F_j·r_(j+1))/σ5，五分钟内前四个相邻对 | asinh后训练整理 |
| price_leads_pressure | sum(r_j·F_(j+1))/σ5 | 同上；与上项分开 |
| quote_concentration | max(Q_j)/sum(Q_j) | 自然[0,1]，保留 |
| sign_continuity | 相邻一分钟收益符号相同的比例 | 自然[0,1]，保留 |
| peak_retrace | log(max(C_j)/C_end)/σ5 | asinh；上涨后回吐代理 |
| path_efficiency | abs(sum(r_j))/(sum(abs(r_j))+ε) | 自然[0,1]，保留 |

新增摘要使用全部五根已完成一分钟；短编码器输入最近180根一分钟，小TCN后按完整五分钟patch聚合，不是随意取一根代表五分钟。主监督仍每小时一份。

## 小时与背景状态

小时10字段：return、log_risk（该小时内五分钟r²和开方再log）、成交额加权pressure、quote/trades surprise、range、BTC收益、BTC残差、absorption、breadth。小时价格形状用小时过去σ；其他尺度不混用。慢输入比较72/168小时，分段读出保留年龄顺序。

24背景字段：1/4/24/168/720小时累计收益；1/24/168/720小时log realized risk；log(rv24/(rv168/√7))；1/4小时成交额加权压力；当前小时quote surprise；过去小时EWMA量额log；单笔规模log；β_BTC7d；BTC/ETH4小时收益；小时市场广度；log市场小时收益分散度；小时与周内sin/cos各两项。

动态方案累计收益分别除以σ_hour·√window再asinh，未做窗口去均值；风险log、短长风险比及活动规模负责保留背景。不存在“quote_surprise24h实际却是单小时”的含糊命名。周期字段与压力、广度保留原界限。β属于条件变量，不是永久把市场剔除的约束。

## 标签与时间合同

小时决策t，入场e=t+5min，到期e+H；G_H=open(e+H)/open(e)−1。v4=√(48·(.5 mean_past24h(r5²)+.5 mean_past7d(r5²)))，在本折训练期按币拟合5%尺度下限。1/4/8h尺度为v4·√(H/4)。

半方差使用e到e+4h相同边界内的1m或5m open-to-open log增量，U=sum(max(r,0)²)、D=sum(min(r,0)²)。T=(U+D)/(v4²+ε)、Q=(U−D)/(v4²+ε)、A=(U−D)/(U+D+.02v4²+ε)；logT目标为log(T+.02)。它们不是终点收益，也不是彼此等价。

四档按abs(Y4)边界0.5/1/2，精确零收益的方向监督为0.5；各档方向先验、幅度先验按本折训练期明确初始化。统一八类按负尾→近零→正尾排列；代表值来自本折训练期。条件方向评价可用事后档位，但主信号推理与选模必须使用预测路由后收益信号。纯路径例外只选自身任务，不冒称条件收益。

所有比较共用最长8h+5m标签成熟purge和未来8h无零成交的标签质量门槛；这只是质量限定评价对象，未来质量标志不进入模型。训练前预热720小时。训练、早停、后续回放按真实时间顺序；无跨折拟合scaler、分箱代表值或尺度下限。
