# v12c：保留原 A 初始输出的 ICSF 强度门控

## 来自 v12a/v12b 的依据

用户于 2026-09-27 回传 `contra_v12b_regions_01`，状态为
`ARCHITECTURE_REGION_AUDIT_COMPLETE`，协议 SHA256 为
`b7d5792496c549237886a620aa6b0bffe6633ff49f743656c41263b3dbd1e4ce`。
以下是服务器报告结果，不是本地独立复现。

| ICSF 相对 norm_only 的 raw MAE 收益 | 完整事故 | C1 | C2 |
| --- | ---: | ---: | ---: |
| 候选 H1–H6 | +0.116298 | −0.088120 | −0.098941 |
| 候选 H7–H12 | +0.182014 | −0.091665 | −0.075718 |
| 非候选全时段 | +0.027063 | +0.006347 | +0.008030 |

前两行事故收益及对照伤害，在周 bootstrap 和四周块敏感性下均未跨零。
共同事故减两类对照平均收益：早期 +0.196737（四周块区间 [0.152203, 0.244535]），
晚期 +0.236286（[0.194376, 0.277354]）；等样本权重也同向。
完整事故全局收益 0.036443 中，非候选区域按全局分母核算为 0.024984，
约占 68.6%。这是区域收支，不能解释为动态图的因果贡献率。
TIID 在候选早期对事故有小幅正收益，对两类伪事故对照有负收益。

这些发现支持测试“按当前状态调节已有事故注入”的假设，不能证明门控一定有效。
ICSF 的两个单事故 softmax 轴均退化为 1，但 V 表示及其下游传播仍可能有用。
因此本轮不重新预测有符号残差，不把非候选传播与晚期响应硬切断，也不同时修改 TIID。

## 模型与对比

保留生产 `igstgnn.py` 及其 checkpoint；训练脚本在严格加载原 A 后附加独立 wrapper。
在 `report_location_v1`、无传感器属性、单事故条件下，原 ICSF 注入等于 `mask * V`。
新最后一帧为 `LayerNorm(H + g * mask * V)`，其中 `g = 2 * sigmoid(logit)`。
输出 logit 初始化为零，所以初始 g 严格为 1；工程检查要求完整原模型预测逐元素相等。
不改 K/TIID 上下文、LayerNorm、图生成代码或解码器。

| 对比 | 可训练部分 | 496 节点生产模型中的新增参数 |
| --- | --- | ---: |
| A | 无 | 0 |
| scalar | 一个全局 logit | 1 |
| node | 100→16→1 的 Tanh MLP | 1,633 |

节点输入为原冻结流量嵌入的最后一帧、历史均值、首尾差，三类距离特征及 report_age/5。
不使用未来 Y、残差标签、样本 ID、cohort 标识或额外多通道包作为预测输入。
本轮隔离“状态依赖门控”与“全局调弱事故模块”，不把模型容量同时扩展到新残差专家。

骨干全部参数冻结，始终处于 eval 模式；梯度经过动态图和解码器传回门控。
这是真实神经网络梯度训练，服务器默认使用 CUDA，不再是纯 CPU 统计。
每 epoch 检查原权重/状态哈希、冻结参数无梯度及门控梯度有限，并记录门控是否更新；
首 epoch 必须有实际更新，后续可能的收敛停滞保留为结果。

## 时间与信息边界

只读原 train 数据包。按 manifest 时间、在读取目标值前确定：

| 用途 | ISO 周 | 事故样本 | 完整共同三元组 |
| --- | --- | ---: | ---: |
| 门控优化 | W01–W16 | 1,520 | 不用于优化 |
| epoch 选择及对照保护 | W18–W25 | 856 | 206 |
| 后段开发审计 | W27–W35 | 1,010 | 378 |

W17、W26 为间隔周。正事故完整 support 必须位于对应时段内；配对评估额外要求两类
对照的完整 support 也在同一时段内。未改变源三元组，只对本实验作显式时间资格筛选。
输出保留原始 ID、源索引、各时段选中与排除列表。初稿较窄的选择时段只留下 34 组三元组，
因此仅据 manifest 元数据扩为八周；没有依据 Y 或门控结果更改时间边界。

优化器仅使用 fit 完整事故的 Y。选择时段事故和两类对照的 Y 用于 epoch 选择与保护检查，
不会进入优化器。最终审计的 Y 不参与 epoch/seed/variant 选择。旧 val 与 test 数组不读取。

**这不是独立时间外验证。** 原 A 已在整个 train 时段训练，旧 checkpoint 使用 val 选过模型，
输入 scaler 也已使用完整 train X；这些时段此前被多轮研究检查过。
本轮只隔离新增门控拟合、选择与后段报告，不能消除继承的开发信息。

## 固定训练及选择规则

两种门控各用 seeds 2025/2026/2027，全部报告，不选最佳种子。
每次 12 epoch，batch 8、评估 batch 16；Adam lr=0.001，eps=1e-8，无 weight decay，
梯度范数上限 5。训练损失是所有节点/12 horizons 的有效 cell MAE（原标准化流量单位）
加上 `0.001 * mean((g-1)^2)`，后一项仅在候选节点上计算。
两种门控使用同一 seed 对应的相同样本排列和相同训练预算。

每 epoch 在选择时段评估 A 与当前模型。四个 cohort 的 all、候选 H1–H6、候选 H7–H12、
noncandidate_all MAE 都必须 ≤ 对应 A MAE × 1.001；缺失指标拒绝该 epoch。
这是 0.1% 相对退化上限的点估计筛选，不是有置信保证的保护。
符合约束的 epoch 按完整事故全局 MAE 最小值选择；严格改善才替换，平局保留较早 epoch。
原 A 对应 epoch 0 始终可选，因此“未选中任何训练后 epoch”是有效的否定结果。

后段报告三种配对差值：A−scalar、A−node、scalar−node（正数表示后者更好）。
每个种子单独给出 pooled cell MAE、等样本权重敏感性、共享周 bootstrap 和四周块区间，
沿用 v12b 的区域核算及事故/对照对比。不将种子当作独立样本拼进 bootstrap。
审计只有九个日历周，区间是固定已拟合模型的逐项描述性不确定性；无多重比较校正，
不能覆盖所有种子变异，也不能用于声称独立泛化或授权打开测试集。

## 运行与产物

在平台分配的 GPU 节点，使用原 `igstgnn` 环境：

```bash
git pull --ff-only origin research/chronological-tiid
bash experiments/chronological/run_incident_strength_gate.sh start contra_v12c_strength_gate_01
```

依次执行单元测试、真实 checkpoint 小样本检查（每时段/cohort 两条样本、两次 epoch、
两个门控），成功后才执行完整六次拟合。小样本结果只用于工程验证，不参与完整实验选择。
训练/评估有周期日志，每 epoch 保存历史、last gate 和优化器状态，最终保存 selected gate。
本版本不支持自动续跑；失败保留现场并使用新运行名。实际 GPU 耗时尚未测量。

```bash
tail -f experiments/chronological_runs/contra_v12c_strength_gate_01.job/run.log
bash experiments/chronological/run_incident_strength_gate.sh status contra_v12c_strength_gate_01
bash experiments/chronological/run_incident_strength_gate.sh report contra_v12c_strength_gate_01
```

`Ctrl+C` 仅退出日志跟踪。运行期间不要拉取代码。status 会提示运行主机，避免跨节点查 PID。
日志与退出码在 `.job`；处理中 `.partial/progress.json`，失败尽量保存 `failure.json`；
成功才原子发布最终输出。

输出包括冻结协议及输入/源码哈希、时间资格清单、所有 seed/variant 的 epoch 历史和门控权重、
A 与训练后模型的每样本区域误差统计和候选 gate 值、共享 bootstrap 权重及充分统计量。
`selected_gate.pt` 仅是适配器权重，必须与原冻结 A 及对应 scalar/node wrapper 一起加载，
不能当作完整模型 checkpoint。

## 验证记录

本地没有真实 A checkpoint；真实 A 的精确初始化及梯度检查由服务器启动器执行。
本地以真实模型代码的小型实例进行合成测试，不把合成训练结果当成科研效果。
测试覆盖 g=1 完整复现（含零支持/off）、仅门控有梯度、节点隐层在首次更新后获得梯度、
LayerNorm 前注入及 TIID 上下文不变、跨时段对照过滤、局部对照伤害拒绝、epoch 0 回退、
未来 Y 不进入预测输入、完整合成训练/选择/配对统计/原子输出、失败现场与启动器停止规则。

2026-09-27：WSL Conda `igstgnn-audit`（Python 3.10、PyTorch 2.3.1+cpu、NumPy 1.24.4）
通过 12 项新增测试及 51 项相关回归，共 63 项；Bash 语法及 Python 编译检查通过。
真实本地 train 输入、上下文及两类对照的冻结哈希全部通过。另用生产尺寸 496 节点模型的
随机权重及四个 cohort 各两条真实输入核验 scalar/node 的原生前向精确复现；用随机投影的
合成目标分别反向更新两次，门控梯度有限且非零、节点隐层在第二步收到梯度，骨干状态不变。
该检查没有使用真实未来 Y 优化，也没有报告真实 MAE；真实 checkpoint/GPU 检查仍由服务器执行。
