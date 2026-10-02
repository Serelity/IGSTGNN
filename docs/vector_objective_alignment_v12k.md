# v12k：向量模块的训练目标与选择目标对照

## 从 v12j 得到的依据

2026-10-02 用户回传 gpu11 的 `contra_v12j_support_audit_01`。提交为
`3a8457c`，36 项预检测试通过，工作流退出码为 0。以下为服务器报告证据，
不是本地独立复现。

1,010 个审计预测窗口中，1,003 个属于预定粗状态支持范围，占候选早期有效
单元的 99.633%；7 个范围外窗口占 0.367%。八个联合格均有 167–211 个拟合
窗口，覆盖全部 16 个拟合周。六个向量模型在 supported 组的候选 H1–H6
收益为 -0.023358 至 -0.015503，四周块区间全部低于零；等窗口点收益也均负。
按拟合期格权重重新加权后，审计收益仍为 -0.018963 至 -0.012270。

这些证据削弱“少数明显超出拟合范围的窗口解释主要损害”的假设，但粗状态
支持不等于高维表示重叠，也没有证明过拟合或排除所有分布变化。现有分组不能
直接成为部署路由。v12g 已观察到选择期候选早期退步仍被全局指标选中，v12i
确认完整拟合期早期有小幅收益；因而本轮检验优化及选择目标的对齐问题。

v12e 已试过标量节点门控的三区域等权损失。v12k 的对象是原 v12f 两个向量
适配器，训练对照为全局与候选早期损失；选择规则作为第二个独立因素，避免
同时修改两项后无法区分贡献。这是一项新的重复开发对照，不预称为修复或成功。

## 冻结的 2 × 2 比较

保留 `state_vector`、`interaction_vector`，各 4,288 个可训练参数；输入、
初始化、原 A 骨干、归一化、图、解码器和 TIID 均沿用 v12f。无新增数据源，
不接入 v8c 路由或 v12j 支持分组。候选处的表示注入仍可能改变非候选及晚期输出。

| 训练数据损失 | global 选择 | candidate_early 选择 |
| --- | --- | --- |
| global：全部有效单元标准化 MAE | 原全局选择规则 | 早期收益门槛及排名 |
| candidate_early：候选 H1–H6 有效单元标准化 MAE | 原全局选择规则 | 早期收益门槛及排名 |

两种损失均加原 `0.001 × mean(r²)` 候选通道正则。早期损失不额外缩放，空
早期目标支持直接报错；其他区域不进入其直接数据损失，但仍接受选择保护。
global 路径直接调用原 v12f 训练函数。它是同预算重跑对照，不把旧 v12f 数值
无条件当成本轮复现结果。

每个架构、损失、种子只训练一条轨迹，两种 selector 同时观察该轨迹。没有
按 selector 再训练、提前停止或改变样本顺序。两个架构、两个损失、三个种子
共 **12 次拟合、144 个正式 epoch、24 个已选端点**。同种子的四条轨迹共享
精确相同的适配器张量初值与逐 epoch 样本排列。

## 选择和保护规则

所有端点保留原 16 项点约束：四 cohort 的全局、候选早期、候选晚期、非候选
MAE 均不超过对应 A 的 1.001 倍；无效指标拒绝该 epoch。epoch 0 始终可选。

- global：按完整事故全局 MAE 选最低值，严格改善才替换，平局保留较早轮。
- candidate_early：先要求完整事故全局 MAE 严格优于 A，且候选 H1–H6 MAE
  严格优于 A，再按候选早期 MAE 选最低值；否则保留 A。它改变早期收益的
  准入要求与排名，不能把该对比解释为只换排名。

因此所有非零已选端点必须在选择期严格改善完整事故全局 MAE。选择保护只是
点估计筛选，不保证审计期无害。拟合指标、审计指标、种子优劣或分组标签不
参与选择。所有 12 条轨迹及 24 个端点先冻结，之后才评估任何审计目标。

## 数据与报告

继承 W01–16 拟合 1,520 个完整事故窗口；W18–25 选择 856 个完整窗口及
206 组共同三元组；W27–35 审计 1,010 个完整窗口及 378 组三元组；W17/W26
保持间隔。只用拟合事故 Y 优化。两类对照 Y 用于选择保护及后续报告。旧 val
和 test 数组不读取；输入哈希仍读取整个 train 文件字节，不能误称完全未接触
包含后段目标的文件。原 A 和 scaler 的广泛 train 使用及旧 val 选择依赖仍存在。

本轮一次保存 A 与所有已选端点的完整拟合、选择、审计逐窗口误差、有效计数、
预测计数、样本 ID、源行号和候选 mask。完整拟合只含正例；选择、审计含四个
cohort。固定模型完整拟合评估与训练在线损失分别记录。epoch 0 的误差必须
精确等于 A，已选选择期指标必须通过相同 batch size 的精确回放。

主关注区域为完整事故候选 H1–H6，另外并列报告全局、候选晚期、非候选及
常规对照。保留全部种子与以下配对比较：每个端点相对 A；固定架构/损失的
selector 差异；固定架构/selector 的损失差异；固定损失/selector 的上下文
交互差异。不挑最佳种子或配置，不以组合改动推断某个单独因素有效。

每阶段分别使用相同日历网格与共享 2,000 次周、四周循环块抽样，报告单元
合并与等预测窗口收益及 95% 逐项区间。等窗口不等于等独立事故；拟合/选择
区间是样本内或选择条件下的描述，审计也已反复用于研究，不是独立确认。
不构造跨阶段配对区间，不设置自动升级训练或测试集访问的门槛。

## 服务器运行

本轮需要重新反向传播，默认使用已分配 GPU。代码提交推送后，在有足够剩余
时间的计算作业中执行；总预算比 v12f 多三条轨迹，最终还会完整评估 24 个
端点，实际耗时以日志为准。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_vector_objective_alignment.sh run contra_v12k_objective_alignment_01
```

这是前台入口。先做设备预检、单元测试，以及两架构 × 两损失 × 一个种子 ×
两轮的真实 checkpoint 工程检查，通过才正式运行。保持运行终端打开，运行中
不要更新代码。另开终端查看：

```bash
tail -f experiments/chronological_runs/contra_v12k_objective_alignment_01.job/run.log
bash experiments/chronological/run_vector_objective_alignment.sh status contra_v12k_objective_alignment_01
bash experiments/chronological/run_vector_objective_alignment.sh report contra_v12k_objective_alignment_01
```

默认设备 `cuda:0`；`V12K_DEVICE` 可以指定另一已分配设备，CPU 仅适合工程
测试。日志保存主机、PID、Slurm ID、退出码和内存观测；没有 summary 时报告
明确提示未完成。已有输出和符号链接（包括悬空链接）均拒绝覆盖。

中断后确认原进程结束，在有效 GPU 作业内恢复到新运行名：

```bash
bash experiments/chronological/run_vector_objective_alignment.sh resume contra_v12k_objective_alignment_02 contra_v12k_objective_alignment_01
```

来源为原 `.partial`，保持只读。每个完整 epoch 原子保存当前适配器、优化器、
随机状态、完整历史以及两个 selector 的历史最佳状态；epoch 0 也保存。
恢复核验代码、协议、输入、样本、设备和版本身份，重放两个选择规则并校验
状态哈希。已完成的拟合不重新优化，最终三阶段评估重做。不接受旧 v12f/v12e
状态，不在代码改变后绕过恢复检查。没有完整 epoch 时，该轨迹从头训练。

输出包含 `summary.json`、`comparisons.csv`、三阶段周统计、
`selected_endpoints_frozen.json`、每条轨迹的 `history.json`、恢复 checkpoint、
两个已选 checkpoint，以及每个端点的三阶段 NPZ。

冻结协议 SHA-256：
`896a2969389c262aaf7b8dea78f969ab1d10337946bc4c5d6a5ba4a15e9620be`。

本地合成工程测试不能替代真实 A、完整数据和服务器 GPU 验证；本轮尚无科学结果。

## 本地验证记录

2026-10-02，在 WSL `igstgnn-audit`（Python 3.10.21、PyTorch 2.3.1+cpu、
NumPy 1.24.4）通过 24 项新增测试和 59 项强度门控、向量模块、区域目标相关
回归，共 83 项。覆盖早期损失梯度支持、两种选择规则与 A 回退、完整小网络
流程、审计目标延迟访问、对照目标不进入优化器、两个历史最佳状态保存、两种
损失的中断恢复、完成来源不重训、共享重采样、空支持、配对权重和启动失败码。
小模型上 global 轨迹的最终及已选权重与原 v12f 训练/选择实现逐张量一致。

真实本地冻结清单的资格重算得到 1,520 / 856 / 1,010 个完整事故窗口，以及
选择、审计阶段 206 / 378 组三元组，与冻结预算一致；此项只读取 manifest，
未读取目标或运行真实模型。Bash 语法、CLI 和新增文件空白检查通过。

```bash
python -m unittest discover -s tests -p 'test_vector_objective_alignment*.py' -v
bash -n experiments/chronological/run_vector_objective_alignment.sh
```
