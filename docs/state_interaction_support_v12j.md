# v12j: 可观测状态支持与跨时段收益

## 依据与问题

v12i 在 gpu08 正常完成，31 项预检测试通过，退出码 0。六个固定向量模型
在完整拟合期候选 H1–H6 的 pooled MAE 收益为 +0.024757 至 +0.035676，
选择期为 -0.009453 至 -0.003559，审计期为 -0.027252 至 -0.018625。
等预测窗口加权也呈现拟合期正、审计期负。五个拟合期四周块区间高于零，
state_vector/2027 的拟合区间跨零。64 窗口回放的 pooled MAE 差最大约
3.8e-7，远小于这些变化，但这不是对全部窗口所有数值效应的排除证明。

因此完整拟合期并非完全没有改善；需要检验后续损害是否主要出现在拟合期
覆盖不足的可观测状态。v12h 共同组/补集的差异不足以直接构造部署路由，
v12e 已试过节点门控的区域损失，v12j 不重复训练或修改选择目标。

本轮问题限定为：在预先固定的粗粒度报告时状态定义下，拟合期与审计期的
支持覆盖和候选早期收益如何变化？它不能识别时间迁移失败的因果机制。

## 信息边界与来源

只读已完成的 `contra_v12i_full_fit_audit_01` 与其原始
`contra_v12f_state_interaction_01`，核对冻结协议、代码身份、样本预算、
保存模型身份，以及所有实际读取 NPZ 的哈希。v12i 记录的 v12f 输入哈希
必须与实际读取来源相同，避免把不同运行的拟合结果和审计结果混用。

读取 v12i 的 A 和九个已选模块完整拟合期数组，读取 v12f 对应的完整事故
审计数组；四路径的 ID、source index、候选 mask 和有效/预测单元数必须
一致，并与 v12i 的各阶段 MAE/支持统计对账。科学比较只报告两个向量臂
相对 A 的六组结果；strength 数组用于保持原四路径来源对账。

另读取原始 `train_manifest.csv`、`station_ids.npy`、`train_context.npz`，
并将 `train_flow.npy` 只读内存映射。所有文件先核对原输入身份。
数值特征只访问原合格拟合/审计窗口的 `X[0:12]`，不访问选择期 X、
未来目标槽位、控制交通、旧 val/test、scaler 或 checkpoint。
整份 train flow 文件的身份哈希会经过包含未来目标的字节；这些字节不作
数组解析、特征构造或阈值拟合。评估收益使用已保存的预测误差与目标有效计数。

本轮只依赖 NumPy 和标准库，在 CPU 上运行。没有模型加载、推理、优化器、
梯度、重新选择 epoch、自动 gate 或部署路由。选择期模块只有聚合误差，
因此不伪造选择期条件收益或周区间。

## 拟合期冻结规则

六种特征只使用报告时可见数据，原始流量不作全 train 标准化或缺失插补：

| 特征 | 定义 |
| --- | --- |
| history_mean | 候选节点 12 个历史槽位的有效单元平均流量 |
| history_trend | 每候选节点后六槽减前六槽有效均值，再对两半均有观测的节点取平均 |
| history_volatility | 每节点历史总体标准差，再对至少两次有效观测的候选节点取平均 |
| history_missing_fraction | 候选历史无效单元数 / (12 × 候选节点数) |
| report_age_minutes | t0 减 report_time，核对报告上下文，范围 (0,5] |
| candidate_node_count | 非零距离支持节点数，与保存模型 mask 核对 |

有效 flow 为有限且非负，零是真实观测；无可用观测的特征保持缺失，不能
以零或全 train 均值替代。波动使用节点内变化，不把不同节点流量均值差
当作时间波动。最后一个 X 槽位必须早于报告时刻。

各单特征分箱采用完整拟合期有限值的 Q25/Q50/Q75，线性分位数；重复和
最小值切点移除，切点相等时进入右侧箱。非恒定特征的最大值切点保留，
因为等于最大值的观测构成非空右侧箱；恒定特征合并为一箱，其余空箱明确保留。
范围外低值、高值和缺失各有独立组，
不得将审计极值裁剪进拟合箱。分箱不用误差、未来 Y 或目标有效计数。

联合状态仅对流量水平、趋势、候选规模作拟合期中位数划分，最多八格。
不构造六特征四分位数的稀疏高维笛卡尔积。每格必须至少有 30 个拟合
预测窗口、覆盖四个不同拟合日历周。这里窗口不是独立事故，30/4 是预定
诊断约定，不是已经验证的安全门槛。

每个阶段窗口按以下优先级归入一个且仅一个支持组：

1. missing_state: 任一特征缺失。
2. outside_fit_range: 任一有限特征超出完整拟合期最小/最大值，或没有有限拟合参照。
3. low_joint_support: 剩余窗口的联合格未满足上述拟合窗口/周数要求。
4. supported: 六特征均有拟合范围支持，且联合格满足要求。

联合格拟合支持只计六特征完整且在范围内的窗口。所有状态规则保持跨模型/
种子相同。supported 只指这些粗粒度摘要的支持，不能证明原始高维输入或
模型表示空间的充分重叠。

## 统计与解释

单特征各箱、四支持组均报告四区域的 A/模块 MAE、有效单元和可评估窗口数、
pooled 与等预测窗口收益、组质量占比及精确贡献。每组误差收益和除以完整
阶段区域或全局有效单元数；不得直接相加组 MAE 或按窗口比例推算单元贡献。
组贡献必须重构完整阶段收益，缺失/范围外/低支持窗口不会被静默丢弃。

按同一阶段完整日历网格保留空组周，六模型使用相同周及四周循环块抽样。
沿用 2,000 次抽样、95% 点区间和至少 95% 有效抽样的原规则。空支持或
抽样分母不足明确报告 undefined/insufficient 状态，不把缺失收益写成零。
拟合区间是样本内描述，审计区间是条件于已保存模型的重复开发描述。

构成核算只在两阶段均有该区域评估支持的 supported 联合格上进行。
以拟合期这些格的单元质量或等窗口质量固定权重，计算审计期标准化收益：

`审计共同格收益 - 拟合共同格收益`

`= (审计共同格收益 - 审计拟合权重标准化收益)`

`+ (审计拟合权重标准化收益 - 拟合共同格收益)`。

两项分别命名为构成项、格内项，仅为描述性代数。报告共同格覆盖质量、
排除的非支持质量和未共享支持质量；没有共同支持时保留未定义结果，不外推。
跨阶段窗口不配对，不构造差值 CI，不将两组单独 CI 当成组间差值检验。

结果用途是判断未来该优先检验适用范围还是修正本身的稳定性，并非自动进入
下一训练。若粗支持内仍退步，则这份粗分箱没有解释该损害；若范围外组损害
更大，也不能证明支持不足是原因或据此直接部署回退。不得选择有利分箱、
种子、周或修改门槛。A/scaler 的既有广泛 train 使用和旧 val 选择依赖仍然存在，
整个审计不是独立确认。

## 输出与服务器运行

输出为 `summary.json`、`state_membership.csv`、`conditional_gains.csv`、
`weekly_gains.csv`、`composition_accounting.csv`。完整发布前使用 `.partial`，
已有 final、partial、job 或符号链接均拒绝覆盖。中断后保留现场，用新运行名重跑。

代码提交推送后，在已分配的有效计算作业中执行，激活已有 igstgnn 环境：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_state_interaction_support_audit.sh run contra_v12j_support_audit_01
```

若来源运行名不同，可显式指定 v12i 与 v12f；来源仍须属于同一核验链：

```bash
bash experiments/chronological/run_state_interaction_support_audit.sh run contra_v12j_support_audit_02 contra_v12i_full_fit_audit_01 contra_v12f_state_interaction_01
```

另开终端查看；运行终端保持前台打开，不在运行中拉代码或修改协议：

```bash
tail -f experiments/chronological_runs/contra_v12j_support_audit_01.job/run.log
bash experiments/chronological/run_state_interaction_support_audit.sh status contra_v12j_support_audit_01
bash experiments/chronological/run_state_interaction_support_audit.sh report contra_v12j_support_audit_01
```

不需要 GPU 分配，但整份 train flow 身份哈希需要读磁盘。启动器记录主机、
PID、Slurm job ID、退出码，先执行 NumPy 预检及本轮测试，再开展正式统计。
本地验证使用合成数据，不声称已读取服务器真实结果或获得 v12j 科学结论。

## 本地验证记录

2026-10-02，在 WSL 的 `igstgnn-audit` conda 环境完成 119 项相关测试：
v12j 新增 36 项，既有完整拟合、迁移、时间稳定性与选择轨迹回归 83 项。
覆盖拟合期规则不受审计特征/误差影响、离散值切点、缺失和空支持、共享周
抽样、精确构成代数、X-only 访问、来源链与重新发布的数组损坏、完成报告、
启动失败码和现场保护。Bash 语法、命令行入口及新增文件空白检查通过。

冻结协议 SHA-256：
`02040548974590f1b3addc2347561839c33aae6cdee2c6468a089f8dd7891b29`。

```bash
python -m unittest discover -s tests -p 'test_state_interaction_*.py' -v
bash -n experiments/chronological/run_state_interaction_support_audit.sh
```
