# v12g：状态交互模块的选择轨迹与审计样本构成诊断

## 基于 v12f 的问题

服务器 v12f 已完成三种模块、三个种子、每组 12 epoch，共 9 组训练和 108 个正式
epoch；97 个 epoch 通过原选择期保护约束。六个向量模型最终都选择了训练后的
状态，因此本次诊断有实际变化的保存模型和误差可以分析。

完整事故审计集上，六个向量模型的全局 MAE 收益很小，按周及四周块置信区间均
跨零；候选节点 H1–H6 的收益区间均位于零以下，非候选区域收益区间均位于零
以上。这里收益定义为 `MAE(A) - MAE(模块)`，正数表示改善。这表明全局微小
正收益伴随着早期事故候选区域退步，选择期保护成立并未保证后段各区域同样安全。

同时，固定 64 个拟合期预测窗口的 selected 探针上，六个向量模型的候选 H1–H6
MAE 改善约为 0.135–0.172 原始单位。修正范数约为原事故向量的 10%–11%，并含有
垂直于原事故向量的分量。模块确实学到了新的修正方向，但这个固定子集的改善
不能代替完整拟合期结果，也不能直接证明后段退步就是过拟合。

完整事故集与匹配事故子集的表现不同，匹配事故的早期候选退步较小，两个对照组
的早期候选点估计反而改善。v12g 因此回答两个具体问题：选择期是否已出现了
“全局改善、候选早期退步”的轨迹；后段完整事故损害主要来自共同三元组内窗口，
还是其余完整事故窗口。根据结果再决定后续模块实验的假设。

## 冻结范围与输入核验

协议文件：`experiments/chronological/state_interaction_transfer_v12g.json`。
SHA-256：`cef7b6b53d78bd11b5af547585a9758015af350e09c93feccb2cc84a582af683`。

本轮只需要 NumPy 和 Python 标准库，在 CPU 上读取完整 v12f 保存结果及来源数据
包的 `train_manifest.csv`。不加载模型、checkpoint 权重或原始流量数组；不重新
训练、不计算梯度、不读取原 val/test 数组。v12f 保存的选择期和审计期结果属于
原 train 内已经使用过的开发时段，本轮继续复用这些结果。

输入必须是完整正式 v12f 运行，而非工程检查、部分完成或不同预算的结果。核验
原 v12f 协议、继承的 v12c 协议、代码及输入身份、每个实际读取文件的哈希，以及
各阶段样本 ID、source index、候选 support 和有效单元计数。manifest 只有在
哈希与原始来源记录相同后才读取。原来源目录只读，审计前后保留其内容和哈希。

来源代码身份固定为 `a1b4bdf` 历史提交中 v12f 记录的完整 18 文件集合，包含
backbone、adapter、数据解释、训练、恢复和运行入口，文件集合及哈希均需一致。
来源的历史哈希不与当前诊断代码混用；本轮共享统计 helper 仅修复空子组索引的
整数 dtype，当前诊断代码哈希单独记录。不要用更新后的代码恢复旧 v12f partial，
其恢复入口会按原代码身份拒绝；本轮只读取已经完整完成的 v12f。

正式输入包含以下固定数量，三种模块和全部种子均需完整存在：

| 阶段 | 完整事故 | 匹配事故／每个对照组 |
| --- | ---: | ---: |
| 拟合 W01–16 | 1,520 | 不用于专家参数拟合 |
| 选择 W18–25 | 856 | 206 |
| 审计 W27–35 | 1,010 | 378 |

原模型 A、训练预算、选择规则和保存的 selected 状态均沿用 v12f。v12g 的协议
决定为 `DIAGNOSTIC_REVIEW_REQUIRED_NO_NEW_MODEL_SELECTION_OR_TEST_AUTHORIZATION`，
诊断结果本身不会产生新模型选择或授权查看 test。

## 三类诊断

第一类诊断重放 108 个 epoch 的选择过程。按原四个 cohort、四个区域的 16 条
点估计约束逐项核对，要求各区域 MAE 不超过 A×1.001，并保持完整事故全局
MAE 严格改善及 epoch 0 回退。记录被拒绝的 cohort/区域、候选 H1–H6 退步，
以及“全局改善同时早期候选退步”的 epoch。区域对全局收益的贡献按有效单元数
加权，不能把三个区域的平均收益直接相加。

原选择结果只保存每 epoch 的聚合指标，没有 selected 模型的选择期逐窗口误差。
因此本轮只能诊断选择期聚合轨迹，不能构造选择期按周置信区间、逐样本分组或
逐窗口完整拟合到选择的迁移曲线。

第二类诊断分解已保存的审计期逐窗口误差。使用以下六个组：

| 组名 | 定义 |
| --- | --- |
| `incident_full` | 原完整事故审计窗口 |
| `incident_full_common` | 完整事故数组中，ID 属于本审计阶段合格共同三元组的窗口 |
| `incident_full_complement` | 完整事故数组中的其余窗口 |
| `incident` | 原保存的匹配事故评估，单独保留用于重放核对和对照描述 |
| `primary_control` | 原保存的主对照窗口 |
| `secondary_control` | 原保存的第二对照窗口 |

common 和 complement 从完整事故数组按 ID 直接切分，保留其原始误差及 support；
不通过两个 cohort 的 MAE 相减推算。complement 的含义是“本审计阶段合格共同
三元组之外”，不等于全局没有匹配对照的独立事故。

原 plan 的三个匹配 cohort 均使用共同匹配数据集位置，但保存数组中的源行号
分别指向正样本、主对照、第二对照。完整事故与第二对照源行号可直接核对 plan；
匹配事故通过正样本 ID 对应原 manifest 行号核验。主对照不新增读取其 manifest，
依靠来源产物哈希及 A/各模块保存结果的源行号精确对齐，不声称重新验证了其
外部 manifest 映射。

common 与原匹配事故保存数组的几何 support、计数必须精确一致。由于原批次
重放可出现数值差异，误差比较使用冻结的 `rtol=1e-5, atol=0.001`；报告最大差异
及聚合 MAE/收益差异，使当前极小收益对数值重放的敏感性可见。

各组按正样本 t0 的日历周汇总；两个对照组也使用其配对事故的周，而非对照
候选时钟的周。全部组和种子共用完整事故审计期的九周日历网格，包括子组没有
样本的周。common 与 complement 的误差和计数必须精确重构完整事故统计，
并报告各组/区域误差收益总和除以完整事故全局有效单元总数的贡献。按周也作
同样核算，全部贡献相加应还原对应完整事故全局收益。按周表同时保存单元合并
与等预测窗口加权的 MAE/收益、区域可评估窗口数，零有效计数窗口不参与后者。

第三类诊断读取已有 initial、last、selected 的固定拟合子集探针，核对其
样本身份和保存指标，把子集收益与选择、审计结果并列描述。这 64 个窗口不是
完整拟合期评估，本轮也没有计算完整拟合期的泛化差距。

## 统计与解释边界

分别保留按有效单元合并的统计和等预测窗口加权的统计。一个预测窗口可能与其他
窗口来自同一事故，等预测窗口加权不等于等独立事故加权，不能把窗口数写成唯一
事故数。两个对照组的目标有效计数可不同，组间平均收益也不能忽略这种差异。

置信区间继承原 v12c 的配对周 bootstrap：2,000 次抽样、seed 12028、95% 点对点
区间，并报告四周循环块 bootstrap 敏感性。每种统计使用相同完整事故周网格；
有效抽样比例至少达到 95%，空或过稀疏的区域明确报告有效抽样不足。按周区间
以已经保存的模型为条件，不包含训练种子和多重比较的不确定性。只有九个审计
日历周，加上窗口时间重叠，限制了区间的解释范围。

共同组与补集的差异属于观察性样本构成分析。事故与配对对照收益的差值可作为
描述性比较，不能解释为事故模块的因果效应。早期候选退步、非候选收益或固定
拟合探针改善，都不能单独识别架构缺陷、信息不足或过拟合为原因。

本轮产出用于定位后续假设，不能据此挑审计表现最好的种子、epoch 或子群，也
不能将再次查看过的开发时段称为独立确认。

## 输出与服务器运行

完整输出包含 `summary.json` 和四张表：

- `selection_epochs.csv`：各 epoch 的选择约束、区域指标和贡献。
- `audit_regions.csv`：全部组、区域和种子的审计统计及比较。
- `audit_weekly.csv`：共用周网格上的两种加权 MAE/收益、有效单元与窗口数和全局贡献。
- `group_membership.csv`：完整事故共同组/补集身份及日历周。

每个种子保存 `audit_weekly_sSEED.npz`，用于重放按周和块抽样统计。具体字段由
冻结协议及脚本中的结果结构定义，空支持会记录状态而非伪造收益。

在服务器上激活已有 NumPy 环境即可运行；入口是前台 CPU 审计，GPU 分配不是
必要条件。使用具有足够资源和剩余时间的有效作业环境，并保留终端运行：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_state_interaction_transfer_audit.sh run contra_v12g_transfer_audit_01
```

默认来源为完整的 `contra_v12f_state_interaction_01`。指定其他来源时，第三个
位置传入完整 v12f 运行名：

```bash
bash experiments/chronological/run_state_interaction_transfer_audit.sh run contra_v12g_transfer_audit_02 contra_v12f_state_interaction_01
```

另开终端查看日志和状态，结束后读取报告：

```bash
tail -f experiments/chronological_runs/contra_v12g_transfer_audit_01.job/run.log
bash experiments/chronological/run_state_interaction_transfer_audit.sh status contra_v12g_transfer_audit_01
bash experiments/chronological/run_state_interaction_transfer_audit.sh report contra_v12g_transfer_audit_01
```

入口先检查完整 v12f summary，记录 host、PID、Slurm job ID 和退出码，执行
v12g 测试后开始审计。NumPy 为唯一数值计算依赖，没有真实 checkpoint 工程
训练或 GPU 检查。输出完整写入前使用 `.partial`；已有 final、partial、job
任意一个都会拒绝覆盖。缺少 summary 时 report 提示先检查 status。

如果审计中断，先核对退出码和进程，再使用新运行名从保存输入重新审计；本轮
没有训练恢复入口。旧输出及 v12f 来源均保留，不需要删除已完成的 v12f 文件。
