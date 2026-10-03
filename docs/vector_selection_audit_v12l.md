# v12l：向量模型的选择失败原因审计

**最新状态（2026-10-03）**：服务器审计已完整完成，20 项预检通过、退出码 0，
原选择过程精确重放。结果与解释见本文末尾“真实服务器结果”；前文保留运行前依据。

## 来自 v12k 的依据

2026-10-02 用户回传 `contra_v12k_objective_alignment_01` 完整服务器日志：
gpu11、Slurm 1135269、提交 `053ed45`，24 项预检测试通过，正式运行约
143 分钟、退出码 0。12 条轨迹共 144 个 epoch，24 个端点全部冻结后才
进行正式审计评估。以下数值来自该服务器日志，不是本地独立复现。

完整事故审计期候选 H1–H6 的收益定义为 `A MAE - endpoint MAE`，单位为原始
流量 MAE。每行包含两个架构、三个种子；两种选择器可能选中同一个模型，
不能把端点数当成独立实验次数。

| 训练损失 | 选择规则 | 回退 A | 未回退端点的早期收益范围 |
| --- | --- | --- | --- |
| global | global | 0/6 | -0.027252 至 -0.018625 |
| global | candidate_early | 4/6 | -0.005476 至 -0.004567 |
| candidate_early | global | 2/6 | -0.027297 至 -0.017819 |
| candidate_early | candidate_early | 4/6 | -0.019147 至 -0.017819 |

全局训练不变时，早期选择相对全局选择的六组配对早期收益区间均高于零，
但四组来自回退 A，另两组仍是相对 A 的负点收益且区间跨零。因此这是减少
损害，尚无稳定的正早期收益。对齐训练损失也未消除拟合至后续时段的反转。

同时，早期损失、种子 2025、第 3 轮在选择期的完整事故全局和早期指标均
有正收益：state_vector 约 +0.000698 / +0.003955，interaction_vector
约 +0.000725 / +0.003827。原规则仍未准入这些轮次，表明至少有其他保护
条件失败。具体失败位置不能从简略日志确定，需要已有 `history.json`。

本轮只回答：哪些条件阻止模型入选，是否存在多个同时失败的条件，以及原
选择过程是否能由保存的指标精确重放。它不直接决定新模块结构。

## 固定分析范围

读取完整正式 v12k 的 27 个 JSON 文件：`summary.json`、
`run_identity.json`、`selected_endpoints_frozen.json`，以及 12 条轨迹各自的
`history.json`、`fit_summary.json`。不读取模型 checkpoint、NPZ/NPY 或
原始数据，不执行训练、推理、梯度计算或新的模型选择。

完整源 summary 的字节包含已经存在的审计期汇总。不能声称未读过任何审计
信息；实现不使用其中的 audit 指标分析拒绝原因，也不评估任何被拒绝轮次的
审计表现。所有统计仅使用既有选择期指标。

逐项保持 v12k 原规则：四 cohort × 四区域共 16 项保护，各项满足
`current <= A * (1 + 0.001) + 1e-12`；空指标拒绝。global 选择器还要求
完整事故全局严格改善 A；candidate_early 选择器另要求早期严格改善 A，
并按早期 MAE 排名。严格优于当前最佳才替换，平局保留较早轮，epoch 0 为 A。

审计对每轮独立计算未经保护条件筛选的全局/早期收益，再重放原来的条件。
这是必要区别：v12k 保存的 `full_*_strictly_better_than_A` 标志已经与
`protected` 相与，不能直接用它筛选“有原始收益但被保护条件拒绝”的轮次。

每个选择决定同时保存：

- 所有重叠的准入障碍，包括各项保护失败及全局/早期未严格改善；
- 一个互斥的程序性原因，依次为无效保护指标、违反保护、全局未改善、早期
  未改善、准入但未击败当前最佳、替换当前最佳；
- 当前目标 MAE、之前最佳目标 MAE、前后最佳 epoch。

互斥原因有固定优先次序，仅用于完整计数，不将第一个原因当作唯一根因。

对每项保护输出 A/current MAE、原始收益/损害、相对损害百分比、名义阈值、
含原数值容差的实际阈值、两种有符号余量及实际阈值超出量。A=0 时相对百分比
为空；无效指标的余量为空，不伪造为零损害。

重点子集在本轮协议中固定为：完整事故全局与候选早期都严格优于 A、但保护
未通过的全部轮次。逐一列出所有失败条件及超限量，不挑其中最好的轮次。
同时报告全体轨迹结果，避免只看此子集造成选择偏差。

保护失败按每条轨迹每个 epoch 计一次，不因两个选择器重复计数。完整报告
失败组合、两两共现、仅一项保护失败的次数，以及各 cohort/区域的失败并集。
一个轮次可以属于多个组；全局 MAE 与区域 MAE 也重叠，不能相加为损害归因。
不同 epoch、种子以及重叠 cohort 不构成独立重复；本轮不计算置信区间或
显著性检验，不修改阈值，不做“放宽约束后重新选模”。

## 来源核验与输出

要求源状态为正式 `VECTOR_OBJECTIVE_ALIGNMENT_COMPARISON_COMPLETE`，工程
检查或 `.partial` 不可用。核对冻结协议、关键生产代码哈希、训练和样本预算、
源运行身份、全部历史长度、原始决定以及最终已选 epoch/指标/状态哈希引用。
两种选择器必须逐轮与原记录完全一致，包括浮点容差、平局和 A 回退。

消费的 JSON 均按源 summary 的输出清单核验 SHA-256，并在发布前复核。已选
checkpoint 的哈希引用在三个记录之间交叉核对，但不打开 checkpoint 文件。
这些是保存产物的一致性检查，不是对源 summary 的外部认证，也不替代模型
预测复算。保留源文件；已有输出、partial、job 或符号链接均不覆盖。

输出采用 `.partial`，成功后改为正式目录；失败记录 `failure.json`：

| 文件 | 内容 |
| --- | --- |
| summary.json | 来源、边界、全部轨迹计数、重放结果与拒绝详情 |
| epoch_metrics.csv | 144 × 16 = 2304 项保护条件及余量 |
| selector_decisions.csv | 144 × 2 = 288 个选择决定及全部障碍 |
| selected_endpoints.csv | 24 个原有端点的 epoch、选择期收益与回退状态 |
| joint_positive_rejections.json | 所有全局与早期均改善但被保护拒绝的轮次 |
| cofailures.json | 16 项保护两两共现的完整 120 项计数 |

审计结果不能证明图传播的因果外溢，也不能排除高维分布变化或证明不可学习。
如果后续发现主要失败条件位于晚期/非候选输出，可以据此提出输出修改范围
受限的结构对照；如果主要仍是事故早期，则应优先考虑跨时间泛化问题。这两
条都需要另立实验，程序不会自动升级训练或访问 test。

## 服务器运行

代码提交推送后，在仓库目录执行。使用现有 `igstgnn` 环境即可；运行本身
仅需 Python 3.10+ 标准库，无需 GPU 或重新训练 v12k。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid || exit 1
conda activate igstgnn || exit 1
bash experiments/chronological/run_vector_selection_audit.sh run contra_v12l_selection_audit_01
```

这是前台入口，先检查源 JSON 完整性并运行本轮标准库测试，再执行审计，
自动记录主机、PID、Slurm ID、日志和退出码。服务器应按平台规则使用允许
的 CPU 作业或现有计算会话。若平台已经选定并激活 `igstgnn`，可省略环境激活
这一行。运行期间如需查看日志，在另一个终端执行下面的可选命令；`Ctrl+C`
只退出该终端的日志跟踪，不会停止原前台作业：

```bash
tail -f experiments/chronological_runs/contra_v12l_selection_audit_01.job/run.log
```

前台作业结束后检查状态并打印报告：

```bash
bash experiments/chronological/run_vector_selection_audit.sh status contra_v12l_selection_audit_01
bash experiments/chronological/run_vector_selection_audit.sh report contra_v12l_selection_audit_01
```

成功交付真实审计结果需要同时满足：

- `Workflow exit code: 0`。
- `status: VECTOR_SELECTION_FAILURE_AUDIT_COMPLETE`。
- `Exact original-selector replay: True`；`summary.json` 中
  `all_final_selection_references_match` 也为 `true`。
- 报告的 `fits/epochs/selector_decisions/endpoints/fallbacks` 前四项为
  `12 144 288 24`；最后一项是实际回退端点数，不在此预填。

完成后回传 `report` 的完整文本，保留原 v12k 目录以及本轮 `.job` 日志、
`summary.json`、三张 CSV 和两份诊断 JSON。先查看全体轨迹与全部失败组合，
再核对早期损失、seed 2025、第 3 轮的两种架构是否出现及具体失败条件。
不得只回传有利轮次，也不能根据失败位置直接宣布某个新结构有效。
若运行失败，回传 `status` 和错误日志，保留已有输出，并用新的运行名重跑。

默认源为 `contra_v12k_objective_alignment_01`。需要换源或中断重跑时，用
新的输出名；第三个参数可指定另一个完整 v12k 源运行名：

```bash
bash experiments/chronological/run_vector_selection_audit.sh run contra_v12l_selection_audit_02 contra_v12k_objective_alignment_01
```

本轮协议 SHA-256：
`52a1a87fc79f71077ba35af6ddc670ba25602c74437ba722a944e302be5e4df7`。
初次交付时尚无真实服务器 v12l 审计结果；本次收到的完成记录见本文末尾。

## 本地验证记录

2026-10-03 在 WSL `igstgnn-audit` 环境完成 21 项新增测试，以及 24 项原
v12k 回归测试，共 45 项全部通过。Bash 语法和命令行入口检查通过。

新增测试包括：精确保护边界与数值容差、原始收益与保护准入分离、并列轮次、
A 回退、重叠失败计数、完整产物流程、源文件只读、哈希及决定被修改时拒绝
发布、缺失 epoch/错误预算、源文件运行中变化、输出保护和启动退出码。
改变源 summary 的既有 audit 汇总后，选择审计的全部统计与结果文件保持相同。

另用原 v12k 选择函数生成 800 轮、1,600 个选择决定，覆盖所有保护条件、
阈值附近、空指标与平局，审计重放及最终 epoch 全部一致。该对照测试单独
依赖原环境的 PyTorch；服务器预检只运行其余 20 项标准库测试。完整合成
产物审计也已用 `python -S` 跑通，确认运行不依赖第三方包。

```bash
python -m unittest discover -s tests -p 'test_vector_selection*.py' -v
python -m unittest discover -s tests -p 'test_vector_objective_alignment*.py' -v
bash -n experiments/chronological/run_vector_selection_audit.sh
```

本地验证使用合成历史与既有实现的一致性检查。真实 v12k JSON 产物仍在服务器，
因此具体失败条件和次数要以服务器 v12l 报告为准。

## 2026-10-03 交付复核

用户确认服务器尚未运行 v12l，本轮先完成交付准备。独立审查分别核对了
原 v12k 选择器与本轮重放、源产物结构与指纹，以及启动器的前台执行、失败
退出码和结果报告，没有发现阻断交付的问题。

在已有 `igstgnn-audit` 环境重新执行本轮 21 项测试和 v12k 24 项回归测试，
45 项全部通过。Bash 语法、标准库命令行入口、三个冻结协议及五个生产脚本
指纹核验通过。补充了环境激活失败即停止、日志跟踪与结果查看分开执行、
成功验收和报告回传说明；模型选择规则与冻结协议保持一致。

本次交付范围为以下八个文件，通过 `research/chronological-tiid` 研究分支同步
到服务器：

- `docs/vector_selection_audit_v12l.md`
- `experiments/chronological/vector_selection_audit_v12l.json`
- `experiments/chronological/vector_selection_audit.py`
- `experiments/chronological/audit_vector_selection.py`
- `experiments/chronological/run_vector_selection_audit.sh`
- `tests/test_vector_selection_audit.py`
- `tests/test_vector_selection_audit_launcher.py`
- `tests/test_vector_selection_reference.py`

上述交付复核仅覆盖本地工程验证，真实来源的端到端结果由下节服务器回传补充。

## 2026-10-03 真实服务器结果

用户回传 `contra_v12l_selection_audit_01` 的完整运行日志：gpu10、Slurm
1137048、提交 `9c6db6e08f04b473ceae6903204b499660cb5336`，开始于
2026-10-03 11:33:20 +08:00，结束于 11:33:23 +08:00。20 项标准库预检
全部通过，审计主程序约 0.605 秒，工作流退出码为 0。

服务器报告 `VECTOR_SELECTION_FAILURE_AUDIT_COMPLETE`，验证 27 个源 JSON，
精确重放 12 条轨迹、144 个 epoch、288 个选择决定和 24 个端点，其中 10 个
端点回退 A。来源 v12k 提交为 `053ed45be5cb50a44386c4823f90a66b98312ec9`。
本地逐条复算日志中的计数与交叉关系一致；这是服务器回传证据，不是本地读取
27 个真实 JSON 或重新推理所得的独立复现。

原始日志按字节保存在
[v12l_服务器日志.txt](../../复现结果/选择失败审计_20261003/v12l_服务器日志.txt)，
SHA-256 为 `b09484317bcd64b2a6b875a767aeb5dc7acd2c2af002166ebee42269c109ac15`。
[本地日志核对记录](../../复现结果/选择失败审计_20261003/v12l_log_review.json)
只保存日志提取与复算，不冒充服务器 `summary.json`。

| 训练损失 | 轨迹 / epoch | 完整事故全局改善 | 完整事故早期改善 | 两者均改善 | 全部保护通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 6 / 72 | 72 | 3 | 3 | 61 |
| candidate_early | 6 / 72 | 18 | 16 | 10 | 4 |
| 合计 | 12 / 144 | 90 | 19 | 13 | 65 |

13 个双正收益 epoch 中，8 个被保护拒绝、5 个通过。8 个被拒者全部来自
candidate_early 训练，且全部违反 **匹配事故子集 `incident` 的候选 H7–H12**
保护。7 个仅此一项失败；`interaction_vector`、seed 2027、第 5 轮还同时
违反匹配事故候选 H1–H6 保护。

完整 144 轮中，共 79 轮保护拒绝，均涉及匹配事故子集。该子集早期失败 66 次、
晚期失败 68 次、两者同时失败 55 次，满足 `66 + 68 - 55 = 79`。完整事故
早期还有 6 次失败，均已包含在上述交集中。两个常规对照、所有非候选区域及
所有全局 MAE 保护的失败次数均为零。保护通过允许至多 0.1% 退化，不能将
零失败解释为各区域都得到改善。

这里 `incident_full` 是选择期全部 856 个事故预测窗口；`incident` 是其中
具有完整共同匹配三元组的 206 个窗口，二者不能混称。全体事故汇总改善不能
保证这个匹配子集也改善，窗口亦不等同于独立事故。

此前关注的 seed 2025、第 3 轮得到如下解释，单位均为原始流量 MAE：

| 架构 | 完整事故全局收益 | 完整事故早期收益 | 匹配事故晚期损害 | 超出原保护阈值 |
| --- | ---: | ---: | ---: | ---: |
| state_vector | +0.000697813 | +0.003954840 | 0.029386296 | 0.010256089 |
| interaction_vector | +0.000725089 | +0.003826723 | 0.031999088 | 0.012868881 |

因此这两轮未入选是原规则正常拒绝了匹配事故晚期的损害。当前日志不支持把
阻碍归到常规对照、非候选输出，或把原规则实现认定为有误。计数只描述这一批
已保存轨迹；不同 epoch、架构、种子与重叠 cohort 不能当成 144 次独立实验。

## 后续研究判断

下一项有限假设是：保留 A 在晚期与非候选节点的原始预测，只允许候选 H1–H6
使用向量适配器输出，检验能否在相同保护条件下保留早期收益。该干预改变
预测函数的作用范围，不证明晚期损害来自图传播，也不能预称解决跨时段泛化。

固定 candidate_early 损失与原正则时，最终输出范围约束不会改变候选早期的
预测值或训练梯度。因此每架构、种子只需一条共享训练轨迹，分别观察两种输出
策略及其选模结果；不能为两种范围重复训练后称为独立的学习机制。

匹配事故早期和两个常规对照早期的保护仍须保留。v12k 已显示符合选择规则的
端点在后段早期仍可能退步；本轮没有评估被拒绝 epoch 的后段表现，不能声称
这些轮次会泛化。下一轮的可审阅设计见
[v12m 输出范围对照设计与实现](vector_output_scope_v12m_plan.md)。随后完整服务器
实验已完成：[v12m结果记录](vector_output_scope_v12m_results.md)显示输出范围保护
成立，但6个受限端点的后段完整事故早期收益均为负；不能将准入增加视为预测改善。
