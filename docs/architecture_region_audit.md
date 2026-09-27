# v12b：已保存的 v12a 误差分区与不确定性分析

## 决策依据

2026-09-27 用户回传服务器 `contra_v12a_architecture_01` 完整报告，工作流退出码为 0。
这是用户提供的真实 checkpoint 报告；本地没有该 checkpoint 或该次 NPZ，不能声称本地独立复现。

| 训练集 cohort | off MAE | norm_only MAE | icsf_only MAE | full MAE |
| --- | ---: | ---: | ---: | ---: |
| incident_full | 21.9707201763 | 21.0663447976 | 21.0299020378 | 21.0293056319 |
| incident | 21.4807915441 | 20.6378301098 | 20.6053864724 | 20.6050045745 |
| primary_control | 21.3248977330 | 20.5686101951 | 20.5693576932 | 20.5699374981 |
| secondary_control | 21.6327378174 | 20.8678583567 | 20.8668581060 | 20.8675984182 |

完整事故集上，norm_only 到 ICSF 的 MAE 差约为 +0.03644，ICSF 到 full 约为 +0.000596。
这说明值得分开查看归一化、ICSF 和 TIID 的固定权重路径，但不能把这些差当成各模块独立贡献。
两类对照的 norm_only 到 full 差分别约为 −0.001327 和 +0.000260。
下一步要定位事故收益和对照伤害出现在哪些节点/时段，以及它们在按周重采样下是否稳定。

v12a 同时显示：非候选节点的 ICSF 局部输出差为零，但解码后的预测仍有变化；
ICSF 改变历史表示后会影响动态图及下游预测。TIID 的直接变化主要在较早时段，
ICSF 的变化延续到 H7–H12。预测变化幅度本身不能证明误差改善。
单事故 ICSF 中 singleton softmax 的退化也是后续结构设计线索，
但本轮先分析已有误差，不据此直接启动新网络训练。

## 固定分析范围

- 只读 v12a 的 summary、四个 train NPZ 和三个冻结的 train CSV manifest；不读原始交通数组、验证/测试数组，不加载模型。
- 核对冻结协议、checkpoint、v12a 审计代码哈希、输入文件哈希、非连续样本 ID、样本顺序和源索引。
- 核对区域/时段计数与误差的分区一致性，复算全部原始 MAE，与 v12a summary 对齐。
- 主描述端点：incident_full 的候选节点 H1–H6，比较 `norm_only − icsf_only`。
- 输出全部七组路径对比、六个基本区域、十二个 horizon 和三个派生区域。

所有 gain 为 **参照路径 MAE − 被比较路径 MAE**，正数表示改善。
主估计量按有效 cell 汇总误差后除以总有效数；另外报告每样本等权的敏感性结果。
JSON 中 `equal_event` 指每个 manifest 样本等权，不把它混称为 pooled MAE。
一个固定路径对比下，各不重叠区域误差差除以全局 cell 数，可加回全局 MAE 差。
这是区域收支核算，不是跨模块因果归因。

事故及对照共用由配对正事故时间确定的 ISO 周抽样权重，保留日历上的空周。
主分析为 2,000 次周 bootstrap（seed 12026）；时间依赖敏感性为连续四周的 circular block bootstrap
（seed 12027），每次抽取相同总周数。使用逐项 95% percentile CI，至少 95% draw 分母有效。
无目标值区域输出 null 和有效 draw 数，不把缺失结果写成零收益。
共同事故收益减两类对照平均收益使用相同 draw，保持配对；这是描述性差异，不是因果效应。

这是看过汇总 MAE 后的探索性分析，复用了训练数据，不是独立确认。
区间没有多重比较校正；按正事故周分组及四周敏感性不能消除跨周对照复用的全部依赖。
不设置“挑显著区域就训练”的自动门槛。获得结果后再决定结构实验如何预先冻结和验证。

## 服务器运行

先进入已有项目并激活 `igstgnn`；使用平台分配的计算节点。这一步只需 CPU。

```bash
git pull --ff-only origin research/chronological-tiid
bash experiments/chronological/run_architecture_region_audit.sh start contra_v12b_regions_01
```

默认读取 `experiments/chronological_runs/contra_v12a_architecture_01`。
如使用另一完整 v12a 运行，可在 start 前设置 `V12A_SOURCE`；工程检查输出会被拒绝。
启动器固定当前 Python 路径，先运行 NumPy 单元测试，测试成功才运行完整统计分析。

```bash
bash experiments/chronological/run_architecture_region_audit.sh status contra_v12b_regions_01
tail -f experiments/chronological_runs/contra_v12b_regions_01.job/run.log
```

`Ctrl+C` 仅退出日志跟踪。status 显示原运行主机，避免在不同节点误查 PID。
完成后：

```bash
bash experiments/chronological/run_architecture_region_audit.sh report contra_v12b_regions_01
```

结果目录包含 `summary.json`、`regional_comparisons.csv`、`weekly_statistics.npz`。
NPZ 保留每周充分统计量和两套 bootstrap 权重，可复算区间。
`.job` 目录保存日志、主机、PID 和退出码；分析过程中 `.partial/progress.json` 可显示阶段，
失败会尽量保留 `failure.json`。成功才把 `.partial` 原子改名为结果目录。
不覆盖旧输出或运行记录；重跑使用 `_02` 等新名字。

## 本地验证边界

测试覆盖不等 cell 数时两种估计量的区别、共享抽样的配对抵消、空周/跨年/四周块、
无效目标区域、非连续样本身份、哈希及分区/MAE 错误、完整合成输入到原子输出、失败与重跑保护。
11 项测试已分别在无 PyTorch 的 WSL Conda 环境（NumPy 1.26.4）和
与服务器一致的 Python 3.10 / NumPy 1.24.4 环境通过，
包括启动器测试失败时停止分析并保存非零退出码的检查；Bash 语法检查通过。
真实冻结 manifest 的哈希和配对关系已核对：3,604 个完整事故样本、3,106 组共同样本，
日历跨度为 2023-W01 至 W35，共 35 周。真实 v12a 区域分析结果仍需服务器运行。
