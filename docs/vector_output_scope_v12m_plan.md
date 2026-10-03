# v12m：共享轨迹下的向量输出范围对照设计与实现

状态：2026-10-03 根据 v12l 完整服务器报告完成设计和代码实现，可执行协议
已在 v12m 结果评估前冻结；完整服务器实验尚未运行。工程验证记录见文末。
它不修改 v12l 的范围和选择规则。来源结果见
[v12l 实测记录](vector_selection_audit_v12l.md)。

## 问题与干预

v12l 的 13 个完整事故全局/早期双正 epoch 中，8 个被保护拒绝；8 个均有
匹配事故晚期损害，7 个仅有此障碍。下一步检验：在相同输入和同一向量参数
轨迹上，限制最终预测的修改范围，能否改变准入结果并获得后段候选早期收益？

设 `A(X,I)` 为同一冻结原始 A checkpoint 对同一历史与报告上下文的预测，
`V_theta(X,I)` 为原向量适配器的完整预测。两种策略为：

- unrestricted：全部输出使用 `V_theta(X,I)`。
- candidate_early_only：候选节点的 H1–H6 使用 `V_theta(X,I)`，其他单元
  使用 `A(X,I)`，以 `torch.where` 直接选择，避免相减再相加的舍入差。

mask 仅由原报告位置候选支持与固定 horizon 构成，不使用未来有效性、Y、
cohort 身份、匹配资格或 v8 路由。事故与两类常规对照采用完全相同的规则。
`incident_data=None` 会关闭原事故路径，不是 A，不能用于这里的参照。

该方案是对既有向量模型输出的范围限制，不能单凭这一组合声称新颖性，亦
不能把按构造保持不变的区域当成泛化改善。

## 一条训练轨迹、两个选模过程

固定 v12k 的 candidate_early MAE、0.001 修正能量正则、原初始化、优化器
和样本顺序。两种输出策略在该目标上具有完全相同的候选预测、损失和参数梯度。
不为它们重复训练。冻结预算：

| 项目 | 数量 |
| --- | ---: |
| 架构 | state_vector、interaction_vector，共 2 个 |
| 种子 | 2025、2026、2027，共 3 个 |
| 每架构 × 种子的共享 early-loss 轨迹 | 1 条，12 epoch |
| 总训练预算 | 6 条轨迹、72 epoch |
| 每轨迹分别选模的输出策略 | 2 种 |
| 端点记录 | 12 个；可能包含回退 A 或相同权重，不是 12 次独立拟合 |

操作上需要从头跑轨迹并保存两种策略各自的历史最佳权重，因为 v12k 未保存
每一个被拒绝 epoch 的模型。新的 run 使用独立目录与恢复身份；不载入旧
v12k 被拒绝轮次后直接查看其审计结果，也不覆盖旧研究证据。

每种策略均沿用 v12k 的 candidate_early 选择器：16 项保护、完整事故全局
与候选早期严格改善、按早期 MAE 排名、严格优于才替换、平局保留早轮、epoch 0
回退 A。保护阈值保持 `A * 1.001 + 1e-12`。先冻结全部端点，再访问后段目标。

在固定 cohort 与有效支持下，受限策略有精确的误差收支恒等式：

`global_gain = (N_candidate_early / N_all) * candidate_early_gain`。

因此受限策略的全局/早期排名在实数算术下相同，本轮不重复设置 global
选择器。实际实现仍执行原完整条件，并保存浮点计算结果；不能假定理论等价
后跳过原规则。匹配事故与常规对照的早期保护仍须真实检查。

## 分开测量输出范围与选模效应

每条轨迹产生 unrestricted 选中权重 U 和 candidate_early_only 选中权重 P。
最终在冻结端点后，评估三个预先定义的配对输出：unrestricted(U)、受限(U)、
受限(P)。其中受限(U)只是同权重的派生诊断，不额外选模：

- 同一个 U 的受限/不受限输出，检验纯输出范围效应；候选早期预测
  应逐元素完全相等，其他区域受限输出应逐元素等于 A。
- 原 unrestricted(U) 与受限(P)，测量完整策略的开发期结果。
- 受限(U) 与受限(P)，说明候选早期的差异来自选中权重不同。

对同一评估区域及有效分母，完整策略收益可以精确分解为：

`MAE[unrestricted(U)] - MAE[受限(P)]`

`= {MAE[unrestricted(U)] - MAE[受限(U)]} + {MAE[受限(U)] - MAE[受限(P)]}`。

第一项是同权重输出替换，第二项是受限输出下选中权重变化。早期第一项为零，
晚期/非候选第二项为零。直接对总效应做配对区间，不将两项区间端点相加。
无需查看 unrestricted(P) 的后段结果；P 对应的原不受限模型可能是旧规则
拒绝的轮次，主问题不需要这个额外比较。

这些路径共享训练数据、参数轨迹与部分端点，只是配对诊断。三个输出的规则、
端点和 checkpoint 引用一起冻结，不挑种子或根据后段表现更换端点。

在精确算术和相同评估支持下，原 unrestricted 准入 epoch 的早期保护已通过，
其受限输出通常仍可准入；受限策略扩大候选集合后，选择期最低早期 MAE 不差
可能只是程序结构的结果。因此选择期改善和更多非零端点不能作为泛化证据。

继承 W01–16 的 1,520 个拟合窗口，W18–25 的 856 个选择窗口及 206 组三元组，
W27–35 的 1,010 个后段窗口及 378 组三元组；间隔周 W17/W26 不变。优化器只
接收拟合事故 Y；对照目标仅用于选择保护与最终报告。每条路径保留相同样本
ID、mask、误差和有效单元计数，报告单元合并和等预测窗口的区域指标。

所有种子、回退比例、保护失败与后段早期结果并列报告，沿用共同日历周和
四周块的配对区间。受限晚期/非候选零变化是工程保证；后段早期相对 A 的
正收益及匹配事故、常规早期保护才是尚待检验的结果。改善准入率不能作为
研究成功标准，也不能靠增加epoch或放松阈值追求通过。

## 实现约束与检查

先冻结可执行协议，再做工程验证：非零适配器下候选早期预测、目标和
梯度等价；晚期与非候选预测精确保留 A；初始化等于 A；三个输出路径支持
一致；两个策略各自保存和恢复最佳状态；原骨干状态不变；所有端点冻结前
不访问审计目标。恢复身份必须包含本轮协议与代码，不能接受旧 v12k 恢复包。

训练可复用原 early-loss 更新，无需额外 A 前向。评估时需取得同批次 A 和
向量预测；缓存或第二次前向必须核对样本、输入、batch和模型身份，并记录
新增前向次数、耗时及内存。A 的计算应无梯度且不能覆写用于训练正则的中间
状态；向量路径的反向传播仍要穿过冻结骨干。

本轮将继续复用已多次查看的开发时段，且 A/scaler 保留原先广泛 train 与
旧 val 的信息依赖。因此即使受限策略改善，也只是后续独立城市/年份确认的
候选，不能宣布事故因果效应或严格的独立时间外泛化。

## 可执行协议与交付接口

协议文件为
[`vector_output_scope_v12m.json`](../experiments/chronological/vector_output_scope_v12m.json)，
SHA-256：`ac3478f320938a2e61af9529704e76b32cc68d3981d3a935e8851174a7782998`。
启动时核对本协议以及继承的 v12k、v12c、v12f、骨干和输入协议，不接受静默改动。
选择阈值、梯度更新、种子、样本划分与原协议保持一致。

主要实现：

- [`vector_output_scope.py`](../experiments/chronological/vector_output_scope.py)：
  输出投影、两个选择器历史重放、区域恒等式与共同周配对区间。
- [`train_vector_output_scope.py`](../experiments/chronological/train_vector_output_scope.py)：
  共享训练、两策略选模、三路径评估、运行身份与恢复校验。
- [`run_vector_output_scope.sh`](../experiments/chronological/run_vector_output_scope.sh)：
  前台启动、预检、小包工程检查、状态和报告；失败时保留 `.partial`。

所有轨迹完成后，先写 `selected_endpoints_frozen.json`（12 个checkpoint引用）
和 `evaluation_paths_frozen.json`（18 个预测函数引用），然后才读取后段评价目标。
三条输出路径的磁盘名字如下：

| 路径 | 输出 | 选中权重 |
| --- | --- | --- |
| `unrestricted_at_unrestricted` | 原向量完整输出 | unrestricted 选择器 |
| `protected_at_unrestricted` | 候选早期向量，其余 A | unrestricted 选择器 |
| `protected_at_protected` | 候选早期向量，其余 A | candidate_early_only 选择器 |

每条路径保存 `<fit>/<path>/<phase>_<cohort>.npz`，含样本身份、区域绝对误差和、
有效计数、候选mask和预测单元数。最终 `summary.json` 包含12个主端点的选中轮次、
全部路径指标、回退和保护失败统计、训练预算、推理次数/耗时/内存及文件指纹；
`comparisons.csv` 与各阶段 `*_weekly_s<seed>.npz` 提供配对收益和区间复核。
逐轮选择证据在各轨迹的 `history.json` 中。保护失败允许一轮同时触发多项；
两种策略共享轨迹，不能将两者的epoch计数相加后作为独立训练量。

恢复包 `last_adapter.pt` 按epoch原子提交共享适配器、Adam、随机状态、完整历史
和两个历史最佳。恢复会精确重放两种选择结果，并检查实际Adam参数组、每个参数
的矩状态、形状、有限性与累计更新步数。写入checkpoint后、发布历史JSON前中断
也可从已提交checkpoint恢复。恢复只读原目录，写入新目录；已完成轨迹不再训练，
最终评估重新执行。旧v12k包、代码/协议/输入/样本/设备身份改变均拒绝恢复。

## 服务器操作

以下用于本轮代码同步到研究分支后，在已分配GPU的计算环境中执行。沿用现有
`igstgnn`环境和真实数据，不另建环境；完整流程先跑本轮测试和2架构×2epoch的
小包检查，全部通过才开始6条正式轨迹。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid || exit 1
conda activate igstgnn || exit 1
bash experiments/chronological/run_vector_output_scope.sh run contra_v12m_output_scope_01
```

默认 `V12M_DEVICE=cuda:0`。这是前台任务，请遵循现有平台计算作业方式。
启动器会记录主机、Slurm编号（如果存在）、Python路径、环境、git提交、退出码
和 `.job/run.log`。从其他窗口查看或结束后生成报告：

```bash
bash experiments/chronological/run_vector_output_scope.sh status contra_v12m_output_scope_01
bash experiments/chronological/run_vector_output_scope.sh report contra_v12m_output_scope_01
```

中断后保留原现场，原任务已停止且代码、环境、输入不变时，恢复到新名字：

```bash
bash experiments/chronological/run_vector_output_scope.sh resume contra_v12m_output_scope_02 contra_v12m_output_scope_01
```

源为原运行的 `.partial`，不能指向新输出目录或符号链接。没有已提交epoch的
轨迹从头开始，其余从最近提交的epoch继续；不增加原72epoch预算。若源身份
不匹配，应保留失败记录并调查原因，不修改协议或checkpoint以绕过校验。

回传完整 `.job/run.log`、`summary.json`、`comparisons.csv`、两个冻结清单，
以及6条轨迹各自的 `history.json` 和 `fit_summary.json`。模型和逐窗口NPZ留在
服务器，必要时再定向核验。分析先确认预算和工程恒等式，再并列读取所有种子
的后段候选早期收益、匹配事故/常规保护和配对区间；不根据结果追加轮次或改阈值。

## 本地验证记录

2026-10-03在WSL的 `igstgnn-audit` 环境、PyTorch 2.3.1 CPU上验证：

- v12m的45项测试通过，原v12k的24项回归测试通过，共69项。
- 非零适配器的输出/梯度等价、两个策略选择不同历史最佳、初始化等于A、
  骨干不变、双清单先于后段目标、禁止计算U(P)目标误差均有检查。
- 中断和历史JSON发布失败后精确恢复、完整恢复不重训，以及错误身份、
  旧恢复包、Adam设置/步数/动量损坏的拒绝均通过。
- Bash语法、命令行入口和启动器错误退出检查通过；协议指纹保持冻结值。

从本地保留的原始A轮ZIP中仅读出 `best_model.pt`，其SHA-256与冻结协议的
`b0c712ad9c00007417ccc6ea6268f373852d04063efba2d15d3f49f5497b8e13`
一致。三套真实train数据及模型代码共19个输入文件通过原指纹核验。随后运行
`--device cpu --check`：496节点、seed 2025、2架构各2epoch，每cohort/阶段
2个窗口；训练总计4epoch，固定4个主端点、6条输出路径，报告
`ENGINEERING_CHECK_PASS`，约73.66秒。

工程检查共执行52个原生A批次和52个向量批次用于策略评估，另有9个A参照批次；
619,008个预测单元通过投影支持检查。运行结束后再次核对77个输出文件指纹，
并对9个阶段/cohort组合验证导出记录的相同支持、保护区域精确相等及区域收支/
效应分解恒等式。进程峰值RSS约1.02GiB。上述单元在重复评价间重叠，不是
619,008个独立样本。

原始测试日志、运行日志及核验记录保存在
[`输出范围对照_20261003`](../../复现结果/输出范围对照_20261003/)，
集中记录为
[`delivery_validation.json`](../../复现结果/输出范围对照_20261003/delivery_validation.json)。
真实小包的
[`summary.json`](../../复现结果/输出范围对照_20261003/v12m_cpu_check_01/summary.json)
SHA-256为 `8390a425d47a6aef6bcdbe6911928fd8f5859a5eda4c43bec9f415d3588f6e10`。

未读取val/test数组。小包检查不执行正式配对区间估计，其数值只用于工程验收，
不支持受限输出具有科学收益。完整6条轨迹/72epoch、GPU执行和后段科学结果
仍须由服务器运行检验；本轮没有据小包结果修改协议、模型或阈值。
