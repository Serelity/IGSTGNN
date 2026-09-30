# v12f：交通状态与事故表示的向量交互对照

## 为什么开展这一步

v12e 已完成两种损失、三个种子、各 12 epoch 的正式比较。72 个 epoch 都通过了
选择期保护约束；global 的三个种子最终选择 epoch 0/0/11，regional 选择 0/2/0。
多数回退到 A 的原因是选择期完整事故全局 MAE 未严格改善，并非保护约束拒绝。

regional 唯一非恒等的 seed 2026，在完整事故审计期的原始单位 MAE 收益
（A 减新模型，正数为改善）如下：

| 区域 | MAE 收益 |
| --- | ---: |
| 全局 | -0.0003230645 |
| 候选 H1–H6 | +0.002744349 |
| 候选 H7–H12 | -0.0016491 |
| 非候选 | -0.000398736 |

其全局收益的按周置信区间约为 [-0.000652578, -0.000030201]。局部早期改善没有
转化为完整事故全局改善，且绝对变化很小。固定 64 个拟合样本上的共享参数梯度，
初始化时总体同向；后续状态存在部分冲突，但不支持把失败归因于普遍、严重的
区域梯度冲突。该子集梯度也不等于整个拟合期梯度或 Adam 实际更新。

结合 v12a 的单报告注意力退化、v12c 的强度学习收益不稳定，以及 v12d/v12e 的
区域响应与权重结果，v12f 检验一个新的、有限的假设：保留原事故路径时，允许
交通状态决定附加向量的方向和内容，是否比仅缩放原事故值向量更有效；额外加入
冻结事故上下文的交互，是否还有增量收益。这是待检验的表示假设，不是已经证明
原架构无法学习，也不以失败结果为理由放宽原选择规则。

## 冻结协议与三个版本

协议文件：`experiments/chronological/incident_state_interaction_v12f.json`。
SHA-256：`d782ed11d2044583d7dda2d0025a6fe24c06ac512ab9465258ccfb894ee0379c`。

记 H 为原模型对本节点历史 flow 的冻结线性输入投影，尚未经过时间或图编码；
V 为原 ICSF 对 `report_location_v1` 事故嵌入的冻结值投影。这里的 V 只来自报告
年龄和预测时钟，不包含事故严重度、类型、文本等新信息。三个版本共享输入特征：

`F = [H_last, mean_time(H), H_last-H_first, 三个距离, report_age_minutes/5]`。

当前 H 宽度为 32，F 宽度为 100。mask 沿用非零距离支持的候选节点定义。

| 版本 | 附加模块与注入 | 可训练参数 |
| --- | --- | ---: |
| strength | 原 v12c 节点门控 F→16 Tanh→1，g=2 sigmoid(logit)；注入 g×mask×V | 1,633 |
| state_vector | u=tanh(Linear(F,32))，r=tanh(Linear(u,32))；原注入上增加 mask×RMS(V)×r | 4,288 |
| interaction_vector | 与 state_vector 相同参数张量；末层输入改为 u×(1+tanh(V))，再产生 r | 4,288 |

原骨干冻结并保持 eval，LayerNorm、动态图、解码器、TIID 上下文均保留。
strength 使用原始训练函数和运算顺序；两个向量版本最后线性层权重和偏置均为零，
初始化附加修正精确为零，因而从 A 开始。其公式为：

`LayerNorm((H_last + mask×V) + mask×RMS(V)×r)`。

两向量版本每个种子采用相同初始化与训练样本排列，二者容量严格匹配；
strength 与向量模块参数量不同，不能把它们的差异全部归因于向量方向能力。
state_vector 也使用报告年龄、距离及 RMS(V) 尺度，因此不是“没有事故信息”的对照。
interaction_vector 与 state_vector 的差异检验额外冻结上下文条件化，不能据此断言
乘性交互优于所有拼接、加法或注意力融合方法。

本轮不接入高影响路由，不增加 speed、occupancy、天气或其他新输入。先单独检验
附加事故表示，避免同时改变识别、专家与数据源而无法解释差异。

## 修正边界、损失和模型选择

向量 r 的每个分量经 tanh 限制，故每个节点的附加修正满足
`||delta||₂ ≤ ||V||₂`。V=0 时 delta 精确为零，不用 epsilon 将其放大。
strength 的 g 在 (0,2)，其相对原注入的修正范数同样不超过原注入范数。
向量变化可以拥有垂直于 V 的分量；strength 只能沿 V 改变注入。

优化目标继续使用原全有效单元标准化 flow MAE。附加正则系数仍为 0.001：
strength 对候选节点取 `(g-1)²` 均值；向量版本对候选节点和通道取 `r²` 均值。
V 非零时，它们均表示相对原注入范数的修正能量。V=0 时向量 r 仍受正则约束，
但其实际 delta 为零。正则系数相同不意味着三个模型的参数梯度大小相同。

注入阶段非候选节点的附加修正为零；原模型的后续图传播、时间计算及解码仍可能
使非候选预测和 H7–H12 发生变化。这里没有声称这些预测精确等于 A，仍需逐区域评估。

沿用 v12c 的三种子 2025/2026/2027、每臂每种子 12 epoch、训练 batch 8、评估
batch 16、Adam 学习率 0.001、eps 1e-8、weight decay 0、梯度裁剪上限 5。
九组拟合全部从头训练；不挑最好种子作为结论。

时间和完整 support 资格不变：拟合 W01–16，共 1,520 个完整事故样本；选择
W18–25，共 856 个完整事故、206 组共同三元组；审计 W27–35，共 1,010 个完整事故、
378 组三元组。W17/W26 为原间隔周。只有拟合事故 Y 进入优化器；对照 Y 可用于
选择保护约束，但不用于参数拟合。

模型选择仍要求完整事故选择期全局 MAE 严格改善，同时四 cohort 的全局、候选
H1–H6、候选 H7–H12、非候选区域点 MAE 均不超过 A×1.001。epoch 0 始终可选。
不得依据局部收益、表示诊断、审计表现或种子表现改变这条规则。

## 表示诊断与报告

在合格拟合样本有序索引上，用 linspace 等位置选取固定 64 条（不足则全取），
不依据标签选样本。三个版本在 initial、last、selected 状态使用同一子集，
保存 `representation_diagnostics` 及每样本、节点数组和汇总统计，包括：

- 区域 MAE，候选修正 RMS，修正范数相对原注入范数的比例。
- 垂直于 V 的修正范数比例，LayerNorm 后的变化 RMS。
- 冻结上下文调制项的均值、标准差以及原注入为零的数量。

V=0 时相对范数未定义，保存 NaN 和明确有效计数，不伪造为零。诊断不更新优化器，
不计算新梯度，并隔离随机数状态。它是固定子集上的描述，不能代替整个拟合期
表现，也不构成图传播的因果归因。末层零初始化使初始隐藏层梯度可能为零，
本身不是训练失效。

正式报告覆盖全部种子、四 cohort 和所有区域，比较各版本相对 A、两个向量版本
各自相对 strength，以及 interaction_vector 相对 state_vector。置信区间沿用
配对周 bootstrap 和四周块 bootstrap；原点估计保护规则不等于风险保证。

原 A 训练使用整个原 train，并曾由原 val 选择；scaler 也使用整个 train X。
本次虽然不读取旧 val/test 数组，不用审计 Y 选模型，后段开发时段已经多次被查看。
因此本实验是重复开发比较，不能声称独立确认，也不授权查看 test。
本文件冻结设计与运行方式，不预先报告 v12f 的科学结果。

2026-09-30 本地 WSL conda 验证：19 项 v12f 模型/流程/启动器测试、33 项原强度
门控/诊断/恢复回归以及 7 项 v12e 回归，共 59 项通过；Bash 语法、CLI 与 diff
空白检查通过。覆盖真实微型架构上的合成数据工程行为，尚未在本地读取真实 A
checkpoint 或运行完整 GPU 数据；服务器入口会先做真实 checkpoint 工程检查。

## 服务器运行与恢复

在已分配 GPU、具有足够剩余时间的服务器环境执行前台入口；九组正式训练会比
v12e 的六组更久，具体时长需由实际日志判断。先更新代码，再开始运行：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_incident_state_interaction.sh run contra_v12f_state_interaction_01
```

入口先执行测试，再用真实 A checkpoint 对三个版本各做一个种子、两个 epoch 的
工程检查，通过后执行九组正式拟合。默认设备 cuda:0；需要 CPU 工程环境时可设置
`V12F_DEVICE=cpu`。这只是设备选择，不改变协议或预算。

运行中另开终端查看日志和状态，完整结束后查看报告：

```bash
tail -f experiments/chronological_runs/contra_v12f_state_interaction_01.job/run.log
bash experiments/chronological/run_incident_state_interaction.sh status contra_v12f_state_interaction_01
bash experiments/chronological/run_incident_state_interaction.sh report contra_v12f_state_interaction_01
```

运行目录保留 host、PID、Slurm job ID 和退出码；结果完整写入前使用 `.partial`。
没有 summary 时 report 提示先查看 status，不把中断过程当作完成。
异常终止后先确认原进程已结束，再在有效 GPU 分配内恢复到新运行名：

```bash
bash experiments/chronological/run_incident_state_interaction.sh resume contra_v12f_state_interaction_02 contra_v12f_state_interaction_01
```

恢复来源 `.partial` 只读，工程检查不会读取恢复状态。正式恢复核验代码、协议、
输入、样本和训练版本身份，从完整 epoch 的模型及优化器状态继续；历史最佳选择
一并保留，已完成拟合不重新优化。不同版本状态不可互用，旧文件和已有输出不覆盖。
代码实质变化后应新开运行，不能绕过身份校验。训练期间保持代码不变。
