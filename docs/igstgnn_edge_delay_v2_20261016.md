# 原 IGSTGNN：第一层历史边界回退模块 v2

开发日期：2026-10-16。入口：`experiments/chronological/train_original_edge_delay_v2.py`。v1源码和运行目录保持原样，新版使用独立身份及目录。

## 设计依据与目标

上一轮服务器结果中，DYNAMIC最后一轮比同检查点NO_ALIGN差约0.06061 MAE；短层的早期历史存在明显路径系数缺失。FIXED的时延几乎整体从0.23841变到0.27452，而边间差异很小。对原Adam的零任务梯度标量计算发现，学习参数从-2被L2向0推动82步，就能产生约0.274519的时延。这支持审计正则项的必要性，不能代替历史实际任务梯度测量。

新版检验的结构假设是：第一层的道路消息应按照边关系读取不同历史位置，并且缺少历史时不应顺带衰减原消息。只在第一层接入用来隔离这一问题，并不代表已证明第一层最优。没有学习率、时延上限或评分宽度搜索。

## 模块与数据流

```text
第一层门控历史 H
    │
    ├── 原三步非线性窗口编码 ── 完整窗口摘要 P(t)
    │                              │
    └── 原图边 + 最后状态及变化 ── 边延迟 τ
                                   │
                    可用历史按 τ 读取；缺失路径回退到 P(t)
                                   │
                  原道路消息 + 零初值有界残差
                                   │
               与原 self / dynamic / static 消息拼接
                                   │
                       原 GCN 投影及一次 dropout
                                   │
                   原 DIF forecast / backcast 与 INH
                                   │
                    后四层和最终读出保持原流程
```

所有未来gap=3递推调用继续使用原卷积。窗口编码产生的三个混合子槽不被当作三个时间点；P(t)表示结束于t的完整三步观察窗口，12步历史只有10个这样的摘要。

## 明确的计算合同

对每条原道路边，`τ = 2 sigmoid(-2 + u_ij + s(H_i,H_j))`。-2为固定buffer，边偏差u初始为0；动态评分器输出层初始为0。FIXED只学习边偏差，DYNAMIC再读取节点历史状态；两个方向沿用原邻接及转置的计算方向，不认证物理上下游。

延迟在一次输入窗口内固定，可跨样本改变。评分使用预测起点之前的最后历史及变化。因此整个预测不读取未来目标，但这不是按每个中间时刻严格前缀因果的流式算子。时间单位是原观察采样步，不是已识别旅行时间。

将插值延迟核记作K_l。k跳路径累计各边离散插值滞后，最终接收点等于源点的返回路径按累计滞后扣除。保留两个算子：

- D_k(P)：仅累加时间索引确实存在的延迟路径。
- C_k(P)：沿完全相同的可用路径系数，但特征全部读取P(t)。

新道路消息为 `M_k_native(P) + tanh(a_k) [D_k(P) - C_k(P)]`。每个道路方向/空间阶数一个a，初始为0。原native消息使用原展开图的原einsum直接计算，保留其浮点运算顺序。

对于完整替换尺度1，等价于可用路径使用过去值，缺失的完整路径使用同刻值；不是逐边夹紧到第一条历史，也不是重新归一化。两跳使用完整路径的累计可用性，不将缺失中间状态伪装成观测。时间上恒定的P保持原道路消息，包含最早边界。该性质只涉及图路径系数/特征的代数计算，不是预测精度、车辆守恒或物理容量保证。

实际尺度在(-1,1)，允许负残差。因此不能把它解释为概率或保证所有有效特征权重非负。零尺度用于成熟模型等价初始化；第一步a获得梯度，随后u及评分器获得梯度。内部延迟初始非零，整体模型初始输出却与原模型完全一致。它不是通过将整个内部路径和尺度同时置零而产生死分支。

## 对照和诊断

三组从同一原模型最佳权重开始：CONTROL纯原模型、FIXED第一层固定边延迟、DYNAMIC第一层状态条件边延迟。保留原Adam、L2、学习率、全局梯度裁剪、dropout、调度器和课程；三组同样重置优化器及调度器，接收相同额外轮数。

这属于成熟模型配对额外微调，不能叫原最佳轮优化器精确续训。原版参数455220；两种新版483998，增加28778。FIXED注册状态评分器但不调用，其参数没有任务梯度也不被Adam衰减；相同注册量不代表相同活跃容量。

每组分别加载BEST和LAST，完整回放后保存17种状态：SOURCE、CONTROL的两种轮次，FIXED两种轮次下的ON/NO_ALIGN/ZERO_BOUNDARY，以及DYNAMIC两种轮次下的ON/NO_ALIGN/ZERO_BOUNDARY/SHUFFLE。

- NO_ALIGN直接恢复原代码路径，保留当前联合训练主干。
- ZERO_BOUNDARY保留同一延迟，取消缺失历史的同刻回退。
- SHUFFLE在同一方向循环错配边延迟；保存实际改变了多少延迟及预测。

只有第一层读取延迟，第一层输入不会受这些探针影响，因此ZERO_BOUNDARY可以保持实际边延迟相同。各探针为同检查点敏感性，可能分布外，不能直接替代删模块重训或真实因果实验。

梯度审计使用原顺序前两个训练批，原全十二时距损失、原掩码与逆变换。先记录原任务梯度，再按原全局范数5裁剪，单独记录Adam随后加上的L2项；grad=None与数值零梯度分开统计。只有backward，没有optimizer.step，结束或异常都恢复张量、已有梯度、训练模式及RNG。旧运行可选审计三组LAST，不重新推断旧训练全过程；两个批的比值不是历史更新的因果份额。

## 服务器同步与第一轮命令

在原仓库、`research/chronological-tiid`分支更新：

```bash
git pull --ff-only origin research/chronological-tiid
```

首轮固定1轮，先补旧运行梯度审计，再完成新版三组实验：

```bash
bash experiments/chronological/run_original_edge_delay_v2.sh \
  --source-run experiments/chronological_runs/contra_original_baseline_s2025 \
  --audit-pilot-run experiments/chronological_runs/contra_original_edge_delay_pilot_s2025 \
  --run-dir experiments/chronological_runs/contra_original_edge_delay_v2_s2025 \
  --epochs 1
```

在已分配的GPU作业内直接运行；不在Slurm作业内则launcher提交sbatch。Python解释器明确指向igstgnn环境，不受当前(base)提示符影响。旧pilot须是完整等预算运行，且原源码、数据和源权重身份一致。缺旧运行时可以省略`--audit-pilot-run`，会清楚记录没有进行旧运行审计。

被中断后以同一组参数恢复，`--epochs`仍是目标总额外轮数，不能换到旧v1目录：

```bash
bash experiments/chronological/run_original_edge_delay_v2.sh \
  --source-run experiments/chronological_runs/contra_original_baseline_s2025 \
  --audit-pilot-run experiments/chronological_runs/contra_original_edge_delay_pilot_s2025 \
  --run-dir experiments/chronological_runs/contra_original_edge_delay_v2_s2025 \
  --resume --epochs 1
```

每轮原子保存last、优化器、调度器、课程、样本顺序和RNG；恢复从最后完整轮次开始，未完成一轮重新计算。相同目录同时运行会被锁拒绝，应使用原任务。已完成1轮后若决定完成5轮预算，保留全部参数，只将目标改为`--resume --epochs 5`，不能把五轮不同结构比较解释成相同计算量。

完成后读取：

```bash
cat experiments/chronological_runs/contra_original_edge_delay_v2_s2025/report.md
```

`summary.json`保存完成预算、回放/恢复证书、计数及全网指标；`gradient_summary.csv`与各gradient JSON记录任务梯度和L2；`mechanisms.json`记录边延迟、跨边/跨样本差异；`boundary_support.csv`记录回退前缺失比例，不能标为本版实际丢失比例；`resources.json`记录训练墙钟及整个arm（训练+验证+重载）峰值显存，不能冒充纯训练步峰值；`timings.json`把完整前向与诊断成本分开。

## 验收和解释范围

本地测试覆盖显式一跳/两跳路径参考、返回路径、边界回退、有限差分、原初始化输出及dropout RNG、主干梯度一致性、分阶段梯度、原未来调用次数、探针恢复和梯度审计异常恢复。另以本地GPU四行训练/验证工程集检查三组训练和断点恢复，不能拿这四行衡量预测改进。

先检查初始精确回放、LAST是否退化、收益是否依赖真正变化的正确边时间关系，以及MAE/RMSE/MAPE与固定困难窗口是否一致。零初值导致BEST至少保留原源水平，但保住源指标不表示新模块产生收益。若CONTROL共同明显退化，须同时审查成熟模型重新优化的协议；边界修补并不足以解释或解决全部主干漂移。

延迟建模本身已有研究，不把重参数化、零初始化或边界补偿单独声称为原创贡献。只有后续同配方结构对照和独立确认支持时，才能讨论预测优势。当前数据加载会反序列化包含其他划分的发布对象数组，但不创建测试loader、不评价测试集；开发验证集已反复用于设计，重叠窗口不视为独立事故重复。
