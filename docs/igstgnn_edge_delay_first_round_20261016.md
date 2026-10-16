# 原版 IGSTGNN：道路边时延第一轮开发

本轮是结构候选的工程开发，尚无正式效果结论。相关方法已有 PDFormer、STDDE 等研究，本实现不宣称“首次提出传播延迟”。

## 改动位置

以只读、哈希核对的 `original_igstgnn_558ab01` 为主干。仅替换五个 DIF 层的**历史道路图消息**；原三步非线性投影、自身均值、动态图/静态图 bank、backcast、残差 INH、事故入口和 gap=3 未来递推保持原调用。

每个节点先经原 3C→3C→ReLU，三个混合子槽求和得到 P(t)，t 为完整历史窗口的结束位置。对道路边 j→i，读取 P_j(t−τ_ij)，再按原图权重汇聚。非整数时延通过相邻历史摘要插值；不将混合子槽或四个未来隐块解释为真实采样时刻。

此处是对完整历史窗口进行边级时间对齐，相较最初“在投影前取单个时刻”的构想，保留了原版已学到的窗口编码。新模型历史消息额外需要更早的窗口；超过可用历史的部分置零屏蔽，不重复第一条观测、不重新归一化可用权重。

每个层/道路分支独立学习静态边 logit；DYNAMIC 再由共享小网络读取两端节点的最新隐藏值及一步变化，产生状态修正。τ=2·sigmoid(logit+修正)，默认一条边最大两步。窗口内 τ 固定，跨输入样本可变；这不是逐历史时刻变化的传播速度模型，也不提供前缀流式因果保证。所有输入仍在预测起点以前，不读取未来标签。

两跳通过延迟算子复合实现，离散插值会形成延迟分布；路径的平均滞后累计，而非直接为 A² 指定一个独立时延。最终回到来源节点的路径在自身累计滞后处扣除，保留原 bank 的最终去对角规则，不在中间传播步骤额外删边。

原邻接及其转置只保留原计算方向。未额外认证道路行驶方向、真实道路距离或车速；学习到的数值是预测滞后，不等于真实旅行时间或拥堵波速。

## 首轮对照

| 名称 | 训练/推理职责 |
|---|---|
| SOURCE | 原最佳检查点回放，不新增源训练 |
| CONTROL | 同检查点原版主干额外微调 |
| FIXED | 训练边级常数时延，跨样本固定 |
| DYNAMIC | 训练边常数及状态决定的时延修正 |
| DYNAMIC_NO_ALIGN | 同 DYNAMIC 最佳检查点，历史调用直接回到原代码 |
| DYNAMIC_SHUFFLE | 同检查点，在每个层/道路分支内循环打乱边时延，图和输入不变 |
| DYNAMIC_MASS_ONLY | 同检查点，保留相同边界可用路径质量，值都取当前摘要；区分时间对齐与边界衰减 |

前三个训练 arm 从同一源权重开始，各完成 5 个额外 epoch，原 batch48、Adam、学习率、调度和原训练 engine 不作搜索；优化器/调度/课程状态在各 arm 相同重置，前五轮使用全部十二时距损失。阶段0参加各自最佳选模，不早停。原源 epoch28 与额外 epoch5 不应相加解释为原训练的精确续训。

CONTROL 注册参数455220，FIXED/DYNAMIC 注册参数599090。FIXED 的状态评分器不活跃；注册量相同不是活跃参数或表达能力完全匹配。DYNAMIC 对 CONTROL 的提升可能包含额外参数/联合训练差异；必须同时查看 FIXED、NO_ALIGN、SHUFFLE、MASS_ONLY，不能仅凭一个全网 MAE 归因“识别了传播延迟”。

## 服务器运行

在仓库更新到包含本轮文件的版本后，先做一轮完整验证集的短预算试运行：

```bash
bash experiments/chronological/run_original_edge_delay.sh \
  --source-run experiments/chronological_runs/contra_original_baseline_s2025 \
  --run-dir experiments/chronological_runs/contra_original_edge_delay_pilot_s2025 \
  --epochs 1
```

检查耗时、显存和报告后，在**相同源码与环境**下扩展到总共5轮（不是再加5轮）：

```bash
bash experiments/chronological/run_original_edge_delay.sh \
  --source-run experiments/chronological_runs/contra_original_baseline_s2025 \
  --run-dir experiments/chronological_runs/contra_original_edge_delay_pilot_s2025 \
  --resume --epochs 5
```

被 kill 后也使用第二条命令；仅恢复已完整提交的 epoch。源运行目录、最佳源权重及原版源码不改写。启动器默认通过 Slurm 调度；已有 Slurm 作业内直接执行。`--arms CONTROL|FIXED|DYNAMIC` 可单独续某组，汇总只有三组完成轮数相同时才标记等预算通过。

仅做工程检查可增加 `--check`，使用四个训练/验证窗口、batch2。这样的报告不能用来判断模型效果。正式试运行不要带 `--check`。

## 产物与判断

根目录输出 `summary.json`、`report.md`、`analysis.json`、源回放和各探针预测。每组目录保存阶段0及每轮验证日志、不可变最佳权重/预测、`last.pt` 和训练历史。`last.pt` 同时保存优化器、调度、课程长度、样本顺序与 Python/NumPy/Torch/CUDA RNG。

时延均值、极值、跨边变化、跨样本变化随每批验证写入 `validation_epoch_*.json` 和探针 timing JSON。完整前向同步计时不含H2D、逆变换、CPU统计及这些诊断；首批可能有冷启动。训练墙钟包括加载、前向、反向、更新和指标，不能拿它与纯前向秒数比较。

优先看 DYNAMIC 是否超过同完成预算 CONTROL 和 FIXED，并核对 RMSE/MAPE、十二时距、真实流量分层、道路、固定源困难ID，以及 MASS_ONLY/SHUFFLE 是否保留收益。真值流量只用于事后分组。无测试集评估；已有验证集属于探索集，窗口可重叠，不作独立样本显著性或事故因果结论。

## 开发文件

- `experiments/chronological/original_edge_delay.py`：时延、插值、路径复合、历史 DIF 包装与可恢复推理干预。
- `experiments/chronological/train_original_edge_delay.py`：源回放、三组等预算训练、检查点续训、同点指标与机制探针。
- `experiments/chronological/run_original_edge_delay.sh`：服务器入口。
- `tests/test_original_edge_delay.py`：整数/分数时延、多跳、回路扣除、边界质量、梯度、原路径、未来递推保持和断点恢复。

已有方法参考：[PDFormer](https://ojs.aaai.org/index.php/AAAI/article/download/25556/25328)、[STDDE](https://arxiv.org/html/2402.01231)。
