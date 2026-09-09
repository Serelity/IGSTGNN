# 论文公式对齐版 v1

本分支 `fix/paper-alignment` 根据 IGSTGNN 论文第4–6页 Eq.6–16、Table 2修正已核实的实现差异。修改前代码保留在 `main` 的 `558ab01`，三个历史结果保持不变；历史结果包未包含完整Git/启动记录，无法仅凭包内日志独立确认其运行提交。本版标记为 `paper_aligned_v1`，不是已经复现论文数值或已经证明预测更准的模型。

## 修改范围

| 问题 | 原实现 | 本版 |
|---|---|---|
| 传感器字段 | 优先Sensor Type；读取Road Width；带单位的数值解析失败后全变0/50 | 按Table 2读取Type、Lane Width、Design Speed Limit；解析单位并记录缺失 |
| ICSF归一化方向 | 在节点维N上softmax | 按Eq.6/7在每节点的事故维上softmax；当前数据M=1 |
| ICSF融合输入 | 学习距离表示和逐通道注意力 | 初始注意力标量、传感器表示、未经学习编码的3通道关系D；输出事故权重 |
| TIID初始上下文 | 直接复用ICSF的注意力×V | Eq.13独立MLP从局部K、传感器表示和D构造上下文 |
| TIID衰减时间 | 对4个隐藏块使用tau=1至4 | 对12个真实预测步使用tau=1至12 |
| 分块输出长度 | 使用seq_len作为未来长度 | 使用horizon、对不足一块的末尾裁剪，保留gap=3的输出通道顺序 |
| 数据重叠检查 | 没有针对实际split的检查入口 | 新增check_overlap.py，输出指纹和重叠计数，可要求重叠时非零退出 |

模型文件为 `src/models/igstgnn.py`，字段解析为 `src/utils/dataloader.py`。`--icsf_dim` 原先未生效，现取消该误导性命令行参数；日志记录真实维度 `num_hidden`。

未改变：原始数组与split、归一化统计量、损失与评价掩码、batch 48/48/24、优化器和学习率计划、warm_epoch=30/cl_epoch=3课程、patience=20。课程和重复位置编码等目前属于待实验验证的问题，未作为已证实错误一并修改。

## 必须理解的实现边界

### 单事故softmax退化

发布样本每条只有一个事故，没有多事故维数据。按论文Eq.6/7，有效节点上的单事故权重必然为1，无效节点为0。这意味着：

- ICSF的Q和融合MLP在该单事故路径上没有有效学习信号；S/D的数值差异不再调节ICSF权重，D仍用于连通mask。
- K及传感器表示还会经过独立TIID分支学习。
- ICSF上下文不再按节点总数分摊，数值变化可能很大，不能将其描述为微小修补。
- 这是公开单事故数据与论文字面公式结合的退化现象。本版没有伪造多事故输入，也没有声称已解决空间异质性。

### TIID与分块主干

主干保留4块、每块3步的结构。解码时将每块隐藏态复制到其三个预测步，对每一步施加TIID，再从原输出头选择相应的块内通道。无TIID时输出与原分块输出一致，已有数值测试；有TIID时12个步分别衰减。

这个分块兼容过程是明确的实现约定，论文没有写出该适配细节。独立上下文MLP使用64维中间层，构造后再mask以避免bias给不连通节点带来影响，也是保持论文语义的工程选择。D沿用发布数据的接近度/方向编码和原连通mask判据，没有重新验证物理路网关系。

默认sigma=1仍按论文设定，H1/H3/H6/H12的系数约为0.6065、0.01111、1.52e-8、5.38e-32。这仅表示显式TIID附加分支的衰减；ICSF经主干传播的事故影响仍然存在。

### 传感器单位与缺失值

Table 2的字段名明确，但表中没有明确列出单位。本版约定车道宽度用m、限速用km/h：`ft×0.3048`、`mph×1.609344`，已有m和km/h不重复换算，裸数视作约定单位。模型接口键`road_width`为兼容现有调用保留，里面实际是Lane Width。

空白或仅有单位而无数字视为缺失，宽度默认0m、限速默认50km/h，日志列出缺失数量。其他非空无法解析值或NaN/Inf会报包含字段和CSV行号的错误。缺失填补策略本身没有被论文唯一确定，当前默认值是保留的工程约定。

真实三城CSV验证：

| 城市 | 节点数 | 宽度不同值数（含默认） | 限速不同值数（含默认） | 宽度/限速缺失 |
|---|---:|---:|---:|---|
| Alameda | 521 | 13 | 4 | 85 / 83 |
| Contra_Costa | 496 | 14 | 6 | 4 / 4 |
| Orange | 990 | 32 | 4 | 149 / 141 |

三城Type均为Mainline。新增TIID参数以及类别/融合层维度改变，**旧checkpoint不能直接加载到新模型，也不能在新特征输入下直接评价旧权重并宣称修复有效**。

## 数据协议仍需单独处理

本版没有消除旧切分的交通帧交叉。发布数组缺少绝对日期和事件ID，无法通过排序现有特征恢复严格时间切分。按共享帧分组技术上可行，但属于新的评估协议，必须另目录建数据并让全部对照模型使用相同协议，不能与已发表表格混为同条件复现。

检查工具读取服务器实际的 `incident_train/val/test.npy` 和stats，而不是推测其内容与本地all切片一致：

```bash
python data/xtraffic/check_overlap.py --dataset Contra_Costa --output Contra_Costa_overlap_paper_v1.json
```

这是CPU与磁盘检查，不使用GPU；按平台规则在允许的数据处理终端运行。对象数组仍需足够主存。输出文件仅创建，不覆盖已有文件；再次运行可省略`--output`直接看stdout，或使用新文件名。

如果某项实验要求没有精确帧交叉：

```bash
python data/xtraffic/check_overlap.py --dataset Contra_Costa --fail-on-overlap
```

有交叉时退出码为2，仍输出报告。不要把这个门禁串到明确采用发布协议的训练前面后忽略其退出状态。即使没有精确重复，也不等于证明日期/事件完全独立。

## 本地验证与未完成的验证

- 先用测试复现字段解析、归一化轴、TIID上下文/时间粒度、horizon和启动参数问题，再修改实现。
- 本地自动测试覆盖单位解析、单事故聚合、TIID逐步衰减与mask/梯度、分块输出对应、完整模型反向、重叠门禁、原有NumPy兼容和图卷积显存回归。
- 2026-09-09完整运行35项测试全部通过，独立复核同样通过；检查了Bash语法与Git差异格式。
- 已使用真实Contra前2条样本、全部496节点和5层模型，在CPU完成一次前向/反向：输出`[2,12,496,1]`，预测/损失/梯度有限；没有执行optimizer更新。这不能作为预测精度结果。
- 本地验证环境为Python3.12、PyTorch2.3.1+cpu、NumPy1.26.4、SciPy1.11.4；服务器继续使用现有Python3.10/PyTorch2.3.1+cu121/NumPy1.24.4，不需要重新安装。
- 尚未验证服务器V100全batch显存或训练后精度。逐步解码比原4块解码增加了预测头激活，因此先做一轮GPU检查。原einsum显存修复保留。

## 服务器执行

先在未运行训练任务时更新到独立分支：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git fetch origin
git switch fix/paper-alignment
git pull --ff-only origin fix/paper-alignment
conda activate igstgnn
```

先提交一轮真实batch=48的Contra检查，仍使用原单卡Slurm脚本：

```bash
sbatch --export=ALL --job-name=igstgnn_paper_smoke --time=00:30:00 experiments/IGSTGNN/run.slurm Contra_Costa --smoke
```

若使用平台网页任务，在已分配的GPU任务内启动：

```bash
bash experiments/IGSTGNN/run.sh Contra_Costa --smoke
```

验收GPU/CUDA正确、日志显示paper_aligned_v1、传感器字段解析日志合理、1轮训练与12步测试正常结束且无非有限值/OOM。旧一轮结果的误差不能作为新模型的一轮验收阈值；新旧结构和初始化已经变化。

通过后，再提交完整Contra训练：

```bash
sbatch --export=ALL --job-name=igstgnn_paper_Contra experiments/IGSTGNN/run.slurm Contra_Costa
```

或者在平台GPU任务内：

```bash
bash experiments/IGSTGNN/run.sh Contra_Costa
```

上限100轮、patience20、seed2025、batch48、原课程30/3保持不变。先观察验证MAE和完整测试指标，确认没有实现异常后再决定扩展城市和随机种子；本次多个公式/输入修正是一组累计修正，指标变化不能分别归因到某个模块，更不能直接称作创新收益。

新结果默认写到：

```text
experiments/igstgnn_paper/Contra_Costa_igstgnn_paper_aligned_v1_s2025_<timestamp>/
```

继续保存Slurm `.out/.err` 和目录内日志/权重；一轮检查与完整训练各自拥有新时间戳，不覆盖旧结果。请求的30分钟/8小时均为任务上限，实际费用依平台分配与计费规则；旧Contra约51分钟的记录不能当成本版耗时承诺。

若之后单独比较固定12步监督，可在独立运行中显式设 `--warm_epoch 101 --max_epochs 100`，避免本轮范围内发生重置。不要用`warm_epoch=0`或`cl_epoch=0`代替关闭课程；本轮默认没有改变此策略。
