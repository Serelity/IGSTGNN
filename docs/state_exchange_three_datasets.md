# GW16 STATE_EXCHANGE 的三数据集扩展

本次固定已有最优方法GW16 STATE_EXCHANGE，不使用仍待验证的GW17结构。
开发机只编写、读取和静态检查；Python、单测和所有数值计算在服务器执行。

## 数据与结构

| 数据集 | 节点 | 总窗口 | 训练 | 验证 | 测试 | 事故类型数 | 描述数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alameda | 521 | 14687 | 10280 | 2203 | 2204 | 6 | 30 |
| Contra_Costa | 496 | 5585 | 3909 | 837 | 839 | 6 | 29 |
| Orange | 990 | 18700 | 13090 | 2805 | 2805 | 6 | 32 |

数据来自各县已有作者model-ready发布包；registry.json固定每套数据六个文件的SHA256、
节点数、总窗口数和按作者70/15/15向下取整划分的数量。没有重新划分日期、重建窗口或拟合归一化。
全样本object array反序列化包含测试载荷；训练只验证和构建训练/验证loader，测试需显式入口。

各县分别从头训练；不把Contra Costa权重直接拿到其他节点轴上推理，也不混合训练三县。
模型继承已有PositionStateModel.forward、HistoryEncoder、CoupledHead和P/Q几何，
保持隐藏32、更新宽度96、两层空间编码、rank8、max_region_size32、top-k8。
位置和区域由本县固定站点元数据按同一道路/方向/里程规则生成。
事故类别ID沿用本地mapping，不跨县共享ID含义；描述embedding按30/29/32类别加各自UNKNOWN扩展。
除此之外不增添GW17的观测校正，也不按新县验证结果改变结构或训练配方。
服务器测试核查Contra相同形状时每张初始权重和每个前向输出逐位恢复已有GW16。

## 两种模型及训练预算

- STATE_EXCHANGE：原GW16配方，从头固定100轮、seed2025、batch48、Adam0.002，
  40/70/90后学习率乘0.5，节点MAE加0.2区域MAE，无早停。
  BEST按验证pooled MAE选择，包含第0轮。
- ORIGINAL_IGSTGNN：冻结558ab01原版模型、原生loader和原生engine；
  复用本项目已有Contra原版复现配方：seed2025、batch48、最多100轮、patience20、
  warm30、课程每3轮推进、里程碑1/38/46/54/62/70/80。
  BEST沿用原生验证pooled MAE，不包括第0轮。
  这不是声称batch48等于作者未覆盖命令行默认值；它与本项目现有原版参照保持一致。

两种模型保留各自已固定配方，输入特征也有差异；比较的是整套方法表现，
不能将全部收益独立归因于状态交换，不能称为完全相同训练配方的机制消融。
每个数据集记录实际注册参数、训练更新数、轮次和耗时。参数量不替代运行时间或显存测量。

Contra Costa已有完整训练和冻结配对测试，不重训、不重新选择或推理。
默认只新增Alameda/Orange各两模型共四次正式训练，并读取已有Contra测试缓存组成主表。

## 指标与测试边界

主表遵循现有原版评估：在同一目标和mask上分别计算12个时距指标，再等权平均MAE/RMSE/MAPE。
RMSE不是pooled RMSE；共同掩码FP64 pooled指标另列，不能与主表混算。
原文表中数值仅可作为外部参考；正式提升率以同样本原版回放为分母。

先对Alameda/Orange完成训练并冻结验证集BEST，再显式调用--split test。
测试入口不提供改选LAST、校准、重训或根据测试延长预算的选项。
test_lock.json在测试loader创建前提交，绑定数据、执行源、runtime、两个BEST与LAST的文件SHA。
每模型缓存原子发布、重读严格复算；中断时已完成模型零新前向复用。
新预测统一C布局后再计算FP32原生指标，避免旧测试的写盘步长差异。
Contra缓存只使用已有固定布局恢复和严格复算，无损、不改旧文件、不重复模型前向。

这是单seed跨三个数据集的表现检验；没有多seed显著性结论。
发布窗口重叠，绝对时间/事故独立性和报告首次可用时间尚未独立认证。
负结果同样保留，不按测试结果挑选县、调整模型或追加轮数。

## 服务器运行

从GitHub固定提交无Git下载到新目录后，在igstgnn环境执行：

```bash
bash experiments/state_exchange_multidata/run.sh
```

流程：服务器单元测试 → 两县各两模型3轮小样本工程检查和验证归档 →
各模型正式batch48完整前向/反向/Adam状态显存检查 → 两县正式训练 →
验证回放 → 冻结BEST的配对测试 → 加入Contra旧缓存 → 三数据集主表。
工程只使用四个训练和四个验证窗口，没有工程测试集入口。
batch48检查在最前面的48个训练窗口上做两次更新，权重随检查丢弃，不进入正式模型或结果表。
检查同时覆盖首次Adam状态分配和状态已常驻时的第二次前向/反向；显存不足时停止，不缩batch。
中断后重复同一命令，恢复完整Adam、调度器、RNG、训练排列及原版课程长度。
candidate第0轮为BEST候选；原版仅提交第0轮的恢复状态，仍不参与原版BEST选择。

默认服务器父目录：
`/seu_share/home/huangkai/220243809/paper/IGSTGNN`。
数据位于其`data/xtraffic/Alameda`、`Contra_Costa`、`Orange`。
旧Contra测试目录：
`IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025_final_test_locked`。
默认新结果位于新代码目录的
`experiments/chronological_runs/gw16_three_datasets_s2025`。
路径保存到新代码目录`gw16_multidata_paths.env`，不依赖旧shell变量。

可先运行单县；第二次沿用同一结果根目录即可补齐另一县：

```bash
GW16_MULTI_DATASETS=Alameda bash experiments/state_exchange_multidata/run.sh
GW16_MULTI_DATASETS=Orange bash experiments/state_exchange_multidata/run.sh
```

部分训练预算的数字是累计目标，不是追加轮数；只有完整协议才会开启正式测试：

```bash
GW16_MULTI_EPOCHS=20 bash experiments/state_exchange_multidata/run.sh
GW16_MULTI_EPOCHS=100 bash experiments/state_exchange_multidata/run.sh
```

数据或缓存确实移动时可设置GW16_MULTI_DATA、GW16_CONTRA_TEST；结果根可设置GW16_MULTI_RUN。
已有训练的恢复必须使用同一固定提交、输入和runtime。batch保持48，脚本不在OOM后自动缩batch。
初始化缺identity时保留现场，设置新的结果根；已提交初始化的第一轮中断可直接恢复。

完成后：

```bash
source ./gw16_multidata_paths.env
cat "$GW16_MULTI_RUN/Alameda/test/report.md"
cat "$GW16_MULTI_RUN/Orange/test/report.md"
cat "$GW16_MULTI_RUN/three_dataset_test/report.md"
```

主表、全部时距和参数分别在three_dataset_test目录下report.md、metrics.csv、per_horizon.csv、comparison.json。
如Contra旧缓存缺失，两个新县结果仍各自保存，脚本会提示补齐GW16_CONTRA_TEST路径后再汇总。
