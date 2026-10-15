# 事故条件消息传播：实现与运行

2026-10-15。论文主 baseline 为原始 IGSTGNN。本版将新算子接入五层 DIF，结构目标是让报告参与消息选择和消息内容；预测效果尚待正式实验。公式及其条件性质见[设计合同](igstgnn_conditional_propagation_contract_20261015.md)。

## 本版改动

一个节点局部报告编码器将历史摘要、三列关系、类别语义和支持 M 编成 R[B,N,32]，整网复用一次。历史摘要来自 latest-normalization 之前的流量嵌入；原 static/dynamic 图来自 latest-normalization 后、尚未受到报告影响的历史。

新 DIF 用接收状态、来源状态、窗口位置/来源类型/生成块号与 R 的联合交互，同时产生有界 logit 增量与 value 增量。每层保留 self、predefined 四组、dynamic 四组、static 两组，共11个 bank。按接收节点分块计算，训练默认启用非重入 checkpoint；没有构造完整的边×通道消息张量。原图构造一次，但 static 二阶展开及 prior 抽取仍按每次核调用执行。

旧 localized convolution 的全窗变换、潜交换器及历史容量 prior 被移除。原 estimation gate、DIF backcast、残差分解、INH 与预测头保留；每层产生四个原生 gap-three 预测特征块，整网仍输出12槽。最终融合器没有接入本版。兼容权重复制不保证复制旧 MIN 的预测，因此各组必须计算自己的阶段0。

对应此前五个问题：联合权重/value 条件化为关系内容提供新入口；状态×报告交互提供选择性表达；时间来源与生成块号显式区分；报告可沿内部消息和残差影响输出；主干与新增参数联合训练。这些是结构回应，关系是否增益、选择是否正确及指标能否提高仍需实验。单次固定输入扰动界不保证全网稳定，也不认证真实容量或车辆守恒。

正式宽度32时模型共有 **433481** 个参数，本地同结构源纯 MIN 为435856，减少2375。HISTORY、M_ONLY、EVENT 注册结构和初始张量相同；活跃的报告参数不同。

## 文件

| 文件 | 作用 |
|---|---|
| [incident_conditional_propagation.py](../src/models/incident_conditional_propagation.py) | 稀疏11-bank条件消息核、硬支持、报告双路径和分块checkpoint |
| [incident_spatial_conditional_propagation.py](../src/models/incident_spatial_conditional_propagation.py) | 单报告编码、五层DIF集成、来源时间元数据、MIN兼容权重转换 |
| [train_min_conditional_propagation.py](../experiments/chronological/train_min_conditional_propagation.py) | 三组配对暖启动、各自阶段0、全模型训练及精确续训 |
| [evaluate_min_conditional_propagation.py](../experiments/chronological/evaluate_min_conditional_propagation.py) | 最佳检查点重放及同检查点报告/logit/value/语义/位置干预 |
| [min_conditional_propagation_v1.json](../experiments/chronological/min_conditional_propagation_v1.json) | 冻结数据口径、单Adam学习率1e-4、完整12时距损失 |

## 三组与选模

| 组 | 输入 |
|---|---|
| HISTORY | 历史；不读取报告 |
| M_ONLY | 历史与原支持M；屏蔽关系数值和全部语义，保留M |
| EVENT | 历史、原支持M、关系数值与类别 |

三组从同一个 MIN 最佳检查点复制兼容权重，新增算子按同一种子初始化，所有参数联合训练。事件最终读出零初始化，因此三组自身阶段0预测相同；第一步主要更新事件读出，后续步骤报告内部因子可获得梯度。使用相同原始训练顺序、batch size、完成轮数与优化器更新预算。

各自阶段0与后续验证参与最佳选模；阶段0不是旧 MIN 验证指标。优化器和scheduler重新建立，不声称从源 MIN 精确续训。新阶段内部保存完整optimizer、scheduler、RNG和数据顺序；`--epochs`是绝对完成目标，目标已达时不继续更新、不重写检查点。

## 服务器运行

在服务器仓库更新代码后，先完成一轮配对试验：

```bash
bash experiments/chronological/run_min_conditional_propagation.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_conditional_p1_bs64_seed2025 \
  --epochs 1
```

launcher在未处于Slurm作业内时提交任务；提交返回不表示训练完成。等待输出 `MIN_CONDITIONAL_PROPAGATION_PAIRED_STAGE_COMPLETE` 后，再运行评估：

```bash
bash experiments/chronological/run_min_conditional_propagation_evaluation.sh \
  --run-dir experiments/chronological_runs/contra_min_conditional_p1_bs64_seed2025
```

评估保存8种新状态及可用的SOURCE_MIN归档参照，输出 `summary.json`、`report.md`、`analysis.json`、`mechanisms.json`、`comparability.json`、`replay.json`、`paired_predictions.npz` 与来源/保护记录。SOURCE_MIN仅从匹配的已有F1 `source_reference` 复用；没有归档就省略，不重新推理源模型。独立训练组差异、同检查点路径依赖与真实事故因果须区分。

后续同一目录续到5轮，将首条命令的 `--epochs 1` 改成 `--epochs 5`。协议、资源选项、源检查点和runtime身份不能改变。默认batch64、接收分块4、checkpoint开启；分块和checkpoint选项用于资源适配，并写入身份，不能在同一个run中修改。没有开展学习率或宽度搜索。

三组目前各自按patience20早停；若触发时完成轮次或更新预算不同，摘要会明确预算不等，配对评估会拒绝这种比较，不能标成同预算主实验。

`--check`只接受工程身份的 MIN-I 源运行，采用4个train/val样本与batch2；不能搭配正式源检查点冒充完整验证。正式源默认目录即上例。源检查点必须与当前Torch/CUDA/device类型和冻结数据/代码身份一致；不需要原GPU型号相同。

## 验证与边界

本地覆盖独立dense公式和梯度对照、同质/零value的报告响应、硬图及断连隔离、节点重排、报告关闭/不可用/零支持精确回退、序列化、三组自身阶段0、两步梯度、历史锚点与生成块来源、连续训练/断点恢复/no-op、中断保护及完整配对评估闭环。工程运行详情见[验证记录](verification/igstgnn_conditional_propagation_implementation_20261015.json)。

最终39项测试全部通过。另用实际发布数据的4个train与4个val样本、496站点、宽度32完成三组各1轮/2次更新和完整配对评估；三组阶段0相同、预算相同，8种新状态及源归档参照重放/保全通过，EVENT_ON干预后精确恢复。此工程检查在CPU、batch2、接收分块32运行，不能外推服务器batch64性能，也不作为预测增益证据。

kernel的tiny CUDA FP16 autocast检查已执行；**完整模型与本版训练入口只验证/使用FP32，不支持整网AMP**。本地GPU和服务器V100/Torch2.3环境不同；尚未认证服务器batch64峰值显存或吞吐，epoch会保存seconds及CUDA峰值分配以供核对。4样本工程指标不能用作预测效果或论文结论。

没有运行正式训练或评价测试集。沿用发布对象加载器时会反序列化全样本对象，包括测试payload；不迭代测试loader。报告实时可用性、关系三列物理方向和跨split独立性仍未认证。开发验证集已参与多轮设计；最终论文需要原IGSTGNN同口径对照与独立评价。
