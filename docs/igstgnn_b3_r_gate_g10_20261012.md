# IGSTGNN B3-R-G1.0：D容量修正的幅度控制

日期：2026-10-12。此版本是原B3-R内部事故容量更新的独立扩展，不改变原模型文件，不修改已有run/checkpoint。原TIID、候选图、受容量及共享收发预算约束的交换、读出与预测头保持一致。

## 开发依据和待检验问题

服务器第84轮最佳B3-R检查点，917行完整验证的同检查点D路径ON/OFF结果：

| 量 | 结果 |
| --- | ---: |
| ON MAE | 22.595564481460897 |
| OFF MAE | 22.59854156660853 |
| 正收益G+ | 0.07731140879777547 |
| 负收益G− | 0.07433432365014277 |
| 净收益 | 0.002977085147632714 |
| 正负收益抵消率 | 98.03681574181115% |
| OFF高估组的全网净贡献 | +0.0529803101035528 |
| OFF低估组的全网净贡献 | −0.05000321713919246 |

OFF高估组承载84.36%的正收益，OFF低估组承载83.54%的负收益。直接报告关联节点占7.99%的有效权重，同时承载约32.88%的正收益和31.45%的负收益。三个空间区域净贡献均为正，尚无依据直接删除其余区域作用。后期时距正负变化都更大，不能直接证明动态事故扩散。

待检验假设是：**控制D的修正强度，能否减少不利修正，同时保留有利修正？** D原本已经读取历史，新门控增加的是幅度控制参数化，没有增加新的历史信息。门值不是经过校准的事故可信度。

## 模型与三个对照

所有组从相同的原B3初始化映射到相同P/D状态，并复制主干、TIID、边界头、交换和所有buffers。训练全部参数，不冻结、不detach、不增加损失或特殊样本权重。

```text
base_logit = P(history_summary)
raw_D = D(history_summary, report_embedding, distances, report_age)
effective_D = report_mask * alpha * raw_D
node_capacity = 0.001 + 0.999 * sigmoid(base_logit + effective_D)
```

| arm | alpha | 参数量（496站） | 目的 |
| --- | --- | ---: | --- |
| `B3R` | 固定1，使用原B3-R实现 | 455780 | 原参数化参照。 |
| `B3R_HALF` | 固定0.5 | 455780 | 判断整体幅度参数化的影响。 |
| `B3R_GATE` | 每节点一个sigmoid门值，四个潜通道共享 | 458309 | 检验随状态调整幅度是否优于固定缩放。 |

门控网络为156→16→1，增加2529参数。输入与D一致：历史末值、历史均值、首尾差、已有时钟嵌入，以及报告编码、空间关系和归一化报告年龄。门控末层权重、偏置均为零，初始alpha严格等于0.5；D末层仍为原零初始化。

因此零D时三组初始预测相同；D非零时，初始GATE仍与HALF逐元素相同。D可正可负，门控只调节幅度，不强迫容量只能下降。P和边界头不乘alpha，readout和最终预测也不乘alpha。未关联节点的修正严格为零；关闭空间报告路径时跳过D和gate，容量回到P，原TIID仍开启。

门值、原始D和有效alpha×D具有尺度不可辨识性：从头训练时，D可以放大补偿固定0.5。因此HALF不保证训练后的容量影响减半，也不能仅凭alpha变小就宣称模型减少使用事故信息。

## 必须保留的诊断

每个epoch的history保留原容量/交换/梯度诊断，并新增：

- 报告关联节点alpha的均值、总体标准差、最小/最大值及低于0.1、高于0.9的比例。分母按sample-node计一次，不复制成四通道。
- 原始masked D绝对幅度、实际masked alpha×D绝对幅度、D幅度加权alpha。
- P/更新后节点及边容量、容量变化、饱和情况、收发预算及受约束交换检查。
- 第一训练batch的任务损失梯度，包括gate。初始D为零时gate梯度为零是预期；D开始变化后，gate末层及前层应逐步获得任务梯度。

alpha均值降低而原D放大，可能只是尺度补偿；alpha近似常数且GATE不优于HALF，不能宣称学到了有效的状态选择。最终仍以共同预算的验证误差、稳定性及后续D路径干预为依据。

## 区域×偏差×时距联合分析

先对已保存的原B3-R最佳检查点探针做离线联合分解：

```bash
bash experiments/chronological/run_b3_r_joint_gain.sh \
  --probe-dir experiments/chronological_runs/contra_b3r_v11_seed2025/b3r_path_probe_20261012_205122_3150be
```

生成独立`b3r_joint_gain_*`目录，成功状态为`B3R_JOINT_GAIN_ANALYSIS_PASS`。`joint_decomposition.csv`包含三空间区域×OFF高估/低估/精确命中×宏平均及12时距，共117行。逐格记录正负收益、净贡献、全网权重、有符号预测移动和绝对变化，并核验各时距与宏平均可加回全网及所有边际。

这一步只读已有预测，无新推理、训练或GPU需求。OFF高估/低估标签依赖验证目标，只用作事后诊断，绝不输入门控。原TIID依然开启，因此是整条D路径敏感性，不是纯事故信息因果收益。

## 服务器启动与续跑

Git更新独立执行，运行脚本内没有Git：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

先工程检查，自动生成独立check目录，运行模型/runner测试及三组各一个小epoch：

```bash
bash experiments/chronological/run_b3_r_gate.sh --check
```

初始化应报告`B3RGATE_INITIALIZATION_PAIR_PASS`，完成应报告`B3RGATE_PAIRED_STAGE_COMPLETE`。工程检查只用前4条训练/验证、batch2，不能排名或用其MAE判断收益。

检查通过后，在新的正式开发目录启动三组同预算筛选：

```bash
bash experiments/chronological/run_b3_r_gate.sh \
  --run-dir experiments/chronological_runs/contra_b3rg_v1_seed2025 \
  --epochs 10
```

`--epochs`为累计目标轮数；相同命令可续跑，已完成的组跳过。中断后重复即可，不删除检查点。仅续某一组时用`--only`，保持原三组身份：

```bash
bash experiments/chronological/run_b3_r_gate.sh \
  --run-dir experiments/chronological_runs/contra_b3rg_v1_seed2025 \
  --only B3R_GATE --epochs 20
```

这不会把另外两组的10轮结果当作20轮对照：报告严格使用共同前缀的最佳值、共同轮次及共同末5轮均值，并另列各组实际完成轮数。尚未启动的组显示partial状态，无完整比较。不要直接调用隐藏worker。

脚本在登录节点自动sbatch，在已有allocation内直接执行，1 GPU/3 CPU/6小时，无`--mem`。Python固定为既有igstgnn环境。GPU SKU名字变化只审计；数据、协议、源码、Torch/CUDA等原身份约束保持严格。可用`B3R_REPO_DIR`、`PYTHON_BIN`覆盖路径；`B3R_GATE_DIRECT=1`仅在已具备运行资源时绕过自动提交。

## 工程与恢复约束

本版独立模型/训练器/协议/诊断，不改原B3-R及原IS1.1文件，所以已有run继续用原入口续跑。新版本从头训练，不能把旧`contra_b3r_v11_seed2025`作为新run目录。

复用旧IS1.1的优化器、训练顺序、每batch随机种子、scheduler、checkpoint及原指标算法，放在独立模块命名空间中，不替换旧模块全局变量。父进程在子worker启动前释放训练/验证数据与图，三组顺序独立进程运行。

单组续跑仍经过父进程run锁。若进程在checkpoint提交后、JSON旁档写完前被杀，以通过身份/状态检查的checkpoint恢复history/summary；截断JSON先保留原始bytes备份，格式合法但与checkpoint冲突的历史不自动覆盖。

10轮只做筛选；后续需要相同预算、多个种子及最佳检查点路径干预。原始IGSTGNN/B0仍是最终比较目标，不能以GATE优于HALF代替改进原IGSTGNN的证据。当前继续使用有限报告、候选传播图和潜容量，Test封存，不宣称在线可用性、真实通行能力或事故生命周期识别。

## 本地验收记录

2026-10-12，本地CPU环境完成50项测试（29.815秒），包括新模型11项、训练/恢复9项、联合分析9项，以及原B3-R和收益分析的21项回归测试。两个新Bash入口通过语法检查，原模型、训练器、协议和探针源码未修改。

新训练入口在真实496站数据上完成三组各一个工程epoch，状态为`B3RGATE_PAIRED_STAGE_COMPLETE`。独立运行原B3-R入口作为复现检查，两个入口的B3-R模型、最佳模型、Adam、scheduler、随机状态及共同数值诊断逐项一致；耗时和新增诊断schema不作为一致性对象。

联合分析入口在本地已有工程探针上生成117行CSV，通过可加性与来源保全检查，无新推理。本地工程验收只使用4条样本，不是服务器917条完整验证的实验结果；新模型预测收益仍待服务器筛选。机器可读记录见`verification/igstgnn_b3_r_gate_g10_20261012.json`。
