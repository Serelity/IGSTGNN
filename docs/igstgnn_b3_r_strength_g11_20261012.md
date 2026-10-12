# IGSTGNN B3-R-G1.1：从恒等强度开始的D路径对照

日期：2026-10-12。目标是检验状态条件的D修正强度能否减少不利修正，同时保留有利修正。本版使用独立模型、训练入口与协议；原B3-R、G1.0及其检查点仍沿用各自原入口。

## 采纳的修改与保留的边界

G1.0的`alpha=sigmoid(G)`初始为0.5。因为D末层为零，三组初始容量与预测本来就相同；问题是D开始学习时，GATE的首步D任务梯度带有0.5的缩放。G1.1改为`alpha=2*sigmoid(G)`，G末层权重和偏置仍置零，使GATE初始alpha=1，与原B3-R的D幅度及首步D任务梯度匹配。固定HALF仍保留为有意的0.5尺度对照。

门控增加的是幅度控制参数化，没有新增历史信息，也不据此宣称具有严格更强的函数类。D本来就读取历史和报告；D与alpha存在尺度补偿，原D可能变大以抵消较小alpha。不能把门值称为校准过的事故可信度，也不能把门均值降低直接解释为事故信息使用减少。

当前保留D/G相同的完整输入，不拆成“D只读事故、G只读历史”。这样先检验一个明确的参数化问题；输入拆分会同时改变信息和表达条件，需要另设版本及对照。OFF高估/低估、未来目标、未来事故状态均不进入D或G。

## 四组怎样比较

```text
z_P = P(history_summary)
z_D = D(history_summary, report_embedding, distances, report_age)
effective_D = report_mask * alpha * z_D
c = 0.001 + 0.999 * sigmoid(z_P + effective_D)
```

| arm | 模式 | 作用 |
| --- | --- | --- |
| `B3R` | `original_D`，alpha固定1 | 原B3-R参数化参照。 |
| `B3R_HALF` | `fixed_half`，alpha固定0.5 | 判断固定缩放及其优化尺度的作用。 |
| `B3R_GATE` | `identity_gate`，alpha=`2*sigmoid(G)` | 每节点一个强度，四个潜通道共享；初始1，连续允许衰减或增强。 |
| `B3R_NO_D` | `pd_report_off_tiid_on` | 从头独立训练并跳过整个D路径，容量只由P产生，原TIID开启。 |

G为156→16→1：历史末值/均值/首尾差、已有时钟嵌入，以及报告编码、空间关系和归一化报告年龄，输入与D一致。GATE只乘masked D logit，不乘P、边界头、读出或最终预测。数学上alpha在(0,2)，诊断允许浮点端点[0,2]；不是原G1.0的[0,1]。

四组从同一个规范B3映射得到共同P/D及保留模块和buffers，D末层仍为零，G末层为零；优化器在公共初始化后创建。所有应学习的参数继续联合训练。D为零时G任务梯度为零是预期，不能为获得非零门梯度擅自改变初始化。

NO_D是独立从头训练对照，不能用ON模型的优化器或学习后的权重直接转换。它也不是“没有事故信息”：原TIID仍开启；另外，D本身包含历史处理，关闭D会同时移除这条联合处理路径。相同检查点ON/OFF评估的是路径敏感性，与这里的重训练问题分开。

完整496站模型的注册参数量：B3R、HALF、NO_D均为455,780，GATE为458,309，增加2,529。NO_D保留公共D参数供初始化核验，但不执行D，D的任务梯度为`None`、不进入实际更新；不能把它视为训练故障。

数据、图、4个潜通道、2子步、eta=0.5、容量及共享收发预算、读出、主预测MAE损失和优化协议保持原值。仍使用train3604/validation917、496站、batch48，100轮上限、patience20及原学习率里程碑。Test封存。

## 需要一起看的诊断

- alpha按报告关联sample-node每次完整forward统计一次；均值、总体标准差、最小/最大值和范围检查采用[0,2]。固定1与0.5是两个参照，不能继续套用G1.0的[0,1]语义。
- 统计alpha低于0.1、高于1.9、低于1及高于1的比例。NO_D标记`not_computed`，没有关联节点时标记`no_associated_nodes`，均不补成门值0。
- 同时报原始masked D、实际masked alpha×D以及D幅度加权alpha，识别尺度补偿；NO_D须标记路径未执行，不把未计算的D写成训练后恰好为零。
- 沿用P与更新后节点/边容量、容量变化、预算激活、真实执行的潜交换及读出诊断。容量变化非零不保证最终通量或预测改变。
- 记录D/G任务梯度：受控首步验收比较GATE与B3R的D梯度，零D时允许G梯度为零；后续重新前向后检查学习路径。训练中的一个零梯度batch不能自动判为失效。

主评价仍是同预算验证MAE及最佳检查点规则；10轮是优化与机制观察节点，不按早期排序淘汰。GATE优于HALF还不足以说明优于原B3-R，更不能代替对原IGSTGNN/B0的最终比较。

## 服务器更新、启动与续跑

先单独更新Git，Bash脚本内不执行Git：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code

git pull --ff-only origin research/chronological-tiid
```

工程检查先运行新模型与runner测试，再运行四组小样本训练；自动创建独立check目录：

```bash
bash experiments/chronological/run_b3_r_strength.sh --check
```

初始化预期为`B3RSTRENGTH_INITIALIZATION_PAIR_PASS`，完成为`B3RSTRENGTH_PAIRED_STAGE_COMPLETE`。工程检查只取前4条train/validation、batch2，不以其MAE比较方案优劣。

通过后新建正式开发目录，四组各训练到累计10轮：

```bash
bash experiments/chronological/run_b3_r_strength.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --epochs 10
```

从同一目录把四组继续到累计20轮：

```bash
bash experiments/chronological/run_b3_r_strength.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --epochs 20
```

如果仅需补跑GATE，使用父进程的`--only`并保留原四组身份：

```bash
bash experiments/chronological/run_b3_r_strength.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --only B3R_GATE --epochs 20
```

`--epochs 20`是绝对累计目标，不是增加20轮。中断后重复相同指令从各组checkpoint恢复；完成的组自动跳过。单组补跑后只比较共同前缀轮次的最佳值、共同轮次与共同末5轮，另列实际完成轮数，不把GATE20轮与其余10轮算成同预算。不要直接调用内部worker。

登录节点自动sbatch，已有Slurm allocation中直接执行，1 GPU/3 CPU/6小时、无`--mem`，日志`b3rstrength-%j.out`；使用固定igstgnn Python，`(base)`提示符也可执行。可用`B3R_REPO_DIR`、`PYTHON_BIN`覆盖路径，`B3R_STRENGTH_DIRECT=1`仅在已有直接运行资源时绕过提交。

## 版本、恢复和后续分析

新协议为`internal_spatial_b3_r_strength_g11.json`，架构`B3-R-G1.1`，初始化映射`b3r_common_pd_identity_gate_nod_g11_v1`，诊断schema为`b3r_strength_diagnostics_g11_v1`。不能把原B3-R或G1.0目录作为新run。源码、数据、图、协议、组选择和模式均进入恢复身份；兼容V100 SKU名称变化作为环境审计。

单组续跑仍经过父进程锁；checkpoint提交后旁档未完成的中断按已校验检查点恢复，保留损坏旁档原始备份，不覆盖合法但冲突的历史。四组在独立子进程中顺序运行。

原B3-R的同检查点探针和离线联合分析可继续在原路径使用，例如：

```bash
bash experiments/chronological/run_b3_r_joint_gain.sh \
  --probe-dir experiments/chronological_runs/contra_b3r_v11_seed2025/b3r_path_probe_20261012_205122_3150be
```

原联合分析只读已有预测，不加载新模型。G1.1最佳检查点的推理探针适配属于后续开发，本版不宣称已经实现，也不把新训练目录交给原B3-R探针。后续需要共同预算、路径干预与配对种子复核；当前不宣称在线报告证据齐全、真实容量辨识或事故因果效果。

## 本地验收

新模型12项测试、新训练器7项测试、原B3-R/G1.0回归30项测试，共49项通过。验证了非零D时初始GATE与原B3-R预测/容量精确一致、零D时首步D任务梯度一致、signed D放大/衰减、HALF复现G1.0，以及NO_D跳过D并保留TIID。

GATE与NO_D分别通过连续训练两轮和一轮后续跑到两轮的逐项一致性检查，覆盖模型、最佳权重、Adam、scheduler与更新数。新协议、四组共同预算报告、非法门值拒绝及恢复身份也已验证。Bash入口通过语法检查。

本地真实496站数据的CPU工程入口已完成四组各一个小epoch，状态为`B3RSTRENGTH_PAIRED_STAGE_COMPLETE`。新入口B3R对照与原入口工程运行的模型、最佳模型、Adam、scheduler、随机状态和共同数值历史逐项一致，耗时与诊断schema除外。工程检查仅4条样本，不能以其MAE排名，也不代表服务器完整917条验证的收益。

机器可读验收记录：`docs/verification/igstgnn_b3_r_strength_g11_20261012.json`。
