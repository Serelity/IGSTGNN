# IGSTGNN IS1.1：内部事故状态与受限空间传播实现

日期：2026-10-11。状态：本地工程验收通过；服务器作业1173577已回传四组累计epoch30，IS11_PAIRED_STAGE_COMPLETE。前30轮最佳MAE排序B1、B0、B2、B3，B3未保持十轮时的最佳值优势；末10轮均值B3最低，但不改选模规则。B2验证裁剪约4.91%，完整机制记录尚待回读。下一步回读分组/机制并保持四组与冻结协议续到目标50轮，见[三十轮分析](igstgnn_internal_spatial_is11_epoch30_20261011.md)。[十轮](igstgnn_internal_spatial_is11_epoch10_20261011.md)与[首轮分析](igstgnn_internal_spatial_is11_first_epoch_20261011.md)保留历史结果；当前不是确认性预测增益。

本版落实用户《IGSTGNN_IS1.0_修改建议与执行清单_20261010.md》的 R1—R5。[IS1.0](igstgnn_internal_spatial_interface_plan_20261010.md) 保留为设计来源，读出和执行状态以本文件为准。

## 1. 内部机制

新路径：合法历史及截止点时钟 → 历史限制先验 → 报告采用与状态更新 → 边交换上限 → 各层空间交换 → 原预测头。没有外部预测残差、M4辅助损失或容量伪标签。

共享状态模块计算：

`a_prior = sigmoid(A(history_summary, cutoff_clock))`

`r = report_mask * sigmoid(G(history_summary, report_embedding, distances, age))`

`a = a_prior + r * (sigmoid(U(...)) - a_prior)`

`c = 0.001 + 0.999 * (1-a)`；内部边 `C_ij=min(c_i,c_j)`，开放边取相连节点的容量。

每次 forward 只更新一次报告状态。先验和开放入口/出口提议只读取 ICSF 前的历史，不读取报告。当前报告仍按既有离线可见性假设使用，不新认证为在线首报。

各层复用原窗口表示，编码4维有界初态。共同提议网络读取两端隐藏特征、当前状态、容量、发送和接收信息；没有报告向量、距离或掩码的独立旁路。每次空间调用执行两次子步，eta=0.5，不对应真实分钟。

B2使用 f=q 后裁剪状态。B3依次执行边容量限制、共享发送、共享接收，再同时更新；不通过clip保持范围。内部出边与出口竞争发送，内部入边与入口竞争接收；受阻余量不自动重分配。

R1 的读出改为：

`z = z_stat + W[ x_final-x_initial, mean_actual_inflow, mean_actual_outflow ]`

W为12→32、无bias、非零Xavier初始化。等量进出时即使位移为零，仍可读出吞吐变化。B3中位移与净流存在代数冗余，当前用共同读出控制对照，不额外引入第五组。相较4→32，每层增加256参数，五层1280。

原统计图、estimation gate、InhBlock、TIID、输出头保留。最后一帧LayerNorm在所有组保留；B2/B3删除失活的ICSF Q/V与融合MLP，K继续供原TIID。五层历史空间计算及DifForecast预测调用全部接入：每个forward共20次调用，每次两子步。新算子只约束自身潜交换，不约束原统计图消息。

## 2. 四组与固定输入

| 组 | 空间机制 | 总参数 | 新模块参数 |
|---|---|---:|---:|
| B0 | 原ICSF | 443645 | 0 |
| B1 | 原ICSF注入乘2*sigmoid(g)，端到端训练 | 445278 | 1633 |
| B2 | 报告更新状态＋同信息普通交换 | 458360 | 17148 |
| B3 | 与B2同参数同信息；改为受限交换 | 458360 | 17148 |

新模块参数不等于总参数差，因为B2/B3删除了旧ICSF分支。优化器包含全部当前模型参数，不调用冻结主干的旧attach_gate。

保持3604/917训练/验证窗口、496站、12时距，原flow、时钟和单触发报告；X=槽0:12，槽12:14排除，Y=槽14:26。训练目标为原单位pooled MAE，选模为全496站horizon-macro MAE。test不读取。

协议：batch48，Adam lr0.002、weight_decay1e-5、eps1e-8，梯度裁剪5；milestones=[1,38,46,54,62,70,80]、gamma0.5，保留第1轮后衰减。seed2025，上限100轮、patience20、严格改善并列取较早轮；按1→10→30绝对轮数推进。

B0/B1初始预测严格相同。四组匹配主干初值相同；B2/B3参数名称、形状和初值完全一致。每个epoch/batch重设共同dropout种子，新模块无dropout；不同算子的初始预测允许不同。

固定图随代码提供，无需旧M4包：候选覆盖476/496站、456条内部边、266条开放边；联合物理认证节点0。开放边保留来源依据；20个图外节点不补造边，仍参与主预测。固定评价14个道路/方向组。

B3/B2为同信息算子比较；B3/B0同时改变内部设计和候选道路信息。四组都保留TIID，不能据此声称报告纯信息增量或真实交通守恒。

## 3. 记录、恢复与判读

每组独立子进程、Adam、scheduler和checkpoint。按epoch原子保存，完成组自动跳过；中途死亡只重跑该组未保存的一轮。父进程存活时记录子进程退出码；-9不自动归因为显存不足。作业整体被杀时，以最后完整checkpoint为恢复边界。

last.pt保存最新模型、优化器、scheduler、最佳模型状态best_model_state、历史和RNG策略。恢复核对数据、图、来源代码、协议、种子、batch、Torch/CUDA等身份。GPU名称另记审计，兼容环境下SXM2/PCIE名称改变允许续跑；不承诺跨硬件位级一致。同run并发启动由系统锁拒绝，进程死亡自动释放。

输出包括：

- 全网、原报告关联、固定候选组件可达包络、图外节点、14道路方向、12时距及有效目标数；空分组为unavailable。可达包络不是预测实际响应或因果影响范围。
- 逐窗口/时距绝对误差、有效数、sample_index及t0，供后续按日期/周配对分析；训练不保留全部预测。
- 采用强度、报告状态更新、容量及其相对历史先验的变化。
- 各层/调用/子步的状态范围、位移、B2裁剪比例与截断量、预算激活/超限、float32及独立float64潜收支误差。
- 内部道路、开放入口、开放出口交换分开记录，边界主导响应不能证明内部道路传播。
- 首个训练batch新增参数和位移/入流/出流读出梯度。B2状态梯度可能被裁剪阻断，通量读出有梯度不等于状态路径稳定。

B2/B3提议网络都读取C，完整报告响应不能全归为硬容量限制。合成验收另固定高提议、关闭原TIID，确认只有受限算子的容量路径仍能影响主预测并获得报告梯度。

先查看B2训练/验证裁剪诊断，再比较MAE。无事后挑选的可接受裁剪率或收益阈值。10轮不淘汰B2；B3继续则保留B0/B2，同预算完成必要B1比较。报告最佳值、共同epoch、末5轮趋势，收益=control_MAE−candidate_MAE。

若值得扩展，先做从头初始化的空间报告关闭重训练，再考虑3个配对种子。模型开关spatial_report_enabled=False保持截止点时钟和原TIID，tiid_report_enabled=False另关TIID。空间OFF也移除U/G的历史处理，是路径消融；同权重推理开关不能替代重训练。条件扩展的训练协议和正式checkpoint干预器尚未启动。

## 4. 服务器运行

先单独更新代码；Git不在Bash脚本中：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

登录节点用Bash提交完整首轮：

```bash
bash experiments/chronological/run_internal_spatial.sh --epochs 1
```

未在作业内时自动sbatch，已有Slurm GPU allocation则直接运行。固定正确工作目录，不用--mem，不含Git、安装或下载。默认现有igstgnn Python、1GPU/3CPU/6小时；不同路径可用IS11_REPO_DIR和PYTHON_BIN环境变量覆盖。

已有GPU allocation也可只运行Python：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/train_internal_spatial.py --epochs 1
```

开头打印Run directory，形如experiments/chronological_runs/contra_is11_时间_随机后缀。默认定位此前../data/chronological/Contra_Costa_v8_dev；如放别处，加--data-dir /实际原数据包目录，不传旧M4 run/history目录。

四组已完成30轮。回读已保存分组/机制后，用同一个实际run续到目标累计50轮：

```bash
bash experiments/chronological/run_internal_spatial.sh \
  --run-dir experiments/chronological_runs/contra_is11_20261011_163218_259a3f --epochs 50
```

--epochs 50是绝对累计目标，本次每组计划新增20轮，保留原patience20早停、上限100及第38/46轮后的学习率下降；全部完成的轮内时间粗估约103分钟。50是执行/回读端点，不改变最终停止规则。不要加--check；它只取首4条训练/验证、batch2，每组两次更新，身份与完整训练隔离。

## 5. 独立的可选数据审查

audit_internal_spatial_optional_data.py只读取既有多通道训练X和元数据，不读目标、验证或test，不改变首轮输入。

本地回读_01包：3604×12×496×3，训练X哈希与物化记录一致。三通道显式非有限/负值缺失率均0，不能证明来源未插补。flow/occupancy/speed顺序仍依据文献和数值范围推断；单位、独立实体、在线报告版本未新认证。后续另立同结构信息实验，本轮不只给B3添加新通道。

## 6. 验收边界

已验收单链、合流、分岔、开放/闭合收支、空图、节点/边重排、零分母、报告读取权限、20次空间覆盖、强制容量路径主预测梯度及保存恢复一致性。真实首4条训练/验证、每组两次更新已在本地CUDA完成，工程数据和检查点保存在独立check目录。

本地Python3.10、Torch2.8/CUDA12.6；服务器沿用既有环境，作业1173577四组累计30轮已回传成功。该标准输出未附Torch/CUDA/GPU型号身份文件，未另行认证环境指纹。使用Torch已有基础算子，无新增pip依赖。没有按本地小样本MAE选模块或淘汰组；服务器各阶段耗时和指标见独立分析。

相关核心检查共67项，65项通过、2项旧Bash平台测试跳过。另检查旧v12c适配器测试发现其context缺少incident_embedding，已在开发前提交bde35ee上复现同一KeyError；该冻结实验接口保持原状。旧综合workflow在此Windows环境还出现OpenMP重复加载，未将其计入通过项。新B1使用独立端到端接口，已验证与原结构初始预测严格相同。对开发前源码的原模型有/无报告前向复核差值均为0。
