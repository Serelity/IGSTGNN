# B3-R V1.1：代码、验收与服务器运行

日期：2026-10-12。首版交付P/D模型、公共初始化、诊断、数据时间清单及独立训练/恢复入口。旧B0—B3模型、训练源码和JSON协议未修改。开发阶段只使用train/validation；服务器完整首轮及预测收益仍待验证。

## 这次实际改变了什么

| 部件 | 实现与范围 |
| --- | --- |
| 容量更新 | [独立模型模块](../src/models/incident_spatial_residual.py)：P产生历史容量logit，D产生报告条件修正，`c=.001+.999*sigmoid(P+m*D)`。D仅末层置零，无采用门G、tanh、新正则或新报告字段。 |
| 容量接口 | 节点容量直接映射到边；内部边取两端最小值，入口/出口取真实端点。不会再次调用旧`state_capacity`把容量当成状态。 |
| 保留部分 | 深复制同一规范未训练B3的主干、TIID、边界、交换层和缓冲区。P复制A隐藏层、取反A末层；D复制U隐藏层、末层归零。优化器在此之后建立。 |
| 交换 | 保留4通道、2子步、eta=.5、84维提议输入、硬边上限、共享收发预算及12→32读出；没有折叠接口或重新分配余量。 |
| 参数 | 全496站B3为458,360，B3-R为455,780，减少2,580；保留模块联合训练。 |
| OFF | 跳过整个D，容量回到历史基线；原TIID继续开启。原始delta标记未计算，有效修正为零。该比较评估整条D路径，不隔离纯报告信息。 |

[训练入口](../experiments/chronological/train_b3_r.py)在独立模块命名空间中复用原IS1.1的worker、epoch循环、优化器、指标和检查点函数，仅替换模型构造、诊断及所选组。不会修改原训练模块的全局变量，也没有复制另一套优化流程。默认仅训练B3/B3R；需要时可选择B3R_OFF。

## 验收与证据

复现入口为[工程核验脚本](verification/verify_b3_r_implementation_20261012.py)，本地CPU结果见[工程记录JSON](verification/igstgnn_b3_r_implementation_20261012.json)，24项测试通过。包含：

- 新模型及旧模型回归检查；合成数据连续2轮与1轮后恢复到2轮的模型、优化器、scheduler完全一致；重复已完成请求不改检查点。
- 实际完整IGSTGNN的受控MAE正例：报告端是容量瓶颈、收发预算有余量，首次D末层有任务梯度、隐藏层为零；更新并重新前向后隐藏层获得梯度。另保留预算饱和、非瓶颈端点的零响应反例。
- 真实496站数据仅取首4条train及首4条validation，各组完成2次更新；核对实际参数量、公共初始化、日志和可读检查点。这些小批量MAE不用于判断结构优劣。

[新诊断](../experiments/chronological/b3_r_diagnostics.py)每次完整前向统计一次logit/容量，并分别给出全图与报告关联条件的分母；交换统计维持5层×4调用×2子步，共40行。内部边瓶颈统计按报告端点/通道计数，两端近似相等及开放边分开记录。容量截断比例不等于最终交换敏感性。

初始化证书包含保留参数/持久缓冲区的名称、形状、dtype及哈希。CPU上每次重建并核对初始ON/OFF完全一致、新历史容量与规范B3空间OFF在容差内对应。数值误差本身通过容差验收，不把浮点误差末位写进严格身份，以支持不同服务器主机续跑。

## 服务器先运行这一项

在服务器仓库目录更新代码；Git只在下面的独立指令中，Bash脚本不执行Git：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
bash experiments/chronological/run_b3_r.sh --check --arms B3 B3R B3R_OFF
```

检查会先运行新模块测试，再对三组执行真实数据小批量训练，预期末尾为`B3R_PAIRED_STAGE_COMPLETE`。这会创建新的`contra_b3r_check_*`目录；它不是正式首轮目录。

脚本在登录节点自动提交sbatch，在已有Slurm allocation中直接运行。沿用1 GPU、3 CPU、6小时，不指定`--mem`；Python固定使用既有igstgnn环境。可通过`B3R_REPO_DIR`和`PYTHON_BIN`覆盖路径。

检查通过后，新建正式开发运行，默认只比较B3与B3-R：

```bash
bash experiments/chronological/run_b3_r.sh \
  --run-dir experiments/chronological_runs/contra_b3r_v11_seed2025 \
  --epochs 1
```

从同一目录续到累计10轮：

```bash
bash experiments/chronological/run_b3_r.sh \
  --run-dir experiments/chronological_runs/contra_b3r_v11_seed2025 \
  --epochs 10
```

`--epochs`是累计目标轮次。中断后重跑同一指令，已完成组/轮次自动跳过，从各自`last.pt`恢复。旧四组继续使用原`run_internal_spatial.sh`及原目录，不把旧B3检查点传给新入口。

空间OFF从头对照应使用独立目录和固定组选择，例如：

```bash
bash experiments/chronological/run_b3_r.sh \
  --run-dir experiments/chronological_runs/contra_b3r_path_v11_seed2025 \
  --arms B3R B3R_OFF --epochs 1
```

之后续跑必须保留同一`--arms B3R B3R_OFF`。改变组、seed、源码、模式或协议将被身份检查拒绝，不能将ON优化器恢复到OFF。这里的ON/OFF对照也是两组从头训练；服务器首轮未确认前不自动扩展全部组合。

## 产物、恢复与研究边界

| 路径（相对运行目录） | 内容 |
| --- | --- |
| `identity.json`、`initialization.json` | 数据/图/新旧源码指纹、映射版本、模式、训练参数范围、诊断schema及初值证书。 |
| `B3/`、`B3R/`、可选`B3R_OFF/` | `last.pt`内含当前及最佳模型、优化器、scheduler、epoch和随机状态；另有history、summary及逐窗口误差。 |
| `session_*/data_time_event_audit.json` | 时间/事件审查，将已确认、假设、缺失和冲突分开。 |
| `session_*/environment*.json` | 设备及环境变化；V100型号变化作为审计字段，不单独阻止续跑。PyTorch/CUDA/设备类型等仍按原身份规则核对。 |
| `session_*/*_process_result.json` | 各独立子进程退出码；失败时保留之前完整epoch检查点和日志。 |
| `latest_comparison.json` | 共同轮次、共同前缀最佳值、各自最佳值及共同末5轮；收益符号为参照MAE减候选MAE。 |

每组独立子进程顺序运行，退出即释放其模型与优化器资源。保留原100轮上限、patience20及学习率计划；10轮只观察优化与机制，不因当时排名落后自动淘汰。

[时间审查器](../experiments/chronological/audit_b3_r_development.py)核对开发清单的窗口/年龄、训练目标与验证截止点的名义顺序、共享目标时槽和事件ID交集。[本地清单记录](verification/igstgnn_b3_r_data_time_event_audit_20261012.json)未发现跨集合共享目标时槽或事件ID；最晚训练目标为2023-08-31 21:35，最早验证截止点为2023-09-01 01:55。报告版本、首报时间语义、数据发布延迟及事件来源认证仍缺失，不标为在线时间审查通过。有实际窗口/跨集合目标冲突时保存审查并停止，不静默更改旧数据。所有结果仍限定为条件离线开发，候选图尚未认证为真实物理图。

后续待服务器完成完整首轮并回传日志，再进行10轮观察和共同预算比较。选中检查点的ON/OFF/ON敏感性探针及多种子复核在后续阶段交付；当前工程结果不代表预测收益、事故因果效果或真实容量辨识。
