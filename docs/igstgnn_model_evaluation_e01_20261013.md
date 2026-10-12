# IGSTGNN 模型评价 E0/E1：现有结果核对与多角度分析

日期：2026-10-13。

本工具落实[完整模型评估与改进决策方案 V1.0](igstgnn_model_evaluation_and_improvement_plan_v1_20261013.md)的 E0/E1。它只读取已有训练产物，使用 CPU 分析 JSON/NPZ；不加载 checkpoint、不调用模型、不新增推理或训练、不读取 test。训练中的模型可以继续运行，评价工具不修改原 run 的身份、历史和 checkpoint。

## 1. 本次回答的问题

1. 四个 G1.1 对照在同一训练轮数下是否具备可比性？缺少哪些证据？
2. 全网 MAE、RMSE、各时距与各区域/道路误差是否给出一致判断？
3. 相同验证窗口中，哪些改善、哪些恶化，严重失败是否减少？
4. 已保存的门控与 D 路径诊断能否定位值得进一步检验的问题？

全网 MAE 仍是既定选模指标。所有次级指标用于描述作用条件、风险和后续模块假设，不据此事后更换选模规则。

## 2. 服务器更新

先在仓库目录单独执行更新指令；运行脚本不包含 Git 操作。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only
```

## 3. 优先评估四组共同完成的第 10 轮

```bash
bash experiments/chronological/run_model_evaluation.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --control B3R \
  --epoch 10
```

`--epoch 10` 固定共同预算，可以在原任务继续训练时评价已经完成的第 10 轮。主比较为 B3R 与同 run 中的 HALF、GATE、NO_D 对照，以原产物中的 arm 名称为准。不同组不各自挑选不同轮数的困难窗口。

不传 `--epoch` 时，工具使用当前主 run 各组共同完成的最小轮数；为了结果便于重复核对，建议在正式回传时显式指定轮数。`--output-dir` 可以指定一个新的独立结果目录；默认会在 run 下创建独立评价目录。

脚本沿用服务器 Python 环境，限制 BLAS/OMP 为单线程。可以在已有作业分配中或允许执行轻量 CPU 分析的节点上直接运行；脚本不会申请 GPU、提交 Slurm 作业或设置 `--mem`。若站点要求所有 CPU 工作均通过调度器执行，按站点规定进入已有分配后运行同一命令。

需要覆盖路径时，可以在调用前设置：

```bash
B3R_REPO_DIR=/path/to/IGSTGNN-code \
PYTHON_BIN=/path/to/python \
bash /path/to/IGSTGNN-code/experiments/chronological/run_model_evaluation.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --epoch 10
```

也可直接使用 Python：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/analyze_model_evaluation.py \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --control B3R \
  --epoch 10
```

## 4. 可选：登记原模型和受限交换对照

原 IGSTGNN/B0 是判断整体改进是否有价值的必要参照。若原 IS1.1 的结果目录仍完整，可增加：

```bash
bash experiments/chronological/run_model_evaluation.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --control B3R \
  --epoch 10 \
  --reference-run-dir experiments/chronological_runs/contra_is11_20261011_163218_259a3f \
  --reference-arms B0 B2 B3
```

这个目录只是历史上记录的候选参照，不能预先保证当前文件完整或训练身份相容。工具先核对已记录的数据身份、共有骨架源码指纹、协议、训练预算、种子、图和环境等证据，以及保存的初始化证书与公共参数初始哈希，再报告比较资格及缺失字段。初始化核验依据保存的证书，不重新构造模型或执行初始化。只有通过相应可比性核对的结果，才能用于模型设计的比较；跨版本缺失证据不能被静默当作一致。

未提供基准结果时，报告仅解释当前四组之间的差异，不能宣布 G1.1 优于原 IGSTGNN。

## 5. 读取什么、生成什么

输入来自已有 `identity.json`、`initialization.json`、各 arm 的 `history.json` 和 `summary.json`，以及指定轮次的 `validation_windows_epoch_NNN.npz` 等已保存产物。本版没有读取时间审计文件，不扩展原有时间可见性结论。评价工具不反序列化 `last.pt` 或 `best.pt`。checkpoint 选择信息来自已保存的历史和摘要，并明确记录无法核实的部分。

输出文件及用途如下；目录路径以控制台和 `summary.json` 为准。

| 输出 | 用途 |
| --- | --- |
| `evaluation_manifest.json` | 评价参数、源文件身份及读取范围 |
| `comparability.json` | 比较资格、冲突字段和缺失证据 |
| `metrics_global.csv` | 全网误差、样本数、注册参数量、已保存轮次耗时与峰值 CUDA 分配字节数 |
| `metrics_horizon.csv` | 12 个时距分别比较 |
| `metrics_region.csv`、`metrics_road.csv` | 全量空间分组与道路结果 |
| `metric_comparisons.csv` | 每个可比模型对的全网、区域、道路组 MAE/RMSE/MAPE 差值；含 macro、pooled 与逐时距结果 |
| `window_statistics.json` | 各模型窗口误差分布、尾部及时间依赖描述 |
| `window_pairs.csv` | 相同窗口的配对收益与伤害 |
| `fixed_difficult_windows.csv` | 由固定控制组定义的困难窗口比较 |
| `mechanism_chain.csv` | 已保存的 alpha、原始 D、实际 D 等诊断及可用性 |
| `checkpoint_selection.csv` | 固定轮数、共同前缀最佳与原选模记录 |
| `training_curves.csv` | 各组全部已完成 history 的训练轨迹；超出共同预算的轮数仅作描述 |
| `data_evidence.csv` | 数据支持范围及不可得信息 |
| `report.md`、`summary.json` | 可阅读报告和机器可读汇总 |

若 `history.json` 等源产物在读取期间发生更新，工具应拒绝把不同快照混成一次评价。等待当前写入结束后重跑同一命令；无需删除 checkpoint、修改原 run 身份或重新训练。

## 6. 如何解读结果

- 主 MAE 为各时距 MAE 的等权平均，不能用窗口 MAE 的简单平均替代；两者分母和问题不同。
- `metric_comparisons.csv` 的收益统一为 `control - candidate`，正值表示候选误差更低。分组有效计数不同、分组缺失或某时距为空时，相应比较标为不可用；MAPE 另核对正目标计数，并以比例值保存。
- `metrics_global.csv` 中的 `registered_parameters`、`epoch_seconds`、`peak_cuda_allocated_bytes` 来自已有摘要/历史，缺失项不补测。注册参数量不等于参与计算或更新的参数量；耗时和峰值显存是原训练记录，不能当作统一硬件、负载和测速协议下的效率基准。
- 窗口分布、尾部、困难集合及窗口胜负比例只纳入全部 12 个时距有效计数均大于 0 的完整窗口。partial 窗口保留在全网主指标和可加收益贡献中，但排除于窗口评分统计；分别报告完整、partial、完全无有效目标及合计排除窗口数。
- 困难窗口由同一固定控制组定义，再在完全相同窗口上比较其他组；不能让各组选各自最差窗口后声称比较了同一批困难样本。
- `report_associated`、`candidate_potential`、`outside_operator` 不是互斥划分，不能将它们的结果直接相加分解全网收益。候选包络也不是真实事故影响标签。
- 窗口 NPZ 保存的是逐窗口、逐时距的绝对误差和有效计数，能计算窗口风险，但不能重建逐站点的高估/低估、点误差尾部或点级正负收益抵消。
- 已保存的机制统计是描述性响应。门控变化不等于报告置信度，潜容量变化不等于真实车辆通行能力变化，训练对照也不等于同 checkpoint 的路径干预。
- 窗口存在时间重叠，917 个窗口不能当作 917 起独立事故；单种子和单验证集的结果不构成独立重复实验。
- 缺失的 alpha/D 诊断、信息版本、真实影响标签、事件 ID 等应标为不可用；不自动补造或以零代替。

本版输出 Markdown、CSV 和 JSON，不提供图表、新模型的 ON/OFF 推理、逐点收益归因、信息扰动或最终 test 评估。这些属于后续任务。

本地现有的小规模工程产物虽然覆盖全部 496 站，但只包含 4 个验证窗口，不能代表正式效果评价。仅开发验收时可显式传入 `--allow-engineering`，结果范围为 `engineering_only`。服务器上述正式命令不要加该选项；真实 917 个验证窗口的评价需要在服务器读取正式产物后完成。

## 7. 回传结果与改进决策

完成后优先回传控制台汇总和新评价目录内的 `report.md`；若需要进一步定位，可再回传相应 CSV/JSON。

评价后先记录一个具体问题，例如“某些较远时距的困难窗口改善，但普通窗口恶化”，再提出一个最小可检验假设。后续先用路径干预区分是否依赖该路径，再用独立训练消融验证设计价值。不要仅凭一个次级指标有利就淘汰对照，或同时修改多个模块导致无法归因。
