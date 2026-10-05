# v13a：有效输入清单与单事故 ICSF 等价核验

日期：2026-10-05。代码、本地验证及日志记录的真实冻结 A／V100 全 fit 核验已完成。
服务器 1520 个窗口的 9,047,040 个预测值逐值完全一致，最大差为 0；18 项服务器测试通过，退出码 0。
已完成日志与本地输入清单核对，尚未收到正式服务器 summary/CSV；详见[服务器结果与证据边界](effective_input_audit_v13a_results.md)。
本阶段回答“哪些输入实际进入模型、哪些计算可以等价简化”，不计算预测误差或训练新模型。
下一阶段的公平重新训练已实现，见[最小信息消融设计](minimal_information_ablation_v13b_draft.md)与[V100 运行说明](minimal_information_v13b_run.md)。

## 冻结范围

协议：`experiments/chronological/effective_input_audit_v13a.json`。
SHA-256：`71aa29cde966505b4359b406cbc9f9db6826c95dc6bd9d52f08f4043a043d0ef`。
该协议在本轮拟合期统计与实际核验之前写入并固定，沿用已知的 1520 个 fit 窗口，未用后段结果挑样本。

- 输入仍为 Contra496、v8、2023 train 包。按报告 t0 和完整支持区间，保留
  `[2023-01-02, 2023-04-24)` 的 1520 个窗口；支持结束允许恰好等于右边界。
- 统计只计算这些窗口的 X 前 12 槽与报告上下文。保存完整的时间资格清单和被实际测量的 sample ID。
- 原始 `train_flow.npy` 的文件哈希会遍历包括 Y 在内的全部字节；程序不解析 Y 做统计、损失或选择。
  train manifest 与 context 的身份元数据覆盖整个 train；不打开 val/test 文件。
- 旧 scaler 与 A 具有更广泛时间依赖，只用于复现原 A 输入/输出，不能充当下一轮过去信息训练的认证。
- 原生产模型、参数格式和 checkpoint 保留，等价实现通过临时独立 wrapper 调用，结束或异常时恢复原模块。

## 实现内容

`audit_effective_inputs.py` 的输入视图直接切取 `flow[indices, :12, :]`，不复用会同时生成 Y 的训练 loader。
它核对 sample/station/scaler 身份、单报告形状、报告年龄、跨日时钟、X/Y 名义时间和完整支持范围。

清单输出缺失/非有限/负值、真实零、取值范围、均值/标准差与是否恒定，分别展示距离全节点值和有支持值。
流量按“窗口中出现的输入单元”加权，重叠时段可能重复计数；这些统计不是重新拟合的去重 scaler。
精确唯一值数最多保存 4096 个，超过后输出下界，不把截断值冒充完整基数。

字段表同时说明实际消费者与可用性：预测时钟是已知日历信息，报告年龄仅为报告后 1–5 分钟的槽内偏移；
位置首报可得性仍为条件假设。静态属性、类型/文本/最终时长和额外交通通道均标明没有进入当前路径。

`SingleIncidentICSF` 仅在 fixed / report_location_v1 / 无静态传感器属性 / FP32 eval inference 下工作。
它直接计算 `LayerNorm(H_last + mask × V)`，保留原 report encoder、K、V 和全部节点的 LayerNorm，
保留 TIID、构图及解码；只跳过 Q、ICSF 融合 MLP 和单元素 softmax。
逐批比较增强历史、K、空传感器表示、距离、最终预测，并记录最大绝对差及是否逐值完全相同。
容差为标准化输出绝对 1e-6、相对 1e-6；超限或非有限值均失败。

脚本核对前后全部参数/buffer 哈希不变、没有梯度或优化步骤。本轮只验证推理等价；
训练时跳过 dropout 会改变 RNG 消耗，所以 wrapper 明确拒绝训练/启用梯度调用。
跳过计算没有从 checkpoint 删除参数，也没有测量加速，不能据参数数量宣称速度收益。

## 本地结果与边界

已在 `igstgnn-audit`（Python 3.10、NumPy 1.24.4、PyTorch 2.3.1+cpu）完成全 fit 输入清单：

| 项目 | 结果 |
|---|---:|
| 预测窗口 / 传感器 | 1520 / 496 |
| 历史流量单元 | 9,047,040 |
| 非有限值 / 负值 | 0 / 0 |
| 真实零流量单元 | 34,510，约 0.38145% |
| 流量范围 / 唯一值数 | 0–993 / 947 |
| 报告年龄 | 1–5 分钟，共 5 种值 |
| 距离第一列 D0 | 全部为零 |
| 每窗口候选节点数 | 最少 8，最多 58，平均 36.6316 |

这只描述本包中该 fit 子集的输入，不能证明源传感器无缺测、Y 完整或其他年份同样如此。
当前结果不支持优先增加复杂插补模块，也不证明报告年龄/距离的预测价值。

另用前两条真实 fit X、496 节点生产架构和明确标记的随机权重，完成 CPU 等价工程检查。
11,904 个预测单元及全部被比较的中间张量完全一致，最大绝对差为 0；状态哈希不变。
该架构被跳过的 Q/融合 MLP 共 1409 个参数，仍保存在原模型中。状态为
`ENGINEERING_CHECK_PASS_RANDOM_WEIGHTS`，不能当作服务器冻结 A 或 V100 的核验结果。

新增 18 项测试覆盖数据隔离、真实零与缺失、跨日、完整支持、来源哈希、恒定/全缺失、
微型真实骨干等价、零/混合/全支持、错误简化拒绝、异常恢复、CUDA 路由与启动失败传播。
另通过原机制 12 项与 chronological 数据处理 6 项回归，共 36 项不同测试通过。
Bash 语法、Python 编译与差异空白检查通过。

本地已有文件，无需压缩：

- [全 fit 清单 summary](../../复现结果/模块输入核验_20261005/fit_inventory_02/summary.json)
- [字段清单 CSV](../../复现结果/模块输入核验_20261005/fit_inventory_02/field_inventory.csv)
- [各站输入质量 CSV](../../复现结果/模块输入核验_20261005/fit_inventory_02/node_input_quality.csv)
- [随机权重工程检查](../../复现结果/模块输入核验_20261005/random_weight_cpu_check_02/summary.json)

`_01` 保留为增加“输入视图拒绝直接访问非 fit 行”约束前的执行记录；上述 `_02` 对应最终交付代码。

## 服务器运行

代码提交到 GitHub 后，在平台分配的 V100 任务终端执行；沿用 igstgnn，不连接 SSH。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
git pull --ff-only origin research/chronological-tiid
bash experiments/chronological/run_effective_input_audit.sh run contra_v13a_inputs_01
```

启动器前台运行，先做测试和两窗口冻结 A 检查，再做 1520 窗口的完整等价核验。
**这里会使用 GPU 做原模型/简化模型推理，优化器步数为 0**：这是预定核验任务，并非 CUDA 未启用。
2026-10-05 作业 1148424 的完整流程约 38 秒，最后全量进度为 25.018 秒；后者不等于纯推理延迟。
全量 CUDA 分配峰值约 1.18 GiB，未做加速基准。以后重跑时保持平台作业存活，进度会记录设备和完成窗口数。

默认复用已有文件：

```text
../data/chronological/Contra_Costa_v8_dev
experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
```

无需旧 v12o 输出、匹配对照包或下载数据。如服务器路径不同，可设置 `V13A_DATA_DIR` 和
`V13A_CHECKPOINT`；冻结哈希仍必须相同。`V13A_DEVICE=cpu` 仅用于明确的 CPU 工程环境，默认是 cuda:0。

在同一运行名上查看状态/报告：

```bash
bash experiments/chronological/run_effective_input_audit.sh status contra_v13a_inputs_01
bash experiments/chronological/run_effective_input_audit.sh report contra_v13a_inputs_01
```

成功状态为 `FIT_INPUT_AND_EQUIVALENCE_COMPLETE`。请直接回传已有的：

```text
experiments/chronological_runs/contra_v13a_inputs_01/summary.json
experiments/chronological_runs/contra_v13a_inputs_01/field_inventory.csv
```

结果目录还包括 `node_input_quality.csv`、`eligibility.csv` 和 `measured_sample_ids.json`；
`.job/run.log` 保存日志、主机、Slurm ID 与退出码。失败保留 `.partial/failure.json`，
重跑使用 `_02`，不覆盖旧结果。仅需 CPU 清单时使用 `inventory` 子命令及新运行名；仅工程检查用 `check`。
