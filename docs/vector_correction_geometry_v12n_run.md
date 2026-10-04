# v12n 交付与服务器运行

**更新：2026-10-04。** 本项目采用本地开发并提交 GitHub、用户在校园网服务器拉取运行、
用户回传结果的流程。服务器使用 **`igstgnn` 环境、V100 显卡**，由用户本次确认。
整个流程不使用 SSH，也不尝试由本地直接连接服务器。

研究定义见 [v12n 方案](vector_correction_geometry_v12n_plan.md)。冻结 JSON 的
`DESIGN_READY_IMPLEMENTATION_PENDING` 是设计时的快照字段，保留原文件及 SHA-256；
它不代表当前实现进度。正式 V100 诊断已在 2026-10-04 完成，退出码 0；
主口径 G 六端点均负、S 三负三正，详见 [服务器结果分析](vector_correction_geometry_v12n_results.md)。
以下保留运行与恢复方法，已经完成的同一诊断无需重复执行。

## 运行前保留的来源

服务器应已有完成的 `experiments/chronological_runs/contra_v12m_output_scope_01`。
程序只接受原 v12m 运行、原协议、原 6 个 P 端点及原数据指纹；重命名目录不改变身份要求。
既有 v12m 的 checkpoint、两份冻结清单、JSON、逐窗口 NPZ 和 weekly NPZ 均需保留。
还需要原 A 的 `best_model.pt` 及原数据包，默认路径与 v12m 启动器相同。

本轮不重新训练 v12m。它从已选权重导出候选早期 A、Y、d、valid 和坐标，核对原误差后
计算 G/S/O、六类贡献、匹配差值及预定周敏感性。只计算 P 的目标误差，不计算 U(eP) 的误差。

## V100 上启动

在平台分配的计算会话内打开终端。以下路径沿用此前成功运行的服务器仓库：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid || exit 1
conda activate igstgnn || exit 1
git log -1 --format='%H %s'
bash experiments/chronological/run_vector_correction_geometry.sh run contra_v12n_geometry_01
```

若平台已经激活 `igstgnn`，可省略激活命令。若原目录不同，先进入实际仓库目录。
不要在本轮运行中拉取代码或修改环境、协议、数据及源产物。
这是前台工作流，依次完成设备检查、v12n 测试、来源预检、冻结小包回放和正式诊断。
启动日志会打印 Python、Torch/NumPy、实际 GPU 型号、commit、主机和 Slurm job。

仅预检来源，或仅做小包检查：

```bash
bash experiments/chronological/run_vector_correction_geometry.sh preflight contra_v12n_geometry_preflight_01
bash experiments/chronological/run_vector_correction_geometry.sh check contra_v12n_geometry_check_01
```

`preflight` 核对输入与既有输出，不做模型推理、不创建实验结果目录。
`check` 使用首个种子、两个架构、各 cohort 原首个不超过 16 窗口批次，保持原推理 batch size。
它没有训练步骤，输出明确为工程检查，不作科学判定。

如果来源目录使用不同名字，可指定第三个参数：

```bash
bash experiments/chronological/run_vector_correction_geometry.sh run contra_v12n_geometry_02 contra_v12m_output_scope_01
```

不同名字仍须指向原冻结来源，不能替换成另一次训练或工程小包。已有 final、partial 或 job
目录均拒绝覆盖；中断后保留原目录，按下文判断能否直接恢复统计。

默认设备为 `cuda:0`。数据被搬到其他目录时，使用 `V12N_DATA_DIR`、`V12N_PRIMARY_DIR`、
`V12N_SECONDARY_DIR`、`V12N_CHECKPOINT` 指定新位置，指纹仍必须一致；无需改源码。
`V12N_DEVICE=cpu` 仅供确有需要的本地/CPU 检查，不根据本地小包耗时推算完整 V100 运行时间。

## 状态与结果回传

另开终端查看日志；退出 `tail` 不影响原前台任务：

```bash
tail -f experiments/chronological_runs/contra_v12n_geometry_01.job/run.log
```

任务结束后：

```bash
bash experiments/chronological/run_vector_correction_geometry.sh status contra_v12n_geometry_01
bash experiments/chronological/run_vector_correction_geometry.sh report contra_v12n_geometry_01
```

成功运行应同时满足工作流退出码 `0`，正式状态
`VECTOR_CORRECTION_GEOMETRY_AUDIT_COMPLETE`，以及
`scientific_status=FIXED_SAMPLE_DIAGNOSTIC_REQUIRES_REVIEW`。
工程小包的 `ENGINEERING_CHECK_PASS` 不能替代这个正式状态。
正式完成表示核算完成，不表示模型改善或允许继续训练。

回传 `report` 的完整文本，并通过平台下载以下小文件：

- `summary.json`、`report.md`：全部固定端点及信息边界；
- `geometry_metrics.csv`、`category_contributions.csv`：全部阶段/群体、两种加权、G/S/O 与贡献；
- `matched_contrasts.csv`、`matched_week_sensitivity.csv`：直接差值、有效支持与删周敏感性；
- `replay_checks.csv`、`signed_export_manifest.json`、`input_manifest.json`：来源与回放验证。

保留大型 `signed_cells/`、`window_geometry/` 和原 checkpoint，无需通过 Git 上传。
若失败，回传工作流退出码、日志末尾 traceback 及 `.partial/failure.json`，不删除现场。

## 统计阶段中断后恢复

只有 `.partial/signed_export_manifest.json` 已完整写出并标记 `SIGNED_EXPORT_COMPLETE`，
才允许恢复统计。该文件表示所有预定端点的导出已齐全并通过回放核验；程序会重查每个文件哈希、
样本预算、端点集合、冻结协议及实现代码。缺少它时，不使用半成品推断完整结果。

代码与协议不变时，可在同一 `igstgnn` 环境从只读导出生成新结果目录：

```bash
python experiments/chronological/audit_vector_correction_geometry.py analyze \
  --export-dir experiments/chronological_runs/contra_v12n_geometry_01.partial \
  --output experiments/chronological_runs/contra_v12n_geometry_stats_02
```

这一入口只使用 NumPy，不加载模型、不重新推理或训练。完整结果目录也可作为统计重放来源。
输出中的 `new_model_inference_this_invocation=false` 与原导出的推理记录分开保存。
直接 Python 入口不生成 shell 的 `.job` 文件；可查看新目录中的 `progress.json`、`summary.json`
和 `report.md`，或执行：

```bash
python experiments/chronological/audit_vector_correction_geometry.py report \
  experiments/chronological_runs/contra_v12n_geometry_stats_02/summary.json
```

若冻结推理阶段中断、没有完整导出清单，使用新运行名重新做冻结回放。
这不是训练断点续跑，不读取优化器或最后一轮状态。修复代码后也不能绕过旧导出的实现身份核验。

## 本地交付验证

实际检查使用 WSL `igstgnn-audit` 的 PyTorch `2.3.1+cpu`、NumPy `1.24.4`；
这与服务器的 `igstgnn`/V100 运行是两个独立环境。

v12n 新增测试覆盖数学边界、配对缺失支持、共同抽样、来源及 checkpoint 调包、
禁止优化器/反向传播、目标变化不影响预测、模型状态不变、统计恢复、启动失败码及目录保护。
相关旧 v12m/v12k 测试用于回归。最终 **30 项 v12n、45 项 v12m、24 项 v12k 测试全部通过，
共 99 项**；日志指纹登记在本地 `delivery_validation.json` 中。

真实 496 站工程小包已完成两个架构、一个种子的冻结回放，9 次 A 前向、18 次适配器前向，
导出 18 份 signed NPZ，发布 45 个带哈希的产物。逐窗口区域误差、pooled MAE 与 G(1)
相对原小包的最大回放差均为 **0**。统计完成事件约在 41.14 秒，进程峰值 RSS 约 686 MiB。
这只是小包工程验证；在该本地交付时点，正式 6 端点尚未运行，也没有新的方向/幅度科研结论。
后续正式服务器结果独立记录于文首链接，不与小包指标混用。

对上述真实导出还执行了独立统计重放，运行时禁止导入 PyTorch；全部端点统计、
匹配差值、类别贡献和模型身份与原结果一致，保持工程检查状态，未产生新的模型推理。

首次检查曾因将新建适配器的随机初值与原训练初始化比较而失败。现在从已认证来源核对
原初始化身份，新建模块只提供键/形状约束，随后严格加载原已选状态；失败现场与修复后
小包均保留。原 v12m 的训练/选择代码及 v12n 冻结协议没有改变。

本地证据保存在 `复现结果/修正方向幅度诊断_20261003/`，不随源代码推送实验数组。
