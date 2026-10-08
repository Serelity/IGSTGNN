# P2 最佳 checkpoint：新增门控分支开启/关闭

作业1163180已完成本步骤：ON与保存预测逐元素一致，模型状态与输入文件未变。
OFF−ON全节点/关联/非关联MAE分别为-0.013984116 / +0.016834267 / -0.016660488。
OFF全节点仍比原fixed高0.074086。本轮初筛未获支持的判断保持。
接续入口为[五组独立逐层关闭](incident_routing_p2_layer_probe.md)，以下保留整组ON/OFF的运行约定。

本步骤已获用户同意。目标是比较同一个训练完成的ACDG模型，在新增事故条件门控残差开启与关闭时
的验证预测差异。不重训练、不更新参数、不改选模结果，不进行系数或层数搜索。

前一阶段作业1163061的保存预测诊断已完成：全节点MAE复算为22.686666（fixed）与22.774736（ACDG）；
共同69/89轮预算内ACDG仍落后；全节点12个时距、9/10周及550/917窗口变差。
这将当前P2版本定位为本次单种子初筛未获支持，尚不能确定分支的即时作用与联合训练轨迹分别有何贡献。

## 固定对照

| 模式 | 使用权重 | 新增条件门控残差 | 其他模块 |
|---|---|---|---|
| ON | acdg/best_model.pt，当前最佳第69轮 | 保留原值 | 保留 |
| OFF | 与ON完全同一份权重 | 五层输出同时置零 | 保留 |
| fixed_saved | 原fixed最佳第99轮的已保存预测 | 原模型无新增分支 | 仅作已有参考，不重新运行 |

使用原917个验证窗口、496站、12个预测步、batch48，以及原有原始流量单位、标签和支持掩码。
OFF通过临时forward hook替换 `incident_condition` 的输出；基础门控、ICSF、TIID、图计算、预测头和
全部参数保持不变。仍会计算条件MLP的候选输出以作统计，但该输出在OFF模式不会进入后续门控。
每次推理在 `eval()` / `inference_mode()` 下运行，不创建优化器。hook即使遇到异常也会移除。

**OFF得到的是ACDG训练过的主干，不是原fixed基线；开关效果也不是重新训练消融的效果。**
五层一起关闭，不能据此归因到某一层。保留事故输入意味着OFF也不是无事故信息模型。

## 启动前与运行中检查

- 核对原训练源码、协议、数据包、两臂身份和完整学习曲线选模记录。
- 核对 `best_model.pt` 的张量状态与 `last_checkpoint.pt` 中保存的 `best_model_state` 完全一致。
- 保留原PyTorch/NumPy/CUDA构建、V100、3线程、batch48、确定性设置。
- ON必须先复现原ACDG最佳预测：最大逐元素绝对差不超过 `1e-5` 原流量单位，
  三个区域的MAE绝对差不超过 `1e-8`。这些阈值在运行前固定；若失败立即停止，OFF不执行。
- 每次推理前后核对完整模型状态哈希；结束时复核原始输入文件、数据包与冻结源码未变。
- 检查五层均处理全部917个窗口、20个batch，并验证无支持节点的条件残差严格为零。

这次新增脚本与测试，不修改被冻结的训练器、模型文件或训练协议。

## 运行

先单独更新仓库：

```text
git pull --ff-only origin research/chronological-tiid
```

在Slurm平台分配1张V100、3核CPU、32GB内存，建议30分钟时限（预留量，并非已测耗时）。
沿用 `igstgnn` 环境，在仓库根目录运行：

```bash
bash experiments/chronological/probe_incident_routing_p2.sh contra_p2_first_epoch_20261008_172731_wNQWZK
```

脚本内没有Git操作，也不调用sbatch。它先执行小型CPU模型单测，再加载服务器现有最佳模型完成
ON/OFF两次验证推理。需要V100是为了在原环境核验ON预测复现；这不是GPU训练任务。
默认数据目录保持 `../data/chronological/Contra_Costa_v8_dev`，可通过 `P2_DATA_DIR` 指定原包位置。

## 输出与解释

每次在配对根目录下新建 `gate_on_off_时间_随机后缀/`，保留 `run.log`、`exit_code`，
`report/` 内包含：

- `probe_config.json`：运行前记录的权重/输入身份、模式和固定容差；
- `on_replay_check.json`：ON复现检查，包括失败时的实际偏差；
- `paired_predictions.npz`：同一顺序下的ON/OFF预测、标签、掩码和样本/节点轴；
- `per_horizon.csv`：全节点、关联、非关联区域的逐时距MAE及差值；
- `gate_statistics.csv`：ON/OFF各层、各区域的logit增量、门值变化和饱和比例；
- `summary.json`：完整汇总、环境和输入输出哈希，成功状态为 `P2_FIXED_CHECKPOINT_GATE_ON_OFF_COMPLETE`。

主要对照量为 `off_minus_on_mae = MAE(OFF) - MAE(ON)`：

- 正值：在这份已训练权重下，新增分支开启使误差更低；即便如此，整体ACDG仍可能不如fixed。
- 负值：关闭分支使误差更低，支持其当前即时作用不利；不等于已找到更好的可推广模型。
- 接近零：当前预测对这一联合开关不敏感；不能单凭此证明分支未学习或完全无用。

`native_gate` 指**同一份ACDG权重下**基础logit的sigmoid，不是fixed模型实际门值。
饱和仅用固定描述性阈值 `gate <= 0.01` 或 `gate >= 0.99` 统计，不用于选模。
OFF的 `proposed_delta` 是OFF隐藏状态下条件MLP的候选输出，`applied_delta`应为零；
ON/OFF后续层隐藏状态可能不同，因此它们的候选输出差异不是单层独立干预效果。
门值统计按历史槽—节点位置计数，不能当作独立实验重复。

固定科学状态为 `POSTHOC_FIXED_WEIGHT_INTERVENTION_NOT_RETRAINED_ABLATION`。
这是复用验证集上的事后计算干预，不识别真实事故因果效应，也不自动升级此前的负向初筛结论。
运行完成后回传末尾汇总即可。

本地验证：PyTorch 2.3.1、NumPy 1.22.4的CPU环境完成8项小型真实模型测试，
覆盖ON逐值保持、OFF与同权重原生门控等价、零初始化开关等价、支持掩码、异常移除hook、
推理状态不变、ON复现失败阻止OFF、对照符号与样本轴检查；Bash语法检查通过。
尚未在本地读取服务器真实checkpoint或执行V100探针；服务器脚本会在原环境重跑这8项测试。
