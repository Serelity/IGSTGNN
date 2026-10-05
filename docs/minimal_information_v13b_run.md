# v13b 最小信息消融：运行与结果交付

日期：2026-10-05。代码与本地验证已完成；服务器 V100 测时和正式预测收益尚未获得。
[冻结设计](minimal_information_ablation_v13b_draft.md)定义 M0 交通＋时钟、M1 加位置、M2 加年龄。
**本轮会从头训练全部交通骨干，实际反向传播并执行 Adam 更新，不再只是冻结权重推理。**

## 已完成的本地验证

- 19 项新增测试：输入隔离、唯一 fit X scaler、完整支持、旧文件访问边界、共同初始化、信息梯度、图缓存迁移、
  区域计数、配对区间、保护门槛、实际小模型训练、全部端点先冻结、断点恢复、损坏拒绝与启动器失败传播。
- 原架构机制 12 项与 chronological 数据处理 6 项回归也通过，共 37 项测试；Python 解析、Bash 语法与差异空白检查通过。
- 真实 496 节点包通过预检：fit=1520、selection=856、audit=1010；匹配 selection/audit 分别 206/378。
- 新 scaler 使用 12140 个唯一名义时槽、6,021,440 个有效输入单元，真实零 23494；
  均值 256.8559268214912，标准差 155.14245102535168。与 v13a 按窗口重复计数的描述统计口径不同。
- 真实数据 CPU 小包：三臂各取 4 个 fit 与 4 个 selection 窗口，训练 2 轮；共 12 次 Adam 更新，
  三臂初始化哈希相同，骨干梯度非零且参数更新，audit 保持关闭。
- 本地工程结果不构成位置/年龄的预测收益，也不提供 V100 的耗时与显存估计。

本地结果在 `复现结果/最小信息消融_v13b_20261005/`。
最终代码的小包为 [real_cpu_check_02/summary.json](../../复现结果/最小信息消融_v13b_20261005/real_cpu_check_02/summary.json)。
`_01` 留作增加运行结束输入身份复核之前的工程记录；不覆盖历史文件。

## 服务器先运行测时包

代码提交到 `research/chronological-tiid` 后，在平台分配的 V100 终端执行。沿用原 igstgnn 环境，无 SSH。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
git pull --ff-only origin research/chronological-tiid
bash experiments/chronological/run_minimal_information.sh pilot contra_v13b_information_pilot_01
```

启动器默认 `cuda:0`，CUDA 不可用会失败，不会悄悄切到 CPU。命令前台运行，保持平台作业存活。
它依次运行新增测试、真实小包 check，再执行三臂各一轮完整 fit＋selection。
pilot 本身执行 `3 × 190 = 570` 次 Adam 更新；前置 check 另有 12 次，分别保存在主目录与 `.job/check/`。
pilot 不评价 audit，也不会自动开始正式 9 次训练。

看日志应出现：

```text
Mode: pilot
Device: cuda:0
"stage": "training"
"gradient_device": "cuda:0"
"backbone_trained": true
Status: V13B_COST_PILOT_COMPLETE
Optimizer steps: 570
```

日志可能有测试输出和进度 JSON，以上字段分布在不同位置。最终每臂记录显存分配峰值、训练加选模耗时、
梯度/参数更新、名义参数量与初始/选中模型哈希。完整预算线性时间估计排除了最终队列评价、部分落盘和排队等成本，
只能用于资源规划；若时间不足，应保留 pilot，不能据 pilot 误差更改模型或门槛。

## 正式开发实验

确认分配时间足以覆盖 pilot 的成本估计后，使用新运行名：

```bash
bash experiments/chronological/run_minimal_information.sh run contra_v13b_information_full_01
```

固定为三臂×三个种子×60 轮，共 540 个训练 epoch、102600 次 Adam 更新；前置 check 的 12 次另计。
全骨干训练的成本不能套用此前冻结骨干适配器的耗时。没有提前停止或结果驱动调参。
程序仅用 selection 全事故全节点误差选模型，九个端点全部冻结后才评价 audit。
成功状态：`V13B_DEVELOPMENT_COMPARISON_COMPLETE`；科学判断看 `comparison.json` 中两个增量各自的门槛结果。
即使门槛通过，数据仍为重复开发集，不自动升级为独立确认或开放 test。

## 状态、恢复与单独预检

```bash
bash experiments/chronological/run_minimal_information.sh status contra_v13b_information_full_01
bash experiments/chronological/run_minimal_information.sh report contra_v13b_information_full_01
```

中断后保留旧 `.partial`，在相同环境、代码、数据和设备配置下恢复到新名字：

```bash
bash experiments/chronological/run_minimal_information.sh resume contra_v13b_information_full_02 contra_v13b_information_full_01
```

恢复只支持正式 run 的 epoch 边界；不覆写源目录。未完成的当前 epoch 从上一个已保存边界重算，
完成的轨迹不再优化；全部端点固定后重新导出评价。版本/数据/设备或选模记录不匹配会拒绝恢复。
不要删除 `.partial`、`run_identity.json`、各臂 `last.pt` 或 `history.json`。

只做 CPU 输入预检或 CUDA 小包，可使用：

```bash
bash experiments/chronological/run_minimal_information.sh preflight contra_v13b_information_preflight_01
bash experiments/chronological/run_minimal_information.sh check contra_v13b_information_check_01
```

默认数据目录沿用已有文件，不需要原 A checkpoint：

```text
../data/chronological/Contra_Costa_v8_dev
../research_artifacts/v6_inputs_20260920/v3_materialized_01
../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
```

需要改路径时使用 `V13B_DATA_DIR`、`V13B_PRIMARY_DIR`、`V13B_SECONDARY_DIR`，输入哈希仍必须一致。
`V13B_DEVICE=cpu` 供明确的本地工程检查使用；服务器训练保持默认 CUDA。

## 回传已有文件

测时包先发这一个文件即可，无需压缩：

```text
experiments/chronological_runs/contra_v13b_information_pilot_01/summary.json
```

正式运行结束后，优先回传以下已有文件：

```text
experiments/chronological_runs/contra_v13b_information_full_01/summary.json
experiments/chronological_runs/contra_v13b_information_full_01/paired_comparisons.csv
experiments/chronological_runs/contra_v13b_information_full_01/weekly_statistics.csv
```

目录还保存 `comparison.json`、`scaler.json`、`eligibility.json`、`run_identity.json`、
`frozen_endpoints.json`，以及各臂/种子的 `history.json`、`best.pt`、`last.pt`、逐窗口误差 CSV。
`.job/run.log` 与 `.job/exit_code` 记录过程和退出状态；失败时 `.partial/failure.json` 保存异常。
若使用了其他运行名，替换以上路径中的目录名，不需要创建压缩包。
