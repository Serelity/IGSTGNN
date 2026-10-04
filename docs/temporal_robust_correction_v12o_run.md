# v12o 运行与交付

本轮真正重新训练 ICSF 向量适配器，骨干 A 冻结。两架构 × 两目标 × 三种子，
正式共 144 epochs / 27360 次 Adam 更新 / 12 个选定端点。
研究依据、公式与成功边界见[设计方案](temporal_robust_correction_v12o_plan.md)。

## 服务器执行

按本项目固定流程：本地开发 → GitHub → 你在校园服务器 git pull。无需 SSH。
进入已分配的 V100 GPU 作业终端，沿用 `igstgnn`；不要在无 GPU 的登录节点运行正式训练。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
git switch research/chronological-tiid
git pull --ff-only origin research/chronological-tiid
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
bash experiments/chronological/run_temporal_robust_correction.sh run contra_v12o_temporal_robust_01
```

`run` 前台执行，依次完成针对性测试、真实小包、正式训练；任何一步失败都会停止并保存退出码。
小包也使用 `cuda:0`；单元测试中的小模型主要在 CPU，属于工程测试。
四个时间块保留在小包中。正式训练不需要先运行 v12m/v12n，不读取它们服务器上的结果目录。

输入沿用既有路径：

- `../data/chronological/Contra_Costa_v8_dev`
- `../research_artifacts/v6_inputs_20260920/v3_materialized_01`
- `../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01`
- `experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt`

默认设备为 `cuda:0`。缺少 CUDA 时立即报错，不自动回退 CPU。
检查日志中的 `actual_device_verified`：parameter/input/prediction 三项应均为 `cuda:0`，
训练时 `autograd_enabled=true`；`epoch_complete` 应显示每轮 190 次更新。
显存和 GPU 利用率可能随数据读取、评估、统计波动，是否训练以实际设备与优化器日志为准。

## 状态、汇总与中断恢复

```bash
bash experiments/chronological/run_temporal_robust_correction.sh status contra_v12o_temporal_robust_01
bash experiments/chronological/run_temporal_robust_correction.sh report contra_v12o_temporal_robust_01
```

成功后 `experiments/chronological_runs/contra_v12o_temporal_robust_01/summary.json` 存在。
未完成时保留 `.partial`、`.job/run.log`、`.job/exit_code`，不要删除后重用名称。
如需从原子 epoch 边界恢复，用另一个名称：

```bash
bash experiments/chronological/run_temporal_robust_correction.sh resume contra_v12o_temporal_robust_02 contra_v12o_temporal_robust_01
```

恢复源为旧 `.partial`，原目录保持只读；仍会先跑小包。设备/环境版本和输入/代码身份必须一致。
完成的轨迹跳过优化，部分轨迹接着下一轮训练；所有端点再次冻结后重新生成评估与汇总。
`optimizer_steps` 是整个有效轨迹预算，`optimizer_steps_this_invocation` 是这次进程实际执行步数。

如只想测试服务器环境，可单独使用新名称：

```bash
bash experiments/chronological/run_temporal_robust_correction.sh check contra_v12o_temporal_robust_preflight_01
```

只检查的结果位于对应 `.job/check/summary.json`；这不是正式科研结果。正式启动请用另一个名称。

## 回传与解读

回传 `.job/run.log`、正式 `summary.json` 和 `comparisons.csv`。如需要独立统计核验，再回传
`*_weekly_s*.npz`、每个端点的 `*_geometry.npz`、区域 NPZ 与两个身份清单；无需上传原始交通数组到 GitHub。

主比较 `PRIMARY ... temporal_vs_erm`：正值意味着 temporal 的 MAE 小于 ERM。
必须同时查看各自 `early_gain_vs_A`，以及 G/S/O、两种权重、两种区间和常规交通保护。
优于 ERM 但仍差于 A 不能认定为有效修正；一两个有利种子也不能代替完整结果。
工程 PASS 与科研增益分开记录；这轮开发区间不能用于独立确认声明。

## 本地验收

16 项新增测试与 75 项 v12m/v12n 回归测试通过；覆盖单位权重与原 ERM 的精确等价、真实梯度、
未来标签隔离、共同抽样、非零已选端点评估、完整/中断恢复一致性、恢复篡改拒绝与启动失败传播。
最终启动脚本另作 4 项复检，Shell 语法与 Python 编译检查通过。

真实 496 站小包在本地 `Torch 2.3.1+cpu / NumPy 1.24.4` 完成：四条轨迹、八轮、八次 Adam 更新；
每个适配器 4288 个训练参数。四个 fit 块各两个窗口，其他阶段/群体各两个窗口。
正式交付代码对应的小包耗时 143.06 秒，进程峰值 RSS 2177264 KiB（约 2.08 GiB）。
19 个输入指纹通过原认证；102 个结果文件、34 个运行文件和 30 个旧冻结文件的哈希逐项核对通过。
配对初始化和首轮适配器/Adam 状态完全一致；全部端点冻结后才评估 audit。

四个小包端点均由保护/选择规则退回 epoch 0=A，不能推断完整实验也会如此，或由此判断新方法有效/无效。
此处状态仅为 `ENGINEERING_CHECK_PASS / NOT_EVALUATED_ENGINEERING_ONLY`。
完整 12 轨迹 V100 实验尚未执行。小包日志、结果与校验记录保存于
`复现结果/时间块稳健训练_20261004/`，主核验记录为 `delivery_validation.json`。
冻结协议 SHA-256：`8cbfe2c038f0a78b107981fb3b83a03ebd8ae05e097016506f9b51df5c8c6854`。
