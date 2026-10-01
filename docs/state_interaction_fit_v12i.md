# v12i：固定已选模型的完整拟合期区域评估

## 本轮依据

v12h 在服务器正常完成。六个向量模型在审计期完整事故与补集的候选 H1–H6
收益均为负，四周块区间低于零，删除任意一周后仍负。共同组收益接近零，
区间和删一周结果跨零。共同组减补集的点估计虽为正，等预测窗口加权的直接
差值区间仍跨零，不能将共同组身份直接用于部署路由。

同时，非候选区域的小幅正贡献抵消了候选早期损害，得到很小的全局正收益。
因此全局收益或原保护约束通过，不等于事故候选早期已经改善。strength 两个
种子回退到 A，只有一个种子改善，也没有构成稳定成功。

此前 v12f 的拟合期表征诊断仅覆盖 64 个固定窗口，不能据此判断完整拟合期
是否改善。v12i 补齐全部 1,520 个合格完整事故拟合窗口，比较固定 A 与九个
已选适配器，回答：候选早期是在拟合期改善、后续阶段丢失收益，还是完整
拟合期本身也未改善？结果只能帮助确定后续诊断方向，不能独自证明失败原因。

## 冻结范围与信息边界

协议文件为 `experiments/chronological/state_interaction_fit_v12i.json`。
SHA-256：`7c5e7090fed6e1b28660accc76331db43b5b42fa306c0c5b6cdcaa23b8bf08ea`。

完整来源目录必须是已完成的 v12f 运行，包含 `summary.json`、
`run_identity.json`、`eligibility.json`，以及三种适配器 `strength`、
`state_vector`、`interaction_vector` 在 seed 2025、2026、2027 下的九个
`selected_gate.pt`。来源保持只读，不能把中断运行的 `.partial` 作为完整结果。

本轮只读取原 train 数据和训练期匹配控制输入用于身份、资格与冻结 scaler
核验。模型推理仅针对完整拟合期正例窗口；不重新读取旧 val/test，不为控制
拟合新专家。审计期区域指标复用 v12f 已保存的完整正例每窗口误差 NPZ。
选择期只保存了已选模型的合并区域 MAE，没有每窗口已选模型误差数组，因此
只复用选择期保存的合并指标，不补造等窗口统计、按周统计或区间，缺失项明确
标记不可用。不能根据本轮结果改选 epoch、筛种子或删窗口。保存的 epoch 0
也是已选结果，必须保留，不能为得到非零改动而改用最后一轮模型；也不推理
最后一轮状态或对 epoch 重新排序。

固定 A 与所有九个已选模型在同一拟合窗口和区域支持上评估。收益定义为
`MAE(A) - MAE(模块)`，正数表示改善；单位是原始 MAE，不是百分比。
候选 H1–H6、候选 H7–H12、非候选及完整全局并列报告，避免用区域间抵消
掩盖早期候选损害。本轮拟合期不再拆分共同组与补集，也不依据旧组间结果
设计新的路由或门槛。

模型使用 `eval` 与无梯度推理，不训练、不开 optimizer、不恢复训练，不重新
选择 epoch。默认在已分配的 GPU 上运行 `cuda:0`，因为这次需重新执行固定
神经网络前向计算；统计汇总仍在 CPU 上完成。小型测试夹具用 CPU Torch
验证流程，不能当作真实数据实验。

九个 checkpoint 均核验保存的来源哈希、arm、seed、selected epoch、冻结
backbone 身份、适配器键与形状、有限张量及语义状态哈希。完整推理前后模型
参数与 buffer 的状态哈希必须不变，epoch 0 的拟合误差必须精确等于 A。
这只检验冻结模型流程，不能把所有种子的 epoch 0 回退说成训练成功。

完整拟合结果再按原清单规则限制到 64 个探针窗口，分别对照原保存的初始
探针与已选探针。原探针 batch size 为 8，新完整推理为 16；逐窗口区域误差
和允许 `rtol=1e-5`、`atol=1e-3` 的冻结数值容差，并报告最大误差和合并 MAE
差异。ID、原始位置、候选 mask、有效单元与预测单元支持仍需精确一致。
这项回放核验不能以 64 窗口结果替代完整 1,520 窗口的评估。

## 解释限制

完整拟合期评估是已选模型的样本内描述，而且已选 epoch 受选择期指标影响。
原 A 与 scaler 已使用更广的 train 时段，A 也曾由旧 val 选出，时间分段不能
消除这些既有信息依赖。拟合、选择、审计的组成与流量状态可能不同，同一事故
也可产生多个重叠预测窗口，因此跨阶段收益差不等于独立估计的泛化误差。

不把本轮称为独立确认、事故因果效应或架构缺陷证明。不建立新的 gate，不
根据观察结果调整区域、阶段、阈值或 seed。若完整拟合期仍没有候选早期收益，
可进一步检查优化目标和修正位置；若拟合期改善而后续未改善，可进一步检查
时间迁移和学习稳定性。两种模式都只是后续研究的线索。

## 服务器运行

完整结果包含 `summary.json`、`phase_metrics.csv`、`weekly_metrics.csv`、
`phase_gain_changes.csv`，以及完整拟合期的 `fit_A.npz` 与九份
`fit_{arm}_s{seed}.npz`。跨阶段收益变化只作描述，不补出选择期缺失的统计。

`summary.results[seed].phases` 保存 fit 与 audit 的合并和等预测窗口统计、
区域贡献及按周结果；`selection_point_context` 仅保留选择期合并点估计。
`phase_gain_changes` 报告后续阶段减拟合期的收益差，不构造跨阶段配对区间；
`saved_probe_replay` 记录上述原探针回放差异与冻结容差。拟合期区间为样本内
描述，不是独立泛化置信保证。

使用已分配且 CUDA 可用的 GPU 作业环境，激活 `igstgnn`。这是前台入口，
保持终端打开。已有 v12f/v12g/v12h 来源和结果无需删除：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_state_interaction_fit_audit.sh run contra_v12i_full_fit_audit_01
```

默认来源为 `contra_v12f_state_interaction_01`。如完整来源运行名不同，通过
第三个位置参数指定：

```bash
bash experiments/chronological/run_state_interaction_fit_audit.sh run contra_v12i_full_fit_audit_02 contra_v12f_state_interaction_02
```

默认设备为 `cuda:0`，需要另一个已分配设备时可设置 `V12I_DEVICE=cuda:1`。
也支持显式设置 `V12I_DEVICE=cpu` 作诊断，但真实完整推理建议使用已分配 GPU。
入口核对三份来源元数据与九个已选 checkpoint，随后记录 host、PID、Slurm
job ID、Python、设备、来源与 Git 提交，执行 Torch/CUDA 预检及 v12i 测试，
通过后才启动固定模型评估。CUDA 不可用或测试失败会写明退出码，不会继续。

另开终端查看日志和状态，完成后打印报告：

```bash
tail -f experiments/chronological_runs/contra_v12i_full_fit_audit_01.job/run.log
bash experiments/chronological/run_state_interaction_fit_audit.sh status contra_v12i_full_fit_audit_01
bash experiments/chronological/run_state_interaction_fit_audit.sh report contra_v12i_full_fit_audit_01
```

完整结果以 `summary.json` 为入口；仅有 `.partial` 或日志不能当作已完成研究
结果。已有 final、partial、job 或相应悬空符号链接均拒绝覆盖。中断后先核对
退出码和日志，再用新运行名从只读完整 v12f 来源重新评估；本轮没有 resume
或 check 训练入口，不需要清空旧实验现场。
