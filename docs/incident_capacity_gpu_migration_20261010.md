# M4.3 GPU型号变化的显式续跑

2026-10-10。用户无法选择服务器分配的GPU型号。回传诊断已确认，原运行与当前环境只有`environment.gpu`不同：`Tesla V100-SXM2-32GB`变为`Tesla V100-PCIE-32GB`。据此增加显式兼容选项，用户可在已分配设备上继续六组共同10轮预算。

之前要求申请原GPU型号的处理不适用于这一约束。型号字段用于运行环境追踪；本次将它与模型、数据、协议、初始化和软件版本的严格检查分开处理。默认续跑仍要求完整身份一致，仅加`--allow-gpu-name-change`时允许这一项差异。

## 实现及边界

仅修改现有诊断器与续跑包装入口。17个冻结训练源文件、模型、协议、学习率及损失规则保持原字节。

显式允许须同时满足：完整身份的唯一差异路径为`environment.gpu`；旧值和新值均为已有非空字符串；设备仍为同一个`cuda:N`。缺失GPU字段、切换CPU、软件版本变化、数据/代码指纹变化及初始化哈希变化均不能通过这个选项放行。

通过后打印`RESUME_GPU_NAME_CHANGE_ACCEPTED`，诊断保留旧/新完整身份，`matching=false`和`resume_accepted=true`分别表达原身份并非完全相等、此次差异已显式接受。`--identity-audit-only`仍在任何更新前停止，即使这一差异可以接受。

原根目录`identity.json`保持原字节。各组仍严格校验已有检查点的来源身份，恢复原模型、Adam和学习率调度状态，按原epoch计划继续更新。新检查点及summary的`identity`表示原运行来源；真实执行环境另存`runtime_segments`，不能把其中的原GPU字段解释成后续轮次的实际设备。

`runtime_segments`按连续epoch段记录：历史无逐段记录的轮次引用原身份，并标明`source=origin_identity`；新段标明`source=checked_resume`，保存实际Python/CUDA/GPU/确定性设置、轮次范围、续跑前检查点SHA256以及本次诊断文件路径与SHA256。后续再次续跑会保留此前各段，包括回到原型号的情况。回读入口检查轮次连续性、硬件变化范围及summary/checkpoint一致性，并在before/after报告中明确`hardware_migration_observed`。

这不是同硬件逐位复现认证。六组应共同完成第6—10轮；结果须保留这次硬件迁移信息。前5轮与后5轮的耗时不能直接用于判断模块加速。训练器本体仍保留旧严格入口；此类迁移统一使用`continue_incident_capacity.py`。

## 服务器命令

单独更新代码，运行入口内没有Git操作：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

在当前已分配GPU的终端执行，无需更换节点或修改服务器配置：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/continue_incident_capacity.py \
  --run-dir experiments/chronological_runs/contra_training_m42_20261010_153544 \
  --history-dir experiments/chronological_runs/contra_v11a_multichannel_history_02 \
  --epochs 10 \
  --allow-gpu-name-change
```

`--epochs 10`表示绝对停止轮次，第5轮检查点继续到第10轮。若仍有其他身份差异，会在任何训练更新前列出并停止。只做诊断时再加`--identity-audit-only`。

## 验收范围

新增10项测试及既有73项回归全部通过，共83项、无跳过。覆盖显式GPU差异放行、默认拒绝、其他字段和缺失GPU字段拒绝、只读审查不改原产物、来源身份不变、Adam/调度器恢复、与严格同硬件对照的更新结果一致、跨多次续跑的环境段保存及异常时内存钩子恢复。

真实数据验证使用原数据首4条与隔离检查点副本；只在测试副本中人为改变原GPU名称，其他输入、状态和冻结源均来自已有运行。六组续跑、产物回读和同设备严格对照用于验证入口正确性，不能声称已经完成真实SXM2→PCIE硬件迁移测试，也不构成预测增益证据。具体验收结果见`incident_capacity_gpu_migration_20261010.json`。

具体结果：只读审查前后32个副本原产物哈希一致；未加选项时仍在更新前拒绝；加选项后六组从epoch2续至epoch3，每组累计6次更新。新检查点和summary中的来源身份与实际GPU记录均通过回读，原来源运行的32个产物未改动。与同设备严格续跑对照比较，六组模型权重、Adam状态和调度器状态逐项完全一致。验证脚本曾因Windows路径键分隔符中断检查，修正核对键后完成剩余回读，训练入口未因此修改。17个冻结训练源文件和协议指纹保持不变。服务器真实迁移续跑及10轮结果仍待用户运行。
