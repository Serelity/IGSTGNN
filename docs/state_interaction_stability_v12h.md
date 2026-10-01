# v12h：审计样本构成对比与时间稳定性

## 本轮依据

v12g 在服务器正常完成。六个向量模型在选择期的完整事故候选 H1–H6 收益
已经为负，说明候选早期退步并非只到后段审计期才出现；原保护规则允许小幅
退步，因此“通过保护约束”不能改写为“事故区域改善”。

审计期完整事故包含 1,010 个预测窗口，其中本阶段合格共同三元组窗口为 378，
补集为 632。六个向量模型的完整事故候选早期收益约为 -0.02725 至 -0.01862
原始 MAE 单位；共同组约为 -0.00609 至 +0.00128，按周区间均跨零；补集约为
-0.03833 至 -0.02739，按周区间均低于零。收益定义为 `MAE(A) - MAE(模块)`，
正数表示改善。

这些结果提示样本构成与时间迁移值得检查，但两个组各自的区间不能代替直接
组间差值的区间。补集也不等于“从未匹配的独立事故”。同一事故可产生多个
预测窗口，窗口数不是唯一事故数，也不能用 378/1,010 作为有效单元的权重。

因此 v12h 不扩大模块、不设计新部署路由、不挑选新模型；只进一步核验已经
保存结果中的组间差值、按周变化与全局收益贡献。

## 冻结输入与范围

协议文件为 `experiments/chronological/state_interaction_stability_v12h.json`。
SHA-256：`cb396d4384fdd0b3eb81920bdfd66ecef827abc93bd25dde092b7ccf1259c5a2`。
输入限于完整 v12g 来源目录中的 `summary.json` 和三个
`audit_weekly_s2025.npz`、`audit_weekly_s2026.npz`、`audit_weekly_s2027.npz`。
输入身份、协议和保存统计由诊断代码核验；来源目录保持只读。

来源固定为 `97a8088` 实现记录的五个诊断代码/协议文件身份，核对已完成状态、
三种 seed、原始阶段样本数、六个审计组和每个实际读取 NPZ 的哈希。历史 NPZ
的有效单元及可评估窗口计数为浮点存储，要求其有限、非负且为精确整数值。
同一周网格、评估支持计数和抽样矩阵必须跨种子保持一致，抽样矩阵还需与原
冻结抽样生成器一致；共同事故组的两种回放必须具有完全一致的支持计数。

来源链核验不等于独立重新读取原始 traffic、mask 或每窗口绝对误差数组。
保存的 weekly NPZ 不包含绝对 MAE 或每周总预测窗口数，本轮不重新验证这些量。
原始 A 和 scaler 已使用更广的训练时段，A 也曾由旧验证集选出，时间分段不能
消除这些已有信息依赖。

本轮只依赖 Python 标准库与 NumPy，在 CPU 上计算保存统计。不读取原始 traffic、
manifest、checkpoint、旧 val 或 test，不加载 Torch，不重新训练、不计算梯度，
也不构造新的门槛或重新选择 epoch。三个模块、三个种子均保留，重点报告六个
向量模型，而不是根据审计结果选一个最优种子。

## 诊断内容

第一类是共同组与补集的直接对比。按同一个完整事故日历周网格，计算
`共同组收益 - 补集收益`，并作配对周 bootstrap 和四周循环块 bootstrap。
两种抽样中组间统计使用同一组抽样周，保留组间共同的时间变化，不能分别抽样
两个组再拼接结果。共同组或补集的空支持明确报告，不将缺失收益写成零。

第二类是时间稳定性。保留各周可评估预测窗口数、有效单元数和区域收益，分别报告按有效
单元合并与等预测窗口加权的结果。逐一移除一个日历周后重新计算汇总收益，
用作“结论是否依赖某一周”的敏感性描述；这不是新的检验或额外确认集。
等预测窗口加权不等于等独立事故加权。

逐周和删一周结果分别计数正、负、零和未定义；这些相关的删除不是九次独立
实验，不用于构造新 gate。全程保留空周；空分母将传播到该次组间抽样，达不到
原 95% 有效抽样比例时保留明确的区间不可用状态，不用零代替缺失。

第三类是全局收益的精确贡献分解。每个子组、区域的误差收益总和除以完整事故
全局有效单元总数，再核对贡献之和能否重构完整事故全局收益。按周也采用对应
完整事故全局有效单元数作分母。不得把各区域平均收益直接相加，也不得按窗口
数比例推算有效单元贡献。候选早期损害与非候选收益需要并列观察。

整体贡献不是各周贡献的简单平均，需按每周完整事故全局有效单元数加权。
共同组与补集可重构窗口加权和单元加权的保存统计；但等窗口区域收益不能在
不同区域之间直接相加，因为其可评估窗口和区域分母不同。

## 解释边界

本轮继续复用原 train 内已经多次查看过的开发时段，只有九个审计日历周。
区间以已经保存的模型为条件，不包含训练种子或多重比较的全部不确定性；
四周块抽样只是对短时序相关性的敏感性检查，不能消除重复使用数据的限制。

直接组间差值能回答两组在这些保存结果中的收益是否不同，但不能说明其原因。
不据此断言补集身份导致退步、架构缺陷、过拟合、特征不足或事故因果效应，
也不把此轮分析称为独立确认。若差值或周敏感性不稳定，应保留这一结果，
不能在看过输出后调整组定义、筛周、改阈值或挑种子。

## 输出与运行

完整结果包含：

- `summary.json`：输入核验、直接组间对比、时间稳定性和贡献核算。
- `composition_contrasts.csv`：共享周网格上的共同组减补集收益及区间。
- `temporal_stability.csv`：按周结果与移除一周的敏感性结果。
- `global_contributions.csv`：精确有效单元加权的全局贡献。

服务器使用已有 NumPy 环境即可，不要求 GPU。请在有足够剩余时间和资源的
有效作业环境中运行，保持前台终端打开：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_state_interaction_stability_audit.sh run contra_v12h_stability_audit_01
```

默认读取 `contra_v12g_transfer_audit_01`。如实际来源运行名不同，第三个位置
传入完整 v12g 运行名：

```bash
bash experiments/chronological/run_state_interaction_stability_audit.sh run contra_v12h_stability_audit_02 contra_v12g_transfer_audit_02
```

另开终端查看日志和状态，结束后打印报告：

```bash
tail -f experiments/chronological_runs/contra_v12h_stability_audit_01.job/run.log
bash experiments/chronological/run_state_interaction_stability_audit.sh status contra_v12h_stability_audit_01
bash experiments/chronological/run_state_interaction_stability_audit.sh report contra_v12h_stability_audit_01
```

入口核对四个来源文件，记录 host、PID、Slurm job ID 与退出码，先执行 NumPy
预检及 v12h 测试，再开始诊断。完整写入前使用 `.partial`；已有 final、partial、
job 或相应符号链接都会拒绝覆盖。`report` 缺少完整 summary 时会提示先检查
`status`，不会把未完成目录当作研究结果。

中断后先核对进程、退出码和日志，再使用新运行名从只读来源重新审计；没有
训练恢复入口。无需删除已完成的 v12g 或旧诊断现场。
