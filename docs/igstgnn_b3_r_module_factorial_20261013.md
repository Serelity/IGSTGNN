# B3-R 模块必要性筛查：TIID × D 四组合

日期：2026-10-13。阶段：同一检查点的条件离线推理诊断。

## 研究目的

IGSTGNN 本身已有事故处理，保留原模块不能成为改进的前提。现在需要判断：新增 D 存在时原 TIID 是否还有增量价值，以及两条路线是否相互补偿。先定位有希望的精简版本，再通过重新训练验证可删除性。

本次开发不改变训练、模型公式、检查点、原始输入或选模规则。使用当前模型在已完成训练轮次中的首个最低全网验证 MAE 检查点。

## 实际模块关系

- B3-R 已替换原 ICSF 空间注入；共享报告编码、K 投影、归一化及 TIID 接口继续使用。
- TIID 在输出隐藏状态上加入带固定时间衰减的报告 context。
- D 读取历史摘要、报告 embedding、距离和年龄，修正潜容量 logit；P 和边界头继续运行。
- 原图卷积与新受限交换在内部相加。四组合中均保持这两条传播路线开启。
- 容量、潜状态和交换值无真实车辆或道路容量单位，不据此声称物理识别。

## 四种状态

| artifact key | TIID | D | 保持的内容 |
|---|---|---|---|
| TIID_ON_D_ON | 开 | 开 | 检查点原始模型 |
| TIID_ON_D_OFF | 开 | 关 | 原 TIID、P、边界、图和交换 |
| TIID_OFF_D_ON | 关 | 开 | 新 D、P、边界、图和交换 |
| TIID_OFF_D_OFF | 关 | 关 | P、边界、图和交换 |

开关直接传入模型 `tiid_report_enabled` 和 `spatial_report_enabled`。报告输入与距离不置零，`report_path_enabled` 属性不修改。这样不会把改变报告支持范围混入路径关闭。

D 同时处理交通历史与报告，因此 D OFF 不是纯事故内容消融。双 OFF 也不是重新训练的“无事故模型”；共享报告编码仍执行，但这两条消费报告的预测路径关闭。

## 回放与保护

1. 支持旧 `b3r_run_v1` 和当前 `b3r_strength_run_g11_v1`，分别重建原协议、模型和初始化证书。
2. 仅允许原来训练时 D 开启的 B3R、B3R_HALF、B3R_GATE。NO_D 的 D 未按同样路径训练，不能事后强行开启。
3. 校验源代码、数据包、候选图、站点顺序、运行身份、history、summary 与 checkpoint；严格载入 `last.pt:best_model_state`。
4. 完整验证集先用原 batch 大小和原顺序回放双 ON。按保存的逐窗口/逐时距误差、计数、时间与最佳 MAE 验证，失败时阻止所有 OFF。
5. 每个 batch：双 ON → 仅 TIID → 仅 D → 双 OFF → 恢复双 ON。五次使用同一批输入，尾 batch 不重分组。
6. 恢复后的预测与容量 context 精确一致；模型 state_dict、输入、梯度、开关均保持。固定 D 时关闭 TIID 不得改变空间 context，也不得影响直接报告支持外的节点。
7. 无报告支持的样本四种状态预测必须完全相同。
8. 完成后复核原训练文件、源代码、数据和图哈希，才写成功摘要。

复现容差沿用旧探针：保存的窗口平均误差 `atol=1e-4, rtol=1e-6`；全网回放 MAE 绝对差不超过 `1e-4`。同一分组的重复 ON 与恢复 ON 要求逐元素精确相同。GPU 名称不属于严格运行身份，可记录不同 SKU；Torch/CUDA、设备类型和线程等仍按原身份验证。

本工具无 batch-size 参数；`--threads` 默认使用原记录。当前问题需要可比的推理干预，增大 batch 的效率试验另行安排。

## 指标与符号

四组合均输出显式有效掩码下的逐时距与宏平均 MAE/RMSE，零流量标签保持有效；空区域报告 unavailable。

设 `M11` 为双 ON、`M10` 为仅 TIID、`M01` 为仅 D、`M00` 为双 OFF 的 MAE：

| 指标 | 公式 |
|---|---|
| D_given_TIID_ON | M10 − M11 |
| D_given_TIID_OFF | M00 − M01 |
| TIID_given_D_ON | M01 − M11 |
| TIID_given_D_OFF | M00 − M10 |
| combined_vs_both_off | M00 − M11 |
| TIID_only_vs_D_only | M01 − M10 |
| MAE 交互量 | M10 + M01 − M11 − M00 |

收益为正表示加入对应路径后误差降低。交互量为正表示当前权重下，两路联合降低 MAE 的幅度超过各自单独作用之和；交互量为负不自动证明模块冗余，绝对误差本身具有非线性。

每个配对还计算逐点正收益、负收益、净收益、抵消比例、预测变化幅度与窗口胜负。保留分道路、报告关联、候选传播包络、仅潜在传播、算子外、无报告样本的结果。区域可能重叠，区域均值不是可相加的全网贡献；`outside_operator` 不代表没有事故。

窗口 MAE 为每个窗口各时距 MAE 的等权均值，仅纳入全部时距都有有效点的完整窗口。各状态分别报告窗口 P95 和各自最差 `ceil(5% × 完整窗口数)` 的平均误差；另以双 ON 的全网误差固定选择最差10%完整窗口，让四组合在同一集合上比较。排序同分按 sample_index，部分有效窗口仍参与全网指标但不参加尾部集合。

重叠窗口不是独立实验重复，不输出 p 值或显著性。固定困难集为诊断集合，不用它替代原选模指标。

## 服务器运行

先单独更新代码，Git 不在运行脚本内：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

优先检查当前 G1.1 运行中的 B3R：

```bash
bash experiments/chronological/run_b3_r_module_probe.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025
```

在已有 Slurm 作业内直接推理；登录节点上自动用 sbatch 提交。脚本无 `--mem`。当前训练若仍占有同一 run 的锁，先让它退出，保护现有检查点。

需要检查成熟旧 B3R 时：

```bash
bash experiments/chronological/run_b3_r_module_probe.sh \
  --run-dir experiments/chronological_runs/contra_b3r_v11_seed2025
```

可选指定已训练的门控模型：

```bash
bash experiments/chronological/run_b3_r_module_probe.sh \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025 \
  --arm B3R_GATE
```

Python 入口同样支持上述参数：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/probe_b3_r_modules.py \
  --run-dir experiments/chronological_runs/contra_b3rg11_v1_seed2025
```

默认使用既定数据和候选图目录，改变数据位置时用 `--data-dir` / `--graph-dir` 指向同一指纹的文件。CPU 工程核验使用 `--device cpu`；原记录设备类型不同会被身份检查拒绝，不默许跨运行环境产生可比性结论。

## 输出与后续决策

新目录为原 run 下 `module_factorial_<arm>_<时间>_<随机后缀>`：

- `provenance.json`：权重、最佳/已完成轮次、身份、环境与输入/源码哈希。
- `replay.json`：双 ON 原窗口回放证据。
- `paired_predictions.npz`：四格预测、target、valid、sample_indices、t0、station_ids 和全部 region mask。
- `analysis.json`：四格指标、六对收益、交互与固定困难集。
- `mechanisms.json`：各格容量/限制活动与门控统计；`capacity_changes.json`：D ON/OFF 容量变化。
- `preservation.json`：原训练产物、数据、图与源文件保护检查。
- `report.md`：全网、时距、道路/区域、正负收益和困难集表。
- `summary.json`：全部检查通过后的成功记录；异常用 `failure.json`，不写成功结论。

结果用于提出候选精简组。如果 TIID 在 D 存在时依赖很弱、或移除后更好，再从头训练 TIID OFF；D 同理。正式删除要求相同训练预算、多 seed 与全网/道路/风险指标支持，不能只凭一次 OFF 探针删除。

下一阶段再研究原统计邻居传播与受限交换的重叠。若控制原图传播，需要保留本地时序编码及 exchange 输入，处理候选图外站点回退，不能把整个 STLocalizedConv 置零。本轮没有实现或执行该空间精简实验。
