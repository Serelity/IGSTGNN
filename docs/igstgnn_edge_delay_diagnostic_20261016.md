# 原IGSTGNN道路时延：best/last只读诊断

本入口在[首轮实现](igstgnn_edge_delay_first_round_20261016.md)上补齐best/last只读机制诊断；不改原模型、首轮训练器、旧运行身份或源检查点，不新增训练及测试集评价。后续结构候选仍需依据服务器诊断结果确定。

## 为什么需要这次诊断

服务器首轮CONTROL、FIXED、DYNAMIC均完成1轮，最佳均为阶段0。旧探针只评价最佳权重，因此尚未解释epoch1训练后退化。本入口分别评价best与last；即使best=0，也不会把阶段0预测冒充训练后预测。

## 服务器命令

更新到包含本入口的提交后，在仓库目录运行：

```bash
bash experiments/chronological/run_original_edge_delay_diagnostic.sh \
  --pilot-run experiments/chronological_runs/contra_original_edge_delay_pilot_s2025
```

默认经Slurm提交；已有Slurm作业内直接执行。默认输出为旧pilot旁的独立时间戳目录，终端最后打印`Output directory`。也可通过`--output-dir`指定一个尚不存在、与输入run及数据目录不重叠的目录。没有`--resume`，不向旧目录写状态。

须使用原pilot的Torch/NumPy/CUDA版本、设备和线程配置；默认从其identity读取设备、线程及种子。`--data-dir`可定位同一哈希认证的数据。原源baseline路径来自pilot identity，要求原目录仍可读且源制品未变化。

只有诊断原工程pilot时才能使用`--check`；它不能把正式837窗口run缩成4窗口。脚本会阻止正在写入、未完成、不同完成预算、历史代码hash不匹配、输入/批次/尺度不一致的pilot。

本地工程验证示例：

```powershell
& 'D:\anaconda\set\envs\stgcn\python.exe' -u experiments/chronological/evaluate_original_edge_delay.py --pilot-run experiments/chronological_runs/original_edge_delay_check_20261016_final --check
```

## 评价状态与复用

| 权重 | 状态 |
|---|---|
| SOURCE | 原最佳权重FULL回放 |
| CONTROL best/last | 原生FULL |
| FIXED best/last | FIXED、NO_ALIGN |
| DYNAMIC best/last | DYNAMIC、NO_ALIGN、MASS_ONLY、SHUFFLE |

共15个具名状态。DYNAMIC best和last在干预后各另做一次ON回放核对。只有完整state hash与模式均相同才复用本次确定性预测，并记录`reused_from`及零新增调用；不能仅凭指标相同复用。相同权重的best与last仍保留各自epoch和文件身份。

SOURCE和各best须逐点复现旧归档预测；best/last的原生FP32 MAE、RMSE、MAPE须与对应epoch日志核对。新报告另按float64统计pooled和时距macro，不将两种精度口径混作精确恒等式。

## 机制统计与定义

- 时延按层/道路方向统计均值、极值、跨边std、跨样本std。跨样本方差使用float64分批Welford合并，防止固定时延因二阶矩抵消产生伪变化。
- 分位数以2048个固定宽度桶给出包含区间；不标为精确分位数。SHUFFLE同时列出实际被错配改变的时延位置数；常数时延时该探针缺乏区分力。
- 历史支持用实际观测时延在float64重算一跳和两跳道路系数质量。参考质量采用同一图、同一空间阶数及最终对角扣除；零参考路径单列。
- `window_end_input_index`是输入12步历史中的0起算位置。`summary_length`明确保留10/8/6/4/2等各层真实摘要长度。
- `missing_weight_fraction`为相对于完整历史参考的缺失系数质量比例，不是边数比例、隐特征强度、物理容量或MAE贡献。
- 原主干可训练参数、主干buffer及新增参数相对确定性安装初值的变化分别记录，防止BN计数器主导“权重变化”；这些范数均不能解释为重要性。

指标分全网、十二时距、道路、真值流量、固定SOURCE困难窗口及当前偏差。困难ID从旧analysis读取并与原SOURCE规则核对，不重选；偏差统一由`DYNAMIC_LAST_NO_ALIGN`定义。正负收益、抵消和预测响应均为描述性配对统计。

## 计时与只读边界

前向hook仅留存时延引用和形状；全部机制归约在同步完整前向计时之后执行。`timings.json`分列`forward_seconds`和`diagnostic_seconds`，前向峰值显存也在归约前读取。首批可能含冷启动，数据搬运、逆变换、哈希和CPU统计不属于前向时间。

本入口不新增反向或优化器，不报告新训练速度。旧训练墙钟直接复用history并保留其包含加载/指标的口径。

输入run使用已有锁文件的只读锁，不创建/写入输入文件；Windows锁文件自身无法由第二句柄读hash，因此`active.lock`内容从hash快照中明确排除。其他输入run文件全部在锁内前后核对。只读推理检查模型张量和RNG；异常移除hooks并仅在新诊断目录保存failure。

仅创建验证loader，但原对象数组会反序列化完整payload；不声称从未读取测试payload。没有测试集预测或指标。

## 产物与解释

输出`report.md`、`summary.json`、`analysis.json`、`paired_predictions.npz`，以及cases/comparisons/horizons/flow/roads/current_bias/fixed_hard/group_horizons CSV、`boundary_support.csv`、`lag_statistics.json`、`parameter_changes.json`、`timings.json`、`replay.json`、`preservation.json`、`provenance.json`。

若失败，先读新目录的`failure.json`；无需重训。只有全部回放、恢复、数据身份和输入保护检查通过才写`EDGE_DELAY_BEST_LAST_DIAGNOSTIC_PASS`；PASS不代表精度胜出。验证集已反复用于探索，窗口可能重叠；推理干预可能分布外，不等于独立重训消融、事故因果或真实旅行时延。

本地测试与CUDA工程验证结果见[验证记录](verification/igstgnn_edge_delay_diagnostic_20261016.json)。本地4窗口结果只验证工程通路，不能替代服务器837窗口的正式开发诊断。
