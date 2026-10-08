# P2 固定检查点：五组逐层关闭门控残差

作业1163180已完成五层同时ON/OFF的推理对照。全关闭相对全开启的MAE变化为：

| 区域 | OFF − ON |
|---|---:|
| 全节点 | -0.013984116 |
| 关联节点 | +0.016834267 |
| 非关联节点 | -0.016660488 |

这提示当前权重下可能同时存在局部收益和间接代价。本步骤检查它们是否集中于特定层。
不根据第2/4层门值变化较大预先归因；完整报告五组结果，不重新选择检查点或自动选出新模型。

## 干预对象

使用现有ACDG最佳第69轮 `best_model.pt`，每组仅将一个分解层内的
`estimation_gate.incident_condition` 输出置零，其余四层的新增分支正常计算。
基础门控、整个分解层、扩散/固有分支、ICSF、TIID、图计算与预测头均保留。

```text
指定层：sigmoid(base_logit + condition_delta) → sigmoid(base_logit)
其余层：继续计算 sigmoid(base_logit + condition_delta)
```

| 输出中的实验名 | 人类编号 | 唯一关闭的模块 |
|---|---:|---|
| layer_1_off | 第1层 | layers.0.estimation_gate.incident_condition |
| layer_2_off | 第2层 | layers.1.estimation_gate.incident_condition |
| layer_3_off | 第3层 | layers.2.estimation_gate.incident_condition |
| layer_4_off | 第4层 | layers.3.estimation_gate.incident_condition |
| layer_5_off | 第5层 | layers.4.estimation_gate.incident_condition |

这是五组独立关闭，不是逐步累积关闭。较早层的干预可以改变后续隐藏状态，后续层的候选门控值
也可能随之变化；“其余层开启”指其计算保留，并不强制它们取全开时的数值。
模型在分解层之前构建动态图，因此不能把此干预称为直接修改动态图生成。

## 运行

先单独更新代码：

```text
git pull --ff-only origin research/chronological-tiid
```

在Slurm平台分配1张V100、3核CPU、32GB内存，建议预留30分钟（资源预留，不是实测耗时）。
沿用 `igstgnn` 环境，在仓库根目录运行：

```bash
bash experiments/chronological/probe_incident_routing_p2_layers.sh contra_p2_first_epoch_20261008_172731_wNQWZK
```

Bash中没有Git操作，不调用sbatch，不安装环境。默认数据目录为
`../data/chronological/Contra_Costa_v8_dev`，可用 `P2_DATA_DIR` 指向原数据包。
共七次完整验证推理：全开复现、五组单层关闭、再次全开恢复检查。没有训练或参数更新。

启动时复用原ON/OFF脚本的检查：冻结源码/协议、原数据包及验证清单、917窗口/496站/12步、
batch48、seed2025、原Torch/NumPy/CUDA构建、V100、3线程与确定性设置。
最佳权重须与原最后检查点中的选模状态一致。当前目录选出的仍是第69轮。

第一次全开须复现已保存的ACDG最佳预测：最大绝对预测差不超过 `1e-5`、
三个区域MAE差不超过 `1e-8`，否则五组关闭均不执行。
每次推理核对权重/缓冲区哈希、全部样本及门控统计覆盖；指定分支的实际增量须为零，
未指定分支保留其候选增量，非关联位置的直接门控增量仍须为零。
最后全开预测须与本次第一次全开逐元素完全一致；完成前再次核对输入文件和数据包未变。
任一检查失败都不会写成功的 `summary.json`。

## 输出与解释

新目录为原配对目录下 `gate_layerwise_时间_随机后缀/`，包含 `run.log`、`exit_code`。
`report/` 内保存：

- `probe_config.json`：运行前记录五组层编号、权重身份、输入哈希和固定容差。
- `on_replay_check.json`、`on_restoration_check.json`：首尾全开检查。
- `all_on_predictions.npz` 与五个 `layer_N_off_predictions.npz`：预测、标签、有效/关联掩码、样本与节点轴。
- `layer_comparisons.csv`：五组 × 三个区域，共15行，包含总MAE及H1–3、H4–6、H7–12差值。
- `per_horizon.csv`：五组 × 三个区域 × 12步，共180行。
- `gate_statistics.csv`：七次推理 × 五层 × 两个支持区域，共70行；记录实际关闭标志、增量与门值统计。
- `summary.json`：完整结果、首尾复现、运行环境及输入输出哈希。

成功状态为 `P2_FIXED_CHECKPOINT_LAYER_GATE_PROBE_COMPLETE`，正常结束的退出码为0。
日志末尾会打印五组、三个区域的总误差与分时段差值，回传这部分即可。

主要比较量：`layer_off_minus_all_on = MAE(仅该层关闭) − MAE(全部开启)`。

- 正值：移除该层分支使当前误差升高，开启该分支在当前其余四层开启时有帮助。
- 负值：移除该层分支使当前误差降低，提示该层分支在这个上下文中存在代价。
- 接近零：当前模型对该单层干预不敏感，不能据此证明该层无用。

每个效应都以其余四层开启为条件，**不能把五层差值相加，当成联合关闭的总效果**。
保留fixed已保存的预测作为背景参考，单层关闭仍是ACDG训练过的主干。
这些差异属于复用验证集的固定权重诊断，不代表重训消融或最佳交互位置，也不证明事故信息的
独立价值或真实事故因果效应。门值统计按隐藏位置计数，不能视为独立重复。

## 本地验证

19项CPU真实小模型测试通过（PyTorch2.3.1、NumPy1.22.4）：共享ON/OFF的9项回归测试，
以及逐层关闭的10项测试。测试覆盖五层分别与独立置零投影模型逐值一致、其余分支保留、
异常清理、首尾复现失败拦截、七次实际推理与NPZ/CSV写入回读、指标符号、轴匹配和空区域。
测试没有加载服务器真实权重；V100和完整数据的结果需本次服务器运行产生。

## 作业1163295：启动检查失败后的修复

2026-10-08 23:13:43，gpu20上的作业1163295在17项CPU单测中有一项报错：
`layer_4_off` 测试用例触发 `Expected zero direct gate change`，退出码1。
正式检查点推理尚未开始，没有产生逐层实验结果；报错用例也不等于已证明第4层模型失效。
原日志未记录失败统计行及数值，无法据此判断差值大小。

旧统计路径分别计算切片的 `sigmoid(base)` 与加法生成张量的 `sigmoid(base + applied)`，
两者可能采用不同内存布局。即使增量为零，不同CPU内核路径的舍入差异也可能触发严格相等检查。
修复将参考值改为 `sigmoid(base + zeros_like(applied))`，让两侧经过相同的加法和布局处理。
仅用于统计的参考路径发生变化，返回模型的增量、模型前向、训练源码和检查点均不改变。
没有放宽零增量、预测复现或恢复容差。

新增测试覆盖切片长度、batch大小及局部/整层零增量，并验证两侧sigmoid输入布局一致；
另一个测试确认极小非零门值变化仍会被拒绝，错误信息会给出层、区域及实际差值。
19项测试本地通过。服务器原报错尚需重跑确认，更新代码后使用原Bash命令即可，输出会另建目录。
