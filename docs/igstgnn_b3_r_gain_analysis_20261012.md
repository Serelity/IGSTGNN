# B3-R：已有D路径探针的离线收益分解

日期：2026-10-12。本阶段分析同一B3-R检查点开启/关闭整个D路径时，哪些位置获得收益、哪些位置受到损害。原TIID始终开启。只读取已经保存的验证预测，不读取检查点、训练数据或Test，不调用模型，不重新推理或训练。

## 服务器运行

先更新代码（Git与运行脚本分开）：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

对已经完成的第84轮最佳检查点路径探针执行：

```bash
bash experiments/chronological/run_b3_r_gain_analysis.sh \
  --probe-dir experiments/chronological_runs/contra_b3r_v11_seed2025/b3r_path_probe_20261012_205122_3150be
```

这是CPU上的NumPy离线分析，不需要申请GPU，也不会自动提交Slurm；已有allocation内同样可运行。脚本固定使用服务器的`igstgnn`环境，当前提示符为`(base)`也可以；无`--mem`和Git命令。用`B3R_REPO_DIR`、`PYTHON_BIN`覆盖仓库及Python路径。

直接使用Python的等价入口：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/analyze_b3_r_path_probe.py \
  --probe-dir experiments/chronological_runs/contra_b3r_v11_seed2025/b3r_path_probe_20261012_205122_3150be
```

程序在原run下创建独立的`b3r_gain_analysis_时间_随机串`目录。可以用`--output-dir`指定一个尚不存在的新目录；禁止覆盖或写入源探针目录。分析历史探针不要求当前父run的检查点继续保持当时状态。

## 输入与校验

读取源探针中的五个文件：`summary.json`、`provenance.json`、`replay.json`、`preservation.json`、`paired_predictions.npz`。核对完成状态、科学范围、最佳轮数与权重哈希、原ON复现、精确恢复标志、输入/模型保全记录及原batch划分。

NPZ的预测、目标和有效掩码必须是`[917,12,496,1]`；关联/候选掩码为`[917,496]`。仅显式标为工程检查的源产物可通过`--allow-engineering`使用小规模轴。无效目标可以为NaN，不会变成有效零值；任何有效目标或预测非有限均阻断。

源全网及已知分组的ON/OFF误差、各时距计数与预测变化均重新计算。源预测变化采用pooled分母，新主分析采用horizon-macro；保留两种口径并分别核对。源探针未保存NPZ生成时哈希，因此本分析的哈希只能证明本次读取前后保全，不能回溯证明NPZ从生成起没有变化。

## 主指标与分区

对有效元素定义：

```text
g = abs(OFF - y) - abs(ON - y)
w = 1 / (12 * N_h)              # N_h为该时距全网有效元素数
G_positive = sum(w * max(g, 0))
G_negative = sum(w * max(-g, 0))
net_gain = G_positive - G_negative
gross_error_gain = G_positive + G_negative
cancellation = 1 - abs(net_gain) / gross_error_gain
```

`gross_error_gain=0`时抵消率不可用。预测绝对变化`sum(w*abs(OFF-ON))`是另一个量，不能替代抵消率分母。例如ON与OFF对称地跨过真实值时，预测改变而绝对误差不变。

保留原始全部验证行及12时距宏平均，不去重主结果。空间分为三个互斥且完备区域：

| 区域 | 定义 |
| --- | --- |
| `direct_report_nodes` | A：直接报告关联节点，包含算子支持范围外的直接关联节点。 |
| `potential_propagation_only_nodes` | B：候选结构包络中扣除A。 |
| `remaining_nodes` | C：A和候选包络以外的其余节点。 |

组内MAE使用该组各时距有效分母；全网贡献使用全网分母。A/B/C的全网净收益、正负项、预测变化贡献及权重占比均可加回全网，程序会核验恒等式。某组缺少任一时距有效目标时，组内宏平均为`null`，保留可用时距；全网分母下的空区域贡献仍为0。

另外报告`direct_outside_operator`、`outside_operator`、`candidate_potential`、`no_report_samples`等诊断组。这些组可能重叠，不能再相加。候选包络不等于已观测的事故影响范围。

## 方向和样本分析

- `bias_metrics`按**OFF预测残差**分为高估`over`、低估`under`、精确命中`exact`。OFF指同一B3-R检查点关闭D，不是原始B0。
- `movement_metrics`分为同侧、严格跨越目标、仅OFF命中、仅ON命中、两者均命中五类；精确零单列，不引入任意epsilon。
- 每条原始窗口与每个截止时间`t0`均报告组内误差、正负收益及对全网收益的贡献。窗口贡献加和与截止时间贡献加和分别恢复全网净收益。
- 唯一截止点敏感性给每行`1/该t0行数`权重，再重算各时距有效分母。它是单独的敏感性口径，不把同t0窗口自动视为同一事故或独立交通情境。

## 输出

| 文件 | 内容 |
| --- | --- |
| `analysis.json` | 全网/空间/方向分类的完整宏平均及逐时距统计、唯一截止点敏感性、来源与哈希。 |
| `README.md` | 自动生成的中文结果说明、A/B/C贡献表及改善/恶化各最多5个代表窗口。 |
| `gain_decomposition.csv` | 空间区域的宏平均与12时距结果。 |
| `bias_decomposition.csv` | 按OFF高估/低估/精确命中分解。 |
| `movement_decomposition.csv` | 按残差跨越/同侧/命中分解。 |
| `window_summary.csv` | 每个原始预测窗口及其贡献、最大预测变化。 |
| `cutoff_summary.csv` | 每个截止时间及其贡献，保留原始行数。 |
| `preservation.json` | 本次分析读取前后五个源文件的哈希。 |
| `failure.json` | 校验或分析失败时记录原因，不宣称完成。 |

CSV中的`contribution_to_global_macro_mae`在单时距行已包含`1/12`，所以12行加和等于对应macro行；组内`off_minus_on`仍是该时距的MAE差。`global_valid_weight_fraction`是宏指标有效元素权重占比，不是实际事故覆盖率。

## 结果怎样推动下一轮模型设计

1. 先看G+、G−、净收益、预测变化四个量，区分正负误差收益抵消与跨越真实值造成的误差不敏感。
2. 看A/B/C的全网贡献及组内收益。如果关联区域收益被其余区域损害抵消，优先研究作用范围/读出控制；这只是候选解释，需要匹配对照验证。
3. 看12时距方向及OFF高估/低估分组，判断是否值得检验时间条件化容量或调整修正方向。远期收益不证明事故扩散；也不能仅凭净收益小定位硬容量约束故障。
4. 若支持动态容量，先明确历史/未来块与容量时间对应，再用原B3-R、新接口静态容量、同接口动态容量进行对照，保持TIID与交换预算一致。

源NPZ没有站点ID/道路映射、事故事件身份或逐窗口逐边的交换轨迹。本工具不生成道路/事件因果归因，不把汇总交换量与个体误差强行关联。跨窗口/站点/时距不视为独立重复；本阶段无显著性检验、测试集选择、跨种子稳定性或真实通行能力恢复结论。

## 本地验收

已有本地工程探针的4条验证样本×496站已通过完整CLI；它只验证程序兼容旧产物，不代表服务器917条完整验证集的研究结果。11项独立测试全部通过（2.637秒），覆盖宏权重、互斥区域、空区域、正负收益抵消、跨越目标、重复截止时间、源指标不一致阻断、源保全及禁止导入Torch的CLI。Bash语法检查通过；既有模型、训练器、协议和路径探针未改动，续训身份不变。
