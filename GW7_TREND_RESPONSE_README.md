# 最后一项缓存诊断：连续水平匹配与历史趋势响应

本工具只使用CPU，复用R_CONTINUE BEST的预测，不训练或重新推理模型。目标是区分：高变化是否主要标记困难度，还是存在值得用结构试验检验的历史趋势响应模式。本轮不开发GW8模型，不扩展GW7预算。

## 服务器执行

在允许CPU计算的服务器节点，更新原GW7分支并运行。所有新增Python导入/语法检查及数值测试均留在服务器；本地只做静态检查和文件字节核对。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW7
conda activate igstgnn
git pull --ff-only origin research/graph-interaction-gw7

export GW7_RUN="$PWD/experiments/chronological_runs/contra_original_gw7_screen_s2025"
export GW7_DATA="$(realpath ../data/xtraffic/Contra_Costa)"
export GW7_TREND_DIR="$GW7_RUN/r_best_trend_$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
set -o pipefail

python -m unittest tests.test_gw7_trend_response -v && \
python -u -m experiments.graph_interaction_gw7.diagnose_trend_response \
  --run-dir "$GW7_RUN" \
  --data-dir "$GW7_DATA" \
  --output-dir "$GW7_TREND_DIR" 2>&1 | tee "$GW7_TREND_DIR.log"
```

默认自动定位与当前run及BEST身份匹配、已完成的剩余误差诊断，打印实际路径。要固定此前回传的某次目录，额外传入 `--history-dir "$GW7_ERROR_DIR"`；变量必须已指向正确目录。不会将LAST或其他run混作BEST。

成功结尾为 `GW7_TREND_RESPONSE_COMPLETE`，缓存pooled MAE应回放约13.618473855。输出目录必须全新，失败/中断后更换tag保留现场，不需要重跑模型。

先回传：

```bash
cat "$GW7_TREND_DIR/report.md"
cat "$GW7_TREND_DIR/matched_overall.csv"
```

进一步查看：

```bash
cat "$GW7_TREND_DIR/by_horizon.csv"
cat "$GW7_TREND_DIR/by_order_block.csv"
```

`by_node.csv`保存节点汇总，`constructed_pairs.csv`保存全部历史配对身份与连续特征，`unmatched_trend_groups.csv`保存未匹配的实际组内偏差，`input_manifest.json/summary.json`保存身份、协议与完成状态。

## 固定的定义与匹配规则

沿用训练输入拟合的每站尺度 `s=max(IQR,1)`，V与此前相同。历史趋势 `T=(最近3槽均值−最早3槽均值)/s`；`T>0.05`为UP，`T<−0.05`为DOWN，边界及其间为FLAT。只读取预测截止点以前的历史。

匹配限于同一节点、同一个作者顺序块，六块沿用837验证窗口的固定等分；不认证为真实日历分块。每个比较内不重复使用窗口×节点，配对不读取预测、真值、误差或目标有效性。

| 比较 | 左端/右端 | 主要连续控制与逐维卡尺 |
|---|---|---|
| volatility_high_vs_low | 训练阈值定义的高V/低V | 全窗口均值/s：0.05 |
| trend_up_vs_down_all_V | UP/DOWN | 最新观测流量/s：0.05；V：0.02；abs(T)：0.05 |
| trend_up_vs_down_high_V | 高V中的UP/DOWN | 同上，两端都必须位于训练最高V桶 |

方向比较以最新观测水平为主要水平控制，避免全窗口均值相同但上升/下降的当前流量不同。全窗口均值、最近3槽均值和最新槽水平的两端均值、平均/最大逐对差都输出；未受卡尺控制的量仍可能失衡。

匹配通过增广路径取得固定约束图中的最大配对数；候选按卡尺归一化L1距离和样本索引排序。它不保证全局最小距离，不能把它称为最优最近邻匹配。`double_caliper_sensitivity`预先固定为所有卡尺乘2，完整报告，不能挑更有利版本替代主版本。

## 评分与共同支持

配对身份固定用于全部12时距，各时距不重新匹配。每对只在两端共同有效的目标上评分；总体先在每对的共同有效时距内平均，再按配对等权平均。每个切片的节点×顺序块至少有5对可评分配对才计入，否则NA。规则不随结果放宽。

同时输出：纯历史候选数、目标可评分候选数、构造配对数、共同目标可评分数、五对门槛后的配对数、节点/节点块数、覆盖率和连续变量平衡。`matched_endpoint_fraction`是计入评分的两端单元数除以该比较可评分候选端点数；`matched_all_units_fraction`的分母为所有可评分窗口×节点单元。

主汇总按配对等权，另输出等节点结果。逐时距共同有效性可能不同，即使配对不变，可评分人群仍可能变化；必须看支持数。六块并非独立重复，不按数百万目标点给IID显著性检验。

## 如何作研究决定

方向比较的 `left_bias/right_bias`是UP与DOWN各自真实的 `prediction−truth`，不是两个活动组的bias差。`underresponse_score`（兼容字段 `aligned_error`）定义为 `(-UP_error+DOWN_error)/2`：正值与历史趋势欠响应相容，但必须同时看UP是否偏低、DOWN是否偏高、节点广泛性和覆盖率。

高V/低V比较的MAE差如果在连续水平匹配后仍为正，可保留“高变化更难”的线索；不等于能回收这么多模型收益。若UP低估、DOWN高估的模式在足够支持、卡尺平衡及多个时距下保留，可进一步检验原DIF内部水平/变化响应结构。若没有相应模式或支持过少，应停止将该模式作为设计依据。

这些偏差模式不能证明因果滞后或编码器失效，未来真实反转/均值回归仍是竞争解释。两种卡尺若结论相反，报告不确定，不再搜索第三种阈值。本次之后应作明确继续/停止决定，不无止境增加诊断或扩展旧GW7训练。

## 身份与执行边界

旧 `history_features.npz` 和训练参数未被旧summary逐一绑定哈希，因此本工具从可信发布X重新计算本地尺度、分箱和V/T/水平，逐值核对缓存；不重算邻域图特征。原发布object数组会反序列化包含测试载荷的完整文件，但只索引训练X与验证X/Y，测试不参与评价。

原模型与GW7历史执行文件均保持字节不变。checkpoint只检查文件哈希，不反序列化；只导入标准库、NumPy和旧的纯NumPy诊断模块，不导入Torch/模型。输入及工具源码哈希前后核对，完成summary最后写入；残留CSV本身不代表成功。
