# R_CONTINUE BEST 剩余误差与历史变化诊断

这是 GW7 之后的只读诊断，不新增训练或推理，不修改模型、checkpoint、原评价归档和旧执行源码。仅依赖标准库与 NumPy，不导入 Torch，不使用 GPU。新增数值测试必须在服务器执行；本地没有执行 Python、导入/语法检查或测试。

## 服务器运行

在允许 CPU 计算的服务器节点执行。以下路径对应已回传的 `contra_original_gw7_screen_s2025`，不要误用旧 GW6 run 或只有工程小样本的目录。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW7
conda activate igstgnn
git pull --ff-only origin research/graph-interaction-gw7

export GW7_RUN="$PWD/experiments/chronological_runs/contra_original_gw7_screen_s2025"
export GW7_DATA="$(realpath ../data/xtraffic/Contra_Costa)"
export GW7_ERROR_DIR="$GW7_RUN/r_best_remaining_$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
set -o pipefail

python -m unittest tests.test_gw7_remaining_error -v && \
python -u -m experiments.graph_interaction_gw7.diagnose_remaining_error \
  --run-dir "$GW7_RUN" \
  --data-dir "$GW7_DATA" \
  --output-dir "$GW7_ERROR_DIR" \
  --chunk-size 16 2>&1 | tee "$GW7_ERROR_DIR.log"
```

脚本自动选择与当前 run 身份、R_CONTINUE BEST 检查点、完整五轮预算一致的最新合格 BEST 评价，打印实际路径。不会把 LAST 当 BEST。如果要指定某次评价，额外传入 `--best-dir /absolute/path/to/eval_best_directory`；不猜写时间戳。

目标是 BEST 第4轮、文件 `best_epoch_004_ef6b4b17370d.pt`，已回传 checkpoint SHA256 为 `568ff5e0dcea220cb28e2e9bec794ef1894411e8dd8acd4bf427bfb2e425ca65`。脚本从当前 summary 与 metadata 核对所选身份，不反序列化权重。pooled MAE 应回放约 **13.618473855**。

成功结尾为 `GW7_REMAINING_ERROR_COMPLETE`。输出目录必须全新；被中断或失败后保留该目录和日志，更换 tag 后重跑分析，不需要重跑模型或删除旧实验。chunk-size 只控制 CPU 特征内存分块，不改变指标定义或模型 batch。

若重开终端丢失环境变量，可恢复 run 并查找已有分析目录：

```bash
export GW7_RUN="/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW7/experiments/chronological_runs/contra_original_gw7_screen_s2025"
find "$GW7_RUN" -maxdepth 2 -path '*/r_best_remaining_*/report.md' -print
```

## 先查看并回传什么

```bash
cat "$GW7_ERROR_DIR/report.md"
cat "$GW7_ERROR_DIR/conditional_contrasts.csv"
cat "$GW7_ERROR_DIR/by_horizon.csv"
cat "$GW7_ERROR_DIR/by_order_block.csv"
```

优先回传前两项；逐时距和顺序块结果较长，可以作为附件补充。另保留 `summary.json`、`input_manifest.json`、`fitted_training_parameters.json`、`node_scales.csv`、`conditional_cells.csv` 和 `history_features.npz`。

## 事先固定的分析定义

仅流量历史通道参与变化计算，时钟两列不当作交通变量。训练索引0–3908，验证索引3909–4745，按CSV原496站点顺序；用原float32反标准化逐行精确核对验证真值与已有归档，再用float64做统计。沿用归档的全局null有效mask，真实历史零不当缺失。

- 每站尺度 `s=max(IQR(训练输入流量),1)`；窗口有重叠，统计对象是训练窗口中的观测暴露，未宣称已去重。尺度和阈值不使用验证/测试拟合。
- 历史水平 L：窗口流量均值；按各站训练期L三分位分桶。
- 本站变化幅度 V：`mean_t abs(diff(x_t/s))`。
- 邻域分歧 D：`mean_t sum_j P_ij * abs(diff(x_j/s_j)-diff(x_i/s_i))`。先逐边取绝对值，再汇总，避免邻居反向变化相互抵消。
- 邻域变化幅度 V_neighbor：`sum_j P_ij V_j`。
- 相对分歧 Q：`D/(V+V_neighbor)`，分母为0时定义0；完全无活动窗口不参与Q条件比较。

主图为去掉self后的 `row_normalize(A+A.T)`，两个敏感性方向为 `row_normalize(A)` 和 `row_normalize(A.T)`。行i表示接收者读取列j，不解释成已认证的物理上下游。无邻居标不可用，不记作低分歧。

除按站点拟合的水平桶外，V/D/Q和邻域V使用训练期全部站点的尺度化三分位。重复阈值合并；等于阈值归低桶，常数特征/空桶如实记录，不抖动拆桶。原尺度和站点尺度化误差同时输出。

## 如何读条件对照

`conditional_contrasts.csv` 中：

1. `volatility_given_level`：同节点、同水平桶，高V相对低V的误差差。
2. `symmetric__D_given_level_local`：再控制本站V，看邻域D。
3. `symmetric__D_given_level_local_neighbor`：同时控制本站和邻域V，检查是否只是邻域更活跃。
4. `symmetric__Q_given_level_local_neighbor`：检查相对分歧，而非只看绝对变化幅度。
5. `row_A/row_AT`：相同定义在两个预测依赖方向上的敏感性。

先对每个窗口×节点的有效12时距求平均误差，再在同节点/控制桶内比较训练定义的最高与最低暴露桶；每端至少5个窗口，否则不匹配。跨匹配单元按 `min(n_low,n_high)` 加权。`weighted_mae_high_minus_low>0` 表示高暴露端剩余误差更大；scaled列是除以训练站点尺度后的相同对照。

必须同时看共同支持覆盖率 `matched_all_fraction`、节点数、连续控制量两端均值及差额。粗桶内仍可能失衡，不能称为已经排除幅度影响。单元不够就是证据不足，不能把NA当成“没有关系”。

`by_order_block.csv` 将作者验证顺序固定分六块，各块独立重建共同支持；阈值不重拟合。块内仍要求每端至少5个窗口，严格分层可能大量NA。各块匹配人群可能不同、样本可能重叠，不能把六块当六个独立重复。`by_horizon.csv` 使用原有12个真实预测时距分别报告同样的条件对照。

若D在控制邻域V后明显减弱，应优先考虑邻域活动幅度解释。若D/Q在足够共同支持下仍有相似模式，才有理由进一步做相对变化消息的受控训练对照。这份诊断没有自动通过阈值，不证明因果机制，也不直接决定GW8结构。

## 数据与执行边界

脚本读取已有预测缓存，不生成修正预测。反序列化原发布 `incident_all.npy` 会包含未使用的测试载荷；代码只索引训练X与验证X/Y，不构建或评价测试批次。原始对象数组只有在固定发布文件哈希核对后才加载。

输入/源码/归档/所选权重的字节哈希在分析前后核对；保持原GW7的执行文件和922文件清单不变，新增工具不会要求重新训练已有run。运行成功仅说明缓存与分析检查完成，不能等同于新模型有效。
