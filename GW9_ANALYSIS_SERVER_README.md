# GW9 读出结果诊断：服务器运行说明

发布分支仍为 `research/branch-readout-gw9`。本次新增离线分析入口，读取已经完成的 GW9 frozen probe；不训练模型，不重新前向，不申请 GPU，也不修改现有 probe、源码证书或筛查结果。使用当前 `igstgnn` 环境中的 NumPy 与 Python 标准库。

GW9 的分支差值读出未通过同容量 SUM 对照，本次诊断不会把 `eligible_for_joint_screen=false` 改为 true。当前任务是分解 SUM 的收益、比较 BRANCH 与 SUM 的修正，并检查剩余误差与已有输入历史特征的关联，供后续设计新假设。

本地只执行静态审阅和文件字节核对，未运行 Python、导入、语法检查、测试或模型。新增测试和实际数据分析由下面的服务器命令执行；运行成功与具体诊断结论以服务器日志和产物为准。

## 更新并运行

在服务器执行。`GW9_ANALYSIS_DIR` 位于 probe 目录旁边，每次使用新目录；不要放入已有 GW6/GW7/GW8/GW9 运行目录内部。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git pull --ff-only origin research/branch-readout-gw9
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW9_REPO_DIR="$PWD"
export GW9_PROBE_RUN="$PWD/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025"
export GW9_ANALYSIS_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW9_ANALYSIS_TAG="$(date +%Y%m%d_%H%M%S)"
export GW9_ANALYSIS_DIR="${GW9_PROBE_RUN}_readout_analysis_${GW9_ANALYSIS_TAG}"

bash experiments/branch_readout_gw9/run_analysis.sh
```

此入口直接进行 CPU 分析，不调用 `sbatch` 或原来的 `run.sh`。按所在服务器对 CPU 作业的管理要求运行。顺序为独立分析源码包校验、四组新增单元测试、来源认证、已有预测分析；任一步失败即停止。无需设置 `GW9_GW8_RUN`、`GW9_DEVICE` 或 `GW9_DIRECT`。

如果暂时没有原始数据，只分析已保存预测：

```bash
unset GW9_ANALYSIS_DATA
export GW9_ANALYSIS_TAG="$(date +%Y%m%d_%H%M%S)"
export GW9_ANALYSIS_DIR="${GW9_PROBE_RUN}_readout_analysis_${GW9_ANALYSIS_TAG}"
bash experiments/branch_readout_gw9/run_analysis.sh
```

未提供数据时，报告会明确跳过输入历史关联。预测归档本身不足以判断残差是否可由历史输入预测。

## 查看与回传

```bash
cat "$GW9_ANALYSIS_DIR/report.md"
cat "$GW9_ANALYSIS_DIR/analysis.json"
cat "$GW9_ANALYSIS_DIR/comparisons.csv"
cat "$GW9_ANALYSIS_DIR/corrections.csv"
```

先回传 `report.md`。需要核对细节时，再回传对应 CSV 和 `analysis.json`。这些变量已由父 shell 显式 export；换 shell 后可以定位输出：

```bash
find /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs \
  -maxdepth 2 -path '*_readout_analysis_*/report.md' -print
```

若输出目录已存在，保留它，换新的 `GW9_ANALYSIS_TAG` 与 `GW9_ANALYSIS_DIR` 重跑。分析不覆盖已有目录，不续跑训练，也不会追加训练预算。

## 产物

| 文件 | 用途 |
|---|---|
| `report.md` | 来源、全局比较、修正形态、时距/流量/道路/支持分组、历史关联和限制 |
| `analysis.json` | 认证来源、指标、分析规则及是否执行历史输入分析 |
| `cases.csv` | FROZEN_R、SUM_CONTROL、BRANCH_READOUT 的总体误差 |
| `comparisons.csv` | SUM−R、BRANCH−R、BRANCH−SUM 的配对收益与损失 |
| `corrections.csv` | 三组预测差值的均值、方差、幅度和与原误差的关系 |
| `by_horizon.csv` | 全部 12 个预测时距 |
| `by_node.csv` | 全部节点，保留原节点标识，不仅展示收益最大的节点 |
| `by_flow.csv` | 固定未来真实值区间，仅供事后诊断 |
| `by_predicted_flow.csv` | 固定 SUM 预测值区间 |
| `by_road.csv` | 原归档道路标签分组 |
| `by_support.csv` | 原归档事故直接支持分组 |
| `by_sample_block.csv` | 原作者样本顺序中每 128 个样本的连续块 |
| `partition_cancellation.csv` | 时距、节点等互斥分组的收益与损失抵消，与点级抵消分开记录 |
| `history_features.csv` | 提供数据时的历史变化、邻域分歧特征、训练输入切点及关联摘要 |
| `by_history.csv` | 提供数据时按历史输入特征分组的误差 |
| `matched_history.csv` | 提供数据时，在相同时距和起点最后流量区间内比较历史特征组 |
| `matched_contrasts.csv` | 提供数据时，同一分层内 Q4−Q1 的 SUM MAE、偏差、配对收益差及两端有效点数 |

默认分析 BEST，并校验 LAST 的来源、轴和归档关系。BEST/LAST 预测相同只算同一份预测证据，不作为重复实验或独立确认。原样本轴的绝对日历顺序未被独立认证，因此连续样本块不称为真实日历时间块。

流量边界固定为 `-inf/0/5/20/50/100/200/314/inf`，各组左闭右开，负值单列以覆盖全部有限预测，不根据本次收益调整。真实值分组使用未来标签，仅用于事后误差诊断；不能据此开发可部署路由规则。SUM 预测分组来自已训练模型输出，仍属于开发验证诊断。所有分组沿用原始预测轴和原有有效值 mask。

提供 `GW9_ANALYSIS_DATA` 后，程序核对原数据身份和字节哈希，仅使用认证 train/validation 划分的历史输入；分组四分位切点只取训练历史。原作者发布格式是 object 数组，载入时会反序列化其中的 test 对象，但不读取或使用 test 历史和标签进行分析，不评估 test。

历史描述量包括：最后流量、历史平均绝对变化、窗口标准差、最后一步变化、窗口末值减首值、节点最后变化与邻域平均最后变化的绝对差、邻域最后变化的加权标准差。邻域直接使用原始 `A_ij` 的行邻接，去对角后按行归一化；不解释为物理上游或传播时延。无邻居节点的两项邻域描述量记为不可用，而不是零。

`matched_history.csv` 仅对历史平均绝对变化与邻域变化分歧两项，按预测时距及起点最后流量固定区间分层；不控制节点身份，也不表示因果调整。`matched_contrasts.csv` 给出同一分层中 Q4−Q1 的原始差值和两端有效点数；报告中的 Higher/Lower/Equal 只计两端均非空的分层。跨时距及特征涉及重叠样本，这些符号计数不是独立重复或显著性检验。训练四分位边界重复时保留空组，不按验证误差移动切点。

报告展示 BRANCH 相对 SUM 全局贡献最大的五个受益节点和五个受损节点，用于定位后续检查位置；`by_node.csv` 保留全部节点。不能把这些事后极值直接作为部署节点选择规则。

误差与输入历史的相关性不等于可预测技能或因果机制。报告不拟合校准量、混合系数、路由器或新模型，也不应用全局或分组均值修正。任何后续模型方案都需要新的明确假设、公平对照与预先固定的失败条件。

## 独立命令

需要绕过 launcher 定位问题时，仍应先做源码校验与新增测试：

```bash
python verify_gw9_analysis_bundle.py
python -m unittest tests.test_gw9_readout_metrics tests.test_gw9_readout_auth \
  tests.test_gw9_readout_history tests.test_gw9_readout_analysis -v

python -u -m experiments.branch_readout_gw9.analyze_readouts \
  --probe-dir "$GW9_PROBE_RUN" \
  --output-dir "$GW9_ANALYSIS_DIR" \
  --data-dir "$GW9_ANALYSIS_DATA"
```

没有原数据时省略最后一行 `--data-dir`，并去掉上一行末尾的续行反斜杠。原有 `GW9_SOURCE_MANIFEST.json` 和训练执行文件全部保留；新增 `GW9_ANALYSIS_SOURCE_MANIFEST.json` 独立认证分析文件及其继承来源。
