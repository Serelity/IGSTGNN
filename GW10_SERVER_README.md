# GW10：剩余误差方向是否可由历史输入预测

发布分支 `research/conditional-residual-gw10`，继承已完成的 GW9。沿用现有 `IGSTGNN-GW9` 目录，保留已认证运行中的绝对路径；无需重新克隆、复制数据或迁移旧实验。

这是一个训练集拟合的中位数残差探针，用来决定是否值得设计下一代模型结构。它不是已经验证的新架构，也不启动失败的 GW9 joint。冻结 GW9 `SUM_CONTROL BEST`，先严格回放其验证预测，再对原训练集推理一次；随后全部查表拟合和评分使用 CPU/NumPy，不更新骨干或 SUM 读出头。

本地没有执行 Python、导入、语法检查、测试、推理或训练。服务器 launcher 先验证源码字节，再运行 38 项新增测试，成功后才执行正式流程；本次代码是否能在实际环境运行，以服务器输出为准。CPU 阶段同时保存多组预测和拟合表，预计需要数 GB 内存；尚无实际耗时或峰值内存测量。

## 服务器更新和运行

在原来的计算节点 `igstgnn` 环境执行。首次导出需要与 GW9 严格回放兼容的 PyTorch/CUDA 环境和 GPU；不申请新的训练轮数。此脚本不自动提交调度器作业，按服务器已有资源管理方式运行。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git fetch origin research/conditional-residual-gw10:refs/remotes/origin/research/conditional-residual-gw10
if git show-ref --verify --quiet refs/heads/research/conditional-residual-gw10; then
  git switch research/conditional-residual-gw10
else
  git switch -c research/conditional-residual-gw10 origin/research/conditional-residual-gw10
fi
git pull --ff-only origin research/conditional-residual-gw10
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW10_PROBE_RUN="$PWD/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025"
export GW10_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW10_TAG="$(date +%Y%m%d_%H%M%S)"
export GW10_CACHE_DIR="${GW10_PROBE_RUN}_gw10_cache_${GW10_TAG}"
export GW10_RUN="${GW10_PROBE_RUN}_gw10_probe_${GW10_TAG}"
export GW10_DEVICE="cuda:0"
export GW10_STAGE="all"

bash experiments/conditional_residual_gw10/run.sh
```

如果 Git 提示存在会被覆盖的本地修改，先保存修改并核对内容，不要使用 `reset --hard`。切换分支不改变旧实验目录；GW10 导出与结果目录是 GW9 原运行目录的同级新目录。

源运行被固定到用户已返回的 replayfix GW9 identity `5e20efbdcca907afda994889cdc460276fdd560e235047a11c20743494d36757`。程序从认证元数据读取实际 SUM BEST 头权重和完整 R 来源，不猜测 BEST 轮次。验证 y、mask、sample/node 轴、支持区、R 和 SUM 预测必须逐点精确一致；不通过时停止，不放宽容差。只有验证回放通过，才推理原顺序的 3909 个训练样本。原训练/验证 batch 大小仍为 48。

## 查看与回传

```bash
cat "$GW10_RUN/report.md"
cat "$GW10_RUN/screening.json"
cat "$GW10_RUN/coverage.csv"
```

先回传这三个文件。需要定位收益时再看：

```bash
cat "$GW10_RUN/comparisons.csv"
cat "$GW10_RUN/audit_comparisons.csv"
cat "$GW10_RUN/directions.csv"
cat "$GW10_RUN/by_horizon.csv"
cat "$GW10_RUN/by_history.csv"
```

换 shell 后变量不会自动保留。可重新 export 原先打印的 `GW10_RUN`，或定位报告：

```bash
find /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs \
  -maxdepth 2 -path '*_gw10_probe_*/report.md' -print
```

## 缓存复用与中断

完整缓存的最后一个产物是 `cache.json`。缓存由来源证书、所选头、源码、数据、完整回放、train/val NPZ 哈希共同绑定；后续 CPU 阶段逐项核验。缓存可复用，避免重复 GPU 推理。

如果导出已经完成，但 CPU 阶段被杀：保留原 `GW10_CACHE_DIR`，设置新的结果目录，然后只执行 CPU 阶段：

```bash
export GW10_STAGE="fit"
export GW10_RUN="${GW10_PROBE_RUN}_gw10_probe_retry_$(date +%Y%m%d_%H%M%S)"
bash experiments/conditional_residual_gw10/run.sh
```

如果 `cache.json` 尚未产生，说明缓存未完整提交。保留不完整目录，换新的 `GW10_CACHE_DIR` 与 `GW10_RUN`，重新导出；当前版本不从中间推理 batch 续跑。CPU 拟合没有训练轮次或优化器状态，也不支持中途接续。已完成的结果不覆盖，重复评分不作为新证据。

## 预先固定的设计

所有修正拟合训练集 `y − SUM` 的中位数，预测时加到 SUM。流量分组只使用起点最后流量，内部边界固定为 `0/5/20/50/100/200/314`，两端覆盖无穷范围。历史变化、邻域分歧的中位数/四分位切点只用对应拟合集的历史输入确定；不按验证结果调参。

| 名称 | 修正条件 |
|---|---|
| SUM | 冻结的原始 SUM，无新增修正 |
| GLOBAL | 全训练集残差中位数 |
| HORIZON | 各时距 |
| NODE_HORIZON | 节点 × 时距 |
| ORIGIN | 节点 × 时距 × 起点流量区间，核心基础对照 |
| OWN2 | ORIGIN 加自身变化幅度高/低 |
| NEIGHBOR2 | ORIGIN 加邻域变化分歧高/低 |
| OWN4 | ORIGIN 加自身变化幅度四分位，与 JOINT 同为四个子状态 |
| JOINT | ORIGIN 加自身变化高/低 × 邻域分歧高/低，唯一主候选 |

GLOBAL 为原始中位数；其余每个 cell 至少 64 个有效点且覆盖至少 3 个长度 128 的作者顺序块，才使用 `n/(n+128)` 加权的 cell 中位数，其余权重给父层。支持不足时完全回退父层：`HORIZON→GLOBAL`、`NODE_HORIZON→HORIZON`、`ORIGIN→NODE_HORIZON`、四个状态探针→ORIGIN。邻域信息不可用也回退 ORIGIN。块覆盖不是独立样本数；实际占用 cell 和支持见模型与 coverage。相同名义状态数不保证观测 cell 数或有效自由度完全相同。

原训练样本 `0:1826` 拟合一次审计映射，排除 `1826:1954` 的 128 个样本，在 `1954:3909` 检查映射迁移。之后始终按预定方案在全训练集重新拟合，统一评分验证集一次，不根据审计或验证结果切换赢家。审计切点也仅来自前段拟合历史。原 R 和 SUM 已用过训练标签，因此这是新增映射的留出审计，不是整个模型的样本外残差实验。原作者样本顺序尚未被独立认证为绝对日历顺序，隔离间隔也不能保证消除窗口重叠或其他依赖。

固定通过条件全部满足才标记 `eligible_for_new_model_design=true`：JOINT 在验证集相对 SUM 和 ORIGIN 的 MAE 均至少改善 0.5%；严格优于 OWN2、NEIGHBOR2、OWN4；训练顺序审计优于 ORIGIN 和 OWN4；验证高流量 ≥314、时距 7–12、真实流量 50–200 三组 MAE 均不劣于 SUM。真实流量只用于事后守护指标，不进入预测条件。

通过只支持进一步设计条件修正模型，不能证明邻域物理机制、架构创新、同训练预算优势或独立确认。失败不改变候选、分箱、收缩参数或预算。只有误差尺度关联、方向命中率或某个节点的事后极值较好，不算通过。

## 产物和数据边界

`protocol.json` 在拟合前保存固定规则；`audit_model.json` 与 `audit_cases.csv`/`audit_comparisons.csv` 保存内部审计；`model.json` 和 `fitted_before_validation.json` 在验证评分前提交拟合表及哈希。其余输出包括 `analysis.json`、`cases.csv`、`comparisons.csv`、`directions.csv`、`groups.csv`、`by_horizon.csv`、`by_node.csv`、`by_sample_block.csv`、`by_history.csv`、`coverage.csv`、`screening.json`、`report.md`。

历史特征沿用 GW9 诊断定义：反归一化后的平均绝对逐步变化，以及节点最后变化与去对角、按行归一化原始邻接加权邻域变化的绝对差。原始行邻域不被解释为物理上游。无邻居节点记不可用。固定作者数据 object 数组会整体反序列化，包括 test 对象；只访问训练/验证内容，不构造 test loader，不使用 test 做拟合或评价。沿用原发布归一化参数，不重新估计。
