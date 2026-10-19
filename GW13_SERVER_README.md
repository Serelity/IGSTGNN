# GW13：检验历史形态是否提供额外误差信息

发布分支 `research/trajectory-diagnostic-gw13`，继续使用服务器原 `IGSTGNN-GW9` checkout 和已有 GW9/GW10 认证文件。这是固定 SUM 预测上的 CPU NumPy 信息诊断：从缓存重新拟合受控中位数查表，检验粗略的历史形态是否比变化幅度和净方向提供额外收益。它不续训 GW12，也不启动新的神经网络。

本地仅做源码静态、字节与 Git 检查，没有执行 Python、导入、语法检查或测试。入口会在服务器先核验源码，再运行四个新测试模块，然后执行诊断；是否通过以服务器实际日志为准。

## 更新与运行

沿用现有 `igstgnn` 环境即可，无需 GPU、安装依赖或新模型推理。脚本固定 NumPy 相关 CPU 线程为 3；可预留 3 CPU、16 GB CPU 内存，这是资源建议，没有测量本轮耗时和峰值内存。脚本不提交调度器。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git fetch origin research/trajectory-diagnostic-gw13:refs/remotes/origin/research/trajectory-diagnostic-gw13
if git show-ref --verify --quiet refs/heads/research/trajectory-diagnostic-gw13; then
  git switch research/trajectory-diagnostic-gw13
else
  git switch -c research/trajectory-diagnostic-gw13 origin/research/trajectory-diagnostic-gw13
fi
git pull --ff-only origin research/trajectory-diagnostic-gw13
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW13_PROBE_RUN="/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025"
export GW13_CACHE="/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025_gw10_cache_20261019_155428"
export GW13_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW13_RUN="$PWD/experiments/chronological_runs/contra_gw13_trajectory_s2025_$(date +%Y%m%d_%H%M%S)"

bash experiments/trajectory_diagnostic_gw13/run.sh
```

Git 若因本地修改拒绝切换，先核对并保存修改。所有来源路径应继续指向原认证文件。`GW13_CACHE` 必须显式设置；脚本在执行任何 Python 前检查输入、绝对路径和全新输出目录。`GW13_PROBE_RUN` 可省略，默认当前仓库内上述原 GW9 运行名；`GW13_DATA` 默认仓库父目录的 `data/xtraffic/Contra_Costa`；`GW13_RUN` 可省略，默认自动生成带时间戳的新目录。建议显式保留 `GW13_RUN`，便于查看结果。

本轮没有训练检查点或 resume 模式，也不读取 `GW12_RESUME`、`GW12_EVALUATE_ONLY` 等旧变量。已有输出目录不会覆盖。若进程中断，保留原目录及失败信息，选择全新 `GW13_RUN` 后重跑同一固定诊断；不要删除来源认证文件或修改候选、切点、支持阈值来补跑。

也可直接调用固定入口，参数只有四个路径：

```bash
python -m experiments.trajectory_diagnostic_gw13.run \
  --probe-dir "$GW13_PROBE_RUN" --cache-dir "$GW13_CACHE" \
  --data-dir "$GW13_DATA" --output-dir "$GW13_RUN"
```

## 固定问题与输入边界

GW12 的末值加平均绝对变化状态读出未通过筛查。本轮检验一个更窄的假设：两个历史窗口即使末值、变化幅度和净变化方向接近，变化的先后顺序是否仍对应不同的 SUM 剩余误差？这里只使用 12 点历史的粗分段形态，不能据此判断完整轨迹模型是否有效。

原始历史文件必须与认证数据身份中的每个文件哈希相符，才允许使用继承的可信 object 数组加载器。它会整体反序列化原始数组，包含测试对象；程序只访问原训练 3909 个、验证 837 个样本的 `x_data`，提取第 0 列交通值，不访问 `y_data` 或测试样本。归一化沿用固定 mean/std。末值和平均绝对变化先按 float64 反归一化计算，再转 float32，必须与 GW10 缓存逐点、dtype、shape 完全相同，不放宽容差。

新增历史描述如下：

| 特征 | 固定定义 | 用途 |
|---|---|---|
| 净方向 | `(最后值−最初值)/总绝对变化`，常量历史为 0 | 幅度分组内四分位净方向对照 |
| 粗形态 | 前 4、中 4、后 4 点均值，按中减前、后减中的正负编码四类 | 唯一候选的四种状态 |
| 转折位置 | 3 点滑动均值的相邻斜率严格异号；多个选绝对斜率变化最大者，同强度选最早 | 仅分组描述，不参与拟合 |

粗形态以 `>=0` 归入非负侧，常量序列归入双非负类，不能把它称作严格上升。转折位置是原 12 点的 0-based 索引 2–9，无严格反转时为 −1；零斜率不跳过，也不算严格反转。

## 受控拟合与内部审查

所有新表从固定缓存中的训练标签拟合。首先按认证作者样本顺序划分：拟合前缀 `[0,1826)`，排除间隔 `[1826,1954)`，审查尾段 `[1954,3909)`。每次拟合都重新估计父级 OWN4、切点和新共享表，禁止把全训练 OWN4 拿来计算前缀拟合残差。

这一尾段仅对新查表拟合是未参与拟合的数据；源 SUM 已经用过完整训练集。作者顺序未被认证为日历顺序，也没有原始时间戳证明窗口之间无重叠，间隔 128 个样本不能保证物理时间隔离。该步骤不能当成完整模型的独立验证。

| 案例 | 相对固定 SUM 的修正 |
|---|---|
| SUM | 无 |
| ORIGIN | 重新拟合的节点×预测步×末值流量档父表 |
| OWN4 | 重新拟合的节点×预测步×末值流量 8 档×变化幅度 4 档父表 |
| CONTEXT | OWN4 加跨节点共享的预测步×末值流量 8 档×幅度 4 档残差表 |
| MAGNITUDE16 | 相同 CONTEXT 再按每个幅度档内部的幅度四分位细分，作为容量对照 |
| DIRECTION16 | 相同 CONTEXT 再按每个幅度档内部的净方向四分位细分，作为顺序信息对照 |
| TRAJECTORY16 | 相同 CONTEXT 再按四种粗形态细分，唯一候选 |

三种子表每个预测步都有相同的 128 个名义单元，即 8 个末值档×4 个幅度档×4 个子状态。各表有效占用、支持覆盖率和有效容量仍可能不同，不能把相同表尺寸等同于公平性已完全成立。新共享上下文不包含日内时间或星期，因此本轮不复刻 GW12 同参数量的时钟条件神经读出对照。

新共享表统一拟合 `r0 = y − SUM − OWN4`。CONTEXT 向零收缩；三种子表直接拟合同一个 r0 并向 CONTEXT 收缩。最终预测是 `SUM + OWN4 + 一张共享表`，不会同时再加一遍 CONTEXT。沿用每单元至少 64 个有效点、至少 3 个作者顺序块、块长 128、收缩强度 128；不满足支持时回退父表。块和点不能视为独立观测。

三个子表在审查尾段各自直接使用合格当前层单元的有效点占比都必须至少为 25%。这是固定的操作性支持阈值，不是统计置信标准。覆盖不足时结果是当前探针支持不足，不能解释成轨迹没有信息。

候选 TRAJECTORY16 必须在内部审查中同时严格优于 OWN4、CONTEXT、MAGNITUDE16、DIRECTION16，并通过上述覆盖条件。只有全部通过，才允许按完整 3909 个训练样本重新拟合一次，先保存模型、切点及哈希，再对新修正进行一次开发验证评分。

审查失败时不对新修正做验证评分，也不执行全训练重拟合。但继承的 GW9/GW10 认证流程仍加载验证标签，并复算固定历史 SUM 指标以验证缓存；不能把整个程序描述成未读取验证标签。

## 最终门槛与结果

若内部审查通过，开发验证中的唯一候选必须同时满足：相对 SUM、OWN4、CONTEXT 各改善至少 0.5% MAE；严格优于 MAGNITUDE16 和 DIRECTION16；真值 ≥314、预测步 7–12、50 ≤ 真值 <200 三组的 MAE 均不劣于 SUM。未来真值只用于评分组，不用于构造特征或切点。

通过只表示可以继续设计、验证相应结构，不代表新模型创新、因果机制、统计显著性或独立确认；本轮不评价测试集。失败不授权改候选、门槛、延长 GW12 或自动启动神经训练。

先回传以下三个文件：

```bash
cat "$GW13_RUN/report.md"
cat "$GW13_RUN/screening.json"
cat "$GW13_RUN/coverage.csv"
```

需要查看形态与转折位置对应的剩余误差时：

```bash
cat "$GW13_RUN/audit_strata.csv"
cat "$GW13_RUN/audit_residual_associations.csv"
cat "$GW13_RUN/audit_paired.csv"
```

`audit_residual_associations.csv` 是残差中位数、正修正比例等描述性相关，不证明有独立预测收益。`audit_paired.csv` 给出预测步和作者顺序块的候选与 OWN4 配对差异，块仍不代表独立样本或日历日期。

完整审查与方法记录在 `analysis.json`、`protocol.json`、`audit_decision.json` 和 `audit_model.json`。内部审查通过后才额外产生 `model.json`、`fitted_before_validation.json` 及 `validation_cases.csv`、`validation_groups.csv`、`validation_strata.csv`、`validation_residual_associations.csv`、`validation_paired.csv`。输入改变、认证失败或中断的部分产物不算完整结果。
