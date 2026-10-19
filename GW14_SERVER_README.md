# GW14：重放历史形态结果并匹配查表支持预算

发布分支 `research/trajectory-check-gw14` 继承原 GW13 提交 `357656fd0bcfac552f3f0074e4fe6dd1848d7092`，继续使用服务器现有 `IGSTGNN-GW9` checkout、原 GW9/GW10 认证文件和已经完成的 GW13 结果。GW14 先认证并重放原 GW13 的 audit 与 development 结果，再检查更严格支持预算下粗历史形态是否仍优于幅度和净方向。原 GW13 的模型、切点、父表、CONTEXT 和失败筛查结论均保留。

这是固定 SUM 缓存上的 CPU NumPy 中位数查表诊断。每个匹配拟合阶段新拟合 MAGNITUDE、DIRECTION、TRAJECTORY 三张共享统计表；audit 不通过时共 3 次新统计拟合，通过后增加全训练阶段，共 6 次。旧 GW13 重放不重新拟合旧表。没有新神经推理、梯度或优化器步骤。

本地只做源码静态、字节与 Git 检查，没有执行 Python、导入、语法检查或测试。服务器启动器先核验源码，再运行四个新测试模块，然后执行诊断；是否通过以服务器实际日志为准。

## 更新与运行

沿用现有 `igstgnn` 环境，无需 GPU 或新增依赖。脚本固定 NumPy 相关 CPU 线程为 3；可预留 3 CPU、16 GB CPU 内存，这是资源建议，尚无本轮耗时或峰值内存实测。脚本不提交调度器。

必须显式选择原来已经完成的 GW13 输出目录，不能自动挑选最新目录。若还在运行 GW13 时的同一个 shell，先确认 `GW13_RUN` 仍是那个原始目录，再执行：

```bash
: "${GW13_RUN:?GW13_RUN must name the completed original GW13 run}"
export GW14_GW13_RUN="$GW13_RUN"
```

如果换了新 shell，可先查找候选 `analysis.json`，核对相应目录中的原报告与来源记录，再将选定的完整目录绝对路径赋给 `GW14_GW13_RUN`：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
find "$PWD/experiments/chronological_runs" -maxdepth 2 -type f -path '*gw13*/analysis.json' -print
# 按核对后的实际路径执行：export GW14_GW13_RUN="/原GW13结果目录的绝对路径"
```

设置好 `GW14_GW13_RUN` 后：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git fetch origin research/trajectory-check-gw14:refs/remotes/origin/research/trajectory-check-gw14
if git show-ref --verify --quiet refs/heads/research/trajectory-check-gw14; then
  git switch research/trajectory-check-gw14
else
  git switch -c research/trajectory-check-gw14 origin/research/trajectory-check-gw14
fi
git pull --ff-only origin research/trajectory-check-gw14
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW14_PROBE_RUN="/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025"
export GW14_CACHE="/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025_gw10_cache_20261019_155428"
export GW14_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW14_RUN="$PWD/experiments/chronological_runs/contra_gw14_trajectory_check_s2025_$(date +%Y%m%d_%H%M%S)"

bash experiments/trajectory_check_gw14/run.sh
```

Git 若因本地修改拒绝切换，先核对并保存修改。所有来源路径应继续指向原认证文件。`GW14_GW13_RUN` 和 `GW14_CACHE` 必须显式设置为绝对路径；前者至少包含 `analysis.json`、`model.json`、`audit_model.json`、`screening.json`，后者包含 `cache.json`、`train.npz`、`val.npz`。后续 Python 认证还会核验相关来源记录与字节身份，文件存在不等于认证通过。

`GW14_PROBE_RUN` 可省略，默认当前仓库内上述原 GW9 运行名；`GW14_DATA` 默认仓库父目录的 `data/xtraffic/Contra_Costa`；`GW14_RUN` 可省略，默认生成带时间戳的新目录。脚本在任何 Python 执行前检查这些输入、绝对路径及全新输出目录。建议显式保留 `GW14_RUN`，便于回传结果。

GW14 不读取旧 `GW13_*`、`GW12_*` 变量；上面同 shell 的显式赋值只是把已核对的旧路径传给新变量。已有输出目录不会覆盖，也没有 resume 模式。若中断，保留原目录及失败信息，选择全新 `GW14_RUN` 后重跑固定诊断，不更改来源文件、候选或门槛。

也可直接调用入口，五个路径参数均为必填：

```bash
python -m experiments.trajectory_check_gw14.run \
  --gw13-dir "$GW14_GW13_RUN" --probe-dir "$GW14_PROBE_RUN" \
  --cache-dir "$GW14_CACHE" --data-dir "$GW14_DATA" --output-dir "$GW14_RUN"
```

## 原 GW13 的认证重放与交叉分组

首先认证原 GW13 的模型、来源记录和认证缓存，再按已保存的 audit、全训练模型重放原结果。这里不会重估原父表、切点或 CONTEXT，也不会覆写原 GW13 的失败筛查。旧 development 重放及其交叉分组属于已有方法复核，不受后面新匹配 audit 门槛控制；不能将 GW14 描述为完全不读取验证标签或不重算旧 development。

沿用原 12 点历史的四类粗形态、四类严格转折组和起点末值流量 8 档，输出 `shape4 × turn4 × lastflow8` 交叉收益；另输出 `shape × turn × lead` 和作者顺序块结果。转折组为无转折、早期位置 2–4、中期位置 5–6、后期位置 7–9。位置使用原 12 点历史的 0-based 索引。三点滑动均值相邻斜率必须严格异号；零斜率不跳过，多个转折取斜率变化绝对值最大者，同强度取最早者。

粗形态仍由前 4、中 4、后 4 点均值的两段差值编码，差值为零归入非负侧；因此“双非负”不等于严格上升。所有流量档、形态和转折均来自预测起点及之前的历史；未来真值仅用于误差评分与既定评分组，不用于切点或匹配。交叉单元和作者块的收益为描述性结果，不用于选择不同候选、重定阈值或证明统计显著性。

原始历史必须通过已有数据身份哈希认证。沿用可信 object 数组加载器会整体反序列化原始数组，其中包含测试对象；程序只索引原训练和验证样本的历史，不访问测试样本或对测试集评分。缓存认证流程使用验证标签复算原 SUM 指标。

## 更严格的训练支持匹配

每个阶段均复用对应原 GW13 模型的 OWN4 父表、训练切点和 CONTEXT，仅新拟合三张匹配共享表。audit 使用原前缀 `[0,1826)` 拟合新表，排除 `[1826,1954)`，在原尾段 `[1954,3909)` 评分；通过匹配 audit 后，才用全部 3909 个训练样本拟合新的全训练匹配表，再对新的校正进行 development 评分。

对每个 `预测步 × 末值流量8档 × 幅度4档` context，按训练支持而非评分误差选择三个 arm 各自相同数量 `k` 的合格子 cell。取被选 cell 的共同训练作者块交集，在每个共同块内为每个 cell 选取相同数量的训练点，使用固定 seed 2025。匹配后每个被选 cell 的总点数 `n`、作者块直方图及收缩权重 `w = n/(n+128)` 均相同；低支持时整个 context 的三个 arm 一起回退原 CONTEXT。预算明细保留原始支持和匹配后支持，不能用原始计数替代匹配预算。

三张新表拟合同一残差 `y − SUM − OWN4`，向原 CONTEXT 收缩。最终预测为 `SUM + OWN4 + 一张共享表`，不会重复添加 CONTEXT。匹配只是训练点预算及部分支持结构相同；三个 arm 划分不同、选中的具体样本身份可以不同，同一预测起点的不同节点仍相关。每个单元的独立起点数量可以不同，预算明细另行报告。单个 seed、相同 cell 数及预算不能证明完整有效表达容量相同或因果公平。

评分采用三个 arm 同时落入已匹配合格子 cell 的共同支持点。新表之间和相应 CONTEXT、OWN4 的比较均使用同一评分 mask；共同覆盖率的分母是该阶段全部有效评分点。匹配交叉表和作者块表也使用这个共同支持子集，不能与全量旧 GW13 表的 MAE 直接当作同一评分集合比较。

## 固定门槛与解释边界

匹配 audit 必须同时满足：共同支持有效评分覆盖率至少 25%；TRAJECTORY 在共同点上严格优于 DIRECTION、MAGNITUDE 和 CONTEXT；各项预算相等不变量全部成立。25% 是固定操作性支持阈值，不是统计置信标准。覆盖不足说明该匹配探针支持不足，不能解释为历史顺序没有信息。

只有全部通过，才允许新全训练匹配拟合和新 development 评分。development 只报告共同覆盖率、TRAJECTORY 是否严格优于 DIRECTION、MAGNITUDE 和 CONTEXT，以及三个既定保护组是否不劣于同 mask 的 OWN4：真值 ≥314、预测步 7–12、50 ≤ 真值 <200。保护组若在共同 mask 下为空，记为不可判，不能当作通过。

原 SUM 已经训练过完整训练集。作者顺序留出只隔离新查表拟合，不能当成整体模型的独立验证；作者顺序未认证为日历顺序，窗口时间重叠也未认证，128 个样本的间隔不能保证物理时间隔离。新 context 没有日内时间或星期条件，不能声称完成 GW12 时钟条件对照。

无论新诊断是否通过，原 GW13 失败结论均保留。新通过只能作为继续研究的诊断信息，不构成独立确认、统计显著性、因果机制或架构有效性结论；不会设置架构授权为真，不评价测试集，也不自动启动新神经训练。失败不授权修改候选、门槛或 seed 补跑。

## 回传结果

先回传三个文件：

```bash
cat "$GW14_RUN/report.md"
cat "$GW14_RUN/checks.json"
cat "$GW14_RUN/matching_summary.csv"
```

需要细看分组时，再查看：

| 文件 | 内容 |
|---|---|
| `raw_audit_cross.csv`、`raw_validation_cross.csv` | 原 GW13 重放的形态 × 转折 × 起点末值流量交叉结果 |
| `cross_summary.csv` | 原重放的形态 × 转折简表，以及 none 按起点流量的组成与贡献；两种视图不能相加 |
| `raw_audit_phase_lead.csv`、`raw_validation_phase_lead.csv` | 原重放的形态 × 转折 × 预测步结果 |
| `raw_audit_blocks.csv`、`raw_validation_blocks.csv` | 原重放的作者顺序块结果 |
| `matched_audit_cross.csv`、`matched_audit_phase_lead.csv`、`matched_audit_blocks.csv` | 匹配 audit 共同支持子集结果 |
| `matched_development_cross.csv`、`matched_development_phase_lead.csv`、`matched_development_blocks.csv` | 通过匹配 audit 后才产生的新 development 共同支持子集结果 |
| `audit_matching_cells.csv` | audit 的选择、原支持和匹配训练预算明细 |
| `full_matching_cells.csv` | 通过匹配 audit 后才产生的全训练匹配预算明细 |
| `matched_audit_model.json` | 新 audit 三张匹配统计表及对应预算记录 |
| `matched_full_model.json` | 通过匹配 audit 后才产生的新全训练三张匹配统计表 |
| `audit_fitted_before_score.json`、`development_fitted_before_score.json` | 对应阶段的新表先保存后评分记录；后者仅在匹配 audit 通过后产生 |

共同支持为空时，匹配指标记为不可用，不生成该阶段的匹配交叉 CSV；原始重放文件仍会保留。块与点均不能视为独立观测。输入改变、认证失败、中断的部分产物不算完整结果；首先根据 `checks.json` 和报告判断是否完整，以及新 development 是否获准评分。
