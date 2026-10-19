# GW11：自身状态的作用位置——内部解耦门控与输出头

发布分支 `research/state-gate-gw11`。沿用服务器现有 `IGSTGNN-GW9` checkout，保留旧运行的绝对路径。GW9、GW10 执行源码、原结果和失败门槛不改；此次是新的固定结构比较，不是续训失败的 GW9 BRANCH 或 GW10 JOINT。

本地仅做静态审阅、字节核对与 Git 检查，没有执行 Python、导入、语法检查、测试、模型或训练。源码包包含 72 项服务器测试；是否通过以服务器日志为准。

## 更新和启动

使用原来的计算节点、`igstgnn` 环境及与认证源一致的运行条件。保留 batch=48、seed=2025、3 CPU 线程，PyTorch/NumPy/CUDA 版本必须与源一致。建议沿用以往 1 GPU、3 CPU、32 GB CPU 内存的资源配置；这是资源预留建议，未测得本轮耗时或峰值显存。本脚本不提交调度器、不安装依赖。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git fetch origin research/state-gate-gw11:refs/remotes/origin/research/state-gate-gw11
if git show-ref --verify --quiet refs/heads/research/state-gate-gw11; then
  git switch research/state-gate-gw11
else
  git switch -c research/state-gate-gw11 origin/research/state-gate-gw11
fi
git pull --ff-only origin research/state-gate-gw11
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW11_PROBE_RUN="$PWD/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025"
export GW11_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW11_TAG="$(date +%Y%m%d_%H%M%S)"
export GW11_RUN="$PWD/experiments/chronological_runs/contra_gw11_state_gate_s2025_${GW11_TAG}"
export GW11_MASK_DIR="${GW11_RUN}_mask_audit"
export GW11_DEVICE="cuda:0"
export GW11_STAGE="all"
unset GW11_RESUME GW11_EVALUATE_ONLY

bash experiments/state_gate_gw11/run.sh
```

脚本从原 probe 同级目录自动寻找已完成 GW10 结果及其 `cache_metadata_sha256` 绑定的完整缓存。只按完成顺序定位，不比较模型分数挑选来源；随后完整认证源码、轴、mask、模型表哈希和原有失败筛查，复算 ORIGIN/OWN4/JOINT 指标。若自动定位失败，显式设置实际旧目录后再运行同一脚本：

```bash
export GW11_CACHE_DIR="你的已完成_GW10_CACHE_DIR_绝对路径"
export GW11_GW10_RUN="你的已完成_GW10_RUN_绝对路径"
bash experiments/state_gate_gw11/run.sh
```

不要把这些变量指向新空目录，不重新导出 GW10 缓存，不覆盖原 GW10 结果。若 Git 因本地修改拒绝切换，先核对和保存修改，不使用 `reset --hard`。

## 掩码审计与停止条件

第一阶段只使用既有 train.npz 中的目标值、valid mask、作者顺序，以及经哈希认证的 GW9 LAST 保存的最后一轮训练排列。LAST 只在 CPU 以 weights-only 方式反序列化，不运行神经网络，不申请新的 GPU 推理。

对两种顺序分别比较“逐 batch 最小值定义的 mask”和“全训练集最小值定义的 mask”，输出差异点数、错误纳入/排除数、节点和值分布。null 数值不同但有效点完全相同是等价 mask，不因此拒绝。零值仅作为数值统计，不擅自解释为缺测。

只有两个明确顺序的逐点 mask 均一致，才允许训练。发现差异会停在审计报告，不自动改变 mask 或评价方式。已有排列审计不证明所有旧轮次、新排列都一致；正式训练每一批继续与完整缓存的目标值和 mask 精确核对。

只做 CPU 审计可以运行：

```bash
export GW11_STAGE="audit"
bash experiments/state_gate_gw11/run.sh
cat "$GW11_MASK_DIR/report.md"
```

此后设回 `GW11_STAGE=all` 会复用完整审计并重新认证。若审计报告 `eligible_for_training=false`，先回传报告，不启动模型阶段。

## 四臂与共同起点

四臂都继承完整的 GW9 `SUM_CONTROL BEST`：认证 GW8 R 骨干，加上已经训练的 GW9 SUM 读出头。没有移除 SUM 头，也没有用更弱的原 R 替代同源对照。来源 identity 固定为 `5e20efbdcca907afda994889cdc460276fdd560e235047a11c20743494d36757`。

| 实验臂 | 唯一新增部分 | 新参数 |
|---|---|---:|
| SUM_CONTINUE | 无，完整 SUM 继续训练 | 0 |
| STATIC_GATE | 节点/时间条件的原生门控 logit 增量 | 725 |
| STATE_READOUT | 自身状态条件的输出增量 | 725 |
| STATE_GATE | 自身状态条件的原生门控 logit 增量，唯一候选 | 725 |

三个新增臂各有五个 `Linear(7,16)→Tanh→Linear(16,1)`，每个 145 参数。输出层 weight/bias 全零，三臂初始新增张量相同。新增参数量相同不保证有效函数容量、输入分布或计算成本完全相同，需连同对照和完整计时解释。

STATE 两臂共同输入：原归一化流量末值、历史平均绝对变化的 `log1p`、预测起点日内时间和星期的 sin/cos 四列，以及 12 个位置的 `0..11/11` 坐标。STATIC 将前两列换成原节点两组嵌入的均值，其余相同；所有输入实际参与计算，没有空置参数凑容量。

GATE 仅观察五层真实执行的 `estimation_gate.fully_connected_layer_2` 输出，加入增量后由原 sigmoid 与历史长度裁剪继续处理；不重复原生门控或主干前向。READOUT 在完整 SUM 输出后加五个同状态 adapter 的平均值，保留未来 12 步。内部历史位置与输出预测位置含义不同，这正是作用位置对照的一部分，不声称两者计算图等价。

源与新增参数全部参与联合训练，固定新增 3 轮，每臂 82×3=246 次更新；沿用原 Adam 学习率、weight decay、梯度裁剪与完整 12 步 MAE。开始时重置 Adam，续跑恢复原状态。每轮四臂共用 `RandomState(2025+epoch)` 排列，并在各臂该轮前重置相同 `2025+10000×epoch` 随机种子。新增模块不含随机层。BEST 包含阶段零，LAST 固定第 3 轮，不早停、不扩预算。

正式更新前四臂全部逐点回放认证 SUM BEST 的完整验证预测，dtype、值、目标和支持轴必须一致；不放宽容差。门控的数学零增量不代替实际回放。

## 查看结果

```bash
cat "$GW11_MASK_DIR/report.md"
cat "$GW11_RUN/report.md"
cat "$GW11_RUN/screening.json"
```

若模型尚未完成，先看：

```bash
cat "$GW11_RUN/summary.json"
cat "$GW11_RUN/STATE_GATE/history.json"
```

完整 BEST/LAST 缓存位于 `result_pointer.json` 所指的新 `evaluation_*/best`、`evaluation_*/last` 子目录。每个目录包含 `paired_predictions.npz`、`metadata.json`、`metrics.json`，保留七个案例：FROZEN_SUM、ORIGIN、OWN4 和四个新训练臂。ORIGIN/OWN4 是原 GW10 表的固定历史参考，不是等梯度预算对照。

固定通过条件：STATE_GATE BEST 相对本次 SUM_CONTINUE、STATIC_GATE、STATE_READOUT 的 pooled MAE 均至少改善 0.5%；不劣于认证 OWN4；LAST 相对自身 BEST 退化不超过 1%；真实流量 ≥314、时距 7–12、真实流量 50–200 三组 MAE 不劣于本次 BEST SUM_CONTINUE。偏差继续报告，原 MAE 目标不变，不增加偏差损失。真实值分组仅用于评价，不输入模型。失败保留，不更换候选、门槛、隐藏维度或轮数。

## 中断续跑和重新导出

保留原 `GW11_RUN` 与 `GW11_MASK_DIR`，不要重新生成 TAG。换 shell 后先按日志重新 export 这两个原绝对路径，来源目录也保持不变。

```bash
export GW11_STAGE="all"
export GW11_RESUME="1"
unset GW11_EVALUATE_ONLY
bash experiments/state_gate_gw11/run.sh
```

每个实验臂每轮完成验证后原子保存完整模型、Adam、scheduler、RNG、历史与 BEST 角色。进入训练前另存绑定运行身份的 `training_started.json`；若已有训练标记或训练记录但缺失任何臂的 `last.pt`，会要求恢复丢失检查点，不从零重建该臂。已提交的“臂×轮次”跳过，未提交轮次从上一检查点重放，仍以每臂累计三轮为上限。三个新增臂并非同时驻留 GPU，按臂顺序构建和释放。若初始化尚未提交 `identity.json` 就中断，保留残留目录，换新 GW11_RUN；不会猜测缺失身份。

训练已完成但最终导出中断时，可仅重新评价并生成新的不可变结果子目录，不增加训练：

```bash
unset GW11_RESUME
export GW11_EVALUATE_ONLY="1"
bash experiments/state_gate_gw11/run.sh
```

## 研究边界

GW10 JOINT 未过筛，OWN4 相对 ORIGIN 的约 0.32% 收益只作为弱动机。旧 P2 已做事故条件门控并报告负结果；此次明确分离自身条件、参数量和作用位置，不将门控本身包装为首次提出。参考 [D²STGNN](https://www.vldb.org/pvldb/vol15/p2733-shao.pdf) 的估计门控、[FiLM](https://arxiv.org/abs/1709.07871) 的条件调制；预测统计量与损失的对应参考 [Gneiting](https://arxiv.org/abs/0912.0902)。本轮主张仍为可证伪结构假设。

高流量真值组的负偏差不单独证明输入条件校准失效；MAE 对应中位数目标，不能默认与均值无偏同时最优。开发验证已经反复用于设计；通过只是单种子开发筛查，独立确认尚未进行，不评估 test。原发布 object 数据会整体反序列化包含 test 对象，但模型只访问训练/验证切片，未创建 test loader。
