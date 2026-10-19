# GW12：冻结完整 SUM，隔离自身状态读出的作用

发布分支 `research/frozen-readout-gw12`。沿用服务器现有 `IGSTGNN-GW9` checkout，保留认证来源的绝对路径。本轮从原 GW9 `SUM_CONTROL BEST` 出发，不从已经退化的 GW11 LAST 出发，也不延长 GW11 训练。

本地仅做文本静态审阅、字节和 Git 检查，没有执行 Python、导入、语法检查、测试、模型或训练。服务器入口先核验源码，再执行四个新测试模块中的 57 项测试，最后进行回放、训练与评价；是否通过以服务器日志为准。

## 更新和首次运行

继续使用原 `igstgnn` 环境和认证运行条件。保留 batch=48、seed=2025、3 CPU 线程，以及与来源相同的 PyTorch/NumPy/CUDA 版本。可沿用以往 1 GPU、3 CPU、32 GB CPU 内存的资源预留；这是建议，没有测得本轮耗时或峰值显存。脚本不提交调度器、不安装依赖。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git fetch origin research/frozen-readout-gw12:refs/remotes/origin/research/frozen-readout-gw12
if git show-ref --verify --quiet refs/heads/research/frozen-readout-gw12; then
  git switch research/frozen-readout-gw12
else
  git switch -c research/frozen-readout-gw12 origin/research/frozen-readout-gw12
fi
git pull --ff-only origin research/frozen-readout-gw12
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW11_RUN="$PWD/experiments/chronological_runs/contra_gw11_state_gate_s2025_20261019_164846"
export GW12_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW12_TAG="$(date +%Y%m%d_%H%M%S)"
export GW12_RUN="$PWD/experiments/chronological_runs/contra_gw12_frozen_readout_s2025_${GW12_TAG}"
export GW12_DEVICE="cuda:0"
export GW12_THREADS="3"
unset GW12_RESUME GW12_EVALUATE_ONLY

bash experiments/frozen_readout_gw12/run.sh
```

若 Git 因本地修改拒绝切换，核对并保存修改后再继续，不执行 `reset --hard`。`GW11_RUN` 必须显式指向已完成的原 GW11 运行，不能省略、不能填新目录。程序从它的身份文件恢复并认证原 GW9、GW10 缓存/结果和掩码审计，不重新选择历史结果或重新拟合查表方法。

只有全新运行允许省略 `GW12_RUN`，入口会生成带时间戳的新目录。建议按上例显式保存变量，便于查看和续跑。已有输出目录不能按 fresh 模式覆盖。

## 研究问题与固定比较

GW11 LAST 消融中，STATE_GATE 的退化主要已体现在共同更新的源模型；STATE_READOUT 在其共同训练后的源模型上有补偿作用，但完整模型仍未超过原 SUM 起点。GW12 因此只回答：固定完整源模型后，自身状态是否比静态条件提供独立的输出修正收益？这项动机不预先证明状态修正有效。

| 案例 | 源模型 | 新增部分 | 新增可优化参数 |
|---|---|---|---:|
| FROZEN_SUM | 原 GW9 SUM BEST，始终固定 | 无 | 0 |
| STATIC_READOUT | 同一固定源模型 | 节点/时间条件输出修正 | 725 |
| STATE_READOUT | 同一固定源模型 | 自身历史状态输出修正，唯一候选 | 725 |

完整源模型包含 GW8 R 骨干及已训练的 GW9 SUM 读出头。两种新增头各有五个 `Linear(7,16)→Tanh→Linear(16,1)`，取平均作为输出增量；第二层 weight/bias 全零，两个头的初始新增张量相同，阶段零均须精确恢复认证 SUM 缓存。

STATE 的七列输入为归一化历史末值、历史平均绝对变化的 `log1p`、预测起点日内时间和星期的 sin/cos 四列，以及 12 个预测位置的 `0..11/11` 坐标。STATIC 将前两列替换为原节点两组嵌入的均值，其余相同。参数量、训练预算相同不保证输入分布或有效函数容量相同，需连同结果解释。

固定源模型始终 `eval`，在 `torch.no_grad` 中每批前向一次，输出由两个新增头共享。为保留来源实际使用的数值执行路径，源参数的 `requires_grad=True` 标志保持原值；它们不进入优化器，不累积梯度，不更新。程序在每轮及整个流程前后检查源张量哈希，验证全程 FROZEN_SUM 与原 GW9 SUM 缓存逐点、dtype 精确一致。冻结语义以优化器成员、梯度和张量哈希证明。

既有加载器会整体反序列化原始 object 数据文件，其中包含测试对象；本轮只创建和遍历训练与验证 loader，不创建测试 loader。

两个新增头固定各训练 3 轮，每轮 82 批、每头 246 次更新，使用完整 12 步原 MAE、原 weight decay 和梯度裁剪。Adam 从新状态开始，沿用 0.002 / 0.001 / 0.001 的学习率序列。每轮共用 `RandomState(2025+epoch)` 排列；不训练源模型，不加入偏差损失，不用未来真值构造输入，不延长预算、不早停。

FROZEN_SUM 是零更新参照，不是与两个头等训练预算的对照；STATIC_READOUT 是状态输入的同参数量、同更新预算对照。BEST 包含阶段零，LAST 固定第 3 轮。GW10 的 OWN4 只作固定历史参考，不参与本次信息探针门槛。

## 筛查含义

唯一候选 STATE_READOUT 必须同时满足：

- BEST MAE 相对 FROZEN_SUM 和同次 STATIC_READOUT 各改善至少 0.5%。
- LAST 相对自身 BEST 的 MAE 退化不超过 1%。
- BEST 在真值 ≥314、预测步 7–12、50 ≤ 真值 <200 三组的 MAE 均不劣于 FROZEN_SUM。

偏差和其他指标同时报告，不作为新的事后选择规则。通过仅记录 `state_information_probe_passed` 和 `eligible_for_frozen_readout_followup`；`candidate_for_confirmation` 与 `independent_confirmation_authorized` 均为 false。它不证明超越 OWN4，不直接授权独立确认，不构成统计显著性或新机制的证据。开发验证集已多次参与设计，本轮不评价测试集。未通过时保留负结果，不改旧门槛或续加轮次。

## 查看结果

```bash
cat "$GW12_RUN/report.md"
cat "$GW12_RUN/screening.json"
cat "$GW12_RUN/summary.json"
```

训练记录位于运行根目录的 `history.json`。完整 BEST/LAST 预测缓存位于 `result_pointer.json` 指向的新 `evaluation_*` 目录，分别保存 `best/paired_predictions.npz`、`best/metadata.json` 和 `last/paired_predictions.npz`、`last/metadata.json`。固定历史参考与两个新训练头应按各自预算解释。

## 中断后续跑

**续跑必须恢复原 GW12_RUN，不重新生成时间戳。** 每个完整轮次将两个头的状态一起原子提交到根目录 `last.pt`，包含 Adam、scheduler 和随机状态。已提交轮次跳过；未提交的轮次从上一完整轮次重新执行，不扩展预设 3 轮预算。

换 shell 后按实际原目录重新设置：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export GW11_RUN="$PWD/experiments/chronological_runs/contra_gw11_state_gate_s2025_20261019_164846"
export GW12_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW12_RUN="这里填原 GW12 运行目录的完整绝对路径"
export GW12_RESUME="1"
unset GW12_EVALUATE_ONLY

bash experiments/frozen_readout_gw12/run.sh
```

查找已有 GW12 身份文件：

```bash
find "$PWD/experiments/chronological_runs" -maxdepth 2 -type f \
  -path '*/contra_gw12_frozen_readout_s2025_*/identity.json' -printf '%h\n'
```

续跑和只评价均要求原目录同时存在 `identity.json` 与 `last.pt`，入口会在任何 Python/测试前检查。输出目录绝对路径也绑定在新身份摘要内，不能把别处检查点搬进新目录冒充续跑。若初始化阶段尚未完整提交，保留原目录并用新的 fresh 目录重新开始；不要删除身份或训练标记来绕过预算核验。

如果 3 轮训练已完成，仅最终评价/导出被杀，保留上述原目录变量后运行：

```bash
unset GW12_RESUME
export GW12_EVALUATE_ONLY="1"
bash experiments/frozen_readout_gw12/run.sh
```

`GW12_RESUME=1` 与 `GW12_EVALUATE_ONLY=1` 不能同时设置。所有模式均先验证继承源码字节；认证回放、掩码、来源路径或运行条件不一致时直接停止，保留报错，不放宽容差。
