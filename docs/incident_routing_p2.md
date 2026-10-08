# P2：事故条件化 EstimationGate

P2 在 IGSTGNN 的 `EstimationGate` logit 上增加事故条件残差：

```text
native_gate_logit + accident_condition_delta
                    → sigmoid
                    → 扩散/固有分支的输入分配
```

P2 保留 ICSF、动态图、扩散/固有解耦主干和 TIID。事故条件使用已有 `report_location_v1` 信息：

- ICSF 已编码的报告年龄和预测时钟；
- 事故—节点关系向量 `distances`；
- 当前分解层的历史隐藏状态。

条件分支最后一层零初始化，因此同一份原始主干权重下，P2 初始输出应与原始 `fixed` 模型一致。断开节点的事故条件增量在输出端乘支持掩码，训练后也保持为零。

## 本地验证边界

本地当前系统 Python 没有安装 PyTorch，因此本机完成了 Python 编译、JSON 解析、`git diff --check` 和不依赖 PyTorch 的运行脚本回归；`tests/test_acdg.py` 需要在服务器 `igstgnn` 环境执行。工程验证不等于科学效果验证。

## 服务器流程

服务器不使用 SSH。代码从 GitHub 拉取，数据目录沿用现有 `Contra_Costa_v8_dev` 包。

### 1. 更新代码并确认环境

```bash
set -euo pipefail
cd <SERVER_REPO>
git fetch origin
git switch research/chronological-tiid
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
git log -1 --format='commit=%H%nsubject=%s'
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

### 2. 运行 P2 工程测试

```bash
python -m unittest tests.test_acdg -v
```

然后在真实数据包上运行确定性 CUDA 检查：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 python experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_acdg_cuda_check_s2025 \
  --protocol experiments/chronological/incident_routing_p2.json \
  --variant acdg \
  --device cuda:0 \
  --seed 2025 \
  --check
```

检查目录中的 `summary.json` 应至少满足：

- `status` 为 `ENGINEERING_CHECK_PASS`；
- `variant` 为 `acdg`；
- `initial_max_abs_difference_from_A` 不超过 `0.001`；
- `environment.device` 为 `cuda:0`；
- `environment.gpu_name` 非空并显示 V100；
- `environment.deterministic_algorithms` 为 `true`；
- `environment.tf32` 为 `false`。

这个检查只确认代码、GPU、梯度和 checkpoint 流程，不能用于判断 P2 是否提高预测精度。

### 3. 先跑固定 baseline 和 P2 各一轮

为估算 V100 单轮耗时，两个变体分别使用独立目录：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 python experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_fixed_p2_s2025 \
  --protocol experiments/chronological/incident_routing_p2.json \
  --variant fixed \
  --device cuda:0 \
  --seed 2025 \
  --stop-after-epoch 1

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 python experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_acdg_p2_s2025 \
  --protocol experiments/chronological/incident_routing_p2.json \
  --variant acdg \
  --device cuda:0 \
  --seed 2025 \
  --stop-after-epoch 1
```

查看两个 `summary.json` 的 `runtime_epochs[0].seconds`、`parameters` 和环境信息后，再决定每个 V100 会话的续跑轮数。续跑保持同一 commit、数据包、协议和输出目录：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 python experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_acdg_p2_s2025 \
  --protocol experiments/chronological/incident_routing_p2.json \
  --variant acdg \
  --device cuda:0 \
  --seed 2025 \
  --resume \
  --stop-after-epoch N
```

`N` 表示累计完成轮数。`fixed` 与 `acdg` 必须使用不同输出目录；一个 run 完成或放弃前不要拉取新代码。

## 结果回传

请回传以下文件中的文本内容或关键字段：

- `contra_fixed_p2_s2025/summary.json`；
- `contra_acdg_p2_s2025/summary.json`；
- 两个目录的 `best_validation_predictions.npz` 是否存在；
- 如任务中断，回传 `last_checkpoint.pt` 是否存在、最后完成 epoch 和日志最后 30 行。

工程状态和科学状态分开记录：完整训练前不比较精度；完整训练后先比较 `all_nodes.mae_macro`，再按事故窗口/无事故窗口、支持节点/无支持节点和 H1–H3/H4–H6/H7–H12 分层分析。
