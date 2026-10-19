# GW8 服务器交付：DIF 内部发送者×时间槽联合核

发布分支：`research/edge-slot-kernel-gw8`。继承 GW7 `a695429de8a17b5dc30c222f39323267fd83c2a8`，只新增 GW8 文件，保留认证历史源码的原始字节。

本版已经实现，尚未执行服务器单元测试、真实批次预检或训练。遵循本地不运行 Python/模型/测试的约定；本地交付检查仅含人工静态审阅、文本/Git 检查和文件字节核对。零初始化精确恢复、梯度与数值边界是待服务器验证的契约，不是已通过的结果。

## 结构与公平比较

新增低维核在第一层原生 DIF 内运行，读取各发送者在原 fc 混合前的隐藏时间槽；接收者查询联合决定“哪个邻居的哪个槽”进入消息。更新在 gcn_updt 后、原 dropout 前注入，继续参与 backcast、层间残差和原生预测递推。原输出头、事故路径与全部原参数继续训练。

| 臂 | 内容 |
|---|---|
| R_CONTINUE | 当前强 R 继续训练 |
| R_LOCAL_KERNEL | 同容量本节点时间核 |
| R_SEPARABLE_KERNEL | 同容量图核，用联合分布的发送者边缘×时间槽边缘 |
| R_JOINT_KERNEL | 同容量发送者×时间槽联合核 |

三个新增臂具有完全相同的参数和初始张量，新增量为 `4*C*16+K`，实际 C=64、K=3 时为 4,099 个参数。联合与可分离核在同输入、同参数下保留相同的两轴边缘，检验的是关联的价值。所有臂保留完整原生 R 路径；新增隐状态输出投影为零初始化。

固定 d=16、每接收者最多 8 个先验发送者。图先验由当前原生各图源的一阶邻接按有效行归一化、等权混合后 top-k 得到；记录保留质量，不使用未来真值路由。槽指本次调用的隐藏状态位置，预测调用含生成状态，不能解释为真实道路时延。

共同来源是既有 GW7 中 R_CONTINUE BEST 第 4 轮，checkpoint SHA-256：

`568ff5e0dcea220cb28e2e9bec794ef1894411e8dd8acd4bf427bfb2e425ca65`

来源认证同时核对完成的 GW7 四臂状态、身份链、BEST/LAST 元数据、预测归档和数据。源码、归档或源目录不应挪动或修改。四臂共同新增 **5 轮、410 次更新**（既有 82 batch/轮），全 12 步损失；开始重置优化器，恢复时完整恢复，BEST 包含第零轮，固定 LAST 为第 5 轮。

## 克隆与设置

在服务器上，从包含 `IGSTGNN-GW7` 与 `data` 的目录操作：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
git clone --single-branch --branch research/edge-slot-kernel-gw8 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW8
cd IGSTGNN-GW8
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export GW8_GW7_RUN="$(realpath ../IGSTGNN-GW7/experiments/chronological_runs/contra_original_gw7_screen_s2025)"
export GW8_DATA="$(realpath ../data/xtraffic/Contra_Costa)"
export GW8_RUN="$PWD/experiments/chronological_runs/contra_original_gw8_screen_s2025"
export GW8_TAG="$(date +%Y%m%d_%H%M%S)"
```

默认无 Slurm allocation 时 launcher 提交 GPU 作业。如果已分配 GPU 资源、直接在 gpu05 运行且不使用 Slurm，在同一 shell 设置 `export GW8_DIRECT=1`。环境变量须在父 shell 中 export；launcher 内设置不会返回父 shell。

## 一次完成预检、训练及评价

```bash
bash experiments/lag_transport_gw8/run.sh all
```

执行顺序是文件字节检查 → 合成模型/协议测试 → batch 48 真实数据零更新前反向预检 → 四臂各 5 轮 → BEST/LAST 配对评价 → 锁定筛查。任何一步失败就停止，不自动缩小 batch、改结构或扩展预算。

若希望先只核查可运行性：

```bash
bash experiments/lag_transport_gw8/run.sh check
```

预检比较 train/eval 预测与 RNG，核对每层原 DIF forecast 的任务梯度、初始新增投影的梯度、完整前反向耗时和显存。训练模式检查会改变临时模型的 BN 缓冲；检查后从认证源重新载入，临时模型被丢弃，源 checkpoint 写入为零。正式训练独立重建模型，不继承预检状态。

`check` 完成后调用 `all` 会再执行一次预检。只想执行一次完整流程可直接用 `all`。所有检查与训练日志都需要服务器实际输出才能确认成功。

来源 BEST/LAST 自动选择与固定源身份匹配的完整归档；存在歧义或需指定时，在启动前设：

```bash
export GW8_GW7_BEST_EVAL="/绝对路径/原GW7运行目录/eval_best_原tag"
export GW8_GW7_LAST_EVAL="/绝对路径/原GW7运行目录/eval_last_原tag"
```

## 中断与评价重跑

保留相同来源、数据、种子、配置和 GW8_RUN，在同一 shell 重新激活环境并执行：

```bash
export GW8_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/lag_transport_gw8/run.sh resume
```

从最后一个原子提交的完整 epoch 恢复；被杀死的未提交 epoch 重放。无需删除 active.lock 文件、checkpoint 或优化器。恢复不会把总预算从 5 轮增加到 10 轮。已完成的臂跳过训练，评价写入新的 tag 子目录。

只重新产生评价与筛查，不训练：

```bash
export GW8_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/lag_transport_gw8/run.sh evaluate
```

## 查看结果

运行结束后，父 shell 中保持相同 GW8_RUN/GW8_TAG：

```bash
cat "$GW8_RUN/summary.json"
cat "$GW8_RUN/eval_best_$GW8_TAG/report.md"
cat "$GW8_RUN/eval_last_$GW8_TAG/report.md"
cat "$GW8_RUN/screen_$GW8_TAG/screening.json"
```

忘记 tag 时先定位已有文件：

```bash
find "$GW8_RUN" -maxdepth 2 \( -name report.md -o -name screening.json \) -print
```

各臂的 `history.json` 保存逐轮记录，`summary.json` 保存阶段零、BEST/LAST、预算与身份；评价目录包含 cases/horizons/groups/comparisons.csv、paired_predictions.npz、metadata.json 和 report.md。预检报告位于 `${GW8_RUN}_verification_${GW8_TAG}.json`。

## 判据与解释

JOINT 的 BEST pooled MAE 必须相对本次 R_CONTINUE、SEPARABLE 都至少改善 0.5%，并优于 LOCAL；LAST 相对自身 BEST 退化不超过 1%；真实值≥314 与 7—12 步组 MAE 均不劣于同次 R。旧 13.61847 是历史起点，不能代替新预算控制。

未超过可分离核，不保留联合关联的必要性主张；未超过本地核，不保留新增跨节点核的必要性主张；失败不自动追加轮数。全部是反复使用的开发验证集上的单种子筛查，没有独立测试结论。

时空联合消息及注意力已有 TraverseNet/STSGCN 等先例。本版尚不保证强创新或性能收益；完整公式、相关方法、失败情形和贡献边界见 [设计说明](docs/igstgnn_edge_slot_kernel_gw8_design.md)。
