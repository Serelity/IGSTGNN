# GW6 服务器代码与运行

本包实现上轮结构方案的联合训练主模型：保留预测起点前各历史窗口、图来源、传播阶数的消息值，直接读取十二个目标时距的 DIF 表征，再与原 INH 表征融合。`stage=joint` 重建 DIF 预测路径；`stage=probe` 是冻结 GW4 后增加输出残差的机制探针。二者不能混作同一实验。

本次本地整理源码并检查发布文件与哈希。以下单元测试、Python 语法/导入检查、真实 batch48 前反向和训练均留在服务器执行；文档中的成功标志是验收条件，不是本次已取得的结果。

## 1. GitHub 拉取与设置路径

本轮分支为 `research/graph-memory-gw6`，仓库为 `https://github.com/Serelity/IGSTGNN`。在服务器建立独立 `IGSTGNN-GW6` checkout。保留原 GW1/GW2/GW4/GW5 运行目录、检查点与认证评估，GW6 会读取它们的证书链，不会自动重建旧缓存。

在原来的 **GW5 工作目录、仍保留正确 `$GW5_RUN` 的终端**，先记录绝对路径：

```bash
export GW6_GW5_RUN="$(realpath "$GW5_RUN")"
test -f "$GW6_GW5_RUN/summary.json"
```

如果已经换过终端，请把 `GW6_GW5_RUN` 设置为实际完成的 GW5 run 绝对路径。它是含 `summary.json` 和 `identity.json` 的实验目录，不是 GW5 仓库目录。此前回传的 GW4 来源位于 `IGSTGNN-code/IGSTGNN-GW4/.../contra_original_gw4_frozen_s2025`；程序读取已保存的路径，不以本地副本替换服务器原件。

在原 GW5 目录记录来源后，拉取新分支；先前捕获的 `GW6_GW5_RUN` 环境变量需要保留，换终端时重新设置它：

```bash
conda activate igstgnn
cd ..
git clone --single-branch --branch research/graph-memory-gw6 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW6
cd IGSTGNN-GW6
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
export GW6_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW6_RUN="$PWD/experiments/chronological_runs/contra_original_gw6_joint_s2025"
export GW6_EPOCHS=5
export GW6_TAG="$(date +%Y%m%d_%H%M%S)"
```

继续使用已有 `igstgnn` 环境和原数据，不执行无关依赖升级。训练默认沿用原 batch48、学习率、优化器和数据划分，新增参数与资源成本写入运行身份与报告。若旧评估自动发现不唯一，可设置 `GW6_GW5_BEST_EVAL`、`GW6_GW5_LAST_EVAL` 为本次 GW5 的对应评估目录，launcher 会同时传给检查和训练。

## 2. 推荐运行：一个作业内完成检查、训练、评估

```bash
bash experiments/graph_memory_gw6/run.sh all
```

未处于 Slurm 作业内时，launcher 自动 `sbatch`，返回作业号；整个 `check → train → BEST → LAST` 都在该作业中顺序执行。提交返回不代表训练已经完成。已有 GPU allocation 时直接执行；仅在已获得 GPU 的非 Slurm 节点，可显式设置 `GW6_DIRECT=1`。默认申请一张 GPU、3 CPU、24 小时；分区、账户和时限由服务器要求设置，未知分区没有写死。

`check` 先用 `verify_gw6_bundle.py` 核对交付哈希，再运行合成单元测试，对四臂各执行一次真实 batch48、完整十二时距原 MAE 前向和反向，不做真实数据 optimizer.step，也不在 OOM 后暗中缩小批量。合成测试会在微型夹具上更新参数。`GW6_VERIFICATION_PASS` 和 `full_native_backward_checked=true` 仅说明工程检查成功。

默认四臂：

| 臂 | 新实验中的作用 |
|---|---|
| `RECURRENT_CONTROL` | 从共同 GW4 检查点继续训练原递推结构，作为本轮主对照 |
| `POOLED_MEMORY` | 历史消息先池化，再直接解码，检验保留结构轴的价值 |
| `ADDITIVE_MEMORY` | 保留历史消息轴，使用加性条件查询 |
| `JOINT_MEMORY` | 保留历史消息轴，使用接收状态与目标时距的乘积查询 |

四臂从 **同一个已完成 GW4 RECEIVER_RESIDUAL 的 BEST 检查点** 开始，重新初始化本轮优化器，每臂完成同样的新训练轮数。默认各 5 轮只是开发试跑；允许 1–100 轮。历史 GW4/GW5 的已报数字是历史参照，不能称作与本轮候选相同总训练预算的控制。各臂参数量与计算量并不强行相等，必须结合实际成本解释收益。

默认 `--curriculum full`：`warm_epoch=101`，在最大 100 个新训练 epoch 内始终使用完整十二时距损失。这是四臂共同的显式配方变化。`--curriculum native` 才沿用原 `warm_epoch=30, cl_epoch=3`；必须使用新 run 目录，不能与 full 配方续接或混合结论。`config.json` 是可读的默认声明，不是 `--config` 输入，真正生效的解析值保存在 `identity.json`。

## 3. 分步骤命令

已有 GPU allocation 时也可以直接调用 Python；先做检查：

```bash
"$PYTHON_BIN" verify_gw6_bundle.py &&
"$PYTHON_BIN" -m unittest tests.test_graph_memory_gw6_model tests.test_graph_memory_gw6_workflow tests.test_graph_memory_gw6_evaluation -v &&
"$PYTHON_BIN" -u -m experiments.graph_memory_gw6.verify \
  --gw5-run "$GW6_GW5_RUN" --data-dir "$GW6_DATA" \
  --batch-size 48 --device cuda:0 --threads 3 --seed 2025 \
  --stage joint --curriculum full --hidden 32 --node-chunk 32 \
  --output "${GW6_RUN}_verification_${GW6_TAG}.json"
```

检查通过后训练：

```bash
"$PYTHON_BIN" -u -m experiments.graph_memory_gw6.train \
  --gw5-run "$GW6_GW5_RUN" --data-dir "$GW6_DATA" \
  --run-dir "$GW6_RUN" --epochs "$GW6_EPOCHS" \
  --device cuda:0 --threads 3 --seed 2025 \
  --stage joint --curriculum full --hidden 32 --node-chunk 32
```

首次运行要求新 run 目录。中断恢复使用同样的命令加 `--resume`；也可设置新的 `GW6_TAG` 后执行 `bash experiments/graph_memory_gw6/run.sh resume`，依次重新检查、恢复训练、评估。`--epochs 5` 始终指本阶段目标总轮数，不是在已完成轮数上再加 5。只有最后完整 epoch 原子提交；未完成 epoch 会从上次完整状态回放。优化器、调度器、随机数、样本顺序和 curriculum 状态一同恢复，不重新初始化优化器。

要把 5 轮延长到 20 轮，原目录与全部身份参数保持一致，使用 `--resume --epochs 20` 或 `GW6_EPOCHS=20` 的 resume launcher。不得降低已保存的预算或在运行中修改源码。`--only` 可以单独恢复已注册臂，最终正式比较仍要求注册臂均完成共同预算。

训练完成后，同一 GPU 终端顺序评估：

```bash
"$PYTHON_BIN" -u -m experiments.graph_memory_gw6.evaluate \
  --run-dir "$GW6_RUN" --data-dir "$GW6_DATA" --device cuda:0 \
  --checkpoint best --output-dir "$GW6_RUN/eval_best_$GW6_TAG" &&
"$PYTHON_BIN" -u -m experiments.graph_memory_gw6.evaluate \
  --run-dir "$GW6_RUN" --data-dir "$GW6_DATA" --device cuda:0 \
  --checkpoint last --output-dir "$GW6_RUN/eval_last_$GW6_TAG"
```

BEST 按各臂本轮验证 MAE 选取，LAST 使用相同已完成 epoch；旧起点不参与本轮 BEST 选择。程序不读取测试集做选模或性能评价。不要在 `sbatch` 提交命令后马上从登录终端运行评估；使用 `all` 或等待训练完成。

## 4. 探针、结果与验收

冻结机制探针需要新目录，显式 `--stage probe --arms POOLED_MEMORY ADDITIVE_MEMORY JOINT_MEMORY`，不能带 `RECURRENT_CONTROL`，因为冻结控制没有可训练参数。launcher 对应：

```bash
export GW6_STAGE=probe
export GW6_ARMS="POOLED_MEMORY ADDITIVE_MEMORY JOINT_MEMORY"
export GW6_RUN="$PWD/experiments/chronological_runs/contra_original_gw6_probe_s2025"
export GW6_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/graph_memory_gw6/run.sh all
```

探针收益仅表示冻结锚点上的附加路径作用，不能作为完整联合模型结论。主模型优先看 `JOINT_MEMORY` 相对本轮 `RECURRENT_CONTROL`，再看 POOLED/ADDITIVE 两个机制对照；接受负收益，并报告每个目标时距、固定困难窗口、时间和显存成本。单种子且多次使用同一开发验证集，仍需独立种子/划分验证。

完成后贴回：

```bash
cat "${GW6_RUN}_verification_${GW6_TAG}.json"
cat "$GW6_RUN/summary.json"
cat "$GW6_RUN/eval_best_$GW6_TAG/report.md"
cat "$GW6_RUN/eval_last_$GW6_TAG/report.md"
```

`GW6_TRAINING_COMPLETE` 仅表示注册预算全部完成，`GW6_EVALUATION_COMPLETE` 仅表示评估产物生成；都不自动判定精度改善。保留 `identity.json`、`source_reference.json`、`budget.json`、各臂 `last.pt`/BEST/history、评估 metadata/配对预测和资源统计，便于核查与重分析。
